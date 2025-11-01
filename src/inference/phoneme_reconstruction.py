"""
Phoneme-to-Sentence Reconstruction for Valerie Visual ASR.

Uses fine-tuned LLM (Qwen3) to reconstruct natural sentences from phoneme sequences.
Implements LoRA-based fine-tuning for efficiency and beam search for generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    GenerationConfig, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, PeftModel
import numpy as np
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import re
import logging

from src.data.phoneme_utils import IDX_TO_PHONEME, PHONEME_TO_IDX
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReconstructionConfig:
    """Configuration for phoneme reconstruction."""
    
    # Model settings
    model_name: str = "Qwen/Qwen3-0.6B"
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Generation settings
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 4
    do_sample: bool = True
    
    # Quantization
    use_quantization: bool = True
    quantization_bits: int = 4
    
    # Processing
    batch_size: int = 8
    max_phoneme_length: int = 200
    
    # Special tokens
    phoneme_separator: str = " "
    sentence_start_token: str = "<|start|>"
    sentence_end_token: str = "<|end|>"
    
    # Confidence scoring
    confidence_method: str = "token_probs"  # token_probs, sequence_prob, entropy
    min_confidence: float = 0.1


class ReconstructionResult(NamedTuple):
    """Result from phoneme reconstruction."""
    
    text: str                    # Reconstructed text
    confidence: float           # Confidence score
    log_prob: float            # Log probability
    phoneme_input: List[str]   # Input phonemes
    tokens: List[str]          # Generated tokens
    attention_weights: Optional[torch.Tensor] = None


class PhonemeToSentenceReconstructor:
    """
    Phoneme-to-sentence reconstruction using fine-tuned LLM.
    
    Uses Qwen3 with LoRA fine-tuning to convert phoneme sequences
    to natural language sentences with proper grammar and punctuation.
    """
    
    def __init__(self, config: ReconstructionConfig, device: str = "auto"):
        self.config = config
        self.device = self._setup_device(device)
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        
        self._load_model()
        self._setup_generation_config()
        
        # Phoneme processing
        self.phoneme_vocab = IDX_TO_PHONEME
        self.phoneme_to_idx = PHONEME_TO_IDX
        
        logger.info(f"✅ PhonemeToSentenceReconstructor initialized:")
        logger.info(f"   Model: {config.model_name}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   LoRA: {config.use_lora}")
        logger.info(f"   Quantization: {config.use_quantization}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        return torch.device(device)
    
    def _load_model(self):
        """Load and setup the LLM."""
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add special tokens if needed
            special_tokens = {
                "pad_token": "<|pad|>",
                "eos_token": "<|endoftext|>",
                "additional_special_tokens": [
                    self.config.sentence_start_token,
                    self.config.sentence_end_token
                ]
            }
            
            num_added = self.tokenizer.add_special_tokens(special_tokens)
            if num_added > 0:
                logger.info(f"Added {num_added} special tokens")
            
            # Quantization config
            quantization_config = None
            if self.config.use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True if self.config.quantization_bits == 4 else False,
                    load_in_8bit=True if self.config.quantization_bits == 8 else False,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto" if self.device.type == "cuda" else None,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            
            # Resize embeddings if special tokens were added
            if num_added > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Apply LoRA if configured
            if self.config.use_lora:
                self._apply_lora()
            
            # Move to device if not using device_map
            if not self.config.use_quantization:
                self.model.to(self.device)
            
            self.model.eval()
            
            logger.info(f"✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _apply_lora(self):
        """Apply LoRA fine-tuning configuration."""
        
        try:
            # LoRA configuration
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Apply LoRA
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            self.model.print_trainable_parameters()
            
            logger.info("✅ LoRA applied successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ LoRA application failed: {e}")
    
    def _setup_generation_config(self):
        """Setup generation configuration."""
        
        self.generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            num_beams=self.config.num_beams,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            length_penalty=1.0,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True
        )
    
    def reconstruct(
        self,
        phoneme_sequences: List[List[str]],
        return_attention: bool = False
    ) -> List[ReconstructionResult]:
        """
        Reconstruct sentences from phoneme sequences.
        
        Args:
            phoneme_sequences: List of phoneme sequences
            return_attention: Whether to return attention weights
            
        Returns:
            List of reconstruction results
        """
        if not phoneme_sequences:
            return []
        
        # Process in batches
        all_results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(phoneme_sequences), batch_size):
            batch = phoneme_sequences[i:i + batch_size]
            batch_results = self._reconstruct_batch(batch, return_attention)
            all_results.extend(batch_results)
        
        return all_results
    
    def _reconstruct_batch(
        self,
        phoneme_sequences: List[List[str]],
        return_attention: bool = False
    ) -> List[ReconstructionResult]:
        """Reconstruct batch of phoneme sequences."""
        
        # Prepare input prompts
        prompts = []
        for phonemes in phoneme_sequences:
            prompt = self._create_prompt(phonemes)
            prompts.append(prompt)
        
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    output_attentions=return_attention,
                    use_cache=True
                )
                
                # Extract generated sequences and scores
                generated_sequences = outputs.sequences
                scores = outputs.scores if hasattr(outputs, 'scores') else None
                attentions = outputs.attentions if return_attention and hasattr(outputs, 'attentions') else None
                
                # Process results
                results = []
                for i, (phonemes, generated_seq) in enumerate(zip(phoneme_sequences, generated_sequences)):
                    
                    # Decode generated text
                    input_length = inputs.input_ids[i].shape[0]
                    generated_tokens = generated_seq[input_length:]  # Remove input tokens
                    
                    generated_text = self.tokenizer.decode(
                        generated_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    # Post-process text
                    processed_text = self._post_process_text(generated_text)
                    
                    # Calculate confidence
                    confidence = self._calculate_confidence(
                        generated_tokens, scores, i if scores else None
                    )
                    
                    # Calculate log probability
                    log_prob = self._calculate_log_probability(
                        generated_tokens, scores, i if scores else None
                    )
                    
                    # Extract attention weights if requested
                    attention_weights = None
                    if return_attention and attentions:
                        attention_weights = self._extract_attention_weights(attentions, i)
                    
                    # Create result
                    result = ReconstructionResult(
                        text=processed_text,
                        confidence=confidence,
                        log_prob=log_prob,
                        phoneme_input=phonemes,
                        tokens=self.tokenizer.convert_ids_to_tokens(generated_tokens),
                        attention_weights=attention_weights
                    )
                    
                    results.append(result)
                
                return results
                
            except Exception as e:
                logger.error(f"❌ Generation failed: {e}")
                # Return empty results
                return [
                    ReconstructionResult(
                        text="",
                        confidence=0.0,
                        log_prob=float('-inf'),
                        phoneme_input=phonemes,
                        tokens=[],
                        attention_weights=None
                    )
                    for phonemes in phoneme_sequences
                ]
    
    def _create_prompt(self, phonemes: List[str]) -> str:
        """Create input prompt from phonemes."""
        
        # Clean phonemes
        cleaned_phonemes = [p for p in phonemes if p in self.phoneme_vocab.values()]
        
        # Join phonemes
        phoneme_text = self.config.phoneme_separator.join(cleaned_phonemes)
        
        # Create prompt
        prompt = f"""Convert the following phoneme sequence to natural English text:

Phonemes: {phoneme_text}
Text: """
        
        return prompt
    
    def _post_process_text(self, text: str) -> str:
        """Post-process generated text."""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special tokens
        text = text.replace(self.config.sentence_start_token, "")
        text = text.replace(self.config.sentence_end_token, "")
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        # Add period if missing
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def _calculate_confidence(
        self,
        generated_tokens: torch.Tensor,
        scores: Optional[Tuple[torch.Tensor]],
        batch_idx: Optional[int]
    ) -> float:
        """Calculate confidence score for generated sequence."""
        
        if scores is None or len(generated_tokens) == 0:
            return self.config.min_confidence
        
        try:
            if self.config.confidence_method == "token_probs":
                # Average token probabilities
                token_probs = []
                for i, token_id in enumerate(generated_tokens):
                    if i < len(scores):
                        step_scores = scores[i]
                        if batch_idx is not None and batch_idx < step_scores.shape[0]:
                            token_prob = F.softmax(step_scores[batch_idx], dim=-1)[token_id]
                            token_probs.append(token_prob.item())
                
                confidence = np.mean(token_probs) if token_probs else self.config.min_confidence
                
            elif self.config.confidence_method == "sequence_prob":
                # Overall sequence probability
                log_probs = []
                for i, token_id in enumerate(generated_tokens):
                    if i < len(scores):
                        step_scores = scores[i]
                        if batch_idx is not None and batch_idx < step_scores.shape[0]:
                            log_prob = F.log_softmax(step_scores[batch_idx], dim=-1)[token_id]
                            log_probs.append(log_prob.item())
                
                avg_log_prob = np.mean(log_probs) if log_probs else float('-inf')
                confidence = np.exp(avg_log_prob)
                
            elif self.config.confidence_method == "entropy":
                # Entropy-based confidence
                entropies = []
                for i, token_id in enumerate(generated_tokens):
                    if i < len(scores):
                        step_scores = scores[i]
                        if batch_idx is not None and batch_idx < step_scores.shape[0]:
                            probs = F.softmax(step_scores[batch_idx], dim=-1)
                            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                            entropies.append(entropy.item())
                
                avg_entropy = np.mean(entropies) if entropies else 10.0
                confidence = 1.0 / (1.0 + avg_entropy)
            
            else:
                confidence = self.config.min_confidence
            
            return float(max(confidence, self.config.min_confidence))
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return self.config.min_confidence
    
    def _calculate_log_probability(
        self,
        generated_tokens: torch.Tensor,
        scores: Optional[Tuple[torch.Tensor]],
        batch_idx: Optional[int]
    ) -> float:
        """Calculate log probability of generated sequence."""
        
        if scores is None or len(generated_tokens) == 0:
            return float('-inf')
        
        try:
            log_probs = []
            for i, token_id in enumerate(generated_tokens):
                if i < len(scores):
                    step_scores = scores[i]
                    if batch_idx is not None and batch_idx < step_scores.shape[0]:
                        log_prob = F.log_softmax(step_scores[batch_idx], dim=-1)[token_id]
                        log_probs.append(log_prob.item())
            
            return sum(log_probs) if log_probs else float('-inf')
            
        except Exception as e:
            logger.warning(f"Log probability calculation failed: {e}")
            return float('-inf')
    
    def _extract_attention_weights(
        self,
        attentions: Tuple[torch.Tensor],
        batch_idx: int
    ) -> Optional[torch.Tensor]:
        """Extract attention weights for visualization."""
        
        try:
            # Average attention across layers and heads
            if attentions and len(attentions) > 0:
                # Get last layer attention
                last_layer_attention = attentions[-1]  # [batch, heads, seq_len, seq_len]
                
                if batch_idx < last_layer_attention.shape[0]:
                    # Average across heads
                    attention = last_layer_attention[batch_idx].mean(dim=0)  # [seq_len, seq_len]
                    return attention.cpu()
            
            return None
            
        except Exception as e:
            logger.warning(f"Attention extraction failed: {e}")
            return None
    
    def reconstruct_single(
        self,
        phonemes: List[str],
        return_attention: bool = False
    ) -> ReconstructionResult:
        """Reconstruct single phoneme sequence."""
        
        results = self.reconstruct([phonemes], return_attention)
        return results[0] if results else ReconstructionResult(
            text="",
            confidence=0.0,
            log_prob=float('-inf'),
            phoneme_input=phonemes,
            tokens=[],
            attention_weights=None
        )
    
    def fine_tune(
        self,
        training_data: List[Tuple[List[str], str]],
        validation_data: Optional[List[Tuple[List[str], str]]] = None,
        num_epochs: int = 3,
        learning_rate: float = 1e-4,
        save_path: Optional[str] = None
    ):
        """
        Fine-tune the model on phoneme-to-text pairs.
        
        Args:
            training_data: List of (phonemes, text) pairs
            validation_data: Optional validation data
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            save_path: Path to save fine-tuned model
        """
        
        logger.info(f"🔧 Starting fine-tuning with {len(training_data)} samples")
        
        # This would implement the fine-tuning logic
        # For now, we'll log that it's not implemented
        logger.warning("⚠️ Fine-tuning not implemented yet. Using pre-trained model.")
        
        # TODO: Implement fine-tuning with LoRA
        # 1. Prepare training dataset
        # 2. Setup optimizer and scheduler
        # 3. Training loop with LoRA updates
        # 4. Validation and checkpointing
        # 5. Save final model
    
    def load_fine_tuned_model(self, model_path: str):
        """Load fine-tuned LoRA weights."""
        
        try:
            if self.config.use_lora:
                self.model = PeftModel.from_pretrained(self.model, model_path)
                logger.info(f"✅ Loaded fine-tuned LoRA weights from {model_path}")
            else:
                logger.warning("⚠️ LoRA not enabled, cannot load fine-tuned weights")
                
        except Exception as e:
            logger.error(f"❌ Failed to load fine-tuned model: {e}")
    
    def save_fine_tuned_model(self, save_path: str):
        """Save fine-tuned LoRA weights."""
        
        try:
            if self.config.use_lora and hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(save_path)
                logger.info(f"✅ Saved fine-tuned LoRA weights to {save_path}")
            else:
                logger.warning("⚠️ No LoRA model to save")
                
        except Exception as e:
            logger.error(f"❌ Failed to save fine-tuned model: {e}")
