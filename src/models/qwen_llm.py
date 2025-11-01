"""
Qwen3-0.6B LLM integration for Valerie Visual ASR.

Implements phoneme-to-sentence reconstruction using Qwen3-0.6B with LoRA fine-tuning
and thinking mode support for enhanced reasoning capabilities.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Dict, Optional, Tuple, Union
from src.utils.logging import get_logger

logger = get_logger(__name__)


class QwenPhonemeToText(nn.Module):
    """
    Qwen3-0.6B based phoneme-to-sentence reconstruction module.
    
    Features:
    - LoRA fine-tuning for efficiency
    - Thinking mode support for complex reasoning
    - Phoneme-aware attention mechanisms
    - Confidence-weighted input processing
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        enable_thinking: bool = True,
        max_new_tokens: int = 32768,
        quantization_config: Optional[Dict] = None,
        device_map: str = "auto"
    ):
        """
        Initialize Qwen3 LLM for phoneme-to-text conversion.
        
        Args:
            model_name: Hugging Face model identifier
            lora_rank: LoRA rank for fine-tuning
            lora_alpha: LoRA alpha parameter
            enable_thinking: Whether to enable Qwen3 thinking mode
            max_new_tokens: Maximum tokens to generate
            quantization_config: Optional quantization configuration
            device_map: Device mapping strategy
        """
        super().__init__()
        
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        
        logger.info(f"🤖 Initializing Qwen3 LLM: {model_name}")
        
        # Setup quantization if specified
        self.quantization_config = None
        if quantization_config:
            self.quantization_config = BitsAndBytesConfig(**quantization_config)
            logger.info("⚡ Using quantization for memory efficiency")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"  # Important for batch generation
        )
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info(f"📝 Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device_map,
            quantization_config=self.quantization_config,
            trust_remote_code=True
        )
        
        # Setup LoRA for fine-tuning
        self._setup_lora(lora_rank, lora_alpha)
        
        # Setup generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.6 if enable_thinking else 0.7,
            top_p=0.95 if enable_thinking else 0.8,
            top_k=20,
            min_p=0.0,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        logger.info("✅ Qwen3 LLM initialization completed")
    
    def _setup_lora(self, lora_rank: int, lora_alpha: int) -> None:
        """Setup LoRA configuration for efficient fine-tuning."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"🔧 LoRA setup complete:")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    def create_phoneme_prompt(
        self,
        phonemes: List[str],
        confidence_scores: Optional[List[float]] = None,
        context: Optional[str] = None
    ) -> str:
        """
        Create a structured prompt for phoneme-to-text conversion.
        
        Args:
            phonemes: List of phoneme strings
            confidence_scores: Optional confidence scores for each phoneme
            context: Optional context information
            
        Returns:
            Formatted prompt string
        """
        prompt = "Convert the following phoneme sequence to natural English text.\n\n"
        
        if context:
            prompt += f"Context: {context}\n\n"
        
        prompt += "Phonemes: "
        
        if confidence_scores and len(confidence_scores) == len(phonemes):
            # Include confidence information
            phoneme_parts = []
            for phoneme, conf in zip(phonemes, confidence_scores):
                if conf < 0.5:
                    phoneme_parts.append(f"[{phoneme}?]")  # Low confidence
                elif conf < 0.8:
                    phoneme_parts.append(f"({phoneme})")   # Medium confidence
                else:
                    phoneme_parts.append(phoneme)          # High confidence
            prompt += " ".join(phoneme_parts)
        else:
            prompt += " ".join(phonemes)
        
        prompt += "\n\nText:"
        
        return prompt
    
    def forward(
        self,
        phoneme_sequences: List[List[str]],
        confidence_scores: Optional[List[List[float]]] = None,
        contexts: Optional[List[str]] = None,
        return_thinking: bool = False
    ) -> Union[List[str], List[Tuple[str, str]]]:
        """
        Convert phoneme sequences to text using Qwen3.
        
        Args:
            phoneme_sequences: Batch of phoneme sequences
            confidence_scores: Optional confidence scores for phonemes
            contexts: Optional context information for each sequence
            return_thinking: Whether to return thinking content
            
        Returns:
            Generated text sequences, optionally with thinking content
        """
        batch_size = len(phoneme_sequences)
        
        # Create prompts
        prompts = []
        for i in range(batch_size):
            conf_scores = confidence_scores[i] if confidence_scores else None
            context = contexts[i] if contexts else None
            
            prompt = self.create_phoneme_prompt(
                phoneme_sequences[i],
                conf_scores,
                context
            )
            prompts.append(prompt)
        
        # Prepare messages for chat template
        messages_batch = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            messages_batch.append(messages)
        
        # Apply chat template
        texts = []
        for messages in messages_batch:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )
            texts.append(text)
        
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode outputs
        results = []
        for i, output in enumerate(outputs):
            # Remove input tokens
            input_length = inputs.input_ids[i].shape[0]
            generated_tokens = output[input_length:].tolist()
            
            if return_thinking and self.enable_thinking:
                # Parse thinking and content
                thinking_content, final_content = self._parse_thinking_output(generated_tokens)
                results.append((final_content, thinking_content))
            else:
                # Just return final content
                content = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                results.append(content.strip())
        
        return results
    
    def _parse_thinking_output(self, token_ids: List[int]) -> Tuple[str, str]:
        """
        Parse Qwen3 thinking output to separate thinking and final content.
        
        Args:
            token_ids: Generated token IDs
            
        Returns:
            Tuple of (final_content, thinking_content)
        """
        try:
            # Find the </think> token (ID 151668 according to Qwen3 docs)
            think_end_id = 151668
            if think_end_id in token_ids:
                think_end_idx = len(token_ids) - token_ids[::-1].index(think_end_id)
                
                thinking_tokens = token_ids[:think_end_idx]
                content_tokens = token_ids[think_end_idx:]
                
                thinking_content = self.tokenizer.decode(thinking_tokens, skip_special_tokens=True).strip()
                final_content = self.tokenizer.decode(content_tokens, skip_special_tokens=True).strip()
            else:
                # No thinking content found
                thinking_content = ""
                final_content = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                
        except (ValueError, IndexError):
            # Fallback if parsing fails
            thinking_content = ""
            final_content = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        
        return final_content, thinking_content
    
    def fine_tune_step(
        self,
        phoneme_sequences: List[List[str]],
        target_texts: List[str],
        confidence_scores: Optional[List[List[float]]] = None
    ) -> torch.Tensor:
        """
        Perform a single fine-tuning step.
        
        Args:
            phoneme_sequences: Batch of phoneme sequences
            target_texts: Target text sequences
            confidence_scores: Optional confidence scores
            
        Returns:
            Loss tensor
        """
        batch_size = len(phoneme_sequences)
        
        # Create training examples
        input_texts = []
        target_texts_formatted = []
        
        for i in range(batch_size):
            conf_scores = confidence_scores[i] if confidence_scores else None
            prompt = self.create_phoneme_prompt(phoneme_sequences[i], conf_scores)
            
            input_texts.append(prompt)
            target_texts_formatted.append(target_texts[i])
        
        # Prepare training data
        messages_batch = []
        for i in range(batch_size):
            messages = [
                {"role": "user", "content": input_texts[i]},
                {"role": "assistant", "content": target_texts_formatted[i]}
            ]
            messages_batch.append(messages)
        
        # Apply chat template and tokenize
        full_texts = []
        for messages in messages_batch:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=self.enable_thinking
            )
            full_texts.append(text)
        
        # Tokenize
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.model.device)
        
        # Forward pass
        outputs = self.model(**inputs, labels=inputs.input_ids)
        
        return outputs.loss
    
    def save_model(self, save_path: str) -> None:
        """Save the fine-tuned LoRA adapter."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"💾 Model saved to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """Load a fine-tuned LoRA adapter."""
        from peft import PeftModel
        
        self.model = PeftModel.from_pretrained(self.model, load_path)
        logger.info(f"📂 Model loaded from {load_path}")


def create_qwen_llm(config) -> QwenPhonemeToText:
    """
    Factory function to create Qwen3 LLM from configuration.
    
    Args:
        config: Model configuration object
        
    Returns:
        Initialized QwenPhonemeToText instance
    """
    quantization_config = None
    
    # Setup quantization if needed for memory efficiency
    if hasattr(config, 'use_quantization') and config.use_quantization:
        quantization_config = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        }
        logger.info("🔧 Using 4-bit quantization for memory efficiency")
    
    return QwenPhonemeToText(
        model_name=config.llm_model_name,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        enable_thinking=getattr(config, 'enable_thinking', True),
        max_new_tokens=getattr(config, 'max_new_tokens', 32768),
        quantization_config=quantization_config
    )


# Example usage and testing
if __name__ == "__main__":
    from src.utils.config import ModelConfig
    
    # Test configuration
    config = ModelConfig()
    config.llm_model_name = "Qwen/Qwen3-0.6B"
    config.lora_rank = 8  # Smaller for testing
    config.lora_alpha = 16
    
    # Create model
    qwen_llm = create_qwen_llm(config)
    
    # Test inference
    test_phonemes = [["HH", "EH", "L", "OW", "W", "ER", "L", "D"]]  # "hello world"
    test_confidence = [[0.9, 0.8, 0.95, 0.7, 0.85, 0.9, 0.8, 0.9]]
    
    results = qwen_llm(test_phonemes, test_confidence, return_thinking=True)
    
    print("Generated text:", results[0][0])
    if results[0][1]:
        print("Thinking process:", results[0][1])
