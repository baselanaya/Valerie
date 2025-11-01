"""
Evaluation metrics for Valerie Visual ASR.

Implements phoneme error rate (PER), word error rate (WER), 
and other metrics for visual speech recognition evaluation.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import editdistance
from collections import Counter
import re
from src.data.phoneme_utils import IDX_TO_PHONEME
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PhonemeErrorRate:
    """
    Phoneme Error Rate (PER) calculation.
    
    Computes character-level edit distance between predicted and target phoneme sequences.
    """
    
    def __init__(self, ignore_tokens: Optional[List[str]] = None):
        """
        Initialize PER calculator.
        
        Args:
            ignore_tokens: Tokens to ignore in calculation (e.g., ['<blank>', '<pad>'])
        """
        self.ignore_tokens = ignore_tokens or ['<blank>', '<pad>', '<unk>']
        self.total_phonemes = 0
        self.total_errors = 0
    
    def reset(self):
        """Reset accumulated statistics."""
        self.total_phonemes = 0
        self.total_errors = 0
    
    def update(
        self,
        predictions: List[List[str]],
        targets: List[List[str]]
    ):
        """
        Update PER with batch of predictions and targets.
        
        Args:
            predictions: List of predicted phoneme sequences
            targets: List of target phoneme sequences
        """
        assert len(predictions) == len(targets), "Predictions and targets must have same length"
        
        for pred_seq, target_seq in zip(predictions, targets):
            # Filter out ignore tokens
            pred_clean = [p for p in pred_seq if p not in self.ignore_tokens]
            target_clean = [t for t in target_seq if t not in self.ignore_tokens]
            
            # Compute edit distance
            if len(target_clean) > 0:
                errors = editdistance.eval(pred_clean, target_clean)
                self.total_errors += errors
                self.total_phonemes += len(target_clean)
    
    def compute(self) -> float:
        """Compute current PER."""
        if self.total_phonemes == 0:
            return 0.0
        return self.total_errors / self.total_phonemes
    
    def compute_batch(
        self,
        predictions: List[List[str]], 
        targets: List[List[str]]
    ) -> float:
        """Compute PER for a single batch."""
        batch_errors = 0
        batch_phonemes = 0
        
        for pred_seq, target_seq in zip(predictions, targets):
            pred_clean = [p for p in pred_seq if p not in self.ignore_tokens]
            target_clean = [t for t in target_seq if t not in self.ignore_tokens]
            
            if len(target_clean) > 0:
                errors = editdistance.eval(pred_clean, target_clean)
                batch_errors += errors
                batch_phonemes += len(target_clean)
        
        return batch_errors / batch_phonemes if batch_phonemes > 0 else 0.0


class WordErrorRate:
    """
    Word Error Rate (WER) calculation.
    
    Computes word-level edit distance between predicted and target sentences.
    """
    
    def __init__(self, phoneme_to_word_converter=None):
        """
        Initialize WER calculator.
        
        Args:
            phoneme_to_word_converter: Function to convert phonemes to words
        """
        self.phoneme_to_word_converter = phoneme_to_word_converter
        self.total_words = 0
        self.total_errors = 0
    
    def reset(self):
        """Reset accumulated statistics."""
        self.total_words = 0
        self.total_errors = 0
    
    def _phonemes_to_words(self, phonemes: List[str]) -> List[str]:
        """Convert phoneme sequence to word sequence."""
        if self.phoneme_to_word_converter:
            return self.phoneme_to_word_converter(phonemes)
        else:
            # Simple fallback: split on silence tokens
            phoneme_str = ' '.join(phonemes)
            # Split on silence markers
            words = re.split(r'\s*<sil>\s*', phoneme_str)
            words = [w.strip() for w in words if w.strip()]
            return words
    
    def update(
        self,
        predictions: List[List[str]],
        targets: List[List[str]]
    ):
        """
        Update WER with batch of predictions and targets.
        
        Args:
            predictions: List of predicted phoneme sequences
            targets: List of target phoneme sequences
        """
        for pred_seq, target_seq in zip(predictions, targets):
            # Convert phonemes to words
            pred_words = self._phonemes_to_words(pred_seq)
            target_words = self._phonemes_to_words(target_seq)
            
            # Compute word-level edit distance
            if len(target_words) > 0:
                errors = editdistance.eval(pred_words, target_words)
                self.total_errors += errors
                self.total_words += len(target_words)
    
    def compute(self) -> float:
        """Compute current WER."""
        if self.total_words == 0:
            return 0.0
        return self.total_errors / self.total_words
    
    def compute_batch(
        self,
        predictions: List[List[str]],
        targets: List[List[str]]
    ) -> float:
        """Compute WER for a single batch."""
        batch_errors = 0
        batch_words = 0
        
        for pred_seq, target_seq in zip(predictions, targets):
            pred_words = self._phonemes_to_words(pred_seq)
            target_words = self._phonemes_to_words(target_seq)
            
            if len(target_words) > 0:
                errors = editdistance.eval(pred_words, target_words)
                batch_errors += errors
                batch_words += len(target_words)
        
        return batch_errors / batch_words if batch_words > 0 else 0.0


class CTCDecoder:
    """
    CTC decoder for converting logits to phoneme sequences.
    
    Implements greedy decoding and basic beam search for CTC outputs.
    """
    
    def __init__(
        self,
        blank_token_id: int = 0,
        vocabulary: Optional[Dict[int, str]] = None
    ):
        """
        Initialize CTC decoder.
        
        Args:
            blank_token_id: ID of blank token
            vocabulary: Mapping from token IDs to phonemes
        """
        self.blank_token_id = blank_token_id
        self.vocabulary = vocabulary or IDX_TO_PHONEME
    
    def greedy_decode(
        self,
        logits: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> List[List[str]]:
        """
        Greedy CTC decoding.
        
        Args:
            logits: CTC logits [B, T, V]
            lengths: Valid sequence lengths [B]
            
        Returns:
            List of decoded phoneme sequences
        """
        batch_size, max_time, vocab_size = logits.shape
        
        # Get best predictions
        predictions = torch.argmax(logits, dim=-1)  # [B, T]
        
        decoded_sequences = []
        
        for b in range(batch_size):
            # Get valid sequence length
            seq_len = lengths[b].item() if lengths is not None else max_time
            pred_seq = predictions[b, :seq_len].cpu().numpy()
            
            # CTC collapse: remove blanks and consecutive duplicates
            decoded = []
            prev_token = None
            
            for token_id in pred_seq:
                if token_id != self.blank_token_id and token_id != prev_token:
                    phoneme = self.vocabulary.get(token_id, '<unk>')
                    decoded.append(phoneme)
                prev_token = token_id
            
            decoded_sequences.append(decoded)
        
        return decoded_sequences
    
    def beam_search_decode(
        self,
        logits: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        beam_width: int = 5
    ) -> List[List[str]]:
        """
        Beam search CTC decoding.
        
        Args:
            logits: CTC logits [B, T, V]
            lengths: Valid sequence lengths [B]
            beam_width: Number of beams to keep
            
        Returns:
            List of decoded phoneme sequences
        """
        # For now, implement simple greedy decoding
        # TODO: Implement full beam search with language model
        return self.greedy_decode(logits, lengths)


class ValerieMetrics:
    """
    Complete metrics suite for Valerie Visual ASR.
    
    Combines PER, WER, and other metrics for comprehensive evaluation.
    """
    
    def __init__(self, blank_token_id: int = 0):
        """
        Initialize metrics suite.
        
        Args:
            blank_token_id: Blank token ID for CTC decoding
        """
        self.per_calculator = PhonemeErrorRate()
        self.wer_calculator = WordErrorRate()
        self.ctc_decoder = CTCDecoder(blank_token_id=blank_token_id)
        
        # Additional metrics
        self.accuracy_sum = 0.0
        self.accuracy_count = 0
        
        logger.info("✅ ValerieMetrics initialized")
    
    def reset(self):
        """Reset all metrics."""
        self.per_calculator.reset()
        self.wer_calculator.reset()
        self.accuracy_sum = 0.0
        self.accuracy_count = 0
    
    def update(
        self,
        ctc_logits: torch.Tensor,
        attention_logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ):
        """
        Update metrics with model outputs.
        
        Args:
            ctc_logits: CTC predictions [B, T, V]
            attention_logits: Attention predictions [B, S, V]
            targets: Target phoneme sequences [B, S]
            input_lengths: Input sequence lengths [B]
            target_lengths: Target sequence lengths [B]
        """
        batch_size = ctc_logits.size(0)
        
        # Decode CTC outputs
        ctc_predictions = self.ctc_decoder.greedy_decode(ctc_logits, input_lengths)
        
        # Decode attention outputs
        attention_pred_ids = torch.argmax(attention_logits, dim=-1)
        attention_predictions = []
        
        for b in range(batch_size):
            target_len = target_lengths[b].item()
            pred_ids = attention_pred_ids[b, :target_len].cpu().numpy()
            pred_phonemes = [self.ctc_decoder.vocabulary.get(id, '<unk>') for id in pred_ids]
            attention_predictions.append(pred_phonemes)
        
        # Convert targets to phonemes
        target_sequences = []
        for b in range(batch_size):
            target_len = target_lengths[b].item()
            target_ids = targets[b, :target_len].cpu().numpy()
            target_phonemes = [self.ctc_decoder.vocabulary.get(id, '<unk>') for id in target_ids]
            target_sequences.append(target_phonemes)
        
        # Update PER (using attention predictions)
        self.per_calculator.update(attention_predictions, target_sequences)
        
        # Update WER (using attention predictions)
        self.wer_calculator.update(attention_predictions, target_sequences)
        
        # Update accuracy (token-level)
        correct_tokens = 0
        total_tokens = 0
        
        for b in range(batch_size):
            target_len = target_lengths[b].item()
            pred_ids = attention_pred_ids[b, :target_len]
            target_ids = targets[b, :target_len]
            
            correct_tokens += (pred_ids == target_ids).sum().item()
            total_tokens += target_len
        
        if total_tokens > 0:
            self.accuracy_sum += correct_tokens
            self.accuracy_count += total_tokens
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        metrics = {
            'per': self.per_calculator.compute(),
            'wer': self.wer_calculator.compute(),
            'accuracy': self.accuracy_sum / self.accuracy_count if self.accuracy_count > 0 else 0.0
        }
        
        return metrics
    
    def compute_batch_metrics(
        self,
        ctc_logits: torch.Tensor,
        attention_logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> Dict[str, float]:
        """Compute metrics for a single batch."""
        batch_size = ctc_logits.size(0)
        
        # Decode predictions
        ctc_predictions = self.ctc_decoder.greedy_decode(ctc_logits, input_lengths)
        
        attention_pred_ids = torch.argmax(attention_logits, dim=-1)
        attention_predictions = []
        target_sequences = []
        
        for b in range(batch_size):
            # Attention predictions
            target_len = target_lengths[b].item()
            pred_ids = attention_pred_ids[b, :target_len].cpu().numpy()
            pred_phonemes = [self.ctc_decoder.vocabulary.get(id, '<unk>') for id in pred_ids]
            attention_predictions.append(pred_phonemes)
            
            # Target sequences
            target_ids = targets[b, :target_len].cpu().numpy()
            target_phonemes = [self.ctc_decoder.vocabulary.get(id, '<unk>') for id in target_ids]
            target_sequences.append(target_phonemes)
        
        # Compute batch metrics
        batch_per = self.per_calculator.compute_batch(attention_predictions, target_sequences)
        batch_wer = self.wer_calculator.compute_batch(attention_predictions, target_sequences)
        
        # Compute batch accuracy
        correct_tokens = 0
        total_tokens = 0
        
        for b in range(batch_size):
            target_len = target_lengths[b].item()
            pred_ids = attention_pred_ids[b, :target_len]
            target_ids = targets[b, :target_len]
            
            correct_tokens += (pred_ids == target_ids).sum().item()
            total_tokens += target_len
        
        batch_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        
        return {
            'per': batch_per,
            'wer': batch_wer,
            'accuracy': batch_accuracy
        }
