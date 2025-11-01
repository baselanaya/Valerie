"""
CTC Beam Search Decoder for Valerie Visual ASR.

Implements efficient beam search decoding with:
- Phoneme language model integration
- Confidence scoring for predictions
- Batch decoding for multiple sequences
- Greedy decoding as fallback
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import heapq
from collections import defaultdict
import logging

from src.data.phoneme_utils import PHONEME_TO_IDX, IDX_TO_PHONEME
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DecodingConfig:
    """Configuration for CTC decoding."""
    
    # Beam search parameters
    beam_width: int = 100
    max_length: int = 1000
    length_penalty: float = 1.0
    
    # Language model integration
    use_language_model: bool = False
    lm_weight: float = 0.3
    word_score: float = 0.0
    
    # Confidence scoring
    confidence_method: str = "max_prob"  # max_prob, mean_prob, entropy
    min_confidence: float = 0.1
    
    # Batch processing
    batch_size: int = 1
    
    # Special tokens
    blank_token_id: int = 0
    sos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


class BeamSearchResult(NamedTuple):
    """Result from beam search decoding."""
    
    sequence: List[int]          # Token IDs
    phonemes: List[str]         # Phoneme strings
    log_prob: float             # Log probability
    confidence: float           # Confidence score
    alignment: List[int]        # Frame-to-token alignment
    length: int                 # Sequence length


class BeamHypothesis:
    """Individual hypothesis in beam search."""
    
    def __init__(
        self,
        sequence: List[int],
        log_prob: float,
        alignment: List[int],
        last_token: int = -1
    ):
        self.sequence = sequence
        self.log_prob = log_prob
        self.alignment = alignment
        self.last_token = last_token
        self.length = len(sequence)
    
    def extend(self, token_id: int, token_log_prob: float, frame_idx: int) -> 'BeamHypothesis':
        """Extend hypothesis with new token."""
        new_sequence = self.sequence + [token_id]
        new_log_prob = self.log_prob + token_log_prob
        new_alignment = self.alignment + [frame_idx]
        
        return BeamHypothesis(
            sequence=new_sequence,
            log_prob=new_log_prob,
            alignment=new_alignment,
            last_token=token_id
        )
    
    def __lt__(self, other):
        """Compare hypotheses by normalized log probability."""
        return self.normalized_score() < other.normalized_score()
    
    def normalized_score(self, length_penalty: float = 1.0) -> float:
        """Compute length-normalized score."""
        if self.length == 0:
            return float('-inf')
        return self.log_prob / (self.length ** length_penalty)


class GreedyDecoder:
    """Simple greedy CTC decoder."""
    
    def __init__(self, config: DecodingConfig):
        self.config = config
        self.blank_token_id = config.blank_token_id
        self.vocabulary = IDX_TO_PHONEME
    
    def decode(
        self,
        logits: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> List[BeamSearchResult]:
        """
        Greedy decode CTC outputs.
        
        Args:
            logits: CTC logits [B, T, V]
            lengths: Valid sequence lengths [B]
            
        Returns:
            List of decoding results
        """
        batch_size, max_time, vocab_size = logits.shape
        
        # Get best predictions
        predictions = torch.argmax(logits, dim=-1)  # [B, T]
        log_probs = F.log_softmax(logits, dim=-1)
        
        results = []
        
        for b in range(batch_size):
            # Get valid sequence length
            seq_len = lengths[b].item() if lengths is not None else max_time
            pred_seq = predictions[b, :seq_len].cpu().numpy()
            prob_seq = log_probs[b, :seq_len].cpu().numpy()
            
            # CTC collapse: remove blanks and consecutive duplicates
            decoded_tokens = []
            decoded_probs = []
            alignment = []
            prev_token = None
            
            for t, (token_id, frame_probs) in enumerate(zip(pred_seq, prob_seq)):
                if token_id != self.blank_token_id and token_id != prev_token:
                    decoded_tokens.append(token_id)
                    decoded_probs.append(frame_probs[token_id])
                    alignment.append(t)
                prev_token = token_id
            
            # Convert to phonemes
            phonemes = [self.vocabulary.get(token_id, '<unk>') for token_id in decoded_tokens]
            
            # Calculate confidence
            if decoded_probs:
                total_log_prob = sum(decoded_probs)
                confidence = self._calculate_confidence(decoded_probs)
            else:
                total_log_prob = float('-inf')
                confidence = 0.0
            
            result = BeamSearchResult(
                sequence=decoded_tokens,
                phonemes=phonemes,
                log_prob=total_log_prob,
                confidence=confidence,
                alignment=alignment,
                length=len(decoded_tokens)
            )
            
            results.append(result)
        
        return results
    
    def _calculate_confidence(self, log_probs: List[float]) -> float:
        """Calculate confidence score from log probabilities."""
        if not log_probs:
            return 0.0
        
        if self.config.confidence_method == "max_prob":
            return float(np.exp(max(log_probs)))
        elif self.config.confidence_method == "mean_prob":
            return float(np.exp(np.mean(log_probs)))
        elif self.config.confidence_method == "entropy":
            probs = np.exp(log_probs)
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            return float(1.0 / (1.0 + entropy))
        else:
            return float(np.exp(np.mean(log_probs)))


class CTCBeamSearchDecoder:
    """
    CTC Beam Search Decoder with language model integration.
    
    Implements efficient beam search with:
    - Prefix beam search algorithm
    - Optional phoneme language model
    - Confidence scoring
    - Batch processing
    """
    
    def __init__(self, config: DecodingConfig, language_model=None):
        self.config = config
        self.language_model = language_model
        self.blank_token_id = config.blank_token_id
        self.vocabulary = IDX_TO_PHONEME
        self.vocab_size = len(self.vocabulary)
        
        # Initialize greedy decoder as fallback
        self.greedy_decoder = GreedyDecoder(config)
        
        logger.info(f"✅ CTCBeamSearchDecoder initialized:")
        logger.info(f"   Beam width: {config.beam_width}")
        logger.info(f"   Max length: {config.max_length}")
        logger.info(f"   Language model: {config.use_language_model}")
        logger.info(f"   Confidence method: {config.confidence_method}")
    
    def decode(
        self,
        logits: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> List[BeamSearchResult]:
        """
        Beam search decode CTC outputs.
        
        Args:
            logits: CTC logits [B, T, V]
            lengths: Valid sequence lengths [B]
            
        Returns:
            List of best decoding results
        """
        batch_size = logits.size(0)
        results = []
        
        for b in range(batch_size):
            # Extract single sequence
            seq_logits = logits[b]  # [T, V]
            seq_length = lengths[b].item() if lengths is not None else seq_logits.size(0)
            seq_logits = seq_logits[:seq_length]
            
            # Perform beam search
            try:
                result = self._beam_search_single(seq_logits)
                results.append(result)
            except Exception as e:
                logger.warning(f"Beam search failed for sequence {b}: {e}")
                # Fallback to greedy decoding
                greedy_results = self.greedy_decoder.decode(
                    logits[b:b+1].unsqueeze(0),
                    lengths[b:b+1] if lengths is not None else None
                )
                results.append(greedy_results[0])
        
        return results
    
    def _beam_search_single(self, logits: torch.Tensor) -> BeamSearchResult:
        """Beam search for single sequence."""
        
        seq_length, vocab_size = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)  # [T, V]
        
        # Initialize beam with empty hypothesis
        beam = [BeamHypothesis(sequence=[], log_prob=0.0, alignment=[])]
        
        for t in range(seq_length):
            frame_log_probs = log_probs[t]  # [V]
            new_beam = []
            
            # Extend each hypothesis in current beam
            for hyp in beam:
                
                # Get top-k tokens for this frame
                top_k_log_probs, top_k_tokens = torch.topk(
                    frame_log_probs, 
                    min(self.config.beam_width, vocab_size)
                )
                
                for token_log_prob, token_id in zip(top_k_log_probs, top_k_tokens):
                    token_id = token_id.item()
                    token_log_prob = token_log_prob.item()
                    
                    if token_id == self.blank_token_id:
                        # Blank token - don't extend sequence
                        new_hyp = BeamHypothesis(
                            sequence=hyp.sequence,
                            log_prob=hyp.log_prob + token_log_prob,
                            alignment=hyp.alignment,
                            last_token=hyp.last_token
                        )
                    else:
                        # Non-blank token
                        if token_id == hyp.last_token:
                            # Same as previous token - CTC collapse
                            new_hyp = BeamHypothesis(
                                sequence=hyp.sequence,
                                log_prob=hyp.log_prob + token_log_prob,
                                alignment=hyp.alignment,
                                last_token=token_id
                            )
                        else:
                            # New token - extend sequence
                            new_hyp = hyp.extend(token_id, token_log_prob, t)
                            
                            # Apply language model score if available
                            if self.config.use_language_model and self.language_model:
                                lm_score = self._get_language_model_score(new_hyp.sequence)
                                new_hyp.log_prob += self.config.lm_weight * lm_score
                    
                    new_beam.append(new_hyp)
            
            # Prune beam to keep only top hypotheses
            beam = self._prune_beam(new_beam)
            
            # Early stopping if max length reached
            if beam and max(hyp.length for hyp in beam) >= self.config.max_length:
                break
        
        # Select best hypothesis
        if not beam:
            # Empty beam - return empty result
            return BeamSearchResult(
                sequence=[],
                phonemes=[],
                log_prob=float('-inf'),
                confidence=0.0,
                alignment=[],
                length=0
            )
        
        best_hyp = max(beam, key=lambda h: h.normalized_score(self.config.length_penalty))
        
        # Convert to phonemes
        phonemes = [self.vocabulary.get(token_id, '<unk>') for token_id in best_hyp.sequence]
        
        # Calculate confidence
        confidence = self._calculate_hypothesis_confidence(best_hyp, logits)
        
        return BeamSearchResult(
            sequence=best_hyp.sequence,
            phonemes=phonemes,
            log_prob=best_hyp.log_prob,
            confidence=confidence,
            alignment=best_hyp.alignment,
            length=best_hyp.length
        )
    
    def _prune_beam(self, hypotheses: List[BeamHypothesis]) -> List[BeamHypothesis]:
        """Prune beam to keep only top hypotheses."""
        
        if len(hypotheses) <= self.config.beam_width:
            return hypotheses
        
        # Group by sequence (for CTC prefix beam search)
        sequence_groups = defaultdict(list)
        for hyp in hypotheses:
            key = tuple(hyp.sequence)
            sequence_groups[key].append(hyp)
        
        # Keep best hypothesis for each unique sequence
        unique_hypotheses = []
        for group in sequence_groups.values():
            best_hyp = max(group, key=lambda h: h.log_prob)
            unique_hypotheses.append(best_hyp)
        
        # Sort by score and keep top-k
        unique_hypotheses.sort(key=lambda h: h.normalized_score(self.config.length_penalty), reverse=True)
        
        return unique_hypotheses[:self.config.beam_width]
    
    def _get_language_model_score(self, sequence: List[int]) -> float:
        """Get language model score for sequence."""
        
        if not self.language_model or not sequence:
            return 0.0
        
        try:
            # Convert to phonemes
            phonemes = [self.vocabulary.get(token_id, '<unk>') for token_id in sequence]
            
            # Get LM score (this would depend on your language model implementation)
            # For now, return a simple unigram score
            score = -len(sequence) * 0.1  # Simple length penalty
            return score
            
        except Exception as e:
            logger.warning(f"Language model scoring failed: {e}")
            return 0.0
    
    def _calculate_hypothesis_confidence(
        self, 
        hypothesis: BeamHypothesis, 
        logits: torch.Tensor
    ) -> float:
        """Calculate confidence score for hypothesis."""
        
        if not hypothesis.sequence or not hypothesis.alignment:
            return 0.0
        
        try:
            # Get probabilities for aligned frames
            log_probs = F.log_softmax(logits, dim=-1)
            
            token_probs = []
            for token_id, frame_idx in zip(hypothesis.sequence, hypothesis.alignment):
                if frame_idx < len(log_probs):
                    token_prob = log_probs[frame_idx, token_id].item()
                    token_probs.append(token_prob)
            
            if not token_probs:
                return 0.0
            
            # Calculate confidence based on method
            if self.config.confidence_method == "max_prob":
                confidence = np.exp(max(token_probs))
            elif self.config.confidence_method == "mean_prob":
                confidence = np.exp(np.mean(token_probs))
            elif self.config.confidence_method == "entropy":
                probs = np.exp(token_probs)
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                confidence = 1.0 / (1.0 + entropy)
            else:
                confidence = np.exp(np.mean(token_probs))
            
            return float(max(confidence, self.config.min_confidence))
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return self.config.min_confidence
    
    def decode_batch(
        self,
        logits: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_multiple: bool = False,
        num_results: int = 1
    ) -> List[List[BeamSearchResult]]:
        """
        Decode batch with multiple results per sequence.
        
        Args:
            logits: CTC logits [B, T, V]
            lengths: Valid sequence lengths [B]
            return_multiple: Whether to return multiple results per sequence
            num_results: Number of results to return per sequence
            
        Returns:
            List of lists of decoding results
        """
        batch_size = logits.size(0)
        all_results = []
        
        for b in range(batch_size):
            # Extract single sequence
            seq_logits = logits[b]
            seq_length = lengths[b].item() if lengths is not None else seq_logits.size(0)
            seq_logits = seq_logits[:seq_length]
            
            if return_multiple:
                # Get multiple hypotheses
                results = self._beam_search_multiple(seq_logits, num_results)
            else:
                # Get single best result
                result = self._beam_search_single(seq_logits)
                results = [result]
            
            all_results.append(results)
        
        return all_results
    
    def _beam_search_multiple(
        self, 
        logits: torch.Tensor, 
        num_results: int
    ) -> List[BeamSearchResult]:
        """Get multiple beam search results for single sequence."""
        
        # Similar to _beam_search_single but return top-k results
        seq_length, vocab_size = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Initialize beam
        beam = [BeamHypothesis(sequence=[], log_prob=0.0, alignment=[])]
        
        for t in range(seq_length):
            frame_log_probs = log_probs[t]
            new_beam = []
            
            for hyp in beam:
                top_k_log_probs, top_k_tokens = torch.topk(
                    frame_log_probs, 
                    min(self.config.beam_width, vocab_size)
                )
                
                for token_log_prob, token_id in zip(top_k_log_probs, top_k_tokens):
                    token_id = token_id.item()
                    token_log_prob = token_log_prob.item()
                    
                    if token_id == self.blank_token_id:
                        new_hyp = BeamHypothesis(
                            sequence=hyp.sequence,
                            log_prob=hyp.log_prob + token_log_prob,
                            alignment=hyp.alignment,
                            last_token=hyp.last_token
                        )
                    else:
                        if token_id == hyp.last_token:
                            new_hyp = BeamHypothesis(
                                sequence=hyp.sequence,
                                log_prob=hyp.log_prob + token_log_prob,
                                alignment=hyp.alignment,
                                last_token=token_id
                            )
                        else:
                            new_hyp = hyp.extend(token_id, token_log_prob, t)
                    
                    new_beam.append(new_hyp)
            
            beam = self._prune_beam(new_beam)
        
        # Sort beam by score and return top-k
        beam.sort(key=lambda h: h.normalized_score(self.config.length_penalty), reverse=True)
        
        results = []
        for i, hyp in enumerate(beam[:num_results]):
            phonemes = [self.vocabulary.get(token_id, '<unk>') for token_id in hyp.sequence]
            confidence = self._calculate_hypothesis_confidence(hyp, logits)
            
            result = BeamSearchResult(
                sequence=hyp.sequence,
                phonemes=phonemes,
                log_prob=hyp.log_prob,
                confidence=confidence,
                alignment=hyp.alignment,
                length=hyp.length
            )
            results.append(result)
        
        return results if results else [BeamSearchResult([], [], float('-inf'), 0.0, [], 0)]
