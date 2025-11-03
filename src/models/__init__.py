"""
Model implementations for Valerie Audio-Only ASR with Ensemble Distillation.

This module contains the core model components:
- AudioFrontend: Mel spectrogram extraction from audio
- AudioPhonemeASR: Complete audio-to-phoneme model (Stage 1)
- ConformerEncoder: Hybrid CNN-Transformer encoder
- HybridCTCAttention: Joint CTC/Attention training head
- EnsembleDistillationModule: Multi-teacher knowledge distillation
- QwenPhonemeToText: LLM for phoneme-to-text reconstruction (Stage 2)

Legacy (deprecated):
- ValerieModel: Old video-based model (use AudioPhonemeASR instead)
"""

from .conformer import ConformerEncoder, ConformerBlock, FeedForward, MultiHeadSelfAttention, ConvolutionModule
from .hybrid_ctc_attention import HybridCTCAttention, AttentionDecoder
from .qwen_llm import QwenPhonemeToText
from .audio_frontend import AudioFrontend, MelSpectrogramExtractor, SpecAugment
from .audio_phoneme_model import AudioPhonemeASR
from .ensemble_distillation import EnsembleDistillationModule, TeacherModel
from .valerie_model import ValerieModel  # Legacy - deprecated

__all__ = [
    # New audio-only models
    'AudioPhonemeASR',
    'AudioFrontend',
    'MelSpectrogramExtractor',
    'SpecAugment',
    'EnsembleDistillationModule',
    'TeacherModel',

    # Core components (reused)
    'ConformerEncoder',
    'ConformerBlock',
    'FeedForward',
    'MultiHeadSelfAttention',
    'ConvolutionModule',
    'HybridCTCAttention',
    'AttentionDecoder',
    'QwenPhonemeToText',

    # Legacy (deprecated)
    'ValerieModel',
]