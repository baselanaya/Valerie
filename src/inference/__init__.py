"""
Inference components for Valerie Visual ASR.

This module contains:
- CTC beam search decoder with confidence scoring
- Phoneme-to-sentence reconstruction using LLM
- Batch inference pipeline for efficient processing
- Real-time inference capabilities
"""

from .ctc_decoder import (
    CTCBeamSearchDecoder,
    GreedyDecoder,
    BeamSearchResult,
    DecodingConfig
)

from .phoneme_reconstruction import (
    PhonemeToSentenceReconstructor,
    ReconstructionConfig,
    ReconstructionResult
)

from .inference_engine import (
    ValerieInferenceEngine,
    InferenceConfig,
    InferenceResult
)

__all__ = [
    # CTC Decoding
    'CTCBeamSearchDecoder',
    'GreedyDecoder',
    'BeamSearchResult',
    'DecodingConfig',
    
    # Phoneme Reconstruction
    'PhonemeToSentenceReconstructor',
    'ReconstructionConfig', 
    'ReconstructionResult',
    
    # Inference Engine
    'ValerieInferenceEngine',
    'InferenceConfig',
    'InferenceResult'
]