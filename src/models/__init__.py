"""
Model implementations for Valerie Visual ASR.

This module contains the core model components:
- SpatioTemporalEmbedding: 3D CNN for video feature extraction
- ConformerEncoder: Hybrid CNN-Transformer encoder
- HybridCTCAttention: Joint CTC/Attention training head
- AudioKnowledgeDistillation: Knowledge distillation from audio models
- ValerieModel: Main model class integrating all components
"""

from .spatio_temporal import SpatioTemporalEmbedding, Conv3dBlock, TemporalPositionalEncoding
from .conformer import ConformerEncoder, ConformerBlock, FeedForward, MultiHeadSelfAttention, ConvolutionModule
from .hybrid_ctc_attention import HybridCTCAttention, AttentionDecoder
from .qwen_llm import QwenPhonemeToText
from .audio_distillation import AudioKnowledgeDistillation, WhisperTeacher, CrossModalAlignment
from .valerie_model import ValerieModel

__all__ = [
    'ValerieModel',
    'SpatioTemporalEmbedding',
    'Conv3dBlock', 
    'TemporalPositionalEncoding',
    'ConformerEncoder',
    'ConformerBlock',
    'FeedForward',
    'MultiHeadSelfAttention', 
    'ConvolutionModule',
    'HybridCTCAttention',
    'AttentionDecoder',
    'QwenPhonemeToText',
    'AudioKnowledgeDistillation',
    'WhisperTeacher',
    'CrossModalAlignment'
]