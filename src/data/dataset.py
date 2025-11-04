"""
Dataset structures for Valerie Audio ASR.

Provides DataSample class for audio-only ASR training.
"""

import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DataSample:
    """
    Data sample structure for audio-only ASR.

    Used throughout the training and inference pipeline for consistent
    data representation.
    """

    # Audio data (primary input)
    audio_waveform: Optional[torch.Tensor] = None  # [T_audio] waveform at 16kHz
    audio_path: Optional[str] = None

    # Text and phoneme data
    transcription: Optional[str] = None
    phoneme_sequence: Optional[List[str]] = None  # List of phoneme symbols
    phoneme_alignment: Optional[List[Tuple[int, int, str]]] = None  # (start, end, phoneme)

    # Metadata
    speaker_id: Optional[str] = None
    language: str = "en"
    duration: Optional[float] = None

    # Quality metrics
    transcription_confidence: Optional[float] = None
    phoneme_confidence: Optional[float] = None

    # Training targets (optional)
    ctc_labels: Optional[torch.Tensor] = None  # CTC target labels
    attention_labels: Optional[torch.Tensor] = None  # Attention decoder targets


# Note: Dataset loaders for LibriSpeech should be implemented as needed.
# The old VoxCeleb2 and AVSpeech video dataset classes have been removed
# as the system is now audio-only.
