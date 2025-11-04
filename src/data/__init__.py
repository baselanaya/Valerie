"""
Data loading and preprocessing for Valerie Audio ASR.

This module contains:
- Dataset classes for audio-only ASR (LibriSpeech, etc.)
- Audio preprocessing and augmentation
- Phoneme processing utilities
- Batch collation functions
"""

from .dataset import (
    DataSample,
)
from .transforms import (
    AudioTransforms,
    MixupAugmentation,
    DataAugmentationPipeline,
    create_augmentation_pipeline
)
from .phoneme_utils import (
    TextToPhonemeConverter,
    ForcedAligner,
    CTCLabelGenerator,
    PhonemeAlignment,
    PhonemeSequence,
    PHONEME_VOCAB,
    PHONEME_TO_IDX,
    IDX_TO_PHONEME,
    process_text_to_ctc_labels,
    create_phoneme_processor
)
from .collate import (
    ValerieCollator,
    CTCCollator,
    BatchData,
    create_collator
)

__all__ = [
    # Dataset classes (audio-only)
    'DataSample',

    # Transforms and augmentation (audio-only)
    'AudioTransforms',
    'MixupAugmentation',
    'DataAugmentationPipeline',
    'create_augmentation_pipeline',

    # Phoneme processing
    'TextToPhonemeConverter',
    'ForcedAligner',
    'CTCLabelGenerator',
    'PhonemeAlignment',
    'PhonemeSequence',
    'PHONEME_VOCAB',
    'PHONEME_TO_IDX',
    'IDX_TO_PHONEME',
    'process_text_to_ctc_labels',
    'create_phoneme_processor',

    # Batch collation (audio-only)
    'ValerieCollator',
    'CTCCollator',
    'BatchData',
    'create_collator'
]