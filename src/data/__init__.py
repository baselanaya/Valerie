"""
Data loading and preprocessing for Valerie Visual ASR.

This module contains:
- Dataset classes for VoxCeleb2 and AVSpeech
- Data preprocessing and augmentation
- Phoneme processing utilities
- Batch collation functions
"""

from .dataset import (
    VoxCeleb2Dataset, 
    EnhancedVoxCeleb2Dataset,
    AVSpeechDataset, 
    DataSample,
    create_voxceleb2_dataset,
    create_enhanced_voxceleb2_dataset,
    create_avspeech_dataset,
    create_dataloader,
    collate_batch,
    collate_enhanced_batch
)
from .transforms import (
    VideoTransforms, 
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
    create_collator,
    pad_sequence_2d,
    create_attention_mask
)

__all__ = [
    # Dataset classes
    'VoxCeleb2Dataset',
    'EnhancedVoxCeleb2Dataset',
    'AVSpeechDataset', 
    'DataSample',
    'create_voxceleb2_dataset',
    'create_enhanced_voxceleb2_dataset',
    'create_avspeech_dataset',
    'create_dataloader',
    'collate_batch',
    'collate_enhanced_batch',
    
    # Transforms and augmentation
    'VideoTransforms',
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
    
    # Batch collation
    'ValerieCollator',
    'CTCCollator',
    'BatchData',
    'create_collator',
    'pad_sequence_2d',
    'create_attention_mask'
]