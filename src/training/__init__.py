"""
Training components for Valerie Visual ASR.

This module contains:
- Loss functions for hybrid CTC/Attention training
- Training loops with mixed precision support
- Evaluation metrics and callbacks
- Learning rate scheduling and optimization
"""

from .losses import (
    HybridCTCAttentionLoss,
    KnowledgeDistillationLoss,
    TemporalConsistencyLoss,
    ValerieLossFunction
)

from .trainer import (
    ValerieTrainer,
    TrainingConfig
)

from .metrics import (
    PhonemeErrorRate,
    WordErrorRate,
    CTCDecoder,
    ValerieMetrics
)

# Ray distributed training removed - use DDP or Accelerate for multi-GPU training

from .ddp_trainer import (
    DDPTrainer,
    DDPConfig,
    setup_ddp_environment,
    quick_ddp_train
)

try:
    from .accelerate_trainer import (
        AccelerateTrainer,
        AccelerateConfig,
        quick_accelerate_train
    )
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

__all__ = [
    # Loss functions
    'HybridCTCAttentionLoss',
    'KnowledgeDistillationLoss', 
    'TemporalConsistencyLoss',
    'ValerieLossFunction',
    
    # Training
    'ValerieTrainer',
    'TrainingConfig',
    
    # Multi-GPU training
    'DDPTrainer',
    'DDPConfig',
    'setup_ddp_environment',
    'quick_ddp_train',
    
    # Ray components removed - use DDP or Accelerate instead
    
    # Metrics
    'PhonemeErrorRate',
    'WordErrorRate', 
    'CTCDecoder',
    'ValerieMetrics'
]

# Add Accelerate components if available
if ACCELERATE_AVAILABLE:
    __all__.extend([
        'AccelerateTrainer',
        'AccelerateConfig', 
        'quick_accelerate_train'
    ])