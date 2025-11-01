"""
Configuration management for Valerie Visual ASR.

Provides dataclasses and utilities for managing model, training, and data configurations
with YAML file support and validation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml
import os
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Input specifications
    input_height: int = 224
    input_width: int = 224
    input_channels: int = 3
    max_sequence_length: int = 500
    
    # 3D CNN parameters
    embed_dim: int = 512
    conv3d_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    conv3d_kernel_size: int = 3
    conv3d_stride: int = 1
    conv3d_padding: int = 1
    
    # Conformer parameters
    conformer_layers: int = 12
    conformer_heads: int = 8
    conformer_dim: int = 512
    conv_kernel_size: int = 31
    ffn_expansion_factor: int = 4
    dropout: float = 0.1
    
    # Phoneme vocabulary
    phoneme_vocab_size: int = 40  # 39 phonemes + blank
    
    # CTC/Attention parameters
    decoder_dim: int = 256
    attention_heads: int = 8
    
    # LLM configuration (now mandatory) - Upgraded to match VALLR scale
    llm_model_name: str = "Qwen/Qwen3-1.7B"
    lora_rank: int = 16
    lora_alpha: int = 32
    enable_thinking: bool = False  # Qwen3 thinking mode
    max_new_tokens: int = 32768  # Qwen3 recommended output length


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Loss weights
    ctc_weight: float = 0.3
    attention_weight: float = 0.7
    distillation_weight: float = 0.1
    temporal_consistency_weight: float = 0.05
    
    # Optimization parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    warmup_steps: int = 4000
    max_epochs: int = 100
    batch_size: int = 8
    gradient_clip_val: float = 1.0
    
    # Training strategy
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    accumulate_grad_batches: int = 1
    
    # Scheduler parameters
    scheduler_type: str = "cosine"  # cosine, linear, exponential
    min_lr: float = 1e-6
    
    # Validation
    val_check_interval: float = 0.25
    patience: int = 10


@dataclass
class DataConfig:
    """Data configuration with VALLR-style training support."""
    
    # Dataset paths
    voxceleb2_root: str = "data"
    avspeech_root: str = "data/avspeech"
    
    # Two-stage training data paths (for generated transcriptions and phonemes)
    transcription_dir: str = "data/transcriptions"
    phoneme_dir: str = "data/phonemes"
    
    # Training methodology
    use_two_stage_training: bool = True  # Use two-stage visual speech recognition training
    training_stage: str = "stage1"  # "stage1" (video→phonemes), "stage2" (phonemes→text), "both"
    
    # Preprocessing parameters
    frame_rate: int = 25
    audio_sample_rate: int = 16000
    
    # Data loading
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2
    max_samples: Optional[int] = None  # Limit dataset size for testing
    quality_filter: bool = True
    
    # Augmentation parameters
    horizontal_flip_prob: float = 0.5
    brightness_range: float = 0.2
    contrast_range: float = 0.2
    noise_std: float = 0.01
    mixup_alpha: float = 0.2
    
    # Quality filtering
    min_face_confidence: float = 0.8
    min_transcription_confidence: float = 0.1  # Lower threshold for Whisper ASR
    min_duration: float = 1.0
    max_duration: float = 10.0
    max_sequence_length: int = 500


@dataclass
class DistillationConfig:
    """Audio knowledge distillation configuration (now mandatory)."""
    
    # Teacher model - Using Whisper Large V3 for best performance
    teacher_model_name: str = "openai/whisper-large-v3"
    teacher_feature_dim: int = 1280  # Whisper Large V3 feature dimension
    
    # Distillation parameters
    temperature: float = 4.0
    feature_matching_weight: float = 1.0  # Weight for feature matching loss
    contrastive_weight: float = 0.1  # Weight for contrastive loss
    alpha: float = 0.7  # Overall weight for distillation loss
    
    # Cross-modal alignment
    use_cca_init: bool = True
    projection_dim: int = 512


class Config:
    """Main configuration class that combines all config components."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or defaults."""
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.distillation = DistillationConfig()
        
        if config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update configurations
        if 'model' in config_dict:
            self._update_dataclass(self.model, config_dict['model'])
        if 'training' in config_dict:
            self._update_dataclass(self.training, config_dict['training'])
        if 'data' in config_dict:
            self._update_dataclass(self.data, config_dict['data'])
        if 'distillation' in config_dict:
            self._update_dataclass(self.distillation, config_dict['distillation'])
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'model': self._dataclass_to_dict(self.model),
            'training': self._dataclass_to_dict(self.training),
            'data': self._dataclass_to_dict(self.data),
            'distillation': self._dataclass_to_dict(self.distillation)
        }
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def _update_dataclass(self, dataclass_obj: Any, update_dict: Dict[str, Any]) -> None:
        """Update dataclass fields from dictionary."""
        for key, value in update_dict.items():
            if hasattr(dataclass_obj, key):
                setattr(dataclass_obj, key, value)
    
    def _dataclass_to_dict(self, dataclass_obj: Any) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        result = {}
        for key, value in dataclass_obj.__dict__.items():
            if isinstance(value, list):
                result[key] = value.copy()
            else:
                result[key] = value
        return result
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Model validation
        assert self.model.input_height > 0, "Input height must be positive"
        assert self.model.input_width > 0, "Input width must be positive"
        assert self.model.embed_dim > 0, "Embedding dimension must be positive"
        assert self.model.conformer_layers > 0, "Number of Conformer layers must be positive"
        
        # Training validation
        assert 0 < self.training.learning_rate < 1, "Learning rate must be between 0 and 1"
        assert self.training.batch_size > 0, "Batch size must be positive"
        assert self.training.max_epochs > 0, "Max epochs must be positive"
        
        # Loss weights should sum to reasonable values
        total_weight = (self.training.ctc_weight + 
                       self.training.attention_weight + 
                       self.training.distillation_weight)
        assert 0.5 <= total_weight <= 2.0, f"Total loss weights seem unreasonable: {total_weight}"
        
        # Data validation
        assert self.data.frame_rate > 0, "Frame rate must be positive"
        assert self.data.num_workers >= 0, "Number of workers must be non-negative"
        
    def __str__(self) -> str:
        """String representation of configuration."""
        return (f"Config(\n"
                f"  Model: {self.model}\n"
                f"  Training: {self.training}\n"
                f"  Data: {self.data}\n"
                f"  Distillation: {self.distillation}\n"
                f")")


def load_config(config_path: str) -> Config:
    """Convenience function to load configuration from file."""
    return Config(config_path)


def create_default_configs() -> None:
    """Create default configuration files."""
    config = Config()
    
    # Create configs directory
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Save base config
    config.save_to_file("configs/base_config.yaml")
    
    # Create VoxCeleb2 specific config
    config.data.voxceleb2_root = "/data/voxceleb2"
    config.data.avspeech_root = ""  # Only VoxCeleb2
    config.save_to_file("configs/voxceleb2_config.yaml")
    
    # Create AVSpeech specific config
    config.data.voxceleb2_root = ""  # Only AVSpeech
    config.data.avspeech_root = "/data/avspeech"
    config.save_to_file("configs/avspeech_config.yaml")
    
    print("Default configuration files created in configs/")


if __name__ == "__main__":
    # Create default configs when run as script
    create_default_configs()