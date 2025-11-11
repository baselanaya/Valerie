"""
Configuration management for Valerie Audio ASR.

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
    """Model architecture configuration for audio-only ASR."""

    # Audio input specifications
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    max_sequence_length: int = 500

    # Student model architecture
    embed_dim: int = 256  # Conformer embedding dimension
    conformer_layers: int = 12  # Number of Conformer layers
    conformer_heads: int = 4  # Attention heads (optimized for 256 dim)
    conv_kernel_size: int = 31  # Convolution kernel size in Conformer
    ffn_expansion_factor: int = 4  # Feed-forward expansion factor
    dropout: float = 0.1

    # Phoneme vocabulary
    vocab_size: int = 40  # 39 phonemes + blank for CTC
    phoneme_vocab_size: int = 40  # Alias for compatibility

    # CTC/Attention hybrid decoder
    decoder_dim: int = 256
    attention_heads: int = 8

    # LLM configuration for Stage 2 (Phoneme → Text)
    llm_model_name: str = "Qwen/Qwen2-0.6B"
    lora_rank: int = 16
    lora_alpha: int = 32
    max_new_tokens: int = 256


@dataclass
class TrainingConfig:
    """Training configuration for audio ASR."""

    # Loss weights
    ctc_weight: float = 1.0  # CTC loss weight
    attention_weight: float = 0.3  # Attention loss weight
    distillation_weight: float = 0.5  # Ensemble distillation weight

    # Optimization parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 1000
    max_epochs: int = 50
    batch_size: int = 16
    gradient_clip_val: float = 1.0

    # Training strategy
    mixed_precision: bool = True  # Use fp16/bf16
    gradient_checkpointing: bool = False
    accumulate_grad_batches: int = 1
    use_amp: bool = True  # Automatic mixed precision

    # Scheduler parameters
    scheduler_type: str = "cosine"  # cosine, linear, exponential
    min_lr: float = 1e-6
    total_steps: int = 50000

    # Validation
    val_check_interval: float = 0.25
    patience: int = 10
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3


@dataclass
class DataConfig:
    """Data configuration for audio-only ASR."""

    # Dataset paths - LibriSpeech
    librispeech_root: str = "./data/librispeech"
    train_split: str = "train-clean-100"  # 100 hours
    val_split: str = "dev-clean"
    test_split: str = "test-clean"
    download: bool = True  # Auto-download LibriSpeech

    # Text corpus for Stage 2 (Phoneme → Text)
    text_datasets: List[str] = field(default_factory=lambda: ["wikitext-103-v1", "bookcorpus"])
    max_text_samples: int = 100000

    # Three-stage training configuration
    training_stage: str = "stage1"  # "stage1" (audio→phonemes), "stage2" (phonemes→text), "stage3" (end-to-end)

    # Audio preprocessing parameters
    sample_rate: int = 16000
    n_mels: int = 80
    use_specaugment: bool = True

    # SpecAugment parameters
    freq_mask_param: int = 27
    time_mask_param: int = 100
    n_freq_masks: int = 2
    n_time_masks: int = 2

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    max_samples: Optional[int] = None  # Limit dataset size for testing

    # Augmentation parameters (audio-only)
    mixup_alpha: float = 0.2
    noise_std: float = 0.01
    time_stretch_range: float = 0.1  # ±10% time stretching
    pitch_shift_range: int = 2  # ±2 semitones

    # Quality filtering
    min_duration: float = 1.0  # seconds
    max_duration: float = 20.0  # seconds
    max_sequence_length: int = 500

    # Error injection for Stage 2 training
    error_config: Dict[str, float] = field(default_factory=lambda: {
        'substitution_prob': 0.05,
        'deletion_prob': 0.05,
        'insertion_prob': 0.05,
        'max_error_rate': 0.15
    })


@dataclass
class DistillationConfig:
    """Ensemble knowledge distillation configuration."""

    # Enable distillation
    enable: bool = True
    warmup_epochs: int = 5  # Train without distillation first

    # Teacher models - Ensemble of 3
    teachers: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            'model_name': 'openai/whisper-large-v3',
            'model_type': 'whisper',
            'weight': 0.4,
            'feature_layer': -1,
            'freeze': True,
            'extract_phoneme_logits': True
        },
        {
            'model_name': 'microsoft/wavlm-large',
            'model_type': 'wavlm',
            'weight': 0.3,
            'feature_layer': -1,
            'freeze': True,
            'extract_phoneme_logits': True
        },
        {
            'model_name': 'facebook/hubert-large-ls960-ft',
            'model_type': 'hubert',
            'weight': 0.3,
            'feature_layer': -1,
            'freeze': True,
            'extract_phoneme_logits': True
        }
    ])

    # Distillation parameters
    projection_dim: int = 512
    temperature: float = 4.0
    aggregation_method: str = "weighted_average"  # weighted_average, attention, max
    feature_matching_loss: str = "mse"  # mse, cosine, huber
    soft_label_loss: str = "kl_div"  # kl_div, mse, js_div

    # Loss weights
    feature_loss_weight: float = 1.0
    soft_label_loss_weight: float = 1.0


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
            elif isinstance(value, dict):
                result[key] = value.copy()
            else:
                result[key] = value
        return result

    def validate(self) -> None:
        """Validate configuration parameters."""
        # Model validation
        assert self.model.sample_rate > 0, "Sample rate must be positive"
        assert self.model.n_mels > 0, "Number of mel bins must be positive"
        assert self.model.embed_dim > 0, "Embedding dimension must be positive"
        assert self.model.conformer_layers > 0, "Number of Conformer layers must be positive"
        assert self.model.vocab_size > 0, "Vocabulary size must be positive"

        # Training validation
        assert 0 < self.training.learning_rate < 1, "Learning rate must be between 0 and 1"
        assert self.training.batch_size > 0, "Batch size must be positive"
        assert self.training.max_epochs > 0, "Max epochs must be positive"

        # Loss weights should be non-negative
        assert self.training.ctc_weight >= 0, "CTC weight must be non-negative"
        assert self.training.attention_weight >= 0, "Attention weight must be non-negative"
        assert self.training.distillation_weight >= 0, "Distillation weight must be non-negative"

        # Data validation
        assert self.data.sample_rate > 0, "Sample rate must be positive"
        assert self.data.num_workers >= 0, "Number of workers must be non-negative"
        assert self.data.min_duration > 0, "Minimum duration must be positive"
        assert self.data.max_duration > self.data.min_duration, "Max duration must be greater than min duration"

        # Distillation validation
        if self.distillation.enable:
            assert len(self.distillation.teachers) > 0, "At least one teacher model required"
            assert self.distillation.temperature > 0, "Temperature must be positive"
            assert self.distillation.projection_dim > 0, "Projection dimension must be positive"

    def __str__(self) -> str:
        """String representation of configuration."""
        return (f"Config(\n"
                f"  Model: embed_dim={self.model.embed_dim}, layers={self.model.conformer_layers}, vocab={self.model.vocab_size}\n"
                f"  Training: lr={self.training.learning_rate}, epochs={self.training.max_epochs}, batch={self.training.batch_size}\n"
                f"  Data: dataset={self.data.train_split}, workers={self.data.num_workers}\n"
                f"  Distillation: enabled={self.distillation.enable}, teachers={len(self.distillation.teachers)}\n"
                f")")


def load_config(config_path: str) -> Config:
    """Convenience function to load configuration from file."""
    return Config(config_path)


def create_default_configs() -> None:
    """Create default configuration files for audio ASR."""
    config = Config()

    # Create configs directory
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)

    # Save base config
    config.save_to_file("configs/base_audio_config.yaml")

    # Create Stage 1 specific config (Audio → Phonemes)
    config.data.training_stage = "stage1"
    config.training.batch_size = 16
    config.training.max_epochs = 50
    config.save_to_file("configs/stage1_config.yaml")

    # Create Stage 2 specific config (Phonemes → Text)
    config.data.training_stage = "stage2"
    config.training.batch_size = 32
    config.training.max_epochs = 20
    config.training.learning_rate = 5e-5
    config.save_to_file("configs/stage2_config.yaml")

    # Create Stage 3 specific config (End-to-End)
    config.data.training_stage = "stage3"
    config.training.batch_size = 8
    config.training.max_epochs = 10
    config.training.learning_rate = 1e-5
    config.save_to_file("configs/stage3_config.yaml")

    print("Default configuration files created in configs/")
    print("  - base_audio_config.yaml")
    print("  - stage1_config.yaml (Audio → Phonemes)")
    print("  - stage2_config.yaml (Phonemes → Text)")
    print("  - stage3_config.yaml (End-to-End)")


if __name__ == "__main__":
    # Create default configs when run as script
    create_default_configs()
