"""
Unit tests for audio ASR configuration management.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from src.utils.config import Config, ModelConfig, TrainingConfig, DataConfig, DistillationConfig


class TestConfig:
    """Test configuration management functionality."""

    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = Config()

        # Check that all sub-configs are created
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.distillation, DistillationConfig)

        # Check some default values for audio-only architecture
        assert config.model.sample_rate == 16000
        assert config.model.n_mels == 80
        assert config.model.embed_dim == 256
        assert config.model.conformer_layers == 12
        assert config.training.learning_rate == 1e-4
        assert config.data.train_split == "train-clean-100"

    def test_config_validation(self):
        """Test configuration validation."""
        config = Config()

        # Valid config should pass
        config.validate()

        # Invalid values should raise assertions
        config.model.sample_rate = -1
        with pytest.raises(AssertionError):
            config.validate()

        config.model.sample_rate = 16000  # Reset
        config.training.learning_rate = 2.0  # Invalid LR
        with pytest.raises(AssertionError):
            config.validate()

    def test_config_save_load(self):
        """Test saving and loading configuration files."""
        config = Config()

        # Modify some values
        config.model.embed_dim = 512
        config.training.batch_size = 32
        config.data.train_split = "train-clean-360"

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name

        try:
            config.save_to_file(config_path)

            # Load and verify
            loaded_config = Config(config_path)
            assert loaded_config.model.embed_dim == 512
            assert loaded_config.training.batch_size == 32
            assert loaded_config.data.train_split == "train-clean-360"
        finally:
            Path(config_path).unlink()


class TestModelConfig:
    """Test model configuration specifically."""

    def test_default_model_params(self):
        """Test default model parameters."""
        config = ModelConfig()

        assert config.sample_rate == 16000
        assert config.n_mels == 80
        assert config.embed_dim == 256
        assert config.conformer_layers == 12
        assert config.conformer_heads == 4
        assert config.vocab_size == 40

    def test_audio_params(self):
        """Test audio-specific parameters."""
        config = ModelConfig()

        assert config.n_fft == 400
        assert config.hop_length == 160
        assert config.sample_rate > 0
        assert config.n_mels > 0

    def test_llm_config(self):
        """Test LLM configuration for Stage 2."""
        config = ModelConfig()

        assert config.llm_model_name == "Qwen/Qwen2-0.6B"
        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert config.max_new_tokens == 256


class TestTrainingConfig:
    """Test training configuration specifically."""

    def test_default_training_params(self):
        """Test default training parameters."""
        config = TrainingConfig()

        assert config.learning_rate == 1e-4
        assert config.batch_size == 16
        assert config.max_epochs == 50
        assert config.gradient_clip_val == 1.0

    def test_loss_weights(self):
        """Test loss weight configuration."""
        config = TrainingConfig()

        assert config.ctc_weight == 1.0
        assert config.attention_weight == 0.3
        assert config.distillation_weight == 0.5

        # All weights should be non-negative
        assert config.ctc_weight >= 0
        assert config.attention_weight >= 0
        assert config.distillation_weight >= 0

    def test_optimizer_params(self):
        """Test optimizer parameters."""
        config = TrainingConfig()

        assert config.weight_decay == 1e-2
        assert config.warmup_steps == 1000
        assert 0 < config.learning_rate < 1

    def test_mixed_precision(self):
        """Test mixed precision training settings."""
        config = TrainingConfig()

        assert config.mixed_precision == True
        assert config.use_amp == True


class TestDataConfig:
    """Test data configuration specifically."""

    def test_default_dataset_params(self):
        """Test default dataset parameters for LibriSpeech."""
        config = DataConfig()

        assert config.librispeech_root == "./data/librispeech"
        assert config.train_split == "train-clean-100"
        assert config.val_split == "dev-clean"
        assert config.test_split == "test-clean"
        assert config.download == True

    def test_audio_preprocessing_params(self):
        """Test audio preprocessing parameters."""
        config = DataConfig()

        assert config.sample_rate == 16000
        assert config.n_mels == 80
        assert config.use_specaugment == True

    def test_specaugment_params(self):
        """Test SpecAugment parameters."""
        config = DataConfig()

        assert config.freq_mask_param == 27
        assert config.time_mask_param == 100
        assert config.n_freq_masks == 2
        assert config.n_time_masks == 2

    def test_augmentation_params(self):
        """Test augmentation parameters for audio."""
        config = DataConfig()

        assert config.mixup_alpha == 0.2
        assert config.noise_std == 0.01
        assert config.time_stretch_range == 0.1
        assert config.pitch_shift_range == 2

    def test_data_loading_params(self):
        """Test data loading parameters."""
        config = DataConfig()

        assert config.num_workers >= 0
        assert config.pin_memory == True
        assert config.prefetch_factor == 2

    def test_quality_filtering(self):
        """Test quality filtering parameters."""
        config = DataConfig()

        assert config.min_duration > 0
        assert config.max_duration > config.min_duration
        assert config.max_sequence_length > 0


class TestDistillationConfig:
    """Test distillation configuration specifically."""

    def test_default_distillation_params(self):
        """Test default distillation parameters."""
        config = DistillationConfig()

        assert config.enable == True
        assert config.warmup_epochs == 5
        assert config.temperature == 4.0
        assert config.projection_dim == 512

    def test_teacher_models_config(self):
        """Test teacher models configuration."""
        config = DistillationConfig()

        # Should have 3 teacher models
        assert len(config.teachers) == 3

        # Check teacher names and types
        teacher_names = [t['model_name'] for t in config.teachers]
        teacher_types = [t['model_type'] for t in config.teachers]

        assert 'openai/whisper-large-v3' in teacher_names
        assert 'microsoft/wavlm-large' in teacher_names
        assert 'facebook/hubert-large-ls960-ft' in teacher_names

        assert 'whisper' in teacher_types
        assert 'wavlm' in teacher_types
        assert 'hubert' in teacher_types

    def test_teacher_weights(self):
        """Test teacher ensemble weights."""
        config = DistillationConfig()

        # Check weights sum to approximately 1.0
        total_weight = sum(t['weight'] for t in config.teachers)
        assert abs(total_weight - 1.0) < 0.01  # Allow small floating point error

        # Each weight should be positive
        for teacher in config.teachers:
            assert teacher['weight'] > 0

    def test_distillation_loss_params(self):
        """Test distillation loss parameters."""
        config = DistillationConfig()

        assert config.aggregation_method in ["weighted_average", "attention", "max"]
        assert config.feature_matching_loss in ["mse", "cosine", "huber"]
        assert config.soft_label_loss in ["kl_div", "mse", "js_div"]
        assert config.feature_loss_weight > 0
        assert config.soft_label_loss_weight > 0


class TestConfigIntegration:
    """Integration tests for complete configuration."""

    def test_stage1_config(self):
        """Test Stage 1 (Audio → Phonemes) configuration."""
        config = Config()
        config.data.training_stage = "stage1"
        config.training.batch_size = 16
        config.training.max_epochs = 50

        config.validate()

        assert config.data.training_stage == "stage1"
        assert config.distillation.enable == True

    def test_stage2_config(self):
        """Test Stage 2 (Phonemes → Text) configuration."""
        config = Config()
        config.data.training_stage = "stage2"
        config.training.batch_size = 32
        config.training.max_epochs = 20
        config.training.learning_rate = 5e-5

        config.validate()

        assert config.data.training_stage == "stage2"
        assert config.model.llm_model_name == "Qwen/Qwen2-0.6B"

    def test_stage3_config(self):
        """Test Stage 3 (End-to-End) configuration."""
        config = Config()
        config.data.training_stage = "stage3"
        config.training.batch_size = 8
        config.training.max_epochs = 10
        config.training.learning_rate = 1e-5

        config.validate()

        assert config.data.training_stage == "stage3"


if __name__ == "__main__":
    pytest.main([__file__])
