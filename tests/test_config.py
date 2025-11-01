"""
Unit tests for configuration management.
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
        
        # Check some default values
        assert config.model.input_height == 224
        assert config.model.input_width == 224
        assert config.training.learning_rate == 1e-4
        assert config.data.frame_rate == 25
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = Config()
        
        # Valid config should pass
        config.validate()
        
        # Invalid values should raise assertions
        config.model.input_height = -1
        with pytest.raises(AssertionError):
            config.validate()
        
        config.model.input_height = 224  # Reset
        config.training.learning_rate = 2.0  # Invalid LR
        with pytest.raises(AssertionError):
            config.validate()
    
    def test_config_save_load(self):
        """Test saving and loading configuration files."""
        config = Config()
        
        # Modify some values
        config.model.embed_dim = 256
        config.training.batch_size = 16
        config.data.num_workers = 4
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            # Save config
            config.save_to_file(config_path)
            
            # Load config
            loaded_config = Config(config_path)
            
            # Check that values were preserved
            assert loaded_config.model.embed_dim == 256
            assert loaded_config.training.batch_size == 16
            assert loaded_config.data.num_workers == 4
            
        finally:
            Path(config_path).unlink()
    
    def test_partial_config_loading(self):
        """Test loading partial configuration (only some sections)."""
        # Create partial config file
        partial_config = {
            'model': {
                'embed_dim': 128,
                'conformer_layers': 6
            },
            'training': {
                'learning_rate': 5e-4
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(partial_config, f)
            config_path = f.name
        
        try:
            config = Config(config_path)
            
            # Check that specified values were loaded
            assert config.model.embed_dim == 128
            assert config.model.conformer_layers == 6
            assert config.training.learning_rate == 5e-4
            
            # Check that unspecified values remain default
            assert config.model.input_height == 224  # Default value
            assert config.data.frame_rate == 25  # Default value
            
        finally:
            Path(config_path).unlink()
    
    def test_config_string_representation(self):
        """Test string representation of configuration."""
        config = Config()
        config_str = str(config)
        
        assert "Config(" in config_str
        assert "Model:" in config_str
        assert "Training:" in config_str
        assert "Data:" in config_str
        assert "Distillation:" in config_str


class TestModelConfig:
    """Test model configuration specifically."""
    
    def test_default_values(self):
        """Test default model configuration values."""
        config = ModelConfig()
        
        assert config.input_height == 224
        assert config.input_width == 224
        assert config.input_channels == 3
        assert config.embed_dim == 512
        assert config.conformer_layers == 12
        assert config.phoneme_vocab_size == 40
    
    def test_conv3d_channels_list(self):
        """Test that conv3d_channels is properly handled as a list."""
        config = ModelConfig()
        
        assert isinstance(config.conv3d_channels, list)
        assert config.conv3d_channels == [64, 128, 256, 512]
        
        # Test modification
        config.conv3d_channels = [32, 64, 128]
        assert config.conv3d_channels == [32, 64, 128]


class TestTrainingConfig:
    """Test training configuration specifically."""
    
    def test_loss_weights(self):
        """Test loss weight configuration."""
        config = TrainingConfig()
        
        assert config.ctc_weight == 0.3
        assert config.attention_weight == 0.7
        assert config.distillation_weight == 0.1
        
        # Test that weights are reasonable
        total_weight = config.ctc_weight + config.attention_weight + config.distillation_weight
        assert 0.5 <= total_weight <= 2.0
    
    def test_optimization_params(self):
        """Test optimization parameters."""
        config = TrainingConfig()
        
        assert 0 < config.learning_rate < 1
        assert config.batch_size > 0
        assert config.max_epochs > 0
        assert config.gradient_clip_val > 0


class TestDataConfig:
    """Test data configuration specifically."""
    
    def test_default_paths(self):
        """Test default dataset paths."""
        config = DataConfig()
        
        assert config.voxceleb2_root == "/data/voxceleb2"
        assert config.avspeech_root == "/data/avspeech"
    
    def test_preprocessing_params(self):
        """Test preprocessing parameters."""
        config = DataConfig()
        
        assert config.frame_rate == 25
        assert config.audio_sample_rate == 16000
        assert config.num_workers >= 0
    
    def test_augmentation_params(self):
        """Test augmentation parameters."""
        config = DataConfig()
        
        assert 0 <= config.horizontal_flip_prob <= 1
        assert config.brightness_range > 0
        assert config.contrast_range > 0
        assert config.noise_std > 0


class TestDistillationConfig:
    """Test distillation configuration specifically."""
    
    def test_teacher_model_config(self):
        """Test teacher model configuration."""
        config = DistillationConfig()
        
        assert config.teacher_model_name == "facebook/wav2vec2-large-960h"
        assert config.teacher_feature_dim > 0
    
    def test_distillation_params(self):
        """Test distillation parameters."""
        config = DistillationConfig()
        
        assert config.temperature > 0
        assert 0 <= config.alpha <= 1
        assert isinstance(config.feature_matching, bool)
        assert isinstance(config.use_cca_init, bool)


if __name__ == "__main__":
    pytest.main([__file__])