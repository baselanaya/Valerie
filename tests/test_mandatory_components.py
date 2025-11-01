"""
Test suite to verify that audio knowledge distillation and phoneme-to-text reconstruction
are now mandatory components of the ValerieModel.
"""

import unittest
import torch
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.models.valerie_model import ValerieModel
from src.utils.logging import setup_logger, get_logger

# Setup logging for tests
setup_logger(use_rich=True)
logger = get_logger(__name__)


class TestMandatoryComponents(unittest.TestCase):
    """Test that mandatory components are always initialized and functional."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.device = torch.device('cpu')  # Use CPU for testing
    
    def test_distillation_always_present(self):
        """Test that distillation component is always initialized."""
        logger.info("🧪 Testing Mandatory Distillation Component")
        
        model = ValerieModel(self.config)
        
        # Distillation should always be present
        self.assertIsNotNone(model.distillation, "Distillation component should always be initialized")
        
        # Should be a neural network module
        self.assertIsInstance(model.distillation, torch.nn.Module, "Distillation should be a PyTorch module")
        
        # Should have distillation parameters
        self.assertTrue(hasattr(model, 'distillation_temperature'), "Should have distillation temperature")
        self.assertTrue(hasattr(model, 'distillation_alpha'), "Should have distillation alpha")
        
        logger.info("   ✅ Distillation component is mandatory and properly initialized")
    
    def test_phoneme_to_text_always_present(self):
        """Test that phoneme-to-text component is always initialized."""
        logger.info("🧪 Testing Mandatory Phoneme-to-Text Component")
        
        model = ValerieModel(self.config)
        
        # Phoneme-to-text should always be present
        self.assertIsNotNone(model.phoneme_to_text, "Phoneme-to-text component should always be initialized")
        
        # Should be a neural network module
        self.assertIsInstance(model.phoneme_to_text, torch.nn.Module, "Phoneme-to-text should be a PyTorch module")
        
        logger.info("   ✅ Phoneme-to-text component is mandatory and properly initialized")
    
    def test_model_info_reflects_mandatory_components(self):
        """Test that model info correctly shows both components as enabled."""
        logger.info("🧪 Testing Model Info for Mandatory Components")
        
        model = ValerieModel(self.config)
        info = model.get_model_info()
        
        # Both components should be marked as enabled
        components = info['components']
        self.assertTrue(components['distillation'], "Distillation should be marked as enabled in model info")
        self.assertTrue(components['phoneme_to_text'], "Phoneme-to-text should be marked as enabled in model info")
        
        logger.info("   ✅ Model info correctly reflects mandatory components")
        logger.info(f"   ✅ Components: {components}")
    
    def test_forward_pass_always_computes_distillation(self):
        """Test that forward pass always computes distillation outputs."""
        logger.info("🧪 Testing Forward Pass Distillation Computation")
        
        model = ValerieModel(self.config)
        model.eval()
        
        # Create test inputs
        batch_size = 2
        video = torch.randn(batch_size, 3, 8, 224, 224)  # [B, C, T, H, W]
        input_lengths = torch.tensor([8, 6])
        
        # Test without teacher features
        with torch.no_grad():
            outputs = model(video, input_lengths)
        
        # Should always have distillation_loss key
        self.assertIn('distillation_loss', outputs, "Forward pass should always include distillation_loss")
        
        # Should be zero when no teacher features provided
        self.assertEqual(outputs['distillation_loss'].item(), 0.0, 
                        "Distillation loss should be zero when no teacher features provided")
        
        # Test with teacher features
        teacher_features = torch.randn(batch_size, 8, self.config.distillation.teacher_feature_dim)
        
        with torch.no_grad():
            outputs = model(video, input_lengths, teacher_features=teacher_features)
        
        # Should compute actual distillation loss
        self.assertGreater(outputs['distillation_loss'].item(), 0.0,
                          "Distillation loss should be positive when teacher features provided")
        
        logger.info("   ✅ Forward pass always computes distillation outputs")
    
    def test_forward_pass_always_computes_text_outputs(self):
        """Test that forward pass always computes text generation outputs."""
        logger.info("🧪 Testing Forward Pass Text Generation")
        
        model = ValerieModel(self.config)
        model.eval()
        
        # Create test inputs
        batch_size = 2
        video = torch.randn(batch_size, 3, 8, 224, 224)  # [B, C, T, H, W]
        input_lengths = torch.tensor([8, 6])
        
        with torch.no_grad():
            outputs = model(video, input_lengths)
        
        # Should always have text_outputs key
        self.assertIn('text_outputs', outputs, "Forward pass should always include text_outputs")
        
        # Should not be None
        self.assertIsNotNone(outputs['text_outputs'], "Text outputs should not be None")
        
        # Should have expected structure
        text_outputs = outputs['text_outputs']
        if isinstance(text_outputs, dict):
            # Should have generated text for each batch item
            self.assertIn('generated_text', text_outputs, "Text outputs should include generated_text")
            self.assertEqual(len(text_outputs['generated_text']), batch_size,
                           "Should have text output for each batch item")
        
        logger.info("   ✅ Forward pass always computes text generation outputs")
    
    def test_generate_text_method_always_works(self):
        """Test that generate_text method always returns outputs."""
        logger.info("🧪 Testing Generate Text Method")
        
        model = ValerieModel(self.config)
        model.eval()
        
        # Create test phoneme logits
        batch_size = 2
        seq_len = 10
        vocab_size = self.config.model.phoneme_vocab_size
        phoneme_logits = torch.randn(batch_size, seq_len, vocab_size)
        input_lengths = torch.tensor([seq_len, seq_len-2])
        
        with torch.no_grad():
            text_outputs = model.generate_text(phoneme_logits, input_lengths)
        
        # Should never return None
        self.assertIsNotNone(text_outputs, "generate_text should never return None")
        
        # Should be a dictionary
        self.assertIsInstance(text_outputs, dict, "Text outputs should be a dictionary")
        
        logger.info("   ✅ Generate text method always returns outputs")
    
    def test_component_freezing_unfreezing(self):
        """Test that mandatory components can be frozen/unfrozen."""
        logger.info("🧪 Testing Component Freezing/Unfreezing")
        
        model = ValerieModel(self.config)
        
        # Test freezing distillation
        model.freeze_component('distillation')
        for param in model.distillation.parameters():
            self.assertFalse(param.requires_grad, "Distillation parameters should be frozen")
        
        model.unfreeze_component('distillation')
        for param in model.distillation.parameters():
            self.assertTrue(param.requires_grad, "Distillation parameters should be unfrozen")
        
        # Test freezing phoneme-to-text
        model.freeze_component('phoneme_to_text')
        for param in model.phoneme_to_text.parameters():
            self.assertFalse(param.requires_grad, "Phoneme-to-text parameters should be frozen")
        
        model.unfreeze_component('phoneme_to_text')
        for param in model.phoneme_to_text.parameters():
            self.assertTrue(param.requires_grad, "Phoneme-to-text parameters should be unfrozen")
        
        logger.info("   ✅ Component freezing/unfreezing works correctly")
    
    def test_model_size_includes_mandatory_components(self):
        """Test that model size includes the mandatory components."""
        logger.info("🧪 Testing Model Size with Mandatory Components")
        
        model = ValerieModel(self.config)
        info = model.get_model_info()
        
        # Model should be quite large due to LLM component
        total_params = info['total_parameters']
        model_size_mb = info['model_size_mb']
        
        # Should be substantial due to Qwen LLM (600M+ parameters)
        self.assertGreater(total_params, 500_000_000, "Model should have 500M+ parameters due to LLM")
        self.assertGreater(model_size_mb, 1000, "Model should be 1GB+ due to mandatory LLM component")
        
        logger.info(f"   ✅ Model size: {total_params:,} parameters ({model_size_mb:.1f} MB)")


def run_mandatory_component_tests():
    """Run all mandatory component tests."""
    
    logger.info("🚀 Running Mandatory Component Tests")
    logger.info("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMandatoryComponents)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary
    logger.info("=" * 60)
    if result.wasSuccessful():
        logger.info("🎉 All mandatory component tests passed!")
        logger.info(f"✅ Ran {result.testsRun} tests successfully")
        logger.info("✅ Audio distillation and phoneme-to-text are now mandatory!")
    else:
        logger.error(f"❌ {len(result.failures)} test(s) failed")
        logger.error(f"❌ {len(result.errors)} test(s) had errors")
        
        for test, error in result.failures + result.errors:
            logger.error(f"   ❌ {test}: {error}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_mandatory_component_tests()
    exit(0 if success else 1)
