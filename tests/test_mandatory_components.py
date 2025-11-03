"""
Test suite for AudioPhonemeASR model with ensemble knowledge distillation.

Tests the new audio-only architecture with optional 3-teacher ensemble distillation.
"""

import unittest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.audio_phoneme_model import AudioPhonemeASR
from src.utils.logging import setup_logger, get_logger

# Setup logging for tests
setup_logger(use_rich=True)
logger = get_logger(__name__)


class TestAudioPhonemeASR(unittest.TestCase):
    """Test AudioPhonemeASR model components and ensemble distillation."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')  # Use CPU for testing

        # Create model without distillation
        self.model = AudioPhonemeASR(
            embed_dim=256,
            conformer_layers=4,  # Reduced for testing
            conformer_heads=4,
            vocab_size=40,
            enable_distillation=False
        )

    def test_model_initialization(self):
        """Test that model initializes correctly."""
        logger.info("🧪 Testing AudioPhonemeASR Initialization")

        # Model should initialize successfully
        self.assertIsNotNone(self.model, "Model should initialize")
        self.assertEqual(self.model.embed_dim, 256, "Embed dim should be 256")
        self.assertEqual(self.model.vocab_size, 40, "Vocab size should be 40")

        logger.info("   ✅ Model initializes correctly")

    def test_audio_frontend_present(self):
        """Test that audio frontend is initialized."""
        logger.info("🧪 Testing Audio Frontend Component")

        # Audio frontend should be present
        self.assertIsNotNone(self.model.audio_frontend, "Audio frontend should be initialized")
        self.assertTrue(hasattr(self.model.audio_frontend, 'mel_extractor'), "Should have mel extractor")

        logger.info("   ✅ Audio frontend present and functional")

    def test_conformer_encoder_present(self):
        """Test that Conformer encoder is initialized."""
        logger.info("🧪 Testing Conformer Encoder")

        # Conformer should be present
        self.assertIsNotNone(self.model.conformer, "Conformer should be initialized")
        self.assertEqual(len(self.model.conformer.layers), 4, "Should have 4 layers")

        logger.info("   ✅ Conformer encoder present")

    def test_forward_pass_with_audio(self):
        """Test forward pass with audio input."""
        logger.info("🧪 Testing Forward Pass with Audio")

        self.model.eval()

        # Create test audio (16kHz, 1 second)
        batch_size = 2
        audio_len = 16000
        audio = torch.randn(batch_size, audio_len)
        audio_lengths = torch.tensor([audio_len, audio_len // 2])

        with torch.no_grad():
            outputs = self.model(audio=audio, audio_lengths=audio_lengths)

        # Check outputs
        self.assertIn('ctc_logits', outputs, "Should have CTC logits")
        self.assertIn('encoder_outputs', outputs, "Should have encoder outputs")

        # Check shapes
        ctc_logits = outputs['ctc_logits']
        self.assertEqual(ctc_logits.shape[0], batch_size, "Batch size should match")
        self.assertEqual(ctc_logits.shape[-1], 40, "Vocab size should be 40")

        logger.info(f"   ✅ Forward pass successful: {audio.shape} -> {ctc_logits.shape}")

    def test_greedy_decoding(self):
        """Test greedy CTC decoding."""
        logger.info("🧪 Testing Greedy Decoding")

        self.model.eval()

        # Create test audio
        audio = torch.randn(2, 16000)

        decoded, confidences = self.model.decode_greedy(audio)

        # Check outputs
        self.assertEqual(len(decoded), 2, "Should have 2 decoded sequences")
        self.assertEqual(len(confidences), 2, "Should have 2 confidence scores")

        logger.info(f"   ✅ Greedy decoding works: {len(decoded)} sequences")

    def test_model_size_info(self):
        """Test model size information."""
        logger.info("🧪 Testing Model Size Information")

        size_info = self.model.get_model_size()

        # Check structure
        self.assertIn('total_parameters', size_info)
        self.assertIn('trainable_parameters', size_info)
        self.assertIn('model_size_mb_fp16', size_info)

        total_params = size_info['total_parameters']
        logger.info(f"   ✅ Model size: {total_params:,} parameters")

    def test_distillation_loss_computation(self):
        """Test distillation loss computation."""
        logger.info("🧪 Testing Distillation Loss Computation")

        self.model.eval()

        # Create test inputs
        batch_size = 2
        seq_len = 50
        encoder_outputs = torch.randn(batch_size, seq_len, 256)
        ctc_logits = torch.randn(batch_size, seq_len, 40)
        audio_waveforms = torch.randn(batch_size, 16000)

        # Compute distillation loss (should return zero when disabled)
        distill_outputs = self.model.compute_distillation_loss(
            encoder_outputs=encoder_outputs,
            ctc_logits=ctc_logits,
            audio_waveforms=audio_waveforms
        )

        # Should return zero loss when distillation disabled
        self.assertEqual(distill_outputs['distillation_loss'].item(), 0.0,
                        "Distillation loss should be zero when disabled")

        logger.info("   ✅ Distillation loss computation works")


def run_tests():
    """Run all tests."""

    logger.info("🚀 Running AudioPhonemeASR Component Tests")
    logger.info("=" * 60)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAudioPhonemeASR)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    # Summary
    logger.info("=" * 60)
    if result.wasSuccessful():
        logger.info("🎉 All AudioPhonemeASR tests passed!")
        logger.info(f"✅ Ran {result.testsRun} tests successfully")
    else:
        logger.error(f"❌ {len(result.failures)} test(s) failed")
        logger.error(f"❌ {len(result.errors)} test(s) had errors")

        for test, error in result.failures + result.errors:
            logger.error(f"   ❌ {test}: {error}")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
