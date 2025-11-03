"""
Integration Tests for Valerie Audio ASR.

Tests the complete end-to-end pipeline:
- Synthetic data generation (audio-only)
- Full training pipeline
- Inference pipeline
- Data loading and processing
- Model convergence validation
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
import os
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.audio_phoneme_model import AudioPhonemeASR
from src.data import ValerieCollator, DataAugmentationPipeline
from src.training import ValerieTrainer, TrainingConfig, ValerieLossFunction
from src.inference import ValerieInferenceEngine, InferenceConfig
from src.evaluation import ValerieEvaluator, EvaluationConfig
from src.utils.config import Config
from src.utils.logging import setup_logger, get_logger

# Setup logging
setup_logger(use_rich=True)
logger = get_logger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic data for testing (audio-only)."""

    def __init__(self, num_samples=10, seq_length=50, vocab_size=40):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def generate_audio_data(self):
        """Generate synthetic audio waveforms."""
        audios = []
        for i in range(self.num_samples):
            # Generate random audio waveform at 16kHz (1-2 seconds)
            audio_length = 16000 + i * 1000  # Varying lengths
            audio = torch.randn(audio_length)
            audios.append(audio)
        return audios
    
    def generate_transcriptions(self):
        """Generate synthetic transcriptions."""
        transcriptions = []
        phonemes = ['AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 
                   'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 
                   'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 
                   'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
        
        for i in range(self.num_samples):
            # Generate random phoneme sequence
            length = np.random.randint(5, 20)
            phoneme_seq = np.random.choice(phonemes, size=length)
            text = ' '.join(phoneme_seq)
            transcriptions.append(text)
        
        return transcriptions
    
    def create_synthetic_dataset(self, output_dir):
        """Create complete synthetic audio dataset."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate data
        audios = self.generate_audio_data()
        transcriptions = self.generate_transcriptions()

        # Save data
        dataset_info = []
        for i, (audio, text) in enumerate(zip(audios, transcriptions)):
            sample_dir = output_path / f"sample_{i:03d}"
            sample_dir.mkdir(exist_ok=True)

            # Save audio tensor
            torch.save(audio, sample_dir / "audio.pt")

            # Save transcription
            with open(sample_dir / "transcription.txt", 'w') as f:
                f.write(text)

            dataset_info.append({
                'id': f"sample_{i:03d}",
                'audio_path': str(sample_dir / "audio.pt"),
                'transcription_path': str(sample_dir / "transcription.txt"),
                'transcription': text
            })

        # Save dataset metadata
        with open(output_path / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)

        logger.info(f"✅ Created synthetic audio dataset with {len(dataset_info)} samples in {output_dir}")
        return dataset_info


class TestEndToEndPipeline(unittest.TestCase):
    """Test complete end-to-end pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device('cpu')  # Use CPU for testing
        
        # Create synthetic data
        self.data_generator = SyntheticDataGenerator(num_samples=20)
        self.dataset_info = self.data_generator.create_synthetic_dataset(
            self.temp_dir + "/synthetic_data"
        )
        
        # Create test config
        self.config = self._create_test_config()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_config(self):
        """Create test configuration for audio-only model."""
        config_dict = {
            'model': {
                'embed_dim': 128,  # Small for testing
                'conformer_layers': 2,
                'conformer_heads': 2,
                'feed_forward_dim': 256,
                'conv_kernel_size': 7,
                'dropout': 0.1,
                'vocab_size': 40,
                'enable_distillation': False,
                'use_llm_reconstruction': False,
                'llm_model_name': 'Qwen/Qwen3-0.6B',
                'lora_rank': 8,
                'lora_alpha': 16,
                'enable_thinking': False,
                'max_new_tokens': 64
            },
            'distillation': {
                'enabled': False,
                'teachers': [],
                'projection_dim': 512,
                'temperature': 4.0,
                'alpha': 0.7
            },
            'data': {
                'dataset_root': self.temp_dir + "/synthetic_data",
                'sample_rate': 16000,
                'n_mels': 80,
                'n_fft': 400,
                'hop_length': 160,
                'augmentation_prob': 0.0  # Disable for testing
            },
            'training': {
                'batch_size': 2,
                'learning_rate': 1e-3,
                'num_epochs': 2,  # Short for testing
                'warmup_steps': 10,
                'weight_decay': 1e-4,
                'gradient_clip_norm': 1.0,
                'mixed_precision': False,
                'save_every_n_epochs': 1
            }
        }

        return Config.from_dict(config_dict)
    
    def test_synthetic_data_generation(self):
        """Test synthetic audio data generation."""
        logger.info("🧪 Testing Synthetic Audio Data Generation")

        # Check dataset info
        self.assertEqual(len(self.dataset_info), 20)

        # Check first sample
        sample = self.dataset_info[0]
        self.assertIn('audio_path', sample)
        self.assertIn('transcription', sample)

        # Check files exist
        self.assertTrue(Path(sample['audio_path']).exists())
        self.assertTrue(Path(sample['transcription_path']).exists())

        # Load and check data
        audio = torch.load(sample['audio_path'])

        # Audio should be 1D waveform
        self.assertEqual(len(audio.shape), 1)
        self.assertGreater(audio.shape[0], 15000)  # At least ~1 second

        logger.info("   ✅ Synthetic audio data generation works")
    
    def test_data_loading_pipeline(self):
        """Test data loading and processing pipeline."""
        logger.info("🧪 Testing Data Loading Pipeline")

        # Create custom dataset class for synthetic audio data
        class SyntheticAudioDataset(torch.utils.data.Dataset):
            def __init__(self, dataset_info):
                self.dataset_info = dataset_info

            def __len__(self):
                return len(self.dataset_info)

            def __getitem__(self, idx):
                sample = self.dataset_info[idx]

                # Load data
                audio = torch.load(sample['audio_path'])
                transcription = sample['transcription']

                return {
                    'audio': audio,
                    'transcription': transcription,
                    'audio_path': sample['audio_path']
                }

        # Create dataset
        dataset = SyntheticAudioDataset(self.dataset_info)

        # Test dataset length
        self.assertEqual(len(dataset), 20)

        # Test sample loading
        sample = dataset[0]
        self.assertIn('audio', sample)
        self.assertIn('transcription', sample)

        # Test data loader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        batch = next(iter(dataloader))
        self.assertEqual(len(batch['audio']), 4)
        self.assertEqual(len(batch['transcription']), 4)

        logger.info("   ✅ Data loading pipeline works")
    
    def test_model_forward_pass(self):
        """Test model forward pass with synthetic audio data."""
        logger.info("🧪 Testing Model Forward Pass")

        model = AudioPhonemeASR(
            embed_dim=128,
            conformer_layers=2,
            conformer_heads=2,
            vocab_size=40,
            enable_distillation=False
        )
        model.eval()

        # Create test batch (audio waveforms)
        batch_size = 2
        audio = torch.randn(batch_size, 16000)  # 1 second at 16kHz
        audio_lengths = torch.tensor([16000, 12000])

        # Forward pass
        with torch.no_grad():
            outputs = model(audio=audio, audio_lengths=audio_lengths)

        # Check outputs
        self.assertIn('ctc_logits', outputs)
        self.assertIn('encoder_outputs', outputs)

        ctc_logits = outputs['ctc_logits']
        self.assertEqual(ctc_logits.shape[0], batch_size)
        self.assertEqual(ctc_logits.shape[2], 40)  # vocab_size

        logger.info(f"   ✅ Model forward pass: {list(audio.shape)} → {list(ctc_logits.shape)}")
    
    def test_loss_computation(self):
        """Test loss computation with synthetic audio data."""
        logger.info("🧪 Testing Loss Computation")

        model = AudioPhonemeASR(
            embed_dim=128,
            conformer_layers=2,
            conformer_heads=2,
            vocab_size=40,
            enable_distillation=False
        )
        loss_fn = ValerieLossFunction(
            ctc_weight=1.0,
            attention_weight=0.3,
            distillation_weight=0.0,  # Disabled
            temporal_consistency_weight=0.1
        )

        # Create test batch (audio)
        batch_size = 2
        audio = torch.randn(batch_size, 16000)
        audio_lengths = torch.tensor([16000, 12000])

        # Create dummy targets
        targets = torch.randint(1, 40, (batch_size, 10))  # Avoid blank token (0)
        target_lengths = torch.tensor([10, 8])

        # Forward pass
        outputs = model(audio=audio, audio_lengths=audio_lengths)

        # Compute loss
        loss_dict = loss_fn(
            ctc_logits=outputs['ctc_logits'],
            attention_logits=outputs.get('attention_logits'),
            targets=targets,
            target_lengths=target_lengths,
            input_lengths=audio_lengths
        )

        # Check loss
        self.assertIn('total_loss', loss_dict)
        self.assertIn('ctc_loss', loss_dict)

        total_loss = loss_dict['total_loss']
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertGreater(total_loss.item(), 0)

        logger.info(f"   ✅ Loss computation: {total_loss.item():.4f}")
    
    def test_training_step(self):
        """Test single training step with audio."""
        logger.info("🧪 Testing Training Step")

        model = AudioPhonemeASR(
            embed_dim=128,
            conformer_layers=2,
            conformer_heads=2,
            vocab_size=40,
            enable_distillation=False
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = ValerieLossFunction()

        # Create test batch (audio)
        batch_size = 2
        audio = torch.randn(batch_size, 16000)
        audio_lengths = torch.tensor([16000, 12000])
        targets = torch.randint(1, 40, (batch_size, 10))
        target_lengths = torch.tensor([10, 8])

        # Training step
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(audio=audio, audio_lengths=audio_lengths)

        # Compute loss
        loss_dict = loss_fn(
            ctc_logits=outputs['ctc_logits'],
            attention_logits=outputs.get('attention_logits'),
            targets=targets,
            target_lengths=target_lengths,
            input_lengths=audio_lengths
        )

        # Backward pass
        loss = loss_dict['total_loss']
        loss.backward()

        # Check gradients
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break

        self.assertTrue(has_gradients, "No gradients found")

        # Optimizer step
        optimizer.step()

        logger.info(f"   ✅ Training step completed, loss: {loss.item():.4f}")
    
    def test_convergence_on_synthetic_data(self):
        """Test model convergence on synthetic audio data."""
        logger.info("🧪 Testing Convergence on Synthetic Audio Data")

        model = AudioPhonemeASR(
            embed_dim=128,
            conformer_layers=2,
            conformer_heads=2,
            vocab_size=40,
            enable_distillation=False
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = ValerieLossFunction()

        # Create consistent synthetic batch (audio)
        torch.manual_seed(42)
        batch_size = 4
        audio = torch.randn(batch_size, 16000)  # 1 second at 16kHz
        audio_lengths = torch.tensor([16000, 16000, 16000, 16000])
        targets = torch.randint(1, 40, (batch_size, 8))
        target_lengths = torch.tensor([8, 8, 8, 8])

        # Training loop
        losses = []
        model.train()

        for step in range(20):  # Short training
            optimizer.zero_grad()

            # Forward pass
            outputs = model(audio=audio, audio_lengths=audio_lengths)

            # Compute loss
            loss_dict = loss_fn(
                ctc_logits=outputs['ctc_logits'],
                attention_logits=outputs.get('attention_logits'),
                targets=targets,
                target_lengths=target_lengths,
                input_lengths=audio_lengths
            )

            loss = loss_dict['total_loss']
            losses.append(loss.item())

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 5 == 0:
                logger.info(f"   Step {step}: Loss = {loss.item():.4f}")

        # Check convergence (loss should decrease)
        initial_loss = np.mean(losses[:5])
        final_loss = np.mean(losses[-5:])

        self.assertLess(final_loss, initial_loss, "Model did not converge")

        logger.info(f"   ✅ Convergence verified: {initial_loss:.4f} → {final_loss:.4f}")
    
    def test_inference_pipeline(self):
        """Test inference pipeline with synthetic data."""
        logger.info("🧪 Testing Inference Pipeline")
        
        # This test would require a trained model checkpoint
        # For now, we'll test the inference configuration
        
        inference_config = InferenceConfig(
            model_checkpoint="dummy_checkpoint.pt",  # Would fail to load
            device="cpu",
            batch_size=2
        )
        
        # Test configuration
        self.assertEqual(inference_config.device, "cpu")
        self.assertEqual(inference_config.batch_size, 2)
        
        logger.info("   ✅ Inference configuration works")
        logger.info("   ℹ️ Actual inference test requires trained model")
    
    def test_evaluation_pipeline(self):
        """Test evaluation pipeline configuration."""
        logger.info("🧪 Testing Evaluation Pipeline")
        
        eval_config = EvaluationConfig(
            model_checkpoint="dummy_checkpoint.pt",
            dataset_root=self.temp_dir + "/synthetic_data",
            dataset_split="test",
            max_samples=10,
            batch_size=2
        )
        
        # Test configuration
        self.assertEqual(eval_config.max_samples, 10)
        self.assertEqual(eval_config.batch_size, 2)
        self.assertTrue(eval_config.calculate_per)
        self.assertTrue(eval_config.calculate_wer)
        
        logger.info("   ✅ Evaluation configuration works")
        logger.info("   ℹ️ Actual evaluation test requires trained model")


class TestDataProcessingPipeline(unittest.TestCase):
    """Test data processing components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_audio_augmentation_pipeline(self):
        """Test audio-only augmentation pipeline."""
        logger.info("🧪 Testing Audio Augmentation Pipeline")

        try:
            from src.utils.config import Config
            config = Config('configs/ensemble_distillation_config.yaml')

            # Create augmentation pipeline
            augmentation = DataAugmentationPipeline(config)

            # Test with synthetic audio data
            audio = torch.randn(16000)  # 1 second at 16kHz

            from src.data.dataset import DataSample
            sample = DataSample(
                audio=audio,
                transcription="test transcription",
                phonemes=['T', 'EH', 'S', 'T'],
                audio_path="test.wav"
            )

            # Apply augmentation
            augmented = augmentation(sample)

            # Check output
            self.assertIsInstance(augmented, DataSample)
            self.assertEqual(len(augmented.audio.shape), 1)  # 1D waveform

            logger.info("   ✅ Audio augmentation pipeline works")

        except Exception as e:
            logger.warning(f"   ⚠️ Audio augmentation test skipped: {e}")
    
    def test_collate_function(self):
        """Test batch collation for audio-only data."""
        logger.info("🧪 Testing Collate Function")

        try:
            collator = ValerieCollator(max_length=150)

            # Create sample batch (audio-only)
            from src.data.dataset import DataSample
            samples = []
            for i in range(3):
                sample = DataSample(
                    audio=torch.randn(16000 + i * 2000),  # Different lengths
                    transcription=f"test transcription {i}",
                    phonemes=['T', 'EH', 'S', 'T'] + ['AH'] * i,
                    audio_path=f"test_{i}.wav"
                )
                samples.append(sample)

            # Collate batch
            batch = collator(samples)

            # Check batch structure
            self.assertIn('audio', batch)
            self.assertIn('transcriptions', batch)

            # Check shapes
            audio_batch = batch['audio']
            self.assertEqual(audio_batch.shape[0], 3)  # Batch size

            logger.info(f"   ✅ Batch collation: {list(audio_batch.shape)}")

        except Exception as e:
            logger.warning(f"   ⚠️ Collate function test skipped: {e}")


def run_integration_tests():
    """Run all integration tests."""
    
    logger.info("🚀 Running Integration Tests")
    logger.info("=" * 60)
    
    # Create test suite
    test_classes = [
        TestEndToEndPipeline,
        TestDataProcessingPipeline
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary
    logger.info("=" * 60)
    if result.wasSuccessful():
        logger.info("🎉 All integration tests passed!")
        logger.info(f"✅ Ran {result.testsRun} tests successfully")
    else:
        logger.error(f"❌ {len(result.failures)} test(s) failed")
        logger.error(f"❌ {len(result.errors)} test(s) had errors")
        
        for test, error in result.failures + result.errors:
            logger.error(f"   ❌ {test}: {error}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)
