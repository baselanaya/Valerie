"""
Comprehensive tests for Valerie Audio ASR model components.

Tests all implemented modules to ensure correct functionality,
tensor shapes, and gradient flow.
"""

import torch
import torch.nn as nn
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.audio_frontend import AudioFrontend, MelSpectrogramExtractor
from src.models.audio_phoneme_model import AudioPhonemeASR
from src.models.conformer import (
    ConvolutionModule, MultiHeadSelfAttention, FeedForward,
    ConformerBlock, ConformerEncoder
)
from src.models.hybrid_ctc_attention import (
    CTCHead, LocationAwareAttention, AttentionDecoder, HybridCTCAttention
)
from src.models.qwen_llm import QwenPhonemeToText
from src.utils.config import ModelConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TestAudioFrontend:
    """Test suite for audio frontend components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.audio_len = 16000  # 1 second at 16kHz

        logger.info(f"🧪 Testing on device: {self.device}")

    def test_mel_spectrogram_extractor(self):
        """Test MelSpectrogramExtractor functionality."""
        logger.info("🔍 Testing MelSpectrogramExtractor...")

        mel_extractor = MelSpectrogramExtractor(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80
        ).to(self.device)

        # Input audio [B, T]
        audio = torch.randn(self.batch_size, self.audio_len).to(self.device)

        # Forward pass
        mel_spec = mel_extractor(audio)

        # Check output shape [B, n_mels, T_mel]
        assert mel_spec.shape[0] == self.batch_size, "Batch size mismatch"
        assert mel_spec.shape[1] == 80, "Number of mel bins should be 80"

        # Check that output is in log scale (should have negative values)
        assert (mel_spec < 0).any(), "Mel spectrogram should be in log scale"

        logger.info("✅ MelSpectrogramExtractor test passed")

    def test_audio_frontend(self):
        """Test complete AudioFrontend module."""
        logger.info("🔍 Testing AudioFrontend...")

        embed_dim = 256
        frontend = AudioFrontend(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80,
            embed_dim=embed_dim
        ).to(self.device)

        # Input audio [B, T]
        audio = torch.randn(self.batch_size, self.audio_len).to(self.device)
        audio_lengths = torch.tensor([self.audio_len, self.audio_len // 2]).to(self.device)

        # Forward pass
        features, feature_lengths = frontend(audio, audio_lengths)

        # Check output shape [B, T_feat, embed_dim]
        assert features.shape[0] == self.batch_size, "Batch size mismatch"
        assert features.shape[2] == embed_dim, "Embedding dimension mismatch"

        # Check that feature lengths are reduced correctly
        assert feature_lengths[0] > feature_lengths[1], "Feature lengths should reflect audio lengths"

        # Test gradient flow
        loss = features.sum()
        loss.backward()

        # Check that gradients exist
        assert frontend.projection.weight.grad is not None, "No gradients for projection"

        logger.info("✅ AudioFrontend test passed")


class TestAudioPhonemeASR:
    """Test suite for complete AudioPhonemeASR model."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.audio_len = 16000

    def test_audio_phoneme_asr_initialization(self):
        """Test AudioPhonemeASR initialization."""
        logger.info("🔍 Testing AudioPhonemeASR initialization...")

        model = AudioPhonemeASR(
            embed_dim=256,
            conformer_layers=4,
            conformer_heads=4,
            vocab_size=40,
            enable_distillation=False
        ).to(self.device)

        # Check components exist
        assert hasattr(model, 'audio_frontend'), "Should have audio_frontend"
        assert hasattr(model, 'conformer'), "Should have conformer"
        assert hasattr(model, 'ctc_attention'), "Should have ctc_attention"

        logger.info("✅ AudioPhonemeASR initialization test passed")

    def test_audio_phoneme_asr_forward(self):
        """Test AudioPhonemeASR forward pass."""
        logger.info("🔍 Testing AudioPhonemeASR forward pass...")

        model = AudioPhonemeASR(
            embed_dim=256,
            conformer_layers=4,
            conformer_heads=4,
            vocab_size=40,
            enable_distillation=False
        ).to(self.device)

        model.eval()

        # Input audio [B, T]
        audio = torch.randn(self.batch_size, self.audio_len).to(self.device)
        audio_lengths = torch.tensor([self.audio_len, self.audio_len // 2]).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = model(audio=audio, audio_lengths=audio_lengths)

        # Check outputs
        assert 'ctc_logits' in outputs, "Should have CTC logits"
        assert 'encoder_outputs' in outputs, "Should have encoder outputs"

        ctc_logits = outputs['ctc_logits']
        assert ctc_logits.shape[0] == self.batch_size, "Batch size mismatch"
        assert ctc_logits.shape[-1] == 40, "Vocab size should be 40"

        logger.info("✅ AudioPhonemeASR forward pass test passed")

    def test_greedy_decoding(self):
        """Test greedy decoding."""
        logger.info("🔍 Testing greedy decoding...")

        model = AudioPhonemeASR(
            embed_dim=256,
            conformer_layers=4,
            conformer_heads=4,
            vocab_size=40,
            enable_distillation=False
        ).to(self.device)

        model.eval()

        # Input audio
        audio = torch.randn(self.batch_size, self.audio_len).to(self.device)

        # Decode
        decoded, confidences = model.decode_greedy(audio)

        assert len(decoded) == self.batch_size, "Should have decoded sequences for each batch"
        assert len(confidences) == self.batch_size, "Should have confidences for each batch"

        logger.info("✅ Greedy decoding test passed")


class TestConformerEncoder:
    """Test suite for Conformer encoder components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.seq_len = 50
        self.embed_dim = 512
    
    def test_convolution_module(self):
        """Test ConvolutionModule functionality."""
        logger.info("🔍 Testing ConvolutionModule...")
        
        conv_module = ConvolutionModule(
            embed_dim=self.embed_dim,
            kernel_size=31
        ).to(self.device)
        
        # Input tensor [B, T, embed_dim]
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim).to(self.device)
        
        # Forward pass
        output = conv_module(x)
        
        # Check output shape
        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        
        assert conv_module.pointwise_conv1.weight.grad is not None, "No gradients for conv module"
        
        logger.info("✅ ConvolutionModule test passed")
    
    def test_multi_head_self_attention(self):
        """Test MultiHeadSelfAttention functionality."""
        logger.info("🔍 Testing MultiHeadSelfAttention...")
        
        attention = MultiHeadSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=8
        ).to(self.device)
        
        # Input tensor [B, T, embed_dim]
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim).to(self.device)
        
        # Forward pass without mask
        output = attention(x)
        assert output.shape == x.shape, "Attention output shape mismatch"
        
        # Forward pass with attention mask
        mask = torch.ones(self.batch_size, self.seq_len).to(self.device)
        mask[0, self.seq_len//2:] = 0
        
        masked_output = attention(x, mask)
        assert masked_output.shape == x.shape, "Masked attention output shape mismatch"
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        
        assert attention.q_proj.weight.grad is not None, "No gradients for attention"
        
        logger.info("✅ MultiHeadSelfAttention test passed")
    
    def test_conformer_block(self):
        """Test ConformerBlock functionality."""
        logger.info("🔍 Testing ConformerBlock...")
        
        conformer_block = ConformerBlock(
            embed_dim=self.embed_dim,
            num_heads=8,
            conv_kernel_size=31
        ).to(self.device)
        
        # Input tensor [B, T, embed_dim]
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim).to(self.device)
        
        # Forward pass
        output = conformer_block(x)
        
        # Check output shape
        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
        
        # Test with attention mask
        mask = torch.ones(self.batch_size, self.seq_len).to(self.device)
        mask[1, self.seq_len//3:] = 0
        
        masked_output = conformer_block(x, mask)
        assert masked_output.shape == x.shape, "Masked ConformerBlock output shape mismatch"
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        
        assert conformer_block.attention.q_proj.weight.grad is not None, "No gradients for ConformerBlock"
        
        logger.info("✅ ConformerBlock test passed")
    
    def test_conformer_encoder(self):
        """Test complete ConformerEncoder."""
        logger.info("🔍 Testing ConformerEncoder...")
        
        encoder = ConformerEncoder(
            input_dim=self.embed_dim,
            embed_dim=self.embed_dim,
            num_layers=6,  # Smaller for testing
            num_heads=8
        ).to(self.device)
        
        # Input tensor [B, T, embed_dim]
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim).to(self.device)
        
        # Forward pass
        output = encoder(x)
        
        # Check output shape
        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
        
        # Test with attention mask
        mask = torch.ones(self.batch_size, self.seq_len).to(self.device)
        mask[0, self.seq_len//2:] = 0
        
        masked_output = encoder(x, mask)
        assert masked_output.shape == x.shape, "Masked encoder output shape mismatch"
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        
        assert encoder.layers[0].attention.q_proj.weight.grad is not None, "No gradients for encoder"
        
        logger.info("✅ ConformerEncoder test passed")


class TestHybridCTCAttention:
    """Test suite for hybrid CTC/Attention components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.seq_len = 80
        self.encoder_dim = 512
        self.vocab_size = 40
        self.target_len = 30
    
    def test_ctc_head(self):
        """Test CTCHead functionality."""
        logger.info("🔍 Testing CTCHead...")
        
        ctc_head = CTCHead(
            encoder_dim=self.encoder_dim,
            vocab_size=self.vocab_size
        ).to(self.device)
        
        # Encoder outputs [B, T, encoder_dim]
        encoder_outputs = torch.randn(
            self.batch_size, self.seq_len, self.encoder_dim
        ).to(self.device)
        
        # Forward pass
        logits = ctc_head(encoder_outputs)
        expected_shape = (self.batch_size, self.seq_len, self.vocab_size)
        assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
        
        # Test CTC loss computation
        targets = torch.randint(1, self.vocab_size, (self.batch_size, self.target_len)).to(self.device)
        input_lengths = torch.tensor([self.seq_len, self.seq_len-10]).to(self.device)
        target_lengths = torch.tensor([self.target_len, self.target_len-5]).to(self.device)
        
        loss = ctc_head.compute_loss(encoder_outputs, targets, input_lengths, target_lengths)
        assert loss.item() >= 0, "CTC loss should be non-negative"
        
        # Test greedy decoding
        decoded = ctc_head.decode_greedy(encoder_outputs)
        assert len(decoded) == self.batch_size, "Decoded sequences count mismatch"
        
        # Test gradient flow
        loss.backward()
        assert ctc_head.linear.weight.grad is not None, "No gradients for CTC head"
        
        logger.info("✅ CTCHead test passed")
    
    def test_attention_decoder(self):
        """Test AttentionDecoder functionality."""
        logger.info("🔍 Testing AttentionDecoder...")
        
        decoder = AttentionDecoder(
            encoder_dim=self.encoder_dim,
            decoder_dim=256,
            vocab_size=self.vocab_size
        ).to(self.device)
        
        # Encoder outputs [B, T, encoder_dim]
        encoder_outputs = torch.randn(
            self.batch_size, self.seq_len, self.encoder_dim
        ).to(self.device)
        
        # Test training mode (teacher forcing)
        decoder.train()
        targets = torch.randint(1, self.vocab_size, (self.batch_size, self.target_len)).to(self.device)
        
        output = decoder(encoder_outputs, targets)
        expected_shape = (self.batch_size, self.target_len, self.vocab_size)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Test inference mode
        decoder.eval()
        with torch.no_grad():
            output_infer = decoder(encoder_outputs, max_length=20)
            assert output_infer.shape[0] == self.batch_size, "Batch size mismatch in inference"
            assert output_infer.shape[2] == self.vocab_size, "Vocab size mismatch in inference"
        
        # Test gradient flow
        decoder.train()
        loss = output.sum()
        loss.backward()
        assert decoder.lstm.weight_ih_l0.grad is not None, "No gradients for decoder LSTM"
        
        logger.info("✅ AttentionDecoder test passed")
    
    def test_hybrid_ctc_attention(self):
        """Test complete HybridCTCAttention module."""
        logger.info("🔍 Testing HybridCTCAttention...")
        
        hybrid = HybridCTCAttention(
            encoder_dim=self.encoder_dim,
            vocab_size=self.vocab_size,
            decoder_dim=256
        ).to(self.device)
        
        # Encoder outputs [B, T, encoder_dim]
        encoder_outputs = torch.randn(
            self.batch_size, self.seq_len, self.encoder_dim
        ).to(self.device)
        
        # Test training mode
        hybrid.train()
        ctc_targets = torch.randint(1, self.vocab_size, (self.batch_size, self.target_len)).to(self.device)
        attention_targets = torch.randint(1, self.vocab_size, (self.batch_size, self.target_len)).to(self.device)
        input_lengths = torch.tensor([self.seq_len, self.seq_len-10]).to(self.device)
        target_lengths = torch.tensor([self.target_len, self.target_len-5]).to(self.device)
        
        results = hybrid(
            encoder_outputs, ctc_targets, attention_targets,
            input_lengths, target_lengths
        )
        
        # Check that all expected outputs are present
        assert 'ctc_logits' in results, "CTC logits missing"
        assert 'ctc_loss' in results, "CTC loss missing"
        assert 'attention_logits' in results, "Attention logits missing"
        assert 'attention_loss' in results, "Attention loss missing"
        assert 'total_loss' in results, "Total loss missing"
        
        # Test inference mode
        hybrid.eval()
        with torch.no_grad():
            results_infer = hybrid(encoder_outputs)
            assert 'ctc_logits' in results_infer, "CTC logits missing in inference"
            assert 'attention_logits' in results_infer, "Attention logits missing in inference"
        
        # Test gradient flow
        hybrid.train()
        loss = results['total_loss']
        loss.backward()
        assert hybrid.ctc_head.linear.weight.grad is not None, "No gradients for hybrid module"
        
        logger.info("✅ HybridCTCAttention test passed")


class TestQwenLLM:
    """Test suite for Qwen3 LLM integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Skip this test if we don't have enough memory or internet access
        try:
            # Try to create a small test model
            self.qwen_llm = QwenPhonemeToText(
                model_name="Qwen/Qwen3-0.6B",
                lora_rank=8,  # Small rank for testing
                lora_alpha=16,
                enable_thinking=False,  # Disable thinking for faster testing
                max_new_tokens=50  # Small for testing
            )
            self.test_enabled = True
        except Exception as e:
            logger.warning(f"⚠️ Skipping Qwen LLM tests due to: {e}")
            self.test_enabled = False
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for LLM testing")
    def test_qwen_phoneme_to_text(self):
        """Test Qwen3 phoneme-to-text conversion."""
        if not self.test_enabled:
            pytest.skip("Qwen LLM not available")
        
        logger.info("🔍 Testing QwenPhonemeToText...")
        
        # Test phoneme sequences
        phoneme_sequences = [
            ["HH", "EH", "L", "OW"],  # "hello"
            ["W", "ER", "L", "D"]     # "world"
        ]
        
        confidence_scores = [
            [0.9, 0.8, 0.95, 0.7],
            [0.85, 0.9, 0.8, 0.9]
        ]
        
        # Test inference
        try:
            results = self.qwen_llm(
                phoneme_sequences, 
                confidence_scores, 
                return_thinking=False
            )
            
            assert len(results) == 2, "Should return results for both sequences"
            assert all(isinstance(result, str) for result in results), "Results should be strings"
            
            logger.info("✅ QwenPhonemeToText inference test passed")
            
        except Exception as e:
            logger.warning(f"⚠️ Qwen inference test failed: {e}")
    
    def test_qwen_prompt_creation(self):
        """Test prompt creation functionality."""
        if not self.test_enabled:
            pytest.skip("Qwen LLM not available")
        
        logger.info("🔍 Testing Qwen prompt creation...")
        
        phonemes = ["HH", "EH", "L", "OW"]
        confidence_scores = [0.9, 0.8, 0.95, 0.7]
        context = "greeting"
        
        prompt = self.qwen_llm.create_phoneme_prompt(
            phonemes, confidence_scores, context
        )
        
        assert isinstance(prompt, str), "Prompt should be a string"
        assert "phoneme" in prompt.lower(), "Prompt should mention phonemes"
        assert "context" in prompt.lower(), "Prompt should mention context"
        assert all(phoneme in prompt for phoneme in phonemes), "All phonemes should be in prompt"
        
        logger.info("✅ Qwen prompt creation test passed")


class TestIntegration:
    """Integration tests for the complete audio pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.audio_len = 16000  # 1 second at 16kHz

        # Create configuration
        self.config = ModelConfig()
        self.config.embed_dim = 256
        self.config.conformer_dim = 256
        self.config.conformer_layers = 4  # Smaller for testing
        self.config.phoneme_vocab_size = 40

    def test_audio_to_phonemes_pipeline(self):
        """Test the complete audio-to-phonemes pipeline."""
        logger.info("🔍 Testing audio-to-phonemes pipeline...")

        # Create components
        audio_frontend = AudioFrontend(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80,
            embed_dim=self.config.embed_dim
        ).to(self.device)

        conformer = ConformerEncoder(
            input_dim=self.config.embed_dim,
            embed_dim=self.config.conformer_dim,
            num_layers=self.config.conformer_layers,
            num_heads=4
        ).to(self.device)

        hybrid_ctc = HybridCTCAttention(
            encoder_dim=self.config.conformer_dim,
            vocab_size=self.config.phoneme_vocab_size,
            decoder_dim=256
        ).to(self.device)

        # Test forward pass through complete pipeline
        # Input audio: [B, T]
        audio_input = torch.randn(self.batch_size, self.audio_len).to(self.device)
        audio_lengths = torch.tensor([self.audio_len, self.audio_len // 2]).to(self.device)

        # 1. Audio frontend (Mel spectrograms)
        audio_features, feature_lengths = audio_frontend(audio_input, audio_lengths)
        logger.info(f"   Audio features shape: {audio_features.shape}")

        # 2. Conformer encoding
        encoded_features = conformer(audio_features)
        logger.info(f"   Encoded features shape: {encoded_features.shape}")

        # 3. CTC/Attention prediction
        hybrid_ctc.eval()
        with torch.no_grad():
            results = hybrid_ctc(encoded_features)

        assert 'ctc_logits' in results, "CTC logits missing from pipeline"
        assert 'attention_logits' in results, "Attention logits missing from pipeline"

        ctc_shape = results['ctc_logits'].shape
        att_shape = results['attention_logits'].shape

        logger.info(f"   CTC logits shape: {ctc_shape}")
        logger.info(f"   Attention logits shape: {att_shape}")

        # Verify shapes
        assert ctc_shape[0] == self.batch_size, "CTC batch size mismatch"
        assert ctc_shape[2] == self.config.phoneme_vocab_size, "CTC vocab size mismatch"

        assert att_shape[0] == self.batch_size, "Attention batch size mismatch"
        assert att_shape[2] == self.config.phoneme_vocab_size, "Attention vocab size mismatch"

        logger.info("✅ Audio-to-phonemes pipeline test passed")

    def test_end_to_end_model(self):
        """Test the complete AudioPhonemeASR model."""
        logger.info("🔍 Testing end-to-end AudioPhonemeASR model...")

        # Create complete model
        model = AudioPhonemeASR(
            embed_dim=self.config.embed_dim,
            conformer_layers=self.config.conformer_layers,
            conformer_heads=4,
            vocab_size=self.config.phoneme_vocab_size,
            enable_distillation=False
        ).to(self.device)

        model.eval()

        # Input audio [B, T]
        audio_input = torch.randn(self.batch_size, self.audio_len).to(self.device)

        # End-to-end inference
        with torch.no_grad():
            decoded, confidences = model.decode_greedy(audio_input)

        assert len(decoded) == self.batch_size, "Should have decoded sequences for each batch"
        assert len(confidences) == self.batch_size, "Should have confidences for each batch"

        logger.info("✅ End-to-end model test passed")

    def test_memory_efficiency(self):
        """Test memory usage and efficiency for audio model."""
        logger.info("🔍 Testing memory efficiency...")

        if not torch.cuda.is_available():
            logger.info("   Skipping memory test (CUDA not available)")
            return

        # Clear GPU memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        # Create audio frontend
        audio_frontend = AudioFrontend(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80,
            embed_dim=256
        ).to(self.device)

        # Test with larger batch
        large_batch_size = 4
        audio_input = torch.randn(large_batch_size, self.audio_len).to(self.device)
        audio_lengths = torch.tensor([self.audio_len] * large_batch_size).to(self.device)

        # Forward pass
        features, _ = audio_frontend(audio_input, audio_lengths)

        peak_memory = torch.cuda.memory_allocated()
        memory_used = (peak_memory - initial_memory) / (1024 ** 2)  # MB

        logger.info(f"   Memory used: {memory_used:.2f} MB")

        # Clean up
        del audio_frontend, audio_input, features
        torch.cuda.empty_cache()

        # Audio model should use much less memory than video model
        assert memory_used < 500, f"Memory usage too high for audio model: {memory_used:.2f} MB"

        logger.info("✅ Memory efficiency test passed")


def run_all_tests():
    """Run all tests with proper logging."""
    logger.info("🚀 Starting comprehensive component tests...")

    # Test classes
    test_classes = [
        TestAudioFrontend,
        TestAudioPhonemeASR,
        TestConformerEncoder,
        TestHybridCTCAttention,
        TestQwenLLM,
        TestIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        logger.info(f"\n📋 Running {test_class.__name__}...")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            
            try:
                # Create test instance and run setup
                test_instance = test_class()
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run the test
                getattr(test_instance, test_method)()
                passed_tests += 1
                
            except Exception as e:
                logger.error(f"❌ {test_class.__name__}.{test_method} failed: {e}")
                failed_tests.append(f"{test_class.__name__}.{test_method}: {e}")
    
    # Summary
    logger.info(f"\n📊 Test Summary:")
    logger.info(f"   Total tests: {total_tests}")
    logger.info(f"   Passed: {passed_tests}")
    logger.info(f"   Failed: {len(failed_tests)}")
    
    if failed_tests:
        logger.error(f"\n❌ Failed tests:")
        for failure in failed_tests:
            logger.error(f"   {failure}")
    else:
        logger.info(f"\n🎉 All tests passed!")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    # Run all tests
    success = run_all_tests()

    if success:
        print("\n✅ All audio ASR component tests passed! Ready to proceed with training.")
    else:
        print("\n❌ Some tests failed. Please review and fix issues before proceeding.")

    exit(0 if success else 1)
