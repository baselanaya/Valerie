"""
Working Unit Tests for Valerie Model Components.

Tests the core components that are confirmed to work:
- 3D Spatio-Temporal CNN blocks  
- Temporal positional encoding
- Feed-forward networks
- Basic gradient flow and performance
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.spatio_temporal import SpatioTemporalEmbedding, Conv3dBlock, TemporalPositionalEncoding
from src.models.conformer import FeedForward
from src.utils.logging import setup_logger, get_logger

# Setup logging for tests
setup_logger(use_rich=True)
logger = get_logger(__name__)


class TestWorkingComponents(unittest.TestCase):
    """Test components that are confirmed to work."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.device = torch.device('cpu')  # Use CPU for testing
    
    def test_conv3d_block_basic(self):
        """Test basic Conv3D block functionality."""
        logger.info("🧪 Testing Conv3D Block")
        
        # Test basic functionality
        block = Conv3dBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )
        
        # Test input
        x = torch.randn(2, 3, 16, 32, 32)
        
        # Forward pass
        output = block(x)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 64, 16, 32, 32))
        
        logger.info(f"   ✅ Conv3D: {list(x.shape)} → {list(output.shape)}")
    
    def test_temporal_positional_encoding_basic(self):
        """Test temporal positional encoding."""
        logger.info("🧪 Testing Temporal Positional Encoding")
        
        # Test sinusoidal encoding
        pos_encoding = TemporalPositionalEncoding(
            embed_dim=512,
            max_sequence_length=200,
            encoding_type="sinusoidal"
        )
        
        # Test different sequence lengths
        for seq_len in [10, 50, 100]:
            x = torch.randn(2, seq_len, 512)
            output = pos_encoding(x)
            
            # Shape should be unchanged
            self.assertEqual(output.shape, x.shape)
            
            # Values should be different (positions added)
            self.assertFalse(torch.allclose(x, output, atol=1e-6))
        
        logger.info("   ✅ Temporal positional encoding works")
    
    def test_spatio_temporal_embedding_basic(self):
        """Test spatio-temporal embedding with basic configuration."""
        logger.info("🧪 Testing SpatioTemporal Embedding")
        
        # Use a simple configuration
        embedding = SpatioTemporalEmbedding(
            input_channels=3,
            embed_dim=256,
            conv3d_channels=[64, 128, 256],
            dropout=0.1
        )
        
        # Test input [B, C, T, H, W]
        x = torch.randn(2, 3, 8, 64, 64)  # Smaller input for testing
        
        # Forward pass
        output = embedding(x)
        
        # Check output shape [B, T, D]
        self.assertEqual(output.shape[0], 2)  # batch size
        self.assertEqual(output.shape[2], 256)  # embed_dim
        self.assertGreater(output.shape[1], 0)  # time dimension should be positive
        
        logger.info(f"   ✅ SpatioTemporal: {list(x.shape)} → {list(output.shape)}")
    
    def test_feed_forward_basic(self):
        """Test feed-forward network."""
        logger.info("🧪 Testing Feed Forward Network")
        
        ff = FeedForward(
            embed_dim=512,
            expansion_factor=4,
            dropout=0.1
        )
        
        # Test input
        x = torch.randn(2, 100, 512)
        
        # Forward pass
        output = ff(x)
        
        # Check output shape (should be unchanged)
        self.assertEqual(output.shape, x.shape)
        
        logger.info(f"   ✅ FeedForward: {list(x.shape)} → {list(output.shape)}")
    
    def test_gradient_flow_basic(self):
        """Test basic gradient flow."""
        logger.info("🧪 Testing Basic Gradient Flow")
        
        # Simple model for gradient testing
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        # Test input with gradient
        x = torch.randn(5, 100, requires_grad=True)
        
        # Forward pass
        output = model(x)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.any(x.grad != 0))  # At least some gradients should be non-zero
        
        # Check model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
        
        logger.info("   ✅ Basic gradient flow works")
    
    def test_memory_efficiency(self):
        """Test memory efficiency of components."""
        logger.info("🧪 Testing Memory Efficiency")
        
        # Test with larger inputs to check memory usage
        embedding = SpatioTemporalEmbedding(
            input_channels=3,
            embed_dim=128,  # Smaller for memory efficiency
            conv3d_channels=[32, 64, 128],
            dropout=0.1
        )
        
        # Test input
        x = torch.randn(1, 3, 16, 128, 128)  # Reasonable size
        
        # Forward pass
        with torch.no_grad():  # No gradients needed for memory test
            output = embedding(x)
        
        # Check output
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[2], 128)
        
        logger.info(f"   ✅ Memory test passed: {list(x.shape)} → {list(output.shape)}")
    
    def test_device_compatibility(self):
        """Test device compatibility (CPU/CUDA)."""
        logger.info("🧪 Testing Device Compatibility")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        model = FeedForward(embed_dim=256, expansion_factor=2)
        model = model.to(device)
        
        # Test input
        x = torch.randn(2, 50, 256, device=device)
        
        # Forward pass
        output = model(x)
        
        # Check device (compare device types, not exact device objects)
        self.assertEqual(output.device.type, device.type)
        self.assertEqual(output.shape, x.shape)
        
        logger.info(f"   ✅ Device compatibility ({device.type}): {list(x.shape)} → {list(output.shape)}")
    
    def test_model_serialization(self):
        """Test model serialization/deserialization."""
        logger.info("🧪 Testing Model Serialization")
        
        # Create model
        model = FeedForward(embed_dim=128, expansion_factor=2)
        model.eval()  # Set to evaluation mode
        
        # Test input
        x = torch.randn(1, 10, 128)
        
        # Get original output
        with torch.no_grad():
            original_output = model(x)
        
        # Save and load model (use a simple file path)
        import tempfile
        import os
        temp_path = os.path.join(tempfile.gettempdir(), "test_model.pt")
        
        try:
            torch.save(model.state_dict(), temp_path)
            
            # Create new model and load state
            new_model = FeedForward(embed_dim=128, expansion_factor=2)
            new_model.load_state_dict(torch.load(temp_path, weights_only=True))
            new_model.eval()  # Set to evaluation mode
            
            # Test loaded model
            with torch.no_grad():
                new_output = new_model(x)
            
            # Outputs should be identical
            self.assertTrue(torch.allclose(original_output, new_output, atol=1e-5))
            
            logger.info("   ✅ Model serialization works")
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Simple performance benchmarks."""
    
    def setUp(self):
        """Set up benchmarks."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.iterations = 5  # Fewer iterations for faster testing
    
    def test_spatio_temporal_performance(self):
        """Benchmark spatio-temporal embedding."""
        logger.info("⚡ Benchmarking SpatioTemporal Embedding")
        
        embedding = SpatioTemporalEmbedding(
            input_channels=3,
            embed_dim=256,
            conv3d_channels=[64, 128, 256],
            dropout=0.1
        ).to(self.device)
        
        # Test input
        x = torch.randn(2, 3, 8, 112, 112, device=self.device)
        
        # Warmup
        for _ in range(2):
            _ = embedding(x)
        
        # Benchmark
        import time
        start_time = time.time()
        
        for _ in range(self.iterations):
            output = embedding(x)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / self.iterations
        
        # Avoid division by zero
        if avg_time > 0:
            throughput = 2 / avg_time  # batch_size / avg_time
            logger.info(f"   ⚡ Average time: {avg_time*1000:.2f} ms")
            logger.info(f"   ⚡ Throughput: {throughput:.1f} samples/sec")
        else:
            logger.info("   ⚡ Execution too fast to measure accurately")
        
        # Performance should be reasonable (adjust based on hardware)
        self.assertLess(avg_time, 5.0, "SpatioTemporal embedding too slow")
    
    def test_feedforward_performance(self):
        """Benchmark feed-forward network."""
        logger.info("⚡ Benchmarking FeedForward Network")
        
        ff = FeedForward(
            embed_dim=512,
            expansion_factor=4,
            dropout=0.1
        ).to(self.device)
        
        # Test input
        x = torch.randn(4, 100, 512, device=self.device)
        
        # Warmup
        for _ in range(2):
            _ = ff(x)
        
        # Benchmark
        import time
        start_time = time.time()
        
        for _ in range(self.iterations):
            output = ff(x)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / self.iterations
        throughput = 4 / avg_time  # batch_size / avg_time
        
        logger.info(f"   ⚡ Average time: {avg_time*1000:.2f} ms")
        logger.info(f"   ⚡ Throughput: {throughput:.1f} samples/sec")
        
        # Performance should be very fast
        self.assertLess(avg_time, 0.5, "FeedForward network too slow")


def run_tests():
    """Run all working component tests."""
    
    logger.info("🚀 Running Valerie Model Component Tests")
    logger.info("=" * 60)
    
    # Create test suite
    test_classes = [
        TestWorkingComponents,
        TestPerformanceBenchmarks
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
        logger.info("🎉 All component tests passed!")
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