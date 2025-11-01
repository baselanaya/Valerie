"""
Simple component tests for Valerie Visual ASR.

Tests individual components without complex dependencies.
"""

import torch
import torch.nn as nn
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_basic_components():
    """Test basic functionality of implemented components."""
    print("🚀 Starting basic component tests...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    # Test 1: Import and basic functionality
    try:
        from src.models.spatio_temporal import Conv3dBlock, TemporalPositionalEncoding, SpatioTemporalEmbedding
        print("✅ Successfully imported spatio_temporal components")
        
        # Test Conv3dBlock
        conv_block = Conv3dBlock(3, 64).to(device)
        test_input = torch.randn(1, 3, 8, 56, 56).to(device)
        output = conv_block(test_input)
        print(f"✅ Conv3dBlock: {test_input.shape} -> {output.shape}")
        
        # Test TemporalPositionalEncoding
        pos_enc = TemporalPositionalEncoding(512, 100).to(device)
        test_seq = torch.randn(1, 20, 512).to(device)
        encoded = pos_enc(test_seq)
        print(f"✅ TemporalPositionalEncoding: {test_seq.shape} -> {encoded.shape}")
        
        # Test SpatioTemporalEmbedding
        embedding = SpatioTemporalEmbedding(3, 512, [64, 128, 256, 512]).to(device)
        video_input = torch.randn(1, 8, 56, 56, 3).to(device)
        embedded = embedding(video_input)
        print(f"✅ SpatioTemporalEmbedding: {video_input.shape} -> {embedded.shape}")
        
    except Exception as e:
        print(f"❌ Spatio-temporal test failed: {e}")
        return False
    
    # Test 2: Conformer components
    try:
        from src.models.conformer import ConvolutionModule, MultiHeadSelfAttention, ConformerBlock
        print("✅ Successfully imported conformer components")
        
        # Test ConvolutionModule
        conv_module = ConvolutionModule(512).to(device)
        test_input = torch.randn(1, 50, 512).to(device)
        output = conv_module(test_input)
        print(f"✅ ConvolutionModule: {test_input.shape} -> {output.shape}")
        
        # Test MultiHeadSelfAttention
        attention = MultiHeadSelfAttention(512, 8).to(device)
        output = attention(test_input)
        print(f"✅ MultiHeadSelfAttention: {test_input.shape} -> {output.shape}")
        
        # Test ConformerBlock
        block = ConformerBlock(512).to(device)
        output = block(test_input)
        print(f"✅ ConformerBlock: {test_input.shape} -> {output.shape}")
        
    except Exception as e:
        print(f"❌ Conformer test failed: {e}")
        return False
    
    # Test 3: CTC/Attention components
    try:
        from src.models.hybrid_ctc_attention import CTCHead, HybridCTCAttention
        print("✅ Successfully imported hybrid CTC/Attention components")
        
        # Test CTCHead
        ctc_head = CTCHead(512, 40).to(device)
        encoder_out = torch.randn(1, 50, 512).to(device)
        ctc_logits = ctc_head(encoder_out)
        print(f"✅ CTCHead: {encoder_out.shape} -> {ctc_logits.shape}")
        
        # Test HybridCTCAttention
        hybrid = HybridCTCAttention(512, 40).to(device)
        hybrid.eval()
        with torch.no_grad():
            results = hybrid(encoder_out)
        print(f"✅ HybridCTCAttention: Generated {len(results)} outputs")
        
    except Exception as e:
        print(f"❌ Hybrid CTC/Attention test failed: {e}")
        return False
    
    # Test 4: Integration test
    try:
        print("🔗 Testing integration pipeline...")
        
        # Create full pipeline
        spatio_temporal = SpatioTemporalEmbedding(3, 512).to(device)
        
        from src.models.conformer import ConformerEncoder
        conformer = ConformerEncoder(512, 512, 4, 8).to(device)  # Smaller for testing
        
        hybrid_ctc = HybridCTCAttention(512, 40).to(device)
        
        # Test forward pass
        video_input = torch.randn(1, 8, 56, 56, 3).to(device)
        
        # Pipeline: Video -> Spatio-temporal -> Conformer -> CTC/Attention
        features = spatio_temporal(video_input)
        encoded = conformer(features)
        
        hybrid_ctc.eval()
        with torch.no_grad():
            results = hybrid_ctc(encoded)
        
        print(f"✅ Full pipeline test passed:")
        print(f"   Video input: {video_input.shape}")
        print(f"   Spatio-temporal features: {features.shape}")
        print(f"   Conformer encoded: {encoded.shape}")
        print(f"   CTC logits: {results['ctc_logits'].shape}")
        print(f"   Attention logits: {results['attention_logits'].shape}")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    
    print("\n🎉 All basic component tests passed!")
    return True


def test_memory_usage():
    """Test memory usage and efficiency."""
    if not torch.cuda.is_available():
        print("⚠️ Skipping memory test (CUDA not available)")
        return True
    
    print("\n💾 Testing memory usage...")
    
    try:
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Create components
        from src.models.spatio_temporal import SpatioTemporalEmbedding
        embedding = SpatioTemporalEmbedding(3, 256, [32, 64, 128, 256]).cuda()
        
        # Test with batch
        video_input = torch.randn(2, 8, 64, 64, 3).cuda()
        output = embedding(video_input)
        
        peak_memory = torch.cuda.memory_allocated()
        memory_used = (peak_memory - initial_memory) / (1024 ** 2)  # MB
        
        print(f"✅ Memory usage test: {memory_used:.2f} MB")
        
        # Cleanup
        del embedding, video_input, output
        torch.cuda.empty_cache()
        
        return memory_used < 500  # Should use less than 500MB for test
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Valerie Component Testing")
    print("=" * 50)
    
    # Run tests
    basic_success = test_basic_components()
    memory_success = test_memory_usage()
    
    print("\n📊 Test Summary:")
    print(f"   Basic components: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"   Memory usage: {'✅ PASS' if memory_success else '❌ FAIL'}")
    
    if basic_success and memory_success:
        print("\n🎉 All tests passed! Components are working correctly.")
        exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        exit(1)
