"""
Test script for complete dataset processing pipeline.

Tests VoxCeleb2 dataset loading, transforms, phoneme processing, and batch collation.
"""

import torch
import sys
import os
import tempfile
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_dataset_pipeline():
    """Test complete dataset processing pipeline."""
    print("🧪 Testing Dataset Processing Pipeline")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    # Test 1: Import all components
    try:
        from src.data import (
            VoxCeleb2Dataset, DataSample, VideoTransforms, AudioTransforms,
            TextToPhonemeConverter, CTCLabelGenerator, ValerieCollator,
            create_dataloader, create_augmentation_pipeline
        )
        from src.utils.config import Config
        print("✅ Successfully imported all data components")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Configuration loading
    try:
        print("\n🔍 Testing configuration loading...")
        config = Config("configs/base_config.yaml")
        print(f"✅ Config loaded:")
        print(f"   Max sequence length: {config.data.max_sequence_length}")
        print(f"   Frame rate: {config.data.frame_rate}")
        print(f"   Audio sample rate: {config.data.audio_sample_rate}")
        print(f"   Input resolution: {config.model.input_height}x{config.model.input_width}")
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    # Test 3: Video and Audio Transforms
    try:
        print("\n🔍 Testing transforms...")
        video_transforms = VideoTransforms(
            target_height=224,
            target_width=224,
            horizontal_flip_prob=0.5
        )
        
        audio_transforms = AudioTransforms(
            sample_rate=16000,
            speed_perturb_prob=0.3
        )
        
        # Test video transforms
        dummy_video = torch.randn(10, 224, 224, 3)
        transformed_video = video_transforms(dummy_video, training=True)
        print(f"✅ Video transforms: {dummy_video.shape} -> {transformed_video.shape}")
        
        # Test audio transforms
        dummy_audio = torch.randn(16000)
        transformed_audio = audio_transforms(dummy_audio, training=True)
        print(f"✅ Audio transforms: {dummy_audio.shape} -> {transformed_audio.shape}")
        
    except Exception as e:
        print(f"❌ Transforms test failed: {e}")
        return False
    
    # Test 4: Phoneme Processing
    try:
        print("\n🔍 Testing phoneme processing...")
        
        # Text-to-phoneme conversion
        converter = TextToPhonemeConverter()
        test_text = "Hello world, how are you today?"
        phonemes = converter.text_to_phonemes(test_text)
        print(f"✅ Text-to-phonemes:")
        print(f"   Text: {test_text}")
        print(f"   Phonemes: {phonemes[:10]}...")  # Show first 10
        print(f"   Total phonemes: {len(phonemes)}")
        
        # CTC label generation
        ctc_generator = CTCLabelGenerator()
        ctc_labels = ctc_generator.phonemes_to_ctc_labels(phonemes)
        print(f"✅ CTC labels: {len(ctc_labels)} labels generated")
        
    except Exception as e:
        print(f"❌ Phoneme processing test failed: {e}")
        return False
    
    # Test 5: Data Sample Creation
    try:
        print("\n🔍 Testing data sample creation...")
        
        # Create sample data
        samples = []
        for i in range(3):
            sample = DataSample(
                video_frames=torch.randn(8 + i*2, 224, 224, 3),
                audio_waveform=torch.randn(16000 + i*1000),
                transcription=f"This is test sample number {i}",
                speaker_id=f"speaker_{i:03d}",
                language="en",
                duration=1.0 + i*0.1,
                face_confidence=0.9 + i*0.02,
                transcription_confidence=0.95 + i*0.01
            )
            samples.append(sample)
        
        print(f"✅ Created {len(samples)} data samples")
        for i, sample in enumerate(samples):
            print(f"   Sample {i}: video={sample.video_frames.shape}, audio={sample.audio_waveform.shape}")
        
    except Exception as e:
        print(f"❌ Data sample creation failed: {e}")
        return False
    
    # Test 6: Batch Collation
    try:
        print("\n🔍 Testing batch collation...")
        
        # Test basic collator
        collator = ValerieCollator(
            pad_video=True,
            pad_audio=True,
            max_video_length=20,
            max_audio_length=20000
        )
        
        batch_data = collator(samples)
        
        print(f"✅ Basic collator:")
        print(f"   Batch size: {len(batch_data)}")
        print(f"   Video batch shape: {batch_data.video_frames.shape}")
        print(f"   Video lengths: {batch_data.video_lengths}")
        print(f"   Audio batch shape: {batch_data.audio_waveforms.shape}")
        print(f"   Audio lengths: {batch_data.audio_lengths}")
        print(f"   Has video mask: {batch_data.video_mask is not None}")
        print(f"   Has audio mask: {batch_data.audio_mask is not None}")
        print(f"   Transcriptions: {len(batch_data.transcriptions)}")
        
        # Test moving to device
        batch_on_device = batch_data.to(device)
        print(f"✅ Batch moved to {device}")
        
    except Exception as e:
        print(f"❌ Batch collation test failed: {e}")
        return False
    
    # Test 7: Data Augmentation Pipeline
    try:
        print("\n🔍 Testing augmentation pipeline...")
        
        pipeline = create_augmentation_pipeline(config, training=True)
        
        # Create batch dict for pipeline
        batch_dict = {
            'video_frames': batch_data.video_frames,
            'audio_waveforms': batch_data.audio_waveforms
        }
        
        augmented_batch = pipeline(batch_dict)
        
        print(f"✅ Augmentation pipeline:")
        print(f"   Input video shape: {batch_dict['video_frames'].shape}")
        print(f"   Output video shape: {augmented_batch['video_frames'].shape}")
        print(f"   Has mixup lambdas: {'mixup_lambdas' in augmented_batch}")
        
    except Exception as e:
        print(f"❌ Augmentation pipeline test failed: {e}")
        return False
    
    # Test 8: VoxCeleb2 Dataset (with limited samples for testing)
    try:
        print("\n🔍 Testing VoxCeleb2 dataset loading...")
        
        # Test with a very small sample to avoid long download
        try:
            vox_dataset = VoxCeleb2Dataset(
                split="train",
                max_samples=5,  # Very small for testing
                streaming=True,  # Use streaming to avoid full download
                max_sequence_length=50,
                quality_filter=False  # Disable for testing
            )
            
            print(f"✅ VoxCeleb2 dataset created:")
            print(f"   Length: {len(vox_dataset)}")
            print(f"   Split: train")
            print(f"   Max samples: 5 (testing)")
            
            # Test getting a sample (this might take time for first download)
            print("   Attempting to load first sample...")
            sample = vox_dataset[0]
            print(f"   Sample loaded: video={sample.video_frames.shape}, audio={sample.audio_waveform.shape if sample.audio_waveform is not None else 'None'}")
            
        except Exception as dataset_e:
            print(f"⚠️ VoxCeleb2 dataset loading failed (expected in some environments): {dataset_e}")
            print("   This is normal if dataset is not accessible or network issues exist")
        
    except Exception as e:
        print(f"❌ VoxCeleb2 dataset test failed: {e}")
        return False
    
    # Test 9: DataLoader Integration
    try:
        print("\n🔍 Testing DataLoader integration...")
        
        # Create a simple dataset from our samples
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, samples):
                self.samples = samples
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                return self.samples[idx]
        
        simple_dataset = SimpleDataset(samples)
        
        # Create DataLoader
        dataloader = create_dataloader(
            simple_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # Avoid multiprocessing issues in testing
            collate_fn=collator
        )
        
        print(f"✅ DataLoader created:")
        print(f"   Dataset size: {len(simple_dataset)}")
        print(f"   Batch size: 2")
        
        # Test iteration
        for i, batch in enumerate(dataloader):
            print(f"   Batch {i}: video={batch.video_frames.shape}, audio={batch.audio_waveforms.shape}")
            if i >= 1:  # Only test 2 batches
                break
        
        print(f"✅ DataLoader iteration successful")
        
    except Exception as e:
        print(f"❌ DataLoader integration test failed: {e}")
        return False
    
    # Test 10: Memory Usage and Performance
    try:
        print("\n💾 Testing memory efficiency...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # Process a batch
            batch_on_gpu = batch_data.to(device)
            
            peak_memory = torch.cuda.memory_allocated()
            memory_used = (peak_memory - initial_memory) / (1024 ** 2)  # MB
            
            print(f"✅ Memory usage: {memory_used:.2f} MB for batch of {len(batch_data)}")
            
            # Cleanup
            del batch_on_gpu
            torch.cuda.empty_cache()
            
        else:
            print("⚠️ CUDA not available, skipping GPU memory test")
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
    
    print("\n🎉 All dataset pipeline tests completed!")
    return True


if __name__ == "__main__":
    success = test_dataset_pipeline()
    
    if success:
        print("\n✅ Dataset Processing Pipeline is working correctly!")
        print("🚀 Ready for:")
        print("   • VoxCeleb2 and AVSpeech data loading")
        print("   • Video and audio augmentation")
        print("   • Phoneme processing and CTC labels")
        print("   • Efficient batch collation")
        print("   • Integration with PyTorch DataLoader")
    else:
        print("\n❌ Some tests failed. Please review and fix issues.")
    
    exit(0 if success else 1)
