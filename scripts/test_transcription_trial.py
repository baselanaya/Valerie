#!/usr/bin/env python3
"""
Trial run of transcription pipeline on VoxCeleb2 data.

This script tests the transcription pipeline on a small subset of your VoxCeleb2 data
to verify everything works correctly before running on the full dataset.
"""

import os
import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import our transcription generator
sys.path.append(str(Path(__file__).parent))
from generate_transcriptions import TranscriptionGenerator
from convert_to_phonemes import PhonemeConverter

def main():
    print("🎯 VoxCeleb2 Transcription Pipeline Trial Run")
    print("=" * 50)
    
    # Set up paths
    data_root = Path("data")
    mp4_dir = data_root / "dev" / "mp4"
    aac_dir = data_root / "dev" / "aac"
    output_dir = Path("data/trial_transcriptions")
    phoneme_dir = Path("data/trial_phonemes")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    phoneme_dir.mkdir(parents=True, exist_ok=True)
    
    # Find first few MP4 files for testing
    mp4_files = list(mp4_dir.rglob("*.mp4"))[:3]  # Test with first 3 files
    
    if not mp4_files:
        print("❌ No MP4 files found in data/dev/mp4")
        print("   Make sure your VoxCeleb2 data is in the correct location")
        return
    
    print(f"📁 Found {len(mp4_files)} MP4 files for testing")
    for mp4_file in mp4_files:
        print(f"   - {mp4_file}")
    
    # Initialize transcription generator with tiny model for speed
    print("\\n🎤 Initializing Whisper (tiny model for fast testing)...")
    generator = TranscriptionGenerator(
        model_name="tiny",  # Fast for testing
        device="auto",
        language="en"
    )
    
    # Process each file
    print("\\n📝 Processing files...")
    results = []
    
    for i, mp4_file in enumerate(mp4_files, 1):
        print(f"\\n--- File {i}/{len(mp4_files)}: {mp4_file.name} ---")
        start_time = time.time()
        
        # Process single file
        result = generator.process_single_file(str(mp4_file), str(output_dir), use_existing_aac=True)
        results.append(result)
        
        # Show results
        processing_time = time.time() - start_time
        status = result.get('status', 'unknown')
        transcription_text = result.get('transcription', {}).get('text', '')
        
        print(f"   Status: {status}")
        print(f"   Processing time: {processing_time:.1f}s")
        if transcription_text:
            print(f"   Transcription: '{transcription_text[:100]}{'...' if len(transcription_text) > 100 else ''}'")
        else:
            print("   Transcription: (empty)")
    
    # Convert to phonemes
    print("\\n🔤 Converting transcriptions to phonemes...")
    phoneme_converter = PhonemeConverter(phoneme_set="arpabet")
    
    # Find generated transcription files
    transcription_files = list(output_dir.glob("*.json"))
    
    if transcription_files:
        for trans_file in transcription_files:
            print(f"   Converting: {trans_file.name}")
            phoneme_result = phoneme_converter.process_transcription_file(
                str(trans_file), str(phoneme_dir)
            )
            
            if phoneme_result.get('status') == 'success':
                phonemes = phoneme_result.get('phonemes', [])
                print(f"   Generated {len(phonemes)} phonemes: {' '.join(phonemes[:10])}{'...' if len(phonemes) > 10 else ''}")
            else:
                print(f"   Failed to convert phonemes")
    
    # Summary
    print("\\n📊 Trial Run Summary")
    print("=" * 30)
    successful = sum(1 for r in results if r.get('status') == 'success')
    failed = sum(1 for r in results if r.get('status') == 'failed')
    empty = sum(1 for r in results if r.get('status') == 'empty')
    
    print(f"Files processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Empty: {empty}")
    
    # Check output files
    transcription_files = list(output_dir.glob("*.json"))
    phoneme_files = list(phoneme_dir.glob("*.json"))
    
    print(f"\\nGenerated files:")
    print(f"Transcriptions: {len(transcription_files)}")
    print(f"Phonemes: {len(phoneme_files)}")
    
    if transcription_files:
        print(f"\\n✅ Trial run successful! Check output files in:")
        print(f"   Transcriptions: {output_dir}")
        print(f"   Phonemes: {phoneme_dir}")
        print(f"\\n🚀 Ready to run full transcription pipeline:")
        print(f"   python scripts/generate_transcriptions.py --data_root data --output_dir data/transcriptions --model base")
    else:
        print(f"\\n⚠️  No transcriptions generated. Check the logs above for issues.")

if __name__ == "__main__":
    main()
