#!/usr/bin/env python3
"""
Generate transcriptions from VoxCeleb2 audio using OpenAI Whisper.

This script addresses the missing transcription problem in VoxCeleb2 dataset:
- VoxCeleb2 text files contain face bounding boxes, NOT transcriptions
- We need actual speech transcriptions for two-stage visual speech recognition training
- This pipeline processes M4A audio files directly to generate transcriptions

Usage:
    python scripts/generate_transcriptions.py --data_root data --output_dir data/transcriptions
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import torch
import whisper
import librosa
import soundfile as sf
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcription_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TranscriptionGenerator:
    """Generate transcriptions from audio using Whisper ASR."""
    
    def __init__(
        self,
        model_name: str = "large-v2",
        device: str = "auto",
        language: str = "en",
        batch_size: int = 1
    ):
        """
        Initialize transcription generator.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
            device: Device to run on ("cuda", "cpu", or "auto")
            language: Language code for transcription
            batch_size: Number of files to process in parallel
        """
        self.model_name = model_name
        self.language = language
        self.batch_size = batch_size
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing Whisper {model_name} on {self.device}")
        
        # Load Whisper model
        try:
            self.model = whisper.load_model(model_name, device=self.device)
            logger.info(f"Whisper {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
            
        # Model specifications
        self.model_specs = {
            "tiny": {"params": "39M", "vram": "~1GB", "speed": "32x"},
            "base": {"params": "74M", "vram": "~1GB", "speed": "16x"},
            "small": {"params": "244M", "vram": "~2GB", "speed": "6x"},
            "medium": {"params": "769M", "vram": "~5GB", "speed": "2x"},
            "large": {"params": "1550M", "vram": "~10GB", "speed": "1x"},
            "large-v2": {"params": "1550M", "vram": "~10GB", "speed": "1x"},
            "large-v3": {"params": "1550M", "vram": "~10GB", "speed": "1x"}
        }
        
        specs = self.model_specs.get(model_name, {"params": "Unknown", "vram": "Unknown", "speed": "Unknown"})
        logger.info(f"Model specs - Params: {specs['params']}, VRAM: {specs['vram']}, Speed: {specs['speed']}")
    
    def extract_audio_from_mp4(self, mp4_path: str, output_path: str, sample_rate: int = 16000) -> bool:
        """
        Extract audio from MP4 file.
        
        Args:
            mp4_path: Path to MP4 file
            output_path: Path to save extracted audio
            sample_rate: Target sample rate
            
        Returns:
            bool: Success status
        """
        try:
            # Load audio from MP4 (librosa can handle video files)
            audio, sr = librosa.load(mp4_path, sr=sample_rate, mono=True)
            
            # Save as WAV file
            sf.write(output_path, audio, sample_rate)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to extract audio from {mp4_path}: {e}")
            return False
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """
        Transcribe audio file using Whisper.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Transcription results with text, segments, and metadata
        """
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_path,
                language=self.language,
                task="transcribe",
                verbose=False,
                condition_on_previous_text=False  # Better for short clips
            )
            
            # Extract key information
            transcription = {
                "text": result["text"].strip(),
                "language": result["language"],
                "duration": result.get("duration", 0.0),
                "segments": result.get("segments", []),
                "words": []
            }
            
            # Extract word-level information (if available)
            for segment in result.get("segments", []):
                # Add segment-level words if available
                if "words" in segment:
                    for word_info in segment["words"]:
                        transcription["words"].append({
                            "word": word_info["word"].strip(),
                            "start": word_info["start"],
                            "end": word_info["end"],
                            "probability": word_info.get("probability", 0.0)
                        })
                else:
                    # Fallback: create word entries from segment text
                    segment_words = segment["text"].strip().split()
                    segment_duration = segment["end"] - segment["start"]
                    word_duration = segment_duration / len(segment_words) if segment_words else 0
                    
                    for i, word in enumerate(segment_words):
                        word_start = segment["start"] + i * word_duration
                        word_end = word_start + word_duration
                        transcription["words"].append({
                            "word": word,
                            "start": word_start,
                            "end": word_end,
                            "probability": 0.5  # Default probability
                        })
            
            return transcription
            
        except Exception as e:
            logger.error(f"❌ Failed to transcribe {audio_path}: {e}")
            return {
                "text": "",
                "language": "unknown",
                "duration": 0.0,
                "segments": [],
                "words": [],
                "error": str(e)
            }
    
    def process_single_file(self, mp4_path: str, output_dir: str, use_existing_aac: bool = True) -> Dict:
        """
        Process a single MP4 file to generate transcription.
        
        Args:
            mp4_path: Path to MP4 file
            output_dir: Directory to save transcription
            use_existing_aac: Whether to use existing AAC file if available
            
        Returns:
            dict: Processing results
        """
        mp4_path = Path(mp4_path)
        output_dir = Path(output_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output paths
        base_name = mp4_path.stem
        transcription_path = output_dir / f"{base_name}.json"
        text_path = output_dir / f"{base_name}.txt"
        audio_path = output_dir / f"{base_name}.wav"
        
        # Check if transcription already exists
        if transcription_path.exists():
            logger.info(f"⏭️  Skipping {base_name} (transcription exists)")
            try:
                with open(transcription_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                logger.warning(f"⚠️  Corrupted transcription file {transcription_path}, regenerating...")
        
        start_time = time.time()
        
        # Step 1: Check for existing AAC file first
        audio_source = None
        if use_existing_aac:
            # Look for corresponding AAC/M4A file in VoxCeleb2 structure
            # MP4 path: data/dev/mp4/id00012/21Uxsk56VDQ/00001.mp4
            # AAC path: data/dev/aac/id00012/21Uxsk56VDQ/00001.m4a
            mp4_parts = mp4_path.parts
            if 'mp4' in mp4_parts:
                # Replace 'mp4' with 'aac' in path
                aac_parts = list(mp4_parts)
                mp4_idx = aac_parts.index('mp4')
                aac_parts[mp4_idx] = 'aac'
                
                # Try both .m4a and .aac extensions
                for ext in ['.m4a', '.aac']:
                    aac_path = Path(*aac_parts).with_suffix(ext)
                    if aac_path.exists():
                        audio_source = str(aac_path)
                        logger.info(f"Using existing audio file: {base_name}{ext}")
                        break
            
            # Fallback: look in same directory
            if audio_source is None:
                for ext in ['.m4a', '.aac']:
                    aac_path = mp4_path.with_suffix(ext)
                    if aac_path.exists():
                        audio_source = str(aac_path)
                        logger.info(f"Using existing audio file: {base_name}{ext}")
                        break
        
        # Step 2: Extract audio from MP4 if no AAC found
        if audio_source is None:
            logger.info(f"Extracting audio from MP4: {base_name}")
            if not self.extract_audio_from_mp4(str(mp4_path), str(audio_path)):
                return {
                    "file": str(mp4_path),
                    "status": "failed",
                    "error": "Audio extraction failed",
                    "processing_time": time.time() - start_time
                }
            audio_source = str(audio_path)
        
        # Step 3: Transcribe audio
        logger.info(f"Transcribing: {base_name}")
        transcription = self.transcribe_audio(audio_source)
        
        # Step 4: Save results
        processing_time = time.time() - start_time
        
        result = {
            "file": str(mp4_path),
            "audio_source": audio_source,
            "transcription": transcription,
            "processing_time": processing_time,
            "status": "success" if transcription["text"] else "empty",
            "model": self.model_name,
            "timestamp": time.time()
        }
        
        # Save detailed JSON
        with open(transcription_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save simple text file
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(transcription["text"])
        
        # Clean up temporary audio file if we extracted it
        if audio_source == str(audio_path) and audio_path.exists():
            audio_path.unlink()
        
        if transcription and transcription.get('text'):
            logger.info(f"Completed {base_name} in {processing_time:.1f}s: '{transcription['text'][:50]}...'")
        
        return result
    
    def process_m4a_file(self, m4a_path: str, output_dir: str) -> Dict:
        """
        Process a single M4A audio file for transcription.
        
        Args:
            m4a_path: Path to M4A audio file
            output_dir: Output directory for transcription files
            
        Returns:
            dict: Processing result with status, transcription, and metadata
        """
        m4a_path = Path(m4a_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = m4a_path.stem
        json_path = output_dir / f"{base_name}.json"
        text_path = output_dir / f"{base_name}.txt"
        
        # Skip if already processed
        if json_path.exists():
            logger.info(f"Skipping {base_name} (already processed)")
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        start_time = time.time()
        
        # Transcribe the M4A file directly
        logger.info(f"Transcribing: {base_name}")
        transcription = self.transcribe_audio(str(m4a_path))
        
        # Process results
        processing_time = time.time() - start_time
        
        if not transcription or not transcription.get("text", "").strip():
            result = {
                "file": str(m4a_path),
                "audio_source": str(m4a_path),
                "status": "empty",
                "transcription": transcription,
                "processing_time": processing_time
            }
        else:
            result = {
                "file": str(m4a_path),
                "audio_source": str(m4a_path),
                "status": "success",
                "transcription": transcription,
                "processing_time": processing_time
            }
        
        # Save JSON result
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save simple text file
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(transcription["text"])
        
        if transcription and transcription.get('text'):
            logger.info(f"Completed {base_name} in {processing_time:.1f}s: '{transcription['text'][:50]}...'")
        
        return result
    
    def process_dataset(
        self,
        data_root: str,
        output_dir: str,
        max_files: Optional[int] = None,
        num_workers: int = 1
    ) -> Dict:
        """
        Process entire VoxCeleb2 dataset.
        
        Args:
            data_root: Root directory of VoxCeleb2 dataset
            output_dir: Directory to save transcriptions
            max_files: Maximum number of files to process (for testing)
            num_workers: Number of parallel workers
            
        Returns:
            dict: Processing statistics
        """
        data_root = Path(data_root)
        output_dir = Path(output_dir)
        
        logger.info(f"Scanning for M4A audio files in {data_root}")
        
        # Find all M4A files (audio files)
        m4a_files = list(data_root.rglob("*.m4a"))
        
        if not m4a_files:
            logger.error(f"No M4A files found in {data_root}")
            return {"status": "failed", "error": "No M4A files found"}
        
        logger.info(f"Found {len(m4a_files)} M4A files")
        
        if max_files:
            m4a_files = m4a_files[:max_files]
            logger.info(f"Processing first {len(m4a_files)} files (limited by max_files)")
        
        # Process files
        results = []
        start_time = time.time()
        
        if num_workers == 1:
            # Single-threaded processing
            for m4a_path in tqdm(m4a_files, desc="Processing files"):
                result = self.process_m4a_file(str(m4a_path), str(output_dir))
                results.append(result)
        else:
            # Multi-threaded processing (careful with GPU memory!)
            logger.info(f"Using {num_workers} workers for parallel processing")
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(self.process_m4a_file, str(m4a_path), str(output_dir))
                    for m4a_path in m4a_files
                ]
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"❌ Worker failed: {e}")
                        results.append({"status": "failed", "error": str(e)})
        
        # Calculate statistics
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.get("status") == "success")
        failed = sum(1 for r in results if r.get("status") == "failed")
        empty = sum(1 for r in results if r.get("status") == "empty")
        
        stats = {
            "total_files": len(m4a_files),
            "successful": successful,
            "failed": failed,
            "empty": empty,
            "total_time": total_time,
            "avg_time_per_file": total_time / len(m4a_files) if m4a_files else 0,
            "model": self.model_name,
            "device": self.device
        }
        
        logger.info(f"Processing completed!")
        logger.info(f"   Total files: {stats['total_files']}")
        logger.info(f"   Successful: {stats['successful']}")
        logger.info(f"   Failed: {stats['failed']}")
        logger.info(f"   Empty: {stats['empty']}")
        logger.info(f"   Total time: {stats['total_time']:.1f}s")
        logger.info(f"   Avg time per file: {stats['avg_time_per_file']:.1f}s")
        
        # Save statistics
        stats_path = output_dir / "transcription_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats


def get_recommended_workers(model_name: str, gpu_memory_gb: float = None) -> int:
    """
    Get recommended number of workers based on model size and GPU memory.
    
    Args:
        model_name: Whisper model name
        gpu_memory_gb: Available GPU memory in GB (auto-detect if None)
        
    Returns:
        int: Recommended number of workers
    """
    if gpu_memory_gb is None:
        # Try to detect GPU memory
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            return 1  # CPU fallback
    
    # Model VRAM requirements (approximate)
    model_vram = {
        "tiny": 1.0,
        "base": 1.0, 
        "small": 2.0,
        "medium": 5.0,
        "large": 10.0,
        "large-v2": 10.0,
        "large-v3": 10.0
    }
    
    required_vram = model_vram.get(model_name, 2.0)
    
    # Leave 1GB buffer for CUDA overhead
    usable_memory = max(1.0, gpu_memory_gb - 1.0)
    
    # Calculate max workers
    max_workers = int(usable_memory / required_vram)
    
    # Practical limits (diminishing returns beyond certain point)
    practical_limits = {
        "tiny": 12,
        "base": 10,
        "small": 6,
        "medium": 3,
        "large": 1,
        "large-v2": 1, 
        "large-v3": 1
    }
    
    recommended = min(max_workers, practical_limits.get(model_name, 4))
    return max(1, recommended)


def main():
    parser = argparse.ArgumentParser(description="Generate transcriptions from VoxCeleb2 audio")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory of VoxCeleb2 dataset (containing M4A audio files)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save transcriptions")
    parser.add_argument("--model", type=str, default="large-v3",
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                       help="Whisper model size")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to run on")
    parser.add_argument("--language", type=str, default="en",
                       help="Language code for transcription")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum number of files to process (for testing)")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Number of parallel workers (auto-optimized based on model and GPU if not specified)")
    parser.add_argument("--no_existing_aac", action="store_true",
                       help="Don't use existing AAC files, extract from MP4")
    
    args = parser.parse_args()
    
    # Auto-optimize workers if not specified
    if args.num_workers is None:
        recommended_workers = get_recommended_workers(args.model)
        args.num_workers = recommended_workers
        logger.info(f"Auto-optimized workers for {args.model} model: {args.num_workers}")
    
    logger.info("Starting transcription generation")
    logger.info(f"   Data root: {args.data_root}")
    logger.info(f"   Output dir: {args.output_dir}")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Device: {args.device}")
    logger.info(f"   Language: {args.language}")
    logger.info(f"   Max files: {args.max_files}")
    logger.info(f"   Workers: {args.num_workers}")
    
    # Show GPU memory info if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB VRAM)")
        
        # Show recommended workers for different models
        if args.num_workers > 1:
            logger.info("   Recommended workers by model:")
            for model in ["tiny", "base", "small", "medium", "large"]:
                rec = get_recommended_workers(model, gpu_memory)
                logger.info(f"     {model}: {rec} workers")
    
    # Initialize generator
    generator = TranscriptionGenerator(
        model_name=args.model,
        device=args.device,
        language=args.language
    )
    
    # Process dataset
    stats = generator.process_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        max_files=args.max_files,
        num_workers=args.num_workers
    )
    
    if stats.get("status") == "failed":
        logger.error(f"❌ Processing failed: {stats.get('error')}")
        sys.exit(1)
    
    logger.info("🎉 Transcription generation completed successfully!")


if __name__ == "__main__":
    main()
