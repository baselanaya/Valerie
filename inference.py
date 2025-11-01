#!/usr/bin/env python3
"""
Inference Script for Valerie Visual ASR
=======================================

Easy-to-use script for running inference on video files.

Usage:
    python inference.py --input video.mp4
    python inference.py --input video.mp4 --model checkpoints/best_model.pt
    python inference.py --batch videos/*.mp4 --output results.json
    python inference.py --real-time --camera 0
"""

import argparse
import sys
import os
from pathlib import Path
import json
import time
import glob
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.inference import ValerieInferenceEngine, InferenceConfig
from src.utils.logging import setup_logger, get_logger

# Setup logging
setup_logger(use_rich=True)
logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Valerie Visual ASR Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input video file or pattern (e.g., 'video.mp4' or 'videos/*.mp4')"
    )
    
    parser.add_argument(
        "--batch",
        type=str,
        nargs="+",
        help="Batch of video files"
    )
    
    parser.add_argument(
        "--real-time", "-rt",
        action="store_true",
        help="Real-time inference mode"
    )
    
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera index for real-time mode"
    )
    
    # Model options
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for inference"
    )
    
    # Processing options
    parser.add_argument(
        "--batch-size", "-bs",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    
    parser.add_argument(
        "--beam-width", "-bw",
        type=int,
        default=100,
        help="Beam width for CTC decoding"
    )
    
    parser.add_argument(
        "--confidence-threshold", "-ct",
        type=float,
        default=0.1,
        help="Minimum confidence threshold"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for results (JSON format)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with debug information"
    )
    
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run performance benchmark"
    )
    
    return parser.parse_args()


def single_video_inference(engine: ValerieInferenceEngine, video_path: str, verbose: bool = False):
    """Run inference on single video."""
    
    logger.info(f"🎬 Processing video: {video_path}")
    
    if not Path(video_path).exists():
        logger.error(f"❌ Video file not found: {video_path}")
        return None
    
    try:
        start_time = time.time()
        result = engine.infer_video(video_path, return_debug_info=verbose)
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Inference completed in {processing_time:.2f}s")
        logger.info(f"📝 Result: '{result.text}'")
        logger.info(f"🎯 Confidence: {result.confidence:.3f}")
        
        if verbose:
            logger.info(f"🔤 Phonemes: {' '.join(result.phonemes)}")
            logger.info(f"📊 Phoneme confidence: {result.phoneme_confidence:.3f}")
            logger.info(f"🔄 Reconstruction confidence: {result.reconstruction_confidence:.3f}")
        
        return {
            'video_path': video_path,
            'text': result.text,
            'confidence': result.confidence,
            'phonemes': result.phonemes,
            'phoneme_confidence': result.phoneme_confidence,
            'reconstruction_confidence': result.reconstruction_confidence,
            'processing_time': processing_time
        }
        
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        return None


def batch_inference(engine: ValerieInferenceEngine, video_paths: list, verbose: bool = False):
    """Run batch inference on multiple videos."""
    
    logger.info(f"📁 Processing {len(video_paths)} videos in batch")
    
    # Filter existing files
    existing_paths = [p for p in video_paths if Path(p).exists()]
    if len(existing_paths) < len(video_paths):
        logger.warning(f"⚠️ {len(video_paths) - len(existing_paths)} videos not found")
    
    if not existing_paths:
        logger.error("❌ No valid video files found")
        return []
    
    try:
        start_time = time.time()
        results = engine.infer_batch(existing_paths, return_debug_info=verbose)
        total_time = time.time() - start_time
        
        logger.info(f"✅ Batch inference completed in {total_time:.2f}s")
        logger.info(f"⚡ Throughput: {len(existing_paths)/total_time:.2f} videos/second")
        
        # Process results
        batch_results = []
        successful = 0
        
        for i, (video_path, result) in enumerate(zip(existing_paths, results)):
            if result.confidence > 0.1:
                successful += 1
                logger.info(f"✅ [{i+1}/{len(existing_paths)}] {Path(video_path).name}: '{result.text}' (conf: {result.confidence:.3f})")
            else:
                logger.warning(f"⚠️ [{i+1}/{len(existing_paths)}] {Path(video_path).name}: Low confidence ({result.confidence:.3f})")
            
            batch_results.append({
                'video_path': video_path,
                'text': result.text,
                'confidence': result.confidence,
                'phonemes': result.phonemes,
                'phoneme_confidence': result.phoneme_confidence,
                'reconstruction_confidence': result.reconstruction_confidence,
                'processing_time': result.processing_time
            })
        
        logger.info(f"📊 Success rate: {successful}/{len(existing_paths)} ({successful/len(existing_paths)*100:.1f}%)")
        
        return batch_results
        
    except Exception as e:
        logger.error(f"❌ Batch inference failed: {e}")
        return []


def real_time_inference(engine: ValerieInferenceEngine, camera_index: int = 0):
    """Run real-time inference from camera."""
    
    logger.info(f"📹 Starting real-time inference from camera {camera_index}")
    
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error(f"❌ Cannot open camera {camera_index}")
            return
        
        logger.info("🎥 Real-time inference started. Press 'q' to quit.")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("❌ Failed to capture frame")
                break
            
            # Process frame
            result = engine.infer_real_time(frame)
            
            if result is not None:
                logger.info(f"📝 Result: '{result.text}' (confidence: {result.confidence:.3f})")
            
            # Display frame
            cv2.imshow('Valerie Visual ASR - Real-time', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        logger.info(f"✅ Real-time inference completed. Processed {frame_count} frames.")
        
    except Exception as e:
        logger.error(f"❌ Real-time inference failed: {e}")


def run_benchmark(engine: ValerieInferenceEngine, test_videos: list):
    """Run performance benchmark."""
    
    if not test_videos:
        logger.warning("⚠️ No test videos provided for benchmark")
        return
    
    logger.info(f"🏁 Running benchmark with {len(test_videos)} videos")
    
    try:
        metrics = engine.benchmark(test_videos, num_runs=3)
        
        logger.info("📊 Benchmark Results:")
        logger.info(f"   Average time: {metrics['avg_time_seconds']:.2f} ± {metrics['std_time_seconds']:.2f} seconds")
        logger.info(f"   Throughput: {metrics['throughput_videos_per_second']:.2f} videos/second")
        logger.info(f"   Success rate: {metrics['success_rate']:.2%}")
        logger.info(f"   Min/Max time: {metrics['min_time_seconds']:.2f}/{metrics['max_time_seconds']:.2f} seconds")
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        return None


def main():
    """Main inference function."""
    
    args = parse_args()
    
    logger.info("🚀 Valerie Visual ASR - Inference")
    logger.info("=" * 50)
    
    # Check model checkpoint
    if not Path(args.model).exists():
        logger.error(f"❌ Model checkpoint not found: {args.model}")
        return 1
    
    # Create inference configuration
    from src.inference.ctc_decoder import DecodingConfig
    from src.inference.phoneme_reconstruction import ReconstructionConfig
    
    decoding_config = DecodingConfig(
        beam_width=args.beam_width,
        min_confidence=args.confidence_threshold
    )
    
    reconstruction_config = ReconstructionConfig()
    
    inference_config = InferenceConfig(
        model_checkpoint=args.model,
        device=args.device,
        batch_size=args.batch_size,
        decoding_config=decoding_config,
        reconstruction_config=reconstruction_config
    )
    
    # Initialize inference engine
    try:
        logger.info("🔧 Initializing inference engine...")
        engine = ValerieInferenceEngine(inference_config)
        
        # Display model info
        model_info = engine.get_model_info()
        logger.info("🏗️ Model Information:")
        logger.info(f"   Parameters: {model_info['total_parameters']:,}")
        logger.info(f"   Model size: {model_info['model_size_mb']:.1f} MB")
        logger.info(f"   Device: {model_info['device']}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize inference engine: {e}")
        return 1
    
    # Determine inference mode
    results = None
    
    if args.real_time:
        # Real-time inference
        real_time_inference(engine, args.camera)
        
    elif args.batch:
        # Batch inference from arguments
        video_paths = args.batch
        results = batch_inference(engine, video_paths, args.verbose)
        
    elif args.input:
        # Single video or pattern
        if '*' in args.input or '?' in args.input:
            # Glob pattern
            video_paths = glob.glob(args.input)
            if not video_paths:
                logger.error(f"❌ No videos found matching pattern: {args.input}")
                return 1
            results = batch_inference(engine, video_paths, args.verbose)
        else:
            # Single video
            result = single_video_inference(engine, args.input, args.verbose)
            results = [result] if result else []
    
    else:
        logger.error("❌ No input specified. Use --input, --batch, or --real-time")
        return 1
    
    # Run benchmark if requested
    if args.benchmark and results:
        video_paths = [r['video_path'] for r in results if r]
        benchmark_results = run_benchmark(engine, video_paths)
        if benchmark_results and args.output:
            # Add benchmark results to output
            results.append({'benchmark': benchmark_results})
    
    # Save results
    if args.output and results:
        try:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"💾 Results saved to: {args.output}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
    
    logger.info("✅ Inference completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
