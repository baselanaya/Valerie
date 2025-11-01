#!/usr/bin/env python3
"""
Evaluation Script for Valerie Visual ASR
========================================

Comprehensive evaluation on test datasets with detailed metrics.

Usage:
    python evaluate.py --model checkpoints/best_model.pt
    python evaluate.py --model checkpoints/best_model.pt --dataset-root data --split test
    python evaluate.py --model checkpoints/best_model.pt --baseline baseline_results.json
    python evaluate.py --model checkpoints/best_model.pt --max-samples 100 --output evaluation_results
"""

import argparse
import sys
import os
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.evaluation import ValerieEvaluator, EvaluationConfig
from src.utils.logging import setup_logger, get_logger

# Setup logging
setup_logger(use_rich=True)
logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Valerie Visual ASR Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model settings
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for evaluation"
    )
    
    # Dataset settings
    parser.add_argument(
        "--dataset-root", "-dr",
        type=str,
        default="data",
        help="Root directory for datasets"
    )
    
    parser.add_argument(
        "--split", "-s",
        type=str,
        default="test",
        choices=["test", "dev", "train"],
        help="Dataset split to evaluate on"
    )
    
    parser.add_argument(
        "--max-samples", "-ms",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)"
    )
    
    # Processing settings
    parser.add_argument(
        "--batch-size", "-bs",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--num-workers", "-nw",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    
    # Metrics settings
    parser.add_argument(
        "--skip-per",
        action="store_true",
        help="Skip PER calculation"
    )
    
    parser.add_argument(
        "--skip-wer", 
        action="store_true",
        help="Skip WER calculation"
    )
    
    parser.add_argument(
        "--skip-bleu",
        action="store_true",
        help="Skip BLEU calculation"
    )
    
    parser.add_argument(
        "--skip-components",
        action="store_true",
        help="Skip component-wise metrics"
    )
    
    # Statistical analysis
    parser.add_argument(
        "--confidence-interval", "-ci",
        type=float,
        default=0.95,
        help="Confidence interval for statistical analysis"
    )
    
    parser.add_argument(
        "--bootstrap-samples", "-bs-samples",
        type=int,
        default=1000,
        help="Number of bootstrap samples for confidence intervals"
    )
    
    # Output settings
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save detailed results"
    )
    
    parser.add_argument(
        "--no-predictions",
        action="store_true", 
        help="Don't save individual predictions"
    )
    
    # Comparison
    parser.add_argument(
        "--baseline", "-bl",
        type=str,
        help="Path to baseline results for comparison"
    )
    
    # Debug settings
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug information"
    )
    
    parser.add_argument(
        "--max-debug-samples",
        type=int,
        default=100,
        help="Maximum number of samples to save debug info for"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    
    args = parse_args()
    
    logger.info("🔍 Valerie Visual ASR - Evaluation")
    logger.info("=" * 50)
    
    # Check model checkpoint
    if not Path(args.model).exists():
        logger.error(f"❌ Model checkpoint not found: {args.model}")
        return 1
    
    # Check dataset
    if not Path(args.dataset_root).exists():
        logger.error(f"❌ Dataset root not found: {args.dataset_root}")
        return 1
    
    # Create evaluation configuration
    eval_config = EvaluationConfig(
        model_checkpoint=args.model,
        device=args.device,
        dataset_root=args.dataset_root,
        dataset_split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        calculate_per=not args.skip_per,
        calculate_wer=not args.skip_wer,
        calculate_bleu=not args.skip_bleu,
        calculate_component_metrics=not args.skip_components,
        confidence_interval=args.confidence_interval,
        bootstrap_samples=args.bootstrap_samples,
        save_results=not args.no_save,
        save_predictions=not args.no_predictions,
        output_dir=args.output,
        baseline_results=args.baseline,
        save_debug_info=args.debug,
        max_debug_samples=args.max_debug_samples
    )
    
    logger.info("⚙️ Evaluation Configuration:")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Dataset: {args.dataset_root}/{args.split}")
    logger.info(f"   Max samples: {args.max_samples or 'All'}")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Output: {args.output}")
    
    # Initialize evaluator
    try:
        logger.info("🔧 Initializing evaluator...")
        evaluator = ValerieEvaluator(eval_config)
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize evaluator: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1
    
    # Run evaluation
    try:
        logger.info("🚀 Starting evaluation...")
        result = evaluator.evaluate()
        
        # Display results
        logger.info("✅ Evaluation completed!")
        logger.info("📊 Results Summary:")
        logger.info(f"   PER: {result.per:.4f} (95% CI: {result.per_ci[0]:.4f}-{result.per_ci[1]:.4f})")
        logger.info(f"   WER: {result.wer:.4f} (95% CI: {result.wer_ci[0]:.4f}-{result.wer_ci[1]:.4f})")
        logger.info(f"   BLEU: {result.bleu:.4f}")
        logger.info(f"   Confidence: {result.confidence:.4f}")
        
        logger.info(f"\n📈 Performance:")
        logger.info(f"   Processing time: {result.processing_time:.2f} seconds")
        logger.info(f"   Throughput: {result.throughput:.2f} samples/second")
        logger.info(f"   Success rate: {result.successful_samples}/{result.num_samples} ({result.successful_samples/result.num_samples*100:.1f}%)")
        
        # Component metrics
        if eval_config.calculate_component_metrics:
            logger.info(f"\n🔧 Component Analysis:")
            logger.info(f"   CTC PER: {result.component_metrics.ctc_per:.4f}")
            logger.info(f"   CTC Confidence: {result.component_metrics.ctc_confidence:.4f}")
            logger.info(f"   Reconstruction BLEU: {result.component_metrics.reconstruction_bleu:.4f}")
            logger.info(f"   Reconstruction Confidence: {result.component_metrics.reconstruction_confidence:.4f}")
        
        # Baseline comparison
        if args.baseline:
            try:
                comparison = evaluator.compare_with_baseline(result, args.baseline)
                
                logger.info(f"\n📊 Baseline Comparison:")
                logger.info(f"   PER improvement: {comparison['per_improvement_percent']:+.2f}%")
                logger.info(f"   WER improvement: {comparison['wer_improvement_percent']:+.2f}%")
                logger.info(f"   BLEU improvement: {comparison['bleu_improvement_percent']:+.2f}%")
                logger.info(f"   Statistical significance: {'Yes' if comparison['statistically_significant'] else 'No'} (p={comparison['per_p_value']:.4f})")
                
            except Exception as e:
                logger.warning(f"⚠️ Baseline comparison failed: {e}")
        
        # Save additional summary
        if not args.no_save:
            summary_file = Path(args.output) / "evaluation_summary.txt"
            with open(summary_file, 'w') as f:
                f.write("Valerie Visual ASR - Evaluation Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Model: {args.model}\n")
                f.write(f"Dataset: {args.dataset_root}/{args.split}\n")
                f.write(f"Samples: {result.num_samples}\n\n")
                f.write(f"PER: {result.per:.4f} (95% CI: {result.per_ci[0]:.4f}-{result.per_ci[1]:.4f})\n")
                f.write(f"WER: {result.wer:.4f} (95% CI: {result.wer_ci[0]:.4f}-{result.wer_ci[1]:.4f})\n")
                f.write(f"BLEU: {result.bleu:.4f}\n")
                f.write(f"Confidence: {result.confidence:.4f}\n\n")
                f.write(f"Processing time: {result.processing_time:.2f} seconds\n")
                f.write(f"Throughput: {result.throughput:.2f} samples/second\n")
                f.write(f"Success rate: {result.successful_samples/result.num_samples*100:.1f}%\n")
            
            logger.info(f"💾 Summary saved to: {summary_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
