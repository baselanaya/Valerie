#!/usr/bin/env python3
"""
Multi-GPU Training Script for Valerie Visual ASR
================================================

Optimized single-machine multi-GPU training using PyTorch native DDP.
Best performance and minimal overhead for 2-8 GPUs.

Usage:
    python train_multi_gpu.py --gpus 0,1,2,3 --batch-size 8
    python train_multi_gpu.py --config configs/base_config.yaml --epochs 50
    python train_multi_gpu.py --accelerate  # Use Accelerate instead of DDP
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import torch
from dataclasses import asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.training.ddp_trainer import DDPTrainer, DDPConfig, setup_ddp_environment
from src.training.accelerate_trainer import AccelerateTrainer, AccelerateConfig
from src.training.trainer import TrainingConfig
from src.models import ValerieModel
from src.data import VoxCeleb2Dataset, ValerieCollator
from src.utils.config import Config
from src.utils.logging import setup_logger, get_logger

# Setup logging
setup_logger(use_rich=True)
logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-GPU training for Valerie Visual ASR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/base_config.yaml",
        help="Path to model configuration file"
    )
    
    # GPU settings
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="GPU IDs to use (e.g., '0,1,2,3' or 'all')"
    )
    
    parser.add_argument(
        "--accelerate",
        action="store_true",
        help="Use Accelerate instead of native DDP"
    )
    
    # Training settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per GPU"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    
    # Data settings
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory for datasets"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for testing)"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/multi_gpu_training",
        help="Output directory for checkpoints and logs"
    )
    
    # Performance settings
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32", "no"],
        help="Mixed precision training"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers per GPU"
    )
    
    # Debugging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()


def parse_gpu_ids(gpu_str: str) -> list:
    """Parse GPU IDs from string."""
    
    if gpu_str is None or gpu_str.lower() == "all":
        return list(range(torch.cuda.device_count()))
    
    try:
        return [int(x.strip()) for x in gpu_str.split(",")]
    except ValueError:
        raise ValueError(f"Invalid GPU specification: {gpu_str}")


def create_datasets(data_root: str, max_samples: int = None):
    """Create training and validation datasets."""
    
    logger.info(f"📂 Loading datasets from {data_root}")
    
    # Training dataset
    train_dataset = VoxCeleb2Dataset(
        voxceleb2_root=data_root,
        split="dev",
        max_samples=max_samples,
        use_video=True,
        use_transcriptions=True
    )
    
    # Validation dataset
    val_dataset = VoxCeleb2Dataset(
        voxceleb2_root=data_root,
        split="test",
        max_samples=max_samples // 10 if max_samples else None,
        use_video=True,
        use_transcriptions=True
    )
    
    logger.info(f"✅ Datasets loaded:")
    logger.info(f"   Training samples: {len(train_dataset)}")
    logger.info(f"   Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def create_data_loaders(train_dataset, val_dataset, batch_size: int, num_workers: int):
    """Create data loaders (for Accelerate)."""
    
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False
    )
    
    return train_loader, val_loader


def main():
    """Main training function."""
    
    args = parse_args()
    
    logger.info("🚀 Multi-GPU Training for Valerie Visual ASR")
    logger.info("=" * 60)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available. Multi-GPU training requires GPUs.")
        return 1
    
    num_gpus = torch.cuda.device_count()
    logger.info(f"🎮 Found {num_gpus} GPU(s)")
    
    for i in range(num_gpus):
        gpu_props = torch.cuda.get_device_properties(i)
        logger.info(f"   GPU {i}: {gpu_props.name} ({gpu_props.total_memory/1e9:.1f} GB)")
    
    # Parse GPU IDs
    gpu_ids = parse_gpu_ids(args.gpus)
    logger.info(f"🎯 Using GPUs: {gpu_ids}")
    
    if len(gpu_ids) == 0:
        logger.error("❌ No GPUs specified for training")
        return 1
    
    # Load model configuration
    logger.info(f"🏗️ Loading model configuration from {args.config}")
    model_config = Config(args.config)
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(args.data_root, args.max_samples)
    
    # Create collate function
    collate_fn = ValerieCollator(model_config)
    
    # Create training configuration
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        warmup_steps=4000,
        max_grad_norm=1.0,
        ctc_weight=0.3,
        attention_weight=0.7,
        distillation_weight=0.1,
        temporal_consistency_weight=0.05
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configurations
    with open(output_dir / "training_config.yaml", 'w') as f:
        yaml.dump(asdict(training_config), f, default_flow_style=False)
    
    logger.info("⚙️ Training Configuration:")
    logger.info(f"   Batch size per GPU: {training_config.batch_size}")
    logger.info(f"   Total batch size: {training_config.batch_size * len(gpu_ids)}")
    logger.info(f"   Learning rate: {training_config.learning_rate}")
    logger.info(f"   Max epochs: {training_config.max_epochs}")
    logger.info(f"   Mixed precision: {args.mixed_precision}")
    
    # Choose training method
    if args.accelerate:
        logger.info("🤗 Using Accelerate for training")
        
        # Check if Accelerate is available
        try:
            from src.training.accelerate_trainer import AccelerateTrainer, AccelerateConfig, ACCELERATE_AVAILABLE
            if not ACCELERATE_AVAILABLE:
                raise ImportError("Accelerate not available")
        except ImportError:
            logger.error("❌ Accelerate not available. Install with: pip install accelerate")
            logger.info("💡 Falling back to native DDP training")
            args.accelerate = False
        
        if args.accelerate:
            # Create model and data loaders
            model = ValerieModel(model_config)
            train_loader, val_loader = create_data_loaders(
                train_dataset, val_dataset, args.batch_size, args.num_workers
            )
            
            # Create Accelerate configuration
            accelerate_config = AccelerateConfig(
                mixed_precision=args.mixed_precision if args.mixed_precision != "fp32" else "no",
                dataloader_num_workers=args.num_workers,
                project_dir=str(output_dir)
            )
            
            # Create trainer
            trainer = AccelerateTrainer(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                training_config=training_config,
                accelerate_config=accelerate_config,
                output_dir=str(output_dir)
            )
    
    if not args.accelerate:
        logger.info("⚡ Using PyTorch native DDP for training")
        
        # Setup DDP environment
        setup_ddp_environment()
        
        # Create model factory
        def model_factory():
            return ValerieModel(model_config)
        
        # Create DDP configuration
        ddp_config = DDPConfig(
            gpus=gpu_ids,
            mixed_precision=args.mixed_precision in ["fp16", "bf16"],
            num_workers=args.num_workers,
            find_unused_parameters=False,
            bucket_cap_mb=25
        )
        
        # Save DDP configuration
        with open(output_dir / "ddp_config.yaml", 'w') as f:
            yaml.dump(asdict(ddp_config), f, default_flow_style=False)
        
        # Create trainer
        trainer = DDPTrainer(
            model_factory=model_factory,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_config=training_config,
            ddp_config=ddp_config,
            collate_fn=collate_fn,
            output_dir=str(output_dir)
        )
    
    # Start training
    logger.info("🚀 Starting multi-GPU training...")
    logger.info(f"   Method: {'Accelerate' if args.accelerate else 'PyTorch DDP'}")
    logger.info(f"   GPUs: {len(gpu_ids)}")
    logger.info(f"   Total batch size: {training_config.batch_size * len(gpu_ids)}")
    
    try:
        start_time = time.time()
        
        if args.accelerate:
            history = trainer.train()
        else:
            trainer.train()
            history = None
        
        training_time = time.time() - start_time
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"⏱️ Total training time: {training_time/3600:.2f} hours")
        
        if history:
            logger.info("📊 Final metrics:")
            if history.get('val_loss'):
                logger.info(f"   Final val loss: {history['val_loss'][-1]:.4f}")
            if history.get('val_per'):
                logger.info(f"   Final val PER: {history['val_per'][-1]:.4f}")
        
        # Save training summary
        summary = {
            'training_time_hours': training_time / 3600,
            'num_gpus': len(gpu_ids),
            'gpu_ids': gpu_ids,
            'method': 'Accelerate' if args.accelerate else 'PyTorch DDP',
            'final_metrics': history[-1] if history else None
        }
        
        with open(output_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"💾 Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    import time
    import json
    exit(main())
