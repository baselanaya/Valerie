#!/usr/bin/env python3
"""
Multi-GPU Training Comparison for Valerie Visual ASR
====================================================

Compare different multi-GPU training methods:
1. PyTorch Native DDP (fastest, most efficient)
2. Accelerate (easiest to use, great features)
3. Ray (for complex distributed setups)

This example demonstrates the trade-offs and shows how to use each method.
"""

import sys
import os
from pathlib import Path
import torch
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.training import (
    DDPTrainer, DDPConfig, setup_ddp_environment,
    TrainingConfig
)
from src.models import ValerieModel
from src.data import VoxCeleb2Dataset, ValerieCollator
from src.utils.config import Config
from src.utils.logging import setup_logger, get_logger

# Setup logging
setup_logger(use_rich=True)
logger = get_logger(__name__)


def create_test_setup():
    """Create small test setup for comparison."""
    
    # Load model configuration
    model_config = Config('configs/base_config.yaml')
    
    # Create small datasets
    train_dataset = VoxCeleb2Dataset(
        voxceleb2_root="data",
        split="dev",
        max_samples=50,  # Very small for testing
        use_video=True,
        use_transcriptions=True
    )
    
    val_dataset = VoxCeleb2Dataset(
        voxceleb2_root="data",
        split="test",
        max_samples=20,  # Very small for testing
        use_video=True,
        use_transcriptions=True
    )
    
    # Create model factory
    def model_factory():
        return ValerieModel(model_config)
    
    # Create collate function
    collate_fn = ValerieCollator(model_config)
    
    # Training configuration
    training_config = TrainingConfig(
        batch_size=2,  # Small batch size for testing
        learning_rate=1e-4,
        max_epochs=3,  # Short training
        warmup_steps=10,
        max_grad_norm=1.0
    )
    
    return model_factory, train_dataset, val_dataset, collate_fn, training_config


def benchmark_ddp_training():
    """Benchmark PyTorch native DDP training."""
    
    logger.info("🚀 Benchmarking PyTorch Native DDP")
    logger.info("=" * 50)
    
    # Check GPU availability
    if torch.cuda.device_count() < 2:
        logger.warning("⚠️ Less than 2 GPUs available, skipping multi-GPU test")
        return None
    
    try:
        # Setup
        model_factory, train_dataset, val_dataset, collate_fn, training_config = create_test_setup()
        
        # Setup DDP environment
        setup_ddp_environment()
        
        # DDP configuration
        ddp_config = DDPConfig(
            gpus=[0, 1],  # Use first 2 GPUs
            mixed_precision=True,
            num_workers=2,
            find_unused_parameters=False
        )
        
        # Create trainer
        trainer = DDPTrainer(
            model_factory=model_factory,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_config=training_config,
            ddp_config=ddp_config,
            collate_fn=collate_fn,
            output_dir="outputs/ddp_benchmark"
        )
        
        # Benchmark training
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        logger.info("✅ PyTorch DDP Results:")
        logger.info(f"   Training time: {training_time:.2f} seconds")
        logger.info(f"   GPUs used: {ddp_config.gpus}")
        logger.info(f"   Backend: {ddp_config.backend}")
        
        return {
            'method': 'PyTorch DDP',
            'training_time': training_time,
            'gpus_used': len(ddp_config.gpus),
            'backend': ddp_config.backend
        }
        
    except Exception as e:
        logger.error(f"❌ DDP training failed: {e}")
        return None


def benchmark_accelerate_training():
    """Benchmark Accelerate training."""
    
    logger.info("\n🤗 Benchmarking Accelerate Training")
    logger.info("=" * 50)
    
    try:
        from src.training import AccelerateTrainer, AccelerateConfig, ACCELERATE_AVAILABLE
        
        if not ACCELERATE_AVAILABLE:
            logger.warning("⚠️ Accelerate not available, skipping test")
            logger.info("💡 Install with: pip install accelerate")
            return None
        
        # Check GPU availability
        if torch.cuda.device_count() < 2:
            logger.warning("⚠️ Less than 2 GPUs available, using single GPU")
        
        # Setup
        model_factory, train_dataset, val_dataset, collate_fn, training_config = create_test_setup()
        
        # Create data loaders
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        # Create model
        model = model_factory()
        
        # Accelerate configuration
        accelerate_config = AccelerateConfig(
            mixed_precision="fp16",
            gradient_accumulation_steps=1,
            dataloader_num_workers=2,
            project_dir="outputs/accelerate_benchmark"
        )
        
        # Create trainer
        trainer = AccelerateTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            training_config=training_config,
            accelerate_config=accelerate_config,
            output_dir="outputs/accelerate_benchmark"
        )
        
        # Benchmark training
        start_time = time.time()
        history = trainer.train()
        training_time = time.time() - start_time
        
        logger.info("✅ Accelerate Results:")
        logger.info(f"   Training time: {training_time:.2f} seconds")
        logger.info(f"   GPUs used: {trainer.accelerator.num_processes}")
        logger.info(f"   Mixed precision: {accelerate_config.mixed_precision}")
        
        return {
            'method': 'Accelerate',
            'training_time': training_time,
            'gpus_used': trainer.accelerator.num_processes,
            'mixed_precision': accelerate_config.mixed_precision
        }
        
    except Exception as e:
        logger.error(f"❌ Accelerate training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def benchmark_ray_training():
    """Benchmark Ray training."""
    
    logger.info("\n☁️ Benchmarking Ray Training")
    logger.info("=" * 50)
    
    try:
        from src.training import DistributedTrainer, DistributedConfig
        
        # Check if Ray is available
        import ray
        
        # Check GPU availability
        if torch.cuda.device_count() < 2:
            logger.warning("⚠️ Less than 2 GPUs available, using single GPU")
        
        # Setup
        model_factory, train_dataset, val_dataset, collate_fn, training_config = create_test_setup()
        
        # Ray configuration
        distributed_config = DistributedConfig(
            num_workers=min(2, torch.cuda.device_count()),
            use_gpu=True,
            gpus_per_worker=1.0,
            cpus_per_worker=2,
            data_sharding=True,
            mixed_precision="fp16"
        )
        
        # Create trainer
        trainer = DistributedTrainer(
            model_factory=model_factory,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_config=training_config,
            distributed_config=distributed_config,
            collate_fn=collate_fn
        )
        
        # Benchmark training
        start_time = time.time()
        result = trainer.train()
        training_time = time.time() - start_time
        
        logger.info("✅ Ray Results:")
        logger.info(f"   Training time: {training_time:.2f} seconds")
        logger.info(f"   Workers: {distributed_config.num_workers}")
        logger.info(f"   GPUs per worker: {distributed_config.gpus_per_worker}")
        
        return {
            'method': 'Ray',
            'training_time': training_time,
            'workers': distributed_config.num_workers,
            'gpus_per_worker': distributed_config.gpus_per_worker
        }
        
    except ImportError:
        logger.warning("⚠️ Ray not available, skipping test")
        logger.info("💡 Install with: pip install ray[train]")
        return None
    except Exception as e:
        logger.error(f"❌ Ray training failed: {e}")
        return None


def main():
    """Run multi-GPU training comparison."""
    
    logger.info("🏁 Multi-GPU Training Method Comparison")
    logger.info("=" * 60)
    
    # Check system
    num_gpus = torch.cuda.device_count()
    logger.info(f"🎮 System: {num_gpus} GPU(s) available")
    
    if num_gpus == 0:
        logger.error("❌ No GPUs available. This comparison requires at least 1 GPU.")
        return 1
    
    for i in range(num_gpus):
        gpu_props = torch.cuda.get_device_properties(i)
        logger.info(f"   GPU {i}: {gpu_props.name}")
    
    # Run benchmarks
    results = []
    
    # 1. PyTorch Native DDP
    ddp_result = benchmark_ddp_training()
    if ddp_result:
        results.append(ddp_result)
    
    # 2. Accelerate
    accelerate_result = benchmark_accelerate_training()
    if accelerate_result:
        results.append(accelerate_result)
    
    # 3. Ray (commented out to avoid complexity)
    # ray_result = benchmark_ray_training()
    # if ray_result:
    #     results.append(ray_result)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 COMPARISON SUMMARY")
    logger.info("=" * 60)
    
    if not results:
        logger.warning("⚠️ No benchmarks completed successfully")
        return 1
    
    # Sort by training time
    results.sort(key=lambda x: x['training_time'])
    
    for i, result in enumerate(results, 1):
        logger.info(f"{i}. {result['method']}:")
        logger.info(f"   Training time: {result['training_time']:.2f}s")
        if 'gpus_used' in result:
            logger.info(f"   GPUs used: {result['gpus_used']}")
        if 'backend' in result:
            logger.info(f"   Backend: {result['backend']}")
        if 'mixed_precision' in result:
            logger.info(f"   Mixed precision: {result['mixed_precision']}")
    
    # Recommendations
    logger.info("\n💡 RECOMMENDATIONS:")
    logger.info("=" * 30)
    
    if num_gpus <= 8:
        logger.info("🥇 **PyTorch Native DDP** - Best performance for single machine")
        logger.info("   ✅ Fastest training speed")
        logger.info("   ✅ Minimal overhead")
        logger.info("   ✅ Battle-tested and stable")
        logger.info("   ✅ No extra dependencies")
        
        logger.info("\n🥈 **Accelerate** - Best for ease of use")
        logger.info("   ✅ Minimal code changes")
        logger.info("   ✅ Excellent documentation")
        logger.info("   ✅ Great for research/prototyping")
        logger.info("   ✅ Built-in logging integration")
    
    if num_gpus > 8:
        logger.info("\n🥉 **Ray** - Best for very large scale")
        logger.info("   ✅ Multi-node support")
        logger.info("   ✅ Fault tolerance")
        logger.info("   ✅ Resource management")
        logger.info("   ⚠️ More complex setup")
    
    logger.info("\n🎯 **For your single machine multi-GPU setup:**")
    logger.info("   Use: python train_multi_gpu.py --gpus 0,1,2,3 --batch-size 8")
    logger.info("   Or:  python train_multi_gpu.py --accelerate --batch-size 8")
    
    return 0


if __name__ == "__main__":
    exit(main())
