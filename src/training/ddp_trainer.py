"""
PyTorch Native DDP Training for Valerie Visual ASR.

Optimized single-machine multi-GPU training using PyTorch's native
DistributedDataParallel (DDP) for maximum performance and minimal overhead.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
import json
from dataclasses import dataclass, asdict
from tqdm import tqdm
import signal
import atexit

from src.training.trainer import TrainingConfig
from src.training.losses import ValerieLossFunction
from src.training.metrics import ValerieMetrics
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DDPConfig:
    """DDP training configuration."""
    
    # GPU configuration
    gpus: List[int] = None  # GPU IDs to use, None = all available
    master_port: str = "12355"
    backend: str = "nccl"  # nccl, gloo
    
    # Performance optimization
    find_unused_parameters: bool = False
    bucket_cap_mb: int = 25
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    
    # Mixed precision
    mixed_precision: bool = True
    amp_backend: str = "native"  # native, apex
    
    # Data loading
    num_workers: int = 4  # Per GPU
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Checkpointing
    checkpoint_freq: int = 1
    save_top_k: int = 5
    
    # Monitoring
    log_freq: int = 100
    profile: bool = False


def cleanup():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_ddp(rank: int, world_size: int, master_port: str, backend: str = "nccl"):
    """Initialize distributed training."""
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    # Set CUDA device
    torch.cuda.set_device(rank)
    
    # Register cleanup
    atexit.register(cleanup)


def train_worker(
    rank: int,
    world_size: int,
    model_factory,
    train_dataset,
    val_dataset,
    training_config: TrainingConfig,
    ddp_config: DDPConfig,
    collate_fn=None,
    output_dir: str = "outputs/ddp_training"
):
    """Training worker function for each GPU."""
    
    try:
        # Setup distributed training
        setup_ddp(rank, world_size, ddp_config.master_port, ddp_config.backend)
        
        # Set device
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        
        # Only log from rank 0
        if rank == 0:
            logger.info(f"🚀 Starting DDP training on {world_size} GPUs")
            logger.info(f"   Backend: {ddp_config.backend}")
            logger.info(f"   Mixed precision: {ddp_config.mixed_precision}")
        
        # Create model
        model = model_factory()
        model = model.to(device)
        
        # Wrap with DDP
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=ddp_config.find_unused_parameters,
            bucket_cap_mb=ddp_config.bucket_cap_mb,
            gradient_as_bucket_view=ddp_config.gradient_as_bucket_view,
            static_graph=ddp_config.static_graph
        )
        
        # Create data loaders with distributed sampling
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            sampler=train_sampler,
            num_workers=ddp_config.num_workers,
            pin_memory=ddp_config.pin_memory,
            persistent_workers=ddp_config.persistent_workers and ddp_config.num_workers > 0,
            prefetch_factor=ddp_config.prefetch_factor if ddp_config.num_workers > 0 else None,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.batch_size,
            sampler=val_sampler,
            num_workers=ddp_config.num_workers,
            pin_memory=ddp_config.pin_memory,
            persistent_workers=ddp_config.persistent_workers and ddp_config.num_workers > 0,
            prefetch_factor=ddp_config.prefetch_factor if ddp_config.num_workers > 0 else None,
            collate_fn=collate_fn
        )
        
        # Setup training components
        loss_fn = ValerieLossFunction(
            ctc_weight=training_config.ctc_weight,
            attention_weight=training_config.attention_weight,
            distillation_weight=training_config.distillation_weight,
            temporal_consistency_weight=training_config.temporal_consistency_weight
        ).to(device)
        
        # Optimizer with learning rate scaling
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate * world_size,  # Scale LR
            weight_decay=training_config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config.max_epochs,
            eta_min=training_config.min_lr
        )
        
        # Mixed precision scaler
        scaler = GradScaler() if ddp_config.mixed_precision else None
        
        # Metrics
        metrics = ValerieMetrics()
        
        # Training state
        best_val_loss = float('inf')
        start_time = time.time()
        
        # Training loop
        for epoch in range(training_config.max_epochs):
            
            # Set epoch for distributed sampler
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            
            # Training phase
            model.train()
            train_losses = []
            metrics.reset()
            
            if rank == 0:
                train_pbar = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch+1}/{training_config.max_epochs}",
                    leave=False
                )
            else:
                train_pbar = train_loader
            
            for batch_idx, batch in enumerate(train_pbar):
                
                # Move batch to device
                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                optimizer.zero_grad()
                
                if scaler is not None:
                    with autocast():
                        outputs = model(batch)
                        loss_dict = loss_fn(outputs, batch)
                        loss = loss_dict['total_loss']
                    
                    # Backward pass
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(batch)
                    loss_dict = loss_fn(outputs, batch)
                    loss = loss_dict['total_loss']
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                    optimizer.step()
                
                # Update metrics
                if 'ctc_logits' in outputs and 'attention_logits' in outputs:
                    metrics.update(
                        outputs['ctc_logits'],
                        outputs['attention_logits'],
                        batch.get('phoneme_targets', batch.get('targets')),
                        batch.get('input_lengths'),
                        batch.get('target_lengths')
                    )
                
                train_losses.append(loss.item())
                
                # Update progress bar
                if rank == 0 and batch_idx % ddp_config.log_freq == 0:
                    train_pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                    })
            
            # Validation phase
            model.eval()
            val_losses = []
            val_metrics = ValerieMetrics()
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc="Validation", leave=False) if rank == 0 else val_loader
                
                for batch in val_pbar:
                    batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    if scaler is not None:
                        with autocast():
                            outputs = model(batch)
                            loss_dict = loss_fn(outputs, batch)
                            loss = loss_dict['total_loss']
                    else:
                        outputs = model(batch)
                        loss_dict = loss_fn(outputs, batch)
                        loss = loss_dict['total_loss']
                    
                    val_losses.append(loss.item())
                    
                    if 'ctc_logits' in outputs and 'attention_logits' in outputs:
                        val_metrics.update(
                            outputs['ctc_logits'],
                            outputs['attention_logits'],
                            batch.get('phoneme_targets', batch.get('targets')),
                            batch.get('input_lengths'),
                            batch.get('target_lengths')
                        )
            
            # Aggregate metrics across GPUs
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            # All-reduce losses
            train_loss_tensor = torch.tensor(avg_train_loss, device=device)
            val_loss_tensor = torch.tensor(avg_val_loss, device=device)
            
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            
            avg_train_loss = train_loss_tensor.item()
            avg_val_loss = val_loss_tensor.item()
            
            # Learning rate step
            scheduler.step()
            
            # Compute metrics
            train_metrics_dict = metrics.compute()
            val_metrics_dict = val_metrics.compute()
            
            # Logging and checkpointing (rank 0 only)
            if rank == 0:
                elapsed_time = time.time() - start_time
                
                logger.info(f"Epoch {epoch+1}/{training_config.max_epochs}:")
                logger.info(f"  Train Loss: {avg_train_loss:.4f}")
                logger.info(f"  Val Loss: {avg_val_loss:.4f}")
                logger.info(f"  Train PER: {train_metrics_dict.get('per', 0.0):.4f}")
                logger.info(f"  Val PER: {val_metrics_dict.get('per', 0.0):.4f}")
                logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
                logger.info(f"  Time: {elapsed_time/60:.1f}m")
                
                # Save checkpoint
                if (epoch + 1) % ddp_config.checkpoint_freq == 0:
                    checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict() if scaler else None,
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                        'config': asdict(training_config),
                        'ddp_config': asdict(ddp_config)
                    }
                    
                    # Save checkpoint
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    checkpoint_path = output_path / f"checkpoint_epoch_{epoch+1:03d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    
                    # Save best model
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_path = output_path / "best_model.pt"
                        torch.save(checkpoint, best_path)
                        logger.info(f"💾 Saved best model with val_loss: {avg_val_loss:.4f}")
            
            # Synchronize all processes
            dist.barrier()
        
        if rank == 0:
            total_time = time.time() - start_time
            logger.info(f"✅ Training completed in {total_time/3600:.2f} hours")
    
    except Exception as e:
        logger.error(f"❌ Training failed on rank {rank}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    finally:
        cleanup()


class DDPTrainer:
    """
    PyTorch native DDP trainer for single machine multi-GPU training.
    """
    
    def __init__(
        self,
        model_factory,
        train_dataset,
        val_dataset,
        training_config: TrainingConfig,
        ddp_config: DDPConfig,
        collate_fn=None,
        output_dir: str = "outputs/ddp_training"
    ):
        """
        Initialize DDP trainer.
        
        Args:
            model_factory: Function that returns a fresh model instance
            train_dataset: Training dataset
            val_dataset: Validation dataset
            training_config: Training configuration
            ddp_config: DDP configuration
            collate_fn: Custom collate function
            output_dir: Output directory for checkpoints
        """
        self.model_factory = model_factory
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_config = training_config
        self.ddp_config = ddp_config
        self.collate_fn = collate_fn
        self.output_dir = output_dir
        
        # Determine GPUs to use
        if ddp_config.gpus is None:
            self.gpus = list(range(torch.cuda.device_count()))
        else:
            self.gpus = ddp_config.gpus
        
        self.world_size = len(self.gpus)
        
        if self.world_size == 0:
            raise RuntimeError("No GPUs available for training")
        
        logger.info("✅ DDPTrainer initialized")
        logger.info(f"   GPUs: {self.gpus}")
        logger.info(f"   World size: {self.world_size}")
        logger.info(f"   Backend: {ddp_config.backend}")
        logger.info(f"   Batch size per GPU: {training_config.batch_size}")
        logger.info(f"   Total batch size: {training_config.batch_size * self.world_size}")
    
    def train(self):
        """Start DDP training."""
        
        logger.info("🚀 Starting PyTorch native DDP training")
        
        # Handle single GPU case
        if self.world_size == 1:
            logger.info("Single GPU training (no DDP needed)")
            train_worker(
                rank=0,
                world_size=1,
                model_factory=self.model_factory,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                training_config=self.training_config,
                ddp_config=self.ddp_config,
                collate_fn=self.collate_fn,
                output_dir=self.output_dir
            )
            return
        
        # Multi-GPU training with spawn
        try:
            mp.spawn(
                train_worker,
                args=(
                    self.world_size,
                    self.model_factory,
                    self.train_dataset,
                    self.val_dataset,
                    self.training_config,
                    self.ddp_config,
                    self.collate_fn,
                    self.output_dir
                ),
                nprocs=self.world_size,
                join=True
            )
            
            logger.info("✅ DDP training completed successfully")
            
        except Exception as e:
            logger.error(f"❌ DDP training failed: {e}")
            raise


def setup_ddp_environment():
    """Setup optimal environment for DDP training."""
    
    # CUDA settings
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(torch.cuda.device_count()))))
    
    # NCCL settings for optimal performance
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")  # Disable InfiniBand for single machine
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")  # Enable P2P for single machine
    os.environ.setdefault("NCCL_TREE_THRESHOLD", "0")  # Use tree algorithm
    
    # PyTorch distributed settings
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    
    # Disable tokenizer parallelism warnings
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    # Set optimal number of threads
    torch.set_num_threads(1)  # Important for multi-GPU
    
    logger.info("🔧 DDP environment configured for optimal performance")


# Utility functions for easy usage

def create_ddp_config(
    gpus: Optional[List[int]] = None,
    mixed_precision: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DDPConfig:
    """Create DDP configuration with sensible defaults."""
    
    return DDPConfig(
        gpus=gpus,
        mixed_precision=mixed_precision,
        num_workers=num_workers,
        **kwargs
    )


def quick_ddp_train(
    model_factory,
    train_dataset,
    val_dataset,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    max_epochs: int = 10,
    gpus: Optional[List[int]] = None,
    output_dir: str = "outputs/ddp_training",
    **kwargs
):
    """Quick DDP training with minimal configuration."""
    
    # Setup environment
    setup_ddp_environment()
    
    # Create configurations
    training_config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        **kwargs
    )
    
    ddp_config = create_ddp_config(gpus=gpus)
    
    # Create trainer
    trainer = DDPTrainer(
        model_factory=model_factory,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_config=training_config,
        ddp_config=ddp_config,
        output_dir=output_dir
    )
    
    # Start training
    trainer.train()
