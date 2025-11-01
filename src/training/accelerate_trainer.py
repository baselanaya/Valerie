"""
Accelerate-based Training for Valerie Visual ASR.

User-friendly multi-GPU training using Hugging Face Accelerate library.
Minimal code changes required, excellent for rapid prototyping and research.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
import json
from dataclasses import dataclass, asdict
from tqdm import tqdm

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed, DistributedDataParallelKwargs
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

from src.training.trainer import TrainingConfig
from src.training.losses import ValerieLossFunction
from src.training.metrics import ValerieMetrics
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AccelerateConfig:
    """Accelerate training configuration."""
    
    # Mixed precision
    mixed_precision: str = "fp16"  # fp16, bf16, no
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Logging
    log_with: Optional[str] = None  # tensorboard, wandb, all
    project_dir: str = "outputs/accelerate_training"
    
    # DDP settings
    find_unused_parameters: bool = False
    bucket_cap_mb: int = 25
    
    # Data loading
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Checkpointing
    save_every: int = 1  # Save every N epochs
    save_total_limit: int = 5
    
    # Other settings
    cpu: bool = False  # Force CPU training
    seed: int = 42


class AccelerateTrainer:
    """
    Accelerate-based trainer for easy multi-GPU training.
    
    Advantages:
    - Minimal code changes
    - Automatic device placement
    - Built-in mixed precision
    - Easy gradient accumulation
    - Excellent logging integration
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader,
        val_dataloader,
        training_config: TrainingConfig,
        accelerate_config: AccelerateConfig,
        output_dir: str = "outputs/accelerate_training"
    ):
        """
        Initialize Accelerate trainer.
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            training_config: Training configuration
            accelerate_config: Accelerate configuration
            output_dir: Output directory for checkpoints
        """
        if not ACCELERATE_AVAILABLE:
            raise ImportError(
                "Accelerate is required for this trainer. "
                "Install with: pip install accelerate"
            )
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.training_config = training_config
        self.accelerate_config = accelerate_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed
        set_seed(accelerate_config.seed)
        
        # Initialize Accelerator
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=accelerate_config.find_unused_parameters,
            bucket_cap_mb=accelerate_config.bucket_cap_mb
        )
        
        self.accelerator = Accelerator(
            mixed_precision=accelerate_config.mixed_precision,
            gradient_accumulation_steps=accelerate_config.gradient_accumulation_steps,
            log_with=accelerate_config.log_with,
            project_dir=str(self.output_dir),
            kwargs_handlers=[ddp_kwargs],
            cpu=accelerate_config.cpu
        )
        
        # Setup logging
        if self.accelerator.is_main_process:
            logger.info("✅ AccelerateTrainer initialized")
            logger.info(f"   Device: {self.accelerator.device}")
            logger.info(f"   Num processes: {self.accelerator.num_processes}")
            logger.info(f"   Mixed precision: {accelerate_config.mixed_precision}")
            logger.info(f"   Gradient accumulation: {accelerate_config.gradient_accumulation_steps}")
        
        # Initialize training components
        self._setup_training_components()
    
    def _setup_training_components(self):
        """Setup optimizer, scheduler, and loss function."""
        
        # Loss function
        self.loss_fn = ValerieLossFunction(
            ctc_weight=self.training_config.ctc_weight,
            attention_weight=self.training_config.attention_weight,
            distillation_weight=self.training_config.distillation_weight,
            temporal_consistency_weight=self.training_config.temporal_consistency_weight
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.training_config.max_epochs,
            eta_min=self.training_config.min_lr
        )
        
        # Prepare everything with Accelerator
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
            self.scheduler
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
            self.scheduler
        )
        
        # Metrics
        self.metrics = ValerieMetrics()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        self.metrics.reset()
        
        # Progress bar (only on main process)
        if self.accelerator.is_main_process:
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.training_config.max_epochs}"
            )
        else:
            progress_bar = self.train_dataloader
        
        for step, batch in enumerate(progress_bar):
            
            with self.accelerator.accumulate(self.model):
                # Forward pass
                outputs = self.model(batch)
                loss_dict = self.loss_fn(outputs, batch)
                loss = loss_dict['total_loss']
                
                # Scale loss for gradient accumulation
                loss = loss / self.accelerate_config.gradient_accumulation_steps
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.max_grad_norm
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Update metrics
            if 'ctc_logits' in outputs and 'attention_logits' in outputs:
                self.metrics.update(
                    outputs['ctc_logits'],
                    outputs['attention_logits'],
                    batch.get('phoneme_targets', batch.get('targets')),
                    batch.get('input_lengths'),
                    batch.get('target_lengths')
                )
            
            # Accumulate loss
            total_loss += loss.item() * self.accelerate_config.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            if self.accelerator.is_main_process and step % 100 == 0:
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
        
        # Gather metrics across processes
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        train_metrics = self.metrics.compute()
        
        return {
            'train_loss': avg_loss,
            **{f'train_{k}': v for k, v in train_metrics.items()}
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        val_metrics = ValerieMetrics()
        
        with torch.no_grad():
            if self.accelerator.is_main_process:
                progress_bar = tqdm(self.val_dataloader, desc="Validation")
            else:
                progress_bar = self.val_dataloader
            
            for batch in progress_bar:
                # Forward pass
                outputs = self.model(batch)
                loss_dict = self.loss_fn(outputs, batch)
                loss = loss_dict['total_loss']
                
                # Update metrics
                if 'ctc_logits' in outputs and 'attention_logits' in outputs:
                    val_metrics.update(
                        outputs['ctc_logits'],
                        outputs['attention_logits'],
                        batch.get('phoneme_targets', batch.get('targets')),
                        batch.get('input_lengths'),
                        batch.get('target_lengths')
                    )
                
                total_loss += loss.item()
                num_batches += 1
        
        # Gather metrics across processes
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        validation_metrics = val_metrics.compute()
        
        return {
            'val_loss': avg_loss,
            **{f'val_{k}': v for k, v in validation_metrics.items()}
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        
        if not self.accelerator.is_main_process:
            return
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'training_config': asdict(self.training_config),
            'accelerate_config': asdict(self.accelerate_config)
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model based on validation loss
        if hasattr(self, 'best_val_loss'):
            if metrics.get('val_loss', float('inf')) < self.best_val_loss:
                self.best_val_loss = metrics['val_loss']
                best_path = self.output_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                logger.info(f"💾 Saved best model with val_loss: {self.best_val_loss:.4f}")
        else:
            self.best_val_loss = metrics.get('val_loss', float('inf'))
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop."""
        
        if self.accelerator.is_main_process:
            logger.info("🚀 Starting Accelerate training")
            logger.info(f"   Training samples: {len(self.train_dataloader.dataset)}")
            logger.info(f"   Validation samples: {len(self.val_dataloader.dataset)}")
            logger.info(f"   Batch size per device: {self.train_dataloader.batch_size}")
            logger.info(f"   Total batch size: {self.train_dataloader.batch_size * self.accelerator.num_processes}")
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_per': [],
            'val_per': [],
            'learning_rate': []
        }
        
        start_time = time.time()
        
        for epoch in range(self.training_config.max_epochs):
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Learning rate step
            self.scheduler.step()
            
            # Update history
            history['train_loss'].append(train_metrics.get('train_loss', 0.0))
            history['val_loss'].append(val_metrics.get('val_loss', 0.0))
            history['train_per'].append(train_metrics.get('train_per', 0.0))
            history['val_per'].append(val_metrics.get('val_per', 0.0))
            history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Logging
            if self.accelerator.is_main_process:
                logger.info(f"Epoch {epoch+1}/{self.training_config.max_epochs}:")
                logger.info(f"  Train Loss: {train_metrics.get('train_loss', 0.0):.4f}")
                logger.info(f"  Val Loss: {val_metrics.get('val_loss', 0.0):.4f}")
                logger.info(f"  Train PER: {train_metrics.get('train_per', 0.0):.4f}")
                logger.info(f"  Val PER: {val_metrics.get('val_per', 0.0):.4f}")
                logger.info(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Log to tracker (if configured)
            if self.accelerator.is_main_process and self.accelerate_config.log_with:
                self.accelerator.log(epoch_metrics, step=epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.accelerate_config.save_every == 0:
                self.save_checkpoint(epoch + 1, epoch_metrics)
            
            # Wait for all processes
            self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            training_time = time.time() - start_time
            logger.info(f"✅ Training completed in {training_time/3600:.2f} hours")
        
        return history
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device)
        
        # Load model state
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.accelerator.is_main_process:
            logger.info(f"📥 Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint


# Utility functions

def create_accelerate_config(
    mixed_precision: str = "fp16",
    gradient_accumulation_steps: int = 1,
    log_with: Optional[str] = None,
    **kwargs
) -> AccelerateConfig:
    """Create Accelerate configuration with sensible defaults."""
    
    return AccelerateConfig(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with=log_with,
        **kwargs
    )


def quick_accelerate_train(
    model: nn.Module,
    train_dataloader,
    val_dataloader,
    learning_rate: float = 1e-4,
    max_epochs: int = 10,
    mixed_precision: str = "fp16",
    output_dir: str = "outputs/accelerate_training",
    **kwargs
):
    """Quick Accelerate training with minimal configuration."""
    
    # Create configurations
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        **kwargs
    )
    
    accelerate_config = create_accelerate_config(
        mixed_precision=mixed_precision
    )
    
    # Create trainer
    trainer = AccelerateTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        training_config=training_config,
        accelerate_config=accelerate_config,
        output_dir=output_dir
    )
    
    # Start training
    return trainer.train()
