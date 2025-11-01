"""
Training loop implementation for Valerie Visual ASR.

Implements mixed precision training, gradient accumulation, learning rate scheduling,
and comprehensive logging for robust model training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
import json
from dataclasses import dataclass, asdict
from tqdm import tqdm

from src.training.losses import ValerieLossFunction
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    optimizer: str = "adamw"  # adamw, adam, sgd
    
    # Training schedule
    max_epochs: int = 100
    warmup_steps: int = 4000
    max_grad_norm: float = 1.0
    accumulate_grad_batches: int = 1
    
    # Mixed precision
    use_mixed_precision: bool = True
    
    # Learning rate scheduling
    scheduler: str = "cosine"  # cosine, linear, exponential, plateau
    min_lr: float = 1e-6
    
    # Validation and checkpointing
    val_check_interval: float = 0.25  # Fraction of epoch
    save_top_k: int = 3
    patience: int = 10
    
    # Loss weights
    ctc_weight: float = 0.3
    attention_weight: float = 0.7
    distillation_weight: float = 0.1
    temporal_consistency_weight: float = 0.05


class ValerieTrainer:
    """
    Main training class for Valerie Visual ASR.
    
    Handles mixed precision training, gradient accumulation, checkpointing,
    and comprehensive logging for visual speech recognition training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: TrainingConfig,
        save_dir: str = "checkpoints",
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Valerie model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            config: Training configuration
            save_dir: Directory to save checkpoints
            device: Training device (auto-detected if None)
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Initialize training components
        self._setup_loss_function()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_mixed_precision()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        logger.info(f"✅ ValerieTrainer initialized:")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Mixed precision: {config.use_mixed_precision}")
        logger.info(f"   Optimizer: {config.optimizer}")
        logger.info(f"   Scheduler: {config.scheduler}")
        logger.info(f"   Max epochs: {config.max_epochs}")
    
    def _setup_loss_function(self):
        """Initialize loss function."""
        self.loss_fn = ValerieLossFunction(
            ctc_weight=self.config.ctc_weight,
            attention_weight=self.config.attention_weight,
            distillation_weight=self.config.distillation_weight,
            temporal_consistency_weight=self.config.temporal_consistency_weight,
            adaptive_weighting=True
        ).to(self.device)
    
    def _setup_optimizer(self):
        """Initialize optimizer."""
        if self.config.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.98),
                eps=1e-9
            )
        elif self.config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _setup_scheduler(self):
        """Initialize learning rate scheduler."""
        if self.config.scheduler.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler.lower() == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.min_lr / self.config.learning_rate,
                total_iters=self.config.max_epochs
            )
        elif self.config.scheduler.lower() == "exponential":
            gamma = (self.config.min_lr / self.config.learning_rate) ** (1.0 / self.config.max_epochs)
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=gamma
            )
        elif self.config.scheduler.lower() == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=self.config.min_lr
            )
        else:
            self.scheduler = None
    
    def _setup_mixed_precision(self):
        """Initialize mixed precision training."""
        if self.config.use_mixed_precision and torch.cuda.is_available():
            self.scaler = GradScaler()
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}
        num_batches = len(self.train_dataloader)
        
        # Progress bar
        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch+1}/{self.config.max_epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with mixed precision
            loss_dict = self._forward_step(batch)
            
            # Backward pass
            self._backward_step(loss_dict['total_loss'])
            
            # Update metrics
            self._update_epoch_losses(epoch_losses, loss_dict)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.config.accumulate_grad_batches == 0:
                self._optimizer_step()
                self.global_step += 1
        
        # Average losses over epoch
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = {}
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                batch = self._move_batch_to_device(batch)
                loss_dict = self._forward_step(batch)
                self._update_epoch_losses(epoch_losses, loss_dict)
        
        # Average losses
        num_batches = len(self.val_dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through model and loss computation."""
        
        if self.use_amp:
            with autocast():
                outputs = self.model(batch)
                loss_dict = self.loss_fn(outputs, batch)
        else:
            outputs = self.model(batch)
            loss_dict = self.loss_fn(outputs, batch)
        
        return loss_dict
    
    def _backward_step(self, loss: torch.Tensor):
        """Backward pass with gradient scaling."""
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.accumulate_grad_batches
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def _optimizer_step(self):
        """Optimizer step with gradient clipping."""
        
        if self.use_amp:
            # Gradient clipping with scaling
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            # Optimizer step
            self.optimizer.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        
        return device_batch
    
    def _update_epoch_losses(
        self,
        epoch_losses: Dict[str, float],
        batch_losses: Dict[str, torch.Tensor]
    ):
        """Update running epoch losses."""
        for key, value in batch_losses.items():
            if isinstance(value, torch.Tensor):
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value.item()
    
    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'val_loss': val_loss,
            'config': asdict(self.config),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"💾 Saved best model with val_loss: {val_loss:.4f}")
        
        # Keep only top-k checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only top-k."""
        checkpoints = list(self.save_dir.glob("checkpoint_epoch_*.pt"))
        
        if len(checkpoints) > self.config.save_top_k:
            # Sort by validation loss (stored in filename or load from checkpoint)
            checkpoints_with_loss = []
            for cp_path in checkpoints:
                try:
                    cp = torch.load(cp_path, map_location='cpu')
                    checkpoints_with_loss.append((cp_path, cp['val_loss']))
                except:
                    # If can't load, assume high loss
                    checkpoints_with_loss.append((cp_path, float('inf')))
            
            # Sort by loss and keep top-k
            checkpoints_with_loss.sort(key=lambda x: x[1])
            to_remove = checkpoints_with_loss[self.config.save_top_k:]
            
            for cp_path, _ in to_remove:
                cp_path.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"📥 Loaded checkpoint from epoch {self.current_epoch}")
    
    def fit(self) -> Dict[str, List[float]]:
        """Main training loop."""
        
        logger.info("🚀 Starting training...")
        logger.info(f"   Training samples: {len(self.train_dataloader.dataset)}")
        logger.info(f"   Validation samples: {len(self.val_dataloader.dataset)}")
        logger.info(f"   Batch size: {self.train_dataloader.batch_size}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            
            # Training
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)
            
            # Validation
            val_losses = self.validate_epoch()
            self.val_losses.append(val_losses)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total_loss'])
                else:
                    self.scheduler.step()
            
            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            logger.info(f"Epoch {epoch+1}/{self.config.max_epochs}:")
            logger.info(f"   Train Loss: {train_losses['total_loss']:.4f}")
            logger.info(f"   Val Loss: {val_losses['total_loss']:.4f}")
            logger.info(f"   Learning Rate: {current_lr:.2e}")
            
            # Checkpointing
            is_best = val_losses['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch + 1, val_losses['total_loss'], is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"⏹️ Early stopping after {self.config.patience} epochs without improvement")
                break
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time/3600:.2f} hours")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
