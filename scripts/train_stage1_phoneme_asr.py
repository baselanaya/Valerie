"""
Stage 1 Training: Audio → Phonemes with Ensemble Distillation

This script trains the audio-to-phoneme ASR model (student) using:
- LibriSpeech dataset (30-100 hours)
- Ensemble knowledge distillation from 3 teacher models:
  1. Whisper Large V3
  2. Wav2Vec2 Large
  3. HuBERT Large

Target: <35% PER (Phoneme Error Rate)

Training Strategy:
1. Train with CTC loss on phoneme alignments (MFA)
2. Add distillation loss from ensemble teachers
3. Fine-tune with hybrid CTC/Attention
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from pathlib import Path
import wandb
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.audio_phoneme_model import AudioPhonemeASR
from src.models.ensemble_distillation import create_ensemble_distillation
from src.training.losses import CTCLoss, AttentionLoss
from src.training.metrics import compute_per, compute_wer
from src.utils.logging import get_logger
from src.data.phoneme_utils import load_librispeech_phoneme_dataset

logger = get_logger(__name__)


class Stage1Trainer:
    """
    Trainer for Stage 1: Audio → Phonemes with Ensemble Distillation.
    """

    def __init__(self, config: dict):
        """
        Initialize Stage 1 trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        logger.info("🚀 Initializing Stage 1 Trainer: Audio → Phonemes")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Mixed Precision: {self.use_amp}")

        # Initialize model
        self._init_model()

        # Initialize datasets
        self._init_datasets()

        # Initialize optimizer and scheduler
        self._init_optimizer()

        # Initialize losses
        self._init_losses()

        # Initialize logging
        self._init_logging()

        logger.info("✅ Stage 1 Trainer initialized")

    def _init_model(self):
        """Initialize student model and distillation module."""
        logger.info("🔧 Initializing model...")

        # Create student model
        self.model = AudioPhonemeASR(
            sample_rate=self.config['audio']['sample_rate'],
            n_fft=self.config['audio']['n_fft'],
            hop_length=self.config['audio']['hop_length'],
            n_mels=self.config['audio']['n_mels'],
            embed_dim=self.config['model']['embed_dim'],
            conformer_layers=self.config['model']['conformer_layers'],
            conformer_heads=self.config['model']['conformer_heads'],
            conv_kernel_size=self.config['model']['conv_kernel_size'],
            vocab_size=self.config['model']['vocab_size'],
            decoder_dim=self.config['model']['decoder_dim'],
            use_specaugment=self.config['audio'].get('use_specaugment', True),
            enable_distillation=False  # Will enable after warmup
        ).to(self.device)

        # Print model info
        model_size = self.model.get_model_size()
        logger.info(f"   Model parameters: {model_size['total_parameters']:,}")
        logger.info(f"   Model size (FP16): {model_size['model_size_mb_fp16']:.2f} MB")

        # Enable distillation if specified
        if self.config['distillation']['enable']:
            logger.info("🎓 Enabling ensemble distillation...")
            self.model.enable_distillation_mode(
                distillation_config=self.config['distillation']
            )
            self.use_distillation = True
        else:
            self.use_distillation = False
            logger.info("⚠️ Distillation disabled (warmup mode)")

    def _init_datasets(self):
        """Initialize LibriSpeech datasets."""
        logger.info("📚 Loading LibriSpeech dataset...")

        # Load training data
        self.train_dataset = load_librispeech_phoneme_dataset(
            root=self.config['data']['librispeech_root'],
            split=self.config['data']['train_split'],
            sample_rate=self.config['audio']['sample_rate'],
            download=self.config['data'].get('download', False)
        )

        # Load validation data
        self.val_dataset = load_librispeech_phoneme_dataset(
            root=self.config['data']['librispeech_root'],
            split=self.config['data']['val_split'],
            sample_rate=self.config['audio']['sample_rate'],
            download=self.config['data'].get('download', False)
        )

        logger.info(f"   Train samples: {len(self.train_dataset)}")
        logger.info(f"   Val samples: {len(self.val_dataset)}")

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True,
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['val_batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )

    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        logger.info("⚙️ Initializing optimizer...")

        # AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=self.config['training']['weight_decay']
        )

        # Cosine annealing with warmup
        warmup_steps = self.config['training']['warmup_steps']
        total_steps = self.config['training']['total_steps']

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

        logger.info(f"   Learning rate: {self.config['training']['learning_rate']}")
        logger.info(f"   Warmup steps: {warmup_steps}")

    def _init_losses(self):
        """Initialize loss functions."""
        self.ctc_loss = nn.CTCLoss(
            blank=0,  # Blank token index
            reduction='mean',
            zero_infinity=True
        )

        # Loss weights
        self.ctc_weight = self.config['training']['ctc_weight']
        self.attention_weight = self.config['training']['attention_weight']
        self.distillation_weight = self.config['training']['distillation_weight']

        logger.info("📉 Loss functions initialized:")
        logger.info(f"   CTC weight: {self.ctc_weight}")
        logger.info(f"   Attention weight: {self.attention_weight}")
        logger.info(f"   Distillation weight: {self.distillation_weight}")

    def _init_logging(self):
        """Initialize logging (WandB)."""
        if self.config['logging']['use_wandb']:
            wandb.init(
                project=self.config['logging']['wandb_project'],
                name=self.config['logging']['experiment_name'],
                config=self.config
            )
            logger.info("📊 WandB logging enabled")
        else:
            logger.info("📊 WandB logging disabled")

        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int):
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number
        """
        self.model.train()

        total_loss = 0.0
        total_ctc_loss = 0.0
        total_attention_loss = 0.0
        total_distillation_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            audio = batch['audio'].to(self.device)
            audio_lengths = batch['audio_lengths'].to(self.device)
            targets = batch['phoneme_targets'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                # Student forward pass
                outputs = self.model(
                    audio=audio,
                    audio_lengths=audio_lengths,
                    targets=targets,
                    target_lengths=target_lengths
                )

                # CTC loss
                ctc_logits = outputs['ctc_logits']
                feature_lengths = outputs['feature_lengths']

                ctc_loss = self.ctc_loss(
                    ctc_logits.transpose(0, 1).log_softmax(2),
                    targets,
                    feature_lengths,
                    target_lengths
                )

                # Attention loss (if available)
                attention_loss = torch.tensor(0.0, device=self.device)
                if 'attention_logits' in outputs and outputs['attention_logits'] is not None:
                    attention_logits = outputs['attention_logits']
                    attention_loss = F.cross_entropy(
                        attention_logits.reshape(-1, attention_logits.size(-1)),
                        targets.reshape(-1),
                        ignore_index=0
                    )

                # Distillation loss
                distillation_loss = torch.tensor(0.0, device=self.device)
                if self.use_distillation and self.model.distillation is not None:
                    distillation_outputs = self.model.compute_distillation_loss(
                        encoder_outputs=outputs['encoder_outputs'],
                        ctc_logits=ctc_logits,
                        audio_waveforms=audio
                    )
                    distillation_loss = distillation_outputs['distillation_loss']

                # Total loss
                loss = (
                    self.ctc_weight * ctc_loss +
                    self.attention_weight * attention_loss +
                    self.distillation_weight * distillation_loss
                )

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            self.scheduler.step()

            # Update statistics
            total_loss += loss.item()
            total_ctc_loss += ctc_loss.item()
            total_attention_loss += attention_loss.item()
            total_distillation_loss += distillation_loss.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ctc': f"{ctc_loss.item():.4f}",
                'distill': f"{distillation_loss.item():.4f}"
            })

            # Log to WandB
            if self.config['logging']['use_wandb'] and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/ctc_loss': ctc_loss.item(),
                    'train/attention_loss': attention_loss.item(),
                    'train/distillation_loss': distillation_loss.item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0]
                })

        # Epoch statistics
        avg_loss = total_loss / len(self.train_loader)
        avg_ctc_loss = total_ctc_loss / len(self.train_loader)
        avg_attention_loss = total_attention_loss / len(self.train_loader)
        avg_distillation_loss = total_distillation_loss / len(self.train_loader)

        logger.info(f"📊 Epoch {epoch} Training Summary:")
        logger.info(f"   Avg Loss: {avg_loss:.4f}")
        logger.info(f"   Avg CTC Loss: {avg_ctc_loss:.4f}")
        logger.info(f"   Avg Attention Loss: {avg_attention_loss:.4f}")
        logger.info(f"   Avg Distillation Loss: {avg_distillation_loss:.4f}")

        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int):
        """
        Validate model.

        Args:
            epoch: Current epoch number
        """
        self.model.eval()

        total_loss = 0.0
        total_per = 0.0
        num_samples = 0

        pbar = tqdm(self.val_loader, desc=f"Validation {epoch}")

        for batch in pbar:
            audio = batch['audio'].to(self.device)
            audio_lengths = batch['audio_lengths'].to(self.device)
            targets = batch['phoneme_targets'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)

            # Forward pass
            outputs = self.model(
                audio=audio,
                audio_lengths=audio_lengths,
                targets=targets,
                target_lengths=target_lengths
            )

            # CTC loss
            ctc_logits = outputs['ctc_logits']
            feature_lengths = outputs['feature_lengths']

            loss = self.ctc_loss(
                ctc_logits.transpose(0, 1).log_softmax(2),
                targets,
                feature_lengths,
                target_lengths
            )

            total_loss += loss.item()

            # Decode and compute PER
            predictions = torch.argmax(ctc_logits, dim=-1)

            for i in range(predictions.shape[0]):
                pred = predictions[i].cpu().numpy()
                target = targets[i].cpu().numpy()

                # Simple PER computation (can be improved)
                per = compute_per(pred, target)
                total_per += per
                num_samples += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'per': f"{total_per/num_samples:.4f}"
            })

        # Validation statistics
        avg_loss = total_loss / len(self.val_loader)
        avg_per = total_per / num_samples

        logger.info(f"📊 Epoch {epoch} Validation Summary:")
        logger.info(f"   Avg Loss: {avg_loss:.4f}")
        logger.info(f"   Avg PER: {avg_per:.4f} ({avg_per*100:.2f}%)")

        # Log to WandB
        if self.config['logging']['use_wandb']:
            wandb.log({
                'val/loss': avg_loss,
                'val/per': avg_per,
                'epoch': epoch
            })

        return avg_loss, avg_per

    def save_checkpoint(self, epoch: int, per: float):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            per: Current PER
        """
        checkpoint_path = self.checkpoint_dir / f"stage1_epoch{epoch}_per{per:.4f}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'per': per,
            'config': self.config
        }, checkpoint_path)

        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

    def train(self):
        """Main training loop."""
        logger.info("🚀 Starting Stage 1 training...")

        best_per = float('inf')

        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{self.config['training']['num_epochs']}")
            logger.info(f"{'='*60}")

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_per = self.validate(epoch)

            # Save checkpoint
            if val_per < best_per:
                best_per = val_per
                self.save_checkpoint(epoch, val_per)
                logger.info(f"🎉 New best PER: {best_per:.4f}")

            # Enable distillation after warmup
            if (not self.use_distillation and
                epoch >= self.config['distillation'].get('warmup_epochs', 5)):
                logger.info("🎓 Enabling ensemble distillation...")
                self.model.enable_distillation_mode(
                    distillation_config=self.config['distillation']
                )
                self.use_distillation = True

        logger.info("\n✅ Stage 1 training completed!")
        logger.info(f"   Best PER: {best_per:.4f} ({best_per*100:.2f}%)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Stage 1: Audio → Phonemes Training")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--resume', type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create trainer and train
    trainer = Stage1Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
