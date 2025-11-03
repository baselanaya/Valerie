"""
Stage 2 Training: Phonemes → Text with LLM

This script fine-tunes the Qwen 0.6B LLM to reconstruct clean text from
phoneme sequences with synthetic errors.

Dataset: WikiText + BookCorpus (unlimited text data)
Strategy:
1. Convert text to phonemes using G2P
2. Add synthetic errors to phoneme sequences
3. Train LLM to reconstruct original text
4. Use LoRA for efficient fine-tuning

Target: >95% reconstruction accuracy
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
import wandb
from datasets import load_dataset

sys.path.append(str(Path(__file__).parent.parent))

from src.models.qwen_llm import QwenPhonemeToText
from src.data.text_to_phoneme import create_text_to_phoneme_pipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PhonemeTextDataset(Dataset):
    """Dataset for phoneme-to-text training."""

    def __init__(
        self,
        texts: list,
        phoneme_pipeline,
        max_length: int = 512,
        inject_errors: bool = True
    ):
        self.texts = texts
        self.phoneme_pipeline = phoneme_pipeline
        self.max_length = max_length
        self.inject_errors = inject_errors

        logger.info(f"📚 PhonemeTextDataset created with {len(texts)} samples")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Convert to phonemes
        clean_phonemes, noisy_phonemes, original_text = self.phoneme_pipeline.process_text(
            text, inject_errors=self.inject_errors
        )

        return {
            'text': original_text,
            'clean_phonemes': ' '.join(clean_phonemes),
            'noisy_phonemes': ' '.join(noisy_phonemes)
        }


class Stage2Trainer:
    """Trainer for Stage 2: Phonemes → Text."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        logger.info("🚀 Initializing Stage 2 Trainer: Phonemes → Text")

        self._init_model()
        self._init_datasets()
        self._init_optimizer()
        self._init_logging()

    def _init_model(self):
        """Initialize Qwen LLM with LoRA."""
        logger.info("🔧 Initializing Qwen LLM...")

        self.model = QwenPhonemeToText(
            model_name=self.config['model']['llm_model_name'],
            lora_rank=self.config['model']['lora_rank'],
            lora_alpha=self.config['model']['lora_alpha'],
            max_new_tokens=self.config['model']['max_new_tokens']
        ).to(self.device)

        logger.info("✅ Qwen LLM initialized with LoRA")

    def _init_datasets(self):
        """Initialize WikiText/BookCorpus datasets."""
        logger.info("📚 Loading text datasets...")

        # Load WikiText
        wikitext = load_dataset("wikitext", "wikitext-103-v1")
        wikitext_texts = [
            item['text'] for item in wikitext['train']
            if len(item['text'].strip()) > 20
        ][:self.config['data']['max_samples']]

        # Create phoneme pipeline
        self.phoneme_pipeline = create_text_to_phoneme_pipeline(
            backend="g2p_en",
            inject_errors=True,
            error_config=self.config['data'].get('error_config')
        )

        # Create datasets
        self.train_dataset = PhonemeTextDataset(
            texts=wikitext_texts[:-1000],
            phoneme_pipeline=self.phoneme_pipeline,
            inject_errors=True
        )

        self.val_dataset = PhonemeTextDataset(
            texts=wikitext_texts[-1000:],
            phoneme_pipeline=self.phoneme_pipeline,
            inject_errors=True
        )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers']
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['val_batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers']
        )

        logger.info(f"   Train samples: {len(self.train_dataset)}")
        logger.info(f"   Val samples: {len(self.val_dataset)}")

    def _init_optimizer(self):
        """Initialize optimizer."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

    def _init_logging(self):
        """Initialize logging."""
        if self.config['logging']['use_wandb']:
            wandb.init(
                project=self.config['logging']['wandb_project'],
                name=self.config['logging']['experiment_name'],
                config=self.config
            )

        self.checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            # Prepare inputs
            noisy_phonemes = batch['noisy_phonemes']
            target_texts = batch['text']

            # Create prompts
            prompts = [
                f"Reconstruct the text from these phonemes: {phonemes}"
                for phonemes in noisy_phonemes
            ]

            # Tokenize
            inputs = self.model.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            labels = self.model.tokenizer(
                target_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Forward pass
            with autocast(enabled=self.use_amp):
                outputs = self.model.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=labels['input_ids']
                )
                loss = outputs.loss

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    def train(self):
        """Main training loop."""
        logger.info("🚀 Starting Stage 2 training...")

        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            train_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch}: Loss = {train_loss:.4f}")

            # Save checkpoint
            if epoch % 5 == 0:
                checkpoint_path = self.checkpoint_dir / f"stage2_epoch{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, checkpoint_path)
                logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

        logger.info("✅ Stage 2 training completed!")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Phonemes → Text Training")
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    trainer = Stage2Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
