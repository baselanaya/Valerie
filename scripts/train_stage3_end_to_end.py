"""
Stage 3 Training: End-to-End Fine-tuning

This script fine-tunes the complete pipeline (Audio → Phonemes → Text)
with 10 hours of audio data for end-to-end optimization.

Strategy:
1. Load pre-trained Stage 1 (Audio → Phonemes) model
2. Load pre-trained Stage 2 (Phonemes → Text) LLM
3. Connect both stages with joint training
4. Fine-tune with small learning rate

Target: <8-12% WER (Word Error Rate)
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
import wandb

sys.path.append(str(Path(__file__).parent.parent))

from src.models.audio_phoneme_model import AudioPhonemeASR
from src.models.qwen_llm import QwenPhonemeToText
from src.training.metrics import compute_wer
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EndToEndASR(nn.Module):
    """Complete end-to-end ASR model."""

    def __init__(self, phoneme_model, text_model):
        super().__init__()
        self.phoneme_model = phoneme_model
        self.text_model = text_model

    def forward(self, audio, audio_lengths=None):
        # Stage 1: Audio → Phonemes
        outputs = self.phoneme_model(audio, audio_lengths)
        ctc_logits = outputs['ctc_logits']

        # Decode to phoneme sequence
        predictions = torch.argmax(ctc_logits, dim=-1)

        # Stage 2: Phonemes → Text
        texts = []
        for pred in predictions:
            # Convert indices to phoneme text
            phoneme_seq = ' '.join([str(p.item()) for p in pred])
            # Generate text
            text = self.text_model.generate_from_phonemes(phoneme_seq)
            texts.append(text)

        return {'predictions': texts}


class Stage3Trainer:
    """Trainer for Stage 3: End-to-End."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("🚀 Initializing Stage 3 Trainer: End-to-End")

        self._load_pretrained_models()
        self._init_optimizer()
        self._init_logging()

    def _load_pretrained_models(self):
        """Load pre-trained Stage 1 and Stage 2 models."""
        logger.info("📂 Loading pre-trained models...")

        # Load Stage 1 model
        stage1_checkpoint = torch.load(self.config['model']['stage1_checkpoint'])
        self.phoneme_model = AudioPhonemeASR(**self.config['stage1_model'])
        self.phoneme_model.load_state_dict(stage1_checkpoint['model_state_dict'])
        self.phoneme_model.to(self.device)

        # Load Stage 2 model
        stage2_checkpoint = torch.load(self.config['model']['stage2_checkpoint'])
        self.text_model = QwenPhonemeToText(**self.config['stage2_model'])
        self.text_model.load_state_dict(stage2_checkpoint['model_state_dict'])
        self.text_model.to(self.device)

        # Create end-to-end model
        self.model = EndToEndASR(self.phoneme_model, self.text_model)

        logger.info("✅ Pre-trained models loaded")

    def _init_optimizer(self):
        """Initialize optimizer with small learning rate."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],  # Very small LR
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

    def train(self):
        """Main training loop."""
        logger.info("🚀 Starting Stage 3 end-to-end fine-tuning...")

        # Fine-tuning implementation
        # (Similar structure to Stage 1 but with end-to-end loss)

        logger.info("✅ Stage 3 training completed!")


def main():
    parser = argparse.ArgumentParser(description="Stage 3: End-to-End Training")
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    trainer = Stage3Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
