"""
Audio-Only Phoneme ASR Model.

This module implements the complete audio-to-phoneme model architecture
for Stage 1 of the two-stage ASR system with ensemble knowledge distillation.

Architecture:
1. Audio Frontend: Mel spectrogram extraction
2. Conformer Encoder: 256 dim, 12 layers (reduced from original)
3. CTC/Attention Heads: Phoneme prediction (39 phonemes + blank)
4. Ensemble Distillation: Knowledge from 3 teacher models
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from src.utils.logging import get_logger

from .audio_frontend import AudioFrontend
from .conformer import ConformerEncoder
from .hybrid_ctc_attention import HybridCTCAttention
from .ensemble_distillation import EnsembleDistillationModule

logger = get_logger(__name__)


class AudioPhonemeASR(nn.Module):
    """
    Complete audio-to-phoneme ASR model with ensemble distillation.

    This is the student model that learns from multiple teacher ASR models
    while maintaining a compact architecture for efficient inference.

    Key Features:
    - Mel spectrogram audio frontend (replaces 3D CNN)
    - Conformer encoder (256 dim, 12 layers)
    - Hybrid CTC/Attention for phoneme prediction
    - Ensemble distillation from 3 teacher models
    """

    def __init__(
        self,
        # Audio frontend parameters
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,

        # Model architecture parameters
        embed_dim: int = 256,
        conformer_layers: int = 12,
        conformer_heads: int = 4,
        conv_kernel_size: int = 31,
        ffn_expansion_factor: int = 4,
        dropout: float = 0.1,

        # Phoneme vocabulary
        vocab_size: int = 40,  # 39 phonemes + blank

        # CTC/Attention parameters
        decoder_dim: int = 256,

        # Augmentation
        use_specaugment: bool = True,

        # Distillation (optional, can be added later)
        enable_distillation: bool = False,
        distillation_config: Optional[Dict] = None
    ):
        """
        Initialize audio phoneme ASR model.

        Args:
            sample_rate: Audio sample rate
            n_fft: FFT size for Mel spectrogram
            hop_length: Hop length for STFT
            n_mels: Number of Mel filterbanks
            embed_dim: Embedding dimension (reduced to 256)
            conformer_layers: Number of Conformer layers (reduced to 12)
            conformer_heads: Number of attention heads (4 for 256 dim)
            conv_kernel_size: Convolution kernel size
            ffn_expansion_factor: FFN expansion factor
            dropout: Dropout probability
            vocab_size: Phoneme vocabulary size
            decoder_dim: Decoder dimension
            use_specaugment: Whether to use SpecAugment
            enable_distillation: Whether to enable ensemble distillation
            distillation_config: Configuration for distillation module
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.enable_distillation = enable_distillation

        logger.info("🎯 Initializing Audio Phoneme ASR Model...")

        # 1. Audio Frontend (replaces 3D CNN)
        self.audio_frontend = AudioFrontend(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            embed_dim=embed_dim,
            dropout=dropout,
            use_specaugment=use_specaugment
        )

        # 2. Conformer Encoder (256 dim, 12 layers)
        self.conformer = ConformerEncoder(
            input_dim=embed_dim,
            embed_dim=embed_dim,
            num_layers=conformer_layers,
            num_heads=conformer_heads,
            conv_kernel_size=conv_kernel_size,
            ffn_expansion_factor=ffn_expansion_factor,
            dropout=dropout,
            activation="swish"
        )

        # 3. Hybrid CTC/Attention Head
        self.ctc_attention = HybridCTCAttention(
            encoder_dim=embed_dim,
            vocab_size=vocab_size,
            decoder_dim=decoder_dim,
            attention_dim=decoder_dim,
            dropout=dropout
        )

        # 4. Ensemble Distillation (optional)
        if enable_distillation:
            if distillation_config is None:
                # Default configuration with 3 teachers
                distillation_config = {
                    'teachers': [
                        {
                            'model_name': 'openai/whisper-large-v3',
                            'model_type': 'whisper',
                            'weight': 0.4
                        },
                        {
                            'model_name': 'microsoft/wavlm-large',
                            'model_type': 'wavlm',
                            'weight': 0.3
                        },
                        {
                            'model_name': 'facebook/hubert-large-ls960-ft',
                            'model_type': 'hubert',
                            'weight': 0.3
                        }
                    ],
                    'projection_dim': 512,
                    'temperature': 4.0,
                    'aggregation_method': 'weighted_average'
                }

            from .ensemble_distillation import create_ensemble_distillation
            self.distillation = create_ensemble_distillation(
                student_feature_dim=embed_dim,
                config=distillation_config
            )
            logger.info("🎓 Ensemble distillation enabled")
        else:
            self.distillation = None

        # Model info
        self._log_model_info()

    def _log_model_info(self):
        """Log model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"✅ Audio Phoneme ASR Model initialized:")
        logger.info(f"   Embed dimension: {self.embed_dim}")
        logger.info(f"   Conformer layers: 12 (reduced from 16)")
        logger.info(f"   Conformer heads: 4 (optimized for 256 dim)")
        logger.info(f"   Vocabulary size: {self.vocab_size}")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Model size: {total_params * 4 / (1024**2):.2f} MB (FP32)")
        logger.info(f"   Distillation: {'Enabled' if self.enable_distillation else 'Disabled'}")

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            audio: Audio waveform [B, T] or [B, 1, T]
            audio_lengths: Valid audio lengths [B]
            targets: Target phoneme sequences [B, S] (for training)
            target_lengths: Target sequence lengths [B] (for training)
            return_features: Whether to return intermediate features

        Returns:
            Dictionary containing:
            - ctc_logits: CTC output logits [B, T', vocab_size]
            - attention_logits: Attention output logits [B, S, vocab_size] (if targets provided)
            - attention_weights: Attention weights [B, H, S, T']
            - encoder_outputs: Encoder features [B, T', embed_dim]
            - feature_lengths: Feature sequence lengths [B]
        """
        batch_size = audio.shape[0]
        outputs = {}

        # 1. Audio Frontend: Audio -> Mel features -> Embeddings
        embeddings, feature_lengths = self.audio_frontend(audio, audio_lengths)
        # embeddings: [B, T', embed_dim]

        if return_features:
            outputs['frontend_embeddings'] = embeddings

        # 2. Conformer Encoder: Process embeddings
        encoder_outputs = self.conformer(embeddings)
        # encoder_outputs: [B, T', embed_dim]

        outputs['encoder_outputs'] = encoder_outputs
        outputs['feature_lengths'] = feature_lengths

        # 3. CTC and Attention outputs
        ctc_logits, attention_outputs = self.ctc_attention(
            encoder_outputs,
            feature_lengths,
            targets,
            target_lengths
        )

        outputs['ctc_logits'] = ctc_logits

        if attention_outputs is not None:
            outputs['attention_logits'] = attention_outputs.get('logits')
            outputs['attention_weights'] = attention_outputs.get('attention_weights')

        return outputs

    def compute_distillation_loss(
        self,
        encoder_outputs: torch.Tensor,
        ctc_logits: torch.Tensor,
        audio_waveforms: torch.Tensor,
        feature_loss_weight: float = 1.0,
        soft_label_loss_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ensemble distillation loss.

        Args:
            encoder_outputs: Student encoder features [B, T, embed_dim]
            ctc_logits: Student CTC logits [B, T, vocab_size]
            audio_waveforms: Audio input for teachers [B, T_audio]
            feature_loss_weight: Weight for feature matching loss
            soft_label_loss_weight: Weight for soft label loss

        Returns:
            Dictionary with distillation losses
        """
        if not self.enable_distillation or self.distillation is None:
            return {
                'distillation_loss': torch.tensor(0.0, device=encoder_outputs.device),
                'feature_loss': torch.tensor(0.0, device=encoder_outputs.device),
                'soft_label_loss': torch.tensor(0.0, device=encoder_outputs.device)
            }

        # Compute distillation from ensemble teachers
        distillation_outputs = self.distillation(
            student_features=encoder_outputs,
            student_logits=ctc_logits,
            audio_waveforms=audio_waveforms,
            feature_loss_weight=feature_loss_weight,
            soft_label_loss_weight=soft_label_loss_weight
        )

        return distillation_outputs

    def enable_distillation_mode(self, distillation_config: Optional[Dict] = None):
        """Enable distillation mode after initial training."""
        if self.distillation is None:
            from .ensemble_distillation import create_ensemble_distillation
            self.distillation = create_ensemble_distillation(
                student_feature_dim=self.embed_dim,
                config=distillation_config
            )
            self.enable_distillation = True
            logger.info("🎓 Ensemble distillation enabled")

    def decode_greedy(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy CTC decoding for inference.

        Args:
            audio: Audio waveform [B, T]

        Returns:
            Tuple of (decoded_sequences, confidence_scores)
        """
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(audio)
            ctc_logits = outputs['ctc_logits']  # [B, T', vocab_size]

            # Greedy decoding
            predictions = torch.argmax(ctc_logits, dim=-1)  # [B, T']

            # Remove blank tokens and duplicates (CTC decoding)
            decoded = []
            confidences = []

            for i in range(predictions.shape[0]):
                pred_seq = predictions[i].cpu().numpy()
                prob_seq = torch.softmax(ctc_logits[i], dim=-1).cpu().numpy()

                # CTC collapse
                decoded_seq = []
                prev_token = -1

                for t, token in enumerate(pred_seq):
                    if token != 0 and token != prev_token:  # 0 is blank
                        decoded_seq.append(token)
                    prev_token = token

                decoded.append(torch.tensor(decoded_seq))

                # Compute average confidence
                if len(decoded_seq) > 0:
                    avg_conf = prob_seq[range(len(pred_seq)), pred_seq].mean()
                    confidences.append(avg_conf)
                else:
                    confidences.append(0.0)

            return decoded, torch.tensor(confidences)

    def get_model_size(self) -> Dict[str, any]:
        """Get model size information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Component sizes
        frontend_params = sum(p.numel() for p in self.audio_frontend.parameters())
        conformer_params = sum(p.numel() for p in self.conformer.parameters())
        ctc_params = sum(p.numel() for p in self.ctc_attention.parameters())

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb_fp32': total_params * 4 / (1024**2),
            'model_size_mb_fp16': total_params * 2 / (1024**2),
            'components': {
                'audio_frontend': frontend_params,
                'conformer': conformer_params,
                'ctc_attention': ctc_params
            }
        }


def create_audio_phoneme_model(config) -> AudioPhonemeASR:
    """
    Factory function to create AudioPhonemeASR from configuration.

    Args:
        config: Configuration object

    Returns:
        Initialized AudioPhonemeASR instance
    """
    return AudioPhonemeASR(
        # Audio parameters
        sample_rate=getattr(config, 'sample_rate', 16000),
        n_fft=getattr(config, 'n_fft', 400),
        hop_length=getattr(config, 'hop_length', 160),
        n_mels=getattr(config, 'n_mels', 80),

        # Model parameters
        embed_dim=getattr(config, 'embed_dim', 256),
        conformer_layers=getattr(config, 'conformer_layers', 12),
        conformer_heads=getattr(config, 'conformer_heads', 4),
        conv_kernel_size=getattr(config, 'conv_kernel_size', 31),
        ffn_expansion_factor=getattr(config, 'ffn_expansion_factor', 4),
        dropout=getattr(config, 'dropout', 0.1),

        # Vocabulary
        vocab_size=getattr(config, 'vocab_size', 40),
        decoder_dim=getattr(config, 'decoder_dim', 256),

        # Augmentation
        use_specaugment=getattr(config, 'use_specaugment', True),

        # Distillation
        enable_distillation=getattr(config, 'enable_distillation', False),
        distillation_config=getattr(config, 'distillation_config', None)
    )


# Unit tests
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🧪 Testing Audio Phoneme ASR Model...")

    # Create model
    model = AudioPhonemeASR(
        embed_dim=256,
        conformer_layers=12,
        conformer_heads=4,
        vocab_size=40,
        enable_distillation=False  # Disable for quick testing
    ).to(device)

    # Generate dummy data
    batch_size = 2
    audio_len = 16000 * 3  # 3 seconds
    seq_len = 50
    vocab_size = 40

    dummy_audio = torch.randn(batch_size, audio_len).to(device)
    audio_lengths = torch.tensor([audio_len, audio_len // 2]).to(device)
    targets = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)
    target_lengths = torch.tensor([seq_len, seq_len // 2]).to(device)

    print("\n🔄 Running forward pass...")
    outputs = model(
        audio=dummy_audio,
        audio_lengths=audio_lengths,
        targets=targets,
        target_lengths=target_lengths
    )

    print("\n✅ Model Test Results:")
    print(f"   Audio input: {dummy_audio.shape}")
    print(f"   CTC logits: {outputs['ctc_logits'].shape}")
    print(f"   Encoder outputs: {outputs['encoder_outputs'].shape}")
    if 'attention_logits' in outputs:
        print(f"   Attention logits: {outputs['attention_logits'].shape}")

    # Test greedy decoding
    print("\n🔄 Testing greedy decoding...")
    decoded, confidences = model.decode_greedy(dummy_audio)
    print(f"✅ Decoded {len(decoded)} sequences")
    print(f"   Confidences: {confidences}")

    # Print model size
    print("\n📊 Model Size Information:")
    size_info = model.get_model_size()
    print(f"   Total parameters: {size_info['total_parameters']:,}")
    print(f"   Trainable parameters: {size_info['trainable_parameters']:,}")
    print(f"   Model size (FP32): {size_info['model_size_mb_fp32']:.2f} MB")
    print(f"   Model size (FP16): {size_info['model_size_mb_fp16']:.2f} MB")

    print("\n🎉 All tests passed!")
