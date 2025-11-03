"""
Audio Frontend for Phoneme ASR.

This module implements audio feature extraction for the phoneme-based ASR system,
replacing the 3D CNN video processing with audio Mel spectrogram extraction.

Features:
- Mel spectrogram extraction
- Log-Mel filterbanks
- Feature normalization
- Data augmentation (SpecAugment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from typing import Optional, Tuple
import math
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MelSpectrogramExtractor(nn.Module):
    """
    Extract Mel spectrogram features from raw audio waveforms.

    Implements standard Mel-scale filterbank features used in modern ASR systems.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        window_fn: str = "hann",
        normalize: bool = True,
        log_scale: bool = True,
        eps: float = 1e-10
    ):
        """
        Initialize Mel spectrogram extractor.

        Args:
            sample_rate: Audio sample rate in Hz
            n_fft: FFT size
            win_length: Window length (default: n_fft)
            hop_length: Hop length for STFT
            n_mels: Number of Mel filterbanks
            f_min: Minimum frequency
            f_max: Maximum frequency (default: sample_rate / 2)
            window_fn: Window function ("hann", "hamming", "blackman")
            normalize: Whether to normalize features
            log_scale: Whether to apply log scaling
            eps: Small value for numerical stability
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2
        self.normalize = normalize
        self.log_scale = log_scale
        self.eps = eps

        # Create Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=self.f_max,
            window_fn=getattr(torch, f"{window_fn}_window"),
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect"
        )

        # Normalization statistics (learned from data)
        self.register_buffer('mean', torch.zeros(n_mels))
        self.register_buffer('std', torch.ones(n_mels))
        self.register_buffer('count', torch.tensor(0))

        logger.info(f"🎵 MelSpectrogramExtractor initialized:")
        logger.info(f"   Sample rate: {sample_rate} Hz")
        logger.info(f"   N_FFT: {n_fft}")
        logger.info(f"   Hop length: {hop_length}")
        logger.info(f"   N_mels: {n_mels}")
        logger.info(f"   Frequency range: {f_min}-{self.f_max} Hz")

    def update_statistics(self, features: torch.Tensor):
        """
        Update running mean and std for normalization.

        Args:
            features: Mel features [B, n_mels, T]
        """
        if not self.training:
            return

        # Compute mean and std across batch and time
        batch_mean = features.mean(dim=(0, 2))  # [n_mels]
        batch_std = features.std(dim=(0, 2))    # [n_mels]

        # Update running statistics
        alpha = 0.01  # Momentum for running average
        self.mean = (1 - alpha) * self.mean + alpha * batch_mean
        self.std = (1 - alpha) * self.std + alpha * batch_std
        self.count += 1

    def forward(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract Mel spectrogram features from audio.

        Args:
            waveform: Audio waveform [B, T] or [B, 1, T]
            lengths: Original lengths of sequences [B] (before padding)

        Returns:
            Tuple of (mel_features, feature_lengths)
            - mel_features: [B, n_mels, T'] where T' = (T - n_fft) / hop_length + 1
            - feature_lengths: [B] (if lengths provided)
        """
        # Ensure correct shape [B, T]
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Extract Mel spectrogram
        mel_spec = self.mel_transform(waveform)  # [B, n_mels, T']

        # Apply log scaling
        if self.log_scale:
            mel_spec = torch.log(mel_spec + self.eps)

        # Update normalization statistics
        if self.training and self.normalize:
            self.update_statistics(mel_spec)

        # Apply normalization
        if self.normalize:
            mel_spec = (mel_spec - self.mean.view(1, -1, 1)) / (self.std.view(1, -1, 1) + self.eps)

        # Compute feature lengths if input lengths provided
        feature_lengths = None
        if lengths is not None:
            feature_lengths = (
                (lengths - self.n_fft) // self.hop_length + 1
            ).long()

        return mel_spec, feature_lengths


class SpecAugment(nn.Module):
    """
    SpecAugment data augmentation for audio features.

    Implements frequency and time masking as described in:
    "SpecAugment: A Simple Data Augmentation Method for ASR" (Park et al., 2019)
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        mask_value: float = 0.0
    ):
        """
        Initialize SpecAugment.

        Args:
            freq_mask_param: Maximum frequency mask size
            time_mask_param: Maximum time mask size
            n_freq_masks: Number of frequency masks to apply
            n_time_masks: Number of time masks to apply
            mask_value: Value to use for masked regions
        """
        super().__init__()

        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.mask_value = mask_value

        # Create masking transforms
        self.freq_mask = T.FrequencyMasking(
            freq_mask_param=freq_mask_param,
            iid_masks=True
        )

        self.time_mask = T.TimeMasking(
            time_mask_param=time_mask_param,
            iid_masks=True
        )

        logger.debug(f"🎭 SpecAugment initialized:")
        logger.debug(f"   Freq masks: {n_freq_masks} x {freq_mask_param}")
        logger.debug(f"   Time masks: {n_time_masks} x {time_mask_param}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to features.

        Args:
            features: Input features [B, n_mels, T]

        Returns:
            Augmented features [B, n_mels, T]
        """
        if not self.training:
            return features

        # Apply frequency masks
        for _ in range(self.n_freq_masks):
            features = self.freq_mask(features)

        # Apply time masks
        for _ in range(self.n_time_masks):
            features = self.time_mask(features)

        return features


class AudioFrontend(nn.Module):
    """
    Complete audio frontend for phoneme ASR.

    Combines Mel spectrogram extraction, normalization, and augmentation
    into a single module that replaces the 3D CNN video processing.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        embed_dim: int = 256,
        dropout: float = 0.1,
        use_specaugment: bool = True,
        spec_augment_config: Optional[dict] = None
    ):
        """
        Initialize audio frontend.

        Args:
            sample_rate: Audio sample rate
            n_fft: FFT size
            hop_length: Hop length for STFT
            n_mels: Number of Mel filterbanks
            embed_dim: Output embedding dimension
            dropout: Dropout probability
            use_specaugment: Whether to use SpecAugment
            spec_augment_config: SpecAugment configuration
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.embed_dim = embed_dim

        # Mel spectrogram extractor
        self.mel_extractor = MelSpectrogramExtractor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalize=True,
            log_scale=True
        )

        # SpecAugment (optional)
        if use_specaugment:
            if spec_augment_config is None:
                spec_augment_config = {
                    'freq_mask_param': 27,
                    'time_mask_param': 100,
                    'n_freq_masks': 2,
                    'n_time_masks': 2
                }
            self.specaugment = SpecAugment(**spec_augment_config)
        else:
            self.specaugment = None

        # Projection layer to embed_dim
        # Input: [B, n_mels, T] -> [B, T, embed_dim]
        self.projection = nn.Sequential(
            nn.Conv1d(n_mels, embed_dim, kernel_size=3, padding=1),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=5000,
            dropout=dropout
        )

        logger.info(f"✅ AudioFrontend initialized:")
        logger.info(f"   Sample rate: {sample_rate} Hz")
        logger.info(f"   N_mels: {n_mels}")
        logger.info(f"   Embed dim: {embed_dim}")
        logger.info(f"   SpecAugment: {use_specaugment}")

    def forward(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process audio waveform to embeddings.

        Args:
            waveform: Audio waveform [B, T] or [B, 1, T]
            lengths: Original lengths [B]

        Returns:
            Tuple of (embeddings, feature_lengths)
            - embeddings: [B, T', embed_dim]
            - feature_lengths: [B]
        """
        # Extract Mel spectrogram
        mel_features, feature_lengths = self.mel_extractor(waveform, lengths)
        # mel_features: [B, n_mels, T']

        # Apply SpecAugment during training
        if self.specaugment is not None and self.training:
            mel_features = self.specaugment(mel_features)

        # Project to embed_dim
        projected = self.projection(mel_features)  # [B, embed_dim, T']

        # Transpose to [B, T', embed_dim]
        projected = projected.transpose(1, 2)

        # Add positional encoding
        embedded = self.positional_encoding(projected)

        return embedded, feature_lengths


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer models.
    """

    def __init__(
        self,
        embed_dim: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        """
        Initialize positional encoding.

        Args:
            embed_dim: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )

        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor [B, T, embed_dim]

        Returns:
            Output with positional encoding [B, T, embed_dim]
        """
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)


def create_audio_frontend(config) -> AudioFrontend:
    """
    Factory function to create AudioFrontend from configuration.

    Args:
        config: Configuration object with audio parameters

    Returns:
        Initialized AudioFrontend instance
    """
    return AudioFrontend(
        sample_rate=getattr(config, 'sample_rate', 16000),
        n_fft=getattr(config, 'n_fft', 400),
        hop_length=getattr(config, 'hop_length', 160),
        n_mels=getattr(config, 'n_mels', 80),
        embed_dim=getattr(config, 'embed_dim', 256),
        dropout=getattr(config, 'dropout', 0.1),
        use_specaugment=getattr(config, 'use_specaugment', True),
        spec_augment_config=getattr(config, 'spec_augment_config', None)
    )


# Unit tests
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🧪 Testing Audio Frontend...")

    # Test MelSpectrogramExtractor
    print("\n🔬 Testing MelSpectrogramExtractor...")
    mel_extractor = MelSpectrogramExtractor(
        sample_rate=16000,
        n_mels=80,
        n_fft=400,
        hop_length=160
    ).to(device)

    # Generate dummy audio (16kHz, 3 seconds)
    batch_size = 2
    audio_len = 16000 * 3
    dummy_audio = torch.randn(batch_size, audio_len).to(device)

    mel_features, _ = mel_extractor(dummy_audio)
    print(f"✅ MelExtractor: {dummy_audio.shape} -> {mel_features.shape}")

    # Test SpecAugment
    print("\n🔬 Testing SpecAugment...")
    specaugment = SpecAugment(
        freq_mask_param=27,
        time_mask_param=100,
        n_freq_masks=2,
        n_time_masks=2
    ).to(device)

    augmented = specaugment(mel_features)
    print(f"✅ SpecAugment: {mel_features.shape} -> {augmented.shape}")

    # Test complete AudioFrontend
    print("\n🔬 Testing AudioFrontend...")
    audio_frontend = AudioFrontend(
        sample_rate=16000,
        n_mels=80,
        embed_dim=256,
        use_specaugment=True
    ).to(device)

    embeddings, feature_lengths = audio_frontend(dummy_audio)
    print(f"✅ AudioFrontend: {dummy_audio.shape} -> {embeddings.shape}")

    # Test with lengths
    lengths = torch.tensor([audio_len, audio_len // 2]).to(device)
    embeddings, feature_lengths = audio_frontend(dummy_audio, lengths)
    print(f"✅ With lengths: {dummy_audio.shape} -> {embeddings.shape}")
    print(f"   Feature lengths: {feature_lengths}")

    print("\n🎉 All audio frontend tests passed!")
