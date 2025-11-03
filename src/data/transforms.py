"""
Data augmentation and preprocessing transforms for Valerie Audio-Only ASR.

Implements audio augmentations for phoneme-based ASR that improve
model robustness while preserving phonetic content.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchaudio.transforms as AT
import numpy as np
import cv2
import random
from typing import Tuple, Optional, Dict, Any, List, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import librosa
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AudioTransforms:
    """Audio-specific transforms for speech processing."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        speed_perturb_prob: float = 0.3,
        speed_range: Tuple[float, float] = (0.9, 1.1),
        volume_perturb_prob: float = 0.5,
        volume_range: Tuple[float, float] = (0.8, 1.2),
        noise_prob: float = 0.3,
        noise_snr_range: Tuple[float, float] = (10, 30),
        spec_augment_prob: float = 0.5,
        freq_mask_param: int = 15,
        time_mask_param: int = 35
    ):
        """
        Initialize audio transforms.
        
        Args:
            sample_rate: Audio sample rate
            speed_perturb_prob: Probability of speed perturbation
            speed_range: Speed perturbation range
            volume_perturb_prob: Probability of volume perturbation
            volume_range: Volume perturbation range
            noise_prob: Probability of adding noise
            noise_snr_range: SNR range for noise addition
            spec_augment_prob: Probability of SpecAugment
            freq_mask_param: Frequency masking parameter
            time_mask_param: Time masking parameter
        """
        self.sample_rate = sample_rate
        self.speed_perturb_prob = speed_perturb_prob
        self.speed_range = speed_range
        self.volume_perturb_prob = volume_perturb_prob
        self.volume_range = volume_range
        self.noise_prob = noise_prob
        self.noise_snr_range = noise_snr_range
        self.spec_augment_prob = spec_augment_prob
        
        # SpecAugment transforms
        self.freq_masking = AT.FrequencyMasking(freq_mask_param)
        self.time_masking = AT.TimeMasking(time_mask_param)
        
        # Mel spectrogram transform
        self.mel_transform = AT.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=80
        )
        
        logger.info(f"✅ AudioTransforms initialized:")
        logger.info(f"   Sample rate: {sample_rate} Hz")
        logger.info(f"   Speed perturbation: {speed_range}")
        logger.info(f"   Volume perturbation: {volume_range}")
        logger.info(f"   Noise SNR range: {noise_snr_range} dB")
    
    def __call__(self, audio_waveform: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Apply audio transforms.
        
        Args:
            audio_waveform: Audio tensor [T] or [1, T]
            training: Whether in training mode
            
        Returns:
            Transformed audio tensor
        """
        if not training:
            return self.normalize_audio(audio_waveform)
        
        # Ensure 1D tensor
        if audio_waveform.dim() == 2:
            audio_waveform = audio_waveform.squeeze(0)
        
        # Apply speed perturbation
        if random.random() < self.speed_perturb_prob:
            audio_waveform = self.speed_perturb(audio_waveform)
        
        # Apply volume perturbation
        if random.random() < self.volume_perturb_prob:
            audio_waveform = self.volume_perturb(audio_waveform)
        
        # Add noise
        if random.random() < self.noise_prob:
            audio_waveform = self.add_noise(audio_waveform)
        
        # Normalize
        audio_waveform = self.normalize_audio(audio_waveform)
        
        return audio_waveform
    
    def speed_perturb(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply speed perturbation to audio."""
        speed_factor = random.uniform(*self.speed_range)
        
        # Convert to numpy for librosa
        audio_np = audio.numpy()
        
        # Apply speed change
        audio_stretched = librosa.effects.time_stretch(audio_np, rate=speed_factor)
        
        return torch.from_numpy(audio_stretched)
    
    def volume_perturb(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply volume perturbation to audio."""
        volume_factor = random.uniform(*self.volume_range)
        return audio * volume_factor
    
    def add_noise(self, audio: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to audio."""
        # Target SNR in dB
        target_snr_db = random.uniform(*self.noise_snr_range)
        
        # Calculate signal power
        signal_power = torch.mean(audio ** 2)
        
        # Calculate noise power for target SNR
        snr_linear = 10 ** (target_snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate and add noise
        noise = torch.randn_like(audio) * torch.sqrt(noise_power)
        
        return audio + noise
    
    def normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1] range."""
        max_val = torch.max(torch.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def spec_augment(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to mel spectrogram."""
        if random.random() < self.spec_augment_prob:
            mel_spec = self.freq_masking(mel_spec)
            mel_spec = self.time_masking(mel_spec)
        return mel_spec


class MixupAugmentation:
    """Mixup augmentation for audio-visual data."""
    
    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        """
        Initialize mixup augmentation.
        
        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying mixup
        """
        self.alpha = alpha
        self.prob = prob
        
        logger.info(f"✅ MixupAugmentation initialized (alpha={alpha}, prob={prob})")
    
    def __call__(
        self, 
        batch_data: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Apply mixup to batch data.
        
        Args:
            batch_data: Batch dictionary with video and audio data
            
        Returns:
            Mixed batch data and mixing coefficients
        """
        if random.random() > self.prob:
            # No mixup, return original batch
            batch_size = batch_data['video_frames'].shape[0]
            return batch_data, torch.ones(batch_size)
        
        batch_size = batch_data['video_frames'].shape[0]
        
        # Sample mixing coefficients
        lam = np.random.beta(self.alpha, self.alpha, batch_size)
        lam = torch.from_numpy(lam).float()
        
        # Create random permutation for mixing
        indices = torch.randperm(batch_size)
        
        # Mix video frames
        if 'video_frames' in batch_data:
            video_frames = batch_data['video_frames']
            mixed_video = lam.view(-1, 1, 1, 1, 1) * video_frames + \
                         (1 - lam).view(-1, 1, 1, 1, 1) * video_frames[indices]
            batch_data['video_frames'] = mixed_video
        
        # Mix audio waveforms
        if 'audio_waveforms' in batch_data:
            audio_waveforms = batch_data['audio_waveforms']
            mixed_audio = lam.view(-1, 1) * audio_waveforms + \
                         (1 - lam).view(-1, 1) * audio_waveforms[indices]
            batch_data['audio_waveforms'] = mixed_audio
        
        return batch_data, lam


class DataAugmentationPipeline:
    """
    Audio-only data augmentation pipeline for Valerie.

    Note: Video transforms removed. For audio augmentation, use AudioFrontend
    with SpecAugment directly, which is more efficient.
    """

    def __init__(self, config, training: bool = True):
        """
        Initialize augmentation pipeline from configuration.

        Args:
            config: Configuration object
            training: Whether pipeline is for training
        """
        self.training = training

        # Audio transforms (video transforms removed)
        self.audio_transforms = AudioTransforms(
            sample_rate=config.data.audio_sample_rate
        )

        # Mixup augmentation
        if training:
            self.mixup = MixupAugmentation(
                alpha=config.data.mixup_alpha,
                prob=0.5
            )
        else:
            self.mixup = None

        logger.info(f"✅ DataAugmentationPipeline initialized (audio-only, training={training})")
    
    def __call__(self, data) -> Union[Dict[str, Any], 'DataSample']:
        """Apply audio-only augmentation pipeline to batch or single sample."""

        # Handle DataSample objects (audio-only)
        if hasattr(data, 'audio_waveform'):
            from src.data.dataset import DataSample

            # Apply audio transforms
            if data.audio_waveform is not None:
                transformed_audio = self.audio_transforms(
                    data.audio_waveform,
                    training=self.training
                )
            else:
                transformed_audio = data.audio_waveform

            # Create new DataSample with transformed audio (video fields removed)
            return DataSample(
                video_frames=None,  # Video removed
                audio_waveform=transformed_audio,
                video_path=None,
                audio_path=data.audio_path if hasattr(data, 'audio_path') else None,
                transcription=data.transcription if hasattr(data, 'transcription') else None,
                phoneme_sequence=data.phoneme_sequence if hasattr(data, 'phoneme_sequence') else None,
                phoneme_alignment=data.phoneme_alignment if hasattr(data, 'phoneme_alignment') else None,
                speaker_id=data.speaker_id if hasattr(data, 'speaker_id') else None,
                language=data.language if hasattr(data, 'language') else None,
                duration=data.duration if hasattr(data, 'duration') else None,
                face_confidence=None,  # Video removed
                transcription_confidence=data.transcription_confidence if hasattr(data, 'transcription_confidence') else None
            )

        # Handle dictionary/batch data (audio-only)
        elif isinstance(data, dict):
            batch_data = data.copy()

            # Apply audio transforms (video processing removed)
            if 'audio_waveforms' in batch_data:
                transformed_audio = []
                for audio in batch_data['audio_waveforms']:
                    transformed_audio.append(
                        self.audio_transforms(audio, training=self.training)
                    )
                batch_data['audio_waveforms'] = torch.stack(transformed_audio)

            # Apply mixup (only during training)
            if self.training and self.mixup is not None:
                batch_data, mixup_lambdas = self.mixup(batch_data)
                batch_data['mixup_lambdas'] = mixup_lambdas

            return batch_data

        else:
            raise TypeError(f"Expected DataSample or dict, got {type(data)}")


# Factory function
def create_augmentation_pipeline(config, training: bool = True) -> DataAugmentationPipeline:
    """Create augmentation pipeline from configuration."""
    return DataAugmentationPipeline(config, training=training)


if __name__ == "__main__":
    # Test audio-only transforms
    from src.utils.config import Config

    print("🧪 Testing audio-only transforms...")

    # Test audio transforms
    print("\n1️⃣ Testing AudioTransforms...")
    audio_transforms = AudioTransforms()
    dummy_audio = torch.randn(16000)  # 1 second at 16kHz

    transformed_audio = audio_transforms(dummy_audio, training=True)
    print(f"   Audio: {dummy_audio.shape} -> {transformed_audio.shape}")

    # Test augmentation pipeline (audio-only)
    print("\n2️⃣ Testing DataAugmentationPipeline (audio-only)...")

    # Create minimal config for testing
    class SimpleConfig:
        class data:
            audio_sample_rate = 16000
            mixup_alpha = 0.2

    config = SimpleConfig()

    pipeline = create_augmentation_pipeline(config, training=True)

    batch_data = {
        'audio_waveforms': torch.randn(2, 16000)
    }

    augmented = pipeline(batch_data)
    print(f"   Pipeline applied successfully!")
    print(f"   Audio shape: {augmented['audio_waveforms'].shape}")

    print("\n✅ Audio-only transform tests completed!")
