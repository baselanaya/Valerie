"""
Data augmentation and preprocessing transforms for Valerie Visual ASR.

Implements video and audio augmentations that preserve lip reading content
while improving model robustness.
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


class VideoTransforms:
    """Video-specific transforms for lip reading."""
    
    def __init__(
        self,
        target_height: int = 224,
        target_width: int = 224,
        horizontal_flip_prob: float = 0.5,
        brightness_range: float = 0.2,
        contrast_range: float = 0.2,
        noise_std: float = 0.01,
        temporal_crop_prob: float = 0.3,
        normalize: bool = True
    ):
        """
        Initialize video transforms.
        
        Args:
            target_height: Target frame height
            target_width: Target frame width
            horizontal_flip_prob: Probability of horizontal flipping
            brightness_range: Brightness adjustment range
            contrast_range: Contrast adjustment range
            noise_std: Standard deviation for Gaussian noise
            temporal_crop_prob: Probability of temporal cropping
            normalize: Whether to normalize to [0, 1]
        """
        self.target_height = target_height
        self.target_width = target_width
        self.horizontal_flip_prob = horizontal_flip_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.temporal_crop_prob = temporal_crop_prob
        self.normalize = normalize
        
        # Create spatial augmentation pipeline
        self.spatial_transforms = A.Compose([
            A.Resize(target_height, target_width),
            A.HorizontalFlip(p=horizontal_flip_prob),
            A.RandomBrightnessContrast(
                brightness_limit=brightness_range,
                contrast_limit=contrast_range,
                p=0.7
            ),
            A.GaussNoise(var_limit=(0.0, noise_std**2), p=0.3),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) if normalize else A.NoOp(),
        ])
        
        logger.info(f"✅ VideoTransforms initialized:")
        logger.info(f"   Target size: {target_height}x{target_width}")
        logger.info(f"   Horizontal flip: {horizontal_flip_prob}")
        logger.info(f"   Brightness/Contrast: ±{brightness_range:.2f}")
        logger.info(f"   Noise std: {noise_std}")
    
    def __call__(self, video_frames: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Apply video transforms.
        
        Args:
            video_frames: Video tensor [T, H, W, C] or [T, C, H, W]
            training: Whether in training mode
            
        Returns:
            Transformed video tensor
        """
        if not training:
            # Only resize during inference
            return self.resize_only(video_frames)
        
        # Ensure correct format [T, H, W, C]
        if video_frames.dim() == 4 and video_frames.shape[1] == 3:
            video_frames = video_frames.permute(0, 2, 3, 1)  # [T, C, H, W] -> [T, H, W, C]
        
        # Apply temporal cropping
        if random.random() < self.temporal_crop_prob:
            video_frames = self.temporal_crop(video_frames)
        
        # Apply spatial transforms frame by frame
        transformed_frames = []
        for frame in video_frames:
            # Convert to numpy for albumentations
            frame_np = frame.numpy() if isinstance(frame, torch.Tensor) else frame
            
            # Ensure uint8 format for albumentations
            if frame_np.dtype != np.uint8:
                frame_np = (frame_np * 255).astype(np.uint8)
            
            # Apply transforms
            transformed = self.spatial_transforms(image=frame_np)['image']
            
            # Convert back to tensor
            if isinstance(transformed, np.ndarray):
                transformed = torch.from_numpy(transformed)
            
            transformed_frames.append(transformed)
        
        # Stack frames
        result = torch.stack(transformed_frames)
        
        # Ensure correct output format [T, H, W, C]
        if result.dim() == 4 and result.shape[-1] != 3:
            result = result.permute(0, 2, 3, 1)
        
        return result
    
    def resize_only(self, video_frames: torch.Tensor) -> torch.Tensor:
        """Apply only resizing (for inference)."""
        # Ensure correct format
        if video_frames.dim() == 4 and video_frames.shape[1] == 3:
            video_frames = video_frames.permute(0, 2, 3, 1)
        
        resized_frames = []
        for frame in video_frames:
            frame_np = frame.numpy() if isinstance(frame, torch.Tensor) else frame
            if frame_np.dtype != np.uint8:
                frame_np = (frame_np * 255).astype(np.uint8)
            
            resized = cv2.resize(frame_np, (self.target_width, self.target_height))
            
            if self.normalize:
                resized = resized.astype(np.float32) / 255.0
                resized = (resized - 0.5) / 0.5  # Normalize to [-1, 1]
            
            resized_frames.append(torch.from_numpy(resized))
        
        return torch.stack(resized_frames)
    
    def temporal_crop(self, video_frames: torch.Tensor, min_ratio: float = 0.8) -> torch.Tensor:
        """Apply temporal cropping to video sequence."""
        T = video_frames.shape[0]
        
        # Random crop length (at least min_ratio of original)
        crop_length = random.randint(int(T * min_ratio), T)
        
        # Random start position
        start_idx = random.randint(0, T - crop_length)
        
        return video_frames[start_idx:start_idx + crop_length]
    
    def mixup_frames(
        self, 
        video1: torch.Tensor, 
        video2: torch.Tensor, 
        alpha: float = 0.2
    ) -> torch.Tensor:
        """Apply mixup augmentation between two video sequences."""
        # Sample mixing coefficient
        lam = np.random.beta(alpha, alpha)
        
        # Ensure same length (pad shorter one)
        T1, T2 = video1.shape[0], video2.shape[0]
        if T1 != T2:
            T_max = max(T1, T2)
            if T1 < T_max:
                padding = torch.zeros(T_max - T1, *video1.shape[1:])
                video1 = torch.cat([video1, padding], dim=0)
            if T2 < T_max:
                padding = torch.zeros(T_max - T2, *video2.shape[1:])
                video2 = torch.cat([video2, padding], dim=0)
        
        # Mix videos
        mixed_video = lam * video1 + (1 - lam) * video2
        
        return mixed_video, lam


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
    """Complete data augmentation pipeline for Valerie."""
    
    def __init__(self, config, training: bool = True):
        """
        Initialize augmentation pipeline from configuration.
        
        Args:
            config: Configuration object
            training: Whether pipeline is for training
        """
        self.training = training
        
        # Video transforms
        self.video_transforms = VideoTransforms(
            target_height=config.model.input_height,
            target_width=config.model.input_width,
            horizontal_flip_prob=config.data.horizontal_flip_prob,
            brightness_range=config.data.brightness_range,
            contrast_range=config.data.contrast_range,
            noise_std=config.data.noise_std
        )
        
        # Audio transforms
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
        
        logger.info(f"✅ DataAugmentationPipeline initialized (training={training})")
    
    def __call__(self, data) -> Union[Dict[str, Any], 'DataSample']:
        """Apply full augmentation pipeline to batch or single sample."""
        
        # Handle DataSample objects
        if hasattr(data, 'video_frames'):
            from src.data.dataset import DataSample
            
            # Apply video transforms
            if data.video_frames is not None:
                transformed_video = self.video_transforms(
                    data.video_frames, 
                    training=self.training
                )
            else:
                transformed_video = data.video_frames
            
            # Apply audio transforms
            if data.audio_waveform is not None:
                transformed_audio = self.audio_transforms(
                    data.audio_waveform, 
                    training=self.training
                )
            else:
                transformed_audio = data.audio_waveform
            
            # Create new DataSample with transformed data
            return DataSample(
                video_frames=transformed_video,
                audio_waveform=transformed_audio,
                video_path=data.video_path,
                audio_path=data.audio_path,
                transcription=data.transcription,
                phoneme_sequence=data.phoneme_sequence,
                phoneme_alignment=data.phoneme_alignment,
                speaker_id=data.speaker_id,
                language=data.language,
                duration=data.duration,
                face_confidence=data.face_confidence,
                transcription_confidence=data.transcription_confidence
            )
        
        # Handle dictionary/batch data
        elif isinstance(data, dict):
            batch_data = data.copy()
            
            # Apply video transforms
            if 'video_frames' in batch_data:
                batch_data['video_frames'] = self.video_transforms(
                    batch_data['video_frames'], 
                    training=self.training
                )
            
            # Apply audio transforms
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
    # Test transforms
    from src.utils.config import Config
    
    config = Config("configs/base_config.yaml")
    
    # Test video transforms
    print("Testing video transforms...")
    video_transforms = VideoTransforms()
    dummy_video = torch.randn(10, 224, 224, 3)  # 10 frames
    
    transformed = video_transforms(dummy_video, training=True)
    print(f"Video: {dummy_video.shape} -> {transformed.shape}")
    
    # Test audio transforms
    print("\nTesting audio transforms...")
    audio_transforms = AudioTransforms()
    dummy_audio = torch.randn(16000)  # 1 second at 16kHz
    
    transformed_audio = audio_transforms(dummy_audio, training=True)
    print(f"Audio: {dummy_audio.shape} -> {transformed_audio.shape}")
    
    # Test augmentation pipeline
    print("\nTesting augmentation pipeline...")
    pipeline = create_augmentation_pipeline(config, training=True)
    
    batch_data = {
        'video_frames': torch.randn(2, 10, 224, 224, 3),
        'audio_waveforms': torch.randn(2, 16000)
    }
    
    augmented = pipeline(batch_data)
    print(f"Pipeline applied successfully!")
    print(f"  Video shape: {augmented['video_frames'].shape}")
    print(f"  Audio shape: {augmented['audio_waveforms'].shape}")
    
    print("\n✅ Transform tests completed!")
