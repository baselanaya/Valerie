"""
Complete Inference Engine for Valerie Visual ASR.

⚠️ **DEPRECATED:** This module uses the old video-based ValerieModel.

For audio-only inference with ensemble distillation, use:
- src.models.audio_phoneme_model.AudioPhonemeASR
- Direct forward pass with audio inputs

Migration example:
```python
from src.models.audio_phoneme_model import AudioPhonemeASR

model = AudioPhonemeASR.from_pretrained("checkpoints/stage1_best.pt")
audio = load_audio("audio.wav")  # [T] at 16kHz
outputs = model(audio.unsqueeze(0))
predictions = model.decode_greedy(audio.unsqueeze(0))
```

This file is kept for backward compatibility but will be removed in future versions.

---

Original docstring:
Combines all components for end-to-end inference:
- Video preprocessing and feature extraction
- Model forward pass (3D CNN + Conformer + CTC/Attention)
- CTC beam search decoding
- Phoneme-to-sentence reconstruction
- Batch processing and real-time inference
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, NamedTuple
from dataclasses import dataclass
import time
import logging

from src.models import ValerieModel
from src.inference.ctc_decoder import CTCBeamSearchDecoder, DecodingConfig, BeamSearchResult
from src.inference.phoneme_reconstruction import PhonemeToSentenceReconstructor, ReconstructionConfig
from src.data.transforms import VideoTransforms, AudioTransforms
from src.utils.config import Config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    
    # Model settings
    model_checkpoint: str = "checkpoints/best_model.pt"
    device: str = "auto"  # auto, cuda, cpu
    
    # Input processing
    video_size: Tuple[int, int] = (224, 224)
    sequence_length: int = 150
    fps: int = 25
    
    # Batch processing
    batch_size: int = 8
    max_batch_size: int = 32
    
    # CTC decoding
    decoding_config: DecodingConfig = None
    
    # Phoneme reconstruction
    reconstruction_config: ReconstructionConfig = None
    
    # Performance
    use_mixed_precision: bool = True
    compile_model: bool = True
    
    # Real-time settings
    real_time_buffer_size: int = 150  # frames
    real_time_overlap: int = 30       # frame overlap
    
    def __post_init__(self):
        if self.decoding_config is None:
            self.decoding_config = DecodingConfig()
        if self.reconstruction_config is None:
            self.reconstruction_config = ReconstructionConfig()


class InferenceResult(NamedTuple):
    """Complete inference result."""
    
    # Final output
    text: str
    confidence: float
    
    # Intermediate results
    phonemes: List[str]
    phoneme_confidence: float
    reconstruction_confidence: float
    
    # Timing
    processing_time: float
    
    # Debug info
    ctc_result: Optional[BeamSearchResult] = None
    raw_logits: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None


class ValerieInferenceEngine:
    """
    Complete inference engine for Valerie Visual ASR.
    
    Provides high-level interface for:
    - Single video inference
    - Batch video processing
    - Real-time streaming inference
    - Performance optimization
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = self._setup_device()
        
        # Load model
        self.model = None
        self.model_config = None
        self._load_model()
        
        # Initialize components
        self.ctc_decoder = CTCBeamSearchDecoder(config.decoding_config)
        self.reconstructor = PhonemeToSentenceReconstructor(
            config.reconstruction_config, 
            device=str(self.device)
        )
        
        # Initialize transforms
        self._setup_transforms()
        
        # Real-time state
        self.real_time_buffer = []
        self.real_time_state = {}
        
        logger.info(f"✅ ValerieInferenceEngine initialized:")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Model: {config.model_checkpoint}")
        logger.info(f"   Mixed precision: {config.use_mixed_precision}")
        logger.info(f"   Batch size: {config.batch_size}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        else:
            device = self.config.device
        
        return torch.device(device)
    
    def _load_model(self):
        """Load the trained Valerie model."""
        
        try:
            # Load checkpoint
            checkpoint_path = Path(self.config.model_checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Extract model config
            if 'config' in checkpoint:
                self.model_config = Config.from_dict(checkpoint['config'])
            else:
                # Fallback to default config
                logger.warning("No model config in checkpoint, using default")
                self.model_config = Config('configs/base_config.yaml')
            
            # Create model
            self.model = ValerieModel(self.model_config)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint  # Assume checkpoint is state dict
            
            # Handle DDP/DataParallel state dict
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            # Model compilation for better performance
            if self.config.compile_model and hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model, mode='default')
                    logger.info("✅ Model compiled for better performance")
                except Exception as e:
                    logger.warning(f"⚠️ Model compilation failed: {e}")
            
            logger.info("✅ Model loaded successfully")
            
            # Log model info
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"   Total parameters: {total_params:,}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _setup_transforms(self):
        """Setup video and audio transforms."""
        
        self.video_transforms = VideoTransforms(
            size=self.config.video_size,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            augmentation_prob=0.0  # No augmentation for inference
        )
        
        self.audio_transforms = AudioTransforms(
            sample_rate=16000,
            n_mels=80,
            augmentation_prob=0.0  # No augmentation for inference
        )
    
    def infer_video(
        self,
        video_path: str,
        return_debug_info: bool = False
    ) -> InferenceResult:
        """
        Infer single video file.
        
        Args:
            video_path: Path to video file
            return_debug_info: Whether to return debug information
            
        Returns:
            Inference result
        """
        start_time = time.time()
        
        try:
            # Load and preprocess video
            video_frames = self._load_video(video_path)
            if video_frames is None:
                return self._empty_result(time.time() - start_time)
            
            # Run inference
            result = self._infer_frames(video_frames, return_debug_info)
            result = result._replace(processing_time=time.time() - start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Video inference failed: {e}")
            return self._empty_result(time.time() - start_time)
    
    def infer_batch(
        self,
        video_paths: List[str],
        return_debug_info: bool = False
    ) -> List[InferenceResult]:
        """
        Infer batch of video files.
        
        Args:
            video_paths: List of video file paths
            return_debug_info: Whether to return debug information
            
        Returns:
            List of inference results
        """
        results = []
        batch_size = self.config.batch_size
        
        # Process in batches
        for i in range(0, len(video_paths), batch_size):
            batch_paths = video_paths[i:i + batch_size]
            batch_results = self._infer_video_batch(batch_paths, return_debug_info)
            results.extend(batch_results)
        
        return results
    
    def _infer_video_batch(
        self,
        video_paths: List[str],
        return_debug_info: bool = False
    ) -> List[InferenceResult]:
        """Infer batch of videos efficiently."""
        
        start_time = time.time()
        
        # Load all videos
        batch_frames = []
        valid_indices = []
        
        for i, video_path in enumerate(video_paths):
            frames = self._load_video(video_path)
            if frames is not None:
                batch_frames.append(frames)
                valid_indices.append(i)
        
        if not batch_frames:
            # Return empty results for all
            return [self._empty_result(0.0) for _ in video_paths]
        
        # Batch inference
        try:
            batch_results = self._infer_frames_batch(batch_frames, return_debug_info)
        except Exception as e:
            logger.error(f"❌ Batch inference failed: {e}")
            batch_results = [self._empty_result(0.0) for _ in batch_frames]
        
        # Map results back to original order
        results = []
        batch_idx = 0
        processing_time = time.time() - start_time
        
        for i in range(len(video_paths)):
            if i in valid_indices:
                result = batch_results[batch_idx]
                result = result._replace(processing_time=processing_time / len(video_paths))
                results.append(result)
                batch_idx += 1
            else:
                results.append(self._empty_result(0.0))
        
        return results
    
    def infer_real_time(
        self,
        frame: np.ndarray,
        audio_chunk: Optional[np.ndarray] = None
    ) -> Optional[InferenceResult]:
        """
        Process single frame for real-time inference.
        
        Args:
            frame: Video frame [H, W, C]
            audio_chunk: Optional audio chunk
            
        Returns:
            Inference result when buffer is ready, None otherwise
        """
        
        # Add frame to buffer
        self.real_time_buffer.append(frame)
        
        # Check if buffer is ready
        if len(self.real_time_buffer) >= self.config.real_time_buffer_size:
            
            # Extract frames for processing
            frames = np.array(self.real_time_buffer[-self.config.real_time_buffer_size:])
            
            # Run inference
            try:
                result = self._infer_frames(frames, return_debug_info=False)
                
                # Remove processed frames (keep overlap)
                overlap = self.config.real_time_overlap
                self.real_time_buffer = self.real_time_buffer[-overlap:]
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Real-time inference failed: {e}")
                return None
        
        return None
    
    def _load_video(self, video_path: str) -> Optional[np.ndarray]:
        """Load and preprocess video file."""
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"❌ Cannot open video: {video_path}")
                return None
            
            frames = []
            frame_count = 0
            target_frames = self.config.sequence_length
            
            # Calculate frame skip for target length
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > target_frames:
                skip = total_frames / target_frames
            else:
                skip = 1
            
            frame_idx = 0
            while len(frames) < target_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx >= int(frame_count * skip):
                    # Resize frame
                    frame = cv2.resize(frame, self.config.video_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    frame_count += 1
                
                frame_idx += 1
            
            cap.release()
            
            if not frames:
                logger.error(f"❌ No frames extracted from: {video_path}")
                return None
            
            # Pad or truncate to target length
            while len(frames) < target_frames:
                frames.append(frames[-1])  # Repeat last frame
            
            frames = frames[:target_frames]  # Truncate if too long
            
            return np.array(frames)  # [T, H, W, C]
            
        except Exception as e:
            logger.error(f"❌ Video loading failed: {e}")
            return None
    
    def _infer_frames(
        self,
        frames: np.ndarray,
        return_debug_info: bool = False
    ) -> InferenceResult:
        """Infer single sequence of frames."""
        
        batch_results = self._infer_frames_batch([frames], return_debug_info)
        return batch_results[0]
    
    def _infer_frames_batch(
        self,
        batch_frames: List[np.ndarray],
        return_debug_info: bool = False
    ) -> List[InferenceResult]:
        """Infer batch of frame sequences."""
        
        try:
            # Prepare batch
            batch_tensor = self._prepare_batch(batch_frames)
            
            # Model forward pass
            with torch.no_grad():
                if self.config.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_tensor)
                else:
                    outputs = self.model(batch_tensor)
            
            # Extract CTC logits and attention logits
            ctc_logits = outputs.get('ctc_logits')
            attention_logits = outputs.get('attention_logits')
            
            if ctc_logits is None:
                raise ValueError("Model did not return CTC logits")
            
            # CTC decoding
            input_lengths = torch.full((len(batch_frames),), ctc_logits.size(1), dtype=torch.long)
            ctc_results = self.ctc_decoder.decode(ctc_logits, input_lengths)
            
            # Phoneme reconstruction
            phoneme_sequences = [result.phonemes for result in ctc_results]
            reconstruction_results = self.reconstructor.reconstruct(phoneme_sequences, return_attention=return_debug_info)
            
            # Combine results
            final_results = []
            for i, (ctc_result, recon_result) in enumerate(zip(ctc_results, reconstruction_results)):
                
                # Calculate overall confidence
                overall_confidence = (ctc_result.confidence + recon_result.confidence) / 2
                
                result = InferenceResult(
                    text=recon_result.text,
                    confidence=overall_confidence,
                    phonemes=ctc_result.phonemes,
                    phoneme_confidence=ctc_result.confidence,
                    reconstruction_confidence=recon_result.confidence,
                    processing_time=0.0,  # Will be set by caller
                    ctc_result=ctc_result if return_debug_info else None,
                    raw_logits=ctc_logits[i] if return_debug_info else None,
                    attention_weights=recon_result.attention_weights if return_debug_info else None
                )
                
                final_results.append(result)
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Frame inference failed: {e}")
            return [self._empty_result(0.0) for _ in batch_frames]
    
    def _prepare_batch(self, batch_frames: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        """Prepare batch tensor for model input."""
        
        batch_tensors = []
        
        for frames in batch_frames:
            # Apply transforms
            video_tensor = self._process_video_frames(frames)
            batch_tensors.append(video_tensor)
        
        # Stack into batch
        video_batch = torch.stack(batch_tensors, dim=0).to(self.device)
        
        # Create input dictionary
        batch_dict = {
            'video': video_batch,
            'input_lengths': torch.full((len(batch_frames),), video_batch.size(2), dtype=torch.long, device=self.device)
        }
        
        return batch_dict
    
    def _process_video_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Process video frames to tensor."""
        
        # frames: [T, H, W, C] numpy array
        # Convert to tensor and normalize
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        
        # Rearrange to [C, T, H, W]
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)
        
        # Apply normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        frames_tensor = (frames_tensor - mean) / std
        
        return frames_tensor
    
    def _empty_result(self, processing_time: float) -> InferenceResult:
        """Create empty result for failed inference."""
        
        return InferenceResult(
            text="",
            confidence=0.0,
            phonemes=[],
            phoneme_confidence=0.0,
            reconstruction_confidence=0.0,
            processing_time=processing_time,
            ctc_result=None,
            raw_logits=None,
            attention_weights=None
        )
    
    def benchmark(
        self,
        test_videos: List[str],
        num_runs: int = 5
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            test_videos: List of test video paths
            num_runs: Number of benchmark runs
            
        Returns:
            Performance metrics
        """
        
        logger.info(f"🔥 Starting inference benchmark with {len(test_videos)} videos")
        
        all_times = []
        successful_inferences = 0
        
        for run in range(num_runs):
            logger.info(f"Run {run + 1}/{num_runs}")
            
            start_time = time.time()
            results = self.infer_batch(test_videos)
            total_time = time.time() - start_time
            
            # Count successful inferences
            run_successful = sum(1 for r in results if r.confidence > 0.1)
            successful_inferences += run_successful
            
            all_times.append(total_time)
            
            logger.info(f"  Time: {total_time:.2f}s, Success: {run_successful}/{len(test_videos)}")
        
        # Calculate metrics
        avg_time = np.mean(all_times)
        std_time = np.std(all_times)
        min_time = np.min(all_times)
        max_time = np.max(all_times)
        
        throughput = len(test_videos) / avg_time  # videos per second
        success_rate = successful_inferences / (num_runs * len(test_videos))
        
        metrics = {
            'avg_time_seconds': avg_time,
            'std_time_seconds': std_time,
            'min_time_seconds': min_time,
            'max_time_seconds': max_time,
            'throughput_videos_per_second': throughput,
            'success_rate': success_rate,
            'total_runs': num_runs,
            'total_videos': len(test_videos)
        }
        
        logger.info("📊 Benchmark Results:")
        logger.info(f"   Average time: {avg_time:.2f} ± {std_time:.2f} seconds")
        logger.info(f"   Throughput: {throughput:.2f} videos/second")
        logger.info(f"   Success rate: {success_rate:.2%}")
        
        return metrics
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information."""
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'device': str(self.device),
            'mixed_precision': self.config.use_mixed_precision,
            'compiled': hasattr(self.model, '_orig_mod'),  # Check if compiled
            'checkpoint_path': self.config.model_checkpoint
        }
