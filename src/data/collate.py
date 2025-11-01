"""
Batch collation functions for Valerie Visual ASR.

Handles variable-length sequences and complex data structures for efficient
batching during training and inference.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from src.data.dataset import DataSample
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BatchData:
    """Structured batch data for Valerie training."""
    
    # Video data
    video_frames: torch.Tensor  # [B, T_max, H, W, C]
    video_lengths: torch.Tensor  # [B]
    video_mask: torch.Tensor  # [B, T_max]
    
    # Audio data (optional)
    audio_waveforms: Optional[torch.Tensor] = None  # [B, T_audio_max]
    audio_lengths: Optional[torch.Tensor] = None  # [B]
    audio_mask: Optional[torch.Tensor] = None  # [B, T_audio_max]
    
    # Text and phoneme data
    transcriptions: Optional[List[str]] = None
    phoneme_sequences: Optional[List[List[str]]] = None
    phoneme_labels: Optional[torch.Tensor] = None  # [B, T_phoneme_max]
    phoneme_lengths: Optional[torch.Tensor] = None  # [B]
    
    # CTC labels
    ctc_labels: Optional[torch.Tensor] = None  # [B, T_ctc_max]
    ctc_lengths: Optional[torch.Tensor] = None  # [B]
    
    # Metadata
    speaker_ids: Optional[List[str]] = None
    durations: Optional[torch.Tensor] = None  # [B]
    languages: Optional[List[str]] = None
    
    # Quality metrics
    face_confidences: Optional[torch.Tensor] = None  # [B]
    transcription_confidences: Optional[torch.Tensor] = None  # [B]
    
    # Augmentation info
    mixup_lambdas: Optional[torch.Tensor] = None  # [B]
    
    def to(self, device: torch.device) -> 'BatchData':
        """Move batch data to device."""
        def move_tensor(tensor):
            return tensor.to(device) if tensor is not None else None
        
        return BatchData(
            video_frames=move_tensor(self.video_frames),
            video_lengths=move_tensor(self.video_lengths),
            video_mask=move_tensor(self.video_mask),
            audio_waveforms=move_tensor(self.audio_waveforms),
            audio_lengths=move_tensor(self.audio_lengths),
            audio_mask=move_tensor(self.audio_mask),
            transcriptions=self.transcriptions,
            phoneme_sequences=self.phoneme_sequences,
            phoneme_labels=move_tensor(self.phoneme_labels),
            phoneme_lengths=move_tensor(self.phoneme_lengths),
            ctc_labels=move_tensor(self.ctc_labels),
            ctc_lengths=move_tensor(self.ctc_lengths),
            speaker_ids=self.speaker_ids,
            durations=move_tensor(self.durations),
            languages=self.languages,
            face_confidences=move_tensor(self.face_confidences),
            transcription_confidences=move_tensor(self.transcription_confidences),
            mixup_lambdas=move_tensor(self.mixup_lambdas)
        )
    
    def __len__(self) -> int:
        """Get batch size."""
        return self.video_frames.shape[0]


class ValerieCollator:
    """Collator for Valerie dataset samples."""
    
    def __init__(
        self,
        pad_video: bool = True,
        pad_audio: bool = True,
        max_video_length: Optional[int] = None,
        max_audio_length: Optional[int] = None,
        video_pad_value: float = 0.0,
        audio_pad_value: float = 0.0,
        return_batch_data: bool = True
    ):
        """
        Initialize collator.
        
        Args:
            pad_video: Whether to pad video sequences
            pad_audio: Whether to pad audio sequences
            max_video_length: Maximum video length (truncate if longer)
            max_audio_length: Maximum audio length (truncate if longer)
            video_pad_value: Value for video padding
            audio_pad_value: Value for audio padding
            return_batch_data: Whether to return BatchData object
        """
        self.pad_video = pad_video
        self.pad_audio = pad_audio
        self.max_video_length = max_video_length
        self.max_audio_length = max_audio_length
        self.video_pad_value = video_pad_value
        self.audio_pad_value = audio_pad_value
        self.return_batch_data = return_batch_data
    
    def __call__(self, batch: List[DataSample]) -> Union[BatchData, Dict[str, Any]]:
        """Collate batch of DataSample objects."""
        
        if not batch:
            raise ValueError("Empty batch")
        
        batch_size = len(batch)
        
        # Collect video data
        video_frames_list = []
        video_lengths = []
        
        for sample in batch:
            if sample.video_frames is not None:
                frames = sample.video_frames
                
                # Ensure correct format [T, H, W, C]
                if frames.dim() == 4 and frames.shape[1] == 3:
                    frames = frames.permute(0, 2, 3, 1)
                
                # Truncate if too long
                if self.max_video_length and frames.shape[0] > self.max_video_length:
                    frames = frames[:self.max_video_length]
                
                video_frames_list.append(frames)
                video_lengths.append(frames.shape[0])
            else:
                # Create dummy video if missing
                dummy_frames = torch.zeros(1, 224, 224, 3)
                video_frames_list.append(dummy_frames)
                video_lengths.append(1)
        
        # Pad video sequences
        if self.pad_video and video_frames_list:
            max_length = max(video_lengths)
            h, w, c = video_frames_list[0].shape[1:]
            
            padded_videos = torch.full(
                (batch_size, max_length, h, w, c),
                self.video_pad_value,
                dtype=video_frames_list[0].dtype
            )
            
            video_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
            
            for i, (frames, length) in enumerate(zip(video_frames_list, video_lengths)):
                padded_videos[i, :length] = frames
                video_mask[i, :length] = True
            
            video_frames_tensor = padded_videos
            video_lengths_tensor = torch.tensor(video_lengths, dtype=torch.long)
        else:
            video_frames_tensor = torch.stack(video_frames_list) if video_frames_list else None
            video_lengths_tensor = torch.tensor(video_lengths, dtype=torch.long)
            video_mask = None
        
        # Collect audio data
        audio_waveforms_list = []
        audio_lengths = []
        has_audio = False
        
        for sample in batch:
            if sample.audio_waveform is not None:
                audio = sample.audio_waveform
                
                # Ensure 1D
                if audio.dim() > 1:
                    audio = audio.squeeze()
                
                # Truncate if too long
                if self.max_audio_length and len(audio) > self.max_audio_length:
                    audio = audio[:self.max_audio_length]
                
                audio_waveforms_list.append(audio)
                audio_lengths.append(len(audio))
                has_audio = True
            else:
                audio_waveforms_list.append(torch.tensor([]))
                audio_lengths.append(0)
        
        # Pad audio sequences
        audio_waveforms_tensor = None
        audio_lengths_tensor = None
        audio_mask = None
        
        if has_audio and self.pad_audio:
            max_audio_length = max(audio_lengths) if audio_lengths else 0
            
            if max_audio_length > 0:
                padded_audio = torch.full(
                    (batch_size, max_audio_length),
                    self.audio_pad_value,
                    dtype=torch.float32
                )
                
                audio_mask = torch.zeros(batch_size, max_audio_length, dtype=torch.bool)
                
                for i, (audio, length) in enumerate(zip(audio_waveforms_list, audio_lengths)):
                    if length > 0:
                        padded_audio[i, :length] = audio
                        audio_mask[i, :length] = True
                
                audio_waveforms_tensor = padded_audio
                audio_lengths_tensor = torch.tensor(audio_lengths, dtype=torch.long)
        
        # Collect text data
        transcriptions = [sample.transcription for sample in batch if sample.transcription]
        phoneme_sequences = [sample.phoneme_sequence for sample in batch if sample.phoneme_sequence]
        
        # Collect metadata
        speaker_ids = [sample.speaker_id for sample in batch]
        durations = torch.tensor([sample.duration or 0.0 for sample in batch], dtype=torch.float32)
        languages = [sample.language or "en" for sample in batch]
        
        # Collect quality metrics
        face_confidences = torch.tensor([
            sample.face_confidence or 0.0 for sample in batch
        ], dtype=torch.float32)
        
        transcription_confidences = torch.tensor([
            sample.transcription_confidence or 0.0 for sample in batch
        ], dtype=torch.float32)
        
        # Create result
        if self.return_batch_data:
            return BatchData(
                video_frames=video_frames_tensor,
                video_lengths=video_lengths_tensor,
                video_mask=video_mask,
                audio_waveforms=audio_waveforms_tensor,
                audio_lengths=audio_lengths_tensor,
                audio_mask=audio_mask,
                transcriptions=transcriptions if transcriptions else None,
                phoneme_sequences=phoneme_sequences if phoneme_sequences else None,
                speaker_ids=speaker_ids,
                durations=durations,
                languages=languages,
                face_confidences=face_confidences,
                transcription_confidences=transcription_confidences
            )
        else:
            # Return dictionary format
            result = {
                'video_frames': video_frames_tensor,
                'video_lengths': video_lengths_tensor,
                'speaker_ids': speaker_ids,
                'durations': durations,
                'face_confidences': face_confidences,
                'transcription_confidences': transcription_confidences
            }
            
            if video_mask is not None:
                result['video_mask'] = video_mask
            
            if audio_waveforms_tensor is not None:
                result['audio_waveforms'] = audio_waveforms_tensor
                result['audio_lengths'] = audio_lengths_tensor
                if audio_mask is not None:
                    result['audio_mask'] = audio_mask
            
            if transcriptions:
                result['transcriptions'] = transcriptions
            
            if phoneme_sequences:
                result['phoneme_sequences'] = phoneme_sequences
            
            result['languages'] = languages
            
            return result


class CTCCollator(ValerieCollator):
    """Specialized collator for CTC training."""
    
    def __init__(
        self,
        phoneme_processor=None,
        max_label_length: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize CTC collator.
        
        Args:
            phoneme_processor: Phoneme processing utilities
            max_label_length: Maximum CTC label length
            **kwargs: Additional arguments for ValerieCollator
        """
        super().__init__(**kwargs)
        self.phoneme_processor = phoneme_processor
        self.max_label_length = max_label_length
    
    def __call__(self, batch: List[DataSample]) -> Union[BatchData, Dict[str, Any]]:
        """Collate batch with CTC labels."""
        
        # Get basic batch data
        batch_data = super().__call__(batch)
        
        # Process CTC labels if phoneme processor is available
        if self.phoneme_processor and hasattr(batch_data, 'transcriptions'):
            ctc_labels_list = []
            ctc_lengths = []
            
            for sample in batch:
                if sample.transcription:
                    # Convert text to CTC labels
                    try:
                        from src.data.phoneme_utils import process_text_to_ctc_labels
                        phonemes, ctc_labels = process_text_to_ctc_labels(sample.transcription)
                        
                        # Truncate if too long
                        if self.max_label_length and len(ctc_labels) > self.max_label_length:
                            ctc_labels = ctc_labels[:self.max_label_length]
                        
                        ctc_labels_list.append(ctc_labels)
                        ctc_lengths.append(len(ctc_labels))
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to process CTC labels: {e}")
                        # Fallback to empty labels
                        ctc_labels_list.append(torch.tensor([0], dtype=torch.long))
                        ctc_lengths.append(1)
                else:
                    # Empty labels for samples without transcription
                    ctc_labels_list.append(torch.tensor([0], dtype=torch.long))
                    ctc_lengths.append(1)
            
            # Pad CTC labels
            if ctc_labels_list:
                max_label_length = max(ctc_lengths)
                padded_labels = torch.zeros(len(batch), max_label_length, dtype=torch.long)
                
                for i, (labels, length) in enumerate(zip(ctc_labels_list, ctc_lengths)):
                    padded_labels[i, :length] = labels
                
                # Update batch data
                if isinstance(batch_data, BatchData):
                    batch_data.ctc_labels = padded_labels
                    batch_data.ctc_lengths = torch.tensor(ctc_lengths, dtype=torch.long)
                else:
                    batch_data['ctc_labels'] = padded_labels
                    batch_data['ctc_lengths'] = torch.tensor(ctc_lengths, dtype=torch.long)
        
        return batch_data


def create_collator(
    collator_type: str = "default",
    config=None,
    **kwargs
) -> ValerieCollator:
    """
    Factory function to create collators.
    
    Args:
        collator_type: Type of collator ("default", "ctc")
        config: Configuration object
        **kwargs: Additional arguments
        
    Returns:
        Configured collator
    """
    if config:
        # Extract parameters from config
        collator_kwargs = {
            'max_video_length': getattr(config.data, 'max_sequence_length', None),
            'max_audio_length': getattr(config.data, 'max_audio_length', None),
            **kwargs
        }
    else:
        collator_kwargs = kwargs
    
    if collator_type == "ctc":
        return CTCCollator(**collator_kwargs)
    else:
        return ValerieCollator(**collator_kwargs)


# Utility functions
def pad_sequence_2d(sequences: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    """Pad 2D sequences to same length."""
    if not sequences:
        return torch.empty(0)
    
    max_len = max(seq.shape[0] for seq in sequences)
    feature_dim = sequences[0].shape[1]
    
    padded = torch.full((len(sequences), max_len, feature_dim), pad_value)
    
    for i, seq in enumerate(sequences):
        length = seq.shape[0]
        padded[i, :length] = seq
    
    return padded


def create_attention_mask(lengths: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    """Create attention mask from sequence lengths."""
    if max_length is None:
        max_length = lengths.max().item()
    
    batch_size = len(lengths)
    mask = torch.arange(max_length).expand(batch_size, max_length) < lengths.unsqueeze(1)
    
    return mask


if __name__ == "__main__":
    # Test collators
    from src.data.dataset import DataSample
    
    print("Testing collators...")
    
    # Create dummy samples
    samples = []
    for i in range(3):
        sample = DataSample(
            video_frames=torch.randn(10 + i*2, 224, 224, 3),
            audio_waveform=torch.randn(16000 + i*1000),
            transcription=f"Hello world sample {i}",
            speaker_id=f"speaker_{i}",
            duration=1.0 + i*0.2
        )
        samples.append(sample)
    
    # Test default collator
    collator = ValerieCollator()
    batch_data = collator(samples)
    
    print(f"✅ Default collator:")
    print(f"   Video shape: {batch_data.video_frames.shape}")
    print(f"   Video lengths: {batch_data.video_lengths}")
    print(f"   Audio shape: {batch_data.audio_waveforms.shape if batch_data.audio_waveforms is not None else None}")
    print(f"   Batch size: {len(batch_data)}")
    
    # Test CTC collator
    ctc_collator = CTCCollator()
    ctc_batch = ctc_collator(samples)
    
    print(f"\n✅ CTC collator:")
    print(f"   Video shape: {ctc_batch.video_frames.shape}")
    print(f"   Has CTC labels: {ctc_batch.ctc_labels is not None}")
    
    print("\n✅ Collator tests completed!")
