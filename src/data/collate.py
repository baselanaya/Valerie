"""
Batch collation functions for Valerie Audio ASR.

Handles variable-length audio sequences and complex data structures for efficient
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
    """Structured batch data for Valerie audio-only training."""

    # Audio data
    audio_waveforms: torch.Tensor  # [B, T_audio_max]
    audio_lengths: torch.Tensor  # [B]
    audio_mask: torch.Tensor  # [B, T_audio_max]

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
    transcription_confidences: Optional[torch.Tensor] = None  # [B]

    # Augmentation info
    mixup_lambdas: Optional[torch.Tensor] = None  # [B]

    def to(self, device: torch.device) -> 'BatchData':
        """Move batch data to device."""
        def move_tensor(tensor):
            return tensor.to(device) if tensor is not None else None

        return BatchData(
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
            transcription_confidences=move_tensor(self.transcription_confidences),
            mixup_lambdas=move_tensor(self.mixup_lambdas)
        )

    def __len__(self) -> int:
        """Get batch size."""
        return self.audio_waveforms.shape[0]


class ValerieCollator:
    """Audio-only collator for Valerie dataset samples."""

    def __init__(
        self,
        pad_audio: bool = True,
        max_audio_length: Optional[int] = None,
        audio_pad_value: float = 0.0,
        return_batch_data: bool = True
    ):
        """
        Initialize audio-only collator.

        Args:
            pad_audio: Whether to pad audio sequences
            max_audio_length: Maximum audio length (truncate if longer)
            audio_pad_value: Value for audio padding
            return_batch_data: Whether to return BatchData object
        """
        self.pad_audio = pad_audio
        self.max_audio_length = max_audio_length
        self.audio_pad_value = audio_pad_value
        self.return_batch_data = return_batch_data

    def __call__(self, batch: List[DataSample]) -> Union[BatchData, Dict[str, Any]]:
        """Collate batch of DataSample objects (audio-only)."""

        if not batch:
            raise ValueError("Empty batch")

        batch_size = len(batch)

        # Collect audio data
        audio_waveforms_list = []
        audio_lengths = []

        for sample in batch:
            if hasattr(sample, 'audio_waveform') and sample.audio_waveform is not None:
                audio = sample.audio_waveform

                # Ensure 1D
                if audio.dim() > 1:
                    audio = audio.squeeze()

                # Truncate if too long
                if self.max_audio_length and len(audio) > self.max_audio_length:
                    audio = audio[:self.max_audio_length]

                audio_waveforms_list.append(audio)
                audio_lengths.append(len(audio))
            else:
                # Create minimal audio if missing
                audio_waveforms_list.append(torch.zeros(1600))  # 0.1s at 16kHz
                audio_lengths.append(1600)

        # Pad audio sequences
        if self.pad_audio:
            max_audio_length = max(audio_lengths)

            padded_audio = torch.full(
                (batch_size, max_audio_length),
                self.audio_pad_value,
                dtype=torch.float32
            )

            audio_mask = torch.zeros(batch_size, max_audio_length, dtype=torch.bool)

            for i, (audio, length) in enumerate(zip(audio_waveforms_list, audio_lengths)):
                padded_audio[i, :length] = audio
                audio_mask[i, :length] = True

            audio_waveforms_tensor = padded_audio
            audio_lengths_tensor = torch.tensor(audio_lengths, dtype=torch.long)
        else:
            audio_waveforms_tensor = torch.stack(audio_waveforms_list)
            audio_lengths_tensor = torch.tensor(audio_lengths, dtype=torch.long)
            audio_mask = torch.ones(batch_size, max(audio_lengths), dtype=torch.bool)

        # Collect text data
        transcriptions = [
            sample.transcription if hasattr(sample, 'transcription') else None
            for sample in batch
        ]
        transcriptions = [t for t in transcriptions if t is not None]

        phoneme_sequences = [
            sample.phoneme_sequence if hasattr(sample, 'phoneme_sequence') else None
            for sample in batch
        ]
        phoneme_sequences = [p for p in phoneme_sequences if p is not None]

        # Collect metadata
        speaker_ids = [
            sample.speaker_id if hasattr(sample, 'speaker_id') else None
            for sample in batch
        ]

        durations = torch.tensor([
            sample.duration if hasattr(sample, 'duration') and sample.duration else 0.0
            for sample in batch
        ], dtype=torch.float32)

        languages = [
            sample.language if hasattr(sample, 'language') and sample.language else "en"
            for sample in batch
        ]

        # Collect quality metrics
        transcription_confidences = torch.tensor([
            sample.transcription_confidence if hasattr(sample, 'transcription_confidence') and sample.transcription_confidence else 1.0
            for sample in batch
        ], dtype=torch.float32)

        # Create result
        if self.return_batch_data:
            return BatchData(
                audio_waveforms=audio_waveforms_tensor,
                audio_lengths=audio_lengths_tensor,
                audio_mask=audio_mask,
                transcriptions=transcriptions if transcriptions else None,
                phoneme_sequences=phoneme_sequences if phoneme_sequences else None,
                speaker_ids=speaker_ids,
                durations=durations,
                languages=languages,
                transcription_confidences=transcription_confidences
            )
        else:
            # Return dictionary format
            result = {
                'audio_waveforms': audio_waveforms_tensor,
                'audio_lengths': audio_lengths_tensor,
                'audio_mask': audio_mask,
                'speaker_ids': speaker_ids,
                'durations': durations,
                'transcription_confidences': transcription_confidences,
                'languages': languages
            }

            if transcriptions:
                result['transcriptions'] = transcriptions

            if phoneme_sequences:
                result['phoneme_sequences'] = phoneme_sequences

            return result


class CTCCollator(ValerieCollator):
    """Specialized audio-only collator for CTC training."""

    def __init__(
        self,
        phoneme_processor=None,
        max_label_length: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize CTC collator.

        Args:
            phoneme_processor: Processor for converting phonemes to indices
            max_label_length: Maximum label length
            **kwargs: Arguments passed to ValerieCollator
        """
        super().__init__(**kwargs)
        self.phoneme_processor = phoneme_processor
        self.max_label_length = max_label_length

    def __call__(self, batch: List[DataSample]) -> Union[BatchData, Dict[str, Any]]:
        """Collate batch with CTC labels."""

        # First collate using parent class
        collated = super().__call__(batch)

        # Add CTC labels if phoneme processor is available
        if self.phoneme_processor and collated.phoneme_sequences:
            ctc_labels_list = []
            ctc_lengths = []

            for phoneme_seq in collated.phoneme_sequences:
                # Convert phonemes to indices
                indices = self.phoneme_processor.encode(phoneme_seq)

                # Truncate if too long
                if self.max_label_length and len(indices) > self.max_label_length:
                    indices = indices[:self.max_label_length]

                ctc_labels_list.append(torch.tensor(indices, dtype=torch.long))
                ctc_lengths.append(len(indices))

            # Pad CTC labels
            if ctc_labels_list:
                max_label_length = max(ctc_lengths)
                padded_labels = torch.full(
                    (len(ctc_labels_list), max_label_length),
                    0,  # Blank token
                    dtype=torch.long
                )

                for i, (labels, length) in enumerate(zip(ctc_labels_list, ctc_lengths)):
                    padded_labels[i, :length] = labels

                collated.ctc_labels = padded_labels
                collated.ctc_lengths = torch.tensor(ctc_lengths, dtype=torch.long)

        return collated


def create_collator(
    config,
    phoneme_processor=None,
    for_ctc: bool = False
) -> Union[ValerieCollator, CTCCollator]:
    """
    Factory function to create appropriate collator.

    Args:
        config: Configuration object
        phoneme_processor: Phoneme processor for CTC
        for_ctc: Whether to create CTC collator

    Returns:
        Collator instance
    """
    max_audio_length = getattr(config.data, 'max_audio_length', None)

    if for_ctc and phoneme_processor:
        max_label_length = getattr(config.data, 'max_label_length', None)
        return CTCCollator(
            phoneme_processor=phoneme_processor,
            max_label_length=max_label_length,
            max_audio_length=max_audio_length
        )
    else:
        return ValerieCollator(
            max_audio_length=max_audio_length
        )


if __name__ == "__main__":
    # Test audio-only collator
    print("🧪 Testing audio-only collator...")

    # Create dummy samples
    samples = []
    for i in range(3):
        sample = DataSample(
            audio_waveform=torch.randn(16000 + i * 1000),
            transcription=f"Test transcription {i}",
            phoneme_sequence=['T', 'EH', 'S', 'T'] + ['AH'] * i,
            speaker_id=f"speaker_{i}",
            language="en",
            duration=1.0 + i * 0.1
        )
        samples.append(sample)

    # Test ValerieCollator
    collator = ValerieCollator()
    batch = collator(samples)

    print(f"\n✅ Collation successful!")
    print(f"   Audio shape: {batch.audio_waveforms.shape}")
    print(f"   Audio lengths: {batch.audio_lengths}")
    print(f"   Audio mask shape: {batch.audio_mask.shape}")
    print(f"   Transcriptions: {len(batch.transcriptions) if batch.transcriptions else 0}")
    print(f"   Phoneme sequences: {len(batch.phoneme_sequences) if batch.phoneme_sequences else 0}")

    print("\n✅ Audio-only collator tests completed!")
