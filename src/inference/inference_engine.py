"""
Audio-only inference engine for Valerie ASR using ensemble distillation.

Replaces the old video-based ValerieInferenceEngine with audio-only approach
using AudioPhonemeASR with 3-teacher ensemble distillation.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
from dataclasses import dataclass

from src.models.audio_phoneme_model import AudioPhonemeASR
from src.models.qwen_llm import QwenPhonemeToText
from src.data.transforms import AudioTransforms
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for audio-only inference."""

    # Model paths
    stage1_checkpoint: str  # Audio → Phonemes model
    stage2_checkpoint: Optional[str] = None  # Phonemes → Text (Qwen LLM)

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Audio parameters
    sample_rate: int = 16000

    # Inference parameters
    batch_size: int = 1
    beam_size: int = 1  # 1 = greedy decoding
    use_llm_reconstruction: bool = False  # Whether to use Stage 2 LLM

    # Output options
    return_confidence: bool = True
    return_phonemes: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.use_llm_reconstruction and not self.stage2_checkpoint:
            raise ValueError("stage2_checkpoint required when use_llm_reconstruction=True")


class AudioInferenceEngine:
    """
    Audio-only inference engine for Valerie ASR.

    Supports two modes:
    1. Audio → Phonemes only (Stage 1)
    2. Audio → Phonemes → Text (Stage 1 + Stage 2)

    Example:
        ```python
        # Initialize engine
        config = InferenceConfig(
            stage1_checkpoint="checkpoints/stage1_best.pt",
            stage2_checkpoint="checkpoints/stage2_best.pt",
            use_llm_reconstruction=True
        )
        engine = AudioInferenceEngine(config)

        # Transcribe audio file
        result = engine.transcribe("audio.wav")
        print(result.text)
        print(f"Confidence: {result.confidence:.2f}")
        ```
    """

    def __init__(self, config: InferenceConfig):
        """
        Initialize audio inference engine.

        Args:
            config: Inference configuration
        """
        self.config = config
        self.device = torch.device(config.device)

        logger.info(f"🚀 Initializing AudioInferenceEngine...")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Stage 1 (Audio→Phonemes): {config.stage1_checkpoint}")
        if config.use_llm_reconstruction:
            logger.info(f"   Stage 2 (Phonemes→Text): {config.stage2_checkpoint}")

        # Load Stage 1: Audio → Phonemes
        self.phoneme_model = self._load_stage1_model()

        # Load Stage 2: Phonemes → Text (optional)
        self.text_model = None
        if config.use_llm_reconstruction:
            self.text_model = self._load_stage2_model()

        # Audio preprocessing
        self.audio_transforms = AudioTransforms(
            sample_rate=config.sample_rate
        )

        logger.info("✅ AudioInferenceEngine initialized successfully!")

    def _load_stage1_model(self) -> AudioPhonemeASR:
        """Load Stage 1 model (Audio → Phonemes)."""
        logger.info("📥 Loading Stage 1 model...")

        try:
            # Load checkpoint
            checkpoint = torch.load(
                self.config.stage1_checkpoint,
                map_location=self.device
            )

            # Get model config
            model_config = checkpoint.get('config', {})

            # Create model
            model = AudioPhonemeASR(
                embed_dim=model_config.get('embed_dim', 256),
                conformer_layers=model_config.get('conformer_layers', 12),
                conformer_heads=model_config.get('conformer_heads', 4),
                vocab_size=model_config.get('vocab_size', 40),
                enable_distillation=False  # Disable for inference
            )

            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()

            logger.info("✅ Stage 1 model loaded successfully")
            return model

        except Exception as e:
            logger.error(f"❌ Failed to load Stage 1 model: {e}")
            raise

    def _load_stage2_model(self) -> QwenPhonemeToText:
        """Load Stage 2 model (Phonemes → Text)."""
        logger.info("📥 Loading Stage 2 model...")

        try:
            model = QwenPhonemeToText.from_pretrained(
                self.config.stage2_checkpoint
            )
            model.to(self.device)
            model.eval()

            logger.info("✅ Stage 2 model loaded successfully")
            return model

        except Exception as e:
            logger.error(f"❌ Failed to load Stage 2 model: {e}")
            raise

    def load_audio(
        self,
        audio_path: Union[str, Path],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Load and preprocess audio file.

        Args:
            audio_path: Path to audio file
            normalize: Whether to normalize audio

        Returns:
            Audio tensor [T]
        """
        # Load audio
        waveform, sr = torchaudio.load(str(audio_path))

        # Resample if needed
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        else:
            waveform = waveform.squeeze(0)

        # Normalize
        if normalize:
            waveform = self.audio_transforms(waveform, training=False)

        return waveform

    @torch.no_grad()
    def transcribe(
        self,
        audio_input: Union[str, Path, torch.Tensor],
        return_phonemes: bool = None,
        return_confidence: bool = None
    ) -> 'TranscriptionResult':
        """
        Transcribe audio to text.

        Args:
            audio_input: Audio file path or tensor [T]
            return_phonemes: Whether to return phoneme sequence
            return_confidence: Whether to return confidence scores

        Returns:
            TranscriptionResult with text and optional metadata
        """
        # Use config defaults if not specified
        if return_phonemes is None:
            return_phonemes = self.config.return_phonemes
        if return_confidence is None:
            return_confidence = self.config.return_confidence

        # Load audio if path provided
        if isinstance(audio_input, (str, Path)):
            audio = self.load_audio(audio_input)
        else:
            audio = audio_input

        # Ensure on correct device
        audio = audio.to(self.device)

        # Add batch dimension
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # [1, T]

        # Stage 1: Audio → Phonemes
        phonemes, confidences = self.phoneme_model.decode_greedy(audio)

        # Get first batch element
        phonemes_seq = phonemes[0]
        confidence_scores = confidences[0] if confidences else None

        # Stage 2: Phonemes → Text (optional)
        if self.config.use_llm_reconstruction and self.text_model:
            text = self.text_model(
                [phonemes_seq],
                [confidence_scores] if confidence_scores else None,
                return_thinking=False
            )[0]
        else:
            # Join phonemes as text
            text = ' '.join(phonemes_seq)

        # Calculate overall confidence
        overall_confidence = None
        if return_confidence and confidence_scores is not None:
            overall_confidence = float(np.mean(confidence_scores))

        return TranscriptionResult(
            text=text,
            phonemes=phonemes_seq if return_phonemes else None,
            confidence=overall_confidence,
            phoneme_confidences=confidence_scores.tolist() if (
                return_confidence and confidence_scores is not None
            ) else None
        )

    @torch.no_grad()
    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        return_phonemes: bool = None,
        return_confidence: bool = None
    ) -> List['TranscriptionResult']:
        """
        Transcribe multiple audio files in batch.

        Args:
            audio_paths: List of audio file paths
            return_phonemes: Whether to return phoneme sequences
            return_confidence: Whether to return confidence scores

        Returns:
            List of TranscriptionResult objects
        """
        results = []

        # Process in batches
        for i in range(0, len(audio_paths), self.config.batch_size):
            batch_paths = audio_paths[i:i + self.config.batch_size]

            for audio_path in batch_paths:
                result = self.transcribe(
                    audio_path,
                    return_phonemes=return_phonemes,
                    return_confidence=return_confidence
                )
                results.append(result)

        return results


@dataclass
class TranscriptionResult:
    """Result from audio transcription."""

    text: str  # Transcribed text
    phonemes: Optional[List[str]] = None  # Phoneme sequence
    confidence: Optional[float] = None  # Overall confidence [0-1]
    phoneme_confidences: Optional[List[float]] = None  # Per-phoneme confidences

    def __str__(self) -> str:
        """String representation."""
        s = f"Text: {self.text}"
        if self.confidence is not None:
            s += f"\nConfidence: {self.confidence:.2%}"
        if self.phonemes:
            s += f"\nPhonemes: {' '.join(self.phonemes)}"
        return s


# Factory function
def create_inference_engine(
    stage1_checkpoint: str,
    stage2_checkpoint: Optional[str] = None,
    device: str = "auto",
    use_llm: bool = False
) -> AudioInferenceEngine:
    """
    Create audio inference engine with simple interface.

    Args:
        stage1_checkpoint: Path to Stage 1 model
        stage2_checkpoint: Path to Stage 2 model (optional)
        device: Device to use ("auto", "cuda", or "cpu")
        use_llm: Whether to use LLM for text reconstruction

    Returns:
        AudioInferenceEngine instance
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = InferenceConfig(
        stage1_checkpoint=stage1_checkpoint,
        stage2_checkpoint=stage2_checkpoint,
        device=device,
        use_llm_reconstruction=use_llm
    )

    return AudioInferenceEngine(config)


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing AudioInferenceEngine...")

    # This is a dummy example - replace with actual checkpoint paths
    try:
        config = InferenceConfig(
            stage1_checkpoint="checkpoints/stage1_best.pt",
            use_llm_reconstruction=False
        )

        engine = AudioInferenceEngine(config)
        print("✅ Engine initialized successfully!")

        # Test with dummy audio
        dummy_audio = torch.randn(16000)  # 1 second
        result = engine.transcribe(dummy_audio)
        print(f"\nTranscription: {result.text}")

    except FileNotFoundError:
        print("ℹ️ Checkpoint files not found - this is expected for testing")
        print("   Use actual checkpoint paths in production")
