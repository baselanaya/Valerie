#!/usr/bin/env python3
"""
ValerieASR - Main entry point for Valerie Audio ASR System

A state-of-the-art audio-only ASR system using ensemble knowledge distillation.

Usage:
    # Simple transcription
    from ValerieASR import ValerieASR

    asr = ValerieASR("checkpoints/valerie_final.safetensors")
    result = asr.transcribe("audio.wav")
    print(result.text)

    # Batch transcription
    results = asr.transcribe_batch(["audio1.wav", "audio2.wav"])

    # With phoneme output
    result = asr.transcribe("audio.wav", return_phonemes=True)
    print(result.phonemes)
"""

import torch
import json
from pathlib import Path
from typing import Union, List, Optional, Dict
from dataclasses import dataclass
from safetensors.torch import load_file

from src.models.audio_phoneme_model import AudioPhonemeASR
from src.models.qwen_llm import QwenPhonemeToText
from src.data.transforms import AudioTransforms
from src.utils.logging import get_logger
import torchaudio

logger = get_logger(__name__)


@dataclass
class TranscriptionResult:
    """Result from audio transcription."""

    text: str
    phonemes: Optional[List[str]] = None
    confidence: Optional[float] = None
    phoneme_confidences: Optional[List[float]] = None

    def __str__(self) -> str:
        """String representation."""
        s = f"Text: {self.text}"
        if self.confidence is not None:
            s += f"\nConfidence: {self.confidence:.2%}"
        if self.phonemes:
            s += f"\nPhonemes: {' '.join(self.phonemes)}"
        return s


class ValerieASR:
    """
    Main Valerie ASR interface.

    A simple, unified interface for audio transcription using the Valerie
    ensemble distillation model.

    Attributes:
        model: The loaded ASR model
        device: Device for inference (cuda/cpu)
        sample_rate: Audio sample rate (16000 Hz)

    Example:
        ```python
        # Initialize
        asr = ValerieASR("checkpoints/valerie_final.safetensors")

        # Transcribe single file
        result = asr.transcribe("speech.wav")
        print(f"Transcription: {result.text}")
        print(f"Confidence: {result.confidence:.2%}")

        # Transcribe multiple files
        results = asr.transcribe_batch(["file1.wav", "file2.wav"])
        for i, result in enumerate(results):
            print(f"File {i+1}: {result.text}")

        # Get phonemes
        result = asr.transcribe("speech.wav", return_phonemes=True)
        print(f"Phonemes: {' '.join(result.phonemes)}")
        ```
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        use_llm: bool = False
    ):
        """
        Initialize Valerie ASR system.

        Args:
            model_path: Path to model checkpoint (.safetensors or .pt)
            config_path: Path to config file (optional, will look for adjacent config.json)
            device: Device to use ("auto", "cuda", or "cpu")
            use_llm: Whether to use LLM for phoneme-to-text reconstruction

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model format is invalid
        """
        self.model_path = Path(model_path)
        self.use_llm = use_llm

        # Auto-detect device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"🚀 Initializing Valerie ASR...")
        logger.info(f"   Model: {self.model_path.name}")
        logger.info(f"   Device: {self.device}")

        # Load configuration
        if config_path is None:
            # Look for config.json in same directory as model
            config_path = self.model_path.parent / "config.json"

        self.config = self._load_config(config_path)

        # Load model
        self.model = self._load_model()

        # Initialize audio preprocessing
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.audio_transforms = AudioTransforms(sample_rate=self.sample_rate)

        # Load LLM if enabled
        self.llm_model = None
        if use_llm:
            llm_path = self.model_path.parent / "qwen_phoneme_model"
            if llm_path.exists():
                logger.info(f"📥 Loading LLM for text reconstruction...")
                self.llm_model = QwenPhonemeToText.from_pretrained(str(llm_path))
                self.llm_model.to(self.device)
                self.llm_model.eval()
                logger.info("✅ LLM loaded successfully")
            else:
                logger.warning(f"⚠️ LLM path not found: {llm_path}")

        logger.info("✅ Valerie ASR initialized successfully!")

    def _load_config(self, config_path: Path) -> Dict:
        """Load model configuration."""
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"📄 Loaded config from {config_path}")
            return config
        else:
            logger.warning(f"⚠️ Config file not found: {config_path}")
            logger.info("   Using default configuration")
            return {
                'embed_dim': 256,
                'conformer_layers': 12,
                'conformer_heads': 4,
                'vocab_size': 40,
                'sample_rate': 16000
            }

    def _load_model(self) -> AudioPhonemeASR:
        """Load ASR model from checkpoint."""
        logger.info(f"📥 Loading model from {self.model_path}...")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            # Create model architecture
            model = AudioPhonemeASR(
                embed_dim=self.config.get('embed_dim', 256),
                conformer_layers=self.config.get('conformer_layers', 12),
                conformer_heads=self.config.get('conformer_heads', 4),
                vocab_size=self.config.get('vocab_size', 40),
                enable_distillation=False  # Disable for inference
            )

            # Load weights based on file format
            if self.model_path.suffix == '.safetensors':
                # Load from safetensors format
                state_dict = load_file(str(self.model_path))
                model.load_state_dict(state_dict, strict=True)
                logger.info("✅ Loaded model from safetensors format")
            elif self.model_path.suffix == '.pt':
                # Load from PyTorch format
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info("✅ Loaded model from PyTorch format")
            else:
                raise ValueError(f"Unsupported model format: {self.model_path.suffix}")

            # Move to device and set to eval mode
            model.to(self.device)
            model.eval()

            return model

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
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
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
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
        return_phonemes: bool = False,
        return_confidence: bool = True
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio_input: Audio file path or tensor [T]
            return_phonemes: Whether to return phoneme sequence
            return_confidence: Whether to return confidence scores

        Returns:
            TranscriptionResult with text and optional metadata

        Example:
            ```python
            # From file
            result = asr.transcribe("speech.wav")
            print(result.text)

            # From tensor
            audio = torch.randn(16000)  # 1 second
            result = asr.transcribe(audio)

            # With phonemes
            result = asr.transcribe("speech.wav", return_phonemes=True)
            print(f"Phonemes: {result.phonemes}")
            ```
        """
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

        # Run inference - get phonemes
        phonemes, confidences = self.model.decode_greedy(audio)

        # Get first batch element
        phonemes_seq = phonemes[0]
        confidence_scores = confidences[0] if confidences else None

        # Convert phonemes to text
        if self.use_llm and self.llm_model:
            # Use LLM for reconstruction
            text = self.llm_model(
                [phonemes_seq],
                [confidence_scores] if confidence_scores else None,
                return_thinking=False
            )[0]
        else:
            # Simple join (no LLM)
            text = ' '.join(phonemes_seq)

        # Calculate overall confidence
        overall_confidence = None
        if return_confidence and confidence_scores is not None:
            overall_confidence = float(confidence_scores.mean().item())

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
        batch_size: int = 8,
        return_phonemes: bool = False,
        return_confidence: bool = True
    ) -> List[TranscriptionResult]:
        """
        Transcribe multiple audio files in batch.

        Args:
            audio_paths: List of audio file paths
            batch_size: Batch size for processing
            return_phonemes: Whether to return phoneme sequences
            return_confidence: Whether to return confidence scores

        Returns:
            List of TranscriptionResult objects

        Example:
            ```python
            files = ["audio1.wav", "audio2.wav", "audio3.wav"]
            results = asr.transcribe_batch(files, batch_size=4)

            for i, result in enumerate(results):
                print(f"File {i+1}: {result.text}")
                print(f"Confidence: {result.confidence:.2%}")
            ```
        """
        results = []

        # Process in batches
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]

            for audio_path in batch_paths:
                result = self.transcribe(
                    audio_path,
                    return_phonemes=return_phonemes,
                    return_confidence=return_confidence
                )
                results.append(result)

        return results

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ValerieASR(\n"
            f"  model={self.model_path.name},\n"
            f"  device={self.device},\n"
            f"  sample_rate={self.sample_rate},\n"
            f"  use_llm={self.use_llm}\n"
            f")"
        )


# Convenience function
def load_asr(
    model_path: Union[str, Path],
    device: str = "auto",
    use_llm: bool = False
) -> ValerieASR:
    """
    Convenience function to load Valerie ASR.

    Args:
        model_path: Path to model checkpoint
        device: Device to use ("auto", "cuda", or "cpu")
        use_llm: Whether to use LLM for text reconstruction

    Returns:
        Initialized ValerieASR instance

    Example:
        ```python
        from ValerieASR import load_asr

        asr = load_asr("checkpoints/valerie_final.safetensors")
        result = asr.transcribe("audio.wav")
        print(result.text)
        ```
    """
    return ValerieASR(model_path, device=device, use_llm=use_llm)


if __name__ == "__main__":
    import sys

    # Simple CLI interface
    if len(sys.argv) < 2:
        print("Usage: python ValerieASR.py <model_path> [audio_file]")
        print("Example: python ValerieASR.py checkpoints/valerie_final.safetensors audio.wav")
        sys.exit(1)

    model_path = sys.argv[1]

    # Load ASR
    print(f"Loading Valerie ASR from {model_path}...")
    asr = load_asr(model_path)
    print(f"✅ Model loaded: {asr}\n")

    # Transcribe if audio file provided
    if len(sys.argv) >= 3:
        audio_file = sys.argv[2]
        print(f"Transcribing {audio_file}...")
        result = asr.transcribe(audio_file, return_phonemes=True)
        print("\n" + "="*60)
        print(result)
        print("="*60)
    else:
        print("Ready for transcription. Call asr.transcribe('audio.wav')")
