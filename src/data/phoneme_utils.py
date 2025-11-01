"""
Phoneme processing utilities for Valerie Visual ASR.

Handles text-to-phoneme conversion using proper libraries (g2p_en, epitran, phonemizer),
forced alignment, and CTC label generation for training visual speech recognition models.
"""

import re
import string
from typing import List, Dict, Tuple, Optional, Union
import torch
import numpy as np
from pathlib import Path
import json
import subprocess
import tempfile
import logging
from dataclasses import dataclass
import librosa
import soundfile as sf

# Import proper phoneme processing libraries
try:
    from g2p_en import G2p
    G2P_AVAILABLE = True
except ImportError:
    G2P_AVAILABLE = False

try:
    import epitran
    EPITRAN_AVAILABLE = True
except ImportError:
    EPITRAN_AVAILABLE = False

try:
    from phonemizer import phonemize
    from phonemizer.backend import EspeakBackend
    PHONEMIZER_AVAILABLE = True
except ImportError:
    PHONEMIZER_AVAILABLE = False

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ARPAbet phoneme set (standard for English ASR)
ARPABET_PHONEMES = [
    # Vowels (monophthongs)
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW',
    # Consonants
    'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH'
]

# Add special tokens
SPECIAL_TOKENS = ['<blank>', '<unk>', '<sil>']
PHONEME_VOCAB = SPECIAL_TOKENS + ARPABET_PHONEMES

# Create phoneme-to-index mapping
PHONEME_TO_IDX = {phoneme: idx for idx, phoneme in enumerate(PHONEME_VOCAB)}
IDX_TO_PHONEME = {idx: phoneme for phoneme, idx in PHONEME_TO_IDX.items()}

# Viseme groups (phonemes that look similar on lips)
VISEME_GROUPS = {
    'silence': ['<blank>', 'SIL'],
    'p_b_m': ['P', 'B', 'M'],
    'f_v': ['F', 'V'],
    'th_dh': ['TH', 'DH'],
    't_d_n_l': ['T', 'D', 'N', 'L'],
    's_z': ['S', 'Z'],
    'sh_zh_ch_jh': ['SH', 'ZH', 'CH', 'JH'],
    'k_g_ng': ['K', 'G', 'NG'],
    'r': ['R'],
    'w_ow_uw': ['W', 'OW', 'UW'],
    'y_iy_ih': ['Y', 'IY', 'IH'],
    'vowels_1': ['AA', 'AO'],
    'vowels_2': ['AE', 'AH', 'EH'],
    'vowels_3': ['AW', 'AY', 'EY', 'OY'],
    'vowels_4': ['ER', 'UH']
}


@dataclass
class PhonemeAlignment:
    """Phoneme alignment information."""
    phoneme: str
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    confidence: float = 1.0


@dataclass
class PhonemeSequence:
    """Complete phoneme sequence with alignment."""
    text: str
    phonemes: List[str]
    alignments: List[PhonemeAlignment]
    duration: float
    frame_rate: float = 25.0


class PhonemeConverter:
    """Professional phoneme converter using multiple libraries for robustness."""
    
    def __init__(self, language: str = "en-us"):
        """
        Initialize phoneme converter with multiple backends.
        
        Args:
            language: Language code (default: "en-us" for English)
        """
        self.language = language
        self.g2p = None
        self.epitran = None
        self.espeak_backend = None
        
        # Initialize available backends
        self._init_backends()
        
        # IPA to ARPAbet mapping for consistency
        self.ipa_to_arpabet = self._create_ipa_to_arpabet_mapping()
        
        logger.info(f"✅ PhonemeConverter initialized:")
        logger.info(f"   Language: {language}")
        logger.info(f"   G2P available: {self.g2p is not None}")
        logger.info(f"   Epitran available: {self.epitran is not None}")
        logger.info(f"   Phonemizer available: {self.espeak_backend is not None}")
    
    def _init_backends(self):
        """Initialize available phoneme conversion backends."""
        
        # Initialize G2P (Grapheme-to-Phoneme)
        if G2P_AVAILABLE:
            try:
                self.g2p = G2p()
                logger.info("✅ G2P backend initialized")
            except Exception as e:
                logger.warning(f"⚠️ G2P initialization failed: {e}")
        
        # Initialize Epitran
        if EPITRAN_AVAILABLE:
            try:
                if self.language.startswith("en"):
                    self.epitran = epitran.Epitran("eng-Latn")
                else:
                    # For other languages, you'd specify appropriate codes
                    self.epitran = epitran.Epitran("eng-Latn")
                logger.info("✅ Epitran backend initialized")
            except Exception as e:
                logger.warning(f"⚠️ Epitran initialization failed: {e}")
        
        # Initialize Phonemizer with eSpeak
        if PHONEMIZER_AVAILABLE:
            try:
                self.espeak_backend = EspeakBackend(
                    language=self.language,
                    punctuation_marks=';:,.!?¡¿—…"«»""',
                    preserve_punctuation=False,
                    with_stress=True
                )
                logger.info("✅ Phonemizer eSpeak backend initialized")
            except Exception as e:
                logger.warning(f"⚠️ Phonemizer initialization failed: {e}")
        
        # Check if at least one backend is available
        if not any([self.g2p, self.epitran, self.espeak_backend]):
            logger.error("❌ No phoneme conversion backends available!")
            raise RuntimeError("No phoneme conversion libraries available. Please install g2p_en, epitran, or phonemizer.")
    
    def _create_ipa_to_arpabet_mapping(self) -> Dict[str, str]:
        """Create mapping from IPA to ARPAbet phonemes."""
        return {
            # Vowels
            'i': 'IY', 'ɪ': 'IH', 'e': 'EY', 'ɛ': 'EH', 'æ': 'AE',
            'ɑ': 'AA', 'ɔ': 'AO', 'o': 'OW', 'ʊ': 'UH', 'u': 'UW',
            'ʌ': 'AH', 'ə': 'AH', 'ɚ': 'ER', 'ɝ': 'ER',
            'aɪ': 'AY', 'aʊ': 'AW', 'ɔɪ': 'OY',
            
            # Consonants
            'p': 'P', 'b': 'B', 't': 'T', 'd': 'D', 'k': 'K', 'g': 'G',
            'f': 'F', 'v': 'V', 'θ': 'TH', 'ð': 'DH', 's': 'S', 'z': 'Z',
            'ʃ': 'SH', 'ʒ': 'ZH', 'h': 'HH', 'm': 'M', 'n': 'N', 'ŋ': 'NG',
            'l': 'L', 'r': 'R', 'w': 'W', 'j': 'Y', 'tʃ': 'CH', 'dʒ': 'JH',
            
            # Stress markers (ignore)
            'ˈ': '', 'ˌ': '', '1': '', '2': '', '0': '',
            
            # Common variants
            'ɹ': 'R', 'ɾ': 'T', 'ʔ': 'T'
        }
    
    def word_to_phonemes(self, word: str) -> List[str]:
        """
        Convert word to ARPAbet phonemes using available backends.
        
        Args:
            word: Input word
            
        Returns:
            List of ARPAbet phonemes
        """
        word = word.strip().upper()
        if not word:
            return []
        
        # Try G2P first (most accurate for English)
        if self.g2p:
            try:
                # G2P expects lowercase input
                phonemes = self.g2p(word.lower())
                
                # G2P returns list of phonemes, clean them up
                clean_phonemes = []
                for p in phonemes:
                    # Remove stress markers and clean up
                    clean_p = p.strip('0123456789').upper()
                    
                    # Handle common G2P variations
                    if clean_p == 'AX':  # G2P uses AX, we use AH
                        clean_p = 'AH'
                    elif clean_p == 'IX':  # G2P uses IX, we use IH
                        clean_p = 'IH'
                    
                    if clean_p in ARPABET_PHONEMES:
                        clean_phonemes.append(clean_p)
                    elif clean_p:  # Non-empty but unknown phoneme
                        logger.debug(f"Unknown G2P phoneme '{clean_p}' for word '{word}'")
                
                if clean_phonemes:
                    return clean_phonemes
                else:
                    logger.debug(f"G2P returned no valid phonemes for '{word}': {phonemes}")
                    
            except Exception as e:
                logger.debug(f"G2P failed for '{word}': {e}")
        
        # Try Epitran (IPA output, need to convert)
        if self.epitran:
            try:
                ipa_phonemes = self.epitran.transliterate(word.lower())
                arpabet_phonemes = self._ipa_to_arpabet(ipa_phonemes)
                if arpabet_phonemes:
                    return arpabet_phonemes
            except Exception as e:
                logger.debug(f"Epitran failed for '{word}': {e}")
        
        # Try Phonemizer as fallback
        if self.espeak_backend:
            try:
                ipa_phonemes = self.espeak_backend.phonemize([word.lower()], strip=True)[0]
                arpabet_phonemes = self._ipa_to_arpabet(ipa_phonemes)
                if arpabet_phonemes:
                    return arpabet_phonemes
            except Exception as e:
                logger.debug(f"Phonemizer failed for '{word}': {e}")
        
        # Final fallback: return unknown token
        logger.warning(f"⚠️ Could not phonemize word: {word}")
        return ['<unk>']
    
    def _ipa_to_arpabet(self, ipa_string: str) -> List[str]:
        """Convert IPA string to ARPAbet phonemes."""
        if not ipa_string:
            return []
        
        # Clean up IPA string
        ipa_string = ipa_string.strip()
        
        # Simple conversion approach
        arpabet_phonemes = []
        i = 0
        while i < len(ipa_string):
            # Try two-character combinations first
            if i + 1 < len(ipa_string):
                two_char = ipa_string[i:i+2]
                if two_char in self.ipa_to_arpabet:
                    mapped = self.ipa_to_arpabet[two_char]
                    if mapped:  # Skip empty mappings
                        arpabet_phonemes.append(mapped)
                    i += 2
                    continue
            
            # Try single character
            single_char = ipa_string[i]
            if single_char in self.ipa_to_arpabet:
                mapped = self.ipa_to_arpabet[single_char]
                if mapped:  # Skip empty mappings
                    arpabet_phonemes.append(mapped)
            elif single_char.isalpha():
                # Unknown phoneme, skip with warning
                logger.debug(f"Unknown IPA phoneme: {single_char}")
            
            i += 1
        
        # Filter to only valid ARPAbet phonemes
        valid_phonemes = [p for p in arpabet_phonemes if p in ARPABET_PHONEMES]
        
        return valid_phonemes if valid_phonemes else ['<unk>']


class TextToPhonemeConverter:
    """Convert text to phoneme sequences using professional libraries."""
    
    def __init__(self, language: str = "en-us"):
        """
        Initialize converter with professional phoneme converter.
        
        Args:
            language: Language code for phoneme conversion
        """
        self.phoneme_converter = PhonemeConverter(language)
        self.language = language
        
        logger.info(f"✅ TextToPhonemeConverter initialized for {language}")
    
    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to phoneme sequence.
        
        Args:
            text: Input text
            
        Returns:
            List of ARPAbet phonemes
        """
        if not text or not text.strip():
            return []
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split into words
        words = text.split()
        
        # Convert each word to phonemes
        all_phonemes = []
        for word in words:
            if word:  # Skip empty words
                phonemes = self.phoneme_converter.word_to_phonemes(word)
                all_phonemes.extend(phonemes)
                
                # Add word boundary for multi-word utterances (optional)
                # This can help with alignment in some cases
                if len(words) > 1:
                    all_phonemes.append('<sil>')
        
        # Remove trailing silence if added
        if all_phonemes and all_phonemes[-1] == '<sil>':
            all_phonemes.pop()
        
        return all_phonemes
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Convert to uppercase
        text = text.upper()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove punctuation except apostrophes
        text = re.sub(r"[^\w\s']", '', text)
        
        # Handle contractions (basic)
        contractions = {
            "WON'T": "WILL NOT",
            "CAN'T": "CAN NOT",
            "N'T": " NOT",
            "'RE": " ARE",
            "'VE": " HAVE",
            "'LL": " WILL",
            "'D": " WOULD",
            "'M": " AM"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text


class ForcedAligner:
    """Forced alignment using Montreal Forced Alignment (MFA)."""
    
    def __init__(
        self,
        mfa_path: Optional[str] = None,
        acoustic_model: str = "english_us_arpa",
        dictionary_path: Optional[str] = None
    ):
        """
        Initialize MFA-based forced aligner.
        
        Args:
            mfa_path: Path to MFA binary
            acoustic_model: Acoustic model name
            dictionary_path: Path to pronunciation dictionary
        """
        self.mfa_path = mfa_path or "mfa"
        self.acoustic_model = acoustic_model
        self.dictionary_path = dictionary_path
        
        # Check if MFA is available
        self.available = self._check_mfa_availability()
        
        if self.available:
            logger.info(f"✅ MFA ForcedAligner initialized")
        else:
            logger.warning("⚠️ MFA not available, using uniform alignment")
    
    def _check_mfa_availability(self) -> bool:
        """Check if MFA is available."""
        try:
            result = subprocess.run([self.mfa_path, "--version"], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def align(
        self,
        audio_path: str,
        text: str,
        frame_rate: float = 25.0
    ) -> List[PhonemeAlignment]:
        """
        Perform forced alignment.
        
        Args:
            audio_path: Path to audio file
            text: Text transcript
            frame_rate: Video frame rate for frame indices
            
        Returns:
            List of phoneme alignments
        """
        if not self.available:
            return self._uniform_alignment(audio_path, text, frame_rate)
        
        try:
            return self._mfa_align(audio_path, text, frame_rate)
        except Exception as e:
            logger.warning(f"⚠️ MFA alignment failed: {e}, using uniform alignment")
            return self._uniform_alignment(audio_path, text, frame_rate)
    
    def _mfa_align(
        self,
        audio_path: str,
        text: str,
        frame_rate: float
    ) -> List[PhonemeAlignment]:
        """Perform MFA alignment."""
        # Create temporary directory for MFA
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # Copy audio file
            audio_file = temp_dir / "audio.wav"
            subprocess.run(["cp", audio_path, str(audio_file)], check=True)
            
            # Create text file
            text_file = temp_dir / "audio.txt"
            with open(text_file, 'w') as f:
                f.write(text)
            
            # Run MFA alignment
            output_dir = temp_dir / "output"
            output_dir.mkdir()
            
            cmd = [
                self.mfa_path, "align",
                str(temp_dir), self.dictionary_path or "english_us_arpa",
                self.acoustic_model, str(output_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"MFA failed: {result.stderr}")
            
            # Parse alignment results
            alignment_file = output_dir / "audio.TextGrid"
            if alignment_file.exists():
                return self._parse_textgrid(alignment_file, frame_rate)
            else:
                raise Exception("No alignment file generated")
    
    def _parse_textgrid(self, textgrid_path: Path, frame_rate: float) -> List[PhonemeAlignment]:
        """Parse TextGrid file from MFA."""
        # This is a simplified parser - you might want to use a proper TextGrid library
        alignments = []
        
        try:
            with open(textgrid_path, 'r') as f:
                content = f.read()
            
            # Extract phone tier (simplified parsing)
            phone_tier_start = content.find('name = "phones"')
            if phone_tier_start == -1:
                raise Exception("No phone tier found")
            
            # Parse intervals (very basic)
            intervals = re.findall(
                r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"',
                content[phone_tier_start:]
            )
            
            for xmin, xmax, phoneme in intervals:
                if phoneme and phoneme != 'sil' and phoneme != 'sp':
                    start_time = float(xmin)
                    end_time = float(xmax)
                    
                    alignment = PhonemeAlignment(
                        phoneme=phoneme.upper(),
                        start_time=start_time,
                        end_time=end_time,
                        start_frame=int(start_time * frame_rate),
                        end_frame=int(end_time * frame_rate)
                    )
                    alignments.append(alignment)
            
            return alignments
            
        except Exception as e:
            logger.warning(f"⚠️ TextGrid parsing failed: {e}")
            return []
    
    def _uniform_alignment(
        self,
        audio_path: str,
        text: str,
        frame_rate: float
    ) -> List[PhonemeAlignment]:
        """Create uniform alignment when MFA is not available."""
        # Get audio duration
        try:
            audio, sr = librosa.load(audio_path)
            duration = len(audio) / sr
        except:
            duration = 3.0  # Fallback duration
        
        # Convert text to phonemes
        converter = TextToPhonemeConverter()
        phonemes = converter.text_to_phonemes(text)
        
        if not phonemes:
            return []
        
        # Create uniform timing
        phoneme_duration = duration / len(phonemes)
        alignments = []
        
        for i, phoneme in enumerate(phonemes):
            start_time = i * phoneme_duration
            end_time = (i + 1) * phoneme_duration
            
            alignment = PhonemeAlignment(
                phoneme=phoneme,
                start_time=start_time,
                end_time=end_time,
                start_frame=int(start_time * frame_rate),
                end_frame=int(end_time * frame_rate),
                confidence=0.5  # Lower confidence for uniform alignment
            )
            alignments.append(alignment)
        
        return alignments


class CTCLabelGenerator:
    """Generate CTC labels from phoneme sequences."""
    
    def __init__(self, blank_token: str = '<blank>', unk_token: str = '<unk>'):
        """Initialize CTC label generator."""
        self.blank_token = blank_token
        self.unk_token = unk_token
        self.blank_idx = PHONEME_TO_IDX[blank_token]
        self.unk_idx = PHONEME_TO_IDX[unk_token]
        
        logger.info(f"✅ CTCLabelGenerator initialized:")
        logger.info(f"   Vocabulary size: {len(PHONEME_VOCAB)}")
        logger.info(f"   Blank token: {blank_token} (idx: {self.blank_idx})")
        logger.info(f"   Unknown token: {unk_token} (idx: {self.unk_idx})")
    
    def phonemes_to_ctc_labels(self, phonemes: List[str]) -> torch.Tensor:
        """
        Convert phoneme sequence to CTC labels.
        
        Args:
            phonemes: List of ARPAbet phonemes
            
        Returns:
            CTC label tensor
        """
        if not phonemes:
            return torch.tensor([self.blank_idx], dtype=torch.long)
        
        # Convert phonemes to indices
        indices = []
        for phoneme in phonemes:
            if phoneme in PHONEME_TO_IDX:
                indices.append(PHONEME_TO_IDX[phoneme])
            else:
                logger.debug(f"⚠️ Unknown phoneme '{phoneme}', using <unk>")
                indices.append(self.unk_idx)
        
        return torch.tensor(indices, dtype=torch.long)
    
    def generate_labels(self, phonemes: List[str]) -> torch.Tensor:
        """
        Generate CTC labels from phoneme sequence (alias for phonemes_to_ctc_labels).
        
        Args:
            phonemes: List of ARPAbet phonemes
            
        Returns:
            CTC label tensor
        """
        return self.phonemes_to_ctc_labels(phonemes)
    
    def alignment_to_ctc_sequence(
        self,
        alignments: List[PhonemeAlignment],
        total_frames: int
    ) -> torch.Tensor:
        """
        Convert phoneme alignments to frame-level CTC sequence.
        
        Args:
            alignments: List of phoneme alignments
            total_frames: Total number of frames
            
        Returns:
            Frame-level CTC sequence
        """
        # Initialize with blank tokens
        ctc_sequence = [self.blank_idx] * total_frames
        
        # Fill in phonemes based on alignment
        for alignment in alignments:
            phoneme_idx = PHONEME_TO_IDX.get(alignment.phoneme, self.blank_idx)
            
            start_frame = max(0, alignment.start_frame)
            end_frame = min(total_frames, alignment.end_frame)
            
            for frame_idx in range(start_frame, end_frame):
                ctc_sequence[frame_idx] = phoneme_idx
        
        return torch.tensor(ctc_sequence, dtype=torch.long)


# Factory functions
def create_phoneme_processor(config) -> Tuple[TextToPhonemeConverter, ForcedAligner, CTCLabelGenerator]:
    """Create phoneme processing components from configuration."""
    
    converter = TextToPhonemeConverter()
    
    aligner = ForcedAligner(
        acoustic_model="english_us_arpa"
    )
    
    ctc_generator = CTCLabelGenerator()
    
    return converter, aligner, ctc_generator


def process_text_to_ctc_labels(
    text: str,
    audio_path: Optional[str] = None,
    total_frames: Optional[int] = None,
    frame_rate: float = 25.0
) -> Tuple[List[str], torch.Tensor]:
    """
    Complete pipeline: text -> phonemes -> CTC labels.
    
    Args:
        text: Input text
        audio_path: Path to audio file (for alignment)
        total_frames: Total number of frames (for frame-level labels)
        frame_rate: Video frame rate
        
    Returns:
        Tuple of (phonemes, ctc_labels)
    """
    converter = TextToPhonemeConverter()
    ctc_generator = CTCLabelGenerator()
    
    # Convert text to phonemes
    phonemes = converter.text_to_phonemes(text)
    
    if total_frames and audio_path:
        # Create frame-level alignment
        aligner = ForcedAligner()
        alignments = aligner.align(audio_path, text, frame_rate)
        ctc_labels = ctc_generator.alignment_to_ctc_sequence(alignments, total_frames)
    else:
        # Create sequence-level labels
        ctc_labels = ctc_generator.phonemes_to_ctc_labels(phonemes)
    
    return phonemes, ctc_labels


if __name__ == "__main__":
    # Test phoneme processing
    print("Testing phoneme processing...")
    
    # Test text-to-phoneme conversion
    converter = TextToPhonemeConverter()
    text = "Hello world, how are you today?"
    phonemes = converter.text_to_phonemes(text)
    print(f"Text: {text}")
    print(f"Phonemes: {phonemes}")
    
    # Test CTC label generation
    ctc_generator = CTCLabelGenerator()
    ctc_labels = ctc_generator.phonemes_to_ctc_labels(phonemes)
    print(f"CTC labels: {ctc_labels}")
    
    # Test complete pipeline
    phonemes, labels = process_text_to_ctc_labels(text)
    print(f"Pipeline result: {len(phonemes)} phonemes, {len(labels)} labels")
    
    print("\n✅ Phoneme processing tests completed!")
