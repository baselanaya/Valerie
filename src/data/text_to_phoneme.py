"""
Text-to-Phoneme Conversion Utilities.

This module provides utilities for converting text to phoneme sequences
and vice versa, which is essential for Stage 2 training (Phonemes → Text).

Features:
- G2P (Grapheme-to-Phoneme) conversion using multiple backends
- Phoneme-to-text mapping
- Synthetic error injection for robustness
- Support for ARPAbet phoneme set
"""

import torch
import re
import unicodedata
from typing import List, Tuple, Dict, Optional
import numpy as np
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ARPAbet phoneme set (39 phonemes + blank)
# Based on CMU Pronouncing Dictionary
ARPABET_PHONEMES = [
    '<blank>',  # 0: Blank token for CTC
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'SIL'
]

# Create phoneme to index mapping
PHONEME_TO_IDX = {p: i for i, p in enumerate(ARPABET_PHONEMES)}
IDX_TO_PHONEME = {i: p for i, p in enumerate(ARPABET_PHONEMES)}


class TextToPhonemeConverter:
    """
    Convert text to phoneme sequences using various G2P methods.

    Supports multiple backends:
    - g2p_en: English G2P library
    - phonemizer: Multi-language phonemizer
    - epitran: IPA-based phonemizer
    """

    def __init__(
        self,
        backend: str = "g2p_en",
        language: str = "en-us",
        use_stress: bool = False
    ):
        """
        Initialize text-to-phoneme converter.

        Args:
            backend: G2P backend ("g2p_en", "phonemizer", "epitran")
            language: Language code
            use_stress: Whether to include stress markers
        """
        self.backend = backend
        self.language = language
        self.use_stress = use_stress

        logger.info(f"📝 Initializing TextToPhonemeConverter:")
        logger.info(f"   Backend: {backend}")
        logger.info(f"   Language: {language}")
        logger.info(f"   Use stress: {use_stress}")

        # Initialize backend
        if backend == "g2p_en":
            try:
                from g2p_en import G2p
                self.g2p = G2p()
                logger.info("✅ g2p_en backend loaded")
            except ImportError:
                logger.error("❌ g2p_en not installed. Install with: pip install g2p-en")
                raise

        elif backend == "phonemizer":
            try:
                from phonemizer import phonemize
                from phonemizer.backend import EspeakBackend
                self.phonemizer = phonemize
                self.espeak_backend = EspeakBackend(
                    language=language,
                    preserve_punctuation=False,
                    with_stress=use_stress
                )
                logger.info("✅ phonemizer backend loaded")
            except ImportError:
                logger.error("❌ phonemizer not installed. Install with: pip install phonemizer")
                raise

        elif backend == "epitran":
            try:
                import epitran
                self.epitran = epitran.Epitran(language)
                logger.info("✅ epitran backend loaded")
            except ImportError:
                logger.error("❌ epitran not installed. Install with: pip install epitran")
                raise

        else:
            raise ValueError(f"Unknown backend: {backend}")

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for phoneme conversion.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove accents
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('utf-8')

        # Remove punctuation (keep spaces)
        text = re.sub(r'[^\w\s]', '', text)

        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to phoneme sequence.

        Args:
            text: Input text

        Returns:
            List of phonemes
        """
        # Normalize text
        text = self.normalize_text(text)

        if not text:
            return []

        phonemes = []

        if self.backend == "g2p_en":
            # g2p_en returns list of phonemes
            phonemes = self.g2p(text)

            # Remove stress markers if not needed
            if not self.use_stress:
                phonemes = [p.rstrip('012') for p in phonemes if p.isalpha() or p.rstrip('012').isalpha()]

        elif self.backend == "phonemizer":
            # phonemizer returns string of phonemes
            phoneme_str = self.phonemizer(
                text,
                backend='espeak',
                language=self.language,
                preserve_punctuation=False,
                with_stress=self.use_stress
            )
            phonemes = phoneme_str.split()

        elif self.backend == "epitran":
            # epitran returns IPA string
            ipa = self.epitran.transliterate(text)
            # Convert IPA to phoneme list (simplified)
            phonemes = list(ipa.replace(' ', ''))

        # Filter to valid ARPAbet phonemes
        valid_phonemes = []
        for p in phonemes:
            p_clean = p.upper().rstrip('012')
            if p_clean in PHONEME_TO_IDX:
                valid_phonemes.append(p_clean)

        return valid_phonemes

    def text_to_indices(self, text: str) -> List[int]:
        """
        Convert text to phoneme indices.

        Args:
            text: Input text

        Returns:
            List of phoneme indices
        """
        phonemes = self.text_to_phonemes(text)
        indices = [PHONEME_TO_IDX.get(p, PHONEME_TO_IDX['<blank>']) for p in phonemes]
        return indices

    def batch_text_to_phonemes(self, texts: List[str]) -> List[List[str]]:
        """
        Convert batch of texts to phoneme sequences.

        Args:
            texts: List of input texts

        Returns:
            List of phoneme sequences
        """
        return [self.text_to_phonemes(text) for text in texts]


class PhonemeToTextConverter:
    """
    Convert phoneme sequences back to text.

    This is used for evaluation and as input for the LLM in Stage 2.
    """

    def __init__(self, use_separator: bool = True):
        """
        Initialize phoneme-to-text converter.

        Args:
            use_separator: Whether to use separator between phonemes
        """
        self.use_separator = use_separator

    def phonemes_to_text(
        self,
        phonemes: List[str],
        separator: str = " "
    ) -> str:
        """
        Convert phoneme sequence to text representation.

        Args:
            phonemes: List of phonemes
            separator: Separator between phonemes

        Returns:
            Text representation of phonemes
        """
        if self.use_separator:
            return separator.join(phonemes)
        else:
            return "".join(phonemes)

    def indices_to_text(
        self,
        indices: List[int],
        separator: str = " "
    ) -> str:
        """
        Convert phoneme indices to text.

        Args:
            indices: List of phoneme indices
            separator: Separator between phonemes

        Returns:
            Text representation
        """
        phonemes = [IDX_TO_PHONEME.get(idx, '<unk>') for idx in indices]
        # Remove blank tokens
        phonemes = [p for p in phonemes if p != '<blank>']
        return self.phonemes_to_text(phonemes, separator)


class SyntheticErrorInjector:
    """
    Inject synthetic errors into phoneme sequences for robust training.

    This makes the LLM more robust to errors from the phoneme ASR model.
    """

    def __init__(
        self,
        substitution_prob: float = 0.05,
        deletion_prob: float = 0.05,
        insertion_prob: float = 0.05,
        max_error_rate: float = 0.15
    ):
        """
        Initialize error injector.

        Args:
            substitution_prob: Probability of phoneme substitution
            deletion_prob: Probability of phoneme deletion
            insertion_prob: Probability of phoneme insertion
            max_error_rate: Maximum overall error rate
        """
        self.substitution_prob = substitution_prob
        self.deletion_prob = deletion_prob
        self.insertion_prob = insertion_prob
        self.max_error_rate = max_error_rate

        logger.debug(f"💉 SyntheticErrorInjector initialized:")
        logger.debug(f"   Substitution: {substitution_prob}")
        logger.debug(f"   Deletion: {deletion_prob}")
        logger.debug(f"   Insertion: {insertion_prob}")

    def inject_errors(
        self,
        phonemes: List[str],
        target_error_rate: Optional[float] = None
    ) -> List[str]:
        """
        Inject synthetic errors into phoneme sequence.

        Args:
            phonemes: Original phoneme sequence
            target_error_rate: Target error rate (overrides default probs)

        Returns:
            Phoneme sequence with errors
        """
        if not phonemes:
            return phonemes

        # Adjust probabilities based on target error rate
        if target_error_rate is not None:
            scale = target_error_rate / (
                self.substitution_prob + self.deletion_prob + self.insertion_prob
            )
            sub_prob = self.substitution_prob * scale
            del_prob = self.deletion_prob * scale
            ins_prob = self.insertion_prob * scale
        else:
            sub_prob = self.substitution_prob
            del_prob = self.deletion_prob
            ins_prob = self.insertion_prob

        noisy_phonemes = []
        error_count = 0

        for i, phoneme in enumerate(phonemes):
            # Check if we've exceeded max error rate
            if error_count / len(phonemes) >= self.max_error_rate:
                noisy_phonemes.append(phoneme)
                continue

            rand = np.random.random()

            # Deletion
            if rand < del_prob:
                error_count += 1
                continue

            # Substitution
            elif rand < del_prob + sub_prob:
                # Replace with random phoneme
                random_phoneme = np.random.choice([
                    p for p in ARPABET_PHONEMES[1:-1]  # Exclude blank and SIL
                    if p != phoneme
                ])
                noisy_phonemes.append(random_phoneme)
                error_count += 1

            # Insertion (before current phoneme)
            elif rand < del_prob + sub_prob + ins_prob:
                random_phoneme = np.random.choice(ARPABET_PHONEMES[1:-1])
                noisy_phonemes.append(random_phoneme)
                noisy_phonemes.append(phoneme)
                error_count += 1

            # No error
            else:
                noisy_phonemes.append(phoneme)

        return noisy_phonemes


class PhonemeDatasetBuilder:
    """
    Build phoneme dataset from text corpus for Stage 2 training.

    Converts text to phonemes with synthetic errors for LLM training.
    """

    def __init__(
        self,
        text_to_phoneme_converter: TextToPhonemeConverter,
        error_injector: Optional[SyntheticErrorInjector] = None
    ):
        """
        Initialize dataset builder.

        Args:
            text_to_phoneme_converter: Text-to-phoneme converter
            error_injector: Error injector (optional)
        """
        self.converter = text_to_phoneme_converter
        self.error_injector = error_injector

    def process_text(
        self,
        text: str,
        inject_errors: bool = True
    ) -> Tuple[List[str], List[str], str]:
        """
        Process text into phoneme training example.

        Args:
            text: Input text
            inject_errors: Whether to inject synthetic errors

        Returns:
            Tuple of (clean_phonemes, noisy_phonemes, original_text)
        """
        # Convert text to phonemes
        clean_phonemes = self.converter.text_to_phonemes(text)

        # Inject errors if requested
        if inject_errors and self.error_injector is not None:
            noisy_phonemes = self.error_injector.inject_errors(clean_phonemes)
        else:
            noisy_phonemes = clean_phonemes

        return clean_phonemes, noisy_phonemes, text

    def build_dataset(
        self,
        texts: List[str],
        inject_errors: bool = True
    ) -> List[Dict[str, any]]:
        """
        Build dataset from list of texts.

        Args:
            texts: List of input texts
            inject_errors: Whether to inject errors

        Returns:
            List of training examples
        """
        dataset = []

        for text in texts:
            clean_phonemes, noisy_phonemes, original_text = self.process_text(
                text, inject_errors
            )

            if clean_phonemes:  # Skip empty sequences
                dataset.append({
                    'text': original_text,
                    'clean_phonemes': clean_phonemes,
                    'noisy_phonemes': noisy_phonemes,
                    'clean_phoneme_text': ' '.join(clean_phonemes),
                    'noisy_phoneme_text': ' '.join(noisy_phonemes)
                })

        logger.info(f"📊 Built dataset with {len(dataset)} examples")
        return dataset


# Utility functions
def create_text_to_phoneme_pipeline(
    backend: str = "g2p_en",
    inject_errors: bool = True,
    error_config: Optional[Dict] = None
) -> PhonemeDatasetBuilder:
    """
    Create complete text-to-phoneme processing pipeline.

    Args:
        backend: G2P backend
        inject_errors: Whether to inject errors
        error_config: Error injection configuration

    Returns:
        PhonemeDatasetBuilder instance
    """
    # Create converter
    converter = TextToPhonemeConverter(backend=backend)

    # Create error injector if needed
    error_injector = None
    if inject_errors:
        if error_config is None:
            error_config = {
                'substitution_prob': 0.05,
                'deletion_prob': 0.05,
                'insertion_prob': 0.05
            }
        error_injector = SyntheticErrorInjector(**error_config)

    # Create dataset builder
    builder = PhonemeDatasetBuilder(converter, error_injector)

    return builder


# Unit tests
if __name__ == "__main__":
    print("🧪 Testing Text-to-Phoneme Conversion...")

    # Test TextToPhonemeConverter
    print("\n📝 Testing TextToPhonemeConverter...")
    try:
        converter = TextToPhonemeConverter(backend="g2p_en")

        test_texts = [
            "Hello world",
            "This is a test",
            "Speech recognition"
        ]

        for text in test_texts:
            phonemes = converter.text_to_phonemes(text)
            indices = converter.text_to_indices(text)
            print(f"✅ '{text}'")
            print(f"   Phonemes: {' '.join(phonemes)}")
            print(f"   Indices: {indices[:10]}...")

    except ImportError:
        print("⚠️ g2p_en not installed, skipping test")

    # Test PhonemeToTextConverter
    print("\n📝 Testing PhonemeToTextConverter...")
    p2t = PhonemeToTextConverter()

    phonemes = ['HH', 'EH', 'L', 'OW']
    text = p2t.phonemes_to_text(phonemes)
    print(f"✅ Phonemes to text: {phonemes} -> '{text}'")

    # Test SyntheticErrorInjector
    print("\n💉 Testing SyntheticErrorInjector...")
    error_injector = SyntheticErrorInjector(
        substitution_prob=0.1,
        deletion_prob=0.1,
        insertion_prob=0.1
    )

    clean_phonemes = ['HH', 'EH', 'L', 'OW', 'W', 'ER', 'L', 'D']
    noisy_phonemes = error_injector.inject_errors(clean_phonemes)

    print(f"✅ Clean: {' '.join(clean_phonemes)}")
    print(f"   Noisy: {' '.join(noisy_phonemes)}")

    # Test PhonemeDatasetBuilder
    print("\n📊 Testing PhonemeDatasetBuilder...")
    try:
        builder = create_text_to_phoneme_pipeline(inject_errors=True)

        test_texts = [
            "Hello world",
            "This is a test"
        ]

        dataset = builder.build_dataset(test_texts)

        for i, example in enumerate(dataset):
            print(f"\n✅ Example {i+1}:")
            print(f"   Text: {example['text']}")
            print(f"   Clean: {example['clean_phoneme_text']}")
            print(f"   Noisy: {example['noisy_phoneme_text']}")

    except ImportError:
        print("⚠️ g2p_en not installed, skipping dataset builder test")

    print("\n🎉 All text-to-phoneme tests completed!")
