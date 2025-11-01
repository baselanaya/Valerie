#!/usr/bin/env python3
"""
Convert transcriptions to phonemes for VALLR-style training.

This script converts text transcriptions to phoneme sequences using:
- G2P (Grapheme-to-Phoneme) conversion
- Multiple phoneme representations (IPA, ARPABET, etc.)
- Phoneme alignment with video frames

Usage:
    python scripts/convert_to_phonemes.py --transcription_dir data/transcriptions --output_dir data/phonemes
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import torch
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhonemeConverter:
    """Convert text transcriptions to phoneme sequences."""
    
    def __init__(self, phoneme_set: str = "arpabet"):
        """
        Initialize phoneme converter.
        
        Args:
            phoneme_set: Type of phoneme representation ("arpabet", "ipa", "custom")
        """
        self.phoneme_set = phoneme_set
        
        # Try to import G2P libraries
        self.g2p_model = None
        self.use_espeak = False
        
        try:
            # Try epitran for IPA phonemes
            if phoneme_set == "ipa":
                import epitran
                self.g2p_model = epitran.Epitran('eng-Latn')
                logger.info("✅ Using epitran for IPA phonemes")
        except ImportError:
            logger.warning("⚠️  epitran not available for IPA phonemes")
        
        try:
            # Try eSpeak for phoneme conversion
            import subprocess
            result = subprocess.run(['espeak', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.use_espeak = True
                logger.info("✅ eSpeak available for phoneme conversion")
        except (ImportError, FileNotFoundError):
            logger.warning("⚠️  eSpeak not available")
        
        # Fallback: Use simple rule-based conversion
        if not self.g2p_model and not self.use_espeak:
            logger.info("📝 Using rule-based phoneme conversion (fallback)")
            self._init_rule_based_converter()
    
    def _init_rule_based_converter(self):
        """Initialize rule-based phoneme converter as fallback."""
        
        # Simple English phoneme mapping (ARPABET-like)
        self.phoneme_map = {
            # Vowels
            'A': 'AE', 'E': 'EH', 'I': 'IH', 'O': 'AO', 'U': 'UH',
            'AA': 'AA', 'AE': 'AE', 'AH': 'AH', 'AO': 'AO', 'AW': 'AW',
            'AY': 'AY', 'EH': 'EH', 'ER': 'ER', 'EY': 'EY', 'IH': 'IH',
            'IY': 'IY', 'OW': 'OW', 'OY': 'OY', 'UH': 'UH', 'UW': 'UW',
            
            # Consonants
            'B': 'B', 'C': 'K', 'D': 'D', 'F': 'F', 'G': 'G', 'H': 'HH',
            'J': 'JH', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'P': 'P',
            'Q': 'K', 'R': 'R', 'S': 'S', 'T': 'T', 'V': 'V', 'W': 'W',
            'X': 'K S', 'Y': 'Y', 'Z': 'Z',
            
            # Digraphs and special cases
            'CH': 'CH', 'SH': 'SH', 'TH': 'TH', 'WH': 'W', 'PH': 'F',
            'GH': 'F', 'CK': 'K', 'NG': 'NG', 'NK': 'NG K'
        }
        
        # Common word-to-phoneme mappings for better accuracy
        self.word_phonemes = {
            'THE': 'DH AH', 'AND': 'AE N D', 'A': 'AH', 'TO': 'T UW',
            'OF': 'AH V', 'IN': 'IH N', 'IS': 'IH Z', 'IT': 'IH T',
            'YOU': 'Y UW', 'THAT': 'DH AE T', 'HE': 'HH IY', 'WAS': 'W AH Z',
            'FOR': 'F AO R', 'ON': 'AA N', 'ARE': 'AA R', 'AS': 'AE Z',
            'WITH': 'W IH TH', 'HIS': 'HH IH Z', 'THEY': 'DH EY',
            'I': 'AY', 'AT': 'AE T', 'BE': 'B IY', 'THIS': 'DH IH S',
            'HAVE': 'HH AE V', 'FROM': 'F R AH M', 'OR': 'AO R',
            'ONE': 'W AH N', 'HAD': 'HH AE D', 'BY': 'B AY', 'WORD': 'W ER D',
            'BUT': 'B AH T', 'NOT': 'N AA T', 'WHAT': 'W AA T',
            'ALL': 'AO L', 'WERE': 'W ER', 'WE': 'W IY', 'WHEN': 'W EH N',
            'YOUR': 'Y AO R', 'CAN': 'K AE N', 'SAID': 'S EH D',
            'THERE': 'DH EH R', 'EACH': 'IY CH', 'WHICH': 'W IH CH',
            'DO': 'D UW', 'HOW': 'HH AW', 'THEIR': 'DH EH R',
            'IF': 'IH F', 'WILL': 'W IH L', 'UP': 'AH P', 'OTHER': 'AH DH ER',
            'ABOUT': 'AH B AW T', 'OUT': 'AW T', 'MANY': 'M EH N IY',
            'THEN': 'DH EH N', 'THEM': 'DH EH M', 'THESE': 'DH IY Z',
            'SO': 'S OW', 'SOME': 'S AH M', 'HER': 'HH ER', 'WOULD': 'W UH D',
            'MAKE': 'M EY K', 'LIKE': 'L AY K', 'INTO': 'IH N T UW',
            'HIM': 'HH IH M', 'HAS': 'HH AE Z', 'TWO': 'T UW',
            'MORE': 'M AO R', 'GO': 'G OW', 'NO': 'N OW', 'WAY': 'W EY',
            'COULD': 'K UH D', 'MY': 'M AY', 'THAN': 'DH AE N',
            'FIRST': 'F ER S T', 'BEEN': 'B IH N', 'CALL': 'K AO L',
            'WHO': 'HH UW', 'ITS': 'IH T S', 'NOW': 'N AW', 'FIND': 'F AY N D',
            'LONG': 'L AO NG', 'DOWN': 'D AW N', 'DAY': 'D EY',
            'GET': 'G EH T', 'COME': 'K AH M', 'MADE': 'M EY D',
            'MAY': 'M EY', 'PART': 'P AA R T'
        }
    
    def text_to_phonemes_epitran(self, text: str) -> List[str]:
        """Convert text to IPA phonemes using epitran."""
        if not self.g2p_model:
            raise ValueError("Epitran not available")
        
        # Clean text
        text = re.sub(r'[^\w\s]', '', text.upper())
        words = text.split()
        
        phonemes = []
        for word in words:
            if word:
                ipa = self.g2p_model.transliterate(word.lower())
                # Convert IPA to space-separated phonemes
                phoneme_list = list(ipa)
                phonemes.extend(phoneme_list)
                phonemes.append(' ')  # Word boundary
        
        return [p for p in phonemes if p.strip()]
    
    def text_to_phonemes_espeak(self, text: str) -> List[str]:
        """Convert text to phonemes using eSpeak."""
        if not self.use_espeak:
            raise ValueError("eSpeak not available")
        
        import subprocess
        
        try:
            # Use eSpeak to get phonemes
            result = subprocess.run([
                'espeak', '-q', '--ipa', text
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                ipa_text = result.stdout.strip()
                # Convert IPA to list of phonemes
                phonemes = list(ipa_text.replace(' ', ''))
                return [p for p in phonemes if p.strip()]
            else:
                raise ValueError(f"eSpeak failed: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"⚠️  eSpeak conversion failed: {e}")
            return []
    
    def text_to_phonemes_rules(self, text: str) -> List[str]:
        """Convert text to phonemes using rule-based approach (fallback)."""
        
        # Clean and normalize text
        text = re.sub(r'[^\w\s]', '', text.upper())
        words = text.split()
        
        phonemes = []
        
        for word in words:
            if not word:
                continue
                
            # Check if word is in dictionary
            if word in self.word_phonemes:
                word_phonemes = self.word_phonemes[word].split()
                phonemes.extend(word_phonemes)
            else:
                # Apply rule-based conversion
                word_phonemes = self._convert_word_to_phonemes(word)
                phonemes.extend(word_phonemes)
            
            # Add word boundary (silence)
            phonemes.append('SIL')
        
        return [p for p in phonemes if p]
    
    def _convert_word_to_phonemes(self, word: str) -> List[str]:
        """Convert single word to phonemes using rules."""
        
        phonemes = []
        i = 0
        
        while i < len(word):
            # Try digraphs first
            if i < len(word) - 1:
                digraph = word[i:i+2]
                if digraph in self.phoneme_map:
                    phonemes.extend(self.phoneme_map[digraph].split())
                    i += 2
                    continue
            
            # Try single character
            char = word[i]
            if char in self.phoneme_map:
                phonemes.extend(self.phoneme_map[char].split())
            else:
                # Unknown character, skip or use placeholder
                logger.debug(f"Unknown character: {char}")
                phonemes.append('UNK')
            
            i += 1
        
        return phonemes
    
    def convert_text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to phonemes using the best available method.
        
        Args:
            text: Input text string
            
        Returns:
            List of phoneme strings
        """
        if not text or not text.strip():
            return []
        
        # Try methods in order of preference
        methods = []
        
        if self.phoneme_set == "ipa" and self.g2p_model:
            methods.append(("epitran", self.text_to_phonemes_epitran))
        
        if self.use_espeak:
            methods.append(("espeak", self.text_to_phonemes_espeak))
        
        methods.append(("rules", self.text_to_phonemes_rules))
        
        for method_name, method_func in methods:
            try:
                phonemes = method_func(text)
                if phonemes:
                    logger.debug(f"✅ Used {method_name} for: '{text[:30]}...' -> {len(phonemes)} phonemes")
                    return phonemes
            except Exception as e:
                logger.warning(f"⚠️  Method {method_name} failed for '{text[:30]}...': {e}")
                continue
        
        logger.warning(f"❌ All phoneme conversion methods failed for: '{text[:30]}...'")
        return []
    
    def process_transcription_file(self, transcription_path: str, output_dir: str) -> Dict:
        """
        Process a single transcription file to generate phonemes.
        
        Args:
            transcription_path: Path to transcription JSON file
            output_dir: Directory to save phoneme files
            
        Returns:
            dict: Processing results
        """
        transcription_path = Path(transcription_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output paths
        base_name = transcription_path.stem
        phoneme_json_path = output_dir / f"{base_name}.json"
        phoneme_text_path = output_dir / f"{base_name}.txt"
        
        try:
            # Load transcription
            with open(transcription_path, 'r', encoding='utf-8') as f:
                transcription_data = json.load(f)
            
            text = transcription_data.get("transcription", {}).get("text", "")
            
            if not text or not text.strip():
                logger.warning(f"⚠️  Empty transcription for {base_name}")
                return {
                    "file": str(transcription_path),
                    "status": "empty",
                    "phonemes": [],
                    "text": ""
                }
            
            # Convert to phonemes
            phonemes = self.convert_text_to_phonemes(text)
            
            if not phonemes:
                logger.warning(f"⚠️  No phonemes generated for {base_name}")
                return {
                    "file": str(transcription_path),
                    "status": "failed",
                    "phonemes": [],
                    "text": text
                }
            
            # Create output data
            phoneme_data = {
                "file": str(transcription_path),
                "original_text": text,
                "phonemes": phonemes,
                "phoneme_count": len(phonemes),
                "phoneme_set": self.phoneme_set,
                "word_count": len(text.split()),
                "status": "success"
            }
            
            # Save JSON file
            with open(phoneme_json_path, 'w', encoding='utf-8') as f:
                json.dump(phoneme_data, f, indent=2, ensure_ascii=False)
            
            # Save simple text file (space-separated phonemes)
            with open(phoneme_text_path, 'w', encoding='utf-8') as f:
                f.write(' '.join(phonemes))
            
            logger.debug(f"✅ Converted {base_name}: {len(text.split())} words -> {len(phonemes)} phonemes")
            
            return phoneme_data
            
        except Exception as e:
            logger.error(f"❌ Failed to process {transcription_path}: {e}")
            return {
                "file": str(transcription_path),
                "status": "failed",
                "error": str(e),
                "phonemes": [],
                "text": ""
            }
    
    def process_transcription_directory(
        self,
        transcription_dir: str,
        output_dir: str,
        max_files: Optional[int] = None
    ) -> Dict:
        """
        Process all transcription files in a directory.
        
        Args:
            transcription_dir: Directory containing transcription JSON files
            output_dir: Directory to save phoneme files
            max_files: Maximum number of files to process
            
        Returns:
            dict: Processing statistics
        """
        transcription_dir = Path(transcription_dir)
        output_dir = Path(output_dir)
        
        logger.info(f"🔍 Scanning for transcription files in {transcription_dir}")
        
        # Find all JSON transcription files
        json_files = list(transcription_dir.rglob("*.json"))
        json_files = [f for f in json_files if not f.name.endswith('_stats.json')]
        
        if not json_files:
            logger.error(f"❌ No transcription JSON files found in {transcription_dir}")
            return {"status": "failed", "error": "No transcription files found"}
        
        logger.info(f"📁 Found {len(json_files)} transcription files")
        
        if max_files:
            json_files = json_files[:max_files]
            logger.info(f"🎯 Processing first {len(json_files)} files (limited by max_files)")
        
        # Process files
        results = []
        
        for json_path in tqdm(json_files, desc="Converting to phonemes"):
            result = self.process_transcription_file(str(json_path), str(output_dir))
            results.append(result)
        
        # Calculate statistics
        successful = sum(1 for r in results if r.get("status") == "success")
        failed = sum(1 for r in results if r.get("status") == "failed")
        empty = sum(1 for r in results if r.get("status") == "empty")
        total_phonemes = sum(r.get("phoneme_count", 0) for r in results)
        total_words = sum(r.get("word_count", 0) for r in results)
        
        stats = {
            "total_files": len(json_files),
            "successful": successful,
            "failed": failed,
            "empty": empty,
            "total_phonemes": total_phonemes,
            "total_words": total_words,
            "avg_phonemes_per_word": total_phonemes / total_words if total_words > 0 else 0,
            "phoneme_set": self.phoneme_set
        }
        
        logger.info(f"🎊 Phoneme conversion completed!")
        logger.info(f"   Total files: {stats['total_files']}")
        logger.info(f"   Successful: {stats['successful']}")
        logger.info(f"   Failed: {stats['failed']}")
        logger.info(f"   Empty: {stats['empty']}")
        logger.info(f"   Total phonemes: {stats['total_phonemes']}")
        logger.info(f"   Total words: {stats['total_words']}")
        logger.info(f"   Avg phonemes/word: {stats['avg_phonemes_per_word']:.1f}")
        
        # Save statistics
        stats_path = output_dir / "phoneme_conversion_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Convert transcriptions to phonemes")
    parser.add_argument("--transcription_dir", type=str, required=True,
                       help="Directory containing transcription JSON files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save phoneme files")
    parser.add_argument("--phoneme_set", type=str, default="arpabet",
                       choices=["arpabet", "ipa", "custom"],
                       help="Type of phoneme representation")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum number of files to process (for testing)")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting phoneme conversion")
    logger.info(f"   Transcription dir: {args.transcription_dir}")
    logger.info(f"   Output dir: {args.output_dir}")
    logger.info(f"   Phoneme set: {args.phoneme_set}")
    logger.info(f"   Max files: {args.max_files}")
    
    # Initialize converter
    converter = PhonemeConverter(phoneme_set=args.phoneme_set)
    
    # Process transcriptions
    stats = converter.process_transcription_directory(
        transcription_dir=args.transcription_dir,
        output_dir=args.output_dir,
        max_files=args.max_files
    )
    
    if stats.get("status") == "failed":
        logger.error(f"❌ Processing failed: {stats.get('error')}")
        sys.exit(1)
    
    logger.info("🎉 Phoneme conversion completed successfully!")


if __name__ == "__main__":
    main()
