# Valerie: Audio ASR with Ensemble Knowledge Distillation

## Overview

Valerie is a state-of-the-art audio-only automatic speech recognition (ASR) system using ensemble knowledge distillation from multiple teacher models. The system implements a three-stage training approach: Audio to Phonemes to Text, achieving high accuracy with a compact student model.

### Key Features

- **Ensemble Distillation**: Learn from 3 teacher models (Whisper Large V3, WavLM Large, HuBERT Large)
- **Audio-Only**: Efficient Mel spectrogram processing with Conformer encoder
- **Compact Model**: 50M parameters (10x smaller than Whisper Large)
- **Three-Stage Training**: Progressive learning from audio to text
- **High Performance**: Target <8-12% WER with LibriSpeech training

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

### Overview

```
Stage 1: Audio → Phonemes (Ensemble Distillation)
  Input: Audio (16kHz) → Mel Spectrogram → Conformer → CTC/Attention → Phonemes
  Teachers: Whisper + WavLM + HuBERT
  Target: <35% PER

Stage 2: Phonemes → Text (LLM Fine-tuning)
  Input: Phonemes + Synthetic Errors → Qwen LLM → Text
  Target: >95% Reconstruction

Stage 3: End-to-End (Optional Joint Training)
  Target: <8-12% WER
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- torchaudio
- transformers (for Qwen LLM)
- CUDA 11.8+ (for GPU training)

### Setup

```bash
# Clone repository
git clone https://github.com/valerie-team/valerie-asr.git
cd valerie-asr

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

### Training

#### Stage 1: Audio to Phonemes

```bash
python scripts/train_stage1_phoneme_asr.py \
  --config configs/ensemble_distillation_config.yaml \
  --data-root data/LibriSpeech/train-clean-100 \
  --output-dir checkpoints/stage1 \
  --use-distillation
```

#### Stage 2: Phonemes to Text

```bash
python scripts/train_stage2_phoneme_to_text.py \
  --config configs/ensemble_distillation_config.yaml \
  --stage1-checkpoint checkpoints/stage1/best_model.pt \
  --text-data data/text_corpus \
  --output-dir checkpoints/stage2
```

#### Stage 3: End-to-End Fine-tuning

```bash
python scripts/train_stage3_end_to_end.py \
  --config configs/ensemble_distillation_config.yaml \
  --stage1-checkpoint checkpoints/stage1/best_model.pt \
  --stage2-checkpoint checkpoints/stage2/best_model.pt \
  --data-root data/LibriSpeech/train-clean-10 \
  --output-dir checkpoints/stage3
```

### Inference

Using the simple ValerieASR interface:

```python
from ValerieASR import load_asr

# Load model (supports .safetensors or .pt formats)
asr = load_asr("checkpoints/valerie_final.safetensors")

# Transcribe single audio file
result = asr.transcribe("audio.wav")
print(f"Text: {result.text}")
print(f"Confidence: {result.confidence:.2%}")

# Batch transcription
results = asr.transcribe_batch(["audio1.wav", "audio2.wav"])
for i, result in enumerate(results):
    print(f"File {i+1}: {result.text}")

# Get phonemes
result = asr.transcribe("audio.wav", return_phonemes=True)
print(f"Phonemes: {' '.join(result.phonemes)}")
```

Or use the CLI:

```bash
# Simple transcription
python ValerieASR.py checkpoints/valerie_final.safetensors audio.wav

# Or use the inference engine directly
python scripts/inference.py \
  --checkpoint checkpoints/valerie_final.safetensors \
  --audio audio.wav \
  --output transcription.txt
```

### Evaluation

```bash
# Evaluate Stage 1 (Phoneme Error Rate)
python scripts/evaluate.py \
  --stage stage1 \
  --checkpoint checkpoints/stage1/best_model.pt \
  --test-data data/LibriSpeech/test-clean

# Evaluate End-to-End (Word Error Rate)
python scripts/evaluate.py \
  --stage stage3 \
  --checkpoint checkpoints/stage3/best_model.pt \
  --test-data data/LibriSpeech/test-clean
```

## Model Components

### AudioFrontend
- Mel spectrogram extraction (80 bins, 16kHz)
- SpecAugment for data augmentation
- Conv1D projection + positional encoding

### Conformer Encoder
- 256 dimensions, 12 layers, 4 attention heads
- Hybrid CNN-Transformer blocks
- Local and global context modeling

### CTC/Attention Hybrid
- Joint training for better alignment
- CTC head for phoneme prediction
- Attention decoder for sequence generation

### Ensemble Distillation
- 3 teacher models with weighted aggregation
- Feature-level and soft-label distillation
- Temperature-scaled knowledge transfer

## Performance Targets

| Metric | Stage 1 | Stage 2 | Stage 3 (Final) |
|--------|---------|---------|-----------------|
| PER | <35% | - | <20% |
| WER | - | - | <8-12% |
| Reconstruction | - | >95% | >97% |
| Model Size | 50M | 600M | 650M |
| Inference Speed | ~20ms/sec | ~50ms/sec | ~70ms/sec |

## Configuration

Main configuration: `configs/ensemble_distillation_config.yaml`

```yaml
model:
  embed_dim: 256
  conformer_layers: 12
  conformer_heads: 4
  vocab_size: 40

distillation:
  enable: true
  teachers:
    - model_name: "openai/whisper-large-v3"
      weight: 0.4
    - model_name: "microsoft/wavlm-large"
      weight: 0.3
    - model_name: "facebook/hubert-large-ls960-ft"
      weight: 0.3
```

## Project Structure

```
Valerie/
├── ValerieASR.py           # Main entry point for inference
├── src/
│   ├── models/              # Model implementations
│   │   ├── audio_phoneme_model.py  # Student model (Stage 1)
│   │   ├── ensemble_distillation.py  # Teacher ensemble
│   │   └── qwen_llm.py      # LLM for Stage 2
│   ├── data/                # Data processing
│   ├── training/            # Training loops
│   ├── inference/           # Inference engine
│   └── utils/               # Utilities (config, logging)
├── scripts/                 # Training scripts
│   ├── train_stage1_phoneme_asr.py
│   ├── train_stage2_phoneme_to_text.py
│   └── train_stage3_end_to_end.py
├── configs/                 # Configuration files
│   └── ensemble_distillation_config.yaml
├── tests/                   # Unit tests
├── README.md               # This file
├── ARCHITECTURE.md         # Detailed architecture
└── HARDWARE_REQUIREMENTS.md  # Hardware requirements
```

## Model Format

Valerie supports both **safetensors** (recommended) and PyTorch `.pt` formats:

- **Safetensors** (.safetensors): Faster loading, safer, better for production
- **PyTorch** (.pt): Traditional format, backward compatible

```python
# Both formats work seamlessly
asr = load_asr("model.safetensors")  # Recommended
asr = load_asr("model.pt")           # Also supported
```

Models are automatically detected and loaded based on file extension.

## Hardware Requirements

For detailed hardware requirements, see [HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md).

### Quick Summary

**Training:**
- Minimum: RTX 3090 (24GB) for Stage 1, RTX 3060 (12GB) for Stage 2
- Recommended: A100 (40GB) for faster training
- Total time: 17-50 hours across all stages
- Total cost (cloud): $30-75 on AWS/GCP, $15-30 on Vast.ai

**Inference:**
- Minimum: RTX 3060 (12 GB) for real-time
- Throughput: 100-200x real-time on RTX 3090
- CPU-only: Possible but ~5x slower than real-time
- Latency: 50-100ms per second of audio (GPU)

## Citation

```bibtex
@misc{valerie2024,
  title={Valerie: Audio ASR with Ensemble Knowledge Distillation},
  author={Valerie Team},
  year={2024},
  howpublished={\url{https://github.com/valerie-team/valerie-asr}}
}
```

## License

This project is licensed under the MIT License - see LICENSE for details.

## Acknowledgments

- Teacher Models: OpenAI Whisper, Microsoft WavLM, Facebook HuBERT
- LLM: Qwen team for Qwen 0.6B model
- Datasets: LibriSpeech, WikiText, BookCorpus
- Frameworks: PyTorch, Transformers
