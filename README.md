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

```python
from src.inference.inference_engine import AudioInferenceEngine, InferenceConfig

# Initialize engine
config = InferenceConfig(
    stage1_checkpoint="checkpoints/stage1/best_model.pt",
    stage2_checkpoint="checkpoints/stage2/best_model.pt",
    use_llm_reconstruction=True
)
engine = AudioInferenceEngine(config)

# Transcribe audio
result = engine.transcribe("audio.wav")
print(f"Text: {result.text}")
print(f"Confidence: {result.confidence:.2%}")
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
├── src/
│   ├── models/              # Model implementations
│   ├── data/                # Data processing
│   ├── training/            # Training loops
│   └── inference/           # Inference engine
├── scripts/                 # Training scripts
├── configs/                 # Configuration files
├── tests/                   # Unit tests
├── README.md               # This file
└── ARCHITECTURE.md         # Detailed architecture
```

## Hardware Requirements

### Training (Stage 1)
- GPU: NVIDIA A100 (40GB) or equivalent
- RAM: 32GB+
- Storage: 100GB+ for LibriSpeech + teacher models
- Time: ~24-48 hours for 50 epochs

### Inference
- GPU: Any CUDA-capable GPU (4GB+ VRAM)
- CPU: Possible but slower (10x)
- Latency: ~20-70ms per second of audio

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
