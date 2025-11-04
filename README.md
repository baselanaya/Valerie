# Valerie: Audio ASR with Ensemble Knowledge Distillation

## Overview

Valerie is a state-of-the-art **audio-only** automatic speech recognition (ASR) system using ensemble knowledge distillation from multiple teacher models. The system implements a three-stage training approach: **Audio → Phonemes → Text**, achieving high accuracy with a compact student model.

### Key Features

- 🎯 **Ensemble Distillation**: Learn from 3 teacher models (Whisper Large V3, Wav2Vec2 Large, HuBERT Large)
- 🎵 **Audio-Only**: Efficient Mel spectrogram processing with Conformer encoder
- 📉 **Compact Model**: 50M parameters (10x smaller than Whisper Large)
- 🎓 **Three-Stage Training**: Progressive learning from audio to text
- 🚀 **High Performance**: Target <8-12% WER with LibriSpeech training

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    THREE-STAGE ARCHITECTURE                 │
└─────────────────────────────────────────────────────────────┘

STAGE 1: Audio → Phonemes (Ensemble Distillation)
────────────────────────────────────────────────────

Input Audio (16kHz)
     ↓
Mel Spectrogram (80 bins)
     ↓
AudioFrontend (Conv1D + Positional Encoding)
     ↓
Conformer Encoder (256 dim, 12 layers)
     ↓
CTC/Attention Hybrid Head
     ↓
Phoneme Sequence (39 ARPAbet phonemes)

Teacher Models (Ensemble):
  • Whisper Large V3 (weight: 0.4)
  • Wav2Vec2 Large (weight: 0.3)
  • HuBERT Large (weight: 0.3)

Training: 30-100h LibriSpeech + Distillation
Target: <35% PER


STAGE 2: Phonemes → Text (LLM Fine-tuning)
────────────────────────────────────────────

Phoneme Sequence + Synthetic Errors
     ↓
Qwen 0.6B LLM (LoRA fine-tuned)
     ↓
Text Reconstruction

Training: Unlimited text data (WikiText + BookCorpus)
Target: >95% Reconstruction Accuracy


STAGE 3: End-to-End Fine-tuning (Optional)
────────────────────────────────────────────

Audio → Phonemes → Text (Joint training)

Training: 10h high-quality paired data
Target: <8-12% WER (Final)
```

## Key Innovations

### 1. Multi-Teacher Ensemble Distillation
- **Problem**: Single teacher provides limited supervision
- **Solution**: Aggregate knowledge from 3 diverse audio ASR models
- **Impact**: Richer feature learning and better generalization

### 2. Two-Stage Architecture
- **Problem**: End-to-end models require large paired audio-text datasets
- **Solution**: Separate phoneme prediction (audio data) from text reconstruction (unlimited text data)
- **Impact**: Better data efficiency and performance

### 3. Conformer Encoder (Compact)
- **Problem**: Large Transformer models are computationally expensive
- **Solution**: Hybrid CNN-Transformer with reduced dimensions (256 dim, 12 layers)
- **Impact**: 10x parameter reduction while maintaining accuracy through distillation

### 4. Synthetic Error Injection
- **Problem**: LLM needs to handle imperfect phoneme predictions
- **Solution**: Add controlled errors (substitution, deletion, insertion) during LLM training
- **Impact**: Robust text reconstruction even with phoneme errors

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

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Data Preparation

```bash
# Download LibriSpeech (for Stage 1)
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz

# Prepare text data (for Stage 2)
python scripts/prepare_text_data.py --output data/text_corpus
```

### 2. Training

#### Stage 1: Audio → Phonemes (with Ensemble Distillation)

```bash
python scripts/train_stage1_phoneme_asr.py \
  --config configs/ensemble_distillation_config.yaml \
  --data-root data/LibriSpeech/train-clean-100 \
  --output-dir checkpoints/stage1 \
  --use-distillation
```

**Key Parameters:**
- Warmup: 5 epochs (CTC only)
- Distillation: 45 epochs (CTC + 3-teacher ensemble)
- Target: <35% PER

#### Stage 2: Phonemes → Text (LLM Fine-tuning)

```bash
python scripts/train_stage2_phoneme_to_text.py \
  --config configs/ensemble_distillation_config.yaml \
  --stage1-checkpoint checkpoints/stage1/best_model.pt \
  --text-data data/text_corpus \
  --output-dir checkpoints/stage2
```

**Key Parameters:**
- Base model: Qwen 0.6B
- LoRA rank: 32, alpha: 64
- Synthetic error rate: 10-15%
- Target: >95% reconstruction

#### Stage 3: End-to-End Fine-tuning (Optional)

```bash
python scripts/train_stage3_end_to_end.py \
  --config configs/ensemble_distillation_config.yaml \
  --stage1-checkpoint checkpoints/stage1/best_model.pt \
  --stage2-checkpoint checkpoints/stage2/best_model.pt \
  --data-root data/LibriSpeech/train-clean-10 \
  --output-dir checkpoints/stage3
```

**Key Parameters:**
- Joint training of both stages
- Small high-quality dataset (10h)
- Target: <8-12% WER

### 3. Inference

```python
from src.inference.inference_engine import AudioInferenceEngine, InferenceConfig

# Initialize engine
config = InferenceConfig(
    stage1_checkpoint="checkpoints/stage1/best_model.pt",
    stage2_checkpoint="checkpoints/stage2/best_model.pt",
    use_llm_reconstruction=True
)
engine = AudioInferenceEngine(config)

# Transcribe audio file
result = engine.transcribe("audio.wav")
print(f"Text: {result.text}")
print(f"Confidence: {result.confidence:.2%}")
```

### 4. Evaluation

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

## Model Architecture Details

### AudioFrontend
- **Input**: Raw audio waveform at 16kHz
- **Output**: Mel spectrogram features (80 bins)
- **Components**:
  - MelSpectrogram extraction (n_fft=400, hop_length=160)
  - SpecAugment (frequency + time masking)
  - Conv1D projection to embedding dimension
  - Positional encoding

### Conformer Encoder
- **Dimensions**: 256
- **Layers**: 12
- **Heads**: 4
- **Components** (per layer):
  - Feed-forward module (expansion factor: 4)
  - Multi-head self-attention
  - Convolution module (kernel size: 31)
  - Feed-forward module
  - Layer normalization

### CTC/Attention Hybrid
- **CTC Head**: Linear projection to phoneme vocabulary (40 tokens)
- **Attention Decoder**: LSTM-based decoder with location-aware attention
- **Training**: Joint CTC + Attention loss (λ_CTC = 0.3, λ_Att = 0.7)

### Ensemble Distillation Module
- **Teachers**: Whisper Large V3, Wav2Vec2 Large, HuBERT Large
- **Distillation Losses**:
  - Feature-level MSE (encoder outputs)
  - Soft label KL divergence (temperature=4.0)
- **Aggregation**: Weighted average (0.4, 0.3, 0.3)

## Performance Targets

| Metric | Stage 1 | Stage 2 | Stage 3 (Final) |
|--------|---------|---------|-----------------|
| **PER** | <35% | - | <20% |
| **WER** | - | - | <8-12% |
| **Reconstruction** | - | >95% | >97% |
| **Model Size** | 50M params | 600M params | 650M params |
| **Inference Speed** | ~20ms/sec | ~50ms/sec | ~70ms/sec |

## Configuration

Main configuration file: `configs/ensemble_distillation_config.yaml`

Key sections:
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
    - model_name: "facebook/wav2vec2-large-960h-lv60-self"
      weight: 0.3
    - model_name: "facebook/hubert-large-ls960-ft"
      weight: 0.3
  temperature: 4.0

training:
  batch_size: 32
  learning_rate: 1e-4
  warmup_epochs: 5
  distillation_epochs: 45
```

## Project Structure

```
Valerie/
├── src/
│   ├── models/
│   │   ├── audio_frontend.py          # Mel spectrogram extraction
│   │   ├── audio_phoneme_model.py     # Complete Stage 1 model
│   │   ├── ensemble_distillation.py   # 3-teacher distillation
│   │   ├── conformer.py                # Conformer encoder
│   │   ├── hybrid_ctc_attention.py     # CTC/Attention head
│   │   └── qwen_llm.py                 # Stage 2 LLM
│   ├── data/
│   │   ├── transforms.py               # Audio augmentation
│   │   ├── collate.py                  # Batch collation
│   │   └── text_to_phoneme.py          # G2P conversion
│   ├── training/
│   │   └── trainer.py                  # Training loops
│   └── inference/
│       └── inference_engine.py         # Audio inference
├── scripts/
│   ├── train_stage1_phoneme_asr.py    # Stage 1 training
│   ├── train_stage2_phoneme_to_text.py # Stage 2 training
│   └── train_stage3_end_to_end.py     # Stage 3 training
├── configs/
│   └── ensemble_distillation_config.yaml
├── tests/
│   ├── test_models.py                  # Model tests
│   └── test_integration.py             # Integration tests
└── README.md
```

## Documentation

- **Architecture Details**: See [README_ENSEMBLE_DISTILLATION.md](README_ENSEMBLE_DISTILLATION.md)
- **Training Guide**: See training scripts in `scripts/`
- **API Reference**: See docstrings in source files

## Hardware Requirements

### Training (Stage 1 with Distillation)
- **GPU**: NVIDIA A100 (40GB) or equivalent
- **RAM**: 32GB+
- **Storage**: 100GB+ for LibriSpeech + teacher models
- **Training Time**: ~24-48 hours for 50 epochs

### Training (Stage 2)
- **GPU**: NVIDIA A100 (40GB) recommended
- **RAM**: 32GB+
- **Storage**: 50GB+ for text corpus
- **Training Time**: ~12-24 hours

### Inference
- **GPU**: Any CUDA-capable GPU (4GB+ VRAM)
- **CPU**: Possible but slower (10x)
- **Latency**: ~20-70ms per second of audio (on GPU)

## Migration from Video-Based System

If you're upgrading from the old video-based Valerie:

1. **Old imports** (deprecated):
   ```python
   from src.models import ValerieModel  # ❌ Deprecated
   from src.data import VideoTransforms  # ❌ Removed
   ```

2. **New imports** (audio-only):
   ```python
   from src.models.audio_phoneme_model import AudioPhonemeASR  # ✅
   from src.data.transforms import AudioTransforms  # ✅
   from src.inference.inference_engine import AudioInferenceEngine  # ✅
   ```

3. **Key changes**:
   - Video frames → Audio waveforms (16kHz)
   - 3D CNN → Mel spectrogram + Conformer
   - Single teacher distillation → 3-teacher ensemble
   - VoxCeleb2/AVSpeech → LibriSpeech

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use Valerie in your research, please cite:

```bibtex
@misc{valerie2024,
  title={Valerie: Audio ASR with Ensemble Knowledge Distillation},
  author={Valerie Team},
  year={2024},
  howpublished={\url{https://github.com/valerie-team/valerie-asr}}
}
```

## Acknowledgments

- **Teacher Models**: OpenAI Whisper, Facebook Wav2Vec2, Facebook HuBERT
- **LLM**: Qwen team for Qwen 0.6B model
- **Datasets**: LibriSpeech, WikiText, BookCorpus
- **Frameworks**: PyTorch, Transformers

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Email: contact@valerie-asr.dev

---

**Note**: This is the audio-only version of Valerie. The old video-based system has been deprecated and migrated to an audio-only ensemble distillation architecture for better performance and efficiency.
