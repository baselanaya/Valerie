# Valerie Architecture

## Overview

Valerie implements a three-stage audio ASR system with ensemble knowledge distillation. This document provides detailed technical specifications and architecture diagrams.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VALERIE ASR SYSTEM                           │
│                Three-Stage Architecture                         │
└─────────────────────────────────────────────────────────────────┘

                           ┌──────────┐
                           │  Audio   │ 16kHz waveform
                           │ (16kHz)  │
                           └────┬─────┘
                                │
                    ┌───────────▼────────────┐
                    │   STAGE 1: Audio → Phonemes  │
                    │  (Ensemble Distillation)     │
                    └───────────┬────────────┘
                                │
                         ┌──────▼───────┐
                         │  Phonemes    │
                         │  Sequence    │
                         └──────┬───────┘
                                │
                    ┌───────────▼────────────┐
                    │   STAGE 2: Phonemes → Text   │
                    │   (LLM Fine-tuning)          │
                    └───────────┬────────────┘
                                │
                         ┌──────▼───────┐
                         │  Text Output │
                         └──────────────┘
```

## Stage 1: Audio to Phonemes

### Architecture

```
Audio Waveform [T]
    │
    ▼
┌──────────────────────────┐
│   Mel Spectrogram        │
│   - n_fft: 400           │
│   - hop_length: 160      │
│   - n_mels: 80           │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   AudioFrontend          │
│   - SpecAugment          │
│   - Conv1D projection    │
│   - Positional encoding  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   Conformer Encoder      │
│   - Layers: 12           │
│   - Dim: 256             │
│   - Heads: 4             │
│   - FFN expansion: 4x    │
└──────────┬───────────────┘
           │
           ├──────────────┐
           │              │
           ▼              ▼
    ┌──────────┐   ┌──────────┐
    │ CTC Head │   │Attention │
    │          │   │ Decoder  │
    └────┬─────┘   └────┬─────┘
         │              │
         └──────┬───────┘
                │
                ▼
         Phoneme Sequence
         [39 ARPAbet phonemes]
```

### Ensemble Distillation

```
┌─────────────────────────────────────────────┐
│         TEACHER ENSEMBLE                    │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────────┐                      │
│  │ Whisper Large V3 │ (weight: 0.4)        │
│  │   1.5B params    │                      │
│  └────────┬─────────┘                      │
│           │                                 │
│  ┌────────▼─────────┐                      │
│  │  WavLM Large     │ (weight: 0.3)        │
│  │   317M params    │                      │
│  └────────┬─────────┘                      │
│           │                                 │
│  ┌────────▼─────────┐                      │
│  │  HuBERT Large    │ (weight: 0.3)        │
│  │   316M params    │                      │
│  └────────┬─────────┘                      │
│           │                                 │
│           ▼                                 │
│    Weighted Ensemble                        │
│    Feature Aggregation                      │
│                                             │
└──────────────┬──────────────────────────────┘
               │
               ▼
        ┌─────────────┐
        │   STUDENT   │
        │ AudioPhoneme│
        │  ASR Model  │
        │  50M params │
        └─────────────┘
```

### Distillation Losses

1. **Feature-Level Distillation**
   - MSE loss between teacher and student encoder outputs
   - Alignment through projection layers
   - Weight: 0.5

2. **Soft-Label Distillation**
   - KL divergence between temperature-scaled predictions
   - Temperature: 4.0
   - Weight: 0.3

3. **Hard-Label Loss**
   - Standard CTC + Attention loss
   - Ground truth phoneme labels
   - Weight: 0.2

## Stage 2: Phonemes to Text

### Architecture

```
Phoneme Sequence + Confidence Scores
    │
    ▼
┌──────────────────────────┐
│  Synthetic Error         │
│  Injection (Training)    │
│  - Substitution: 5%      │
│  - Deletion: 5%          │
│  - Insertion: 5%         │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   Prompt Construction    │
│   "Convert phonemes to   │
│    readable text..."     │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   Qwen 0.6B LLM          │
│   LoRA Fine-tuned        │
│   - Rank: 32             │
│   - Alpha: 64            │
│   - Target modules: all  │
└──────────┬───────────────┘
           │
           ▼
     Text Output
```

### Training Data

**Text Sources:**
- WikiText-103 (100M+ words)
- BookCorpus (800M+ words)
- Custom phonetic datasets

**Processing:**
1. Text → Phonemes (g2p_en)
2. Add synthetic errors (10-15%)
3. Train LLM to reconstruct original text
4. Validate on clean phoneme sequences

## Stage 3: End-to-End Fine-tuning

### Joint Training

```
Audio Input
    │
    ├─► Stage 1 Model (frozen/fine-tuned)
    │       │
    │       ▼
    │   Phonemes
    │       │
    └─────► Stage 2 Model (fine-tuned)
            │
            ▼
        Text Output
```

**Training Strategy:**
- Initial: Freeze Stage 1, train Stage 2
- Middle: Unfreeze top layers of Stage 1
- Final: Joint end-to-end training
- Dataset: LibriSpeech train-clean-10 (10 hours)

## Model Components

### AudioFrontend

**Input:** Raw audio waveform [B, T_audio]
**Output:** Features [B, T_feat, 256]

**Components:**
1. MelSpectrogramExtractor
   - Sample rate: 16000 Hz
   - FFT size: 400
   - Hop length: 160 (10ms)
   - Mel bins: 80
   - Log scale

2. SpecAugment
   - Frequency masking: max 15 bins
   - Time masking: max 35 frames
   - Probability: 50%

3. Projection
   - Conv1D (80 → 256 channels)
   - Kernel size: 3
   - Stride: 1

4. Positional Encoding
   - Sinusoidal embeddings
   - Max sequence: 3000 frames

### Conformer Block

**Structure (per layer):**

```
Input [B, T, 256]
    │
    ▼
┌─────────────────┐
│ Feed-Forward    │ expansion=4
│ (Swish + GLU)   │
└────────┬────────┘
         │ (1/2 scaling)
         ▼
┌─────────────────┐
│ Multi-Head      │ heads=4
│ Self-Attention  │
└────────┬────────┘
         │ + residual
         ▼
┌─────────────────┐
│ Convolution     │ kernel=31
│ (Depthwise +    │
│  Pointwise)     │
└────────┬────────┘
         │ + residual
         ▼
┌─────────────────┐
│ Feed-Forward    │ expansion=4
│ (Swish + GLU)   │
└────────┬────────┘
         │ (1/2 scaling)
         ▼
┌─────────────────┐
│ Layer Norm      │
└────────┬────────┘
         │
         ▼
    Output [B, T, 256]
```

**12 blocks total, stacked sequentially**

### CTC/Attention Hybrid

**CTC Branch:**
- Linear projection: 256 → 40 (vocab size)
- Log-softmax activation
- CTC loss with blank token

**Attention Branch:**
- LSTM decoder: 2 layers, 256 hidden
- Location-aware attention mechanism
- Attention context: 512 dim
- Output projection: 512 → 40

**Joint Decoding:**
- CTC weight: 0.3
- Attention weight: 0.7
- Beam search with both scores

## Training Details

### Stage 1: Audio to Phonemes

**Phase 1: Warmup (5 epochs)**
- Loss: CTC only
- Optimizer: Adam (lr=1e-4)
- Batch size: 32
- Goal: Initialize encoder

**Phase 2: Distillation (45 epochs)**
- Loss: CTC + Attention + Distillation
- CTC weight: 0.2
- Attention weight: 0.1
- Distillation weight: 0.7
- Optimizer: Adam (lr=1e-4 → 1e-5)
- Scheduler: Cosine annealing
- Batch size: 32
- Gradient clipping: 1.0

**Data Augmentation:**
- Speed perturbation: 0.9-1.1x
- Volume perturbation: 0.8-1.2x
- Noise injection: SNR 10-30 dB
- SpecAugment: On features

### Stage 2: Phonemes to Text

**Training:**
- Model: Qwen 0.6B + LoRA
- LoRA rank: 32, alpha: 64
- Batch size: 64
- Learning rate: 2e-4
- Epochs: 10
- Optimizer: AdamW (weight decay: 0.01)

**Data:**
- Training samples: 1M text samples
- Validation: 10K samples
- Synthetic error rate: 10-15%

### Stage 3: End-to-End

**Joint Training:**
- Batch size: 16 (smaller due to both models)
- Learning rate: 5e-5
- Epochs: 20
- Freeze schedule:
  - Epochs 1-5: Stage 1 frozen
  - Epochs 6-15: Top 3 layers unfrozen
  - Epochs 16-20: Full fine-tuning

## Performance Metrics

### Stage 1 Targets

| Dataset | PER | CTC Accuracy | Attention Accuracy |
|---------|-----|--------------|-------------------|
| LibriSpeech train | <30% | >75% | >80% |
| LibriSpeech test-clean | <35% | >70% | >75% |
| LibriSpeech test-other | <45% | >60% | >65% |

### Stage 2 Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Reconstruction Rate | >95% | Exact text match |
| Edit Distance | <3% | Character-level |
| Phoneme Tolerance | ±2 errors | Per sequence |

### Stage 3 Targets

| Dataset | WER | Accuracy | Speed (RTF) |
|---------|-----|----------|-------------|
| test-clean | <8% | >92% | 0.1x |
| test-other | <12% | >88% | 0.1x |

## Implementation Notes

### Memory Optimization

- Gradient checkpointing for Conformer
- Mixed precision training (FP16)
- Batch accumulation for large batch sizes
- Teacher model caching (pre-compute features)

### Inference Optimization

- Model quantization (INT8) for deployment
- ONNX export for production
- Batch inference for throughput
- Streaming support for real-time

### Compute Requirements

**Training:**
- Stage 1: ~48 hours on A100 (40GB)
- Stage 2: ~24 hours on A100 (40GB)
- Stage 3: ~12 hours on A100 (40GB)
- Total: ~84 hours (~3.5 days)

**Inference:**
- GPU: 20-70ms per second of audio
- CPU: 200-700ms per second of audio
- Memory: 2GB (Stage 1) + 4GB (Stage 2)

## Code Structure

```
src/models/
├── audio_frontend.py          # Mel spectrogram + SpecAugment
├── audio_phoneme_model.py     # Complete Stage 1 model
├── ensemble_distillation.py   # 3-teacher distillation
├── conformer.py                # Conformer encoder
├── hybrid_ctc_attention.py     # CTC/Attention head
└── qwen_llm.py                 # Stage 2 LLM

src/data/
├── dataset.py                  # DataSample structure
├── transforms.py               # Audio augmentation
├── collate.py                  # Batch collation
└── text_to_phoneme.py          # G2P conversion

src/training/
├── trainer.py                  # Base trainer
├── stage1_trainer.py           # Stage 1 specific
├── stage2_trainer.py           # Stage 2 specific
└── stage3_trainer.py           # Stage 3 specific

src/inference/
└── inference_engine.py         # Production inference
```

## References

### Teacher Models

1. **Whisper Large V3**
   - Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision"
   - Model: openai/whisper-large-v3
   - Parameters: 1.5B

2. **WavLM Large**
   - Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing"
   - Model: microsoft/wavlm-large
   - Parameters: 317M

3. **HuBERT Large**
   - Hsu et al., "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction"
   - Model: facebook/hubert-large-ls960-ft
   - Parameters: 316M

### Student Architecture

1. **Conformer**
   - Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition"
   - Modifications: Reduced dimensions for efficiency

2. **Hybrid CTC/Attention**
   - Watanabe et al., "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition"

### Knowledge Distillation

- Hinton et al., "Distilling the Knowledge in a Neural Network"
- Sanh et al., "DistilBERT, a distilled version of BERT"
- Multi-teacher ensemble distillation techniques

## License

MIT License - See LICENSE file for details
