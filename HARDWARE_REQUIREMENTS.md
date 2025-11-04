# Valerie ASR - Hardware Requirements

This document outlines the estimated hardware requirements for training and inference with the Valerie audio ASR system using ensemble knowledge distillation.

## Model Sizes

### Student Model (AudioPhonemeASR)
- **Parameters**: ~50M
  - Conformer Encoder: ~45M (12 layers, 256 dim, 4 heads)
  - CTC Head: ~10K (256 → 40 vocab)
  - Attention Decoder: ~5M (256 dim)
- **Memory (fp32)**: ~200 MB
- **Memory (fp16)**: ~100 MB

### Teacher Models (frozen during training)
1. **Whisper Large V3**
   - Parameters: 1.55B
   - Memory (fp16): ~3.1 GB

2. **WavLM Large**
   - Parameters: 317M
   - Memory (fp16): ~635 MB

3. **HuBERT Large**
   - Parameters: 317M
   - Memory (fp16): ~635 MB

**Total Teacher Memory**: ~4.4 GB (fp16, frozen)

### Stage 2: Qwen LLM with LoRA
- **Base Model**: Qwen2-0.6B
- **Parameters (trainable with LoRA)**: ~1.2M (rank=16, alpha=32)
- **Total Parameters**: 600M
- **Memory (with LoRA)**: ~2.4 GB (fp16)

---

## Training Hardware Requirements

### Stage 1: Audio → Phonemes (with Ensemble Distillation)

**Minimum Configuration:**
- **GPU**: 1x NVIDIA RTX 3090 (24 GB) or A5000 (24 GB)
- **VRAM Usage**:
  - Student Model: 200 MB (fp32)
  - Teacher Models: 4.4 GB (fp16, frozen)
  - Optimizer States (AdamW): 800 MB (student only)
  - Gradients: 200 MB
  - Batch Data (batch_size=16):
    - Audio waveforms: ~500 MB
    - Mel spectrograms: ~200 MB
    - Teacher features: ~1.5 GB (cached)
  - Forward/Backward Activations: ~2 GB
  - **Total**: ~9.8 GB
- **RAM**: 32 GB
- **Storage**: 500 GB SSD (for LibriSpeech + checkpoints)
- **Training Time**: ~24-36 hours (50 epochs, batch_size=16)

**Recommended Configuration:**
- **GPU**: 1x NVIDIA A100 (40 GB) or RTX 4090 (24 GB)
- **VRAM Usage**:
  - Increased batch_size=32: ~15 GB
  - Better throughput and convergence
- **RAM**: 64 GB
- **Storage**: 1 TB NVMe SSD
- **Training Time**: ~12-18 hours (50 epochs, batch_size=32)

**Optimal Configuration (Multi-GPU):**
- **GPU**: 2x NVIDIA A100 (40 GB each) or 4x RTX 3090 (24 GB each)
- **VRAM per GPU**: ~18 GB (batch_size=32 per GPU)
- **RAM**: 128 GB
- **Storage**: 2 TB NVMe SSD
- **Training Time**: ~6-9 hours (50 epochs, distributed training)

---

### Stage 2: Phonemes → Text (LoRA Fine-tuning)

**Minimum Configuration:**
- **GPU**: 1x RTX 3060 (12 GB) or RTX 3070 (8 GB)
- **VRAM Usage**:
  - Qwen Model (fp16): 2.4 GB
  - LoRA Adapters: 5 MB
  - Optimizer States: 5 MB
  - Batch Data (batch_size=32): 500 MB
  - Activations: 1 GB
  - **Total**: ~4 GB
- **RAM**: 16 GB
- **Storage**: 100 GB SSD
- **Training Time**: ~4-6 hours (20 epochs, batch_size=32)

**Recommended Configuration:**
- **GPU**: 1x RTX 3090 (24 GB) or A5000 (24 GB)
- **VRAM Usage**: ~6 GB (batch_size=64)
- **RAM**: 32 GB
- **Storage**: 200 GB SSD
- **Training Time**: ~2-3 hours (20 epochs, batch_size=64)

---

### Stage 3: End-to-End Fine-tuning

**Minimum Configuration:**
- **GPU**: 1x NVIDIA A100 (40 GB)
- **VRAM Usage**:
  - Full Pipeline: ~12 GB
  - Batch_size=8
  - **Total**: ~18 GB
- **RAM**: 64 GB
- **Storage**: 500 GB SSD
- **Training Time**: ~6-8 hours (10 epochs, 10 hours LibriSpeech)

**Recommended Configuration:**
- **GPU**: 2x NVIDIA A100 (40 GB each)
- **VRAM per GPU**: ~20 GB (batch_size=8 per GPU)
- **RAM**: 128 GB
- **Storage**: 1 TB NVMe SSD
- **Training Time**: ~3-4 hours (10 epochs, distributed)

---

## Inference Hardware Requirements

### Real-time Inference (1x speedup)
- **GPU**: RTX 3060 (12 GB) or better
- **VRAM**: 2 GB (student model + buffer)
- **RAM**: 8 GB
- **Latency**: ~50-100 ms per second of audio
- **Throughput**: ~10-20x real-time on RTX 3090

### Batch Inference
- **GPU**: RTX 3070 (8 GB) or better
- **VRAM**: 4 GB (batch_size=32)
- **RAM**: 16 GB
- **Throughput**: ~100-200x real-time on RTX 3090

### CPU-only Inference (not recommended)
- **CPU**: Modern 8-core processor (Intel i7/AMD Ryzen 7)
- **RAM**: 16 GB
- **Latency**: ~2-5 seconds per second of audio (0.2-0.5x real-time)

---

## Training Dataset Requirements

### LibriSpeech
- **train-clean-100**: 100 hours, ~30 GB
- **train-clean-360**: 360 hours, ~100 GB
- **train-other-500**: 500 hours, ~150 GB
- **dev-clean**: 5.4 hours, ~1.5 GB
- **test-clean**: 5.4 hours, ~1.5 GB

**Recommended**: Start with train-clean-100 (30 GB)

### Text Corpus (Stage 2)
- **WikiText-103**: ~500 MB
- **BookCorpus**: ~4 GB
- **Total**: ~5 GB

---

## Summary Table

| Stage | Min GPU | Min VRAM | Rec GPU | Rec VRAM | Training Time | Cost/Hour |
|-------|---------|----------|---------|----------|---------------|-----------|
| Stage 1 | RTX 3090 | 24 GB | A100 40GB | 40 GB | 12-36 hours | $1.50-$3.00 |
| Stage 2 | RTX 3060 | 12 GB | RTX 3090 | 24 GB | 2-6 hours | $0.50-$1.50 |
| Stage 3 | A100 40GB | 40 GB | 2x A100 | 40 GB/each | 3-8 hours | $2.00-$4.00 |
| **Total** | - | - | - | - | **17-50 hours** | **$30-75** |

**Note**: Costs based on cloud GPU pricing (AWS/GCP p3/g4 instances)

---

## Optimization Tips

### Memory Optimization
1. **Gradient Checkpointing**: Reduce activation memory by 50%
   - Increases training time by ~20%
   - Essential for batch_size > 16 on 24 GB GPUs

2. **Mixed Precision (fp16/bf16)**:
   - Reduces memory by 50%
   - Speeds up training by 2-3x on modern GPUs
   - Already enabled in config

3. **Gradient Accumulation**:
   - Simulate larger batch sizes with limited VRAM
   - Example: batch_size=8, accumulation_steps=4 → effective batch_size=32

4. **Teacher Feature Caching**:
   - Cache teacher outputs to disk
   - Reduces VRAM by 4.4 GB but adds I/O overhead

### Compute Optimization
1. **DataLoader Workers**: Use num_workers=4-8 for better CPU-GPU pipeline
2. **Pin Memory**: Enable pin_memory=True for faster data transfer
3. **Compile Models**: Use torch.compile() on PyTorch 2.0+ for 20-30% speedup
4. **Multi-GPU Training**: Use DistributedDataParallel for near-linear scaling

---

## Cloud Provider Recommendations

### AWS
- **Stage 1**: p3.2xlarge (V100 16 GB, $3.06/hr) or g5.2xlarge (A10G 24 GB, $1.21/hr)
- **Stage 2**: g4dn.xlarge (T4 16 GB, $0.53/hr)
- **Stage 3**: p4d.24xlarge (8x A100 40 GB, $32.77/hr) split across team

### Google Cloud
- **Stage 1**: n1-standard-8 + 1x V100 ($2.48/hr) or 1x A100 ($3.67/hr)
- **Stage 2**: n1-standard-4 + 1x T4 ($0.35/hr)
- **Stage 3**: a2-highgpu-2g (2x A100 40 GB, $6.80/hr)

### Vast.ai / RunPod (Budget-friendly)
- **Stage 1**: RTX 3090 ($0.30-0.50/hr) or RTX 4090 ($0.60-0.90/hr)
- **Stage 2**: RTX 3060 Ti ($0.15-0.25/hr)
- **Stage 3**: 2x RTX 3090 ($0.60-1.00/hr)

**Total Training Cost (Vast.ai)**: $15-30 for all 3 stages

---

## Benchmarks (Measured)

Hardware benchmarks will be added after training runs. Expected metrics:

| Hardware | Stage 1 Speed | Stage 2 Speed | Inference Speed |
|----------|---------------|---------------|-----------------|
| RTX 3090 | 0.8s/step | 0.2s/step | 150x real-time |
| RTX 4090 | 0.5s/step | 0.15s/step | 200x real-time |
| A100 40GB | 0.4s/step | 0.12s/step | 250x real-time |
| V100 16GB | 1.2s/step | 0.3s/step | 100x real-time |

---

## Recommendations by Use Case

### Academic Research / Single GPU
- **Budget**: RTX 3090 (24 GB) - $1,500
- **Training Time**: 2-3 days total
- **Cost**: $0 (own hardware)

### Small Team / Fast Iteration
- **Cloud**: 1x A100 40 GB for all stages
- **Training Time**: 1 day total
- **Cost**: $50-75 (cloud rental)

### Production / Multi-GPU
- **Cloud**: 2-4x A100 40 GB with distributed training
- **Training Time**: 6-12 hours total
- **Cost**: $100-150 (cloud rental)

### Inference-Only Deployment
- **Edge**: RTX 3060 (12 GB) or better
- **Cloud**: T4 instances ($0.35/hr) for API serving
- **Throughput**: 100+ hours audio/hour on RTX 3090
