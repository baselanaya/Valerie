# Valerie: Enhanced Visual ASR Language Model

## Overview

Valerie is a state-of-the-art visual automatic speech recognition (ASR) language model designed to improve upon existing lip reading technologies. The model implements a two-stage approach: **Video → Phonemes → Sentences**, incorporating four key architectural innovations that address fundamental limitations in current visual ASR systems.

## Key Innovations

### 1. 3D Spatio-Temporal Embedding
- **Problem**: Traditional 2D frame-by-frame processing misses temporal lip movement dynamics
- **Solution**: 3D CNN blocks capture coarticulation effects and lip movement transitions
- **Impact**: Better phoneme boundary detection and temporal sequence modeling

### 2. Conformer Architecture
- **Problem**: Vision Transformers lack local feature modeling capabilities
- **Solution**: Hybrid CNN-Transformer blocks combining local and global context
- **Impact**: Improved phoneme prediction accuracy through better feature representation

### 3. Hybrid CTC/Attention Training
- **Problem**: CTC-only training struggles with sequence modeling and alignment
- **Solution**: Joint CTC and attention-based training with shared encoder
- **Impact**: Better temporal alignment and sequence generation capabilities

### 4. Audio Knowledge Distillation
- **Problem**: Visual-only models lack the rich information available in audio signals
- **Solution**: Distill knowledge from pre-trained audio ASR models into visual encoder
- **Impact**: Improved accuracy while maintaining visual-only inference capability

## Architecture Overview

```
Input Video Frames (224x224x3)
         ↓
3D Spatio-Temporal Embedding
    (3D CNN + Temporal Embeddings)
         ↓
Conformer Encoder
    (CNN + Transformer Blocks)
         ↓
Hybrid CTC/Attention Head
    (Joint Training)
         ↓
Phoneme Sequences
         ↓
Fine-tuned LLM (LoRA)
         ↓
Final Sentences
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

### Setup
```bash
# Clone repository
git clone https://github.com/valerie-team/valerie-visual-asr.git
cd valerie-visual-asr

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Data Preparation
```bash
# Prepare VoxCeleb2 dataset
python scripts/prepare_data.py --dataset voxceleb2 --config configs/voxceleb2_config.yaml

# Prepare AVSpeech dataset
python scripts/prepare_data.py --dataset avspeech --config configs/avspeech_config.yaml
```

### 2. Training
```bash
# Train on VoxCeleb2
python scripts/train.py --config configs/voxceleb2_config.yaml --experiment-name voxceleb2_exp

# Train on AVSpeech
python scripts/train.py --config configs/avspeech_config.yaml --experiment-name avspeech_exp
```

### 3. Evaluation
```bash
# Evaluate trained model
python scripts/evaluate.py --config configs/base_config.yaml --checkpoint path/to/checkpoint.ckpt
```

### 4. Inference
```bash
# Run inference on video file
python scripts/inference.py --checkpoint path/to/checkpoint.ckpt --input video.mp4

# Real-time inference
python scripts/inference.py --checkpoint path/to/checkpoint.ckpt --input camera --real-time
```

## Multi-GPU Training

Valerie supports three different multi-GPU training methods optimized for different use cases:

### 1. PyTorch Native DDP (Recommended for Single Machine)
**Best performance for 2-8 GPUs on single machine**

```bash
# Use all available GPUs
python train_multi_gpu.py --gpus all --batch-size 8

# Use specific GPUs
python train_multi_gpu.py --gpus 0,1,2,3 --batch-size 8 --epochs 50

# With mixed precision
python train_multi_gpu.py --gpus 0,1 --batch-size 8 --mixed-precision fp16
```

**Advantages:**
- ✅ **Fastest**: Direct NCCL communication, minimal overhead
- ✅ **Most Stable**: Battle-tested by PyTorch community  
- ✅ **No Dependencies**: Built into PyTorch
- ✅ **95%+ Scaling Efficiency**

### 2. Accelerate (Easiest to Use)
**Best for research and rapid prototyping**

```bash
# Install Accelerate
pip install accelerate

# Use Accelerate for training
python train_multi_gpu.py --accelerate --batch-size 8 --mixed-precision fp16

# With advanced features
python train_multi_gpu.py --accelerate --batch-size 4 --epochs 100
```

**Advantages:**
- ✅ **User-Friendly**: Minimal code changes required
- ✅ **Feature-Rich**: Built-in mixed precision, logging, etc.
- ✅ **Great Documentation**: Excellent examples and guides
- ✅ **Active Development**: Regular updates and improvements

### 3. Alternative: PyTorch Lightning (Optional)
**For those who prefer high-level abstractions**

```bash
# Install PyTorch Lightning
pip install pytorch-lightning

# Lightning provides similar functionality to Accelerate
# with additional features for research workflows
```

**When to Consider:**
- 🔬 **Research-Focused**: Built for ML research workflows
- 📊 **Experiment Tracking**: Built-in logging and monitoring
- 🎛️ **High-Level API**: Abstracts away training loops
- ⚠️ **Learning Curve**: Requires understanding Lightning patterns

### Performance Comparison

| Method | Setup | Performance | Scaling | Memory | Best For |
|--------|-------|-------------|---------|---------|----------|
| **PyTorch DDP** | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Single machine |
| **Accelerate** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Research/prototyping |
| **Lightning** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Research workflows |

### Configuration Examples

```yaml
# configs/multi_gpu_config.yaml
training_method: "ddp"  # ddp, accelerate

gpu:
  gpu_ids: [0, 1, 2, 3]  # Specific GPUs or null for all
  mixed_precision: "fp16"

training:
  batch_size: 8  # Per GPU
  learning_rate: 1e-4
  max_epochs: 50

ddp:
  backend: "nccl"
  find_unused_parameters: false
  bucket_cap_mb: 25
```

## Configuration

The system uses YAML configuration files for easy customization:

- `configs/base_config.yaml`: Base configuration with default parameters
- `configs/voxceleb2_config.yaml`: VoxCeleb2-specific settings
- `configs/avspeech_config.yaml`: AVSpeech-specific settings

Key configuration sections:
- **model**: Architecture parameters (Conformer layers, embedding dimensions, etc.)
- **training**: Training hyperparameters (learning rate, batch size, loss weights)
- **data**: Dataset paths and preprocessing parameters
- **distillation**: Knowledge distillation settings

## Datasets

### VoxCeleb2
- **Scale**: 1M+ utterances from 6,112 celebrities
- **Diversity**: Wide range of speakers, accents, and recording conditions
- **Usage**: Primary training dataset for speaker diversity

### AVSpeech
- **Scale**: 270k training segments, 22k test segments
- **Quality**: Clean audio-visual segments with face coordinates
- **Usage**: High-quality segments for model validation

## Model Architecture

### Technical Specifications
- **Encoder**: 12-layer Conformer with 8 attention heads
- **3D CNN**: 4-layer network with residual connections
- **Vocabulary**: 40 phonemes (39 English phonemes + blank token)
- **Parameters**: ~500M total parameters
- **Training**: Mixed precision (FP16) with gradient checkpointing

### Performance Targets
- **Primary Goal**: Achieve better than 18.7% WER on standard benchmarks
- **Efficiency**: Maintain data efficiency with <1% additional labeled data
- **Speed**: Real-time inference capability on modern GPUs
- **Robustness**: Handle diverse speakers, lighting conditions, and poses

## Hardware Requirements

### Architecture Overview for Memory Planning

**Complete Model Stack:**
- **3D Spatio-Temporal CNN**: ~50M parameters, 3D convolutions on video frames
- **Conformer Encoder (12 layers)**: ~300M parameters, attention + convolution blocks
- **Hybrid CTC/Attention Head**: ~20M parameters, dual prediction heads
- **Whisper Large V3 Teacher**: ~1550M parameters (frozen during distillation)
- **Qwen3-0.6B LLM**: ~600M parameters + LoRA adapters (~2M)
- **Cross-Modal Alignment**: ~10M parameters for visual-audio projection
- **Total Trainable**: ~980M parameters (~3.9GB FP32, ~2GB FP16)

### Distributed Training Architecture

**Ray + Accelerate Integration:**
- **Data Sharding**: Distribute VoxCeleb2 (119GB) + AVSpeech (~500GB) across nodes
- **Model Sharding**: Split large components (Conformer, Whisper) across GPUs
- **Pipeline Parallelism**: Stage-wise execution (3D CNN → Conformer → CTC/Attention)
- **Mixed Precision**: FP16 training with gradient scaling
- **Gradient Accumulation**: Simulate large batch sizes with limited memory

### Single Node Setups

#### Development/Prototyping (Single GPU)
- **GPU**: 1x NVIDIA RTX 4090 (24GB VRAM) or A6000 (48GB)
- **CPU**: 16+ cores (Intel i9-13900K or AMD Ryzen 9 7950X)
- **RAM**: 64GB DDR4/DDR5
- **Storage**: 2TB NVMe SSD (Gen4 recommended)
- **Network**: 1Gbps minimum for dataset streaming
- **Training Time**: ~3-4 weeks (with gradient checkpointing)
- **Limitations**: Batch size ≤4, model sharding required

#### Research/Small Team (Dual GPU)
- **GPU**: 2x NVIDIA RTX 4090 (24GB each) or 2x A100 (40GB each)
- **CPU**: 24+ cores (Intel Xeon W or AMD Threadripper)
- **RAM**: 128GB DDR4/DDR5
- **Storage**: 4TB NVMe SSD RAID0 + 8TB HDD backup
- **Network**: 10Gbps for efficient data loading
- **Training Time**: ~1.5-2 weeks
- **Features**: Data parallelism, larger batch sizes (8-16)

#### Production/Large Team (Multi-GPU)
- **GPU**: 4x NVIDIA A100 (80GB each) or 8x A100 (40GB each)
- **CPU**: 64+ cores (Intel Xeon Platinum or AMD EPYC)
- **RAM**: 256GB+ DDR4 ECC
- **Storage**: 8TB NVMe SSD RAID0 + 16TB enterprise HDD
- **Network**: 25Gbps+ with InfiniBand for multi-node
- **Training Time**: ~4-7 days
- **Features**: Full pipeline parallelism, model sharding, large batch sizes (32-64)

### Multi-Node Distributed Training

#### Ray Cluster Configuration
```yaml
# 4-Node A100 Cluster Example
head_node:
  gpu: 8x A100 80GB
  cpu: 128 cores (2x AMD EPYC 7742)
  ram: 1TB DDR4 ECC
  storage: 16TB NVMe SSD
  network: 100Gbps InfiniBand

worker_nodes: 3x
  gpu: 8x A100 80GB each
  cpu: 128 cores each
  ram: 1TB each
  storage: 8TB NVMe SSD each
  network: 100Gbps InfiniBand

total_resources:
  gpus: 32x A100 80GB (2.56TB total VRAM)
  cores: 512 CPU cores
  ram: 4TB total RAM
  training_time: ~1-2 days
```

### Memory Requirements Breakdown

#### Training Memory per GPU (FP16)
- **Model Parameters**: ~2GB (with model sharding: ~500MB per GPU)
- **Optimizer States**: ~4GB (Adam: 2x model size)
- **Gradients**: ~2GB (same as model parameters)
- **Activations**: ~8-16GB (depends on batch size and sequence length)
- **3D CNN Activations**: ~4-8GB (video frames + temporal features)
- **Whisper Cache**: ~2-4GB (frozen teacher model features)
- **Data Buffers**: ~2-4GB (batch loading and augmentation)
- **Total per GPU**: ~24-40GB (requires 40GB+ VRAM for safety)

#### Data Storage Requirements
- **VoxCeleb2 Processed**: ~300GB (video frames + audio + transcripts)
- **AVSpeech Processed**: ~500GB (video segments + face coordinates)
- **Phoneme Alignments**: ~50GB (MFA outputs and CTC labels)
- **Model Checkpoints**: ~100GB (multiple training stages)
- **Logs and Metrics**: ~20GB (tensorboard, wandb, training logs)
- **Working Space**: ~200GB (temporary files, data preprocessing)
- **Total Storage**: ~1.2TB minimum, 2TB recommended

### Cloud Computing Options

#### AWS (Optimized for Ray)
- **p4d.24xlarge**: 8x A100 40GB, 96 vCPUs, 1.1TB RAM (~$32/hour)
- **p4de.24xlarge**: 8x A100 80GB, 96 vCPUs, 1.1TB RAM (~$40/hour)
- **Multi-node setup**: 4x p4de.24xlarge (~$160/hour, ~$3,840/day)
- **Storage**: Amazon FSx for Lustre (high-performance parallel filesystem)
- **Network**: 400Gbps network performance, EFA for MPI

#### Google Cloud Platform
- **a2-ultragpu-8g**: 8x A100 40GB, 96 vCPUs, 1.4TB RAM (~$30/hour)
- **a2-megagpu-16g**: 16x A100 40GB, 96 vCPUs, 1.4TB RAM (~$55/hour)
- **Multi-node setup**: 2x a2-megagpu-16g (~$110/hour, ~$2,640/day)
- **Storage**: Google Cloud Filestore or Persistent Disk SSD
- **Network**: Up to 100Gbps with GPUDirect-RDMA

#### Microsoft Azure
- **ND96amsr_A100_v4**: 8x A100 80GB, 96 vCPUs, 1.9TB RAM (~$35/hour)
- **ND96isr_H100_v5**: 8x H100 80GB, 96 vCPUs, 1.9TB RAM (~$45/hour)
- **Multi-node setup**: 4x ND96amsr_A100_v4 (~$140/hour, ~$3,360/day)
- **Storage**: Azure NetApp Files or Premium SSD
- **Network**: 200Gbps InfiniBand with SR-IOV

### Performance Optimization Strategies

#### Ray Configuration
```python
# Ray cluster optimization
ray.init(
    address="ray://head-node:10001",
    runtime_env={
        "pip": ["accelerate", "transformers", "datasets"],
        "env_vars": {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
    }
)

# Data sharding across nodes
@ray.remote(num_gpus=1)
class DataWorker:
    def load_shard(self, shard_id): 
        # Load VoxCeleb2/AVSpeech shard
        pass
```

#### Accelerate Integration
```python
# Multi-GPU training with model sharding
accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=4,
    dataloader_config=DataLoaderConfiguration(
        split_batches=True,
        dispatch_batches=True
    )
)

# Model sharding for large components
device_map = {
    "spatio_temporal": 0,
    "conformer.layers.0-5": 1,
    "conformer.layers.6-11": 2,
    "ctc_attention": 3,
    "whisper_teacher": 4,
    "qwen_llm": [5, 6, 7]  # Multi-GPU for LLM
}
```

#### Training Optimizations
- **Gradient Checkpointing**: Reduce memory by 50% at 20% speed cost
- **Mixed Precision**: FP16 training reduces memory by 40-50%
- **Data Pipeline**: Prefetch with 4-8 workers per GPU
- **Model Parallelism**: Split Conformer layers across GPUs
- **Dynamic Batching**: Variable sequence lengths with smart padding

### Estimated Training Costs

| Configuration | Time | AWS Cost | GCP Cost | Azure Cost |
|---------------|------|----------|----------|------------|
| Single RTX 4090 | 3-4 weeks | N/A | N/A | N/A |
| 2x A100 40GB | 1-2 weeks | ~$10,000 | ~$8,000 | ~$9,000 |
| 4x A100 80GB | 4-7 days | ~$15,000 | ~$12,000 | ~$13,000 |
| 8x H100 80GB | 2-3 days | ~$20,000 | ~$18,000 | ~$19,000 |

*Costs include compute, storage, and network transfer. Spot instances can reduce costs by 60-80%.*

### Monitoring and Profiling

#### Resource Monitoring
- **Ray Dashboard**: Cluster utilization, task scheduling
- **NVIDIA-SMI**: GPU utilization, memory usage
- **Weights & Biases**: Training metrics, system monitoring
- **TensorBoard**: Loss curves, gradient norms
- **Prometheus + Grafana**: Infrastructure monitoring

#### Performance Profiling
- **PyTorch Profiler**: Identify bottlenecks
- **NVIDIA Nsight**: GPU kernel optimization  
- **Ray Memory Profiler**: Memory usage across nodes
- **Data Loading Profiler**: I/O bottleneck detection

## Development

### Project Structure
```
valerie-visual-asr/
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── data/              # Data loading and preprocessing
│   ├── training/          # Training pipeline
│   ├── inference/         # Inference and decoding
│   └── utils/             # Utilities and configuration
├── configs/               # Configuration files
├── scripts/               # Command-line scripts
├── tests/                 # Unit and integration tests
└── notebooks/             # Analysis and visualization
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_config.py

# Run with coverage
pytest --cov=src tests/
```

### Code Style
```bash
# Format code
black src/ tests/ scripts/

# Check style
flake8 src/ tests/ scripts/

# Type checking
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{valerie2024,
  title={Valerie: Enhanced Visual ASR Language Model with Conformer Architecture and Knowledge Distillation},
  author={[Your Name]},
  journal={arXiv preprint},
  year={2024}
}
```

## Acknowledgments

- VoxCeleb2 dataset creators for providing diverse speaker data
- AVSpeech dataset team for clean audio-visual segments
- Conformer architecture developers for the hybrid CNN-Transformer design
- Open source community for PyTorch and related libraries

## Contact

For questions and support, please open an issue on GitHub or contact us at contact@valerie-asr.com.