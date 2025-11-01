#!/usr/bin/env python3
"""
Training Requirements Calculator for Valerie Visual ASR
======================================================

Calculate training time, memory requirements, and resource needs based on:
- Dataset size (VoxCeleb2 dev: 331GB, test: 11GB)
- Model parameters (~500M parameters)
- Hardware configuration
- Training settings

Based on your actual VoxCeleb2 dataset:
- Dev folder: 331 GB (355,947,559,822 bytes) - 3,312,264 files, 459,892 folders
- Test folder: 11 GB (11,882,803,533 bytes) - 108,711 files, 15,090 folders
"""

import argparse
import math
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

@dataclass
class DatasetStats:
    """Dataset statistics."""
    name: str
    size_gb: float
    size_bytes: int
    num_files: int
    num_folders: int
    estimated_samples: int
    estimated_hours: float

@dataclass
class ModelStats:
    """Model parameter statistics."""
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    model_size_mb: float
    memory_per_sample_mb: float

@dataclass
class HardwareConfig:
    """Hardware configuration."""
    num_gpus: int
    gpu_memory_gb: int
    gpu_name: str
    cpu_cores: int
    ram_gb: int
    storage_type: str  # SSD, NVMe, HDD

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size_per_gpu: int
    sequence_length: int
    num_epochs: int
    mixed_precision: bool
    gradient_accumulation: int
    num_workers_per_gpu: int

class TrainingCalculator:
    """Calculate training requirements and estimates."""
    
    def __init__(self):
        # VoxCeleb2 dataset statistics (from user's data)
        self.voxceleb2_dev = DatasetStats(
            name="VoxCeleb2 Dev",
            size_gb=331.0,
            size_bytes=355_947_559_822,
            num_files=3_312_264,
            num_folders=459_892,
            estimated_samples=1_092_009,  # Based on VoxCeleb2 paper
            estimated_hours=2300.0  # Estimated from dataset size
        )
        
        self.voxceleb2_test = DatasetStats(
            name="VoxCeleb2 Test", 
            size_gb=11.0,
            size_bytes=11_882_803_533,
            num_files=108_711,
            num_folders=15_090,
            estimated_samples=36_237,  # Based on VoxCeleb2 paper
            estimated_hours=118.0  # Estimated from dataset size
        )
        
        # Valerie model statistics
        self.model = ModelStats(
            total_parameters=500_000_000,  # ~500M parameters
            trainable_parameters=500_000_000,
            frozen_parameters=0,
            model_size_mb=2000.0,  # ~2GB in FP32
            memory_per_sample_mb=50.0  # Estimated memory per sample
        )
    
    def calculate_memory_requirements(
        self, 
        hardware: HardwareConfig,
        training: TrainingConfig
    ) -> Dict[str, float]:
        """Calculate memory requirements."""
        
        # Model memory (FP32)
        model_memory_mb = self.model.model_size_mb
        
        # Mixed precision reduces model memory by ~50%
        if training.mixed_precision:
            model_memory_mb *= 0.5
        
        # Optimizer memory (AdamW needs ~3x model parameters)
        optimizer_memory_mb = model_memory_mb * 3
        
        # Gradient memory (same as model)
        gradient_memory_mb = model_memory_mb
        
        # Batch memory
        batch_memory_mb = (
            training.batch_size_per_gpu * 
            self.model.memory_per_sample_mb *
            training.sequence_length / 150  # Normalize to 150 frames
        )
        
        # Total memory per GPU
        total_memory_per_gpu_mb = (
            model_memory_mb + 
            optimizer_memory_mb + 
            gradient_memory_mb + 
            batch_memory_mb +
            1000  # Buffer for CUDA overhead
        )
        
        # DDP communication buffers (additional 10-20%)
        if hardware.num_gpus > 1:
            total_memory_per_gpu_mb *= 1.15
        
        return {
            'model_memory_mb': model_memory_mb,
            'optimizer_memory_mb': optimizer_memory_mb,
            'gradient_memory_mb': gradient_memory_mb,
            'batch_memory_mb': batch_memory_mb,
            'total_per_gpu_mb': total_memory_per_gpu_mb,
            'total_per_gpu_gb': total_memory_per_gpu_mb / 1024,
            'total_all_gpus_gb': (total_memory_per_gpu_mb * hardware.num_gpus) / 1024,
            'memory_utilization': total_memory_per_gpu_mb / (hardware.gpu_memory_gb * 1024)
        }
    
    def calculate_training_time(
        self,
        hardware: HardwareConfig,
        training: TrainingConfig,
        dataset: DatasetStats
    ) -> Dict[str, float]:
        """Calculate training time estimates."""
        
        # Samples per epoch
        samples_per_epoch = dataset.estimated_samples
        
        # Effective batch size
        effective_batch_size = (
            training.batch_size_per_gpu * 
            hardware.num_gpus * 
            training.gradient_accumulation
        )
        
        # Steps per epoch
        steps_per_epoch = math.ceil(samples_per_epoch / effective_batch_size)
        
        # Estimate time per step based on hardware
        if "H100" in hardware.gpu_name:
            base_time_per_step = 0.8  # seconds
        elif "A100" in hardware.gpu_name:
            base_time_per_step = 1.2  # seconds
        elif "RTX 4090" in hardware.gpu_name:
            base_time_per_step = 1.8  # seconds
        elif "RTX 3090" in hardware.gpu_name:
            base_time_per_step = 2.5  # seconds
        else:
            base_time_per_step = 3.0  # seconds (conservative)
        
        # Adjust for batch size and sequence length
        time_per_step = base_time_per_step * (
            training.batch_size_per_gpu / 8 *  # Normalize to batch size 8
            training.sequence_length / 150     # Normalize to 150 frames
        )
        
        # Multi-GPU efficiency (not perfectly linear)
        if hardware.num_gpus > 1:
            efficiency = min(0.95, 0.85 + 0.1 * math.log2(hardware.num_gpus))
            time_per_step /= (hardware.num_gpus * efficiency)
        
        # Time calculations
        time_per_epoch_hours = (steps_per_epoch * time_per_step) / 3600
        total_training_hours = time_per_epoch_hours * training.num_epochs
        total_training_days = total_training_hours / 24
        
        return {
            'samples_per_epoch': samples_per_epoch,
            'effective_batch_size': effective_batch_size,
            'steps_per_epoch': steps_per_epoch,
            'time_per_step_seconds': time_per_step,
            'time_per_epoch_hours': time_per_epoch_hours,
            'total_training_hours': total_training_hours,
            'total_training_days': total_training_days,
            'gpu_efficiency': efficiency if hardware.num_gpus > 1 else 1.0
        }
    
    def calculate_storage_requirements(
        self,
        training: TrainingConfig,
        dataset: DatasetStats
    ) -> Dict[str, float]:
        """Calculate storage requirements."""
        
        # Dataset storage (already downloaded)
        dataset_storage_gb = dataset.size_gb
        
        # Preprocessing cache (estimated)
        preprocessing_cache_gb = dataset_storage_gb * 0.3
        
        # Model checkpoints (save every epoch)
        checkpoint_size_gb = self.model.model_size_mb / 1024 * 2  # Model + optimizer
        total_checkpoint_storage_gb = checkpoint_size_gb * min(training.num_epochs, 10)  # Keep max 10
        
        # Logs and metrics
        logs_storage_gb = 1.0  # Conservative estimate
        
        # Total storage needed
        total_storage_gb = (
            dataset_storage_gb + 
            preprocessing_cache_gb + 
            total_checkpoint_storage_gb + 
            logs_storage_gb
        )
        
        return {
            'dataset_storage_gb': dataset_storage_gb,
            'preprocessing_cache_gb': preprocessing_cache_gb,
            'checkpoint_storage_gb': total_checkpoint_storage_gb,
            'logs_storage_gb': logs_storage_gb,
            'total_storage_gb': total_storage_gb,
            'recommended_free_space_gb': total_storage_gb * 1.5  # 50% buffer
        }
    
    def calculate_costs(
        self,
        training_time_hours: float,
        hardware: HardwareConfig
    ) -> Dict[str, float]:
        """Calculate training costs for cloud providers."""
        
        # Cloud pricing per GPU-hour (approximate, as of 2024)
        pricing = {
            'H100': {'aws': 4.0, 'gcp': 3.5, 'azure': 3.8},
            'A100_80GB': {'aws': 3.2, 'gcp': 2.8, 'azure': 3.0},
            'A100_40GB': {'aws': 2.4, 'gcp': 2.0, 'azure': 2.2},
            'RTX_4090': {'aws': 1.5, 'gcp': 1.3, 'azure': 1.4},  # If available
            'default': {'aws': 2.0, 'gcp': 1.8, 'azure': 1.9}
        }
        
        # Determine GPU type for pricing
        gpu_type = 'default'
        if 'H100' in hardware.gpu_name:
            gpu_type = 'H100'
        elif 'A100' in hardware.gpu_name:
            if '80GB' in hardware.gpu_name:
                gpu_type = 'A100_80GB'
            else:
                gpu_type = 'A100_40GB'
        elif 'RTX 4090' in hardware.gpu_name:
            gpu_type = 'RTX_4090'
        
        # Calculate costs
        costs = {}
        for provider, price_per_gpu_hour in pricing[gpu_type].items():
            total_cost = training_time_hours * hardware.num_gpus * price_per_gpu_hour
            costs[f'{provider}_cost_usd'] = total_cost
            costs[f'{provider}_cost_with_spot_usd'] = total_cost * 0.3  # 70% discount for spot instances
        
        return costs
    
    def generate_report(
        self,
        hardware: HardwareConfig,
        training: TrainingConfig,
        use_test_set: bool = False
    ) -> Dict:
        """Generate comprehensive training report."""
        
        dataset = self.voxceleb2_test if use_test_set else self.voxceleb2_dev
        
        # Calculate all requirements
        memory_reqs = self.calculate_memory_requirements(hardware, training)
        training_time = self.calculate_training_time(hardware, training, dataset)
        storage_reqs = self.calculate_storage_requirements(training, dataset)
        costs = self.calculate_costs(training_time['total_training_hours'], hardware)
        
        # Check feasibility
        feasible = memory_reqs['memory_utilization'] < 0.9  # <90% memory usage
        
        return {
            'dataset': asdict(dataset),
            'model': asdict(self.model),
            'hardware': asdict(hardware),
            'training_config': asdict(training),
            'memory_requirements': memory_reqs,
            'training_time': training_time,
            'storage_requirements': storage_reqs,
            'estimated_costs': costs,
            'feasible': feasible,
            'recommendations': self._generate_recommendations(memory_reqs, training_time, hardware, training)
        }
    
    def _generate_recommendations(
        self,
        memory_reqs: Dict,
        training_time: Dict,
        hardware: HardwareConfig,
        training: TrainingConfig
    ) -> List[str]:
        """Generate optimization recommendations."""
        
        recommendations = []
        
        # Memory recommendations
        if memory_reqs['memory_utilization'] > 0.9:
            recommendations.append("⚠️ High memory usage. Consider reducing batch size or enabling gradient checkpointing")
        
        if memory_reqs['memory_utilization'] > 1.0:
            recommendations.append("❌ Insufficient GPU memory. Reduce batch size or use gradient accumulation")
        
        # Performance recommendations
        if training_time['total_training_days'] > 7:
            recommendations.append("⏰ Long training time. Consider using more GPUs or reducing dataset size")
        
        if hardware.num_gpus > 1 and training_time['gpu_efficiency'] < 0.8:
            recommendations.append("📊 Low multi-GPU efficiency. Check data loading and communication overhead")
        
        # Configuration recommendations
        if not training.mixed_precision:
            recommendations.append("🚀 Enable mixed precision (FP16) to reduce memory usage by ~50%")
        
        if training.batch_size_per_gpu < 4:
            recommendations.append("📈 Small batch size may reduce training efficiency. Consider gradient accumulation")
        
        if training.num_workers_per_gpu < 4:
            recommendations.append("💾 Increase data loading workers to prevent I/O bottlenecks")
        
        return recommendations


def main():
    """Main function to run calculations."""
    
    parser = argparse.ArgumentParser(description="Calculate Valerie training requirements")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--gpu-memory", type=int, default=24, help="GPU memory in GB")
    parser.add_argument("--gpu-name", type=str, default="RTX 4090", help="GPU name")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--use-test", action="store_true", help="Use test set instead of dev set")
    parser.add_argument("--output", type=str, help="Output JSON file")
    
    args = parser.parse_args()
    
    # Create configurations
    hardware = HardwareConfig(
        num_gpus=args.gpus,
        gpu_memory_gb=args.gpu_memory,
        gpu_name=args.gpu_name,
        cpu_cores=args.gpus * 8,  # Estimate 8 cores per GPU
        ram_gb=args.gpus * 32,    # Estimate 32GB RAM per GPU
        storage_type="NVMe"
    )
    
    training = TrainingConfig(
        batch_size_per_gpu=args.batch_size,
        sequence_length=150,  # Standard sequence length
        num_epochs=args.epochs,
        mixed_precision=True,
        gradient_accumulation=1,
        num_workers_per_gpu=4
    )
    
    # Generate report
    calculator = TrainingCalculator()
    report = calculator.generate_report(hardware, training, args.use_test)
    
    # Print report
    print("🔍 VALERIE VISUAL ASR - TRAINING REQUIREMENTS ANALYSIS")
    print("=" * 80)
    
    print(f"\n📊 DATASET: {report['dataset']['name']}")
    print(f"   Size: {report['dataset']['size_gb']:.1f} GB")
    print(f"   Files: {report['dataset']['num_files']:,}")
    print(f"   Estimated samples: {report['dataset']['estimated_samples']:,}")
    print(f"   Estimated hours: {report['dataset']['estimated_hours']:.1f}")
    
    print(f"\n🏗️ MODEL: Valerie Visual ASR")
    print(f"   Parameters: {report['model']['total_parameters']:,}")
    print(f"   Model size: {report['model']['model_size_mb']:.1f} MB")
    
    print(f"\n🎮 HARDWARE: {hardware.gpu_name}")
    print(f"   GPUs: {hardware.num_gpus}")
    print(f"   GPU Memory: {hardware.gpu_memory_gb} GB each")
    print(f"   Total GPU Memory: {hardware.num_gpus * hardware.gpu_memory_gb} GB")
    
    print(f"\n⚙️ TRAINING CONFIG:")
    print(f"   Batch size per GPU: {training.batch_size_per_gpu}")
    print(f"   Total batch size: {report['training_time']['effective_batch_size']}")
    print(f"   Epochs: {training.num_epochs}")
    print(f"   Mixed precision: {training.mixed_precision}")
    
    print(f"\n💾 MEMORY REQUIREMENTS:")
    memory = report['memory_requirements']
    print(f"   Model memory: {memory['model_memory_mb']:.1f} MB")
    print(f"   Optimizer memory: {memory['optimizer_memory_mb']:.1f} MB")
    print(f"   Batch memory: {memory['batch_memory_mb']:.1f} MB")
    print(f"   Total per GPU: {memory['total_per_gpu_gb']:.1f} GB")
    print(f"   Memory utilization: {memory['memory_utilization']:.1%}")
    
    print(f"\n⏱️ TRAINING TIME:")
    time_est = report['training_time']
    print(f"   Steps per epoch: {time_est['steps_per_epoch']:,}")
    print(f"   Time per epoch: {time_est['time_per_epoch_hours']:.1f} hours")
    print(f"   Total training time: {time_est['total_training_hours']:.1f} hours ({time_est['total_training_days']:.1f} days)")
    if hardware.num_gpus > 1:
        print(f"   Multi-GPU efficiency: {time_est['gpu_efficiency']:.1%}")
    
    print(f"\n💿 STORAGE REQUIREMENTS:")
    storage = report['storage_requirements']
    print(f"   Dataset: {storage['dataset_storage_gb']:.1f} GB")
    print(f"   Preprocessing cache: {storage['preprocessing_cache_gb']:.1f} GB")
    print(f"   Checkpoints: {storage['checkpoint_storage_gb']:.1f} GB")
    print(f"   Total needed: {storage['total_storage_gb']:.1f} GB")
    print(f"   Recommended free space: {storage['recommended_free_space_gb']:.1f} GB")
    
    print(f"\n💰 ESTIMATED CLOUD COSTS:")
    costs = report['estimated_costs']
    print(f"   AWS (on-demand): ${costs['aws_cost_usd']:,.0f}")
    print(f"   AWS (spot): ${costs['aws_cost_with_spot_usd']:,.0f}")
    print(f"   GCP (on-demand): ${costs['gcp_cost_usd']:,.0f}")
    print(f"   GCP (spot): ${costs['gcp_cost_with_spot_usd']:,.0f}")
    
    print(f"\n✅ FEASIBILITY: {'✅ FEASIBLE' if report['feasible'] else '❌ NOT FEASIBLE'}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"   {rec}")
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Report saved to: {args.output}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
