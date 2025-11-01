"""
Model Analysis and Architecture Visualization for Valerie Visual ASR.

Provides tools for:
- Model architecture diagram generation
- Parameter analysis and distribution
- Layer-wise computational analysis
- Model comparison and profiling
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import graphviz
from torchviz import make_dot
from torchsummary import summary
import logging
from collections import defaultdict

from src.models import ValerieModel
from src.utils.config import Config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelAnalyzer:
    """
    Comprehensive model analysis and visualization.
    
    Provides detailed analysis of model architecture, parameters,
    and computational characteristics.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.model_info = self._analyze_model()
    
    def _analyze_model(self) -> Dict[str, Any]:
        """Analyze model structure and parameters."""
        
        info = {
            'total_parameters': 0,
            'trainable_parameters': 0,
            'layers': [],
            'layer_types': defaultdict(int),
            'parameter_distribution': {}
        }
        
        # Analyze parameters
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            info['total_parameters'] += num_params
            
            if param.requires_grad:
                info['trainable_parameters'] += num_params
            
            # Store layer info
            layer_info = {
                'name': name,
                'shape': list(param.shape),
                'num_parameters': num_params,
                'trainable': param.requires_grad,
                'dtype': str(param.dtype)
            }
            info['layers'].append(layer_info)
            
            # Count layer types
            layer_type = name.split('.')[0] if '.' in name else name
            info['layer_types'][layer_type] += num_params
        
        return info
    
    def print_model_summary(self):
        """Print comprehensive model summary."""
        
        logger.info("🏗️ Valerie Model Analysis")
        logger.info("=" * 60)
        
        # Basic info
        logger.info(f"Total Parameters: {self.model_info['total_parameters']:,}")
        logger.info(f"Trainable Parameters: {self.model_info['trainable_parameters']:,}")
        logger.info(f"Non-trainable Parameters: {self.model_info['total_parameters'] - self.model_info['trainable_parameters']:,}")
        
        # Model size
        model_size_mb = self.model_info['total_parameters'] * 4 / (1024 * 1024)  # Assume float32
        logger.info(f"Model Size: {model_size_mb:.1f} MB")
        
        logger.info("\n📊 Component Analysis:")
        
        # Component breakdown
        total_params = self.model_info['total_parameters']
        for component, num_params in sorted(self.model_info['layer_types'].items(), 
                                          key=lambda x: x[1], reverse=True):
            percentage = (num_params / total_params) * 100
            logger.info(f"   {component}: {num_params:,} ({percentage:.1f}%)")
        
        # Layer details (top 10 largest)
        logger.info("\n🔍 Largest Layers:")
        sorted_layers = sorted(self.model_info['layers'], 
                              key=lambda x: x['num_parameters'], reverse=True)
        
        for i, layer in enumerate(sorted_layers[:10]):
            percentage = (layer['num_parameters'] / total_params) * 100
            logger.info(f"   {i+1}. {layer['name']}: {layer['num_parameters']:,} ({percentage:.1f}%)")
    
    def plot_parameter_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot parameter distribution across model components.
        
        Args:
            save_path: Path to save plot
            
        Returns:
            matplotlib Figure
        """
        
        # Prepare data
        components = list(self.model_info['layer_types'].keys())
        param_counts = list(self.model_info['layer_types'].values())
        
        # Create pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
        wedges, texts, autotexts = ax1.pie(param_counts, labels=components, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        
        ax1.set_title('Parameter Distribution by Component')
        
        # Bar chart
        bars = ax2.bar(range(len(components)), param_counts, color=colors)
        ax2.set_title('Parameter Count by Component')
        ax2.set_xlabel('Component')
        ax2.set_ylabel('Number of Parameters')
        ax2.set_xticks(range(len(components)))
        ax2.set_xticklabels(components, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, param_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}', ha='center', va='bottom', rotation=90)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Parameter distribution plot saved to {save_path}")
        
        return fig
    
    def analyze_layer_shapes(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Analyze and visualize layer shapes and dimensions.
        
        Args:
            save_path: Path to save plot
            
        Returns:
            matplotlib Figure
        """
        
        # Extract layer dimensions
        layer_names = []
        layer_dims = []
        layer_params = []
        
        for layer in self.model_info['layers']:
            if len(layer['shape']) > 1:  # Skip scalar parameters
                layer_names.append(layer['name'].split('.')[-1])  # Get last part of name
                layer_dims.append(len(layer['shape']))
                layer_params.append(layer['num_parameters'])
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Dimension distribution
        dim_counts = {}
        for dim in layer_dims:
            dim_counts[dim] = dim_counts.get(dim, 0) + 1
        
        dims = list(dim_counts.keys())
        counts = list(dim_counts.values())
        
        bars1 = ax1.bar(dims, counts, color='skyblue', alpha=0.7)
        ax1.set_title('Distribution of Layer Dimensions')
        ax1.set_xlabel('Number of Dimensions')
        ax1.set_ylabel('Number of Layers')
        ax1.set_xticks(dims)
        
        # Add value labels
        for bar, count in zip(bars1, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')
        
        # Parameter count by dimension
        dim_params = defaultdict(int)
        for dim, params in zip(layer_dims, layer_params):
            dim_params[dim] += params
        
        dims_sorted = sorted(dim_params.keys())
        params_by_dim = [dim_params[d] for d in dims_sorted]
        
        bars2 = ax2.bar(dims_sorted, params_by_dim, color='lightcoral', alpha=0.7)
        ax2.set_title('Parameters by Layer Dimension')
        ax2.set_xlabel('Number of Dimensions')
        ax2.set_ylabel('Total Parameters')
        ax2.set_xticks(dims_sorted)
        
        # Add value labels
        for bar, params in zip(bars2, params_by_dim):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{params:,}', ha='center', va='bottom', rotation=90)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Layer shapes analysis saved to {save_path}")
        
        return fig
    
    def create_architecture_diagram(
        self,
        input_shape: Tuple[int, ...] = (2, 3, 16, 224, 224),
        save_path: Optional[str] = None
    ) -> Optional[graphviz.Digraph]:
        """
        Create visual architecture diagram.
        
        Args:
            input_shape: Input tensor shape for tracing
            save_path: Path to save diagram
            
        Returns:
            Graphviz diagram or None
        """
        
        try:
            # Create dummy input
            dummy_input = torch.randn(*input_shape)
            
            # Forward pass to create computation graph
            self.model.eval()
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    # For ValerieModel, we need to provide the right input format
                    if len(input_shape) == 5:  # [B, C, T, H, W]
                        input_lengths = torch.tensor([input_shape[2]] * input_shape[0])
                        output = self.model(dummy_input, input_lengths)
                    else:
                        output = self.model(dummy_input)
                else:
                    output = self.model(dummy_input)
            
            # Create visualization
            if isinstance(output, dict):
                # Take first output for visualization
                output_tensor = list(output.values())[0]
            else:
                output_tensor = output
            
            # Generate graph
            dot = make_dot(output_tensor, params=dict(self.model.named_parameters()),
                          show_attrs=True, show_saved=True)
            
            if save_path:
                dot.render(save_path, format='png', cleanup=True)
                logger.info(f"💾 Architecture diagram saved to {save_path}.png")
            
            return dot
            
        except Exception as e:
            logger.warning(f"⚠️ Could not create architecture diagram: {e}")
            return None
    
    def profile_model_performance(
        self,
        input_shape: Tuple[int, ...] = (1, 3, 16, 224, 224),
        num_runs: int = 10
    ) -> Dict[str, float]:
        """
        Profile model performance (inference time, memory usage).
        
        Args:
            input_shape: Input tensor shape
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary of performance metrics
        """
        
        device = next(self.model.parameters()).device
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape, device=device)
        
        # Warmup
        self.model.eval()
        with torch.no_grad():
            for _ in range(3):
                if len(input_shape) == 5:  # [B, C, T, H, W]
                    input_lengths = torch.tensor([input_shape[2]] * input_shape[0], device=device)
                    _ = self.model(dummy_input, input_lengths)
                else:
                    _ = self.model(dummy_input)
        
        # Measure inference time
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                if len(input_shape) == 5:
                    input_lengths = torch.tensor([input_shape[2]] * input_shape[0], device=device)
                    _ = self.model(dummy_input, input_lengths)
                else:
                    _ = self.model(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_inference_time = (end_time - start_time) / num_runs
        
        # Memory usage
        if device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
            memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # MB
        else:
            memory_allocated = 0
            memory_reserved = 0
        
        # Model size
        model_size_mb = sum(p.numel() * 4 for p in self.model.parameters()) / (1024 ** 2)
        
        performance = {
            'avg_inference_time_ms': avg_inference_time * 1000,
            'throughput_samples_per_sec': input_shape[0] / avg_inference_time,
            'memory_allocated_mb': memory_allocated,
            'memory_reserved_mb': memory_reserved,
            'model_size_mb': model_size_mb,
            'device': str(device)
        }
        
        logger.info("⚡ Model Performance Profile:")
        logger.info(f"   Average inference time: {performance['avg_inference_time_ms']:.2f} ms")
        logger.info(f"   Throughput: {performance['throughput_samples_per_sec']:.1f} samples/sec")
        logger.info(f"   Memory allocated: {performance['memory_allocated_mb']:.1f} MB")
        logger.info(f"   Model size: {performance['model_size_mb']:.1f} MB")
        
        return performance
    
    def compare_with_baseline(
        self,
        baseline_info: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare model with baseline architecture.
        
        Args:
            baseline_info: Baseline model information
            save_path: Path to save comparison plot
            
        Returns:
            matplotlib Figure
        """
        
        # Prepare comparison data
        metrics = ['total_parameters', 'trainable_parameters', 'model_size_mb']
        current_values = [
            self.model_info['total_parameters'],
            self.model_info['trainable_parameters'],
            self.model_info['total_parameters'] * 4 / (1024 ** 2)
        ]
        baseline_values = [
            baseline_info.get('total_parameters', 0),
            baseline_info.get('trainable_parameters', 0),
            baseline_info.get('model_size_mb', 0)
        ]
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, current_values, width, label='Current Model', color='skyblue')
        bars2 = ax.bar(x + width/2, baseline_values, width, label='Baseline Model', color='lightcoral')
        
        ax.set_title('Model Comparison')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels(['Total Params', 'Trainable Params', 'Size (MB)'])
        ax.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:,.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Model comparison saved to {save_path}")
        
        return fig


def analyze_model_components(
    model: nn.Module,
    save_dir: str = "model_analysis"
) -> Dict[str, Any]:
    """
    Convenience function to analyze model components.
    
    Args:
        model: PyTorch model to analyze
        save_dir: Directory to save analysis results
        
    Returns:
        Dictionary of analysis results
    """
    
    analyzer = ModelAnalyzer(model)
    
    # Print summary
    analyzer.print_model_summary()
    
    # Create visualizations
    from pathlib import Path
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Parameter distribution
    analyzer.plot_parameter_distribution(str(save_path / "parameter_distribution.png"))
    
    # Layer shapes
    analyzer.analyze_layer_shapes(str(save_path / "layer_shapes.png"))
    
    # Architecture diagram
    analyzer.create_architecture_diagram(save_path=str(save_path / "architecture"))
    
    # Performance profile
    performance = analyzer.profile_model_performance()
    
    return {
        'model_info': analyzer.model_info,
        'performance': performance
    }


def create_architecture_diagram(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 16, 224, 224),
    save_path: Optional[str] = None
) -> Optional[graphviz.Digraph]:
    """
    Convenience function to create architecture diagram.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        save_path: Path to save diagram
        
    Returns:
        Graphviz diagram or None
    """
    
    analyzer = ModelAnalyzer(model)
    return analyzer.create_architecture_diagram(input_shape, save_path)
