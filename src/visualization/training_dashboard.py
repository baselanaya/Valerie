"""
Training Progress Monitoring Dashboard for Valerie Visual ASR.

Real-time training monitoring with:
- Loss curves and metrics tracking
- Learning rate scheduling visualization
- Model performance analysis
- Interactive training dashboard
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import json
from pathlib import Path
import logging

from src.utils.logging import get_logger

logger = get_logger(__name__)


class MetricsTracker:
    """
    Track and store training metrics over time.
    
    Stores metrics history for visualization and analysis.
    """
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_per': [],
            'val_per': [],
            'train_wer': [],
            'val_wer': [],
            'train_bleu': [],
            'val_bleu': [],
            'learning_rate': [],
            'epoch': [],
            'step': [],
            'timestamp': []
        }
        
        self.best_metrics = {
            'best_val_loss': float('inf'),
            'best_val_per': float('inf'),
            'best_val_wer': float('inf'),
            'best_val_bleu': 0.0,
            'best_epoch': 0
        }
    
    def update(
        self,
        epoch: int,
        step: int,
        metrics_dict: Dict[str, float],
        learning_rate: Optional[float] = None
    ):
        """
        Update metrics with new values.
        
        Args:
            epoch: Current epoch
            step: Current step
            metrics_dict: Dictionary of metric values
            learning_rate: Current learning rate
        """
        
        # Add epoch, step, and timestamp
        self.metrics['epoch'].append(epoch)
        self.metrics['step'].append(step)
        self.metrics['timestamp'].append(datetime.now().isoformat())
        
        if learning_rate is not None:
            self.metrics['learning_rate'].append(learning_rate)
        
        # Update metrics
        for key, value in metrics_dict.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        # Update best metrics
        if 'val_loss' in metrics_dict:
            if metrics_dict['val_loss'] < self.best_metrics['best_val_loss']:
                self.best_metrics['best_val_loss'] = metrics_dict['val_loss']
                self.best_metrics['best_epoch'] = epoch
        
        if 'val_per' in metrics_dict:
            if metrics_dict['val_per'] < self.best_metrics['best_val_per']:
                self.best_metrics['best_val_per'] = metrics_dict['val_per']
        
        if 'val_wer' in metrics_dict:
            if metrics_dict['val_wer'] < self.best_metrics['best_val_wer']:
                self.best_metrics['best_val_wer'] = metrics_dict['val_wer']
        
        if 'val_bleu' in metrics_dict:
            if metrics_dict['val_bleu'] > self.best_metrics['best_val_bleu']:
                self.best_metrics['best_val_bleu'] = metrics_dict['val_bleu']
    
    def get_history(self, metric: str) -> List[float]:
        """Get history for specific metric."""
        return self.metrics.get(metric, [])
    
    def get_latest(self, metric: str) -> Optional[float]:
        """Get latest value for specific metric."""
        history = self.get_history(metric)
        return history[-1] if history else None
    
    def save_metrics(self, save_path: str):
        """Save metrics to JSON file."""
        data = {
            'metrics': self.metrics,
            'best_metrics': self.best_metrics
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"💾 Metrics saved to {save_path}")
    
    def load_metrics(self, load_path: str):
        """Load metrics from JSON file."""
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        self.metrics = data['metrics']
        self.best_metrics = data['best_metrics']
        
        logger.info(f"📂 Metrics loaded from {load_path}")


class TrainingDashboard:
    """
    Interactive training dashboard for real-time monitoring.
    
    Provides comprehensive visualization of training progress.
    """
    
    def __init__(self, save_dir: str = "training_visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_tracker = MetricsTracker()
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 10
        })
    
    def plot_training_curves(
        self,
        metrics_tracker: Optional[MetricsTracker] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training and validation loss curves.
        
        Args:
            metrics_tracker: Metrics tracker to use (default: self.metrics_tracker)
            save_path: Path to save plot
            
        Returns:
            matplotlib Figure
        """
        
        if metrics_tracker is None:
            metrics_tracker = self.metrics_tracker
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = metrics_tracker.get_history('epoch')
        
        # Loss curves
        train_loss = metrics_tracker.get_history('train_loss')
        val_loss = metrics_tracker.get_history('val_loss')
        
        if train_loss and epochs:
            axes[0, 0].plot(epochs[:len(train_loss)], train_loss, 'b-', label='Train Loss', linewidth=2)
        if val_loss and epochs:
            axes[0, 0].plot(epochs[:len(val_loss)], val_loss, 'r-', label='Val Loss', linewidth=2)
        
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # PER curves
        train_per = metrics_tracker.get_history('train_per')
        val_per = metrics_tracker.get_history('val_per')
        
        if train_per and epochs:
            axes[0, 1].plot(epochs[:len(train_per)], train_per, 'g-', label='Train PER', linewidth=2)
        if val_per and epochs:
            axes[0, 1].plot(epochs[:len(val_per)], val_per, 'orange', label='Val PER', linewidth=2)
        
        axes[0, 1].set_title('Phoneme Error Rate (PER)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PER')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # WER curves
        train_wer = metrics_tracker.get_history('train_wer')
        val_wer = metrics_tracker.get_history('val_wer')
        
        if train_wer and epochs:
            axes[1, 0].plot(epochs[:len(train_wer)], train_wer, 'purple', label='Train WER', linewidth=2)
        if val_wer and epochs:
            axes[1, 0].plot(epochs[:len(val_wer)], val_wer, 'brown', label='Val WER', linewidth=2)
        
        axes[1, 0].set_title('Word Error Rate (WER)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('WER')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        learning_rates = metrics_tracker.get_history('learning_rate')
        
        if learning_rates and epochs:
            # Ensure same length
            min_len = min(len(epochs), len(learning_rates))
            axes[1, 1].plot(epochs[:min_len], learning_rates[:min_len], 'red', linewidth=2)
            axes[1, 1].set_yscale('log')
        
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Training curves saved to {save_path}")
        
        return fig
    
    def create_interactive_dashboard(
        self,
        metrics_tracker: Optional[MetricsTracker] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive training dashboard with Plotly.
        
        Args:
            metrics_tracker: Metrics tracker to use
            save_path: Path to save HTML dashboard
            
        Returns:
            Plotly Figure
        """
        
        if metrics_tracker is None:
            metrics_tracker = self.metrics_tracker
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss Curves', 'Error Rates', 'BLEU Scores', 'Learning Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        epochs = metrics_tracker.get_history('epoch')
        
        # Loss curves
        train_loss = metrics_tracker.get_history('train_loss')
        val_loss = metrics_tracker.get_history('val_loss')
        
        if train_loss:
            fig.add_trace(
                go.Scatter(x=epochs[:len(train_loss)], y=train_loss, name='Train Loss', 
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
        
        if val_loss:
            fig.add_trace(
                go.Scatter(x=epochs[:len(val_loss)], y=val_loss, name='Val Loss',
                          line=dict(color='red', width=2)),
                row=1, col=1
            )
        
        # Error rates
        train_per = metrics_tracker.get_history('train_per')
        val_per = metrics_tracker.get_history('val_per')
        train_wer = metrics_tracker.get_history('train_wer')
        val_wer = metrics_tracker.get_history('val_wer')
        
        if train_per:
            fig.add_trace(
                go.Scatter(x=epochs[:len(train_per)], y=train_per, name='Train PER',
                          line=dict(color='green', width=2)),
                row=1, col=2
            )
        
        if val_per:
            fig.add_trace(
                go.Scatter(x=epochs[:len(val_per)], y=val_per, name='Val PER',
                          line=dict(color='orange', width=2)),
                row=1, col=2
            )
        
        if train_wer:
            fig.add_trace(
                go.Scatter(x=epochs[:len(train_wer)], y=train_wer, name='Train WER',
                          line=dict(color='purple', width=2, dash='dash')),
                row=1, col=2
            )
        
        if val_wer:
            fig.add_trace(
                go.Scatter(x=epochs[:len(val_wer)], y=val_wer, name='Val WER',
                          line=dict(color='brown', width=2, dash='dash')),
                row=1, col=2
            )
        
        # BLEU scores
        train_bleu = metrics_tracker.get_history('train_bleu')
        val_bleu = metrics_tracker.get_history('val_bleu')
        
        if train_bleu:
            fig.add_trace(
                go.Scatter(x=epochs[:len(train_bleu)], y=train_bleu, name='Train BLEU',
                          line=dict(color='teal', width=2)),
                row=2, col=1
            )
        
        if val_bleu:
            fig.add_trace(
                go.Scatter(x=epochs[:len(val_bleu)], y=val_bleu, name='Val BLEU',
                          line=dict(color='pink', width=2)),
                row=2, col=1
            )
        
        # Learning rate
        learning_rates = metrics_tracker.get_history('learning_rate')
        
        if learning_rates:
            fig.add_trace(
                go.Scatter(x=epochs[:len(learning_rates)], y=learning_rates, name='Learning Rate',
                          line=dict(color='red', width=2)),
                row=2, col=2
            )
        
        # Update layout
        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Error Rate", row=1, col=2)
        fig.update_yaxes(title_text="BLEU Score", row=2, col=1)
        fig.update_yaxes(title_text="Learning Rate", type="log", row=2, col=2)
        
        fig.update_layout(
            title="Valerie Visual ASR - Training Dashboard",
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"💾 Interactive dashboard saved to {save_path}")
        
        return fig
    
    def plot_model_performance_summary(
        self,
        metrics_tracker: Optional[MetricsTracker] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create summary plot of model performance.
        
        Args:
            metrics_tracker: Metrics tracker to use
            save_path: Path to save plot
            
        Returns:
            matplotlib Figure
        """
        
        if metrics_tracker is None:
            metrics_tracker = self.metrics_tracker
        
        # Get best metrics
        best_metrics = metrics_tracker.best_metrics
        
        # Create performance summary
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Best metrics bar chart
        metrics_names = ['Val Loss', 'Val PER', 'Val WER', 'Val BLEU']
        metrics_values = [
            best_metrics['best_val_loss'],
            best_metrics['best_val_per'],
            best_metrics['best_val_wer'],
            best_metrics['best_val_bleu']
        ]
        
        colors = ['red', 'orange', 'purple', 'teal']
        bars = axes[0, 0].bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        axes[0, 0].set_title('Best Validation Metrics')
        axes[0, 0].set_ylabel('Metric Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
        
        # Training progress
        epochs = metrics_tracker.get_history('epoch')
        val_loss = metrics_tracker.get_history('val_loss')
        
        if val_loss and epochs:
            axes[0, 1].plot(epochs[:len(val_loss)], val_loss, 'b-', linewidth=2)
            axes[0, 1].axhline(y=best_metrics['best_val_loss'], color='r', linestyle='--', 
                              label=f"Best: {best_metrics['best_val_loss']:.4f}")
            axes[0, 1].axvline(x=best_metrics['best_epoch'], color='g', linestyle='--',
                              label=f"Best Epoch: {best_metrics['best_epoch']}")
        
        axes[0, 1].set_title('Validation Loss Progress')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Validation Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error rate comparison
        val_per = metrics_tracker.get_history('val_per')
        val_wer = metrics_tracker.get_history('val_wer')
        
        if val_per and epochs:
            axes[1, 0].plot(epochs[:len(val_per)], val_per, 'orange', label='PER', linewidth=2)
        if val_wer and epochs:
            axes[1, 0].plot(epochs[:len(val_wer)], val_wer, 'purple', label='WER', linewidth=2)
        
        axes[1, 0].set_title('Error Rate Comparison')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Error Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training statistics
        stats_text = f"""
        Training Statistics:
        
        Total Epochs: {len(epochs) if epochs else 0}
        Best Epoch: {best_metrics['best_epoch']}
        
        Best Validation Loss: {best_metrics['best_val_loss']:.4f}
        Best Validation PER: {best_metrics['best_val_per']:.4f}
        Best Validation WER: {best_metrics['best_val_wer']:.4f}
        Best Validation BLEU: {best_metrics['best_val_bleu']:.4f}
        """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Performance summary saved to {save_path}")
        
        return fig
    
    def update_dashboard(
        self,
        epoch: int,
        step: int,
        metrics_dict: Dict[str, float],
        learning_rate: Optional[float] = None,
        save_plots: bool = True
    ):
        """
        Update dashboard with new metrics.
        
        Args:
            epoch: Current epoch
            step: Current step
            metrics_dict: Dictionary of metrics
            learning_rate: Current learning rate
            save_plots: Whether to save updated plots
        """
        
        # Update metrics tracker
        self.metrics_tracker.update(epoch, step, metrics_dict, learning_rate)
        
        if save_plots:
            # Save updated plots
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Training curves
            curves_path = self.save_dir / f"training_curves_{timestamp}.png"
            self.plot_training_curves(save_path=str(curves_path))
            
            # Interactive dashboard
            dashboard_path = self.save_dir / f"dashboard_{timestamp}.html"
            self.create_interactive_dashboard(save_path=str(dashboard_path))
            
            # Performance summary
            summary_path = self.save_dir / f"performance_summary_{timestamp}.png"
            self.plot_model_performance_summary(save_path=str(summary_path))
            
            # Save metrics
            metrics_path = self.save_dir / f"metrics_{timestamp}.json"
            self.metrics_tracker.save_metrics(str(metrics_path))


def create_training_plots(
    metrics_tracker: MetricsTracker,
    save_dir: str = "training_plots"
) -> Dict[str, plt.Figure]:
    """
    Convenience function to create all training plots.
    
    Args:
        metrics_tracker: Metrics tracker with training history
        save_dir: Directory to save plots
        
    Returns:
        Dictionary of matplotlib Figures
    """
    
    dashboard = TrainingDashboard(save_dir)
    
    plots = {
        'training_curves': dashboard.plot_training_curves(metrics_tracker),
        'performance_summary': dashboard.plot_model_performance_summary(metrics_tracker)
    }
    
    return plots
