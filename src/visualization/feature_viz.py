"""
Feature Map Visualization for Valerie Visual ASR.

Visualizes 3D CNN feature maps and intermediate representations:
- Spatio-temporal feature evolution
- Channel-wise feature analysis
- Activation patterns visualization
- Feature importance heatmaps
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import cv2
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

from src.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureVisualizer:
    """
    Comprehensive feature visualization for 3D CNN components.
    
    Provides visualization methods for:
    - 3D convolutional feature maps
    - Temporal feature evolution
    - Channel activation analysis
    - Feature importance mapping
    """
    
    def __init__(self, save_dir: str = "feature_visualizations"):
        self.save_dir = save_dir
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style for consistent plots."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("viridis")
        
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 10
        })
    
    def visualize_3d_features(
        self,
        features: torch.Tensor,
        title: str = "3D Feature Maps",
        save_path: Optional[str] = None,
        max_channels: int = 16,
        time_slice: Optional[int] = None
    ) -> plt.Figure:
        """
        Visualize 3D CNN feature maps.
        
        Args:
            features: Feature tensor [B, C, T, H, W] or [C, T, H, W]
            title: Plot title
            save_path: Path to save plot
            max_channels: Maximum number of channels to show
            time_slice: Specific time slice to visualize (None for middle)
            
        Returns:
            matplotlib Figure
        """
        
        # Handle batch dimension
        if features.dim() == 5:
            feat = features[0].cpu().numpy()  # Take first batch
        else:
            feat = features.cpu().numpy()
        
        C, T, H, W = feat.shape
        
        # Select time slice
        if time_slice is None:
            time_slice = T // 2
        
        # Select channels to visualize
        num_channels = min(C, max_channels)
        channel_indices = np.linspace(0, C-1, num_channels, dtype=int)
        
        # Calculate grid size
        cols = min(4, num_channels)
        rows = (num_channels + cols - 1) // cols
        
        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if num_channels == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, ch_idx in enumerate(channel_indices):
            ax = axes[i]
            
            # Get feature map for this channel and time slice
            feature_map = feat[ch_idx, time_slice]
            
            # Normalize for visualization
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            
            # Display feature map
            im = ax.imshow(feature_map, cmap='viridis', aspect='auto')
            ax.set_title(f"Channel {ch_idx} (t={time_slice})")
            ax.axis('off')
            
            # Add colorbar to last subplot
            if i == num_channels - 1:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Activation', rotation=270, labelpad=15)
        
        # Hide unused subplots
        for idx in range(num_channels, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f"{title} (Time Slice {time_slice})", fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 3D feature visualization saved to {save_path}")
        
        return fig
    
    def plot_temporal_feature_evolution(
        self,
        features: torch.Tensor,
        channel_idx: int = 0,
        title: str = "Temporal Feature Evolution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot how features evolve over time for a specific channel.
        
        Args:
            features: Feature tensor [B, C, T, H, W] or [C, T, H, W]
            channel_idx: Channel to visualize
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            matplotlib Figure
        """
        
        # Handle batch dimension
        if features.dim() == 5:
            feat = features[0].cpu().numpy()  # Take first batch
        else:
            feat = features.cpu().numpy()
        
        C, T, H, W = feat.shape
        
        if channel_idx >= C:
            channel_idx = 0
            logger.warning(f"Channel index {channel_idx} out of range, using channel 0")
        
        # Get feature maps for all time steps
        channel_features = feat[channel_idx]  # [T, H, W]
        
        # Calculate number of time steps to show
        max_time_steps = min(T, 8)
        time_indices = np.linspace(0, T-1, max_time_steps, dtype=int)
        
        # Create subplots
        cols = min(4, max_time_steps)
        rows = (max_time_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if max_time_steps == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, t_idx in enumerate(time_indices):
            ax = axes[i]
            
            # Get feature map at this time step
            feature_map = channel_features[t_idx]
            
            # Normalize
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            
            # Display
            im = ax.imshow(feature_map, cmap='plasma', aspect='auto')
            ax.set_title(f"t = {t_idx}")
            ax.axis('off')
            
            # Add colorbar to last subplot
            if i == max_time_steps - 1:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Activation', rotation=270, labelpad=15)
        
        # Hide unused subplots
        for idx in range(max_time_steps, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f"{title} - Channel {channel_idx}", fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Temporal evolution plot saved to {save_path}")
        
        return fig
    
    def analyze_channel_activations(
        self,
        features: torch.Tensor,
        save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Analyze activation patterns across channels.
        
        Args:
            features: Feature tensor [B, C, T, H, W] or [C, T, H, W]
            save_path: Path to save analysis
            
        Returns:
            Dictionary of activation statistics
        """
        
        # Handle batch dimension
        if features.dim() == 5:
            feat = features[0].cpu().numpy()  # Take first batch
        else:
            feat = features.cpu().numpy()
        
        C, T, H, W = feat.shape
        
        # Compute statistics per channel
        channel_means = []
        channel_stds = []
        channel_max = []
        channel_sparsity = []
        
        for c in range(C):
            channel_feat = feat[c]  # [T, H, W]
            
            # Statistics
            mean_activation = channel_feat.mean()
            std_activation = channel_feat.std()
            max_activation = channel_feat.max()
            sparsity = (channel_feat < 0.01).mean()  # Fraction of near-zero activations
            
            channel_means.append(mean_activation)
            channel_stds.append(std_activation)
            channel_max.append(max_activation)
            channel_sparsity.append(sparsity)
        
        analysis = {
            'channel_means': np.array(channel_means),
            'channel_stds': np.array(channel_stds),
            'channel_max': np.array(channel_max),
            'channel_sparsity': np.array(channel_sparsity),
            'num_channels': C,
            'feature_shape': (T, H, W)
        }
        
        # Create analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Mean activations
        axes[0, 0].bar(range(C), channel_means, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Mean Activation per Channel')
        axes[0, 0].set_xlabel('Channel')
        axes[0, 0].set_ylabel('Mean Activation')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Standard deviations
        axes[0, 1].bar(range(C), channel_stds, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Activation Std per Channel')
        axes[0, 1].set_xlabel('Channel')
        axes[0, 1].set_ylabel('Std Activation')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Max activations
        axes[1, 0].bar(range(C), channel_max, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Max Activation per Channel')
        axes[1, 0].set_xlabel('Channel')
        axes[1, 0].set_ylabel('Max Activation')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sparsity
        axes[1, 1].bar(range(C), channel_sparsity, color='gold', alpha=0.7)
        axes[1, 1].set_title('Sparsity per Channel')
        axes[1, 1].set_xlabel('Channel')
        axes[1, 1].set_ylabel('Sparsity (fraction near-zero)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            analysis_path = save_path.replace('.png', '_analysis.png')
            plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
            
            # Save numerical data
            data_path = save_path.replace('.png', '_data.npz')
            np.savez(data_path, **analysis)
            
            logger.info(f"💾 Channel analysis saved to {analysis_path}")
        
        logger.info("📊 Channel Activation Analysis:")
        logger.info(f"   Mean activation: {np.mean(channel_means):.4f} ± {np.std(channel_means):.4f}")
        logger.info(f"   Average sparsity: {np.mean(channel_sparsity):.3f}")
        logger.info(f"   Most active channel: {np.argmax(channel_means)} (mean: {np.max(channel_means):.4f})")
        
        return analysis
    
    def create_feature_importance_heatmap(
        self,
        features: torch.Tensor,
        spatial_importance: Optional[torch.Tensor] = None,
        title: str = "Feature Importance Heatmap",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create heatmap showing feature importance across spatial dimensions.
        
        Args:
            features: Feature tensor [B, C, T, H, W] or [C, T, H, W]
            spatial_importance: Pre-computed importance weights [H, W]
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            matplotlib Figure
        """
        
        # Handle batch dimension
        if features.dim() == 5:
            feat = features[0].cpu().numpy()  # Take first batch
        else:
            feat = features.cpu().numpy()
        
        C, T, H, W = feat.shape
        
        # Compute spatial importance if not provided
        if spatial_importance is None:
            # Use average activation magnitude across channels and time
            importance = np.mean(np.abs(feat), axis=(0, 1))  # [H, W]
        else:
            importance = spatial_importance.cpu().numpy()
        
        # Normalize importance
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(importance, cmap='hot', aspect='auto', interpolation='bilinear')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Importance Score', rotation=270, labelpad=15)
        
        # Labels and title
        ax.set_title(title)
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        
        # Add grid
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Feature importance heatmap saved to {save_path}")
        
        return fig
    
    def create_3d_feature_animation(
        self,
        features: torch.Tensor,
        channel_idx: int = 0,
        title: str = "3D Feature Animation",
        save_path: Optional[str] = None,
        fps: int = 5
    ) -> Optional[FuncAnimation]:
        """
        Create animation showing temporal evolution of 3D features.
        
        Args:
            features: Feature tensor [B, C, T, H, W] or [C, T, H, W]
            channel_idx: Channel to animate
            title: Animation title
            save_path: Path to save animation (MP4 or GIF)
            fps: Frames per second
            
        Returns:
            Animation object or None if save_path provided
        """
        
        # Handle batch dimension
        if features.dim() == 5:
            feat = features[0].cpu().numpy()  # Take first batch
        else:
            feat = features.cpu().numpy()
        
        C, T, H, W = feat.shape
        
        if channel_idx >= C:
            channel_idx = 0
            logger.warning(f"Channel index {channel_idx} out of range, using channel 0")
        
        # Get channel features
        channel_features = feat[channel_idx]  # [T, H, W]
        
        # Normalize all frames together for consistent scaling
        vmin = channel_features.min()
        vmax = channel_features.max()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Initialize plot
        im = ax.imshow(channel_features[0], cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f"{title} - Channel {channel_idx} - Frame 0")
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activation', rotation=270, labelpad=15)
        
        def animate(frame):
            """Animation function."""
            im.set_array(channel_features[frame])
            ax.set_title(f"{title} - Channel {channel_idx} - Frame {frame}")
            return [im]
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=T, interval=1000//fps, blit=True, repeat=True)
        
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=fps)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=fps)
            else:
                logger.warning("Unsupported animation format. Use .gif or .mp4")
                return anim
            
            logger.info(f"💾 3D feature animation saved to {save_path}")
            plt.close(fig)
            return None
        
        return anim


def visualize_3d_features(
    features: torch.Tensor,
    title: str = "3D Feature Maps",
    save_path: Optional[str] = None,
    max_channels: int = 16
) -> plt.Figure:
    """
    Convenience function to visualize 3D CNN features.
    
    Args:
        features: Feature tensor [B, C, T, H, W] or [C, T, H, W]
        title: Plot title
        save_path: Path to save plot
        max_channels: Maximum channels to show
        
    Returns:
        matplotlib Figure
    """
    
    visualizer = FeatureVisualizer()
    return visualizer.visualize_3d_features(features, title, save_path, max_channels)


def plot_feature_maps(
    features: torch.Tensor,
    title: str = "Feature Maps",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Convenience function to plot feature maps.
    
    Args:
        features: Feature tensor
        title: Plot title
        save_path: Path to save plot
        
    Returns:
        matplotlib Figure
    """
    
    visualizer = FeatureVisualizer()
    return visualizer.visualize_3d_features(features, title, save_path)
