"""
Attention Weight Visualization for Valerie Visual ASR.

Visualizes attention patterns in the model:
- Multi-head attention weights
- Cross-modal attention (video-audio)
- Temporal attention patterns
- Interactive attention heatmaps
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

from src.utils.logging import get_logger

logger = get_logger(__name__)


class AttentionVisualizer:
    """
    Comprehensive attention visualization for Valerie model.
    
    Provides various visualization methods for:
    - Self-attention in Conformer layers
    - Cross-attention in decoder
    - Temporal attention patterns
    - Multi-head attention analysis
    """
    
    def __init__(self, save_dir: str = "attention_visualizations"):
        self.save_dir = save_dir
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style for consistent plots."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Configure matplotlib
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11
        })
    
    def plot_attention_weights(
        self,
        attention_weights: torch.Tensor,
        input_tokens: Optional[List[str]] = None,
        output_tokens: Optional[List[str]] = None,
        title: str = "Attention Weights",
        save_path: Optional[str] = None,
        head_idx: int = 0
    ) -> plt.Figure:
        """
        Plot attention weights as heatmap.
        
        Args:
            attention_weights: Attention tensor [B, H, T_out, T_in] or [H, T_out, T_in]
            input_tokens: Input token labels
            output_tokens: Output token labels  
            title: Plot title
            save_path: Path to save plot
            head_idx: Which attention head to visualize
            
        Returns:
            matplotlib Figure
        """
        
        # Handle different tensor dimensions
        if attention_weights.dim() == 4:
            # [B, H, T_out, T_in] -> take first batch, specified head
            attn = attention_weights[0, head_idx].cpu().numpy()
        elif attention_weights.dim() == 3:
            # [H, T_out, T_in] -> take specified head
            attn = attention_weights[head_idx].cpu().numpy()
        else:
            # [T_out, T_in] -> use as is
            attn = attention_weights.cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        im = ax.imshow(attn, cmap='Blues', aspect='auto', interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15)
        
        # Set labels
        if input_tokens:
            ax.set_xticks(range(len(input_tokens)))
            ax.set_xticklabels(input_tokens, rotation=45, ha='right')
        else:
            ax.set_xlabel('Input Position')
        
        if output_tokens:
            ax.set_yticks(range(len(output_tokens)))
            ax.set_yticklabels(output_tokens)
        else:
            ax.set_ylabel('Output Position')
        
        ax.set_title(f"{title} (Head {head_idx})")
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Attention plot saved to {save_path}")
        
        return fig
    
    def plot_multi_head_attention(
        self,
        attention_weights: torch.Tensor,
        input_tokens: Optional[List[str]] = None,
        output_tokens: Optional[List[str]] = None,
        title: str = "Multi-Head Attention",
        save_path: Optional[str] = None,
        max_heads: int = 8
    ) -> plt.Figure:
        """
        Plot multiple attention heads in a grid.
        
        Args:
            attention_weights: Attention tensor [B, H, T_out, T_in] or [H, T_out, T_in]
            input_tokens: Input token labels
            output_tokens: Output token labels
            title: Plot title
            save_path: Path to save plot
            max_heads: Maximum number of heads to show
            
        Returns:
            matplotlib Figure
        """
        
        # Handle tensor dimensions
        if attention_weights.dim() == 4:
            attn = attention_weights[0].cpu().numpy()  # Take first batch
        else:
            attn = attention_weights.cpu().numpy()
        
        num_heads = min(attn.shape[0], max_heads)
        
        # Calculate grid size
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols
        
        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if num_heads == 1:
            axes = [axes]
        elif rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for head_idx in range(num_heads):
            ax = axes[head_idx]
            
            # Plot attention for this head
            im = ax.imshow(attn[head_idx], cmap='Blues', aspect='auto', interpolation='nearest')
            ax.set_title(f"Head {head_idx}")
            
            # Set labels for first row/column only
            if head_idx >= num_heads - cols:  # Bottom row
                if input_tokens and len(input_tokens) <= 20:
                    ax.set_xticks(range(len(input_tokens)))
                    ax.set_xticklabels(input_tokens, rotation=45, ha='right', fontsize=8)
                else:
                    ax.set_xlabel('Input Position')
            
            if head_idx % cols == 0:  # First column
                if output_tokens and len(output_tokens) <= 20:
                    ax.set_yticks(range(len(output_tokens)))
                    ax.set_yticklabels(output_tokens, fontsize=8)
                else:
                    ax.set_ylabel('Output Position')
            
            # Add colorbar to last subplot
            if head_idx == num_heads - 1:
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Attention', rotation=270, labelpad=15)
        
        # Hide unused subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Multi-head attention plot saved to {save_path}")
        
        return fig
    
    def create_interactive_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        input_tokens: Optional[List[str]] = None,
        output_tokens: Optional[List[str]] = None,
        title: str = "Interactive Attention Heatmap",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive attention heatmap with Plotly.
        
        Args:
            attention_weights: Attention tensor [B, H, T_out, T_in] or [T_out, T_in]
            input_tokens: Input token labels
            output_tokens: Output token labels
            title: Plot title
            save_path: Path to save HTML file
            
        Returns:
            Plotly Figure
        """
        
        # Handle tensor dimensions
        if attention_weights.dim() >= 3:
            attn = attention_weights[0, 0].cpu().numpy()  # Take first batch, first head
        else:
            attn = attention_weights.cpu().numpy()
        
        # Create labels
        if input_tokens is None:
            input_tokens = [f"In_{i}" for i in range(attn.shape[1])]
        if output_tokens is None:
            output_tokens = [f"Out_{i}" for i in range(attn.shape[0])]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=attn,
            x=input_tokens,
            y=output_tokens,
            colorscale='Blues',
            hoverongaps=False,
            hovertemplate='Input: %{x}<br>Output: %{y}<br>Attention: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Input Tokens",
            yaxis_title="Output Tokens",
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"💾 Interactive attention heatmap saved to {save_path}")
        
        return fig
    
    def analyze_attention_patterns(
        self,
        attention_weights: torch.Tensor,
        input_tokens: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Analyze attention patterns and compute statistics.
        
        Args:
            attention_weights: Attention tensor [B, H, T_out, T_in]
            input_tokens: Input token labels
            save_path: Path to save analysis results
            
        Returns:
            Dictionary of attention statistics
        """
        
        # Handle tensor dimensions
        if attention_weights.dim() == 4:
            attn = attention_weights[0].cpu().numpy()  # Take first batch
        else:
            attn = attention_weights.cpu().numpy()
        
        num_heads, seq_out, seq_in = attn.shape
        
        # Compute statistics
        stats = {
            'entropy': [],
            'sparsity': [],
            'diagonal_attention': [],
            'max_attention': [],
            'attention_spread': []
        }
        
        for head_idx in range(num_heads):
            head_attn = attn[head_idx]
            
            # Entropy (higher = more uniform attention)
            entropy = -np.sum(head_attn * np.log(head_attn + 1e-8), axis=1).mean()
            stats['entropy'].append(entropy)
            
            # Sparsity (fraction of near-zero weights)
            sparsity = (head_attn < 0.01).mean()
            stats['sparsity'].append(sparsity)
            
            # Diagonal attention (self-attention strength)
            if seq_out == seq_in:
                diagonal = np.diag(head_attn).mean()
                stats['diagonal_attention'].append(diagonal)
            
            # Maximum attention weight
            max_attn = head_attn.max()
            stats['max_attention'].append(max_attn)
            
            # Attention spread (standard deviation)
            spread = head_attn.std()
            stats['attention_spread'].append(spread)
        
        # Average across heads
        analysis = {
            'avg_entropy': np.mean(stats['entropy']),
            'avg_sparsity': np.mean(stats['sparsity']),
            'avg_max_attention': np.mean(stats['max_attention']),
            'avg_attention_spread': np.mean(stats['attention_spread']),
            'num_heads': num_heads,
            'sequence_lengths': (seq_out, seq_in)
        }
        
        if seq_out == seq_in:
            analysis['avg_diagonal_attention'] = np.mean(stats['diagonal_attention'])
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"💾 Attention analysis saved to {save_path}")
        
        logger.info("📊 Attention Pattern Analysis:")
        logger.info(f"   Average entropy: {analysis['avg_entropy']:.3f}")
        logger.info(f"   Average sparsity: {analysis['avg_sparsity']:.3f}")
        logger.info(f"   Average max attention: {analysis['avg_max_attention']:.3f}")
        
        return analysis
    
    def plot_temporal_attention_evolution(
        self,
        attention_sequence: List[torch.Tensor],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot how attention patterns evolve over time/layers.
        
        Args:
            attention_sequence: List of attention tensors from different layers/steps
            save_path: Path to save plot
            
        Returns:
            matplotlib Figure
        """
        
        # Extract attention statistics over time
        entropies = []
        max_attentions = []
        sparsities = []
        
        for attn in attention_sequence:
            if attn.dim() >= 3:
                attn = attn[0, 0].cpu().numpy()  # Take first batch, first head
            else:
                attn = attn.cpu().numpy()
            
            # Compute statistics
            entropy = -np.sum(attn * np.log(attn + 1e-8), axis=1).mean()
            max_attn = attn.max()
            sparsity = (attn < 0.01).mean()
            
            entropies.append(entropy)
            max_attentions.append(max_attn)
            sparsities.append(sparsity)
        
        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Entropy evolution
        axes[0].plot(entropies, 'b-o', linewidth=2, markersize=6)
        axes[0].set_title('Attention Entropy Evolution')
        axes[0].set_xlabel('Layer/Step')
        axes[0].set_ylabel('Entropy')
        axes[0].grid(True, alpha=0.3)
        
        # Max attention evolution
        axes[1].plot(max_attentions, 'r-s', linewidth=2, markersize=6)
        axes[1].set_title('Maximum Attention Evolution')
        axes[1].set_xlabel('Layer/Step')
        axes[1].set_ylabel('Max Attention')
        axes[1].grid(True, alpha=0.3)
        
        # Sparsity evolution
        axes[2].plot(sparsities, 'g-^', linewidth=2, markersize=6)
        axes[2].set_title('Attention Sparsity Evolution')
        axes[2].set_xlabel('Layer/Step')
        axes[2].set_ylabel('Sparsity')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Temporal attention evolution saved to {save_path}")
        
        return fig


def plot_attention_weights(
    attention_weights: torch.Tensor,
    input_tokens: Optional[List[str]] = None,
    output_tokens: Optional[List[str]] = None,
    title: str = "Attention Weights",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Convenience function to plot attention weights.
    
    Args:
        attention_weights: Attention tensor
        input_tokens: Input token labels
        output_tokens: Output token labels
        title: Plot title
        save_path: Path to save plot
        
    Returns:
        matplotlib Figure
    """
    
    visualizer = AttentionVisualizer()
    return visualizer.plot_attention_weights(
        attention_weights, input_tokens, output_tokens, title, save_path
    )


def create_attention_heatmap(
    attention_weights: torch.Tensor,
    input_tokens: Optional[List[str]] = None,
    output_tokens: Optional[List[str]] = None,
    title: str = "Attention Heatmap",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Convenience function to create interactive attention heatmap.
    
    Args:
        attention_weights: Attention tensor
        input_tokens: Input token labels
        output_tokens: Output token labels
        title: Plot title
        save_path: Path to save HTML
        
    Returns:
        Plotly Figure
    """
    
    visualizer = AttentionVisualizer()
    return visualizer.create_interactive_attention_heatmap(
        attention_weights, input_tokens, output_tokens, title, save_path
    )
