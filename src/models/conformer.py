"""
Conformer Architecture for Valerie Visual ASR.

Implements the Conformer encoder architecture that combines convolutional layers
with self-attention for better local and global context modeling in lip reading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ConvolutionModule(nn.Module):
    """
    Conformer Convolution Module with depthwise separable convolutions.
    
    This module uses depthwise separable convolutions with GLU activation
    to capture local temporal patterns while maintaining computational efficiency.
    """
    
    def __init__(
        self,
        embed_dim: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
        activation: str = "swish"
    ):
        """
        Initialize convolution module.
        
        Args:
            embed_dim: Embedding dimension
            kernel_size: Convolution kernel size (should be odd)
            dropout: Dropout probability
            activation: Activation function ("swish", "gelu", or "relu")
        """
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size should be odd for 'same' padding"
        
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Pointwise convolution (expansion)
        self.pointwise_conv1 = nn.Conv1d(embed_dim, 2 * embed_dim, kernel_size=1)
        
        # GLU (Gated Linear Unit)
        self.glu = nn.GLU(dim=1)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, groups=embed_dim
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(embed_dim)
        
        # Activation function
        if activation == "swish":
            self.activation = nn.SiLU()  # Swish activation
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Pointwise convolution (projection)
        self.pointwise_conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        logger.debug(f"🔧 ConvolutionModule initialized: embed_dim={embed_dim}, kernel_size={kernel_size}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolution module.
        
        Args:
            x: Input tensor of shape [B, T, embed_dim]
            
        Returns:
            Output tensor of shape [B, T, embed_dim]
        """
        # Layer normalization
        x = self.layer_norm(x)
        
        # Transpose for conv1d: [B, T, embed_dim] -> [B, embed_dim, T]
        x = x.transpose(1, 2)
        
        # Pointwise expansion + GLU
        x = self.pointwise_conv1(x)  # [B, 2*embed_dim, T]
        x = self.glu(x)              # [B, embed_dim, T]
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        
        # Batch normalization and activation
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # Pointwise projection
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # Transpose back: [B, embed_dim, T] -> [B, T, embed_dim]
        x = x.transpose(1, 2)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with relative positional encoding.
    
    Implements efficient self-attention with relative position bias
    for better temporal modeling in sequential data.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_relative_position: int = 100
    ):
        """
        Initialize multi-head self-attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_relative_position: Maximum relative position for positional encoding
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_relative_position = max_relative_position
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Relative positional embeddings
        self.relative_position_k = nn.Parameter(
            torch.randn(2 * max_relative_position + 1, self.head_dim)
        )
        self.relative_position_v = nn.Parameter(
            torch.randn(2 * max_relative_position + 1, self.head_dim)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
        logger.debug(f"🔧 MultiHeadSelfAttention initialized: embed_dim={embed_dim}, num_heads={num_heads}")
    
    def _init_weights(self):
        """Initialize attention weights."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
    
    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        """Get relative position indices."""
        range_vec = torch.arange(seq_len)
        relative_positions = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        relative_positions = torch.clamp(
            relative_positions, 
            -self.max_relative_position, 
            self.max_relative_position
        ) + self.max_relative_position
        return relative_positions
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through multi-head self-attention.
        
        Args:
            x: Input tensor of shape [B, T, embed_dim]
            mask: Optional attention mask of shape [B, T] or [B, T, T]
            
        Returns:
            Output tensor of shape [B, T, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation: [B, T, H, D] -> [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add relative positional encoding
        relative_positions = self._get_relative_positions(seq_len).to(x.device)
        
        # Relative position bias for keys
        relative_k = self.relative_position_k[relative_positions]  # [T, T, head_dim]
        relative_scores_k = torch.einsum('bhid,ijd->bhij', q, relative_k)
        scores = scores + relative_scores_k
        
        # Apply attention mask if provided
        if mask is not None:
            if mask.dim() == 2:  # [B, T] -> [B, 1, 1, T]
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:  # [B, T, T] -> [B, 1, T, T]
                mask = mask.unsqueeze(1)
            
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        
        # Add relative positional encoding for values
        relative_v = self.relative_position_v[relative_positions]  # [T, T, head_dim]
        relative_context_v = torch.einsum('bhij,ijd->bhid', attention_weights, relative_v)
        context = context + relative_context_v
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_proj(context)
        
        return output


class FeedForward(nn.Module):
    """
    Feed-forward network with configurable activation and expansion factor.
    """
    
    def __init__(
        self,
        embed_dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        activation: str = "swish"
    ):
        """
        Initialize feed-forward network.
        
        Args:
            embed_dim: Embedding dimension
            expansion_factor: Hidden dimension expansion factor
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        
        hidden_dim = embed_dim * expansion_factor
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "swish":
            self.activation = nn.SiLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor of shape [B, T, embed_dim]
            
        Returns:
            Output tensor of shape [B, T, embed_dim]
        """
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class ConformerBlock(nn.Module):
    """
    Conformer block with macaron-style feed-forward layers.
    
    Architecture: FFN -> MHSA -> Conv -> FFN with residual connections.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        conv_kernel_size: int = 31,
        ffn_expansion_factor: int = 4,
        dropout: float = 0.1,
        activation: str = "swish"
    ):
        """
        Initialize Conformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            conv_kernel_size: Convolution kernel size
            ffn_expansion_factor: Feed-forward expansion factor
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        
        # First feed-forward (half step)
        self.ffn1 = FeedForward(
            embed_dim, ffn_expansion_factor, dropout, activation
        )
        
        # Multi-head self-attention
        self.attention = MultiHeadSelfAttention(
            embed_dim, num_heads, dropout
        )
        
        # Convolution module
        self.conv = ConvolutionModule(
            embed_dim, conv_kernel_size, dropout, activation
        )
        
        # Second feed-forward (half step)
        self.ffn2 = FeedForward(
            embed_dim, ffn_expansion_factor, dropout, activation
        )
        
        # Layer normalization for final output
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        logger.debug(f"🔧 ConformerBlock initialized: embed_dim={embed_dim}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Conformer block.
        
        Args:
            x: Input tensor of shape [B, T, embed_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape [B, T, embed_dim]
        """
        # Macaron-style: half-step FFN
        x = x + 0.5 * self.ffn1(x)
        
        # Multi-head self-attention with residual
        x = x + self.attention(x, mask)
        
        # Convolution module with residual
        x = x + self.conv(x)
        
        # Final half-step FFN
        x = x + 0.5 * self.ffn2(x)
        
        # Final layer normalization
        x = self.layer_norm(x)
        
        return x


class ConformerEncoder(nn.Module):
    """
    Complete Conformer encoder with multiple Conformer blocks.
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        conv_kernel_size: int = 31,
        ffn_expansion_factor: int = 4,
        dropout: float = 0.1,
        activation: str = "swish"
    ):
        """
        Initialize Conformer encoder.
        
        Args:
            input_dim: Input feature dimension
            embed_dim: Embedding dimension
            num_layers: Number of Conformer layers
            num_heads: Number of attention heads
            conv_kernel_size: Convolution kernel size
            ffn_expansion_factor: Feed-forward expansion factor
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Input projection if needed
        if input_dim != embed_dim:
            self.input_projection = nn.Linear(input_dim, embed_dim)
        else:
            self.input_projection = nn.Identity()
        
        # Conformer layers
        self.layers = nn.ModuleList([
            ConformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                conv_kernel_size=conv_kernel_size,
                ffn_expansion_factor=ffn_expansion_factor,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])
        
        # Output layer normalization
        self.output_layer_norm = nn.LayerNorm(embed_dim)
        
        logger.info(f"✅ ConformerEncoder initialized:")
        logger.info(f"   Input dimension: {input_dim}")
        logger.info(f"   Embedding dimension: {embed_dim}")
        logger.info(f"   Number of layers: {num_layers}")
        logger.info(f"   Number of heads: {num_heads}")
        logger.info(f"   Convolution kernel size: {conv_kernel_size}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Conformer encoder.
        
        Args:
            x: Input tensor of shape [B, T, input_dim]
            mask: Optional attention mask of shape [B, T]
            
        Returns:
            Encoded features of shape [B, T, embed_dim]
        """
        # Input projection
        x = self.input_projection(x)
        
        # Pass through Conformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final layer normalization
        x = self.output_layer_norm(x)
        
        return x


def create_conformer_encoder(config) -> ConformerEncoder:
    """
    Factory function to create ConformerEncoder from configuration.
    
    Args:
        config: Model configuration object
        
    Returns:
        Initialized ConformerEncoder instance
    """
    return ConformerEncoder(
        input_dim=config.embed_dim,  # From spatio-temporal embedding
        embed_dim=config.conformer_dim,
        num_layers=config.conformer_layers,
        num_heads=config.conformer_heads,
        conv_kernel_size=config.conv_kernel_size,
        ffn_expansion_factor=config.ffn_expansion_factor,
        dropout=config.dropout
    )


# Unit tests and examples
if __name__ == "__main__":
    # Test the modules
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test ConvolutionModule
    print("🧪 Testing ConvolutionModule...")
    conv_module = ConvolutionModule(embed_dim=512, kernel_size=31).to(device)
    test_input = torch.randn(2, 100, 512).to(device)  # [B, T, embed_dim]
    output = conv_module(test_input)
    print(f"✅ ConvolutionModule: {test_input.shape} -> {output.shape}")
    
    # Test MultiHeadSelfAttention
    print("\n🧪 Testing MultiHeadSelfAttention...")
    attention = MultiHeadSelfAttention(embed_dim=512, num_heads=8).to(device)
    output = attention(test_input)
    print(f"✅ MultiHeadSelfAttention: {test_input.shape} -> {output.shape}")
    
    # Test ConformerBlock
    print("\n🧪 Testing ConformerBlock...")
    conformer_block = ConformerBlock(embed_dim=512).to(device)
    output = conformer_block(test_input)
    print(f"✅ ConformerBlock: {test_input.shape} -> {output.shape}")
    
    # Test complete ConformerEncoder
    print("\n🧪 Testing ConformerEncoder...")
    encoder = ConformerEncoder(
        input_dim=512,
        embed_dim=512,
        num_layers=6,  # Smaller for testing
        num_heads=8
    ).to(device)
    
    output = encoder(test_input)
    print(f"✅ ConformerEncoder: {test_input.shape} -> {output.shape}")
    
    # Test with attention mask
    mask = torch.ones(2, 100).to(device)
    mask[0, 50:] = 0  # Mask second half of first sequence
    masked_output = encoder(test_input, mask)
    print(f"✅ With attention mask: {test_input.shape} -> {masked_output.shape}")
    
    print("\n🎉 All tests passed!")
