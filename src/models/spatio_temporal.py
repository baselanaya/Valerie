"""
3D Spatio-Temporal Embedding Module for Valerie Visual ASR.

This module implements 3D CNN blocks with residual connections and temporal
position embeddings to capture lip movement dynamics and coarticulation effects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Conv3dBlock(nn.Module):
    """
    3D CNN block with batch normalization, ReLU activation, and residual connections.
    
    This block captures spatio-temporal features from video sequences while
    maintaining gradient flow through residual connections.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (1, 1, 1),
        use_residual: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize 3D CNN block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: 3D convolution kernel size (T, H, W)
            stride: Convolution stride (T, H, W)
            padding: Convolution padding (T, H, W)
            use_residual: Whether to use residual connections
            dropout: Dropout probability
        """
        super().__init__()
        
        self.use_residual = use_residual and (in_channels == out_channels)
        
        # Main convolution path
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False  # BatchNorm handles bias
        )
        
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        
        # Residual connection projection if needed
        if self.use_residual and in_channels != out_channels:
            self.residual_projection = nn.Conv3d(
                in_channels, out_channels, 
                kernel_size=1, stride=stride, bias=False
            )
        else:
            self.residual_projection = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 3D CNN block.
        
        Args:
            x: Input tensor of shape [B, C, T, H, W]
            
        Returns:
            Output tensor of shape [B, out_channels, T', H', W']
        """
        identity = x
        
        # Main path
        out = self.conv3d(x)
        out = self.batch_norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.use_residual:
            identity = self.residual_projection(identity)
            # Handle different spatial dimensions due to stride
            if identity.shape != out.shape:
                # Adaptive pooling to match dimensions
                identity = F.adaptive_avg_pool3d(identity, out.shape[2:])
            out = out + identity
        
        return out


class TemporalPositionalEncoding(nn.Module):
    """
    Temporal positional encoding for video sequences.
    
    Supports both sinusoidal and learnable position embeddings to help
    the model understand temporal relationships in lip movements.
    """
    
    def __init__(
        self,
        embed_dim: int,
        max_sequence_length: int = 500,
        encoding_type: str = "sinusoidal",
        dropout: float = 0.1
    ):
        """
        Initialize temporal positional encoding.
        
        Args:
            embed_dim: Embedding dimension
            max_sequence_length: Maximum sequence length to support
            encoding_type: Type of encoding ("sinusoidal" or "learnable")
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_sequence_length = max_sequence_length
        self.encoding_type = encoding_type
        self.dropout = nn.Dropout(dropout)
        
        if encoding_type == "sinusoidal":
            # Create sinusoidal position embeddings
            pe = torch.zeros(max_sequence_length, embed_dim)
            position = torch.arange(0, max_sequence_length).unsqueeze(1).float()
            
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                               -(math.log(10000.0) / embed_dim))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            # Register as buffer (not a parameter)
            self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, embed_dim]
            
        elif encoding_type == "learnable":
            # Create learnable position embeddings
            self.pe = nn.Parameter(torch.randn(1, max_sequence_length, embed_dim))
            
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape [B, T, embed_dim]
            mask: Optional padding mask of shape [B, T]
            
        Returns:
            Output tensor with positional encoding added
        """
        batch_size, seq_len, embed_dim = x.shape
        
        if seq_len > self.max_sequence_length:
            logger.warning(f"⚠️ Sequence length {seq_len} exceeds max length {self.max_sequence_length}")
            # Truncate or interpolate position embeddings
            if self.encoding_type == "sinusoidal":
                # Interpolate sinusoidal embeddings
                pe = F.interpolate(
                    self.pe.transpose(1, 2), 
                    size=seq_len, 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
            else:
                # Interpolate learnable embeddings
                pe = F.interpolate(
                    self.pe.transpose(1, 2), 
                    size=seq_len, 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
        else:
            pe = self.pe[:, :seq_len, :]
        
        # Add positional encoding
        x = x + pe
        
        # Apply mask if provided
        if mask is not None:
            # mask: [B, T] -> [B, T, 1]
            mask = mask.unsqueeze(-1).float()
            x = x * mask
        
        return self.dropout(x)


class SpatioTemporalEmbedding(nn.Module):
    """
    Complete 3D spatio-temporal embedding module.
    
    Combines 3D CNN blocks for feature extraction with temporal positional
    encoding to create rich spatio-temporal representations of lip movements.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        embed_dim: int = 512,
        conv3d_channels: list = None,
        kernel_sizes: list = None,
        strides: list = None,
        paddings: list = None,
        max_sequence_length: int = 500,
        positional_encoding: str = "sinusoidal",
        dropout: float = 0.1
    ):
        """
        Initialize spatio-temporal embedding module.
        
        Args:
            input_channels: Number of input channels (typically 3 for RGB)
            embed_dim: Final embedding dimension
            conv3d_channels: List of channel sizes for each 3D CNN layer
            kernel_sizes: List of kernel sizes for each layer
            strides: List of strides for each layer
            paddings: List of padding values for each layer
            max_sequence_length: Maximum temporal sequence length
            positional_encoding: Type of positional encoding
            dropout: Dropout probability
        """
        super().__init__()
        
        # Default architecture if not specified
        if conv3d_channels is None:
            conv3d_channels = [64, 128, 256, embed_dim]
        if kernel_sizes is None:
            kernel_sizes = [(3, 3, 3)] * len(conv3d_channels)
        if strides is None:
            strides = [(1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 1, 1)]
        if paddings is None:
            paddings = [(1, 1, 1)] * len(conv3d_channels)
        
        self.embed_dim = embed_dim
        self.conv3d_channels = conv3d_channels
        
        # Build 3D CNN layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        for i, out_channels in enumerate(conv3d_channels):
            layer = Conv3dBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_sizes[i] if i < len(kernel_sizes) else (3, 3, 3),
                stride=strides[i] if i < len(strides) else (1, 1, 1),
                padding=paddings[i] if i < len(paddings) else (1, 1, 1),
                use_residual=True,
                dropout=dropout
            )
            self.conv_layers.append(layer)
            in_channels = out_channels
        
        # Global average pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # Keep temporal, pool spatial
        
        # Temporal positional encoding
        self.temporal_pos_encoding = TemporalPositionalEncoding(
            embed_dim=embed_dim,
            max_sequence_length=max_sequence_length,
            encoding_type=positional_encoding,
            dropout=dropout
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        logger.info(f"✅ SpatioTemporalEmbedding initialized:")
        logger.info(f"   Input channels: {input_channels}")
        logger.info(f"   Embedding dimension: {embed_dim}")
        logger.info(f"   Conv3D channels: {conv3d_channels}")
        logger.info(f"   Positional encoding: {positional_encoding}")
    
    def forward(
        self, 
        video_frames: torch.Tensor, 
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through spatio-temporal embedding.
        
        Args:
            video_frames: Input video tensor of shape [B, T, H, W, C] or [B, C, T, H, W]
            padding_mask: Optional padding mask of shape [B, T]
            
        Returns:
            Embedded features of shape [B, T', embed_dim]
        """
        # Handle different input formats
        if video_frames.dim() == 5:
            if video_frames.shape[-1] == 3:  # [B, T, H, W, C]
                video_frames = video_frames.permute(0, 4, 1, 2, 3)  # -> [B, C, T, H, W]
        
        batch_size = video_frames.shape[0]
        
        logger.debug(f"🔍 Input shape: {video_frames.shape}")
        
        # Pass through 3D CNN layers
        x = video_frames
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            logger.debug(f"🔍 After conv layer {i}: {x.shape}")
        
        # Global spatial pooling: [B, C, T, H, W] -> [B, C, T, 1, 1]
        x = self.global_pool(x)
        
        # Reshape for temporal processing: [B, C, T, 1, 1] -> [B, T, C]
        x = x.squeeze(-1).squeeze(-1).transpose(1, 2)
        
        logger.debug(f"🔍 After spatial pooling: {x.shape}")
        
        # Add temporal positional encoding
        x = self.temporal_pos_encoding(x, padding_mask)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        logger.debug(f"🔍 Final output shape: {x.shape}")
        
        return x
    
    def get_output_sequence_length(self, input_sequence_length: int) -> int:
        """
        Calculate output sequence length given input sequence length.
        
        Args:
            input_sequence_length: Input temporal sequence length
            
        Returns:
            Output temporal sequence length
        """
        length = input_sequence_length
        
        # Calculate length reduction through conv layers
        for i, conv_layer in enumerate(self.conv_layers):
            stride = conv_layer.conv3d.stride[0]  # Temporal stride
            padding = conv_layer.conv3d.padding[0]  # Temporal padding
            kernel_size = conv_layer.conv3d.kernel_size[0]  # Temporal kernel
            
            length = (length + 2 * padding - kernel_size) // stride + 1
        
        return length


# Factory function for easy creation
def create_spatio_temporal_embedding(config) -> SpatioTemporalEmbedding:
    """
    Factory function to create SpatioTemporalEmbedding from configuration.
    
    Args:
        config: Model configuration object
        
    Returns:
        Initialized SpatioTemporalEmbedding instance
    """
    return SpatioTemporalEmbedding(
        input_channels=config.input_channels,
        embed_dim=config.embed_dim,
        conv3d_channels=config.conv3d_channels,
        max_sequence_length=config.max_sequence_length,
        dropout=config.dropout
    )


# Unit tests and examples
if __name__ == "__main__":
    # Test the modules
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test Conv3dBlock
    print("🧪 Testing Conv3dBlock...")
    conv_block = Conv3dBlock(3, 64, kernel_size=(3, 3, 3)).to(device)
    test_input = torch.randn(2, 3, 16, 112, 112).to(device)  # [B, C, T, H, W]
    output = conv_block(test_input)
    print(f"✅ Conv3dBlock: {test_input.shape} -> {output.shape}")
    
    # Test TemporalPositionalEncoding
    print("\n🧪 Testing TemporalPositionalEncoding...")
    pos_enc = TemporalPositionalEncoding(embed_dim=512, max_sequence_length=100).to(device)
    test_features = torch.randn(2, 16, 512).to(device)  # [B, T, embed_dim]
    encoded = pos_enc(test_features)
    print(f"✅ TemporalPositionalEncoding: {test_features.shape} -> {encoded.shape}")
    
    # Test complete SpatioTemporalEmbedding
    print("\n🧪 Testing SpatioTemporalEmbedding...")
    embedding = SpatioTemporalEmbedding(
        input_channels=3,
        embed_dim=512,
        conv3d_channels=[64, 128, 256, 512]
    ).to(device)
    
    test_video = torch.randn(2, 16, 224, 224, 3).to(device)  # [B, T, H, W, C]
    embedded = embedding(test_video)
    print(f"✅ SpatioTemporalEmbedding: {test_video.shape} -> {embedded.shape}")
    
    # Test with padding mask
    padding_mask = torch.ones(2, embedded.shape[1]).to(device)
    padding_mask[0, 10:] = 0  # Mask second half of first sequence
    masked_embedded = embedding.temporal_pos_encoding(embedded, padding_mask)
    print(f"✅ With padding mask: {embedded.shape} -> {masked_embedded.shape}")
    
    print("\n🎉 All tests passed!")
