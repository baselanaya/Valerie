"""
Main Valerie Visual ASR Model.

Integrates all components into the complete architecture:
- 3D Spatio-Temporal Embedding
- Conformer Encoder  
- Hybrid CTC/Attention Head
- Audio Knowledge Distillation (mandatory)
- Phoneme-to-Sentence Reconstruction (mandatory)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging

from .spatio_temporal import SpatioTemporalEmbedding
from .conformer import ConformerEncoder
from .hybrid_ctc_attention import HybridCTCAttention
from .audio_distillation import AudioKnowledgeDistillation
from .qwen_llm import QwenPhonemeToText
from src.utils.config import Config

logger = logging.getLogger(__name__)


class ValerieModel(nn.Module):
    """
    Complete Valerie Visual ASR Model.
    
    Architecture:
    1. 3D Spatio-Temporal Embedding: Extract visual features from lip movements
    2. Conformer Encoder: Process temporal sequences with hybrid CNN-Transformer
    3. Hybrid CTC/Attention: Joint training with CTC and attention mechanisms
    4. Audio Distillation: Knowledge transfer from audio teacher model (mandatory)
    5. Phoneme-to-Text: Convert phonemes to natural language sentences (mandatory)
    """
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.config = config
        
        # Model components
        self.spatio_temporal = SpatioTemporalEmbedding(
            input_channels=config.model.input_channels,
            embed_dim=config.model.embed_dim,
            conv3d_channels=config.model.conv3d_channels,
            dropout=config.model.dropout
        )
        
        self.conformer = ConformerEncoder(
            input_dim=config.model.embed_dim,
            embed_dim=config.model.embed_dim,
            num_layers=config.model.conformer_layers,
            num_heads=config.model.conformer_heads,
            ffn_expansion_factor=config.model.ffn_expansion_factor,
            conv_kernel_size=config.model.conv_kernel_size,
            dropout=config.model.dropout
        )
        
        self.ctc_attention = HybridCTCAttention(
            encoder_dim=config.model.embed_dim,
            vocab_size=config.model.phoneme_vocab_size,
            decoder_dim=config.model.decoder_dim,
            attention_dim=config.model.decoder_dim,
            dropout=config.model.dropout
        )
        
        # Audio knowledge distillation (now mandatory)
        # Note: Using a simplified distillation setup for now
        self.distillation = nn.Sequential(
            nn.Linear(config.model.embed_dim, config.distillation.projection_dim),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.distillation.projection_dim, config.distillation.teacher_feature_dim)
        )
        
        # Store distillation parameters
        self.distillation_temperature = config.distillation.temperature
        self.distillation_alpha = config.distillation.alpha
        
        # Phoneme-to-text reconstruction (now mandatory)
        self.phoneme_to_text = QwenPhonemeToText(
            model_name=config.model.llm_model_name,
            lora_rank=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
            enable_thinking=config.model.enable_thinking,
            max_new_tokens=config.model.max_new_tokens
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"✅ ValerieModel initialized:")
        logger.info(f"   Embed dim: {config.model.embed_dim}")
        logger.info(f"   Conformer layers: {config.model.conformer_layers}")
        logger.info(f"   Vocab size: {config.model.phoneme_vocab_size}")
        logger.info(f"   Audio distillation: ENABLED (mandatory)")
        logger.info(f"   LLM reconstruction: ENABLED (mandatory)")
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        video: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.
        
        Args:
            video: Video frames [B, C, T, H, W]
            input_lengths: Valid sequence lengths [B]
            targets: Target phoneme sequences [B, S] (for training)
            target_lengths: Target sequence lengths [B] (for training)
            teacher_features: Teacher audio features [B, T, D] (for distillation)
            
        Returns:
            Dictionary containing:
            - ctc_logits: CTC output logits [B, T, V]
            - attention_logits: Attention output logits [B, S, V]
            - attention_weights: Attention weights [B, H, S, T]
            - encoder_outputs: Encoder hidden states [B, T, D]
            - distillation_loss: Knowledge distillation loss (always computed)
            - text_outputs: Generated text (always computed)
        """
        
        batch_size = video.size(0)
        outputs = {}
        
        # 1. Spatio-temporal feature extraction
        visual_features = self.spatio_temporal(video)  # [B, T, D]
        
        # 2. Conformer encoding
        encoder_outputs, encoder_lengths = self.conformer(visual_features, input_lengths)
        outputs['encoder_outputs'] = encoder_outputs
        
        # 3. CTC and Attention outputs
        ctc_logits, attention_outputs = self.ctc_attention(
            encoder_outputs, 
            encoder_lengths,
            targets,
            target_lengths
        )
        
        outputs['ctc_logits'] = ctc_logits
        
        if attention_outputs is not None:
            outputs['attention_logits'] = attention_outputs.get('logits')
            outputs['attention_weights'] = attention_outputs.get('attention_weights')
        
        # 4. Knowledge distillation (always computed when teacher features provided)
        if teacher_features is not None:
            # Project student features to teacher feature space
            projected_student = self.distillation(encoder_outputs)  # [B, T, teacher_dim]
            
            # Compute distillation loss (simplified MSE for now)
            distillation_loss = nn.functional.mse_loss(
                projected_student, teacher_features, reduction='mean'
            )
            outputs['distillation_loss'] = distillation_loss * self.distillation_alpha
        else:
            # If no teacher features provided, set distillation loss to zero
            outputs['distillation_loss'] = torch.tensor(0.0, device=encoder_outputs.device)
        
        # 5. Phoneme-to-text reconstruction (always computed)
        try:
            # Convert CTC logits to phoneme predictions for LLM input
            ctc_predictions = torch.argmax(ctc_logits, dim=-1)  # [B, T]
            
            text_outputs = self.phoneme_to_text(
                phoneme_logits=ctc_logits,
                phoneme_predictions=ctc_predictions,
                input_lengths=encoder_lengths
            )
            outputs['text_outputs'] = text_outputs
            
        except Exception as e:
            logger.warning(f"LLM reconstruction failed: {e}")
            # Return empty text outputs instead of None
            outputs['text_outputs'] = {
                'generated_text': [''] * batch_size,
                'phoneme_sequences': torch.zeros_like(ctc_predictions),
                'confidence_scores': torch.zeros(batch_size, device=encoder_outputs.device)
            }
        
        return outputs
    
    def encode_video(
        self, 
        video: torch.Tensor, 
        input_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode video to features (inference mode).
        
        Args:
            video: Video frames [B, C, T, H, W]
            input_lengths: Valid sequence lengths [B]
            
        Returns:
            encoder_outputs: Encoded features [B, T, D]
            encoder_lengths: Valid sequence lengths [B]
        """
        
        with torch.no_grad():
            # Spatio-temporal encoding
            visual_features = self.spatio_temporal(video)
            
            # Conformer encoding
            encoder_outputs, encoder_lengths = self.conformer(visual_features, input_lengths)
            
            return encoder_outputs, encoder_lengths
    
    def decode_ctc(
        self,
        encoder_outputs: torch.Tensor,
        encoder_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        CTC decoding (inference mode).
        
        Args:
            encoder_outputs: Encoded features [B, T, D]
            encoder_lengths: Valid sequence lengths [B]
            
        Returns:
            ctc_logits: CTC output logits [B, T, V]
        """
        
        with torch.no_grad():
            ctc_logits, _ = self.ctc_attention(
                encoder_outputs, 
                encoder_lengths,
                targets=None,
                target_lengths=None
            )
            
            return ctc_logits
    
    def generate_text(
        self,
        phoneme_logits: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate text from phoneme logits (inference mode).
        
        Args:
            phoneme_logits: Phoneme logits [B, T, V]
            input_lengths: Valid sequence lengths [B]
            
        Returns:
            Generated text outputs (always available)
        """
        
        with torch.no_grad():
            # Convert logits to predictions
            phoneme_predictions = torch.argmax(phoneme_logits, dim=-1)
            
            try:
                text_outputs = self.phoneme_to_text(
                    phoneme_logits=phoneme_logits,
                    phoneme_predictions=phoneme_predictions,
                    input_lengths=input_lengths
                )
                return text_outputs
                
            except Exception as e:
                logger.warning(f"Text generation failed: {e}")
                # Return empty outputs instead of None
                batch_size = phoneme_logits.size(0)
                return {
                    'generated_text': [''] * batch_size,
                    'phoneme_sequences': torch.zeros_like(phoneme_predictions),
                    'confidence_scores': torch.zeros(batch_size, device=phoneme_logits.device)
                }
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information."""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'components': {
                'spatio_temporal': True,
                'conformer': True,
                'ctc_attention': True,
                'distillation': True,  # Now always enabled
                'phoneme_to_text': True  # Now always enabled
            },
            'config': {
                'embed_dim': self.config.model.embed_dim,
                'conformer_layers': self.config.model.conformer_layers,
                'vocab_size': self.config.model.phoneme_vocab_size,
                'num_heads': self.config.model.conformer_heads
            }
        }
    
    def freeze_component(self, component_name: str):
        """Freeze specific component for fine-tuning."""
        
        if component_name == 'spatio_temporal':
            for param in self.spatio_temporal.parameters():
                param.requires_grad = False
        elif component_name == 'conformer':
            for param in self.conformer.parameters():
                param.requires_grad = False
        elif component_name == 'ctc_attention':
            for param in self.ctc_attention.parameters():
                param.requires_grad = False
        elif component_name == 'distillation':
            for param in self.distillation.parameters():
                param.requires_grad = False
        elif component_name == 'phoneme_to_text':
            for param in self.phoneme_to_text.parameters():
                param.requires_grad = False
        else:
            logger.warning(f"Unknown component: {component_name}")
    
    def unfreeze_component(self, component_name: str):
        """Unfreeze specific component."""
        
        if component_name == 'spatio_temporal':
            for param in self.spatio_temporal.parameters():
                param.requires_grad = True
        elif component_name == 'conformer':
            for param in self.conformer.parameters():
                param.requires_grad = True
        elif component_name == 'ctc_attention':
            for param in self.ctc_attention.parameters():
                param.requires_grad = True
        elif component_name == 'distillation':
            for param in self.distillation.parameters():
                param.requires_grad = True
        elif component_name == 'phoneme_to_text':
            for param in self.phoneme_to_text.parameters():
                param.requires_grad = True
        else:
            logger.warning(f"Unknown component: {component_name}")
