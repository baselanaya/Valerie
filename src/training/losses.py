"""
Loss functions for Valerie Visual ASR training.

Implements hybrid CTC/Attention loss, knowledge distillation loss, 
and temporal consistency loss for robust visual speech recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import logging
from src.utils.logging import get_logger

logger = get_logger(__name__)


class HybridCTCAttentionLoss(nn.Module):
    """
    Hybrid CTC/Attention loss with adaptive weighting.
    
    Combines CTC loss for alignment-free training with attention loss
    for improved accuracy. Supports adaptive weighting based on training progress.
    """
    
    def __init__(
        self,
        ctc_weight: float = 0.3,
        attention_weight: float = 0.7,
        adaptive_weighting: bool = True,
        blank_token_id: int = 0,
        reduction: str = 'mean'
    ):
        """
        Initialize hybrid loss function.
        
        Args:
            ctc_weight: Weight for CTC loss component
            attention_weight: Weight for attention loss component  
            adaptive_weighting: Whether to adapt weights during training
            blank_token_id: ID of blank token for CTC
            reduction: Loss reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        
        self.ctc_weight = ctc_weight
        self.attention_weight = attention_weight
        self.adaptive_weighting = adaptive_weighting
        self.blank_token_id = blank_token_id
        self.reduction = reduction
        
        # Initialize loss functions
        self.ctc_loss = nn.CTCLoss(
            blank=blank_token_id,
            reduction=reduction,
            zero_infinity=True
        )
        
        self.attention_loss = nn.CrossEntropyLoss(
            ignore_index=-1,  # Ignore padding tokens
            reduction=reduction
        )
        
        # Adaptive weighting parameters
        self.training_step = 0
        self.warmup_steps = 4000
        
        logger.info(f"✅ HybridCTCAttentionLoss initialized:")
        logger.info(f"   CTC weight: {ctc_weight}")
        logger.info(f"   Attention weight: {attention_weight}")
        logger.info(f"   Adaptive weighting: {adaptive_weighting}")
    
    def forward(
        self,
        ctc_logits: torch.Tensor,
        attention_logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        attention_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hybrid CTC/Attention loss.
        
        Args:
            ctc_logits: CTC predictions [B, T, V]
            attention_logits: Attention predictions [B, S, V] 
            targets: Target phoneme sequences [B, S]
            input_lengths: Input sequence lengths [B]
            target_lengths: Target sequence lengths [B]
            attention_targets: Attention targets (if different from targets)
            
        Returns:
            Dictionary containing individual and total losses
        """
        batch_size = ctc_logits.size(0)
        device = ctc_logits.device
        
        # Compute CTC loss
        # CTC expects [T, B, V] format
        ctc_logits_t = ctc_logits.transpose(0, 1).log_softmax(dim=-1)
        
        ctc_loss_val = self.ctc_loss(
            ctc_logits_t,
            targets,
            input_lengths,
            target_lengths
        )
        
        # Compute attention loss
        if attention_targets is None:
            attention_targets = targets
        
        # Flatten for cross entropy
        attention_logits_flat = attention_logits.view(-1, attention_logits.size(-1))
        attention_targets_flat = attention_targets.view(-1)
        
        # Mask padding tokens (-1)
        mask = attention_targets_flat != -1
        if mask.sum() > 0:
            attention_loss_val = self.attention_loss(
                attention_logits_flat[mask],
                attention_targets_flat[mask]
            )
        else:
            attention_loss_val = torch.tensor(0.0, device=device)
        
        # Get current weights (adaptive or fixed)
        ctc_w, att_w = self._get_current_weights()
        
        # Compute total loss
        total_loss = ctc_w * ctc_loss_val + att_w * attention_loss_val
        
        # Update training step for adaptive weighting
        if self.training:
            self.training_step += 1
        
        return {
            'total_loss': total_loss,
            'ctc_loss': ctc_loss_val,
            'attention_loss': attention_loss_val,
            'ctc_weight': torch.tensor(ctc_w, device=device),
            'attention_weight': torch.tensor(att_w, device=device)
        }
    
    def _get_current_weights(self) -> Tuple[float, float]:
        """Get current loss weights (adaptive or fixed)."""
        
        if not self.adaptive_weighting:
            return self.ctc_weight, self.attention_weight
        
        # Adaptive weighting: start with higher CTC weight, gradually shift to attention
        if self.training_step < self.warmup_steps:
            # During warmup, emphasize CTC for alignment learning
            progress = self.training_step / self.warmup_steps
            ctc_w = self.ctc_weight + (0.7 - self.ctc_weight) * (1 - progress)
            att_w = self.attention_weight + (0.3 - self.attention_weight) * (1 - progress)
        else:
            # After warmup, use configured weights
            ctc_w = self.ctc_weight
            att_w = self.attention_weight
        
        # Normalize weights
        total_w = ctc_w + att_w
        return ctc_w / total_w, att_w / total_w


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge distillation loss for audio-visual alignment.
    
    Distills knowledge from audio teacher model to visual student model
    using feature alignment and soft target matching.
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        feature_weight: float = 0.3,
        reduction: str = 'mean'
    ):
        """
        Initialize knowledge distillation loss.
        
        Args:
            temperature: Temperature for soft target distillation
            alpha: Weight for distillation loss vs hard target loss
            feature_weight: Weight for feature alignment loss
            reduction: Loss reduction method
        """
        super().__init__()
        
        self.temperature = temperature
        self.alpha = alpha
        self.feature_weight = feature_weight
        self.reduction = reduction
        
        self.kl_loss = nn.KLDivLoss(reduction=reduction)
        self.mse_loss = nn.MSELoss(reduction=reduction)
        
        logger.info(f"✅ KnowledgeDistillationLoss initialized:")
        logger.info(f"   Temperature: {temperature}")
        logger.info(f"   Alpha: {alpha}")
        logger.info(f"   Feature weight: {feature_weight}")
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        hard_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute knowledge distillation loss.
        
        Args:
            student_logits: Student model predictions [B, T, V]
            teacher_logits: Teacher model predictions [B, T, V]
            student_features: Student features [B, T, D]
            teacher_features: Teacher features [B, T, D]
            hard_targets: Ground truth targets [B, T]
            
        Returns:
            Dictionary containing distillation losses
        """
        device = student_logits.device
        
        # Soft target distillation
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Feature alignment loss
        feature_loss = self.mse_loss(student_features, teacher_features)
        
        # Hard target loss (if provided)
        hard_loss = torch.tensor(0.0, device=device)
        if hard_targets is not None:
            hard_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                hard_targets.view(-1),
                ignore_index=-1,
                reduction=self.reduction
            )
        
        # Combine losses
        total_loss = (
            self.alpha * soft_loss + 
            (1 - self.alpha) * hard_loss + 
            self.feature_weight * feature_loss
        )
        
        return {
            'total_loss': total_loss,
            'soft_loss': soft_loss,
            'hard_loss': hard_loss,
            'feature_loss': feature_loss
        }


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for smooth predictions.
    
    Encourages smooth transitions in predictions over time to reduce
    flickering and improve temporal stability.
    """
    
    def __init__(
        self,
        consistency_weight: float = 0.1,
        smoothing_kernel_size: int = 3,
        reduction: str = 'mean'
    ):
        """
        Initialize temporal consistency loss.
        
        Args:
            consistency_weight: Weight for consistency loss
            smoothing_kernel_size: Kernel size for temporal smoothing
            reduction: Loss reduction method
        """
        super().__init__()
        
        self.consistency_weight = consistency_weight
        self.smoothing_kernel_size = smoothing_kernel_size
        self.reduction = reduction
        
        # Create smoothing kernel
        self.register_buffer(
            'smoothing_kernel',
            torch.ones(1, 1, smoothing_kernel_size) / smoothing_kernel_size
        )
        
        logger.info(f"✅ TemporalConsistencyLoss initialized:")
        logger.info(f"   Consistency weight: {consistency_weight}")
        logger.info(f"   Smoothing kernel size: {smoothing_kernel_size}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute temporal consistency loss.
        
        Args:
            predictions: Model predictions [B, T, V]
            targets: Ground truth targets [B, T] (optional)
            input_lengths: Valid sequence lengths [B] (optional)
            
        Returns:
            Dictionary containing consistency losses
        """
        batch_size, seq_len, vocab_size = predictions.shape
        device = predictions.device
        
        # Convert logits to probabilities
        probs = F.softmax(predictions, dim=-1)
        
        # Compute temporal differences
        temporal_diff = torch.diff(probs, dim=1)  # [B, T-1, V]
        
        # L2 norm of temporal differences (encourages smoothness)
        consistency_loss = torch.mean(temporal_diff ** 2)
        
        # Apply input length masking if provided
        if input_lengths is not None:
            mask = torch.zeros(batch_size, seq_len - 1, device=device)
            for i, length in enumerate(input_lengths):
                if length > 1:
                    mask[i, :min(length-1, seq_len-1)] = 1.0
            
            # Apply mask and normalize
            masked_diff = temporal_diff * mask.unsqueeze(-1)
            consistency_loss = masked_diff.pow(2).sum() / (mask.sum() * vocab_size + 1e-8)
        
        # Smoothing loss (simple moving average)
        if seq_len >= self.smoothing_kernel_size:
            # Simple moving average smoothing
            kernel_size = self.smoothing_kernel_size
            padding = kernel_size // 2
            
            # Apply average pooling for smoothing
            smoothed = F.avg_pool1d(
                probs.transpose(1, 2),  # [B, V, T]
                kernel_size=kernel_size,
                stride=1,
                padding=padding
            ).transpose(1, 2)  # [B, T, V]
            
            smoothing_loss = F.mse_loss(probs, smoothed, reduction=self.reduction)
        else:
            smoothing_loss = torch.tensor(0.0, device=device)
        
        total_loss = self.consistency_weight * (consistency_loss + smoothing_loss)
        
        return {
            'total_loss': total_loss,
            'consistency_loss': consistency_loss,
            'smoothing_loss': smoothing_loss
        }


class ValerieLossFunction(nn.Module):
    """
    Complete loss function for Valerie Visual ASR.
    
    Combines hybrid CTC/Attention loss, knowledge distillation loss,
    and temporal consistency loss for comprehensive training.
    """
    
    def __init__(
        self,
        ctc_weight: float = 0.3,
        attention_weight: float = 0.7,
        distillation_weight: float = 0.1,
        temporal_consistency_weight: float = 0.05,
        adaptive_weighting: bool = True,
        blank_token_id: int = 0
    ):
        """
        Initialize complete Valerie loss function.
        
        Args:
            ctc_weight: Weight for CTC loss
            attention_weight: Weight for attention loss
            distillation_weight: Weight for knowledge distillation
            temporal_consistency_weight: Weight for temporal consistency
            adaptive_weighting: Whether to use adaptive loss weighting
            blank_token_id: Blank token ID for CTC
        """
        super().__init__()
        
        self.distillation_weight = distillation_weight
        self.temporal_consistency_weight = temporal_consistency_weight
        
        # Initialize component losses
        self.hybrid_loss = HybridCTCAttentionLoss(
            ctc_weight=ctc_weight,
            attention_weight=attention_weight,
            adaptive_weighting=adaptive_weighting,
            blank_token_id=blank_token_id
        )
        
        self.distillation_loss = KnowledgeDistillationLoss()
        self.temporal_loss = TemporalConsistencyLoss()
        
        logger.info(f"✅ ValerieLossFunction initialized:")
        logger.info(f"   Distillation weight: {distillation_weight}")
        logger.info(f"   Temporal consistency weight: {temporal_consistency_weight}")
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute complete Valerie loss.
        
        Args:
            outputs: Model outputs dictionary containing:
                - ctc_logits: CTC predictions
                - attention_logits: Attention predictions  
                - student_features: Visual features
                - teacher_logits: Teacher predictions (optional)
                - teacher_features: Teacher features (optional)
            targets: Target dictionary containing:
                - phoneme_targets: Target phoneme sequences
                - input_lengths: Input sequence lengths
                - target_lengths: Target sequence lengths
                
        Returns:
            Dictionary containing all loss components
        """
        device = list(outputs.values())[0].device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}
        
        # 1. Hybrid CTC/Attention Loss
        hybrid_losses = self.hybrid_loss(
            ctc_logits=outputs['ctc_logits'],
            attention_logits=outputs['attention_logits'],
            targets=targets['phoneme_targets'],
            input_lengths=targets['input_lengths'],
            target_lengths=targets['target_lengths']
        )
        
        total_loss += hybrid_losses['total_loss']
        loss_dict.update({f'hybrid_{k}': v for k, v in hybrid_losses.items()})
        
        # 2. Knowledge Distillation Loss (if teacher outputs available)
        if 'teacher_logits' in outputs and 'teacher_features' in outputs:
            distill_losses = self.distillation_loss(
                student_logits=outputs['attention_logits'],
                teacher_logits=outputs['teacher_logits'],
                student_features=outputs['student_features'],
                teacher_features=outputs['teacher_features'],
                hard_targets=targets['phoneme_targets']
            )
            
            total_loss += self.distillation_weight * distill_losses['total_loss']
            loss_dict.update({f'distill_{k}': v for k, v in distill_losses.items()})
        
        # 3. Temporal Consistency Loss
        temporal_losses = self.temporal_loss(
            predictions=outputs['attention_logits'],
            targets=targets['phoneme_targets'],
            input_lengths=targets['input_lengths']
        )
        
        total_loss += self.temporal_consistency_weight * temporal_losses['total_loss']
        loss_dict.update({f'temporal_{k}': v for k, v in temporal_losses.items()})
        
        # Add total loss
        loss_dict['total_loss'] = total_loss
        
        return loss_dict
