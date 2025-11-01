"""
Audio Knowledge Distillation for Valerie Visual ASR.

This module implements knowledge distillation from pre-trained audio ASR models
(Whisper) to improve the visual encoder's performance while maintaining 
visual-only inference capability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, WhisperProcessor
from typing import Optional, Tuple, Dict, List
import numpy as np
from sklearn.cross_decomposition import CCA
from src.utils.logging import get_logger

logger = get_logger(__name__)


class WhisperTeacher(nn.Module):
    """
    Whisper-based teacher model for audio knowledge distillation.
    
    Extracts rich audio features from pre-trained Whisper encoder
    to guide the visual encoder training.
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        feature_layer: int = -1,
        freeze_teacher: bool = True
    ):
        """
        Initialize Whisper teacher model.
        
        Args:
            model_name: Whisper model identifier from Hugging Face
            feature_layer: Which encoder layer to extract features from (-1 for last)
            freeze_teacher: Whether to freeze teacher parameters
        """
        super().__init__()
        
        self.model_name = model_name
        self.feature_layer = feature_layer
        self.freeze_teacher = freeze_teacher
        
        logger.info(f"🎤 Loading Whisper teacher model: {model_name}")
        
        # Load Whisper model and processor
        self.whisper_model = WhisperModel.from_pretrained(model_name)
        self.processor = WhisperProcessor.from_pretrained(model_name)
        
        # Get model dimensions
        self.audio_feature_dim = self.whisper_model.config.d_model
        self.sample_rate = self.processor.feature_extractor.sampling_rate
        
        # Freeze teacher if specified
        if freeze_teacher:
            for param in self.whisper_model.parameters():
                param.requires_grad = False
            self.whisper_model.eval()
            logger.info("🔒 Teacher model frozen for distillation")
        
        logger.info(f"✅ Whisper teacher initialized:")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Feature dimension: {self.audio_feature_dim}")
        logger.info(f"   Sample rate: {self.sample_rate}")
        logger.info(f"   Feature layer: {feature_layer}")
    
    def preprocess_audio(self, audio_waveforms: torch.Tensor) -> torch.Tensor:
        """
        Preprocess audio waveforms for Whisper.
        
        Args:
            audio_waveforms: Audio tensor of shape [B, T] or [B, 1, T]
            
        Returns:
            Preprocessed audio features
        """
        # Ensure correct shape [B, T]
        if audio_waveforms.dim() == 3:
            audio_waveforms = audio_waveforms.squeeze(1)
        
        batch_size = audio_waveforms.shape[0]
        
        # Process each audio in batch
        processed_audios = []
        for i in range(batch_size):
            audio = audio_waveforms[i].cpu().numpy()
            
            # Use Whisper processor to extract mel spectrogram
            inputs = self.processor(
                audio, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            )
            processed_audios.append(inputs.input_features)
        
        # Stack and move to device
        input_features = torch.cat(processed_audios, dim=0)
        return input_features.to(audio_waveforms.device)
    
    def forward(self, audio_waveforms: torch.Tensor) -> torch.Tensor:
        """
        Extract features from Whisper encoder.
        
        Args:
            audio_waveforms: Audio tensor of shape [B, T]
            
        Returns:
            Audio features of shape [B, T_audio, feature_dim]
        """
        # Preprocess audio
        input_features = self.preprocess_audio(audio_waveforms)
        
        # Extract features from Whisper encoder
        with torch.no_grad() if self.freeze_teacher else torch.enable_grad():
            encoder_outputs = self.whisper_model.encoder(
                input_features,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Get features from specified layer
        if self.feature_layer == -1:
            audio_features = encoder_outputs.last_hidden_state
        else:
            audio_features = encoder_outputs.hidden_states[self.feature_layer]
        
        return audio_features


class CrossModalAlignment(nn.Module):
    """
    Cross-modal alignment module for visual-audio feature alignment.
    
    Uses learnable projections and optional CCA initialization to align
    visual and audio feature spaces.
    """
    
    def __init__(
        self,
        visual_dim: int,
        audio_dim: int,
        projection_dim: int = 512,
        use_cca_init: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize cross-modal alignment.
        
        Args:
            visual_dim: Visual feature dimension
            audio_dim: Audio feature dimension  
            projection_dim: Common projection space dimension
            use_cca_init: Whether to use CCA for initialization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.projection_dim = projection_dim
        self.use_cca_init = use_cca_init
        
        # Visual feature projection
        self.visual_projector = nn.Sequential(
            nn.Linear(visual_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Audio feature projection  
        self.audio_projector = nn.Sequential(
            nn.Linear(audio_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Temperature for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
        # CCA initialization will be done during first forward pass
        self.cca_initialized = False
        
        logger.info(f"🔗 CrossModalAlignment initialized:")
        logger.info(f"   Visual dim: {visual_dim} -> {projection_dim}")
        logger.info(f"   Audio dim: {audio_dim} -> {projection_dim}")
        logger.info(f"   Use CCA init: {use_cca_init}")
    
    def initialize_with_cca(
        self, 
        visual_features: torch.Tensor, 
        audio_features: torch.Tensor
    ):
        """
        Initialize projections using Canonical Correlation Analysis.
        
        Args:
            visual_features: Sample visual features [N, visual_dim]
            audio_features: Sample audio features [N, audio_dim]
        """
        if not self.use_cca_init or self.cca_initialized:
            return
        
        logger.info("🔄 Initializing cross-modal alignment with CCA...")
        
        # Convert to numpy for CCA
        visual_np = visual_features.detach().cpu().numpy()
        audio_np = audio_features.detach().cpu().numpy()
        
        # Apply CCA
        n_components = min(self.projection_dim, visual_np.shape[1], audio_np.shape[1])
        cca = CCA(n_components=n_components)
        
        try:
            cca.fit(visual_np, audio_np)
            
            # Initialize visual projector weights
            if hasattr(cca, 'x_weights_'):
                visual_weights = torch.from_numpy(cca.x_weights_.T).float()
                with torch.no_grad():
                    self.visual_projector[-1].weight[:n_components] = visual_weights
            
            # Initialize audio projector weights  
            if hasattr(cca, 'y_weights_'):
                audio_weights = torch.from_numpy(cca.y_weights_.T).float()
                with torch.no_grad():
                    self.audio_projector[-1].weight[:n_components] = audio_weights
            
            logger.info(f"✅ CCA initialization completed with {n_components} components")
            
        except Exception as e:
            logger.warning(f"⚠️ CCA initialization failed: {e}")
        
        self.cca_initialized = True
    
    def forward(
        self, 
        visual_features: torch.Tensor, 
        audio_features: torch.Tensor,
        return_similarity: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Project features to common space and optionally compute similarity.
        
        Args:
            visual_features: Visual features [B, T_v, visual_dim]
            audio_features: Audio features [B, T_a, audio_dim]
            return_similarity: Whether to return cross-modal similarity
            
        Returns:
            Tuple of (projected_visual, projected_audio, similarity_matrix)
        """
        # Flatten temporal dimensions for projection
        batch_size_v, seq_len_v, _ = visual_features.shape
        batch_size_a, seq_len_a, _ = audio_features.shape
        
        visual_flat = visual_features.view(-1, self.visual_dim)
        audio_flat = audio_features.view(-1, self.audio_dim)
        
        # Initialize CCA if needed
        if not self.cca_initialized and self.training:
            self.initialize_with_cca(visual_flat, audio_flat)
        
        # Project to common space
        visual_proj = self.visual_projector(visual_flat)
        audio_proj = self.audio_projector(audio_flat)
        
        # Reshape back
        visual_proj = visual_proj.view(batch_size_v, seq_len_v, self.projection_dim)
        audio_proj = audio_proj.view(batch_size_a, seq_len_a, self.projection_dim)
        
        # Compute similarity if requested
        similarity = None
        if return_similarity:
            # L2 normalize for cosine similarity
            visual_norm = F.normalize(visual_proj, p=2, dim=-1)
            audio_norm = F.normalize(audio_proj, p=2, dim=-1)
            
            # Compute similarity matrix [B, T_v, T_a]
            similarity = torch.bmm(visual_norm, audio_norm.transpose(1, 2)) / self.temperature
        
        return visual_proj, audio_proj, similarity


class AudioKnowledgeDistillation(nn.Module):
    """
    Complete audio knowledge distillation module.
    
    Integrates visual encoder with audio teacher (Whisper) and implements
    various distillation losses for improved visual ASR performance.
    """
    
    def __init__(
        self,
        visual_encoder: nn.Module,
        visual_feature_dim: int,
        audio_teacher_model: str = "openai/whisper-large-v3",
        projection_dim: int = 512,
        distillation_temperature: float = 4.0,
        feature_matching_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        use_cca_init: bool = True,
        freeze_visual_encoder: bool = False
    ):
        """
        Initialize audio knowledge distillation.
        
        Args:
            visual_encoder: Pre-trained visual encoder (e.g., SpatioTemporalEmbedding + Conformer)
            visual_feature_dim: Output dimension of visual encoder
            audio_teacher_model: Whisper model name for teacher
            projection_dim: Common projection space dimension
            distillation_temperature: Temperature for knowledge distillation
            feature_matching_weight: Weight for feature matching loss
            contrastive_weight: Weight for contrastive loss
            use_cca_init: Whether to use CCA initialization
            freeze_visual_encoder: Whether to freeze visual encoder during distillation
        """
        super().__init__()
        
        self.visual_encoder = visual_encoder
        self.visual_feature_dim = visual_feature_dim
        self.projection_dim = projection_dim
        self.distillation_temperature = distillation_temperature
        self.feature_matching_weight = feature_matching_weight
        self.contrastive_weight = contrastive_weight
        
        # Audio teacher model
        self.audio_teacher = WhisperTeacher(
            model_name=audio_teacher_model,
            freeze_teacher=True
        )
        
        # Cross-modal alignment
        self.alignment = CrossModalAlignment(
            visual_dim=visual_feature_dim,
            audio_dim=self.audio_teacher.audio_feature_dim,
            projection_dim=projection_dim,
            use_cca_init=use_cca_init
        )
        
        # Freeze visual encoder if specified
        if freeze_visual_encoder:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            logger.info("🔒 Visual encoder frozen during distillation")
        
        logger.info(f"🎓 AudioKnowledgeDistillation initialized:")
        logger.info(f"   Teacher: {audio_teacher_model}")
        logger.info(f"   Visual feature dim: {visual_feature_dim}")
        logger.info(f"   Audio feature dim: {self.audio_teacher.audio_feature_dim}")
        logger.info(f"   Projection dim: {projection_dim}")
        logger.info(f"   Temperature: {distillation_temperature}")
    
    def compute_feature_matching_loss(
        self, 
        visual_proj: torch.Tensor, 
        audio_proj: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute feature matching loss between visual and audio projections.
        
        Args:
            visual_proj: Projected visual features [B, T_v, proj_dim]
            audio_proj: Projected audio features [B, T_a, proj_dim]
            
        Returns:
            Feature matching loss
        """
        # Temporal alignment via adaptive pooling or attention
        batch_size = visual_proj.shape[0]
        
        # Simple approach: global average pooling
        visual_global = visual_proj.mean(dim=1)  # [B, proj_dim]
        audio_global = audio_proj.mean(dim=1)    # [B, proj_dim]
        
        # MSE loss between global features
        feature_loss = F.mse_loss(visual_global, audio_global)
        
        return feature_loss
    
    def compute_contrastive_loss(
        self,
        visual_proj: torch.Tensor,
        audio_proj: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss for cross-modal alignment.
        
        Args:
            visual_proj: Projected visual features [B, T_v, proj_dim]
            audio_proj: Projected audio features [B, T_a, proj_dim]
            
        Returns:
            Contrastive loss
        """
        batch_size = visual_proj.shape[0]
        
        # Global pooling
        visual_global = F.normalize(visual_proj.mean(dim=1), p=2, dim=1)
        audio_global = F.normalize(audio_proj.mean(dim=1), p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.mm(visual_global, audio_global.t()) / self.distillation_temperature
        
        # Labels for positive pairs (diagonal)
        labels = torch.arange(batch_size, device=similarity.device)
        
        # Contrastive loss (InfoNCE)
        loss_v2a = F.cross_entropy(similarity, labels)
        loss_a2v = F.cross_entropy(similarity.t(), labels)
        
        contrastive_loss = (loss_v2a + loss_a2v) / 2
        
        return contrastive_loss
    
    def forward(
        self,
        video_frames: torch.Tensor,
        audio_waveforms: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through distillation module.
        
        Args:
            video_frames: Video input [B, T, H, W, C] or [B, C, T, H, W]
            audio_waveforms: Audio input [B, T_audio] (optional, for distillation)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with visual features and distillation losses
        """
        results = {}
        
        # Extract visual features
        visual_features = self.visual_encoder(video_frames)
        results['visual_features'] = visual_features
        
        if audio_waveforms is not None and self.training:
            # Extract audio features from teacher
            audio_features = self.audio_teacher(audio_waveforms)
            
            # Cross-modal alignment
            visual_proj, audio_proj, similarity = self.alignment(
                visual_features, audio_features, return_similarity=True
            )
            
            # Compute distillation losses
            feature_loss = self.compute_feature_matching_loss(visual_proj, audio_proj)
            contrastive_loss = self.compute_contrastive_loss(visual_proj, audio_proj)
            
            # Total distillation loss
            total_distillation_loss = (
                self.feature_matching_weight * feature_loss +
                self.contrastive_weight * contrastive_loss
            )
            
            results.update({
                'audio_features': audio_features,
                'visual_proj': visual_proj,
                'audio_proj': audio_proj,
                'similarity': similarity,
                'feature_matching_loss': feature_loss,
                'contrastive_loss': contrastive_loss,
                'distillation_loss': total_distillation_loss
            })
            
            if return_features:
                results.update({
                    'visual_features_raw': visual_features,
                    'audio_features_raw': audio_features
                })
        
        return results
    
    def save_alignment(self, save_path: str):
        """Save the cross-modal alignment module."""
        torch.save(self.alignment.state_dict(), save_path)
        logger.info(f"💾 Cross-modal alignment saved to {save_path}")
    
    def load_alignment(self, load_path: str):
        """Load the cross-modal alignment module."""
        self.alignment.load_state_dict(torch.load(load_path))
        logger.info(f"📂 Cross-modal alignment loaded from {load_path}")


def create_audio_distillation(
    visual_encoder: nn.Module,
    config
) -> AudioKnowledgeDistillation:
    """
    Factory function to create AudioKnowledgeDistillation from configuration.
    
    Args:
        visual_encoder: Pre-trained visual encoder
        config: Configuration object with distillation parameters
        
    Returns:
        Initialized AudioKnowledgeDistillation instance
    """
    return AudioKnowledgeDistillation(
        visual_encoder=visual_encoder,
        visual_feature_dim=config.conformer_dim,
        audio_teacher_model=getattr(config, 'teacher_model_name', "openai/whisper-large-v3"),
        projection_dim=getattr(config, 'projection_dim', 512),
        distillation_temperature=getattr(config, 'temperature', 4.0),
        feature_matching_weight=getattr(config, 'feature_matching_weight', 1.0),
        contrastive_weight=getattr(config, 'contrastive_weight', 0.1),
        use_cca_init=getattr(config, 'use_cca_init', True)
    )


# Unit tests and examples
if __name__ == "__main__":
    # Test the modules
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test WhisperTeacher
    print("🧪 Testing WhisperTeacher...")
    teacher = WhisperTeacher("openai/whisper-tiny").to(device)  # Use tiny for testing
    
    # Generate dummy audio (16kHz, 5 seconds)
    sample_rate = 16000
    audio_length = 5 * sample_rate
    dummy_audio = torch.randn(2, audio_length).to(device)
    
    audio_features = teacher(dummy_audio)
    print(f"✅ WhisperTeacher: {dummy_audio.shape} -> {audio_features.shape}")
    
    # Test CrossModalAlignment
    print("\n🧪 Testing CrossModalAlignment...")
    alignment = CrossModalAlignment(
        visual_dim=512,
        audio_dim=teacher.audio_feature_dim,
        projection_dim=256
    ).to(device)
    
    visual_features = torch.randn(2, 50, 512).to(device)
    visual_proj, audio_proj, similarity = alignment(
        visual_features, audio_features, return_similarity=True
    )
    
    print(f"✅ CrossModalAlignment:")
    print(f"   Visual: {visual_features.shape} -> {visual_proj.shape}")
    print(f"   Audio: {audio_features.shape} -> {audio_proj.shape}")
    print(f"   Similarity: {similarity.shape}")
    
    print("\n🎉 All distillation tests passed!")
