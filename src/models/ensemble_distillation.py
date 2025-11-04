"""
Ensemble Knowledge Distillation for Phoneme ASR.

This module implements ensemble distillation from multiple pre-trained ASR teacher models
to a smaller student model for efficient phoneme recognition with improved accuracy.

Supports multiple teacher models including:
- Whisper Large V3 (OpenAI)
- WavLM Large (Microsoft)
- HuBERT Large (Facebook)
- Any other ASR model from Hugging Face
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    WhisperModel, WhisperProcessor,
    WavLMModel, Wav2Vec2Processor,
    HubertModel, HubertProcessor,
    AutoModel, AutoProcessor
)
from typing import Optional, Tuple, Dict, List, Union
import numpy as np
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TeacherModel(nn.Module):
    """
    Wrapper for a single teacher ASR model.

    Supports loading various pre-trained models from Hugging Face
    and extracting intermediate features for distillation.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "whisper",
        feature_layer: int = -1,
        freeze: bool = True,
        extract_phoneme_logits: bool = True
    ):
        """
        Initialize teacher model.

        Args:
            model_name: Model identifier from Hugging Face
            model_type: Type of model ("whisper", "wavlm", "hubert", "auto")
            feature_layer: Which encoder layer to extract features from (-1 for last)
            freeze: Whether to freeze teacher parameters
            extract_phoneme_logits: Whether to extract phoneme predictions
        """
        super().__init__()

        self.model_name = model_name
        self.model_type = model_type.lower()
        self.feature_layer = feature_layer
        self.freeze = freeze
        self.extract_phoneme_logits = extract_phoneme_logits

        logger.info(f"🎓 Loading teacher model: {model_name} (type: {model_type})")

        # Load model and processor based on type
        if self.model_type == "whisper":
            self.model = WhisperModel.from_pretrained(model_name)
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.sample_rate = self.processor.feature_extractor.sampling_rate
            self.feature_dim = self.model.config.d_model

        elif self.model_type == "wavlm":
            self.model = WavLMModel.from_pretrained(model_name)
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.sample_rate = self.processor.feature_extractor.sampling_rate
            self.feature_dim = self.model.config.hidden_size

        elif self.model_type == "hubert":
            self.model = HubertModel.from_pretrained(model_name)
            self.processor = HubertProcessor.from_pretrained(model_name)
            self.sample_rate = self.processor.feature_extractor.sampling_rate
            self.feature_dim = self.model.config.hidden_size

        elif self.model_type == "auto":
            self.model = AutoModel.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.sample_rate = getattr(
                self.processor.feature_extractor, 'sampling_rate', 16000
            )
            self.feature_dim = self.model.config.hidden_size

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Add phoneme prediction head if needed
        if extract_phoneme_logits:
            # 39 English phonemes + blank token
            self.phoneme_head = nn.Linear(self.feature_dim, 40)
        else:
            self.phoneme_head = None

        # Freeze teacher if specified
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            logger.info("🔒 Teacher model frozen")

        logger.info(f"✅ Teacher initialized:")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Feature dimension: {self.feature_dim}")
        logger.info(f"   Sample rate: {self.sample_rate}")

    def preprocess_audio(
        self,
        audio_waveforms: torch.Tensor,
        sample_rate: Optional[int] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Preprocess audio waveforms for the teacher model.

        Args:
            audio_waveforms: Audio tensor of shape [B, T] or [B, 1, T]
            sample_rate: Sample rate of input audio (if different from model)

        Returns:
            Preprocessed audio features
        """
        # Ensure correct shape [B, T]
        if audio_waveforms.dim() == 3:
            audio_waveforms = audio_waveforms.squeeze(1)

        batch_size = audio_waveforms.shape[0]
        device = audio_waveforms.device

        if sample_rate is None:
            sample_rate = self.sample_rate

        # Process each audio in batch
        processed_audios = []
        for i in range(batch_size):
            audio = audio_waveforms[i].cpu().numpy()

            # Use processor to extract features
            if self.model_type == "whisper":
                inputs = self.processor(
                    audio,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                )
                processed_audios.append(inputs.input_features)
            else:
                # WavLM, HuBERT, etc.
                inputs = self.processor(
                    audio,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                )
                processed_audios.append(inputs.input_values)

        # Stack and move to device
        if self.model_type == "whisper":
            input_features = torch.cat(processed_audios, dim=0).to(device)
            return input_features
        else:
            input_values = torch.cat(processed_audios, dim=0).to(device)
            return {"input_values": input_values.squeeze(1)}

    def forward(
        self,
        audio_waveforms: torch.Tensor,
        return_all_layers: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features and predictions from teacher model.

        Args:
            audio_waveforms: Audio tensor of shape [B, T]
            return_all_layers: Whether to return features from all layers

        Returns:
            Dictionary containing:
            - features: Encoder features [B, T', feature_dim]
            - phoneme_logits: Phoneme predictions [B, T', 40] (if enabled)
            - all_layers: List of features from all layers (if requested)
        """
        # Preprocess audio
        inputs = self.preprocess_audio(audio_waveforms)

        # Extract features from model
        with torch.no_grad() if self.freeze else torch.enable_grad():
            if self.model_type == "whisper":
                outputs = self.model.encoder(
                    inputs,
                    output_hidden_states=return_all_layers or (self.feature_layer != -1),
                    return_dict=True
                )
            else:
                # WavLM, HuBERT, etc.
                outputs = self.model(
                    **inputs,
                    output_hidden_states=return_all_layers or (self.feature_layer != -1),
                    return_dict=True
                )

        # Get features from specified layer
        if self.feature_layer == -1:
            features = outputs.last_hidden_state
        else:
            features = outputs.hidden_states[self.feature_layer]

        results = {"features": features}

        # Extract phoneme logits if enabled
        if self.extract_phoneme_logits and self.phoneme_head is not None:
            with torch.no_grad() if self.freeze else torch.enable_grad():
                phoneme_logits = self.phoneme_head(features)
            results["phoneme_logits"] = phoneme_logits

        # Return all layers if requested
        if return_all_layers:
            results["all_layers"] = outputs.hidden_states

        return results


class EnsembleDistillationModule(nn.Module):
    """
    Ensemble distillation from multiple teacher models to a student model.

    Combines knowledge from multiple pre-trained ASR models using:
    - Feature-level distillation (encoder features)
    - Soft label distillation (phoneme predictions)
    - Weighted ensemble aggregation
    """

    def __init__(
        self,
        teacher_configs: List[Dict[str, any]],
        student_feature_dim: int,
        projection_dim: int = 512,
        distillation_temperature: float = 4.0,
        ensemble_weights: Optional[List[float]] = None,
        aggregation_method: str = "weighted_average",
        feature_matching_loss: str = "mse",
        soft_label_loss: str = "kl_div"
    ):
        """
        Initialize ensemble distillation module.

        Args:
            teacher_configs: List of teacher configurations, each with:
                - model_name: Hugging Face model identifier
                - model_type: Model type ("whisper", "wavlm", "hubert")
                - weight: Optional weight for this teacher (default: 1.0)
            student_feature_dim: Student model feature dimension
            projection_dim: Common projection space dimension
            distillation_temperature: Temperature for soft label distillation
            ensemble_weights: Weights for each teacher (default: equal weights)
            aggregation_method: How to aggregate teacher outputs
                - "weighted_average": Weighted average of teacher predictions
                - "attention": Learn attention weights dynamically
                - "max": Take maximum across teachers
            feature_matching_loss: Type of feature loss ("mse", "cosine", "huber")
            soft_label_loss: Type of soft label loss ("kl_div", "mse", "js_div")
        """
        super().__init__()

        self.student_feature_dim = student_feature_dim
        self.projection_dim = projection_dim
        self.distillation_temperature = distillation_temperature
        self.aggregation_method = aggregation_method
        self.feature_matching_loss_type = feature_matching_loss
        self.soft_label_loss_type = soft_label_loss

        logger.info(f"🎓 Initializing Ensemble Distillation with {len(teacher_configs)} teachers")

        # Initialize teacher models
        self.teachers = nn.ModuleList()
        self.teacher_names = []

        for i, config in enumerate(teacher_configs):
            teacher = TeacherModel(
                model_name=config['model_name'],
                model_type=config.get('model_type', 'whisper'),
                feature_layer=config.get('feature_layer', -1),
                freeze=config.get('freeze', True),
                extract_phoneme_logits=config.get('extract_phoneme_logits', True)
            )
            self.teachers.append(teacher)
            self.teacher_names.append(config['model_name'])

        # Set ensemble weights
        if ensemble_weights is None:
            # Equal weights by default
            self.ensemble_weights = [1.0 / len(teacher_configs)] * len(teacher_configs)
        else:
            assert len(ensemble_weights) == len(teacher_configs)
            # Normalize weights
            total = sum(ensemble_weights)
            self.ensemble_weights = [w / total for w in ensemble_weights]

        logger.info(f"📊 Ensemble weights: {self.ensemble_weights}")

        # Create projection layers for each teacher to common space
        self.teacher_projectors = nn.ModuleList()
        for teacher in self.teachers:
            projector = nn.Sequential(
                nn.Linear(teacher.feature_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(projection_dim, projection_dim)
            )
            self.teacher_projectors.append(projector)

        # Student projector
        self.student_projector = nn.Sequential(
            nn.Linear(student_feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim)
        )

        # Attention-based aggregation (optional)
        if aggregation_method == "attention":
            self.attention_weights = nn.Sequential(
                nn.Linear(projection_dim, len(teacher_configs)),
                nn.Softmax(dim=-1)
            )
        else:
            self.attention_weights = None

        # Temperature parameter (learnable)
        self.temperature = nn.Parameter(
            torch.tensor(distillation_temperature)
        )

        logger.info(f"✅ Ensemble Distillation initialized:")
        logger.info(f"   Teachers: {', '.join([t.split('/')[-1] for t in self.teacher_names])}")
        logger.info(f"   Projection dim: {projection_dim}")
        logger.info(f"   Aggregation: {aggregation_method}")
        logger.info(f"   Temperature: {distillation_temperature}")

    def aggregate_teacher_outputs(
        self,
        teacher_features: List[torch.Tensor],
        teacher_logits: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate outputs from multiple teachers.

        Args:
            teacher_features: List of teacher features [B, T, D_proj]
            teacher_logits: List of teacher phoneme logits [B, T, V]

        Returns:
            Tuple of (aggregated_features, aggregated_logits)
        """
        batch_size = teacher_features[0].shape[0]

        if self.aggregation_method == "weighted_average":
            # Weighted average based on fixed weights
            agg_features = sum(
                w * feat for w, feat in zip(self.ensemble_weights, teacher_features)
            )
            agg_logits = sum(
                w * logit for w, logit in zip(self.ensemble_weights, teacher_logits)
            )

        elif self.aggregation_method == "attention":
            # Learn attention weights dynamically
            # Stack all teacher features
            stacked_features = torch.stack(teacher_features, dim=-2)  # [B, T, N_teachers, D]
            stacked_logits = torch.stack(teacher_logits, dim=-2)  # [B, T, N_teachers, V]

            # Compute attention weights
            avg_features = stacked_features.mean(dim=1)  # [B, N_teachers, D]
            attn_weights = self.attention_weights(avg_features)  # [B, N_teachers]
            attn_weights = attn_weights.unsqueeze(1).unsqueeze(-1)  # [B, 1, N_teachers, 1]

            # Apply attention
            agg_features = (stacked_features * attn_weights).sum(dim=-2)
            agg_logits = (stacked_logits * attn_weights).sum(dim=-2)

        elif self.aggregation_method == "max":
            # Take maximum across teachers
            stacked_features = torch.stack(teacher_features, dim=0)
            stacked_logits = torch.stack(teacher_logits, dim=0)
            agg_features = stacked_features.max(dim=0)[0]
            agg_logits = stacked_logits.max(dim=0)[0]

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        return agg_features, agg_logits

    def compute_feature_matching_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute feature matching loss between student and teacher.

        Args:
            student_features: Student features [B, T, D_proj]
            teacher_features: Teacher features [B, T, D_proj]

        Returns:
            Feature matching loss
        """
        if self.feature_matching_loss_type == "mse":
            loss = F.mse_loss(student_features, teacher_features)

        elif self.feature_matching_loss_type == "cosine":
            # Cosine embedding loss
            loss = 1 - F.cosine_similarity(
                student_features, teacher_features, dim=-1
            ).mean()

        elif self.feature_matching_loss_type == "huber":
            loss = F.smooth_l1_loss(student_features, teacher_features)

        else:
            raise ValueError(f"Unknown feature loss: {self.feature_matching_loss_type}")

        return loss

    def compute_soft_label_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute soft label distillation loss.

        Args:
            student_logits: Student phoneme logits [B, T, V]
            teacher_logits: Teacher phoneme logits [B, T, V]

        Returns:
            Soft label loss
        """
        # Apply temperature scaling
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        if self.soft_label_loss_type == "kl_div":
            # KL divergence loss
            loss = F.kl_div(
                student_probs, teacher_probs,
                reduction='batchmean', log_target=False
            ) * (self.temperature ** 2)

        elif self.soft_label_loss_type == "mse":
            # MSE on probabilities
            loss = F.mse_loss(
                F.softmax(student_logits, dim=-1),
                teacher_probs
            )

        elif self.soft_label_loss_type == "js_div":
            # Jensen-Shannon divergence
            student_probs_soft = F.softmax(student_logits / self.temperature, dim=-1)
            avg_probs = 0.5 * (student_probs_soft + teacher_probs)

            kl1 = F.kl_div(
                student_probs, avg_probs.detach(),
                reduction='batchmean', log_target=False
            )
            kl2 = F.kl_div(
                teacher_probs.log(), avg_probs.detach(),
                reduction='batchmean', log_target=False
            )
            loss = 0.5 * (kl1 + kl2) * (self.temperature ** 2)

        else:
            raise ValueError(f"Unknown soft label loss: {self.soft_label_loss_type}")

        return loss

    def forward(
        self,
        student_features: torch.Tensor,
        student_logits: torch.Tensor,
        audio_waveforms: torch.Tensor,
        feature_loss_weight: float = 1.0,
        soft_label_loss_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble distillation.

        Args:
            student_features: Student encoder features [B, T, D_student]
            student_logits: Student phoneme logits [B, T, V]
            audio_waveforms: Audio input [B, T_audio] for teachers
            feature_loss_weight: Weight for feature matching loss
            soft_label_loss_weight: Weight for soft label loss

        Returns:
            Dictionary with losses and intermediate outputs
        """
        batch_size = student_features.shape[0]
        device = student_features.device

        # Extract features from all teachers
        teacher_outputs = []
        for teacher in self.teachers:
            with torch.no_grad():
                outputs = teacher(audio_waveforms)
            teacher_outputs.append(outputs)

        # Project all features to common space
        teacher_features_proj = []
        teacher_logits_list = []

        for i, outputs in enumerate(teacher_outputs):
            # Handle sequence length mismatch via interpolation
            teacher_feat = outputs['features']  # [B, T_teacher, D_teacher]

            # Interpolate to match student sequence length
            if teacher_feat.shape[1] != student_features.shape[1]:
                teacher_feat = F.interpolate(
                    teacher_feat.transpose(1, 2),
                    size=student_features.shape[1],
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)

            # Project to common space
            teacher_feat_proj = self.teacher_projectors[i](teacher_feat)
            teacher_features_proj.append(teacher_feat_proj)

            # Get phoneme logits
            if 'phoneme_logits' in outputs:
                teacher_logits = outputs['phoneme_logits']

                # Interpolate logits if needed
                if teacher_logits.shape[1] != student_logits.shape[1]:
                    teacher_logits = F.interpolate(
                        teacher_logits.transpose(1, 2),
                        size=student_logits.shape[1],
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)

                teacher_logits_list.append(teacher_logits)

        # Aggregate teacher outputs
        aggregated_features, aggregated_logits = self.aggregate_teacher_outputs(
            teacher_features_proj, teacher_logits_list
        )

        # Project student features
        student_features_proj = self.student_projector(student_features)

        # Compute losses
        feature_loss = self.compute_feature_matching_loss(
            student_features_proj, aggregated_features
        )

        soft_label_loss = self.compute_soft_label_loss(
            student_logits, aggregated_logits
        )

        # Total distillation loss
        total_loss = (
            feature_loss_weight * feature_loss +
            soft_label_loss_weight * soft_label_loss
        )

        return {
            'distillation_loss': total_loss,
            'feature_loss': feature_loss,
            'soft_label_loss': soft_label_loss,
            'aggregated_features': aggregated_features,
            'aggregated_logits': aggregated_logits,
            'student_features_proj': student_features_proj,
            'teacher_features': teacher_features_proj,
            'teacher_logits': teacher_logits_list
        }


# Factory function for easy initialization
def create_ensemble_distillation(
    student_feature_dim: int,
    config: Optional[Dict] = None
) -> EnsembleDistillationModule:
    """
    Factory function to create ensemble distillation module.

    Args:
        student_feature_dim: Student model feature dimension
        config: Configuration dictionary with distillation settings

    Returns:
        Initialized EnsembleDistillationModule
    """
    if config is None:
        # Default configuration with 3 teachers
        config = {
            'teachers': [
                {
                    'model_name': 'openai/whisper-large-v3',
                    'model_type': 'whisper',
                    'weight': 0.4
                },
                {
                    'model_name': 'microsoft/wavlm-large',
                    'model_type': 'wavlm',
                    'weight': 0.3
                },
                {
                    'model_name': 'facebook/hubert-large-ls960-ft',
                    'model_type': 'hubert',
                    'weight': 0.3
                }
            ],
            'projection_dim': 512,
            'temperature': 4.0,
            'aggregation_method': 'weighted_average',
            'feature_matching_loss': 'mse',
            'soft_label_loss': 'kl_div'
        }

    return EnsembleDistillationModule(
        teacher_configs=config['teachers'],
        student_feature_dim=student_feature_dim,
        projection_dim=config.get('projection_dim', 512),
        distillation_temperature=config.get('temperature', 4.0),
        ensemble_weights=[t.get('weight', 1.0) for t in config['teachers']],
        aggregation_method=config.get('aggregation_method', 'weighted_average'),
        feature_matching_loss=config.get('feature_matching_loss', 'mse'),
        soft_label_loss=config.get('soft_label_loss', 'kl_div')
    )


# Unit tests
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🧪 Testing Ensemble Distillation Module...")

    # Use smaller models for testing
    config = {
        'teachers': [
            {
                'model_name': 'openai/whisper-tiny',
                'model_type': 'whisper',
                'weight': 0.5
            },
            {
                'model_name': 'microsoft/wavlm-base',
                'model_type': 'wavlm',
                'weight': 0.5
            }
        ],
        'projection_dim': 256,
        'temperature': 4.0,
        'aggregation_method': 'weighted_average'
    }

    # Create ensemble distillation module
    student_dim = 256
    ensemble = create_ensemble_distillation(student_dim, config).to(device)

    # Generate dummy data
    batch_size = 2
    seq_len = 100
    audio_len = 16000 * 5  # 5 seconds
    vocab_size = 40

    student_features = torch.randn(batch_size, seq_len, student_dim).to(device)
    student_logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
    audio_waveforms = torch.randn(batch_size, audio_len).to(device)

    # Forward pass
    print("\n🔄 Running forward pass...")
    outputs = ensemble(
        student_features=student_features,
        student_logits=student_logits,
        audio_waveforms=audio_waveforms
    )

    print("\n✅ Ensemble Distillation Test Results:")
    print(f"   Total loss: {outputs['distillation_loss'].item():.4f}")
    print(f"   Feature loss: {outputs['feature_loss'].item():.4f}")
    print(f"   Soft label loss: {outputs['soft_label_loss'].item():.4f}")
    print(f"   Aggregated features shape: {outputs['aggregated_features'].shape}")
    print(f"   Aggregated logits shape: {outputs['aggregated_logits'].shape}")
    print(f"   Number of teachers: {len(outputs['teacher_features'])}")

    print("\n🎉 All tests passed!")
