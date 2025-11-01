"""
Visualization and Analysis Tools for Valerie Visual ASR.

This module contains:
- Attention weight visualization
- Feature map visualization for 3D CNN
- Training progress monitoring
- Model analysis and interpretation tools
"""

from .attention_viz import (
    AttentionVisualizer,
    plot_attention_weights,
    create_attention_heatmap
)

from .feature_viz import (
    FeatureVisualizer,
    visualize_3d_features,
    plot_feature_maps
)

from .training_dashboard import (
    TrainingDashboard,
    MetricsTracker,
    create_training_plots
)

from .model_analysis import (
    ModelAnalyzer,
    analyze_model_components,
    create_architecture_diagram
)

__all__ = [
    # Attention visualization
    'AttentionVisualizer',
    'plot_attention_weights',
    'create_attention_heatmap',
    
    # Feature visualization
    'FeatureVisualizer',
    'visualize_3d_features',
    'plot_feature_maps',
    
    # Training monitoring
    'TrainingDashboard',
    'MetricsTracker',
    'create_training_plots',
    
    # Model analysis
    'ModelAnalyzer',
    'analyze_model_components',
    'create_architecture_diagram'
]
