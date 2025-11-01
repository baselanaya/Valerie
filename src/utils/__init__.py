"""
Utility functions for Valerie Visual ASR.

This module provides:
- Configuration management
- Logging utilities
- Visualization tools for attention and features
- Common helper functions
"""

from .config import Config, ModelConfig, TrainingConfig, DataConfig
from .logging import setup_logger, get_logger, log_model_summary

__all__ = [
    'Config',
    'ModelConfig',
    'TrainingConfig', 
    'DataConfig',
    'setup_logger',
    'get_logger',
    'log_model_summary'
]