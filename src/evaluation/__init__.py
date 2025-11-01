"""
Evaluation components for Valerie Visual ASR.

This module contains:
- Comprehensive evaluation pipeline for test sets
- Component-wise performance analysis
- Statistical analysis and confidence intervals
- Performance benchmarking and comparison tools
"""

from .evaluator import (
    ValerieEvaluator,
    EvaluationConfig,
    EvaluationResult,
    ComponentMetrics
)

__all__ = [
    # Core evaluation
    'ValerieEvaluator',
    'EvaluationConfig',
    'EvaluationResult',
    'ComponentMetrics'
]
