"""
Pair Selection Module

Strategies for selecting image pairs for reconstruction initialization
and incremental view addition.

Components:
- InitializationPairSelector: Select best initialization pairs
- PairSelectionConfig: Configuration for pair selection
- Scoring utilities: Quality metrics for pair evaluation
"""

from .initialization import (
    InitializationPairSelector,
    ScoringConfig as PairSelectionConfig,
    create_monument_pair_selector
)

from .scoring import (
    PairScorer,
    PairQualityMetrics,
    score_camera_connectivity,
    validate_pair_quality
)


__all__ = [
    'InitializationPairSelector',
    'PairSelectionConfig',
    'create_monument_pair_selector',
    'PairScorer',
    'PairQualityMetrics',
    'score_camera_connectivity',
    'validate_pair_quality',
]


__version__ = '1.0.0'