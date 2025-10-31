

__all__ = [
    # Base classes
    'BaseSelector',
    'SelectionResult',
    'SelectionMode',
    'CompositeScore',
    'ScoreComponent',
    
    # Pair selection
    'InitializationPairSelector',
    'PairSelectionConfig',
    
    # Scoring
    'PairScorer',
    'PairQualityMetrics',
    'score_camera_connectivity',
    'validate_pair_quality',
]

__version__ = '1.0.0'


# ============================================================================
# algorithms/selection/strategies/__init__.py
# ============================================================================

"""
Selection Strategies

Base classes and interfaces for all selection strategies.
"""

from .base_selector import (
    BaseSelector,
    SelectionResult,
    SelectionMode,
    CompositeScore,
    ScoreComponent,
    normalize_score,
    apply_threshold
)


__all__ = [
    'BaseSelector',
    'SelectionResult',
    'SelectionMode',
    'CompositeScore',
    'ScoreComponent',
    'normalize_score',
    'apply_threshold',
]

