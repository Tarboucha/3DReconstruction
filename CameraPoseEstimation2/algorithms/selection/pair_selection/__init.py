
# ============================================================================
# algorithms/selection/pair_selection/__init__.py
# ============================================================================

"""
Pair Selection Module

Strategies for selecting image pairs for reconstruction initialization
and incremental view addition.
"""

# Note: The actual InitializationPairSelector implementation exists in
# algorithms/selection/pair_selection/initialization.py
# We just re-export it here for convenience

# Import will work once initialization.py is refactored
# For now, this provides the module structure

try:
    from .initialization import (
        InitializationPairSelector,
        PairSelectionConfig
    )
except ImportError:
    # Fallback if old structure
    print("Warning: Using legacy pair selector")
    from CameraPoseEstimation2.algorithms.selection.pair_selection.initialization import (
        InitializationPairSelector
    )
    
    class PairSelectionConfig:
        """Placeholder config"""
        pass

from .scoring import (
    PairScorer,
    PairQualityMetrics,
    score_camera_connectivity,
    validate_pair_quality
)


__all__ = [
    'InitializationPairSelector',
    'PairSelectionConfig',
    'PairScorer',
    'PairQualityMetrics',
    'score_camera_connectivity',
    'validate_pair_quality',
]