# ============================================================================
# algorithms/selection/__init__.py
# ============================================================================

"""
Selection Module

Intelligent selection strategies for reconstruction including:
- Initial pair selection for two-view initialization
- Next view selection for incremental reconstruction
- Quality-based candidate ranking
- Multi-criteria scoring

Components:
- Pair Selection: Select best image pairs for reconstruction
- View Selection: Select next cameras to add
- Scoring Utilities: Quality metrics and scoring functions
- Base Strategies: Abstract base classes for selection

Usage:
    # Initial pair selection
    from algorithms.selection import InitializationPairSelector
    
    selector = InitializationPairSelector()
    result = selector.select_best_pair(matches_data)
    
    # Next view selection
    from algorithms.selection import NextViewSelector
    
    selector = NextViewSelector()
    next_camera = selector.select_next_view(
        reconstruction_state,
        matches_data
    )
    
    # Quality scoring
    from algorithms.selection import PairScorer
    
    scorer = PairScorer()
    score = scorer.score_pair(pts1, pts2, img_size1, img_size2)

Examples:
    # Example 1: Select initialization pair
    >>> from algorithms.selection import InitializationPairSelector
    >>> 
    >>> selector = InitializationPairSelector(
    ...     min_matches=50,
    ...     min_inlier_ratio=0.4
    ... )
    >>> 
    >>> result = selector.select_best_pair(matches_data)
    >>> print(f"Best pair: {result.selected_item}")
    >>> print(f"Score: {result.score:.3f}")
    
    # Example 2: Select next camera for reconstruction
    >>> from algorithms.selection import NextViewSelector
    >>> 
    >>> selector = NextViewSelector()
    >>> result = selector.select_next_view(
    ...     existing_cameras={'img1.jpg', 'img2.jpg'},
    ...     matches_data=matches
    ... )
    >>> print(f"Next camera: {result.selected_item}")
    
    # Example 3: Score a specific pair
    >>> from algorithms.selection import PairScorer
    >>> 
    >>> scorer = PairScorer()
    >>> scores = scorer.score_pair(
    ...     pts1, pts2,
    ...     image_size1=(1920, 1080),
    ...     image_size2=(1920, 1080)
    ... )
    >>> print(f"Total score: {scores['total_score']:.3f}")
"""

# Base classes
from .strategies import (
    BaseSelector,
    SelectionResult,
    SelectionMode,
    CompositeScore,
    ScoreComponent
)

# Pair selection
from .pair_selection import (
    InitializationPairSelector,
    PairSelectionConfig
)

# Scoring utilities
from .pair_selection.scoring import (
    PairScorer,
    PairQualityMetrics,
    score_camera_connectivity,
    validate_pair_quality
)


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