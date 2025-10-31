"""
Data loaders and transformers.

This module provides utilities for loading and transforming raw data
into standardized formats.

Components:
    - MatchQualityStandardizer: Normalize match quality scores
    
Usage:
    from data.loaders import MatchQualityStandardizer
    from core.structures import ScoreType
    
    # Create standardizer
    standardizer = MatchQualityStandardizer()
    
    # Standardize quality scores
    standardized_quality = standardizer.standardize_quality(
        score=50.0,
        score_type=ScoreType.DISTANCE,
        num_matches=150
    )
"""

from .match_loader import MatchQualityStandardizer


__all__ = [
    'MatchQualityStandardizer',
]


__version__ = '1.0.0'