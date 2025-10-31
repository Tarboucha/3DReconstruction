"""
Matchers module: Feature detection and matching.

Provides:
- Dense matchers: DKM, RoMa, LoFTR (full-image matching)
- Sparse matchers: SIFT, ORB, AKAZE with FLANN/BruteForce
- Flexible pipeline: Mix any detector with any matcher

Examples:
    >>> # High-level API
    >>> from FeatureMatchingExtraction.matchers import create_matcher
    >>> matcher = create_matcher('DKM', device='cuda')
    >>> result = matcher.match(img1, img2)

    >>> # Dense matcher directly
    >>> from FeatureMatchingExtraction.matchers import create_dense_matcher
    >>> matcher = create_dense_matcher('DKM', symmetric=False)

    >>> # Build custom pipeline
    >>> from FeatureMatchingExtraction.matchers import SparsePipeline, SIFTDetector, FLANNMatcher
    >>> pipeline = SparsePipeline(detector=SIFTDetector(), matcher=FLANNMatcher())
"""

from .base_matcher import BaseMatcher, MatchResult
from .dense_matcher import DenseMatcher, create_dense_matcher
from .sparse_pipeline import SparsePipeline
from .factory import create_matcher

# Detectors
from .detectors.base_detector import BaseDetector, DetectionResult
from .detectors.traditional import SIFTDetector, ORBDetector, AKAZEDetector
from .detectors.deep_learning import SuperPointDetector, ALIKEDDetector

# Descriptor matchers
from .matching_algorithms.descriptor_matcher import (
    BruteForceMatcher,
    FLANNMatcher,
    MatchingResult
)
from .matching_algorithms.lightglue_matcher import LightGlueMatcher

__all__ = [
    # Base classes
    'BaseMatcher',
    'MatchResult',
    'BaseDetector',
    'DetectionResult',

    # High-level API
    'create_matcher',
    'create_dense_matcher',

    # Matchers
    'DenseMatcher',
    'SparsePipeline',

    # Detectors
    'SIFTDetector',
    'ORBDetector',
    'AKAZEDetector',
    'SuperPointDetector',
    'ALIKEDDetector',

    # Descriptor matchers
    'BruteForceMatcher',
    'FLANNMatcher',
    'LightGlueMatcher',
    'MatchingResult',
]

__version__ = '0.1.0'
