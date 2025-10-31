"""
Dense Matcher Adapter for Sparse Pipeline

Adapts dense matchers (DKM, RoMa, LoFTR) to work with the sparse pipeline.
Dense matchers don't have separate detection/matching stages - they process
entire image pairs at once. This adapter makes them compatible with the
sparse pipeline architecture.
"""

import numpy as np
import cv2
import time
from typing import Optional, Tuple

try:
    from .dense_matcher import DenseMatcher
    HAS_DENSE = True
except ImportError:
    HAS_DENSE = False

from .detectors.base_detector import BaseDetector, DetectionResult
from .base_matcher import BaseMatcher, MatchResult


class DenseDetectorAdapter(BaseDetector):
    """
    Fake detector for dense matchers

    Dense matchers don't need detection - they process entire images.
    This adapter just stores the image for later use by the matcher.
    """

    def __init__(self, method: str = 'DKM'):
        self.method = method
        self._name = method
        self._descriptor_type = 'dense'

    @property
    def name(self) -> str:
        return self._name

    @property
    def descriptor_type(self) -> str:
        return self._descriptor_type

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Store image without actual detection

        Returns empty keypoints - actual matching happens in the matcher stage
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Return empty detection - the image is stored in the matcher
        return DetectionResult(
            keypoints=np.zeros((0, 2), dtype=np.float32),
            descriptors=None,
            scores=np.zeros(0, dtype=np.float32),
            detection_time=0.0,
            image_shape=image.shape[:2]
        )


class DenseMatcherAdapter(BaseMatcher):
    """
    Adapter to run dense matcher in sparse pipeline

    Converts dense matches (pixel coordinates) to sparse format (DMatch objects)
    """

    def __init__(
        self,
        method: str = 'DKM',
        device: str = 'cuda',
        resize_to: Optional[Tuple[int, int]] = (896, 672),
        weights: str = 'gim_dkm_100h',
        max_matches: int = 8000,
        matcher_type: Optional[str] = None,  # Ignored, for compatibility
        **kwargs  # Ignore other kwargs
    ):
        if not HAS_DENSE:
            raise ImportError("Dense matching requires torch. Install with: pip install torch")

        self._name = method
        self.method = method
        self.device = device
        self.resize_to = resize_to
        self.max_matches = max_matches

        # Create dense matcher
        self.dense_matcher = DenseMatcher(
            method=method,
            device=device,
            resize_to=resize_to,
            weights=weights,
            max_matches=max_matches
        )

        # Store images for matching
        self.image1 = None
        self.image2 = None

    @property
    def name(self) -> str:
        return self._name

    def set_images(self, image1: np.ndarray, image2: np.ndarray):
        """Store images for matching"""
        self.image1 = image1
        self.image2 = image2

    def match(
        self,
        descriptors1: np.ndarray,
        descriptors2: np.ndarray,
        kp1: Optional[np.ndarray] = None,
        kp2: Optional[np.ndarray] = None,
        image1: Optional[np.ndarray] = None,
        image2: Optional[np.ndarray] = None
    ) -> MatchResult:
        """
        Run dense matching and convert to sparse format

        Note: Ignores descriptors/keypoints from detection stage since
        dense matching processes the entire images.
        """
        # Use stored images if not provided
        if image1 is None:
            image1 = self.image1
        if image2 is None:
            image2 = self.image2

        if image1 is None or image2 is None:
            raise ValueError("Images must be provided for dense matching")

        # Ensure RGB
        if len(image1.shape) == 2:
            image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
        if len(image2.shape) == 2:
            image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)

        # Run dense matching
        start_time = time.time()
        dense_result = self.dense_matcher.match(image1, image2)
        matching_time = time.time() - start_time

        # Extract matches: Nx4 array [x1, y1, x2, y2]
        matches_array = dense_result.matches

        if len(matches_array) == 0:
            return MatchResult(
                matches=[],
                match_confidence=np.array([]),
                num_matches=0,
                matching_time=matching_time
            )

        # Convert to DMatch format
        # For dense matches, we create synthetic keypoint indices
        matches = []
        confidences = []

        for i, (x1, y1, x2, y2) in enumerate(matches_array):
            match = cv2.DMatch(
                _queryIdx=i,
                _trainIdx=i,
                _distance=0.0  # Dense matches don't have distance
            )
            matches.append(match)
            confidences.append(1.0)  # Dense matches are already filtered

        return MatchResult(
            matches=matches,
            match_confidence=np.array(confidences, dtype=np.float32),
            num_matches=len(matches),
            matching_time=matching_time,
            # Store actual coordinates for reconstruction
            metadata={
                'dense_matches': matches_array,  # Store original Nx4 array
                'method': self.method
            }
        )


def create_dense_pipeline(
    method: str = 'DKM',
    device: str = 'cuda',
    resize_to: Optional[Tuple[int, int]] = (896, 672),
    weights: str = 'gim_dkm_100h',
    max_matches: int = 8000
):
    """
    Create detector + matcher pair for dense matching

    Returns:
        tuple: (detector, matcher) compatible with sparse pipeline
    """
    detector = DenseDetectorAdapter(method=method)

    matcher = DenseMatcherAdapter(
        method=method,
        device=device,
        resize_to=resize_to,
        weights=weights,
        max_matches=max_matches
    )

    return detector, matcher
