# matchers/sparse_pipeline.py

import time
import numpy as np
from .base_matcher import BaseMatcher, MatchResult
from .detectors.base_detector import BaseDetector


class SparsePipeline(BaseMatcher):
    """
    Flexible sparse matching pipeline.
    Combines any detector with any compatible matcher.
    """
    
    def __init__(self, detector: BaseDetector, matcher):
        """
        Args:
            detector: Any BaseDetector (SIFT, ALIKED, SuperPoint, etc.)
            matcher: Any descriptor matcher (BruteForce, FLANN, LightGlue)
        """
        self.detector = detector
        self.matcher = matcher
        
        # Validate compatibility
        self._validate_compatibility()
    
    def _validate_compatibility(self):
        """Check if detector and matcher are compatible"""
        # LightGlue only works with SuperPoint
        if self.matcher.name == "LightGlue":
            if self.detector.name != "SuperPoint":
                raise ValueError(
                    f"LightGlue only works with SuperPoint detector, "
                    f"got {self.detector.name}"
                )
    
    @property
    def name(self):
        return f"{self.detector.name}+{self.matcher.name}"
    
    def match(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult:
        """Match using detector + matcher pipeline"""
        start_time = time.time()
        
        # Detect features
        det1 = self.detector.detect(img1)
        det2 = self.detector.detect(img2)
        
        if len(det1.keypoints) < 2 or len(det2.keypoints) < 2:
            return MatchResult(
                keypoints1=np.array([]),
                keypoints2=np.array([]),
                confidence=np.array([]),
                method=self.name,
                time_seconds=time.time() - start_time,
                num_features1=len(det1.keypoints),
                num_features2=len(det2.keypoints)
            )
        
        # Match descriptors
        if self.matcher.name == "LightGlue":
            # LightGlue needs keypoints too
            match_result = self.matcher.match(
                det1.keypoints, det1.descriptors,
                det2.keypoints, det2.descriptors
            )
        else:
            # Descriptor-only matching
            match_result = self.matcher.match(det1.descriptors, det2.descriptors)
        
        # Extract matched keypoint coordinates
        if len(match_result.matches_idx) > 0:
            idx1 = match_result.matches_idx[:, 0]
            idx2 = match_result.matches_idx[:, 1]
            
            keypoints1 = det1.keypoints[idx1]
            keypoints2 = det2.keypoints[idx2]
            confidence = match_result.confidence
        else:
            keypoints1 = np.array([])
            keypoints2 = np.array([])
            confidence = np.array([])
        
        elapsed = time.time() - start_time
        
        return MatchResult(
            keypoints1=keypoints1,
            keypoints2=keypoints2,
            confidence=confidence,
            method=self.name,
            time_seconds=elapsed,
            num_features1=len(det1.keypoints),
            num_features2=len(det2.keypoints),
            metadata={
                'detector': self.detector.name,
                'matcher': self.matcher.name,
                'num_raw_matches': match_result.num_raw_matches,
                'num_filtered_matches': len(match_result.matches_idx)
            }
        )