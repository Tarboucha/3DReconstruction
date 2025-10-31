# matchers/descriptor_matcher.py

import cv2
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class MatchingResult:
    """Result from matching"""
    matches_idx: np.ndarray    # (M, 2) - indices into keypoints1 and keypoints2
    confidence: np.ndarray     # (M,) - match confidence
    num_raw_matches: int       # Before filtering


class DescriptorMatcher(ABC):
    """Base for descriptor-based matchers"""
    
    @abstractmethod
    def match(self, desc1: np.ndarray, desc2: np.ndarray) -> MatchingResult:
        """Match descriptors"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass


class BruteForceMatcher(DescriptorMatcher):
    """Brute force matcher with ratio test"""
    
    def __init__(self, ratio_test=0.75, norm_type='L2'):
        self.ratio_test = ratio_test
        
        if norm_type == 'L2':
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif norm_type == 'HAMMING':
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
    
    @property
    def name(self):
        return "BruteForce"
    
    def match(self, desc1: np.ndarray, desc2: np.ndarray) -> MatchingResult:
        # KNN match
        raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Ratio test
        good_indices = []
        confidences = []
        
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_test * n.distance:
                    good_indices.append([m.queryIdx, m.trainIdx])
                    # Confidence based on ratio
                    conf = 1.0 - (m.distance / (n.distance + 1e-8))
                    confidences.append(conf)
        
        if len(good_indices) > 0:
            matches_idx = np.array(good_indices, dtype=np.int32)
            confidence = np.array(confidences, dtype=np.float32)
        else:
            matches_idx = np.zeros((0, 2), dtype=np.int32)
            confidence = np.zeros(0, dtype=np.float32)
        
        return MatchingResult(
            matches_idx=matches_idx,
            confidence=confidence,
            num_raw_matches=len(raw_matches)
        )


class FLANNMatcher(DescriptorMatcher):
    """FLANN-based matcher"""
    
    def __init__(self, ratio_test=0.75, descriptor_type='float32'):
        self.ratio_test = ratio_test
        
        if descriptor_type == 'float32':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        else:  # binary descriptors
            FLANN_INDEX_LSH = 6
            index_params = dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=6,
                key_size=12,
                multi_probe_level=1
            )
        
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    @property
    def name(self):
        return "FLANN"
    
    def match(self, desc1: np.ndarray, desc2: np.ndarray) -> MatchingResult:
        # Convert binary descriptors if needed
        if desc1.dtype == np.uint8:
            desc1 = desc1.astype(np.float32)
            desc2 = desc2.astype(np.float32)
        
        raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        good_indices = []
        confidences = []
        
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_test * n.distance:
                    good_indices.append([m.queryIdx, m.trainIdx])
                    conf = 1.0 - (m.distance / (n.distance + 1e-8))
                    confidences.append(conf)
        
        if len(good_indices) > 0:
            matches_idx = np.array(good_indices, dtype=np.int32)
            confidence = np.array(confidences, dtype=np.float32)
        else:
            matches_idx = np.zeros((0, 2), dtype=np.int32)
            confidence = np.zeros(0, dtype=np.float32)
        
        return MatchingResult(
            matches_idx=matches_idx,
            confidence=confidence,
            num_raw_matches=len(raw_matches)
        )