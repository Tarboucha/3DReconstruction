from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Tuple

@dataclass
class MatchResult:
    """Unified result format for all matchers"""
    keypoints1: np.ndarray      # (N, 2) - x, y pixel coordinates
    keypoints2: np.ndarray      # (N, 2)
    confidence: np.ndarray      # (N,) - match confidence [0, 1]
    method: str                 # 'SIFT', 'DKM', etc.
    time_seconds: float         # Total matching time
    
    # Optional metadata
    num_features1: int = 0      # Total features detected in img1
    num_features2: int = 0      # Total features detected in img2
    metadata: dict = None       # Method-specific data


class BaseMatcher(ABC):
    """Base interface for all matchers"""
    
    @abstractmethod
    def match(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult:
        """
        Match two images.
        
        Args:
            img1: First image (H, W, 3) BGR
            img2: Second image (H, W, 3) BGR
            
        Returns:
            MatchResult with correspondences
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Matcher name"""
        pass