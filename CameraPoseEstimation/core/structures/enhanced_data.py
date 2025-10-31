
import numpy as np
import cv2
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Union
from .score_type import ScoreType

@dataclass
class EnhancedDMatch:
    """
    Enhanced version of cv2.DMatch with additional metadata.
    
    This extends the basic OpenCV DMatch with:
    - Explicit score type tracking
    - Standardized quality score (0-1, comparable across methods)
    - Source method tracking
    - Additional metadata fields
    
    Attributes:
        queryIdx: Index of keypoint in first image
        trainIdx: Index of keypoint in second image
        score: Primary matching score (interpretation depends on score_type)
        score_type: Type of score (DISTANCE, CONFIDENCE, SIMILARITY)
        imgIdx: Image index (for multi-image matching)
        raw_distance: Original distance value if available
        confidence: Confidence score if available
        standardized_quality: Normalized quality score (0-1, higher is better)
        source_method: Method that produced this match (e.g., 'lightglue', 'orb')
    """
    
    queryIdx: int
    trainIdx: int
    score: float
    score_type: ScoreType
    imgIdx: int = 0
    raw_distance: Optional[float] = None
    confidence: Optional[float] = None
    standardized_quality: Optional[float] = None
    source_method: Optional[str] = None
    
    @property
    def distance(self) -> float:
        """
        Backward compatibility with cv2.DMatch.distance
        
        Returns a distance-like score regardless of actual score_type.
        For confidence scores, inverts to distance-like semantics.
        """
        if self.score_type == ScoreType.DISTANCE:
            return self.score
        elif self.score_type == ScoreType.CONFIDENCE:
            # Invert confidence to distance-like score
            return 1.0 - self.score
        elif self.score_type == ScoreType.SIMILARITY:
            # Invert similarity to distance-like score
            return 1.0 - self.score
        else:
            return self.score
    
    def get_quality_score(self, higher_is_better: bool = True) -> float:
        """
        Get normalized quality score with consistent semantics.
        
        Args:
            higher_is_better: If True, return score where higher is better
                            If False, return score where lower is better
        
        Returns:
            Quality score with requested semantics
        """
        if self.standardized_quality is not None:
            # Use pre-computed standardized quality
            return self.standardized_quality if higher_is_better else (1.0 - self.standardized_quality)
        
        # Fall back to score conversion
        if self.score_type == ScoreType.CONFIDENCE or self.score_type == ScoreType.SIMILARITY:
            # Higher score = better match
            return self.score if higher_is_better else (1.0 - self.score)
        else:  # DISTANCE
            # Lower score = better match, need to invert
            normalized = 1.0 - min(self.score, 1.0)
            return normalized if higher_is_better else (1.0 - normalized)
    
    def to_cv2_dmatch(self) -> cv2.DMatch:
        """
        Convert to standard cv2.DMatch for compatibility.
        
        Returns:
            cv2.DMatch object with this match's data
        """
        match = cv2.DMatch()
        match.queryIdx = self.queryIdx
        match.trainIdx = self.trainIdx
        match.distance = self.distance
        match.imgIdx = self.imgIdx
        return match
    
    def __repr__(self):
        return (f"EnhancedDMatch(query={self.queryIdx}, train={self.trainIdx}, "
                f"score={self.score:.3f}, type={self.score_type.value}, "
                f"quality={self.standardized_quality:.3f if self.standardized_quality else 'None'})")


