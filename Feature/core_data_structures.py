"""
Core data structures and enums for the feature matching system.

This module contains the fundamental data classes and enumerations used
throughout the feature detection and matching pipeline.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import time


class DetectorType(Enum):
    """Enumeration of available detector types"""
    SIFT = "SIFT"
    ORB = "ORB"
    SURF = "SURF"
    AKAZE = "AKAZE"
    BRISK = "BRISK"
    SUPERPOINT = "SuperPoint"
    LIGHTGLUE = "LightGlue"
    DISK = "DISK"
    LOFTR = "LoFTR"
    HARRIS = "Harris"
    GOODFEATURES = "GoodFeatures"


class ScoreType(Enum):
    """Enumeration of score types"""
    DISTANCE = "distance"      # Lower is better (traditional matchers)
    CONFIDENCE = "confidence"  # Higher is better (deep learning matchers)
    SIMILARITY = "similarity"  # Higher is better


@dataclass
class FeatureData:
    """Container for feature detection results"""
    keypoints: List[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]
    method: str
    confidence_scores: Optional[List[float]] = None
    detection_time: float = 0.0
    raw_image: Optional[np.ndarray] = None
    
    def __len__(self):
        return len(self.keypoints)
    
    def to_serializable(self) -> Dict:
        """Convert to serializable format"""
        return {
            'keypoints': keypoints_to_serializable(self.keypoints),
            'descriptors': self.descriptors.tolist() if self.descriptors is not None else None,
            'method': self.method,
            'confidence_scores': self.confidence_scores,
            'detection_time': self.detection_time
            # Note: raw_image is not serialized to avoid large file sizes
        }


@dataclass
class EnhancedDMatch:
    """Enhanced DMatch with multiple score support"""
    queryIdx: int
    trainIdx: int
    score: float
    score_type: ScoreType
    imgIdx: int = 0
    raw_distance: Optional[float] = None  # Original distance if available
    confidence: Optional[float] = None    # Confidence score if available
    
    @property
    def distance(self) -> float:
        """Backward compatibility with cv2.DMatch.distance"""
        if self.score_type == ScoreType.DISTANCE:
            return self.score
        elif self.score_type == ScoreType.CONFIDENCE:
            # Convert confidence to distance-like score (invert and scale)
            return 1.0 - self.score
        else:
            return self.score
    
    def to_cv2_dmatch(self) -> cv2.DMatch:
        """Convert to cv2.DMatch for compatibility"""
        match = cv2.DMatch()
        match.queryIdx = self.queryIdx
        match.trainIdx = self.trainIdx
        match.distance = self.distance
        match.imgIdx = self.imgIdx
        return match
    
    def get_quality_score(self, higher_is_better: bool = True) -> float:
        """Get normalized quality score"""
        if self.score_type == ScoreType.CONFIDENCE or self.score_type == ScoreType.SIMILARITY:
            return self.score if higher_is_better else (1.0 - self.score)
        else:  # DISTANCE
            return (1.0 - min(self.score, 1.0)) if higher_is_better else self.score


@dataclass
class MatchData:
    """Enhanced container for feature matching results"""
    matches: List[Union[cv2.DMatch, EnhancedDMatch]]
    filtered_matches: Optional[List[Union[cv2.DMatch, EnhancedDMatch]]] = None
    homography: Optional[np.ndarray] = None
    fundamental_matrix: Optional[np.ndarray] = None
    method: str = "unknown"
    matching_time: float = 0.0
    score_type: ScoreType = ScoreType.DISTANCE
    match_confidences: Optional[np.ndarray] = None  # Raw confidence scores
    keypoint_confidences: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (kp1_conf, kp2_conf)
    
    def get_best_matches(self) -> List[Union[cv2.DMatch, EnhancedDMatch]]:
        """Return the best available matches"""
        return self.filtered_matches if self.filtered_matches else self.matches
    
    def get_match_scores(self, use_filtered: bool = True) -> np.ndarray:
        """Get match scores as numpy array"""
        matches = self.get_best_matches() if use_filtered else self.matches
        if not matches:
            return np.array([])
        
        if isinstance(matches[0], EnhancedDMatch):
            return np.array([m.score for m in matches])
        else:
            return np.array([m.distance for m in matches])
    
    def filter_by_score(self, threshold: float, top_k: Optional[int] = None) -> 'MatchData':
        """Filter matches by score threshold"""
        matches = self.matches.copy()
        
        if self.score_type == ScoreType.DISTANCE:
            # Lower distance is better
            filtered = [m for m in matches if self._get_score(m) <= threshold]
            filtered.sort(key=self._get_score)
        else:
            # Higher confidence/similarity is better
            filtered = [m for m in matches if self._get_score(m) >= threshold]
            filtered.sort(key=self._get_score, reverse=True)
        
        if top_k:
            filtered = filtered[:top_k]
        
        new_match_data = MatchData(
            matches=self.matches,
            filtered_matches=filtered,
            method=self.method,
            score_type=self.score_type,
            match_confidences=self.match_confidences,
            keypoint_confidences=self.keypoint_confidences
        )
        return new_match_data
    
    def _get_score(self, match: Union[cv2.DMatch, EnhancedDMatch]) -> float:
        """Get score from match object"""
        if isinstance(match, EnhancedDMatch):
            return match.score
        else:
            return match.distance
    
    def to_cv2_matches(self) -> List[cv2.DMatch]:
        """Convert all matches to cv2.DMatch format"""
        cv2_matches = []
        for match in self.get_best_matches():
            if isinstance(match, EnhancedDMatch):
                cv2_matches.append(match.to_cv2_dmatch())
            else:
                cv2_matches.append(match)
        return cv2_matches


def keypoints_to_serializable(keypoints: List[cv2.KeyPoint]) -> List[Dict]:
    """Convert keypoints to serializable format"""
    return [
        {
            'pt': kp.pt,
            'angle': kp.angle,
            'class_id': kp.class_id,
            'octave': kp.octave,
            'response': kp.response,
            'size': kp.size
        }
        for kp in keypoints
    ]


def keypoints_from_serializable(keypoints_data: List[Dict]) -> List[cv2.KeyPoint]:
    """Convert serialized keypoints back to cv2.KeyPoint objects"""
    keypoints = []
    for kp_data in keypoints_data:
        kp = cv2.KeyPoint(
            x=kp_data['pt'][0],
            y=kp_data['pt'][1],
            size=kp_data['size'],
            angle=kp_data['angle'],
            response=kp_data['response'],
            octave=kp_data['octave'],
            class_id=kp_data['class_id']
        )
        keypoints.append(kp)
    return keypoints