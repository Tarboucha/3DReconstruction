"""
Core data structures and enums for the feature matching system.

This module contains the fundamental data classes and enumerations used
throughout the feature detection and matching pipeline.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass, field
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
    raw_distance: Optional[float] = None
    confidence: Optional[float] = None
    source_method: Optional[str] = None 
    
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

# In core_data_structures.py - ADD THIS NEW CLASS

@dataclass
class MultiMethodMatchData:
    """
    Container for matches from multiple methods applied to the same image pair.
    
    FULL API COMPATIBILITY with MatchData - can be used anywhere MatchData is used.
    """
    
    # Method-separated matches with their metadata
    per_method_matches: Dict[str, MatchData] = field(default_factory=dict)
    
    # Combined keypoint lists (for unified indexing)
    all_keypoints1: List[cv2.KeyPoint] = field(default_factory=list)
    all_keypoints2: List[cv2.KeyPoint] = field(default_factory=list)
    
    # Index mapping: method -> (offset1, offset2)
    method_offsets: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    
    # Filtering results (applied across all methods) - COMPATIBLE WITH MatchData
    homography: Optional[np.ndarray] = None
    fundamental_matrix: Optional[np.ndarray] = None
    filtered_match_indices: Optional[List[int]] = None
    
    # ✅ ADD: Computed properties for compatibility
    _matching_time: float = 0.0
    _match_confidences: Optional[np.ndarray] = None
    _keypoint_confidences: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    # ========================================================================
    # CORE API - Matches Access (MUST HAVE)
    # ========================================================================
    
    @property
    def matches(self) -> List[EnhancedDMatch]:
        """
        Get all matches (unfiltered) - COMPATIBLE WITH MatchData
        
        Returns all matches from all methods combined.
        """
        return self.get_all_matches()
    
    @property
    def filtered_matches(self) -> Optional[List[EnhancedDMatch]]:
        """
        Get filtered matches - COMPATIBLE WITH MatchData
        
        Returns None if no filtering applied, otherwise filtered matches.
        """
        if self.filtered_match_indices is None:
            return None
        return self.get_filtered_matches()
    
    @filtered_matches.setter
    def filtered_matches(self, value: Optional[List[EnhancedDMatch]]):
        """Allow setting filtered matches"""
        if value is None:
            self.filtered_match_indices = None
        else:
            # Find indices of these matches in all_matches
            all_matches = self.get_all_matches()
            self.filtered_match_indices = [
                i for i, m in enumerate(all_matches) if m in value
            ]
    
    def get_best_matches(self) -> List[EnhancedDMatch]:
        """
        Get best matches - COMPATIBLE WITH MatchData
        
        Returns filtered matches if available, otherwise all matches.
        This is the MOST COMMONLY USED method!
        """
        if self.filtered_match_indices is not None:
            return self.get_filtered_matches()
        return self.get_all_matches()
    
    def get_all_matches(self, apply_offsets: bool = True) -> List[EnhancedDMatch]:
        """Get all matches from all methods combined"""
        all_matches = []
        
        for method, match_data in self.per_method_matches.items():
            offset1, offset2 = self.method_offsets[method]
            
            for match in match_data.matches:
                if apply_offsets:
                    if isinstance(match, EnhancedDMatch):
                        new_match = EnhancedDMatch(
                            queryIdx=match.queryIdx + offset1,
                            trainIdx=match.trainIdx + offset2,
                            score=match.score,
                            score_type=match.score_type,
                            raw_distance=match.raw_distance,
                            confidence=match.confidence,
                            source_method=method
                        )
                    else:
                        new_match = EnhancedDMatch(
                            queryIdx=match.queryIdx + offset1,
                            trainIdx=match.trainIdx + offset2,
                            score=match.distance,
                            score_type=ScoreType.DISTANCE,
                            raw_distance=match.distance,
                            source_method=method
                        )
                    all_matches.append(new_match)
                else:
                    all_matches.append(match)
        
        return all_matches
    
    def get_filtered_matches(self) -> List[EnhancedDMatch]:
        """Get filtered matches"""
        if self.filtered_match_indices is None:
            return self.get_all_matches()
        
        all_matches = self.get_all_matches()
        return [all_matches[i] for i in self.filtered_match_indices]
    
    # ========================================================================
    # SCORE API - COMPATIBLE WITH MatchData
    # ========================================================================
    
    @property
    def score_type(self) -> ScoreType:
        """
        Get dominant score type - COMPATIBLE WITH MatchData
        
        Returns CONFIDENCE if any method uses it, otherwise DISTANCE.
        """
        if not self.per_method_matches:
            return ScoreType.DISTANCE
        
        score_types = set(md.score_type for md in self.per_method_matches.values())
        
        if ScoreType.CONFIDENCE in score_types:
            return ScoreType.CONFIDENCE
        if ScoreType.SIMILARITY in score_types:
            return ScoreType.SIMILARITY
        return ScoreType.DISTANCE
    
    @property
    def method(self) -> str:
        """
        Get method name - COMPATIBLE WITH MatchData
        
        Returns combined method string like "Multi(SIFT,ORB,LightGlue)"
        """
        methods = ','.join(self.get_methods())
        return f"Multi({methods})"
    
    @property
    def matching_time(self) -> float:
        """Get total matching time - COMPATIBLE WITH MatchData"""
        if self._matching_time > 0:
            return self._matching_time
        # Sum times from all methods
        return sum(md.matching_time for md in self.per_method_matches.values())
    
    @matching_time.setter
    def matching_time(self, value: float):
        """Allow setting matching time"""
        self._matching_time = value
    
    @property
    def match_confidences(self) -> Optional[np.ndarray]:
        """Get match confidences - COMPATIBLE WITH MatchData"""
        if self._match_confidences is not None:
            return self._match_confidences
        
        # Aggregate from all methods
        all_confs = []
        for method, match_data in self.per_method_matches.items():
            if match_data.match_confidences is not None:
                all_confs.append(match_data.match_confidences)
        
        if all_confs:
            return np.concatenate(all_confs)
        return None
    
    @match_confidences.setter
    def match_confidences(self, value: Optional[np.ndarray]):
        """Allow setting match confidences"""
        self._match_confidences = value
    
    @property
    def keypoint_confidences(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get keypoint confidences - COMPATIBLE WITH MatchData"""
        return self._keypoint_confidences
    
    @keypoint_confidences.setter
    def keypoint_confidences(self, value: Optional[Tuple[np.ndarray, np.ndarray]]):
        """Allow setting keypoint confidences"""
        self._keypoint_confidences = value
    
    def get_match_scores(self, use_filtered: bool = True) -> np.ndarray:
        """
        Get match scores as array - COMPATIBLE WITH MatchData
        
        Args:
            use_filtered: Use filtered matches if available
            
        Returns:
            Array of scores
        """
        matches = self.get_best_matches() if use_filtered else self.get_all_matches()
        
        if not matches:
            return np.array([])
        
        scores = []
        for match in matches:
            if isinstance(match, EnhancedDMatch):
                scores.append(match.score)
            else:
                scores.append(match.distance)
        
        return np.array(scores)
    
    def filter_by_score(self, threshold: float, top_k: Optional[int] = None) -> 'MultiMethodMatchData':
        """
        Filter matches by score - COMPATIBLE WITH MatchData
        
        Returns a NEW MultiMethodMatchData with filtering applied.
        """
        # Create new instance with same structure
        filtered = MultiMethodMatchData()
        filtered.all_keypoints1 = self.all_keypoints1
        filtered.all_keypoints2 = self.all_keypoints2
        filtered.per_method_matches = {}
        filtered.method_offsets = self.method_offsets.copy()
        filtered.homography = self.homography
        filtered.fundamental_matrix = self.fundamental_matrix
        filtered._matching_time = self._matching_time
        
        # Filter each method independently
        for method, match_data in self.per_method_matches.items():
            filtered_method_data = match_data.filter_by_score(threshold, top_k)
            filtered.per_method_matches[method] = filtered_method_data
        
        return filtered
    
    def to_cv2_matches(self) -> List[cv2.DMatch]:
        """
        Convert to cv2.DMatch list - COMPATIBLE WITH MatchData
        """
        cv2_matches = []
        for match in self.get_best_matches():
            if isinstance(match, EnhancedDMatch):
                cv2_matches.append(match.to_cv2_dmatch())
            else:
                cv2_matches.append(match)
        return cv2_matches
    
    # ========================================================================
    # MULTI-METHOD SPECIFIC API
    # ========================================================================
    
    def add_method_matches(self, method: str, match_data: MatchData, 
                          offset1: int, offset2: int):
        """Add matches from a specific method"""
        self.per_method_matches[method] = match_data
        self.method_offsets[method] = (offset1, offset2)
    
    def get_methods(self) -> List[str]:
        """Get list of methods"""
        return list(self.per_method_matches.keys())
    
    def get_method_matches(self, method: str, with_offsets: bool = False) -> List:
        """Get matches from specific method"""
        if method not in self.per_method_matches:
            return []
        
        matches = self.per_method_matches[method].matches
        
        if with_offsets:
            offset1, offset2 = self.method_offsets[method]
            adjusted = []
            for m in matches:
                if isinstance(m, EnhancedDMatch):
                    new_m = EnhancedDMatch(
                        queryIdx=m.queryIdx + offset1,
                        trainIdx=m.trainIdx + offset2,
                        score=m.score,
                        score_type=m.score_type,
                        raw_distance=m.raw_distance,
                        confidence=m.confidence,
                        source_method=method
                    )
                else:
                    new_m = EnhancedDMatch(
                        queryIdx=m.queryIdx + offset1,
                        trainIdx=m.trainIdx + offset2,
                        score=m.distance,
                        score_type=ScoreType.DISTANCE,
                        raw_distance=m.distance,
                        source_method=method
                    )
                adjusted.append(new_m)
            return adjusted
        
        return matches
    
    def get_score_types_by_method(self) -> Dict[str, ScoreType]:
        """Get score type for each method"""
        return {
            method: match_data.score_type 
            for method, match_data in self.per_method_matches.items()
        }
    
    def has_mixed_score_types(self) -> bool:
        """Check if methods have different score types"""
        score_types = set(self.get_score_types_by_method().values())
        return len(score_types) > 1
    
    def get_match_count_by_method(self) -> Dict[str, int]:
        """Get match count per method"""
        return {
            method: len(match_data.matches)
            for method, match_data in self.per_method_matches.items()
        }
    
    def get_stats(self) -> Dict[str, Dict]:
        """Get detailed statistics per method"""
        stats = {}
        
        for method, match_data in self.per_method_matches.items():
            scores = match_data.get_match_scores()
            
            stats[method] = {
                'num_matches': len(match_data.matches),
                'score_type': match_data.score_type.value,
                'offset1': self.method_offsets[method][0],
                'offset2': self.method_offsets[method][1]
            }
            
            if len(scores) > 0:
                stats[method]['score_mean'] = float(np.mean(scores))
                stats[method]['score_std'] = float(np.std(scores))
                stats[method]['score_min'] = float(np.min(scores))
                stats[method]['score_max'] = float(np.max(scores))
        
        return stats
    
    # ========================================================================
    # SPECIAL METHODS
    # ========================================================================
    
    def __len__(self) -> int:
        """Total number of matches"""
        return len(self.get_all_matches())
    
    def __repr__(self) -> str:
        methods = ', '.join(self.get_methods())
        total = len(self)
        filtered = len(self.get_filtered_matches()) if self.filtered_match_indices else total
        return f"MultiMethodMatchData(methods=[{methods}], total={total}, filtered={filtered})"
    
    def __bool__(self) -> bool:
        """True if has any matches"""
        return len(self) > 0
