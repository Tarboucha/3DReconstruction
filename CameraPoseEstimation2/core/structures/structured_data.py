
import numpy as np
import cv2
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Union
from .score_type import ScoreType
from .enhanced_data import EnhancedDMatch

@dataclass
class StructuredMatchData:
    """
    Rich container for feature matching results between two images.
    
    This is the primary data structure for representing matches in the
    pose estimation pipeline. It preserves all per-match information
    while providing convenient access methods.
    
    Attributes:
        matches: List of individual match objects with full metadata
        keypoints1: Keypoints detected in first image
        keypoints2: Keypoints detected in second image
        method: Matching method used (e.g., 'lightglue', 'orb', 'sift')
        score_type: Type of scores in matches
        num_matches: Total number of matches
        standardized_pair_quality: Overall quality score for this pair (0-1)
        match_quality_stats: Statistics about match quality distribution
        homography: Homography matrix if computed (3x3)
        fundamental_matrix: Fundamental matrix if computed (3x3)
        inlier_mask: Boolean mask indicating inlier matches
        matching_time: Time taken for matching (seconds)
    """
    
    # Core match data
    matches: List[EnhancedDMatch]
    keypoints1: List[cv2.KeyPoint]
    keypoints2: List[cv2.KeyPoint]
    
    # Metadata
    method: str
    score_type: ScoreType
    num_matches: int
    
    # Quality metrics
    standardized_pair_quality: float
    match_quality_stats: Dict[str, float]
    
    # Geometric verification (optional)
    homography: Optional[np.ndarray] = None
    fundamental_matrix: Optional[np.ndarray] = None
    inlier_mask: Optional[np.ndarray] = None
    
    # Timing
    matching_time: float = 0.0

    image1_size: Optional[Tuple[int, int]] = None
    image2_size: Optional[Tuple[int, int]] = None
    
    # =========================================================================
    # Backward Compatibility Properties
    # =========================================================================
    
    @property
    def correspondences(self) -> np.ndarray:
        """Get correspondences as (N, 4) array of [x1, y1, x2, y2]"""
        if len(self.matches) == 0:
            return np.empty((0, 4), dtype=np.float32)
        
        corr_list = []
        for match in self.matches:
            idx1 = match.queryIdx
            idx2 = match.trainIdx
            
            # Handle both numpy arrays and KeyPoint lists
            if isinstance(self.keypoints1, np.ndarray):
                pt1 = self.keypoints1[idx1]
            else:
                pt1 = self.keypoints1[idx1].pt if hasattr(self.keypoints1[idx1], 'pt') else self.keypoints1[idx1]
            
            if isinstance(self.keypoints2, np.ndarray):
                pt2 = self.keypoints2[idx2]
            else:
                pt2 = self.keypoints2[idx2].pt if hasattr(self.keypoints2[idx2], 'pt') else self.keypoints2[idx2]
            
            corr_list.append([pt1[0], pt1[1], pt2[0], pt2[1]])
    
        return np.array(corr_list, dtype=np.float32)
    
    @property
    def pts1(self) -> np.ndarray:
        """
        Get matched points from first image.
        
        Returns:
            Array of shape (N, 2) containing (x, y) coordinates
        """
        if not self.matches:
            return np.array([]).reshape(0, 2)
        
        # Handle both numpy arrays and KeyPoint lists, always index by matches
        if isinstance(self.keypoints1, np.ndarray):
            indices = [m.queryIdx for m in self.matches]
            return self.keypoints1[indices].astype(np.float32)
        else:
            return np.array([self.keypoints1[m.queryIdx].pt for m in self.matches], 
                           dtype=np.float32)
    
    @property
    def pts2(self) -> np.ndarray:
        """
        Get matched points from second image.
        
        Returns:
            Array of shape (N, 2) containing (x, y) coordinates
        """
        if not self.matches:
            return np.array([]).reshape(0, 2)
        
        # Handle both numpy arrays and KeyPoint lists, always index by matches
        if isinstance(self.keypoints2, np.ndarray):
            indices = [m.trainIdx for m in self.matches]
            return self.keypoints2[indices].astype(np.float32)
        else:
            return np.array([self.keypoints2[m.trainIdx].pt for m in self.matches], 
                           dtype=np.float32)
    
    @property
    def match_scores(self) -> np.ndarray:
        """
        Get all raw match scores.
        
        Returns:
            Array of shape (N,) containing match scores
        """
        return np.array([m.score for m in self.matches], dtype=np.float32)
    
    @property
    def match_qualities(self) -> np.ndarray:
        """
        Get standardized quality scores for all matches.
        
        Returns:
            Array of shape (N,) with normalized quality scores (0-1)
        """
        qualities = [m.standardized_quality for m in self.matches 
                    if m.standardized_quality is not None]
        
        if not qualities:
            # Fallback: use raw scores
            return self.match_scores
        
        return np.array(qualities, dtype=np.float32)
    
    def __getitem__(self, key):
        """Support dictionary-style access: data['correspondences']"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in StructuredMatchData")

    def __contains__(self, key):
        """Support 'in' operator: 'correspondences' in data"""
        return hasattr(self, key)

    def get(self, key, default=None):
        """Dict-style get with default"""
        return getattr(self, key, default)

    def keys(self):
        """Return available attributes"""
        return [k for k in dir(self) if not k.startswith('_')]

    # =========================================================================
    # Filtering and Manipulation
    # =========================================================================
    
    def get_top_k_matches(self, k: int) -> 'StructuredMatchData':
        """
        Get top-k matches by quality.
        
        Args:
            k: Number of top matches to return
            
        Returns:
            New StructuredMatchData with only top-k matches
        """
        if k >= len(self.matches):
            return self
        
        # Sort by standardized quality (or score if quality not available)
        sorted_matches = sorted(
            self.matches,
            key=lambda m: m.standardized_quality if m.standardized_quality is not None else m.score,
            reverse=True
        )
        
        top_matches = sorted_matches[:k]
        
        return StructuredMatchData(
            matches=top_matches,
            keypoints1=self.keypoints1,
            keypoints2=self.keypoints2,
            method=self.method,
            score_type=self.score_type,
            num_matches=len(top_matches),
            standardized_pair_quality=self.standardized_pair_quality,
            match_quality_stats=self._recompute_stats(top_matches),
            homography=self.homography,
            fundamental_matrix=self.fundamental_matrix,
            matching_time=self.matching_time
        )
    
    def filter_by_quality(self, min_quality: float) -> 'StructuredMatchData':
        """
        Filter matches by minimum quality threshold.
        
        Args:
            min_quality: Minimum standardized quality score (0-1)
            
        Returns:
            New StructuredMatchData with filtered matches
        """
        filtered = [m for m in self.matches 
                   if m.standardized_quality and m.standardized_quality >= min_quality]
        
        if not filtered:
            filtered = []
        
        return StructuredMatchData(
            matches=filtered,
            keypoints1=self.keypoints1,
            keypoints2=self.keypoints2,
            method=self.method,
            score_type=self.score_type,
            num_matches=len(filtered),
            standardized_pair_quality=self.standardized_pair_quality,
            match_quality_stats=self._recompute_stats(filtered),
            homography=self.homography,
            fundamental_matrix=self.fundamental_matrix,
            matching_time=self.matching_time
        )
    
    def filter_by_inliers(self) -> 'StructuredMatchData':
        """
        Filter to only inlier matches (requires inlier_mask).
        
        Returns:
            New StructuredMatchData with only inlier matches
            
        Raises:
            ValueError: If inlier_mask is not set
        """
        if self.inlier_mask is None:
            raise ValueError("No inlier mask available")
        
        inlier_matches = [m for i, m in enumerate(self.matches) 
                         if i < len(self.inlier_mask) and self.inlier_mask[i]]
        
        return StructuredMatchData(
            matches=inlier_matches,
            keypoints1=self.keypoints1,
            keypoints2=self.keypoints2,
            method=self.method,
            score_type=self.score_type,
            num_matches=len(inlier_matches),
            standardized_pair_quality=self.standardized_pair_quality,
            match_quality_stats=self._recompute_stats(inlier_matches),
            homography=self.homography,
            fundamental_matrix=self.fundamental_matrix,
            inlier_mask=None,  # Reset mask since we're now all inliers
            matching_time=self.matching_time
        )
    
    def _recompute_stats(self, matches: List[EnhancedDMatch]) -> Dict[str, float]:
        """Recompute quality statistics for a subset of matches"""
        if not matches:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
        
        qualities = [m.standardized_quality for m in matches 
                    if m.standardized_quality is not None]
        
        if not qualities:
            scores = [m.score for m in matches]
            qualities = scores
        
        qualities = np.array(qualities)
        
        return {
            'mean': float(np.mean(qualities)),
            'std': float(np.std(qualities)),
            'min': float(np.min(qualities)),
            'max': float(np.max(qualities)),
            'median': float(np.median(qualities))
        }
    
    # =========================================================================
    # Conversion Methods
    # =========================================================================
    
    def to_cv2_matches(self) -> List[cv2.DMatch]:
        """
        Convert to list of cv2.DMatch objects.
        
        Returns:
            List of standard OpenCV DMatch objects
        """
        return [m.to_cv2_dmatch() for m in self.matches]
    
    def to_dict(self) -> Dict:
        """
        Convert to dictionary format for serialization.
        
        Returns:
            Dictionary representation suitable for JSON/pickle
        """
        return {
            'num_matches': self.num_matches,
            'method': self.method,
            'score_type': self.score_type.value,
            'standardized_pair_quality': self.standardized_pair_quality,
            'match_quality_stats': self.match_quality_stats,
            'correspondences': self.correspondences.tolist(),
            'match_scores': self.match_scores.tolist(),
            'match_qualities': self.match_qualities.tolist(),
            'matching_time': self.matching_time,
            'has_homography': self.homography is not None,
            'has_fundamental': self.fundamental_matrix is not None
        }
    
    # =========================================================================
    # Analysis Methods
    # =========================================================================
    
    def get_spatial_distribution(self) -> Dict[str, float]:
        """
        Analyze spatial distribution of matches in images.
        
        Returns:
            Dictionary with distribution statistics
        """
        pts1 = self.pts1
        pts2 = self.pts2
        
        if len(pts1) == 0:
            return {'coverage': 0.0, 'std_x1': 0.0, 'std_y1': 0.0, 
                   'std_x2': 0.0, 'std_y2': 0.0}
        
        return {
            'coverage': self._compute_coverage(pts1),
            'std_x1': float(np.std(pts1[:, 0])),
            'std_y1': float(np.std(pts1[:, 1])),
            'std_x2': float(np.std(pts2[:, 0])),
            'std_y2': float(np.std(pts2[:, 1])),
            'mean_displacement_x': float(np.mean(pts2[:, 0] - pts1[:, 0])),
            'mean_displacement_y': float(np.mean(pts2[:, 1] - pts1[:, 1]))
        }
    
    def _compute_coverage(self, pts: np.ndarray) -> float:
        """Compute how well matches cover the image"""
        if len(pts) < 4:
            return 0.0
        
        # Compute convex hull area as proxy for coverage
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts)
            return float(hull.volume)  # Area in 2D
        except:
            # Fallback: bounding box area
            min_x, min_y = pts.min(axis=0)
            max_x, max_y = pts.max(axis=0)
            return float((max_x - min_x) * (max_y - min_y))
    
    def get_quality_histogram(self, bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get histogram of match qualities.
        
        Args:
            bins: Number of histogram bins
            
        Returns:
            (counts, bin_edges) tuple
        """
        qualities = self.match_qualities
        if len(qualities) == 0:
            return np.array([]), np.array([])
        
        return np.histogram(qualities, bins=bins, range=(0, 1))
    
    # =========================================================================
    # String Representation
    # =========================================================================
    
    def __repr__(self):
        return (f"StructuredMatchData(matches={self.num_matches}, "
                f"method='{self.method}', "
                f"quality={self.standardized_pair_quality:.3f}, "
                f"score_type={self.score_type.value})")
    
    def __str__(self):
        return self.summary()
    
    def summary(self) -> str:
        """Get human-readable summary"""
        lines = [
            "StructuredMatchData Summary",
            "=" * 50,
            f"Method: {self.method}",
            f"Score Type: {self.score_type.value}",
            f"Number of Matches: {self.num_matches}",
            f"Overall Quality: {self.standardized_pair_quality:.3f}",
            "",
            "Match Quality Statistics:",
            f"  Mean: {self.match_quality_stats['mean']:.3f}",
            f"  Std:  {self.match_quality_stats['std']:.3f}",
            f"  Min:  {self.match_quality_stats['min']:.3f}",
            f"  Max:  {self.match_quality_stats['max']:.3f}",
            "",
            f"Keypoints: {len(self.keypoints1)} / {len(self.keypoints2)}",
            f"Matching Time: {self.matching_time:.3f}s",
        ]
        
        if self.homography is not None:
            lines.append("Homography: Available")
        if self.fundamental_matrix is not None:
            lines.append("Fundamental Matrix: Available")
        if self.inlier_mask is not None:
            inlier_count = np.sum(self.inlier_mask)
            lines.append(f"Inliers: {inlier_count}/{len(self.inlier_mask)}")
        
        return "\n".join(lines)

    def has_match_qualities(self) -> bool:
        """
        Check if per-match quality scores are available.
        
        Returns:
            bool: True if match_qualities is populated, False otherwise
        """
        return self.match_qualities is not None and len(self.match_qualities) > 0


    def has_keypoints(self) -> bool:
        """
        Check if keypoint information is available.
        
        Returns:
            bool: True if keypoints are available for both images
        """
        return (self.keypoints1 is not None and 
                self.keypoints2 is not None and
                len(self.keypoints1) > 0 and 
                len(self.keypoints2) > 0)


    def has_matches(self) -> bool:
        """
        Check if match objects are available (vs just point coordinates).
        
        Returns:
            bool: True if matches list is populated
        """
        return self.matches is not None and len(self.matches) > 0


    def get_quality_stats(self) -> dict:
        """
        Get statistics about match qualities.
        
        Returns:
            dict: Statistics including mean, std, min, max, median
                Returns empty dict if qualities not available
        """
        if not self.has_match_qualities():
            return {}
        
        import numpy as np
        
        qualities = np.array(self.match_qualities)
        
        return {
            'mean': float(np.mean(qualities)),
            'std': float(np.std(qualities)),
            'min': float(np.min(qualities)),
            'max': float(np.max(qualities)),
            'median': float(np.median(qualities)),
            'q25': float(np.percentile(qualities, 25)),
            'q75': float(np.percentile(qualities, 75))
        }


    def filter_by_quality(self, min_quality: float = 0.0, max_quality: float = 1.0):
        """
        Filter matches based on quality threshold.
        
        Args:
            min_quality: Minimum quality threshold (inclusive)
            max_quality: Maximum quality threshold (inclusive)
        
        Returns:
            StructuredMatchData: New instance with filtered matches
        
        Raises:
            ValueError: If match qualities are not available
        """
        if not self.has_match_qualities():
            raise ValueError("Cannot filter by quality - match qualities not available")
        
        import numpy as np
        from structures import StructuredMatchData, EnhancedDMatch
        
        # Find indices of matches that pass quality filter
        mask = np.logical_and(
            np.array(self.match_qualities) >= min_quality,
            np.array(self.match_qualities) <= max_quality
        )
        
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            # Return empty match data
            return StructuredMatchData(
                image1=self.image1,
                image2=self.image2,
                pts1=np.array([]),
                pts2=np.array([]),
                descriptors1=self.descriptors1,
                descriptors2=self.descriptors2
            )
        
        # Filter all arrays
        filtered_pts1 = self.pts1[indices]
        filtered_pts2 = self.pts2[indices]
        filtered_qualities = [self.match_qualities[i] for i in indices]
        
        # Filter matches if available
        filtered_matches = None
        if self.has_matches():
            filtered_matches = [self.matches[i] for i in indices]
        
        # Filter keypoints if available
        filtered_kp1 = None
        filtered_kp2 = None
        if self.has_keypoints():
            filtered_kp1 = [self.keypoints1[i] for i in indices]
            filtered_kp2 = [self.keypoints2[i] for i in indices]
        
        # Create new instance
        return StructuredMatchData(
            image1=self.image1,
            image2=self.image2,
            pts1=filtered_pts1,
            pts2=filtered_pts2,
            matches=filtered_matches,
            keypoints1=filtered_kp1,
            keypoints2=filtered_kp2,
            descriptors1=self.descriptors1,
            descriptors2=self.descriptors2,
            match_qualities=filtered_qualities,
            standardized_pair_quality=self.standardized_pair_quality  # Keep same overall quality
        )


    def get_top_matches(self, k: int):
        """
        Get the top k matches by quality.
        
        Args:
            k: Number of top matches to return
        
        Returns:
            StructuredMatchData: New instance with only top k matches
        
        Raises:
            ValueError: If match qualities are not available
        """
        if not self.has_match_qualities():
            raise ValueError("Cannot get top matches - match qualities not available")
        
        import numpy as np
        
        # Get indices of top k matches
        qualities = np.array(self.match_qualities)
        top_indices = np.argsort(qualities)[-k:][::-1]  # Descending order
        
        # Filter using these indices
        filtered_pts1 = self.pts1[top_indices]
        filtered_pts2 = self.pts2[top_indices]
        filtered_qualities = [self.match_qualities[i] for i in top_indices]
        
        filtered_matches = None
        if self.has_matches():
            filtered_matches = [self.matches[i] for i in top_indices]
        
        filtered_kp1 = None
        filtered_kp2 = None
        if self.has_keypoints():
            filtered_kp1 = [self.keypoints1[i] for i in top_indices]
            filtered_kp2 = [self.keypoints2[i] for i in top_indices]
        
        return StructuredMatchData(
            image1=self.image1,
            image2=self.image2,
            pts1=filtered_pts1,
            pts2=filtered_pts2,
            matches=filtered_matches,
            keypoints1=filtered_kp1,
            keypoints2=filtered_kp2,
            descriptors1=self.descriptors1,
            descriptors2=self.descriptors2,
            match_qualities=filtered_qualities,
            standardized_pair_quality=self.standardized_pair_quality
        )




def create_minimal_match_data(correspondences: np.ndarray,
                              method: str = "unknown",
                              quality: float = 0.5) -> StructuredMatchData:
    """
    Create minimal StructuredMatchData from correspondence array.
    
    Useful for converting old format data or creating test data.
    
    Args:
        correspondences: Array of shape (N, 4) with [x1, y1, x2, y2]
        method: Matching method name
        quality: Overall quality score
        
    Returns:
        StructuredMatchData with minimal keypoints and matches
    """
    if len(correspondences) == 0:
        correspondences = np.array([]).reshape(0, 4)
    
    n = len(correspondences)
    
    # Create minimal keypoints
    kp1 = [cv2.KeyPoint(x=float(c[0]), y=float(c[1]), size=1.0) 
           for c in correspondences]
    kp2 = [cv2.KeyPoint(x=float(c[2]), y=float(c[3]), size=1.0) 
           for c in correspondences]
    
    # Create matches
    matches = [
        EnhancedDMatch(
            queryIdx=i,
            trainIdx=i,
            score=quality,
            score_type=ScoreType.CONFIDENCE,
            standardized_quality=quality,
            source_method=method
        )
        for i in range(n)
    ]
    
    # Quality stats
    stats = {
        'mean': quality,
        'std': 0.0,
        'min': quality,
        'max': quality,
        'median': quality
    }
    
    return StructuredMatchData(
        matches=matches,
        keypoints1=kp1,
        keypoints2=kp2,
        method=method,
        score_type=ScoreType.CONFIDENCE,
        num_matches=n,
        standardized_pair_quality=quality,
        match_quality_stats=stats
    )
