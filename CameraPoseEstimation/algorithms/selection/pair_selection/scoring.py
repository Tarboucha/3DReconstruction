"""
Quality Scoring Utilities

Functions for scoring image pairs and views based on various quality criteria.
Used by selection strategies to rank candidates.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Set
from dataclasses import dataclass


@dataclass
class PairQualityMetrics:
    """Quality metrics for an image pair"""
    num_matches: int
    inlier_ratio: float
    coverage_score: float
    distribution_score: float
    baseline_score: float
    overlap_score: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'num_matches': self.num_matches,
            'inlier_ratio': self.inlier_ratio,
            'coverage_score': self.coverage_score,
            'distribution_score': self.distribution_score,
            'baseline_score': self.baseline_score,
            'overlap_score': self.overlap_score
        }


class PairScorer:
    """
    Scores image pairs for reconstruction suitability.
    
    Considers multiple factors:
    - Number of matches
    - Inlier ratio (quality)
    - Spatial coverage
    - Point distribution
    - Baseline (parallax)
    - Overlap
    """
    
    def __init__(self,
                 match_weight: float = 0.25,
                 quality_weight: float = 0.25,
                 coverage_weight: float = 0.20,
                 distribution_weight: float = 0.15,
                 baseline_weight: float = 0.10,
                 overlap_weight: float = 0.05):
        """
        Initialize pair scorer with component weights.
        
        Args:
            match_weight: Weight for number of matches
            quality_weight: Weight for inlier ratio
            coverage_weight: Weight for spatial coverage
            distribution_weight: Weight for point distribution
            baseline_weight: Weight for baseline/parallax
            overlap_weight: Weight for image overlap
        """
        self.weights = {
            'matches': match_weight,
            'quality': quality_weight,
            'coverage': coverage_weight,
            'distribution': distribution_weight,
            'baseline': baseline_weight,
            'overlap': overlap_weight
        }
        
        # Normalize weights
        total = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total
    
    def score_pair(self,
                  pts1: np.ndarray,
                  pts2: np.ndarray,
                  image_size1: Tuple[int, int],
                  image_size2: Tuple[int, int],
                  inlier_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute comprehensive quality score for an image pair.
        
        Args:
            pts1: Points in first image (Nx2)
            pts2: Points in second image (Nx2)
            image_size1: Size of first image (width, height)
            image_size2: Size of second image (width, height)
            inlier_mask: Optional mask indicating inliers
            
        Returns:
            Dictionary with scores and total
        """
        if inlier_mask is not None:
            inlier_pts1 = pts1[inlier_mask]
            inlier_pts2 = pts2[inlier_mask]
            inlier_ratio = np.sum(inlier_mask) / len(inlier_mask)
        else:
            inlier_pts1 = pts1
            inlier_pts2 = pts2
            inlier_ratio = 1.0
        
        # Compute individual scores
        match_score = self._score_match_count(len(pts1))
        quality_score = inlier_ratio
        coverage_score = self._score_coverage(inlier_pts1, inlier_pts2, 
                                             image_size1, image_size2)
        distribution_score = self._score_distribution(inlier_pts1, inlier_pts2)
        baseline_score = self._score_baseline(inlier_pts1, inlier_pts2)
        overlap_score = self._score_overlap(inlier_pts1, inlier_pts2,
                                           image_size1, image_size2)
        
        # Weighted total
        total_score = (
            match_score * self.weights['matches'] +
            quality_score * self.weights['quality'] +
            coverage_score * self.weights['coverage'] +
            distribution_score * self.weights['distribution'] +
            baseline_score * self.weights['baseline'] +
            overlap_score * self.weights['overlap']
        )
        
        return {
            'total_score': total_score,
            'match_score': match_score,
            'quality_score': quality_score,
            'coverage_score': coverage_score,
            'distribution_score': distribution_score,
            'baseline_score': baseline_score,
            'overlap_score': overlap_score,
            'num_matches': len(pts1),
            'num_inliers': len(inlier_pts1),
            'inlier_ratio': inlier_ratio
        }
    
    def _score_match_count(self, num_matches: int) -> float:
        """Score based on number of matches"""
        # Sigmoid function: more matches = better, with diminishing returns
        optimal = 200  # Optimal number of matches
        if num_matches < 30:
            return 0.0
        
        score = 1.0 / (1.0 + np.exp(-0.01 * (num_matches - optimal)))
        return float(np.clip(score, 0.0, 1.0))
    
    def _score_coverage(self,
                       pts1: np.ndarray,
                       pts2: np.ndarray,
                       size1: Tuple[int, int],
                       size2: Tuple[int, int]) -> float:
        """Score based on spatial coverage of matches"""
        if len(pts1) < 10:
            return 0.0
        
        # Compute coverage in both images
        coverage1 = self._compute_coverage(pts1, size1)
        coverage2 = self._compute_coverage(pts2, size2)
        
        # Return average
        return (coverage1 + coverage2) / 2.0
    
    def _compute_coverage(self,
                         pts: np.ndarray,
                         image_size: Tuple[int, int]) -> float:
        """Compute spatial coverage score for points in one image"""
        width, height = image_size
        
        # Divide image into grid cells
        grid_size = 4  # 4x4 grid
        cell_width = width / grid_size
        cell_height = height / grid_size
        
        # Count occupied cells
        occupied = set()
        for pt in pts:
            cell_x = int(pt[0] / cell_width)
            cell_y = int(pt[1] / cell_height)
            cell_x = np.clip(cell_x, 0, grid_size - 1)
            cell_y = np.clip(cell_y, 0, grid_size - 1)
            occupied.add((cell_x, cell_y))
        
        # Coverage ratio
        coverage = len(occupied) / (grid_size * grid_size)
        
        return float(coverage)
    
    def _score_distribution(self,
                           pts1: np.ndarray,
                           pts2: np.ndarray) -> float:
        """Score based on uniformity of point distribution"""
        if len(pts1) < 10:
            return 0.0
        
        # Compute distribution uniformity using standard deviation
        # Lower std deviation of distances = more uniform = better
        
        def distribution_metric(pts):
            if len(pts) < 2:
                return 0.0
            
            # Compute centroid
            centroid = np.mean(pts, axis=0)
            
            # Compute distances from centroid
            distances = np.linalg.norm(pts - centroid, axis=1)
            
            # Coefficient of variation (lower is more uniform)
            if np.mean(distances) > 0:
                cv = np.std(distances) / np.mean(distances)
                # Convert to score (lower cv = higher score)
                score = 1.0 / (1.0 + cv)
            else:
                score = 0.0
            
            return score
        
        dist1 = distribution_metric(pts1)
        dist2 = distribution_metric(pts2)
        
        return (dist1 + dist2) / 2.0
    
    def _score_baseline(self,
                       pts1: np.ndarray,
                       pts2: np.ndarray) -> float:
        """
        Score based on baseline (parallax/disparity).
        
        Higher disparity = better for triangulation.
        """
        if len(pts1) < 10:
            return 0.0
        
        # Compute average disparity
        disparities = np.linalg.norm(pts1 - pts2, axis=1)
        mean_disparity = np.mean(disparities)
        
        # Score: moderate disparity is best
        # Too low = bad triangulation
        # Too high = poor overlap
        optimal_disparity = 100.0  # pixels
        
        if mean_disparity < 10:
            return 0.0
        
        # Gaussian-like scoring around optimal
        score = np.exp(-0.5 * ((mean_disparity - optimal_disparity) / 50.0)**2)
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _score_overlap(self,
                      pts1: np.ndarray,
                      pts2: np.ndarray,
                      size1: Tuple[int, int],
                      size2: Tuple[int, int]) -> float:
        """
        Score based on image overlap.
        
        Good overlap is important for matching but not too much.
        """
        if len(pts1) == 0:
            return 0.0
        
        width1, height1 = size1
        width2, height2 = size2
        
        # Estimate overlap from match distribution
        # If matches are concentrated in certain regions, overlap is good
        
        # Normalize points to [0, 1]
        pts1_norm = pts1.copy()
        pts1_norm[:, 0] /= width1
        pts1_norm[:, 1] /= height1
        
        pts2_norm = pts2.copy()
        pts2_norm[:, 0] /= width2
        pts2_norm[:, 1] /= height2
        
        # Compute spread in normalized coordinates
        spread1 = np.std(pts1_norm, axis=0)
        spread2 = np.std(pts2_norm, axis=0)
        
        avg_spread = (np.mean(spread1) + np.mean(spread2)) / 2.0
        
        # Moderate spread = good overlap
        # Too little = poor overlap
        # Too much = nearly identical views
        optimal_spread = 0.3
        
        score = np.exp(-10 * (avg_spread - optimal_spread)**2)
        
        return float(np.clip(score, 0.0, 1.0))


def score_camera_connectivity(camera_id: str,
                              matches_data: Dict,
                              existing_cameras: Optional[Set[str]] = None) -> Dict[str, float]:
    """
    Score a camera based on its connectivity to other cameras.
    
    Args:
        camera_id: Camera to score
        matches_data: All match data
        existing_cameras: Set of cameras already in reconstruction
        
    Returns:
        Dictionary with connectivity scores
    """
    if existing_cameras is None:
        existing_cameras = set()
    
    # Count connections
    total_connections = 0
    existing_connections = 0
    total_matches = 0
    
    for pair_key, pair_data in matches_data.items():
        img1, img2 = pair_key
        
        if camera_id in pair_key:
            total_connections += 1
            num_matches = len(pair_data.get('correspondences', []))
            total_matches += num_matches
            
            # Check if connects to existing camera
            other_camera = img2 if img1 == camera_id else img1
            if other_camera in existing_cameras:
                existing_connections += 1
    
    # Compute scores
    connectivity_score = min(1.0, total_connections / 10.0)  # Normalized
    existing_connectivity = existing_connections / max(1, total_connections)
    match_score = min(1.0, total_matches / 1000.0)  # Normalized
    
    return {
        'total_connections': total_connections,
        'existing_connections': existing_connections,
        'connectivity_score': connectivity_score,
        'existing_connectivity': existing_connectivity,
        'total_matches': total_matches,
        'match_score': match_score
    }


def validate_pair_quality(pts1: np.ndarray,
                         pts2: np.ndarray,
                         min_matches: int = 30,
                         min_coverage: float = 0.2) -> Tuple[bool, List[str]]:
    """
    Validate if a pair meets minimum quality requirements.
    
    Args:
        pts1: Points in first image
        pts2: Points in second image
        min_matches: Minimum number of matches required
        min_coverage: Minimum coverage score required
        
    Returns:
        (is_valid, list of warnings)
    """
    warnings = []
    
    # Check match count
    if len(pts1) < min_matches:
        warnings.append(f"Insufficient matches: {len(pts1)} < {min_matches}")
    
    # Check if points exist
    if len(pts1) == 0 or len(pts2) == 0:
        warnings.append("No matches found")
        return False, warnings
    
    # Check for degenerate configuration
    if len(np.unique(pts1, axis=0)) < 4:
        warnings.append("Too few unique points in image 1")
    
    if len(np.unique(pts2, axis=0)) < 4:
        warnings.append("Too few unique points in image 2")
    
    # Check spread
    spread1 = np.std(pts1, axis=0)
    spread2 = np.std(pts2, axis=0)
    
    if np.any(spread1 < 10) or np.any(spread2 < 10):
        warnings.append("Points too concentrated (poor distribution)")
    
    is_valid = len(warnings) == 0 or (len(pts1) >= min_matches)
    
    return is_valid, warnings