"""
Pose Validation Utilities

This module provides validation functions for camera poses and 3D-2D correspondences.
It includes checks for geometric validity, reprojection errors, depth constraints,
and quality assessment.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class PoseValidationResult:
    """Result of pose validation"""
    is_valid: bool
    quality_score: float  # 0-1 scale
    warnings: List[str]
    errors: List[str]
    metrics: Dict[str, float]


class PoseValidator:
    """
    Validates camera poses and provides quality assessment.
    
    This class checks:
    - Rotation matrix validity (orthogonality, det=1)
    - Translation vector constraints (depth bounds)
    - Reprojection errors
    - Geometric consistency
    - Coverage and distribution of inliers
    """
    
    def __init__(self,
                 max_reprojection_error: float = 10.0,
                 min_depth: float = 0.1,
                 max_depth: float = 1000.0,
                 min_inliers: int = 10,
                 min_inlier_ratio: float = 0.3):
        """
        Initialize pose validator.
        
        Args:
            max_reprojection_error: Maximum allowed reprojection error (pixels)
            min_depth: Minimum depth constraint (meters)
            max_depth: Maximum depth constraint (meters)
            min_inliers: Minimum number of inliers required
            min_inlier_ratio: Minimum ratio of inliers to total points
        """
        self.max_reprojection_error = max_reprojection_error
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
    
    def validate_pose(self,
                     R: np.ndarray,
                     t: np.ndarray,
                     points_3d: np.ndarray,
                     points_2d: np.ndarray,
                     camera_matrix: np.ndarray,
                     inlier_mask: Optional[np.ndarray] = None) -> PoseValidationResult:
        """
        Comprehensive pose validation.
        
        Args:
            R: Rotation matrix (3x3)
            t: Translation vector (3x1) or (3,)
            points_3d: 3D points used for estimation (Nx3)
            points_2d: 2D points used for estimation (Nx2)
            camera_matrix: Camera intrinsic matrix (3x3)
            inlier_mask: Optional mask indicating inliers
            
        Returns:
            PoseValidationResult with validation outcome and metrics
        """
        warnings = []
        errors = []
        metrics = {}
        
        # 1. Validate rotation matrix
        if not self._is_valid_rotation_matrix(R):
            errors.append("Invalid rotation matrix (not orthogonal or det != 1)")
        
        # 2. Validate translation
        t = t.reshape(-1)
        t_norm = np.linalg.norm(t)
        metrics['translation_norm'] = float(t_norm)
        
        if t_norm < self.min_depth:
            warnings.append(f"Translation very small: {t_norm:.3f} < {self.min_depth}")
        
        if t_norm > self.max_depth:
            warnings.append(f"Translation very large: {t_norm:.3f} > {self.max_depth}")
        
        # 3. Validate inliers
        if inlier_mask is not None:
            inlier_count = np.sum(inlier_mask)
            inlier_ratio = inlier_count / len(inlier_mask)
            
            metrics['inlier_count'] = int(inlier_count)
            metrics['inlier_ratio'] = float(inlier_ratio)
            
            if inlier_count < self.min_inliers:
                errors.append(f"Insufficient inliers: {inlier_count} < {self.min_inliers}")
            
            if inlier_ratio < self.min_inlier_ratio:
                warnings.append(f"Low inlier ratio: {inlier_ratio:.2%} < {self.min_inlier_ratio:.2%}")
        
        # 4. Compute reprojection errors
        rvec, _ = cv2.Rodrigues(R)
        reproj_errors = self._compute_reprojection_errors(
            points_3d, points_2d, rvec, t.reshape(3, 1), camera_matrix
        )
        
        if inlier_mask is not None:
            inlier_errors = reproj_errors[inlier_mask]
        else:
            inlier_errors = reproj_errors
        
        if len(inlier_errors) > 0:
            metrics['mean_reprojection_error'] = float(np.mean(inlier_errors))
            metrics['median_reprojection_error'] = float(np.median(inlier_errors))
            metrics['max_reprojection_error'] = float(np.max(inlier_errors))
            
            if metrics['mean_reprojection_error'] > self.max_reprojection_error:
                warnings.append(
                    f"High reprojection error: {metrics['mean_reprojection_error']:.2f} > "
                    f"{self.max_reprojection_error}"
                )
        
        # 5. Check point depths
        depth_metrics = self._validate_point_depths(points_3d, R, t)
        metrics.update(depth_metrics)
        
        if depth_metrics['num_behind_camera'] > 0:
            warnings.append(f"{depth_metrics['num_behind_camera']} points behind camera")
        
        # 6. Check distribution of inliers
        if inlier_mask is not None and np.sum(inlier_mask) > 0:
            distribution_metrics = self._assess_inlier_distribution(
                points_2d[inlier_mask]
            )
            metrics.update(distribution_metrics)
            
            if distribution_metrics['coverage_score'] < 0.3:
                warnings.append("Poor inlier distribution across image")
        
        # 7. Compute quality score
        quality_score = self._compute_quality_score(metrics, len(errors), len(warnings))
        
        is_valid = len(errors) == 0
        
        return PoseValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            warnings=warnings,
            errors=errors,
            metrics=metrics
        )
    
    def _is_valid_rotation_matrix(self, R: np.ndarray) -> bool:
        """Check if matrix is a valid rotation matrix."""
        if R.shape != (3, 3):
            return False
        
        # Check orthogonality: R^T * R = I
        should_be_identity = np.dot(R.T, R)
        identity = np.eye(3)
        
        if not np.allclose(should_be_identity, identity, atol=1e-3):
            return False
        
        # Check determinant = 1
        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-3):
            return False
        
        return True
    
    def _compute_reprojection_errors(self,
                                    points_3d: np.ndarray,
                                    points_2d: np.ndarray,
                                    rvec: np.ndarray,
                                    tvec: np.ndarray,
                                    camera_matrix: np.ndarray) -> np.ndarray:
        """Compute reprojection error for each point."""
        # Reshape for OpenCV
        points_3d_cv = points_3d.reshape(-1, 1, 3)
        
        # Project 3D points to 2D
        projected_points, _ = cv2.projectPoints(
            points_3d_cv,
            rvec,
            tvec,
            camera_matrix,
            None
        )
        
        # Compute Euclidean distances
        projected_points = projected_points.reshape(-1, 2)
        points_2d_reshaped = points_2d.reshape(-1, 2)
        
        errors = np.linalg.norm(projected_points - points_2d_reshaped, axis=1)
        
        return errors
    
    def _validate_point_depths(self,
                              points_3d: np.ndarray,
                              R: np.ndarray,
                              t: np.ndarray) -> Dict[str, float]:
        """Validate depths of 3D points in camera frame."""
        # Transform points to camera frame
        points_cam = (R @ points_3d.T).T + t.reshape(1, 3)
        
        # Get depths (Z coordinates)
        depths = points_cam[:, 2]
        
        # Check constraints
        num_behind_camera = np.sum(depths < 0)
        num_too_close = np.sum((depths > 0) & (depths < self.min_depth))
        num_too_far = np.sum(depths > self.max_depth)
        
        valid_depths = depths[(depths > self.min_depth) & (depths < self.max_depth)]
        
        metrics = {
            'num_behind_camera': int(num_behind_camera),
            'num_too_close': int(num_too_close),
            'num_too_far': int(num_too_far),
            'num_valid_depths': int(len(valid_depths))
        }
        
        if len(valid_depths) > 0:
            metrics['mean_depth'] = float(np.mean(valid_depths))
            metrics['median_depth'] = float(np.median(valid_depths))
            metrics['min_depth'] = float(np.min(valid_depths))
            metrics['max_depth'] = float(np.max(valid_depths))
        
        return metrics
    
    def _assess_inlier_distribution(self, points_2d: np.ndarray) -> Dict[str, float]:
        """Assess spatial distribution of inlier points."""
        if len(points_2d) < 3:
            return {'coverage_score': 0.0}
        
        # Compute bounding box
        min_coords = np.min(points_2d, axis=0)
        max_coords = np.max(points_2d, axis=0)
        
        spread = max_coords - min_coords
        
        # Compute coverage score based on spread
        # Higher spread = better coverage
        coverage_score = min(1.0, np.prod(spread) / (1000 * 1000))  # Normalized to typical image size
        
        # Compute spatial variance
        centroid = np.mean(points_2d, axis=0)
        distances = np.linalg.norm(points_2d - centroid, axis=1)
        
        return {
            'coverage_score': float(coverage_score),
            'spread_x': float(spread[0]),
            'spread_y': float(spread[1]),
            'mean_distance_from_center': float(np.mean(distances)),
            'std_distance_from_center': float(np.std(distances))
        }
    
    def _compute_quality_score(self,
                              metrics: Dict[str, float],
                              num_errors: int,
                              num_warnings: int) -> float:
        """
        Compute overall quality score (0-1).
        
        Factors:
        - Inlier ratio
        - Reprojection error
        - Point distribution
        - Errors and warnings
        """
        score = 1.0
        
        # Penalize errors and warnings
        score -= num_errors * 0.5
        score -= num_warnings * 0.1
        
        # Reward high inlier ratio
        if 'inlier_ratio' in metrics:
            score *= metrics['inlier_ratio']
        
        # Penalize high reprojection error
        if 'mean_reprojection_error' in metrics:
            error_penalty = min(1.0, metrics['mean_reprojection_error'] / self.max_reprojection_error)
            score *= (1.0 - 0.5 * error_penalty)
        
        # Reward good coverage
        if 'coverage_score' in metrics:
            score *= (0.5 + 0.5 * metrics['coverage_score'])
        
        return max(0.0, min(1.0, score))


def validate_correspondences(points_3d: np.ndarray,
                            points_2d: np.ndarray,
                            min_points: int = 4) -> Tuple[bool, str]:
    """
    Validate 2D-3D correspondences for pose estimation.
    
    Args:
        points_3d: 3D points (Nx3)
        points_2d: 2D points (Nx2)
        min_points: Minimum required points
        
    Returns:
        (is_valid, error_message)
    """
    # Check if arrays are valid
    if not isinstance(points_3d, np.ndarray) or not isinstance(points_2d, np.ndarray):
        return False, "Points must be numpy arrays"
    
    # Check for empty arrays
    if points_3d.size == 0 or points_2d.size == 0:
        return False, "Empty point arrays"
    
    # Reshape if needed
    points_3d = points_3d.reshape(-1, 3)
    points_2d = points_2d.reshape(-1, 2)
    
    # Check minimum number of points
    if len(points_3d) < min_points or len(points_2d) < min_points:
        return False, f"Insufficient points: need at least {min_points}"
    
    # Check matching lengths
    if len(points_3d) != len(points_2d):
        return False, f"Mismatched lengths: {len(points_3d)} 3D points, {len(points_2d)} 2D points"
    
    # Check for NaN or Inf
    if np.any(np.isnan(points_3d)) or np.any(np.isinf(points_3d)):
        return False, "3D points contain NaN or Inf"
    
    if np.any(np.isnan(points_2d)) or np.any(np.isinf(points_2d)):
        return False, "2D points contain NaN or Inf"
    
    return True, ""


def check_pose_consistency(R1: np.ndarray,
                          t1: np.ndarray,
                          R2: np.ndarray,
                          t2: np.ndarray,
                          max_angle_diff: float = 10.0,
                          max_translation_ratio: float = 5.0) -> Dict[str, any]:
    """
    Check consistency between two camera poses.
    
    Useful for verifying incremental pose estimation or comparing
    different estimation methods.
    
    Args:
        R1, t1: First pose
        R2, t2: Second pose
        max_angle_diff: Maximum allowed rotation angle difference (degrees)
        max_translation_ratio: Maximum allowed translation magnitude ratio
        
    Returns:
        Dictionary with consistency check results
    """
    # Compute relative rotation
    R_rel = R2 @ R1.T
    
    # Compute rotation angle
    trace = np.trace(R_rel)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    angle_deg = np.degrees(angle)
    
    # Compute translation ratio
    t1_norm = np.linalg.norm(t1)
    t2_norm = np.linalg.norm(t2)
    
    if t1_norm > 1e-6:
        translation_ratio = t2_norm / t1_norm
    else:
        translation_ratio = float('inf')
    
    # Check consistency
    rotation_consistent = angle_deg < max_angle_diff
    translation_consistent = translation_ratio < max_translation_ratio
    
    is_consistent = rotation_consistent and translation_consistent
    
    return {
        'is_consistent': is_consistent,
        'rotation_angle_diff': angle_deg,
        'translation_ratio': translation_ratio,
        'rotation_consistent': rotation_consistent,
        'translation_consistent': translation_consistent
    }