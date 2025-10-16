"""
Camera Pose Estimation Module

This module provides comprehensive camera pose estimation from 2D-3D correspondences
using various Perspective-n-Point (PnP) algorithms with robust RANSAC estimation.

Key Features:
- Multiple PnP algorithms (P3P, EPnP, ITERATIVE, SQPNP)
- RANSAC-based robust estimation with outlier rejection
- Pose validation and quality assessment
- Integration with triangulation and bundle adjustment
- Incremental reconstruction support

Usage:
    estimator = PoseEstimator()
    result = estimator.estimate(points_3d, points_2d, camera_matrix)
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from enum import Enum

from CameraPoseEstimation2.core.interfaces.base_estimator import RANSACEstimator, EstimationResult, EstimationStatus


class PnPMethod(Enum):
    """Supported PnP methods"""
    P3P = "P3P"              # 3-point algorithm (minimal solver)
    EPNP = "EPNP"            # Efficient PnP (4+ points)
    ITERATIVE = "ITERATIVE"  # Iterative refinement
    SQPNP = "SQPNP"          # SQPnP algorithm
    AP3P = "AP3P"            # Alternative P3P
    IPPE = "IPPE"            # Infinitesimal Plane-based Pose
    IPPE_SQUARE = "IPPE_SQUARE"  # IPPE for square targets


class PoseEstimatorConfig:
    """Configuration for camera pose estimation"""
    
    # RANSAC parameters
    RANSAC_REPROJECTION_ERROR = 3.0    # Pixels - tighter for accurate poses
    RANSAC_CONFIDENCE = 0.999           # High confidence for monuments
    RANSAC_MAX_ITERATIONS = 2000        # Sufficient iterations
    
    # Quality thresholds
    MIN_INLIERS = 10                    # Minimum inliers for valid pose
    MIN_INLIER_RATIO = 0.3              # Minimum inlier ratio
    MIN_3D_POINTS = 4                   # Minimum 3D-2D correspondences
    
    # Validation thresholds
    MAX_REPROJECTION_ERROR = 10.0       # Maximum allowed reprojection error
    MIN_DEPTH = 0.1                     # Minimum depth (meters)
    MAX_DEPTH = 1000.0                  # Maximum depth (meters)
    MIN_TRIANGULATION_ANGLE = 1.0       # Minimum angle for triangulation (degrees)
    
    # Refinement parameters
    ENABLE_REFINEMENT = True            # Enable iterative refinement
    REFINEMENT_MAX_ITERATIONS = 20      # Refinement iterations
    REFINEMENT_TERMINATION_EPS = 1e-6   # Convergence threshold


class PoseEstimator(RANSACEstimator):
    """
    Camera pose estimation from 2D-3D correspondences using PnP algorithms.
    
    This class estimates camera pose (rotation R and translation t) from known
    3D points and their 2D projections in the image. It supports multiple PnP
    algorithms and includes robust RANSAC estimation for handling outliers.
    
    The estimator follows the project's architecture by extending RANSACEstimator
    and implementing the standard estimation interface.
    """
    
    def __init__(self, 
                 method: str = 'EPNP',
                 threshold: float = None,
                 confidence: float = None,
                 max_iterations: int = None,
                 **config):
        """
        Initialize pose estimator.
        
        Args:
            method: PnP method to use ('P3P', 'EPNP', 'ITERATIVE', 'SQPNP')
            threshold: RANSAC reprojection error threshold (default: from config)
            confidence: RANSAC confidence (default: from config)
            max_iterations: Maximum RANSAC iterations (default: from config)
            **config: Additional configuration parameters
        """
        self.config = PoseEstimatorConfig()
        
        # Initialize parent with RANSAC parameters
        super().__init__(
            threshold=threshold or self.config.RANSAC_REPROJECTION_ERROR,
            confidence=confidence or self.config.RANSAC_CONFIDENCE,
            max_iterations=max_iterations or self.config.RANSAC_MAX_ITERATIONS,
            **config
        )
        
        self.method = self._validate_method(method)
        self.cv2_flags = self._get_cv2_flags(self.method)
    
    def _validate_method(self, method: str) -> str:
        """Validate and normalize PnP method name."""
        method_upper = method.upper()
        
        # Check if method is valid
        valid_methods = [m.value for m in PnPMethod]
        if method_upper not in valid_methods:
            raise ValueError(f"Invalid PnP method: {method}. Valid methods: {valid_methods}")
        
        return method_upper
    
    def _get_cv2_flags(self, method: str) -> int:
        """Get OpenCV flags for the specified PnP method."""
        method_map = {
            'P3P': cv2.SOLVEPNP_P3P,
            'EPNP': cv2.SOLVEPNP_EPNP,
            'ITERATIVE': cv2.SOLVEPNP_ITERATIVE,
            'SQPNP': cv2.SOLVEPNP_SQPNP,
            'AP3P': cv2.SOLVEPNP_AP3P,
            'IPPE': cv2.SOLVEPNP_IPPE,
            'IPPE_SQUARE': cv2.SOLVEPNP_IPPE_SQUARE
        }
        
        return method_map.get(method, cv2.SOLVEPNP_EPNP)
    
    def get_algorithm_name(self) -> str:
        """Get algorithm name for logging."""
        return f"PoseEstimator_{self.method}"
    
    def estimate(self,
                points_3d: np.ndarray,
                points_2d: np.ndarray,
                camera_matrix: np.ndarray,
                dist_coeffs: Optional[np.ndarray] = None,
                use_ransac: bool = True,
                initial_rvec: Optional[np.ndarray] = None,
                initial_tvec: Optional[np.ndarray] = None) -> EstimationResult:
        """
        Estimate camera pose from 2D-3D correspondences.
        
        Args:
            points_3d: 3D points in world coordinates (Nx3)
            points_2d: Corresponding 2D points in image (Nx2)
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeffs: Distortion coefficients (optional)
            use_ransac: Use RANSAC for robust estimation
            initial_rvec: Initial rotation vector for iterative methods
            initial_tvec: Initial translation vector for iterative methods
            
        Returns:
            EstimationResult containing:
                - R: Rotation matrix (3x3)
                - t: Translation vector (3x1)
                - rvec: Rotation vector (3x1)
                - tvec: Translation vector (3x1)
                - inliers: Inlier mask
                - inlier_count: Number of inliers
                - reprojection_error: Mean reprojection error
        """
        # Input validation
        is_valid, error_msg = self.validate_input(
            points_3d, points_2d, camera_matrix, dist_coeffs
        )
        
        if not is_valid:
            return EstimationResult(
                success=False,
                status=EstimationStatus.FAILED,
                metadata={'error': error_msg}
            )
        
        # Prepare inputs
        points_3d = np.asarray(points_3d, dtype=np.float32)
        points_2d = np.asarray(points_2d, dtype=np.float32)
        
        # Ensure correct shape
        if points_3d.shape[1] != 3:
            points_3d = points_3d.T  # Convert 3xN to Nx3
        if points_2d.shape[1] != 2:
            points_2d = points_2d.T  # Convert 2xN to Nx2
        
        # Reshape for OpenCV
        points_3d_cv = points_3d.reshape(-1, 1, 3)
        points_2d_cv = points_2d.reshape(-1, 1, 2)
        
        try:
            if use_ransac:
                result = self._estimate_with_ransac(
                    points_3d_cv, points_2d_cv, camera_matrix, dist_coeffs
                )
            else:
                result = self._estimate_direct(
                    points_3d_cv, points_2d_cv, camera_matrix, dist_coeffs,
                    initial_rvec, initial_tvec
                )
            
            # Refine if requested and successful
            if result.success and self.config.ENABLE_REFINEMENT:
                result = self._refine_pose(
                    result, points_3d_cv, points_2d_cv, camera_matrix, dist_coeffs
                )
            
            return result
            
        except Exception as e:
            return EstimationResult(
                success=False,
                status=EstimationStatus.FAILED,
                metadata={'error': f'Pose estimation failed: {str(e)}'}
            )
    
    def _estimate_with_ransac(self,
                             points_3d: np.ndarray,
                             points_2d: np.ndarray,
                             camera_matrix: np.ndarray,
                             dist_coeffs: Optional[np.ndarray]) -> EstimationResult:
        """Estimate pose using RANSAC for robust outlier rejection."""
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            camera_matrix,
            dist_coeffs if dist_coeffs is not None else None,
            flags=self.cv2_flags,
            reprojectionError=self.threshold,
            confidence=self.confidence,
            iterationsCount=self.max_iterations
        )
        
        if not success or inliers is None:
            return EstimationResult(
                success=False,
                status=EstimationStatus.INSUFFICIENT_INLIERS,
                metadata={'error': 'PnP RANSAC failed to find solution'}
            )
        
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Flatten inliers array
        inlier_mask = np.zeros(len(points_3d), dtype=bool)
        inlier_mask[inliers.flatten()] = True
        
        inlier_count = np.sum(inlier_mask)
        inlier_ratio = inlier_count / len(points_3d)
        
        # Check quality thresholds
        if inlier_count < self.config.MIN_INLIERS:
            return EstimationResult(
                success=False,
                status=EstimationStatus.INSUFFICIENT_INLIERS,
                metadata={
                    'inlier_count': inlier_count,
                    'required': self.config.MIN_INLIERS
                }
            )
        
        if inlier_ratio < self.config.MIN_INLIER_RATIO:
            return EstimationResult(
                success=False,
                status=EstimationStatus.LOW_QUALITY,
                metadata={
                    'inlier_ratio': inlier_ratio,
                    'required': self.config.MIN_INLIER_RATIO
                }
            )
        
        # Compute reprojection error
        reproj_error = self._compute_reprojection_error(
            points_3d[inlier_mask], points_2d[inlier_mask],
            rvec, tvec, camera_matrix, dist_coeffs
        )
        
        return EstimationResult(
            success=True,
            status=EstimationStatus.SUCCESS,
            data={
                'R': R,
                't': tvec,
                'rvec': rvec,
                'tvec': tvec,
                'inliers': inlier_mask,
                'inlier_count': inlier_count,
                'inlier_ratio': inlier_ratio,
                'reprojection_error': reproj_error,
                'method': self.method
            },
            metadata={
                'algorithm': self.get_algorithm_name(),
                'use_ransac': True,
                'num_correspondences': len(points_3d)
            }
        )
    
    def _estimate_direct(self,
                        points_3d: np.ndarray,
                        points_2d: np.ndarray,
                        camera_matrix: np.ndarray,
                        dist_coeffs: Optional[np.ndarray],
                        initial_rvec: Optional[np.ndarray],
                        initial_tvec: Optional[np.ndarray]) -> EstimationResult:
        """Direct pose estimation without RANSAC."""
        
        # For iterative method, use initial guess if provided
        use_extrinsic_guess = (initial_rvec is not None and 
                              initial_tvec is not None and 
                              self.method == 'ITERATIVE')
        
        success, rvec, tvec = cv2.solvePnP(
            points_3d,
            points_2d,
            camera_matrix,
            dist_coeffs if dist_coeffs is not None else None,
            rvec=initial_rvec,
            tvec=initial_tvec,
            useExtrinsicGuess=use_extrinsic_guess,
            flags=self.cv2_flags
        )
        
        if not success:
            return EstimationResult(
                success=False,
                status=EstimationStatus.FAILED,
                metadata={'error': 'PnP estimation failed'}
            )
        
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # All points considered inliers in direct estimation
        inlier_mask = np.ones(len(points_3d), dtype=bool)
        
        # Compute reprojection error
        reproj_error = self._compute_reprojection_error(
            points_3d, points_2d, rvec, tvec, camera_matrix, dist_coeffs
        )
        
        return EstimationResult(
            success=True,
            status=EstimationStatus.SUCCESS,
            data={
                'R': R,
                't': tvec,
                'rvec': rvec,
                'tvec': tvec,
                'inliers': inlier_mask,
                'inlier_count': len(points_3d),
                'inlier_ratio': 1.0,
                'reprojection_error': reproj_error,
                'method': self.method
            },
            metadata={
                'algorithm': self.get_algorithm_name(),
                'use_ransac': False,
                'num_correspondences': len(points_3d)
            }
        )
    
    def _refine_pose(self,
                    result: EstimationResult,
                    points_3d: np.ndarray,
                    points_2d: np.ndarray,
                    camera_matrix: np.ndarray,
                    dist_coeffs: Optional[np.ndarray]) -> EstimationResult:
        """Refine pose estimate using iterative optimization."""
        
        if not result.success:
            return result
        
        # Extract inliers
        inliers = result.data['inliers']
        points_3d_inliers = points_3d[inliers]
        points_2d_inliers = points_2d[inliers]
        
        # Use current estimate as initial guess
        initial_rvec = result.data['rvec']
        initial_tvec = result.data['tvec']
        
        # Refine using iterative PnP
        try:
            success, rvec_refined, tvec_refined = cv2.solvePnP(
                points_3d_inliers,
                points_2d_inliers,
                camera_matrix,
                dist_coeffs if dist_coeffs is not None else None,
                rvec=initial_rvec,
                tvec=initial_tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                R_refined, _ = cv2.Rodrigues(rvec_refined)
                
                # Compute refined reprojection error
                reproj_error_refined = self._compute_reprojection_error(
                    points_3d_inliers, points_2d_inliers,
                    rvec_refined, tvec_refined, camera_matrix, dist_coeffs
                )
                
                # Only accept if refinement improves the result
                if reproj_error_refined < result.data['reprojection_error']:
                    result.data['R'] = R_refined
                    result.data['t'] = tvec_refined
                    result.data['rvec'] = rvec_refined
                    result.data['tvec'] = tvec_refined
                    result.data['reprojection_error'] = reproj_error_refined
                    result.metadata['refined'] = True
        
        except Exception as e:
            # Refinement failed, but keep original estimate
            result.metadata['refinement_error'] = str(e)
        
        return result
    
    def _compute_reprojection_error(self,
                                   points_3d: np.ndarray,
                                   points_2d: np.ndarray,
                                   rvec: np.ndarray,
                                   tvec: np.ndarray,
                                   camera_matrix: np.ndarray,
                                   dist_coeffs: Optional[np.ndarray]) -> float:
        """Compute mean reprojection error."""
        
        # Project 3D points to 2D
        projected_points, _ = cv2.projectPoints(
            points_3d,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs if dist_coeffs is not None else None
        )
        
        # Compute Euclidean distances
        projected_points = projected_points.reshape(-1, 2)
        points_2d_reshaped = points_2d.reshape(-1, 2)
        
        errors = np.linalg.norm(projected_points - points_2d_reshaped, axis=1)
        
        return float(np.mean(errors))
    
    def validate_input(self,
                      points_3d: np.ndarray,
                      points_2d: np.ndarray,
                      camera_matrix: np.ndarray,
                      dist_coeffs: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """
        Validate input data for pose estimation.
        
        Args:
            points_3d: 3D points
            points_2d: 2D points
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            
        Returns:
            (is_valid, error_message)
        """
        # Check if inputs are valid arrays
        if not isinstance(points_3d, np.ndarray) or not isinstance(points_2d, np.ndarray):
            return False, "Points must be numpy arrays"
        
        # Check shapes
        if points_3d.size == 0 or points_2d.size == 0:
            return False, "Empty point arrays"
        
        # Check minimum number of points
        n_points = min(len(points_3d.reshape(-1, 3)), len(points_2d.reshape(-1, 2)))
        
        if n_points < self.config.MIN_3D_POINTS:
            return False, f"Insufficient correspondences: {n_points} < {self.config.MIN_3D_POINTS}"
        
        # Check that we have matching number of 2D and 3D points
        if len(points_3d.reshape(-1, 3)) != len(points_2d.reshape(-1, 2)):
            return False, "Mismatched number of 2D and 3D points"
        
        # Validate camera matrix
        if not isinstance(camera_matrix, np.ndarray) or camera_matrix.shape != (3, 3):
            return False, "Invalid camera matrix shape"
        
        # Check camera matrix values
        if np.any(np.isnan(camera_matrix)) or np.any(np.isinf(camera_matrix)):
            return False, "Camera matrix contains NaN or Inf values"
        
        # Validate distortion coefficients if provided
        if dist_coeffs is not None:
            if not isinstance(dist_coeffs, np.ndarray):
                return False, "Distortion coefficients must be numpy array"
            
            if np.any(np.isnan(dist_coeffs)) or np.any(np.isinf(dist_coeffs)):
                return False, "Distortion coefficients contain NaN or Inf values"
        
        return True, ""
    
    def validate_result(self, result: EstimationResult) -> bool:
        """
        Validate estimation result.
        
        Args:
            result: Estimation result to validate
            
        Returns:
            True if result is valid
        """
        if not result.success:
            return False
        
        # Check if required data is present
        required_keys = ['R', 't', 'rvec', 'tvec', 'inliers', 'reprojection_error']
        if not all(key in result.data for key in required_keys):
            return False
        
        R = result.data['R']
        t = result.data['t']
        
        # Validate rotation matrix
        if not self._is_valid_rotation_matrix(R):
            return False
        
        # Check reprojection error
        if result.data['reprojection_error'] > self.config.MAX_REPROJECTION_ERROR:
            return False
        
        # Check translation vector
        t_norm = np.linalg.norm(t)
        if t_norm < self.config.MIN_DEPTH or t_norm > self.config.MAX_DEPTH:
            return False
        
        return True
    
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
    
    # RANSACEstimator abstract methods (for potential custom RANSAC implementation)
    
    def estimate_model(self, sample_points: Any) -> Any:
        """Estimate model from minimal sample (required by RANSACEstimator)."""
        # This would be implemented if we wanted custom RANSAC
        # Currently using OpenCV's built-in RANSAC
        raise NotImplementedError("Using OpenCV's RANSAC implementation")
    
    def compute_residuals(self, model: Any, all_points: Any) -> np.ndarray:
        """Compute residuals (required by RANSACEstimator)."""
        # This would be implemented if we wanted custom RANSAC
        raise NotImplementedError("Using OpenCV's RANSAC implementation")


# Convenience functions

def estimate_pose(points_3d: np.ndarray,
                 points_2d: np.ndarray,
                 camera_matrix: np.ndarray,
                 method: str = 'EPNP',
                 **kwargs) -> Dict:
    """
    Convenience function for camera pose estimation.
    
    Args:
        points_3d: 3D points in world coordinates (Nx3)
        points_2d: Corresponding 2D points in image (Nx2)
        camera_matrix: Camera intrinsic matrix (3x3)
        method: PnP method ('P3P', 'EPNP', 'ITERATIVE', 'SQPNP')
        **kwargs: Additional parameters for PoseEstimator
        
    Returns:
        Dictionary with pose estimation results
    """
    estimator = PoseEstimator(method=method, **kwargs)
    result = estimator.estimate(points_3d, points_2d, camera_matrix)
    
    # Convert EstimationResult to dict for backward compatibility
    if result.success:
        return {
            'success': True,
            'R': result.data['R'],
            't': result.data['t'],
            'rvec': result.data['rvec'],
            'tvec': result.data['tvec'],
            'inliers': result.data['inliers'],
            'inlier_count': result.data['inlier_count'],
            'inlier_ratio': result.data['inlier_ratio'],
            'reprojection_error': result.data['reprojection_error'],
            'method': result.data['method']
        }
    else:
        return {
            'success': False,
            'error': result.metadata.get('error', 'Pose estimation failed')
        }