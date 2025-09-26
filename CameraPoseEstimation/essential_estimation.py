import cv2
import numpy as np
from typing import Tuple, Dict, Optional
import warnings

class MatrixEstimationConfig:
    """Configuration for essential matrix estimation optimized for monuments"""
    
    # RANSAC parameters for monument images
    RANSAC_THRESHOLD = 1.5          # Slightly higher for outdoor monument conditions
    RANSAC_CONFIDENCE = 0.999       # High confidence for reliable reconstruction
    RANSAC_MAX_ITERATIONS = 5000    # More iterations for challenging scenes
    
    # Quality thresholds
    MIN_INLIERS = 30               # Higher minimum for monument stability
    MIN_INLIER_RATIO = 0.4         # Higher ratio for monument reliability
    
    # Camera estimation defaults
    DEFAULT_FOCAL_RATIO = 1.2      # Standard assumption for monuments

class EssentialMatrixEstimator:
    """
    Essential matrix estimation for monument reconstruction
    Handles both calibrated and uncalibrated cases
    """
    
    def __init__(self):
        """
        Initialize essential matrix estimator
        
        Args:
            camera_matrix: Camera intrinsic matrix (3x3). If None, will estimate from image size.
        """
        self.config = MatrixEstimationConfig()
    
    def estimate_camera_matrix(self, image_size: Tuple[int, int], 
                             focal_ratio: float = None) -> np.ndarray:
        """
        Estimate camera matrix from image size for uncalibrated monument images
        
        Args:
            image_size: (width, height) of images
            focal_ratio: Focal length as ratio of image width (default: 1.2)
            
        Returns:
            Estimated camera matrix (3x3)
        """
        if focal_ratio is None:
            focal_ratio = self.config.DEFAULT_FOCAL_RATIO
        
        width, height = image_size[:2]
        focal_length = width * focal_ratio
        
        camera_matrix = np.array([
            [focal_length, 0, width/2.0],
            [0, focal_length, height/2.0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        return camera_matrix
    
    def estimate(self, pts1: np.ndarray, pts2: np.ndarray,
                image_size1: Tuple[int, int],
                image_size2: Tuple[int, int],
                method: str = 'RANSAC',
                camera_matrix1: Optional[np.ndarray] = None,
                camera_matrix2: Optional[np.ndarray] = None) -> Dict:
        """
        Estimate essential matrix from monument feature correspondences
        
        Args:
            pts1: Feature points in first monument image (Nx2)
            pts2: Feature points in second monument image (Nx2)
            image_size: Image size (width, height) for camera estimation if needed
            method: Estimation method ('RANSAC', '5POINT', '8POINT')
            
        Returns:
            Dictionary with estimation results and quality metrics
        """
        if len(pts1) < 8:
            return {
                'success': False, 
                'error': f'Insufficient correspondences: {len(pts1)} < 8 points needed for essential matrix'
            }
        
        camera1_estimated = False
        camera2_estimated = False
        
        # Ensure camera matrix is available
        if camera_matrix1 is None:
            camera_matrix1 = self.estimate_camera_matrix(image_size1)
            camera1_estimated = True
        
        
        if camera_matrix2 is None:
            camera_matrix2 = self.estimate_camera_matrix(image_size2)
            camera2_estimated = True
        # Normalize points using camera matrix
        try:
            pts1_norm = cv2.undistortPoints(
                pts1.reshape(-1, 1, 2), 
                camera_matrix1, 
                None
            ).reshape(-1, 2)
            
            pts2_norm = cv2.undistortPoints(
                pts2.reshape(-1, 1, 2), 
                camera_matrix2, 
                None
            ).reshape(-1, 2)
        except Exception as e:
            return {
                'success': False,
                'error': f'Point normalization failed: {str(e)}'
            }
        
        # Estimate essential matrix
        try:
            if method == 'RANSAC':
                E, mask = cv2.findEssentialMat(
                    pts1_norm, pts2_norm,
                    focal=1.0, pp=(0.0, 0.0),
                    method=cv2.RANSAC,
                    prob=self.config.RANSAC_CONFIDENCE,
                    threshold=self.config.RANSAC_THRESHOLD,
                    maxIters=self.config.RANSAC_MAX_ITERATIONS
                )
            elif method == '5POINT':
                E, mask = cv2.findEssentialMat(
                    pts1_norm, pts2_norm,
                    focal=1.0, pp=(0.0, 0.0),
                    method=cv2.FM_5POINT
                )
            elif method == '8POINT':
                E, mask = cv2.findEssentialMat(
                    pts1_norm, pts2_norm,
                    focal=1.0, pp=(0.0, 0.0),
                    method=cv2.FM_8POINT
                )
            else:
                return {'success': False, 'error': f'Unknown method: {method}'}
        
        except Exception as e:
            return {
                'success': False,
                'error': f'Essential matrix estimation failed: {str(e)}'
            }
        
        if E is None or mask is None:
            return {
                'success': False,
                'error': 'Essential matrix estimation returned None - possibly degenerate configuration'
            }
        
        # Calculate statistics
        inlier_count = np.sum(mask)
        inlier_ratio = inlier_count / len(pts1)
        
        # Quality validation for monument reconstruction
        if inlier_count < self.config.MIN_INLIERS:
            return {
                'success': False,
                'error': f'Too few inliers for reliable monument reconstruction: {inlier_count} < {self.config.MIN_INLIERS}'
            }
        
        if inlier_ratio < self.config.MIN_INLIER_RATIO:
            return {
                'success': False,
                'error': f'Low inlier ratio for monument reconstruction: {inlier_ratio:.1%} < {self.config.MIN_INLIER_RATIO:.1%}'
            }
        
        # Assess essential matrix quality
        quality_assessment = self._assess_essential_matrix_quality(E)
        
        return {
            'success': True,
            'essential_matrix': E,
            'inlier_mask': mask,
            'camera_matrices': [camera_matrix1, camera_matrix2],
            'camera_estimated': [camera1_estimated, camera2_estimated],
            'method': method,
            'num_points': len(pts1),
            'num_inliers': inlier_count,
            'inlier_ratio': inlier_ratio,
            'quality_assessment': quality_assessment,
            'matrix_type': 'essential'
        }
    
    def _assess_essential_matrix_quality(self, E: np.ndarray) -> Dict:
        """
        Assess quality of essential matrix for monument reconstruction
        
        Args:
            E: Essential matrix
            
        Returns:
            Quality assessment dictionary
        """
        if E is None or E.shape != (3, 3):
            return {
                'quality_score': 0.0,
                'is_valid': False,
                'warnings': ['Matrix is None or wrong shape']
            }
        
        try:
            # Compute SVD for analysis
            U, S, Vt = np.linalg.svd(E)
        except Exception:
            return {
                'quality_score': 0.0,
                'is_valid': False,
                'warnings': ['SVD decomposition failed']
            }
        
        s1, s2, s3 = S
        warnings_list = []
        quality_score = 0.0
        
        # Check singular values (ideal: two equal, one zero)
        if s1 > 1e-6:  # Avoid division by zero
            sigma_ratio = s2 / s1
            sigma3_ratio = s3 / s1
            
            # Two singular values should be approximately equal
            if abs(sigma_ratio - 1.0) < 0.1:
                quality_score += 0.5
            else:
                warnings_list.append(f'Singular values not equal: σ2/σ1 = {sigma_ratio:.3f}')
            
            # Third singular value should be close to zero
            if sigma3_ratio < 0.1:
                quality_score += 0.3
            else:
                warnings_list.append(f'Third singular value too large: σ3/σ1 = {sigma3_ratio:.3f}')
        else:
            warnings_list.append('Degenerate matrix: largest singular value near zero')
        
        # Check matrix rank (should be 2)
        rank = np.linalg.matrix_rank(E, tol=1e-6)
        if rank == 2:
            quality_score += 0.2
        else:
            warnings_list.append(f'Matrix rank is {rank}, should be 2')
        
        is_valid = len(warnings_list) == 0 or quality_score > 0.5
        
        return {
            'quality_score': quality_score,
            'is_valid': is_valid,
            'singular_values': S.tolist(),
            'matrix_rank': rank,
            'warnings': warnings_list
        }

def estimate_essential_matrix(pts1: np.ndarray, pts2: np.ndarray,
                            image_size: Tuple[int, int],
                            camera_matrix: Optional[np.ndarray] = None,
                            method: str = 'RANSAC') -> Dict:
    """
    Convenience function for essential matrix estimation
    
    Args:
        pts1: Points in first monument image (Nx2)
        pts2: Points in second monument image (Nx2)
        image_size: Image size (width, height)
        camera_matrix: Camera intrinsic matrix (estimated if None)
        method: Estimation method ('RANSAC', '5POINT', '8POINT')
        
    Returns:
        Essential matrix estimation results
    """
    estimator = EssentialMatrixEstimator(camera_matrix)
    return estimator.estimate(pts1, pts2, image_size, method)

def validate_correspondences_for_monument(pts1: np.ndarray, pts2: np.ndarray,
                                        image_size: Tuple[int, int]) -> Dict:
    """
    Validate correspondences for reliable monument essential matrix estimation
    
    Args:
        pts1: Points in first monument image (Nx2)
        pts2: Points in second monument image (Nx2)
        image_size: Image dimensions (width, height)
        
    Returns:
        Validation results with monument-specific recommendations
    """
    validation = {
        'valid': True,
        'quality_level': 'unknown',
        'warnings': [],
        'errors': [],
        'recommendations': [],
        'statistics': {}
    }
    
    if len(pts1) != len(pts2):
        validation['errors'].append('Mismatched point array lengths')
        validation['valid'] = False
        return validation
    
    width, height = image_size
    num_points = len(pts1)
    
    # Check minimum points
    if num_points < MatrixEstimationConfig.MIN_INLIERS:
        validation['errors'].append(
            f'Insufficient points: {num_points} < {MatrixEstimationConfig.MIN_INLIERS} '
            f'(monuments require higher point density)'
        )
        validation['valid'] = False
    
    # Monument-specific distribution analysis
    pts1_spread = np.std(pts1, axis=0)
    pts2_spread = np.std(pts2, axis=0)
    
    # Coverage analysis
    coverage_ratio_1 = (pts1_spread[0] * pts1_spread[1]) / (width * height)
    coverage_ratio_2 = (pts2_spread[0] * pts2_spread[1]) / (width * height)
    min_coverage = 0.02  # At least 2% of image area
    
    if coverage_ratio_1 < min_coverage:
        validation['warnings'].append(
            f'Limited coverage in first image ({coverage_ratio_1:.1%})'
        )
    
    if coverage_ratio_2 < min_coverage:
        validation['warnings'].append(
            f'Limited coverage in second image ({coverage_ratio_2:.1%})'
        )
    
    # Baseline analysis
    point_displacements = np.linalg.norm(pts2 - pts1, axis=1)
    mean_displacement = np.mean(point_displacements)
    
    if mean_displacement < 8.0:
        validation['warnings'].append(
            f'Small baseline ({mean_displacement:.1f}px) - may affect 3D reconstruction'
        )
        validation['recommendations'].append('Use images with greater viewpoint separation')
    
    if mean_displacement > min(width, height) * 0.4:
        validation['warnings'].append(
            f'Large baseline ({mean_displacement:.1f}px) - may reduce reliability'
        )
    
    # Quality assessment
    quality_factors = []
    
    if num_points >= MatrixEstimationConfig.MIN_INLIERS * 1.5:
        quality_factors.append('sufficient_points')
    
    if coverage_ratio_1 > 0.05 and coverage_ratio_2 > 0.05:
        quality_factors.append('good_coverage')
    
    if 10.0 <= mean_displacement <= min(width, height) * 0.25:
        quality_factors.append('adequate_baseline')
    
    quality_score = len(quality_factors) / 3.0
    
    if quality_score >= 0.8:
        validation['quality_level'] = 'excellent'
    elif quality_score >= 0.6:
        validation['quality_level'] = 'good'
    elif quality_score >= 0.4:
        validation['quality_level'] = 'fair'
    else:
        validation['quality_level'] = 'poor'
    
    # Statistics
    validation['statistics'] = {
        'num_correspondences': num_points,
        'coverage_ratio_1': float(coverage_ratio_1),
        'coverage_ratio_2': float(coverage_ratio_2),
        'mean_displacement': float(mean_displacement),
        'quality_score': float(quality_score),
        'quality_factors': quality_factors
    }
    
    return validation















