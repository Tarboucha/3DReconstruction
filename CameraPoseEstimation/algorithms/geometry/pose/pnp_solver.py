"""
High-Level PnP Solver

This module provides a high-level interface for solving the Perspective-n-Point (PnP)
problem. It wraps the PoseEstimator with additional functionality for incremental
reconstruction, including integration with observations and reconstruction state.

This solver is designed to work seamlessly with the reconstruction pipeline.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
import warnings

from .pose_estimator import PoseEstimator, PoseEstimatorConfig
from .validators import PoseValidator, validate_correspondences


class PnPSolver:
    """
    High-level PnP solver for incremental reconstruction.
    
    This class provides an easy-to-use interface for camera pose estimation
    that integrates with the reconstruction pipeline. It handles:
    - Extracting 2D-3D correspondences from observations
    - Robust pose estimation with RANSAC
    - Pose validation and quality assessment
    - Integration with reconstruction state
    
    Usage:
        solver = PnPSolver()
        result = solver.solve_pnp(points_3d, points_2d, camera_matrix)
        
        # Or use with reconstruction state
        result = solver.add_camera_to_reconstruction(
            new_image, reconstruction_state, matches
        )
    """
    
    def __init__(self,
                 method: str = 'EPNP',
                 reprojection_error: float = 3.0,
                 confidence: float = 0.999,
                 max_iterations: int = 2000,
                 min_inliers: int = 10,
                 min_inlier_ratio: float = 0.3):
        """
        Initialize PnP solver.
        
        Args:
            method: PnP method ('P3P', 'EPNP', 'ITERATIVE', 'SQPNP')
            reprojection_error: RANSAC reprojection error threshold (pixels)
            confidence: RANSAC confidence level
            max_iterations: Maximum RANSAC iterations
            min_inliers: Minimum number of inliers required
            min_inlier_ratio: Minimum ratio of inliers
        """
        self.estimator = PoseEstimator(
            method=method,
            threshold=reprojection_error,
            confidence=confidence,
            max_iterations=max_iterations
        )
        
        self.validator = PoseValidator(
            max_reprojection_error=reprojection_error * 2,  # More lenient for validation
            min_inliers=min_inliers,
            min_inlier_ratio=min_inlier_ratio
        )
        
        self.config = PoseEstimatorConfig()
    
    def solve_pnp(self,
                 points_3d: np.ndarray,
                 points_2d: np.ndarray,
                 camera_matrix: np.ndarray,
                 dist_coeffs: Optional[np.ndarray] = None,
                 method: Optional[str] = None,
                 use_ransac: bool = True,
                 validate: bool = True) -> Dict[str, Any]:
        """
        Solve PnP problem to estimate camera pose.
        
        Args:
            points_3d: 3D points in world coordinates (Nx3)
            points_2d: Corresponding 2D points in image (Nx2)
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeffs: Distortion coefficients (optional)
            method: PnP method (if None, uses default from initialization)
            use_ransac: Use RANSAC for robust estimation
            validate: Perform pose validation
            
        Returns:
            Dictionary containing:
                - success: Whether estimation succeeded
                - R: Rotation matrix (3x3)
                - t: Translation vector (3x1)
                - rvec: Rotation vector (3x1)
                - tvec: Translation vector (3x1)
                - inliers: Boolean mask of inliers
                - inlier_count: Number of inliers
                - inlier_ratio: Ratio of inliers
                - reprojection_error: Mean reprojection error
                - validation: Validation results (if validate=True)
        """
        # Validate input correspondences
        is_valid, error_msg = validate_correspondences(
            points_3d, points_2d, min_points=self.config.MIN_3D_POINTS
        )
        
        if not is_valid:
            return {
                'success': False,
                'error': error_msg
            }
        
        # Use specified method or default
        if method is not None:
            # Temporarily change method
            original_method = self.estimator.method
            self.estimator.method = method.upper()
            self.estimator.cv2_flags = self.estimator._get_cv2_flags(method.upper())
        
        # Estimate pose
        result = self.estimator.estimate(
            points_3d=points_3d,
            points_2d=points_2d,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            use_ransac=use_ransac
        )
        
        # Restore original method if changed
        if method is not None:
            self.estimator.method = original_method
            self.estimator.cv2_flags = self.estimator._get_cv2_flags(original_method)
        
        # Convert EstimationResult to dictionary
        if not result.success:
            return {
                'success': False,
                'error': result.metadata.get('error', 'PnP estimation failed'),
                'status': result.status.value
            }
        
        output = {
            'success': True,
            'R': result.data['R'],
            't': result.data['t'],
            'rvec': result.data['rvec'],
            'tvec': result.data['tvec'],
            'inliers': result.data['inliers'],
            'inlier_count': result.data['inlier_count'],
            'inlier_ratio': result.data['inlier_ratio'],
            'reprojection_error': result.data['reprojection_error'],
            'method': result.data['method'],
            'metadata': result.metadata
        }
        
        # Perform validation if requested
        if validate:
            validation_result = self.validator.validate_pose(
                R=result.data['R'],
                t=result.data['t'],
                points_3d=points_3d,
                points_2d=points_2d,
                camera_matrix=camera_matrix,
                inlier_mask=result.data['inliers']
            )
            
            output['validation'] = {
                'is_valid': validation_result.is_valid,
                'quality_score': validation_result.quality_score,
                'warnings': validation_result.warnings,
                'errors': validation_result.errors,
                'metrics': validation_result.metrics
            }
            
            # Update success based on validation
            if not validation_result.is_valid:
                output['success'] = False
                output['error'] = 'Pose validation failed: ' + '; '.join(validation_result.errors)
        
        return output
    
    def add_camera_to_reconstruction(self,
                                    new_image_id: str,
                                    reconstruction_state: Dict[str, Any],
                                    feature_matches: Dict[str, Any],
                                    camera_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Add a new camera to an existing reconstruction using PnP.
        
        This method extracts 2D-3D correspondences from the reconstruction state
        and feature matches, then estimates the pose of the new camera.
        
        Args:
            new_image_id: ID of the new image to add
            reconstruction_state: Current reconstruction state with cameras and 3D points
            feature_matches: Feature matches between new image and existing images
            camera_matrix: Camera intrinsic matrix (if None, uses from reconstruction)
            
        Returns:
            Dictionary with pose estimation results and updated observations
        """
        # Extract camera matrix
        if camera_matrix is None:
            # Try to get from reconstruction state
            if 'camera_matrix' in reconstruction_state:
                camera_matrix = reconstruction_state['camera_matrix']
            else:
                return {
                    'success': False,
                    'error': 'No camera matrix provided or found in reconstruction'
                }
        
        # Extract 2D-3D correspondences
        correspondences = self._extract_correspondences(
            new_image_id,
            reconstruction_state,
            feature_matches
        )
        
        if len(correspondences['points_3d']) < self.config.MIN_3D_POINTS:
            return {
                'success': False,
                'error': f'Insufficient 2D-3D correspondences: {len(correspondences["points_3d"])} < {self.config.MIN_3D_POINTS}'
            }
        
        # Solve PnP
        result = self.solve_pnp(
            points_3d=correspondences['points_3d'],
            points_2d=correspondences['points_2d'],
            camera_matrix=camera_matrix,
            use_ransac=True,
            validate=True
        )
        
        if not result['success']:
            return result
        
        # Add observations for the new camera
        inlier_indices = np.where(result['inliers'])[0]
        new_observations = []
        
        for idx in inlier_indices:
            obs = {
                'image_id': new_image_id,
                'point_id': correspondences['point_ids'][idx],
                'coords_2d': correspondences['points_2d'][idx],
                'feature_id': correspondences.get('feature_ids', [None])[idx]
            }
            new_observations.append(obs)
        
        result['observations'] = new_observations
        result['num_observations'] = len(new_observations)
        
        return result
    
    def _extract_correspondences(self,
                                new_image_id: str,
                                reconstruction_state: Dict[str, Any],
                                feature_matches: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract 2D-3D correspondences from reconstruction state and matches.
        
        This finds feature matches between the new image and existing images,
        then looks up the corresponding 3D points that have already been triangulated.
        
        Args:
            new_image_id: ID of new image
            reconstruction_state: Current reconstruction
            feature_matches: Feature matches
            
        Returns:
            Dictionary with arrays of 2D points, 3D points, and point IDs
        """
        points_3d = []
        points_2d = []
        point_ids = []
        feature_ids = []
        
        # Get existing cameras and points
        cameras = reconstruction_state.get('cameras', {})
        points_3d_all = reconstruction_state.get('points_3d', {}).get('points_3d', np.array([]))
        observations = reconstruction_state.get('observations', {})
        
        # For each existing camera
        for existing_image_id in cameras.keys():
            if existing_image_id == new_image_id:
                continue
            
            # Get matches between new image and this existing image
            pair_key = tuple(sorted([new_image_id, existing_image_id]))
            
            if pair_key not in feature_matches:
                continue
            
            matches = feature_matches[pair_key]
            
            # Determine which points belong to new vs existing image
            if matches.get('image1') == new_image_id:
                new_pts = matches.get('pts1', np.array([]))
                existing_pts = matches.get('pts2', np.array([]))
            else:
                new_pts = matches.get('pts2', np.array([]))
                existing_pts = matches.get('pts1', np.array([]))
            
            if len(new_pts) == 0:
                continue
            
            # Find which of these matches have corresponding 3D points
            if existing_image_id in observations:
                for obs in observations[existing_image_id]:
                    point_id = obs['point_id']
                    observed_2d = np.array(obs.get('image_point', obs.get('coords_2d', [])))
                    
                    # Find matching 2D point in existing_pts
                    distances = np.linalg.norm(existing_pts - observed_2d, axis=1)
                    closest_idx = np.argmin(distances)
                    
                    if distances[closest_idx] < 2.0:  # Threshold for matching
                        # Found a 2D-3D correspondence
                        if point_id < points_3d_all.shape[1]:
                            points_3d.append(points_3d_all[:, point_id])
                            points_2d.append(new_pts[closest_idx])
                            point_ids.append(point_id)
                            feature_ids.append(obs.get('feature_id'))
        
        return {
            'points_3d': np.array(points_3d) if points_3d else np.empty((0, 3)),
            'points_2d': np.array(points_2d) if points_2d else np.empty((0, 2)),
            'point_ids': point_ids,
            'feature_ids': feature_ids
        }
    
    def estimate_pose_from_observations(self,
                                       observations: List[Dict[str, Any]],
                                       points_3d_dict: Dict[int, np.ndarray],
                                       camera_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Estimate camera pose from a list of observations.
        
        Args:
            observations: List of observations, each with 'point_id' and 'coords_2d'
            points_3d_dict: Dictionary mapping point IDs to 3D coordinates
            camera_matrix: Camera intrinsic matrix
            
        Returns:
            Pose estimation result
        """
        # Extract correspondences from observations
        points_3d = []
        points_2d = []
        valid_point_ids = []
        
        for obs in observations:
            point_id = obs['point_id']
            
            if point_id in points_3d_dict:
                points_3d.append(points_3d_dict[point_id])
                points_2d.append(obs['coords_2d'])
                valid_point_ids.append(point_id)
        
        if len(points_3d) < self.config.MIN_3D_POINTS:
            return {
                'success': False,
                'error': f'Insufficient observations with 3D points: {len(points_3d)}'
            }
        
        # Solve PnP
        result = self.solve_pnp(
            points_3d=np.array(points_3d),
            points_2d=np.array(points_2d),
            camera_matrix=camera_matrix
        )
        
        if result['success']:
            result['point_ids'] = valid_point_ids
        
        return result
    
    def refine_pose(self,
                   R_init: np.ndarray,
                   t_init: np.ndarray,
                   points_3d: np.ndarray,
                   points_2d: np.ndarray,
                   camera_matrix: np.ndarray,
                   inlier_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Refine an existing pose estimate.
        
        Args:
            R_init: Initial rotation matrix
            t_init: Initial translation vector
            points_3d: 3D points
            points_2d: 2D points
            camera_matrix: Camera intrinsic matrix
            inlier_mask: Optional mask to use only inliers
            
        Returns:
            Refined pose estimation result
        """
        # Filter to inliers if provided
        if inlier_mask is not None:
            points_3d = points_3d[inlier_mask]
            points_2d = points_2d[inlier_mask]
        
        # Convert R to rvec
        rvec_init, _ = cv2.Rodrigues(R_init)
        tvec_init = t_init.reshape(3, 1)
        
        # Estimate with iterative refinement
        result = self.estimator.estimate(
            points_3d=points_3d,
            points_2d=points_2d,
            camera_matrix=camera_matrix,
            use_ransac=False,
            initial_rvec=rvec_init,
            initial_tvec=tvec_init
        )
        
        if not result.success:
            return {
                'success': False,
                'error': 'Refinement failed'
            }
        
        return {
            'success': True,
            'R': result.data['R'],
            't': result.data['t'],
            'rvec': result.data['rvec'],
            'tvec': result.data['tvec'],
            'reprojection_error': result.data['reprojection_error']
        }


# Convenience function for backward compatibility

def solve_pnp_ransac(points_3d: np.ndarray,
                    points_2d: np.ndarray,
                    camera_matrix: np.ndarray,
                    reprojection_error: float = 3.0,
                    confidence: float = 0.999,
                    max_iterations: int = 2000) -> Dict[str, Any]:
    """
    Convenience function for PnP with RANSAC.
    
    Args:
        points_3d: 3D points (Nx3)
        points_2d: 2D points (Nx2)
        camera_matrix: Camera intrinsic matrix (3x3)
        reprojection_error: RANSAC threshold
        confidence: RANSAC confidence
        max_iterations: Maximum iterations
        
    Returns:
        PnP solution dictionary
    """
    solver = PnPSolver(
        reprojection_error=reprojection_error,
        confidence=confidence,
        max_iterations=max_iterations
    )
    
    return solver.solve_pnp(
        points_3d=points_3d,
        points_2d=points_2d,
        camera_matrix=camera_matrix,
        use_ransac=True
    )