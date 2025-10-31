"""
Bundle Adjustment Cost Functions

This module provides cost functions and residual computation for bundle adjustment.
It supports:
- Reprojection error calculation
- Robust loss functions (Huber, Cauchy, etc.)
- Jacobian computation (for optimization)
- Per-camera intrinsics handling
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from scipy.sparse import lil_matrix
from enum import Enum


class LossFunction(Enum):
    """Robust loss functions for bundle adjustment"""
    NONE = "none"
    HUBER = "huber"
    CAUCHY = "cauchy"
    SOFT_L1 = "soft_l1"
    ARCTAN = "arctan"


class BACostFunction:
    """
    Cost function for bundle adjustment.
    
    Computes reprojection errors and their derivatives for optimization.
    Supports per-camera intrinsics and robust loss functions.
    """
    
    def __init__(self, 
                 loss_function: str = 'huber',
                 loss_threshold: float = 2.0):
        """
        Initialize cost function.
        
        Args:
            loss_function: Robust loss function ('huber', 'cauchy', 'soft_l1', 'arctan', 'none')
            loss_threshold: Threshold for robust loss function
        """
        self.loss_function = loss_function.lower()
        self.loss_threshold = loss_threshold
    
    def compute_residuals(self,
                         params: np.ndarray,
                         param_structure: Dict,
                         observations: List[Dict],
                         cameras_dict: Dict,
                         optimize_intrinsics: bool = True) -> np.ndarray:
        """
        Compute reprojection residuals for all observations.
        
        Args:
            params: Flattened parameter vector containing:
                    - Camera parameters (rotation, translation, intrinsics)
                    - 3D point coordinates
            param_structure: Dictionary describing parameter structure
            observations: List of observations (camera_id, point_id, image_point)
            cameras_dict: Dictionary of camera data
            optimize_intrinsics: Whether intrinsics are being optimized
            
        Returns:
            Residual vector (2N,) where N is number of observations
        """
        # Unpack parameters
        camera_params = self._unpack_camera_params(
            params, param_structure, optimize_intrinsics
        )
        points_3d = self._unpack_points(params, param_structure)
        
        # Compute residuals for each observation
        residuals = []
        
        for obs in observations:
            camera_id = obs['camera_id']
            point_id = obs['point_id']
            observed_2d = np.array(obs['image_point'])
            
            # Skip if point index is out of bounds
            if point_id >= points_3d.shape[1]:
                continue
            
            # Get camera parameters
            if camera_id in camera_params:
                cam = camera_params[camera_id]
            else:
                # Use original camera if not being optimized
                cam = cameras_dict[camera_id]
            
            # Get 3D point
            point_3d = points_3d[:, point_id]
            
            # Project point
            projected_2d = self._project_point(
                point_3d, cam['R'], cam['t'], cam['K']
            )
            
            # Compute residual
            residual = observed_2d - projected_2d
            residuals.extend(residual)
        
        return np.array(residuals)
    
    def compute_weighted_residuals(self,
                                   residuals: np.ndarray) -> np.ndarray:
        """
        Apply robust loss function to residuals.
        
        Args:
            residuals: Raw residuals
            
        Returns:
            Weighted residuals
        """
        if self.loss_function == 'none':
            return residuals
        
        # Compute residual norms
        residuals_2d = residuals.reshape(-1, 2)
        norms = np.linalg.norm(residuals_2d, axis=1)
        
        # Apply robust loss function
        if self.loss_function == 'huber':
            weights = self._huber_weights(norms)
        elif self.loss_function == 'cauchy':
            weights = self._cauchy_weights(norms)
        elif self.loss_function == 'soft_l1':
            weights = self._soft_l1_weights(norms)
        elif self.loss_function == 'arctan':
            weights = self._arctan_weights(norms)
        else:
            weights = np.ones_like(norms)
        
        # Apply weights to residuals
        weighted = residuals_2d * weights[:, np.newaxis]
        
        return weighted.flatten()
    
    def compute_cost(self, residuals: np.ndarray) -> float:
        """
        Compute total cost from residuals.
        
        Args:
            residuals: Residual vector
            
        Returns:
            Total cost (RMSE or robust cost)
        """
        if self.loss_function == 'none':
            # Standard RMSE
            return np.sqrt(np.mean(residuals**2))
        else:
            # Robust cost
            weighted = self.compute_weighted_residuals(residuals)
            return np.sqrt(np.mean(weighted**2))
    
    def _project_point(self,
                      point_3d: np.ndarray,
                      R: np.ndarray,
                      t: np.ndarray,
                      K: np.ndarray) -> np.ndarray:
        """
        Project 3D point to 2D image coordinates.

        Args:
            point_3d: 3D point (3,)
            R: Rotation matrix (3x3)
            t: Translation vector (3,) or (3,1)
            K: Camera intrinsic matrix (3x3)

        Returns:
            2D image point (2,)
        """
        # Check for NaN/Inf in input
        if not np.all(np.isfinite(point_3d)):
            return np.array([0.0, 0.0])

        # Transform to camera coordinates
        t = t.reshape(3)
        point_cam = R @ point_3d + t

        # Check for NaN/Inf after transformation
        if not np.all(np.isfinite(point_cam)):
            return np.array([0.0, 0.0])

        # Project to image plane
        if point_cam[2] <= 1e-10:  # Guard against near-zero depth
            # Point behind camera or too close
            return np.array([0.0, 0.0])

        # Normalize with guard against division by near-zero
        depth = max(abs(point_cam[2]), 1e-10)
        point_normalized = point_cam / depth

        # Apply intrinsics
        point_2d = K @ point_normalized

        # Final NaN/Inf check
        if not np.all(np.isfinite(point_2d[:2])):
            return np.array([0.0, 0.0])

        return point_2d[:2]
    
    def _unpack_camera_params(self,
                             params: np.ndarray,
                             param_structure: Dict,
                             optimize_intrinsics: bool) -> Dict:
        """Unpack camera parameters from parameter vector."""
        cameras = {}
        
        camera_ids = param_structure['camera_ids']
        camera_param_size = param_structure['camera_param_size']
        
        for i, cam_id in enumerate(camera_ids):
            start_idx = i * camera_param_size
            
            # Extract rotation vector (3)
            rvec = params[start_idx:start_idx + 3]
            R, _ = cv2.Rodrigues(rvec)
            
            # Extract translation (3)
            t = params[start_idx + 3:start_idx + 6]
            
            if optimize_intrinsics:
                # Extract intrinsics (fx, fy, cx, cy)
                fx, fy, cx, cy = params[start_idx + 6:start_idx + 10]
                K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
            else:
                # Use original intrinsics
                K = param_structure['original_intrinsics'][cam_id]
            
            cameras[cam_id] = {
                'R': R,
                't': t,
                'K': K
            }
        
        return cameras
    
    def _unpack_points(self,
                      params: np.ndarray,
                      param_structure: Dict) -> np.ndarray:
        """Unpack 3D points from parameter vector."""
        points_start = param_structure['points_start']
        points_3d_flat = params[points_start:]
        
        num_points = len(points_3d_flat) // 3
        points_3d = points_3d_flat.reshape(3, num_points)
        
        return points_3d
    
    # Robust loss weight functions
    
    def _huber_weights(self, norms: np.ndarray) -> np.ndarray:
        """Huber loss weights."""
        weights = np.ones_like(norms)
        outliers = norms > self.loss_threshold
        weights[outliers] = self.loss_threshold / norms[outliers]
        return weights
    
    def _cauchy_weights(self, norms: np.ndarray) -> np.ndarray:
        """Cauchy loss weights."""
        c_squared = self.loss_threshold ** 2
        weights = c_squared / (c_squared + norms**2)
        return weights
    
    def _soft_l1_weights(self, norms: np.ndarray) -> np.ndarray:
        """Soft L1 loss weights."""
        weights = 1.0 / np.sqrt(1 + (norms / self.loss_threshold)**2)
        return weights
    
    def _arctan_weights(self, norms: np.ndarray) -> np.ndarray:
        """Arctan loss weights."""
        c = self.loss_threshold
        weights = c / (norms + 1e-8) * np.arctan(norms / c)
        return weights


class ParameterBuilder:
    """
    Builds parameter vectors and structures for bundle adjustment.
    
    Handles the packing/unpacking of camera poses, intrinsics, and 3D points
    into a single optimization parameter vector.
    """
    
    @staticmethod
    def build_parameter_vector(camera_ids: List[str],
                               cameras_dict: Dict,
                               points_3d: np.ndarray,
                               optimize_intrinsics: bool = True,
                               fixed_cameras: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Build parameter vector for optimization.
        
        Args:
            camera_ids: List of camera IDs to optimize
            cameras_dict: Dictionary of camera data
            points_3d: 3D points (3xN)
            optimize_intrinsics: Whether to include intrinsics in parameters
            fixed_cameras: List of camera IDs to keep fixed
            
        Returns:
            (params, param_structure)
            - params: Flattened parameter vector
            - param_structure: Dictionary describing structure
        """
        if fixed_cameras is None:
            fixed_cameras = []
        
        # Determine camera parameter size
        if optimize_intrinsics:
            camera_param_size = 10  # rvec(3) + t(3) + K(4)
        else:
            camera_param_size = 6   # rvec(3) + t(3)
        
        # Build camera parameters
        camera_params = []
        optimized_camera_ids = []
        original_intrinsics = {}
        
        for cam_id in camera_ids:
            if cam_id in fixed_cameras:
                continue
            
            optimized_camera_ids.append(cam_id)
            cam = cameras_dict[cam_id]
            
            # Rotation vector
            rvec, _ = cv2.Rodrigues(cam['R'])
            camera_params.extend(rvec.flatten())
            
            # Translation
            camera_params.extend(cam['t'].flatten()[:3])
            
            # Intrinsics
            if optimize_intrinsics:
                K = cam['K']
                camera_params.extend([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
            
            # Store original intrinsics for reference
            original_intrinsics[cam_id] = cam['K'].copy()
        
        # Build point parameters
        points_flat = points_3d.flatten('F')  # Fortran order (column-major)
        
        # Combine parameters
        params = np.concatenate([camera_params, points_flat])
        
        # Build structure description
        param_structure = {
            'camera_ids': optimized_camera_ids,
            'camera_param_size': camera_param_size,
            'points_start': len(camera_params),
            'num_points': points_3d.shape[1],
            'optimize_intrinsics': optimize_intrinsics,
            'original_intrinsics': original_intrinsics,
            'fixed_cameras': fixed_cameras
        }
        
        return params, param_structure
    
    @staticmethod
    def unpack_parameters(params: np.ndarray,
                         param_structure: Dict) -> Tuple[Dict, np.ndarray]:
        """
        Unpack optimized parameters.
        
        Args:
            params: Optimized parameter vector
            param_structure: Parameter structure description
            
        Returns:
            (cameras_dict, points_3d)
        """
        cost_fn = BACostFunction()
        
        cameras = cost_fn._unpack_camera_params(
            params, param_structure, param_structure['optimize_intrinsics']
        )
        
        points_3d = cost_fn._unpack_points(params, param_structure)
        
        return cameras, points_3d


def compute_reprojection_error(points_3d: np.ndarray,
                               points_2d: np.ndarray,
                               R: np.ndarray,
                               t: np.ndarray,
                               K: np.ndarray) -> np.ndarray:
    """
    Compute reprojection error for a set of points.
    
    Args:
        points_3d: 3D points (Nx3) or (3xN)
        points_2d: 2D observations (Nx2) or (2xN)
        R: Rotation matrix (3x3)
        t: Translation vector (3,) or (3x1)
        K: Camera intrinsic matrix (3x3)
        
    Returns:
        Reprojection errors (N,)
    """
    # Ensure correct shapes
    if points_3d.shape[0] == 3:
        points_3d = points_3d.T
    if points_2d.shape[0] == 2:
        points_2d = points_2d.T
    
    # Project 3D points
    rvec, _ = cv2.Rodrigues(R)
    t = t.reshape(3, 1)
    
    projected, _ = cv2.projectPoints(
        points_3d.reshape(-1, 1, 3),
        rvec,
        t,
        K,
        None
    )
    projected = projected.reshape(-1, 2)
    
    # Compute errors
    errors = np.linalg.norm(projected - points_2d, axis=1)
    
    return errors


def compute_mean_reprojection_error(points_3d: np.ndarray,
                                   observations: List[Dict],
                                   cameras_dict: Dict) -> float:
    """
    Compute mean reprojection error across all observations.
    
    Args:
        points_3d: 3D points (3xN)
        observations: List of observations
        cameras_dict: Dictionary of cameras
        
    Returns:
        Mean reprojection error in pixels
    """
    errors = []
    
    for obs in observations:
        camera_id = obs['camera_id']
        point_id = obs['point_id']
        observed_2d = np.array(obs['image_point'])
        
        if point_id >= points_3d.shape[1]:
            continue
        
        cam = cameras_dict[camera_id]
        point_3d = points_3d[:, point_id]
        
        # Project
        rvec, _ = cv2.Rodrigues(cam['R'])
        projected, _ = cv2.projectPoints(
            point_3d.reshape(1, 1, 3),
            rvec,
            cam['t'].reshape(3, 1),
            cam['K'],
            None
        )
        projected = projected.reshape(2)
        
        # Error
        error = np.linalg.norm(observed_2d - projected)
        errors.append(error)
    
    return float(np.mean(errors)) if errors else 0.0