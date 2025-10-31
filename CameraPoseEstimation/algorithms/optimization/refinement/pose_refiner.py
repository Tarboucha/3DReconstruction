"""
Iterative Pose Refinement

Refines camera poses through iterative optimization using various methods:
- Gauss-Newton refinement
- Levenberg-Marquardt
- Non-linear least squares
- Alternating optimization

Used for:
- Refining poses after PnP estimation
- Improving poses between BA iterations
- Local pose adjustments
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import least_squares
import time

from CameraPoseEstimation2.core.interfaces.base_optimizer import BaseOptimizer, OptimizationResult, OptimizationStatus
from CameraPoseEstimation2.logger import get_logger

logger = get_logger("optimization.pose_refiner")


class PoseRefinerConfig:
    """Configuration for pose refinement"""
    
    # Optimization parameters
    MAX_ITERATIONS = 20
    FUNCTION_TOLERANCE = 1e-6
    GRADIENT_TOLERANCE = 1e-6
    
    # Convergence criteria
    MIN_COST_REDUCTION = 1e-4
    MIN_PARAM_CHANGE = 1e-6
    
    # What to optimize
    OPTIMIZE_ROTATION = True
    OPTIMIZE_TRANSLATION = True
    OPTIMIZE_INTRINSICS = False  # Usually False for refinement
    
    # Robust loss
    USE_ROBUST_LOSS = True
    ROBUST_LOSS_TYPE = 'huber'
    ROBUST_LOSS_THRESHOLD = 1.5  # Tighter than BA


class PoseRefiner(BaseOptimizer):
    """
    Iterative pose refinement for individual cameras.
    
    Refines camera pose using non-linear optimization with observations.
    More focused than full bundle adjustment - only optimizes one camera
    at a time while keeping structure fixed.
    
    Use cases:
    - After PnP estimation
    - Between incremental BA iterations
    - Quick local pose improvement
    - When you want to refine pose without full BA
    """
    
    def __init__(self, **config):
        """
        Initialize pose refiner.
        
        Args:
            **config: Configuration overrides
        """
        super().__init__(**config)
        self.config = PoseRefinerConfig()
        
        # Override config
        for key, value in config.items():
            if hasattr(self.config, key.upper()):
                setattr(self.config, key.upper(), value)
    

    def validate_input(self, camera_id: str, R_init: np.ndarray, t_init: np.ndarray, 
                    K: np.ndarray, points_3d: np.ndarray, points_2d: np.ndarray, 
                    **kwargs) -> Tuple[bool, str]:
        """Validate input for pose refinement"""
        if points_3d.shape[0] < 4:
            return False, f"Need at least 4 points, got {points_3d.shape[0]}"
        if points_3d.shape[0] != points_2d.shape[0]:
            return False, "Mismatch between 3D and 2D points"
        if R_init.shape != (3, 3):
            return False, f"Invalid rotation matrix shape: {R_init.shape}"
        if K.shape != (3, 3):
            return False, f"Invalid camera matrix shape: {K.shape}"
        return True, ""

    def compute_residuals(self, params: np.ndarray, points_3d: np.ndarray, 
                        points_2d: np.ndarray, K: np.ndarray, 
                        optimize_intrinsics: bool) -> np.ndarray:
        """Compute residuals (delegates to internal method)"""
        return self._compute_residuals(params, points_3d, points_2d, K, optimize_intrinsics)

    def get_algorithm_name(self) -> str:
        """Get algorithm name."""
        return "PoseRefiner"
    
    def refine_pose(self,
                   camera_id: str,
                   R_init: np.ndarray,
                   t_init: np.ndarray,
                   K: np.ndarray,
                   points_3d: np.ndarray,
                   points_2d: np.ndarray,
                   optimize_intrinsics: bool = False) -> OptimizationResult:
        """
        Refine a single camera pose.
        
        Args:
            camera_id: Camera identifier
            R_init: Initial rotation matrix (3x3)
            t_init: Initial translation vector (3,) or (3x1)
            K: Camera intrinsic matrix (3x3)
            points_3d: 3D points visible in this camera (Nx3)
            points_2d: Corresponding 2D observations (Nx2)
            optimize_intrinsics: Whether to optimize K
            
        Returns:
            OptimizationResult with refined pose
        """
        start_time = time.time()
        
        # Validate inputs
        if points_3d.shape[0] < 4:
            return OptimizationResult(
                success=False,
                status=OptimizationStatus.INVALID_INPUT,
                metadata={'error': f'Need at least 4 points, got {points_3d.shape[0]}'}
            )
        
        if points_3d.shape[0] != points_2d.shape[0]:
            return OptimizationResult(
                success=False,
                status=OptimizationStatus.INVALID_INPUT,
                metadata={'error': 'Mismatch between 3D and 2D points'}
            )
        
        logger.info(f"Refining pose for {camera_id}:")
        logger.info(f"  Points: {len(points_3d)}")
        logger.info(f"  Optimize intrinsics: {optimize_intrinsics}")
        
        # Build parameter vector
        params = self._build_parameters(R_init, t_init, K, optimize_intrinsics)
        
        # Compute initial cost
        initial_residuals = self._compute_residuals(
            params, points_3d, points_2d, K, optimize_intrinsics
        )
        initial_cost = np.sqrt(np.mean(initial_residuals**2))
        
        logger.info(f"  Initial cost: {initial_cost:.3f}px")
        
        try:
            # Run optimization
            if self.config.USE_ROBUST_LOSS:
                result = least_squares(
                    fun=self._compute_residuals,
                    x0=params,
                    args=(points_3d, points_2d, K, optimize_intrinsics),
                    method='trf',
                    max_nfev=self.config.MAX_ITERATIONS * len(params),
                    ftol=self.config.FUNCTION_TOLERANCE,
                    xtol=self.config.GRADIENT_TOLERANCE,
                    loss='huber',
                    f_scale=self.config.ROBUST_LOSS_THRESHOLD,
                    verbose=0
                )
            else:
                result = least_squares(
                    fun=self._compute_residuals,
                    x0=params,
                    args=(points_3d, points_2d, K, optimize_intrinsics),
                    method='trf',
                    max_nfev=self.config.MAX_ITERATIONS * len(params),
                    ftol=self.config.FUNCTION_TOLERANCE,
                    xtol=self.config.GRADIENT_TOLERANCE,
                    verbose=0
                )
            
            # Unpack optimized parameters
            R_refined, t_refined, K_refined = self._unpack_parameters(
                result.x, optimize_intrinsics
            )
            
            # Compute final cost
            final_residuals = self._compute_residuals(
                result.x, points_3d, points_2d, K, optimize_intrinsics
            )
            final_cost = np.sqrt(np.mean(final_residuals**2))
            
            runtime = time.time() - start_time
            
            logger.info(f"  Final cost: {final_cost:.3f}px ({result.nfev} iterations, {runtime:.2f}s)")
            logger.info(f"  Cost reduction: {initial_cost - final_cost:.3f}px")
            
            # Check if improvement is significant
            cost_reduction = initial_cost - final_cost
            if cost_reduction < self.config.MIN_COST_REDUCTION:
                status = OptimizationStatus.CONVERGED
            else:
                status = OptimizationStatus.SUCCESS
            
            return OptimizationResult(
                success=True,
                status=status,
                optimized_params={
                    'R': R_refined,
                    't': t_refined,
                    'K': K_refined,
                    'rvec': cv2.Rodrigues(R_refined)[0],
                    'tvec': t_refined.reshape(3, 1)
                },
                initial_cost=initial_cost,
                final_cost=final_cost,
                num_iterations=result.nfev,
                residuals=final_residuals,
                runtime=runtime,
                metadata={
                    'algorithm': self.get_algorithm_name(),
                    'camera_id': camera_id,
                    'num_points': len(points_3d),
                    'optimize_intrinsics': optimize_intrinsics,
                    'cost_reduction': cost_reduction
                }
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                status=OptimizationStatus.FAILED,
                initial_cost=initial_cost if 'initial_cost' in locals() else 0.0,
                runtime=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def refine_multiple_poses(self,
                             cameras: Dict[str, Dict],
                             points_3d: np.ndarray,
                             observations: Dict[str, List[Dict]],
                             optimize_intrinsics: bool = False) -> Dict[str, OptimizationResult]:
        """
        Refine multiple camera poses independently.
        
        Args:
            cameras: Dictionary of cameras {cam_id: {'R', 't', 'K'}}
            points_3d: All 3D points (3xN)
            observations: Observations per camera {cam_id: [obs1, obs2, ...]}
            optimize_intrinsics: Whether to optimize intrinsics
            
        Returns:
            Dictionary of results per camera
        """
        results = {}
        
        logger.info(f"\nRefining {len(cameras)} cameras independently:")
        print("="*70)
        
        for cam_id, cam_data in cameras.items():
            if cam_id not in observations or len(observations[cam_id]) == 0:
                logger.info(f"  {cam_id}: No observations, skipping")
                continue
            
            # Extract 2D-3D correspondences for this camera
            cam_points_3d = []
            cam_points_2d = []
            
            for obs in observations[cam_id]:
                point_id = obs['point_id']
                if point_id < points_3d.shape[1]:
                    cam_points_3d.append(points_3d[:, point_id])
                    cam_points_2d.append(obs.get('coords_2d', obs.get('image_point')))
            
            if len(cam_points_3d) < 4:
                logger.info(f"  {cam_id}: Insufficient points ({len(cam_points_3d)}), skipping")
                continue
            
            # Refine this camera
            result = self.refine_pose(
                camera_id=cam_id,
                R_init=cam_data['R'],
                t_init=cam_data['t'],
                K=cam_data['K'],
                points_3d=np.array(cam_points_3d),
                points_2d=np.array(cam_points_2d),
                optimize_intrinsics=optimize_intrinsics
            )
            
            results[cam_id] = result
        
        print("="*70)
        
        return results
    
    def _build_parameters(self,
                         R: np.ndarray,
                         t: np.ndarray,
                         K: np.ndarray,
                         optimize_intrinsics: bool) -> np.ndarray:
        """Build parameter vector for optimization."""
        # Rotation vector
        rvec, _ = cv2.Rodrigues(R)
        params = list(rvec.flatten())
        
        # Translation
        t = t.flatten()
        params.extend(t)
        
        # Intrinsics (if optimizing)
        if optimize_intrinsics:
            params.extend([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])  # fx, fy, cx, cy
        
        return np.array(params)
    
    def _unpack_parameters(self,
                          params: np.ndarray,
                          optimize_intrinsics: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unpack parameter vector."""
        # Rotation
        rvec = params[:3]
        R, _ = cv2.Rodrigues(rvec)
        
        # Translation
        t = params[3:6]
        
        # Intrinsics
        if optimize_intrinsics:
            fx, fy, cx, cy = params[6:10]
            K = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])
        else:
            K = None  # Will use original K
        
        return R, t, K
    
    def _compute_residuals(self,
                          params: np.ndarray,
                          points_3d: np.ndarray,
                          points_2d: np.ndarray,
                          K_original: np.ndarray,
                          optimize_intrinsics: bool) -> np.ndarray:
        """Compute reprojection residuals."""
        # Unpack parameters
        rvec = params[:3]
        tvec = params[3:6]
        
        if optimize_intrinsics:
            fx, fy, cx, cy = params[6:10]
            K = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])
        else:
            K = K_original
        
        # Project points
        projected, _ = cv2.projectPoints(
            points_3d.reshape(-1, 1, 3),
            rvec,
            tvec,
            K,
            None
        )
        projected = projected.reshape(-1, 2)
        
        # Compute residuals
        residuals = (projected - points_2d).flatten()
        
        return residuals
    
    def optimize(self, *args, **kwargs) -> OptimizationResult:
        """
        Optimize - wrapper for refine_pose.
        
        Required by BaseOptimizer interface.
        """
        # Extract standard arguments
        if 'camera_id' in kwargs:
            return self.refine_pose(*args, **kwargs)
        else:
            raise NotImplementedError("Use refine_pose() or refine_multiple_poses()")
    
    def compute_cost(self, *args, **kwargs) -> float:
        """Compute cost (implemented via residuals)."""
        raise NotImplementedError("Use refine_pose() which computes cost internally")


# Convenience functions

def refine_camera_pose(camera_id: str,
                      R: np.ndarray,
                      t: np.ndarray,
                      K: np.ndarray,
                      points_3d: np.ndarray,
                      points_2d: np.ndarray,
                      **config) -> Dict[str, Any]:
    """
    Convenience function for pose refinement.
    
    Args:
        camera_id: Camera identifier
        R: Rotation matrix
        t: Translation vector
        K: Intrinsic matrix
        points_3d: 3D points
        points_2d: 2D observations
        **config: Additional configuration
        
    Returns:
        Dictionary with refined pose
    """
    refiner = PoseRefiner(**config)
    
    result = refiner.refine_pose(
        camera_id=camera_id,
        R_init=R,
        t_init=t,
        K=K,
        points_3d=points_3d,
        points_2d=points_2d
    )
    
    if result.success:
        return {
            'success': True,
            'R': result.optimized_params['R'],
            't': result.optimized_params['t'],
            'K': result.optimized_params['K'],
            'initial_error': result.initial_cost,
            'final_error': result.final_cost,
            'improvement': result.get_cost_reduction()
        }
    else:
        return {
            'success': False,
            'error': result.metadata.get('error', 'Refinement failed')
        }