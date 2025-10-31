"""
Incremental Bundle Adjustment

Optimizes reconstruction after adding each new camera view.
Balances accuracy with computational efficiency by optimizing
only recent cameras while refining all 3D points.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from scipy.optimize import least_squares
import time

from CameraPoseEstimation2.core.interfaces.base_optimizer import BaseOptimizer, OptimizationResult, OptimizationStatus
from .cost_functions import BACostFunction, ParameterBuilder, compute_mean_reprojection_error
from CameraPoseEstimation2.logger import get_logger

logger = get_logger("optimization.incremental_ba")


class IncrementalBundleAdjustmentConfig:
    """Configuration for incremental bundle adjustment"""
    
    # Optimization parameters
    MAX_ITERATIONS = 50  # Fewer than global BA for speed
    FUNCTION_TOLERANCE = 1e-5
    GRADIENT_TOLERANCE = 1e-5
    
    # Robust loss
    USE_ROBUST_LOSS = True
    ROBUST_LOSS_TYPE = 'huber'
    ROBUST_LOSS_THRESHOLD = 2.0
    
    # Incremental BA strategy
    NUM_RECENT_CAMERAS = 3  # Number of recent cameras to optimize
    OPTIMIZE_ALL_POINTS = True  # Always optimize all points
    OPTIMIZE_INTRINSICS = True
    
    # Minimum requirements
    MIN_POINTS_FOR_BA = 10
    MIN_OBSERVATIONS = 20


class IncrementalBundleAdjustment(BaseOptimizer):
    """
    Incremental bundle adjustment for sequential reconstruction.
    
    Strategy:
    - Optimize only recent N cameras (efficiency)
    - Optimize all 3D points (accuracy)
    - Optional per-camera intrinsic optimization
    - Use robust loss for outlier handling
    
    Use after:
    - Adding each new camera via PnP
    - Triangulating new points
    
    Benefits:
    - Prevents error accumulation
    - Much faster than global BA
    - Good accuracy for incremental reconstruction
    """
    
    def __init__(self, **config):
        """
        Initialize incremental bundle adjuster.
        
        Args:
            **config: Configuration overrides
        """
        super().__init__(**config)
        self.config = IncrementalBundleAdjustmentConfig()
        
        # Override config
        for key, value in config.items():
            if hasattr(self.config, key.upper()):
                setattr(self.config, key.upper(), value)
        
        self.cost_function = BACostFunction(
            loss_function=self.config.ROBUST_LOSS_TYPE,
            loss_threshold=self.config.ROBUST_LOSS_THRESHOLD
        )
    
    # Add to CameraPoseEstimation2/algorithms/optimization/bundle_adjustment/two_view.py

    def validate_input(self, cameras, points_3d, observations, **kwargs) -> Tuple[bool, str]:
        """Validate input for bundle adjustment"""
        if len(cameras) < 2:
            return False, f"Need at least 2 cameras, got {len(cameras)}"
        if points_3d.shape[1] < 4:
            return False, f"Need at least 4 points, got {points_3d.shape[1]}"
        if len(observations) < 4:
            return False, f"Need at least 4 observations, got {len(observations)}"
        return True, ""

    def compute_residuals(self, params: np.ndarray, param_structure: Dict, 
                        observations: List[Dict], cameras_dict: Dict) -> np.ndarray:
        """Compute residuals (delegates to cost function)"""
        return self._compute_residuals(params, param_structure, observations, cameras_dict)

    def get_algorithm_name(self) -> str:
        """Get algorithm name."""
        return "IncrementalBundleAdjustment"
    
    def optimize(self,
                cameras: Dict[str, Dict],
                points_3d: np.ndarray,
                observations: List[Dict],
                new_camera_id: str,
                optimize_intrinsics: bool = None,
                num_recent_cameras: int = None) -> OptimizationResult:
        """
        Perform incremental bundle adjustment after adding new camera.
        
        Args:
            cameras: Dictionary of all cameras
            points_3d: All 3D points (3xN)
            observations: All observations
            new_camera_id: ID of newly added camera
            optimize_intrinsics: Whether to optimize intrinsics
            num_recent_cameras: Number of recent cameras to optimize
            
        Returns:
            OptimizationResult with optimized parameters
        """
        start_time = time.time()
        
        # Use config defaults if not specified
        if optimize_intrinsics is None:
            optimize_intrinsics = self.config.OPTIMIZE_INTRINSICS
        if num_recent_cameras is None:
            num_recent_cameras = self.config.NUM_RECENT_CAMERAS
        
        # Validate inputs
        if len(cameras) < 2:
            return OptimizationResult(
                success=False,
                status=OptimizationStatus.INVALID_INPUT,
                metadata={'error': 'Need at least 2 cameras for BA'}
            )
        
        if points_3d.shape[1] < self.config.MIN_POINTS_FOR_BA:
            return OptimizationResult(
                success=False,
                status=OptimizationStatus.INVALID_INPUT,
                metadata={'error': f'Insufficient points: {points_3d.shape[1]} < {self.config.MIN_POINTS_FOR_BA}'}
            )
        
        if len(observations) < self.config.MIN_OBSERVATIONS:
            return OptimizationResult(
                success=False,
                status=OptimizationStatus.INVALID_INPUT,
                metadata={'error': f'Insufficient observations: {len(observations)} < {self.config.MIN_OBSERVATIONS}'}
            )
        
        # Select cameras to optimize
        camera_ids = list(cameras.keys())
        camera_ids_to_optimize = self._select_cameras_to_optimize(
            camera_ids, new_camera_id, num_recent_cameras
        )
        fixed_cameras = [cid for cid in camera_ids if cid not in camera_ids_to_optimize]
        
        logger.info(f"Incremental BA after adding {new_camera_id}:")
        logger.info(f"  Optimizing {len(camera_ids_to_optimize)}/{len(camera_ids)} cameras")
        logger.info(f"  Optimizing {points_3d.shape[1]} points")
        logger.info(f"  Total observations: {len(observations)}")
        
        # Build parameter vector
        params, param_structure = ParameterBuilder.build_parameter_vector(
            camera_ids=camera_ids_to_optimize,
            cameras_dict=cameras,
            points_3d=points_3d,
            optimize_intrinsics=optimize_intrinsics,
            fixed_cameras=[]  # Will handle fixed cameras separately
        )
        
        # Compute initial cost
        initial_residuals = self._compute_residuals(
            params, param_structure, observations, cameras
        )
        initial_cost = self.cost_function.compute_cost(initial_residuals)
        
        logger.info(f"  Initial cost: {initial_cost:.3f}")
        
        try:
            # Run optimization
            if self.config.USE_ROBUST_LOSS:
                result = least_squares(
                    fun=self._compute_residuals,
                    x0=params,
                    args=(param_structure, observations, cameras),
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
                    args=(param_structure, observations, cameras),
                    method='trf',
                    max_nfev=self.config.MAX_ITERATIONS * len(params),
                    ftol=self.config.FUNCTION_TOLERANCE,
                    xtol=self.config.GRADIENT_TOLERANCE,
                    verbose=0
                )
            
            # Unpack optimized parameters
            optimized_cameras, optimized_points = ParameterBuilder.unpack_parameters(
                result.x, param_structure
            )
            
            # Merge with fixed cameras
            for cam_id in fixed_cameras:
                optimized_cameras[cam_id] = cameras[cam_id]
            
            # Compute final cost
            final_residuals = self._compute_residuals(
                result.x, param_structure, observations, cameras
            )
            final_cost = self.cost_function.compute_cost(final_residuals)
            
            runtime = time.time() - start_time
            
            logger.info(f"  Final cost: {final_cost:.3f} ({result.nfev} iterations, {runtime:.2f}s)")
            logger.info(f"  Cost reduction: {initial_cost - final_cost:.3f}")
            
            # Determine status
            if result.success:
                status = OptimizationStatus.CONVERGED
            elif result.nfev >= self.config.MAX_ITERATIONS * len(params):
                status = OptimizationStatus.MAX_ITERATIONS
            else:
                status = OptimizationStatus.SUCCESS
            
            return OptimizationResult(
                success=True,
                status=status,
                optimized_params={
                    'cameras': optimized_cameras,
                    'points_3d': optimized_points
                },
                initial_cost=initial_cost,
                final_cost=final_cost,
                num_iterations=result.nfev,
                residuals=final_residuals,
                runtime=runtime,
                metadata={
                    'algorithm': self.get_algorithm_name(),
                    'new_camera_id': new_camera_id,
                    'optimized_cameras': camera_ids_to_optimize,
                    'fixed_cameras': fixed_cameras,
                    'optimize_intrinsics': optimize_intrinsics,
                    'num_observations': len(observations),
                    'scipy_result': result
                }
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                status=OptimizationStatus.FAILED,
                initial_cost=initial_cost,
                runtime=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def _select_cameras_to_optimize(self,
                                   all_camera_ids: List[str],
                                   new_camera_id: str,
                                   num_recent: int) -> List[str]:
        """
        Select which cameras to optimize.
        
        Strategy:
        - Always include the new camera
        - Include N-1 most recent cameras
        - Keep earlier cameras fixed for stability
        
        Args:
            all_camera_ids: List of all camera IDs
            new_camera_id: Newly added camera
            num_recent: Total number of recent cameras to optimize
            
        Returns:
            List of camera IDs to optimize
        """
        # Find index of new camera
        if new_camera_id not in all_camera_ids:
            # If not found, optimize last N cameras
            return all_camera_ids[-num_recent:]
        
        new_idx = all_camera_ids.index(new_camera_id)
        
        # Select recent cameras including new one
        start_idx = max(0, new_idx - num_recent + 1)
        selected = all_camera_ids[start_idx:new_idx + 1]
        
        # Ensure we have the new camera
        if new_camera_id not in selected:
            selected.append(new_camera_id)
        
        return selected
    
    def _compute_residuals(self,
                          params: np.ndarray,
                          param_structure: Dict,
                          observations: List[Dict],
                          cameras_dict: Dict) -> np.ndarray:
        """Compute reprojection residuals."""
        residuals = self.cost_function.compute_residuals(
            params,
            param_structure,
            observations,
            cameras_dict,
            param_structure['optimize_intrinsics']
        )
        
        return residuals
    
    def compute_cost(self, *args, **kwargs) -> float:
        """Compute optimization cost."""
        raise NotImplementedError("Use optimize() method directly")
    
    def check_convergence(self, *args, **kwargs) -> bool:
        """Check convergence."""
        return True


def adjust_after_new_camera(reconstruction_state: Dict,
                            new_camera_id: str,
                            optimize_intrinsics: bool = True,
                            num_recent_cameras: int = 3,
                            **config) -> Dict[str, Any]:
    """
    Convenience function for incremental BA after adding new camera.
    
    Args:
        reconstruction_state: Current reconstruction state
        new_camera_id: ID of newly added camera
        optimize_intrinsics: Whether to optimize intrinsics
        num_recent_cameras: Number of recent cameras to optimize
        **config: Additional configuration
        
    Returns:
        Updated reconstruction state dictionary
    """
    # Extract data from reconstruction state
    cameras = reconstruction_state.get('cameras', {})
    points_3d = reconstruction_state.get('points_3d', {}).get('points_3d', np.array([]))
    observations_dict = reconstruction_state.get('observations', {})
    
    # Convert observations to list format
    observations = []
    for cam_id, cam_obs in observations_dict.items():
        for obs in cam_obs:
            observations.append({
                'camera_id': cam_id,
                'point_id': obs['point_id'],
                'image_point': obs.get('image_point', obs.get('coords_2d'))
            })
    
    # Run incremental BA
    adjuster = IncrementalBundleAdjustment(**config)
    
    result = adjuster.optimize(
        cameras=cameras,
        points_3d=points_3d,
        observations=observations,
        new_camera_id=new_camera_id,
        optimize_intrinsics=optimize_intrinsics,
        num_recent_cameras=num_recent_cameras
    )
    
    if result.success:
        # Update reconstruction state
        reconstruction_state['cameras'] = result.optimized_params['cameras']
        reconstruction_state['points_3d']['points_3d'] = result.optimized_params['points_3d']
        
        # Add optimization history
        if 'optimization_history' not in reconstruction_state:
            reconstruction_state['optimization_history'] = []
        
        reconstruction_state['optimization_history'].append({
            'type': 'incremental',
            'new_camera': new_camera_id,
            'optimized_cameras': result.metadata['optimized_cameras'],
            'initial_cost': result.initial_cost,
            'final_cost': result.final_cost,
            'iterations': result.num_iterations,
            'runtime': result.runtime
        })
        
        return reconstruction_state
    else:
        logger.info(f"Warning: Incremental BA failed: {result.metadata.get('error')}")
        return reconstruction_state