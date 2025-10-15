"""
Two-View Bundle Adjustment

Optimizes camera poses and 3D points for the initial two-view reconstruction.
This is typically used after essential matrix estimation and triangulation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import least_squares
import warnings

from core.interfaces.base_optimizer import BaseOptimizer, OptimizationResult, OptimizationStatus
from .cost_functions import BACostFunction, ParameterBuilder, compute_mean_reprojection_error


class TwoViewBundleAdjustmentConfig:
    """Configuration for two-view bundle adjustment"""
    
    # Optimization parameters
    MAX_ITERATIONS = 100
    FUNCTION_TOLERANCE = 1e-6
    GRADIENT_TOLERANCE = 1e-6
    
    # Robust loss
    USE_ROBUST_LOSS = True
    ROBUST_LOSS_TYPE = 'huber'
    ROBUST_LOSS_THRESHOLD = 2.0
    
    # What to optimize
    OPTIMIZE_POSES = True
    OPTIMIZE_POINTS = True
    OPTIMIZE_INTRINSICS = True
    FIX_FIRST_CAMERA = True  # Typically fix first camera for gauge freedom


class TwoViewBundleAdjustment(BaseOptimizer):
    """
    Bundle adjustment for two-view reconstruction.
    
    Optimizes:
    - Camera poses (rotation and translation)
    - Camera intrinsics (optional, per-camera)
    - 3D point positions
    
    Typically used after:
    - Essential matrix estimation
    - Initial triangulation
    
    Before:
    - Adding more views
    - Dense reconstruction
    """
    
    def __init__(self, **config):
        """
        Initialize two-view bundle adjuster.
        
        Args:
            **config: Configuration overrides
        """
        super().__init__(**config)
        self.config = TwoViewBundleAdjustmentConfig()
        
        # Override config if provided
        for key, value in config.items():
            if hasattr(self.config, key.upper()):
                setattr(self.config, key.upper(), value)
        
        self.cost_function = BACostFunction(
            loss_function=self.config.ROBUST_LOSS_TYPE,
            loss_threshold=self.config.ROBUST_LOSS_THRESHOLD
        )
    
    def get_algorithm_name(self) -> str:
        """Get algorithm name."""
        return "TwoViewBundleAdjustment"
    
    def optimize(self,
                cameras: Dict[str, Dict],
                points_3d: np.ndarray,
                observations: List[Dict],
                optimize_intrinsics: bool = None,
                fix_first_camera: bool = None) -> OptimizationResult:
        """
        Perform two-view bundle adjustment.
        
        Args:
            cameras: Dictionary of camera data {image_id: {'R', 't', 'K'}}
            points_3d: 3D points (3xN)
            observations: List of observations [{'camera_id', 'point_id', 'image_point'}]
            optimize_intrinsics: Whether to optimize intrinsics (uses config if None)
            fix_first_camera: Whether to fix first camera (uses config if None)
            
        Returns:
            OptimizationResult with optimized parameters
        """
        import time
        start_time = time.time()
        
        # Use config defaults if not specified
        if optimize_intrinsics is None:
            optimize_intrinsics = self.config.OPTIMIZE_INTRINSICS
        if fix_first_camera is None:
            fix_first_camera = self.config.FIX_FIRST_CAMERA
        
        # Validate inputs
        if len(cameras) != 2:
            return OptimizationResult(
                success=False,
                status=OptimizationStatus.INVALID_INPUT,
                metadata={'error': f'Two-view BA requires exactly 2 cameras, got {len(cameras)}'}
            )
        
        if points_3d.shape[1] < 10:
            return OptimizationResult(
                success=False,
                status=OptimizationStatus.INVALID_INPUT,
                metadata={'error': f'Insufficient points for BA: {points_3d.shape[1]} < 10'}
            )
        
        # Determine which cameras to fix
        camera_ids = list(cameras.keys())
        fixed_cameras = [camera_ids[0]] if fix_first_camera else []
        
        # Build parameter vector
        params, param_structure = ParameterBuilder.build_parameter_vector(
            camera_ids=camera_ids,
            cameras_dict=cameras,
            points_3d=points_3d,
            optimize_intrinsics=optimize_intrinsics,
            fixed_cameras=fixed_cameras
        )
        
        # Compute initial cost
        initial_residuals = self._compute_residuals(
            params, param_structure, observations, cameras
        )
        initial_cost = self.cost_function.compute_cost(initial_residuals)
        
        print(f"Two-view BA: Optimizing {len(camera_ids)} cameras, {points_3d.shape[1]} points")
        print(f"  Initial cost: {initial_cost:.3f}")
        
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
            
            # Restore fixed camera if needed
            if fix_first_camera:
                optimized_cameras[camera_ids[0]] = cameras[camera_ids[0]]
            
            # Compute final cost
            final_residuals = self._compute_residuals(
                result.x, param_structure, observations, cameras
            )
            final_cost = self.cost_function.compute_cost(final_residuals)
            
            runtime = time.time() - start_time
            
            print(f"  Final cost: {final_cost:.3f} ({result.nfev} iterations, {runtime:.2f}s)")
            print(f"  Cost reduction: {initial_cost - final_cost:.3f}")
            
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
                    'optimize_intrinsics': optimize_intrinsics,
                    'fix_first_camera': fix_first_camera,
                    'num_cameras': len(cameras),
                    'num_points': points_3d.shape[1],
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
        """Compute optimization cost (required by BaseOptimizer)."""
        # This would be implemented if needed for cost evaluation
        raise NotImplementedError("Use optimize() method directly")
    
    def check_convergence(self, *args, **kwargs) -> bool:
        """Check convergence (handled by scipy.optimize)."""
        return True  # Handled internally by least_squares


# Convenience function

def adjust_two_view(cameras: Dict[str, Dict],
                   points_3d: np.ndarray,
                   observations: List[Dict],
                   optimize_intrinsics: bool = True,
                   fix_first_camera: bool = True,
                   **config) -> Dict[str, Any]:
    """
    Convenience function for two-view bundle adjustment.
    
    Args:
        cameras: Dictionary of camera data
        points_3d: 3D points (3xN)
        observations: List of observations
        optimize_intrinsics: Whether to optimize intrinsics
        fix_first_camera: Whether to fix first camera
        **config: Additional configuration
        
    Returns:
        Dictionary with optimization results
    """
    adjuster = TwoViewBundleAdjustment(**config)
    
    result = adjuster.optimize(
        cameras=cameras,
        points_3d=points_3d,
        observations=observations,
        optimize_intrinsics=optimize_intrinsics,
        fix_first_camera=fix_first_camera
    )
    
    if result.success:
        return {
            'success': True,
            'cameras': result.optimized_params['cameras'],
            'points_3d': result.optimized_params['points_3d'],
            'initial_cost': result.initial_cost,
            'final_cost': result.final_cost,
            'cost_reduction': result.get_cost_reduction(),
            'num_iterations': result.num_iterations,
            'runtime': result.runtime
        }
    else:
        return {
            'success': False,
            'error': result.metadata.get('error', 'Optimization failed'),
            'initial_cost': result.initial_cost
        }