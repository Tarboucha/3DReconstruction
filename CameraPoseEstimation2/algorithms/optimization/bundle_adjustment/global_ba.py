"""
Global Bundle Adjustment

Final optimization of the entire reconstruction including all cameras and 3D points.
This is typically performed after all cameras have been added to refine the complete
reconstruction before dense reconstruction or export.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from scipy.optimize import least_squares
import time

from CameraPoseEstimation2.core.interfaces.base_optimizer import BaseOptimizer, OptimizationResult, OptimizationStatus
from .cost_functions import BACostFunction, ParameterBuilder, compute_mean_reprojection_error


class GlobalBundleAdjustmentConfig:
    """Configuration for global bundle adjustment"""
    
    # Optimization parameters
    MAX_ITERATIONS = 200  # More iterations for global refinement
    FUNCTION_TOLERANCE = 1e-6
    GRADIENT_TOLERANCE = 1e-6
    
    # Robust loss
    USE_ROBUST_LOSS = True
    ROBUST_LOSS_TYPE = 'huber'
    ROBUST_LOSS_THRESHOLD = 2.0
    
    # What to optimize
    OPTIMIZE_ALL_CAMERAS = True
    OPTIMIZE_ALL_POINTS = True
    OPTIMIZE_INTRINSICS = True
    FIX_FIRST_CAMERA = True  # Fix first camera for gauge freedom
    
    # Minimum requirements
    MIN_CAMERAS = 2
    MIN_POINTS = 50
    MIN_OBSERVATIONS = 100


class GlobalBundleAdjustment(BaseOptimizer):
    """
    Global bundle adjustment for complete reconstruction refinement.
    
    Optimizes:
    - All camera poses
    - All 3D points
    - Per-camera intrinsics (optional)
    
    Use when:
    - All cameras have been added
    - Final refinement before export
    - Preparation for dense reconstruction
    
    Benefits:
    - Globally consistent reconstruction
    - Distributes error across entire structure
    - Best possible accuracy
    
    Considerations:
    - Computationally expensive
    - Only run once or twice per reconstruction
    - Fix first camera to avoid gauge ambiguity
    """
    
    def __init__(self, **config):
        """
        Initialize global bundle adjuster.
        
        Args:
            **config: Configuration overrides
        """
        super().__init__(**config)
        self.config = GlobalBundleAdjustmentConfig()
        
        # Override config
        for key, value in config.items():
            if hasattr(self.config, key.upper()):
                setattr(self.config, key.upper(), value)
        
        self.cost_function = BACostFunction(
            loss_function=self.config.ROBUST_LOSS_TYPE,
            loss_threshold=self.config.ROBUST_LOSS_THRESHOLD
        )
    
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
        return "GlobalBundleAdjustment"
    
    def optimize(self,
                cameras: Dict[str, Dict],
                points_3d: np.ndarray,
                observations: List[Dict],
                optimize_intrinsics: bool = None,
                fix_first_camera: bool = None) -> OptimizationResult:
        """
        Perform global bundle adjustment on entire reconstruction.
        
        Args:
            cameras: Dictionary of all cameras
            points_3d: All 3D points (3xN)
            observations: All observations
            optimize_intrinsics: Whether to optimize intrinsics
            fix_first_camera: Whether to fix first camera
            
        Returns:
            OptimizationResult with globally optimized parameters
        """
        start_time = time.time()
        
        # Use config defaults if not specified
        if optimize_intrinsics is None:
            optimize_intrinsics = self.config.OPTIMIZE_INTRINSICS
        if fix_first_camera is None:
            fix_first_camera = self.config.FIX_FIRST_CAMERA
        
        # Validate inputs
        if len(cameras) < self.config.MIN_CAMERAS:
            return OptimizationResult(
                success=False,
                status=OptimizationStatus.INVALID_INPUT,
                metadata={'error': f'Insufficient cameras: {len(cameras)} < {self.config.MIN_CAMERAS}'}
            )
        
        if points_3d.shape[1] < self.config.MIN_POINTS:
            return OptimizationResult(
                success=False,
                status=OptimizationStatus.INVALID_INPUT,
                metadata={'error': f'Insufficient points: {points_3d.shape[1]} < {self.config.MIN_POINTS}'}
            )
        
        if len(observations) < self.config.MIN_OBSERVATIONS:
            return OptimizationResult(
                success=False,
                status=OptimizationStatus.INVALID_INPUT,
                metadata={'error': f'Insufficient observations: {len(observations)} < {self.config.MIN_OBSERVATIONS}'}
            )
        
        print(f"\n{'='*70}")
        print(f"GLOBAL BUNDLE ADJUSTMENT")
        print(f"{'='*70}")
        print(f"Cameras: {len(cameras)}")
        print(f"Points: {points_3d.shape[1]}")
        print(f"Observations: {len(observations)}")
        print(f"Optimize intrinsics: {optimize_intrinsics}")
        print(f"Fix first camera: {fix_first_camera}")
        
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
        
        print(f"Parameter vector size: {len(params)}")
        
        # Compute initial cost
        initial_residuals = self._compute_residuals(
            params, param_structure, observations, cameras
        )
        initial_cost = self.cost_function.compute_cost(initial_residuals)
        initial_mean_error = compute_mean_reprojection_error(
            points_3d, observations, cameras
        )
        
        print(f"\nInitial state:")
        print(f"  Cost: {initial_cost:.3f}")
        print(f"  Mean reprojection error: {initial_mean_error:.2f}px")
        
        try:
            # Run optimization
            print(f"\nRunning optimization...")
            
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
                    verbose=2  # Show optimization progress
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
                    verbose=2
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
            
            # Compute final mean error
            final_observations = []
            for obs in observations:
                final_observations.append({
                    'camera_id': obs['camera_id'],
                    'point_id': obs['point_id'],
                    'image_point': obs['image_point']
                })
            final_mean_error = compute_mean_reprojection_error(
                optimized_points, final_observations, optimized_cameras
            )
            
            runtime = time.time() - start_time
            
            print(f"\nOptimization complete!")
            print(f"  Iterations: {result.nfev}")
            print(f"  Runtime: {runtime:.2f}s")
            print(f"\nFinal state:")
            print(f"  Cost: {final_cost:.3f}")
            print(f"  Mean reprojection error: {final_mean_error:.2f}px")
            print(f"\nImprovement:")
            print(f"  Cost reduction: {initial_cost - final_cost:.3f} ({(initial_cost - final_cost)/initial_cost*100:.1f}%)")
            print(f"  Error reduction: {initial_mean_error - final_mean_error:.2f}px")
            print(f"{'='*70}\n")
            
            # Determine status
            if result.success:
                status = OptimizationStatus.CONVERGED
            elif result.nfev >= self.config.MAX_ITERATIONS * len(params):
                status = OptimizationStatus.MAX_ITERATIONS
                print("Warning: Reached maximum iterations")
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
                    'initial_mean_error': initial_mean_error,
                    'final_mean_error': final_mean_error,
                    'error_reduction': initial_mean_error - final_mean_error,
                    'scipy_result': result
                }
            )
            
        except Exception as e:
            print(f"\n✗ Global BA failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
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
        """Compute optimization cost."""
        raise NotImplementedError("Use optimize() method directly")
    
    def check_convergence(self, *args, **kwargs) -> bool:
        """Check convergence."""
        return True


def adjust_global(reconstruction_state: Dict,
                 optimize_intrinsics: bool = True,
                 fix_first_camera: bool = True,
                 **config) -> Dict[str, Any]:
    """
    Convenience function for global bundle adjustment.
    
    Args:
        reconstruction_state: Complete reconstruction state
        optimize_intrinsics: Whether to optimize intrinsics
        fix_first_camera: Whether to fix first camera
        **config: Additional configuration
        
    Returns:
        Globally optimized reconstruction state
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
    
    # Run global BA
    adjuster = GlobalBundleAdjustment(**config)
    
    result = adjuster.optimize(
        cameras=cameras,
        points_3d=points_3d,
        observations=observations,
        optimize_intrinsics=optimize_intrinsics,
        fix_first_camera=fix_first_camera
    )
    
    if result.success:
        # Update reconstruction state
        reconstruction_state['cameras'] = result.optimized_params['cameras']
        reconstruction_state['points_3d']['points_3d'] = result.optimized_params['points_3d']
        
        # Add optimization history
        if 'optimization_history' not in reconstruction_state:
            reconstruction_state['optimization_history'] = []
        
        reconstruction_state['optimization_history'].append({
            'type': 'global',
            'all_cameras': list(cameras.keys()),
            'initial_cost': result.initial_cost,
            'final_cost': result.final_cost,
            'iterations': result.num_iterations,
            'runtime': result.runtime,
            'initial_error': result.metadata['initial_mean_error'],
            'final_error': result.metadata['final_mean_error']
        })
        
        return reconstruction_state
    else:
        print(f"Warning: Global BA failed: {result.metadata.get('error')}")
        return reconstruction_state