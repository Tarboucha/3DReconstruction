"""
Bundle Adjustment for Monument Reconstruction
============================================

Incremental and global bundle adjustment for optimizing camera poses,
3D points, and per-camera intrinsics in monument reconstruction pipeline.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import warnings
from dataclasses import dataclass

@dataclass
class BundleAdjustmentConfig:
    """Configuration for bundle adjustment optimized for monuments"""
    
    # Optimization parameters
    MAX_ITERATIONS = 100
    FUNCTION_TOLERANCE = 1e-6#6
    GRADIENT_TOLERANCE = 1e-6#6
    
    # Robust loss function
    USE_ROBUST_LOSS = True
    ROBUST_LOSS_THRESHOLD = 2.0  # Huber loss threshold
    
    # What to optimize
    OPTIMIZE_POSES = True
    OPTIMIZE_POINTS = True
    OPTIMIZE_INTRINSICS = True
    
    # Incremental BA thresholds
    MIN_POINTS_FOR_INCREMENTAL = 10
    MAX_REPROJECTION_ERROR = 3.0

class IncrementalBundleAdjuster:
    """
    Incremental bundle adjustment after adding each new view
    Each camera has its own intrinsics (for random online images)
    """
    
    def __init__(self):
        """Initialize incremental bundle adjuster"""
        self.config = BundleAdjustmentConfig()
        
    def adjust_after_new_view(self, reconstruction_state: Dict,
                            new_image_id: str,
                            optimize_intrinsics: bool = True) -> Dict:
        """
        Perform incremental bundle adjustment after adding a new view
        
        Args:
            reconstruction_state: Current reconstruction state
            new_image_id: ID of newly added image
            optimize_intrinsics: Whether to optimize camera intrinsics
            
        Returns:
            Updated reconstruction state with optimized parameters
        """
        cameras = reconstruction_state['cameras']
        points_3d = reconstruction_state['points_3d']['points_3d']
        observations = reconstruction_state.get('observations', {})
        
        # Only adjust if we have sufficient data
        if len(cameras) < 2 or points_3d.shape[1] < self.config.MIN_POINTS_FOR_INCREMENTAL:
            return reconstruction_state
        
        print(f"Running incremental BA after adding {new_image_id}...")
        print(f"  Cameras: {len(cameras)}, Points: {points_3d.shape[1]}")
        
        # For incremental BA, optimize only recent cameras + all points
        recent_cameras = self._get_recent_cameras(cameras, new_image_id, max_recent=3)
        
        # Prepare optimization data
        optimization_data = self._prepare_incremental_data(
            recent_cameras, points_3d, observations, optimize_intrinsics
        )
        
        if optimization_data is None:
            return reconstruction_state
        
        # Run optimization
        result = self._run_bundle_adjustment(optimization_data, reconstruction_state)
        
        if result['success']:
            # Update reconstruction state
            updated_state = self._update_reconstruction_state(
                reconstruction_state, result, recent_cameras, optimize_intrinsics
            )
            
            print(f"  BA converged: error {result['initial_cost']:.3f} → {result['final_cost']:.3f}")
            return updated_state
        else:
            print(f"  BA failed: {result.get('error', 'Unknown error')}")
            return reconstruction_state
    
    def _get_recent_cameras(self, cameras: Dict, new_image_id: str, max_recent: int = 3) -> List[str]:
        """Get list of recent cameras to optimize in incremental BA"""
        camera_ids = list(cameras.keys())
        
        # Always include the new camera
        recent_cameras = [new_image_id]
        
        # Add other recent cameras (excluding new one)
        other_cameras = [cam_id for cam_id in camera_ids if cam_id != new_image_id]
        
        # Add most recent cameras up to max_recent
        recent_cameras.extend(other_cameras[-(max_recent-1):])
        
        return recent_cameras
    
    def _prepare_incremental_data(self, camera_ids: List[str], 
                                points_3d: np.ndarray,
                                observations: Dict,
                                optimize_intrinsics: bool) -> Optional[Dict]:
        """Prepare data for incremental bundle adjustment"""
        
        # Collect observations for recent cameras
        camera_observations = []
        
        for cam_id in camera_ids:
            if cam_id in observations:
                cam_obs = observations[cam_id]
                for obs in cam_obs:
                    camera_observations.append({
                        'camera_id': cam_id,
                        'point_id': obs['point_id'],
                        'image_point': obs['image_point']
                    })
        
        if len(camera_observations) < 10:  # Need sufficient observations
            return None
        
        return {
            'camera_ids': camera_ids,
            'points_3d': points_3d,
            'observations': camera_observations,
            'optimize_intrinsics': optimize_intrinsics,
            'adjustment_type': 'incremental'
        }
    
    def _run_bundle_adjustment(self, optimization_data: Dict, reconstruction_state: Dict) -> Dict:
        """Run the actual bundle adjustment optimization"""
        
        camera_ids = optimization_data['camera_ids']
        points_3d = optimization_data['points_3d']
        observations = optimization_data['observations']
        optimize_intrinsics = optimization_data['optimize_intrinsics']
        
        # Build parameter vector
        params, param_structure = self._build_parameter_vector(
            camera_ids, points_3d, reconstruction_state, optimize_intrinsics
        )
        
        # Build observation vector
        observation_vector = self._build_observation_vector(observations)
        
        if len(observation_vector) == 0:
            return {'success': False, 'error': 'No valid observations'}
        
        # Calculate initial cost
        initial_residuals = self._calculate_residuals(params, param_structure, observations, reconstruction_state)
        initial_cost = np.sqrt(np.mean(initial_residuals**2))
        
        try:
            # Run least squares optimization
            if self.config.USE_ROBUST_LOSS:
                result = least_squares(
                    fun=self._calculate_residuals,
                    x0=params,
                    args=(param_structure, observations, reconstruction_state),
                    method='trf',
                    max_nfev=self.config.MAX_ITERATIONS * len(params),
                    ftol=self.config.FUNCTION_TOLERANCE,
                    xtol=self.config.GRADIENT_TOLERANCE,
                    loss='huber',
                    f_scale=self.config.ROBUST_LOSS_THRESHOLD
                )
            else:
                result = least_squares(
                    fun=self._calculate_residuals,
                    x0=params,
                    args=(param_structure, observations, reconstruction_state),
                    method='lm',
                    max_nfev=self.config.MAX_ITERATIONS * len(params),
                    ftol=self.config.FUNCTION_TOLERANCE,
                    xtol=self.config.GRADIENT_TOLERANCE,
                    loss='linear',
                    verbose=True
                )
            
            final_cost = np.sqrt(np.mean(result.fun**2))
            
            return {
                'success': result.success,
                'optimized_params': result.x,
                'param_structure': param_structure,
                'initial_cost': initial_cost,
                'final_cost': final_cost,
                'iterations': result.nfev,
                'termination_reason': result.message
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Optimization failed: {str(e)}'
            }
    
    def _build_parameter_vector(self, camera_ids: List[str], points_3d: np.ndarray,
                              reconstruction_state: Dict, optimize_intrinsics: bool) -> Tuple[np.ndarray, Dict]:
        """
        Build parameter vector for optimization with per-camera intrinsics
        
        Parameter organization:
        [camera1_pose(6), camera1_intrinsics(4), camera2_pose(6), camera2_intrinsics(4), ..., points_3d(3*N)]
        """
        params = []
        param_structure = {
            'camera_params': {},
            'points_start_idx': 0,
            'optimize_intrinsics': optimize_intrinsics,
            'total_cameras': len(camera_ids)
        }
        
        param_idx = 0
        
        # Each camera: 6 DOF pose + 4 intrinsic parameters (if optimizing)
        for cam_id in camera_ids:
            # Camera pose (rotation vector + translation vector)
            camera_data = reconstruction_state['cameras'].get(cam_id, {})
            
            # Get initial pose from reconstruction state
            if 'R' in camera_data and 't' in camera_data:
                R = camera_data['R']
                t = camera_data['t'].flatten()
                rvec = cv2.Rodrigues(R)[0].flatten()
            else:
                # Fallback to identity
                rvec = np.zeros(3)
                t = np.zeros(3)
            
            params.extend(rvec)
            params.extend(t)
            
            param_structure['camera_params'][cam_id] = {
                'rvec_idx': param_idx,
                'tvec_idx': param_idx + 3
            }
            param_idx += 6
            
            # Per-camera intrinsics
            if optimize_intrinsics:
                # Get initial intrinsics from reconstruction state or use defaults
                if 'K' in camera_data:
                    K = camera_data['K']
                    fx, fy = K[0, 0], K[1, 1]
                    cx, cy = K[0, 2], K[1, 2]
                else:
                    # Default values for random online images
                    fx = fy = 1000.0  # Reasonable default focal length
                    cx = cy = 500.0   # Assume center principal point
                
                params.extend([fx, fy, cx, cy])
                
                param_structure['camera_params'][cam_id]['intrinsics_idx'] = param_idx
                param_idx += 4
        
        # 3D points (3 coordinates each)
        param_structure['points_start_idx'] = param_idx
        if points_3d.shape[0] == 4:  # Homogeneous coordinates
            points_flat = points_3d[:3].ravel()  # Only X,Y,Z
        else:
            points_flat = points_3d.ravel()
        
        params.extend(points_flat)
        
        return np.array(params), param_structure
    
    def _build_observation_vector(self, observations: List[Dict]) -> np.ndarray:
        """Build observation vector from image point measurements"""
        observation_vector = []
        
        for obs in observations:
            image_point = obs['image_point']
            observation_vector.extend([image_point[0], image_point[1]])
        
        return np.array(observation_vector)
    
    def _calculate_residuals(self, params: np.ndarray, param_structure: Dict,
                           observations: List[Dict], reconstruction_state: Dict) -> np.ndarray:
        """
        Calculate reprojection residuals with per-camera intrinsics
        
        For each observation, projects the 3D point using the camera's specific
        intrinsics and pose, then computes the difference from observed 2D location.
        """
        residuals = []
        
        # Extract 3D points from parameter vector
        points_start = param_structure['points_start_idx']
        points_3d_flat = params[points_start:]
        
        for obs in observations:
            camera_id = obs['camera_id']
            point_id = obs['point_id']
            observed_point = obs['image_point']
            
            # Get camera parameters
            if camera_id not in param_structure['camera_params']:
                residuals.extend([100.0, 100.0])  # Large error for missing camera
                continue
                
            cam_params = param_structure['camera_params'][camera_id]
            rvec = params[cam_params['rvec_idx']:cam_params['rvec_idx']+3]
            tvec = params[cam_params['tvec_idx']:cam_params['tvec_idx']+3]
            
            # Get camera intrinsics - each camera has its own
            if param_structure['optimize_intrinsics']:
                # Use per-camera intrinsics from parameter vector
                intrinsics_idx = cam_params['intrinsics_idx']
                fx, fy, cx, cy = params[intrinsics_idx:intrinsics_idx+4]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            else:
                # Use fixed per-camera intrinsics from reconstruction state
                camera_data = reconstruction_state['cameras'].get(camera_id, {})
                K = camera_data.get('K')
                if K is None:
                    residuals.extend([100.0, 100.0])  # No intrinsics available
                    continue
            
            # Get 3D point
            if point_id * 3 + 2 >= len(points_3d_flat):
                residuals.extend([100.0, 100.0])  # Point out of range
                continue
                
            point_3d = points_3d_flat[point_id*3:(point_id+1)*3]
            
            # Project 3D point to image using this camera's specific intrinsics
            try:
                projected_points, _ = cv2.projectPoints(
                    point_3d.reshape(1, 1, 3),
                    rvec, tvec, K, None
                )
                projected_point = projected_points[0, 0]
                
                # Calculate reprojection error
                residual_x = projected_point[0] - observed_point[0]
                residual_y = projected_point[1] - observed_point[1]
                
                residuals.extend([residual_x, residual_y])
                
            except Exception:
                # Projection failed - add large residuals
                residuals.extend([100.0, 100.0])
        
        return np.array(residuals)
    
    def _update_reconstruction_state(self, reconstruction_state: Dict,
                                   optimization_result: Dict,
                                   optimized_cameras: List[str],
                                   optimize_intrinsics: bool) -> Dict:
        """Update reconstruction state with optimized per-camera parameters"""
        
        updated_state = reconstruction_state.copy()
        params = optimization_result['optimized_params']
        param_structure = optimization_result['param_structure']
        
        # Update each camera's pose and intrinsics
        for cam_id in optimized_cameras:
            if cam_id in param_structure['camera_params']:
                cam_params = param_structure['camera_params'][cam_id]
                
                # Update pose
                rvec = params[cam_params['rvec_idx']:cam_params['rvec_idx']+3]
                tvec = params[cam_params['tvec_idx']:cam_params['tvec_idx']+3]
                
                R = cv2.Rodrigues(rvec)[0]
                t = tvec.reshape(3, 1)
                
                updated_state['cameras'][cam_id]['R'] = R
                updated_state['cameras'][cam_id]['t'] = t
                
                # Update per-camera intrinsics
                if optimize_intrinsics and 'intrinsics_idx' in cam_params:
                    intrinsics_idx = cam_params['intrinsics_idx']
                    fx, fy, cx, cy = params[intrinsics_idx:intrinsics_idx+4]
                    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    updated_state['cameras'][cam_id]['K'] = K
        
        # Update 3D points
        points_start = param_structure['points_start_idx']
        points_3d_flat = params[points_start:]
        
        num_points = len(points_3d_flat) // 3
        updated_points_3d = points_3d_flat.reshape(3, num_points)
        updated_state['points_3d']['points_3d'] = updated_points_3d
        
        # Add optimization metadata
        updated_state['optimization_history'] = updated_state.get('optimization_history', [])
        updated_state['optimization_history'].append({
            'type': 'incremental',
            'cameras_optimized': optimized_cameras,
            'initial_cost': optimization_result['initial_cost'],
            'final_cost': optimization_result['final_cost'],
            'iterations': optimization_result['iterations']
        })
        
        return updated_state

class GlobalBundleAdjuster:
    """
    Global bundle adjustment for final optimization of entire reconstruction
    Each camera maintains its own intrinsics
    """
    
    def __init__(self):
        """Initialize global bundle adjuster"""
        self.config = BundleAdjustmentConfig()
    
    def adjust_global(self, reconstruction_state: Dict,
                     optimize_intrinsics: bool = True,
                     fix_first_camera: bool = True) -> Dict:
        """
        Perform global bundle adjustment on entire reconstruction
        
        Args:
            reconstruction_state: Complete reconstruction state
            optimize_intrinsics: Whether to optimize per-camera intrinsics
            fix_first_camera: Whether to fix first camera pose (recommended)
            
        Returns:
            Globally optimized reconstruction state
        """
        cameras = reconstruction_state['cameras']
        points_3d = reconstruction_state['points_3d']['points_3d']
        observations = reconstruction_state.get('observations', {})
        
        print(f"Running global bundle adjustment...")
        print(f"  Cameras: {len(cameras)}, Points: {points_3d.shape[1]}")
        
        # Prepare all data for global optimization
        optimization_data = self._prepare_global_data(
            cameras, points_3d, observations, optimize_intrinsics, fix_first_camera
        )
        
        if optimization_data is None:
            print("  Insufficient data for global BA")
            return reconstruction_state
        
        # Use same optimization framework as incremental
        incremental_adjuster = IncrementalBundleAdjuster()
        result = incremental_adjuster._run_bundle_adjustment(optimization_data, reconstruction_state)
        
        if result['success']:
            # Update reconstruction state
            updated_state = incremental_adjuster._update_reconstruction_state(
                reconstruction_state, result, list(cameras.keys()), optimize_intrinsics
            )
            
            print(f"  Global BA converged: error {result['initial_cost']:.3f} → {result['final_cost']:.3f}")
            
            # Mark as global optimization
            if updated_state['optimization_history']:
                updated_state['optimization_history'][-1]['type'] = 'global'
                
            return updated_state
        else:
            print(f"  Global BA failed: {result.get('error', 'Unknown error')}")
            return reconstruction_state
    
    def _prepare_global_data(self, cameras: Dict, points_3d: np.ndarray,
                           observations: Dict, optimize_intrinsics: bool,
                           fix_first_camera: bool) -> Optional[Dict]:
        """Prepare data for global bundle adjustment"""
        
        # Collect all observations
        all_observations = []
        for cam_id, cam_obs in observations.items():
            for obs in cam_obs:
                all_observations.append({
                    'camera_id': cam_id,
                    'point_id': obs['point_id'],
                    'image_point': obs['image_point']
                })
        
        if len(all_observations) < 50:  # Need sufficient observations for global BA
            return None
        
        camera_ids = list(cameras.keys())
        
        return {
            'camera_ids': camera_ids,
            'points_3d': points_3d,
            'observations': all_observations,
            'optimize_intrinsics': optimize_intrinsics,
            'fix_first_camera': fix_first_camera,
            'adjustment_type': 'global'
        }


