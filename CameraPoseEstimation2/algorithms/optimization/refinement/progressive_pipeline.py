"""
Progressive Refinement Pipeline

Combines multiple refinement strategies into a progressive multi-stage pipeline:
1. Pose refinement
2. Intrinsics learning
3. Structure refinement
4. Alternating optimization

Used for iterative improvement of reconstruction quality through multiple passes.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import time

from core.interfaces.base_optimizer import BaseOptimizer, OptimizationResult, OptimizationStatus
from .pose_refiner import PoseRefiner
from .intrinsics_refiner import ProgressiveIntrinsicsLearner
from .structure_refiner import StructureRefiner


class ProgressiveRefinementConfig:
    """Configuration for progressive refinement pipeline"""
    
    # Pipeline stages
    ENABLE_POSE_REFINEMENT = True
    ENABLE_INTRINSICS_LEARNING = True
    ENABLE_STRUCTURE_REFINEMENT = True
    
    # Iteration control
    MAX_ITERATIONS = 5
    MIN_IMPROVEMENT = 1e-3  # Minimum cost reduction to continue
    
    # Stage ordering
    REFINEMENT_ORDER = [
        'structure',    # Clean structure first
        'pose',         # Refine poses
        'intrinsics',   # Update intrinsics
        'structure'     # Clean again
    ]
    
    # Alternating optimization
    ALTERNATE_POSE_STRUCTURE = True
    ALTERNATE_ITERATIONS = 2


class ProgressiveRefinementPipeline(BaseOptimizer):
    """
    Progressive multi-stage refinement pipeline.
    
    Strategy:
    1. Refine structure (remove outliers)
    2. Refine camera poses
    3. Update intrinsics progressively
    4. Refine structure again
    5. Repeat until convergence
    
    Benefits:
    - Incremental improvement
    - Each stage helps the next
    - Robust to poor initialization
    - Tracks convergence
    
    Use cases:
    - After adding each camera (light refinement)
    - Before bundle adjustment (prepare for BA)
    - Between BA iterations (progressive improvement)
    - Final polish after global BA
    """
    
    def __init__(self, **config):
        """
        Initialize progressive refinement pipeline.
        
        Args:
            **config: Configuration overrides
        """
        super().__init__(**config)
        self.config = ProgressiveRefinementConfig()
        
        # Override config
        for key, value in config.items():
            if hasattr(self.config, key.upper()):
                setattr(self.config, key.upper(), value)
        
        # Initialize component refiners
        self.pose_refiner = PoseRefiner()
        self.intrinsics_learner = ProgressiveIntrinsicsLearner()
        self.structure_refiner = StructureRefiner()
        
        # Track history
        self.refinement_history = []
    
    def get_algorithm_name(self) -> str:
        """Get algorithm name."""
        return "ProgressiveRefinementPipeline"
    
    def refine_reconstruction(self,
                            reconstruction_state: Dict,
                            num_iterations: Optional[int] = None) -> OptimizationResult:
        """
        Progressively refine entire reconstruction.
        
        Args:
            reconstruction_state: Complete reconstruction state
            num_iterations: Number of refinement iterations (None = use config)
            
        Returns:
            OptimizationResult with refined reconstruction
        """
        start_time = time.time()
        
        if num_iterations is None:
            num_iterations = self.config.MAX_ITERATIONS
        
        print("\n" + "="*70)
        print("PROGRESSIVE REFINEMENT PIPELINE")
        print("="*70)
        print(f"Iterations: {num_iterations}")
        print(f"Stages: {', '.join(self.config.REFINEMENT_ORDER)}")
        
        # Extract components
        cameras = reconstruction_state['cameras']
        points_3d = reconstruction_state['points_3d']['points_3d']
        observations = reconstruction_state.get('observations', {})
        
        initial_num_points = points_3d.shape[1]
        
        print(f"\nInitial state:")
        print(f"  Cameras: {len(cameras)}")
        print(f"  Points: {initial_num_points}")
        
        # Initialize intrinsics if needed
        self._initialize_intrinsics(cameras, reconstruction_state.get('image_info', {}))
        
        # Progressive refinement loop
        prev_cost = float('inf')
        converged = False
        
        for iteration in range(num_iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{num_iterations}")
            print(f"{'='*70}")
            
            iter_start = time.time()
            
            # Run refinement stages
            for stage in self.config.REFINEMENT_ORDER:
                if stage == 'pose' and self.config.ENABLE_POSE_REFINEMENT:
                    cameras = self._refine_poses_stage(cameras, points_3d, observations)
                
                elif stage == 'intrinsics' and self.config.ENABLE_INTRINSICS_LEARNING:
                    cameras = self._refine_intrinsics_stage(cameras, points_3d, observations)
                
                elif stage == 'structure' and self.config.ENABLE_STRUCTURE_REFINEMENT:
                    points_3d, observations = self._refine_structure_stage(
                        points_3d, cameras, observations
                    )
            
            # Compute current cost (mean reprojection error)
            current_cost = self._compute_reconstruction_cost(
                points_3d, cameras, observations
            )
            
            # Check convergence
            improvement = prev_cost - current_cost
            
            iter_time = time.time() - iter_start
            
            print(f"\nIteration {iteration + 1} summary:")
            print(f"  Cost: {current_cost:.3f}px")
            print(f"  Improvement: {improvement:.3f}px")
            print(f"  Points: {points_3d.shape[1]}")
            print(f"  Time: {iter_time:.2f}s")
            
            # Record history
            self.refinement_history.append({
                'iteration': iteration + 1,
                'cost': current_cost,
                'improvement': improvement,
                'num_points': points_3d.shape[1],
                'runtime': iter_time
            })
            
            if improvement < self.config.MIN_IMPROVEMENT:
                print(f"\n✓ Converged (improvement < {self.config.MIN_IMPROVEMENT})")
                converged = True
                break
            
            prev_cost = current_cost
        
        # Update reconstruction state
        reconstruction_state['cameras'] = cameras
        reconstruction_state['points_3d']['points_3d'] = points_3d
        reconstruction_state['observations'] = observations
        
        total_runtime = time.time() - start_time
        final_num_points = points_3d.shape[1]
        
        print(f"\n{'='*70}")
        print("REFINEMENT COMPLETE")
        print(f"{'='*70}")
        print(f"Final state:")
        print(f"  Points: {final_num_points} (removed {initial_num_points - final_num_points})")
        print(f"  Final cost: {current_cost:.3f}px")
        print(f"  Total runtime: {total_runtime:.2f}s")
        print(f"  Converged: {converged}")
        
        return OptimizationResult(
            success=True,
            status=OptimizationStatus.CONVERGED if converged else OptimizationStatus.SUCCESS,
            optimized_params=reconstruction_state,
            initial_cost=float('inf') if not self.refinement_history else self.refinement_history[0]['cost'],
            final_cost=current_cost,
            num_iterations=len(self.refinement_history),
            runtime=total_runtime,
            metadata={
                'algorithm': self.get_algorithm_name(),
                'converged': converged,
                'history': self.refinement_history,
                'initial_points': initial_num_points,
                'final_points': final_num_points,
                'points_removed': initial_num_points - final_num_points
            }
        )
    
    def refine_after_camera_addition(self,
                                    reconstruction_state: Dict,
                                    new_camera_id: str,
                                    light_refinement: bool = True) -> Dict:
        """
        Light refinement after adding a new camera.
        
        Args:
            reconstruction_state: Current reconstruction
            new_camera_id: ID of newly added camera
            light_refinement: Whether to do light (fast) refinement
            
        Returns:
            Updated reconstruction state
        """
        print(f"\nLight refinement after adding {new_camera_id}:")
        
        cameras = reconstruction_state['cameras']
        points_3d = reconstruction_state['points_3d']['points_3d']
        observations = reconstruction_state.get('observations', {})
        
        # 1. Refine only the new camera's pose
        if new_camera_id in cameras and new_camera_id in observations:
            print(f"  Refining pose for {new_camera_id}...")
            
            # Extract observations for new camera
            cam_obs = observations[new_camera_id]
            cam_points_3d = []
            cam_points_2d = []
            
            for obs in cam_obs:
                point_id = obs['point_id']
                if point_id < points_3d.shape[1]:
                    cam_points_3d.append(points_3d[:, point_id])
                    cam_points_2d.append(obs.get('coords_2d', obs.get('image_point')))
            
            if len(cam_points_3d) >= 10:
                result = self.pose_refiner.refine_pose(
                    camera_id=new_camera_id,
                    R_init=cameras[new_camera_id]['R'],
                    t_init=cameras[new_camera_id]['t'],
                    K=cameras[new_camera_id]['K'],
                    points_3d=np.array(cam_points_3d),
                    points_2d=np.array(cam_points_2d)
                )
                
                if result.success:
                    cameras[new_camera_id]['R'] = result.optimized_params['R']
                    cameras[new_camera_id]['t'] = result.optimized_params['t']
        
        # 2. Light structure cleanup (if not light refinement)
        if not light_refinement:
            print("  Refining structure...")
            result = self.structure_refiner.refine_structure(
                points_3d, cameras, observations,
                remove_outliers=True,
                merge_points=False  # Skip merging for speed
            )
            
            if result.success:
                points_3d = result.optimized_params['points_3d']
                observations = result.optimized_params['observations']
        
        # Update reconstruction
        reconstruction_state['cameras'] = cameras
        reconstruction_state['points_3d']['points_3d'] = points_3d
        reconstruction_state['observations'] = observations
        
        return reconstruction_state
    
    def _initialize_intrinsics(self, cameras: Dict, image_info: Dict):
        """Initialize intrinsics learner for all cameras."""
        for cam_id, cam_data in cameras.items():
            if cam_id in image_info:
                self.intrinsics_learner.initialize_intrinsics(
                    cam_id,
                    image_info[cam_id]['size'],
                    cam_data['K']
                )
    
    def _refine_poses_stage(self,
                           cameras: Dict,
                           points_3d: np.ndarray,
                           observations: Dict) -> Dict:
        """Refine all camera poses."""
        print("\n--- Pose Refinement Stage ---")
        
        results = self.pose_refiner.refine_multiple_poses(
            cameras, points_3d, observations
        )
        
        # Update cameras with refined poses
        for cam_id, result in results.items():
            if result.success:
                cameras[cam_id]['R'] = result.optimized_params['R']
                cameras[cam_id]['t'] = result.optimized_params['t']
        
        return cameras
    
    def _refine_intrinsics_stage(self,
                                cameras: Dict,
                                points_3d: np.ndarray,
                                observations: Dict) -> Dict:
        """Refine camera intrinsics progressively."""
        print("\n--- Intrinsics Refinement Stage ---")
        
        for cam_id, cam_data in cameras.items():
            if cam_id not in observations:
                continue
            
            # Extract observations for this camera
            cam_obs = observations[cam_id]
            cam_points_3d = []
            cam_points_2d = []
            
            for obs in cam_obs:
                point_id = obs['point_id']
                if point_id < points_3d.shape[1]:
                    cam_points_3d.append(points_3d[:, point_id])
                    cam_points_2d.append(obs.get('coords_2d', obs.get('image_point')))
            
            if len(cam_points_3d) >= 20:  # Need enough points for intrinsics
                estimate = self.intrinsics_learner.refine_intrinsics(
                    cam_id,
                    cam_data['R'],
                    cam_data['t'],
                    np.array(cam_points_3d),
                    np.array(cam_points_2d)
                )
                
                if estimate is not None:
                    cameras[cam_id]['K'] = estimate.to_matrix()
        
        return cameras
    
    def _refine_structure_stage(self,
                               points_3d: np.ndarray,
                               cameras: Dict,
                               observations: Dict) -> Tuple[np.ndarray, Dict]:
        """Refine 3D structure."""
        print("\n--- Structure Refinement Stage ---")
        
        result = self.structure_refiner.refine_structure(
            points_3d, cameras, observations,
            remove_outliers=True,
            merge_points=True
        )
        
        if result.success:
            return (result.optimized_params['points_3d'],
                   result.optimized_params['observations'])
        
        return points_3d, observations
    
    def _compute_reconstruction_cost(self,
                                    points_3d: np.ndarray,
                                    cameras: Dict,
                                    observations: Dict) -> float:
        """Compute mean reprojection error across reconstruction."""
        import cv2
        
        all_errors = []
        
        for cam_id, cam_obs in observations.items():
            if cam_id not in cameras:
                continue
            
            cam = cameras[cam_id]
            rvec, _ = cv2.Rodrigues(cam['R'])
            tvec = cam['t'].reshape(3, 1)
            
            for obs in cam_obs:
                point_id = obs['point_id']
                if point_id >= points_3d.shape[1]:
                    continue
                
                point_3d = points_3d[:, point_id]
                coords_2d = np.array(obs.get('coords_2d', obs.get('image_point')))
                
                # Project
                projected, _ = cv2.projectPoints(
                    point_3d.reshape(1, 1, 3),
                    rvec, tvec, cam['K'], None
                )
                projected = projected.reshape(2)
                
                error = np.linalg.norm(projected - coords_2d)
                all_errors.append(error)
        
        return float(np.mean(all_errors)) if all_errors else 0.0
    
    def optimize(self, *args, **kwargs) -> OptimizationResult:
        """Optimize - wrapper for refine_reconstruction."""
        return self.refine_reconstruction(*args, **kwargs)
    
    def compute_cost(self, *args, **kwargs) -> float:
        """Compute cost."""
        if 'reconstruction_state' in kwargs:
            state = kwargs['reconstruction_state']
            return self._compute_reconstruction_cost(
                state['points_3d']['points_3d'],
                state['cameras'],
                state.get('observations', {})
            )
        return 0.0