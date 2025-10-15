"""
Refinement Module

Iterative refinement algorithms for improving reconstruction quality.

Components:
- Pose Refiner: Refine camera poses individually
- Intrinsics Learner: Progressive intrinsics estimation
- Structure Refiner: Clean and improve 3D point cloud
- Progressive Pipeline: Multi-stage refinement workflow

Features:
- Independent pose refinement
- Progressive intrinsics learning with uncertainty tracking
- Structure cleanup and outlier removal
- Multi-stage iterative refinement pipeline
- Convergence tracking

Usage:
    # Refine a single camera pose
    from algorithms.optimization.refinement import PoseRefiner
    
    refiner = PoseRefiner()
    result = refiner.refine_pose(
        camera_id='img_005.jpg',
        R_init=R, t_init=t, K=K,
        points_3d=points_3d,
        points_2d=points_2d
    )
    
    # Progressive intrinsics learning
    from algorithms.optimization.refinement import ProgressiveIntrinsicsLearner
    
    learner = ProgressiveIntrinsicsLearner()
    learner.initialize_intrinsics('img_005.jpg', (1920, 1080))
    
    # Refine with new observations
    estimate = learner.refine_intrinsics(
        'img_005.jpg', R, t, points_3d, points_2d
    )
    
    # Refine structure
    from algorithms.optimization.refinement import StructureRefiner
    
    refiner = StructureRefiner()
    result = refiner.refine_structure(
        points_3d, cameras, observations
    )
    
    # Full progressive refinement pipeline
    from algorithms.optimization.refinement import ProgressiveRefinementPipeline
    
    pipeline = ProgressiveRefinementPipeline()
    result = pipeline.refine_reconstruction(reconstruction_state)

Examples:
    # Example 1: Refine pose after PnP
    >>> from algorithms.optimization.refinement import refine_camera_pose
    >>> 
    >>> result = refine_camera_pose(
    ...     camera_id='img_010.jpg',
    ...     R=R_from_pnp, t=t_from_pnp, K=K,
    ...     points_3d=visible_points_3d,
    ...     points_2d=observed_points_2d
    ... )
    >>> 
    >>> if result['success']:
    ...     R_refined = result['R']
    ...     t_refined = result['t']
    ...     print(f"Improvement: {result['improvement']:.2f}px")
    
    # Example 2: Progressive intrinsics learning
    >>> from algorithms.optimization.refinement import ProgressiveIntrinsicsLearner
    >>> 
    >>> learner = ProgressiveIntrinsicsLearner()
    >>> 
    >>> # Initialize for each camera
    >>> for cam_id, cam_data in cameras.items():
    ...     learner.initialize_intrinsics(
    ...         cam_id, 
    ...         image_info[cam_id]['size'],
    ...         cam_data['K']
    ...     )
    >>> 
    >>> # Refine during reconstruction
    >>> for cam_id in cameras:
    ...     estimate = learner.refine_intrinsics(
    ...         cam_id, R, t, points_3d, points_2d
    ...     )
    ...     cameras[cam_id]['K'] = estimate.to_matrix()
    
    # Example 3: Structure cleanup
    >>> from algorithms.optimization.refinement import StructureRefiner
    >>> 
    >>> refiner = StructureRefiner(
    ...     max_reprojection_error=3.0,
    ...     min_track_length=2,
    ...     enable_merging=True
    ... )
    >>> 
    >>> result = refiner.refine_structure(
    ...     points_3d, cameras, observations
    ... )
    >>> 
    >>> print(f"Points: {result.metadata['initial_points']} → "
    ...       f"{result.metadata['final_points']}")
    >>> print(f"Removed: {result.metadata['points_removed']}")
    
    # Example 4: Full progressive refinement
    >>> from algorithms.optimization.refinement import ProgressiveRefinementPipeline
    >>> 
    >>> pipeline = ProgressiveRefinementPipeline(
    ...     max_iterations=5,
    ...     enable_pose_refinement=True,
    ...     enable_intrinsics_learning=True,
    ...     enable_structure_refinement=True
    ... )
    >>> 
    >>> result = pipeline.refine_reconstruction(reconstruction_state)
    >>> 
    >>> print(f"Converged: {result.metadata['converged']}")
    >>> print(f"Final cost: {result.final_cost:.2f}px")
    >>> 
    >>> # Check history
    >>> for entry in result.metadata['history']:
    ...     print(f"  Iteration {entry['iteration']}: "
    ...           f"cost={entry['cost']:.2f}, "
    ...           f"improvement={entry['improvement']:.2f}")

Integration Pattern:
    ```
    Essential Matrix → Triangulation → Two-View BA
                                            ↓
                                    [Pose Refinement]  ← Light refinement
                                            ↓
                            For Each New Camera:
                                PnP → Add Camera
                                    ↓
                            [Pose Refinement]  ← Refine new pose
                                    ↓
                            Triangulate New Points
                                    ↓
                            [Structure Refinement]  ← Clean structure
                                    ↓
                            [Intrinsics Update]  ← Progressive learning
                                    ↓
                            Incremental BA
                                    ↓
                        All Cameras Added
                                    ↓
                    [Progressive Refinement]  ← Multi-stage
                                    ↓
                            Global BA
    ```
"""

# Main classes
from .pose_refiner import (
    PoseRefiner,
    PoseRefinerConfig,
    refine_camera_pose
)

from .intrinsics_refiner import (
    ProgressiveIntrinsicsLearner,
    ProgressiveIntrinsicsLearnerConfig,
    IntrinsicsEstimate
)

from .structure_refiner import (
    StructureRefiner,
    StructureRefinerConfig,
    StructureQualityMetrics
)

from .progressive_pipeline import (
    ProgressiveRefinementPipeline,
    ProgressiveRefinementConfig
)


__all__ = [
    # Pose refinement
    'PoseRefiner',
    'PoseRefinerConfig',
    'refine_camera_pose',
    
    # Intrinsics learning
    'ProgressiveIntrinsicsLearner',
    'ProgressiveIntrinsicsLearnerConfig',
    'IntrinsicsEstimate',
    
    # Structure refinement
    'StructureRefiner',
    'StructureRefinerConfig',
    'StructureQualityMetrics',
    
    # Progressive pipeline
    'ProgressiveRefinementPipeline',
    'ProgressiveRefinementConfig',
]


__version__ = '1.0.0'


# Module metadata
__author__ = 'Refinement Module Team'
__description__ = 'Iterative refinement algorithms for 3D reconstruction'