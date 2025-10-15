"""
Optimization Module

Algorithms for refining reconstruction through iterative optimization.

Submodules:
- bundle_adjustment: Bundle adjustment for camera poses and 3D structure
- refinement: Pose and structure refinement utilities

Key Features:
- Two-view, incremental, and global bundle adjustment
- Per-camera intrinsic optimization
- Robust loss functions (Huber, Cauchy, etc.)
- Efficient sparse optimization
- Integration with reconstruction pipeline

Usage:
    # Bundle adjustment
    from algorithms.optimization import (
        TwoViewBundleAdjustment,
        IncrementalBundleAdjustment,
        GlobalBundleAdjustment
    )
    
    # Two-view optimization
    two_view_ba = TwoViewBundleAdjustment()
    result = two_view_ba.optimize(cameras, points_3d, observations)
    
    # Incremental optimization (after adding camera)
    incremental_ba = IncrementalBundleAdjustment()
    result = incremental_ba.optimize(
        cameras, points_3d, observations,
        new_camera_id='image_010.jpg'
    )
    
    # Global optimization (final refinement)
    global_ba = GlobalBundleAdjustment()
    result = global_ba.optimize(cameras, points_3d, observations)
    
    # Convenience functions with reconstruction state
    from algorithms.optimization import (
        adjust_two_view,
        adjust_after_new_camera,
        adjust_global
    )
    
    # Update reconstruction state directly
    reconstruction = adjust_after_new_camera(
        reconstruction_state=reconstruction,
        new_camera_id='image_010.jpg'
    )

Integration Pattern:
    ```
    Essential Matrix → Triangulation → [Two-View BA]
                                              ↓
                                        Initial Cameras
                                              ↓
                  ┌───────────────────────────┘
                  ↓
        For Each New Camera:
            PnP → Add Camera → Triangulate → [Incremental BA]
                  ↓
        All Cameras Added → [Global BA] → Export
    ```

Examples:
    # Example 1: Two-view initialization
    >>> from algorithms.optimization import adjust_two_view
    >>> 
    >>> # After essential matrix and triangulation
    >>> result = adjust_two_view(
    ...     cameras={'img1.jpg': cam1, 'img2.jpg': cam2},
    ...     points_3d=initial_points,
    ...     observations=initial_obs,
    ...     optimize_intrinsics=True,
    ...     fix_first_camera=True
    ... )
    >>> 
    >>> if result['success']:
    ...     optimized_cameras = result['cameras']
    ...     optimized_points = result['points_3d']
    
    # Example 2: Incremental BA during reconstruction
    >>> from algorithms.optimization import adjust_after_new_camera
    >>> 
    >>> # After adding each new camera
    >>> for new_image in remaining_images:
    ...     # Add camera via PnP
    ...     reconstruction = add_camera(new_image, reconstruction)
    ...     
    ...     # Triangulate new points
    ...     reconstruction = triangulate_new_points(reconstruction)
    ...     
    ...     # Optimize recent cameras + all points
    ...     reconstruction = adjust_after_new_camera(
    ...         reconstruction_state=reconstruction,
    ...         new_camera_id=new_image,
    ...         num_recent_cameras=3  # Optimize last 3 cameras
    ...     )
    
    # Example 3: Final global refinement
    >>> from algorithms.optimization import adjust_global
    >>> 
    >>> # After all cameras added
    >>> final_reconstruction = adjust_global(
    ...     reconstruction_state=reconstruction,
    ...     optimize_intrinsics=True,
    ...     fix_first_camera=True
    ... )
    >>> 
    >>> print(f"Final reprojection error: {final_reconstruction['optimization_history'][-1]['final_error']:.2f}px")
"""

# Bundle adjustment
from .bundle_adjustment import (
    # Classes
    TwoViewBundleAdjustment,
    IncrementalBundleAdjustment,
    GlobalBundleAdjustment,
    
    # Config
    TwoViewBundleAdjustmentConfig,
    IncrementalBundleAdjustmentConfig,
    GlobalBundleAdjustmentConfig,
    
    # Convenience functions
    adjust_two_view,
    adjust_after_new_camera,
    adjust_global,
    
    # Cost functions
    BACostFunction,
    ParameterBuilder,
    LossFunction,
    compute_reprojection_error,
    compute_mean_reprojection_error
)

# Refinement
from .refinement import (
    # Classes
    PoseRefiner,
    ProgressiveIntrinsicsLearner,
    StructureRefiner,
    ProgressiveRefinementPipeline,
    
    # Config
    PoseRefinerConfig,
    ProgressiveIntrinsicsLearnerConfig,
    StructureRefinerConfig,
    ProgressiveRefinementConfig,
    
    # Data classes
    IntrinsicsEstimate,
    StructureQualityMetrics,
    
    # Convenience functions
    refine_camera_pose
)


__all__ = [
    # Bundle Adjustment Classes
    'TwoViewBundleAdjustment',
    'IncrementalBundleAdjustment',
    'GlobalBundleAdjustment',
    
    # Bundle Adjustment Configuration
    'TwoViewBundleAdjustmentConfig',
    'IncrementalBundleAdjustmentConfig',
    'GlobalBundleAdjustmentConfig',
    
    # Bundle Adjustment Functions
    'adjust_two_view',
    'adjust_after_new_camera',
    'adjust_global',
    
    # Cost Functions & Utilities
    'BACostFunction',
    'ParameterBuilder',
    'LossFunction',
    'compute_reprojection_error',
    'compute_mean_reprojection_error',
    
    # Refinement Classes
    'PoseRefiner',
    'ProgressiveIntrinsicsLearner',
    'StructureRefiner',
    'ProgressiveRefinementPipeline',
    
    # Refinement Configuration
    'PoseRefinerConfig',
    'ProgressiveIntrinsicsLearnerConfig',
    'StructureRefinerConfig',
    'ProgressiveRefinementConfig',
    
    # Refinement Data Classes
    'IntrinsicsEstimate',
    'StructureQualityMetrics',
    
    # Refinement Functions
    'refine_camera_pose',
]


__version__ = '1.0.0'


# Module metadata
__author__ = 'Bundle Adjustment Team'
__description__ = 'Bundle adjustment algorithms for 3D reconstruction optimization'