"""
Camera Pose Estimation Module

This module provides comprehensive camera pose estimation from 2D-3D correspondences.

Components:
- PoseEstimator: Main pose estimator with multiple PnP algorithms
- PnPSolver: High-level solver with reconstruction integration
- PoseValidator: Pose validation and quality assessment utilities
- validators: Additional validation functions

Key Features:
- Multiple PnP methods (P3P, EPnP, ITERATIVE, SQPNP)
- Robust RANSAC estimation
- Pose validation and refinement
- Seamless integration with reconstruction pipeline
- Quality metrics and assessment

Usage:
    # Basic pose estimation
    from algorithms.geometry.pose import PoseEstimator
    
    estimator = PoseEstimator(method='EPNP')
    result = estimator.estimate(points_3d, points_2d, camera_matrix)
    
    # High-level PnP solver
    from algorithms.geometry.pose import PnPSolver
    
    solver = PnPSolver()
    result = solver.solve_pnp(points_3d, points_2d, camera_matrix)
    
    # With validation
    from algorithms.geometry.pose import PoseValidator
    
    validator = PoseValidator()
    validation = validator.validate_pose(R, t, points_3d, points_2d, camera_matrix)
    
    # Convenience functions
    from algorithms.geometry.pose import estimate_pose, solve_pnp_ransac
    
    result = estimate_pose(points_3d, points_2d, camera_matrix, method='EPNP')

Examples:
    # Example 1: Basic pose estimation
    >>> from algorithms.geometry.pose import PoseEstimator
    >>> import numpy as np
    >>> 
    >>> # Your 3D-2D correspondences
    >>> points_3d = np.random.rand(20, 3) * 10
    >>> points_2d = np.random.rand(20, 2) * 1000
    >>> K = np.eye(3)
    >>> K[0, 0] = K[1, 1] = 1000  # focal length
    >>> K[0, 2] = 500  # cx
    >>> K[1, 2] = 500  # cy
    >>> 
    >>> estimator = PoseEstimator(method='EPNP')
    >>> result = estimator.estimate(points_3d, points_2d, K)
    >>> 
    >>> if result.success:
    ...     print(f"Rotation:\\n{result.data['R']}")
    ...     print(f"Translation: {result.data['t'].T}")
    ...     print(f"Inliers: {result.data['inlier_count']}/{len(points_3d)}")
    ...     print(f"Reprojection error: {result.data['reprojection_error']:.2f}px")
    
    # Example 2: Adding camera to reconstruction
    >>> from algorithms.geometry.pose import PnPSolver
    >>> 
    >>> solver = PnPSolver()
    >>> result = solver.add_camera_to_reconstruction(
    ...     new_image_id='image_010.jpg',
    ...     reconstruction_state=reconstruction,
    ...     feature_matches=matches,
    ...     camera_matrix=K
    ... )
    >>> 
    >>> if result['success']:
    ...     print(f"Camera added successfully")
    ...     print(f"Observations: {result['num_observations']}")
    ...     print(f"Quality score: {result['validation']['quality_score']:.2f}")
    
    # Example 3: Pose validation
    >>> from algorithms.geometry.pose import PoseValidator
    >>> 
    >>> validator = PoseValidator(
    ...     max_reprojection_error=5.0,
    ...     min_inliers=15,
    ...     min_inlier_ratio=0.4
    ... )
    >>> 
    >>> validation = validator.validate_pose(
    ...     R=result.data['R'],
    ...     t=result.data['t'],
    ...     points_3d=points_3d,
    ...     points_2d=points_2d,
    ...     camera_matrix=K,
    ...     inlier_mask=result.data['inliers']
    ... )
    >>> 
    >>> print(f"Valid: {validation.is_valid}")
    >>> print(f"Quality: {validation.quality_score:.2f}")
    >>> if validation.warnings:
    ...     print(f"Warnings: {validation.warnings}")
    >>> if validation.errors:
    ...     print(f"Errors: {validation.errors}")

Integration:
    The pose estimation module integrates seamlessly with other pipeline components:
    
    1. **After Essential Matrix**: Use PnP when you have 3D points from triangulation
    2. **Incremental Reconstruction**: Add new cameras one at a time using PnPSolver
    3. **Before Bundle Adjustment**: Pose estimates are refined by BA
    4. **Quality Control**: Use validators to filter poor pose estimates
"""

# Main classes
from .pose_estimator import (
    PoseEstimator,
    PoseEstimatorConfig,
    PnPMethod,
    estimate_pose
)

from .pnp_solver import (
    PnPSolver,
    solve_pnp_ransac
)

from .validators import (
    PoseValidator,
    PoseValidationResult,
    validate_correspondences,
    check_pose_consistency
)


__all__ = [
    # Main estimator
    'PoseEstimator',
    'PoseEstimatorConfig',
    'PnPMethod',
    
    # High-level solver
    'PnPSolver',
    
    # Validators
    'PoseValidator',
    'PoseValidationResult',
    
    # Convenience functions
    'estimate_pose',
    'solve_pnp_ransac',
    'validate_correspondences',
    'check_pose_consistency',
]


__version__ = '1.0.0'


# Module metadata
__author__ = 'Camera Pose Estimation Team'
__description__ = 'Camera pose estimation from 2D-3D correspondences using PnP algorithms'