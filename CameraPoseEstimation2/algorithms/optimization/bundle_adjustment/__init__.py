"""
Bundle Adjustment Module

Comprehensive bundle adjustment implementations for 3D reconstruction optimization.

Components:
- Two-View BA: Initial two-view optimization
- Incremental BA: After each new view is added
- Global BA: Final optimization of entire reconstruction
- Cost Functions: Reprojection error computation and robust loss

Usage:
    # Two-view bundle adjustment
    from algorithms.optimization.bundle_adjustment import TwoViewBundleAdjustment
    
    adjuster = TwoViewBundleAdjustment()
    result = adjuster.optimize(cameras, points_3d, observations)
    
    # Incremental bundle adjustment
    from algorithms.optimization.bundle_adjustment import IncrementalBundleAdjustment
    
    adjuster = IncrementalBundleAdjustment()
    result = adjuster.optimize(cameras, points_3d, observations, new_camera_id='img_005.jpg')
    
    # Global bundle adjustment
    from algorithms.optimization.bundle_adjustment import GlobalBundleAdjustment
    
    adjuster = GlobalBundleAdjustment()
    result = adjuster.optimize(cameras, points_3d, observations)
    
    # Convenience functions
    from algorithms.optimization.bundle_adjustment import (
        adjust_two_view,
        adjust_after_new_camera,
        adjust_global
    )
    
    # With reconstruction state
    reconstruction_state = adjust_after_new_camera(
        reconstruction_state, 'new_image.jpg'
    )
"""

# Main classes
from .two_view import (
    TwoViewBundleAdjustment,
    TwoViewBundleAdjustmentConfig,
    adjust_two_view
)

from .incremental import (
    IncrementalBundleAdjustment,
    IncrementalBundleAdjustmentConfig,
    adjust_after_new_camera
)

from .global_ba import (
    GlobalBundleAdjustment,
    GlobalBundleAdjustmentConfig,
    adjust_global
)

# Cost functions and utilities
from .cost_functions import (
    BACostFunction,
    ParameterBuilder,
    LossFunction,
    compute_reprojection_error,
    compute_mean_reprojection_error
)


__all__ = [
    # Two-view BA
    'TwoViewBundleAdjustment',
    'TwoViewBundleAdjustmentConfig',
    'adjust_two_view',
    
    # Incremental BA
    'IncrementalBundleAdjustment',
    'IncrementalBundleAdjustmentConfig',
    'adjust_after_new_camera',
    
    # Global BA
    'GlobalBundleAdjustment',
    'GlobalBundleAdjustmentConfig',
    'adjust_global',
    
    # Cost functions
    'BACostFunction',
    'ParameterBuilder',
    'LossFunction',
    'compute_reprojection_error',
    'compute_mean_reprojection_error',
]


__version__ = '1.0.0'