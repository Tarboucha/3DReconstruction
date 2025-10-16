"""
Algorithms Module

Core geometric and optimization algorithms for 3D reconstruction.

Submodules:
- geometry: Geometric algorithms (triangulation, essential matrix, pose estimation)
- optimization: Optimization algorithms (bundle adjustment, refinement)
- selection: Selection strategies (pair selection, next view selection)

Usage:
    from algorithms.geometry import TriangulationEngine, EssentialMatrixEstimator, PoseEstimator
    from algorithms.optimization import TwoViewBundleAdjustment, GlobalBundleAdjustment
    from algorithms.selection import InitializationPairSelector
"""

# Geometry algorithms
from CameraPoseEstimation2.algorithms.geometry import (
    # Triangulation
    TriangulationEngine,
    TriangulationConfig,
    
    # Essential Matrix
    EssentialMatrixEstimator,
    MatrixEstimationConfig,
    
    # Pose Estimation
    PoseEstimator,
    PoseEstimatorConfig,
    PnPSolver,
    PoseValidator,
)

# Optimization algorithms
from CameraPoseEstimation2.algorithms.optimization import (
    # Bundle Adjustment
    TwoViewBundleAdjustment,
    IncrementalBundleAdjustment,
    GlobalBundleAdjustment,
    
    # BA Configs
    TwoViewBundleAdjustmentConfig,
    IncrementalBundleAdjustmentConfig,
    GlobalBundleAdjustmentConfig,
    
    # BA Functions
    adjust_two_view,
    adjust_after_new_camera,
    adjust_global,
    
    # Cost functions
    BACostFunction,
    ParameterBuilder,
    LossFunction,
    compute_reprojection_error,
    compute_mean_reprojection_error,
    
    # Refinement
    PoseRefiner,
    ProgressiveIntrinsicsLearner,
    StructureRefiner,
    EssentialMatrixRefiner,
    ProgressiveRefinementPipeline,
    
    # Refinement Configs
    PoseRefinerConfig,
    ProgressiveIntrinsicsLearnerConfig,
    StructureRefinerConfig,
    EssentialMatrixRefinerConfig,
    ProgressiveRefinementConfig,
    
    # Refinement Data Classes
    IntrinsicsEstimate,
    StructureQualityMetrics,
    
    # Refinement Functions
    refine_camera_pose,
)

# Selection algorithms
from CameraPoseEstimation2.algorithms.selection import (
    InitializationPairSelector,
    PairSelectionConfig,
    PairScorer,
    PairQualityMetrics,
)


__all__ = [
    # Geometry - Triangulation
    'TriangulationEngine',
    'TriangulationConfig',
    
    # Geometry - Essential Matrix
    'EssentialMatrixEstimator',
    'MatrixEstimationConfig',
    
    # Geometry - Pose Estimation
    'PoseEstimator',
    'PoseEstimatorConfig',
    'PnPSolver',
    'PoseValidator',
    
    # Optimization - Bundle Adjustment Classes
    'TwoViewBundleAdjustment',
    'IncrementalBundleAdjustment',
    'GlobalBundleAdjustment',
    
    # Optimization - BA Configs
    'TwoViewBundleAdjustmentConfig',
    'IncrementalBundleAdjustmentConfig',
    'GlobalBundleAdjustmentConfig',
    
    # Optimization - BA Functions
    'adjust_two_view',
    'adjust_after_new_camera',
    'adjust_global',
    
    # Optimization - Cost Functions
    'BACostFunction',
    'ParameterBuilder',
    'LossFunction',
    'compute_reprojection_error',
    'compute_mean_reprojection_error',
    
    # Optimization - Refinement Classes
    'PoseRefiner',
    'ProgressiveIntrinsicsLearner',
    'StructureRefiner',
    'ProgressiveRefinementPipeline',
    
    # Optimization - Refinement Configs
    'PoseRefinerConfig',
    'ProgressiveIntrinsicsLearnerConfig',
    'StructureRefinerConfig',
    'ProgressiveRefinementConfig',
    
    # Optimization - Refinement Data
    'IntrinsicsEstimate',
    'StructureQualityMetrics',
    
    # Optimization - Refinement Functions
    'refine_camera_pose',
    
    # Selection
    'InitializationPairSelector',
    'PairSelectionConfig',
    'PairScorer',
    'PairQualityMetrics',
]


__version__ = '1.0.0'