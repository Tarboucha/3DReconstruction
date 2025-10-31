"""
Geometry Module

Geometric algorithms for 3D reconstruction.

Components:
- Triangulation: 3D point triangulation from multiple views
- Essential Matrix: Relative pose estimation
- Pose Estimation: Camera pose from 3D-2D correspondences (PnP)

Usage:
    from algorithms.geometry import (
        TriangulationEngine,
        EssentialMatrixEstimator,
        PoseEstimator,
    )
"""

# Triangulation
from .triangulation import (
    TriangulationEngine,
    TriangulationConfig,
)

# Essential Matrix
from .essential import   (
    EssentialMatrixEstimator,
    MatrixEstimationConfig,
    )


# Pose Estimation
from .pose import (
    PoseEstimator,
    PoseEstimatorConfig,
    PnPSolver,
    PoseValidator,
)


__all__ = [
    # Triangulation
    'TriangulationEngine',
    'TriangulationConfig',
    
    # Essential Matrix
    'EssentialMatrixEstimator',
    
    # Pose Estimation
    'PoseEstimator',
    'PoseEstimatorConfig',
    'PnPSolver',
    'PoseValidator',
]


__version__ = '1.0.0'