"""
Essential Matrix Estimation Module

Estimates the essential matrix from 2D-2D point correspondences for
relative camera pose estimation.

Components:
- EssentialMatrixEstimator: Main estimator for essential matrix estimation
- MatrixEstimationConfig: Configuration for estimation parameters

Usage:
    from algorithms.geometry.essential import EssentialMatrixEstimator
    
    estimator = EssentialMatrixEstimator()
    result = estimator.estimate(pts1, pts2, camera_matrix)
"""

from .essential_estimation import (
    EssentialMatrixEstimator,
    MatrixEstimationConfig,
)


__all__ = [
    'EssentialMatrixEstimator',
    'MatrixEstimationConfig',
]


__version__ = '1.0.0'