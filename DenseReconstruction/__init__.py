"""
Dense Reconstruction Module
============================

Production-grade dense reconstruction pipeline leveraging established libraries:
- COLMAP PatchMatch MVS (primary, requires CUDA)
- OpenMVS (CPU fallback)
- Open3D TSDF fusion and mesh processing
- Optional neural depth estimation

Architecture:
    methods/     - MVS algorithms (COLMAP, OpenMVS, neural)
    fusion/      - Depth map fusion (TSDF)
    mesh/        - Surface extraction and processing
    io/          - Format conversion and export
    utils/       - Utility functions

Example:
    >>> from DenseReconstruction import DenseReconstructionPipeline
    >>> from CameraPoseEstimation.pipeline import IncrementalReconstructionPipeline
    >>>
    >>> # Get sparse reconstruction
    >>> sparse_pipeline = IncrementalReconstructionPipeline(...)
    >>> sparse_result = sparse_pipeline.run()
    >>>
    >>> # Run dense reconstruction
    >>> dense_pipeline = DenseReconstructionPipeline(method='auto')
    >>> mesh = dense_pipeline.run(
    >>>     sparse_reconstruction=sparse_result,
    >>>     image_folder='./images',
    >>>     output_dir='./dense_output'
    >>> )
"""

from .pipeline import DenseReconstructionPipeline, DenseReconstructionConfig

__version__ = "2.0.0"
__all__ = [
    'DenseReconstructionPipeline',
    'DenseReconstructionConfig'
]
