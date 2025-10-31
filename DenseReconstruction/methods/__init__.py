"""
Dense Reconstruction Methods
============================

Wrappers for different dense reconstruction algorithms:
- COLMAP PatchMatch MVS (GPU-accelerated)
- OpenMVS (CPU-based)
- Neural depth estimation (MiDaS/ZoeDepth)
"""

from .colmap_mvs import COLMAPDenseReconstruction
from .openmvs_mvs import OpenMVSDenseReconstruction
from .neural_depth import NeuralDepthEstimation

__all__ = [
    'COLMAPDenseReconstruction',
    'OpenMVSDenseReconstruction',
    'NeuralDepthEstimation'
]
