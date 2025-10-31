"""
OpenMVS Dense Reconstruction Wrapper
====================================

Wraps OpenMVS for CPU-based dense reconstruction.
Requires OpenMVS command-line tools installed.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List
import subprocess
import shutil


class OpenMVSDenseReconstruction:
    """
    OpenMVS Multi-View Stereo wrapper.

    CPU-based dense reconstruction using OpenMVS.
    Requires OpenMVS CLI tools (DensifyPointCloud, ReconstructMesh).

    Installation: https://github.com/cdcseacave/openMVS
    """

    def __init__(self, config):
        """Initialize OpenMVS dense reconstruction"""
        self.config = config
        self.logger = logging.getLogger("DenseReconstruction.OpenMVS")

        # Check for OpenMVS executables
        self.densify_exe = shutil.which('DensifyPointCloud')
        self.reconstruct_exe = shutil.which('ReconstructMesh')

        if not self.densify_exe:
            raise RuntimeError(
                "OpenMVS 'DensifyPointCloud' not found in PATH. "
                "Please install OpenMVS: https://github.com/cdcseacave/openMVS"
            )

    def compute_depth_maps(self, workspace: Dict, output_dir: Path) -> List[Dict]:
        """
        Compute dense depth maps using OpenMVS.

        Args:
            workspace: MVS workspace with OpenMVS scene file
            output_dir: Output directory

        Returns:
            List of depth map dictionaries
        """
        self.logger.info("Running OpenMVS dense reconstruction...")

        mvs_scene = workspace['openmvs_scene']
        output_scene = output_dir / "scene_dense.mvs"

        # Run DensifyPointCloud
        resolution_level = self._get_resolution_level()

        cmd = [
            self.densify_exe,
            str(mvs_scene),
            '-o', str(output_scene),
            '--resolution-level', str(resolution_level),
            '--number-views', '5',
            '--max-resolution', '3200'
        ]

        self.logger.info(f"Running: {' '.join(map(str, cmd))}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"OpenMVS DensifyPointCloud failed: {result.stderr}")

        self.logger.info("✓ OpenMVS densification complete")

        # OpenMVS outputs a dense point cloud, not individual depth maps
        # We'll return a synthetic depth map list
        # The actual dense point cloud is in scene_dense.mvs

        return [{
            'image_name': 'openmvs_dense',
            'depth_map': None,  # OpenMVS doesn't expose individual depth maps easily
            'point_cloud_path': str(output_scene)
        }]

    def _get_resolution_level(self) -> int:
        """Get resolution level based on quality setting"""
        quality_map = {
            'low': 2,
            'medium': 1,
            'high': 0,
            'ultra': 0
        }
        return quality_map.get(self.config.depth_map_quality, 1)
