"""
COLMAP Dense Reconstruction Wrapper
====================================

Wraps COLMAP's PatchMatch MVS for dense depth estimation.
Requires pycolmap and CUDA.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import shutil

try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False


class COLMAPDenseReconstruction:
    """
    COLMAP PatchMatch Multi-View Stereo wrapper.

    Uses COLMAP's state-of-the-art PatchMatch algorithm for dense depth estimation.
    Requires CUDA for GPU acceleration.

    Reference:
        Schönberger, J. L., Zheng, E., Frahm, J. M., & Pollefeys, M. (2016).
        Pixelwise view selection for unstructured multi-view stereo.
        ECCV 2016.
    """

    def __init__(self, config):
        """
        Initialize COLMAP dense reconstruction.

        Args:
            config: DenseReconstructionConfig object
        """
        if not HAS_PYCOLMAP:
            raise ImportError(
                "pycolmap is required for COLMAP dense reconstruction. "
                "Install with: pip install pycolmap"
            )

        self.config = config
        self.logger = logging.getLogger("DenseReconstruction.COLMAP")

        # Check for COLMAP CLI (fallback if pycolmap doesn't have CUDA)
        self.has_colmap_cli = shutil.which('colmap') is not None

    def compute_depth_maps(self, workspace: Dict, output_dir: Path) -> List[Dict]:
        """
        Compute dense depth maps using COLMAP PatchMatch.

        Args:
            workspace: MVS workspace with cameras and images
            output_dir: Output directory for depth maps

        Returns:
            List of depth map dictionaries with keys:
                - 'image_name': str
                - 'depth_map': np.ndarray (H, W)
                - 'normal_map': np.ndarray (H, W, 3)
                - 'confidence_map': np.ndarray (H, W)
        """
        self.logger.info("Running COLMAP PatchMatch MVS...")

        colmap_workspace = workspace['colmap_workspace']

        # Check if we should use pycolmap or CLI
        use_cli = self.has_colmap_cli and not self._has_cuda_pycolmap()

        if use_cli:
            self.logger.info("Using COLMAP CLI (pycolmap CUDA not available)")
            return self._compute_depth_maps_cli(colmap_workspace, output_dir)
        else:
            self.logger.info("Using pycolmap API")
            return self._compute_depth_maps_api(colmap_workspace, output_dir)

    def _has_cuda_pycolmap(self) -> bool:
        """Check if pycolmap was built with CUDA support"""
        try:
            # Try to check CUDA availability
            # Note: This is a heuristic, actual check may vary
            return hasattr(pycolmap, 'patch_match_stereo')
        except:
            return False

    def _compute_depth_maps_api(self, colmap_workspace: Path, output_dir: Path) -> List[Dict]:
        """Compute depth maps using pycolmap API"""
        try:
            # Undistort images first
            mvs_path = output_dir / "colmap_mvs"
            mvs_path.mkdir(parents=True, exist_ok=True)

            sparse_path = colmap_workspace / "sparse"
            image_path = colmap_workspace / "images"

            self.logger.info("Undistorting images...")
            pycolmap.undistort_images(
                output_path=str(mvs_path),
                input_path=str(sparse_path / "0"),
                image_path=str(image_path)
            )

            # Run patch match stereo
            self.logger.info("Running PatchMatch stereo...")
            pycolmap.patch_match_stereo(workspace_path=str(mvs_path))

            # Run stereo fusion
            self.logger.info("Fusing depth maps...")
            dense_ply = mvs_path / "dense.ply"
            pycolmap.stereo_fusion(
                output_path=str(dense_ply),
                workspace_path=str(mvs_path)
            )

            # Read depth maps
            depth_maps = self._read_colmap_depth_maps(mvs_path / "stereo" / "depth_maps")

            self.logger.info(f"✓ Computed {len(depth_maps)} depth maps")
            return depth_maps

        except Exception as e:
            self.logger.error(f"PatchMatch failed with pycolmap API: {e}")
            # Try CLI fallback
            if self.has_colmap_cli:
                self.logger.info("Falling back to COLMAP CLI...")
                return self._compute_depth_maps_cli(colmap_workspace, output_dir)
            else:
                raise

    def _compute_depth_maps_cli(self, colmap_workspace: Path, output_dir: Path) -> List[Dict]:
        """Compute depth maps using COLMAP command-line interface"""
        mvs_path = output_dir / "colmap_mvs"
        mvs_path.mkdir(parents=True, exist_ok=True)

        sparse_path = colmap_workspace / "sparse"
        image_path = colmap_workspace / "images"

        # 1. Undistort images
        self.logger.info("Undistorting images (CLI)...")
        cmd = [
            'colmap', 'image_undistorter',
            '--image_path', str(image_path),
            '--input_path', str(sparse_path / "0"),
            '--output_path', str(mvs_path),
            '--output_type', 'COLMAP'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"COLMAP undistortion failed: {result.stderr}")

        # 2. Patch match stereo
        self.logger.info("Running PatchMatch stereo (CLI)...")

        # Set quality parameters based on config
        quality_params = self._get_quality_params()

        cmd = [
            'colmap', 'patch_match_stereo',
            '--workspace_path', str(mvs_path),
            '--workspace_format', 'COLMAP',
            '--PatchMatchStereo.geom_consistency', 'true',
            '--PatchMatchStereo.window_radius', str(quality_params['window_radius']),
            '--PatchMatchStereo.num_samples', str(quality_params['num_samples']),
            '--PatchMatchStereo.num_iterations', str(quality_params['num_iterations']),
            '--PatchMatchStereo.filter', 'true',
            '--PatchMatchStereo.filter_min_ncc', '0.1'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"COLMAP PatchMatch failed: {result.stderr}")

        # 3. Stereo fusion
        self.logger.info("Fusing depth maps (CLI)...")
        dense_ply = mvs_path / "dense.ply"

        cmd = [
            'colmap', 'stereo_fusion',
            '--workspace_path', str(mvs_path),
            '--workspace_format', 'COLMAP',
            '--input_type', 'geometric',
            '--output_path', str(dense_ply)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.logger.warning(f"COLMAP fusion failed: {result.stderr}")

        # Read depth maps
        depth_maps = self._read_colmap_depth_maps(mvs_path / "stereo" / "depth_maps")

        self.logger.info(f"✓ Computed {len(depth_maps)} depth maps using CLI")
        return depth_maps

    def _get_quality_params(self) -> Dict:
        """Get quality parameters based on config"""
        quality_presets = {
            'low': {
                'window_radius': 3,
                'num_samples': 7,
                'num_iterations': 3
            },
            'medium': {
                'window_radius': 5,
                'num_samples': 15,
                'num_iterations': 5
            },
            'high': {
                'window_radius': 7,
                'num_samples': 15,
                'num_iterations': 7
            },
            'ultra': {
                'window_radius': 9,
                'num_samples': 15,
                'num_iterations': 10
            }
        }

        return quality_presets.get(self.config.depth_map_quality, quality_presets['medium'])

    def _read_colmap_depth_maps(self, depth_map_dir: Path) -> List[Dict]:
        """
        Read COLMAP depth maps from directory.

        COLMAP stores depth maps in a binary format (.bin or .geometric.bin files).
        """
        depth_maps = []

        if not depth_map_dir.exists():
            self.logger.warning(f"Depth map directory not found: {depth_map_dir}")
            return depth_maps

        # Look for .geometric.bin or .photometric.bin files
        depth_files = list(depth_map_dir.glob("*.geometric.bin"))
        if not depth_files:
            depth_files = list(depth_map_dir.glob("*.photometric.bin"))

        self.logger.info(f"Found {len(depth_files)} depth map files")

        for depth_file in depth_files:
            try:
                # Extract image name from depth map filename
                # Format: image_name.jpg.geometric.bin
                image_name = depth_file.name.replace('.geometric.bin', '').replace('.photometric.bin', '')

                # Read binary depth map using COLMAP format
                depth_map = self._read_colmap_depth_bin(depth_file)

                if depth_map is not None:
                    depth_maps.append({
                        'image_name': image_name,
                        'depth_map': depth_map,
                        'normal_map': None,  # Would need to read normal map separately
                        'confidence_map': None  # Would need to read confidence map separately
                    })

            except Exception as e:
                self.logger.warning(f"Failed to read depth map {depth_file}: {e}")
                continue

        return depth_maps

    def _read_colmap_depth_bin(self, filepath: Path) -> Optional[np.ndarray]:
        """
        Read COLMAP binary depth map format.

        Format:
            - width (int32)
            - height (int32)
            - num_channels (int32) - typically 1 for depth
            - data (float32 array of size width * height * num_channels)
        """
        try:
            with open(filepath, 'rb') as f:
                # Read header
                width = np.fromfile(f, dtype=np.int32, count=1)[0]
                height = np.fromfile(f, dtype=np.int32, count=1)[0]
                channels = np.fromfile(f, dtype=np.int32, count=1)[0]

                # Read data
                num_elements = width * height * channels
                data = np.fromfile(f, dtype=np.float32, count=num_elements)

                # Reshape
                if channels == 1:
                    depth_map = data.reshape((height, width))
                else:
                    depth_map = data.reshape((height, width, channels))

                return depth_map

        except Exception as e:
            self.logger.error(f"Failed to read binary depth map: {e}")
            return None
