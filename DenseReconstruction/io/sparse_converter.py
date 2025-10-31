"""
Sparse Reconstruction Converter
================================

Convert sparse reconstruction from CameraPoseEstimation to MVS formats (COLMAP, OpenMVS).
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import shutil


class SparseReconstructionConverter:
    """
    Convert sparse reconstruction to various MVS formats.

    Supports:
    - COLMAP format (for pycolmap or COLMAP CLI)
    - OpenMVS format (for OpenMVS CLI)
    - Generic workspace (for neural methods)
    """

    def __init__(self):
        """Initialize converter"""
        self.logger = logging.getLogger("DenseReconstruction.IO")

    def convert(self,
               sparse_reconstruction,
               image_folder: Union[str, Path],
               output_dir: Union[str, Path],
               method: str = 'colmap',
               image_list: Optional[List[str]] = None) -> Dict:
        """
        Convert sparse reconstruction to MVS format.

        Args:
            sparse_reconstruction: Sparse reconstruction from CameraPoseEstimation
                                  (Reconstruction object or dict)
            image_folder: Path to images
            output_dir: Output directory
            method: Target format ('colmap', 'openmvs', 'neural')
            image_list: Optional list of images to include

        Returns:
            Dictionary with workspace paths and camera data
        """
        self.logger.info(f"Converting sparse reconstruction to {method} format...")

        output_dir = Path(output_dir)
        image_folder = Path(image_folder)

        # Extract data from sparse reconstruction
        cameras, points = self._extract_reconstruction_data(sparse_reconstruction)

        # Filter by image list if provided
        if image_list:
            cameras = {k: v for k, v in cameras.items() if k in image_list}

        self.logger.info(f"Processing {len(cameras)} cameras, {len(points)} points")

        # Convert to target format
        if method == 'colmap':
            workspace = self._convert_to_colmap(cameras, points, image_folder, output_dir)
        elif method == 'openmvs':
            workspace = self._convert_to_openmvs(cameras, points, image_folder, output_dir)
        else:
            # Generic workspace for neural methods
            workspace = {
                'cameras': cameras,
                'points': points,
                'image_folder': str(image_folder),
                'method': method
            }

        return workspace

    def _extract_reconstruction_data(self, reconstruction) -> tuple:
        """
        Extract cameras and points from reconstruction object.

        Handles both Reconstruction objects and dict format.
        """
        cameras = {}
        points = []

        # Check if it's a Reconstruction object or dict
        if hasattr(reconstruction, 'cameras'):
            # Reconstruction object
            for cam_id, camera in reconstruction.cameras.items():
                cameras[cam_id] = {
                    'R': camera.R,
                    't': camera.t,
                    'K': camera.K,
                    'width': camera.width,
                    'height': camera.height
                }

            if hasattr(reconstruction, 'points'):
                for point_id, point in reconstruction.points.items():
                    points.append(point.coords)

        elif isinstance(reconstruction, dict):
            # Dictionary format
            if 'cameras' in reconstruction:
                cameras = reconstruction['cameras']

            if 'points_3d' in reconstruction:
                points_data = reconstruction['points_3d']
                if isinstance(points_data, np.ndarray):
                    if points_data.ndim == 2:
                        points = [points_data[:, i] for i in range(points_data.shape[1])]
                    else:
                        points = list(points_data)
                elif isinstance(points_data, dict):
                    points = [p['coords'] if isinstance(p, dict) else p for p in points_data.values()]

        return cameras, points

    def _convert_to_colmap(self, cameras: Dict, points: List,
                          image_folder: Path, output_dir: Path) -> Dict:
        """
        Convert to COLMAP format.

        COLMAP format structure:
            colmap_workspace/
                images/           -> symlink to image_folder
                sparse/
                    0/
                        cameras.txt
                        images.txt
                        points3D.txt
        """
        colmap_dir = output_dir / "colmap_workspace"
        sparse_dir = colmap_dir / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        # Create symlink to images
        images_link = colmap_dir / "images"
        if not images_link.exists():
            try:
                images_link.symlink_to(image_folder.absolute(), target_is_directory=True)
            except:
                # Fallback: copy images (slower)
                self.logger.warning("Could not create symlink, copying images...")
                shutil.copytree(image_folder, images_link)

        # Write COLMAP text files
        self._write_colmap_cameras(cameras, sparse_dir / "cameras.txt")
        self._write_colmap_images(cameras, sparse_dir / "images.txt")
        self._write_colmap_points(points, sparse_dir / "points3D.txt")

        self.logger.info(f"✓ COLMAP format written to {colmap_dir}")

        return {
            'colmap_workspace': colmap_dir,
            'cameras': cameras,
            'points': points,
            'image_folder': str(image_folder)
        }

    def _write_colmap_cameras(self, cameras: Dict, output_file: Path):
        """Write COLMAP cameras.txt"""
        with open(output_file, 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"# Number of cameras: {len(cameras)}\n")

            for i, (cam_id, cam_data) in enumerate(cameras.items(), 1):
                K = cam_data['K']
                w = cam_data.get('width', 1920)
                h = cam_data.get('height', 1080)

                # SIMPLE_RADIAL model: fx, cx, cy, k
                fx = K[0, 0]
                cx = K[0, 2]
                cy = K[1, 2]

                f.write(f"{i} SIMPLE_RADIAL {w} {h} {fx} {cx} {cy} 0.0\n")

    def _write_colmap_images(self, cameras: Dict, output_file: Path):
        """Write COLMAP images.txt"""
        with open(output_file, 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {len(cameras)}\n")

            for i, (cam_id, cam_data) in enumerate(cameras.items(), 1):
                # Convert R, t to quaternion
                R = cam_data['R']
                t = cam_data['t'].flatten()

                qw, qx, qy, qz = self._rotation_matrix_to_quaternion(R)

                f.write(f"{i} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {i} {cam_id}\n")
                f.write("\n")  # Empty line for POINTS2D

    def _write_colmap_points(self, points: List, output_file: Path):
        """Write COLMAP points3D.txt"""
        with open(output_file, 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write(f"# Number of points: {len(points)}\n")

            for i, point in enumerate(points, 1):
                if isinstance(point, np.ndarray):
                    x, y, z = point
                else:
                    x, y, z = point['coords'] if isinstance(point, dict) else point

                # Default color: white, error: 0, empty track
                f.write(f"{i} {x} {y} {z} 255 255 255 0.0\n")

    def _convert_to_openmvs(self, cameras: Dict, points: List,
                           image_folder: Path, output_dir: Path) -> Dict:
        """
        Convert to OpenMVS format.

        OpenMVS uses a binary .mvs format. We'll use the InterfaceCOLMAP tool
        to convert from COLMAP format.
        """
        self.logger.info("Converting to OpenMVS format via COLMAP...")

        # First convert to COLMAP
        colmap_workspace = self._convert_to_colmap(cameras, points, image_folder, output_dir)

        # Convert COLMAP to OpenMVS using InterfaceCOLMAP
        mvs_file = output_dir / "scene.mvs"

        import shutil
        interface_colmap = shutil.which('InterfaceCOLMAP')

        if interface_colmap:
            import subprocess
            cmd = [
                interface_colmap,
                '-i', str(colmap_workspace['colmap_workspace']),
                '-o', str(mvs_file)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                self.logger.info(f"✓ OpenMVS format written to {mvs_file}")
            else:
                self.logger.warning(f"InterfaceCOLMAP failed: {result.stderr}")
                self.logger.info("Will use COLMAP format as fallback")

        return {
            'openmvs_scene': mvs_file if mvs_file.exists() else None,
            'colmap_workspace': colmap_workspace['colmap_workspace'],  # Fallback
            'cameras': cameras,
            'points': points,
            'image_folder': str(image_folder)
        }

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> tuple:
        """
        Convert rotation matrix to quaternion (w, x, y, z).

        Uses Shepperd's method for numerical stability.
        """
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return (w, x, y, z)
