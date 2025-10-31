"""
TSDF Fusion
===========

Truncated Signed Distance Function (TSDF) fusion for depth map integration.
Uses Open3D's modern tensor-based TSDF implementation.

Based on:
    - Curless & Levoy, "A Volumetric Method for Building Complex Models from Range Images", SIGGRAPH 1996
    - KinectFusion: Newcombe et al., ISMAR 2011
"""

import numpy as np
import logging
from typing import Dict, List, Optional
import open3d as o3d


class TSDFFusion:
    """
    TSDF volume fusion for integrating multiple depth maps.

    Fuses depth maps from multiple views into a single 3D volume,
    then extracts a point cloud or mesh.
    """

    def __init__(self, voxel_size: float = 0.01, sdf_trunc: float = 0.04):
        """
        Initialize TSDF fusion.

        Args:
            voxel_size: Size of voxels in meters (default: 1cm)
            sdf_trunc: Truncation distance for SDF in meters (default: 4cm)
        """
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc
        self.logger = logging.getLogger("DenseReconstruction.TSDF")

        self.logger.info(f"TSDF initialized: voxel_size={voxel_size}m, sdf_trunc={sdf_trunc}m")

    def fuse_depth_maps(self, depth_maps: List[Dict],
                       cameras: Dict) -> Optional[o3d.geometry.PointCloud]:
        """
        Fuse multiple depth maps into a single point cloud.

        Args:
            depth_maps: List of depth map dictionaries with keys:
                       - 'image_name': str
                       - 'depth_map': np.ndarray
                       - 'camera': camera data (optional)
            cameras: Dictionary of camera data by image name

        Returns:
            Fused point cloud (Open3D PointCloud)
        """
        self.logger.info(f"Fusing {len(depth_maps)} depth maps...")

        if not depth_maps:
            self.logger.warning("No depth maps to fuse")
            return None

        # Create TSDF volume using modern VoxelBlockGrid
        try:
            volume = o3d.t.geometry.VoxelBlockGrid(
                attr_names=('tsdf', 'weight', 'color'),
                attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
                attr_channels=((1,), (1,), (3,)),
                voxel_size=self.voxel_size,
                block_resolution=16,
                block_count=100000,
                device=o3d.core.Device('CPU:0')
            )

            # Integrate each depth map
            integrated_count = 0

            for i, depth_data in enumerate(depth_maps):
                try:
                    image_name = depth_data['image_name']
                    depth_map = depth_data.get('depth_map')

                    if depth_map is None:
                        continue

                    # Get camera parameters
                    if 'camera' in depth_data:
                        camera = depth_data['camera']
                    elif image_name in cameras:
                        camera = cameras[image_name]
                    else:
                        self.logger.warning(f"No camera data for {image_name}, skipping")
                        continue

                    # Integrate this depth map
                    self._integrate_depth_map(volume, depth_map, camera, image_name)
                    integrated_count += 1

                    if (i + 1) % 10 == 0:
                        self.logger.info(f"  Integrated {i+1}/{len(depth_maps)} depth maps")

                except Exception as e:
                    self.logger.warning(f"Failed to integrate {depth_data.get('image_name', '?')}: {e}")
                    continue

            self.logger.info(f"✓ Successfully integrated {integrated_count}/{len(depth_maps)} depth maps")

            # Extract point cloud from TSDF
            point_cloud = volume.extract_point_cloud()

            # Convert to legacy PointCloud for compatibility
            pcd_legacy = o3d.geometry.PointCloud()
            pcd_legacy.points = o3d.utility.Vector3dVector(point_cloud.point.positions.numpy())

            if point_cloud.point.colors is not None:
                pcd_legacy.colors = o3d.utility.Vector3dVector(point_cloud.point.colors.numpy())

            self.logger.info(f"✓ Extracted {len(pcd_legacy.points)} points from TSDF")

            return pcd_legacy

        except Exception as e:
            self.logger.error(f"TSDF fusion failed: {e}")
            # Fallback: simple depth map to point cloud conversion
            return self._simple_depth_fusion(depth_maps, cameras)

    def _integrate_depth_map(self, volume, depth_map: np.ndarray,
                            camera: Dict, image_name: str):
        """
        Integrate a single depth map into the TSDF volume.

        Args:
            volume: TSDF VoxelBlockGrid
            depth_map: Depth map (H, W)
            camera: Camera parameters (R, t, K)
            image_name: Image identifier
        """
        # Get camera parameters
        R = camera.get('R', np.eye(3))
        t = camera.get('t', np.zeros(3)).reshape(3, 1)
        K = camera.get('K')

        if K is None:
            raise ValueError(f"Camera intrinsics missing for {image_name}")

        # Create extrinsic matrix [R|t]
        extrinsic = np.hstack([R, t])
        extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])  # 4x4

        # Convert intrinsics to Open3D format
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        h, w = depth_map.shape
        intrinsic.set_intrinsics(
            width=w,
            height=h,
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2]
        )

        # Convert depth map to Open3D tensor format
        depth_tensor = o3d.t.geometry.Image(depth_map.astype(np.float32))

        # Integrate
        volume.integrate(
            depth=depth_tensor,
            intrinsics=o3d.core.Tensor(intrinsic.intrinsic_matrix),
            extrinsics=o3d.core.Tensor(np.linalg.inv(extrinsic)),  # Open3D uses world-to-camera
            depth_scale=1.0,
            depth_max=self.sdf_trunc * 3.0
        )

    def _simple_depth_fusion(self, depth_maps: List[Dict],
                            cameras: Dict) -> Optional[o3d.geometry.PointCloud]:
        """
        Simple fallback: convert depth maps to point clouds and merge.

        Used when TSDF fusion fails.
        """
        self.logger.info("Using simple depth fusion fallback...")

        all_points = []
        all_colors = []

        for depth_data in depth_maps:
            try:
                depth_map = depth_data.get('depth_map')
                image_name = depth_data['image_name']

                if depth_map is None:
                    continue

                camera = depth_data.get('camera', cameras.get(image_name))
                if camera is None:
                    continue

                # Convert depth map to 3D points
                points = self._depth_to_points(depth_map, camera)

                if points is not None and len(points) > 0:
                    all_points.append(points)

            except Exception as e:
                self.logger.warning(f"Failed to convert depth map: {e}")
                continue

        if not all_points:
            self.logger.error("No points generated from depth maps")
            return None

        # Merge all points
        merged_points = np.vstack(all_points)

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_points)

        # Remove statistical outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        self.logger.info(f"✓ Simple fusion: {len(pcd.points)} points")

        return pcd

    def _depth_to_points(self, depth_map: np.ndarray, camera: Dict) -> Optional[np.ndarray]:
        """
        Convert a depth map to 3D points in world coordinates.

        Args:
            depth_map: Depth map (H, W)
            camera: Camera parameters

        Returns:
            Points (N, 3)
        """
        h, w = depth_map.shape
        K = camera['K']
        R = camera.get('R', np.eye(3))
        t = camera.get('t', np.zeros(3)).reshape(3, 1)

        # Create pixel grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u = u.flatten()
        v = v.flatten()
        depth = depth_map.flatten()

        # Filter valid depths
        valid = depth > 0
        u = u[valid]
        v = v[valid]
        depth = depth[valid]

        if len(depth) == 0:
            return None

        # Backproject to camera coordinates
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth

        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        # Transform to world coordinates
        points_world = (R.T @ (points_cam.T - t)).T

        return points_world
