"""
Mesh Processing
===============

Surface extraction and mesh post-processing using Open3D.
"""

import numpy as np
import logging
from typing import Optional, Literal
import open3d as o3d


class MeshProcessor:
    """
    Mesh extraction and post-processing.

    Provides surface reconstruction from point clouds and various
    mesh improvement operations.
    """

    def __init__(self):
        """Initialize mesh processor"""
        self.logger = logging.getLogger("DenseReconstruction.Mesh")

    def extract_mesh(self,
                    point_cloud: o3d.geometry.PointCloud,
                    method: Literal['marching_cubes', 'poisson', 'ball_pivoting'] = 'marching_cubes') -> Optional[o3d.geometry.TriangleMesh]:
        """
        Extract mesh from point cloud.

        Args:
            point_cloud: Input point cloud
            method: Surface reconstruction method

        Returns:
            Triangle mesh
        """
        self.logger.info(f"Extracting mesh using {method}...")

        if len(point_cloud.points) == 0:
            self.logger.error("Point cloud is empty")
            return None

        try:
            if method == 'poisson':
                return self._poisson_reconstruction(point_cloud)
            elif method == 'ball_pivoting':
                return self._ball_pivoting(point_cloud)
            elif method == 'marching_cubes':
                return self._marching_cubes(point_cloud)
            else:
                raise ValueError(f"Unknown method: {method}")

        except Exception as e:
            self.logger.error(f"Mesh extraction failed: {e}")
            return None

    def _poisson_reconstruction(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """
        Poisson surface reconstruction.

        Requires point cloud with normals.
        """
        # Estimate normals if not present
        if not pcd.has_normals():
            self.logger.info("Estimating normals...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)

        # Poisson reconstruction
        self.logger.info("Running Poisson reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9, width=0, scale=1.1, linear_fit=False
        )

        # Remove low-density vertices (outliers)
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.01)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        self.logger.info(f"✓ Poisson mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

        return mesh

    def _ball_pivoting(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """
        Ball pivoting algorithm.

        Good for unstructured point clouds.
        """
        # Estimate normals if not present
        if not pcd.has_normals():
            pcd.estimate_normals()

        # Estimate point cloud density
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)

        # Ball radii (multi-scale)
        radii = [avg_dist * r for r in [1.0, 2.0, 4.0]]

        self.logger.info(f"Running ball pivoting with radii: {radii}...")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(radii)
        )

        self.logger.info(f"✓ Ball pivoting mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

        return mesh

    def _marching_cubes(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """
        Marching cubes on voxel grid.

        Fast method that works without normals.
        """
        self.logger.info("Creating voxel grid...")

        # Create voxel grid
        voxel_size = self._estimate_voxel_size(pcd)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

        # For marching cubes, we need to use a different approach
        # Create mesh using alpha shapes as a proxy (Open3D doesn't have direct marching cubes)
        self.logger.info("Creating mesh using alpha shapes...")

        # Compute alpha value
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        alpha = avg_dist * 3.0

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

        self.logger.info(f"✓ Alpha shape mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

        return mesh

    def post_process_mesh(self,
                         mesh: o3d.geometry.TriangleMesh,
                         remove_outliers: bool = True,
                         simplify: bool = True,
                         target_triangles: int = 100000,
                         smooth: bool = True,
                         smooth_iterations: int = 1) -> o3d.geometry.TriangleMesh:
        """
        Post-process mesh with various operations.

        Args:
            mesh: Input mesh
            remove_outliers: Remove disconnected components
            simplify: Reduce triangle count
            target_triangles: Target number of triangles
            smooth: Apply smoothing
            smooth_iterations: Number of smoothing iterations

        Returns:
            Processed mesh
        """
        self.logger.info("Post-processing mesh...")

        processed_mesh = mesh

        # Remove outliers (small disconnected components)
        if remove_outliers:
            self.logger.info("Removing outliers...")
            processed_mesh = self._remove_mesh_outliers(processed_mesh)

        # Simplify
        if simplify and len(processed_mesh.triangles) > target_triangles:
            self.logger.info(f"Simplifying mesh to ~{target_triangles} triangles...")
            processed_mesh = self._simplify_mesh(processed_mesh, target_triangles)

        # Smooth
        if smooth:
            self.logger.info(f"Smoothing mesh ({smooth_iterations} iterations)...")
            processed_mesh = processed_mesh.filter_smooth_simple(number_of_iterations=smooth_iterations)

        # Compute normals
        processed_mesh.compute_vertex_normals()

        self.logger.info(
            f"✓ Final mesh: {len(processed_mesh.vertices)} vertices, "
            f"{len(processed_mesh.triangles)} triangles"
        )

        return processed_mesh

    def _remove_mesh_outliers(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Remove small disconnected components"""
        # Cluster connected components
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        if len(cluster_n_triangles) == 0:
            return mesh

        # Keep only largest cluster
        largest_cluster_idx = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        mesh.remove_triangles_by_mask(triangles_to_remove)

        # Remove unreferenced vertices
        mesh.remove_unreferenced_vertices()

        return mesh

    def _simplify_mesh(self, mesh: o3d.geometry.TriangleMesh,
                      target_triangles: int) -> o3d.geometry.TriangleMesh:
        """Simplify mesh using quadric decimation"""
        current_triangles = len(mesh.triangles)

        if current_triangles <= target_triangles:
            return mesh

        # Compute reduction ratio
        reduction_ratio = target_triangles / current_triangles

        # Quadric decimation
        simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)

        self.logger.info(
            f"Reduced from {current_triangles} to {len(simplified.triangles)} triangles "
            f"({len(simplified.triangles)/current_triangles*100:.1f}%)"
        )

        return simplified

    def _estimate_voxel_size(self, pcd: o3d.geometry.PointCloud) -> float:
        """Estimate appropriate voxel size for point cloud"""
        if len(pcd.points) == 0:
            return 0.01

        # Use average nearest neighbor distance
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)

        # Voxel size is ~2x average point spacing
        voxel_size = avg_dist * 2.0

        return float(voxel_size)
