"""
Point Cloud Processing Module
============================

This module handles conversion from depth maps to point clouds,
filtering, and point cloud processing operations.

Author: Photogrammetry Pipeline
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict, Optional, Union
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import open3d as o3d
import warnings
warnings.filterwarnings('ignore')

class PointCloudProcessor:
    """Point cloud generation and processing"""
    
    def __init__(self):
        self.point_clouds = []
        self.merged_cloud = None
        
    def depth_map_to_point_cloud(self, depth_map: np.ndarray,
                                color_image: np.ndarray,
                                camera_matrix: np.ndarray,
                                camera_pose: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert depth map to colored 3D point cloud
        
        Args:
            depth_map: Dense depth map
            color_image: Corresponding color image
            camera_matrix: Camera intrinsic matrix
            camera_pose: Camera pose {'R': R, 't': t}
            
        Returns:
            points_3d: 3D points (3, N)
            colors: RGB colors (3, N)
        """
        height, width = depth_map.shape
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Valid depth mask
        valid_mask = ~np.isnan(depth_map) & (depth_map > 0)
        
        # Extract valid coordinates and depths
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depth_valid = depth_map[valid_mask]
        
        # Convert to 3D coordinates in camera frame
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        x_cam = (u_valid - cx) * depth_valid / fx
        y_cam = (v_valid - cy) * depth_valid / fy
        z_cam = depth_valid
        
        points_cam = np.vstack([x_cam, y_cam, z_cam])
        
        # Transform to world coordinates if pose is provided
        if camera_pose is not None:
            R, t = camera_pose['R'], camera_pose['t']
            points_world = R.T @ (points_cam - t)
        else:
            points_world = points_cam
        
        # Extract colors
        if len(color_image.shape) == 3:
            colors = color_image[valid_mask].T  # (3, N)
        else:
            # Grayscale image
            gray_colors = color_image[valid_mask]
            colors = np.vstack([gray_colors, gray_colors, gray_colors])
        
        return points_world, colors
    
    def filter_point_cloud(self, points: np.ndarray, colors: np.ndarray = None,
                          filter_params: Dict = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Filter point cloud to remove outliers and noise
        
        Args:
            points: 3D points (3, N) or (N, 3)
            colors: Optional colors (3, N) or (N, 3)
            filter_params: Filtering parameters
            
        Returns:
            filtered_points: Cleaned 3D points
            filtered_colors: Cleaned colors (if provided)
        """
        if filter_params is None:
            filter_params = {
                'statistical_outlier_removal': True,
                'radius_outlier_removal': True,
                'voxel_downsampling': True,
                'statistical_nb_neighbors': 20,
                'statistical_std_ratio': 2.0,
                'radius_outlier_nb_points': 16,
                'radius_outlier_radius': 0.05,
                'voxel_size': 0.01
            }
        
        # Ensure points are in (N, 3) format
        if points.shape[0] == 3:
            points = points.T
        if colors is not None and colors.shape[0] == 3:
            colors = colors.T
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0 if colors.max() > 1 else colors)
        
        print(f"Initial point cloud: {len(pcd.points)} points")
        
        # Statistical outlier removal
        if filter_params.get('statistical_outlier_removal', False):
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=filter_params['statistical_nb_neighbors'],
                std_ratio=filter_params['statistical_std_ratio']
            )
            print(f"After statistical outlier removal: {len(pcd.points)} points")
        
        # Radius outlier removal
        if filter_params.get('radius_outlier_removal', False):
            pcd, _ = pcd.remove_radius_outlier(
                nb_points=filter_params['radius_outlier_nb_points'],
                radius=filter_params['radius_outlier_radius']
            )
            print(f"After radius outlier removal: {len(pcd.points)} points")
        
        # Voxel downsampling
        if filter_params.get('voxel_downsampling', False):
            pcd = pcd.voxel_down_sample(voxel_size=filter_params['voxel_size'])
            print(f"After voxel downsampling: {len(pcd.points)} points")
        
        # Convert back to numpy
        filtered_points = np.asarray(pcd.points)
        filtered_colors = np.asarray(pcd.colors) if len(pcd.colors) > 0 else None
        
        return filtered_points, filtered_colors
    
    def merge_point_clouds(self, point_clouds: List[Tuple[np.ndarray, np.ndarray]],
                          registration_method: str = 'icp') -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge multiple point clouds into a single cloud
        
        Args:
            point_clouds: List of (points, colors) tuples
            registration_method: Method for registration ('icp', 'none')
            
        Returns:
            merged_points: Combined 3D points
            merged_colors: Combined colors
        """
        if not point_clouds:
            return None, None
        
        if len(point_clouds) == 1:
            return point_clouds[0]
        
        print(f"Merging {len(point_clouds)} point clouds...")
        
        # Start with first cloud
        merged_points, merged_colors = point_clouds[0]
        
        # Ensure correct format
        if merged_points.shape[0] == 3:
            merged_points = merged_points.T
        if merged_colors is not None and merged_colors.shape[0] == 3:
            merged_colors = merged_colors.T
        
        # Merge additional clouds
        for i, (points, colors) in enumerate(point_clouds[1:], 1):
            if points.shape[0] == 3:
                points = points.T
            if colors is not None and colors.shape[0] == 3:
                colors = colors.T
            
            if registration_method == 'icp':
                # Register using ICP
                points_registered, colors_registered = self._register_point_clouds_icp(
                    merged_points, points, merged_colors, colors
                )
                points = points_registered
                colors = colors_registered
            
            # Combine clouds
            merged_points = np.vstack([merged_points, points])
            if merged_colors is not None and colors is not None:
                merged_colors = np.vstack([merged_colors, colors])
            elif colors is not None:
                merged_colors = colors
            
            print(f"  Merged cloud {i}: {len(merged_points)} total points")
        
        return merged_points, merged_colors
    
    def _register_point_clouds_icp(self, target_points: np.ndarray, source_points: np.ndarray,
                                  target_colors: np.ndarray = None, source_colors: np.ndarray = None,
                                  max_correspondence_distance: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Register two point clouds using ICP
        
        Args:
            target_points: Target point cloud (N, 3)
            source_points: Source point cloud (M, 3)
            target_colors: Target colors (N, 3)
            source_colors: Source colors (M, 3)
            max_correspondence_distance: Maximum distance for correspondences
            
        Returns:
            registered_points: Registered source points
            registered_colors: Registered source colors
        """
        # Create Open3D point clouds
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points)
        
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_points)
        
        if target_colors is not None:
            target_pcd.colors = o3d.utility.Vector3dVector(target_colors / 255.0 if target_colors.max() > 1 else target_colors)
        if source_colors is not None:
            source_pcd.colors = o3d.utility.Vector3dVector(source_colors / 255.0 if source_colors.max() > 1 else source_colors)
        
        # Estimate normals
        target_pcd.estimate_normals()
        source_pcd.estimate_normals()
        
        # Initial alignment (optional - could use feature matching)
        initial_transform = np.eye(4)
        
        # ICP registration
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, max_correspondence_distance, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        # Apply transformation
        source_pcd.transform(result.transformation)
        
        # Extract registered points and colors
        registered_points = np.asarray(source_pcd.points)
        registered_colors = np.asarray(source_pcd.colors) if len(source_pcd.colors) > 0 else source_colors
        
        if registered_colors is not None and registered_colors.max() <= 1:
            registered_colors = registered_colors * 255.0
        
        print(f"  ICP registration: fitness = {result.fitness:.3f}, RMSE = {result.inlier_rmse:.3f}")
        
        return registered_points, registered_colors
    
    def compute_point_cloud_normals(self, points: np.ndarray, 
                                   method: str = 'pca',
                                   search_radius: float = 0.1,
                                   max_neighbors: int = 30) -> np.ndarray:
        """
        Compute point cloud normals
        
        Args:
            points: 3D points (N, 3) or (3, N)
            method: Normal estimation method ('pca', 'weighted_pca')
            search_radius: Search radius for neighbors
            max_neighbors: Maximum number of neighbors
            
        Returns:
            normals: Point normals (N, 3)
        """
        if points.shape[0] == 3:
            points = points.T
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate normals
        if method == 'pca':
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=search_radius, max_nn=max_neighbors
                )
            )
        elif method == 'weighted_pca':
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamRadius(radius=search_radius)
            )
        
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(max_neighbors)
        
        normals = np.asarray(pcd.normals)
        
        print(f"Computed normals for {len(normals)} points")
        
        return normals
    
    def analyze_point_cloud_quality(self, points: np.ndarray, colors: np.ndarray = None) -> Dict:
        """
        Analyze quality of point cloud
        
        Args:
            points: 3D points (N, 3) or (3, N)
            colors: Optional colors
            
        Returns:
            Quality metrics dictionary
        """
        if points.shape[0] == 3:
            points = points.T
        
        metrics = {}
        
        # Basic statistics
        metrics['num_points'] = len(points)
        metrics['bounds'] = {
            'min': points.min(axis=0).tolist(),
            'max': points.max(axis=0).tolist(),
            'range': (points.max(axis=0) - points.min(axis=0)).tolist()
        }
        
        # Point density analysis
        if len(points) > 100:
            # Sample subset for efficiency
            sample_size = min(1000, len(points))
            sample_indices = np.random.choice(len(points), sample_size, replace=False)
            sample_points = points[sample_indices]
            
            # Build KDTree and find nearest neighbors
            tree = KDTree(sample_points)
            distances, _ = tree.query(sample_points, k=2)  # k=2 to exclude self
            nearest_distances = distances[:, 1]  # Second closest (first is self)
            
            metrics['density'] = {
                'mean_nearest_distance': float(np.mean(nearest_distances)),
                'median_nearest_distance': float(np.median(nearest_distances)),
                'std_nearest_distance': float(np.std(nearest_distances))
            }
        
        # Outlier detection
        if len(points) > 50:
            clustering = DBSCAN(eps=0.1, min_samples=5)
            cluster_labels = clustering.fit_predict(points)
            
            num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            num_outliers = list(cluster_labels).count(-1)
            
            metrics['clustering'] = {
                'num_clusters': num_clusters,
                'num_outliers': num_outliers,
                'outlier_ratio': num_outliers / len(points)
            }
        
        # Color analysis (if available)
        if colors is not None:
            if colors.shape[0] == 3:
                colors = colors.T
            
            metrics['color'] = {
                'mean_rgb': colors.mean(axis=0).tolist(),
                'std_rgb': colors.std(axis=0).tolist(),
                'brightness_range': [float(colors.min()), float(colors.max())]
            }
        
        return metrics
    
    def visualize_point_cloud(self, points: np.ndarray, colors: np.ndarray = None,
                             normals: np.ndarray = None, title: str = "Point Cloud"):
        """
        Visualize point cloud with optional colors and normals
        
        Args:
            points: 3D points (N, 3) or (3, N)
            colors: Optional colors (N, 3) or (3, N)
            normals: Optional normals (N, 3)
            title: Plot title
        """
        if points.shape[0] == 3:
            points = points.T
        if colors is not None and colors.shape[0] == 3:
            colors = colors.T
        
        fig = plt.figure(figsize=(15, 10))
        
        # Main point cloud plot
        ax1 = fig.add_subplot(221, projection='3d')
        
        if colors is not None:
            if colors.max() > 1:
                colors_norm = colors / 255.0
            else:
                colors_norm = colors
            ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                       c=colors_norm, s=1, alpha=0.6)
        else:
            ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                       s=1, alpha=0.6)
        
        ax1.set_title("Point Cloud")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        
        # Density visualization
        ax2 = fig.add_subplot(222)
        
        # Project to XY plane for density visualization
        hist, xedges, yedges = np.histogram2d(points[:, 0], points[:, 1], bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        ax2.imshow(hist.T, extent=extent, origin='lower', cmap='hot')
        ax2.set_title("Point Density (XY Projection)")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        
        # Normals visualization (if available)
        if normals is not None:
            ax3 = fig.add_subplot(223, projection='3d')
            
            # Subsample for visualization
            step = max(1, len(points) // 1000)
            points_sub = points[::step]
            normals_sub = normals[::step]
            
            ax3.scatter(points_sub[:, 0], points_sub[:, 1], points_sub[:, 2], 
                       s=1, alpha=0.3)
            
            # Draw normal vectors
            for i in range(0, len(points_sub), 10):  # Further subsample
                p = points_sub[i]
                n = normals_sub[i] * 0.1  # Scale normal vectors
                ax3.plot([p[0], p[0] + n[0]], [p[1], p[1] + n[1]], 
                        [p[2], p[2] + n[2]], 'r-', alpha=0.7)
            
            ax3.set_title("Point Cloud with Normals")
            ax3.set_xlabel("X")
            ax3.set_ylabel("Y")
            ax3.set_zlabel("Z")
        
        # Statistics
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        
        # Calculate and display statistics
        quality_metrics = self.analyze_point_cloud_quality(points, colors)
        
        stats_text = f"Point Cloud Statistics\n\n"
        stats_text += f"Total points: {quality_metrics['num_points']:,}\n"
        stats_text += f"Bounds (min): {[f'{x:.2f}' for x in quality_metrics['bounds']['min']]}\n"
        stats_text += f"Bounds (max): {[f'{x:.2f}' for x in quality_metrics['bounds']['max']]}\n"
        stats_text += f"Range: {[f'{x:.2f}' for x in quality_metrics['bounds']['range']]}\n"
        
        if 'density' in quality_metrics:
            stats_text += f"\nDensity:\n"
            stats_text += f"  Mean neighbor dist: {quality_metrics['density']['mean_nearest_distance']:.4f}\n"
            stats_text += f"  Median neighbor dist: {quality_metrics['density']['median_nearest_distance']:.4f}\n"
        
        if 'clustering' in quality_metrics:
            stats_text += f"\nClustering:\n"
            stats_text += f"  Clusters: {quality_metrics['clustering']['num_clusters']}\n"
            stats_text += f"  Outliers: {quality_metrics['clustering']['num_outliers']}\n"
            stats_text += f"  Outlier ratio: {quality_metrics['clustering']['outlier_ratio']:.2%}\n"
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def export_point_cloud(self, points: np.ndarray, colors: np.ndarray = None,
                          normals: np.ndarray = None, filename: str = "point_cloud.ply") -> bool:
        """
        Export point cloud to PLY format
        
        Args:
            points: 3D points (N, 3) or (3, N)
            colors: Optional colors (N, 3) or (3, N)
            normals: Optional normals (N, 3)
            filename: Output filename
            
        Returns:
            Success status
        """
        try:
            if points.shape[0] == 3:
                points = points.T
            if colors is not None and colors.shape[0] == 3:
                colors = colors.T
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            if colors is not None:
                if colors.max() > 1:
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            if normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(normals)
            
            # Save
            success = o3d.io.write_point_cloud(filename, pcd)
            
            if success:
                print(f"Point cloud exported to {filename}")
            else:
                print(f"Failed to export point cloud to {filename}")
            
            return success
            
        except Exception as e:
            print(f"Error exporting point cloud: {e}")
            return False
    
    def downsample_point_cloud(self, points: np.ndarray, colors: np.ndarray = None,
                              method: str = 'voxel', **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Downsample point cloud using various methods
        
        Args:
            points: 3D points (N, 3) or (3, N)
            colors: Optional colors (N, 3) or (3, N)
            method: Downsampling method ('voxel', 'uniform', 'farthest')
            **kwargs: Method-specific parameters
            
        Returns:
            downsampled_points: Downsampled points
            downsampled_colors: Downsampled colors
        """
        if points.shape[0] == 3:
            points = points.T
        if colors is not None and colors.shape[0] == 3:
            colors = colors.T
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0 if colors.max() > 1 else colors)
        
        original_size = len(pcd.points)
        
        if method == 'voxel':
            voxel_size = kwargs.get('voxel_size', 0.01)
            pcd_down = pcd.voxel_down_sample(voxel_size)
        
        elif method == 'uniform':
            every_k_points = kwargs.get('every_k_points', 10)
            pcd_down = pcd.uniform_down_sample(every_k_points)
        
        elif method == 'farthest':
            # Custom farthest point sampling
            num_points = kwargs.get('num_points', len(points) // 10)
            indices = self._farthest_point_sampling(points, num_points)
            pcd_down = pcd.select_by_index(indices)
        
        else:
            raise ValueError(f"Unknown downsampling method: {method}")
        
        # Extract results
        downsampled_points = np.asarray(pcd_down.points)
        downsampled_colors = np.asarray(pcd_down.colors) if len(pcd_down.colors) > 0 else None
        
        if downsampled_colors is not None and downsampled_colors.max() <= 1:
            downsampled_colors = downsampled_colors * 255.0
        
        print(f"Downsampled from {original_size} to {len(downsampled_points)} points "
              f"({len(downsampled_points)/original_size:.1%} kept)")
        
        return downsampled_points, downsampled_colors
    
    def _farthest_point_sampling(self, points: np.ndarray, num_points: int) -> List[int]:
        """
        Farthest point sampling for downsampling
        
        Args:
            points: Input points (N, 3)
            num_points: Number of points to sample
            
        Returns:
            List of selected indices
        """
        if num_points >= len(points):
            return list(range(len(points)))
        
        # Start with random point
        selected_indices = [np.random.randint(len(points))]
        distances = np.full(len(points), np.inf)
        
        for _ in range(num_points - 1):
            # Update distances to selected points
            last_selected = selected_indices[-1]
            new_distances = np.linalg.norm(points - points[last_selected], axis=1)
            distances = np.minimum(distances, new_distances)
            
            # Select point with maximum distance
            next_idx = np.argmax(distances)
            selected_indices.append(next_idx)
            distances[next_idx] = 0  # Mark as selected
        
        return selected_indices

# Example usage and testing
def test_point_cloud_processing():
    """Test point cloud processing with synthetic data"""
    
    print("Testing point cloud processing...")
    
    # Generate synthetic point cloud
    np.random.seed(42)
    
    # Create a synthetic 3D surface (sphere)
    n_points = 5000
    phi = np.random.uniform(0, 2*np.pi, n_points)
    theta = np.random.uniform(0, np.pi, n_points)
    radius = 1.0 + np.random.normal(0, 0.1, n_points)  # Add noise
    
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    points = np.column_stack([x, y, z])
    
    # Generate colors based on position
    colors = np.zeros((n_points, 3))
    colors[:, 0] = (x - x.min()) / (x.max() - x.min()) * 255  # Red channel
    colors[:, 1] = (y - y.min()) / (y.max() - y.min()) * 255  # Green channel
    colors[:, 2] = (z - z.min()) / (z.max() - z.min()) * 255  # Blue channel
    
    # Initialize processor
    processor = PointCloudProcessor()
    
    # Test filtering
    print("\nTesting point cloud filtering...")
    filtered_points, filtered_colors = processor.filter_point_cloud(
        points, colors,
        filter_params={
            'statistical_outlier_removal': True,
            'voxel_downsampling': True,
            'voxel_size': 0.05
        }
    )
    
    # Test normal computation
    print("\nTesting normal computation...")
    normals = processor.compute_point_cloud_normals(filtered_points)
    
    # Test visualization
    print("\nVisualizing results...")
    processor.visualize_point_cloud(filtered_points, filtered_colors, normals, 
                                   "Test Point Cloud")
    
    # Test downsampling
    print("\nTesting downsampling...")
    downsampled_points, downsampled_colors = processor.downsample_point_cloud(
        filtered_points, filtered_colors,
        method='voxel', voxel_size=0.1
    )
    
    # Test export
    print("\nTesting export...")
    success = processor.export_point_cloud(
        downsampled_points, downsampled_colors, 
        filename="test_point_cloud.ply"
    )
    
    if success:
        print("Point cloud processing test completed successfully!")
    else:
        print("Point cloud export failed!")

if __name__ == "__main__":
    test_point_cloud_processing()