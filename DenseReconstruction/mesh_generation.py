"""
Mesh Generation Module
=====================

This module handles surface reconstruction from point clouds,
mesh processing, and texture mapping.

Author: Photogrammetry Pipeline
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict, Optional, Union
from scipy.spatial import Delaunay
import trimesh
import open3d as o3d
import warnings
warnings.filterwarnings('ignore')

class MeshGenerator:
    """Surface reconstruction and mesh generation"""
    
    def __init__(self):
        self.meshes = []
        self.textured_meshes = []
        
    def create_mesh_poisson(self, points: np.ndarray, normals: np.ndarray = None,
                          depth: int = 9, width: int = 0, scale: float = 1.1,
                          linear_fit: bool = False) -> trimesh.Trimesh:
        """
        Create mesh using Poisson surface reconstruction
        
        Args:
            points: 3D points (N, 3)
            normals: Point normals (N, 3) - estimated if None
            depth: Poisson reconstruction depth (higher = more detail)
            width: Octree width (0 = auto)
            scale: Scale factor for reconstruction
            linear_fit: Whether to use linear fitting
            
        Returns:
            Generated mesh
        """
        # Ensure points are in correct format
        if points.shape[0] == 3:
            points = points.T
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate normals if not provided
        if normals is None:
            print("Estimating point normals...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(100)
        else:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        
        print(f"Running Poisson reconstruction (depth={depth})...")
        
        # Poisson reconstruction
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
        )
        
        # Remove low-density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh_o3d.remove_vertices_by_mask(vertices_to_remove)
        
        # Convert to trimesh
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        
        if len(vertices) > 0 and len(faces) > 0:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            print(f"Generated mesh: {len(vertices)} vertices, {len(faces)} faces")
            return mesh
        else:
            print("Failed to generate mesh - no vertices or faces")
            return None
    
    def create_mesh_ball_pivoting(self, points: np.ndarray, normals: np.ndarray = None,
                                 radii: List[float] = None) -> trimesh.Trimesh:
        """
        Create mesh using ball pivoting algorithm
        
        Args:
            points: 3D points (N, 3)
            normals: Point normals (N, 3) - estimated if None
            radii: List of ball radii to try
            
        Returns:
            Generated mesh
        """
        if points.shape[0] == 3:
            points = points.T
        
        if radii is None:
            # Estimate appropriate radii based on point density
            distances = []
            for i in range(min(100, len(points))):
                dists = np.linalg.norm(points - points[i], axis=1)
                distances.append(np.sort(dists)[1])  # Nearest neighbor distance
            
            avg_distance = np.mean(distances)
            radii = [avg_distance * r for r in [0.5, 1.0, 2.0, 4.0]]
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate normals if not provided
        if normals is None:
            print("Estimating point normals...")
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(100)
        else:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        
        print(f"Running ball pivoting with radii: {radii}")
        
        # Ball pivoting algorithm
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        
        # Convert to trimesh
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        
        if len(vertices) > 0 and len(faces) > 0:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            print(f"Generated mesh: {len(vertices)} vertices, {len(faces)} faces")
            return mesh
        else:
            print("Failed to generate mesh - no vertices or faces")
            return None
    
    def create_mesh_alpha_shape(self, points: np.ndarray, alpha: float = 0.03) -> trimesh.Trimesh:
        """
        Create mesh using alpha shape algorithm
        
        Args:
            points: 3D points (N, 3)
            alpha: Alpha parameter (smaller = more detailed)
            
        Returns:
            Generated mesh
        """
        if points.shape[0] == 3:
            points = points.T
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        print(f"Running alpha shape with alpha={alpha}")
        
        # Alpha shape
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha
        )
        
        # Convert to trimesh
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        
        if len(vertices) > 0 and len(faces) > 0:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            print(f"Generated mesh: {len(vertices)} vertices, {len(faces)} faces")
            return mesh
        else:
            print("Failed to generate mesh - no vertices or faces")
            return None
    
    def create_mesh_delaunay(self, points: np.ndarray) -> trimesh.Trimesh:
        """
        Create mesh using 3D Delaunay triangulation
        
        Args:
            points: 3D points (N, 3)
            
        Returns:
            Generated mesh
        """
        if points.shape[0] == 3:
            points = points.T
        
        print("Computing 3D Delaunay triangulation...")
        
        # Compute Delaunay triangulation
        tri = Delaunay(points)
        
        # Extract surface triangles (simplified - just use all tetrahedra faces)
        faces = []
        for simplex in tri.simplices:
            # Each tetrahedron has 4 triangular faces
            faces.extend([
                [simplex[0], simplex[1], simplex[2]],
                [simplex[0], simplex[1], simplex[3]],
                [simplex[0], simplex[2], simplex[3]],
                [simplex[1], simplex[2], simplex[3]]
            ])
        
        faces = np.array(faces)
        
        # Remove duplicate faces (interior faces appear twice)
        unique_faces = []
        face_set = set()
        for face in faces:
            face_tuple = tuple(sorted(face))
            if face_tuple not in face_set:
                unique_faces.append(face)
                face_set.add(face_tuple)
        
        if len(unique_faces) > 0:
            mesh = trimesh.Trimesh(vertices=points, faces=unique_faces)
            print(f"Delaunay mesh: {len(points)} vertices, {len(unique_faces)} faces")
            return mesh
        else:
            print("Failed to generate Delaunay mesh")
            return None
    
    def simplify_mesh(self, mesh: trimesh.Trimesh, target_faces: int = None, 
                     reduction_ratio: float = 0.5) -> trimesh.Trimesh:
        """
        Simplify mesh by reducing face count
        
        Args:
            mesh: Input mesh
            target_faces: Target number of faces (if None, use reduction_ratio)
            reduction_ratio: Fraction of faces to keep
            
        Returns:
            Simplified mesh
        """
        if mesh is None:
            return None
        
        print(f"Simplifying mesh from {len(mesh.faces)} faces...")
        
        if target_faces is None:
            target_faces = int(len(mesh.faces) * reduction_ratio)
        
        # Use mesh decimation
        simplified = mesh.simplify_quadric_decimation(target_faces)
        
        print(f"Simplified to {len(simplified.faces)} faces")
        return simplified
    
    def smooth_mesh(self, mesh: trimesh.Trimesh, iterations: int = 5) -> trimesh.Trimesh:
        """
        Smooth mesh using Laplacian smoothing
        
        Args:
            mesh: Input mesh
            iterations: Number of smoothing iterations
            
        Returns:
            Smoothed mesh
        """
        if mesh is None:
            return None
        
        print(f"Smoothing mesh with {iterations} iterations...")
        
        # Apply Laplacian smoothing
        smoothed = mesh.smoothed(iterations=iterations)
        
        return smoothed
    
    def repair_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Repair mesh by fixing common issues
        
        Args:
            mesh: Input mesh
            
        Returns:
            Repaired mesh
        """
        if mesh is None:
            return None
        
        print("Repairing mesh...")
        
        # Remove duplicate vertices
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        
        # Fill holes (if small)
        mesh.fill_holes()
        
        # Fix normals
        mesh.fix_normals()
        
        print("Mesh repair completed")
        return mesh
    
    def texture_mesh(self, mesh: trimesh.Trimesh, 
                    images: List[np.ndarray],
                    camera_poses: List[Dict],
                    camera_matrix: np.ndarray,
                    texture_method: str = 'face_projection') -> trimesh.Trimesh:
        """
        Apply texture to mesh from multiple views
        
        Args:
            mesh: Input mesh
            images: List of texture images
            camera_poses: Camera poses for each image
            camera_matrix: Camera intrinsic matrix
            texture_method: Method for texture mapping
            
        Returns:
            Textured mesh
        """
        if mesh is None:
            return None
        
        print("Applying texture to mesh...")
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        if texture_method == 'face_projection':
            # Project vertices to best view for each face
            face_colors = np.zeros((len(faces), 3))
            
            for face_idx, face in enumerate(faces):
                # Get face center
                face_center = np.mean(vertices[face], axis=0)
                
                # Find best camera view for this face
                best_view = 0
                min_distance = float('inf')
                
                for view_idx, pose in enumerate(camera_poses):
                    # Camera position in world coordinates
                    camera_pos = -pose['R'].T @ pose['t']
                    distance = np.linalg.norm(face_center - camera_pos.ravel())
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_view = view_idx
                
                # Project face center to best view
                R, t = camera_poses[best_view]['R'], camera_poses[best_view]['t']
                point_cam = R @ face_center.reshape(3, 1) + t
                
                if point_cam[2] > 0:  # In front of camera
                    # Project to image
                    point_2d = camera_matrix @ point_cam
                    u = int(point_2d[0] / point_2d[2])
                    v = int(point_2d[1] / point_2d[2])
                    
                    # Sample color from image
                    image = images[best_view]
                    if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                        if len(image.shape) == 3:
                            face_colors[face_idx] = image[v, u] / 255.0
                        else:
                            gray_val = image[v, u] / 255.0
                            face_colors[face_idx] = [gray_val, gray_val, gray_val]
            
            # Apply colors to mesh
            mesh.visual.face_colors = (face_colors * 255).astype(np.uint8)
        
        elif texture_method == 'vertex_projection':
            # Project each vertex to best view
            vertex_colors = np.zeros((len(vertices), 3))
            
            for vertex_idx, vertex in enumerate(vertices):
                # Find best camera view for this vertex
                best_view = 0
                min_distance = float('inf')
                
                for view_idx, pose in enumerate(camera_poses):
                    # Camera position in world coordinates
                    camera_pos = -pose['R'].T @ pose['t']
                    distance = np.linalg.norm(vertex - camera_pos.ravel())
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_view = view_idx
                
                # Project vertex to best view
                R, t = camera_poses[best_view]['R'], camera_poses[best_view]['t']
                point_cam = R @ vertex.reshape(3, 1) + t
                
                if point_cam[2] > 0:  # In front of camera
                    # Project to image
                    point_2d = camera_matrix @ point_cam
                    u = int(point_2d[0] / point_2d[2])
                    v = int(point_2d[1] / point_2d[2])
                    
                    # Sample color from image
                    image = images[best_view]
                    if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                        if len(image.shape) == 3:
                            vertex_colors[vertex_idx] = image[v, u] / 255.0
                        else:
                            gray_val = image[v, u] / 255.0
                            vertex_colors[vertex_idx] = [gray_val, gray_val, gray_val]
            
            # Apply colors to mesh
            mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)
        
        return mesh
    
    def analyze_mesh_quality(self, mesh: trimesh.Trimesh) -> Dict:
        """
        Analyze quality of mesh
        
        Args:
            mesh: Input mesh
            
        Returns:
            Quality metrics dictionary
        """
        if mesh is None:
            return {'error': 'No mesh provided'}
        
        metrics = {}
        
        # Basic statistics
        metrics['num_vertices'] = len(mesh.vertices)
        metrics['num_faces'] = len(mesh.faces)
        metrics['volume'] = float(mesh.volume) if mesh.is_volume else 0.0
        metrics['surface_area'] = float(mesh.area)
        
        # Geometric properties
        metrics['bounds'] = {
            'min': mesh.bounds[0].tolist(),
            'max': mesh.bounds[1].tolist(),
            'extents': mesh.extents.tolist()
        }
        
        # Topological properties
        metrics['topology'] = {
            'is_watertight': mesh.is_watertight,
            'is_winding_consistent': mesh.is_winding_consistent,
            'euler_number': mesh.euler_number,
            'genus': (2 - mesh.euler_number) // 2 if mesh.is_watertight else None
        }
        
        # Mesh quality metrics
        if len(mesh.faces) > 0:
            # Face areas
            face_areas = mesh.area_faces
            metrics['face_quality'] = {
                'min_area': float(np.min(face_areas)),
                'max_area': float(np.max(face_areas)),
                'mean_area': float(np.mean(face_areas)),
                'area_std': float(np.std(face_areas))
            }
            
            # Edge lengths
            edges = mesh.edges_unique
            edge_vectors = mesh.vertices[edges[:, 1]] - mesh.vertices[edges[:, 0]]
            edge_lengths = np.linalg.norm(edge_vectors, axis=1)
            
            metrics['edge_quality'] = {
                'min_length': float(np.min(edge_lengths)),
                'max_length': float(np.max(edge_lengths)),
                'mean_length': float(np.mean(edge_lengths)),
                'length_std': float(np.std(edge_lengths))
            }
            
            # Face angles (triangle quality)
            face_angles = []
            for face in mesh.faces:
                v0, v1, v2 = mesh.vertices[face]
                
                # Calculate angles
                e1 = v1 - v0
                e2 = v2 - v0
                e3 = v2 - v1
                
                # Angle at v0
                angle1 = np.arccos(np.clip(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)), -1, 1))
                # Angle at v1
                angle2 = np.arccos(np.clip(np.dot(-e1, e3) / (np.linalg.norm(e1) * np.linalg.norm(e3)), -1, 1))
                # Angle at v2
                angle3 = np.pi - angle1 - angle2
                
                face_angles.extend([angle1, angle2, angle3])
            
            face_angles = np.array(face_angles)
            metrics['angle_quality'] = {
                'min_angle_deg': float(np.degrees(np.min(face_angles))),
                'max_angle_deg': float(np.degrees(np.max(face_angles))),
                'mean_angle_deg': float(np.degrees(np.mean(face_angles))),
                'angle_std_deg': float(np.degrees(np.std(face_angles)))
            }
        
        return metrics
    
    def visualize_mesh(self, mesh: trimesh.Trimesh, title: str = "Mesh Visualization"):
        """
        Visualize mesh with quality analysis
        
        Args:
            mesh: Input mesh
            title: Plot title
        """
        if mesh is None:
            print("No mesh to visualize")
            return
        
        fig = plt.figure(figsize=(15, 10))
        
        # Main mesh plot
        ax1 = fig.add_subplot(221, projection='3d')
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Plot mesh wireframe (subsample for performance)
        face_step = max(1, len(faces) // 1000)
        for i in range(0, len(faces), face_step):
            face = faces[i]
            triangle = vertices[face]
            # Close the triangle
            triangle_closed = np.vstack([triangle, triangle[0]])
            ax1.plot3D(*triangle_closed.T, 'b-', alpha=0.3, linewidth=0.5)
        
        ax1.set_title("Mesh Wireframe")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        
        # Face area distribution
        ax2 = fig.add_subplot(222)
        face_areas = mesh.area_faces
        ax2.hist(face_areas, bins=50, alpha=0.7, edgecolor='black')
        ax2.set_xlabel("Face Area")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Face Area Distribution")
        
        # Edge length distribution
        ax3 = fig.add_subplot(223)
        edges = mesh.edges_unique
        edge_vectors = mesh.vertices[edges[:, 1]] - mesh.vertices[edges[:, 0]]
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        ax3.hist(edge_lengths, bins=50, alpha=0.7, edgecolor='black')
        ax3.set_xlabel("Edge Length")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Edge Length Distribution")
        
        # Statistics
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        
        # Calculate and display statistics
        quality_metrics = self.analyze_mesh_quality(mesh)
        
        stats_text = f"Mesh Statistics\n\n"
        stats_text += f"Vertices: {quality_metrics['num_vertices']:,}\n"
        stats_text += f"Faces: {quality_metrics['num_faces']:,}\n"
        stats_text += f"Volume: {quality_metrics['volume']:.3f}\n"
        stats_text += f"Surface Area: {quality_metrics['surface_area']:.3f}\n"
        
        if 'topology' in quality_metrics:
            topo = quality_metrics['topology']
            stats_text += f"\nTopology:\n"
            stats_text += f"  Watertight: {topo['is_watertight']}\n"
            stats_text += f"  Winding Consistent: {topo['is_winding_consistent']}\n"
            stats_text += f"  Euler Number: {topo['euler_number']}\n"
            if topo['genus'] is not None:
                stats_text += f"  Genus: {topo['genus']}\n"
        
        if 'face_quality' in quality_metrics:
            face_qual = quality_metrics['face_quality']
            stats_text += f"\nFace Quality:\n"
            stats_text += f"  Area range: {face_qual['min_area']:.6f} - {face_qual['max_area']:.6f}\n"
            stats_text += f"  Mean area: {face_qual['mean_area']:.6f}\n"
        
        if 'angle_quality' in quality_metrics:
            angle_qual = quality_metrics['angle_quality']
            stats_text += f"\nAngle Quality:\n"
            stats_text += f"  Angle range: {angle_qual['min_angle_deg']:.1f}° - {angle_qual['max_angle_deg']:.1f}°\n"
            stats_text += f"  Mean angle: {angle_qual['mean_angle_deg']:.1f}°\n"
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def export_mesh(self, mesh: trimesh.Trimesh, filename: str = "mesh.obj") -> bool:
        """
        Export mesh to file
        
        Args:
            mesh: Input mesh
            filename: Output filename
            
        Returns:
            Success status
        """
        if mesh is None:
            print("No mesh to export")
            return False
        
        try:
            mesh.export(filename)
            print(f"Mesh exported to {filename}")
            return True
            
        except Exception as e:
            print(f"Error exporting mesh: {e}")
            return False
    
    def create_mesh_from_depth_map(self, depth_map: np.ndarray, 
                                  color_image: np.ndarray,
                                  camera_matrix: np.ndarray,
                                  method: str = 'triangulation') -> trimesh.Trimesh:
        """
        Create mesh directly from depth map
        
        Args:
            depth_map: Input depth map
            color_image: Corresponding color image
            camera_matrix: Camera intrinsic matrix
            method: Mesh generation method ('triangulation', 'marching_cubes')
            
        Returns:
            Generated mesh
        """
        height, width = depth_map.shape
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Valid depth mask
        valid_mask = ~np.isnan(depth_map) & (depth_map > 0)
        
        if method == 'triangulation':
            # Convert depth map to 3D points
            fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
            cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
            
            # Create vertex array
            vertices = []
            colors = []
            vertex_indices = np.full((height, width), -1, dtype=int)
            
            vertex_idx = 0
            for y in range(height):
                for x in range(width):
                    if valid_mask[y, x]:
                        depth = depth_map[y, x]
                        
                        # Convert to 3D
                        x_3d = (x - cx) * depth / fx
                        y_3d = (y - cy) * depth / fy
                        z_3d = depth
                        
                        vertices.append([x_3d, y_3d, z_3d])
                        
                        # Add color
                        if len(color_image.shape) == 3:
                            colors.append(color_image[y, x])
                        else:
                            gray_val = color_image[y, x]
                            colors.append([gray_val, gray_val, gray_val])
                        
                        vertex_indices[y, x] = vertex_idx
                        vertex_idx += 1
            
            vertices = np.array(vertices)
            colors = np.array(colors)
            
            # Create faces by connecting neighboring valid pixels
            faces = []
            for y in range(height - 1):
                for x in range(width - 1):
                    # Check if we have a valid quad
                    indices = [
                        vertex_indices[y, x],
                        vertex_indices[y, x + 1],
                        vertex_indices[y + 1, x],
                        vertex_indices[y + 1, x + 1]
                    ]
                    
                    # Only create faces if we have valid vertices
                    valid_indices = [i for i in indices if i >= 0]
                    
                    if len(valid_indices) >= 3:
                        if len(valid_indices) == 4:
                            # Create two triangles from quad
                            faces.append([indices[0], indices[1], indices[2]])
                            faces.append([indices[1], indices[3], indices[2]])
                        elif len(valid_indices) == 3:
                            # Create single triangle
                            faces.append([i for i in indices if i >= 0])
            
            faces = np.array(faces)
            
            if len(vertices) > 0 and len(faces) > 0:
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                mesh.visual.vertex_colors = colors
                
                print(f"Created mesh from depth map: {len(vertices)} vertices, {len(faces)} faces")
                return mesh
            else:
                print("Failed to create mesh from depth map")
                return None
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compare_meshes(self, mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh) -> Dict:
        """
        Compare two meshes and return quality metrics
        
        Args:
            mesh1: First mesh
            mesh2: Second mesh
            
        Returns:
            Comparison metrics
        """
        comparison = {}
        
        if mesh1 is None or mesh2 is None:
            comparison['error'] = 'One or both meshes are None'
            return comparison
        
        # Basic statistics comparison
        comparison['vertex_count'] = {
            'mesh1': len(mesh1.vertices),
            'mesh2': len(mesh2.vertices),
            'ratio': len(mesh2.vertices) / max(1, len(mesh1.vertices))
        }
        
        comparison['face_count'] = {
            'mesh1': len(mesh1.faces),
            'mesh2': len(mesh2.faces),
            'ratio': len(mesh2.faces) / max(1, len(mesh1.faces))
        }
        
        comparison['volume'] = {
            'mesh1': float(mesh1.volume) if mesh1.is_volume else 0.0,
            'mesh2': float(mesh2.volume) if mesh2.is_volume else 0.0,
            'ratio': (mesh2.volume if mesh2.is_volume else 0.0) / max(1e-10, mesh1.volume if mesh1.is_volume else 1e-10)
        }
        
        comparison['surface_area'] = {
            'mesh1': float(mesh1.area),
            'mesh2': float(mesh2.area),
            'ratio': mesh2.area / max(1e-10, mesh1.area)
        }
        
        # Bounds comparison
        comparison['bounds'] = {
            'mesh1_extents': mesh1.extents.tolist(),
            'mesh2_extents': mesh2.extents.tolist(),
            'extents_ratio': (mesh2.extents / np.maximum(mesh1.extents, 1e-10)).tolist()
        }
        
        return comparison

# Example usage and testing
def test_mesh_generation():
    """Test mesh generation with synthetic data"""
    
    print("Testing mesh generation...")
    
    # Generate synthetic point cloud (sphere)
    np.random.seed(42)
    n_points = 2000
    
    # Create noisy sphere
    phi = np.random.uniform(0, 2*np.pi, n_points)
    theta = np.random.uniform(0, np.pi, n_points)
    radius = 1.0 + np.random.normal(0, 0.05, n_points)
    
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    points = np.column_stack([x, y, z])
    
    # Initialize mesh generator
    mesh_gen = MeshGenerator()
    
    # Test different mesh generation methods
    methods = ['poisson', 'ball_pivoting', 'alpha_shape']
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Testing {method} method")
        print('='*50)
        
        if method == 'poisson':
            mesh = mesh_gen.create_mesh_poisson(points, depth=7)
        elif method == 'ball_pivoting':
            mesh = mesh_gen.create_mesh_ball_pivoting(points)
        elif method == 'alpha_shape':
            mesh = mesh_gen.create_mesh_alpha_shape(points, alpha=0.1)
        
        if mesh is not None:
            print(f"Generated mesh with {method}")
            
            # Test mesh processing
            print("Testing mesh processing...")
            
            # Repair mesh
            mesh = mesh_gen.repair_mesh(mesh)
            
            # Simplify mesh
            simplified_mesh = mesh_gen.simplify_mesh(mesh, reduction_ratio=0.7)
            
            # Smooth mesh
            smoothed_mesh = mesh_gen.smooth_mesh(simplified_mesh, iterations=3)
            
            # Visualize results
            mesh_gen.visualize_mesh(smoothed_mesh, f"Test Mesh - {method}")
            
            # Test export
            success = mesh_gen.export_mesh(smoothed_mesh, f"test_mesh_{method}.obj")
            if success:
                print(f"Successfully exported {method} mesh")
            
            # Compare with original
            if method == 'poisson':  # Use first method as reference
                original_mesh = mesh
            else:
                comparison = mesh_gen.compare_meshes(original_mesh, smoothed_mesh)
                print(f"Comparison with Poisson mesh: {comparison}")
        
        else:
            print(f"Failed to generate mesh with {method}")
    
    print("\nMesh generation testing completed!")

if __name__ == "__main__":
    test_mesh_generation()