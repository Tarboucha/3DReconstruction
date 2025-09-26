import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict, Optional, Union
from scipy.spatial import KDTree, Delaunay
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
import trimesh
import open3d as o3d
import warnings
warnings.filterwarnings('ignore')

class DenseReconstruction:
    """Dense 3D reconstruction and mesh generation"""
    
    def __init__(self, camera_matrices: List[np.ndarray] = None):
        """
        Initialize dense reconstruction
        
        Args:
            camera_matrices: List of camera intrinsic matrices for each view
        """
        self.camera_matrices = camera_matrices or []
        self.sparse_points = None
        self.dense_points = None
        self.point_colors = None
        self.mesh = None
        self.depth_maps = {}
        
    def compute_stereo_depth(self, img1: np.ndarray, img2: np.ndarray,
                           camera_matrix: np.ndarray, 
                           R: np.ndarray, t: np.ndarray,
                           stereo_params: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dense depth map using stereo matching
        
        Args:
            img1: First rectified image
            img2: Second rectified image  
            camera_matrix: Camera intrinsic matrix
            R: Rotation matrix between cameras
            t: Translation vector between cameras
            stereo_params: Parameters for stereo matching
            
        Returns:
            depth_map: Dense depth map
            disparity_map: Disparity map
        """
        if stereo_params is None:
            stereo_params = {
                'numDisparities': 64,
                'blockSize': 11,
                'P1': 8 * 3 * 11**2,
                'P2': 32 * 3 * 11**2,
                'disp12MaxDiff': 1,
                'uniquenessRatio': 10,
                'speckleWindowSize': 100,
                'speckleRange': 32,
                'preFilterCap': 63,
                'minDisparity': 0
            }
        
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            gray1, gray2 = img1, img2
        
        # Rectify images
        gray1_rect, gray2_rect, Q = self._rectify_stereo_pair(
            gray1, gray2, camera_matrix, R, t
        )
        
        # Create stereo matcher
        stereo = cv2.StereoSGBM_create(
            minDisparity=stereo_params['minDisparity'],
            numDisparities=stereo_params['numDisparities'],
            blockSize=stereo_params['blockSize'],
            P1=stereo_params['P1'],
            P2=stereo_params['P2'],
            disp12MaxDiff=stereo_params['disp12MaxDiff'],
            uniquenessRatio=stereo_params['uniquenessRatio'],
            speckleWindowSize=stereo_params['speckleWindowSize'],
            speckleRange=stereo_params['speckleRange'],
            preFilterCap=stereo_params['preFilterCap']
        )
        
        # Compute disparity
        print("Computing stereo disparity...")
        disparity = stereo.compute(gray1_rect, gray2_rect).astype(np.float32) / 16.0
        
        # Filter invalid disparities
        disparity[disparity <= stereo_params['minDisparity']] = np.nan
        disparity[disparity >= stereo_params['numDisparities']] = np.nan
        
        # Convert disparity to depth
        baseline = np.linalg.norm(t)
        focal_length = camera_matrix[0, 0]
        
        depth_map = np.zeros_like(disparity)
        valid_mask = ~np.isnan(disparity) & (disparity > 0)
        depth_map[valid_mask] = (focal_length * baseline) / disparity[valid_mask]
        depth_map[~valid_mask] = np.nan
        
        return depth_map, disparity
    
    def _rectify_stereo_pair(self, img1: np.ndarray, img2: np.ndarray,
                           camera_matrix: np.ndarray, R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Rectify stereo image pair
        
        Args:
            img1: First image
            img2: Second image
            camera_matrix: Camera intrinsic matrix
            R: Rotation matrix between cameras
            t: Translation vector between cameras
            
        Returns:
            img1_rect: Rectified first image
            img2_rect: Rectified second image
            Q: Reprojection matrix
        """
        image_size = (img1.shape[1], img1.shape[0])
        
        # Stereo rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            camera_matrix, None,  # Camera 1
            camera_matrix, None,  # Camera 2 (assuming same intrinsics)
            image_size,
            R, t,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )
        
        # Create rectification maps
        map1x, map1y = cv2.initUndistortRectifyMap(
            camera_matrix, None, R1, P1, image_size, cv2.CV_32FC1
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            camera_matrix, None, R2, P2, image_size, cv2.CV_32FC1
        )
        
        # Apply rectification
        img1_rect = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
        img2_rect = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
        
        return img1_rect, img2_rect, Q
    
    def multi_view_stereo(self, images: List[np.ndarray],
                         camera_poses: List[Dict],
                         camera_matrix: np.ndarray,
                         reference_view: int = 0) -> Dict[int, np.ndarray]:
        """
        Multi-view stereo reconstruction
        
        Args:
            images: List of input images
            camera_poses: List of camera poses [{'R': R, 't': t}, ...]
            camera_matrix: Camera intrinsic matrix
            reference_view: Index of reference view
            
        Returns:
            Dictionary of depth maps for each view
        """
        depth_maps = {}
        
        print(f"Computing multi-view stereo with {len(images)} views...")
        
        # Reference image
        ref_image = images[reference_view]
        ref_pose = camera_poses[reference_view]
        
        for i, (image, pose) in enumerate(zip(images, camera_poses)):
            if i == reference_view:
                continue
            
            print(f"Processing view pair: {reference_view} -> {i}")
            
            # Compute relative pose
            R_rel = pose['R'] @ ref_pose['R'].T
            t_rel = pose['t'] - R_rel @ ref_pose['t']
            
            # Compute depth map
            depth_map, disparity = self.compute_stereo_depth(
                ref_image, image, camera_matrix, R_rel, t_rel
            )
            
            depth_maps[i] = {
                'depth': depth_map,
                'disparity': disparity,
                'baseline': np.linalg.norm(t_rel),
                'valid_pixels': np.sum(~np.isnan(depth_map))
            }
            
            print(f"  Depth map {i}: {depth_maps[i]['valid_pixels']} valid pixels")
        
        return depth_maps
    
    def fuse_depth_maps(self, depth_maps: Dict[int, np.ndarray],
                       fusion_method: str = 'weighted_average') -> np.ndarray:
        """
        Fuse multiple depth maps into a single consistent depth map
        
        Args:
            depth_maps: Dictionary of depth maps
            fusion_method: Method for fusion ('weighted_average', 'median', 'best_baseline')
            
        Returns:
            Fused depth map
        """
        if not depth_maps:
            return None
        
        # Get image dimensions from first depth map
        first_depth = list(depth_maps.values())[0]['depth']
        height, width = first_depth.shape
        
        if fusion_method == 'weighted_average':
            # Weight by baseline length and number of valid pixels
            weighted_sum = np.zeros((height, width))
            weight_sum = np.zeros((height, width))
            
            for view_id, depth_data in depth_maps.items():
                depth = depth_data['depth']
                baseline = depth_data['baseline']
                valid_mask = ~np.isnan(depth)
                
                # Weight by baseline (longer baseline = more accurate depth)
                weight = baseline * valid_mask.astype(float)
                
                weighted_sum += np.nan_to_num(depth) * weight
                weight_sum += weight
            
            fused_depth = np.divide(weighted_sum, weight_sum, 
                                  out=np.full_like(weighted_sum, np.nan),
                                  where=weight_sum>0)
        
        elif fusion_method == 'median':
            # Stack all depth maps and take median
            depth_stack = []
            for depth_data in depth_maps.values():
                depth_stack.append(depth_data['depth'])
            
            depth_array = np.stack(depth_stack, axis=2)
            fused_depth = np.nanmedian(depth_array, axis=2)
        
        elif fusion_method == 'best_baseline':
            # Use depth from view with best baseline
            best_view = max(depth_maps.keys(), 
                          key=lambda k: depth_maps[k]['baseline'])
            fused_depth = depth_maps[best_view]['depth']
        
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        return fused_depth
    
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
    
    def create_mesh_poisson(self, points: np.ndarray, normals: np.ndarray = None,
                          depth: int = 9) -> trimesh.Trimesh:
        """
        Create mesh using Poisson surface reconstruction
        
        Args:
            points: 3D points (N, 3)
            normals: Point normals (N, 3) - estimated if None
            depth: Poisson reconstruction depth
            
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
            pcd, depth=depth, width=0, scale=1.1, linear_fit=False
        )
        
        # Remove low-density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh_o3d.remove_vertices_by_mask(vertices_to_remove)
        
        # Convert to trimesh
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        print(f"Generated mesh: {len(vertices)} vertices, {len(faces)} faces")
        
        return mesh
    
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
        
        mesh = trimesh.Trimesh(vertices=points, faces=unique_faces)
        
        print(f"Delaunay mesh: {len(points)} vertices, {len(unique_faces)} faces")
        
        return mesh
    
    def texture_mesh(self, mesh: trimesh.Trimesh, 
                    images: List[np.ndarray],
                    camera_poses: List[Dict],
                    camera_matrix: np.ndarray) -> trimesh.Trimesh:
        """
        Apply texture to mesh from multiple views
        
        Args:
            mesh: Input mesh
            images: List of texture images
            camera_poses: Camera poses for each image
            camera_matrix: Camera intrinsic matrix
            
        Returns:
            Textured mesh
        """
        print("Applying texture to mesh...")
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Create UV coordinates and texture atlas (simplified approach)
        # In practice, you'd use more sophisticated UV unwrapping
        
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
        
        return mesh
    
    def visualize_dense_reconstruction(self, points: np.ndarray = None,
                                     colors: np.ndarray = None,
                                     mesh: trimesh.Trimesh = None,
                                     title: str = "Dense Reconstruction"):
        """
        Visualize dense reconstruction results
        
        Args:
            points: 3D points to display
            colors: Point colors
            mesh: Mesh to display
            title: Plot title
        """
        fig = plt.figure(figsize=(15, 5))
        
        # Point cloud visualization
        if points is not None:
            ax1 = fig.add_subplot(131, projection='3d')
            
            if points.shape[0] == 3:
                points = points.T
            
            if colors is not None:
                if colors.shape[0] == 3:
                    colors = colors.T
                if colors.max() > 1:
                    colors = colors / 255.0
                ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=colors, s=1, alpha=0.6)
            else:
                ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          s=1, alpha=0.6)
            
            ax1.set_title("Dense Point Cloud")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")
        
        # Mesh visualization
        if mesh is not None:
            ax2 = fig.add_subplot(132, projection='3d')
            
            vertices = mesh.vertices
            faces = mesh.faces
            
            # Plot mesh wireframe
            for face in faces[::10]:  # Subsample for performance
                triangle = vertices[face]
                ax2.plot3D(*triangle.T, 'b-', alpha=0.3, linewidth=0.5)
            
            ax2.set_title("Generated Mesh")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_zlabel("Z")
        
        # Statistics
        if points is not None or mesh is not None:
            ax3 = fig.add_subplot(133)
            ax3.axis('off')
            
            stats_text = f"Dense Reconstruction Statistics\n\n"
            
            if points is not None:
                stats_text += f"Point Cloud:\n"
                stats_text += f"  - Points: {len(points):,}\n"
                stats_text += f"  - Bounds: [{points.min():.2f}, {points.max():.2f}]\n\n"
            
            if mesh is not None:
                stats_text += f"Mesh:\n"
                stats_text += f"  - Vertices: {len(mesh.vertices):,}\n"
                stats_text += f"  - Faces: {len(mesh.faces):,}\n"
                stats_text += f"  - Volume: {mesh.volume:.3f}\n"
                stats_text += f"  - Surface Area: {mesh.area:.3f}\n"
            
            ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

# Complete dense reconstruction pipeline
def complete_dense_reconstruction_pipeline(sparse_reconstruction: Dict,
                                        images: List[np.ndarray],
                                        reconstruction_params: Dict = None) -> Dict:
    """
    Complete dense reconstruction pipeline
    
    Args:
        sparse_reconstruction: Results from sparse reconstruction
        images: List of input images
        reconstruction_params: Parameters for dense reconstruction
        
    Returns:
        Dense reconstruction results
    """
    if reconstruction_params is None:
        reconstruction_params = {
            'stereo_method': 'multi_view',
            'fusion_method': 'weighted_average',
            'mesh_method': 'poisson',
            'filter_point_cloud': True,
            'apply_texture': True,
            'poisson_depth': 9
        }
    
    print("Starting dense reconstruction pipeline...")
    
    # Extract sparse reconstruction data
    camera_matrix = sparse_reconstruction['camera_matrix']
    sparse_points = sparse_reconstruction['points_3d']
    
    # Camera poses (first camera at origin, second camera relative)
    camera_poses = [
        {'R': np.eye(3), 't': np.zeros((3, 1))},
        {'R': sparse_reconstruction['rotation'], 't': sparse_reconstruction['translation']}
    ]
    
    # Initialize dense reconstructor
    dense_recon = DenseReconstruction([camera_matrix, camera_matrix])
    
    print(f"Input: {len(images)} images, {sparse_points.shape[1] if sparse_points is not None else 0} sparse points")
    
    # Step 1: Compute depth maps
    print("\n" + "="*50)
    print("STEP 1: Dense Stereo Matching")
    print("="*50)
    
    if reconstruction_params['stereo_method'] == 'multi_view':
        depth_maps = dense_recon.multi_view_stereo(
            images, camera_poses, camera_matrix, reference_view=0
        )
    else:
        # Simple two-view stereo
        depth_map, disparity = dense_recon.compute_stereo_depth(
            images[0], images[1], camera_matrix, 
            camera_poses[1]['R'], camera_poses[1]['t']
        )
        depth_maps = {1: {'depth': depth_map, 'disparity': disparity}}
    
    if not depth_maps:
        return {'error': 'Failed to compute depth maps'}
    
    # Step 2: Fuse depth maps
    print("\n" + "="*50)
    print("STEP 2: Depth Map Fusion")
    print("="*50)
    
    if len(depth_maps) > 1:
        fused_depth = dense_recon.fuse_depth_maps(
            depth_maps, reconstruction_params['fusion_method']
        )
    else:
        fused_depth = list(depth_maps.values())[0]['depth']
    
    valid_depth_pixels = np.sum(~np.isnan(fused_depth))
    print(f"Fused depth map: {valid_depth_pixels:,} valid pixels")
    
    # Step 3: Generate dense point cloud
    print("\n" + "="*50)
    print("STEP 3: Dense Point Cloud Generation")
    print("="*50)
    
    dense_points, point_colors = dense_recon.depth_map_to_point_cloud(
        fused_depth, images[0], camera_matrix, camera_poses[0]
    )
    
    print(f"Generated {dense_points.shape[1]:,} dense points")
    
    # Step 4: Filter point cloud
    if reconstruction_params.get('filter_point_cloud', True):
        print("\n" + "="*50)
        print("STEP 4: Point Cloud Filtering")
        print("="*50)
        
        filtered_points, filtered_colors = dense_recon.filter_point_cloud(
            dense_points, point_colors
        )
        dense_points = filtered_points.T
        point_colors = filtered_colors.T if filtered_colors is not None else None
    
    # Step 5: Mesh generation
    print("\n" + "="*50)
    print("STEP 5: Mesh Generation")
    print("="*50)
    
    if reconstruction_params['mesh_method'] == 'poisson':
        mesh = dense_recon.create_mesh_poisson(
            dense_points.T, depth=reconstruction_params.get('poisson_depth', 9)
        )
    elif reconstruction_params['mesh_method'] == 'delaunay':
        mesh = dense_recon.create_mesh_delaunay(dense_points.T)
    else:
        mesh = None
        print("Skipping mesh generation")
    
    # Step 6: Apply texture
    if reconstruction_params.get('apply_texture', True) and mesh is not None:
        print("\n" + "="*50)
        print("STEP 6: Texture Mapping")
        print("="*50)
        
        textured_mesh = dense_recon.texture_mesh(
            mesh, images, camera_poses, camera_matrix
        )
    else:
        textured_mesh = mesh
    
    # Step 7: Visualization and results
    print("\n" + "="*50)
    print("STEP 7: Results and Visualization")
    print("="*50)
    
    # Calculate statistics
    stats = {
        'sparse_points': sparse_points.shape[1] if sparse_points is not None else 0,
        'dense_points': dense_points.shape[1],
        'valid_depth_pixels': int(valid_depth_pixels),
        'depth_map_coverage': valid_depth_pixels / (fused_depth.shape[0] * fused_depth.shape[1]),
        'point_density_increase': dense_points.shape[1] / max(1, sparse_points.shape[1] if sparse_points is not None else 1)
    }
    
    if textured_mesh is not None:
        stats.update({
            'mesh_vertices': len(textured_mesh.vertices),
            'mesh_faces': len(textured_mesh.faces),
            'mesh_volume': float(textured_mesh.volume),
            'mesh_surface_area': float(textured_mesh.area)
        })
    
    print(f"Dense reconstruction completed!")
    print(f"  - Sparse points: {stats['sparse_points']:,}")
    print(f"  - Dense points: {stats['dense_points']:,}")
    print(f"  - Density increase: {stats['point_density_increase']:.1f}x")
    print(f"  - Depth coverage: {stats['depth_map_coverage']:.1%}")
    
    if textured_mesh is not None:
        print(f"  - Mesh vertices: {stats['mesh_vertices']:,}")
        print(f"  - Mesh faces: {stats['mesh_faces']:,}")
    
    # Visualize results
    dense_recon.visualize_dense_reconstruction(
        dense_points, point_colors, textured_mesh, 
        "Dense Reconstruction Results"
    )
    
    # Compile results
    results = {
        'success': True,
        'dense_points': dense_points,
        'point_colors': point_colors,
        'depth_maps': depth_maps,
        'fused_depth_map': fused_depth,
        'mesh': textured_mesh,
        'camera_poses': camera_poses,
        'camera_matrix': camera_matrix,
        'stats': stats,
        'parameters': reconstruction_params
    }
    
    return results

# Advanced mesh processing utilities
class MeshProcessing:
    """Advanced mesh processing and optimization"""
    
    @staticmethod
    def simplify_mesh(mesh: trimesh.Trimesh, target_faces: int = None, 
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
        print(f"Simplifying mesh from {len(mesh.faces)} faces...")
        
        if target_faces is None:
            target_faces = int(len(mesh.faces) * reduction_ratio)
        
        # Use mesh decimation
        simplified = mesh.simplify_quadric_decimation(target_faces)
        
        print(f"Simplified to {len(simplified.faces)} faces")
        return simplified
    
    @staticmethod
    def smooth_mesh(mesh: trimesh.Trimesh, iterations: int = 5) -> trimesh.Trimesh:
        """
        Smooth mesh using Laplacian smoothing
        
        Args:
            mesh: Input mesh
            iterations: Number of smoothing iterations
            
        Returns:
            Smoothed mesh
        """
        print(f"Smoothing mesh with {iterations} iterations...")
        
        # Apply Laplacian smoothing
        smoothed = mesh.smoothed(iterations=iterations)
        
        return smoothed
    
    @staticmethod
    def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Repair mesh by fixing common issues
        
        Args:
            mesh: Input mesh
            
        Returns:
            Repaired mesh
        """
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
    
    @staticmethod
    def create_mesh_from_points_advanced(points: np.ndarray, 
                                       method: str = 'ball_pivoting',
                                       **kwargs) -> trimesh.Trimesh:
        """
        Create mesh from points using advanced algorithms
        
        Args:
            points: 3D points (N, 3)
            method: Reconstruction method ('ball_pivoting', 'alpha_shape', 'convex_hull')
            **kwargs: Method-specific parameters
            
        Returns:
            Generated mesh
        """
        if points.shape[0] == 3:
            points = points.T
        
        print(f"Creating mesh from {len(points)} points using {method}...")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()
        
        if method == 'ball_pivoting':
            # Ball pivoting algorithm
            radii = kwargs.get('radii', [0.005, 0.01, 0.02, 0.04])
            mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
        
        elif method == 'alpha_shape':
            # Alpha shape
            alpha = kwargs.get('alpha', 0.03)
            mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha
            )
        
        elif method == 'convex_hull':
            # Convex hull
            hull, _ = pcd.compute_convex_hull()
            mesh_o3d = hull
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert to trimesh
        if len(mesh_o3d.vertices) > 0:
            vertices = np.asarray(mesh_o3d.vertices)
            faces = np.asarray(mesh_o3d.triangles)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            print(f"Generated mesh: {len(vertices)} vertices, {len(faces)} faces")
            return mesh
        else:
            print("Failed to generate mesh")
            return None

# Quality assessment for dense reconstruction
class ReconstructionQuality:
    """Quality assessment for dense reconstruction results"""
    
    @staticmethod
    def assess_point_cloud_quality(points: np.ndarray, colors: np.ndarray = None) -> Dict:
        """
        Assess quality of point cloud
        
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
    
    @staticmethod
    def assess_mesh_quality(mesh: trimesh.Trimesh) -> Dict:
        """
        Assess quality of mesh
        
        Args:
            mesh: Input mesh
            
        Returns:
            Quality metrics dictionary
        """
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
        
        return metrics
    
    @staticmethod
    def compare_reconstructions(sparse_points: np.ndarray, 
                              dense_points: np.ndarray,
                              mesh: trimesh.Trimesh = None) -> Dict:
        """
        Compare sparse and dense reconstruction quality
        
        Args:
            sparse_points: Sparse 3D points
            dense_points: Dense 3D points  
            mesh: Optional mesh
            
        Returns:
            Comparison metrics
        """
        comparison = {}
        
        # Point count comparison
        sparse_count = sparse_points.shape[1] if sparse_points.shape[0] == 3 else sparse_points.shape[0]
        dense_count = dense_points.shape[1] if dense_points.shape[0] == 3 else dense_points.shape[0]
        
        comparison['point_counts'] = {
            'sparse': sparse_count,
            'dense': dense_count,
            'density_increase': dense_count / max(1, sparse_count)
        }
        
        # Coverage comparison
        if sparse_points.shape[0] == 3:
            sparse_points = sparse_points.T
        if dense_points.shape[0] == 3:
            dense_points = dense_points.T
        
        sparse_bounds = [sparse_points.min(axis=0), sparse_points.max(axis=0)]
        dense_bounds = [dense_points.min(axis=0), dense_points.max(axis=0)]
        
        sparse_volume = np.prod(sparse_bounds[1] - sparse_bounds[0])
        dense_volume = np.prod(dense_bounds[1] - dense_bounds[0])
        
        comparison['coverage'] = {
            'sparse_bounding_volume': float(sparse_volume),
            'dense_bounding_volume': float(dense_volume),
            'volume_ratio': dense_volume / max(sparse_volume, 1e-10)
        }
        
        # Mesh comparison
        if mesh is not None:
            mesh_metrics = ReconstructionQuality.assess_mesh_quality(mesh)
            comparison['mesh'] = mesh_metrics
        
        return comparison

# Export utilities
class ReconstructionExporter:
    """Export reconstruction results to various formats"""
    
    @staticmethod
    def export_point_cloud(points: np.ndarray, colors: np.ndarray = None,
                         filename: str = "point_cloud.ply") -> bool:
        """
        Export point cloud to PLY format
        
        Args:
            points: 3D points (N, 3) or (3, N)
            colors: Optional colors (N, 3) or (3, N)
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
    
    @staticmethod
    def export_mesh(mesh: trimesh.Trimesh, filename: str = "mesh.obj") -> bool:
        """
        Export mesh to file
        
        Args:
            mesh: Input mesh
            filename: Output filename
            
        Returns:
            Success status
        """
        try:
            mesh.export(filename)
            print(f"Mesh exported to {filename}")
            return True
            
        except Exception as e:
            print(f"Error exporting mesh: {e}")
            return False
    
    @staticmethod
    def create_reconstruction_report(results: Dict, 
                                   output_file: str = "reconstruction_report.txt") -> bool:
        """
        Create detailed reconstruction report
        
        Args:
            results: Dense reconstruction results
            output_file: Output report filename
            
        Returns:
            Success status
        """
        try:
            with open(output_file, 'w') as f:
                f.write("DENSE RECONSTRUCTION REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # General information
                f.write("GENERAL INFORMATION\n")
                f.write("-" * 20 + "\n")
                f.write(f"Success: {results.get('success', False)}\n")
                f.write(f"Processing Parameters: {results.get('parameters', {})}\n\n")
                
                # Statistics
                if 'stats' in results:
                    stats = results['stats']
                    f.write("RECONSTRUCTION STATISTICS\n")
                    f.write("-" * 25 + "\n")
                    for key, value in stats.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
                
                # Quality assessment
                if 'dense_points' in results:
                    points = results['dense_points']
                    colors = results.get('point_colors')
                    
                    quality = ReconstructionQuality.assess_point_cloud_quality(points, colors)
                    f.write("POINT CLOUD QUALITY\n")
                    f.write("-" * 20 + "\n")
                    for key, value in quality.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
                
                # Mesh quality
                if 'mesh' in results and results['mesh'] is not None:
                    mesh_quality = ReconstructionQuality.assess_mesh_quality(results['mesh'])
                    f.write("MESH QUALITY\n")
                    f.write("-" * 12 + "\n")
                    for key, value in mesh_quality.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
                
                f.write("Report generated successfully.\n")
            
            print(f"Reconstruction report saved to {output_file}")
            return True
            
        except Exception as e:
            print(f"Error creating report: {e}")
            return False

# Example usage and integration
def main():
    """Example usage of dense reconstruction"""
    
    print("Testing dense reconstruction pipeline...")
    
    # This would typically come from your sparse reconstruction
    # For testing, we'll create some synthetic data
    test_dense_reconstruction()

def test_dense_reconstruction():
    """Test dense reconstruction with synthetic data"""
    
    print("Creating synthetic test data...")
    
    # Create synthetic images and sparse reconstruction results
    # (In practice, these would come from your previous pipeline)
    
    # Synthetic camera parameters
    width, height = 640, 480
    focal_length = width * 1.2
    camera_matrix = np.array([
        [focal_length, 0, width/2],
        [0, focal_length, height/2],
        [0, 0, 1]
    ])
    
    # Synthetic sparse reconstruction results
    sparse_reconstruction = {
        'success': True,
        'camera_matrix': camera_matrix,
        'rotation': np.array([[0.9, -0.1, 0.1], [0.1, 0.95, 0.05], [-0.1, 0.05, 0.99]]),
        'translation': np.array([[0.5], [0.1], [0.2]]),
        'points_3d': np.random.randn(3, 100) * 2 + np.array([[0], [0], [5]])
    }
    
    # Create synthetic images (normally you'd have real images)
    print("Creating synthetic images...")
    img1 = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    images = [img1, img2]
    
    # Test dense reconstruction
    print("Running dense reconstruction pipeline...")
    
    try:
        results = complete_dense_reconstruction_pipeline(
            sparse_reconstruction, images,
            reconstruction_params={
                'stereo_method': 'multi_view',
                'fusion_method': 'weighted_average', 
                'mesh_method': 'poisson',
                'filter_point_cloud': True,
                'apply_texture': True,
                'poisson_depth': 7  # Lower depth for faster processing
            }
        )
        
        if results.get('success'):
            print("\nDense reconstruction successful!")
            
            # Export results
            print("Exporting results...")
            exporter = ReconstructionExporter()
            
            # Export point cloud
            if 'dense_points' in results:
                exporter.export_point_cloud(
                    results['dense_points'], 
                    results.get('point_colors'),
                    "dense_point_cloud.ply"
                )
            
            # Export mesh
            if 'mesh' in results and results['mesh'] is not None:
                exporter.export_mesh(results['mesh'], "reconstruction_mesh.obj")
            
            # Create report
            exporter.create_reconstruction_report(results)
            
            print("Dense reconstruction test completed successfully!")
            
        else:
            print("Dense reconstruction failed:", results.get('error'))
            
    except Exception as e:
        print(f"Error in dense reconstruction test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()