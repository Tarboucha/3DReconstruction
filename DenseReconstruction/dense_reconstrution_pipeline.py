"""
Dense Reconstruction Pipeline - Main Module
==========================================

This module orchestrates the complete dense reconstruction pipeline,
integrating stereo matching, point cloud processing, and mesh generation.

Author: Photogrammetry Pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from stereo_matching import StereoMatcher
from point_cloud_processing import PointCloudProcessor
from mesh_generation import MeshGenerator

class DenseReconstructionPipeline:
    """Complete dense reconstruction pipeline"""
    
    def __init__(self):
        self.stereo_matcher = StereoMatcher()
        self.point_cloud_processor = PointCloudProcessor()
        self.mesh_generator = MeshGenerator()
        
        # Store intermediate results
        self.depth_maps = {}
        self.fused_depth_map = None
        self.dense_points = None
        self.point_colors = None
        self.final_mesh = None
        
    def run_complete_pipeline(self, sparse_reconstruction: Dict,
                            images: List[np.ndarray],
                            reconstruction_params: Dict = None) -> Dict:
        """
        Run the complete dense reconstruction pipeline
        
        Args:
            sparse_reconstruction: Results from sparse reconstruction
            images: List of input images
            reconstruction_params: Parameters for dense reconstruction
            
        Returns:
            Complete reconstruction results
        """
        if reconstruction_params is None:
            reconstruction_params = self._get_default_params()
        
        print("="*60)
        print("DENSE RECONSTRUCTION PIPELINE")
        print("="*60)
        
        # Validate inputs
        if not self._validate_inputs(sparse_reconstruction, images):
            return {'error': 'Invalid inputs'}
        
        # Extract sparse reconstruction data
        camera_matrix = sparse_reconstruction['camera_matrix']
        sparse_points = sparse_reconstruction['points_3d']
        
        # Camera poses (first camera at origin, second camera relative)
        camera_poses = [
            {'R': np.eye(3), 't': np.zeros((3, 1))},
            {'R': sparse_reconstruction['rotation'], 't': sparse_reconstruction['translation']}
        ]
        
        print(f"Input: {len(images)} images, {sparse_points.shape[1] if sparse_points is not None else 0} sparse points")
        
        # Step 1: Dense stereo matching
        print("\n" + "="*50)
        print("STEP 1: Dense Stereo Matching")
        print("="*50)
        
        stereo_results = self._run_stereo_matching(
            images, camera_poses, camera_matrix, reconstruction_params
        )
        
        if not stereo_results['success']:
            return stereo_results
        
        # Step 2: Point cloud generation
        print("\n" + "="*50)
        print("STEP 2: Point Cloud Generation")
        print("="*50)
        
        point_cloud_results = self._run_point_cloud_generation(
            stereo_results, images, camera_poses, camera_matrix, reconstruction_params
        )
        
        if not point_cloud_results['success']:
            return point_cloud_results
        
        # Step 3: Mesh generation
        print("\n" + "="*50)
        print("STEP 3: Mesh Generation")
        print("="*50)
        
        mesh_results = self._run_mesh_generation(
            point_cloud_results, images, camera_poses, camera_matrix, reconstruction_params
        )
        
        if not mesh_results['success']:
            return mesh_results
        
        # Step 4: Final processing and results
        print("\n" + "="*50)
        print("STEP 4: Final Processing")
        print("="*50)
        
        final_results = self._compile_final_results(
            stereo_results, point_cloud_results, mesh_results, 
            sparse_reconstruction, reconstruction_params
        )
        
        # Visualization
        if reconstruction_params.get('visualize', True):
            self._visualize_results(final_results)
        
        return final_results
    
    def _get_default_params(self) -> Dict:
        """Get default reconstruction parameters"""
        return {
            'stereo_method': 'multi_view',
            'fusion_method': 'weighted_average',
            'mesh_method': 'poisson',
            'filter_point_cloud': True,
            'apply_texture': True,
            'poisson_depth': 9,
            'visualize': True,
            'export_results': True,
            'stereo_params': {
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
            },
            'filter_params': {
                'statistical_outlier_removal': True,
                'radius_outlier_removal': True,
                'voxel_downsampling': True,
                'statistical_nb_neighbors': 20,
                'statistical_std_ratio': 2.0,
                'radius_outlier_nb_points': 16,
                'radius_outlier_radius': 0.05,
                'voxel_size': 0.01
            }
        }
    
    def _validate_inputs(self, sparse_reconstruction: Dict, images: List[np.ndarray]) -> bool:
        """Validate input parameters"""
        if not sparse_reconstruction.get('success', False):
            print("Error: Sparse reconstruction failed")
            return False
        
        if len(images) < 2:
            print("Error: Need at least 2 images")
            return False
        
        required_keys = ['camera_matrix', 'rotation', 'translation']
        for key in required_keys:
            if key not in sparse_reconstruction:
                print(f"Error: Missing {key} in sparse reconstruction")
                return False
        
        return True
    
    def _run_stereo_matching(self, images: List[np.ndarray], 
                           camera_poses: List[Dict],
                           camera_matrix: np.ndarray,
                           params: Dict) -> Dict:
        """Run stereo matching step"""
        
        if params['stereo_method'] == 'multi_view':
            self.depth_maps = self.stereo_matcher.multi_view_stereo(
                images, camera_poses, camera_matrix, reference_view=0
            )
        else:
            # Simple two-view stereo
            depth_map, disparity = self.stereo_matcher.compute_stereo_depth(
                images[0], images[1], camera_matrix, 
                camera_poses[1]['R'], camera_poses[1]['t'],
                params.get('stereo_params', {})
            )
            self.depth_maps = {1: {'depth': depth_map, 'disparity': disparity}}
        
        if not self.depth_maps:
            return {'success': False, 'error': 'Failed to compute depth maps'}
        
        # Fuse depth maps
        if len(self.depth_maps) > 1:
            self.fused_depth_map = self.stereo_matcher.fuse_depth_maps(
                self.depth_maps, params['fusion_method']
            )
        else:
            self.fused_depth_map = list(self.depth_maps.values())[0]['depth']
        
        valid_depth_pixels = np.sum(~np.isnan(self.fused_depth_map))
        print(f"Fused depth map: {valid_depth_pixels:,} valid pixels")
        
        # Get depth statistics
        depth_stats = self.stereo_matcher.get_depth_statistics(self.depth_maps)
        
        return {
            'success': True,
            'depth_maps': self.depth_maps,
            'fused_depth_map': self.fused_depth_map,
            'valid_pixels': valid_depth_pixels,
            'depth_stats': depth_stats
        }
    
    def _run_point_cloud_generation(self, stereo_results: Dict,
                                   images: List[np.ndarray],
                                   camera_poses: List[Dict],
                                   camera_matrix: np.ndarray,
                                   params: Dict) -> Dict:
        """Run point cloud generation step"""
        
        # Generate dense point cloud
        self.dense_points, self.point_colors = self.point_cloud_processor.depth_map_to_point_cloud(
            self.fused_depth_map, images[0], camera_matrix, camera_poses[0]
        )
        
        print(f"Generated {self.dense_points.shape[1]:,} dense points")
        
        # Filter point cloud if requested
        if params.get('filter_point_cloud', True):
            filtered_points, filtered_colors = self.point_cloud_processor.filter_point_cloud(
                self.dense_points, self.point_colors, params.get('filter_params', {})
            )
            
            self.dense_points = filtered_points.T
            self.point_colors = filtered_colors.T if filtered_colors is not None else None
        
        # Compute normals
        normals = self.point_cloud_processor.compute_point_cloud_normals(self.dense_points)
        
        # Analyze quality
        quality_metrics = self.point_cloud_processor.analyze_point_cloud_quality(
            self.dense_points, self.point_colors
        )
        
        return {
            'success': True,
            'dense_points': self.dense_points,
            'point_colors': self.point_colors,
            'normals': normals,
            'quality_metrics': quality_metrics
        }
    
    def _run_mesh_generation(self, point_cloud_results: Dict,
                           images: List[np.ndarray],
                           camera_poses: List[Dict],
                           camera_matrix: np.ndarray,
                           params: Dict) -> Dict:
        """Run mesh generation step"""
        
        # Generate mesh
        if params['mesh_method'] == 'poisson':
            self.final_mesh = self.mesh_generator.create_mesh_poisson(
                self.dense_points.T, 
                point_cloud_results['normals'],
                depth=params.get('poisson_depth', 9)
            )
        elif params['mesh_method'] == 'ball_pivoting':
            self.final_mesh = self.mesh_generator.create_mesh_ball_pivoting(
                self.dense_points.T, 
                point_cloud_results['normals']
            )
        elif params['mesh_method'] == 'alpha_shape':
            self.final_mesh = self.mesh_generator.create_mesh_alpha_shape(
                self.dense_points.T, 
                alpha=params.get('alpha', 0.03)
            )
        elif params['mesh_method'] == 'delaunay':
            self.final_mesh = self.mesh_generator.create_mesh_delaunay(
                self.dense_points.T
            )
        else:
            return {'success': False, 'error': f"Unknown mesh method: {params['mesh_method']}"}
        
        if self.final_mesh is None:
            return {'success': False, 'error': 'Failed to generate mesh'}
        
        # Repair mesh
        self.final_mesh = self.mesh_generator.repair_mesh(self.final_mesh)
        
        # Apply texture if requested
        if params.get('apply_texture', True):
            self.final_mesh = self.mesh_generator.texture_mesh(
                self.final_mesh, images, camera_poses, camera_matrix
            )
        
        # Analyze mesh quality
        mesh_quality = self.mesh_generator.analyze_mesh_quality(self.final_mesh)
        
        return {
            'success': True,
            'mesh': self.final_mesh,
            'mesh_quality': mesh_quality
        }
    
    def _compile_final_results(self, stereo_results: Dict, 
                             point_cloud_results: Dict,
                             mesh_results: Dict,
                             sparse_reconstruction: Dict,
                             params: Dict) -> Dict:
        """Compile final results"""
        
        # Calculate comprehensive statistics
        stats = {
            'sparse_points': sparse_reconstruction['points_3d'].shape[1] if sparse_reconstruction['points_3d'] is not None else 0,
            'dense_points': self.dense_points.shape[1],
            'valid_depth_pixels': stereo_results['valid_pixels'],
            'depth_map_coverage': stereo_results['valid_pixels'] / (self.fused_depth_map.shape[0] * self.fused_depth_map.shape[1]),
            'point_density_increase': self.dense_points.shape[1] / max(1, sparse_reconstruction['points_3d'].shape[1] if sparse_reconstruction['points_3d'] is not None else 1)
        }
        
        if self.final_mesh is not None:
            stats.update({
                'mesh_vertices': len(self.final_mesh.vertices),
                'mesh_faces': len(self.final_mesh.faces),
                'mesh_volume': float(self.final_mesh.volume) if self.final_mesh.is_volume else 0.0,
                'mesh_surface_area': float(self.final_mesh.area)
            })
        
        print(f"Dense reconstruction completed!")
        print(f"  - Sparse points: {stats['sparse_points']:,}")
        print(f"  - Dense points: {stats['dense_points']:,}")
        print(f"  - Density increase: {stats['point_density_increase']:.1f}x")
        print(f"  - Depth coverage: {stats['depth_map_coverage']:.1%}")
        
        if self.final_mesh is not None:
            print(f"  - Mesh vertices: {stats['mesh_vertices']:,}")
            print(f"  - Mesh faces: {stats['mesh_faces']:,}")
        
        # Compile complete results
        results = {
            'success': True,
            'dense_points': self.dense_points,
            'point_colors': self.point_colors,
            'depth_maps': self.depth_maps,
            'fused_depth_map': self.fused_depth_map,
            'mesh': self.final_mesh,
            'stats': stats,
            'parameters': params,
            'detailed_results': {
                'stereo': stereo_results,
                'point_cloud': point_cloud_results,
                'mesh': mesh_results
            }
        }
        
        return results
    
    def _visualize_results(self, results: Dict):
        """Visualize reconstruction results"""
        
        print("\nGenerating visualizations...")
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Depth map visualization
        ax1 = fig.add_subplot(2, 3, 1)
        depth_map = results['fused_depth_map']
        valid_depths = depth_map[~np.isnan(depth_map)]
        
        if len(valid_depths) > 0:
            depth_filtered = np.copy(depth_map)
            depth_min, depth_max = np.percentile(valid_depths, [5, 95])
            depth_filtered[depth_filtered < depth_min] = np.nan
            depth_filtered[depth_filtered > depth_max] = np.nan
            
            im1 = ax1.imshow(depth_filtered, cmap='jet')
            ax1.set_title('Fused Depth Map')
            plt.colorbar(im1, ax=ax1, label='Depth')
        
        # 2. Point cloud visualization
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        points = results['dense_points']
        colors = results['point_colors']
        
        if points is not None:
            if points.shape[0] == 3:
                points = points.T
            
            # Subsample for visualization
            step = max(1, len(points) // 5000)
            points_sub = points[::step]
            
            if colors is not None:
                if colors.shape[0] == 3:
                    colors = colors.T
                colors_sub = colors[::step]
                if colors_sub.max() > 1:
                    colors_sub = colors_sub / 255.0
                ax2.scatter(points_sub[:, 0], points_sub[:, 1], points_sub[:, 2], 
                          c=colors_sub, s=1, alpha=0.6)
            else:
                ax2.scatter(points_sub[:, 0], points_sub[:, 1], points_sub[:, 2], 
                          s=1, alpha=0.6)
        
        ax2.set_title('Dense Point Cloud')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # 3. Mesh visualization
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        mesh = results['mesh']
        
        if mesh is not None:
            vertices = mesh.vertices
            faces = mesh.faces
            
            # Plot mesh wireframe (subsample for performance)
            face_step = max(1, len(faces) // 500)
            for i in range(0, len(faces), face_step):
                face = faces[i]
                triangle = vertices[face]
                triangle_closed = np.vstack([triangle, triangle[0]])
                ax3.plot3D(*triangle_closed.T, 'b-', alpha=0.3, linewidth=0.5)
        
        ax3.set_title('Generated Mesh')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        # 4. Depth statistics
        ax4 = fig.add_subplot(2, 3, 4)
        if len(valid_depths) > 0:
            ax4.hist(valid_depths, bins=50, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Depth')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Depth Distribution')
        
        # 5. Point density analysis
        ax5 = fig.add_subplot(2, 3, 5)
        if points is not None:
            # Project to XY plane for density visualization
            hist, xedges, yedges = np.histogram2d(points[:, 0], points[:, 1], bins=50)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im5 = ax5.imshow(hist.T, extent=extent, origin='lower', cmap='hot')
            ax5.set_title('Point Density (XY Projection)')
            ax5.set_xlabel('X')
            ax5.set_ylabel('Y')
            plt.colorbar(im5, ax=ax5, label='Point Count')
        
        # 6. Statistics summary
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        stats = results['stats']
        stats_text = f"Dense Reconstruction Statistics\n\n"
        
        # Basic statistics
        stats_text += f"Points:\n"
        stats_text += f"  Sparse: {stats['sparse_points']:,}\n"
        stats_text += f"  Dense: {stats['dense_points']:,}\n"
        stats_text += f"  Density increase: {stats['point_density_increase']:.1f}x\n\n"
        
        # Depth map statistics
        stats_text += f"Depth Map:\n"
        stats_text += f"  Valid pixels: {stats['valid_depth_pixels']:,}\n"
        stats_text += f"  Coverage: {stats['depth_map_coverage']:.1%}\n"
        
        if len(valid_depths) > 0:
            stats_text += f"  Depth range: {np.min(valid_depths):.2f} - {np.max(valid_depths):.2f}\n"
            stats_text += f"  Mean depth: {np.mean(valid_depths):.2f}\n\n"
        
        # Mesh statistics
        if mesh is not None:
            stats_text += f"Mesh:\n"
            stats_text += f"  Vertices: {stats['mesh_vertices']:,}\n"
            stats_text += f"  Faces: {stats['mesh_faces']:,}\n"
            stats_text += f"  Volume: {stats['mesh_volume']:.3f}\n"
            stats_text += f"  Surface area: {stats['mesh_surface_area']:.3f}\n"
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.suptitle('Dense Reconstruction Results', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def export_results(self, results: Dict, output_dir: str = "./output/") -> Dict:
        """Export reconstruction results to files"""
        
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        export_status = {}
        
        # Export point cloud
        if results['dense_points'] is not None:
            success = self.point_cloud_processor.export_point_cloud(
                results['dense_points'], 
                results['point_colors'],
                filename=os.path.join(output_dir, "dense_point_cloud.ply")
            )
            export_status['point_cloud'] = success
        
        # Export mesh
        if results['mesh'] is not None:
            success = self.mesh_generator.export_mesh(
                results['mesh'],
                filename=os.path.join(output_dir, "reconstruction_mesh.obj")
            )
            export_status['mesh'] = success
        
        # Export depth map
        if results['fused_depth_map'] is not None:
            depth_map = results['fused_depth_map']
            
            # Save as numpy array
            np.save(os.path.join(output_dir, "depth_map.npy"), depth_map)
            
            # Save as image (normalized)
            valid_depths = depth_map[~np.isnan(depth_map)]
            if len(valid_depths) > 0:
                depth_normalized = np.copy(depth_map)
                depth_min, depth_max = np.percentile(valid_depths, [1, 99])
                depth_normalized = (depth_normalized - depth_min) / (depth_max - depth_min)
                depth_normalized = np.clip(depth_normalized, 0, 1)
                depth_normalized[np.isnan(depth_map)] = 0
                
                import cv2
                depth_image = (depth_normalized * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, "depth_map.png"), depth_image)
            
            export_status['depth_map'] = True
        
        # Export statistics report
        report_path = os.path.join(output_dir, "reconstruction_report.txt")
        success = self._create_reconstruction_report(results, report_path)
        export_status['report'] = success
        
        print(f"Results exported to: {output_dir}")
        for item, status in export_status.items():
            print(f"  {item}: {'✓' if status else '✗'}")
        
        return export_status
    
    def _create_reconstruction_report(self, results: Dict, output_file: str) -> bool:
        """Create detailed reconstruction report"""
        
        try:
            with open(output_file, 'w') as f:
                f.write("DENSE RECONSTRUCTION REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # General information
                f.write("GENERAL INFORMATION\n")
                f.write("-" * 20 + "\n")
                f.write(f"Success: {results.get('success', False)}\n")
                f.write(f"Processing Parameters: {results.get('parameters', {})}\n\n")
                
                # Main statistics
                if 'stats' in results:
                    stats = results['stats']
                    f.write("MAIN STATISTICS\n")
                    f.write("-" * 15 + "\n")
                    for key, value in stats.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
                
                # Detailed results
                if 'detailed_results' in results:
                    detailed = results['detailed_results']
                    
                    # Stereo results
                    if 'stereo' in detailed:
                        f.write("STEREO MATCHING RESULTS\n")
                        f.write("-" * 23 + "\n")
                        stereo_stats = detailed['stereo'].get('depth_stats', {})
                        for key, value in stereo_stats.items():
                            f.write(f"{key}: {value}\n")
                        f.write("\n")
                    
                    # Point cloud results
                    if 'point_cloud' in detailed:
                        f.write("POINT CLOUD RESULTS\n")
                        f.write("-" * 19 + "\n")
                        pc_quality = detailed['point_cloud'].get('quality_metrics', {})
                        for key, value in pc_quality.items():
                            f.write(f"{key}: {value}\n")
                        f.write("\n")
                    
                    # Mesh results
                    if 'mesh' in detailed:
                        f.write("MESH RESULTS\n")
                        f.write("-" * 12 + "\n")
                        mesh_quality = detailed['mesh'].get('mesh_quality', {})
                        for key, value in mesh_quality.items():
                            f.write(f"{key}: {value}\n")
                        f.write("\n")
                
                f.write("Report generated successfully.\n")
            
            return True
            
        except Exception as e:
            print(f"Error creating report: {e}")
            return False

# Convenience function for easy usage
def run_dense_reconstruction(sparse_reconstruction: Dict,
                           images: List[np.ndarray],
                           **kwargs) -> Dict:
    """
    Convenience function to run dense reconstruction
    
    Args:
        sparse_reconstruction: Results from sparse reconstruction
        images: List of input images
        **kwargs: Additional parameters
        
    Returns:
        Dense reconstruction results
    """
    
    # Create default parameters and update with kwargs
    params = {
        'stereo_method': 'multi_view',
        'fusion_method': 'weighted_average',
        'mesh_method': 'poisson',
        'filter_point_cloud': True,
        'apply_texture': True,
        'poisson_depth': 9,
        'visualize': True,
        'export_results': True
    }
    params.update(kwargs)
    
    # Run pipeline
    pipeline = DenseReconstructionPipeline()
    results = pipeline.run_complete_pipeline(sparse_reconstruction, images, params)
    
    # Export results if requested
    if params.get('export_results', True) and results.get('success', False):
        pipeline.export_results(results)
    
    return results

# Example usage and testing
def test_dense_reconstruction_pipeline():
    """Test the complete dense reconstruction pipeline"""
    
    print("Testing dense reconstruction pipeline...")
    
    # Create synthetic sparse reconstruction results
    # (In practice, these would come from your camera pose estimation)
    
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
    
    # Create synthetic images
    print("Creating synthetic images...")
    img1 = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    images = [img1, img2]
    
    # Test pipeline
    print("Running dense reconstruction pipeline...")
    
    try:
        results = run_dense_reconstruction(
            sparse_reconstruction, 
            images,
            stereo_method='multi_view',
            fusion_method='weighted_average',
            mesh_method='poisson',
            filter_point_cloud=True,
            apply_texture=True,
            poisson_depth=7,  # Lower depth for faster processing
            visualize=True,
            export_results=True
        )
        
        if results.get('success'):
            print("\n" + "="*50)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*50)
            
            stats = results['stats']
            print(f"Dense points generated: {stats['dense_points']:,}")
            print(f"Density increase: {stats['point_density_increase']:.1f}x")
            
            if results['mesh'] is not None:
                print(f"Mesh vertices: {stats['mesh_vertices']:,}")
                print(f"Mesh faces: {stats['mesh_faces']:,}")
            
            print("\nCheck the ./output/ directory for exported files!")
            
        else:
            print("Dense reconstruction failed:", results.get('error'))
            
    except Exception as e:
        print(f"Error in dense reconstruction pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dense_reconstruction_pipeline()