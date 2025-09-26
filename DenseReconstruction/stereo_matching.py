"""
Stereo Matching Module
=====================

This module handles dense stereo matching and depth map computation
for photogrammetry applications.

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class StereoMatcher:
    """Dense stereo matching and depth computation"""
    
    def __init__(self, camera_matrices: List[np.ndarray] = None):
        """
        Initialize stereo matcher
        
        Args:
            camera_matrices: List of camera intrinsic matrices for each view
        """
        self.camera_matrices = camera_matrices or []
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
    
    def visualize_depth_map(self, depth_map: np.ndarray, title: str = "Depth Map"):
        """
        Visualize depth map
        
        Args:
            depth_map: Depth map to visualize
            title: Plot title
        """
        plt.figure(figsize=(12, 8))
        
        # Original depth map
        plt.subplot(2, 2, 1)
        plt.imshow(depth_map, cmap='jet')
        plt.title('Raw Depth Map')
        plt.colorbar(label='Depth (units)')
        
        # Filtered depth map (remove extreme values)
        valid_depths = depth_map[~np.isnan(depth_map)]
        if len(valid_depths) > 0:
            depth_filtered = np.copy(depth_map)
            depth_min, depth_max = np.percentile(valid_depths, [5, 95])
            depth_filtered[depth_filtered < depth_min] = np.nan
            depth_filtered[depth_filtered > depth_max] = np.nan
            
            plt.subplot(2, 2, 2)
            plt.imshow(depth_filtered, cmap='jet')
            plt.title('Filtered Depth Map')
            plt.colorbar(label='Depth (units)')
        
        # Depth histogram
        plt.subplot(2, 2, 3)
        if len(valid_depths) > 0:
            plt.hist(valid_depths, bins=50, alpha=0.7)
            plt.xlabel('Depth')
            plt.ylabel('Frequency')
            plt.title('Depth Distribution')
        
        # Statistics
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        stats_text = f"Depth Map Statistics\n\n"
        stats_text += f"Valid pixels: {np.sum(~np.isnan(depth_map)):,}\n"
        stats_text += f"Total pixels: {depth_map.size:,}\n"
        stats_text += f"Coverage: {np.sum(~np.isnan(depth_map))/depth_map.size:.1%}\n"
        
        if len(valid_depths) > 0:
            stats_text += f"Min depth: {np.min(valid_depths):.2f}\n"
            stats_text += f"Max depth: {np.max(valid_depths):.2f}\n"
            stats_text += f"Mean depth: {np.mean(valid_depths):.2f}\n"
            stats_text += f"Std depth: {np.std(valid_depths):.2f}\n"
        
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def get_depth_statistics(self, depth_maps: Dict[int, Dict]) -> Dict:
        """
        Get comprehensive statistics for depth maps
        
        Args:
            depth_maps: Dictionary of depth map data
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'num_depth_maps': len(depth_maps),
            'individual_stats': {},
            'overall_stats': {}
        }
        
        all_valid_pixels = 0
        all_pixels = 0
        all_baselines = []
        
        for view_id, depth_data in depth_maps.items():
            depth_map = depth_data['depth']
            valid_pixels = depth_data['valid_pixels']
            baseline = depth_data['baseline']
            
            individual_stats = {
                'valid_pixels': valid_pixels,
                'total_pixels': depth_map.size,
                'coverage': valid_pixels / depth_map.size,
                'baseline': baseline
            }
            
            # Depth range statistics
            valid_depths = depth_map[~np.isnan(depth_map)]
            if len(valid_depths) > 0:
                individual_stats.update({
                    'min_depth': float(np.min(valid_depths)),
                    'max_depth': float(np.max(valid_depths)),
                    'mean_depth': float(np.mean(valid_depths)),
                    'std_depth': float(np.std(valid_depths))
                })
            
            stats['individual_stats'][view_id] = individual_stats
            
            all_valid_pixels += valid_pixels
            all_pixels += depth_map.size
            all_baselines.append(baseline)
        
        # Overall statistics
        stats['overall_stats'] = {
            'total_valid_pixels': all_valid_pixels,
            'total_pixels': all_pixels,
            'average_coverage': all_valid_pixels / all_pixels if all_pixels > 0 else 0,
            'mean_baseline': np.mean(all_baselines) if all_baselines else 0,
            'baseline_range': [float(np.min(all_baselines)), float(np.max(all_baselines))] if all_baselines else [0, 0]
        }
        
        return stats

def test_with_images():    
    """Test stereo matching with synthetic data"""
    
    print("Testing stereo matching...")
    
    # Create synthetic stereo pair
    height, width = 480, 640
    
    # Create synthetic images with some texture
    img1 = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img2 = np.roll(img1, 10, axis=1)  # Simple horizontal shift
    
    # Add some noise
    noise = np.random.randint(-20, 20, img2.shape, dtype=np.int16)
    img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Camera parameters
    focal_length = width * 1.2
    camera_matrix = np.array([
        [focal_length, 0, width/2],
        [0, focal_length, height/2],
        [0, 0, 1]
    ])
    
    # Camera pose (simple baseline)
    R = np.eye(3)
    t = np.array([[0.1], [0.0], [0.0]])  # 10cm baseline
    
    # Initialize stereo matcher
    stereo_matcher = StereoMatcher([camera_matrix, camera_matrix])
    
    # Compute depth map
    depth_map, disparity = stereo_matcher.compute_stereo_depth(
        img1, img2, camera_matrix, R, t
    )
    
    # Visualize results
    stereo_matcher.visualize_depth_map(depth_map, "Test Stereo Depth Map")
    
    print("Stereo matching test completed!")

# Example usage and testing
def test_stereo_matching():
    """Test stereo matching with synthetic data"""
    
    print("Testing stereo matching...")
    
    # Create synthetic stereo pair
    height, width = 480, 640
    
    # Create synthetic images with some texture
    img1 = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img2 = np.roll(img1, 10, axis=1)  # Simple horizontal shift
    
    # Add some noise
    noise = np.random.randint(-20, 20, img2.shape, dtype=np.int16)
    img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Camera parameters
    focal_length = width * 1.2
    camera_matrix = np.array([
        [focal_length, 0, width/2],
        [0, focal_length, height/2],
        [0, 0, 1]
    ])
    
    # Camera pose (simple baseline)
    R = np.eye(3)
    t = np.array([[0.1], [0.0], [0.0]])  # 10cm baseline
    
    # Initialize stereo matcher
    stereo_matcher = StereoMatcher([camera_matrix, camera_matrix])
    
    # Compute depth map
    depth_map, disparity = stereo_matcher.compute_stereo_depth(
        img1, img2, camera_matrix, R, t
    )
    
    # Visualize results
    stereo_matcher.visualize_depth_map(depth_map, "Test Stereo Depth Map")
    
    print("Stereo matching test completed!")

if __name__ == "__main__":
    test_stereo_matching()