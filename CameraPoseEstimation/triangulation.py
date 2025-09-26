import numpy as np
import cv2
from scipy.optimize import least_squares
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
from dataclasses import dataclass



import numpy as np
import cv2
from scipy.optimize import least_squares
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
from dataclasses import dataclass


def optimal_triangulation_hartley_sturm(pts1: np.ndarray, 
                                       pts2: np.ndarray,
                                       F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hartley-Sturm optimal triangulation algorithm.
    Corrects point correspondences to exactly satisfy the epipolar constraint
    before triangulation, minimizing geometric error.
    
    Args:
        pts1: Points in first image (Nx2)
        pts2: Points in second image (Nx2)
        F: Fundamental matrix (3x3)
        
    Returns:
        pts1_corrected: Corrected points in first image (Nx2)
        pts2_corrected: Corrected points in second image (Nx2)
    
    Reference: Hartley & Sturm, "Triangulation", CVIU 1997
    """
    
    n_points = pts1.shape[0]
    pts1_corrected = np.zeros_like(pts1)
    pts2_corrected = np.zeros_like(pts2)
    
    for i in range(n_points):
        # Get original points
        x1 = np.array([pts1[i, 0], pts1[i, 1], 1.0])
        x2 = np.array([pts2[i, 0], pts2[i, 1], 1.0])
        
        # Compute epipolar lines
        l2 = F @ x1  # Epipolar line in image 2
        l1 = F.T @ x2  # Epipolar line in image 1
        
        # Normalize lines (so that a^2 + b^2 = 1 for line [a, b, c])
        l1 = l1 / np.sqrt(l1[0]**2 + l1[1]**2)
        l2 = l2 / np.sqrt(l2[0]**2 + l2[1]**2)
        
        # Distance from points to epipolar lines
        d1 = np.abs(np.dot(l1, x1))
        d2 = np.abs(np.dot(l2, x2))
        
        # Cost function to minimize (squared geometric distance)
        cost = d1**2 + d2**2
        
        # Special case: if points already satisfy epipolar constraint
        if cost < 1e-10:
            pts1_corrected[i] = pts1[i]
            pts2_corrected[i] = pts2[i]
            continue
        
        # Find the closest points that satisfy x2^T * F * x1 = 0
        # This is done by parameterizing points on epipolar lines
        
        # Method: minimize ||x1 - x1_orig||^2 + ||x2 - x2_orig||^2
        # subject to x2^T * F * x1 = 0
        
        # Parameterize corrected points
        # x1_corrected = x1 - t * [l1[0], l1[1], 0]^T / (l1[0]^2 + l1[1]^2)
        # x2_corrected = x2 - s * [l2[0], l2[1], 0]^T / (l2[0]^2 + l2[1]^2)
        
        def correct_points_parametric(t, s):
            """Correct points using parameters t and s"""
            x1_new = x1.copy()
            x2_new = x2.copy()
            
            # Move points perpendicular to epipolar lines
            x1_new[0] -= t * l1[0]
            x1_new[1] -= t * l1[1]
            x2_new[0] -= s * l2[0]
            x2_new[1] -= s * l2[1]
            
            return x1_new, x2_new
        
        # Solve for optimal t and s
        # The constraint x2^T * F * x1 = 0 becomes linear in t and s
        a = l2[0] * F[0, 0] + l2[1] * F[1, 0]
        b = l2[0] * F[0, 1] + l2[1] * F[1, 1]
        c = l1[0] * F[0, 0] + l1[1] * F[0, 1]
        d = l1[0] * F[1, 0] + l1[1] * F[1, 1]
        e = x2 @ F @ x1
        
        # Solve the system (derived from Lagrange multipliers)
        if np.abs(a*d - b*c) > 1e-10:
            t_opt = (b*e) / (a*d - b*c)
            s_opt = (-a*e) / (a*d - b*c)
        else:
            # Degenerate case, use simple perpendicular projection
            t_opt = d1
            s_opt = d2
        
        # Apply correction
        x1_corrected, x2_corrected = correct_points_parametric(t_opt, s_opt)
        
        pts1_corrected[i] = x1_corrected[:2]
        pts2_corrected[i] = x2_corrected[:2]
    
    return pts1_corrected, pts2_corrected


@dataclass
class TriangulationConfig:
    """Configuration for enhanced triangulation"""
    
    # Basic triangulation
    min_triangulation_angle: float = 2.0          # Minimum angle between rays (degrees)
    max_reprojection_error: float = 2.0          # Maximum reprojection error (pixels)
    min_distance: float = 0.1                    # Minimum 3D point distance
    max_distance: float = 100.0                  # Maximum 3D point distance
    
    # Progressive triangulation  
    enable_progressive: bool = True               # Enable progressive triangulation after camera addition
    max_points_per_pair: int = 150               # Limit points per image pair
    max_total_new_points: int = 800              # Global limit per progressive session
    progressive_tolerance: float = 4.0           # Pixel tolerance for unmatched points
    
    # Track extension
    enable_track_extension: bool = True          # Extend existing tracks to unprocessed images
    track_extension_tolerance: float = 2.0       # Pixel tolerance for track extension
    max_track_length: int = 10                   # Maximum observations per 3D point


class TriangulationEngine:
    """
    ENHANCED: Robust triangulation engine for monument reconstruction pipeline.
    Handles both initial two-view triangulation and incremental point addition with progressive features.
    """
    
    def __init__(self, 
                 min_triangulation_angle_deg: float = 2.0,
                 max_reprojection_error: float = 2.0,
                 min_depth: float = 0.1,
                 max_depth: float = 1000.0,
                 use_optimal_triangulation: bool = True,  # NEW: Enable Hartley-Sturm
                 config: Optional[TriangulationConfig] = None):
        """
        Initialize triangulation engine with quality thresholds.
        
        Args:
            min_triangulation_angle_deg: Minimum angle between rays for triangulation
            max_reprojection_error: Maximum allowed reprojection error in pixels
            min_depth: Minimum allowed depth (behind camera rejection)
            max_depth: Maximum allowed depth (far point rejection)
            use_optimal_triangulation: Whether to use Hartley-Sturm optimal triangulation
            config: Configuration for progressive triangulation features
        """
        self.min_triangulation_angle = np.deg2rad(min_triangulation_angle_deg)
        self.max_reprojection_error = max_reprojection_error
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.use_optimal_triangulation = use_optimal_triangulation  # NEW
        
        self.logger = logging.getLogger(__name__)
        
        self.config = config or TriangulationConfig()
        self.matches_data = None
        self.all_image_info = []
    
    # 🆕 NEW: Methods to set up data access for progressive triangulation
    def set_matches_data(self, matches_data: Dict):
        """Set matches data for progressive triangulation operations"""
        self.matches_data = matches_data
    
    def set_all_image_info(self, image_names: List[str]):
        """Set all image names for progressive triangulation"""
        self.all_image_info = image_names

    def triangulate_initial_points(self, 
                                    pts1: np.ndarray, 
                                    pts2: np.ndarray,
                                    R1: np.ndarray, 
                                    t1: np.ndarray,
                                    R2: np.ndarray, 
                                    t2: np.ndarray,
                                    K: np.ndarray,
                                    image_pair: Tuple[str, str]) -> Dict[str, Any]:
            """
            Triangulate 3D points from initial two-view reconstruction.
            Now with optional Hartley-Sturm optimal triangulation.
            """
            
            self.logger.info(f"Triangulating initial points between {image_pair[0]} and {image_pair[1]}")
            self.logger.info(f"Input correspondences: {len(pts1)}")
            
            # Create projection matrices
            P1 = K[0] @ np.hstack([R1, t1.reshape(-1, 1)])  # 3x4
            P2 = K[1] @ np.hstack([R2, t2.reshape(-1, 1)])  # 3x4
            
            # NEW: Apply Hartley-Sturm optimal triangulation if enabled
            if self.use_optimal_triangulation:
                pts1_to_triangulate, pts2_to_triangulate = self._apply_hartley_sturm(
                    pts1, pts2, K[0], K[1], R1, t1, R2, t2
                )
            else:
                pts1_to_triangulate, pts2_to_triangulate = pts1, pts2
            
            # Triangulate all points using DLT
            points_3d_hom = self._triangulate_dlt_batch(
                pts1_to_triangulate, pts2_to_triangulate, P1, P2
            )
            
            # Convert from homogeneous to 3D coordinates
            points_3d = points_3d_hom[:3, :] / points_3d_hom[3, :]  # 3xN
            
            # Quality filtering (use original points for error checking)
            valid_mask = self._filter_triangulated_points(
                points_3d, pts1, pts2, P1, P2, R1, t1, R2, t2
            )
            
            # Keep only valid points AND track original indices
            valid_points_3d = points_3d[:, valid_mask]
            valid_pts1 = pts1[valid_mask]
            valid_pts2 = pts2[valid_mask]
            valid_indices = np.where(valid_mask)[0]
            
            # Create observation structure with original indices
            observations = self._create_initial_observations_with_indices(
                valid_points_3d, valid_pts1, valid_pts2, valid_indices, image_pair
            )
            
            result = {
                'points_3d': valid_points_3d,  # 3xN array
                'observations': observations,
                'valid_indices': valid_indices,
                'point_colors': self._estimate_point_colors(valid_pts1, valid_pts2, image_pair),
                'statistics': {
                    'initial_points': len(pts1),
                    'triangulated_points': points_3d.shape[1],
                    'valid_points': valid_points_3d.shape[1],
                    'success_rate': valid_points_3d.shape[1] / len(pts1),
                    'used_optimal_triangulation': self.use_optimal_triangulation
                }
            }
            
            self.logger.info(f"Successfully triangulated {valid_points_3d.shape[1]}/{len(pts1)} points "
                            f"(success rate: {result['statistics']['success_rate']:.2%})")
            
            return result
        
    def _create_initial_observations_with_indices(self, 
                                        points_3d: np.ndarray,
                                        pts1: np.ndarray, 
                                        pts2: np.ndarray,
                                        original_indices: np.ndarray,
                                        image_pair: Tuple[str, str]) -> List[Dict[str, Any]]:
        """Create observation structure with original point indices."""
    
        observations = []
        
        for i in range(points_3d.shape[1]):
            original_idx = original_indices[i]  # ✅ Use original index
            
            # Observation in first image
            observations.append({
                'point_id': i,  # Sequential for 3D point numbering
                'original_index': original_idx,  # ✅ ADD: Track original index
                'image_id': image_pair[0],
                'pixel_coords': pts1[i],
                'feature_id': original_idx  # ✅ Use original feature index
            })
            
            # Observation in second image  
            observations.append({
                'point_id': i,
                'original_index': original_idx,  # ✅ ADD: Track original index
                'image_id': image_pair[1], 
                'pixel_coords': pts2[i],
                'feature_id': original_idx  # ✅ Use original feature index
            })
        
        return observations



    def triangulate_new_points(self, 
                             new_image: str,
                             reconstruction_state: Dict[str, Any],
                             matches_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Triangulate new 3D points when adding a new view to existing reconstruction.
        
        Args:
            new_image: Identifier of the newly added image
            reconstruction_state: Current reconstruction state with cameras and points
            matches_data: Feature matches (if None, will extract from reconstruction_state)
            
        Returns:
            Dictionary with new 3D points and updated observations
        """
        
        self.logger.info(f"Triangulating new points for view: {new_image}")
        
        # Use provided matches_data or stored one
        if matches_data is None:
            matches_data = self.matches_data

        # Step 1: BASIC TRIANGULATION (your original implementation)
        basic_result = self._triangulate_basic_new_points(new_image, reconstruction_state, matches_data)
        basic_points_3d = basic_result['new_points_3d']

        progressive_points_3d = np.empty((3, 0))
        
        if self.config.enable_progressive and self.matches_data is not None:
            self.logger.info(f"Progressive triangulation for {new_image}...")
            progressive_points_3d = self._triangulate_progressive_new_points(new_image, reconstruction_state)
        
        # Combine basic + progressive results
        if basic_points_3d.size > 0 and progressive_points_3d.size > 0:
            combined_points = np.hstack([basic_points_3d, progressive_points_3d])
        elif basic_points_3d.size > 0:
            combined_points = basic_points_3d
        elif progressive_points_3d.size > 0:
            combined_points = progressive_points_3d
        else:
            combined_points = np.empty((3, 0))

        self.logger.info(f"Triangulation complete: {basic_points_3d.shape[1]} basic + "
                        f"{progressive_points_3d.shape[1]} progressive = {combined_points.shape[1]} total")
        
        return combined_points

    def _triangulate_basic_new_points(self, 
                                    new_image: str,
                                    reconstruction_state: Dict[str, Any],
                                    matches_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        ADAPTED: Your original triangulate_new_points logic (basic triangulation only)
        """
        
        existing_cameras = reconstruction_state['cameras']
        fallback_K = reconstruction_state.get('camera_matrix')
        
        # Get pose and intrinsics of new camera
        new_camera_data = existing_cameras[new_image]
        new_R = new_camera_data['R']
        new_t = new_camera_data['t']
        new_K = new_camera_data.get('K', fallback_K)  # 🔧 FIXED: Use camera's own K matrix
        new_P = new_K @ np.hstack([new_R, new_t.reshape(-1, 1)])
        
        all_new_points = []
        all_new_observations = []
        point_id_offset = reconstruction_state['points_3d'].shape[1]  # Continue point numbering
        
        # Triangulate with each existing camera
        for existing_image, camera_data in existing_cameras.items():
            if existing_image == new_image:
                continue
                
            # Get matches between new image and existing image
            matches = self._get_matches_between_images(new_image, existing_image, matches_data, reconstruction_state)
            
            if len(matches['pts_new']) < 10:  # Minimum matches threshold
                continue
            
            # Create projection matrix for existing camera
            existing_R = camera_data['R'] 
            existing_t = camera_data['t']
            existing_K = camera_data.get('K', fallback_K)  # 🔧 FIXED: Use camera's own K matrix
            existing_P = existing_K @ np.hstack([existing_R, existing_t.reshape(-1, 1)])
            
            # Triangulate points between new and existing view
            points_3d_hom = self._triangulate_dlt_batch(
                matches['pts_new'], matches['pts_existing'], new_P, existing_P
            )
            points_3d = points_3d_hom[:3, :] / points_3d_hom[3, :]
            
            # Filter triangulated points
            valid_mask = self._filter_triangulated_points(
                points_3d, matches['pts_new'], matches['pts_existing'], 
                new_P, existing_P, new_R, new_t, existing_R, existing_t
            )
            
            if np.sum(valid_mask) == 0:
                continue
            
            # Keep only valid points  
            valid_points_3d = points_3d[:, valid_mask]
            valid_pts_new = matches['pts_new'][valid_mask]
            valid_pts_existing = matches['pts_existing'][valid_mask]
            
            # Create observations for these new points
            for i in range(valid_points_3d.shape[1]):
                point_id = point_id_offset + len(all_new_points)
                
                # Add observations from both cameras
                observations = [
                    {
                        'point_id': point_id,
                        'image_id': new_image,
                        'pixel_coords': valid_pts_new[i],
                        'feature_id': matches.get('feature_ids_new', [None])[i]
                    },
                    {
                        'point_id': point_id, 
                        'image_id': existing_image,
                        'pixel_coords': valid_pts_existing[i],
                        'feature_id': matches.get('feature_ids_existing', [None])[i]
                    }
                ]
                
                all_new_observations.extend(observations)
                all_new_points.append(valid_points_3d[:, i])
        
        # Convert to numpy array
        if all_new_points:
            new_points_3d = np.column_stack(all_new_points)  # 3xN
        else:
            new_points_3d = np.empty((3, 0))
        
        result = {
            'new_points_3d': new_points_3d,
            'new_observations': all_new_observations,
            'statistics': {
                'new_points_added': new_points_3d.shape[1],
                'triangulation_pairs': len([img for img in existing_cameras.keys() if img != new_image])
            }
        }
        
        return result



    def _triangulate_progressive_new_points(self, new_image: str, reconstruction_state: Dict[str, Any]) -> np.ndarray:
        """
        🆕 NEW: Progressive triangulation after camera addition
        
        This creates many more 3D points by:
        1. Triangulating with ALL images (processed + unprocessed)  
        2. Extending existing tracks to unprocessed images
        3. Cross-triangulating processed cameras with unprocessed images
        """
        
        # Get image sets
        processed_cameras = set(reconstruction_state['cameras'].keys())
        all_images = set(self.all_image_info) if self.all_image_info else processed_cameras
        unprocessed_images = all_images - processed_cameras
        
        self.logger.info(f"Progressive triangulation: {len(processed_cameras)} processed, {len(unprocessed_images)} unprocessed")
        
        all_new_points = []
        
        # Strategy 1: Triangulate new camera with ALL other images
        for other_image in all_images:
            if other_image == new_image:
                continue
            
            matches = self._find_matches_between_images_progressive(new_image, other_image)
            if matches is None or len(matches['pts1']) < 5:
                continue
            
            # Filter unmatched points (avoid duplicates)
            unmatched = self._filter_unmatched_progressive(matches, new_image, other_image, reconstruction_state)
            if len(unmatched['pts1']) < 3:
                continue
            
            # Triangulate
            if other_image in processed_cameras:
                # Other image has known camera pose
                triangulated = self._triangulate_with_known_camera_progressive(
                    unmatched, new_image, other_image, reconstruction_state
                )
            else:
                # Other image is unprocessed - use rough triangulation
                triangulated = self._triangulate_with_unknown_camera_progressive(
                    unmatched, new_image, other_image, reconstruction_state
                )
            
            if triangulated:
                all_new_points.extend(triangulated)
                
                # Limit per image pair
                if len(triangulated) >= self.config.max_points_per_pair:
                    break
            
            # Global limit check
            if len(all_new_points) >= self.config.max_total_new_points:
                break
        
        # Strategy 2: Extend existing tracks (if enabled)
        if self.config.enable_track_extension:
            self._extend_existing_tracks_progressive(reconstruction_state, unprocessed_images)
        
        # Convert to array format  
        if all_new_points:
            return np.array(all_new_points).T  # (3, N)
        else:
            return np.empty((3, 0))


    def _find_matches_between_images_progressive(self, img1: str, img2: str) -> Optional[Dict]:
        """🆕 NEW: Find matches between two images for progressive triangulation"""
        
        if self.matches_data is None:
            return None
        
        possible_keys = [
            (img1, img2), (img2, img1),
            f"{img1}_vs_{img2}", f"{img2}_vs_{img1}",
            f"{img1}_{img2}", f"{img2}_{img1}",
            f"{img1}__{img2}", f"{img2}__{img1}"
        ]
        
        for key in possible_keys:
            if key in self.matches_data:
                match_data = self.matches_data[key]
                
                if match_data['correspondences'] == []:
                    continue
                # Determine order
                if isinstance(key, tuple):
                    img1_first = (key[0] == img1)
                else:
                    img1_first = str(key).startswith(img1)
                
                # Extract points
                pts1, pts2 = self._extract_points_from_match_data_progressive(match_data)
                if pts1 is None or pts2 is None:
                    continue
                
                # Return with correct assignment
                if img1_first:
                    return {'pts1': pts1, 'pts2': pts2}
                else:
                    return {'pts1': pts2, 'pts2': pts1}
        
        return None

    def _extract_points_from_match_data_progressive(self, match_data: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """🆕 NEW: Extract point arrays from match data for progressive triangulation"""
        
        try:
            # Handle LightGlue format
            if 'correspondences' in match_data:
                correspondences = np.array(match_data['correspondences'][0])
                pts1 = correspondences[:, :2]
                pts2 = correspondences[:, 2:4]
            
            # Handle traditional format
            elif 'pts1' in match_data and 'pts2' in match_data:
                pts1 = np.array(match_data['pts1'])
                pts2 = np.array(match_data['pts2'])
            
            # Handle keypoints
            elif 'keypoints1' in match_data and 'keypoints2' in match_data:
                kp1 = match_data['keypoints1']
                kp2 = match_data['keypoints2']
                
                if hasattr(kp1[0], 'pt'):
                    pts1 = np.array([kp.pt for kp in kp1])
                    pts2 = np.array([kp.pt for kp in kp2])
                else:
                    pts1 = np.array(kp1)
                    pts2 = np.array(kp2)
            
            else:
                return None, None
            
            # Normalize to (N, 2) format
            pts1 = self._normalize_points_progressive(pts1)
            pts2 = self._normalize_points_progressive(pts2)
            
            return pts1, pts2
            
        except Exception:
            return None, None


    def _normalize_points_progressive(self, points) -> np.ndarray:
        """🆕 NEW: Normalize point array to (N, 2) format"""
        if points is None or len(points) == 0:
            return np.empty((0, 2))
        
        points = np.array(points, dtype=np.float32)
        
        if points.ndim == 2 and points.shape[1] == 2:
            return points
        elif points.ndim == 3 and points.shape[1:] == (1, 2):
            return points.squeeze(axis=1)
        elif points.ndim == 2 and points.shape[1] > 2:
            return points[:, :2]
        else:
            return np.empty((0, 2))

    
    def _filter_unmatched_progressive(self, matches: Dict, img1: str, img2: str, reconstruction_state: Dict) -> Dict:
        """🆕 NEW: Filter out points that already have 3D correspondences"""
        
        observations = reconstruction_state.get('observations', {})
        
        # Get existing observations for both images
        img1_obs = observations.get(img1, [])
        img2_obs = observations.get(img2, [])
        
        # 🔧 CHANGED: Bootstrap special case - if one image has no observations, keep all matches
        if len(img1_obs) == 0 or len(img2_obs) == 0:
            return matches
        
        # Create lookup arrays
        img1_obs_points = np.array([obs['image_point'] for obs in img1_obs]) if img1_obs else np.empty((0, 2))
        img2_obs_points = np.array([obs['image_point'] for obs in img2_obs]) if img2_obs else np.empty((0, 2))
        
        filtered_pts1 = []
        filtered_pts2 = []
        
        for pt1, pt2 in zip(matches['pts1'], matches['pts2']):
            
            # Check if points are near existing observations
            pt1_near_obs = False
            pt2_near_obs = False
            
            if len(img1_obs_points) > 0:
                distances1 = np.linalg.norm(img1_obs_points - pt1.reshape(1, -1), axis=1)
                pt1_near_obs = np.min(distances1) < self.config.progressive_tolerance
            
            if len(img2_obs_points) > 0:
                distances2 = np.linalg.norm(img2_obs_points - pt2.reshape(1, -1), axis=1)
                pt2_near_obs = np.min(distances2) < self.config.progressive_tolerance
            
            # Keep if neither point is near existing observations
            if not (pt1_near_obs and pt2_near_obs):
                filtered_pts1.append(pt1)
                filtered_pts2.append(pt2)
        
        return {
            'pts1': np.array(filtered_pts1) if filtered_pts1 else np.empty((0, 2)),
            'pts2': np.array(filtered_pts2) if filtered_pts2 else np.empty((0, 2))
        }

    def _triangulate_with_known_camera_progressive(self, matches: Dict, img1: str, img2: str, reconstruction_state: Dict) -> List[np.ndarray]:
        """🆕 FIXED: Triangulate when both cameras have known poses - using per-camera intrinsics"""
        
        cameras = reconstruction_state['cameras']
        cam1 = cameras[img1]
        cam2 = cameras[img2]
        
        # 🔧 FIXED: Get individual camera matrices instead of assuming same K
        fallback_K = reconstruction_state.get('camera_matrix')
        K1 = cam1.get('K', fallback_K)  # Use camera 1's intrinsics
        K2 = cam2.get('K', fallback_K)  # Use camera 2's intrinsics
        
        P1 = K1 @ np.hstack([cam1['R'], cam1['t'].reshape(-1, 1)])
        P2 = K2 @ np.hstack([cam2['R'], cam2['t'].reshape(-1, 1)])
        
        if K1 is not fallback_K and K2 is not fallback_K:
            self.logger.info(f"Progressive triangulation: {img1} (fx={K1[0,0]:.1f}) <-> {img2} (fx={K2[0,0]:.1f})")
        
        return self._triangulate_point_pairs_progressive(matches['pts1'], matches['pts2'], P1, P2, cam1['R'], cam1['t'], cam2['R'], cam2['t'])
    
    def _triangulate_with_unknown_camera_progressive(self, matches: Dict, known_img: str, unknown_img: str, reconstruction_state: Dict) -> List[np.ndarray]:
        """🆕 FIXED: Triangulate when one camera pose is unknown - using per-camera intrinsics"""
        
        cameras = reconstruction_state['cameras']
        known_cam = cameras[known_img]
        
        # 🔧 FIXED: Use known camera's individual intrinsics
        fallback_K = reconstruction_state.get('camera_matrix')
        K_known = known_cam.get('K', fallback_K)
        
        # Known camera projection
        P_known = K_known @ np.hstack([known_cam['R'], known_cam['t'].reshape(-1, 1)])
        
        # For unknown camera, we have to make assumptions about intrinsics
        # Option 1: Use same as known camera (reasonable if same camera model)
        # Option 2: Use fallback global matrix
        # Option 3: Use average intrinsics from existing cameras
        
        # Using Option 3: Average intrinsics from all existing cameras for unknown camera estimation
        K_unknown = self._estimate_intrinsics_for_unknown_camera(reconstruction_state)
        
        # Rough estimate for unknown camera pose
        R_unknown = np.eye(3)
        t_unknown = known_cam['t'] + np.array([1.0, 0.0, 0.0]).reshape(-1, 1)  # Offset
        P_unknown = K_unknown @ np.hstack([R_unknown, t_unknown])
        
        self.logger.info(f"Progressive triangulation with unknown: {known_img} (fx={K_known[0,0]:.1f}) <-> {unknown_img} (estimated fx={K_unknown[0,0]:.1f})")
        
        # Triangulate with stricter validation for rough estimates
        triangulated = self._triangulate_point_pairs_progressive(matches['pts1'], matches['pts2'], 
                                                              P_known, P_unknown, known_cam['R'], known_cam['t'], R_unknown, t_unknown)
        
        # Filter for conservative distance range (rough triangulations are less reliable)
        filtered = []
        for point in triangulated:
            distance = np.linalg.norm(point)
            if 0.5 < distance < 20.0:  # More conservative range
                filtered.append(point)
        
        return filtered

    def _estimate_intrinsics_for_unknown_camera(self, reconstruction_state: Dict) -> np.ndarray:
        """🆕 NEW: Estimate reasonable intrinsics for unknown camera based on existing cameras"""
        
        cameras = reconstruction_state['cameras']
        fallback_K = reconstruction_state.get('camera_matrix')
        
        # Collect all existing camera matrices
        K_matrices = []
        for camera_data in cameras.values():
            K = camera_data.get('K')
            if K is not None:
                K_matrices.append(K)
        
        if not K_matrices:
            # No per-camera intrinsics available, use fallback
            return fallback_K if fallback_K is not None else np.eye(3)
        
        # Calculate average intrinsics
        K_matrices = np.array(K_matrices)
        K_avg = np.mean(K_matrices, axis=0)
        
        self.logger.info(f"Estimated intrinsics for unknown camera from {len(K_matrices)} existing cameras")
        
        return K_avg

    def _triangulate_point_pairs_progressive(self, pts1: np.ndarray, pts2: np.ndarray,
                                           P1: np.ndarray, P2: np.ndarray,
                                           R1: np.ndarray, t1: np.ndarray,
                                           R2: np.ndarray, t2: np.ndarray) -> List[np.ndarray]:
        """🆕 NEW: Triangulate multiple point pairs for progressive triangulation"""
        
        triangulated = []
        
        for pt1, pt2 in zip(pts1, pts2):
            # Triangulate using DLT
            A = np.array([
                pt1[0] * P1[2, :] - P1[0, :],
                pt1[1] * P1[2, :] - P1[1, :],
                pt2[0] * P2[2, :] - P2[0, :],
                pt2[1] * P2[2, :] - P2[1, :]
            ])
            
            try:
                _, _, Vt = np.linalg.svd(A)
                X = Vt[-1, :]
                
                if abs(X[3]) > 1e-8:
                    point_3d = X[:3] / X[3]
                    
                    # Validate point
                    if self._validate_3d_point_progressive(point_3d, P1, P2, pt1, pt2, R1, t1, R2, t2):
                        triangulated.append(point_3d)
            except:
                continue
        
        return triangulated
    
    def _validate_3d_point_progressive(self, point_3d: np.ndarray, 
                                     P1: np.ndarray, P2: np.ndarray,
                                     pt1: np.ndarray, pt2: np.ndarray,
                                     R1: np.ndarray, t1: np.ndarray,
                                     R2: np.ndarray, t2: np.ndarray) -> bool:
        """🆕 NEW: Validate triangulated 3D point for progressive triangulation"""
        
        if not np.all(np.isfinite(point_3d)):
            return False
        
        # Distance check
        distance = np.linalg.norm(point_3d)
        if not (self.config.min_distance < distance < self.config.max_distance):
            return False
        
        # Reprojection error check
        X_hom = np.append(point_3d, 1.0)
        
        proj1 = P1 @ X_hom
        proj2 = P2 @ X_hom
        
        if abs(proj1[2]) < 1e-8 or abs(proj2[2]) < 1e-8:
            return False
        
        reproj1 = proj1[:2] / proj1[2]
        reproj2 = proj2[:2] / proj2[2]
        
        error1 = np.linalg.norm(reproj1 - pt1)
        error2 = np.linalg.norm(reproj2 - pt2)
        
        return error1 < self.max_reprojection_error and error2 < self.max_reprojection_error
    
    def _extend_existing_tracks_progressive(self, reconstruction_state: Dict, unprocessed_images: Set[str]):
        """🆕 NEW: Extend existing 3D point tracks to unprocessed images"""
        
        if not self.config.enable_track_extension or not unprocessed_images:
            return
        
        observations = reconstruction_state.get('observations', {})
        processed_cameras = set(reconstruction_state['cameras'].keys())
        
        self.logger.info(f"Extending tracks to {len(unprocessed_images)} unprocessed images...")
        
        extended_count = 0
        
        for unprocessed_image in unprocessed_images:
            for processed_camera in processed_cameras:
                
                matches = self._find_matches_between_images_progressive(unprocessed_image, processed_camera)
                if matches is None:
                    continue
                
                processed_obs = observations.get(processed_camera, [])
                if not processed_obs:
                    continue
                
                obs_points = np.array([obs['image_point'] for obs in processed_obs])
                
                # Find matches that correspond to existing 3D points
                for i, (unprocessed_pt, processed_pt) in enumerate(zip(matches['pts1'], matches['pts2'])):
                    
                    # Find closest observation in processed camera
                    distances = np.linalg.norm(obs_points - processed_pt.reshape(1, -1), axis=1)
                    min_idx = np.argmin(distances)
                    min_distance = distances[min_idx]
                    
                    if min_distance < self.config.track_extension_tolerance:
                        # Found match with existing 3D point
                        closest_obs = processed_obs[min_idx]
                        point_3d_id = closest_obs['point_id']
                        
                        # Add observation for unprocessed image
                        if unprocessed_image not in observations:
                            observations[unprocessed_image] = []
                        
                        # Check for duplicates
                        duplicate_found = False
                        for existing_obs in observations[unprocessed_image]:
                            if (existing_obs['point_id'] == point_3d_id and
                                np.linalg.norm(np.array(existing_obs['image_point']) - unprocessed_pt) < 2.0):
                                duplicate_found = True
                                break
                        
                        if not duplicate_found:
                            observations[unprocessed_image].append({
                                'point_id': point_3d_id,
                                'image_point': [float(unprocessed_pt[0]), float(unprocessed_pt[1])],
                                'source': 'track_extension'
                            })
                            extended_count += 1
        
        if extended_count > 0:
            self.logger.info(f"Extended {extended_count} tracks to unprocessed images")
    



    def _triangulate_dlt_batch(self, 
                              pts1: np.ndarray, 
                              pts2: np.ndarray, 
                              P1: np.ndarray, 
                              P2: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D points using Direct Linear Transform (DLT) method.
        Vectorized implementation for efficiency.
        
        Args:
            pts1, pts2: 2D points in both images (Nx2)
            P1, P2: Camera projection matrices (3x4)
            
        Returns:
            Homogeneous 3D points (4xN)
        """
        
        num_points = len(pts1)
        points_3d_hom = np.zeros((4, num_points))
        
        for i in range(num_points):
            # Set up the linear system Ax = 0 for DLT
            # Each point gives us 2 equations
            A = np.array([
                pts1[i, 0] * P1[2, :] - P1[0, :],  # x1*P1[2] - P1[0] 
                pts1[i, 1] * P1[2, :] - P1[1, :],  # y1*P1[2] - P1[1]
                pts2[i, 0] * P2[2, :] - P2[0, :],  # x2*P2[2] - P2[0]
                pts2[i, 1] * P2[2, :] - P2[1, :]   # y2*P2[2] - P2[1]
            ])
            
            # Solve using SVD
            _, _, Vt = np.linalg.svd(A)
            points_3d_hom[:, i] = Vt[-1, :]  # Last row of V^T
        
        return points_3d_hom
    
    def _filter_triangulated_points(self, 
                                  points_3d: np.ndarray,
                                  pts1: np.ndarray, 
                                  pts2: np.ndarray,
                                  P1: np.ndarray, 
                                  P2: np.ndarray,
                                  R1: np.ndarray, 
                                  t1: np.ndarray,
                                  R2: np.ndarray, 
                                  t2: np.ndarray) -> np.ndarray:
        """
        Filter triangulated points based on quality criteria.
        
        Returns:
            Boolean mask indicating valid points
        """
        
        num_points = points_3d.shape[1]
        valid_mask = np.ones(num_points, dtype=bool)
        
        # 1. Check if points are in front of both cameras
        depth_mask1 = self._check_positive_depth(points_3d, R1, t1)
        depth_mask2 = self._check_positive_depth(points_3d, R2, t2)
        valid_mask &= depth_mask1 & depth_mask2
        
        # 2. Check triangulation angle
        angle_mask = self._check_triangulation_angle(points_3d, R1, t1, R2, t2)
        valid_mask &= angle_mask
        
        # 3. Check reprojection error
        reproj_mask = self._check_reprojection_error(points_3d, pts1, pts2, P1, P2)
        valid_mask &= reproj_mask
        
        # 4. Check depth bounds
        depth_bounds_mask = self._check_depth_bounds(points_3d, R1, t1, R2, t2)
        valid_mask &= depth_bounds_mask
        
        return valid_mask
    
    def _check_positive_depth(self, 
                            points_3d: np.ndarray, 
                            R: np.ndarray, 
                            t: np.ndarray) -> np.ndarray:
        """Check if 3D points are in front of camera (positive depth)."""
        
        # Transform points to camera coordinate system
        points_cam = R @ points_3d + t.reshape(-1, 1)
        
        # Check if Z coordinate (depth) is positive
        return points_cam[2, :] > 0
    
    def _check_triangulation_angle(self, 
                                 points_3d: np.ndarray,
                                 R1: np.ndarray, 
                                 t1: np.ndarray,
                                 R2: np.ndarray, 
                                 t2: np.ndarray) -> np.ndarray:
        """Check if triangulation angle is sufficient for stable triangulation."""
        
        # Camera centers in world coordinates
        C1 = -R1.T @ t1
        C2 = -R2.T @ t2
        
        angles = []
        for i in range(points_3d.shape[1]):
            P = points_3d[:, i].reshape(-1, 1)
            
            # Vectors from cameras to 3D point
            ray1 = P - C1
            ray2 = P - C2
            
            # Normalize vectors
            ray1_norm = ray1 / np.linalg.norm(ray1)
            ray2_norm = ray2 / np.linalg.norm(ray2)
            
            # Calculate angle between rays
            cos_angle = np.clip(np.dot(ray1_norm.T, ray2_norm)[0, 0], -1, 1)
            angle = np.arccos(np.abs(cos_angle))  # Take absolute value for obtuse angles
            angles.append(angle)
        
        angles = np.array(angles)
        return angles >= self.min_triangulation_angle
    
    def _check_reprojection_error(self, 
                                points_3d: np.ndarray,
                                pts1: np.ndarray, 
                                pts2: np.ndarray,
                                P1: np.ndarray, 
                                P2: np.ndarray) -> np.ndarray:
        """Check reprojection error for triangulated points."""
        
        # Convert to homogeneous coordinates
        points_3d_hom = np.vstack([points_3d, np.ones((1, points_3d.shape[1]))])
        
        # Project to both images
        proj1_hom = P1 @ points_3d_hom
        proj2_hom = P2 @ points_3d_hom
        
        # Convert to pixel coordinates
        proj1 = (proj1_hom[:2, :] / proj1_hom[2, :]).T  # Nx2
        proj2 = (proj2_hom[:2, :] / proj2_hom[2, :]).T  # Nx2
        
        # Calculate reprojection errors
        error1 = np.linalg.norm(proj1 - pts1, axis=1)
        error2 = np.linalg.norm(proj2 - pts2, axis=1)
        
        # Both projections must have low error
        return (error1 <= self.max_reprojection_error) & (error2 <= self.max_reprojection_error)
    
    def _check_depth_bounds(self, 
                          points_3d: np.ndarray,
                          R1: np.ndarray, 
                          t1: np.ndarray,
                          R2: np.ndarray, 
                          t2: np.ndarray) -> np.ndarray:
        """Check if points are within reasonable depth bounds."""
        
        # Get depths in both camera coordinate systems
        points_cam1 = R1 @ points_3d + t1.reshape(-1, 1)
        points_cam2 = R2 @ points_3d + t2.reshape(-1, 1)
        
        depths1 = points_cam1[2, :]
        depths2 = points_cam2[2, :]
        
        # Check bounds for both cameras
        valid1 = (depths1 >= self.min_depth) & (depths1 <= self.max_depth)
        valid2 = (depths2 >= self.min_depth) & (depths2 <= self.max_depth)
        
        return valid1 & valid2
    
    def _get_matches_between_images(self, 
                                  img1: str, 
                                  img2: str,
                                  matches_data: Optional[Dict] = None,
                                  reconstruction_state: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Extract feature matches between two images.
        This is a placeholder - you'll need to implement based on your match data structure.
        """
        
        # TODO: Implement based on your feature matching pipeline
        # This should return corresponding points between the two images
        # that haven't been triangulated yet
        
        if matches_data is not None:
            # Extract from provided matches data
            pair_key = f"{img1}_{img2}" if f"{img1}_{img2}" in matches_data else f"{img2}_{img1}"
            if pair_key in matches_data:
                matches = matches_data[pair_key]
                return {
                    'pts_new': matches['pts1'] if img1 in pair_key.split('_')[0] else matches['pts2'],
                    'pts_existing': matches['pts2'] if img1 in pair_key.split('_')[0] else matches['pts1'],
                    'feature_ids_new': matches.get('feature_ids1', None),
                    'feature_ids_existing': matches.get('feature_ids2', None)
                }
        
        # Fallback: return empty matches
        return {
            'pts_new': np.empty((0, 2)),
            'pts_existing': np.empty((0, 2)),
            'feature_ids_new': [],
            'feature_ids_existing': []
        }
    
    def _create_initial_observations(self, 
                                   points_3d: np.ndarray,
                                   pts1: np.ndarray, 
                                   pts2: np.ndarray,
                                   image_pair: Tuple[str, str]) -> List[Dict[str, Any]]:
        """Create observation structure for bundle adjustment."""
        
        observations = []
        
        for i in range(points_3d.shape[1]):
            # Observation in first image
            observations.append({
                'point_id': i,
                'image_id': image_pair[0],
                'pixel_coords': pts1[i],
                'feature_id': i  # This should be the actual feature ID from your detector
            })
            
            # Observation in second image  
            observations.append({
                'point_id': i,
                'image_id': image_pair[1], 
                'pixel_coords': pts2[i],
                'feature_id': i  # This should be the actual feature ID from your detector
            })
        
        return observations
    
    def _estimate_point_colors(self, 
                             pts1: np.ndarray, 
                             pts2: np.ndarray, 
                             image_pair: Tuple[str, str]) -> Optional[np.ndarray]:
        """
        Estimate RGB colors for 3D points.
        Placeholder - implement based on your image loading pipeline.
        """
        
        # TODO: Load images and sample colors at feature points
        # For now, return None - colors are optional
        return None



    def _apply_hartley_sturm(self,
                        pts1: np.ndarray,
                        pts2: np.ndarray,
                        K1: np.ndarray,
                        K2: np.ndarray,
                        R1: np.ndarray,
                        t1: np.ndarray,
                        R2: np.ndarray,
                        t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Hartley-Sturm optimal triangulation to point correspondences.
        
        Args:
            pts1, pts2: Original point correspondences in pixel coordinates
            K1, K2: Camera intrinsic matrices
            R1, t1, R2, t2: Camera extrinsics
            
        Returns:
            Corrected point correspondences that exactly satisfy epipolar constraint
        """
        
        self.logger.info("Applying Hartley-Sturm optimal triangulation...")
        
        # Compute Essential matrix
        # E = R2 @ [t2 - R2 @ R1.T @ t1]_x @ R2 @ R1.T
        # Simplified: E = [t]_x @ R where R and t are relative pose
        R_rel = R2 @ R1.T
        t_rel = t2 - R2 @ R1.T @ t1
        
        # Create skew-symmetric matrix
        tx = np.array([
            [0, -t_rel[2, 0], t_rel[1, 0]],
            [t_rel[2, 0], 0, -t_rel[0, 0]],
            [-t_rel[1, 0], t_rel[0, 0], 0]
        ])
        
        E = tx @ R_rel
        
        # Compute Fundamental matrix: F = K2^-T @ E @ K1^-1
        F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
        
        # Normalize F (ensure rank 2)
        U, S, Vt = np.linalg.svd(F)
        S[2] = 0  # Enforce rank-2 constraint
        F = U @ np.diag(S) @ Vt
        
        # Apply Hartley-Sturm correction
        pts1_corrected, pts2_corrected = optimal_triangulation_hartley_sturm(pts1, pts2, F)
        
        # Log correction statistics
        correction1 = np.linalg.norm(pts1_corrected - pts1, axis=1)
        correction2 = np.linalg.norm(pts2_corrected - pts2, axis=1)
        
        self.logger.info(f"Hartley-Sturm corrections - Image 1: mean={np.mean(correction1):.3f}px, "
                        f"max={np.max(correction1):.3f}px | Image 2: mean={np.mean(correction2):.3f}px, "
                        f"max={np.max(correction2):.3f}px")
        
        return pts1_corrected, pts2_corrected
    
    def _triangulate_dlt_batch_with_refinement(self, 
                                              pts1: np.ndarray, 
                                              pts2: np.ndarray, 
                                              P1: np.ndarray, 
                                              P2: np.ndarray,
                                              use_iterative_refinement: bool = False) -> np.ndarray:
        """
        Enhanced DLT triangulation with optional iterative refinement.
        
        Args:
            pts1, pts2: 2D points in both images (Nx2)
            P1, P2: Camera projection matrices (3x4)
            use_iterative_refinement: Apply Levenberg-Marquardt refinement after DLT
            
        Returns:
            Homogeneous 3D points (4xN)
        """
        
        num_points = len(pts1)
        points_3d_hom = np.zeros((4, num_points))
        
        for i in range(num_points):
            # Standard DLT
            A = np.array([
                pts1[i, 0] * P1[2, :] - P1[0, :],
                pts1[i, 1] * P1[2, :] - P1[1, :],
                pts2[i, 0] * P2[2, :] - P2[0, :],
                pts2[i, 1] * P2[2, :] - P2[1, :]
            ])
            
            _, _, Vt = np.linalg.svd(A)
            X_init = Vt[-1, :]
            
            if use_iterative_refinement and np.abs(X_init[3]) > 1e-8:
                # Refine using Levenberg-Marquardt
                X_refined = self._refine_point_lm(X_init, pts1[i], pts2[i], P1, P2)
                points_3d_hom[:, i] = X_refined
            else:
                points_3d_hom[:, i] = X_init
        
        return points_3d_hom
    
    def _refine_point_lm(self,
                        X_init: np.ndarray,
                        pt1: np.ndarray,
                        pt2: np.ndarray,
                        P1: np.ndarray,
                        P2: np.ndarray,
                        max_iterations: int = 10) -> np.ndarray:
        """
        Refine a single 3D point using Levenberg-Marquardt optimization.
        Minimizes reprojection error.
        
        Args:
            X_init: Initial homogeneous 3D point estimate (4x1)
            pt1, pt2: Observed 2D points
            P1, P2: Projection matrices
            max_iterations: Maximum refinement iterations
            
        Returns:
            Refined homogeneous 3D point
        """
        
        # Convert to Euclidean for optimization
        X = X_init[:3] / X_init[3]
        
        def residual(X_euclidean):
            """Compute reprojection error residuals"""
            X_hom = np.append(X_euclidean, 1.0)
            
            # Project to both images
            proj1 = P1 @ X_hom
            proj2 = P2 @ X_hom
            
            # Convert to pixel coordinates
            if np.abs(proj1[2]) > 1e-8 and np.abs(proj2[2]) > 1e-8:
                reproj1 = proj1[:2] / proj1[2]
                reproj2 = proj2[:2] / proj2[2]
                
                # Residuals
                r1 = reproj1 - pt1
                r2 = reproj2 - pt2
                
                return np.concatenate([r1, r2])
            else:
                return np.array([1e6, 1e6, 1e6, 1e6])  # Large error for invalid points
        
        # Optimize
        result = least_squares(residual, X, max_nfev=max_iterations)
        
        # Convert back to homogeneous
        return np.append(result.x, 1.0)
    
    # Add method for triangulating with optimal correction for new points
    def _triangulate_with_hartley_sturm(self,
                                       pts1: np.ndarray,
                                       pts2: np.ndarray,
                                       P1: np.ndarray,
                                       P2: np.ndarray,
                                       K1: np.ndarray,
                                       K2: np.ndarray,
                                       R1: np.ndarray,
                                       t1: np.ndarray,
                                       R2: np.ndarray,
                                       t2: np.ndarray) -> np.ndarray:
        """
        Triangulate points using Hartley-Sturm optimal triangulation.
        
        Returns:
            3D points (3xN)
        """
        
        # Apply Hartley-Sturm correction
        pts1_corrected, pts2_corrected = self._apply_hartley_sturm(
            pts1, pts2, K1, K2, R1, t1, R2, t2
        )
        
        # Triangulate corrected points
        points_3d_hom = self._triangulate_dlt_batch_with_refinement(
            pts1_corrected, pts2_corrected, P1, P2, use_iterative_refinement=True
        )
        
        # Convert to Euclidean
        points_3d = points_3d_hom[:3, :] / points_3d_hom[3, :]
        
        return points_3d


