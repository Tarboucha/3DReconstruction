# ADAPTED TRIANGULATION ENGINE FOR NEW PIPELINE FRAMEWORK
# Key changes:
# 1. Added set_provider() method for provider pattern
# 2. Fixed K parameter handling (list of two matrices)
# 3. Integrated provider access for progressive triangulation
# 4. Cleaned up method signatures for pipeline compatibility

import numpy as np
import cv2
from scipy.optimize import least_squares
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
from dataclasses import dataclass

from core.interfaces import IMatchDataProvider


class TriangulationEngine:
    """
    ADAPTED: Triangulation engine integrated with provider pattern for pipeline.
    """
    
    def __init__(self, 
                 min_triangulation_angle_deg: float = 2.0,
                 max_reprojection_error: float = 2.0,
                 min_depth: float = 0.1,
                 max_depth: float = 1000.0,
                 use_optimal_triangulation: bool = True,
                 config: Optional['TriangulationConfig'] = None):
        """Initialize triangulation engine with quality thresholds."""
        
        # Provider integration
        self.provider: Optional[IMatchDataProvider] = None
        
        # Triangulation parameters
        self.min_triangulation_angle = np.deg2rad(min_triangulation_angle_deg)
        self.max_reprojection_error = max_reprojection_error
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.use_optimal_triangulation = use_optimal_triangulation
        
        self.logger = logging.getLogger(__name__)
        
        # Configuration for progressive features
        self.config = config or TriangulationConfig()
        
        # Legacy attributes (for backward compatibility)
        self.matches_data = None
        self.all_image_info = []
    
    # ============================================================================
    # PROVIDER PATTERN INTEGRATION
    # ============================================================================
    
    def set_provider(self, provider: IMatchDataProvider):
        """
        Set the data provider for accessing match data.
        Required for provider pattern integration with pipeline.
        
        Args:
            provider: IMatchDataProvider instance
        """
        self.provider = provider
        self.logger.info(f"Provider set: {provider.__class__.__name__}")
        
        # Update image info from provider
        if self.provider:
            self.all_image_info = list(self.provider.get_all_images())
            self.logger.info(f"Loaded {len(self.all_image_info)} images from provider")
    
    # ============================================================================
    # INITIAL TWO-VIEW TRIANGULATION (FIXED K PARAMETER)
    # ============================================================================
    
    def triangulate_initial_points(self, 
                                    pts1: np.ndarray, 
                                    pts2: np.ndarray,
                                    R1: np.ndarray, 
                                    t1: np.ndarray,
                                    R2: np.ndarray, 
                                    t2: np.ndarray,
                                    K: List[np.ndarray],  # 🔧 FIXED: Now expects list [K1, K2]
                                    image_pair: Tuple[str, str]) -> Dict[str, Any]:
        """
        Triangulate 3D points from initial two-view reconstruction.
        
        Args:
            pts1, pts2: Point correspondences (Nx2)
            R1, t1: First camera pose
            R2, t2: Second camera pose
            K: List of two intrinsic matrices [K1, K2]  # 🔧 CHANGED
            image_pair: Tuple of image identifiers
            
        Returns:
            Dictionary with triangulated points and observations
        """
        
        self.logger.info(f"Triangulating initial points: {image_pair[0]} <-> {image_pair[1]}")
        self.logger.info(f"Input correspondences: {len(pts1)}")
        
        # 🔧 FIXED: Extract individual K matrices
        K1, K2 = K[0], K[1]
        
        # Create projection matrices
        P1 = K1 @ np.hstack([R1, t1.reshape(-1, 1)])  # 3x4
        P2 = K2 @ np.hstack([R2, t2.reshape(-1, 1)])  # 3x4
        
        # Apply Hartley-Sturm optimal triangulation if enabled
        if self.use_optimal_triangulation:
            pts1_to_triangulate, pts2_to_triangulate = self._apply_hartley_sturm(
                pts1, pts2, K1, K2, R1, t1, R2, t2
            )
        else:
            pts1_to_triangulate, pts2_to_triangulate = pts1, pts2
        
        # Triangulate all points using DLT
        points_3d_hom = self._triangulate_dlt_batch(
            pts1_to_triangulate, pts2_to_triangulate, P1, P2
        )
        
        # Convert from homogeneous to 3D coordinates
        points_3d = points_3d_hom[:3, :] / points_3d_hom[3, :]  # 3xN
        
        # Quality filtering
        valid_mask = self._filter_triangulated_points(
            points_3d, pts1, pts2, P1, P2, R1, t1, R2, t2
        )
        
        # Keep only valid points
        valid_points_3d = points_3d[:, valid_mask]
        valid_pts1 = pts1[valid_mask]
        valid_pts2 = pts2[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        # Create observation structure
        observations = self._create_initial_observations_with_indices(
            valid_points_3d, valid_pts1, valid_pts2, valid_indices, image_pair
        )
        
        result = {
            'points_3d': valid_points_3d,  # 3xN array
            'observations': observations,
            'valid_indices': valid_indices,
            'point_colors': None,  # Optional
            'statistics': {
                'initial_points': len(pts1),
                'triangulated_points': points_3d.shape[1],
                'valid_points': valid_points_3d.shape[1],
                'success_rate': valid_points_3d.shape[1] / len(pts1) if len(pts1) > 0 else 0.0,
                'used_optimal_triangulation': self.use_optimal_triangulation
            }
        }
        
        self.logger.info(
            f"Successfully triangulated {valid_points_3d.shape[1]}/{len(pts1)} points "
            f"(success rate: {result['statistics']['success_rate']:.2%})"
        )
        
        return result
    
    # ============================================================================
    # NEW VIEW TRIANGULATION (USES PROVIDER)
    # ============================================================================
    
    def triangulate_new_points(self, 
                               new_image: str,
                               reconstruction_state: Dict[str, Any]) -> np.ndarray:
        """
        Triangulate new 3D points when adding a new view.
        Uses provider for accessing match data.
        
        Args:
            new_image: Identifier of the newly added image
            reconstruction_state: Current reconstruction state
            
        Returns:
            New 3D points as (3, N) array
        """
        
        self.logger.info(f"Triangulating new points for view: {new_image}")
        
        # Basic triangulation with existing cameras
        basic_points_3d = self._triangulate_basic_new_points(
            new_image, reconstruction_state
        )
        
        # Progressive triangulation (if enabled and provider available)
        progressive_points_3d = np.empty((3, 0))
        if self.config.enable_progressive and self.provider is not None:
            self.logger.info("Running progressive triangulation...")
            progressive_points_3d = self._triangulate_progressive_new_points(
                new_image, reconstruction_state
            )
        
        # Combine results
        if basic_points_3d.size > 0 and progressive_points_3d.size > 0:
            combined_points = np.hstack([basic_points_3d, progressive_points_3d])
        elif basic_points_3d.size > 0:
            combined_points = basic_points_3d
        elif progressive_points_3d.size > 0:
            combined_points = progressive_points_3d
        else:
            combined_points = np.empty((3, 0))
        
        self.logger.info(
            f"Triangulation complete: {basic_points_3d.shape[1]} basic + "
            f"{progressive_points_3d.shape[1]} progressive = {combined_points.shape[1]} total"
        )
        
        return combined_points
    
    def _triangulate_basic_new_points(self, 
                                      new_image: str,
                                      reconstruction_state: Dict[str, Any]) -> np.ndarray:
        """
        Basic triangulation with existing cameras using provider.
        
        Args:
            new_image: New image to triangulate with
            reconstruction_state: Current reconstruction state
            
        Returns:
            New 3D points as (3, N) array
        """
        
        existing_cameras = reconstruction_state['cameras']
        fallback_K = reconstruction_state.get('camera_matrix')
        
        # Get new camera data
        new_camera_data = existing_cameras[new_image]
        new_R = new_camera_data['R']
        new_t = new_camera_data['t']
        new_K = new_camera_data.get('K', fallback_K)
        new_P = new_K @ np.hstack([new_R, new_t.reshape(-1, 1)])
        
        all_new_points = []
        
        # Triangulate with each existing camera
        for existing_image in existing_cameras.keys():
            if existing_image == new_image:
                continue
            
            # Get matches using provider
            matches = self._get_matches_between_images_from_provider(
                new_image, existing_image
            )
            
            if matches is None or len(matches['pts_new']) < 10:
                continue
            
            # Get existing camera data
            camera_data = existing_cameras[existing_image]
            existing_R = camera_data['R']
            existing_t = camera_data['t']
            existing_K = camera_data.get('K', fallback_K)
            existing_P = existing_K @ np.hstack([existing_R, existing_t.reshape(-1, 1)])
            
            # Triangulate
            points_3d_hom = self._triangulate_dlt_batch(
                matches['pts_new'], matches['pts_existing'], new_P, existing_P
            )
            points_3d = points_3d_hom[:3, :] / points_3d_hom[3, :]
            
            # Filter
            valid_mask = self._filter_triangulated_points(
                points_3d, matches['pts_new'], matches['pts_existing'],
                new_P, existing_P, new_R, new_t, existing_R, existing_t
            )
            
            if np.sum(valid_mask) > 0:
                valid_points_3d = points_3d[:, valid_mask]
                all_new_points.extend(valid_points_3d.T)
        
        # Convert to array
        if all_new_points:
            return np.array(all_new_points).T  # (3, N)
        else:
            return np.empty((3, 0))
    
    # ============================================================================
    # PROVIDER-BASED MATCH ACCESS
    # ============================================================================
    
    def _get_matches_between_images_from_provider(self, 
                                                   img1: str, 
                                                   img2: str) -> Optional[Dict]:
        """
        Get matches between two images using provider.
        
        Args:
            img1, img2: Image identifiers
            
        Returns:
            Dictionary with 'pts_new' and 'pts_existing' or None
        """
        
        if self.provider is None:
            self.logger.warning("No provider available for match access")
            return None
        
        # Try both pair orientations
        pair_keys = [(img1, img2), (img2, img1)]
        
        for pair_key in pair_keys:
            if self.provider.has_pair(pair_key):
                match_data = self.provider.get_match_data(pair_key)
                
                if match_data.num_matches < 8:
                    continue
                
                # Determine which points belong to which image
                if pair_key == (img1, img2):
                    pts_new = match_data.pts1
                    pts_existing = match_data.pts2
                else:
                    pts_new = match_data.pts2
                    pts_existing = match_data.pts1
                
                return {
                    'pts_new': pts_new,
                    'pts_existing': pts_existing,
                    'feature_ids_new': None,
                    'feature_ids_existing': None
                }
        
        return None
    
    def _find_matches_between_images_progressive(self, 
                                                 img1: str, 
                                                 img2: str) -> Optional[Dict]:
        """
        Find matches for progressive triangulation using provider.
        
        Args:
            img1, img2: Image identifiers
            
        Returns:
            Dictionary with 'pts1' and 'pts2' or None
        """
        
        if self.provider is None:
            return None
        
        # Try both orientations
        pair_keys = [(img1, img2), (img2, img1)]
        
        for pair_key in pair_keys:
            if self.provider.has_pair(pair_key):
                match_data = self.provider.get_match_data(pair_key)
                
                if match_data.num_matches < 5:
                    continue
                
                # Return with correct assignment
                if pair_key == (img1, img2):
                    return {'pts1': match_data.pts1, 'pts2': match_data.pts2}
                else:
                    return {'pts1': match_data.pts2, 'pts2': match_data.pts1}
        
        return None
    
    # ============================================================================
    # PROGRESSIVE TRIANGULATION
    # ============================================================================
    
    def _triangulate_progressive_new_points(self, 
                                            new_image: str, 
                                            reconstruction_state: Dict[str, Any]) -> np.ndarray:
        """
        Progressive triangulation - creates more 3D points by triangulating with
        both processed and unprocessed images.
        
        Args:
            new_image: Newly added image
            reconstruction_state: Current reconstruction state
            
        Returns:
            Progressive 3D points as (3, N) array
        """
        
        processed_cameras = set(reconstruction_state['cameras'].keys())
        all_images = set(self.all_image_info) if self.all_image_info else processed_cameras
        unprocessed_images = all_images - processed_cameras
        
        self.logger.info(
            f"Progressive: {len(processed_cameras)} processed, "
            f"{len(unprocessed_images)} unprocessed"
        )
        
        all_new_points = []
        
        # Triangulate with all other images
        for other_image in all_images:
            if other_image == new_image:
                continue
            
            matches = self._find_matches_between_images_progressive(new_image, other_image)
            if matches is None or len(matches['pts1']) < 5:
                continue
            
            # Filter out already triangulated points
            unmatched = self._filter_unmatched_progressive(
                matches, new_image, other_image, reconstruction_state
            )
            if len(unmatched['pts1']) < 3:
                continue
            
            # Triangulate based on whether other camera is processed
            if other_image in processed_cameras:
                triangulated = self._triangulate_with_known_camera_progressive(
                    unmatched, new_image, other_image, reconstruction_state
                )
            else:
                triangulated = self._triangulate_with_unknown_camera_progressive(
                    unmatched, new_image, other_image, reconstruction_state
                )
            
            if triangulated:
                all_new_points.extend(triangulated)
                
                if len(triangulated) >= self.config.max_points_per_pair:
                    break
            
            if len(all_new_points) >= self.config.max_total_new_points:
                break
        
        # Convert to array
        if all_new_points:
            return np.array(all_new_points).T  # (3, N)
        else:
            return np.empty((3, 0))
    
    def _filter_unmatched_progressive(self, 
                                     matches: Dict, 
                                     img1: str, 
                                     img2: str, 
                                     reconstruction_state: Dict) -> Dict:
        """Filter out points that already have 3D correspondences."""
        
        observations = reconstruction_state.get('observations', {})
        
        img1_obs = observations.get(img1, [])
        img2_obs = observations.get(img2, [])
        
        # Special case: if one image has no observations, keep all
        if len(img1_obs) == 0 or len(img2_obs) == 0:
            return matches
        
        # Create lookup arrays
        img1_obs_points = np.array([obs['image_point'] for obs in img1_obs])
        img2_obs_points = np.array([obs['image_point'] for obs in img2_obs])
        
        filtered_pts1 = []
        filtered_pts2 = []
        
        for pt1, pt2 in zip(matches['pts1'], matches['pts2']):
            # Check proximity to existing observations
            distances1 = np.linalg.norm(img1_obs_points - pt1.reshape(1, -1), axis=1)
            distances2 = np.linalg.norm(img2_obs_points - pt2.reshape(1, -1), axis=1)
            
            pt1_near_obs = np.min(distances1) < self.config.progressive_tolerance
            pt2_near_obs = np.min(distances2) < self.config.progressive_tolerance
            
            # Keep if neither point is near existing observations
            if not (pt1_near_obs and pt2_near_obs):
                filtered_pts1.append(pt1)
                filtered_pts2.append(pt2)
        
        return {
            'pts1': np.array(filtered_pts1) if filtered_pts1 else np.empty((0, 2)),
            'pts2': np.array(filtered_pts2) if filtered_pts2 else np.empty((0, 2))
        }
    
    def _triangulate_with_known_camera_progressive(self, 
                                                   matches: Dict, 
                                                   img1: str, 
                                                   img2: str, 
                                                   reconstruction_state: Dict) -> List[np.ndarray]:
        """Triangulate when both cameras have known poses."""
        
        cameras = reconstruction_state['cameras']
        cam1 = cameras[img1]
        cam2 = cameras[img2]
        
        fallback_K = reconstruction_state.get('camera_matrix')
        K1 = cam1.get('K', fallback_K)
        K2 = cam2.get('K', fallback_K)
        
        P1 = K1 @ np.hstack([cam1['R'], cam1['t'].reshape(-1, 1)])
        P2 = K2 @ np.hstack([cam2['R'], cam2['t'].reshape(-1, 1)])
        
        return self._triangulate_point_pairs_progressive(
            matches['pts1'], matches['pts2'], P1, P2,
            cam1['R'], cam1['t'], cam2['R'], cam2['t']
        )
    
    def _triangulate_with_unknown_camera_progressive(self, 
                                                     matches: Dict, 
                                                     known_img: str, 
                                                     unknown_img: str, 
                                                     reconstruction_state: Dict) -> List[np.ndarray]:
        """Triangulate when one camera pose is unknown (rough estimation)."""
        
        cameras = reconstruction_state['cameras']
        known_cam = cameras[known_img]
        
        fallback_K = reconstruction_state.get('camera_matrix')
        K_known = known_cam.get('K', fallback_K)
        
        P_known = K_known @ np.hstack([known_cam['R'], known_cam['t'].reshape(-1, 1)])
        
        # Estimate intrinsics for unknown camera
        K_unknown = self._estimate_intrinsics_for_unknown_camera(reconstruction_state)
        
        # Rough pose estimate
        R_unknown = np.eye(3)
        t_unknown = known_cam['t'] + np.array([1.0, 0.0, 0.0]).reshape(-1, 1)
        P_unknown = K_unknown @ np.hstack([R_unknown, t_unknown])
        
        # Triangulate with conservative filtering
        triangulated = self._triangulate_point_pairs_progressive(
            matches['pts1'], matches['pts2'], P_known, P_unknown,
            known_cam['R'], known_cam['t'], R_unknown, t_unknown
        )
        
        # Filter for conservative distance range
        filtered = []
        for point in triangulated:
            distance = np.linalg.norm(point)
            if 0.5 < distance < 20.0:
                filtered.append(point)
        
        return filtered
    
    def _estimate_intrinsics_for_unknown_camera(self, 
                                               reconstruction_state: Dict) -> np.ndarray:
        """Estimate intrinsics for unknown camera from existing cameras."""
        
        cameras = reconstruction_state['cameras']
        fallback_K = reconstruction_state.get('camera_matrix')
        
        K_matrices = []
        for camera_data in cameras.values():
            K = camera_data.get('K')
            if K is not None:
                K_matrices.append(K)
        
        if not K_matrices:
            return fallback_K if fallback_K is not None else np.eye(3)
        
        # Average intrinsics
        K_avg = np.mean(np.array(K_matrices), axis=0)
        return K_avg
    
    def _triangulate_point_pairs_progressive(self, 
                                            pts1: np.ndarray, 
                                            pts2: np.ndarray,
                                            P1: np.ndarray, 
                                            P2: np.ndarray,
                                            R1: np.ndarray, 
                                            t1: np.ndarray,
                                            R2: np.ndarray, 
                                            t2: np.ndarray) -> List[np.ndarray]:
        """Triangulate multiple point pairs."""
        
        triangulated = []
        
        for pt1, pt2 in zip(pts1, pts2):
            # DLT triangulation
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
                    
                    # Validate
                    if self._validate_3d_point_progressive(
                        point_3d, P1, P2, pt1, pt2, R1, t1, R2, t2
                    ):
                        triangulated.append(point_3d)
            except:
                continue
        
        return triangulated
    
    def _validate_3d_point_progressive(self, 
                                      point_3d: np.ndarray,
                                      P1: np.ndarray, 
                                      P2: np.ndarray,
                                      pt1: np.ndarray, 
                                      pt2: np.ndarray,
                                      R1: np.ndarray, 
                                      t1: np.ndarray,
                                      R2: np.ndarray, 
                                      t2: np.ndarray) -> bool:
        """Validate 3D point for progressive triangulation."""
        
        if not np.all(np.isfinite(point_3d)):
            return False
        
        # Distance check
        distance = np.linalg.norm(point_3d)
        if not (self.min_depth < distance < self.max_depth):
            return False
        
        # Reprojection error
        X_hom = np.append(point_3d, 1.0)
        
        proj1 = P1 @ X_hom
        proj2 = P2 @ X_hom
        
        if abs(proj1[2]) < 1e-8 or abs(proj2[2]) < 1e-8:
            return False
        
        reproj1 = proj1[:2] / proj1[2]
        reproj2 = proj2[:2] / proj2[2]
        
        error1 = np.linalg.norm(reproj1 - pt1)
        error2 = np.linalg.norm(reproj2 - pt2)
        
        return (error1 < self.max_reprojection_error and 
                error2 < self.max_reprojection_error)
    
    # ============================================================================
    # HELPER METHODS (Keep your existing implementations)
    # ============================================================================
    
    def _create_initial_observations_with_indices(self, 
                                                  points_3d: np.ndarray,
                                                  pts1: np.ndarray, 
                                                  pts2: np.ndarray,
                                                  original_indices: np.ndarray,
                                                  image_pair: Tuple[str, str]) -> List[Dict]:
        """Create observations with original indices."""
        observations = []
        
        for i in range(points_3d.shape[1]):
            original_idx = original_indices[i]
            
            observations.append({
                'point_id': i,
                'original_index': original_idx,
                'image_id': image_pair[0],
                'pixel_coords': pts1[i],
                'feature_id': original_idx
            })
            
            observations.append({
                'point_id': i,
                'original_index': original_idx,
                'image_id': image_pair[1],
                'pixel_coords': pts2[i],
                'feature_id': original_idx
            })
        
        return observations
    
    # Keep all your existing helper methods:
    # - _triangulate_dlt_batch
    # - _filter_triangulated_points
    # - _check_positive_depth
    # - _check_triangulation_angle
    # - _check_reprojection_error
    # - _check_depth_bounds
    # - _apply_hartley_sturm
    # etc.


# Keep your existing TriangulationConfig dataclass unchanged
@dataclass
class TriangulationConfig:
    """Configuration for enhanced triangulation"""
    min_triangulation_angle: float = 2.0
    max_reprojection_error: float = 2.0
    min_distance: float = 0.1
    max_distance: float = 100.0
    enable_progressive: bool = True
    max_points_per_pair: int = 150
    max_total_new_points: int = 800
    progressive_tolerance: float = 4.0
    enable_track_extension: bool = True
    track_extension_tolerance: float = 2.0
    max_track_length: int = 10