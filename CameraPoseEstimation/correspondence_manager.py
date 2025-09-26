"""
3D Correspondence Management Module

This module handles all aspects of finding, creating, and managing 2D-3D correspondences
for incremental Structure-from-Motion pipelines.

Classes:
- CorrespondenceConfig: Configuration for correspondence finding
- CorrespondenceFinder: Main correspondence finding logic
- PreTriangulator: Creates new 3D points before PnP
- ImageSelector: Smart image selection based on 3D overlap
- CorrespondenceDiagnostics: Debugging and analysis tools
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
import copy


@dataclass
class CorrespondenceConfig:
    """Configuration for 3D correspondence operations"""
    # Correspondence matching
    base_tolerance: float = 2.0              # Base pixel tolerance
    fallback_tolerances: List[float] = None  # Progressive tolerances
    min_correspondences_strict: int = 15     # Preferred minimum
    min_correspondences_fallback: int = 6    # Absolute minimum
    
    # Pre-triangulation
    enable_pre_triangulation: bool = True
    max_new_points_per_camera: int = 50      # Limit points per camera pair
    triangulation_distance_range: Tuple[float, float] = (0.1, 100.0)
    
    # Image selection
    overlap_weight: float = 0.6              # Weight for 3D overlap score
    match_quality_weight: float = 0.3        # Weight for match quality
    coverage_weight: float = 0.1             # Weight for total coverage
    
    def __post_init__(self):
        if self.fallback_tolerances is None:
            self.fallback_tolerances = [2.0, 3.0, 4.0, 5.0, 7.0]


class MatchExtractor:
    """Handles extraction of point matches from various data formats"""
    
    @staticmethod
    def find_matches_between_images(img1: str, img2: str, matches_data: Dict) -> Optional[Dict]:
        """Find matches between two specific images with flexible key handling"""
        
        # Try different key formats
        possible_keys = [
            (img1, img2), (img2, img1),
            f"{img1}_vs_{img2}", f"{img2}_vs_{img1}",
            f"{img1}_{img2}", f"{img2}_{img1}",
            f"{img1}__{img2}", f"{img2}__{img1}"
        ]
        
        for key in possible_keys:
            if key in matches_data:
                match_data = matches_data[key]
                
                # Determine image order
                if isinstance(key, tuple):
                    img1_is_first = (key[0] == img1)
                else:
                    img1_is_first = str(key).startswith(img1)
                
                # Extract points
                points = MatchExtractor._extract_point_arrays(match_data, img1_is_first)
                if points is not None:
                    return points
        
        return None
    
    @staticmethod
    def _extract_point_arrays(match_data: Dict, img1_is_first: bool) -> Optional[Dict]:
        """Extract point arrays from various match data formats"""
        try:
            # Handle LightGlue correspondences
            if 'correspondences' in match_data:
                correspondences = np.array(match_data['correspondences'])
                if correspondences.ndim == 2 and correspondences.shape[1] >= 4:
                    pts1 = correspondences[:, :2]
                    pts2 = correspondences[:, 2:4]
                else:
                    return None
            
            # Handle traditional format
            elif 'pts1' in match_data and 'pts2' in match_data:
                pts1 = np.array(match_data['pts1'])
                pts2 = np.array(match_data['pts2'])
            
            # Handle keypoints
            elif 'keypoints1' in match_data and 'keypoints2' in match_data:
                kp1, kp2 = match_data['keypoints1'], match_data['keypoints2']
                
                if hasattr(kp1[0], 'pt'):  # cv2.KeyPoint objects
                    pts1 = np.array([kp.pt for kp in kp1])
                    pts2 = np.array([kp.pt for kp in kp2])
                else:
                    pts1 = np.array(kp1)
                    pts2 = np.array(kp2)
            else:
                return None
            
            # Normalize arrays
            pts1 = MatchExtractor._normalize_points(pts1)
            pts2 = MatchExtractor._normalize_points(pts2)
            
            if len(pts1) == 0 or len(pts2) == 0:
                return None
            
            # Return based on image order
            if img1_is_first:
                return {'img1_points': pts1, 'img2_points': pts2}
            else:
                return {'img1_points': pts2, 'img2_points': pts1}
        
        except Exception as e:
            print(f"Error extracting points: {e}")
            return None
    
    @staticmethod
    def _normalize_points(points) -> np.ndarray:
        """Normalize point array to (N, 2) format"""
        if points is None or len(points) == 0:
            return np.empty((0, 2))
        
        points = np.array(points, dtype=np.float32)
        
        if points.ndim == 1:
            return points.reshape(1, 2) if len(points) == 2 else np.empty((0, 2))
        elif points.ndim == 2:
            if points.shape[1] == 2:
                return points
            elif points.shape[1] > 2:
                return points[:, :2]
            elif points.shape[0] == 2 and points.shape[1] > 2:
                return points.T
        elif points.ndim == 3 and points.shape[1:] == (1, 2):
            return points.squeeze(axis=1)
        
        return np.empty((0, 2))


class PreTriangulator:
    """Handles pre-triangulation of 3D points before PnP"""
    
    def __init__(self, config: CorrespondenceConfig):
        self.config = config
    
    def triangulate_with_all_cameras(self, new_image: str, reconstruction_state: Dict,
                                   matches_data: Dict) -> int:
        """
        Triangulate new 3D points by matching new image with ALL existing cameras
        """
        if not self.config.enable_pre_triangulation:
            return 0
        
        print(f"Pre-triangulating 3D points for {new_image}...")
        
        existing_cameras = list(reconstruction_state['cameras'].keys())
        all_new_triangulated = []
        
        for existing_camera in existing_cameras:
            print(f"  Processing matches with {existing_camera}...")
            
            # Find matches
            matches = MatchExtractor.find_matches_between_images(
                new_image, existing_camera, matches_data
            )
            
            if matches is None or len(matches['img1_points']) == 0:
                continue
            
            # Filter unmatched points
            unmatched = self._filter_unmatched_points(
                matches, existing_camera, reconstruction_state, new_image
            )
            
            if len(unmatched['new_points']) == 0:
                continue
            
            # Triangulate
            triangulated = self._triangulate_point_pairs(
                unmatched, existing_camera, reconstruction_state
            )
            
            if triangulated:
                all_new_triangulated.extend(triangulated)
                print(f"    Triangulated {len(triangulated)} new points")
            
            # Limit total new points
            if len(all_new_triangulated) >= 100:  # Global limit
                break
        
        # Add to reconstruction
        if all_new_triangulated:
            self._add_points_to_reconstruction(
                reconstruction_state, all_new_triangulated, new_image
            )
        
        return len(all_new_triangulated)
    
    def _filter_unmatched_points(self, matches: Dict, existing_camera: str,
                               reconstruction_state: Dict, new_image: str) -> Dict:
        """Filter out points that already have 3D correspondences"""
        
        observations = reconstruction_state.get('observations', {})
        
        # Determine which points belong to new vs existing image
        if existing_camera in matches:  # Direct camera key
            new_points = matches[new_image] if new_image in matches else matches['img1_points']
            existing_points = matches[existing_camera] if existing_camera in matches else matches['img2_points']
        else:  # Standard format
            new_points = matches['img1_points']
            existing_points = matches['img2_points']
        
        if existing_camera not in observations or not observations[existing_camera]:
            return {'new_points': new_points, 'existing_points': existing_points}
        
        # Filter based on existing observations
        obs_points = np.array([obs['image_point'] for obs in observations[existing_camera]])
        
        filtered_new = []
        filtered_existing = []
        
        for new_pt, existing_pt in zip(new_points, existing_points):
            # Check if existing point is far from observations (unmatched)
            distances = np.linalg.norm(obs_points - existing_pt.reshape(1, -1), axis=1)
            min_distance = np.min(distances)
            
            if min_distance > 4.0:  # Not near existing observations
                filtered_new.append(new_pt)
                filtered_existing.append(existing_pt)
        
        return {
            'new_points': np.array(filtered_new) if filtered_new else np.empty((0, 2)),
            'existing_points': np.array(filtered_existing) if filtered_existing else np.empty((0, 2))
        }
    
    def _triangulate_point_pairs(self, matches: Dict, existing_camera: str,
                               reconstruction_state: Dict) -> List[Dict]:
        """Triangulate 3D points from point pairs"""
        
        new_points = matches['new_points']
        existing_points = matches['existing_points']
        
        if len(new_points) == 0:
            return []
        
        # Get camera parameters
        cameras = reconstruction_state['cameras']
        K = cameras[existing_camera].get('K', reconstruction_state.get('camera_matrix'))
        R_exist = cameras[existing_camera]['R']
        t_exist = cameras[existing_camera]['t']
        
        # Assume new camera at origin for rough triangulation
        R_new = np.eye(3)
        t_new = np.zeros((3, 1))
        
        # Create projection matrices
        P_new = K @ np.hstack([R_new, t_new])
        P_exist = K @ np.hstack([R_exist, t_exist])
        
        triangulated = []
        
        for i, (new_pt, exist_pt) in enumerate(zip(new_points, existing_points)):
            point_3d = self._triangulate_dlt(new_pt, exist_pt, P_new, P_exist)
            
            if self._validate_3d_point(point_3d):
                triangulated.append({
                    'point_3d': point_3d,
                    'new_image_point': new_pt,
                    'existing_image_point': exist_pt,
                    'existing_camera': existing_camera
                })
        
        return triangulated[:self.config.max_new_points_per_camera]
    
    def _triangulate_dlt(self, pt1: np.ndarray, pt2: np.ndarray, 
                        P1: np.ndarray, P2: np.ndarray) -> Optional[np.ndarray]:
        """Direct Linear Transform triangulation"""
        try:
            A = np.array([
                pt1[0] * P1[2, :] - P1[0, :],
                pt1[1] * P1[2, :] - P1[1, :],
                pt2[0] * P2[2, :] - P2[0, :],
                pt2[1] * P2[2, :] - P2[1, :]
            ])
            
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1, :]
            
            if abs(X[3]) > 1e-8:
                return X[:3] / X[3]
            
            return None
        except:
            return None
    
    def _validate_3d_point(self, point_3d: Optional[np.ndarray]) -> bool:
        """Validate triangulated 3D point"""
        if point_3d is None:
            return False
        
        if not np.all(np.isfinite(point_3d)):
            return False
        
        distance = np.linalg.norm(point_3d)
        return self.config.triangulation_distance_range[0] < distance < self.config.triangulation_distance_range[1]
    
    def _add_points_to_reconstruction(self, reconstruction_state: Dict, 
                                    new_triangulated: List[Dict], new_image: str):
        """Add triangulated points to reconstruction state"""
        
        # Extract 3D points
        new_3d_points = np.array([tp['point_3d'] for tp in new_triangulated]).T
        
        # Add to points_3d structure
        points_3d_data = reconstruction_state.get('points_3d', {})
        if isinstance(points_3d_data, dict) and 'points_3d' in points_3d_data:
            current_points = points_3d_data['points_3d']
            updated_points = np.hstack([current_points, new_3d_points])
            reconstruction_state['points_3d']['points_3d'] = updated_points
        else:
            current_points = points_3d_data if hasattr(points_3d_data, 'shape') else np.empty((3, 0))
            updated_points = np.hstack([current_points, new_3d_points]) if current_points.size > 0 else new_3d_points
            reconstruction_state['points_3d'] = updated_points
        
        # Update observations
        self._update_observations(reconstruction_state, new_triangulated, new_image)
    
    def _update_observations(self, reconstruction_state: Dict, 
                           new_triangulated: List[Dict], new_image: str):
        """Update observation structure with new points"""
        
        observations = reconstruction_state.get('observations', {})
        if new_image not in observations:
            observations[new_image] = []
        
        # Get starting point ID
        points_3d_data = reconstruction_state.get('points_3d', {})
        if isinstance(points_3d_data, dict) and 'points_3d' in points_3d_data:
            total_points = points_3d_data['points_3d'].shape[1]
        else:
            total_points = points_3d_data.shape[1] if hasattr(points_3d_data, 'shape') else 0
        
        starting_id = total_points - len(new_triangulated)
        
        for i, tp in enumerate(new_triangulated):
            point_id = starting_id + i
            
            # Add for new image
            observations[new_image].append({
                'point_id': point_id,
                'image_point': [float(tp['new_image_point'][0]), float(tp['new_image_point'][1])],
                'source': 'pre_triangulation'
            })
            
            # Add for existing camera
            existing_camera = tp['existing_camera']
            if existing_camera not in observations:
                observations[existing_camera] = []
            
            observations[existing_camera].append({
                'point_id': point_id,
                'image_point': [float(tp['existing_image_point'][0]), float(tp['existing_image_point'][1])],
                'source': 'pre_triangulation'
            })


class CorrespondenceFinder:
    """Main class for finding 2D-3D correspondences"""
    
    def __init__(self, config: CorrespondenceConfig):
        self.config = config
    
    def find_correspondences_with_fallback(self, new_image: str, reconstruction_state: Dict,
                                         matches_data: Dict) -> Dict[str, Any]:
        """
        Find correspondences with progressive fallback tolerances
        """
        print(f"Finding correspondences for {new_image}...")
        
        best_result = None
        
        for tolerance in self.config.fallback_tolerances:
            print(f"  Trying tolerance: {tolerance:.1f} pixels")
            
            result = self._find_correspondences_with_tolerance(
                new_image, reconstruction_state, matches_data, tolerance
            )
            
            print(f"    Found {result['num_correspondences']} correspondences")
            
            # Check sufficiency
            if result['num_correspondences'] >= self.config.min_correspondences_strict:
                print(f"    ✅ Sufficient correspondences found")
                return result
            elif result['num_correspondences'] >= self.config.min_correspondences_fallback:
                if best_result is None or result['num_correspondences'] > best_result['num_correspondences']:
                    best_result = result
                    print(f"    💾 Stored as fallback")
        
        # Return best result or empty
        if best_result is not None:
            print(f"  Using fallback: {best_result['num_correspondences']} correspondences")
            return best_result
        else:
            print(f"  ❌ Insufficient correspondences found")
            return self._empty_result()
    
    def _find_correspondences_with_tolerance(self, new_image: str, reconstruction_state: Dict,
                                           matches_data: Dict, tolerance: float) -> Dict[str, Any]:
        """Find correspondences using specific tolerance"""
        
        existing_cameras = set(reconstruction_state['cameras'].keys())
        all_correspondences = []
        
        # Get 3D points and observations
        points_3d = self._get_points_3d(reconstruction_state)
        if points_3d.size == 0:
            return self._empty_result()
        
        observations = reconstruction_state.get('observations', {})
        
        # Process each existing camera
        for existing_camera in existing_cameras:
            if existing_camera not in observations or not observations[existing_camera]:
                continue
            
            # Find matches
            matches = MatchExtractor.find_matches_between_images(
                new_image, existing_camera, matches_data
            )
            
            if matches is None:
                continue
            
            # Find correspondences
            camera_correspondences = self._match_with_tolerance(
                matches, existing_camera, observations[existing_camera], 
                points_3d, tolerance, new_image
            )
            
            all_correspondences.extend(camera_correspondences)
        
        # Filter and format
        filtered = self._filter_correspondences(all_correspondences)
        return self._format_result(filtered)
    
    def _match_with_tolerance(self, matches: Dict, existing_camera: str,
                            existing_observations: List[Dict], points_3d: np.ndarray,
                            tolerance: float, new_image: str) -> List[Dict]:
        """Match points with specific tolerance"""
        
        correspondences = []
        obs_points = np.array([obs['image_point'] for obs in existing_observations])
        
        # Determine point assignment based on image names in matches
        new_points = matches['img1_points']
        existing_points = matches['img2_points']
        
        for new_pt, existing_pt in zip(new_points, existing_points):
            # Find closest observation
            distances = np.linalg.norm(obs_points - existing_pt.reshape(1, -1), axis=1)
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]
            
            if min_distance < tolerance:
                closest_obs = existing_observations[min_idx]
                point_3d_id = closest_obs['point_id']
                
                if 0 <= point_3d_id < points_3d.shape[1]:
                    confidence = max(0.3, 1.0 - (min_distance / tolerance))
                    
                    correspondences.append({
                        'point_3d_id': point_3d_id,
                        'point_3d': points_3d[:, point_3d_id],
                        'point_2d': new_pt,
                        'confidence': confidence,
                        'source_camera': existing_camera,
                        'match_distance': min_distance
                    })
        
        return correspondences
    
    def _get_points_3d(self, reconstruction_state: Dict) -> np.ndarray:
        """Extract 3D points from reconstruction state"""
        points_3d_data = reconstruction_state.get('points_3d', {})
        
        if isinstance(points_3d_data, dict) and 'points_3d' in points_3d_data:
            return points_3d_data['points_3d']
        else:
            return points_3d_data if hasattr(points_3d_data, 'shape') else np.empty((3, 0))
    
    def _filter_correspondences(self, correspondences: List[Dict]) -> List[Dict]:
        """Filter and deduplicate correspondences"""
        if not correspondences:
            return []
        
        # Remove duplicates (same 3D point)
        unique = {}
        for corr in correspondences:
            point_id = corr['point_3d_id']
            if point_id not in unique or corr['confidence'] > unique[point_id]['confidence']:
                unique[point_id] = corr
        
        # Basic validation
        filtered = []
        for corr in unique.values():
            if (np.all(np.isfinite(corr['point_3d'])) and 
                np.all(np.isfinite(corr['point_2d'])) and
                0.01 < np.linalg.norm(corr['point_3d']) < 500):
                filtered.append(corr)
        
        return filtered
    
    def _format_result(self, correspondences: List[Dict]) -> Dict[str, Any]:
        """Format correspondences for PnP solver"""
        if not correspondences:
            return self._empty_result()
        
        points_3d = np.array([c['point_3d'] for c in correspondences], dtype=np.float32)
        points_2d = np.array([c['point_2d'] for c in correspondences], dtype=np.float32)
        point_ids = np.array([c['point_3d_id'] for c in correspondences], dtype=int)
        confidences = np.array([c['confidence'] for c in correspondences], dtype=np.float32)
        
        return {
            'points_3d': points_3d,
            'points_2d': points_2d,
            'point_ids': point_ids,
            'confidence_scores': confidences,
            'num_correspondences': len(correspondences),
            'mean_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty correspondence result"""
        return {
            'points_3d': np.empty((0, 3), dtype=np.float32),
            'points_2d': np.empty((0, 2), dtype=np.float32),
            'point_ids': np.empty(0, dtype=int),
            'confidence_scores': np.empty(0, dtype=np.float32),
            'num_correspondences': 0,
            'mean_confidence': 0.0,
            'min_confidence': 0.0,
            'max_confidence': 0.0
        }


class ImageSelector:
    """Smart image selection based on 3D point overlap potential"""
    
    def __init__(self, config: CorrespondenceConfig):
        self.config = config
    
    def select_best_image(self, reconstruction_state: Dict, remaining_images: Set[str],
                         matches_data: Dict) -> Optional[str]:
        """Select best next image based on 3D overlap potential"""
        
        if not remaining_images:
            return None
        
        print(f"Selecting best image from {len(remaining_images)} candidates...")
        
        existing_cameras = set(reconstruction_state['cameras'].keys())
        candidate_scores = []
        
        for candidate in remaining_images:
            score = self._score_candidate(candidate, existing_cameras, 
                                        reconstruction_state, matches_data)
            candidate_scores.append(score)
            
            print(f"  {candidate}: score={score['total_score']:.3f}, "
                  f"est_corr={score['estimated_correspondences']}")
        
        # Filter valid candidates
        valid = [s for s in candidate_scores if s['estimated_correspondences'] > 0]
        
        if not valid:
            return None
        
        # Select best
        best = max(valid, key=lambda x: x['total_score'])
        
        print(f"✅ Selected: {best['image']} (score: {best['total_score']:.3f})")
        return best['image']
    
    def _score_candidate(self, candidate: str, existing_cameras: Set[str],
                        reconstruction_state: Dict, matches_data: Dict) -> Dict[str, Any]:
        """Score a candidate image"""
        
        observations = reconstruction_state.get('observations', {})
        total_matches = 0
        estimated_correspondences = 0
        match_qualities = []
        
        # Analyze matches with each existing camera
        for existing_camera in existing_cameras:
            matches = MatchExtractor.find_matches_between_images(
                candidate, existing_camera, matches_data
            )
            
            if matches is None:
                continue
            
            num_matches = len(matches['img1_points'])
            total_matches += num_matches
            match_qualities.append(0.7)  # Default quality
            
            # Estimate potential correspondences
            if existing_camera in observations:
                obs_points = np.array([obs['image_point'] for obs in observations[existing_camera]])
                if len(obs_points) > 0:
                    # Count matches near observations
                    existing_points = matches['img2_points']
                    for existing_pt in existing_points:
                        distances = np.linalg.norm(obs_points - existing_pt.reshape(1, -1), axis=1)
                        min_dist = np.min(distances)
                        
                        # Progressive scoring based on tolerance
                        for tolerance in [3.0, 5.0, 7.0]:
                            if min_dist < tolerance:
                                weight = 1.0 - (min_dist / tolerance) * 0.5
                                estimated_correspondences += weight
                                break
        
        # Calculate scores
        overlap_score = min(1.0, estimated_correspondences / 20.0)
        match_quality = np.mean(match_qualities) if match_qualities else 0.0
        coverage_score = min(1.0, total_matches / 100.0)
        
        total_score = (
            self.config.overlap_weight * overlap_score +
            self.config.match_quality_weight * match_quality +
            self.config.coverage_weight * coverage_score
        )
        
        return {
            'image': candidate,
            'total_score': total_score,
            'overlap_score': overlap_score,
            'match_quality': match_quality,
            'estimated_correspondences': int(estimated_correspondences),
            'total_matches': total_matches
        }


class CorrespondenceDiagnostics:
    """Diagnostics and debugging tools for correspondence issues"""
    
    @staticmethod
    def diagnose_failure(failed_image: str, reconstruction_state: Dict, 
                        matches_data: Dict) -> Dict[str, Any]:
        """Diagnose why correspondence finding failed"""
        
        print(f"\n=== DIAGNOSING FAILURE: {failed_image} ===")
        
        existing_cameras = list(reconstruction_state['cameras'].keys())
        observations = reconstruction_state.get('observations', {})
        points_3d = CorrespondenceFinder._get_points_3d_static(reconstruction_state)
        
        diagnosis = {
            'failed_image': failed_image,
            'num_cameras': len(existing_cameras),
            'num_3d_points': points_3d.shape[1] if points_3d.size > 0 else 0,
            'camera_analysis': {},
            'recommendations': []
        }
        
        total_matches = 0
        
        for camera in existing_cameras:
            matches = MatchExtractor.find_matches_between_images(
                failed_image, camera, matches_data
            )
            
            num_matches = len(matches['img1_points']) if matches else 0
            num_obs = len(observations.get(camera, []))
            
            diagnosis['camera_analysis'][camera] = {
                'matches': num_matches,
                'observations': num_obs,
                'potential_overlap': min(num_matches, num_obs) if num_matches > 0 else 0
            }
            
            total_matches += num_matches
            print(f"  {camera}: {num_matches} matches, {num_obs} observations")
        
        # Generate recommendations
        if total_matches == 0:
            diagnosis['recommendations'].append("No feature matches - check feature detection")
        elif total_matches < 20:
            diagnosis['recommendations'].append("Few matches - try different images")
        elif diagnosis['num_3d_points'] < 50:
            diagnosis['recommendations'].append("Few 3D points - use aggressive triangulation")
        else:
            diagnosis['recommendations'].append("Try relaxed tolerance or different initialization")
        
        return diagnosis
    
    @staticmethod
    def _get_points_3d_static(reconstruction_state: Dict) -> np.ndarray:
        """Static version of get_points_3d for diagnostics"""
        points_3d_data = reconstruction_state.get('points_3d', {})
        
        if isinstance(points_3d_data, dict) and 'points_3d' in points_3d_data:
            return points_3d_data['points_3d']
        else:
            return points_3d_data if hasattr(points_3d_data, 'shape') else np.empty((3, 0))


# Main interface class
class CorrespondenceManager:
    """
    Main interface for all 3D correspondence operations
    Coordinates between PreTriangulator, CorrespondenceFinder, and ImageSelector
    """
    
    def __init__(self, config: Optional[CorrespondenceConfig] = None):
        self.config = config or CorrespondenceConfig()
        
        # Initialize components
        self.pre_triangulator = PreTriangulator(self.config)
        self.correspondence_finder = CorrespondenceFinder(self.config)
        self.image_selector = ImageSelector(self.config)
        self.diagnostics = CorrespondenceDiagnostics()
    
    def select_next_image(self, reconstruction_state: Dict, remaining_images: Set[str],
                         matches_data: Dict) -> Optional[str]:
        """Select the best next image to add to reconstruction"""
        return self.image_selector.select_best_image(
            reconstruction_state, remaining_images, matches_data
        )
    
    def prepare_correspondences_for_new_view(self, new_image: str, reconstruction_state: Dict,
                                           matches_data: Dict) -> Dict[str, Any]:
        """
        Complete correspondence preparation pipeline for adding a new view
        
        Returns correspondence data ready for PnP solver
        """
        print(f"\n=== PREPARING CORRESPONDENCES FOR {new_image} ===")
        
        # Step 1: Pre-triangulation to create more 3D points
        if self.config.enable_pre_triangulation:
            print("Step 1: Pre-triangulation...")
            new_points_created = self.pre_triangulator.triangulate_with_all_cameras(
                new_image, reconstruction_state, matches_data
            )
            print(f"  Created {new_points_created} new 3D points")
        else:
            print("Step 1: Pre-triangulation disabled")
        
        # Step 2: Find 2D-3D correspondences with fallback tolerance
        print("Step 2: Finding correspondences...")
        correspondences = self.correspondence_finder.find_correspondences_with_fallback(
            new_image, reconstruction_state, matches_data
        )
        
        print(f"  Found {correspondences['num_correspondences']} correspondences")
        
        # Step 3: Validate sufficiency
        if correspondences['num_correspondences'] < self.config.min_correspondences_fallback:
            print(f"❌ Insufficient correspondences: {correspondences['num_correspondences']}")
            
            # Run diagnostics
            diagnosis = self.diagnostics.diagnose_failure(new_image, reconstruction_state, matches_data)
            print("Diagnostic recommendations:")
            for rec in diagnosis['recommendations']:
                print(f"  - {rec}")
            
            return correspondences  # Return empty/insufficient result
        
        print(f"✅ Sufficient correspondences for PnP")
        return correspondences
    
    def update_observations_after_pnp(self, reconstruction_state: Dict, new_image: str,
                                    correspondences: Dict[str, Any]) -> Dict[str, Any]:
        """Update observation structure after successful PnP"""
        
        observations = reconstruction_state.get('observations', {})
        if new_image not in observations:
            observations[new_image] = []
        
        # Add observations from correspondences used in PnP
        if correspondences['num_correspondences'] > 0:
            points_2d = correspondences['points_2d']
            point_ids = correspondences['point_ids']
            
            added_count = 0
            for i in range(correspondences['num_correspondences']):
                observation = {
                    'point_id': int(point_ids[i]),
                    'image_point': [float(points_2d[i][0]), float(points_2d[i][1])],
                    'source': 'pnp_correspondence'
                }
                
                # Avoid duplicates
                if not self._observation_exists(observations[new_image], observation):
                    observations[new_image].append(observation)
                    added_count += 1
            
            print(f"Added {added_count} observations for {new_image}")
        
        return reconstruction_state
    
    def assess_remaining_images(self, remaining_images: Set[str], reconstruction_state: Dict,
                              matches_data: Dict) -> float:
        """Assess likelihood of success for remaining images"""
        
        if not remaining_images:
            return 0.0
        
        scores = []
        for candidate in remaining_images:
            score_info = self.image_selector._score_candidate(
                candidate, set(reconstruction_state['cameras'].keys()),
                reconstruction_state, matches_data
            )
            scores.append(score_info['total_score'])
        
        return np.mean(scores)
    
    def get_reconstruction_statistics(self, reconstruction_state: Dict) -> Dict[str, Any]:
        """Get comprehensive statistics about the current reconstruction"""
        
        cameras = reconstruction_state.get('cameras', {})
        observations = reconstruction_state.get('observations', {})
        points_3d = self.correspondence_finder._get_points_3d(reconstruction_state)
        
        stats = {
            'num_cameras': len(cameras),
            'num_3d_points': points_3d.shape[1] if points_3d.size > 0 else 0,
            'total_observations': sum(len(obs) for obs in observations.values()),
            'observations_per_camera': {
                cam_id: len(obs) for cam_id, obs in observations.items()
            },
            'points_3d_stats': {}
        }
        
        if points_3d.size > 0:
            distances = np.linalg.norm(points_3d, axis=0)
            stats['points_3d_stats'] = {
                'mean_distance': float(np.mean(distances)),
                'std_distance': float(np.std(distances)),
                'min_distance': float(np.min(distances)),
                'max_distance': float(np.max(distances))
            }
        
        # Calculate coverage metrics
        if stats['num_cameras'] > 0:
            obs_counts = list(stats['observations_per_camera'].values())
            stats['avg_observations_per_camera'] = np.mean(obs_counts)
            stats['min_observations_per_camera'] = np.min(obs_counts)
            stats['max_observations_per_camera'] = np.max(obs_counts)
        
        return stats
    
    def _observation_exists(self, observations_list: List[Dict], new_observation: Dict,
                           tolerance: float = 2.0) -> bool:
        """Check if observation already exists to avoid duplicates"""
        
        new_point_id = new_observation['point_id']
        new_image_point = np.array(new_observation['image_point'])
        
        for existing_obs in observations_list:
            if existing_obs['point_id'] == new_point_id:
                existing_image_point = np.array(existing_obs['image_point'])
                distance = np.linalg.norm(new_image_point - existing_image_point)
                if distance < tolerance:
                    return True
        
        return False
    
    def print_statistics(self, reconstruction_state: Dict):
        """Print formatted reconstruction statistics"""
        
        stats = self.get_reconstruction_statistics(reconstruction_state)
        
        print(f"\n{'='*50}")
        print("RECONSTRUCTION STATISTICS")
        print(f"{'='*50}")
        print(f"Cameras: {stats['num_cameras']}")
        print(f"3D Points: {stats['num_3d_points']}")
        print(f"Total Observations: {stats['total_observations']}")
        
        if stats['num_cameras'] > 0:
            print(f"Avg observations per camera: {stats['avg_observations_per_camera']:.1f}")
        
        if stats['points_3d_stats']:
            p3d = stats['points_3d_stats']
            print(f"3D point distances: {p3d['mean_distance']:.2f} ± {p3d['std_distance']:.2f}")
            print(f"Distance range: [{p3d['min_distance']:.2f}, {p3d['max_distance']:.2f}]")
        
        print("Observations per camera:")
        for cam_id, count in stats['observations_per_camera'].items():
            print(f"  {cam_id}: {count}")
        
        print(f"{'='*50}")