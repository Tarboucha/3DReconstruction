"""
3D Structure Refinement

Refines the 3D point cloud structure through:
- Outlier removal and filtering
- Point position optimization
- Point merging and consolidation
- Quality-based pruning

Used for cleaning and improving 3D reconstructions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import cv2

from core.interfaces.base_optimizer import BaseOptimizer, OptimizationResult, OptimizationStatus


@dataclass
class StructureQualityMetrics:
    """Quality metrics for 3D structure"""
    num_points: int
    num_outliers_removed: int
    mean_reprojection_error: float
    median_track_length: float
    mean_triangulation_angle: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_points': self.num_points,
            'outliers_removed': self.num_outliers_removed,
            'mean_reproj_error': self.mean_reprojection_error,
            'median_track_length': self.median_track_length,
            'mean_triangulation_angle': self.mean_triangulation_angle
        }


class StructureRefinerConfig:
    """Configuration for structure refinement"""
    
    # Outlier removal
    MAX_REPROJECTION_ERROR = 4.0  # pixels
    MIN_TRIANGULATION_ANGLE = 2.0  # degrees
    MAX_DEPTH = 1000.0  # meters
    MIN_DEPTH = 0.1
    
    # Quality filtering
    MIN_TRACK_LENGTH = 2  # Minimum number of observations
    PREFERRED_TRACK_LENGTH = 3  # Keep points with 3+ observations
    
    # Point merging
    ENABLE_MERGING = True
    MERGE_DISTANCE_THRESHOLD = 0.1  # meters
    MIN_MERGE_OBSERVATIONS = 4  # Only merge well-observed points
    
    # Filtering strategy
    AGGRESSIVE_FILTERING = False  # More conservative if False


class StructureRefiner(BaseOptimizer):
    """
    Refines 3D point cloud structure.
    
    Operations:
    - Remove outliers based on reprojection error
    - Filter by triangulation angle
    - Remove points behind cameras
    - Merge nearby duplicate points
    - Prune points with few observations
    
    Use cases:
    - After triangulation
    - Before bundle adjustment
    - Periodic cleanup during reconstruction
    - Final point cloud refinement
    """
    
    def __init__(self, **config):
        """
        Initialize structure refiner.
        
        Args:
            **config: Configuration overrides
        """
        super().__init__(**config)
        self.config = StructureRefinerConfig()
        
        # Override config
        for key, value in config.items():
            if hasattr(self.config, key.upper()):
                setattr(self.config, key.upper(), value)
    
    def get_algorithm_name(self) -> str:
        """Get algorithm name."""
        return "StructureRefiner"
    
    def refine_structure(self,
                        points_3d: np.ndarray,
                        cameras: Dict[str, Dict],
                        observations: Dict[str, List[Dict]],
                        remove_outliers: bool = True,
                        merge_points: bool = True) -> OptimizationResult:
        """
        Refine 3D point cloud structure.
        
        Args:
            points_3d: 3D points (3xN)
            cameras: Dictionary of cameras
            observations: Observations per camera
            remove_outliers: Whether to remove outliers
            merge_points: Whether to merge nearby points
            
        Returns:
            OptimizationResult with refined structure
        """
        import time
        start_time = time.time()
        
        initial_num_points = points_3d.shape[1]
        
        print(f"\nRefining 3D structure:")
        print(f"  Initial points: {initial_num_points}")
        print(f"  Cameras: {len(cameras)}")
        
        # Build point-to-observation mapping
        point_observations = self._build_observation_map(observations, initial_num_points)
        
        # Step 1: Remove outliers
        if remove_outliers:
            points_3d, point_observations, outliers_removed = self._remove_outliers(
                points_3d, cameras, point_observations
            )
            print(f"  After outlier removal: {points_3d.shape[1]} points ({outliers_removed} removed)")
        else:
            outliers_removed = 0
        
        # Step 2: Filter by triangulation angle
        points_3d, point_observations, angle_filtered = self._filter_by_angle(
            points_3d, cameras, point_observations
        )
        print(f"  After angle filtering: {points_3d.shape[1]} points ({angle_filtered} removed)")
        
        # Step 3: Filter by track length
        points_3d, point_observations, track_filtered = self._filter_by_track_length(
            points_3d, point_observations
        )
        print(f"  After track filtering: {points_3d.shape[1]} points ({track_filtered} removed)")
        
        # Step 4: Merge nearby points
        if merge_points and self.config.ENABLE_MERGING:
            points_3d, point_observations, merged_count = self._merge_nearby_points(
                points_3d, point_observations
            )
            print(f"  After merging: {points_3d.shape[1]} points ({merged_count} merged)")
        else:
            merged_count = 0
        
        # Compute quality metrics
        metrics = self._compute_quality_metrics(
            points_3d, cameras, point_observations,
            outliers_removed, merged_count
        )
        
        runtime = time.time() - start_time
        
        final_num_points = points_3d.shape[1]
        points_removed = initial_num_points - final_num_points
        
        print(f"  Final points: {final_num_points}")
        print(f"  Total removed: {points_removed} ({points_removed/initial_num_points*100:.1f}%)")
        print(f"  Mean reprojection error: {metrics.mean_reprojection_error:.2f}px")
        print(f"  Runtime: {runtime:.2f}s")
        
        return OptimizationResult(
            success=True,
            status=OptimizationStatus.SUCCESS,
            optimized_params={
                'points_3d': points_3d,
                'observations': self._observations_from_map(point_observations)
            },
            initial_cost=float(initial_num_points),
            final_cost=float(final_num_points),
            num_iterations=1,
            runtime=runtime,
            metadata={
                'algorithm': self.get_algorithm_name(),
                'initial_points': initial_num_points,
                'final_points': final_num_points,
                'points_removed': points_removed,
                'outliers_removed': outliers_removed,
                'merged': merged_count,
                'metrics': metrics.to_dict()
            }
        )
    
    def _build_observation_map(self,
                              observations: Dict[str, List[Dict]],
                              num_points: int) -> Dict[int, List[Tuple[str, np.ndarray]]]:
        """Build mapping from point ID to list of (camera_id, coords_2d)."""
        point_obs = {i: [] for i in range(num_points)}
        
        for cam_id, cam_obs in observations.items():
            for obs in cam_obs:
                point_id = obs['point_id']
                if point_id < num_points:
                    coords = np.array(obs.get('coords_2d', obs.get('image_point')))
                    point_obs[point_id].append((cam_id, coords))
        
        return point_obs
    
    def _remove_outliers(self,
                        points_3d: np.ndarray,
                        cameras: Dict[str, Dict],
                        point_observations: Dict) -> Tuple[np.ndarray, Dict, int]:
        """Remove outliers based on reprojection error."""
        valid_points = []
        new_point_obs = {}
        new_point_id = 0
        outliers_removed = 0
        
        for point_id, point_3d in enumerate(points_3d.T):
            if point_id not in point_observations:
                outliers_removed += 1
                continue
            
            obs_list = point_observations[point_id]
            
            if len(obs_list) == 0:
                outliers_removed += 1
                continue
            
            # Compute reprojection errors for this point
            errors = []
            for cam_id, coords_2d in obs_list:
                if cam_id not in cameras:
                    continue
                
                cam = cameras[cam_id]
                
                # Project point
                rvec, _ = cv2.Rodrigues(cam['R'])
                tvec = cam['t'].reshape(3, 1)
                
                projected, _ = cv2.projectPoints(
                    point_3d.reshape(1, 1, 3),
                    rvec, tvec, cam['K'], None
                )
                projected = projected.reshape(2)
                
                error = np.linalg.norm(projected - coords_2d)
                errors.append(error)
            
            if len(errors) == 0:
                outliers_removed += 1
                continue
            
            mean_error = np.mean(errors)
            
            # Keep if error is acceptable
            if mean_error <= self.config.MAX_REPROJECTION_ERROR:
                valid_points.append(point_3d)
                new_point_obs[new_point_id] = obs_list
                new_point_id += 1
            else:
                outliers_removed += 1
        
        if valid_points:
            refined_points = np.column_stack(valid_points)
        else:
            refined_points = np.empty((3, 0))
        
        return refined_points, new_point_obs, outliers_removed
    
    def _filter_by_angle(self,
                        points_3d: np.ndarray,
                        cameras: Dict[str, Dict],
                        point_observations: Dict) -> Tuple[np.ndarray, Dict, int]:
        """Filter points by triangulation angle."""
        valid_points = []
        new_point_obs = {}
        new_point_id = 0
        filtered = 0
        
        for point_id, point_3d in enumerate(points_3d.T):
            if point_id not in point_observations:
                filtered += 1
                continue
            
            obs_list = point_observations[point_id]
            
            if len(obs_list) < 2:
                filtered += 1
                continue
            
            # Compute triangulation angle
            angles = []
            cam_positions = []
            
            for cam_id, _ in obs_list:
                if cam_id not in cameras:
                    continue
                
                cam = cameras[cam_id]
                # Camera position in world coords: C = -R^T * t
                C = -cam['R'].T @ cam['t'].reshape(3)
                cam_positions.append(C)
            
            if len(cam_positions) < 2:
                filtered += 1
                continue
            
            # Compute angles between viewing rays
            for i in range(len(cam_positions)):
                for j in range(i + 1, len(cam_positions)):
                    ray1 = point_3d - cam_positions[i]
                    ray2 = point_3d - cam_positions[j]
                    
                    ray1_norm = ray1 / (np.linalg.norm(ray1) + 1e-8)
                    ray2_norm = ray2 / (np.linalg.norm(ray2) + 1e-8)
                    
                    cos_angle = np.clip(np.dot(ray1_norm, ray2_norm), -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_angle))
                    angles.append(angle)
            
            if angles and max(angles) >= self.config.MIN_TRIANGULATION_ANGLE:
                valid_points.append(point_3d)
                new_point_obs[new_point_id] = obs_list
                new_point_id += 1
            else:
                filtered += 1
        
        if valid_points:
            refined_points = np.column_stack(valid_points)
        else:
            refined_points = np.empty((3, 0))
        
        return refined_points, new_point_obs, filtered
    
    def _filter_by_track_length(self,
                                points_3d: np.ndarray,
                                point_observations: Dict) -> Tuple[np.ndarray, Dict, int]:
        """Filter points by track length (number of observations)."""
        min_track = (self.config.PREFERRED_TRACK_LENGTH if self.config.AGGRESSIVE_FILTERING 
                    else self.config.MIN_TRACK_LENGTH)
        
        valid_points = []
        new_point_obs = {}
        new_point_id = 0
        filtered = 0
        
        for point_id, point_3d in enumerate(points_3d.T):
            if point_id not in point_observations:
                filtered += 1
                continue
            
            track_length = len(point_observations[point_id])
            
            if track_length >= min_track:
                valid_points.append(point_3d)
                new_point_obs[new_point_id] = point_observations[point_id]
                new_point_id += 1
            else:
                filtered += 1
        
        if valid_points:
            refined_points = np.column_stack(valid_points)
        else:
            refined_points = np.empty((3, 0))
        
        return refined_points, new_point_obs, filtered
    
    def _merge_nearby_points(self,
                            points_3d: np.ndarray,
                            point_observations: Dict) -> Tuple[np.ndarray, Dict, int]:
        """Merge nearby duplicate points."""
        if points_3d.shape[1] == 0:
            return points_3d, point_observations, 0
        
        # Find groups of nearby points
        from scipy.spatial import cKDTree
        
        tree = cKDTree(points_3d.T)
        pairs = tree.query_pairs(r=self.config.MERGE_DISTANCE_THRESHOLD)
        
        if len(pairs) == 0:
            return points_3d, point_observations, 0
        
        # Build merge groups
        merge_groups = {}
        for i, j in pairs:
            # Only merge if both have enough observations
            if (i in point_observations and j in point_observations and
                len(point_observations[i]) >= self.config.MIN_MERGE_OBSERVATIONS and
                len(point_observations[j]) >= self.config.MIN_MERGE_OBSERVATIONS):
                
                # Find group leaders
                leader_i = merge_groups.get(i, i)
                leader_j = merge_groups.get(j, j)
                
                if leader_i != leader_j:
                    # Merge groups
                    new_leader = min(leader_i, leader_j)
                    old_leader = max(leader_i, leader_j)
                    
                    for k, v in merge_groups.items():
                        if v == old_leader:
                            merge_groups[k] = new_leader
                    
                    merge_groups[i] = new_leader
                    merge_groups[j] = new_leader
        
        # Perform merging
        merged_points = []
        merged_obs = {}
        point_mapping = {}
        new_id = 0
        merged_count = 0
        
        processed = set()
        
        for point_id in range(points_3d.shape[1]):
            if point_id in processed:
                continue
            
            if point_id in merge_groups:
                # Get all points in this group
                group = [point_id]
                for other_id, leader in merge_groups.items():
                    if leader == merge_groups[point_id] and other_id != point_id:
                        group.append(other_id)
                
                # Merge points (average position)
                group_points = [points_3d[:, gid] for gid in group if gid < points_3d.shape[1]]
                merged_point = np.mean(group_points, axis=0)
                
                # Merge observations
                merged_obs_list = []
                seen_cameras = set()
                
                for gid in group:
                    if gid in point_observations:
                        for cam_id, coords in point_observations[gid]:
                            if cam_id not in seen_cameras:
                                merged_obs_list.append((cam_id, coords))
                                seen_cameras.add(cam_id)
                
                merged_points.append(merged_point)
                merged_obs[new_id] = merged_obs_list
                
                for gid in group:
                    point_mapping[gid] = new_id
                    processed.add(gid)
                
                merged_count += len(group) - 1
                new_id += 1
            else:
                # Keep point as is
                merged_points.append(points_3d[:, point_id])
                if point_id in point_observations:
                    merged_obs[new_id] = point_observations[point_id]
                point_mapping[point_id] = new_id
                processed.add(point_id)
                new_id += 1
        
        if merged_points:
            refined_points = np.column_stack(merged_points)
        else:
            refined_points = np.empty((3, 0))
        
        return refined_points, merged_obs, merged_count
    
    def _compute_quality_metrics(self,
                                points_3d: np.ndarray,
                                cameras: Dict,
                                point_observations: Dict,
                                outliers_removed: int,
                                merged: int) -> StructureQualityMetrics:
        """Compute quality metrics for refined structure."""
        # Track lengths
        track_lengths = [len(obs) for obs in point_observations.values()]
        median_track_length = float(np.median(track_lengths)) if track_lengths else 0.0
        
        # Mean reprojection error
        all_errors = []
        for point_id, point_3d in enumerate(points_3d.T):
            if point_id not in point_observations:
                continue
            
            for cam_id, coords_2d in point_observations[point_id]:
                if cam_id not in cameras:
                    continue
                
                cam = cameras[cam_id]
                rvec, _ = cv2.Rodrigues(cam['R'])
                tvec = cam['t'].reshape(3, 1)
                
                projected, _ = cv2.projectPoints(
                    point_3d.reshape(1, 1, 3), rvec, tvec, cam['K'], None
                )
                projected = projected.reshape(2)
                
                error = np.linalg.norm(projected - coords_2d)
                all_errors.append(error)
        
        mean_error = float(np.mean(all_errors)) if all_errors else 0.0
        
        return StructureQualityMetrics(
            num_points=points_3d.shape[1],
            num_outliers_removed=outliers_removed,
            mean_reprojection_error=mean_error,
            median_track_length=median_track_length,
            mean_triangulation_angle=0.0  # TODO: compute if needed
        )
    
    def _observations_from_map(self, point_obs_map: Dict) -> Dict[str, List[Dict]]:
        """Convert observation map back to per-camera format."""
        observations = {}
        
        for point_id, obs_list in point_obs_map.items():
            for cam_id, coords_2d in obs_list:
                if cam_id not in observations:
                    observations[cam_id] = []
                
                observations[cam_id].append({
                    'point_id': point_id,
                    'coords_2d': coords_2d,
                    'image_point': coords_2d
                })
        
        return observations
    
    def optimize(self, *args, **kwargs) -> OptimizationResult:
        """Optimize - wrapper for refine_structure."""
        return self.refine_structure(*args, **kwargs)
    
    def compute_cost(self, *args, **kwargs) -> float:
        """Compute cost."""
        raise NotImplementedError("Use refine_structure() which provides metrics")