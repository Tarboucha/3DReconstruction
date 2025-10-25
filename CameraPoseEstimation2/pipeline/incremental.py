"""
Incremental 3D Reconstruction Pipeline

Main pipeline orchestrating the complete reconstruction process using modular components
from the algorithms/ structure.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import pickle
import json
import copy
import cv2

# Optional scipy for quaternion conversion
try:
    from scipy.spatial.transform import Rotation
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    # Scipy warning will be shown only when COLMAP export is used

# Geometry modules
from CameraPoseEstimation2.algorithms.geometry.triangulation import TriangulationEngine
from CameraPoseEstimation2.algorithms.geometry.essential import EssentialMatrixEstimator
from CameraPoseEstimation2.algorithms.geometry.pose import PnPSolver, PoseEstimator

# Optimization modules
from CameraPoseEstimation2.algorithms.optimization.bundle_adjustment import (
    TwoViewBundleAdjustment,
    IncrementalBundleAdjustment, 
    GlobalBundleAdjustment,
    adjust_two_view,
    adjust_after_new_camera,
    adjust_global
)
from CameraPoseEstimation2.algorithms.optimization.refinement import (
    PoseRefiner,
    ProgressiveIntrinsicsLearner,
    StructureRefiner,
    ProgressiveRefinementPipeline,
    EssentialMatrixRefiner,
    EssentialMatrixRefinerConfig
)

# Selection modules
from CameraPoseEstimation2.algorithms.selection.pair_selection import InitializationPairSelector

# Data and interfaces
from CameraPoseEstimation2.data import create_provider
from CameraPoseEstimation2.core.interfaces import IMatchDataProvider


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Camera:
    """Camera data structure with individual intrinsics"""
    R: np.ndarray
    t: np.ndarray
    K: np.ndarray  # Each camera has its own intrinsic matrix
    image_id: str
    width: int = None
    height: int = None
    
    @property
    def P(self) -> np.ndarray:
        """Projection matrix P = K[R|t]"""
        return self.K @ np.hstack([self.R, self.t])
    
    @property
    def center(self) -> np.ndarray:
        """Camera center in world coordinates"""
        return -self.R.T @ self.t
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'R': self.R,
            't': self.t,
            'K': self.K,
            'image_id': self.image_id,
            'width': self.width,
            'height': self.height
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'Camera':
        """Create from dictionary"""
        return Camera(
            R=data['R'],
            t=data['t'],
            K=data['K'],
            image_id=data['image_id'],
            width=data.get('width'),
            height=data.get('height')
        )


@dataclass
class Point3D:
    """3D point data structure"""
    coords: np.ndarray
    color: np.ndarray = None
    error: float = 0.0
    track_length: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'coords': self.coords,
            'color': self.color,
            'error': self.error,
            'track_length': self.track_length
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'Point3D':
        """Create from dictionary"""
        return Point3D(
            coords=data['coords'],
            color=data.get('color'),
            error=data.get('error', 0.0),
            track_length=data.get('track_length', 0)
        )


@dataclass
class Observation:
    """2D observation of a 3D point"""
    camera_id: str
    point_id: int
    coords_2d: np.ndarray
    feature_id: int = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'camera_id': self.camera_id,
            'point_id': self.point_id,
            'coords_2d': self.coords_2d,
            'feature_id': self.feature_id
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'Observation':
        """Create from dictionary"""
        return Observation(
            camera_id=data['camera_id'],
            point_id=data['point_id'],
            coords_2d=data['coords_2d'],
            feature_id=data.get('feature_id')
        )


class Reconstruction:
    """
    Central reconstruction data structure managing cameras, points, and observations.
    
    This class maintains the complete state of the 3D reconstruction including:
    - Camera poses and intrinsics
    - 3D point cloud
    - 2D-3D correspondences (observations)
    - Indexing structures for efficient queries
    """
    
    def __init__(self):
        self.cameras: Dict[str, Camera] = {}
        self.points: Dict[int, Point3D] = {}
        self.observations: Dict[Tuple[str, int], Observation] = {}
        
        # Indexing structures for efficient queries
        self.camera_to_points: Dict[str, Set[int]] = {}
        self.point_to_cameras: Dict[int, Set[str]] = {}
        
        # Metadata
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'num_cameras': 0,
            'num_points': 0,
            'num_observations': 0
        }
        
        # Next available point ID
        self._next_point_id = 0
    
    def add_camera(self, camera_id: str, R: np.ndarray, t: np.ndarray, 
                   K: np.ndarray, width: int = None, height: int = None) -> None:
        """Add a camera to the reconstruction"""
        camera = Camera(R=R, t=t, K=K, image_id=camera_id, width=width, height=height)
        self.cameras[camera_id] = camera
        self.camera_to_points[camera_id] = set()
        self.metadata['num_cameras'] = len(self.cameras)
        print(f"✓ Added camera: {camera_id}")
    
    def add_point(self, coords: np.ndarray, color: np.ndarray = None) -> int:
        """Add a 3D point and return its ID"""
        point_id = self._next_point_id
        self._next_point_id += 1
        
        point = Point3D(coords=coords, color=color)
        self.points[point_id] = point
        self.point_to_cameras[point_id] = set()
        self.metadata['num_points'] = len(self.points)
        
        return point_id
    
    def add_observation(self, camera_id: str, point_id: int, 
                       coords_2d: np.ndarray, feature_id: int = None) -> None:
        """Add a 2D observation of a 3D point"""
        obs = Observation(camera_id=camera_id, point_id=point_id, 
                         coords_2d=coords_2d, feature_id=feature_id)
        
        self.observations[(camera_id, point_id)] = obs
        self.camera_to_points[camera_id].add(point_id)
        self.point_to_cameras[point_id].add(camera_id)
        
        # Update track length
        self.points[point_id].track_length = len(self.point_to_cameras[point_id])
        self.metadata['num_observations'] = len(self.observations)
    
    def get_camera_observations(self, camera_id: str) -> List[Observation]:
        """Get all observations for a camera"""
        point_ids = self.camera_to_points.get(camera_id, set())
        return [self.observations[(camera_id, pid)] for pid in point_ids 
                if (camera_id, pid) in self.observations]
    
    def get_point_observations(self, point_id: int) -> List[Observation]:
        """Get all observations of a point"""
        camera_ids = self.point_to_cameras.get(point_id, set())
        return [self.observations[(cid, point_id)] for cid in camera_ids 
                if (cid, point_id) in self.observations]
    
    def get_3d_2d_correspondences(self, camera_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 3D-2D correspondences for a camera (for PnP).
        
        Returns:
            points_3d: (N, 3) array of 3D points
            points_2d: (N, 2) array of 2D observations
        """
        observations = self.get_camera_observations(camera_id)
        
        points_3d = []
        points_2d = []
        
        for obs in observations:
            if obs.point_id in self.points:
                points_3d.append(self.points[obs.point_id].coords)
                points_2d.append(obs.coords_2d)
        
        return np.array(points_3d), np.array(points_2d)
    
    def to_dict(self) -> Dict:
        """Convert reconstruction to dictionary for serialization"""
        return {
            'cameras': {cid: cam.to_dict() for cid, cam in self.cameras.items()},
            'points': {pid: pt.to_dict() for pid, pt in self.points.items()},
            'observations': {f"{cid}_{pid}": obs.to_dict() 
                           for (cid, pid), obs in self.observations.items()},
            'metadata': self.metadata
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'Reconstruction':
        """Create reconstruction from dictionary"""
        reconstruction = Reconstruction()
        
        # Load cameras
        for cid, cam_data in data['cameras'].items():
            cam = Camera.from_dict(cam_data)
            reconstruction.cameras[cid] = cam
            reconstruction.camera_to_points[cid] = set()
        
        # Load points
        for pid_str, pt_data in data['points'].items():
            pid = int(pid_str)
            pt = Point3D.from_dict(pt_data)
            reconstruction.points[pid] = pt
            reconstruction.point_to_cameras[pid] = set()
            reconstruction._next_point_id = max(reconstruction._next_point_id, pid + 1)
        
        # Load observations
        for key, obs_data in data['observations'].items():
            obs = Observation.from_dict(obs_data)
            reconstruction.observations[(obs.camera_id, obs.point_id)] = obs
            reconstruction.camera_to_points[obs.camera_id].add(obs.point_id)
            reconstruction.point_to_cameras[obs.point_id].add(obs.camera_id)
        
        reconstruction.metadata = data.get('metadata', {})
        
        return reconstruction
    
    def save(self, filepath: str) -> None:
        """Save reconstruction to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.to_dict(), f)
        print(f"✓ Saved reconstruction to: {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'Reconstruction':
        """Load reconstruction from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Loaded reconstruction from: {filepath}")
        return Reconstruction.from_dict(data)
    
    def summary(self) -> str:
        """Get reconstruction summary"""
        lines = [
            "=" * 70,
            "RECONSTRUCTION SUMMARY",
            "=" * 70,
            f"Cameras: {len(self.cameras)}",
            f"Points: {len(self.points)}",
            f"Observations: {len(self.observations)}",
            ""
        ]
        
        if self.cameras:
            lines.append("Camera List:")
            for cid, cam in self.cameras.items():
                num_obs = len(self.camera_to_points.get(cid, set()))
                lines.append(f"  - {cid}: {num_obs} observations")
        
        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
# MAIN INCREMENTAL RECONSTRUCTION PIPELINE
# ============================================================================

class IncrementalReconstructionPipeline:
    """
    Complete incremental 3D reconstruction pipeline.
    
    Pipeline stages:
    1. Pair Selection - Select best initial image pair
    2. Essential Matrix - Estimate relative pose
    3. Triangulation - Recover 3D points
    4. Two-View BA - Optimize initial reconstruction
    5. For each new camera:
        a. PnP - Estimate camera pose
        b. Triangulation - Add new 3D points
        c. Incremental BA - Optimize recent cameras
    6. Global BA - Final optimization
    7. Export - Save results
    """
    
    def __init__(self, 
                 provider: IMatchDataProvider,
                 output_dir: str = "./output"):
        """
        Initialize pipeline.
        
        Args:
            provider: Data provider for matches and images
            output_dir: Directory for output files
        """
        self.provider = provider
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize reconstruction
        self.reconstruction = Reconstruction()
        
        # Initialize modules
        self.pair_selector = InitializationPairSelector()
        self.essential_estimator = EssentialMatrixEstimator()
        self.triangulation_engine = TriangulationEngine()
        self.pnp_solver = PnPSolver()
        self.essential_refiner = EssentialMatrixRefiner()
        
        # Optimization modules
        self.two_view_ba = TwoViewBundleAdjustment()
        self.incremental_ba = IncrementalBundleAdjustment()
        self.global_ba = GlobalBundleAdjustment()
        
        # Refinement modules
        self.pose_refiner = PoseRefiner()
        self.intrinsics_learner = ProgressiveIntrinsicsLearner()
        self.structure_refiner = StructureRefiner()
        
        print("✓ Pipeline initialized")
        print(f"  Output directory: {output_dir}")
    
    def run(self) -> Reconstruction:
        """
        Run complete reconstruction pipeline.
        
        Returns:
            Final reconstruction
        """
        print("\n" + "=" * 70)
        print("STARTING INCREMENTAL RECONSTRUCTION PIPELINE")
        print("=" * 70)
        
        # Stage 1: Initialize with two views
        print("\n[STAGE 1] TWO-VIEW INITIALIZATION")
        self._initialize_two_view()
        
        # Stage 2: Add remaining cameras incrementally
        print("\n[STAGE 2] INCREMENTAL CAMERA ADDITION")
        self._add_cameras_incrementally()
        
        # Stage 3: Global refinement
        print("\n[STAGE 3] GLOBAL REFINEMENT")
        self._global_refinement()
        
        # Stage 4: Export results
        print("\n[STAGE 4] EXPORT")
        self._export_results()
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE!")
        print("=" * 70)
        print(self.reconstruction.summary())
        
        return self.reconstruction
    
    def _initialize_two_view(self) -> None:
        """Initialize reconstruction with two views using iterative refinement"""
        print("\n--- Selecting Initial Image Pair ---")
        
        # 1. Select best pair
        img1, img2, pts1, pts2, img_size1, img_size2 = self._select_and_prepare_initial_pair()
        
        # 2. Initial estimation
        K1_init, K2_init = self._estimate_initial_intrinsics(pts1, pts2, img_size1, img_size2)
        
        # 3. Refine with iterative optimization
        K1_final, K2_final, R_final, t_final, points_3d, valid_pts1, valid_pts2 = \
            self._refine_essential_matrix(pts1, pts2, K1_init, K2_init, img_size1, img_size2)
        
        # 4. Add to reconstruction
        self._add_initial_cameras(img1, img2, K1_final, K2_final, R_final, t_final, 
                                  img_size1, img_size2)
        self._add_initial_points(points_3d, valid_pts1, valid_pts2, img1, img2)
        
        # 5. Optional: two-view BA for further refinement
        self._run_two_view_bundle_adjustment()
        
        print(f"\n✓ Two-view initialization complete!")
        print(f"  Cameras: {len(self.reconstruction.cameras)}")
        print(f"  Points: {len(self.reconstruction.points)}")
        print(f"  Observations: {len(self.reconstruction.observations)}")
    
    def _select_and_prepare_initial_pair(self) -> Tuple:
        """Select and prepare initial image pair"""
        # Get matches data from provider
        matches_data = self._get_matches_data_from_provider()
        
        # Select best pair
        pair_result = self.pair_selector.find_best_pair(matches_data)
        
        if not pair_result or 'best_pair' not in pair_result:
            raise RuntimeError("Failed to select initial pair: No valid pairs found")
        
        img1, img2 = pair_result['best_pair']
        print(f"✓ Selected pair: {img1} <-> {img2}")
        
        # Get score information
        score_result = pair_result['best_score_result']
        print(f"  Score: {score_result.get('total_score', 0):.3f}")
        print(f"  Matches: {score_result.get('num_matches', 0)}")
        
        # Get match data
        match_data = self.provider.get_match_data((img1, img2))
        pts1, pts2 = match_data['pts1'], match_data['pts2']
        
        # Get image sizes
        img_info1 = self.provider.get_image_info(img1)
        img_info2 = self.provider.get_image_info(img2)
        img_size1 = (img_info1['width'], img_info1['height'])
        img_size2 = (img_info2['width'], img_info2['height'])
        
        print(f"  Points: {len(pts1)}")
        
        return img1, img2, pts1, pts2, img_size1, img_size2
    
    def _get_matches_data_from_provider(self) -> Dict:
        """Convert provider data to format expected by pair selector"""
        matches_data = {}
        
        # Get all images
        all_images = self.provider.get_all_images()
        
        # Create pairs and get match data
        for i, img1 in enumerate(all_images):
            for j, img2 in enumerate(all_images[i+1:], i+1):
                try:
                    match_data = self.provider.get_match_data((img1, img2))
                    if match_data and 'pts1' in match_data and 'pts2' in match_data:
                        # Convert to format expected by pair selector
                        pts1 = match_data['pts1']
                        pts2 = match_data['pts2']
                        
                        # Create correspondences in format [x1, y1, x2, y2]
                        correspondences = np.column_stack([pts1, pts2])
                        
                        # Get image info for sizes
                        img_info1 = self.provider.get_image_info(img1)
                        img_info2 = self.provider.get_image_info(img2)
                        
                        matches_data[(img1, img2)] = {
                            'correspondences': correspondences,
                            'image1_size': (img_info1['width'], img_info1['height']),
                            'image2_size': (img_info2['width'], img_info2['height']),
                            'score_type': 'distance',  # Use string to avoid import conflicts
                            'method': 'unknown'  # Default method
                        }
                except Exception as e:
                    # Skip pairs that don't have match data
                    continue
        
        return matches_data
    
    def _estimate_initial_intrinsics(self,
                                     pts1: np.ndarray,
                                     pts2: np.ndarray,
                                     img_size1: Tuple[int, int],
                                     img_size2: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate initial camera intrinsics"""
        print(f"\n--- Initial Essential Matrix Estimation ---")
        
        essential_result = self.essential_estimator.estimate(
            pts1, pts2, 
            image_size1=img_size1,
            image_size2=img_size2
        )
        
        if not essential_result.get('success', False):
            print(f"⚠ Initial estimation failed, using default intrinsics")
            K1 = self._estimate_default_K(img_size1)
            K2 = self._estimate_default_K(img_size2)
        else:
            K1 = essential_result['camera_matrices'][0]
            K2 = essential_result['camera_matrices'][1]
            print(f"✓ Initial intrinsics estimated")
        
        return K1, K2
    
    def _refine_essential_matrix(self,
                                 pts1: np.ndarray,
                                 pts2: np.ndarray,
                                 K1_init: np.ndarray,
                                 K2_init: np.ndarray,
                                 img_size1: Tuple[int, int],
                                 img_size2: Tuple[int, int]) -> Tuple:
        """Refine essential matrix with iterative optimization"""
        print(f"\n--- Iterative Refinement ---")
        
        refinement_result = self.essential_refiner.refine(
            pts1, pts2,
            K1_init, K2_init,
            img_size1, img_size2
        )
        
        if refinement_result['success']:
            print(f"✓ Refinement successful")
            return (
                refinement_result['K1'],
                refinement_result['K2'],
                refinement_result['R'],
                refinement_result['t'],
                refinement_result['points_3d'],
                refinement_result['valid_pts1'],
                refinement_result['valid_pts2']
            )
        else:
            print(f"⚠ Refinement failed, using fallback")
            return self._fallback_estimation(pts1, pts2, K1_init, K2_init)
    
    def _fallback_estimation(self,
                            pts1: np.ndarray,
                            pts2: np.ndarray,
                            K1: np.ndarray,
                            K2: np.ndarray) -> Tuple:
        """Fallback estimation if refinement fails"""
        print(f"--- Fallback: Direct Estimation ---")
        
        # Estimate essential matrix
        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K1, None).reshape(-1, 2)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K2, None).reshape(-1, 2)
        
        E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, focal=1.0, pp=(0.0, 0.0),
                                       method=cv2.RANSAC, prob=0.999, threshold=0.001)
        
        # Recover pose
        _, R, t, pose_mask = cv2.recoverPose(E, pts1_norm, pts2_norm, focal=1.0, pp=(0.0, 0.0))
        
        # Get valid points
        inliers = (mask.ravel() > 0) & (pose_mask.ravel() > 0)
        valid_pts1 = pts1[inliers]
        valid_pts2 = pts2[inliers]
        
        # Triangulate
        result = self.triangulation_engine.triangulate_initial_points(
            valid_pts1, valid_pts2,
            R1=np.eye(3), t1=np.zeros((3, 1)),
            R2=R, t2=t,
            K=[K1, K2],
            image_pair=('cam1', 'cam2')
        )
        
        points_3d = result['points_3d'] if isinstance(result, dict) else result
        
        print(f"✓ Fallback: {points_3d.shape[1]} points triangulated")
        
        return K1, K2, R, t, points_3d, valid_pts1, valid_pts2
    
    def _add_initial_cameras(self,
                            img1: str, img2: str,
                            K1: np.ndarray, K2: np.ndarray,
                            R: np.ndarray, t: np.ndarray,
                            img_size1: Tuple[int, int],
                            img_size2: Tuple[int, int]) -> None:
        """Add initial two cameras to reconstruction"""
        print(f"\n--- Adding Cameras ---")
        
        self.reconstruction.add_camera(
            camera_id=img1,
            R=np.eye(3),
            t=np.zeros((3, 1)),
            K=K1,
            width=img_size1[0],
            height=img_size1[1]
        )
        
        self.reconstruction.add_camera(
            camera_id=img2,
            R=R,
            t=t,
            K=K2,
            width=img_size2[0],
            height=img_size2[1]
        )
    
    def _add_initial_points(self,
                           points_3d: np.ndarray,
                           pts1: np.ndarray,
                           pts2: np.ndarray,
                           img1: str,
                           img2: str) -> None:
        """Add initial 3D points and observations"""
        print(f"--- Adding Points and Observations ---")
        
        for i in range(points_3d.shape[1]):
            point_id = self.reconstruction.add_point(points_3d[:, i])
            
            if i < len(pts1):
                self.reconstruction.add_observation(img1, point_id, pts1[i])
                self.reconstruction.add_observation(img2, point_id, pts2[i])
        
        print(f"✓ Added {points_3d.shape[1]} points")
    
    def _run_two_view_bundle_adjustment(self) -> None:
        """Run two-view bundle adjustment (optional final refinement)"""
        print(f"\n--- Two-View Bundle Adjustment ---")
        
        state = self._reconstruction_to_state()
        ba_result = adjust_two_view(
            cameras=state['cameras'],
            points_3d=state['points_3d'],
            observations=state['observations'],
            optimize_intrinsics=True,
            fix_first_camera=True
        )
        
        if ba_result['success']:
            self._update_reconstruction_from_state(ba_result)
            print(f"✓ Bundle adjustment completed")
            print(f"  Final error: {ba_result.get('final_error', 'N/A'):.2f}px")
        else:
            print(f"⚠ Bundle adjustment failed, using refined estimates")
    
    def _estimate_default_K(self, img_size: Tuple[int, int]) -> np.ndarray:
        """Estimate default camera intrinsics from image size"""
        width, height = img_size
        focal = max(width, height)
        
        K = np.array([
            [focal, 0, width / 2],
            [0, focal, height / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        return K
    
    def _add_cameras_incrementally(self) -> None:
        """Add remaining cameras incrementally"""
        print("\n--- Getting Candidate Cameras ---")
        
        # Get all images from provider
        all_images = set(self.provider.get_all_images())
        registered_images = set(self.reconstruction.cameras.keys())
        unregistered_images = all_images - registered_images
        
        print(f"Total images: {len(all_images)}")
        print(f"Registered: {len(registered_images)}")
        print(f"Remaining: {len(unregistered_images)}")
        
        if not unregistered_images:
            print("✓ All cameras already registered!")
            return
        
        # Add cameras one by one
        iteration = 0
        while unregistered_images:
            iteration += 1
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}: Adding Camera {len(registered_images) + 1}/{len(all_images)}")
            print(f"{'='*70}")
            
            # 1. Select next best camera
            next_camera = self._select_next_camera(unregistered_images)
            
            if next_camera is None:
                print(f"⚠ Cannot add more cameras (no valid candidates)")
                break
            
            print(f"\n--- Selected Camera: {next_camera} ---")
            
            # 2. Estimate pose using PnP
            success = self._add_camera_pnp(next_camera)
            
            if not success:
                print(f"✗ Failed to add camera {next_camera}")
                unregistered_images.remove(next_camera)
                continue
            
            # 3. Triangulate new points
            print(f"\n--- Triangulating New Points ---")
            num_new_points = self._triangulate_new_points(next_camera)
            print(f"✓ Added {num_new_points} new 3D points")
            
            # 4. Incremental bundle adjustment
            print(f"\n--- Incremental Bundle Adjustment ---")
            ba_result = adjust_after_new_camera(
                reconstruction_state=self._reconstruction_to_state(),
                new_camera_id=next_camera,
                num_recent_cameras=3  # Optimize last 3 cameras
            )
            
            if ba_result['success']:
                self._update_reconstruction_from_state(ba_result)
                print(f"✓ Incremental BA completed")
                print(f"  Final error: {ba_result.get('final_error', 'N/A'):.2f}px")
            else:
                print(f"⚠ Incremental BA failed")
            
            # Update sets
            registered_images.add(next_camera)
            unregistered_images.remove(next_camera)
            
            print(f"\n✓ Camera {next_camera} added successfully!")
            print(f"  Total cameras: {len(self.reconstruction.cameras)}")
            print(f"  Total points: {len(self.reconstruction.points)}")
            print(f"  Total observations: {len(self.reconstruction.observations)}")
            
            # Save checkpoint every 5 cameras
            if iteration % 5 == 0:
                checkpoint_file = os.path.join(self.output_dir, 
                                              f"checkpoint_{iteration}.pkl")
                self.reconstruction.save(checkpoint_file)
                print(f"  💾 Checkpoint saved: {checkpoint_file}")
        
        print(f"\n✓ Incremental reconstruction complete!")
        print(f"  Final: {len(self.reconstruction.cameras)} cameras, "
              f"{len(self.reconstruction.points)} points")
    
    def _select_next_camera(self, candidates: Set[str]) -> Optional[str]:
        """Select next best camera to add based on number of 3D-2D correspondences"""
        best_camera = None
        best_score = 0
        
        for camera_id in candidates:
            # Count potential 3D-2D correspondences
            num_correspondences = 0
            
            for registered_cam in self.reconstruction.cameras.keys():
                # Check if we have matches between this candidate and registered camera
                try:
                    match_data = self.provider.get_match_data((camera_id, registered_cam))
                    if match_data and 'pts1' in match_data:
                        num_correspondences += len(match_data['pts1'])
                except:
                    pass
            
            if num_correspondences > best_score:
                best_score = num_correspondences
                best_camera = camera_id
        
        if best_score >= 20:  # Minimum correspondences needed
            print(f"  Selected: {best_camera} ({best_score} potential correspondences)")
            return best_camera
        
        return None
    
    def _add_camera_pnp(self, camera_id: str) -> bool:
        """Add camera using PnP pose estimation"""
        print(f"--- Estimating Pose with PnP ---")
        
        # Collect 3D-2D correspondences from all registered cameras
        all_points_3d = []
        all_points_2d = []
        
        for registered_cam in self.reconstruction.cameras.keys():
            try:
                # Get matches between new camera and registered camera
                match_data = self.provider.get_match_data((camera_id, registered_cam))
                
                if not match_data or 'pts1' not in match_data:
                    continue
                
                pts_new = match_data['pts1']  # Points in new camera
                pts_reg = match_data['pts2']  # Points in registered camera
                
                # Find which pts_reg correspond to existing 3D points
                registered_obs = self.reconstruction.get_camera_observations(registered_cam)
                
                for obs in registered_obs:
                    point_3d = self.reconstruction.points[obs.point_id].coords
                    point_2d_reg = obs.coords_2d
                    
                    # Find matching point in pts_reg
                    distances = np.linalg.norm(pts_reg - point_2d_reg, axis=1)
                    if len(distances) > 0 and np.min(distances) < 2.0:  # 2 pixel threshold
                        match_idx = np.argmin(distances)
                        all_points_3d.append(point_3d)
                        all_points_2d.append(pts_new[match_idx])
            
            except Exception as e:
                continue
        
        if len(all_points_3d) < 10:
            print(f"✗ Insufficient correspondences: {len(all_points_3d)}")
            return False
        
        points_3d = np.array(all_points_3d)
        points_2d = np.array(all_points_2d)
        
        print(f"  Found {len(points_3d)} 3D-2D correspondences")
        
        # Get image info for intrinsics estimation
        img_info = self.provider.get_image_info(camera_id)
        img_size = (img_info['width'], img_info['height'])
        
        # Initial K estimate
        K_init = np.array([
            [img_size[0], 0, img_size[0]/2],
            [0, img_size[0], img_size[1]/2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Solve PnP with RANSAC
        pnp_result = self.pnp_solver.solve_pnp(
            points_3d=points_3d,
            points_2d=points_2d,
            camera_matrix=K_init,
            use_ransac=True
        )
        
        if not pnp_result['success']:
            print(f"✗ PnP failed: {pnp_result.get('message', 'Unknown error')}")
            return False
        
        R = pnp_result['R']
        t = pnp_result['t']
        inliers = pnp_result.get('inliers', np.ones(len(points_3d), dtype=bool))
        
        print(f"✓ PnP solved successfully")
        print(f"  Inliers: {np.sum(inliers)}/{len(points_3d)}")
        print(f"  Reprojection error: {pnp_result.get('reprojection_error', 'N/A'):.2f}px")
        
        # Add camera to reconstruction
        self.reconstruction.add_camera(
            camera_id=camera_id,
            R=R,
            t=t,
            K=K_init,
            width=img_size[0],
            height=img_size[1]
        )
        
        # Add observations for inlier correspondences
        for i, is_inlier in enumerate(inliers):
            if is_inlier:
                # Find the point_id for this 3D point
                point_3d = points_3d[i]
                for point_id, point in self.reconstruction.points.items():
                    if np.allclose(point.coords, point_3d, atol=1e-6):
                        self.reconstruction.add_observation(
                            camera_id, point_id, points_2d[i]
                        )
                        break
        
        return True
    
    def _triangulate_new_points(self, new_camera_id: str) -> int:
        """Triangulate new points visible in the new camera"""
        new_camera = self.reconstruction.cameras[new_camera_id]
        num_new_points = 0
        
        # Try to triangulate with each existing camera
        for existing_cam_id in self.reconstruction.cameras.keys():
            if existing_cam_id == new_camera_id:
                continue
            
            try:
                # Get matches
                match_data = self.provider.get_match_data((new_camera_id, existing_cam_id))
                
                if not match_data or 'pts1' not in match_data:
                    continue
                
                pts_new = match_data['pts1']
                pts_existing = match_data['pts2']
                
                # Filter out points that are already in reconstruction
                existing_obs = self.reconstruction.get_camera_observations(existing_cam_id)
                existing_2d_points = np.array([obs.coords_2d for obs in existing_obs])
                
                new_pts_mask = np.ones(len(pts_new), dtype=bool)
                for i, pt in enumerate(pts_existing):
                    if len(existing_2d_points) > 0:
                        distances = np.linalg.norm(existing_2d_points - pt, axis=1)
                        if np.min(distances) < 2.0:  # Already observed
                            new_pts_mask[i] = False
                
                pts_new_filtered = pts_new[new_pts_mask]
                pts_existing_filtered = pts_existing[new_pts_mask]
                
                if len(pts_new_filtered) < 10:
                    continue
                
                # Triangulate
                existing_camera = self.reconstruction.cameras[existing_cam_id]
                
                result = self.triangulation_engine.triangulate_initial_points(
                    pts_existing_filtered, pts_new_filtered,
                    R1=existing_camera.R, t1=existing_camera.t,
                    R2=new_camera.R, t2=new_camera.t,
                    K=[existing_camera.K, new_camera.K],
                    image_pair=(existing_cam_id, new_camera_id)
                )
                
                if isinstance(result, dict):
                    points_3d = result['points_3d']
                else:
                    points_3d = result
                
                # Add new points
                for i in range(points_3d.shape[1]):
                    point_id = self.reconstruction.add_point(points_3d[:, i])
                    
                    if i < len(pts_existing_filtered):
                        self.reconstruction.add_observation(
                            existing_cam_id, point_id, pts_existing_filtered[i]
                        )
                        self.reconstruction.add_observation(
                            new_camera_id, point_id, pts_new_filtered[i]
                        )
                    
                    num_new_points += 1
                
            except Exception as e:
                continue
        
        return num_new_points
    
    def _global_refinement(self) -> None:
        """Perform global bundle adjustment"""
        print("\n--- Structure Refinement ---")
        
        # Refine structure (remove outliers, merge duplicates)
        structure_refiner = StructureRefiner()
        
        state = self._reconstruction_to_state()
        
        # Adapt observations list -> per-camera dict expected by StructureRefiner
        observations_per_camera: Dict[str, List[Dict]] = {}
        for obs in state['observations']:
            cam_id = obs['camera_id']
            observations_per_camera.setdefault(cam_id, []).append(obs)
        
        refinement_result = structure_refiner.refine_structure(
            points_3d=state['points_3d'],
            cameras=state['cameras'],
            observations=observations_per_camera,
            remove_outliers=True,
            merge_points=True
        )
        
        if refinement_result.success:
            print(f"✓ Structure refined")
            print(f"  Points before: {refinement_result.metadata.get('initial_points', 'N/A')}")
            print(f"  Points after: {refinement_result.metadata.get('final_points', 'N/A')}")
            print(f"  Removed: {refinement_result.metadata.get('points_removed', 0)}")
            
            # Update reconstruction with refined structure
            refined = refinement_result.optimized_params
            self._update_reconstruction_from_state({
                'points_3d': refined['points_3d'],
                'observations': refined.get('observations', state['observations'])
            })
        else:
            print(f"⚠ Structure refinement failed")
        
        print(f"\n--- Global Bundle Adjustment ---")
        
        # Global BA (ensure points_3d nested structure expected by adjust_global)
        state_for_ba = self._reconstruction_to_state()
        # Ensure points_3d nested dict structure
        if isinstance(state_for_ba.get('points_3d'), np.ndarray):
            state_for_ba = dict(state_for_ba)
            state_for_ba['points_3d'] = {'points_3d': state_for_ba['points_3d']}
        # Ensure observations are per-camera dict as expected by adjust_global
        if isinstance(state_for_ba.get('observations'), list):
            obs_per_cam: Dict[str, List[Dict]] = {}
            for obs in state_for_ba['observations']:
                cam_id = obs['camera_id']
                obs_per_cam.setdefault(cam_id, []).append(obs)
            state_for_ba['observations'] = obs_per_cam
        ba_result = adjust_global(
            reconstruction_state=state_for_ba,
            optimize_intrinsics=True,
            fix_first_camera=True
        )
        
        # adjust_global returns an updated reconstruction_state (no 'success' key)
        if isinstance(ba_result, dict) and 'cameras' in ba_result and 'points_3d' in ba_result:
            self._update_reconstruction_from_state({
                'cameras': ba_result.get('cameras', {}),
                'points_3d': ba_result.get('points_3d', {}).get('points_3d', state_for_ba['points_3d']['points_3d']),
                'metadata': {'optimization_history': ba_result.get('optimization_history', [])}
            })
            print(f"✓ Global bundle adjustment completed")
            hist = ba_result.get('optimization_history', [])
            if hist:
                last = hist[-1]
                print(f"  Initial error: {last.get('initial_error', 'N/A')}")
                print(f"  Final error: {last.get('final_error', 'N/A')}")
                print(f"  Iterations: {last.get('iterations', 'N/A')}")
        else:
            print(f"⚠ Global bundle adjustment did not return expected state")
        
        # Final quality assessment
        print(f"\n--- Final Quality Assessment ---")
        self._assess_reconstruction_quality()
        
        print(f"\n✓ Global refinement complete!")
    
    def _assess_reconstruction_quality(self) -> None:
        """Assess final reconstruction quality"""
        num_cameras = len(self.reconstruction.cameras)
        num_points = len(self.reconstruction.points)
        num_observations = len(self.reconstruction.observations)
        
        # Calculate average track length
        track_lengths = [p.track_length for p in self.reconstruction.points.values()]
        avg_track_length = np.mean(track_lengths) if track_lengths else 0
        
        # Calculate average observations per camera
        avg_obs_per_camera = num_observations / num_cameras if num_cameras > 0 else 0
        
        # Calculate reprojection errors
        total_error = 0
        num_errors = 0
        
        for (cam_id, point_id), obs in self.reconstruction.observations.items():
            camera = self.reconstruction.cameras[cam_id]
            point_3d = self.reconstruction.points[point_id].coords
            
            # Project point
            point_3d_cam = camera.R @ point_3d + camera.t.ravel()
            
            if point_3d_cam[2] > 0:  # Point in front of camera
                point_2d_proj = camera.K @ point_3d_cam
                point_2d_proj = point_2d_proj[:2] / point_2d_proj[2]
                
                error = np.linalg.norm(point_2d_proj - obs.coords_2d)
                total_error += error
                num_errors += 1
        
        avg_reproj_error = total_error / num_errors if num_errors > 0 else 0
        
        print(f"  Cameras: {num_cameras}")
        print(f"  Points: {num_points}")
        print(f"  Observations: {num_observations}")
        print(f"  Avg track length: {avg_track_length:.1f}")
        print(f"  Avg obs/camera: {avg_obs_per_camera:.1f}")
        print(f"  Avg reprojection error: {avg_reproj_error:.2f}px")
        
        # Store metrics in reconstruction metadata
        self.reconstruction.metadata.update({
            'final_quality': {
                'avg_track_length': float(avg_track_length),
                'avg_obs_per_camera': float(avg_obs_per_camera),
                'avg_reprojection_error': float(avg_reproj_error),
                'num_cameras': num_cameras,
                'num_points': num_points,
                'num_observations': num_observations
            }
        })
    
    def _export_results(self) -> None:
        """Export reconstruction results"""
        print("\n--- Exporting Results ---")
        
        # 1. Save reconstruction (pickle format)
        reconstruction_file = os.path.join(self.output_dir, "reconstruction.pkl")
        self.reconstruction.save(reconstruction_file)
        print(f"✓ Saved reconstruction: {reconstruction_file}")
        
        # 2. Export to JSON
        json_file = os.path.join(self.output_dir, "reconstruction.json")
        self._export_json(json_file)
        print(f"✓ Saved JSON: {json_file}")
        
        # 3. Export to COLMAP format
        colmap_dir = os.path.join(self.output_dir, "colmap")
        os.makedirs(colmap_dir, exist_ok=True)
        self._export_colmap(colmap_dir)
        print(f"✓ Saved COLMAP: {colmap_dir}")
        
        # 4. Export point cloud (PLY format)
        ply_file = os.path.join(self.output_dir, "point_cloud.ply")
        self._export_ply(ply_file)
        print(f"✓ Saved PLY: {ply_file}")
        
        # 5. Export summary report
        report_file = os.path.join(self.output_dir, "reconstruction_report.txt")
        self._export_report(report_file)
        print(f"✓ Saved report: {report_file}")
        
        print(f"\n✓ All results exported to: {self.output_dir}")
    
    # ========================================================================
    # HELPER METHODS FOR STATE CONVERSION
    # ========================================================================
    
    def _reconstruction_to_state(self) -> Dict:
        """
        Convert Reconstruction object to state dictionary for BA modules.
        
        Returns:
            State dict with format expected by bundle adjustment modules
        """
        # Build cameras dict
        cameras = {}
        for cam_id, camera in self.reconstruction.cameras.items():
            cameras[cam_id] = {
                'R': camera.R,
                't': camera.t,
                'K': camera.K,
                'width': camera.width,
                'height': camera.height
            }
        
        # Build points_3d array
        if self.reconstruction.points:
            point_ids = sorted(self.reconstruction.points.keys())
            points_3d = np.column_stack([
                self.reconstruction.points[pid].coords 
                for pid in point_ids
            ])
        else:
            points_3d = np.empty((3, 0))
        
        # Build observations list
        observations = []
        for (cam_id, point_id), obs in self.reconstruction.observations.items():
            observations.append({
                'camera_id': cam_id,
                'point_id': point_id,
                'image_point': obs.coords_2d
            })
        
        return {
            'cameras': cameras,
            'points_3d': points_3d,
            'observations': observations,
            'metadata': self.reconstruction.metadata
        }
    
    def _update_reconstruction_from_state(self, state: Dict) -> None:
        """
        Update Reconstruction object from state dictionary returned by BA.
        
        Args:
            state: State dict with optimized cameras, points, observations
        """
        # Update cameras
        if 'cameras' in state:
            for cam_id, cam_data in state['cameras'].items():
                if cam_id in self.reconstruction.cameras:
                    camera = self.reconstruction.cameras[cam_id]
                    camera.R = cam_data['R']
                    camera.t = cam_data['t']
                    camera.K = cam_data.get('K', camera.K)
        
        # Update points
        if 'points_3d' in state:
            points_3d = state['points_3d']
            
            if points_3d.shape[1] == len(self.reconstruction.points):
                # Update existing points
                for i, point_id in enumerate(sorted(self.reconstruction.points.keys())):
                    self.reconstruction.points[point_id].coords = points_3d[:, i]
            else:
                # Points were filtered/modified - need to rebuild
                # This is more complex and might need observation remapping
                print(f"  ⚠ Warning: Point count changed "
                      f"({len(self.reconstruction.points)} → {points_3d.shape[1]})")
        
        # Update metadata
        if 'metadata' in state:
            self.reconstruction.metadata.update(state['metadata'])
    
    def _export_json(self, filepath: str) -> None:
        """Export reconstruction to JSON format"""
        data = {
            'metadata': self.reconstruction.metadata,
            'cameras': {},
            'points': {},
            'observations': []
        }
        
        # Cameras
        for cam_id, camera in self.reconstruction.cameras.items():
            data['cameras'][cam_id] = {
                'R': camera.R.tolist(),
                't': camera.t.tolist(),
                'K': camera.K.tolist(),
                'width': camera.width,
                'height': camera.height,
                'center': camera.center.tolist()
            }
        
        # Points
        for point_id, point in self.reconstruction.points.items():
            data['points'][str(point_id)] = {
                'coords': point.coords.tolist(),
                'color': point.color.tolist() if point.color is not None else None,
                'track_length': point.track_length,
                'error': point.error
            }
        
        # Observations
        for (cam_id, point_id), obs in self.reconstruction.observations.items():
            data['observations'].append({
                'camera_id': cam_id,
                'point_id': point_id,
                'coords_2d': obs.coords_2d.tolist()
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _export_colmap(self, output_dir: str) -> None:
        """Export reconstruction to COLMAP format"""
        if not HAS_SCIPY:
            print("  ⚠ Warning: scipy not available, using identity quaternions for rotation")
        
        # cameras.txt
        cameras_file = os.path.join(output_dir, "cameras.txt")
        with open(cameras_file, 'w') as f:
            f.write("# Camera list with one line per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            
            for i, (cam_id, camera) in enumerate(self.reconstruction.cameras.items()):
                # PINHOLE model: fx, fy, cx, cy
                fx = camera.K[0, 0]
                fy = camera.K[1, 1]
                cx = camera.K[0, 2]
                cy = camera.K[1, 2]
                
                f.write(f"{i} PINHOLE {camera.width} {camera.height} "
                       f"{fx} {fy} {cx} {cy}\n")
        
        # images.txt
        images_file = os.path.join(output_dir, "images.txt")
        with open(images_file, 'w') as f:
            f.write("# Image list with two lines per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            
            for i, (cam_id, camera) in enumerate(self.reconstruction.cameras.items()):
                # Convert rotation matrix to quaternion
                if HAS_SCIPY:
                    quat = Rotation.from_matrix(camera.R).as_quat()  # [x, y, z, w]
                    qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
                else:
                    # Fallback: use identity quaternion
                    qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
                    print(f"  Warning: Using identity quaternion for {cam_id} (scipy not available)")
                
                tx, ty, tz = camera.t.ravel()
                
                f.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {cam_id}\n")
                
                # Write 2D points
                obs_list = self.reconstruction.get_camera_observations(cam_id)
                points_line = " ".join([f"{obs.coords_2d[0]} {obs.coords_2d[1]} {obs.point_id}" 
                                       for obs in obs_list])
                f.write(f"{points_line}\n")
        
        # points3D.txt
        points_file = os.path.join(output_dir, "points3D.txt")
        with open(points_file, 'w') as f:
            f.write("# 3D point list with one line per point:\n")
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
                   "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            
            for point_id, point in self.reconstruction.points.items():
                x, y, z = point.coords
                
                # Color (default to white if not available)
                if point.color is not None:
                    r, g, b = point.color
                else:
                    r, g, b = 128, 128, 128
                
                # Get track
                obs_list = self.reconstruction.get_point_observations(point_id)
                track = " ".join([f"{obs.camera_id} 0" for obs in obs_list])
                
                f.write(f"{point_id} {x} {y} {z} {r} {g} {b} {point.error} {track}\n")
    
    def _export_ply(self, filepath: str) -> None:
        """Export point cloud to PLY format"""
        points = []
        colors = []
        
        for point in self.reconstruction.points.values():
            points.append(point.coords)
            
            if point.color is not None:
                colors.append(point.color)
            else:
                colors.append([128, 128, 128])
        
        points = np.array(points)
        colors = np.array(colors, dtype=np.uint8)
        
        with open(filepath, 'w') as f:
            # Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # Data
            for point, color in zip(points, colors):
                f.write(f"{point[0]} {point[1]} {point[2]} "
                       f"{color[0]} {color[1]} {color[2]}\n")
    
    def _export_report(self, filepath: str) -> None:
        """Export summary report"""
        with open(filepath, 'w') as f:
            f.write("="*70 + "\n")
            f.write("3D RECONSTRUCTION REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Metadata
            f.write("METADATA\n")
            f.write("-"*70 + "\n")
            f.write(f"Created: {self.reconstruction.metadata.get('created_at', 'N/A')}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            # Statistics
            f.write("STATISTICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Cameras: {len(self.reconstruction.cameras)}\n")
            f.write(f"Points: {len(self.reconstruction.points)}\n")
            f.write(f"Observations: {len(self.reconstruction.observations)}\n\n")
            
            # Quality metrics
            if 'final_quality' in self.reconstruction.metadata:
                quality = self.reconstruction.metadata['final_quality']
                f.write("QUALITY METRICS\n")
                f.write("-"*70 + "\n")
                f.write(f"Avg track length: {quality.get('avg_track_length', 'N/A'):.2f}\n")
                f.write(f"Avg observations per camera: {quality.get('avg_obs_per_camera', 'N/A'):.1f}\n")
                f.write(f"Avg reprojection error: {quality.get('avg_reprojection_error', 'N/A'):.2f}px\n\n")
            
            # Camera list
            f.write("CAMERAS\n")
            f.write("-"*70 + "\n")
            for cam_id, camera in self.reconstruction.cameras.items():
                num_obs = len(self.reconstruction.camera_to_points.get(cam_id, set()))
                f.write(f"  {cam_id}:\n")
                f.write(f"    Observations: {num_obs}\n")
                f.write(f"    Focal length: {camera.K[0,0]:.1f}px\n")
                f.write(f" Position: [{camera.center[0].item():.2f}, {camera.center[1].item():.2f}, {camera.center[2].item():.2f}]\n\n")
            
            f.write("="*70 + "\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the incremental reconstruction pipeline.
    
    To run:
        python incremental.py
    
    Or integrate into your own script:
        from pipeline.incremental import IncrementalReconstructionPipeline
        from data import create_provider
        
        provider = create_provider("structured", results_dir="./match_results")
        pipeline = IncrementalReconstructionPipeline(provider, output_dir="./reconstruction_output")
        reconstruction = pipeline.run()
    """
    
    import sys
    
    # Configuration
    MATCH_RESULTS_DIR = "./results"  # Directory with match results
    OUTPUT_DIR = "./reconstruction_output"  # Output directory
    
    print("="*70)
    print("INCREMENTAL 3D RECONSTRUCTION PIPELINE")
    print("="*70)
    print(f"Match results: {MATCH_RESULTS_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*70)
    
    try:
        # Create data provider
        print("\n[1/2] Creating data provider...")
        provider = create_provider(
            provider_type="structured",
            results_dir=MATCH_RESULTS_DIR,
            cache_size=100
        )
        
        print(f"✓ Provider initialized")
        print(f"  Available images: {len(list(provider.get_all_images()))}")
        
        # Create and run pipeline
        print("\n[2/2] Running reconstruction pipeline...")
        pipeline = IncrementalReconstructionPipeline(
            provider=provider,
            output_dir=OUTPUT_DIR
        )
        
        reconstruction = pipeline.run()
        
        print("\n" + "="*70)
        print("✓ RECONSTRUCTION COMPLETE!")
        print("="*70)
        print(reconstruction.summary())
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print(f"\nPlease ensure match results exist in: {MATCH_RESULTS_DIR}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)