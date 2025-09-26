import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import pickle
import json
import copy
import cv2

from .essential_estimation import MatrixEstimationConfig,  EssentialMatrixEstimator
from .triangulation import TriangulationEngine
from .bundle_adjusment import BundleAdjustmentConfig, IncrementalBundleAdjuster, GlobalBundleAdjuster
from .pair_selector import InitializationPairSelector
from .pose_recovery import PoseRecovery, PnPSolver
from .intrinsics_estimator import ProgressiveLearningIntrinsicsEstimator
from .quality_assessment import assess_reconstruction_quality
from .iterative_refinement_for_camera import IterativeRefinementPipeline


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


@dataclass
class Point3D:
    """3D point data structure"""
    coords: np.ndarray
    color: np.ndarray = None
    error: float = 0.0
    track_length: int = 0
    

@dataclass
class Observation:
    """2D observation of a 3D point"""
    camera_id: str
    point_id: int
    coords_2d: np.ndarray
    feature_id: int = None


class Reconstruction:
    """
    Central reconstruction data structure managing cameras, points, and observations
    """
    
    def __init__(self):
        self.cameras: Dict[str, Camera] = {}
        self.points: Dict[int, Point3D] = {}
        self.observations: Dict[Tuple[str, int], Observation] = {}
        
        # Indexing structures for efficient queries
        self.camera_to_points: Dict[str, Set[int]] = {}
        self.point_to_cameras: Dict[int, Set[str]] = {}
        
        # Metadata
        self.initialization_info = {}
        self.next_point_id = 0
        
        
    def add_camera(self, camera_id: str, R: np.ndarray, t: np.ndarray, 
                   K: np.ndarray, width: int = None, height: int = None) -> Camera:
        """Add or update a camera - K is now required"""
        if K is None:
            raise ValueError(f"Camera matrix K is required for camera {camera_id}")
        
        camera = Camera(R=R, t=t, K=K, image_id=camera_id, width=width, height=height)
        self.cameras[camera_id] = camera
        
        if camera_id not in self.camera_to_points:
            self.camera_to_points[camera_id] = set()
            
        return camera

    def get_camera_matrix(self, camera_id: str) -> np.ndarray:
        """Get the intrinsic matrix for a specific camera"""
        if camera_id not in self.cameras:
            raise ValueError(f"Camera {camera_id} not found")
        return self.cameras[camera_id].K

    def get_all_camera_matrices(self) -> Dict[str, np.ndarray]:
        """Get all camera intrinsic matrices"""
        return {cam_id: cam.K for cam_id, cam in self.cameras.items()}

    def add_point(self, coords: np.ndarray, color: np.ndarray = None) -> int:
        """Add a new 3D point and return its ID"""
        point_id = self.next_point_id
        self.points[point_id] = Point3D(coords=coords, color=color)
        self.point_to_cameras[point_id] = set()
        self.next_point_id += 1
        return point_id
    
    def add_observation(self, camera_id: str, point_id: int, 
                       coords_2d: np.ndarray, feature_id: int = None):
        """Add an observation linking a camera to a 3D point"""
        if camera_id not in self.cameras:
            raise ValueError(f"Camera {camera_id} not found")
        if point_id not in self.points:
            raise ValueError(f"Point {point_id} not found")
            
        obs = Observation(
            camera_id=camera_id,
            point_id=point_id,
            coords_2d=coords_2d,
            feature_id=feature_id
        )
        
        self.observations[(camera_id, point_id)] = obs
        self.camera_to_points[camera_id].add(point_id)
        self.point_to_cameras[point_id].add(camera_id)
        
        # Update track length
        self.points[point_id].track_length = len(self.point_to_cameras[point_id])
        
    def get_camera_observations(self, camera_id: str) -> List[Observation]:
        """Get all observations for a camera"""
        return [self.observations[(camera_id, pid)] 
                for pid in self.camera_to_points.get(camera_id, set())]
    
    def get_point_observations(self, point_id: int) -> List[Observation]:
        """Get all observations of a point"""
        return [self.observations[(cid, point_id)] 
                for cid in self.point_to_cameras.get(point_id, set())]
    
    def remove_point(self, point_id: int):
        """Remove a point and all its observations"""
        if point_id not in self.points:
            return
            
        # Remove observations
        for camera_id in self.point_to_cameras[point_id]:
            self.observations.pop((camera_id, point_id), None)
            self.camera_to_points[camera_id].discard(point_id)
            
        # Remove point
        del self.points[point_id]
        del self.point_to_cameras[point_id]
        
    def get_points_array(self) -> np.ndarray:
        """Get all 3D points as numpy array"""
        if not self.points:
            return np.empty((3, 0))
        return np.column_stack([p.coords for p in self.points.values()])
    
    def get_statistics(self) -> Dict:
        """Get reconstruction statistics"""
        track_lengths = [p.track_length for p in self.points.values()]
        errors = [p.error for p in self.points.values() if p.error > 0]
        
        return {
            'num_cameras': len(self.cameras),
            'num_points': len(self.points),
            'num_observations': len(self.observations),
            'mean_track_length': np.mean(track_lengths) if track_lengths else 0,
            'max_track_length': max(track_lengths) if track_lengths else 0,
            'mean_reprojection_error': np.mean(errors) if errors else 0,
            'median_reprojection_error': np.median(errors) if errors else 0
        }
    
    def to_legacy_format(self) -> Dict:
        """Convert Reconstruction to legacy dictionary format for compatibility"""
        # Convert cameras to legacy format
        legacy_cameras = {}
        for cam_id, camera in self.cameras.items():
            legacy_cameras[cam_id] = {
                'R': camera.R,
                't': camera.t,
                'K': camera.K  # Each camera has its own K
            }
        
        # Convert points to legacy format (3xN array)
        points_3d_array = self.get_points_array()
        
        # Convert observations to legacy format
        legacy_observations = {}
        for cam_id in self.cameras:
            legacy_observations[cam_id] = []
            for point_id in self.camera_to_points[cam_id]:
                obs = self.observations[(cam_id, point_id)]
                legacy_observations[cam_id].append({
                    'point_id': point_id,
                    'image_point': [float(obs.coords_2d[0]), float(obs.coords_2d[1])]
                })
        
        # For backward compatibility, we can include a "default" camera matrix
        # This could be the first camera's K or an average
        default_K = None
        if self.cameras:
            first_camera = next(iter(self.cameras.values()))
            default_K = first_camera.K
        
        return {
            'cameras': legacy_cameras,
            'points_3d': {'points_3d': points_3d_array},
            'observations': legacy_observations,
            'camera_matrix': default_K,  # For backward compatibility only
            'initialization_info': self.initialization_info
        }

class MainPosePipeline:
    
    def __init__(self):
        self.pair_selector= InitializationPairSelector()
        self.essential_estimator = EssentialMatrixEstimator()
        self.triangulation_engine = TriangulationEngine()
        self.pose_recovery = PoseRecovery()
        self.pnp_solver = PnPSolver()
        self.progressiv_intrinsics_estimator = ProgressiveLearningIntrinsicsEstimator()
        self.incremental_bundle_adjuster = IncrementalBundleAdjuster()
        self.global_bundle_adjuster = GlobalBundleAdjuster()
        self.iterative_adjuster = IterativeRefinementPipeline()
        self.reconstruction = None

    def process_monument_reconstruction(self, matches_pickle_file, output_directory='./pose_results', chosen_images=None):
        """Complete monument pose estimation pipeline"""
        self.matches_pickle = matches_pickle_file
        self.triangulation_engine.set_matches_data(self.matches_pickle['matches_data'])
        self.triangulation_engine.set_all_image_info(self.matches_pickle['image_info'])

        # PHASE 1: Initialize with best pair + initial BA
        self._initialize_two_view_with_ba(chosen_images=chosen_images)  # No return needed
        
        # Save checkpoint after initialization
        with open('saved_variable.pkl', 'wb') as f:
            pickle.dump(self.reconstruction.to_legacy_format(), f)
        
        # PHASE 2: Sequential addition with incremental BA  
        self._add_views_with_incremental_ba()  # No parameters or return needed
        
        # PHASE 3: Final global optimization
        self._global_optimization()  # No parameters or return needed
        
        # PHASE 4: EXPORT FOR DENSE RECONSTRUCTION
        export_data = self._export_for_dense_reconstruction(output_directory)
        
        return {
            'success': True,
            'reconstruction': self.reconstruction,  # Return the Reconstruction object
            'export_data': export_data,
            'statistics': self.reconstruction.get_statistics(),
            'output_files': {
                'main_pickle': os.path.join(output_directory, 'optimized_camera_poses.pkl'),
                'json_backup': os.path.join(output_directory, 'camera_poses.json'),
                'summary_report': os.path.join(output_directory, 'reconstruction_summary.txt')
            }
        }

    def _initialize_two_view_with_ba(self, chosen_images=None):
        """
        Initialize reconstruction with two views, including:
        1. Initial triangulation with conservative thresholds
        2. Bundle adjustment to optimize cameras and points
        3. Re-triangulation of initially rejected points with relaxed thresholds
        4. Validation and filtering of all points
        5. Bootstrap triangulation with unprocessed images
        """
        save = False
        # Initialize Reconstruction object
        self.reconstruction = Reconstruction()
        
        # 1. Select best initialization pair
        if chosen_images is None:
            best_pair = self.pair_selector.get_best_pair_for_pipeline(self.matches_pickle['matches_data'], image_size1=self.matches_pickle['image_info'],
                                                                       image_size2=self.matches_pickle['image_info'])
        else: 
            best_pair = self.pair_selector.get_selected_pair_for_pipeline(
                self.matches_pickle['matches_data'], selected_pair=chosen_images, image1_size=self.matches_pickle['image_info'][chosen_images[0]]['size'],
                  image2_size=self.matches_pickle['image_info'][chosen_images[1]]['size']
            )
        
        image1, image2 = best_pair['image_pair']
        
        # 2. Initial essential matrix estimation (for initial camera matrices)
        print("\n=== INITIAL ESSENTIAL MATRIX ESTIMATION ===")
        essential_result = self.essential_estimator.estimate(
            best_pair['pts1'], best_pair['pts2'], 
            image_size1=self.matches_pickle['image_info'][image1]['size'], 
            image_size2=self.matches_pickle['image_info'][image2]['size']
        )
        
        if not essential_result['success']:
            print(f"Initial estimation failed: {essential_result.get('error', 'Unknown error')}")
            print("Attempting with default camera matrices...")
            
            # Fallback: use default camera matrices
            image_size1 = self.matches_pickle['image_info'][image1]['size']
            image_size2 = self.matches_pickle['image_info'][image2]['size']
            K1_init = self.essential_estimator.estimate_camera_matrix(image_size1)
            K2_init = self.essential_estimator.estimate_camera_matrix(image_size2)
        else:
            # Use initial estimate as starting point
            K1_init = essential_result['camera_matrices'][0]
            K2_init = essential_result['camera_matrices'][1]
            
            print(f"Initial essential matrix estimation:")
            print(f"  Method: {essential_result['method']}")
            print(f"  Points: {essential_result['num_points']}")
            print(f"  Inliers: {essential_result['num_inliers']} ({essential_result['inlier_ratio']:.2%})")
        
        # 3. ITERATIVE REFINEMENT - This is the key addition
        print("\n=== ITERATIVE REFINEMENT ===")
        
        # Get inlier points from initial estimation (or use all if initial failed)
        if essential_result['success']:
            mask = essential_result['inlier_mask'].ravel().astype(bool)
            inlier_pts1 = best_pair['pts1'][mask]
            inlier_pts2 = best_pair['pts2'][mask]
        else:
            inlier_pts1 = best_pair['pts1']
            inlier_pts2 = best_pair['pts2']
                
        # original_inlier_pts1 = inlier_pts1.copy()
        # original_inlier_pts2 = inlier_pts2.copy()
        triangulated_mask = np.zeros(len(inlier_pts1), dtype=bool)  # Track which points were triangulated
        # Run iterative refinement
        refinement_result = self.iterative_adjuster.iterative_refinement_with_relaxation(
            inlier_pts1, inlier_pts2,
            K1_init=K1_init,
            K2_init=K2_init,
            image_size1=self.matches_pickle['image_info'][image1]['size'],
            image_size2=self.matches_pickle['image_info'][image2]['size']
        )
        

        # 4. Use refined results or fall back to initial
        if refinement_result and refinement_result.get('success'):
            print("\n✓ Using refined camera matrices and pose")
            
            # Extract refined parameters
            K1_final = refinement_result['K1']
            K2_final = refinement_result['K2']
            R_final = refinement_result['R']
            t_final = refinement_result['t']
            triangulated_mask = refinement_result['triangulated_mask']

            # Use refined points if available
            if 'points_3d' in refinement_result:
                points_3d = refinement_result['points_3d']
                valid_pts1 = refinement_result.get('valid_pts1', inlier_pts1[:points_3d.shape[1]])
                valid_pts2 = refinement_result.get('valid_pts2', inlier_pts2[:points_3d.shape[1]])
                initial_point_count = points_3d.shape[1]
                
                # if 'mask_inliers' in refinement_result:
                #     # The refinement mask tells us which of the inlier points were successfully triangulated
                #     refinement_mask = refinement_result['mask_inliers']
                #     # Map back to original inliers
                #     triangulated_indices = np.where(refinement_mask)[0]
                #     for idx in triangulated_indices[:initial_point_count]:
                #         if idx < len(triangulated_mask):
                #             triangulated_mask[idx] = True
                # else:
                #     # Assume first N points were triangulated
                #     triangulated_mask[:initial_point_count] = True


                print(f"Refined triangulation: {initial_point_count} points")
                print(f"Refined K1 focal: {K1_final[0,0]:.1f}")
                print(f"Refined K2 focal: {K2_final[0,0]:.1f}")
                
                # Skip additional triangulation since refinement already did it
                skip_triangulation = True
            else:
                skip_triangulation = False
                
            # Update essential result with refined values
            essential_result['camera_matrices'] = [K1_final, K2_final]
            essential_result['camera_estimated'] = [True, True]
            
            # Create pose result with refined values
            pose_result = {
                'R': R_final,
                't': t_final,
                'success': True
            }
            
        else:
            print("\n⚠ Refinement failed, falling back to initial estimation")
            
            if not essential_result['success']:
                raise RuntimeError("Both initial estimation and refinement failed")
            
            # Fall back to initial estimation
            skip_triangulation = False
            
            # Recover pose from initial essential matrix
            pose_result = self.pose_recovery.recover_from_essential(
                essential_result['essential_matrix'],
                inlier_pts1, 
                inlier_pts2,
                essential_result['camera_matrices']
            )
            
            K1_final = essential_result['camera_matrices'][0]
            K2_final = essential_result['camera_matrices'][1]
            R_final = pose_result['R']
            t_final = pose_result['t']
        
        # 5. Add cameras to reconstruction with final parameters
        print("\n=== ADDING CAMERAS TO RECONSTRUCTION ===")
        
        # First camera at origin
        self.reconstruction.add_camera(
            camera_id=image1,
            R=np.eye(3),
            t=np.zeros((3,1)),
            K=K1_final
        )
        
        # Second camera with relative pose
        self.reconstruction.add_camera(
            camera_id=image2,
            R=R_final,
            t=t_final,
            K=K2_final
        )
        
        print(f"Added camera {image1} at origin")
        print(f"Added camera {image2} with relative pose")
        
        # 6. Handle triangulation (if not already done by refinement)
        if not skip_triangulation:
            print("\n=== TRIANGULATION ===")
            print(f"Triangulating {len(inlier_pts1)} inlier correspondences...")
            
            triangulated_result = self.triangulation_engine.triangulate_initial_points(
                inlier_pts1,
                inlier_pts2,
                np.eye(3),
                np.zeros((3,1)),
                R_final,
                t_final,
                [K1_final, K2_final],
                best_pair['image_pair']
            )
            
            # Extract triangulated points
            if isinstance(triangulated_result, dict) and 'points_3d' in triangulated_result:
                points_3d = triangulated_result['points_3d']
                initial_stats = triangulated_result.get('statistics', {})
            else:
                points_3d = triangulated_result
                initial_stats = {}
            
            initial_point_count = points_3d.shape[1]
            valid_pts1 = inlier_pts1[:initial_point_count]  
            valid_pts2 = inlier_pts2[:initial_point_count]
            
            #triangulated_mask[:initial_point_count] = True

            print(f"Successfully triangulated: {initial_point_count}/{len(inlier_pts1)} points")
            print(f"Initial success rate: {100.0 * initial_point_count / len(inlier_pts1):.1f}%")
        
        # 7. Add points and observations to reconstruction
        for i in range(points_3d.shape[1]):
            point_id = self.reconstruction.add_point(points_3d[:, i])
            
            # Add observations
            if i < len(valid_pts1):
                self.reconstruction.add_observation(image1, point_id, valid_pts1[i])
                self.reconstruction.add_observation(image2, point_id, valid_pts2[i])
        
        # 8. Store initialization info
        self.reconstruction.initialization_info = {
            'method': 'iterative_refinement' if refinement_result and refinement_result.get('success') else essential_result.get('method', 'unknown'),
            'num_inliers': len(inlier_pts1),
            'initial_triangulated': initial_point_count,
            'initial_rejected': len(inlier_pts1) - initial_point_count,
            'refinement_iterations': refinement_result.get('iteration', 0) if refinement_result else 0,
            'final_score': refinement_result.get('score', 0) if refinement_result else 0,
            'camera_estimated': [True, True],  # Both were estimated/refined
            'quality_assessment': essential_result.get('quality_assessment', {})
        }
        
        # all_original_correspondences = {
        #     'pts1': best_pair['pts1'].copy(),
        #     'pts2': best_pair['pts2'].copy(),
        #     'inlier_pts1': original_inlier_pts1.copy(),
        #     'inlier_pts2': original_inlier_pts2.copy(),
        #     'triangulated_mask': triangulated_mask,  # Now properly defined
        #     'image1': image1,
        #     'image2': image2
        # }

        # 10. Convert to legacy format for bundle adjustment
        reconstruction_state = self.reconstruction.to_legacy_format()
        
        print(f"\n=== INITIALIZATION SUMMARY ===")
        print(f"Cameras: {len(self.reconstruction.cameras)}")
        print(f"3D Points: {len(self.reconstruction.points)}")
        print(f"Observations: {len(self.reconstruction.observations)}")
        
        # 10. INITIAL BUNDLE ADJUSTMENT
        print("\n=== BUNDLE ADJUSTMENT ===")
        print(f"Optimizing {len(self.reconstruction.cameras)} cameras and {initial_point_count} points...")
        
        if save:
            reconstruction_state = self.incremental_bundle_adjuster.adjust_after_new_view(
                reconstruction_state,
                image2,
                optimize_intrinsics=essential_result['camera_estimated']
            )
        
            with open('reconstruction_after_ba.pkl', 'wb') as f:
                pickle.dump(reconstruction_state, f)
            print("Reconstruction state saved to reconstruction_after_ba.pkl")
        else :
            with open('reconstruction_after_ba.pkl', 'rb') as f:
                reconstruction_state = pickle.load(f)
            print("Loaded reconstruction state from reconstruction_after_ba.pkl")

        print("Bundle adjustment complete")
        
        # 11. RE-TRIANGULATE INITIALLY REJECTED POINTS
        print("\n=== RE-TRIANGULATION WITH OPTIMIZED CAMERAS ===")
        
        num_rejected = len(inlier_pts1) - initial_point_count
        if num_rejected > 0:
            print(f"Re-evaluating {num_rejected} initially rejected points...")
            
            # Get points that were initially rejected
            rejected_indices = np.where(~triangulated_mask)[0]
            rejected_pts1 = inlier_pts1[rejected_indices]
            rejected_pts2 = inlier_pts2[rejected_indices]
            
            # Get optimized camera parameters
            cameras = reconstruction_state['cameras']
            cam1 = cameras[image1]
            cam2 = cameras[image2]
            
            # Calculate adaptive depth bounds based on existing points
            existing_points = reconstruction_state['points_3d']['points_3d']
            min_depth, max_depth = self._calculate_adaptive_depth_bounds(existing_points, cameras)
            
            # Create triangulation engine with relaxed thresholds
            retriangulation_engine = TriangulationEngine(
                min_triangulation_angle_deg=0.5,  # Relaxed from 2.0
                max_reprojection_error=4.0,       # Relaxed from 2.0
                min_depth=min_depth,               # Adaptive
                max_depth=max_depth                # Adaptive
            )
            
            # Re-triangulate with optimized cameras
            retriangulated_result = retriangulation_engine.triangulate_initial_points(
                pts1=rejected_pts1,
                pts2=rejected_pts2,
                R1=cam1['R'],
                t1=cam1['t'],
                R2=cam2['R'],
                t2=cam2['t'],
                K=[cam1.get('K', essential_result['camera_matrices'][0]),
                cam2.get('K', essential_result['camera_matrices'][1])],
                image_pair=(image1, image2)
            )
            
            recovered_points = retriangulated_result['points_3d']
            num_recovered = recovered_points.shape[1] if recovered_points.size > 0 else 0
            
            print(f"Successfully recovered: {num_recovered}/{num_rejected} points")
            print(f"Recovery rate: {100.0 * num_recovered / num_rejected:.1f}%")
            
            # Add recovered points to reconstruction state
            if num_recovered > 0:
                current_points = reconstruction_state['points_3d']['points_3d']
                current_point_count = current_points.shape[1]
                
                # Append recovered points
                updated_points = np.hstack([current_points, recovered_points])
                reconstruction_state['points_3d']['points_3d'] = updated_points
                
                # Add observations for recovered points
                observations = reconstruction_state.get('observations', {})
                if image1 not in observations:
                    observations[image1] = []
                if image2 not in observations:
                    observations[image2] = []
                
                # Use the actual recovered 2D points (subset of rejected points that succeeded)
                if 'observations' in retriangulated_result:
                    # Extract which rejected points were successfully triangulated
                    for obs in retriangulated_result['observations']:
                        if obs['image_id'] == image1:
                            pt_idx = obs.get('point_id', 0)
                            if pt_idx < len(rejected_pts1):
                                observations[image1].append({
                                    'point_id': current_point_count + pt_idx,
                                    'image_point': rejected_pts1[pt_idx].tolist(),
                                    'source': 'retriangulation'
                                })
                        elif obs['image_id'] == image2:
                            pt_idx = obs.get('point_id', 0)
                            if pt_idx < len(rejected_pts2):
                                observations[image2].append({
                                    'point_id': current_point_count + pt_idx,
                                    'image_point': rejected_pts2[pt_idx].tolist(),
                                    'source': 'retriangulation'
                                })
                
                reconstruction_state['observations'] = observations
        
        # 12. VALIDATE ALL POINTS AFTER BA AND RE-TRIANGULATION
        print("\n=== POINT VALIDATION ===")
        total_points_before_validation = reconstruction_state['points_3d']['points_3d'].shape[1]
        print(f"Validating {total_points_before_validation} total points...")
        
        reconstruction_state = self._validate_points_after_ba(
            reconstruction_state,
            max_reprojection_error=3.0,
            min_triangulation_angle=1.0
        )
        
        # 13. Update reconstruction from validated state
        self._update_reconstruction_from_validated_state(reconstruction_state)
        
        points_after_validation = len(self.reconstruction.points)
        points_removed = total_points_before_validation - points_after_validation
        
        print(f"Validation complete:")
        print(f"  Kept: {points_after_validation} points")
        print(f"  Removed: {points_removed} points")
        
        # 14. Save checkpoint after initialization
        print("\n=== SAVING CHECKPOINT ===")
        with open('saved_variable.pkl', 'wb') as f:
            pickle.dump(self.reconstruction.to_legacy_format(), f)
        print("Checkpoint saved to saved_variable.pkl")
        
        # 15. BOOTSTRAP TRIANGULATION WITH UNPROCESSED IMAGES
        print("\n=== BOOTSTRAP PROGRESSIVE TRIANGULATION ===")
        
        processed_images = set(self.reconstruction.cameras.keys())
        all_images = set(self.matches_pickle['image_info'].keys())
        bootstrap_images = all_images - processed_images
        
        print(f"Bootstrapping with {len(bootstrap_images)} additional unprocessed images...")
        
        if len(bootstrap_images) > 0:
            # Convert to legacy format for bootstrap
            reconstruction_state = self.reconstruction.to_legacy_format()
            
            bootstrap_points_added = self._bootstrap_triangulate_with_both_cameras(
                image1, image2, reconstruction_state, bootstrap_images
            )
            
            # Update reconstruction from bootstrap results
            self._update_reconstruction_from_state(reconstruction_state)
            
            final_points = len(self.reconstruction.points)
            
            print(f"\nBootstrap triangulation complete:")
            print(f"  Points before bootstrap: {points_after_validation}")
            print(f"  Bootstrap points added: {bootstrap_points_added}")
            print(f"  Total points: {final_points}")
            
            # Run BA again if we added significant points
            if bootstrap_points_added > 50:
                print("\nRunning bundle adjustment after bootstrap...")
                reconstruction_state = self.reconstruction.to_legacy_format()
                reconstruction_state = self.incremental_bundle_adjuster.adjust_after_new_view(
                    reconstruction_state,
                    image1,
                    optimize_intrinsics=False
                )
                self._update_reconstruction_from_state(reconstruction_state)
                print("Post-bootstrap BA complete")
        
        # 16. Final summary
        print("\n" + "="*60)
        print("INITIALIZATION COMPLETE")
        print("="*60)
        final_stats = self.reconstruction.get_statistics()
        print(f"Final reconstruction statistics:")
        print(f"  Cameras: {final_stats['num_cameras']}")
        print(f"  3D Points: {final_stats['num_points']}")
        print(f"  Observations: {final_stats['num_observations']}")
        print(f"  Mean track length: {final_stats['mean_track_length']:.2f}")
        
        # Summary of point sources
        init_info = self.reconstruction.initialization_info
        print(f"\nPoint source breakdown:")
        print(f"  Initial triangulation: {init_info.get('initial_triangulated', 0)}")
        print(f"  Recovered after BA: {num_recovered if num_rejected > 0 else 0}")
        print(f"  Bootstrap triangulation: {bootstrap_points_added if len(bootstrap_images) > 0 else 0}")
        print(f"  Removed by validation: {points_removed}")
        print("="*60)

    def _validate_points_after_ba(self, reconstruction_state: Dict, 
                                    max_reprojection_error: float = 3.0,
                                    min_triangulation_angle: float = 1.0) -> Dict:
            """
            Validate and filter 3D points after bundle adjustment.
            
            This is CRITICAL because BA can produce:
            - Points with high reprojection errors
            - Points behind cameras
            - Points at infinity
            - Points with poor triangulation angles
            """
            
            print("\n=== Validating points after BA ===")
            
            cameras = reconstruction_state['cameras']
            points_3d = reconstruction_state['points_3d']['points_3d']
            observations = reconstruction_state.get('observations', {})
            
            if points_3d.shape[1] == 0:
                return reconstruction_state
            
            # Calculate quality metrics for each point
            point_quality = {}
            
            for point_id in range(points_3d.shape[1]):
                point_3d = points_3d[:, point_id]
                
                # Get all observations of this point
                point_observations = []
                for cam_id, cam_obs in observations.items():
                    for obs in cam_obs:
                        if obs['point_id'] == point_id:
                            point_observations.append({
                                'camera_id': cam_id,
                                'image_point': np.array(obs['image_point'])
                            })
                
                if len(point_observations) < 2:
                    # Point needs at least 2 observations
                    point_quality[point_id] = {'valid': False, 'reason': 'insufficient_observations'}
                    continue
                
                # Calculate reprojection errors
                reprojection_errors = []
                triangulation_angles = []
                
                for obs in point_observations:
                    cam_id = obs['camera_id']
                    if cam_id not in cameras:
                        continue
                        
                    cam_data = cameras[cam_id]
                    R = cam_data['R']
                    t = cam_data['t']
                    K = cam_data.get('K', reconstruction_state.get('camera_matrix'))
                    
                    if K is None:
                        continue
                    
                    # Project point
                    rvec = cv2.Rodrigues(R)[0]
                    projected, _ = cv2.projectPoints(
                        point_3d.reshape(1, 3),
                        rvec, t, K, None
                    )
                    projected_point = projected[0, 0]
                    
                    # Calculate error
                    error = np.linalg.norm(projected_point - obs['image_point'])
                    reprojection_errors.append(error)
                    
                    # Check if point is in front of camera
                    point_cam = R @ point_3d + t.flatten()
                    if point_cam[2] <= 0:
                        point_quality[point_id] = {'valid': False, 'reason': 'behind_camera'}
                        break
                
                # Calculate triangulation angle between first two observations
                if len(point_observations) >= 2:
                    cam1_id = point_observations[0]['camera_id']
                    cam2_id = point_observations[1]['camera_id']
                    
                    cam1 = cameras[cam1_id]
                    cam2 = cameras[cam2_id]
                    
                    # Camera centers
                    C1 = -cam1['R'].T @ cam1['t']
                    C2 = -cam2['R'].T @ cam2['t']
                    
                    # Rays to point
                    ray1 = point_3d.reshape(-1, 1) - C1
                    ray2 = point_3d.reshape(-1, 1) - C2
                    
                    # Angle between rays
                    cos_angle = np.dot(ray1.T, ray2)[0, 0] / (np.linalg.norm(ray1) * np.linalg.norm(ray2))
                    angle = np.arccos(np.clip(np.abs(cos_angle), 0, 1))
                    triangulation_angles.append(np.degrees(angle))
                
                # Determine if point is valid
                if point_quality.get(point_id, {}).get('valid') is False:
                    continue  # Already marked invalid
                    
                mean_error = np.mean(reprojection_errors) if reprojection_errors else float('inf')
                min_angle = np.min(triangulation_angles) if triangulation_angles else 0
                
                # Check quality criteria
                distance = np.linalg.norm(point_3d)
                
                is_valid = (
                    mean_error < max_reprojection_error and
                    min_angle > min_triangulation_angle and
                    0.1 < distance < 100.0 and  # Reasonable distance range
                    np.all(np.isfinite(point_3d))  # No NaN or inf
                )
                
                point_quality[point_id] = {
                    'valid': is_valid,
                    'mean_reprojection_error': mean_error,
                    'min_triangulation_angle': min_angle,
                    'distance': distance,
                    'num_observations': len(point_observations)
                }
            
            # Filter points and update reconstruction state
            valid_point_ids = [pid for pid, quality in point_quality.items() if quality['valid']]
            invalid_count = len(point_quality) - len(valid_point_ids)
            
            print(f"Point validation results:")
            print(f"  Total points: {len(point_quality)}")
            print(f"  Valid points: {len(valid_point_ids)}")
            print(f"  Removed points: {invalid_count}")
            
            if invalid_count > 0:
                # Show reasons for removal
                removal_reasons = {}
                for pid, quality in point_quality.items():
                    if not quality['valid']:
                        reason = quality.get('reason', 'quality_threshold')
                        removal_reasons[reason] = removal_reasons.get(reason, 0) + 1
                
                print("  Removal reasons:")
                for reason, count in removal_reasons.items():
                    print(f"    {reason}: {count}")
            
            # Create filtered reconstruction state
            filtered_state = self._create_filtered_state(
                reconstruction_state, valid_point_ids, point_quality
            )
            
            return filtered_state

    
    def _create_filtered_state(self, reconstruction_state: Dict, 
                              valid_point_ids: List[int],
                              point_quality: Dict) -> Dict:
        """Create new reconstruction state with only valid points"""
        
        filtered_state = reconstruction_state.copy()
        
        # Filter 3D points
        old_points = reconstruction_state['points_3d']['points_3d']
        new_points = []
        
        # Create mapping from old to new point IDs
        old_to_new_id = {}
        new_id = 0
        
        for old_id in valid_point_ids:
            new_points.append(old_points[:, old_id])
            old_to_new_id[old_id] = new_id
            new_id += 1
        
        if new_points:
            filtered_state['points_3d']['points_3d'] = np.column_stack(new_points)
        else:
            filtered_state['points_3d']['points_3d'] = np.empty((3, 0))
        
        # Update observations with new point IDs
        filtered_observations = {}
        
        for cam_id, cam_obs in reconstruction_state.get('observations', {}).items():
            filtered_cam_obs = []
            
            for obs in cam_obs:
                old_point_id = obs['point_id']
                
                if old_point_id in old_to_new_id:
                    # Update observation with new point ID
                    new_obs = obs.copy()
                    new_obs['point_id'] = old_to_new_id[old_point_id]
                    
                    # Add quality metrics
                    if old_point_id in point_quality:
                        new_obs['quality'] = point_quality[old_point_id]
                    
                    filtered_cam_obs.append(new_obs)
            
            if filtered_cam_obs:
                filtered_observations[cam_id] = filtered_cam_obs
        
        filtered_state['observations'] = filtered_observations
        
        # Add validation metadata
        filtered_state['validation_info'] = {
            'total_points_before': len(point_quality),
            'valid_points': len(valid_point_ids),
            'removed_points': len(point_quality) - len(valid_point_ids),
            'mean_quality': np.mean([q['mean_reprojection_error'] 
                                    for q in point_quality.values() 
                                    if q['valid'] and 'mean_reprojection_error' in q])
        }
        
        return filtered_state
    
    def _update_reconstruction_from_validated_state(self, validated_state: Dict):
        """
        Properly update Reconstruction object from validated and filtered state.
        This replaces the incomplete _update_reconstruction_from_state method.
        """
        
        # Clear and rebuild reconstruction with validated data
        self.reconstruction = Reconstruction()
        
        # Re-add all cameras with optimized poses and intrinsics
        for cam_id, cam_data in validated_state['cameras'].items():
            self.reconstruction.add_camera(
                camera_id=cam_id,
                R=cam_data['R'],
                t=cam_data['t'],
                K=cam_data['K']
            )
        
        # Add validated 3D points
        points_3d = validated_state['points_3d']['points_3d']
        observations = validated_state.get('observations', {})
        
        # Track which observations have been added
        point_id_mapping = {}
        
        for old_point_id in range(points_3d.shape[1]):
            point_coords = points_3d[:, old_point_id]
            new_point_id = self.reconstruction.add_point(point_coords)
            point_id_mapping[old_point_id] = new_point_id
        
        # Add observations
        for cam_id, cam_obs in observations.items():
            for obs in cam_obs:
                old_point_id = obs['point_id']
                if old_point_id in point_id_mapping:
                    new_point_id = point_id_mapping[old_point_id]
                    self.reconstruction.add_observation(
                        cam_id, 
                        new_point_id,
                        np.array(obs['image_point'])
                    )
        
        # Restore metadata
        if 'initialization_info' in validated_state:
            self.reconstruction.initialization_info = validated_state['initialization_info']
        
        if 'validation_info' in validated_state:
            self.reconstruction.initialization_info['validation'] = validated_state['validation_info']

    def _calculate_adaptive_depth_bounds(self, existing_points: np.ndarray, 
                                        cameras: Dict) -> Tuple[float, float]:
        """
        Calculate adaptive depth bounds based on existing reconstruction.
        
        Args:
            existing_points: Current 3D points (3xN array)
            cameras: Dictionary of camera parameters
            
        Returns:
            Tuple of (min_depth, max_depth) appropriate for the scene scale
        """
        
        # If we have existing points, use their distribution
        if existing_points.shape[1] > 10:
            # Calculate distances from origin (roughly camera center for first cam)
            distances = np.linalg.norm(existing_points, axis=0)
            
            # Use percentiles to be robust to outliers
            min_dist = np.percentile(distances, 5)
            max_dist = np.percentile(distances, 95)
            
            # Extend bounds for re-triangulation (more permissive)
            min_depth = min_dist * 0.3  # Allow closer points
            max_depth = max_dist * 2.0  # Allow farther points
            
            print(f"  Adaptive depth bounds: [{min_depth:.2f}, {max_depth:.2f}]")
            print(f"  Based on existing point range: [{min_dist:.2f}, {max_dist:.2f}]")
            
        else:
            # Fallback: use baseline between cameras
            if len(cameras) >= 2:
                cam_list = list(cameras.values())
                cam1, cam2 = cam_list[0], cam_list[1]
                
                # Camera centers
                C1 = -cam1['R'].T @ cam1['t']
                C2 = -cam2['R'].T @ cam2['t']
                baseline = np.linalg.norm(C1 - C2)
                
                # Depth relative to baseline
                min_depth = baseline * 0.5
                max_depth = baseline * 100.0
                
                print(f"  Baseline-based depth bounds: [{min_depth:.2f}, {max_depth:.2f}]")
                print(f"  Camera baseline: {baseline:.2f}")
            else:
                # Conservative defaults
                min_depth = 0.1
                max_depth = 1000.0
                print(f"  Using default depth bounds: [{min_depth:.2f}, {max_depth:.2f}]")
        
        return min_depth, max_depth

    def _add_views_with_incremental_ba(self):
        """Add views incrementally - works directly with self.reconstruction"""
        processed_images = set(self.reconstruction.cameras.keys())
        remaining_images = set(self.matches_pickle['image_info'].keys()) - processed_images
        
        while remaining_images:
            # Select best next image using internal reconstruction
            selected_image = self._select_best_next_image(remaining_images)
            print(f"\n--- Adding view: {selected_image} ---")
            
            # Estimate intrinsics using internal reconstruction
            K_heuristic = self._estimate_intrinsics_for_new_view(selected_image)
            
            # Find correspondences using internal reconstruction
            correspondences_3d_2d = self._find_correspondences_with_existing_3d(selected_image)

            if correspondences_3d_2d['num_correspondences'] < 15:
                print(f"Insufficient correspondences for {selected_image}")
                remaining_images.remove(selected_image)
                continue
            
            # Solve PnP
            pnp_result = self.pnp_solver.solve_pnp(
                correspondences_3d_2d['points_3d'],
                correspondences_3d_2d['points_2d'],
                K_heuristic
            )
            
            if not pnp_result['success']:
                print(f"PnP failed for {selected_image}")
                continue
            
            # Add camera directly to reconstruction
            self.reconstruction.add_camera(
                camera_id=selected_image,
                R=pnp_result['R'],
                t=pnp_result['t'],
                K=K_heuristic
            )
            
            # Triangulate new points directly into reconstruction
            self._triangulate_new_points(selected_image)
            
            # Add observations from correspondences
            self._add_observations_from_correspondences(selected_image, correspondences_3d_2d)
            
            # Bundle adjustment on internal reconstruction
            self._run_incremental_ba(selected_image, optimize_intrinsics=True)
            
            processed_images.add(selected_image)
            
            stats = self.reconstruction.get_statistics()
            print(f"Camera {selected_image} added!")
            print(f"Total cameras: {stats['num_cameras']}")
            print(f"Total 3D points: {stats['num_points']}")

    def _retriangulate_filtered_points(self, 
                                      reconstruction_state: Dict,
                                      original_correspondences: Dict,
                                      initial_mask: np.ndarray,
                                      original_K_matrices: List[np.ndarray]) -> Dict:
        """
        Re-triangulate points that were initially filtered out, using optimized camera poses.
        
        After BA, the camera poses are more accurate, so points that previously had:
        - High reprojection error
        - Poor triangulation angle  
        - Ambiguous depth
        might now be valid.
        """
        
        # Get optimized camera poses from BA
        cameras = reconstruction_state['cameras']
        image1 = original_correspondences['image1']
        image2 = original_correspondences['image2']
        
        cam1 = cameras[image1]
        cam2 = cameras[image2]
        
        # Use optimized intrinsics if available, otherwise use original
        K1 = cam1.get('K', original_K_matrices[0])
        K2 = cam2.get('K', original_K_matrices[1])
        
        # Create projection matrices with optimized poses
        P1 = K1 @ np.hstack([cam1['R'], cam1['t'].reshape(-1, 1)])
        P2 = K2 @ np.hstack([cam2['R'], cam2['t'].reshape(-1, 1)])
        
        # Find points that were initially rejected
        rejected_indices = np.where(~initial_mask)[0]
        
        if len(rejected_indices) == 0:
            return {'num_recovered': 0, 'recovered_points': np.empty((3, 0))}
        
        print(f"  Re-evaluating {len(rejected_indices)} initially filtered points...")
        
        # Get the correspondence for rejected points
        rejected_pts1 = original_correspondences['pts1'][rejected_indices]
        rejected_pts2 = original_correspondences['pts2'][rejected_indices]
        
        # Re-triangulate with optimized cameras
        recovered_points = []
        recovered_observations = []
        
        for i, idx in enumerate(rejected_indices):
            pt1 = rejected_pts1[i]
            pt2 = rejected_pts2[i]
            
            # Triangulate single point pair
            point_3d = self._triangulate_single_point(pt1, pt2, P1, P2)
            
            if point_3d is None:
                continue
            
            # Validate with relaxed criteria (BA has already improved geometry)
            if self._validate_retriangulated_point(
                point_3d, pt1, pt2, P1, P2, 
                cam1['R'], cam1['t'], cam2['R'], cam2['t'],
                max_reprojection_error=4.0,  # Slightly more relaxed
                min_triangulation_angle=0.5   # More permissive
            ):
                recovered_points.append(point_3d)
                recovered_observations.append({
                    'pt1': pt1,
                    'pt2': pt2,
                    'original_idx': idx
                })
        
        print(f"  Successfully recovered {len(recovered_points)} points")
        
        if recovered_points:
            # Add recovered points to reconstruction state
            self._add_recovered_points_to_state(
                reconstruction_state,
                recovered_points,
                recovered_observations,
                image1, image2
            )
            
            # Show improvement statistics
            self._show_recovery_statistics(
                recovered_points, 
                rejected_pts1, rejected_pts2,
                P1, P2
            )
        
        return {
            'num_recovered': len(recovered_points),
            'recovered_points': np.column_stack(recovered_points) if recovered_points else np.empty((3, 0)),
            'recovered_observations': recovered_observations
        }


    # def _add_views_with_incremental_ba(self, reconstruction_state):
        
    #     processed_images = set(reconstruction_state['cameras'].keys())
    #     remaining_images = set(self.matches_pickle['image_info'].keys()) - processed_images
        
    #     while remaining_images:

    #         selected_image = self.select_best_next_image(
    #             reconstruction_state, remaining_images
    #         )
    #         print(f"\n--- Adding view: {selected_image} ---")

    #         K_heuristic = self.progressiv_intrinsics_estimator.estimate_intrinsics_with_progressive_learning(
    #             selected_image, 
    #             reconstruction_state)
            
    #         # 2. FIND CORRESPONDENCES FOR SELECTED IMAGE
    #         correspondences_3d_2d = self.find_correspondences_with_existing_3d(
    #             selected_image, reconstruction_state
    #         )
            
    #         if correspondences_3d_2d['num_correspondences'] < 15:
    #             print(f"Insufficient correspondences for {selected_image}, removing from candidates...")
    #             remaining_images.remove(selected_image)
    #             continue
            
    #         # 3. SOLVE PnP
    #         pnp_result = self.pnp_solver.solve_pnp(
    #             correspondences_3d_2d['points_3d'],
    #             correspondences_3d_2d['points_2d'],
    #             K_heuristic
    #         )
            
    #         if not pnp_result['success']:
    #             print(f"PnP failed for {selected_image}, skipping...")
    #             continue
            
    #         # 3. Add camera to reconstruction
    #         reconstruction_state['cameras'][selected_image] = {
    #             'K': K_heuristic.copy(),
    #             'R': pnp_result['R'], 
    #             't': pnp_result['t']
    #         }
            
    #         # 4. Triangulate new points from this view
    #         new_points_3d = self.triangulation_engine.triangulate_new_points(selected_image, reconstruction_state)
    #         reconstruction_state['points_3d'] = np.hstack([
    #             reconstruction_state['points_3d'], new_points_3d
    #         ])
            
    #         # 5. UPDATE OBSERVATIONS
    #         reconstruction_state = update_observation_structure(
    #             reconstruction_state, selected_image, correspondences_3d_2d
    #         )
            
    #         # 6. ✨ INCREMENTAL BUNDLE ADJUSTMENT AFTER EACH VIEW ✨
    #         print(f"Running incremental BA after adding {selected_image}...")
    #         reconstruction_state = self.incremental_bundle_adjuster.adjust_after_new_view(
    #             reconstruction_state,
    #             selected_image,
    #             reconstruction_state['camera_matrix'],
    #             optimize_intrinsics=True  # Optimize K matrix each time
    #         )
            
    #         processed_images.add(selected_image)
            
    #         print(f"Camera {selected_image} added and optimized!")
    #         print(f"Total cameras: {len(reconstruction_state['cameras'])}")
    #         print(f"Total 3D points: {reconstruction_state['points_3d'].shape[1]}")
        
    #     return reconstruction_state

    #triangulation helping methods

    def _bootstrap_triangulate_with_both_cameras(self, camera1: str, camera2: str,
                                            reconstruction_state: Dict[str, Any],
                                            bootstrap_images: set) -> int:
        """
        🆕 IMPROVED: Bootstrap triangulation using BOTH initialized cameras
        
        This triangulates between BOTH known cameras and all unprocessed images,
        maximizing the number of initial 3D points.
        """
        
        print(f"Bootstrap triangulating with both cameras: {camera1} and {camera2}")
        
        # Get camera data for both initialized cameras
        cam1_data = reconstruction_state['cameras'][camera1]
        cam2_data = reconstruction_state['cameras'][camera2]
        
        cam1_R, cam1_t = cam1_data['R'], cam1_data['t']
        cam2_R, cam2_t = cam2_data['R'], cam2_data['t']
        
        fallback_K = reconstruction_state.get('camera_matrix')
        cam1_K = cam1_data.get('K', fallback_K)
        cam2_K = cam2_data.get('K', fallback_K)
        
        points_added = 0
        
        # Get current point count for new point IDs
        current_points_data = reconstruction_state.get('points_3d', {})
        if isinstance(current_points_data, dict) and 'points_3d' in current_points_data:
            current_point_count = current_points_data['points_3d'].shape[1]
        else:
            current_point_count = current_points_data.shape[1] if hasattr(current_points_data, 'shape') else 0
        
        print(f"Starting from {current_point_count} existing points")
        
        # Strategy 1: Triangulate camera1 with all bootstrap images
        print(f"\n--- Triangulating {camera1} with bootstrap images ---")
        points_from_cam1 = self._triangulate_one_camera_with_bootstrap_images(
            camera1, cam1_R, cam1_t, cam1_K, 
            reconstruction_state, bootstrap_images, 
            current_point_count + points_added
        )
        points_added += points_from_cam1
        
        # Strategy 2: Triangulate camera2 with all bootstrap images  
        print(f"\n--- Triangulating {camera2} with bootstrap images ---")
        points_from_cam2 = self._triangulate_one_camera_with_bootstrap_images(
            camera2, cam2_R, cam2_t, cam2_K,
            reconstruction_state, bootstrap_images,
            current_point_count + points_added
        )
        points_added += points_from_cam2
        
        print(f"\nBootstrap summary:")
        print(f"  Points from {camera1}: {points_from_cam1}")
        print(f"  Points from {camera2}: {points_from_cam2}")
        print(f"  Total bootstrap points: {points_added}")
        
        return points_added

    def _triangulate_one_camera_with_bootstrap_images(self, anchor_camera: str,
                                                    anchor_R: np.ndarray, anchor_t: np.ndarray, anchor_K: np.ndarray,
                                                    reconstruction_state: Dict[str, Any], bootstrap_images: set,
                                                    starting_point_id: int) -> int:
        """
        Triangulate one known camera with all bootstrap images
        """
        
        points_added = 0
        max_points_per_camera = 200  # Limit per anchor camera
        
        for bootstrap_image in bootstrap_images:
            
            # Find matches between anchor and bootstrap image
            matches = self.triangulation_engine._find_matches_between_images_progressive(
                anchor_camera, bootstrap_image
            )
            
            if matches is None or len(matches['pts1']) < 8:
                continue
            
            print(f"  {anchor_camera} <-> {bootstrap_image}: {len(matches['pts1'])} matches")
            
            # Filter matches that don't already have 3D correspondences
            unmatched = self.triangulation_engine._filter_unmatched_progressive(
                matches, anchor_camera, bootstrap_image, reconstruction_state
            )
            
            if len(unmatched['pts1']) < 5:
                continue
            
            # Estimate pose for bootstrap image
            bootstrap_R, bootstrap_t, bootstrap_K = self._estimate_rough_pose_for_bootstrap(
                bootstrap_image, anchor_camera, anchor_R, anchor_t, anchor_K,pts1=unmatched['pts1'], pts2=unmatched['pts2'],
            )
            
            # Create projection matrices
            anchor_P = anchor_K @ np.hstack([anchor_R, anchor_t.reshape(-1, 1)])
            bootstrap_P = bootstrap_K @ np.hstack([bootstrap_R, bootstrap_t.reshape(-1, 1)])
            
            # Triangulate points
            triangulated = self.triangulation_engine._triangulate_point_pairs_progressive(
                unmatched['pts1'], unmatched['pts2'], 
                anchor_P, bootstrap_P,
                anchor_R, anchor_t, bootstrap_R, bootstrap_t
            )
            
            if triangulated:
                # Limit points per bootstrap image
                max_points_per_pair = 80
                if len(triangulated) > max_points_per_pair:
                    triangulated = triangulated[:max_points_per_pair]
                
                print(f"    Triangulated {len(triangulated)} points")
                
                # Add to reconstruction
                self._add_bootstrap_points_to_reconstruction(
                    reconstruction_state, triangulated, anchor_camera, bootstrap_image,
                    unmatched, starting_point_id + points_added
                )
                
                points_added += len(triangulated)
            
            # Global limit per anchor camera
            if points_added >= max_points_per_camera:
                print(f"    Reached limit for {anchor_camera} ({points_added} points)")
                break
        
        return points_added

    def _update_reconstruction_from_state(self, state: Dict):
        """Update Reconstruction object from legacy state dictionary"""
        # Update cameras
        for cam_id, cam_data in state['cameras'].items():
            if cam_id in self.reconstruction.cameras:
                camera = self.reconstruction.cameras[cam_id]
                camera.R = cam_data['R']
                camera.t = cam_data['t']
                if 'K' in cam_data:
                    camera.K = cam_data['K']
        
        # Update points (more complex due to potential reindexing)
        if 'points_3d' in state:
            if isinstance(state['points_3d'], dict):
                new_points = state['points_3d']['points_3d']
            else:
                new_points = state['points_3d']
            
            # For now, assuming point IDs remain consistent
            # In a full implementation, you'd need to handle point ID mapping
            for point_id, point in self.reconstruction.points.items():
                if point_id < new_points.shape[1]:
                    point.coords = new_points[:, point_id]

    def _triangulate_new_points(self, new_camera_id: str):
        """Triangulate new points directly into reconstruction"""
        new_camera = self.reconstruction.cameras[new_camera_id]
        
        # Convert to legacy format for triangulation engine
        legacy_state = self.reconstruction.to_legacy_format()
        
        # Use existing triangulation engine
        new_points_3d = self.triangulation_engine.triangulate_new_points(
            new_camera_id, legacy_state
        )
        
        # Add new points to reconstruction
        if new_points_3d.shape[1] > 0:
            for i in range(new_points_3d.shape[1]):
                point_id = self.reconstruction.add_point(new_points_3d[:, i])
                # Note: Observations should be added by triangulation engine or separately
    
    def _add_observations_from_correspondences(self, camera_id: str, 
                                            correspondences_3d_2d: Dict[str, Any]):
        """Add observations from PnP correspondences to reconstruction"""
        if correspondences_3d_2d['num_correspondences'] == 0:
            return
        
        points_2d = correspondences_3d_2d['points_2d']
        point_ids = correspondences_3d_2d['point_ids']
        
        for i in range(correspondences_3d_2d['num_correspondences']):
            point_2d = points_2d[i]
            point_id = point_ids[i]
            
            # Add observation to reconstruction
            self.reconstruction.add_observation(
                camera_id, point_id, point_2d
            )

    def _run_incremental_ba(self, new_camera_id: str, optimize_intrinsics: bool = True):
        """Run incremental BA on internal reconstruction"""
        # Convert to legacy format for BA
        legacy_state = self.reconstruction.to_legacy_format()
        
        # Run BA
        optimized_state = self.incremental_bundle_adjuster.adjust_after_new_view(
            legacy_state,
            new_camera_id,
            self.reconstruction.camera_matrix,
            optimize_intrinsics=optimize_intrinsics
        )
        
        # Update reconstruction from optimized state
        self._update_reconstruction_from_state(optimized_state)

    def _prepare_export_data_from_reconstruction(self) -> Dict:
        """Prepare export data directly from internal reconstruction"""
        export_data = {
            'camera_poses': {},
            'camera_matrix': self.reconstruction.camera_matrix,
            'points_3d': self.reconstruction.get_points_array(),
            'reconstruction_metadata': {
                'statistics': self.reconstruction.get_statistics(),
                'processing_timestamp': datetime.now().isoformat()
            }
        }
        
        # Export camera data
        for cam_id, camera in self.reconstruction.cameras.items():
            export_data['camera_poses'][cam_id] = {
                'R': camera.R,
                't': camera.t,
                'K': camera.K,
                'projection_matrix': camera.P,
                'world_position': camera.center,
                'image_size': (camera.width, camera.height) if camera.width else None
            }
        
        return export_data
    # def _estimate_rough_pose_for_bootstrap(
    #         self, bootstrap_image: str, 
    #                                     anchor_R: np.ndarray, anchor_t: np.ndarray,
    #                                     anchor_K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     🆕 NEW: Estimate rough pose for bootstrap image
        
    #     Since we don't have the actual pose yet, we estimate a reasonable pose
    #     based on the anchor camera. This will be rough but good enough for bootstrap triangulation.
    #     """
        
    #     # Strategy 1: Random offset from anchor camera
    #     # This assumes cameras are roughly in a circle around the monument
        
    #     # Random rotation around Y-axis (vertical rotation around monument)
    #     theta = np.random.uniform(0, 2*np.pi)  # Random angle
    #     rotation_y = np.array([
    #         [np.cos(theta), 0, np.sin(theta)],
    #         [0, 1, 0],
    #         [-np.sin(theta), 0, np.cos(theta)]
    #     ])
        
    #     # Small random rotation for variety
    #     small_rotation = np.random.uniform(-0.2, 0.2, 3)  # Small angles
    #     R_small = self._rodrigues_rotation(small_rotation)
        
    #     # Combine rotations
    #     bootstrap_R = rotation_y @ anchor_R @ R_small
        
    #     # Translation: offset from anchor position
    #     offset_distance = np.random.uniform(0.5, 2.0)  # Random distance offset
    #     offset_direction = np.random.uniform(-1, 1, 3)  # Random direction
    #     offset_direction = offset_direction / np.linalg.norm(offset_direction)  # Normalize
        
    #     bootstrap_t = anchor_t + offset_direction.reshape(-1, 1) * offset_distance
        
    #     # For intrinsics, use same as anchor (reasonable assumption for same camera setup)
    #     bootstrap_K = anchor_K.copy()
        
    #     return bootstrap_R, bootstrap_t, bootstrap_K



    def _estimate_rough_pose_for_bootstrap(
        self, bootstrap_image: str, anchor_camera: str,
        anchor_R: np.ndarray, anchor_t: np.ndarray, 
        anchor_K: np.ndarray,
        pts1,
        pts2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        🆕 IMPROVED: Estimate rough pose for bootstrap image using your EssentialMatrixEstimator
        
        Uses existing matches and your monument-optimized essential matrix estimation 
        with proper quality checks and validation.
        """
        
        try:

            anchor_image_size = self.matches_pickle['image_info'][anchor_camera]['size']
            bootstrap_image_size = self.matches_pickle['image_info'][bootstrap_image]['size']

            # Use your EssentialMatrixEstimator with the provided camera matrix
            result = self.essential_estimator.estimate(pts1, pts2, 
                                                       image_size1=anchor_image_size, 
                                                       image_size2=bootstrap_image_size, 
                                                       method='RANSAC')
            
            if not result['success']:
                print(f"Warning: Essential matrix estimation failed: {result['error']}")
                return self._fallback_circular_pose(anchor_R, anchor_t, anchor_K)
            
            # Extract results from your estimator
            E = result['essential_matrix']
            inlier_mask = result['inlier_mask']
            
            # Quality check using your assessment
            quality = result['quality_assessment']
            if not quality['is_valid']:
                print(f"Warning: Poor essential matrix quality (score: {quality['quality_score']:.2f})")
                for warning in quality['warnings']:
                    print(f"  - {warning}")
                
                # Still try to use it, but warn user
                if quality['quality_score'] < 0.3:
                    print("Quality too low, falling back to circular trajectory")
                    return self._fallback_circular_pose(anchor_R, anchor_t, anchor_K)
            
            # Get inlier correspondences
            inlier_pts1 = pts1[inlier_mask.ravel() == 1]
            inlier_pts2 = pts2[inlier_mask.ravel() == 1]
            
            print(f"Essential matrix estimated with {result['num_inliers']} inliers "
                f"({result['inlier_ratio']:.1%} ratio, quality: {quality['quality_score']:.2f})")
            
            # Recover pose from Essential Matrix using the camera matrix
            _, R_rel, t_rel, pose_mask = cv2.recoverPose(
                E, inlier_pts1, inlier_pts2, anchor_K
            )
            
            # Validate the recovered pose
            if not self._validate_recovered_pose(R_rel, t_rel):
                print("Warning: Recovered pose validation failed, using circular fallback")
                return self._fallback_circular_pose(anchor_R, anchor_t, anchor_K)
            
            # Convert relative pose to absolute pose
            # Essential matrix gives us pose of bootstrap relative to anchor
            # We need: bootstrap_pose = anchor_pose ∘ relative_pose
            
            # Anchor pose as transformation matrix
            anchor_T = np.eye(4)
            anchor_T[:3, :3] = anchor_R
            anchor_T[:3, 3:4] = anchor_t
            
            # Relative transformation  
            rel_T = np.eye(4)
            rel_T[:3, :3] = R_rel
            rel_T[:3, 3:4] = t_rel
            
            # Bootstrap absolute pose
            bootstrap_T = anchor_T @ rel_T
            
            bootstrap_R = bootstrap_T[:3, :3]
            bootstrap_t = bootstrap_T[:3, 3:4]
            bootstrap_K = anchor_K.copy()  # Use same camera intrinsics
            
            print("✅ Successfully estimated bootstrap pose using your EssentialMatrixEstimator")
            print(f"   Matrix quality score: {quality['quality_score']:.2f}")
            
            return bootstrap_R, bootstrap_t, bootstrap_K
            
        except Exception as e:
            print(f"Essential matrix estimation failed with error: {e}")
            print("Falling back to circular trajectory assumption")
            return self._fallback_circular_pose(anchor_R, anchor_t, anchor_K)




    def _rodrigues_rotation(self, rotation_vector: np.ndarray) -> np.ndarray:
        """Helper: Convert rotation vector to rotation matrix using Rodrigues formula"""
        theta = np.linalg.norm(rotation_vector)
        if theta < 1e-8:
            return np.eye(3)
        
        k = rotation_vector / theta
        K = np.array([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ])
        
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return R

    def _add_bootstrap_points_to_reconstruction(self, reconstruction_state: Dict, 
                                            triangulated_points: List[np.ndarray],
                                            anchor_camera: str, bootstrap_image: str,
                                            unmatched_points: Dict, starting_point_id: int):
        """
        🆕 NEW: Add bootstrap triangulated points to reconstruction
        """
        
        if not triangulated_points:
            return
        
        # Convert to array format
        new_3d_points = np.array(triangulated_points).T  # (3, N)
        
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
        
        # Add observations
        observations = reconstruction_state.get('observations', {})
        
        # Ensure bootstrap image has observation list
        if bootstrap_image not in observations:
            observations[bootstrap_image] = []
        
        for i, point_3d in enumerate(triangulated_points):
            point_id = starting_point_id + i
            
            # Add observation for anchor camera
            observations[anchor_camera].append({
                'point_id': point_id,
                'image_point': [float(unmatched_points['pts1'][i][0]), float(unmatched_points['pts1'][i][1])],
                'source': 'bootstrap_triangulation'
            })
            
            # Add observation for bootstrap image
            observations[bootstrap_image].append({
                'point_id': point_id,
                'image_point': [float(unmatched_points['pts2'][i][0]), float(unmatched_points['pts2'][i][1])],
                'source': 'bootstrap_triangulation'
            })

    # end

    def _estimate_intrinsics_for_new_view(self, selected_image: str) -> np.ndarray:
        """Estimate intrinsics for new view using progressive learning"""
        # Convert reconstruction to legacy format for compatibility
        legacy_state = self.reconstruction.to_legacy_format()
        
        K_heuristic = self.progressiv_intrinsics_estimator.estimate_intrinsics_with_progressive_learning(
            selected_image, 
            legacy_state
        )
        
        return K_heuristic

    def _global_optimization(self):
        """Global optimization - works directly with self.reconstruction"""
        print("\n" + "="*50)
        print("FINAL GLOBAL BUNDLE ADJUSTMENT")
        print("="*50)
        
        print("Optimizing all cameras and points globally...")
        
        # Convert to legacy format only for bundle adjuster
        legacy_state = self.reconstruction.to_legacy_format()
        
        # Run global BA
        optimized_state = self.global_bundle_adjuster.adjust_global(
            legacy_state,
            optimize_intrinsics=True
        )
        
        # Update reconstruction from optimized state
        self._update_reconstruction_from_state(optimized_state)
        
        # Report results
        stats = self.reconstruction.get_statistics()
        print(f"Global BA complete:")
        print(f"  Cameras: {stats['num_cameras']}")
        print(f"  Points: {stats['num_points']}")
        print(f"  Mean reprojection error: {stats['mean_reprojection_error']:.3f}")


    def _select_best_next_image(self, remaining_images: Set[str]) -> str:
        """Select best next image using pair selector"""
        if not remaining_images:
            return None
        
        existing_cameras = set(self.reconstruction.cameras.keys())
        
        print(f"\n=== SELECTING NEXT IMAGE ===")
        print(f"Evaluating {len(remaining_images)} candidate images...")
        
        # Use the pair selector to get next best cameras
        camera_candidates = self.pair_selector.get_next_cameras_to_add(
            existing_cameras, 
            max_new_cameras=min(5, len(remaining_images))
        )
        
        if not camera_candidates:
            print("No valid candidates found by pair selector")
            return None
        
        # Filter candidates to only include remaining images
        valid_candidates = [
            candidate for candidate in camera_candidates 
            if candidate['camera_id'] in remaining_images
        ]
        
        if not valid_candidates:
            print("No remaining images found in pair selector candidates")
            return None
        
        # Select the best candidate
        best_candidate = valid_candidates[0]  # Already sorted by score
        selected_image = best_candidate['camera_id']
        
        print(f"Selected: {selected_image}")
        print(f"  Score: {best_candidate['overall_score']:.3f}")
        print(f"  Connections: {best_candidate['num_connections']}")
        best_pair = best_candidate['best_pair']
        print(f"  Best pair: {best_pair['pair_key']} (score: {best_pair['score_result']['total_score']:.3f})")
        
        return selected_image



    def _assess_reconstruction_quality(self, reconstruction_state):
        """
        Assess final reconstruction quality with per-camera intrinsics
        """
        quality_metrics = {}
        
        cameras = reconstruction_state.get('cameras', {})
        points_3d = reconstruction_state.get('points_3d', {}).get('points_3d')
        
        # Basic metrics
        quality_metrics['num_cameras'] = len(cameras)
        quality_metrics['num_points'] = points_3d.shape[1] if points_3d is not None else 0
        
        # Per-camera intrinsics quality
        camera_intrinsics = {}
        focal_lengths = []
        
        for cam_id, camera_data in cameras.items():
            if 'K' in camera_data:
                K = camera_data['K']
                fx, fy = K[0,0], K[1,1]
                cx, cy = K[0,2], K[1,2]
                
                camera_intrinsics[cam_id] = {
                    'fx': fx, 'fy': fy, 
                    'cx': cx, 'cy': cy,
                    'aspect_ratio': fx/fy if fy != 0 else 1.0
                }
                focal_lengths.extend([fx, fy])
        
        quality_metrics['camera_intrinsics'] = camera_intrinsics
        
        # Focal length statistics across all cameras
        if focal_lengths:
            quality_metrics['focal_length_stats'] = {
                'mean': np.mean(focal_lengths),
                'std': np.std(focal_lengths),
                'min': np.min(focal_lengths),
                'max': np.max(focal_lengths),
                'range': np.max(focal_lengths) - np.min(focal_lengths)
            }
        
        # Reprojection error from optimization history
        opt_history = reconstruction_state.get('optimization_history', [])
        if opt_history:
            final_opt = opt_history[-1]
            quality_metrics['final_reprojection_error'] = final_opt.get('final_cost', 0)
            quality_metrics['optimization_improvement'] = (
                final_opt.get('initial_cost', 0) - final_opt.get('final_cost', 0)
            )
        
        return quality_metrics
    
    def _export_for_dense_reconstruction(self, output_directory):
        """Export - uses self.reconstruction directly"""
        print("\n" + "="*50)
        print("EXPORTING FOR DENSE RECONSTRUCTION")  
        print("="*50)
        
        # Prepare export data from internal reconstruction
        export_data = self._prepare_export_data_from_reconstruction()
        
        # Save as pickle
        pickle_file = os.path.join(output_directory, 'optimized_camera_poses.pkl')
        self._save_poses_pickle(export_data, pickle_file)
        
        # Additional formats
        self._export_additional_formats(export_data, output_directory)
        
        print(f"✅ Poses exported for dense reconstruction:")
        print(f"   Main file: {pickle_file}")
        
        return export_data

    def _save_poses_pickle(self, export_data, pickle_file):
        """Save poses in pickle format for dense reconstruction"""
        
        try:
            with open(pickle_file, 'wb') as f:
                pickle.dump(export_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"Saved optimized poses: {pickle_file}")
            
            # Verify the pickle file
            with open(pickle_file, 'rb') as f:
                test_load = pickle.load(f)
            
            print(f"✓ Pickle file verified - {len(test_load['camera_poses'])} cameras")
            
        except Exception as e:
            print(f"❌ Failed to save pickle file: {e}")
            raise

    def _export_additional_formats(self, export_data, output_directory):
        """Export additional formats for debugging and compatibility"""
        
        # 1. JSON format (human-readable)
        json_file = os.path.join(output_directory, 'camera_poses.json')
        json_data = self._convert_to_json_serializable(export_data)
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # 2. COLMAP format (industry standard)
        self._export_colmap_format(export_data, output_directory)
        
        # 3. Summary report
        report_file = os.path.join(output_directory, 'reconstruction_summary.txt')
        self._create_summary_report(export_data, report_file)


    
    def _find_correspondences_with_existing_3d(self, new_image: str) -> Dict[str, Any]:
        """Find 2D-3D correspondences using internal reconstruction"""
        
        # Check if reconstruction exists
        if self.reconstruction is None or len(self.reconstruction.cameras) == 0:
            print(f"No existing reconstruction to match against")
            return self._empty_correspondence_result()
        
        existing_images = set(self.reconstruction.cameras.keys())
        
        print(f"Finding correspondences for {new_image}...")
        
        # Get matches between new image and existing images
        matches_with_existing = self._get_matches_with_existing_images(
            new_image, existing_images
        )
        
        if not matches_with_existing:
            print(f"No matches found between {new_image} and existing images")
            return self._empty_correspondence_result()
        
        # Extract 2D-3D correspondences using Reconstruction
        correspondences = self._extract_2d_3d_correspondences_from_reconstruction(
            matches_with_existing, new_image
        )
        
        # Filter by quality
        filtered_correspondences = _filter_correspondences_by_quality(correspondences)
        
        # Format for PnP
        result = _format_for_pnp_solver(filtered_correspondences)
        
        print(f"Found {result['num_correspondences']} correspondences for {new_image}")
        if result['num_correspondences'] > 0:
            print(f"Average confidence: {np.mean(result['confidence_scores']):.3f}")
        
        return result

    @staticmethod
    def _empty_correspondence_result() -> Dict[str, Any]:
        """Return empty result when no correspondences found."""
        return {
            'points_3d': np.empty((0, 3)),
            'points_2d': np.empty((0, 2)),
            'point_ids': np.empty(0, dtype=int),
            'confidence_scores': np.empty(0),
            'match_sources': np.empty(0, dtype='U50'),
            'num_correspondences': 0
        }
    

    def _extract_2d_3d_correspondences_from_reconstruction(self, 
                                                        matches_with_existing: List[Dict],
                                                        new_image: str) -> List[Dict]:
        """Extract 2D-3D correspondences using the Reconstruction object"""
        correspondences = []
        
        for match_info in matches_with_existing:
            existing_image = match_info['existing_image']
            
            # Get camera from reconstruction
            if existing_image not in self.reconstruction.cameras:
                continue
            
            # Get observations for the existing camera
            existing_observations = self.reconstruction.get_camera_observations(existing_image)
            
            # Get point arrays for new and existing images
            new_points, existing_points = self._get_point_arrays_from_match(match_info)
            
            if len(new_points) == 0 or len(existing_points) == 0:
                continue
            
            # Create spatial index for faster matching
            observation_index = {}
            for obs in existing_observations:
                key = (round(obs.coords_2d[0], 1), round(obs.coords_2d[1], 1))
                observation_index[key] = obs
            
            # Find correspondences
            for i, (new_pt, existing_pt) in enumerate(zip(new_points, existing_points)):
                # Find matching observation
                matched_obs = self._find_observation_for_point(
                    existing_pt, observation_index, tolerance=3.0
                )
                
                if matched_obs is not None:
                    # Get the 3D point from reconstruction
                    point_3d = self.reconstruction.points[matched_obs.point_id]
                    
                    correspondence = {
                        'point_3d_id': matched_obs.point_id,
                        'point_3d': point_3d.coords,
                        'point_2d': new_pt,
                        'confidence': self._get_match_confidence(match_info['match_data'], i),
                        'source_image': existing_image
                    }
                    correspondences.append(correspondence)
        
        return correspondences
    
    def _get_point_arrays_from_match(self, match_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Get normalized point arrays for new and existing images from various match formats."""
        match_data = match_info['match_data']
        
        # Handle different match data formats
        if 'correspondences' in match_data:
            correspondences = match_data['correspondences'][0]
            pts1 = np.array([[corr[0], corr[1]] for corr in correspondences])
            pts2 = np.array([[corr[2], corr[3]] for corr in correspondences])
        else:
            pts1 = match_data.get('pts1')
            pts2 = match_data.get('pts2')
            
            if pts1 is not None:
                pts1 = np.array(pts1)
            if pts2 is not None:
                pts2 = np.array(pts2)
        
        # Check if we have valid points
        if pts1 is None or pts2 is None or len(pts1) == 0 or len(pts2) == 0:
            return np.array([]), np.array([])
        
        # Determine which is new vs existing based on candidate position
        if match_info['candidate_is_first']:
            new_points = self._normalize_point_array(pts1)
            existing_points = self._normalize_point_array(pts2)
        else:
            new_points = self._normalize_point_array(pts2)
            existing_points = self._normalize_point_array(pts1)
        
        return new_points, existing_points

    @staticmethod
    def _normalize_point_array(points) -> np.ndarray:
        """Normalize point arrays to (N, 2) format."""
        if points is None or len(points) == 0:
            return np.empty((0, 2))
        
        points = np.array(points)
        
        # Handle different input formats
        if points.ndim == 3 and points.shape[1] == 1:
            # Shape (N, 1, 2) -> (N, 2)
            points = points.squeeze(axis=1)
        elif points.ndim == 2 and points.shape[0] == 2 and points.shape[1] > 2:
            # Shape (2, N) -> (N, 2) - transpose
            points = points.T
        elif points.ndim == 1:
            # Invalid 1D array
            return np.empty((0, 2))
        elif points.ndim == 2 and points.shape[1] != 2:
            # Invalid shape like (N, 1) or (N, 3+)
            if points.shape[1] == 1:
                return np.empty((0, 2))  # Can't make 2D points from 1D
            else:
                # Take only first 2 columns if more than 2
                points = points[:, :2]
        
        # Ensure we have the right shape
        if points.ndim != 2 or points.shape[1] != 2:
            return np.empty((0, 2))
        
        return points

    def _find_observation_for_point(self, point_2d: np.ndarray, 
                                observation_index: Dict, 
                                tolerance: float = 3.0) -> Optional[Observation]:
        """Find observation matching a 2D point"""
        key = (round(point_2d[0], 1), round(point_2d[1], 1))
        
        # Quick lookup first
        if key in observation_index:
            return observation_index[key]
        
        # Fallback to exhaustive search within tolerance
        for obs_key, obs in observation_index.items():
            distance = np.linalg.norm(point_2d - obs.coords_2d)
            if distance < tolerance:
                return obs
        
        return None


    @staticmethod
    def _get_match_confidence(match_data: Dict, match_index: int) -> float:
        """Extract confidence score for a specific match."""
        if 'confidence_scores' in match_data:
            confidences = match_data['confidence_scores']
            if match_index < len(confidences):
                return confidences[match_index]
        elif 'distances' in match_data:
            distances = match_data['distances']
            if match_index < len(distances):
                max_dist = np.max(distances) if len(distances) > 0 else 1.0
                return 1.0 - (distances[match_index] / max_dist)
        elif 'scores' in match_data:
            scores = match_data['scores']
            if match_index < len(scores):
                return scores[match_index]
        
        return 0.7  # Default confidence

    def _validate_recovered_pose(self, R: np.ndarray, t: np.ndarray) -> bool:
        """Validate recovered pose from essential matrix"""
        # Check rotation matrix is valid
        if np.linalg.det(R) < 0:
            return False
        
        # Check baseline is reasonable
        baseline = np.linalg.norm(t)
        if baseline < 1e-6:
            return False
            
        return True

    def _fallback_circular_pose(self, anchor_R: np.ndarray, 
                            anchor_t: np.ndarray, 
                            anchor_K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fallback pose estimation using circular trajectory assumption"""
        # Random rotation around Y-axis
        theta = np.random.uniform(0, 2*np.pi)
        rotation_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        bootstrap_R = rotation_y @ anchor_R
        
        # Offset translation
        offset_distance = np.random.uniform(0.5, 2.0)
        offset_direction = np.random.uniform(-1, 1, 3)
        offset_direction = offset_direction / np.linalg.norm(offset_direction)
        
        bootstrap_t = anchor_t + offset_direction.reshape(-1, 1) * offset_distance
        bootstrap_K = anchor_K.copy()
        
        return bootstrap_R, bootstrap_t, bootstrap_K

    def _convert_to_json_serializable(self, data: Dict) -> Dict:
        """Convert numpy arrays to lists for JSON serialization"""
        # Implementation needed
        pass

    def _export_colmap_format(self, export_data: Dict, output_directory: str):
        """Export in COLMAP format"""
        # Implementation needed
        pass

    def _create_summary_report(self, export_data: Dict, report_file: str):
        """Create summary report"""
        # Implementation needed
        pass



@dataclass
class CorrespondenceFilterConfig:
    """Configuration for correspondence filtering"""
    min_confidence: float = 0.3              # Minimum match confidence
    max_reprojection_error: float = 5.0      # Maximum allowed reprojection error (pixels)
    min_triangulation_angle: float = 2.0     # Minimum triangulation angle (degrees)
    max_distance_to_camera: float = 100.0    # Maximum distance from camera center
    min_distance_to_camera: float = 0.1      # Minimum distance from camera center
    remove_outliers: bool = True             # Remove statistical outliers
    outlier_threshold: float = 2.0           # Standard deviations for outlier removal
    max_correspondences: int = 500           # Maximum number of correspondences to keep

def _filter_correspondences_by_quality(correspondences: List[Dict], 
                                     config: Optional[CorrespondenceFilterConfig] = None) -> List[Dict]:
    """
    Filter 2D-3D correspondences based on quality metrics.
    
    Args:
        correspondences: List of correspondence dictionaries
        config: Filtering configuration parameters
    
    Returns:
        List of filtered high-quality correspondences
    """
    
    if not correspondences:
        return []
    
    if config is None:
        config = CorrespondenceFilterConfig()
    
    print(f"  Filtering {len(correspondences)} correspondences...")
    
    filtered_correspondences = []
    
    # Step 1: Basic quality filtering
    for corr in correspondences:
        # Filter by confidence
        if corr['confidence'] < config.min_confidence:
            continue
        
        # Filter by 3D point distance (avoid points too close/far from cameras)
        point_3d = corr['point_3d']
        distance = np.linalg.norm(point_3d)
        
        if distance < config.min_distance_to_camera or distance > config.max_distance_to_camera:
            continue
        
        # Check for valid 2D coordinates
        point_2d = corr['point_2d']
        if not _is_valid_2d_point(point_2d):
            continue
        
        filtered_correspondences.append(corr)
    
    print(f"    After basic filtering: {len(filtered_correspondences)}")
    
    if len(filtered_correspondences) == 0:
        return []
    
    # Step 2: Remove statistical outliers based on confidence scores
    if config.remove_outliers and len(filtered_correspondences) > 10:
        filtered_correspondences = _remove_confidence_outliers(
            filtered_correspondences, config.outlier_threshold
        )
        print(f"    After outlier removal: {len(filtered_correspondences)}")
    
    # Step 3: Remove duplicate correspondences (same 3D point observed multiple times)
    filtered_correspondences = _remove_duplicate_correspondences(filtered_correspondences)
    print(f"    After duplicate removal: {len(filtered_correspondences)}")
    
    # Step 4: Limit number of correspondences (keep best ones)
    if len(filtered_correspondences) > config.max_correspondences:
        # Sort by confidence and keep top N
        filtered_correspondences.sort(key=lambda x: x['confidence'], reverse=True)
        filtered_correspondences = filtered_correspondences[:config.max_correspondences]
        print(f"    Limited to top {config.max_correspondences} correspondences")
    
    # Step 5: Ensure good geometric distribution
    filtered_correspondences = _ensure_geometric_distribution(filtered_correspondences)
    print(f"    After geometric distribution: {len(filtered_correspondences)}")
    
    return filtered_correspondences


def _is_valid_2d_point(point_2d: np.ndarray, 
                       min_coord: float = 0.0, 
                       max_coord: float = 10000.0) -> bool:
    """Check if 2D point has valid coordinates"""
    
    if len(point_2d) != 2:
        return False
    
    x, y = float(point_2d[0]), float(point_2d[1])
    
    # Check for NaN or infinite values
    if not (np.isfinite(x) and np.isfinite(y)):
        return False
    
    # Check reasonable coordinate range
    if not (min_coord <= x <= max_coord and min_coord <= y <= max_coord):
        return False
    
    return True


def _remove_confidence_outliers(correspondences: List[Dict], 
                               threshold: float = 2.0) -> List[Dict]:
    """Remove correspondences with outlier confidence scores"""
    
    if len(correspondences) <= 3:
        return correspondences
    
    # Extract confidence scores
    confidences = np.array([corr['confidence'] for corr in correspondences])
    
    # Calculate statistics
    mean_conf = np.mean(confidences)
    std_conf = np.std(confidences)
    
    if std_conf == 0:  # All confidences are the same
        return correspondences
    
    # Keep correspondences within threshold standard deviations
    filtered_correspondences = []
    for corr in correspondences:
        z_score = abs(corr['confidence'] - mean_conf) / std_conf
        if z_score <= threshold:
            filtered_correspondences.append(corr)
    
    return filtered_correspondences


def _remove_duplicate_correspondences(correspondences: List[Dict], 
                                    tolerance: float = 1e-6) -> List[Dict]:
    """Remove correspondences that refer to the same 3D point"""
    
    if len(correspondences) <= 1:
        return correspondences
    
    # Group correspondences by 3D point ID
    point_groups = {}
    
    for corr in correspondences:
        point_id = corr['point_3d_id']
        
        if point_id not in point_groups:
            point_groups[point_id] = []
        
        point_groups[point_id].append(corr)
    
    # For each 3D point, keep only the correspondence with highest confidence
    filtered_correspondences = []
    
    for point_id, group in point_groups.items():
        if len(group) == 1:
            filtered_correspondences.append(group[0])
        else:
            # Keep the one with highest confidence
            best_corr = max(group, key=lambda x: x['confidence'])
            filtered_correspondences.append(best_corr)
    
    return filtered_correspondences


def _ensure_geometric_distribution(correspondences: List[Dict], 
                                 min_spread_ratio: float = 0.3) -> List[Dict]:
    """Ensure correspondences are well distributed geometrically in the image"""
    
    if len(correspondences) <= 10:
        return correspondences  # Too few points to worry about distribution
    
    # Extract 2D points
    points_2d = np.array([corr['point_2d'] for corr in correspondences])
    
    # Calculate image bounds
    min_x, min_y = np.min(points_2d, axis=0)
    max_x, max_y = np.max(points_2d, axis=0)
    
    # Calculate current spread
    spread_x = max_x - min_x
    spread_y = max_y - min_y
    
    # Check if distribution is reasonable
    image_diagonal = np.sqrt(spread_x**2 + spread_y**2)
    
    if image_diagonal < 100:  # Points are too clustered
        print("    Warning: Correspondences are clustered in small region")
    
    # For now, just return all correspondences
    # In advanced implementation, you could implement spatial binning
    # to ensure more uniform distribution
    
    return correspondences


def _format_for_pnp_solver(correspondences: List[Dict]) -> Dict[str, Any]:
    """
    Format filtered correspondences for PnP solver input.
    
    Args:
        correspondences: List of filtered correspondence dictionaries
    
    Returns:
        Dictionary formatted for PnP solver with all required fields
    """
    
    if not correspondences:
        return _create_empty_pnp_result()
    
    print(f"  Formatting {len(correspondences)} correspondences for PnP...")
    
    #Extract arrays
    points_3d = []
    points_2d = []
    point_ids = []
    confidence_scores = []
    match_sources = []
    
    for corr in correspondences:
        #3D world point
        points_3d.append(corr['point_3d'])
        
        #2D image point
        points_2d.append(corr['point_2d'])
        
        #Point ID in reconstruction
        point_ids.append(corr['point_3d_id'])
        
        #Match confidence
        confidence_scores.append(corr['confidence'])
        
        #Source information (which existing image this correspondence came from)
        match_sources.append(corr.get('source_image', 'unknown'))
    
    #Convert to numpy arrays
    points_3d = np.array(points_3d, dtype=np.float32)  # (N, 3)
    points_2d = np.array(points_2d, dtype=np.float32)  # (N, 2)
    point_ids = np.array(point_ids, dtype=int)         # (N,)
    confidence_scores = np.array(confidence_scores, dtype=np.float32)  # (N,)
    match_sources = np.array(match_sources, dtype='U50')  # (N,)
    
    #Validate array shapes
    n_correspondences = len(points_3d)
    
    assert points_3d.shape == (n_correspondences, 3), f"points_3d shape mismatch: {points_3d.shape}"
    assert points_2d.shape == (n_correspondences, 2), f"points_2d shape mismatch: {points_2d.shape}"
    assert len(point_ids) == n_correspondences, f"point_ids length mismatch: {len(point_ids)}"
    assert len(confidence_scores) == n_correspondences, f"confidence_scores length mismatch: {len(confidence_scores)}"
    
    #Calculate quality statistics
    quality_stats = _calculate_correspondence_quality_stats(
        points_3d, points_2d, confidence_scores
    )
    
    #Create result dictionary
    result = {
        #Core data for PnP solver
        'points_3d': points_3d,
        'points_2d': points_2d,
        'point_ids': point_ids,
        'confidence_scores': confidence_scores,
        'num_correspondences': n_correspondences,
        
        #Additional metadata
        'match_sources': match_sources,
        'quality_stats': quality_stats,
        
        #Summary information
        'mean_confidence': np.mean(confidence_scores) if n_correspondences > 0 else 0.0,
        'min_confidence': np.min(confidence_scores) if n_correspondences > 0 else 0.0,
        'max_confidence': np.max(confidence_scores) if n_correspondences > 0 else 0.0,
        'std_confidence': np.std(confidence_scores) if n_correspondences > 0 else 0.0,
    }
    
    print(f"    Final correspondences: {n_correspondences}")
    print(f"    Confidence range: [{result['min_confidence']:.3f}, {result['max_confidence']:.3f}]")
    print(f"    Mean confidence: {result['mean_confidence']:.3f}")
    
    return result


def _create_empty_pnp_result() -> Dict[str, Any]:
    """Create empty result when no correspondences are available"""
    
    return {
        'points_3d': np.empty((0, 3), dtype=np.float32),
        'points_2d': np.empty((0, 2), dtype=np.float32),
        'point_ids': np.empty(0, dtype=int),
        'confidence_scores': np.empty(0, dtype=np.float32),
        'num_correspondences': 0,
        'match_sources': np.empty(0, dtype='U50'),
        'quality_stats': {},
        'mean_confidence': 0.0,
        'min_confidence': 0.0,
        'max_confidence': 0.0,
        'std_confidence': 0.0,
    }


def _calculate_correspondence_quality_stats(points_3d: np.ndarray, 
                                            points_2d: np.ndarray, 
                                            confidence_scores: np.ndarray) -> Dict[str, Any]:
    """Calculate quality statistics for the correspondence set"""
    
    if len(points_3d) == 0:
        return {}
    
    #3D point statistics
    point_distances = np.linalg.norm(points_3d, axis=1)
    
    #2D point distribution
    points_2d_center = np.mean(points_2d, axis=0)
    distances_from_center = np.linalg.norm(points_2d - points_2d_center, axis=1)
    
    #Geometric spread
    min_2d = np.min(points_2d, axis=0)
    max_2d = np.max(points_2d, axis=0)
    image_span = max_2d - min_2d
    
    stats = {
        #3D point statistics
        'mean_3d_distance': float(np.mean(point_distances)),
        'std_3d_distance': float(np.std(point_distances)),
        'min_3d_distance': float(np.min(point_distances)),
        'max_3d_distance': float(np.max(point_distances)),
        
        #2D point distribution
        'mean_distance_from_center': float(np.mean(distances_from_center)),
        'std_distance_from_center': float(np.std(distances_from_center)),
        'image_span_x': float(image_span[0]) if len(image_span) > 0 else 0.0,
        'image_span_y': float(image_span[1]) if len(image_span) > 1 else 0.0,
        
        #Confidence statistics
        'confidence_uniformity': float(1.0 / (1.0 + np.std(confidence_scores))),  # Higher = more uniform
        
        #Overall quality assessment
        'geometric_diversity': float(_assess_geometric_diversity(points_2d)),
        'depth_diversity': float(_assess_depth_diversity(points_3d)),
    }
    
    return stats


def _assess_geometric_diversity(points_2d: np.ndarray) -> float:
    """Assess how well distributed the 2D points are geometrically"""
    
    if len(points_2d) < 4:
        return 0.5  # Not enough points to assess
    
    # Calculate convex hull area as fraction of bounding box area
    try:
        from scipy.spatial import ConvexHull
        
        if len(points_2d) >= 3:
            hull = ConvexHull(points_2d)
            hull_area = hull.volume  # In 2D, volume is area
            
            # Calculate bounding box area
            min_coords = np.min(points_2d, axis=0)
            max_coords = np.max(points_2d, axis=0)
            bbox_area = np.prod(max_coords - min_coords)
            
            if bbox_area > 0:
                diversity = hull_area / bbox_area
                return min(1.0, diversity)
    
    except ImportError:
        # Fallback: use standard deviation of coordinates
        std_coords = np.std(points_2d, axis=0)
        diversity = np.mean(std_coords) / 1000.0  # Normalize assuming ~1000px images
        return min(1.0, diversity)
    
    return 0.5  # Default moderate diversity


def _assess_depth_diversity(points_3d: np.ndarray) -> float:
    """Assess how well distributed the 3D points are in depth"""
    
    if len(points_3d) < 2:
        return 0.5
    
    # Calculate distances from origin (camera center during initialization)
    distances = np.linalg.norm(points_3d, axis=1)
    
    # Depth diversity based on coefficient of variation
    if np.mean(distances) > 0:
        depth_diversity = np.std(distances) / np.mean(distances)
        return min(1.0, depth_diversity)
    
    return 0.5


# # Integration helper functions
# def validate_correspondence_format(correspondences_result: Dict[str, Any]) -> bool:
#     """Validate that correspondence result has correct format for PnP"""
    
#     required_fields = ['points_3d', 'points_2d', 'point_ids', 'confidence_scores', 'num_correspondences']
    
#     for field in required_fields:
#         if field not in correspondences_result:
#             print(f"Missing required field: {field}")
#             return False
    
#     # Check array shapes
#     n = correspondences_result['num_correspondences']
    
#     if correspondences_result['points_3d'].shape != (n, 3):
#         print(f"Invalid points_3d shape: {correspondences_result['points_3d'].shape}, expected ({n}, 3)")
#         return False
    
#     if correspondences_result['points_2d'].shape != (n, 2):
#         print(f"Invalid points_2d shape: {correspondences_result['points_2d'].shape}, expected ({n}, 2)")
#         return False
    
#     if len(correspondences_result['point_ids']) != n:
#         print(f"Invalid point_ids length: {len(correspondences_result['point_ids'])}, expected {n}")
#         return False
    
#     if len(correspondences_result['confidence_scores']) != n:
#         print(f"Invalid confidence_scores length: {len(correspondences_result['confidence_scores'])}, expected {n}")
#         return False
    
#     return True


# def print_correspondence_summary(correspondences_result: Dict[str, Any]) -> None:
#     """Print a detailed summary of correspondence finding results"""
    
#     print(f"\n{'='*50}")
#     print("CORRESPONDENCE SUMMARY")
#     print(f"{'='*50}")
    
#     n = correspondences_result['num_correspondences']
#     print(f"Total correspondences: {n}")
    
#     if n == 0:
#         print("❌ No correspondences found!")
#         return
    
#     print(f"Confidence: mean={correspondences_result['mean_confidence']:.3f}, "
#           f"std={correspondences_result['std_confidence']:.3f}")
#     print(f"Confidence range: [{correspondences_result['min_confidence']:.3f}, "
#           f"{correspondences_result['max_confidence']:.3f}]")
    
#     if 'quality_stats' in correspondences_result and correspondences_result['quality_stats']:
#         stats = correspondences_result['quality_stats']
#         print(f"\nQuality Assessment:")
#         print(f"  Geometric diversity: {stats.get('geometric_diversity', 0):.3f}")
#         print(f"  Depth diversity: {stats.get('depth_diversity', 0):.3f}")
#         print(f"  Image span: {stats.get('image_span_x', 0):.1f} × {stats.get('image_span_y', 0):.1f} pixels")
#         print(f"  3D distance range: [{stats.get('min_3d_distance', 0):.2f}, {stats.get('max_3d_distance', 0):.2f}]")
    
#     # Assessment
#     if n >= 20 and correspondences_result['mean_confidence'] > 0.6:
#         print("✅ Excellent correspondence quality")
#     elif n >= 10 and correspondences_result['mean_confidence'] > 0.4:
#         print("🟡 Good correspondence quality")
#     elif n >= 6:
#         print("⚠️ Marginal correspondence quality")
#     else:
#         print("❌ Poor correspondence quality - PnP may fail")
    
#     print(f"{'='*50}")




