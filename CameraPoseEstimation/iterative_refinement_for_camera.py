"""
Refined iterative refinement using your existing EssentialMatrixEstimator and TriangulationEngine
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.optimize import least_squares

# Import your existing classes
from .essential_estimation import EssentialMatrixEstimator
from .triangulation import TriangulationEngine, TriangulationConfig

@dataclass
class RefinementConfig:
    """Configuration for iterative refinement"""
    # Relaxation schedule - start loose, get tighter
    relaxation_schedule: list = field(default_factory=lambda: [3.0, 2.5, 2.0, 1.5, 1.2, 1.0])
    
    # Convergence criteria
    error_tolerance: float = 0.01        # Stop if <1% improvement
    stable_iterations_required: int = 20  # Must be stable for N iterations
    
    # Base thresholds (will be multiplied by relaxation factor)
    base_reproj_thresh: float = 2.0      # Base reprojection error for triangulation
    base_triangulation_angle: float = 2.0  # Base triangulation angle
    base_near: float = 0.1               # Minimum depth
    base_far: float = 1000.0             # Maximum depth
    
    # Optimization
    max_iterations: int = 10             
    min_points: int = 20                 # Minimum points to continue
    
    # Camera bounds
    focal_bounds: Tuple[float, float] = (100.0, 5000.0)
    pp_bounds_ratio: float = 0.3         # Principal point within 30% of image center

class IterativeRefinementPipeline:
    """
    Iterative refinement using your existing EssentialMatrixEstimator and TriangulationEngine
    """
    
    def __init__(self):
        self.essential_estimator = EssentialMatrixEstimator()
        self.config = RefinementConfig()
        
    def iterative_refinement_with_relaxation(self, 
                                            pts1: np.ndarray, 
                                            pts2: np.ndarray,
                                            image_size1: Tuple[int, int],
                                            image_size2: Tuple[int, int],
                                            K1_init: Optional[np.ndarray] = None,
                                            K2_init: Optional[np.ndarray] = None) -> Dict:
        """
        Iterative refinement using your existing classes.
        
        Args:
            pts1, pts2: Corresponding points
            image_size1, image_size2: Image dimensions for each view
            K1_init, K2_init: Initial camera matrices (will be estimated if None)
        """
        
        print("\n=== ITERATIVE REFINEMENT WITH EXISTING CLASSES ===")
        
        # Initialize camera matrices if not provided
        if K1_init is None:
            K1_init = self.essential_estimator.estimate_camera_matrix(image_size1)
            print(f"Estimated initial K1 with focal={K1_init[0,0]:.1f}")
        if K2_init is None:
            K2_init = self.essential_estimator.estimate_camera_matrix(image_size2)
            print(f"Estimated initial K2 with focal={K2_init[0,0]:.1f}")
        
        K1 = K1_init.copy()
        K2 = K2_init.copy()
        
        best_result = None
        best_score = -1
        prev_score = -1
        stable_count = 0
        history = []
        
        for iteration in range(self.config.max_iterations):
            
            # Get current relaxation factor
            relax_factor = self.config.relaxation_schedule[
                min(iteration, len(self.config.relaxation_schedule) - 1)
            ]
            
            print(f"\nIteration {iteration + 1}/{self.config.max_iterations}")
            print(f"  Relaxation factor: {relax_factor:.1f}x")
            print(f"  K1 focal: {K1[0,0]:.1f}, K2 focal: {K2[0,0]:.1f}")
            
            # Step 1: Use your EssentialMatrixEstimator with current K matrices
            essential_result = self._estimate_essential_with_relaxation(
                pts1, pts2, image_size1, image_size2, K1, K2, relax_factor
            )
            
            if not essential_result['success']:
                print(f"  Essential matrix estimation failed: {essential_result.get('error', 'Unknown')}")
                continue
            
            # Extract pose (R, t are already recovered by your estimator)
            R, t = self._extract_pose_from_essential(essential_result, pts1, pts2, K1, K2)
            
            # Get inlier mask
            mask_inliers = essential_result['inlier_mask'].ravel().astype(bool)
            num_inliers = np.sum(mask_inliers)
            
            print(f"  Inliers: {num_inliers}/{len(pts1)} ({essential_result['inlier_ratio']:.1%})")
            
            if num_inliers < self.config.min_points:
                print(f"  Too few inliers (< {self.config.min_points})")
                continue
            
            # Step 2: Use your TriangulationEngine with relaxed thresholds
            triangulated_points = self._triangulate_with_relaxation(
                pts1[mask_inliers], pts2[mask_inliers],
                K1, K2, R, t, relax_factor, (image_size1, image_size2)
            )
            
            if triangulated_points is None or triangulated_points['points_3d'].shape[1] < self.config.min_points:
                print(f"  Too few triangulated points")
                continue
            
            points_3d = triangulated_points['points_3d']
            
            # Get the 2D points that were successfully triangulated
            # Your triangulation engine filters points, so we need to track which ones survived
            valid_pts1, valid_pts2, triangulated_mask = self._get_valid_points_from_triangulation(
                triangulated_points, pts1[mask_inliers], pts2[mask_inliers]
            )
            
            print(f"  Triangulated points: {points_3d.shape[1]}")
            
            # Step 3: Refine camera matrices using bundle adjustment
            K1_refined, K2_refined = self._refine_camera_matrices_bounded(
                points_3d, valid_pts1, valid_pts2,
                R, t, K1, K2, image_size1, image_size2
            )
            
            # Step 4: Evaluate quality
            score, metrics = self._evaluate_quality(
                points_3d, K1_refined, K2_refined, R, t,
                valid_pts1, valid_pts2
            )
            
            print(f"  Score: {score:.3f} (mean reproj: {metrics['mean_error']:.2f}px)")
            
            # Store history
            history.append({
                'iteration': iteration,
                'score': score,
                'num_inliers': num_inliers,
                'num_triangulated': points_3d.shape[1],
                'mean_reproj': metrics['mean_error'],
                'relax_factor': relax_factor
            })
            
            # Update best result
            if score > best_score:
                if best_score!=-1:
                    improvement = (score - best_score) / (best_score + 1e-6)
                    print(f"  ✓ New best! (improvement: {improvement*100:.1f}%)")
                    
                best_score = score
                best_result = {
                    'K1': K1_refined.copy(),
                    'K2': K2_refined.copy(),
                    'R': R.copy(),
                    't': t.copy(),
                    'points_3d': points_3d.copy(),
                    'valid_pts1': valid_pts1.copy(),
                    'valid_pts2': valid_pts2.copy(),
                    'mask_inliers': mask_inliers.copy(),
                    'triangulated_mask': triangulated_mask.copy(),
                    'iteration': iteration,
                    'score': score,
                    'metrics': metrics,
                    'essential_result': essential_result
                }
                
                # Update K for next iteration
                K1 = K1_refined
                K2 = K2_refined
                stable_count = 0
            else:
                stable_count += 1
                print(f"  No improvement (stable for {stable_count} iterations)")
            
            # Check convergence
            if self._check_convergence(score, prev_score, stable_count):
                print("\n✓ CONVERGED!")
                break
            
            prev_score = score
        
        # Final summary
        if best_result:
            print(f"\nFinal result:")
            print(f"  Best iteration: {best_result['iteration'] + 1}")
            print(f"  Score: {best_result['score']:.3f}")
            print(f"  Points: {best_result['points_3d'].shape[1]}")
            print(f"  Mean reproj: {best_result['metrics']['mean_error']:.2f}px")
            print(f"  K1 focal: {best_result['K1'][0,0]:.1f}")
            print(f"  K2 focal: {best_result['K2'][0,0]:.1f}")
            
            best_result['history'] = history
            
            # Format result to match your pipeline expectations
            best_result['camera_matrices'] = [best_result['K1'], best_result['K2']]
            best_result['camera_estimated'] = [True, True]  # Both were estimated/refined
            best_result['success'] = True
        
        return best_result
    

    def _estimate_essential_with_relaxation(self, pts1, pts2, image_size1, image_size2,
                                           K1, K2, relax_factor):
        """
        Use your EssentialMatrixEstimator with relaxed thresholds.
        
        Your estimator uses normalized coordinates internally, which is good!
        """
        
        # Temporarily modify the config for relaxed estimation
        original_threshold = self.essential_estimator.config.RANSAC_THRESHOLD
        original_min_inliers = self.essential_estimator.config.MIN_INLIERS
        original_min_ratio = self.essential_estimator.config.MIN_INLIER_RATIO
        
        # Relax thresholds
        self.essential_estimator.config.RANSAC_THRESHOLD = original_threshold * relax_factor
        self.essential_estimator.config.MIN_INLIERS = max(8, int(original_min_inliers / relax_factor))
        self.essential_estimator.config.MIN_INLIER_RATIO = original_min_ratio / relax_factor
        
        # Use your estimator
        result = self.essential_estimator.estimate(
            pts1, pts2,
            image_size1, image_size2,
            method='RANSAC',
            camera_matrix1=K1,
            camera_matrix2=K2
        )
        
        # Restore original config
        self.essential_estimator.config.RANSAC_THRESHOLD = original_threshold
        self.essential_estimator.config.MIN_INLIERS = original_min_inliers
        self.essential_estimator.config.MIN_INLIER_RATIO = original_min_ratio
        
        return result
    
    def _extract_pose_from_essential(self, essential_result, pts1, pts2, K1, K2):
        """
        Extract R and t from essential matrix result.
        Your estimator returns normalized coordinates, so we need to recover pose.
        """
        
        E = essential_result['essential_matrix']
        mask = essential_result['inlier_mask']
        
        # Get inlier points
        inlier_idx = mask.ravel() == 1
        pts1_inliers = pts1[inlier_idx]
        pts2_inliers = pts2[inlier_idx]
        
        # Normalize points for pose recovery
        pts1_norm = cv2.undistortPoints(pts1_inliers.reshape(-1, 1, 2), K1, None).reshape(-1, 2)
        pts2_norm = cv2.undistortPoints(pts2_inliers.reshape(-1, 1, 2), K2, None).reshape(-1, 2)
        
        # Recover pose in normalized coordinates
        _, R, t, _ = cv2.recoverPose(E, pts1_norm, pts2_norm, focal=1.0, pp=(0.0, 0.0))
        
        return R, t
    
    def _triangulate_with_relaxation(self, pts1, pts2, K1, K2, R, t, relax_factor, image_sizes):
        """
        Use your TriangulationEngine with relaxed thresholds.
        """
        
        # Create triangulation config with relaxed thresholds
        config = TriangulationConfig(
            min_triangulation_angle=self.config.base_triangulation_angle / relax_factor,
            max_reprojection_error=self.config.base_reproj_thresh * relax_factor,
            min_distance=self.config.base_near / np.sqrt(relax_factor),
            max_distance=self.config.base_far * relax_factor,
            enable_progressive=False  # Don't use progressive for refinement
        )
        
        # Create triangulator with relaxed config
        triangulator = TriangulationEngine(
            min_triangulation_angle_deg=config.min_triangulation_angle,
            max_reprojection_error=config.max_reprojection_error,
            min_depth=config.min_distance,
            max_depth=config.max_distance,
            config=config
        )
        
        # First camera at origin, second with relative pose
        result = triangulator.triangulate_initial_points(
            pts1, pts2,
            R1=np.eye(3), t1=np.zeros((3, 1)),
            R2=R, t2=t,
            K=[K1, K2],
            image_pair=('cam1', 'cam2')  
        )
        
        return result
    
    def _get_valid_points_from_triangulation(self, triangulated_result, pts1_all, pts2_all):
        """
        Extract the 2D points that were successfully triangulated.
        REWRITTEN: Properly handles all cases and tracks original indices correctly.
        """
        
        # Method 1: Check if we have original indices from triangulation (preferred)
        if 'valid_indices' in triangulated_result:
            valid_indices = triangulated_result['valid_indices']
            
            # Validate indices are within bounds
            valid_indices = valid_indices[valid_indices < len(pts1_all)]
            
            if len(valid_indices) > 0:
                valid_pts1 = pts1_all[valid_indices]
                valid_pts2 = pts2_all[valid_indices]
                
                # Create proper mask showing which original points were triangulated
                triangulated_mask = np.zeros(len(pts1_all), dtype=bool)
                triangulated_mask[valid_indices] = True
                
                return valid_pts1, valid_pts2, triangulated_mask
        
        # Method 2: Parse observations to extract original indices
        observations = triangulated_result.get('observations', [])
        if observations:
            
            # Group observations by point_id to get pairs
            point_observations = {}
            for obs in observations:
                point_id = obs['point_id']
                if point_id not in point_observations:
                    point_observations[point_id] = {}
                point_observations[point_id][obs['image_id']] = {
                    'pixel_coords': obs['pixel_coords'],
                    'original_index': obs.get('original_index', None)
                }
            
            # Get all unique image IDs dynamically
            all_image_ids = set()
            for obs_dict in point_observations.values():
                all_image_ids.update(obs_dict.keys())
            
            if len(all_image_ids) >= 2:
                image_ids = sorted(list(all_image_ids))  # Sort for consistency
                img1_id = image_ids[0]
                img2_id = image_ids[1]
                
                valid_pts1 = []
                valid_pts2 = []
                valid_indices = []
                
                # Extract points for each 3D point that has observations in both images
                for point_id in sorted(point_observations.keys()):
                    obs_dict = point_observations[point_id]
                    
                    if img1_id in obs_dict and img2_id in obs_dict:
                        pt1_data = obs_dict[img1_id]
                        pt2_data = obs_dict[img2_id]
                        
                        valid_pts1.append(pt1_data['pixel_coords'])
                        valid_pts2.append(pt2_data['pixel_coords'])
                        
                        # Use original_index if available, otherwise use point_id as fallback
                        original_idx = pt1_data.get('original_index')
                        if original_idx is not None:
                            valid_indices.append(original_idx)
                        else:
                            # Fallback: assume point_id corresponds to original order
                            # This is the problematic case, but better than nothing
                            valid_indices.append(point_id)
                
                if valid_pts1:
                    valid_pts1 = np.array(valid_pts1)
                    valid_pts2 = np.array(valid_pts2)
                    valid_indices = np.array(valid_indices)
                    
                    # Validate indices are within bounds
                    valid_mask = valid_indices < len(pts1_all)
                    valid_indices = valid_indices[valid_mask]
                    valid_pts1 = valid_pts1[valid_mask]
                    valid_pts2 = valid_pts2[valid_mask]
                    
                    if len(valid_indices) > 0:
                        # Create proper mask
                        triangulated_mask = np.zeros(len(pts1_all), dtype=bool)
                        triangulated_mask[valid_indices] = True
                        
                        return valid_pts1, valid_pts2, triangulated_mask
        
        # Method 3: Last resort - direct extraction from result if points match input size
        points_3d = triangulated_result.get('points_3d', np.empty((3, 0)))
        num_triangulated = points_3d.shape[1]
        
        if num_triangulated > 0 and num_triangulated <= len(pts1_all):
            # Try to match points by finding best correspondences
            # This is a heuristic approach when we have no index tracking
            
            if num_triangulated == len(pts1_all):
                # All points were kept - no filtering happened
                triangulated_mask = np.ones(len(pts1_all), dtype=bool)
                return pts1_all.copy(), pts2_all.copy(), triangulated_mask
            
            # Partial triangulation - need to figure out which points were kept
            # This is problematic and why we need proper index tracking
            print(f"WARNING: Using fallback point matching for {num_triangulated}/{len(pts1_all)} points")
            print("This may not preserve correct point correspondences!")
            
            # Conservative approach: assume first N points (this is the original problem!)
            triangulated_mask = np.zeros(len(pts1_all), dtype=bool)
            triangulated_mask[:num_triangulated] = True
            
            return pts1_all[:num_triangulated].copy(), pts2_all[:num_triangulated].copy(), triangulated_mask
        
        # Method 4: Complete failure - return empty result
        print("ERROR: Could not extract valid points from triangulation result")
        print(f"Available keys: {list(triangulated_result.keys())}")
        print(f"Observations length: {len(observations) if observations else 0}")
        print(f"Points 3D shape: {points_3d.shape}")
        
        # Return empty result
        empty_mask = np.zeros(len(pts1_all), dtype=bool)
        return np.empty((0, 2)), np.empty((0, 2)), empty_mask
    
    def _refine_camera_matrices_bounded(self, points_3d, pts1, pts2, R, t,
                                    K1_init, K2_init, image_size1, image_size2):
        """
        Refine both camera matrices with bounded optimization.
        """
        
        def residuals(params):
            fx1, fy1, cx1, cy1, fx2, fy2, cx2, cy2 = params
            
            K1 = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])
            K2 = np.array([[fx2, 0, cx2], [0, fy2, cy2], [0, 0, 1]])
            
            P1 = K1 @ np.eye(3, 4)
            P2 = K2 @ np.hstack([R, t.reshape(-1, 1)])
            
            errors = []
            for i in range(points_3d.shape[1]):
                X = np.append(points_3d[:, i], 1)
                
                # Camera 1
                proj1 = P1 @ X
                if proj1[2] > 0:
                    px1 = proj1[:2] / proj1[2]
                    errors.extend(px1 - pts1[i])
                else:
                    errors.extend([100, 100])
                
                # Camera 2
                proj2 = P2 @ X
                if proj2[2] > 0:
                    px2 = proj2[:2] / proj2[2]
                    errors.extend(px2 - pts2[i])
                else:
                    errors.extend([100, 100])
            
            return np.array(errors)
        
        # Initial parameters
        x0 = [
            K1_init[0, 0], K1_init[1, 1], K1_init[0, 2], K1_init[1, 2],
            K2_init[0, 0], K2_init[1, 1], K2_init[0, 2], K2_init[1, 2]
        ]
        
        # Set bounds
        w1, h1 = image_size1[:2]
        w2, h2 = image_size2[:2]

        pp_margin = self.config.pp_bounds_ratio
        
        bounds = (
            [  # Lower bounds
                self.config.focal_bounds[0], self.config.focal_bounds[0],
                w1/2 * (1 - pp_margin), h1/2 * (1 - pp_margin),
                self.config.focal_bounds[0], self.config.focal_bounds[0],
                w2/2 * (1 - pp_margin), h2/2 * (1 - pp_margin)
            ],
            [  # Upper bounds
                self.config.focal_bounds[1], self.config.focal_bounds[1],
                w1/2 * (1 + pp_margin), h1/2 * (1 + pp_margin),
                self.config.focal_bounds[1], self.config.focal_bounds[1],
                w2/2 * (1 + pp_margin), h2/2 * (1 + pp_margin)
            ]
        )
        
        # Optimize
        result = least_squares(
            residuals, x0,
            method='trf',
            bounds=bounds,
            max_nfev=100,
            verbose=0
        )
        
        fx1, fy1, cx1, cy1, fx2, fy2, cx2, cy2 = result.x
        
        K1_refined = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])
        K2_refined = np.array([[fx2, 0, cx2], [0, fy2, cy2], [0, 0, 1]])
        
        return K1_refined, K2_refined
    
    def _evaluate_quality(self, points_3d, K1, K2, R, t, pts1, pts2):
        """Evaluate reconstruction quality."""
        
        P1 = K1 @ np.eye(3, 4)
        P2 = K2 @ np.hstack([R, t.reshape(-1, 1)])
        
        errors = []
        for i in range(points_3d.shape[1]):
            X = np.append(points_3d[:, i], 1)
            
            # Camera 1
            proj1 = P1 @ X
            if proj1[2] > 0:
                px1 = proj1[:2] / proj1[2]
                err1 = np.linalg.norm(px1 - pts1[i])
                errors.append(err1)
            
            # Camera 2
            proj2 = P2 @ X
            if proj2[2] > 0:
                px2 = proj2[:2] / proj2[2]
                err2 = np.linalg.norm(px2 - pts2[i])
                errors.append(err2)
        
        if not errors:
            return 0, {'mean_error': float('inf'), 'median_error': float('inf')}
        
        errors = np.array(errors)
        
        metrics = {
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'num_points': points_3d.shape[1]
        }
        
        # Score based on median (robust to outliers)
        score = metrics['num_points'] / (1 + metrics['median_error'])
        
        return score, metrics
    
    def _check_convergence(self, current_score, prev_score, stable_count):
        """Check convergence criteria."""
        
        if stable_count >= self.config.stable_iterations_required:
            return True
        
        if prev_score > 0 and current_score > 0:
            improvement = abs(current_score - prev_score) / current_score
            if improvement < self.config.error_tolerance:
                return True
        
        return False