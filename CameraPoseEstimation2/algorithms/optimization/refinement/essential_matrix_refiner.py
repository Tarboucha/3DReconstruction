import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy.optimize import least_squares

from CameraPoseEstimation2.algorithms.geometry.essential import EssentialMatrixEstimator
from CameraPoseEstimation2.algorithms.geometry.triangulation import TriangulationEngine, TriangulationConfig
from CameraPoseEstimation2.core.interfaces import OptimizationResult, OptimizationStatus


@dataclass
class EssentialMatrixRefinerConfig:
    """Configuration for essential matrix iterative refinement"""
    
    # Relaxation schedule - start loose, get tighter
    relaxation_schedule: list = field(default_factory=lambda: [3.0, 2.5, 2.0, 1.5, 1.2, 1.0])
    
    # Convergence criteria
    error_tolerance: float = 0.01              # Stop if <1% improvement
    stable_iterations_required: int = 2        # Must be stable for N iterations
    
    # Base thresholds (will be multiplied by relaxation factor)
    base_reproj_thresh: float = 2.0            # Base reprojection error
    base_triangulation_angle: float = 2.0      # Base triangulation angle (degrees)
    base_near: float = 0.1                     # Minimum depth
    base_far: float = 1000.0                   # Maximum depth
    
    # Optimization
    max_iterations: int = 10                   # Max refinement iterations
    min_points: int = 20                       # Minimum points to continue
    
    # Camera bounds
    focal_bounds: Tuple[float, float] = (100.0, 5000.0)
    pp_bounds_ratio: float = 0.3               # Principal point within 30% of center


class EssentialMatrixRefiner:
    """
    Iterative refinement of essential matrix estimation.
    
    This refiner progressively relaxes triangulation thresholds to recover
    more 3D points while simultaneously optimizing camera intrinsics and pose.
    
    Process:
    1. Start with initial K1, K2 estimates
    2. For each relaxation factor:
        a. Triangulate points with current thresholds
        b. Optimize camera parameters
        c. Update K1, K2, R, t
        d. Check convergence
    3. Return refined parameters and 3D points
    """
    
    def __init__(self, config: Optional[EssentialMatrixRefinerConfig] = None):
        """
        Initialize refiner.
        
        Args:
            config: Refinement configuration
        """
        self.config = config or EssentialMatrixRefinerConfig()
        self.essential_estimator = EssentialMatrixEstimator()
    
    def refine(self,
               pts1: np.ndarray,
               pts2: np.ndarray,
               K1_init: np.ndarray,
               K2_init: np.ndarray,
               image_size1: Tuple[int, int],
               image_size2: Tuple[int, int]) -> Dict:
        """
        Iteratively refine essential matrix estimation.
        
        Args:
            pts1: Points in first image (N, 2)
            pts2: Points in second image (N, 2)
            K1_init: Initial intrinsics for camera 1
            K2_init: Initial intrinsics for camera 2
            image_size1: Image 1 dimensions (width, height)
            image_size2: Image 2 dimensions (width, height)
            
        Returns:
            Dictionary with refined parameters:
                - success: bool
                - K1, K2: Refined intrinsics
                - R, t: Refined pose
                - points_3d: Triangulated 3D points
                - valid_pts1, valid_pts2: Valid 2D points
                - num_iterations: Number of iterations performed
        """
        print(f"Starting essential matrix iterative refinement...")
        print(f"  Input points: {len(pts1)}")
        print(f"  Relaxation schedule: {self.config.relaxation_schedule}")
        
        # Initialize
        K1_current = K1_init.copy()
        K2_current = K2_init.copy()
        
        best_result = None
        best_num_points = 0
        
        stable_count = 0
        prev_error = float('inf')
        
        # Iterate through relaxation schedule
        for iteration, relax_factor in enumerate(self.config.relaxation_schedule):
            print(f"\n--- Iteration {iteration + 1}/{len(self.config.relaxation_schedule)} "
                  f"(relaxation={relax_factor:.1f}) ---")
            
            # 1. Estimate essential matrix with current K
            E, R, t, inliers = self._estimate_essential_and_pose(
                pts1, pts2, K1_current, K2_current
            )
            
            if inliers is None or np.sum(inliers) < self.config.min_points:
                print(f"  ✗ Insufficient inliers: {np.sum(inliers) if inliers is not None else 0}")
                continue
            
            # 2. Triangulate with relaxed thresholds
            result = self._triangulate_with_relaxation(
                pts1[inliers], pts2[inliers],
                K1_current, K2_current,
                R, t,
                relax_factor,
                image_size1, image_size2
            )
            
            if result is None or result['points_3d'].shape[1] < self.config.min_points:
                print(f"  ✗ Triangulation failed or too few points")
                continue
            
            points_3d = result['points_3d']
            valid_pts1 = result['valid_pts1']
            valid_pts2 = result['valid_pts2']
            
            print(f"  ✓ Triangulated {points_3d.shape[1]} points")
            
            # 3. Optimize camera parameters
            optimized = self._optimize_cameras(
                points_3d, valid_pts1, valid_pts2,
                K1_current, K2_current, R, t,
                image_size1, image_size2
            )
            
            if optimized is None:
                print(f"  ✗ Optimization failed")
                continue
            
            # Update parameters
            K1_current = optimized['K1']
            K2_current = optimized['K2']
            R = optimized['R']
            t = optimized['t']
            error = optimized['error']
            
            print(f"  ✓ Optimized: error={error:.3f}px")
            print(f"    K1 focal: {K1_current[0,0]:.1f}")
            print(f"    K2 focal: {K2_current[0,0]:.1f}")
            
            # Track best result
            if points_3d.shape[1] > best_num_points:
                best_num_points = points_3d.shape[1]
                best_result = {
                    'success': True,
                    'K1': K1_current.copy(),
                    'K2': K2_current.copy(),
                    'R': R.copy(),
                    't': t.copy(),
                    'points_3d': points_3d.copy(),
                    'valid_pts1': valid_pts1.copy(),
                    'valid_pts2': valid_pts2.copy(),
                    'error': error,
                    'num_iterations': iteration + 1
                }
            
            # Check convergence
            improvement = (prev_error - error) / prev_error if prev_error > 0 else 0
            
            if improvement < self.config.error_tolerance:
                stable_count += 1
                print(f"  → Stable ({stable_count}/{self.config.stable_iterations_required})")
                
                if stable_count >= self.config.stable_iterations_required:
                    print(f"\n✓ Converged after {iteration + 1} iterations")
                    break
            else:
                stable_count = 0
            
            prev_error = error
        
        if best_result is None:
            print(f"\n✗ Refinement failed - no valid result")
            return {'success': False}
        
        print(f"\n✓ Refinement complete!")
        print(f"  Best result: {best_num_points} points, error={best_result['error']:.3f}px")
        
        return best_result
    
    def _estimate_essential_and_pose(self,
                                     pts1: np.ndarray,
                                     pts2: np.ndarray,
                                     K1: np.ndarray,
                                     K2: np.ndarray) -> Tuple:
        """Estimate essential matrix and recover pose"""
        # Normalize points
        pts1_norm = cv2.undistortPoints(
            pts1.reshape(-1, 1, 2), K1, None
        ).reshape(-1, 2)
        
        pts2_norm = cv2.undistortPoints(
            pts2.reshape(-1, 1, 2), K2, None
        ).reshape(-1, 2)
        
        # Estimate essential matrix
        E, inliers = cv2.findEssentialMat(
            pts1_norm, pts2_norm,
            focal=1.0, pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=0.001
        )
        
        if E is None or inliers is None:
            return None, None, None, None
        
        # Recover pose
        _, R, t, _ = cv2.recoverPose(
            E, pts1_norm, pts2_norm,
            focal=1.0, pp=(0.0, 0.0)
        )
        
        return E, R, t, inliers.ravel() > 0
    
    def _triangulate_with_relaxation(self,
                                     pts1: np.ndarray,
                                     pts2: np.ndarray,
                                     K1: np.ndarray,
                                     K2: np.ndarray,
                                     R: np.ndarray,
                                     t: np.ndarray,
                                     relax_factor: float,
                                     image_size1: Tuple[int, int],
                                     image_size2: Tuple[int, int]) -> Optional[Dict]:
        """Triangulate with relaxed thresholds"""
        # Create config with relaxed thresholds
        config = TriangulationConfig(
            min_triangulation_angle=self.config.base_triangulation_angle / relax_factor,
            max_reprojection_error=self.config.base_reproj_thresh * relax_factor,
            min_distance=self.config.base_near / np.sqrt(relax_factor),
            max_distance=self.config.base_far * relax_factor,
            enable_progressive=False
        )
        
        # Create triangulator
        triangulator = TriangulationEngine(
            min_triangulation_angle_deg=config.min_triangulation_angle,
            max_reprojection_error=config.max_reprojection_error,
            min_depth=config.min_distance,
            max_depth=config.max_distance,
            config=config
        )
        
        # Triangulate
        try:
            result = triangulator.triangulate_initial_points(
                pts1, pts2,
                R1=np.eye(3), t1=np.zeros((3, 1)),
                R2=R, t2=t.reshape(3, 1),
                K=[K1, K2],
                image_pair=('cam1', 'cam2')
            )
            
            if isinstance(result, dict):
                points_3d = result['points_3d']
            else:
                points_3d = result
            
            # Get valid points (those that were triangulated)
            num_valid = points_3d.shape[1]
            valid_pts1 = pts1[:num_valid]
            valid_pts2 = pts2[:num_valid]
            
            return {
                'points_3d': points_3d,
                'valid_pts1': valid_pts1,
                'valid_pts2': valid_pts2
            }
            
        except Exception as e:
            print(f"  Triangulation error: {e}")
            return None
    
    def _optimize_cameras(self,
                         points_3d: np.ndarray,
                         pts1: np.ndarray,
                         pts2: np.ndarray,
                         K1: np.ndarray,
                         K2: np.ndarray,
                         R: np.ndarray,
                         t: np.ndarray,
                         image_size1: Tuple[int, int],
                         image_size2: Tuple[int, int]) -> Optional[Dict]:
        """Optimize camera parameters"""
        
        # Pack parameters: [fx1, fy1, cx1, cy1, fx2, fy2, cx2, cy2, rvec, t]
        rvec, _ = cv2.Rodrigues(R)
        
        x0 = np.concatenate([
            [K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]],  # K1
            [K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2]],  # K2
            rvec.ravel(),                               # R
            t.ravel()                                   # t
        ])
        
        # Define bounds
        w1, h1 = image_size1
        w2, h2 = image_size2
        
        bounds_lower = [
            self.config.focal_bounds[0], self.config.focal_bounds[0],
            w1 * (0.5 - self.config.pp_bounds_ratio), h1 * (0.5 - self.config.pp_bounds_ratio),
            self.config.focal_bounds[0], self.config.focal_bounds[0],
            w2 * (0.5 - self.config.pp_bounds_ratio), h2 * (0.5 - self.config.pp_bounds_ratio),
            -np.inf, -np.inf, -np.inf,  # rvec
            -np.inf, -np.inf, -np.inf   # t
        ]
        
        bounds_upper = [
            self.config.focal_bounds[1], self.config.focal_bounds[1],
            w1 * (0.5 + self.config.pp_bounds_ratio), h1 * (0.5 + self.config.pp_bounds_ratio),
            self.config.focal_bounds[1], self.config.focal_bounds[1],
            w2 * (0.5 + self.config.pp_bounds_ratio), h2 * (0.5 + self.config.pp_bounds_ratio),
            np.inf, np.inf, np.inf,     # rvec
            np.inf, np.inf, np.inf      # t
        ]
        
        # Optimize
        try:
            result = least_squares(
                self._reprojection_error,
                x0,
                args=(points_3d, pts1, pts2),
                bounds=(bounds_lower, bounds_upper),
                method='trf',
                ftol=1e-6,
                max_nfev=100
            )
            
            # Unpack optimized parameters
            params = result.x
            
            K1_opt = np.array([
                [params[0], 0, params[2]],
                [0, params[1], params[3]],
                [0, 0, 1]
            ])
            
            K2_opt = np.array([
                [params[4], 0, params[6]],
                [0, params[5], params[7]],
                [0, 0, 1]
            ])
            
            R_opt, _ = cv2.Rodrigues(params[8:11])
            t_opt = params[11:14].reshape(3, 1)
            
            # Compute final error
            error = np.sqrt(np.mean(result.fun ** 2))
            
            return {
                'K1': K1_opt,
                'K2': K2_opt,
                'R': R_opt,
                't': t_opt,
                'error': error
            }
            
        except Exception as e:
            print(f"  Optimization error: {e}")
            return None
    
    def _reprojection_error(self,
                           params: np.ndarray,
                           points_3d: np.ndarray,
                           pts1: np.ndarray,
                           pts2: np.ndarray) -> np.ndarray:
        """Compute reprojection error for optimization"""
        
        # Unpack parameters
        K1 = np.array([
            [params[0], 0, params[2]],
            [0, params[1], params[3]],
            [0, 0, 1]
        ])
        
        K2 = np.array([
            [params[4], 0, params[6]],
            [0, params[5], params[7]],
            [0, 0, 1]
        ])
        
        R, _ = cv2.Rodrigues(params[8:11])
        t = params[11:14].reshape(3, 1)
        
        # Project to camera 1 (at origin)
        P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        
        # Project to camera 2
        P2 = K2 @ np.hstack([R, t])
        
        # Compute reprojection errors
        errors = []
        
        for i in range(points_3d.shape[1]):
            pt_3d = np.append(points_3d[:, i], 1.0)
            
            # Project to camera 1
            pt_2d_1 = P1 @ pt_3d
            pt_2d_1 = pt_2d_1[:2] / pt_2d_1[2]
            error1 = pts1[i] - pt_2d_1
            
            # Project to camera 2
            pt_2d_2 = P2 @ pt_3d
            pt_2d_2 = pt_2d_2[:2] / pt_2d_2[2]
            error2 = pts2[i] - pt_2d_2
            
            errors.extend([error1[0], error1[1], error2[0], error2[1]])
        
        return np.array(errors)