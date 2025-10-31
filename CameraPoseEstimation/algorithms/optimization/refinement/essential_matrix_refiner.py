from unittest import result
import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy.optimize import least_squares

from CameraPoseEstimation2.algorithms.geometry.essential import EssentialMatrixEstimator
from CameraPoseEstimation2.algorithms.geometry.triangulation import TriangulationEngine, TriangulationConfig
from CameraPoseEstimation2.core.interfaces import OptimizationResult, OptimizationStatus
from CameraPoseEstimation2.logger import get_logger

logger = get_logger("optimization.essential_refiner")


@dataclass
class EssentialMatrixRefinerConfig:
    """Configuration for essential matrix iterative refinement"""
    
    # Relaxation schedule - start loose, get tighter
    relaxation_schedule: list = field(default_factory=lambda: [3.0, 2.5, 2.0, 1.5, 1.2, 1.0])
    
    # Convergence criteria
    error_tolerance: float = 0.01              # Stop if <1% improvement
    stable_iterations_required: int = 2        # Must be stable for N iterations
    
    # Base thresholds (will be multiplied by relaxation factor)
    base_reproj_thresh: float = 3.0            # Base reprojection error
    base_triangulation_angle: float = 1.0      # Base triangulation angle (degrees)
    base_near: float = 5                     # Minimum depth
    base_far: float = 500                   # Maximum depth
    
    # Optimization
    max_iterations: int = 10                   # Max refinement iterations
    min_points: int = 50                       # Minimum points to continue
    
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
    
    def test_vizu_on_image(self, valid_pts1: np.ndarray, valid_pts2: np.ndarray):
    
        import matplotlib.pyplot as plt
        import cv2

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        img_name1 = 'g01f09eae8523020ec8f9d6b1452305c1c02b066eda4ddbd5135158740033e45302806e322816a39cc80f6dbb123bf1fb33f9f2d3e7e4f4b2ea28030c5efd731f_1280.jpg'
        img_name2 = 'gcd1467f565bc12140098b6ccb6bf28976106640446360cb017c442d5ef376db1da0033abcf3b9e4e8ca78ff8ebcfab098a1241bf4cbc7f4617da46a45fd9f4e7_1280.jpg'

        # Read images (remove extra .jpg)
        img1 = cv2.imread('./images/statue_of_liberty_images/' + img_name1)
        img2 = cv2.imread('./images/statue_of_liberty_images/' + img_name2)

        # Convert BGR to RGB for matplotlib
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Plot image 1 with points
        ax1.imshow(img1_rgb)
        ax1.scatter(valid_pts1[:, 0], valid_pts1[:, 1], c='red', s=2, alpha=0.7)
        ax1.set_title(f'Image 1: {len(valid_pts1)} triangulated points')
        ax1.axis('off')

        # Plot image 2 with points
        ax2.imshow(img2_rgb)
        ax2.scatter(valid_pts2[:, 0], valid_pts2[:, 1], c='red', s=2, alpha=0.7)
        ax2.set_title(f'Image 2: {len(valid_pts2)} triangulated points')
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

    def refine(self,
            pts1: np.ndarray,
            pts2: np.ndarray,
            K1_init: np.ndarray,
            K2_init: np.ndarray,
            image_size1: Tuple[int, int],
            image_size2: Tuple[int, int],
            initial_inliers: Optional[np.ndarray] = None) -> Dict:
        """
        Iteratively refine essential matrix estimation.
        
        Args:
            pts1: Points in first image (N, 2)
            pts2: Points in second image (N, 2)
            K1_init: Initial intrinsics for camera 1
            K2_init: Initial intrinsics for camera 2
            image_size1: Image 1 dimensions (width, height)
            image_size2: Image 2 dimensions (width, height)
            initial_inliers: Optional mask from initial RANSAC
            
        Returns:
            Dictionary with refined parameters
        """
        logger.info(f"Starting essential matrix iterative refinement...")
        logger.info(f"  Input points: {len(pts1)}")
        logger.info(f"  Relaxation schedule: {self.config.relaxation_schedule}")
        
        if initial_inliers.ndim == 2:
            initial_inliers = initial_inliers.ravel()
        
        # Initialize working point set
        if initial_inliers is not None:
            pts1_working = pts1[initial_inliers].copy()
            pts2_working = pts2[initial_inliers].copy()
            logger.info(f"Using {len(pts1_working)} initial inliers for refinement")
        else:
            pts1_working = pts1.copy()
            pts2_working = pts2.copy()
            logger.info(f"No initial mask provided, using all {len(pts1)} points")

        K1_current = K1_init.copy()
        K2_current = K2_init.copy()
        
        best_result = None
        best_num_points = 0
        
        stable_count = 0
        prev_error = float('inf')
        
        for iteration, relax_factor in enumerate(self.config.relaxation_schedule):
            print(f"\n--- Iteration {iteration + 1}/{len(self.config.relaxation_schedule)} "
                f"(relaxation={relax_factor:.1f}) ---")
            logger.info(f"  Working with {len(pts1_working)} points")
            
            E, R, t, inliers = self._estimate_essential_and_pose(
                pts1_working, pts2_working, K1_current, K2_current
            )
            
            if inliers is None or np.sum(inliers) < self.config.min_points:
                logger.info(f"  ✗ Insufficient inliers: {np.sum(inliers) if inliers is not None else 0}")
                continue
            
            pts1_iter = pts1_working[inliers] 
            pts2_iter = pts2_working[inliers]
            logger.info(f"  Filtered to {len(pts1_iter)} inliers for this iteration")
            
            result = self._triangulate_with_relaxation(
                pts1_iter, pts2_iter,
                K1_current, K2_current,
                R, t,
                relax_factor,
                image_size1, image_size2
            )
            
            if result is None or result['points_3d'].shape[1] < self.config.min_points:
                logger.info(f"  ✗ Triangulation failed or too few points")
                continue
            
            points_3d = result['points_3d']
            valid_pts1 = result['valid_pts1']
            valid_pts2 = result['valid_pts2']
            self.test_vizu_on_image(valid_pts1,valid_pts2)
            logger.info(f"  ✓ Triangulated {points_3d.shape[1]} points")
            
            optimized = self._optimize_cameras(
                points_3d, valid_pts1, valid_pts2,
                K1_current, K2_current, R, t,
                image_size1, image_size2
            )
            
            if optimized is None:
                logger.info(f"  ✗ Optimization failed")
                continue
            
            K1_current = optimized['K1']
            K2_current = optimized['K2']
            R = optimized['R']
            t = optimized['t']
            error = optimized['error']
            
            pts1_working = pts1_iter.copy()
            pts2_working = pts2_iter.copy()
            
            logger.info(f"  ✓ Optimized: error={error:.3f}px")
            logger.info(f"    K1 focal: {K1_current[0,0]:.1f}")
            logger.info(f"    K2 focal: {K2_current[0,0]:.1f}")
            logger.info(f"  → Updated working set to {len(pts1_working)} points")
            
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
                logger.info(f"  → Stable ({stable_count}/{self.config.stable_iterations_required})")
                
                if stable_count >= self.config.stable_iterations_required:
                    logger.info(f"\n✓ Converged after {iteration + 1} iterations")
                    break
            else:
                stable_count = 0
            
            prev_error = error
        
        if best_result is None:
            logger.info(f"\n✗ Refinement failed - no valid result")
            return {'success': False}
        
        logger.info(f"\n✓ Refinement complete!")
        logger.info(f"  Best result: {best_num_points} points, error={best_result['error']:.3f}px")
        
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

        if np.linalg.norm(t) < 1e-6:
            return None, None, None, None

        
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
            min_triangulation_angle=self.config.base_triangulation_angle * relax_factor,
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
                valid_indices = result['valid_indices'] 
            else:
                points_3d = result
                valid_indices = np.arange(points_3d.shape[1])
            
            # Get valid points (those that were triangulated)
            valid_pts1 = pts1[valid_indices]
            valid_pts2 = pts2[valid_indices]
                            
            return {
                'points_3d': points_3d,
                'valid_pts1': valid_pts1,
                'valid_pts2': valid_pts2
            }
            
        except Exception as e:
            logger.info(f"  Triangulation error: {e}")
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
        t_magnitude = np.linalg.norm(t)

        rvec_bound = 3.15  # Slightly more than π
    
        # Translation: allow 5x variation from initial estimate
        t_bound_factor = 5.0
        t_lower = -t_magnitude * t_bound_factor
        t_upper = t_magnitude * t_bound_factor

        bounds_lower = [
            self.config.focal_bounds[0], self.config.focal_bounds[0],
            w1 * (0.5 - self.config.pp_bounds_ratio), h1 * (0.5 - self.config.pp_bounds_ratio),
            self.config.focal_bounds[0], self.config.focal_bounds[0],
            w2 * (0.5 - self.config.pp_bounds_ratio), h2 * (0.5 - self.config.pp_bounds_ratio),
            -rvec_bound, -rvec_bound, -rvec_bound,
            t_lower, t_lower, t_lower
        ]
        
        bounds_upper = [
            self.config.focal_bounds[1], self.config.focal_bounds[1],
            w1 * (0.5 + self.config.pp_bounds_ratio), h1 * (0.5 + self.config.pp_bounds_ratio),
            self.config.focal_bounds[1], self.config.focal_bounds[1],
            w2 * (0.5 + self.config.pp_bounds_ratio), h2 * (0.5 + self.config.pp_bounds_ratio),
            rvec_bound, rvec_bound, rvec_bound,
            t_upper, t_upper, t_upper
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
            logger.info(f"  Optimization error: {e}")
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