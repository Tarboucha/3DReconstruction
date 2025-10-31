"""
Progressive Intrinsics Learning

Progressively estimates and refines camera intrinsics through iterative
reconstruction. Particularly useful for:
- Uncalibrated image collections
- Images from different cameras
- Per-camera intrinsic estimation
- Uncertain initial calibration

Uses multiple observations across reconstruction to improve intrinsic estimates.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

from CameraPoseEstimation2.core.interfaces.base_optimizer import BaseOptimizer, OptimizationResult, OptimizationStatus
from CameraPoseEstimation2.logger import get_logger

logger = get_logger("optimization.intrinsics_refiner")


@dataclass
class IntrinsicsEstimate:
    """Estimated intrinsic parameters with uncertainty"""
    fx: float
    fy: float
    cx: float
    cy: float
    confidence: float  # 0-1 scale
    num_observations: int
    reprojection_error: float
    
    def to_matrix(self) -> np.ndarray:
        """Convert to camera matrix"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
    
    def __repr__(self) -> str:
        return (f"IntrinsicsEstimate(fx={self.fx:.1f}, fy={self.fy:.1f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}, conf={self.confidence:.2f})")


class ProgressiveIntrinsicsLearnerConfig:
    """Configuration for progressive intrinsics learning"""
    
    # Initial estimation
    INITIAL_FOCAL_RATIO = 1.2  # Focal length as ratio of image width
    
    # Refinement parameters
    MAX_ITERATIONS = 20
    CONVERGENCE_THRESHOLD = 1e-4
    
    # Constraints
    MIN_FOCAL_LENGTH = 100.0
    MAX_FOCAL_LENGTH = 10000.0
    MIN_OBSERVATIONS = 10  # Minimum points to refine intrinsics
    
    # Learning rate (for incremental updates)
    LEARNING_RATE = 0.1  # How much to trust new estimates
    
    # Outlier rejection
    MAX_REPROJECTION_ERROR = 5.0


class ProgressiveIntrinsicsLearner(BaseOptimizer):
    """
    Progressive learning of camera intrinsics during reconstruction.
    
    Strategy:
    1. Start with estimated intrinsics (from image size)
    2. Refine as more observations become available
    3. Update incrementally with confidence weighting
    4. Track uncertainty and convergence
    
    Use cases:
    - Uncalibrated image collections
    - Mixed camera types
    - Uncertain calibration
    - Online calibration refinement
    """
    
    def __init__(self, **config):
        """
        Initialize progressive intrinsics learner.
        
        Args:
            **config: Configuration overrides
        """
        super().__init__(**config)
        self.config = ProgressiveIntrinsicsLearnerConfig()
        
        # Override config
        for key, value in config.items():
            if hasattr(self.config, key.upper()):
                setattr(self.config, key.upper(), value)
        
        # Track intrinsics per camera
        self.intrinsics_history: Dict[str, List[IntrinsicsEstimate]] = {}
        self.current_intrinsics: Dict[str, IntrinsicsEstimate] = {}
    
    # Add to CameraPoseEstimation2/algorithms/optimization/refinement/intrinsics_refiner.py

    def validate_input(self, cameras: Dict, points_3d: np.ndarray, 
                    observations: List[Dict], **kwargs) -> Tuple[bool, str]:
        """Validate input for intrinsics learning"""
        if len(cameras) < 2:
            return False, f"Need at least 2 cameras, got {len(cameras)}"
        if points_3d.shape[1] < 10:
            return False, f"Need at least 10 points for intrinsics, got {points_3d.shape[1]}"
        if len(observations) < 20:
            return False, f"Need at least 20 observations, got {len(observations)}"
        return True, ""

    def compute_residuals(self, params: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Compute residuals - placeholder for abstract method"""
        # This class uses internal optimization, residuals computed internally
        return np.array([])

    def get_algorithm_name(self) -> str:
        """Get algorithm name."""
        return "ProgressiveIntrinsicsLearner"
    
    def initialize_intrinsics(self,
                             camera_id: str,
                             image_size: Tuple[int, int],
                             initial_K: Optional[np.ndarray] = None) -> IntrinsicsEstimate:
        """
        Initialize intrinsics for a camera.
        
        Args:
            camera_id: Camera identifier
            image_size: Image dimensions (width, height)
            initial_K: Optional initial camera matrix
            
        Returns:
            Initial intrinsics estimate
        """
        width, height = image_size
        
        if initial_K is not None:
            # Use provided matrix
            fx = initial_K[0, 0]
            fy = initial_K[1, 1]
            cx = initial_K[0, 2]
            cy = initial_K[1, 2]
            confidence = 0.5  # Medium confidence for provided K
        else:
            # Estimate from image size
            focal_length = width * self.config.INITIAL_FOCAL_RATIO
            fx = fy = focal_length
            cx = width / 2.0
            cy = height / 2.0
            confidence = 0.3  # Low confidence for estimated K
        
        estimate = IntrinsicsEstimate(
            fx=fx, fy=fy, cx=cx, cy=cy,
            confidence=confidence,
            num_observations=0,
            reprojection_error=0.0
        )
        
        self.current_intrinsics[camera_id] = estimate
        self.intrinsics_history[camera_id] = [estimate]
        
        logger.info(f"Initialized intrinsics for {camera_id}:")
        logger.info(f"  {estimate}")
        
        return estimate
    
    def refine_intrinsics(self,
                         camera_id: str,
                         R: np.ndarray,
                         t: np.ndarray,
                         points_3d: np.ndarray,
                         points_2d: np.ndarray,
                         fix_principal_point: bool = True) -> IntrinsicsEstimate:
        """
        Refine intrinsics for a camera using observations.
        
        Args:
            camera_id: Camera identifier
            R: Camera rotation matrix
            t: Camera translation vector
            points_3d: 3D points (Nx3)
            points_2d: 2D observations (Nx2)
            fix_principal_point: Whether to fix cx, cy
            
        Returns:
            Refined intrinsics estimate
        """
        if len(points_3d) < self.config.MIN_OBSERVATIONS:
            logger.info(f"  {camera_id}: Insufficient observations ({len(points_3d)}), keeping current")
            return self.current_intrinsics.get(camera_id)
        
        # Get current estimate
        current = self.current_intrinsics.get(camera_id)
        if current is None:
            logger.info(f"  {camera_id}: No current intrinsics, cannot refine")
            return None
        
        logger.info(f"\nRefining intrinsics for {camera_id}:")
        logger.info(f"  Current: {current}")
        logger.info(f"  Observations: {len(points_3d)}")
        
        # Build initial parameter vector
        K_init = current.to_matrix()
        
        if fix_principal_point:
            # Only optimize focal lengths
            params = np.array([current.fx, current.fy])
            fixed_cx, fixed_cy = current.cx, current.cy
        else:
            # Optimize all intrinsics
            params = np.array([current.fx, current.fy, current.cx, current.cy])
            fixed_cx, fixed_cy = None, None
        
        # Convert R to rvec
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.reshape(3, 1)
        
        try:
            # Optimize intrinsics
            from scipy.optimize import least_squares
            
            result = least_squares(
                fun=self._intrinsics_residuals,
                x0=params,
                args=(rvec, tvec, points_3d, points_2d, fix_principal_point, fixed_cx, fixed_cy),
                method='trf',
                max_nfev=self.config.MAX_ITERATIONS * len(params),
                ftol=self.config.CONVERGENCE_THRESHOLD,
                verbose=0
            )
            
            # Unpack optimized intrinsics
            if fix_principal_point:
                fx_new, fy_new = result.x
                cx_new, cy_new = current.cx, current.cy
            else:
                fx_new, fy_new, cx_new, cy_new = result.x
            
            # Apply constraints
            fx_new = np.clip(fx_new, self.config.MIN_FOCAL_LENGTH, self.config.MAX_FOCAL_LENGTH)
            fy_new = np.clip(fy_new, self.config.MIN_FOCAL_LENGTH, self.config.MAX_FOCAL_LENGTH)
            
            # Compute final reprojection error
            K_new = np.array([[fx_new, 0, cx_new],
                             [0, fy_new, cy_new],
                             [0, 0, 1]])
            
            projected, _ = cv2.projectPoints(
                points_3d.reshape(-1, 1, 3), rvec, tvec, K_new, None
            )
            projected = projected.reshape(-1, 2)
            errors = np.linalg.norm(projected - points_2d, axis=1)
            mean_error = np.mean(errors)
            
            # Compute confidence based on error and number of observations
            confidence = self._compute_confidence(mean_error, len(points_3d))
            
            # Create new estimate
            new_estimate = IntrinsicsEstimate(
                fx=fx_new, fy=fy_new, cx=cx_new, cy=cy_new,
                confidence=confidence,
                num_observations=len(points_3d),
                reprojection_error=mean_error
            )
            
            # Blend with previous estimate using confidence-weighted average
            blended_estimate = self._blend_estimates(current, new_estimate)
            
            # Update current and history
            self.current_intrinsics[camera_id] = blended_estimate
            self.intrinsics_history[camera_id].append(blended_estimate)
            
            logger.info(f"  New estimate: {new_estimate}")
            logger.info(f"  Blended: {blended_estimate}")
            logger.info(f"  Reprojection error: {mean_error:.2f}px")
            
            return blended_estimate
            
        except Exception as e:
            logger.info(f"  Warning: Refinement failed: {e}")
            return current
    
    def update_camera_intrinsics(self,
                                reconstruction_state: Dict,
                                camera_id: str) -> Dict:
        """
        Update camera intrinsics in reconstruction state.
        
        Args:
            reconstruction_state: Current reconstruction
            camera_id: Camera to update
            
        Returns:
            Updated reconstruction state
        """
        if camera_id not in self.current_intrinsics:
            logger.info(f"No intrinsics learned for {camera_id}")
            return reconstruction_state
        
        estimate = self.current_intrinsics[camera_id]
        
        # Update in reconstruction
        if camera_id in reconstruction_state['cameras']:
            reconstruction_state['cameras'][camera_id]['K'] = estimate.to_matrix()
            logger.info(f"Updated intrinsics for {camera_id} in reconstruction")
        
        return reconstruction_state
    
    def get_intrinsics(self, camera_id: str) -> Optional[np.ndarray]:
        """
        Get current intrinsics matrix for a camera.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Camera matrix or None
        """
        if camera_id in self.current_intrinsics:
            return self.current_intrinsics[camera_id].to_matrix()
        return None
    
    def get_convergence_info(self, camera_id: str) -> Dict[str, Any]:
        """
        Get convergence information for a camera.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Dictionary with convergence metrics
        """
        if camera_id not in self.intrinsics_history:
            return {'converged': False, 'message': 'No history'}
        
        history = self.intrinsics_history[camera_id]
        
        if len(history) < 2:
            return {'converged': False, 'iterations': len(history)}
        
        # Check convergence by comparing recent estimates
        recent = history[-5:] if len(history) >= 5 else history
        
        fx_values = [e.fx for e in recent]
        fy_values = [e.fy for e in recent]
        
        fx_std = np.std(fx_values)
        fy_std = np.std(fy_values)
        
        converged = (fx_std < 10.0 and fy_std < 10.0)  # Focal length stable within 10 pixels
        
        return {
            'converged': converged,
            'iterations': len(history),
            'fx_std': fx_std,
            'fy_std': fy_std,
            'current_confidence': history[-1].confidence,
            'current_error': history[-1].reprojection_error
        }
    
    def _intrinsics_residuals(self,
                             params: np.ndarray,
                             rvec: np.ndarray,
                             tvec: np.ndarray,
                             points_3d: np.ndarray,
                             points_2d: np.ndarray,
                             fix_principal_point: bool,
                             fixed_cx: Optional[float],
                             fixed_cy: Optional[float]) -> np.ndarray:
        """Compute residuals for intrinsics optimization."""
        if fix_principal_point:
            fx, fy = params
            cx, cy = fixed_cx, fixed_cy
        else:
            fx, fy, cx, cy = params
        
        K = np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])
        
        # Project points
        projected, _ = cv2.projectPoints(
            points_3d.reshape(-1, 1, 3), rvec, tvec, K, None
        )
        projected = projected.reshape(-1, 2)
        
        # Residuals
        residuals = (projected - points_2d).flatten()
        
        return residuals
    
    def _compute_confidence(self, reprojection_error: float, num_observations: int) -> float:
        """
        Compute confidence score for intrinsics estimate.
        
        Lower error + more observations = higher confidence
        """
        # Error component (0-1, lower error = higher score)
        error_score = 1.0 / (1.0 + reprojection_error / 2.0)
        
        # Observation count component (0-1, more observations = higher score)
        obs_score = min(1.0, num_observations / 100.0)
        
        # Combined confidence
        confidence = 0.6 * error_score + 0.4 * obs_score
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _blend_estimates(self,
                        previous: IntrinsicsEstimate,
                        new: IntrinsicsEstimate) -> IntrinsicsEstimate:
        """
        Blend previous and new estimates using confidence weighting.
        
        Higher confidence estimates get more weight.
        """
        # Weighted average based on confidence
        prev_weight = previous.confidence
        new_weight = new.confidence
        total_weight = prev_weight + new_weight
        
        if total_weight == 0:
            return new
        
        alpha = new_weight / total_weight
        
        # Blend parameters
        fx_blend = (1 - alpha) * previous.fx + alpha * new.fx
        fy_blend = (1 - alpha) * previous.fy + alpha * new.fy
        cx_blend = (1 - alpha) * previous.cx + alpha * new.cx
        cy_blend = (1 - alpha) * previous.cy + alpha * new.cy
        
        # Blended confidence (take maximum)
        confidence_blend = max(previous.confidence, new.confidence)
        
        return IntrinsicsEstimate(
            fx=fx_blend, fy=fy_blend, cx=cx_blend, cy=cy_blend,
            confidence=confidence_blend,
            num_observations=new.num_observations,
            reprojection_error=new.reprojection_error
        )
    
    def optimize(self, *args, **kwargs) -> OptimizationResult:
        """Optimize - wrapper for refine_intrinsics."""
        raise NotImplementedError("Use refine_intrinsics() method")
    
    def compute_cost(self, *args, **kwargs) -> float:
        """Compute cost."""
        raise NotImplementedError("Use refine_intrinsics() which computes error internally")