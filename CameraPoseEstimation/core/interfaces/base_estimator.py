"""
Base interface for all estimation algorithms.

This defines the contract for algorithms that estimate geometric relationships
(essential matrix, fundamental matrix, homography, pose, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from CameraPoseEstimation2.logger import get_logger

logger = get_logger("core.interfaces")


class EstimationStatus(Enum):
    """Status codes for estimation results"""
    SUCCESS = "success"
    INSUFFICIENT_POINTS = "insufficient_points"
    DEGENERATE_CONFIG = "degenerate_configuration"
    NO_SOLUTION = "no_solution"
    NUMERICAL_INSTABILITY = "numerical_instability"
    FAILED = "failed"


@dataclass
class EstimationResult:
    """
    Result of an estimation algorithm.
    
    Attributes:
        success: Whether estimation succeeded
        status: Status code from EstimationStatus
        model: Estimated model (e.g., essential matrix, pose)
        inliers: Boolean mask or indices of inlier points
        num_inliers: Number of inlier points
        residuals: Residual errors for each point
        confidence: Confidence score [0, 1]
        metadata: Additional algorithm-specific information
    """
    success: bool
    status: EstimationStatus
    model: Optional[Any] = None
    inliers: Optional[np.ndarray] = None
    num_inliers: int = 0
    residuals: Optional[np.ndarray] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __bool__(self) -> bool:
        """Allow truthiness check"""
        return self.success
    
    def get_inlier_ratio(self) -> float:
        """
        Get ratio of inliers to total points.
        
        Returns:
            float: Inlier ratio [0, 1]
        """
        if self.inliers is None:
            return 0.0
        
        total = len(self.inliers)
        return self.num_inliers / total if total > 0 else 0.0
    
    def print_summary(self):
        """Print estimation result summary"""
        print("="*60)
        logger.info("ESTIMATION RESULT")
        print("="*60)
        logger.info(f"Status: {'✅ ' if self.success else '❌ '}{self.status.value}")
        logger.info(f"Inliers: {self.num_inliers}")
        
        if self.inliers is not None:
            logger.info(f"Inlier ratio: {self.get_inlier_ratio():.2%}")
        
        if self.confidence > 0:
            logger.info(f"Confidence: {self.confidence:.3f}")
        
        if self.residuals is not None:
            logger.info(f"Mean residual: {np.mean(self.residuals):.3f}")
            logger.info(f"Median residual: {np.median(self.residuals):.3f}")
        
        if self.metadata:
            logger.info("\nMetadata:")
            for key, value in self.metadata.items():
                logger.info(f"  {key}: {value}")
        print("="*60)


class BaseEstimator(ABC):
    """
    Abstract base class for all estimation algorithms.
    
    This provides a common interface for algorithms that estimate
    geometric relationships from point correspondences.
    
    Examples:
        - EssentialMatrixEstimator
        - FundamentalMatrixEstimator
        - HomographyEstimator
        - PoseEstimator
        - TriangulationEstimator
    
    Design Principles:
    - Single Responsibility: Each estimator does ONE type of estimation
    - Configurable: Parameters set via constructor or methods
    - Testable: Easy to test with mock data
    - Consistent: All estimators follow same interface
    """
    
    def __init__(self, **config):
        """
        Initialize estimator with configuration.
        
        Args:
            **config: Algorithm-specific configuration parameters
        """
        self.config = config
        self._is_configured = False
    
    # ========================================================================
    # CORE ESTIMATION METHOD (Required)
    # ========================================================================
    
    @abstractmethod
    def estimate(self, *args, **kwargs) -> EstimationResult:
        """
        Perform estimation.
        
        This is the main method that implementations must provide.
        Arguments depend on the specific estimation algorithm.
        
        Returns:
            EstimationResult: Estimation result with model and metadata
        
        Example:
            # Essential matrix estimation
            result = estimator.estimate(pts1, pts2, K1, K2)
            
            # Pose estimation
            result = estimator.estimate(points_3d, points_2d, K)
        """
        pass
    
    # ========================================================================
    # VALIDATION METHODS (Required)
    # ========================================================================
    
    @abstractmethod
    def validate_input(self, *args, **kwargs) -> Tuple[bool, str]:
        """
        Validate input data before estimation.
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        
        Example:
            is_valid, msg = estimator.validate_input(pts1, pts2)
            if not is_valid:
                raise ValueError(msg)
        """
        pass
    
    @abstractmethod
    def validate_result(self, result: EstimationResult) -> bool:
        """
        Validate estimation result.
        
        Args:
            result: Estimation result to validate
        
        Returns:
            bool: True if result is valid
        """
        pass
    
    # ========================================================================
    # CONFIGURATION METHODS (Optional but recommended)
    # ========================================================================
    
    def configure(self, **config):
        """
        Update estimator configuration.
        
        Args:
            **config: Configuration parameters to update
        """
        self.config.update(config)
        self._is_configured = True
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            dict: Current configuration parameters
        """
        return self.config.copy()
    
    def reset(self):
        """
        Reset estimator to initial state.
        
        Override in subclasses if they maintain internal state.
        """
        pass
    
    # ========================================================================
    # INFORMATION METHODS (Optional but recommended)
    # ========================================================================
    
    def get_min_points(self) -> int:
        """
        Get minimum number of points required.
        
        Returns:
            int: Minimum number of point correspondences
        
        Example:
            # 5-point algorithm
            return 5
            
            # 8-point algorithm  
            return 8
        """
        return 0  # Override in subclasses
    
    def get_algorithm_name(self) -> str:
        """
        Get name of the estimation algorithm.
        
        Returns:
            str: Human-readable algorithm name
        """
        return self.__class__.__name__
    
    def supports_weights(self) -> bool:
        """
        Check if estimator supports weighted points.
        
        Returns:
            bool: True if weights are supported
        """
        return False  # Override in subclasses that support weights
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def estimate_with_validation(self, *args, **kwargs) -> EstimationResult:
        """
        Estimate with automatic input validation.
        
        This is a convenience method that validates input before estimation.
        
        Returns:
            EstimationResult: Estimation result
        """
        # Validate input
        is_valid, error_msg = self.validate_input(*args, **kwargs)
        
        if not is_valid:
            return EstimationResult(
                success=False,
                status=EstimationStatus.FAILED,
                metadata={'error': error_msg}
            )
        
        # Perform estimation
        result = self.estimate(*args, **kwargs)
        
        # Validate result
        if not self.validate_result(result):
            result.success = False
            result.metadata['validation_failed'] = True
        
        return result
    
    def __repr__(self) -> str:
        """String representation"""
        return f"{self.get_algorithm_name()}(config={self.config})"


class RANSACEstimator(BaseEstimator):
    """
    Base class for RANSAC-based estimators.
    
    This extends BaseEstimator with RANSAC-specific functionality.
    Many geometric estimators use RANSAC, so this provides common infrastructure.
    """
    
    def __init__(self, 
                 threshold: float = 1.0,
                 confidence: float = 0.99,
                 max_iterations: int = 1000,
                 **config):
        """
        Initialize RANSAC estimator.
        
        Args:
            threshold: Inlier threshold (e.g., reprojection error)
            confidence: Desired confidence level
            max_iterations: Maximum RANSAC iterations
            **config: Additional configuration
        """
        super().__init__(**config)
        self.threshold = threshold
        self.confidence = confidence
        self.max_iterations = max_iterations
    
    @abstractmethod
    def estimate_model(self, sample_points: Any) -> Any:
        """
        Estimate model from minimal sample.
        
        Args:
            sample_points: Minimal sample of points
        
        Returns:
            Estimated model from this sample
        """
        pass
    
    @abstractmethod
    def compute_residuals(self, model: Any, all_points: Any) -> np.ndarray:
        """
        Compute residuals for all points given a model.
        
        Args:
            model: Estimated model
            all_points: All point correspondences
        
        Returns:
            np.ndarray: Residuals for each point
        """
        pass
    
    def compute_num_iterations(self, inlier_ratio: float) -> int:
        """
        Compute required number of RANSAC iterations.
        
        Args:
            inlier_ratio: Current inlier ratio estimate
        
        Returns:
            int: Number of iterations needed
        """
        if inlier_ratio <= 0 or inlier_ratio >= 1:
            return self.max_iterations
        
        min_points = self.get_min_points()
        prob = inlier_ratio ** min_points
        
        if prob <= 0:
            return self.max_iterations
        
        num_iters = int(np.log(1 - self.confidence) / np.log(1 - prob))
        return min(num_iters, self.max_iterations)
    
    def get_inliers(self, residuals: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Get inlier mask from residuals.
        
        Args:
            residuals: Residual errors
        
        Returns:
            Tuple[np.ndarray, int]: (inlier_mask, num_inliers)
        """
        inlier_mask = residuals < self.threshold
        num_inliers = np.sum(inlier_mask)
        return inlier_mask, num_inliers