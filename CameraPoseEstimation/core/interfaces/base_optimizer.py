"""
Base interface for optimization algorithms.

This defines the contract for algorithms that refine estimates through
iterative optimization (bundle adjustment, pose refinement, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
from CameraPoseEstimation2.logger import get_logger

logger = get_logger("core.interfaces")


class OptimizationStatus(Enum):
    """Status codes for optimization results"""
    SUCCESS = "success"
    MAX_ITERATIONS = "max_iterations_reached"
    CONVERGED = "converged"
    DIVERGED = "diverged"
    NUMERICAL_ERROR = "numerical_error"
    INVALID_INPUT = "invalid_input"
    FAILED = "failed"


@dataclass
class OptimizationResult:
    """
    Result of an optimization algorithm.
    
    Attributes:
        success: Whether optimization succeeded
        status: Status code from OptimizationStatus
        optimized_params: Optimized parameters (e.g., camera poses, 3D points)
        initial_cost: Cost before optimization
        final_cost: Cost after optimization
        num_iterations: Number of iterations performed
        residuals: Final residuals
        convergence_history: History of cost values per iteration
        runtime: Optimization runtime in seconds
        metadata: Additional algorithm-specific information
    """
    success: bool
    status: OptimizationStatus
    optimized_params: Optional[Any] = None
    initial_cost: float = 0.0
    final_cost: float = 0.0
    num_iterations: int = 0
    residuals: Optional[np.ndarray] = None
    convergence_history: List[float] = field(default_factory=list)
    runtime: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __bool__(self) -> bool:
        """Allow truthiness check"""
        return self.success
    
    def get_cost_reduction(self) -> float:
        """
        Get absolute cost reduction.
        
        Returns:
            float: Initial cost - final cost
        """
        return self.initial_cost - self.final_cost
    
    def get_relative_cost_reduction(self) -> float:
        """
        Get relative cost reduction.
        
        Returns:
            float: (initial - final) / initial
        """
        if self.initial_cost == 0:
            return 0.0
        return (self.initial_cost - self.final_cost) / self.initial_cost
    
    def print_summary(self):
        """Print optimization result summary"""
        print("="*60)
        logger.info("OPTIMIZATION RESULT")
        print("="*60)
        logger.info(f"Status: {'✅ ' if self.success else '❌ '}{self.status.value}")
        logger.info(f"Iterations: {self.num_iterations}")
        logger.info(f"Runtime: {self.runtime:.3f}s")
        
        logger.info(f"\nCost:")
        logger.info(f"  Initial: {self.initial_cost:.6f}")
        logger.info(f"  Final: {self.final_cost:.6f}")
        print(f"  Reduction: {self.get_cost_reduction():.6f} "
              f"({self.get_relative_cost_reduction():.2%})")
        
        if self.residuals is not None:
            logger.info(f"\nResiduals:")
            logger.info(f"  Mean: {np.mean(self.residuals):.6f}")
            logger.info(f"  Median: {np.median(self.residuals):.6f}")
            logger.info(f"  Max: {np.max(self.residuals):.6f}")
        
        if self.metadata:
            logger.info("\nMetadata:")
            for key, value in self.metadata.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.6f}")
                else:
                    logger.info(f"  {key}: {value}")
        print("="*60)


class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms.
    
    This provides a common interface for algorithms that refine estimates
    through iterative optimization.
    
    Examples:
        - BundleAdjustment
        - PoseOptimizer
        - StructureRefiner
        - LocalBundleAdjustment
        - GlobalBundleAdjustment
    
    Design Principles:
    - Single Responsibility: Each optimizer does ONE type of optimization
    - Configurable: Parameters set via constructor
    - Observable: Provides callbacks for monitoring progress
    - Consistent: All optimizers follow same interface
    """
    
    def __init__(self,
                 max_iterations: int = 100,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 **config):
        """
        Initialize optimizer with configuration.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            verbose: Whether to print progress
            **config: Algorithm-specific configuration
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.config = config
        
        # Callbacks
        self._iteration_callback: Optional[Callable] = None
        self._convergence_callback: Optional[Callable] = None
        
        # State
        self._current_iteration = 0
        self._current_cost = float('inf')
        self._converged = False
    
    # ========================================================================
    # CORE OPTIMIZATION METHOD (Required)
    # ========================================================================
    
    @abstractmethod
    def optimize(self, *args, **kwargs) -> OptimizationResult:
        """
        Perform optimization.
        
        This is the main method that implementations must provide.
        Arguments depend on the specific optimization algorithm.
        
        Returns:
            OptimizationResult: Optimization result with optimized parameters
        
        Example:
            # Bundle adjustment
            result = optimizer.optimize(cameras, points_3d, observations)
            
            # Pose refinement
            result = optimizer.optimize(pose, points_3d, points_2d, K)
        """
        pass
    
    # ========================================================================
    # COST COMPUTATION (Required)
    # ========================================================================
    
    @abstractmethod
    def compute_cost(self, params: Any, *args, **kwargs) -> float:
        """
        Compute optimization cost/error for given parameters.
        
        Args:
            params: Current parameter values
            *args: Additional data needed for cost computation
        
        Returns:
            float: Total cost
        """
        pass
    
    @abstractmethod
    def compute_residuals(self, params: Any, *args, **kwargs) -> np.ndarray:
        """
        Compute residuals for given parameters.
        
        Args:
            params: Current parameter values
            *args: Additional data needed for residual computation
        
        Returns:
            np.ndarray: Residuals
        """
        pass
    
    # ========================================================================
    # VALIDATION (Required)
    # ========================================================================
    
    @abstractmethod
    def validate_input(self, *args, **kwargs) -> tuple[bool, str]:
        """
        Validate input before optimization.
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        pass
    
    # ========================================================================
    # CONVERGENCE (Optional but recommended)
    # ========================================================================
    
    def check_convergence(self, 
                         current_cost: float, 
                         previous_cost: float) -> bool:
        """
        Check if optimization has converged.
        
        Args:
            current_cost: Cost at current iteration
            previous_cost: Cost at previous iteration
        
        Returns:
            bool: True if converged
        """
        if previous_cost == 0:
            return False
        
        relative_change = abs(current_cost - previous_cost) / previous_cost
        return relative_change < self.tolerance
    
    def should_terminate(self) -> tuple[bool, OptimizationStatus]:
        """
        Check if optimization should terminate.
        
        Returns:
            Tuple[bool, OptimizationStatus]: (should_stop, reason)
        """
        if self._current_iteration >= self.max_iterations:
            return True, OptimizationStatus.MAX_ITERATIONS
        
        if self._converged:
            return True, OptimizationStatus.CONVERGED
        
        if np.isnan(self._current_cost) or np.isinf(self._current_cost):
            return True, OptimizationStatus.NUMERICAL_ERROR
        
        return False, OptimizationStatus.SUCCESS
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    def configure(self, **config):
        """
        Update optimizer configuration.
        
        Args:
            **config: Configuration parameters to update
        """
        if 'max_iterations' in config:
            self.max_iterations = config.pop('max_iterations')
        if 'tolerance' in config:
            self.tolerance = config.pop('tolerance')
        if 'verbose' in config:
            self.verbose = config.pop('verbose')
        
        self.config.update(config)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            dict: Current configuration
        """
        return {
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance,
            'verbose': self.verbose,
            **self.config
        }
    
    # ========================================================================
    # CALLBACKS & MONITORING
    # ========================================================================
    
    def set_iteration_callback(self, callback: Callable):
        """
        Set callback to be called after each iteration.
        
        Args:
            callback: Function(iteration, cost, params) -> None
        """
        self._iteration_callback = callback
    
    def set_convergence_callback(self, callback: Callable):
        """
        Set callback to be called when converged.
        
        Args:
            callback: Function(result) -> None
        """
        self._convergence_callback = callback
    
    def _notify_iteration(self, iteration: int, cost: float, params: Any):
        """Notify iteration callback"""
        if self._iteration_callback is not None:
            self._iteration_callback(iteration, cost, params)
        
        if self.verbose:
            logger.info(f"Iteration {iteration}: cost = {cost:.6f}")
    
    def _notify_convergence(self, result: OptimizationResult):
        """Notify convergence callback"""
        if self._convergence_callback is not None:
            self._convergence_callback(result)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def reset(self):
        """Reset optimizer state"""
        self._current_iteration = 0
        self._current_cost = float('inf')
        self._converged = False
    
    def get_algorithm_name(self) -> str:
        """
        Get name of the optimization algorithm.
        
        Returns:
            str: Algorithm name
        """
        return self.__class__.__name__
    
    def supports_weights(self) -> bool:
        """
        Check if optimizer supports weighted observations.
        
        Returns:
            bool: True if weights supported
        """
        return False  # Override in subclasses
    
    def supports_robust_loss(self) -> bool:
        """
        Check if optimizer supports robust loss functions.
        
        Returns:
            bool: True if robust loss supported
        """
        return False  # Override in subclasses
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"{self.get_algorithm_name()}("
                f"max_iter={self.max_iterations}, "
                f"tol={self.tolerance})")


class IterativeOptimizer(BaseOptimizer):
    """
    Base class for iterative optimization algorithms.
    
    This extends BaseOptimizer with iteration management,
    making it easier to implement iterative algorithms.
    """
    
    def optimize(self, initial_params: Any, *args, **kwargs) -> OptimizationResult:
        """
        Perform iterative optimization.
        
        Args:
            initial_params: Initial parameter values
            *args, **kwargs: Algorithm-specific arguments
        
        Returns:
            OptimizationResult: Optimization result
        """
        # Validate input
        is_valid, error_msg = self.validate_input(initial_params, *args, **kwargs)
        if not is_valid:
            return OptimizationResult(
                success=False,
                status=OptimizationStatus.INVALID_INPUT,
                metadata={'error': error_msg}
            )
        
        # Initialize
        self.reset()
        start_time = time.time()
        
        current_params = initial_params
        previous_cost = float('inf')
        convergence_history = []
        
        # Compute initial cost
        self._current_cost = self.compute_cost(current_params, *args, **kwargs)
        initial_cost = self._current_cost
        convergence_history.append(self._current_cost)
        
        # Optimization loop
        while self._current_iteration < self.max_iterations:
            # Perform one iteration
            current_params = self._iteration_step(current_params, *args, **kwargs)
            
            # Compute new cost
            previous_cost = self._current_cost
            self._current_cost = self.compute_cost(current_params, *args, **kwargs)
            convergence_history.append(self._current_cost)
            
            self._current_iteration += 1
            
            # Notify callbacks
            self._notify_iteration(self._current_iteration, self._current_cost, current_params)
            
            # Check convergence
            self._converged = self.check_convergence(self._current_cost, previous_cost)
            
            # Check termination
            should_stop, status = self.should_terminate()
            if should_stop:
                break
        
        # Compute final residuals
        residuals = self.compute_residuals(current_params, *args, **kwargs)
        
        # Create result
        runtime = time.time() - start_time
        result = OptimizationResult(
            success=True,
            status=status if 'status' in locals() else OptimizationStatus.SUCCESS,
            optimized_params=current_params,
            initial_cost=initial_cost,
            final_cost=self._current_cost,
            num_iterations=self._current_iteration,
            residuals=residuals,
            convergence_history=convergence_history,
            runtime=runtime
        )
        
        # Notify convergence
        self._notify_convergence(result)
        
        return result
    
    @abstractmethod
    def _iteration_step(self, current_params: Any, *args, **kwargs) -> Any:
        """
        Perform one iteration step.
        
        Args:
            current_params: Current parameter values
            *args, **kwargs: Additional arguments
        
        Returns:
            Updated parameters
        """
        pass