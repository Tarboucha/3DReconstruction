"""
Core interfaces for dependency injection.

This module defines the abstract base classes (contracts) that all
implementations must follow. This enables:
- Dependency Injection: Components receive dependencies via interfaces
- Loose Coupling: Implementations can be swapped without changing code
- Testability: Easy to create mocks for testing
- Extensibility: New implementations just implement the interface

Usage:
    from core.interfaces import IMatchDataProvider, BaseEstimator, BaseOptimizer
    
    # Implement a new provider
    class MyProvider(IMatchDataProvider):
        def get_match_data(self, pair):
            # Your implementation
            pass
    
    # Implement a new estimator
    class MyEstimator(BaseEstimator):
        def estimate(self, *args):
            # Your implementation
            pass
"""

# Provider interface
from .base_provider import (
    IMatchDataProvider,
    ValidationResult
)

# Estimator interfaces
from .base_estimator import (
    BaseEstimator,
    RANSACEstimator,
    EstimationResult,
    EstimationStatus
)

# Optimizer interfaces
from .base_optimizer import (
    BaseOptimizer,
    IterativeOptimizer,
    OptimizationResult,
    OptimizationStatus
)


__all__ = [
    # Provider
    'IMatchDataProvider',
    'ValidationResult',
    
    # Estimator
    'BaseEstimator',
    'RANSACEstimator',
    'EstimationResult',
    'EstimationStatus',
    
    # Optimizer
    'BaseOptimizer',
    'IterativeOptimizer',
    'OptimizationResult',
    'OptimizationStatus',
]


__version__ = '1.0.0'