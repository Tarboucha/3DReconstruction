"""
Data access layer for 3D reconstruction pipeline.

This module provides:
- Data providers: Abstract data access from various sources
- Data loaders: Transform and standardize raw data
- I/O utilities: Low-level file operations (future)

Architecture:
    User Code
        ↓
    IMatchDataProvider (interface)
        ↓
    Providers (structured, mock, etc.)
        ↓
    Loaders (quality standardization)
        ↓
    I/O (file operations)
        ↓
    Raw Data

Usage:
    from data import create_provider, StructuredMatchDataProvider
    from data import MatchQualityStandardizer
    
    # Create provider
    provider = create_provider('./results/')
    
    # Use provider
    match_data = provider.get_match_data(('img1.jpg', 'img2.jpg'))
"""

# Re-export commonly used classes for convenience
from CameraPoseEstimation2.core.interfaces import IMatchDataProvider, ValidationResult

# Providers
from .providers import (
    StructuredMatchDataProvider,
    MockProvider,
    create_provider,
    ProviderFactory,
)

# Loaders
from .loaders import (
    MatchQualityStandardizer,
)


__all__ = [
    # Interface
    'IMatchDataProvider',
    'ValidationResult',
    
    # Providers
    'StructuredMatchDataProvider',
    'MockProvider',
    'create_provider',
    'ProviderFactory',
    
    # Loaders
    'MatchQualityStandardizer',
]


__version__ = '1.0.0'