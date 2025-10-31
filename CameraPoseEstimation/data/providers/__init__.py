"""
Data providers for match data access.

This module provides implementations of IMatchDataProvider for accessing
match data from different sources.

Available Providers:
    - StructuredMatchDataProvider: Read from batch files on disk
    - MockProvider: Generate synthetic data for testing
    - (Future) DatabaseProvider: Read from database
    - (Future) CloudProvider: Read from cloud storage

Usage:
    from data.providers import create_provider, StructuredMatchDataProvider
    
    # Create provider from batch files
    provider = create_provider('./results/')
    
    # Or create directly
    provider = StructuredMatchDataProvider(
        batch_dir='./results/',
        cache_size=100
    )
    
    # Use provider
    match_data = provider.get_match_data(('img1.jpg', 'img2.jpg'))
"""

# Import interface from core
from CameraPoseEstimation2.core.interfaces import IMatchDataProvider, ValidationResult

# Import provider implementations
from .structured_provider import StructuredMatchDataProvider
from .mock_provider import MockProvider
from .factory import create_provider, ProviderFactory
from .folder_provider import FolderMatchDataProvider, is_folder_format

__all__ = [
    # Interface (re-export for convenience)
    'IMatchDataProvider',
    'ValidationResult',
    
    # Provider implementations
    'StructuredMatchDataProvider',
    'MockProvider',
    'FolderMatchDataProvider',
    
    # Factory
    'create_provider',
    'ProviderFactory',
    'is_folder_format',
]


# Version
__version__ = '1.0.0'


# Convenience function
def list_available_providers():
    """
    List all available provider implementations.
    
    Returns:
        dict: Provider name -> class mapping
    """
    return {
        'structured': StructuredMatchDataProvider,
        'mock': MockProvider,
        # Add more as they're implemented
        # 'database': DatabaseProvider,
        # 'cloud': CloudProvider,
    }


def get_provider_info(provider_name: str) -> str:
    """
    Get information about a provider.
    
    Args:
        provider_name: Name of provider ('structured', 'mock', etc.)
    
    Returns:
        str: Provider description
    """
    info = {
        'structured': 'Read match data from batch files on disk',
        'mock': 'Generate synthetic match data for testing',
    }
    
    return info.get(provider_name, 'Unknown provider')