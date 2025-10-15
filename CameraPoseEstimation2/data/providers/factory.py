"""
Factory for creating match data providers with automatic format detection.

File: CameraPoseEstimation/data_providers/factory.py
"""

import glob
from pathlib import Path
from typing import Optional, Dict, Any
import pickle

from .base_provider import IMatchDataProvider
from .structured_provider import StructuredDataProvider


class ProviderFactory:
    """
    Factory for creating appropriate match data providers.
    
    Automatically detects data format and creates the right provider.
    """
    
    @staticmethod
    def auto_detect(path: str, **kwargs) -> IMatchDataProvider:
        """
        Automatically detect format and create appropriate provider.
        
        Args:
            path: Path to data (directory or file)
            **kwargs: Additional arguments to pass to provider
            
        Returns:
            Appropriate IMatchDataProvider implementation
            
        Raises:
            ValueError: If format cannot be detected
        """
        path_obj = Path(path)
        
        print(f"Auto-detecting data format at: {path}")
        
        # Check if it's a directory with batch files (FeatureMatchingExtraction format)
        if path_obj.is_dir():
            if ProviderFactory._is_structured_format(path_obj):
                print("  → Detected: FeatureMatchingExtraction batch format")
                return StructuredDataProvider(path, **kwargs)
            
            # Add more format checks here as needed
            # elif ProviderFactory._is_legacy_format(path_obj):
            #     return LegacyDataProvider(path, **kwargs)
        
        # Check if it's a single file
        elif path_obj.is_file():
            if ProviderFactory._is_legacy_pickle(path_obj):
                print("  → Detected: Legacy pickle format")
                # return LegacyDataProvider(path, **kwargs)
                raise NotImplementedError("Legacy format not yet implemented")
        
        raise ValueError(
            f"Could not detect data format at: {path}\n"
            f"Expected:\n"
            f"  - Directory with *_batch_*.pkl and *_image_metadata.pkl files\n"
            f"  - Legacy pickle file (not yet implemented)"
        )
    
    @staticmethod
    def _is_structured_format(path: Path) -> bool:
        """Check if directory contains FeatureMatchingExtraction batch files"""
        # Look for signature files
        batch_files = list(path.glob("*_batch_*.pkl"))
        metadata_files = list(path.glob("*_image_metadata.pkl"))
        
        return bool(batch_files and metadata_files)
    
    @staticmethod
    def _is_legacy_pickle(path: Path) -> bool:
        """Check if file is a legacy pickle format"""
        if not path.suffix == '.pkl':
            return False
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # Check for legacy format indicators
            # (has 'matches_data' and 'image_info' at top level)
            if isinstance(data, dict):
                has_matches = 'matches_data' in data
                has_images = 'image_info' in data
                return has_matches and has_images
        except:
            return False
        
        return False
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> IMatchDataProvider:
        """
        Create provider from configuration dictionary.
        
        Args:
            config: Configuration with keys:
                - provider_type: 'structured', 'legacy', 'mock', 'database'
                - path: Path to data
                - Other provider-specific options
                
        Returns:
            Configured provider
            
        Example:
            config = {
                'provider_type': 'structured',
                'path': './results/',
                'cache_size': 200,
                'preload_all': True
            }
            provider = ProviderFactory.create_from_config(config)
        """
        provider_type = config.get('provider_type', 'auto')
        path = config.get('path')
        
        if not path:
            raise ValueError("Config must specify 'path'")
        
        # Remove provider_type and path from kwargs
        kwargs = {k: v for k, v in config.items() 
                 if k not in ['provider_type', 'path']}
        
        if provider_type == 'auto':
            return ProviderFactory.auto_detect(path, **kwargs)
        
        elif provider_type == 'structured':
            return StructuredDataProvider(path, **kwargs)
        
        elif provider_type == 'mock':
            from .mock_provider import MockProvider
            return MockProvider(**kwargs)
        
        # Add more provider types as they're implemented
        # elif provider_type == 'legacy':
        #     return LegacyDataProvider(path, **kwargs)
        # elif provider_type == 'database':
        #     return DatabaseProvider(path, **kwargs)
        
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
    
    @staticmethod
    def create_with_fallback(primary_config: Dict, 
                           fallback_config: Dict) -> IMatchDataProvider:
        """
        Create provider with fallback.
        
        Tries primary provider first, falls back to fallback if it fails.
        
        Args:
            primary_config: Config for primary provider
            fallback_config: Config for fallback provider
            
        Returns:
            Primary or fallback provider
        """
        try:
            print("Attempting primary provider...")
            return ProviderFactory.create_from_config(primary_config)
        except Exception as e:
            print(f"Primary provider failed: {e}")
            print("Falling back to backup provider...")
            return ProviderFactory.create_from_config(fallback_config)


# =============================================================================
# Convenience functions for common use cases
# =============================================================================

def create_provider(path: str, **kwargs) -> IMatchDataProvider:
    """
    Simple convenience function to create a provider.
    
    Automatically detects format.
    
    Args:
        path: Path to data
        **kwargs: Provider options (cache_size, preload_all, etc.)
        
    Returns:
        Configured provider
        
    Example:
        provider = create_provider('./results/', cache_size=200)
    """
    return ProviderFactory.auto_detect(path, **kwargs)


def create_structured_provider(results_dir: str,
                              cache_size: int = 100,
                              preload: bool = False,
                              min_quality: Optional[float] = None) -> IMatchDataProvider:
    """
    Create a StructuredDataProvider with common settings.
    
    Args:
        results_dir: Directory with batch files
        cache_size: LRU cache size
        preload: Whether to load all data immediately
        min_quality: Filter pairs below this quality
        
    Returns:
        StructuredDataProvider instance
    """
    return StructuredDataProvider(
        results_dir=results_dir,
        cache_size=cache_size,
        preload_all=preload,
        min_quality=min_quality
    )


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Example 1: Auto-detect format
    provider = create_provider('./results/')
    provider.print_summary()
    
    # Example 2: From config
    config = {
        'provider_type': 'structured',
        'path': './results/',
        'cache_size': 200,
        'min_quality': 0.6
    }
    provider = ProviderFactory.create_from_config(config)
    
    # Example 3: With fallback
    primary = {'provider_type': 'structured', 'path': './results/new/'}
    fallback = {'provider_type': 'structured', 'path': './results/backup/'}
    provider = ProviderFactory.create_with_fallback(primary, fallback)
    
    # Validate and use
    validation = provider.validate()
    if validation:
        print("✓ Provider validated successfully")
        
        # Get best pairs for initialization
        best_pairs = provider.get_best_pairs(k=5)
        print(f"\nTop 5 pairs for initialization:")
        for i, pair in enumerate(best_pairs, 1):
            match_data = provider.get_match_data(pair)
            print(f"  {i}. {pair[0]} <-> {pair[1]}")
            print(f"     Quality: {match_data.standardized_pair_quality:.3f}, "
                  f"Matches: {match_data.num_matches}")
    else:
        print("✗ Validation failed")
        validation.print_report()