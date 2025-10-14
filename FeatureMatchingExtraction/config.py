"""
Configuration management for the feature detection system.

This module provides predefined configurations, validation, and
configuration management utilities.
"""

from typing import Dict, List, Any, Optional, Union
import json
import os
from .core_data_structures import DetectorType


# =============================================================================
# Default Configurations
# =============================================================================


DEFAULT_CONFIG = {
    'methods': ['SuperPoint'],  #  Changed from ['lightglue']
    'max_features': 2048,
    'combine_strategy': 'best',
    'detector_params': {
        'SuperPoint': {
            'keypoint_threshold': 0.005,
            'nms_radius': 4
        }
    },
    'matcher_config': {
        'SuperPoint': 'lightglue'  #   NEW: Explicitly pair with LightGlue
    },
    'lightglue_configs': {
        'SuperPoint': {
            'confidence_threshold': 0.2,
            'filter_threshold': 0.1
        }
    },
    'filtering': {
        'use_adaptive_filtering': True,
        'ransac_threshold': 4.0,
        'top_k': 500
    }
}


PRESET_CONFIGS = {
    'fast': {
        'methods': ['ORB'],
        'max_features': 1000,
        'combine_strategy': 'best',
        'detector_params': {
            'ORB': {
                'scale_factor': 1.5,
                'n_levels': 6,
                'edge_threshold': 31
            }
        },
        'matcher_config': {
            'ORB': 'bf'  #   Explicit matcher
        }
    },
    
    'balanced': {
        'methods': ['SIFT', 'ORB'],
        'max_features': 2000,
        'combine_strategy': 'independent',
        'detector_params': {
            'SIFT': {'contrast_threshold': 0.04},
            'ORB': {'scale_factor': 1.2, 'n_levels': 8}
        },
        'matcher_config': {
            'SIFT': 'flann',
            'ORB': 'bf'
        }
    },
    
    'accurate': {
        'methods': ['SIFT', 'AKAZE', 'BRISK'],
        'max_features': 3000,
        'combine_strategy': 'independent',
        'detector_params': {
            'SIFT': {'contrast_threshold': 0.03},
            'AKAZE': {'threshold': 0.0005},
            'BRISK': {'threshold': 20}
        },
        'matcher_config': {
            'SIFT': 'flann',
            'AKAZE': 'bf',
            'BRISK': 'bf'
        }
    },
    
    'deep_learning': {
        'methods': ['SuperPoint', 'DISK'],  #   Changed from ['lightglue']
        'max_features': 2048,
        'combine_strategy': 'independent',
        'detector_params': {
            'SuperPoint': {'keypoint_threshold': 0.005},
            'DISK': {}
        },
        'matcher_config': {
            'SuperPoint': 'lightglue',  #   Explicit pairing
            'DISK': 'lightglue'
        }
    },
    
    'robust': {
        'methods': ['SIFT', 'AKAZE', 'SuperPoint'],  #   Changed
        'max_features': 2500,
        'combine_strategy': 'independent',
        'detector_params': {
            'SIFT': {'contrast_threshold': 0.035},
            'AKAZE': {'threshold': 0.0008},
            'SuperPoint': {}
        },
        'matcher_config': {
            'SIFT': 'flann',
            'AKAZE': 'bf',
            'SuperPoint': 'lightglue'  #   Explicit pairing
        }
    }
}


DETECTOR_SPECIFIC_CONFIGS = {
    'SIFT': {
        'max_features': 5000,
        'contrast_threshold': 0.04,
        'edge_threshold': 10,
        'sigma': 1.6
    },
    'ORB': {
        'max_features': 5000,
        'scale_factor': 1.2,
        'n_levels': 8,
        'edge_threshold': 31
    },
    'AKAZE': {
        'threshold': 0.001,
        'n_octaves': 4,
        'descriptor_type': 'MLDB'
    },
    'BRISK': {
        'threshold': 30,
        'octaves': 3,
        'pattern_scale': 1.0
    },
    'Harris': {
        'quality_level': 0.01,
        'min_distance': 10,
        'block_size': 3,
        'k': 0.04
    },
    'SuperPoint': {
        'max_features': 2048,
        'keypoint_threshold': 0.005,
        'nms_radius': 4
    },
    'DISK': {
        'max_features': 2048
    },
    'ALIKED': {
        'max_features': 2048,
        'model_name': 'aliked-n16'
    }
}


MATCHER_SPECIFIC_CONFIGS = {
    'FLANN': {
        'ratio_threshold': 0.7,
        'algorithm': 'kdtree',
        'trees': 5,
        'checks': 50
    },
    'BF': {
        'norm_type': 'NORM_L2',
        'cross_check': False,
        'ratio_threshold': None
    },
    'LightGlue': {
        'features': 'superpoint',
        'confidence_threshold': 0.2,
        'max_num_keypoints': 2048,
        'filter_threshold': 0.1
    }
}


# =============================================================================
# Configuration Functions
# =============================================================================

def get_default_config() -> Dict[str, Any]:
    """Get a copy of the default configuration"""
    import copy
    return copy.deepcopy(DEFAULT_CONFIG)


def create_config_from_preset(preset: str) -> Dict[str, Any]:
    """
    Create configuration from a preset
    
    Args:
        preset: Preset name ('fast', 'balanced', 'accurate', 'deep_learning', 'robust', 'debug')
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValueError: If preset is not available
    """
    if preset not in PRESET_CONFIGS:
        available = ', '.join(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset: {preset}. Available: {available}")
    
    import copy
    base_config = copy.deepcopy(DEFAULT_CONFIG)
    preset_config = copy.deepcopy(PRESET_CONFIGS[preset])
    
    # Merge configurations (preset overrides default)
    return merge_configs(base_config, preset_config)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration
    """
    import copy
    merged = copy.deepcopy(base_config)
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    
    return merged

def validate_matcher_config(config: Dict[str, Any]) -> bool:
    """Validate matcher configuration"""
    for method, matcher in config.get('matcher_config', {}).items():
        if method == 'lightglue':
            raise ValueError("'lightglue' is not a valid detector. Use 'SuperPoint' with matcher='lightglue'")
    return True

def validate_config(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate configuration and return any issues
    
    Args:
        config: Configuration to validate
        
    Returns:
        Dictionary with validation results:
        {
            'errors': [list of error messages],
            'warnings': [list of warning messages]
        }
    """
    errors = []
    warnings = []
    
    # Check required fields
    required_fields = ['methods', 'max_features', 'combine_strategy']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate methods
    if 'methods' in config:
        if not isinstance(config['methods'], list):
            errors.append("'methods' must be a list")
        else:
            valid_methods = [d.value for d in DetectorType] + ['lightglue']
            for method in config['methods']:
                if str(method) not in valid_methods:
                    warnings.append(f"Unknown method: {method}")
    
    # Validate max_features
    if 'max_features' in config:
        if not isinstance(config['max_features'], int) or config['max_features'] <= 0:
            errors.append("'max_features' must be a positive integer")
    
    # Validate combine_strategy
    if 'combine_strategy' in config:
        valid_strategies = ['merge', 'best', 'weighted']
        if config['combine_strategy'] not in valid_strategies:
            errors.append(f"'combine_strategy' must be one of: {valid_strategies}")
    
    # Validate detector_params
    if 'detector_params' in config:
        if not isinstance(config['detector_params'], dict):
            errors.append("'detector_params' must be a dictionary")
        else:
            for detector, params in config['detector_params'].items():
                if not isinstance(params, dict):
                    errors.append(f"Parameters for {detector} must be a dictionary")
    
    # Check for potentially incompatible combinations
    if 'methods' in config and len(config['methods']) > 1:
        methods = [str(m) for m in config['methods']]
        if 'lightglue' in methods and len(methods) > 1:
            warnings.append("LightGlue works best when used alone")
        
        # Check for binary vs float descriptor mixing
        binary_methods = ['ORB', 'BRISK', 'AKAZE']
        float_methods = ['SIFT', 'SuperPoint', 'DISK', 'ALIKED']
        
        has_binary = any(m in binary_methods for m in methods)
        has_float = any(m in float_methods for m in methods)
        
        if has_binary and has_float and config.get('combine_strategy') == 'merge':
            warnings.append("Mixing binary and float descriptors with 'merge' strategy may cause issues")
    
    return {'errors': errors, 'warnings': warnings}


def print_config(config: Dict[str, Any], title: str = "Configuration"):
    """
    Pretty print a configuration
    
    Args:
        config: Configuration to print
        title: Title for the printout
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    def print_dict(d, indent=0):
        for key, value in d.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                print_dict(value, indent + 1)
            elif isinstance(value, list):
                print(f"{prefix}{key}: {value}")
            else:
                print(f"{prefix}{key}: {value}")
    
    print_dict(config)


def save_config(config: Dict[str, Any], filepath: str):
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration to save
        filepath: Path to save file
    """
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        filepath: Path to configuration file
        
    Returns:
        Loaded configuration
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    print(f"Configuration loaded from: {filepath}")
    return config


def get_detector_config(detector_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a specific detector
    
    Args:
        detector_type: Type of detector
        
    Returns:
        Default configuration for the detector
    """
    return DETECTOR_SPECIFIC_CONFIGS.get(detector_type, {})


def get_matcher_config(matcher_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a specific matcher
    
    Args:
        matcher_type: Type of matcher
        
    Returns:
        Default configuration for the matcher
    """
    return MATCHER_SPECIFIC_CONFIGS.get(matcher_type, {})


def create_custom_config(methods: List[str], 
                        max_features: int = 2000,
                        **kwargs) -> Dict[str, Any]:
    """
    Create a custom configuration with specified methods
    
    Args:
        methods: List of detection methods
        max_features: Maximum features per method
        **kwargs: Additional configuration parameters
        
    Returns:
        Custom configuration
    """
    config = get_default_config()
    config['methods'] = methods
    config['max_features'] = max_features
    
    # Auto-configure detector parameters
    detector_params = {}
    for method in methods:
        if method in DETECTOR_SPECIFIC_CONFIGS:
            detector_params[method] = DETECTOR_SPECIFIC_CONFIGS[method].copy()
            detector_params[method]['max_features'] = max_features
    
    config['detector_params'] = detector_params
    
    # Apply any additional parameters
    for key, value in kwargs.items():
        if key in config:
            if isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
        else:
            config[key] = value
    
    return config


def get_available_presets() -> List[str]:
    """Get list of available preset configurations"""
    return list(PRESET_CONFIGS.keys())


def describe_preset(preset: str) -> str:
    """
    Get description of a preset configuration
    
    Args:
        preset: Preset name
        
    Returns:
        Description string
    """
    descriptions = {
        'fast': "Optimized for speed using ORB detector with reduced features",
        'balanced': "Good balance of speed and accuracy using SIFT and ORB",
        'accurate': "Maximum accuracy using multiple traditional detectors",
        'deep_learning': "State-of-the-art accuracy using LightGlue and SuperPoint",
        'robust': "Combines traditional and deep learning methods for robustness",
        'debug': "Minimal configuration for testing and debugging"
    }
    
    return descriptions.get(preset, "No description available")


def print_available_presets():
    """Print all available presets with descriptions"""
    print("\nAvailable Configuration Presets:")
    print("=" * 40)
    
    for preset in get_available_presets():
        description = describe_preset(preset)
        methods = PRESET_CONFIGS[preset]['methods']
        print(f"\n{preset}:")
        print(f"  Description: {description}")
        print(f"  Methods: {methods}")
        print(f"  Max Features: {PRESET_CONFIGS[preset]['max_features']}")


# =============================================================================
# Configuration Validation and Auto-Adjustment
# =============================================================================

def auto_adjust_config_for_hardware(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Auto-adjust configuration based on available hardware and libraries
    
    Args:
        config: Input configuration
        
    Returns:
        Adjusted configuration with warnings
    """
    import copy
    adjusted_config = copy.deepcopy(config)
    warnings = []
    
    # Check for PyTorch availability
    try:
        import torch
        torch_available = True
        gpu_available = torch.cuda.is_available()
    except ImportError:
        torch_available = False
        gpu_available = False
    
    # Check for LightGlue availability
    try:
        import LightGlue.lightglue
        lightglue_available = True
    except ImportError:
        lightglue_available = False
    
    # Adjust methods based on availability
    adjusted_methods = []
    for method in adjusted_config.get('methods', []):
        method_str = str(method)
        
        if method_str.lower() == 'lightglue':
            if not lightglue_available:
                warnings.append("LightGlue not available, removing from methods")
                continue
        elif method_str in ['SuperPoint', 'DISK', 'ALIKED']:
            if not torch_available:
                warnings.append(f"{method_str} requires PyTorch, removing from methods")
                continue
        
        adjusted_methods.append(method)
    
    # Fallback to SIFT if no methods remain
    if not adjusted_methods:
        adjusted_methods = ['SIFT']
        warnings.append("No available methods, falling back to SIFT")
    
    adjusted_config['methods'] = adjusted_methods
    
    # Adjust max_features for performance
    if not gpu_available and 'lightglue' in [str(m).lower() for m in adjusted_methods]:
        if adjusted_config.get('max_features', 0) > 1000:
            adjusted_config['max_features'] = 1000
            warnings.append("Reduced max_features for CPU-only LightGlue processing")
    
    # Print warnings
    if warnings:
        print("Configuration auto-adjustments:")
        for warning in warnings:
            print(f"  - {warning}")
    
    return adjusted_config