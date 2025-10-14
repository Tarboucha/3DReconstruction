"""
Matcher Factory - Intelligent matcher selection and initialization

This module handles all matcher creation logic, using the compatibility
configuration to select appropriate matchers for each detector.
"""

import cv2
from typing import Optional, Dict, Any
from .matcher_compatibility import MatcherCompatibilityManager
from .feature_matchers import (
    EnhancedBFMatcher, 
    EnhancedFLANNMatcher,
    LightGlueMatcher,
    create_traditional_matcher,
    create_deep_learning_matcher
)


class MatcherFactory:
    """
    Factory for creating appropriate matchers for detectors
    
    Uses compatibility configuration to select the best matcher
    and initialize it with correct parameters.
    
    Usage:
        factory = MatcherFactory()
        matcher = factory.create_matcher('SIFT')  # Returns FLANN matcher
        matcher = factory.create_matcher('ALIKED', matcher_type='lightglue')
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize matcher factory
        
        Args:
            config: Optional configuration with:
                - matcher_config: Dict mapping detector -> matcher type
                - lightglue_configs: Dict with LightGlue configurations
                - matcher_params: Dict with general matcher parameters
        """
        self.config = config or {}
        self.compat_manager = MatcherCompatibilityManager()
        
        # Cache for matcher parameters
        self.matcher_config = self.config.get('matcher_config', {})
        self.lightglue_configs = self.config.get('lightglue_configs', {})
        self.matcher_params = self.config.get('matcher_params', {})
    
    def create_matcher(self, 
                      detector_name: str,
                      matcher_type: Optional[str] = None,
                      **override_params) -> Optional[Any]:
        """
        Create appropriate matcher for a detector
        
        Args:
            detector_name: Name of detector (e.g., 'SIFT', 'ALIKED')
            matcher_type: Specific matcher to use (None = auto-select)
            **override_params: Override default parameters
            
        Returns:
            Initialized matcher instance or None if creation fails
            
        Example:
            >>> factory = MatcherFactory()
            >>> matcher = factory.create_matcher('SIFT')  # Auto-selects FLANN
            >>> matcher = factory.create_matcher('ALIKED', 'lightglue')
        """
        # 1. Determine matcher type to use
        if matcher_type is None:
            matcher_type = self._determine_matcher_type(detector_name)
        
        if matcher_type is None:
            print(f"  Could not determine matcher type for {detector_name}")
            return None
        
        # 2. Validate compatibility
        if not self.compat_manager.is_compatible(detector_name, matcher_type):
            print(f" {detector_name} is not compatible with {matcher_type}")
            compatible = self.compat_manager.get_compatible_matchers(detector_name)
            print(f"Compatible matchers: {', '.join(compatible)}")
            return None
        
        # 3. Create matcher with appropriate parameters
        try:
            matcher = self._create_matcher_instance(
                detector_name, 
                matcher_type, 
                **override_params
            )
            
            if matcher is not None:
                print(f"Created {matcher_type} matcher for {detector_name}")
            
            return matcher
            
        except Exception as e:
            print(f" Error creating {matcher_type} matcher for {detector_name}: {e}")
            return None
    
    def _determine_matcher_type(self, detector_name: str) -> Optional[str]:
        """
        Determine which matcher to use for a detector
        
        Priority:
        1. Explicit config (matcher_config)
        2. Recommended from compatibility manager
        3. Default from compatibility manager
        """
        # Check explicit configuration
        if detector_name in self.matcher_config:
            return self.matcher_config[detector_name]
        
        # Get recommended matcher
        recommended = self.compat_manager.get_recommended_matcher(detector_name)
        if recommended:
            return recommended
        
        # Fallback to default
        return self.compat_manager.get_default_matcher(detector_name)
    
    def _create_matcher_instance(self,
                                detector_name: str,
                                matcher_type: str,
                                **override_params) -> Optional[Any]:
        """
        Create actual matcher instance with correct parameters
        
        Args:
            detector_name: Detector name
            matcher_type: Type of matcher to create
            **override_params: Override parameters
            
        Returns:
            Matcher instance
        """
        matcher_type_lower = matcher_type.lower()
        
        # Get base parameters from compatibility config
        base_params = self.compat_manager.get_matcher_params(detector_name, matcher_type)
        
        # Merge with user config and overrides
        if matcher_type_lower in self.matcher_params:
            base_params.update(self.matcher_params[matcher_type_lower])
        base_params.update(override_params)
        
        # Create matcher based on type
        if matcher_type_lower == 'lightglue':
            return self._create_lightglue_matcher(detector_name, base_params)
        
        elif matcher_type_lower == 'flann':
            return self._create_flann_matcher(detector_name, base_params)
        
        elif matcher_type_lower == 'bf':
            return self._create_bf_matcher(detector_name, base_params)
        
        else:
            print(f"Unknown matcher type: {matcher_type}")
            return None
    
    def _create_lightglue_matcher(self, 
                                 detector_name: str, 
                                 params: Dict[str, Any]) -> Optional[LightGlueMatcher]:
        """Create LightGlue matcher with proper configuration"""
        # Determine feature type for LightGlue
        feature_type = detector_name.lower()
        
        if feature_type not in ['superpoint', 'disk', 'aliked']:
            print(f"{detector_name} not compatible with LightGlue")
            return None
        
        # Get LightGlue-specific config
        lg_config = self.lightglue_configs.get(detector_name, {})
        
        # Build parameters
        lightglue_params = {
            'features': feature_type,
            'max_num_keypoints': params.get('max_num_keypoints', 2048),
            'confidence_threshold': lg_config.get('confidence_threshold', 0.2),
            'filter_threshold': lg_config.get('filter_threshold', 0.1),
            'weights_path': lg_config.get('weights_path'),
            'auto_download': lg_config.get('auto_download', True)
        }
        
        # Create matcher
        matcher = create_deep_learning_matcher('LightGlue', **lightglue_params)
        
        return matcher
    
    def _create_flann_matcher(self,
                             detector_name: str,
                             params: Dict[str, Any]) -> EnhancedFLANNMatcher:
        """Create FLANN matcher with proper configuration"""
        # Default parameters
        flann_params = {
            'ratio_threshold': params.get('ratio_threshold', 0.7),
            'algorithm': params.get('algorithm', 'kdtree'),
            'trees': params.get('trees', 5),
            'checks': params.get('checks', 50)
        }
        
        # Binary descriptors need LSH algorithm
        if detector_name in ['ORB', 'BRISK', 'AKAZE']:
            flann_params['algorithm'] = 'lsh'
        
        return create_traditional_matcher('FLANN', **flann_params)
    
    def _create_bf_matcher(self,
                          detector_name: str,
                          params: Dict[str, Any]) -> EnhancedBFMatcher:
        """Create BF matcher with proper norm type"""
        # Determine norm type based on descriptor type
        if detector_name in ['ORB', 'BRISK', 'AKAZE']:
            norm_type = cv2.NORM_HAMMING
        else:
            norm_type = cv2.NORM_L2
        
        # Allow override
        norm_type_str = params.get('norm_type', None)
        if norm_type_str:
            if norm_type_str.lower() == 'hamming':
                norm_type = cv2.NORM_HAMMING
            elif norm_type_str.lower() == 'l2':
                norm_type = cv2.NORM_L2
        
        bf_params = {
            'norm_type': norm_type,
            'cross_check': params.get('cross_check', False),
            'ratio_threshold': params.get('ratio_threshold', 0.7)
        }
        
        return EnhancedBFMatcher(**bf_params)
    
    def get_matcher_info(self, detector_name: str) -> Dict[str, Any]:
        """
        Get information about matcher selection for a detector
        
        Returns:
            Dict with matcher selection info
        """
        matcher_type = self._determine_matcher_type(detector_name)
        
        info = {
            'detector': detector_name,
            'selected_matcher': matcher_type,
            'is_compatible': self.compat_manager.is_compatible(detector_name, matcher_type) if matcher_type else False,
            'compatible_matchers': self.compat_manager.get_compatible_matchers(detector_name),
            'recommended_matcher': self.compat_manager.get_recommended_matcher(detector_name),
            'default_matcher': self.compat_manager.get_default_matcher(detector_name)
        }
        
        return info
    
    def print_compatibility_info(self, detector_name: str):
        """Print compatibility information for a detector"""
        info = self.get_matcher_info(detector_name)
        
        print(f"\n{'='*60}")
        print(f"Matcher Info for {detector_name}")
        print(f"{'='*60}")
        print(f"Selected matcher: {info['selected_matcher']}")
        print(f"Compatible: {info['is_compatible']}")
        print(f"Recommended: {info['recommended_matcher']}")
        print(f"Default: {info['default_matcher']}")
        print(f"All compatible: {', '.join(info['compatible_matchers'])}")
        print(f"{'='*60}\n")


# =============================================================================
# Convenience Functions
# =============================================================================

def create_matcher_for_detector(detector_name: str, 
                                config: Optional[Dict[str, Any]] = None,
                                **kwargs) -> Optional[Any]:
    """
    Convenience function to create matcher for a detector
    
    Args:
        detector_name: Name of detector
        config: Optional configuration dict
        **kwargs: Override parameters
        
    Returns:
        Initialized matcher or None
        
    Example:
        >>> matcher = create_matcher_for_detector('SIFT')
        >>> matcher = create_matcher_for_detector('ALIKED', matcher_type='lightglue')
    """
    factory = MatcherFactory(config)
    return factory.create_matcher(detector_name, **kwargs)


def get_recommended_matcher(detector_name: str) -> Optional[str]:
    """Get recommended matcher name for a detector"""
    compat_manager = MatcherCompatibilityManager()
    return compat_manager.get_recommended_matcher(detector_name)


def validate_detector_matcher_combination(detector_name: str, 
                                          matcher_name: str) -> bool:
    """Check if detector-matcher combination is valid"""
    compat_manager = MatcherCompatibilityManager()
    return compat_manager.is_compatible(detector_name, matcher_name)