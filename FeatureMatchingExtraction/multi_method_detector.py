"""
Multi-Method Feature Detector - Simplified for New API

This class manages multiple feature detectors. In the new API,
each method stays independent (no combining/merging).

Store this as: multi_method_detector.py
"""

import numpy as np
from typing import List, Dict, Optional, Union

from .core_data_structures import FeatureData, DetectorType
from .traditional_detectors import create_traditional_detector
from .deep_learning_detectors import create_deep_learning_detector
from .base_classes import BaseFeatureDetector


class MultiMethodFeatureDetector:
    """
    Manages multiple feature detection methods
    
    Simplified for new API - no combining, no merging.
    Each method stays independent.
    
    Usage:
        detector = MultiMethodFeatureDetector(
            methods=['SIFT', 'ORB'],
            max_features_per_method=2000,
            detector_params={'SIFT': {'contrast_threshold': 0.04}}
        )
        
        # Detect with all methods
        all_features = detector.detect_all(image)
        # Returns: {'SIFT': FeatureData(...), 'ORB': FeatureData(...)}
        
        # Or detect with specific method
        sift_features = detector.detectors['SIFT'].detect(image)
    """
    
    def __init__(self, 
                 methods: List[Union[str, DetectorType]], 
                 max_features_per_method: int = 2000,
                 combine_strategy: str = "independent",  # Only for backward compatibility
                 detector_params: Optional[Dict[str, Dict]] = None):
        """
        Initialize multi-method detector
        
        Args:
            methods: List of detector names (e.g., ['SIFT', 'ORB', 'AKAZE'])
            max_features_per_method: Max features per method
            combine_strategy: Ignored (kept for backward compatibility)
            detector_params: Per-method detector parameters
        """
        self.methods = [str(m) for m in methods]
        self.max_features_per_method = max_features_per_method
        self.detector_params = detector_params or {}
        
        # Initialize detectors
        self.detectors = self._initialize_detectors()
        
        # Validate that we have at least one working detector
        if not self.detectors:
            raise ValueError("No detectors could be initialized!")
        
        print(f" Initialized {len(self.detectors)} detectors: {list(self.detectors.keys())}")
    
    def _initialize_detectors(self) -> Dict[str, 'BaseFeatureDetector']:
        """
        Initialize all requested detectors
        
        Returns:
            Dictionary mapping method name to detector instance
        """
        detectors = {}
        
        # Traditional methods
        traditional = ['SIFT', 'ORB', 'AKAZE', 'BRISK', 'KAZE']
        
        # Deep learning methods
        deep_learning = ['SuperPoint', 'DISK', 'ALIKED']
        
        for method_str in self.methods:
            try:
                # Get parameters for this method
                params = self.detector_params.get(method_str, {})
                params['max_features'] = self.max_features_per_method
                
                # Create detector
                if method_str in traditional:
                    detector = create_traditional_detector(method_str, **params)
                    detectors[method_str] = detector
                    print(f"   {method_str} detector initialized")
                    
                elif method_str in deep_learning:
                    try:
                        detector = create_deep_learning_detector(method_str, **params)
                    except ImportError:
                        print(f"    {method_str} requires additional dependencies")
                        continue
                    detectors[method_str] = detector
                    print(f"   {method_str} detector initialized")
                    
                else:
                    print(f"    Unknown detector: {method_str}")
                    
            except Exception as e:
                print(f"   Failed to initialize {method_str}: {e}")
        
        return detectors
    
    def detect_all(self, image: np.ndarray) -> Dict[str, FeatureData]:
        """
        Detect features using all initialized methods
        
        Each method runs independently and returns its own FeatureData.
        
        Args:
            image: Input image (RGB or grayscale)
            
        Returns:
            Dictionary mapping method name to FeatureData
            
        Example:
            >>> all_features = detector.detect_all(image)
            >>> sift_features = all_features['SIFT']
            >>> print(f"SIFT found {len(sift_features)} keypoints")
        """
        results = {}
        
        for name, detector in self.detectors.items():
            print(f"  Detecting with {name}...")
            try:
                features = detector.detect(image)
                results[name] = features
                print(f"     {len(features)} features in {features.detection_time:.3f}s")
            except Exception as e:
                print(f"     Failed: {e}")
                # Store empty FeatureData for failed methods
                results[name] = FeatureData([], None, name, raw_image=image)
        
        return results
    
    def get_detector(self, method_name: str) -> Optional['BaseFeatureDetector']:
        """
        Get a specific detector by name
        
        Args:
            method_name: Name of the method (e.g., 'SIFT')
            
        Returns:
            Detector instance or None if not found
        """
        return self.detectors.get(method_name)
    
    def has_method(self, method_name: str) -> bool:
        """Check if a method is available"""
        return method_name in self.detectors
    
    def __repr__(self) -> str:
        """String representation"""
        methods_str = ', '.join(self.detectors.keys())
        return f"MultiMethodFeatureDetector(methods=[{methods_str}])"


# =============================================================================
# Convenience Function
# =============================================================================

def create_multi_detector(methods: List[str], 
                         max_features: int = 2000,
                         **detector_params) -> MultiMethodFeatureDetector:
    """
    Convenience function to create a multi-method detector
    
    Args:
        methods: List of method names
        max_features: Max features per method
        **detector_params: Parameters for specific detectors
        
    Returns:
        Configured MultiMethodFeatureDetector
        
    Example:
        >>> detector = create_multi_detector(
        ...     methods=['SIFT', 'ORB'],
        ...     max_features=2000,
        ...     SIFT={'contrast_threshold': 0.04},
        ...     ORB={'scale_factor': 1.2}
        ... )
    """
    return MultiMethodFeatureDetector(
        methods=methods,
        max_features_per_method=max_features,
        detector_params=detector_params
    )