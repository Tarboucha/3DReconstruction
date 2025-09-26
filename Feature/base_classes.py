"""
Base classes and interfaces for feature detection and matching.

This module defines the abstract base classes that all feature detectors
and matchers must implement.
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
from .core_data_structures import FeatureData, MatchData


class BaseFeatureDetector(ABC):
    """Abstract base class for all feature detectors"""
    
    def __init__(self, max_features: int = 5000, **kwargs):
        self.max_features = max_features
        self.name = self.__class__.__name__
        
    @abstractmethod
    def detect(self, image: np.ndarray) -> FeatureData:
        """
        Detect features in an image
        
        Args:
            image: Input image (RGB or grayscale)
            
        Returns:
            FeatureData object containing keypoints and descriptors
        """
        pass
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for feature detection
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    
    def postprocess_features(self, keypoints: List[cv2.KeyPoint], 
                           descriptors: Optional[np.ndarray]) -> tuple:
        """
        Post-process detected features (e.g., limit number, sort by response)
        
        Args:
            keypoints: Detected keypoints
            descriptors: Feature descriptors
            
        Returns:
            Tuple of (processed_keypoints, processed_descriptors)
        """
        if self.max_features and len(keypoints) > self.max_features:
            # Sort by response and keep top features
            sorted_pairs = sorted(
                zip(keypoints, descriptors if descriptors is not None else [None] * len(keypoints)),
                key=lambda x: x[0].response,
                reverse=True
            )[:self.max_features]
            
            keypoints = [pair[0] for pair in sorted_pairs]
            if descriptors is not None:
                descriptors = np.array([pair[1] for pair in sorted_pairs])
        
        return keypoints, descriptors


class BaseFeatureMatcher(ABC):
    """Abstract base class for feature matchers"""
    
    @abstractmethod
    def match(self, features1: FeatureData, features2: FeatureData) -> MatchData:
        """
        Match features between two sets
        
        Args:
            features1: Features from first image
            features2: Features from second image
            
        Returns:
            MatchData object containing matches
        """
        pass
    
    def validate_features(self, features1: FeatureData, features2: FeatureData) -> bool:
        """
        Validate that features can be matched
        
        Args:
            features1: Features from first image
            features2: Features from second image
            
        Returns:
            True if features are valid for matching
        """
        return (len(features1) > 0 and len(features2) > 0 and
                features1.descriptors is not None and features2.descriptors is not None)


class BasePairMatcher(ABC):
    """Abstract base class for pair-based matchers (like LightGlue, LoFTR)"""
    
    @abstractmethod
    def match_images_directly(self, img1: np.ndarray, img2: np.ndarray) -> tuple:
        """
        Match images directly without separate feature detection
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Tuple of (features1, features2, match_data)
        """
        pass
    
    def process_pair(self, img1: np.ndarray, img2: np.ndarray) -> tuple:
        """Alias for match_images_directly for consistency"""
        return self.match_images_directly(img1, img2)