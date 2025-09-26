"""
Traditional feature detectors (SIFT, ORB, AKAZE, BRISK, Harris).

This module contains implementations of classical computer vision
feature detection algorithms.
"""

import cv2
import numpy as np
import time
from typing import List, Optional
from .base_classes import BaseFeatureDetector
from .core_data_structures import FeatureData


class SIFTDetector(BaseFeatureDetector):
    """SIFT (Scale-Invariant Feature Transform) feature detector"""
    
    def __init__(self, max_features: int = 5000, contrast_threshold: float = 0.04,
                 edge_threshold: float = 10, sigma: float = 1.6):
        """
        Initialize SIFT detector
        
        Args:
            max_features: Maximum number of features to detect
            contrast_threshold: Threshold for filtering weak features
            edge_threshold: Threshold for filtering edge-like features
            sigma: Gaussian sigma for the first octave
        """
        super().__init__(max_features)
        self.detector = cv2.SIFT_create(
            nfeatures=max_features,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )
    
    def detect(self, image: np.ndarray) -> FeatureData:
        start_time = time.time()
        gray = self.preprocess_image(image)
        
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        keypoints, descriptors = self.postprocess_features(keypoints, descriptors)
        
        return FeatureData(
            keypoints=keypoints,
            descriptors=descriptors,
            method="SIFT",
            detection_time=time.time() - start_time,
            raw_image=image
        )


class ORBDetector(BaseFeatureDetector):
    """ORB (Oriented FAST and Rotated BRIEF) feature detector"""
    
    def __init__(self, max_features: int = 5000, scale_factor: float = 1.2,
                 n_levels: int = 8, edge_threshold: int = 31):
        """
        Initialize ORB detector
        
        Args:
            max_features: Maximum number of features to detect
            scale_factor: Pyramid decimation ratio
            n_levels: Number of pyramid levels
            edge_threshold: Size of border where features are not detected
        """
        super().__init__(max_features)
        self.detector = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=edge_threshold
        )
    
    def detect(self, image: np.ndarray) -> FeatureData:
        start_time = time.time()
        gray = self.preprocess_image(image)
        
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        keypoints, descriptors = self.postprocess_features(keypoints, descriptors)
        
        return FeatureData(
            keypoints=keypoints,
            descriptors=descriptors,
            method="ORB",
            detection_time=time.time() - start_time,
            raw_image=image
        )


class AKAZEDetector(BaseFeatureDetector):
    """AKAZE (Accelerated-KAZE) feature detector"""
    
    def __init__(self, max_features: int = 5000, threshold: float = 0.001,
                 n_octaves: int = 4, descriptor_type: int = cv2.AKAZE_DESCRIPTOR_MLDB):
        """
        Initialize AKAZE detector
        
        Args:
            max_features: Maximum number of features to detect
            threshold: Detector response threshold
            n_octaves: Maximum number of octaves
            descriptor_type: Type of descriptor to use
        """
        super().__init__(max_features)
        self.detector = cv2.AKAZE_create(
            threshold=threshold,
            nOctaves=n_octaves,
            descriptor_type=descriptor_type
        )
    
    def detect(self, image: np.ndarray) -> FeatureData:
        start_time = time.time()
        gray = self.preprocess_image(image)
        
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        keypoints, descriptors = self.postprocess_features(keypoints, descriptors)
        
        return FeatureData(
            keypoints=keypoints,
            descriptors=descriptors,
            method="AKAZE",
            detection_time=time.time() - start_time,
            raw_image=image
        )


class BRISKDetector(BaseFeatureDetector):
    """BRISK (Binary Robust Invariant Scalable Keypoints) feature detector"""
    
    def __init__(self, max_features: int = 5000, threshold: int = 30,
                 octaves: int = 3, pattern_scale: float = 1.0):
        """
        Initialize BRISK detector
        
        Args:
            max_features: Maximum number of features to detect
            threshold: AGAST detection threshold
            octaves: Detection octaves
            pattern_scale: Apply this scale to the pattern used for sampling
        """
        super().__init__(max_features)
        self.detector = cv2.BRISK_create(
            thresh=threshold,
            octaves=octaves,
            patternScale=pattern_scale
        )
    
    def detect(self, image: np.ndarray) -> FeatureData:
        start_time = time.time()
        gray = self.preprocess_image(image)
        
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        keypoints, descriptors = self.postprocess_features(keypoints, descriptors)
        
        return FeatureData(
            keypoints=keypoints,
            descriptors=descriptors,
            method="BRISK",
            detection_time=time.time() - start_time,
            raw_image=image
        )


class HarrisCornerDetector(BaseFeatureDetector):
    """Harris corner detector with SIFT descriptors"""
    
    def __init__(self, max_features: int = 5000, quality_level: float = 0.01,
                 min_distance: float = 10, block_size: int = 3, k: float = 0.04):
        """
        Initialize Harris corner detector
        
        Args:
            max_features: Maximum number of features to detect
            quality_level: Parameter characterizing minimal accepted quality
            min_distance: Minimum possible Euclidean distance between corners
            block_size: Size of averaging block for computing derivative covariation
            k: Harris detector free parameter
        """
        super().__init__(max_features)
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.k = k
        
        # Use SIFT for descriptor computation
        self.descriptor_extractor = cv2.SIFT_create()
        
    def detect(self, image: np.ndarray) -> FeatureData:
        start_time = time.time()
        gray = self.preprocess_image(image)
        
        # Detect Harris corners
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            useHarrisDetector=True,
            k=self.k
        )
        
        # Convert to KeyPoint format
        keypoints = []
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                kp = cv2.KeyPoint(x=float(x), y=float(y), size=10)
                keypoints.append(kp)
        
        # Compute SIFT descriptors for Harris corners
        if keypoints:
            keypoints, descriptors = self.descriptor_extractor.compute(gray, keypoints)
            keypoints, descriptors = self.postprocess_features(keypoints, descriptors)
        else:
            descriptors = None
        
        return FeatureData(
            keypoints=keypoints,
            descriptors=descriptors,
            method="Harris",
            detection_time=time.time() - start_time,
            raw_image=image
        )


class GoodFeaturesToTrackDetector(BaseFeatureDetector):
    """Shi-Tomasi corner detector with SIFT descriptors"""
    
    def __init__(self, max_features: int = 5000, quality_level: float = 0.01,
                 min_distance: float = 10, block_size: int = 3):
        """
        Initialize Shi-Tomasi corner detector
        
        Args:
            max_features: Maximum number of features to detect
            quality_level: Parameter characterizing minimal accepted quality
            min_distance: Minimum possible Euclidean distance between corners
            block_size: Size of averaging block
        """
        super().__init__(max_features)
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        
        # Use SIFT for descriptor computation
        self.descriptor_extractor = cv2.SIFT_create()
        
    def detect(self, image: np.ndarray) -> FeatureData:
        start_time = time.time()
        gray = self.preprocess_image(image)
        
        # Detect Shi-Tomasi corners
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            useHarrisDetector=False
        )
        
        # Convert to KeyPoint format
        keypoints = []
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                kp = cv2.KeyPoint(x=float(x), y=float(y), size=10)
                keypoints.append(kp)
        
        # Compute SIFT descriptors
        if keypoints:
            keypoints, descriptors = self.descriptor_extractor.compute(gray, keypoints)
            keypoints, descriptors = self.postprocess_features(keypoints, descriptors)
        else:
            descriptors = None
        
        return FeatureData(
            keypoints=keypoints,
            descriptors=descriptors,
            method="GoodFeatures",
            detection_time=time.time() - start_time,
            raw_image=image
        )


# Factory function for easy detector creation
def create_traditional_detector(detector_type: str, **kwargs) -> BaseFeatureDetector:
    """
    Factory function to create traditional feature detectors
    
    Args:
        detector_type: Type of detector ('SIFT', 'ORB', 'AKAZE', 'BRISK', 'Harris', 'GoodFeatures')
        **kwargs: Additional parameters for the detector
        
    Returns:
        Initialized detector instance
        
    Raises:
        ValueError: If detector_type is not supported
    """
    detector_map = {
        'SIFT': SIFTDetector,
        'ORB': ORBDetector,
        'AKAZE': AKAZEDetector,
        'BRISK': BRISKDetector,
        'Harris': HarrisCornerDetector,
        'GoodFeatures': GoodFeaturesToTrackDetector
    }
    
    if detector_type not in detector_map:
        available = ', '.join(detector_map.keys())
        raise ValueError(f"Unknown detector type: {detector_type}. Available: {available}")
    
    return detector_map[detector_type](**kwargs)