"""
Utility functions for feature detection and matching.

This module contains core helper functions for image processing,
filtering, analysis, and data structures.
"""

import cv2
import numpy as np
import json
import pickle
import time
import os
import glob
from pathlib import Path
from dataclasses import dataclass, field, is_dataclass, asdict
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Union, Any, Iterator
from enum import Enum

from .core_data_structures import FeatureData, MatchData, EnhancedDMatch, MultiMethodMatchData, ScoreType


# =============================================================================
# Size Validation and Image Helpers
# =============================================================================

def validate_size(size: Tuple[int, int], name: str = "size") -> Tuple[int, int]:
    """
    Validate and return size tuple.
    
    Args:
        size: (width, height) tuple
        name: Parameter name for error messages
        
    Returns:
        Validated (width, height) tuple
        
    Raises:
        ValueError: If size is invalid
    """
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError(f"{name} must be a tuple of (width, height)")
    
    width, height = size
    
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError(f"{name} must contain integers, got ({type(width)}, {type(height)})")
    
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} must be positive, got ({width}, {height})")
    
    return (width, height)

def image_size_from_shape(image: np.ndarray) -> Tuple[int, int]:
    """
    Extract (width, height) from image array.
    
    Args:
        image: Image array with shape (height, width) or (height, width, channels)
        
    Returns:
        Tuple of (width, height)
        
    Example:
        >>> img = np.zeros((480, 640, 3))  # shape is (height, width, channels)
        >>> size = image_size_from_shape(img)
        >>> print(size)  # (640, 480) - width, height
    """
    if image.ndim < 2:
        raise ValueError(f"Image must have at least 2 dimensions, got {image.ndim}")
    
    height, width = image.shape[:2]
    return (width, height)

def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                 interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize image to target size with clear documentation.
    
    Args:
        image: Input image array (height, width) or (height, width, channels)
        target_size: Target size as (width, height) - API convention
        interpolation: OpenCV interpolation method
        
    Returns:
        Resized image
        
    Note:
        target_size is (width, height) as per API convention.
        OpenCV's cv2.resize also expects (width, height).
    """
    width, height = validate_size(target_size, "target_size")
    resized = cv2.resize(image, (width, height), interpolation=interpolation)
    return resized

def print_size_info(image: np.ndarray, label: str = "Image"):
    """
    Print size information in both conventions for debugging.
    
    Args:
        image: Image array
        label: Label for the printout
    """
    height, width = image.shape[:2]
    channels = image.shape[2] if image.ndim > 2 else 1
    
    print(f"{label}:")
    print(f"  Shape (NumPy):     {image.shape} (height, width, channels)")
    print(f"  Size (API):        ({width}, {height}) (width, height)")
    print(f"  Resolution:        {width} Ã— {height} pixels")
    print(f"  Aspect ratio:      {width/height:.2f}:1")

# =============================================================================
# Filtering Functions
# =============================================================================

def enhanced_filter_matches_with_homography(
    kp1: List[cv2.KeyPoint], 
    kp2: List[cv2.KeyPoint],
    matches: Union[List[Union[cv2.DMatch, EnhancedDMatch]], MultiMethodMatchData],
    match_data: Union[MatchData, MultiMethodMatchData],
    ransac_threshold: float = 5.0,
    confidence: float = 0.99,
    max_iters: int = 2000
) -> Tuple[List[Union[cv2.DMatch, EnhancedDMatch]], np.ndarray]:
    """
    Enhanced filter matches using RANSAC homography with score awareness
    
    Handles both single-method (MatchData) and multi-method (MultiMethodMatchData) cases.
    For multi-method, each match retains its original score_type from its source method.
    """
    if isinstance(match_data, MultiMethodMatchData):
        matches_to_filter = match_data.get_filtered_matches()
    else:
        matches_to_filter = match_data.get_best_matches()
    
    if len(matches_to_filter) < 4:
        return [], None
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches_to_filter]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches_to_filter]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_threshold, 
                                  confidence=confidence, maxIters=max_iters)
    
    if H is None or mask is None:
        return [], None
    
    mask = mask.ravel()
    filtered_matches = [m for m, keep in zip(matches_to_filter, mask) if keep]
    
    return filtered_matches, H

def adaptive_match_filtering(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    matches: List[Union[cv2.DMatch, EnhancedDMatch]],  # ✅ Direct list parameter
    match_data: Optional[MatchData] = None,  # ✅ Optional MatchData
    ransac_threshold: float = 4.0
) -> Tuple[List[Union[cv2.DMatch, EnhancedDMatch]], Optional[np.ndarray], Dict[str, Any]]:
    """
    Adaptive match filtering with homography
    
    Args:
        kp1: Keypoints from image 1
        kp2: Keypoints from image 2
        matches: List of matches (cv2.DMatch or EnhancedDMatch)
        match_data: Optional MatchData object (for additional info)
        ransac_threshold: RANSAC threshold
    
    Returns:
        (filtered_matches, homography, filter_info)
    """
    
    # Use matches directly (not from match_data)
    if not matches or len(matches) < 4:
        return [], None, {'method': 'none', 'reason': 'insufficient_matches'}
    
    # Rest of the function stays the same...
    filtered_matches, H = enhanced_filter_matches_with_homography(
        kp1, kp2, matches, match_data, ransac_threshold
    )
    
    filter_info = {
        'method': 'homography_ransac',
        'original_count': len(matches),
        'filtered_count': len(filtered_matches),
        'ransac_threshold': ransac_threshold,
        'homography_found': H is not None
    }
    
    return filtered_matches, H, filter_info

def calculate_reprojection_error(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    matches: List[Union[cv2.DMatch, EnhancedDMatch]],
    homography: np.ndarray
) -> np.ndarray:
    """Calculate reprojection errors for matches given homography"""
    if homography is None or len(matches) == 0:
        return np.array([])
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    projected_pts = cv2.perspectiveTransform(pts1, homography).reshape(-1, 2)
    errors = np.linalg.norm(projected_pts - pts2, axis=1)
    
    return errors

# =============================================================================
# Low-Level Serialization Helpers (used by all classes)
# =============================================================================

def keypoint_to_dict(kp: cv2.KeyPoint) -> Dict[str, Any]:
    """Convert cv2.KeyPoint to serializable dictionary"""
    return {
        'pt': kp.pt,
        'size': kp.size,
        'angle': kp.angle,
        'response': kp.response,
        'octave': kp.octave,
        'class_id': kp.class_id
    }

def dict_to_keypoint(d: Dict[str, Any]) -> cv2.KeyPoint:
    """Convert dictionary back to cv2.KeyPoint"""
    kp = cv2.KeyPoint(
        x=float(d['pt'][0]),
        y=float(d['pt'][1]),
        size=float(d['size']),
        angle=float(d['angle']),
        response=float(d['response']),
        octave=int(d['octave']),
        class_id=int(d['class_id'])
    )
    return kp

def keypoints_to_list(keypoints: List[cv2.KeyPoint]) -> List[Dict[str, Any]]:
    """Convert list of keypoints to serializable list"""
    return [keypoint_to_dict(kp) for kp in keypoints]

def list_to_keypoints(data: List[Dict[str, Any]]) -> List[cv2.KeyPoint]:
    """Convert serializable list back to keypoints"""
    return [dict_to_keypoint(d) for d in data]

__all__ = [
    # Size helpers
    'validate_size',
    'image_size_from_shape',
    'resize_image',
    'print_size_info',

    
    # Filtering
    'enhanced_filter_matches_with_homography',
    'adaptive_match_filtering',
    'calculate_reprojection_error',

    'keypoint_to_dict',
    'dict_to_keypoint',
    'keypoints_to_list',
    'list_to_keypoints',

]