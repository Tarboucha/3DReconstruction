"""
Utility functions for feature detection and matching.

This module contains helper functions for filtering, visualization,
analysis, image loading, and other utility operations.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
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
    
    # cv2.resize expects (width, height) - matches our API convention
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
    print(f"  Resolution:        {width} × {height} pixels")
    print(f"  Aspect ratio:      {width/height:.2f}:1")

class ImageSourceType(Enum):
    """Types of image sources"""
    SYNTHETIC = "synthetic"
    FOLDER = "folder"
    SINGLE_IMAGE = "single_image"
    IMAGE_LIST = "image_list"
    CUSTOM = "custom"


@dataclass
class ImageInfo:
    """Unified image information container"""
    image: np.ndarray
    identifier: str  # filename, index, or custom ID
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_type: ImageSourceType = ImageSourceType.CUSTOM
    
    @property
    def size(self) -> Tuple[int, int]:
        """Return (width, height)"""
        return (self.image.shape[1], self.image.shape[0])
    
    @property
    def channels(self) -> int:
        return self.image.shape[2] if len(self.image.shape) > 2 else 1


class ImageSource(ABC):
    """Abstract base class for image sources"""
    
    @abstractmethod
    def get_images(self) -> Iterator[ImageInfo]:
        """Yield ImageInfo objects"""
        pass
    
    @abstractmethod
    def get_image_pairs(self) -> Iterator[Tuple[ImageInfo, ImageInfo]]:
        """Yield pairs of ImageInfo objects for matching"""
        pass
    
    @abstractmethod
    def get_source_info(self) -> Dict[str, Any]:
        """Return information about this source"""
        pass


class FolderImageSource(ImageSource):
    """Load images from a folder"""
    
    def __init__(self, folder_path: str, 
                 max_images: Optional[int] = None,
                 resize_to: Optional[Tuple[int, int]] = None,
                 image_extensions: List[str] = None):
        """
        Initialize folder image source
        
        Args:
            folder_path: Path to folder containing images
            max_images: Maximum number of images to load (None for all)
            resize_to: Resize images to (width, height) - API convention
                      Example: (1920, 1080) means width=1920, height=1080
            image_extensions: List of file extensions to search for
        """
        self.folder_path = Path(folder_path)
        self.max_images = max_images
        
        # ✅ Validate and store resize_to as (width, height)
        if resize_to is not None:
            self.resize_to = validate_size(resize_to, "resize_to")
        else:
            self.resize_to = None
            
        self.image_extensions = image_extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        if not self.folder_path.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")
    
    def get_images(self) -> Iterator[ImageInfo]:
        """Get images from folder"""
        # Find all image files
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(glob.glob(str(self.folder_path / f"*{ext}")))
        
        image_files = sorted(list(set(image_files)))
        
        if self.max_images:
            image_files = image_files[:self.max_images]
        
        for img_file in image_files:
            try:
                image = cv2.imread(img_file)
                if image is None:
                    print(f"Warning: Could not load image {img_file}")
                    continue
                
                # Convert BGR to RGB for consistency
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # ✅ Get original size using helper function
                original_size = image_size_from_shape(image)  # (width, height)
                
                # ✅ Resize if requested - using helper function
                if self.resize_to:
                    image = resize_image(image, self.resize_to)
                
                yield ImageInfo(
                    image=image,
                    identifier=os.path.basename(img_file),
                    metadata={
                        'filepath': img_file,
                        'original_size': original_size,  # (width, height)
                        'resized': self.resize_to is not None,
                        'final_size': image_size_from_shape(image)  # (width, height)
                    },
                    source_type=ImageSourceType.FOLDER
                )
                
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
    
    def get_source_info(self) -> Dict[str, Any]:
        """Get information about this image source"""
        return {
            'type': 'folder',
            'folder_path': str(self.folder_path),
            'extensions': self.image_extensions,
            'resize_to': self.resize_to,  # (width, height) if set
            'max_images': self.max_images
        }

    def get_image_pairs(self) -> Iterator[Tuple[ImageInfo, ImageInfo]]:
        """Generate pairs of images for matching benchmarks"""
        images = list(self.get_images())
        
        # Create pairs from consecutive images
        for i in range(len(images) - 1):
            yield (images[i], images[i + 1])
        
        # Add some additional pairs for more robust testing
        if len(images) > 2:
            # First with last
            yield (images[0], images[-1])
            # Middle pairs
            if len(images) > 3:
                mid = len(images) // 2
                yield (images[mid], images[mid + 1])
    
    def get_image_count(self) -> int:
        """Get total number of images in folder"""
        return len(list(self.get_images()))
    
    def get_image_list(self) -> List[ImageInfo]:
        """Get all images as a list (loads all into memory)"""
        return list(self.get_images())


class SingleImageSource(ImageSource):
    """Single image source"""
 
    def __init__(self, image_path: str, resize_to: Optional[Tuple[int, int]] = None):
        """
        Initialize single image source
        
        Args:
            image_path: Path to the image file
            resize_to: Resize image to (width, height) - API convention
        """
        self.image_path = image_path
        
        # ✅ Validate and store resize_to as (width, height)
        if resize_to is not None:
            self.resize_to = validate_size(resize_to, "resize_to")
        else:
            self.resize_to = None
        
        if not os.path.exists(image_path):
            raise ValueError(f"Image file does not exist: {image_path}")
    
    def get_images(self) -> Iterator[ImageInfo]:
        """Get the single image"""
        try:
            image = cv2.imread(self.image_path)
            if image is None:
                raise ValueError(f"Could not load image: {self.image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # ✅ Get original size
            original_size = image_size_from_shape(image)
            
            # ✅ Resize if requested
            if self.resize_to:
                image = resize_image(image, self.resize_to)
            
            yield ImageInfo(
                image=image,
                identifier=os.path.basename(self.image_path),
                metadata={
                    'filepath': self.image_path,
                    'original_size': original_size,  # (width, height)
                    'resized': self.resize_to is not None,
                    'final_size': image_size_from_shape(image)  # (width, height)
                },
                source_type=ImageSourceType.SINGLE_IMAGE
            )
            
        except Exception as e:
            print(f"Error loading {self.image_path}: {e}")
    
    def get_image_pairs(self) -> Iterator[Tuple[ImageInfo, ImageInfo]]:
        """For single image, create a transformed version for pairing"""
        images = list(self.get_images())
        if images:
            img1 = images[0]
            
            # ✅ Use size from helper function
            width, height = image_size_from_shape(img1.image)
            
            # Apply transformation
            M = cv2.getRotationMatrix2D((width/2, height/2), 10, 0.95)
            img2_array = cv2.warpAffine(img1.image, M, (width, height))
            
            img2 = ImageInfo(
                image=img2_array,
                identifier=f"{img1.identifier}_transformed",
                metadata=img1.metadata.copy(),
                source_type=img1.source_type
            )
            img2.metadata['transformation'] = 'rotation_10deg_scale_0.95'
            yield (img1, img2)

    def get_source_info(self) -> Dict[str, Any]:
        """Get information about this image source"""
        return {
            'type': 'single_image',
            'image_path': self.image_path,
            'resize_to': self.resize_to
        }


# =============================================================================
# Convenience Functions for Image Loading
# =============================================================================

def load_images_from_folder(folder_path: str, 
                           max_images: Optional[int] = None,
                           resize_to: Optional[Tuple[int, int]] = None,
                           extensions: List[str] = None) -> List[Tuple[np.ndarray, str]]:
    """
    Load images from a folder and return as list of (image, filename) tuples
    
    Args:
        folder_path: Path to folder containing images
        max_images: Maximum number of images to load
        resize_to: Resize to (width, height) - API convention
                  Example: (1920, 1080) for Full HD
        extensions: File extensions to search for
    
    Returns:
        List of (image_array, filename) tuples
    """
    source = FolderImageSource(folder_path, max_images, resize_to, extensions)
    return [(img_info.image, img_info.identifier) for img_info in source.get_images()]



def load_single_image(image_path: str, 
                     resize_to: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, str]:
    """
    Load a single image
    
    Args:
        image_path: Path to image file
        resize_to: Resize to (width, height) - API convention
    
    Returns:
        Tuple of (image_array, filename)
    """
    source = SingleImageSource(image_path, resize_to)
    img_info = next(source.get_images())
    return img_info.image, img_info.identifier


def get_images_from_folder(folder_path: str, extensions: List[str] = None) -> List[str]:
    """
    Extract all image file paths from a given folder
    
    Args:
        folder_path: Path to the folder containing images
        extensions: List of extensions to search for
    
    Returns:
        List of full paths to image files
    """
    if extensions is None:
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
    
    image_paths = []
    
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return []
    
    # Search for all image files with different extensions
    for extension in extensions:
        pattern = os.path.join(folder_path, extension)
        # Case-insensitive search
        image_paths.extend(glob.glob(pattern))
        image_paths.extend(glob.glob(pattern.upper()))

    # Remove duplicates and sort
    image_paths = sorted(list(set(image_paths)))
    
    return image_paths


# =============================================================================
# Filtering Functions
# =============================================================================
def enhanced_filter_matches_with_homography(
    kp1: List[cv2.KeyPoint], 
    kp2: List[cv2.KeyPoint],
    matches: Union[List[Union[cv2.DMatch, EnhancedDMatch]], MultiMethodMatchData],  # ✓ Accept new type
    match_data: Union[MatchData, MultiMethodMatchData],  # ✓ Accept new type
    ransac_threshold: float = 5.0,
    confidence: float = 0.99,
    max_iters: int = 2000
) -> Tuple[List[Union[cv2.DMatch, EnhancedDMatch]], np.ndarray]:
    """Enhanced filter matches using RANSAC homography with score awareness"""
    
    # Handle MultiMethodMatchData
    if isinstance(matches, MultiMethodMatchData):
        matches = matches.get_all_matches()  # Get actual match list
    elif isinstance(match_data, MultiMethodMatchData):
        matches = match_data.get_all_matches()
    
    if len(matches) < 4:
        return matches, None    
    
    # CRITICAL FIX: Validate ALL match indices before processing
    max_idx1 = len(kp1) - 1
    max_idx2 = len(kp2) - 1
    
    valid_matches = []
    invalid_count = 0
    
    for m in matches:
        # Check if indices are within bounds
        if m.queryIdx <= max_idx1 and m.trainIdx <= max_idx2 and m.queryIdx >= 0 and m.trainIdx >= 0:
            valid_matches.append(m)
        else:
            invalid_count += 1
            if invalid_count <= 5:  # Only print first 5 to avoid spam
                print(f"Warning: Invalid match - queryIdx={m.queryIdx} (max={max_idx1}), "
                      f"trainIdx={m.trainIdx} (max={max_idx2})")
    
    if invalid_count > 0:
        print(f"Total invalid matches filtered: {invalid_count}/{len(matches)}")
    
    # Check if we have enough valid matches
    if len(valid_matches) < 4:
        print(f"Error: Only {len(valid_matches)} valid matches after filtering (need 4+ for homography)")
        print(f"  Keypoints: kp1={len(kp1)}, kp2={len(kp2)}")
        print(f"  Original matches: {len(matches)}")
        return valid_matches if valid_matches else matches[:min(len(matches), 4)], None
    
    # Use only valid matches from here on
    matches = valid_matches
    
    # Convert matches to points - now guaranteed to be safe
    try:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    except IndexError as e:
        # This should never happen after validation, but handle it anyway
        print(f"CRITICAL: Index error after validation: {e}")
        print(f"  kp1 length: {len(kp1)}, kp2 length: {len(kp2)}")
        print(f"  Match count: {len(matches)}")
        if matches:
            print(f"  queryIdx range: [{min(m.queryIdx for m in matches)}, {max(m.queryIdx for m in matches)}]")
            print(f"  trainIdx range: [{min(m.trainIdx for m in matches)}, {max(m.trainIdx for m in matches)}]")
        return matches, None
    except Exception as e:
        print(f"Unexpected error extracting points: {e}")
        return matches, None
    
    # Find homography with RANSAC
    try:
        homography, mask = cv2.findHomography(
            src_pts, dst_pts, 
            cv2.RANSAC, 
            ransac_threshold,
            confidence=confidence,
            maxIters=max_iters
        )
    except cv2.error as e:
        print(f"OpenCV error computing homography: {e}")
        return matches, None
    
    # Filter matches based on inliers
    filtered_matches = []
    if mask is not None:
        for i, match in enumerate(matches):
            if mask[i]:
                filtered_matches.append(match)
    else:
        # No mask returned - homography estimation failed
        print("Warning: Homography estimation failed, returning unfiltered matches")
        return matches, homography
    
    # Sort filtered matches by quality score
    if filtered_matches and match_data.score_type == ScoreType.CONFIDENCE:
        # Higher confidence is better
        filtered_matches.sort(
            key=lambda x: x.score if isinstance(x, EnhancedDMatch) else 1.0 - x.distance, 
            reverse=True
        )
    elif filtered_matches:
        # Lower distance is better
        filtered_matches.sort(
            key=lambda x: x.score if isinstance(x, EnhancedDMatch) else x.distance
        )
    
    inlier_ratio = len(filtered_matches) / len(matches) if matches else 0
    print(f"Homography filtering: {len(matches)} → {len(filtered_matches)} matches (inlier ratio: {inlier_ratio:.2%})")
    
    return filtered_matches, homography


def adaptive_match_filtering(
    match_data: Union[MatchData, MultiMethodMatchData],  # ✓ Accept new type
    adaptive_threshold: bool = True,
    top_k: Optional[int] = None,
    percentile_threshold: float = 75.0
) -> Union[MatchData, MultiMethodMatchData]:
    """Adaptive filtering based on score type and distribution"""
    
    # Handle MultiMethodMatchData
    if isinstance(match_data, MultiMethodMatchData):
        # Apply filtering to each method separately to respect score types
        filtered_multi = MultiMethodMatchData()
        filtered_multi.all_keypoints1 = match_data.all_keypoints1
        filtered_multi.all_keypoints2 = match_data.all_keypoints2
        
        for method in match_data.get_methods():
            method_match_data = match_data.per_method_matches[method]
            offset1, offset2 = match_data.method_offsets[method]
            
            # Filter this method's matches
            if len(method_match_data.matches) > 0:
                scores = method_match_data.get_match_scores(use_filtered=False)
                
                if adaptive_threshold and len(scores) > 5:
                    if method_match_data.score_type == ScoreType.CONFIDENCE:
                        threshold = np.percentile(scores, 100 - percentile_threshold)
                    else:
                        threshold = np.percentile(scores, percentile_threshold)
                else:
                    threshold = 0.2 if method_match_data.score_type == ScoreType.CONFIDENCE else 0.8
                
                filtered = method_match_data.filter_by_score(threshold, top_k)
                filtered_multi.add_method_matches(method, filtered, offset1, offset2)
        
        return filtered_multi
    
    # Original MatchData handling
    if not match_data.matches:
        return match_data
    
    scores = match_data.get_match_scores(use_filtered=False)
    
    if adaptive_threshold and len(scores) > 5:
        if match_data.score_type == ScoreType.CONFIDENCE:
            threshold = np.percentile(scores, 100 - percentile_threshold)
        else:
            threshold = np.percentile(scores, percentile_threshold)
    else:
        threshold = 0.2 if match_data.score_type == ScoreType.CONFIDENCE else 0.8
    
    filtered_data = match_data.filter_by_score(threshold, top_k)
    return filtered_data



def remove_duplicate_matches(
    matches: List[Union[cv2.DMatch, EnhancedDMatch]],
    distance_threshold: float = 1.0
) -> List[Union[cv2.DMatch, EnhancedDMatch]]:
    """Remove duplicate matches based on keypoint distance"""
    if not matches:
        return matches
    
    unique_matches = []
    used_query_idx = set()
    used_train_idx = set()
    
    # Sort by score quality first
    if isinstance(matches[0], EnhancedDMatch):
        if matches[0].score_type == ScoreType.CONFIDENCE:
            sorted_matches = sorted(matches, key=lambda x: x.score, reverse=True)
        else:
            sorted_matches = sorted(matches, key=lambda x: x.score)
    else:
        sorted_matches = sorted(matches, key=lambda x: x.distance)
    
    for match in sorted_matches:
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        
        # Check if indices are already used
        if query_idx not in used_query_idx and train_idx not in used_train_idx:
            unique_matches.append(match)
            used_query_idx.add(query_idx)
            used_train_idx.add(train_idx)
    
    return unique_matches


# =============================================================================
# Correspondence Extraction
# =============================================================================

def extract_correspondences(
    kp1: List[cv2.KeyPoint], 
    kp2: List[cv2.KeyPoint],
    matches: Union[List[Union[cv2.DMatch, EnhancedDMatch]], MultiMethodMatchData],  # ✓ Accept new type
    include_scores: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Enhanced correspondence extraction with optional score information"""

    if isinstance(matches, MultiMethodMatchData):
        matches = matches.get_filtered_matches()
    
    if not matches:
        if include_scores:
            return np.array([]).reshape(0, 4), np.array([])
        return np.array([]).reshape(0, 4)
    
    correspondences = []
    scores = []
    
    for match in matches:
        if isinstance(match, EnhancedDMatch):
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            correspondences.append([pt1[0], pt1[1], pt2[0], pt2[1]])
            scores.append(match.score)
        else:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            correspondences.append([pt1[0], pt1[1], pt2[0], pt2[1]])
            scores.append(match.distance)
    
    correspondences = np.array(correspondences)
    scores = np.array(scores)
    
    if include_scores:
        return correspondences, scores
    return correspondences

# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_matches_with_scores(
    img1: np.ndarray, img2: np.ndarray,
    kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint],
    match_data: MatchData, title: str = "Matches",
    max_matches: int = 50, show_histogram: bool = True,
    figsize: Tuple[int, int] = (15, 8)
):
    """Enhanced visualization showing score information"""
    matches = match_data.get_best_matches()[:max_matches]
    
    if not matches:
        print("No matches to visualize")
        return
    
    # Convert to cv2.DMatch for drawing
    cv2_matches = []
    for match in matches:
        if isinstance(match, EnhancedDMatch):
            cv2_matches.append(match.to_cv2_dmatch())
        else:
            cv2_matches.append(match)
    
    # Draw matches
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, cv2_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    plt.figure(figsize=figsize)
    plt.imshow(img_matches)
    
    # Create title with score information
    scores = match_data.get_match_scores()
    if len(scores) > 0:
        score_info = f"({match_data.score_type.value}: {np.mean(scores):.3f}±{np.std(scores):.3f})"
        full_title = f"{title} - {len(matches)} matches {score_info}"
    else:
        full_title = f"{title} - {len(matches)} matches"
    
    plt.title(full_title)
    plt.axis('off')
    
    # Add score histogram as inset
    if show_histogram and len(scores) > 5:
        try:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            axins = inset_axes(plt.gca(), width="30%", height="20%", loc='upper right')
            axins.hist(scores, bins=min(20, len(scores)//2), alpha=0.7)
            axins.set_title(f'{match_data.score_type.value} distribution', fontsize=8)
            axins.tick_params(labelsize=6)
        except ImportError:
            print("matplotlib.axes_grid1 not available for histogram inset")
    
    plt.tight_layout()
    plt.show()


def visualize_keypoints(
    image: np.ndarray, 
    features: FeatureData, 
    title: str = "Keypoints",
    max_keypoints: int = 500,
    color_by_response: bool = True,
    figsize: Tuple[int, int] = (12, 8)
):
    """Visualize detected keypoints"""
    keypoints = features.keypoints[:max_keypoints]
    
    plt.figure(figsize=figsize)
    plt.imshow(image)
    
    if keypoints:
        # Extract keypoint coordinates and responses
        points = np.array([kp.pt for kp in keypoints])
        responses = np.array([kp.response for kp in keypoints])
        
        if color_by_response and len(responses) > 0:
            # Color by response strength
            scatter = plt.scatter(
                points[:, 0], points[:, 1], 
                c=responses, cmap='viridis', 
                s=20, alpha=0.7
            )
            plt.colorbar(scatter, label='Response')
        else:
            plt.scatter(points[:, 0], points[:, 1], c='red', s=20, alpha=0.7)
    
    plt.title(f"{title} - {len(keypoints)} keypoints ({features.method})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# =============================================================================
# Analysis Functions
# =============================================================================

class MatchQualityAnalyzer:
    """Analyze and compare match quality across different methods"""
    
    @staticmethod
    def analyze_match_data(match_data: Union[MatchData, MultiMethodMatchData]) -> Dict[str, Any]:
        """Analyze match data and return quality metrics"""
        
        # Handle MultiMethodMatchData
        if isinstance(match_data, MultiMethodMatchData):
            # Return aggregated analysis across all methods
            all_matches = match_data.get_filtered_matches()
            
            if len(all_matches) == 0:
                return {
                    'num_matches': 0,
                    'quality_score': 0.0,
                    'method': 'Multi-Method',
                    'per_method': {}
                }
            
            # Analyze each method separately
            per_method_analysis = {}
            for method in match_data.get_methods():
                method_data = match_data.per_method_matches[method]
                per_method_analysis[method] = MatchQualityAnalyzer.analyze_match_data(method_data)
            
            # Compute overall quality as weighted average
            total_matches = sum(a['num_matches'] for a in per_method_analysis.values())
            if total_matches > 0:
                weighted_quality = sum(
                    a['num_matches'] * a['quality_score'] 
                    for a in per_method_analysis.values()
                ) / total_matches
            else:
                weighted_quality = 0.0
            
            return {
                'num_matches': len(all_matches),
                'quality_score': weighted_quality,
                'method': 'Multi-Method',
                'per_method': per_method_analysis,
                'methods': match_data.get_methods()
            }
        
        # Original MatchData handling
        scores = match_data.get_match_scores()
        
        if len(scores) == 0:
            return {
                'num_matches': 0,
                'score_type': match_data.score_type.value,
                'quality_score': 0.0,
                'method': match_data.method
            }
        
        analysis = {
            'num_matches': len(scores),
            'score_type': match_data.score_type.value,
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'median_score': float(np.median(scores)),
            'method': match_data.method
        }
        
        # Calculate normalized quality score
        if match_data.score_type == ScoreType.CONFIDENCE:
            quality_score = np.mean(scores)
        else:  # DISTANCE
            max_reasonable_dist = 2.0
            normalized_scores = 1.0 - np.clip(scores / max_reasonable_dist, 0, 1)
            quality_score = np.mean(normalized_scores)
        
        analysis['quality_score'] = float(quality_score)
        
        if len(scores) >= 5:
            analysis['percentiles'] = {
                '25th': float(np.percentile(scores, 25)),
                '75th': float(np.percentile(scores, 75)),
                '90th': float(np.percentile(scores, 90)),
                '95th': float(np.percentile(scores, 95))
            }
        
        return analysis
    
    @staticmethod
    def compare_methods(results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare results from multiple methods"""
        comparison = {}
        
        for method_name, result in results.items():
            if 'match_data' in result:
                match_data = result['match_data']
                analysis = MatchQualityAnalyzer.analyze_match_data(match_data)
                comparison[method_name] = analysis
        
        # Find best method by quality score
        if comparison:
            best_method = max(
                comparison.keys(), 
                key=lambda x: comparison[x]['quality_score']
            )
            comparison['best_method'] = best_method
            comparison['best_quality'] = comparison[best_method]['quality_score']
        
        return comparison
    
    @staticmethod
    def generate_report(comparison: Dict[str, Any]) -> str:
        """Generate a readable report from comparison results"""
        if not comparison:
            return "No results to analyze."
        
        report = ["Feature Matching Analysis Report", "=" * 40, ""]
        
        # Summary
        if 'best_method' in comparison:
            report.append(f"Best Method: {comparison['best_method']}")
            report.append(f"Best Quality Score: {comparison['best_quality']:.3f}")
            report.append("")
        
        # Detailed results
        report.append("Detailed Results:")
        report.append("-" * 20)
        
        for method_name, analysis in comparison.items():
            if method_name in ['best_method', 'best_quality']:
                continue
                
            report.append(f"\n{method_name}:")
            report.append(f"  Matches: {analysis['num_matches']}")
            report.append(f"  Quality Score: {analysis['quality_score']:.3f}")
            report.append(f"  Score Type: {analysis['score_type']}")
            report.append(f"  Mean Score: {analysis['mean_score']:.3f}")
            report.append(f"  Std Score: {analysis['std_score']:.3f}")
            
            if 'percentiles' in analysis:
                report.append(f"  90th Percentile: {analysis['percentiles']['90th']:.3f}")
        
        return "\n".join(report)


# =============================================================================
# Geometric Verification Functions
# =============================================================================

def verify_matches_with_fundamental_matrix(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint], 
    matches: List[Union[cv2.DMatch, EnhancedDMatch]],
    method: int = cv2.FM_RANSAC,
    ransac_threshold: float = 3.0,
    confidence: float = 0.99
) -> Tuple[List[Union[cv2.DMatch, EnhancedDMatch]], np.ndarray]:
    """Verify matches using fundamental matrix estimation"""
    if len(matches) < 8:  # Need at least 8 points for fundamental matrix
        return matches, None
    
    # Extract points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Find fundamental matrix
    F, mask = cv2.findFundamentalMat(
        pts1, pts2, method, ransac_threshold, confidence
    )
    
    # Filter matches
    good_matches = []
    if mask is not None:
        for i, match in enumerate(matches):
            if mask[i]:
                good_matches.append(match)
    
    return good_matches, F


def calculate_reprojection_error(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    matches: List[Union[cv2.DMatch, EnhancedDMatch]],
    homography: np.ndarray
) -> np.ndarray:
    """Calculate reprojection errors for matches given homography"""
    if homography is None or len(matches) == 0:
        return np.array([])
    
    # Extract points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Project points from image 1 to image 2
    projected_pts = cv2.perspectiveTransform(pts1, homography).reshape(-1, 2)
    
    # Calculate distances
    errors = np.linalg.norm(projected_pts - pts2, axis=1)
    
    return errors


# =============================================================================
# Performance Monitoring
# =============================================================================

def benchmark_detector_performance(
    detector, 
    images: List[np.ndarray], 
    num_runs: int = 3
) -> Dict[str, Any]:
    """Benchmark detector performance on a set of images"""
    times = []
    feature_counts = []
    
    for run in range(num_runs):
        run_times = []
        run_counts = []
        
        for img in images:
            start_time = time.time()
            features = detector.detect(img)
            detection_time = time.time() - start_time
            
            run_times.append(detection_time)
            run_counts.append(len(features))
        
        times.extend(run_times)
        feature_counts.extend(run_counts)
    
    return {
        'detector': detector.__class__.__name__,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'mean_features': np.mean(feature_counts),
        'std_features': np.std(feature_counts),
        'total_images': len(images),
        'total_runs': num_runs
    }


# =============================================================================
# Save/Load Functions
# =============================================================================

def save_enhanced_results(results: Any, 
                         filepath: Union[str, Path], 
                         format: str = 'json',
                         include_metadata: bool = True,
                         create_dirs: bool = True) -> bool:
    """
    Save benchmark results in various formats with proper serialization
    
    Args:
        results: Results object to save (dict, dataclass, or any serializable object)
        filepath: Path where to save the file
        format: Output format ('json', 'pickle', 'both')
        include_metadata: Whether to include saving metadata
        create_dirs: Whether to create parent directories if they don't exist
        
    Returns:
        bool: True if successful, False otherwise
    """
    filepath = Path(filepath)
    
    try:
        # Create parent directories if needed
        if create_dirs:
            filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata if requested
        if include_metadata:
            if isinstance(results, dict):
                results = results.copy()  # Don't modify original
                results['_metadata'] = {
                    'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'saved_by': 'feature_detection_system',
                    'format_version': '1.0'
                }
        
        # Handle different formats
        success = True
        
        if format in ['json', 'both']:
            json_path = filepath.with_suffix('.json')
            success &= _save_as_json(results, json_path)
            
        if format in ['pickle', 'both']:
            pickle_path = filepath.with_suffix('.pkl')
            success &= _save_as_pickle(results, pickle_path)
        
        return success
        
    except Exception as e:
        print(f"Error saving results to {filepath}: {e}")
        return False


def load_enhanced_results(filepath: Union[str, Path], 
                         format: str = 'auto') -> Optional[Any]:
    """
    Load benchmark results from file
    
    Args:
        filepath: Path to the results file
        format: Format to load ('json', 'pickle', 'auto')
        
    Returns:
        Loaded results object or None if failed
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"Error: File {filepath} does not exist")
        return None
    
    try:
        # Auto-detect format from extension
        if format == 'auto':
            if filepath.suffix.lower() == '.json':
                format = 'json'
            elif filepath.suffix.lower() in ['.pkl', '.pickle']:
                format = 'pickle'
            else:
                # Try JSON first, then pickle
                try:
                    return _load_from_json(filepath)
                except:
                    return _load_from_pickle(filepath)
        
        # Load specific format
        if format == 'json':
            return _load_from_json(filepath)
        elif format == 'pickle':
            return _load_from_pickle(filepath)
        else:
            print(f"Error: Unknown format '{format}'")
            return None
            
    except Exception as e:
        print(f"Error loading results from {filepath}: {e}")
        return None


def _save_as_json(data: Any, filepath: Path) -> bool:
    """Save data as JSON with custom serialization"""
    try:
        serializable_data = _make_json_serializable(data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved as JSON: {filepath}")
        return True
        
    except Exception as e:
        print(f"Error saving JSON to {filepath}: {e}")
        return False


def _save_as_pickle(data: Any, filepath: Path) -> bool:
    """Save data as pickle (preserves exact object structure)"""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Results saved as pickle: {filepath}")
        return True
        
    except Exception as e:
        print(f"Error saving pickle to {filepath}: {e}")
        return False


def _load_from_json(filepath: Path) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_from_pickle(filepath: Path) -> Any:
    """Load data from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def _make_json_serializable(obj: Any) -> Any:
    """
    Convert complex objects to JSON-serializable format
    
    Handles:
    - NumPy arrays and scalars
    - Dataclasses
    - Custom classes with __dict__
    - Nested structures
    """
    if obj is None:
        return None
    elif isinstance(obj, (bool, int, float, str)):
        return obj
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)  # Convert set to list
    elif isinstance(obj, np.ndarray):
        return {
            '_type': 'numpy_array',
            'data': obj.tolist(),
            'dtype': str(obj.dtype),
            'shape': obj.shape
        }
    elif type(obj).__name__ == 'MultiMethodMatchData':
        return {
            '_type': 'MultiMethodMatchData',
            'methods': obj.get_methods(),
            'stats': obj.get_stats(),
            'total_matches': len(obj),
            'per_method_matches': {
                method: _make_json_serializable(obj.per_method_matches[method])
                for method in obj.get_methods()
            }
        }
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert numpy scalars to Python scalars
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif is_dataclass(obj):
        return {
            '_type': 'dataclass',
            'class_name': obj.__class__.__name__,
            'data': _make_json_serializable(asdict(obj))
        }
    elif hasattr(obj, '__dict__'):
        # Handle custom classes
        return {
            '_type': 'custom_object',
            'class_name': obj.__class__.__name__,
            'data': _make_json_serializable(obj.__dict__)
        }
    elif hasattr(obj, '_asdict'):
        # Handle namedtuples
        return {
            '_type': 'namedtuple',
            'class_name': obj.__class__.__name__,
            'data': _make_json_serializable(obj._asdict())
        }
    else:
        # Fallback: try to convert to string
        try:
            return str(obj)
        except:
            return f"<non-serializable: {type(obj).__name__}>"


def save_benchmark_summary(results: Dict[str, Any], 
                          output_dir: str = "benchmark_results",
                          filename: str = None) -> str:
    """
    Save a human-readable summary of benchmark results
    
    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save the summary
        filename: Custom filename (auto-generated if None)
        
    Returns:
        str: Path to saved summary file
    """
    if filename is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"benchmark_summary_{timestamp}.txt"
    
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            _write_benchmark_summary(results, f)
        
        print(f"Benchmark summary saved: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"Error saving benchmark summary: {e}")
        return ""


def _write_benchmark_summary(results: Dict[str, Any], file_handle):
    """Write formatted benchmark summary to file"""
    file_handle.write("FEATURE DETECTION BENCHMARK SUMMARY\n")
    file_handle.write("=" * 50 + "\n\n")
    
    # Basic info
    file_handle.write(f"Timestamp: {results.get('timestamp', 'Unknown')}\n")
    file_handle.write(f"Methods: {', '.join(results.get('methods', []))}\n")
    file_handle.write(f"Benchmark Types: {', '.join(results.get('benchmark_types', []))}\n")
    
    # Image source info
    source_info = results.get('image_source_info', {})
    file_handle.write(f"Image Source: {source_info.get('type', 'Unknown')}\n")
    
    if 'folder_path' in source_info:
        file_handle.write(f"Folder: {source_info['folder_path']}\n")
    
    file_handle.write(f"Number of Images: {results.get('num_images', 'Unknown')}\n\n")
    
    # Performance results
    benchmarks = results.get('benchmarks', {})
    
    if 'performance' in benchmarks:
        performance = benchmarks['performance']
        summary = performance.get('summary', {})
        
        file_handle.write("PERFORMANCE RESULTS:\n")
        file_handle.write("-" * 30 + "\n")
        
        # Table header
        file_handle.write(f"{'Method':<12} {'Success':<8} {'Avg Time':<10} {'Features':<10} {'FPS':<8} {'Memory':<10}\n")
        file_handle.write("-" * 64 + "\n")
        
        for method, stats in summary.items():
            if 'error' not in stats:
                success_rate = stats['success_rate'] * 100
                avg_time = stats['time_stats']['mean']
                avg_features = stats['feature_stats']['mean']
                fps = stats['avg_fps']
                memory = stats.get('memory_stats', {}).get('mean', 0)
                
                file_handle.write(f"{method:<12} {success_rate:<8.1f}% {avg_time:<10.3f} {avg_features:<10.0f} {fps:<8.1f} {memory:<10.1f}\n")
            else:
                file_handle.write(f"{method:<12} {'ERROR':<8} {'-':<10} {'-':<10} {'-':<8} {'-':<10}\n")
        
        # Statistical comparisons
        comparisons = performance.get('statistical_comparisons', {})
        if comparisons:
            file_handle.write(f"\nSTATISTICAL COMPARISONS:\n")
            file_handle.write("-" * 25 + "\n")
            
            for comp_name, comp_data in comparisons.items():
                if 'statistical_test' in comp_data:
                    test = comp_data['statistical_test']
                    significance = "significant" if test['significant'] else "not significant"
                    file_handle.write(f"{comp_name}: p={test['p_value']:.4f} ({significance})\n")
    
    # Accuracy results
    if 'accuracy' in benchmarks:
        accuracy = benchmarks['accuracy']
        summary = accuracy.get('summary', {})
        
        file_handle.write(f"\nACCURACY RESULTS:\n")
        file_handle.write("-" * 20 + "\n")
        
        for method, stats in summary.items():
            if 'error' not in stats:
                file_handle.write(f"\n{method}:\n")
                file_handle.write(f"  Success Rate: {stats['success_rate']*100:.1f}%\n")
                file_handle.write(f"  Avg Quality: {stats['avg_quality']:.3f}\n")
                file_handle.write(f"  Avg Matches: {stats['avg_matches']:.1f}\n")
            else:
                file_handle.write(f"\n{method}: ERROR\n")
    
    # Comprehensive analysis
    if 'comprehensive_analysis' in results:
        analysis = results['comprehensive_analysis']
        rankings = analysis.get('method_rankings', {})
        
        file_handle.write(f"\nOVERALL RANKINGS:\n")
        file_handle.write("-" * 20 + "\n")
        
        if 'best_performance' in rankings:
            file_handle.write(f"Best Performance: {rankings['best_performance']}\n")
        if 'best_accuracy' in rankings:
            file_handle.write(f"Best Accuracy: {rankings['best_accuracy']}\n")
        
        if 'by_combined_score' in rankings:
            file_handle.write(f"\nCombined Rankings:\n")
            for i, (method, scores) in enumerate(rankings['by_combined_score'][:5]):
                file_handle.write(f"  {i+1}. {method}: {scores['combined_score']:.3f}\n")


def export_results_csv(results: Dict[str, Any], 
                      output_dir: str = "benchmark_results",
                      filename: str = None) -> str:
    """
    Export benchmark results to CSV format for analysis in spreadsheet software
    
    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save CSV
        filename: Custom filename (auto-generated if None)
        
    Returns:
        str: Path to saved CSV file
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas not available for CSV export")
        return ""
    
    if filename is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"benchmark_results_{timestamp}.csv"
    
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Extract performance data
        performance_data = []
        
        benchmarks = results.get('benchmarks', {})
        if 'performance' in benchmarks:
            summary = benchmarks['performance']['summary']
            
            for method, stats in summary.items():
                if 'error' not in stats:
                    row = {
                        'Method': method,
                        'Success_Rate': stats['success_rate'],
                        'Avg_Time_s': stats['time_stats']['mean'],
                        'Std_Time_s': stats['time_stats']['std'],
                        'Avg_Features': stats['feature_stats']['mean'],
                        'Std_Features': stats['feature_stats']['std'],
                        'FPS': stats['avg_fps'],
                        'Memory_MB': stats.get('memory_stats', {}).get('mean', 0)
                    }
                    performance_data.append(row)
        
        if performance_data:
            df = pd.DataFrame(performance_data)
            df.to_csv(output_path, index=False)
            print(f"Results exported to CSV: {output_path}")
            return str(output_path)
        else:
            print("No performance data to export")
            return ""
            
    except Exception as e:
        print(f"Error exporting to CSV: {e}")
        return ""


# Convenience function for common use case
def quick_save_results(results: Any, name: str, output_dir: str = "results") -> bool:
    """
    Quick save function with sensible defaults
    
    Args:
        results: Results to save
        name: Base name for the files
        output_dir: Output directory
        
    Returns:
        bool: Success status
    """
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"{name}_{timestamp}"
    filepath = Path(output_dir) / filename
    
    # Save in both JSON and pickle formats
    success = save_enhanced_results(results, filepath, format='both')
    
    # Also save summary if it's benchmark results
    if isinstance(results, dict) and 'benchmarks' in results:
        save_benchmark_summary(results, output_dir, f"{name}_summary_{timestamp}.txt")
        export_results_csv(results, output_dir, f"{name}_data_{timestamp}.csv")
    
    return success