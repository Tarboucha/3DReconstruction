"""
Image Management Module with Batch Loading and Smart Caching

This module provides efficient image loading for batch processing:
- Metadata-based scanning (doesn't load images upfront)
- Smart batch loading (only loads images needed for current batch)
- Cache management (reuses images across pairs in same batch)
- Memory efficient (bounded memory regardless of dataset size)

Key Classes:
- ImageMetadata: Lightweight metadata (no pixel data)
- ImageCache: Manages loaded images with LRU-style eviction
- BatchImageLoader: Loads batches of images efficiently
- FolderImageSource: Scans folders and creates pairs from metadata
"""

import cv2
import numpy as np
import glob
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set, Iterator, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


# =============================================================================
# ENUMS AND BASIC DATA STRUCTURES
# =============================================================================

class ImageSourceType(Enum):
    """Types of image sources"""
    SYNTHETIC = "synthetic"
    FOLDER = "folder"
    SINGLE_IMAGE = "single_image"
    IMAGE_LIST = "image_list"
    CUSTOM = "custom"


@dataclass
class ImageMetadata:
    """
    Lightweight image metadata (no pixel data!)
    
    Used for scanning and pair creation without loading images.
    Minimal memory footprint: ~500 bytes per image vs ~10MB for actual image.
    """
    filepath: Path
    identifier: str
    original_size: Optional[Tuple[int, int]] = None  # (width, height)
    file_size_bytes: Optional[int] = None
    source_type: ImageSourceType = ImageSourceType.FOLDER
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        """Make hashable for use in sets/dicts"""
        return hash(self.identifier)
    
    def __eq__(self, other):
        """Equality based on identifier"""
        if not isinstance(other, ImageMetadata):
            return False
        return self.identifier == other.identifier


@dataclass
class ImageInfo:
    """
    Full image information with pixel data
    
    Used during actual processing. Contains the loaded image array.
    """
    image: np.ndarray
    identifier: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_type: ImageSourceType = ImageSourceType.CUSTOM
    
    @property
    def size(self) -> Tuple[int, int]:
        """Return (width, height)"""
        return (self.image.shape[1], self.image.shape[0])
    
    @property
    def channels(self) -> int:
        return self.image.shape[2] if len(self.image.shape) > 2 else 1
    
    @property
    def memory_size_mb(self) -> float:
        """Approximate memory size in MB"""
        return self.image.nbytes / (1024 * 1024)


# =============================================================================
# IMAGE CACHE MANAGER
# =============================================================================

class ImageCache:
    """
    Manages loaded images with size-based eviction
    
    Keeps track of loaded images and automatically evicts when cache is full.
    Uses simple FIFO eviction strategy.
    
    Example:
        >>> cache = ImageCache(max_size_mb=500)
        >>> cache.add('img1.jpg', image_array)
        >>> img = cache.get('img1.jpg')
        >>> cache.clear()
    """
    
    def __init__(self, max_size_mb: float = 500):
        """
        Initialize cache
        
        Args:
            max_size_mb: Maximum cache size in megabytes
        """
        self.max_size_mb = max_size_mb
        self.cache: Dict[str, ImageInfo] = {}
        self.load_order: List[str] = []  # Track insertion order for FIFO
        self._current_size_mb = 0.0
    
    def add(self, identifier: str, image_info: ImageInfo):
        """Add image to cache"""
        # Don't add if already in cache
        if identifier in self.cache:
            return
        
        image_size = image_info.memory_size_mb
        
        # Evict if necessary
        while self._current_size_mb + image_size > self.max_size_mb and self.load_order:
            self._evict_oldest()
        
        # Add to cache
        self.cache[identifier] = image_info
        self.load_order.append(identifier)
        self._current_size_mb += image_size
    
    def get(self, identifier: str) -> Optional[ImageInfo]:
        """Get image from cache"""
        return self.cache.get(identifier)
    
    def contains(self, identifier: str) -> bool:
        """Check if image is in cache"""
        return identifier in self.cache
    
    def _evict_oldest(self):
        """Evict oldest image from cache"""
        if not self.load_order:
            return
        
        oldest_id = self.load_order.pop(0)
        if oldest_id in self.cache:
            evicted = self.cache.pop(oldest_id)
            self._current_size_mb -= evicted.memory_size_mb
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self.load_order.clear()
        self._current_size_mb = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'num_images': len(self.cache),
            'current_size_mb': round(self._current_size_mb, 2),
            'max_size_mb': self.max_size_mb,
            'usage_percent': round(100 * self._current_size_mb / self.max_size_mb, 1)
        }
    
    def __len__(self):
        return len(self.cache)
    
    def __repr__(self):
        return f"ImageCache({len(self)} images, {self._current_size_mb:.1f}/{self.max_size_mb}MB)"


# =============================================================================
# BATCH IMAGE LOADER
# =============================================================================

class BatchImageLoader:
    """
    Efficiently loads batches of images with smart caching
    
    Key features:
    - Identifies unique images needed for a batch of pairs
    - Loads only necessary images
    - Maintains cache for reuse within and across batches
    - Handles resize operations
    
    Example:
        >>> loader = BatchImageLoader(cache_size_mb=500)
        >>> pairs_metadata = [(meta1, meta2), (meta2, meta3), ...]
        >>> loader.load_batch(pairs_metadata, resize_to=(1024, 768))
        >>> img1 = loader.get_image('img1.jpg')
        >>> loader.clear_batch()
    """
    
    def __init__(self, cache_size_mb: float = 500):
        """
        Initialize batch loader
        
        Args:
            cache_size_mb: Maximum cache size in megabytes
        """
        self.cache = ImageCache(max_size_mb=cache_size_mb)
    
    def load_batch(
        self,
        pairs_metadata: List[Tuple[ImageMetadata, ImageMetadata]],
        resize_to: Optional[Tuple[int, int]] = None
    ) -> int:
        """
        Load all images needed for a batch of pairs
        
        Args:
            pairs_metadata: List of (metadata1, metadata2) tuples
            resize_to: Optional (width, height) to resize images
        
        Returns:
            Number of images loaded (excludes cached)
        """
        # Find unique images needed
        unique_metadata = self._find_unique_images(pairs_metadata)
        
        # Load images not already in cache
        num_loaded = 0
        for metadata in unique_metadata:
            if not self.cache.contains(metadata.identifier):
                image_info = self._load_single_image(metadata, resize_to)
                if image_info:
                    self.cache.add(metadata.identifier, image_info)
                    num_loaded += 1
        
        return num_loaded
    
    def _find_unique_images(
        self,
        pairs_metadata: List[Tuple[ImageMetadata, ImageMetadata]]
    ) -> Set[ImageMetadata]:
        """Find unique images needed for batch"""
        unique = set()
        for meta1, meta2 in pairs_metadata:
            unique.add(meta1)
            unique.add(meta2)
        return unique
    
    def _load_single_image(
        self,
        metadata: ImageMetadata,
        resize_to: Optional[Tuple[int, int]] = None
    ) -> Optional[ImageInfo]:
        """Load a single image from disk"""
        try:
            # Load image
            image = cv2.imread(str(metadata.filepath))
            if image is None:
                print(f"⚠️  Could not load: {metadata.filepath}")
                return None
            
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Store original size
            original_size = (image.shape[1], image.shape[0])  # (width, height)
            
            # Resize if requested
            if resize_to is not None:
                image = cv2.resize(image, resize_to, interpolation=cv2.INTER_LINEAR)
            
            # Create ImageInfo
            image_info = ImageInfo(
                image=image,
                identifier=metadata.identifier,
                metadata={
                    'filepath': str(metadata.filepath),
                    'original_size': original_size,
                    'resized': resize_to is not None,
                    'final_size': (image.shape[1], image.shape[0]),
                    **metadata.metadata
                },
                source_type=metadata.source_type
            )
            
            return image_info
            
        except Exception as e:
            print(f"⚠️  Error loading {metadata.filepath}: {e}")
            return None
    
    def get_image(self, identifier: str) -> Optional[ImageInfo]:
        """Get loaded image by identifier"""
        return self.cache.get(identifier)
    
    def clear_batch(self):
        """Clear cache (call after batch processing)"""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()


# =============================================================================
# FOLDER IMAGE SOURCE WITH METADATA SUPPORT
# =============================================================================

class FolderImageSource:
    """
    Efficient folder-based image source with metadata scanning
    
    Two modes:
    1. Metadata mode: Scan folder and collect metadata only (fast, low memory)
    2. Full mode: Load all images (backward compatible, high memory)
    
    Example:
        >>> source = FolderImageSource('./images')
        >>> 
        >>> # Metadata mode (efficient)
        >>> metadata_list = source.get_metadata_list()  # Fast, ~1MB for 1000 images
        >>> 
        >>> # Full mode (backward compatible)
        >>> images = source.get_image_list()  # Slow, loads all images
    """
    
    def __init__(
        self,
        folder_path: str,
        max_images: Optional[int] = None,
        resize_to: Optional[Tuple[int, int]] = None,
        image_extensions: List[str] = None
    ):
        """
        Initialize folder image source
        
        Args:
            folder_path: Path to folder containing images
            max_images: Maximum number of images to load (None for all)
            resize_to: Resize images to (width, height)
            image_extensions: List of file extensions to search for
        """
        self.folder_path = Path(folder_path)
        self.max_images = max_images
        self.resize_to = resize_to
        self.image_extensions = image_extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        if not self.folder_path.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")
    
    def get_metadata_list(self) -> List[ImageMetadata]:
        """
        Get metadata for all images WITHOUT loading pixel data
        
        This is FAST and uses minimal memory (~500 bytes per image).
        Use this for batch processing with lazy loading.
        
        Returns:
            List of ImageMetadata objects
        """
        # Find all image files
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(glob.glob(str(self.folder_path / f"*{ext}")))
            image_files.extend(glob.glob(str(self.folder_path / f"*{ext.upper()}")))
        
        image_files = sorted(list(set(image_files)))
        
        if self.max_images:
            image_files = image_files[:self.max_images]
        
        # Create metadata objects
        metadata_list = []
        for img_file in image_files:
            try:
                filepath = Path(img_file)
                
                # Get file size
                file_size = filepath.stat().st_size
                
                # Create metadata (no image loading!)
                metadata = ImageMetadata(
                    filepath=filepath,
                    identifier=filepath.name,
                    file_size_bytes=file_size,
                    source_type=ImageSourceType.FOLDER,
                    metadata={'folder': str(self.folder_path)}
                )
                
                metadata_list.append(metadata)
                
            except Exception as e:
                print(f"⚠️  Error reading metadata for {img_file}: {e}")
        
        return metadata_list
    
    def get_image_list(self) -> List[ImageInfo]:
        """
        Get all images as a list (loads all into memory)
        
        BACKWARD COMPATIBLE: Same as old behavior.
        WARNING: Loads ALL images into RAM! Use get_metadata_list() instead.
        
        Returns:
            List of ImageInfo objects with loaded images
        """
        metadata_list = self.get_metadata_list()
        
        images = []
        for metadata in metadata_list:
            try:
                # Load image
                image = cv2.imread(str(metadata.filepath))
                if image is None:
                    print(f"⚠️  Could not load image {metadata.filepath}")
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                original_size = (image.shape[1], image.shape[0])
                
                if self.resize_to:
                    image = cv2.resize(image, self.resize_to, interpolation=cv2.INTER_LINEAR)
                
                image_info = ImageInfo(
                    image=image,
                    identifier=metadata.identifier,
                    metadata={
                        'filepath': str(metadata.filepath),
                        'original_size': original_size,
                        'resized': self.resize_to is not None,
                        'final_size': (image.shape[1], image.shape[0])
                    },
                    source_type=ImageSourceType.FOLDER
                )
                
                images.append(image_info)
                
            except Exception as e:
                print(f"⚠️  Error loading {metadata.filepath}: {e}")
        
        return images
    
    def get_source_info(self) -> Dict[str, Any]:
        """Get information about this image source"""
        return {
            'type': 'folder',
            'folder_path': str(self.folder_path),
            'extensions': self.image_extensions,
            'resize_to': self.resize_to,
            'max_images': self.max_images
        }


# =============================================================================
# PAIR CREATION HELPERS
# =============================================================================

def create_pairs_from_metadata(
    metadata_list: List[ImageMetadata],
    pair_mode: str = 'consecutive'
) -> List[Tuple[ImageMetadata, ImageMetadata]]:
    """
    Create pairs from metadata (no image loading)
    
    Args:
        metadata_list: List of ImageMetadata objects
        pair_mode: 'consecutive', 'first', or 'all'
    
    Returns:
        List of (metadata1, metadata2) tuples
    """
    if len(metadata_list) < 2:
        return []
    
    pairs = []
    
    if pair_mode == 'consecutive':
        # (img1,img2), (img2,img3), (img3,img4), ...
        for i in range(len(metadata_list) - 1):
            pairs.append((metadata_list[i], metadata_list[i + 1]))
    
    elif pair_mode == 'first':
        # (img1,img2), (img1,img3), (img1,img4), ...
        first = metadata_list[0]
        for i in range(1, len(metadata_list)):
            pairs.append((first, metadata_list[i]))
    
    elif pair_mode == 'all':
        # All possible pairs (combinatorial)
        from itertools import combinations
        pairs = list(combinations(metadata_list, 2))
    
    else:
        raise ValueError(f"Unknown pair_mode: {pair_mode}")
    
    return pairs


def analyze_batch_reuse(
    pairs_metadata: List[Tuple[ImageMetadata, ImageMetadata]]
) -> Dict[str, Any]:
    """
    Analyze how many unique images are needed for a batch
    
    Useful for understanding caching efficiency.
    
    Args:
        pairs_metadata: List of metadata pairs
    
    Returns:
        Dictionary with analysis results
    """
    if not pairs_metadata:
        return {'num_pairs': 0, 'unique_images': 0, 'reuse_ratio': 0.0}
    
    # Count total image references
    total_refs = len(pairs_metadata) * 2
    
    # Find unique images
    unique = set()
    for meta1, meta2 in pairs_metadata:
        unique.add(meta1.identifier)
        unique.add(meta2.identifier)
    
    num_unique = len(unique)
    reuse_ratio = (total_refs - num_unique) / total_refs if total_refs > 0 else 0.0
    
    return {
        'num_pairs': len(pairs_metadata),
        'total_image_refs': total_refs,
        'unique_images': num_unique,
        'reuse_ratio': reuse_ratio,
        'avg_reuse_per_image': total_refs / num_unique if num_unique > 0 else 0.0
    }


def estimate_batch_memory(
    pairs_metadata: List[Tuple[ImageMetadata, ImageMetadata]],
    avg_image_size_mb: float = 10.0
) -> Dict[str, Any]:
    """
    Estimate memory required for a batch
    
    Args:
        pairs_metadata: List of metadata pairs
        avg_image_size_mb: Average image size in MB (default: 10MB for 4K image)
    
    Returns:
        Memory estimates
    """
    analysis = analyze_batch_reuse(pairs_metadata)
    
    num_unique = analysis['unique_images']
    estimated_memory_mb = num_unique * avg_image_size_mb
    
    return {
        'unique_images': num_unique,
        'avg_image_size_mb': avg_image_size_mb,
        'estimated_memory_mb': round(estimated_memory_mb, 2),
        'estimated_memory_gb': round(estimated_memory_mb / 1024, 2)
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def scan_folder_quick(folder_path: str, extensions: List[str] = None) -> Dict[str, Any]:
    """
    Quick scan of folder to get image count and size
    
    Args:
        folder_path: Path to folder
        extensions: Image extensions to look for
    
    Returns:
        Dictionary with folder statistics
    """
    extensions = extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder not found: {folder_path}")
    
    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(str(folder / f"*{ext}")))
        image_files.extend(glob.glob(str(folder / f"*{ext.upper()}")))
    
    image_files = list(set(image_files))
    
    # Calculate total size
    total_size_bytes = sum(Path(f).stat().st_size for f in image_files)
    
    return {
        'folder': str(folder),
        'num_images': len(image_files),
        'total_size_mb': round(total_size_bytes / (1024 * 1024), 2),
        'total_size_gb': round(total_size_bytes / (1024 * 1024 * 1024), 2),
        'avg_file_size_mb': round(total_size_bytes / len(image_files) / (1024 * 1024), 2) if image_files else 0
    }


# =============================================================================
# TESTING AND EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """Test the image management system"""
    
    print("="*70)
    print("IMAGE MANAGER TEST")
    print("="*70)
    
    # Test 1: Metadata scanning
    print("\n1. Testing metadata scanning...")
    try:
        source = FolderImageSource('./test_images')
        metadata_list = source.get_metadata_list()
        print(f"✓ Found {len(metadata_list)} images")
        
        if metadata_list:
            print(f"  Sample: {metadata_list[0].identifier}")
            print(f"  File size: {metadata_list[0].file_size_bytes / 1024:.2f} KB")
    except Exception as e:
        print(f"✗ {e}")
    
    # Test 2: Pair creation
    print("\n2. Testing pair creation...")
    if len(metadata_list) >= 2:
        pairs = create_pairs_from_metadata(metadata_list[:10], 'consecutive')
        print(f"✓ Created {len(pairs)} pairs")
        
        # Analyze reuse
        analysis = analyze_batch_reuse(pairs)
        print(f"  Unique images: {analysis['unique_images']}")
        print(f"  Reuse ratio: {analysis['reuse_ratio']:.1%}")
    
    # Test 3: Batch loading
    print("\n3. Testing batch loading...")
    loader = BatchImageLoader(cache_size_mb=100)
    
    if len(pairs) > 0:
        batch = pairs[:3]
        num_loaded = loader.load_batch(batch, resize_to=(640, 480))
        print(f"✓ Loaded {num_loaded} new images")
        print(f"  Cache stats: {loader.get_cache_stats()}")
    
    print("\n" + "="*70)
    print("✓ ALL TESTS COMPLETE")
    print("="*70)