"""
Folder-Based Match Data Provider

Implements IMatchDataProvider for folder-based feature matching output.
Integrates with ProviderFactory for automatic format detection.

Expected Input Structure:
    input_folder/
    ├── image_metadata.pkl           # Image metadata (root level)
    └── pairs/
        ├── (img1_img2).pkl          # Match data for each pair
        ├── (img1_img3).pkl
        └── ...

File: CameraPoseEstimation2/data/providers/folder_provider.py
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from collections import OrderedDict
import cv2

from CameraPoseEstimation2.core.interfaces import IMatchDataProvider, ValidationResult
from CameraPoseEstimation2.core.structures import StructuredMatchData, EnhancedDMatch
from CameraPoseEstimation2.core.structures import ScoreType as CPEScoreType

class LRUCache:
    """Simple LRU cache for match data"""
    
    def __init__(self, capacity: int = 100):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
    
    def clear(self):
        self.cache.clear()
    
    def size(self):
        return len(self.cache)


class FolderMatchDataProvider(IMatchDataProvider):
    """
    Provider for folder-based feature matching output.
    
    Reads from folder structure with metadata/ and pairs/ subdirectories.
    Implements IMatchDataProvider interface for seamless integration.
    
    Features:
    - Lazy loading: Only loads data when requested
    - LRU caching: Keeps recently accessed pairs in memory
    - Quality filtering: Built-in filtering by quality/match count
    - Auto-detection: Works with ProviderFactory.auto_detect()
    """
    
    def __init__(self, 
                 input_folder: str,
                 cache_size: int = 100,
                 preload_all: bool = False,
                 min_quality: Optional[float] = None,
                 min_matches: Optional[int] = None):
        """
        Initialize provider.
        
        Args:
            input_folder: Root folder with image_metadata.pkl and pairs/ subdir
            cache_size: LRU cache size
            preload_all: Load all pairs into memory immediately
            min_quality: Minimum quality threshold (filters during indexing)
            min_matches: Minimum match count threshold
        """
        self.input_folder = Path(input_folder)
        self.pairs_folder = self.input_folder / 'pairs'
        
        self.min_quality = min_quality
        self.min_matches = min_matches
        
        # Validate structure
        self._validate_structure()
        
        # Initialize cache
        self._cache = LRUCache(capacity=cache_size)
        
        # Load image metadata
        self._image_info = self._load_image_metadata()
        
        # Index pairs (lightweight - just file paths and metadata)
        self._pair_to_file = {}  # (img1, img2) -> file_path
        self._pair_metadata = {}  # (img1, img2) -> {quality, num_matches, method}
        self._image_to_pairs = {}  # img -> [(img1, img2), ...]
        
        self._index_pairs()
        
        print(f"✓ FolderMatchDataProvider initialized")
        print(f"  Images: {len(self._image_info)}")
        print(f"  Pairs: {len(self._pair_to_file)}")
        
        # Preload if requested
        if preload_all:
            self._preload_all()
    
    def __len__(self) -> int:
        """Return number of pairs"""
        return len(self._pair_to_file)


    def items(self):
        """Iterate over (pair_key, match_data) tuples"""
        for pair_key in self._pair_to_file.keys():
            yield pair_key, self.get_match_data(pair_key)

    def keys(self):
        """Return iterator over pair keys"""
        return self._pair_to_file.keys()

    def values(self):
        """Iterate over match data"""
        for pair_key in self._pair_to_file.keys():
            yield self.get_match_data(pair_key)

    def __iter__(self):
        """Iterate over pair keys"""
        return iter(self._pair_to_file.keys())

    def __getitem__(self, key):
        """Get match data by pair key: provider[pair_key]"""
        key_mapping = {
            'image1_size': 'image1_size',
            'image2_size': 'image2_size',
            'correspondences': 'correspondences',
            'num_matches': 'num_matches',
            # add other common keys
        }
        
        if key in key_mapping:
            attr_name = key_mapping[key]
            if hasattr(self, attr_name):
                value = getattr(self, attr_name)
                if value is None and key in ['image1_size', 'image2_size']:
                    return (640, 480)  # Default fallback
                return value
        
        if hasattr(self, key):
            return getattr(self, key)
    
        raise KeyError(f"'{key}' not found in StructuredMatchData")

    def _validate_structure(self):
        """Validate folder structure"""
        if not self.input_folder.exists():
            raise FileNotFoundError(f"Input folder not found: {self.input_folder}")
        
        if not self.pairs_folder.exists():
            raise FileNotFoundError(f"Pairs folder not found: {self.pairs_folder}")
        
        # Check for metadata pickle in root folder
        metadata_file = self.input_folder / 'image_metadata.pkl'
        if not metadata_file.exists():
            # Try alternative names
            alt_files = list(self.input_folder.glob('*metadata*.pkl'))
            if not alt_files:
                raise FileNotFoundError(f"No metadata pickle found in {self.input_folder}")
    
    def _load_image_metadata(self) -> Dict:
        """Load image metadata"""
        # Look for metadata file in root folder
        metadata_file = self.input_folder / 'image_metadata.pkl'
        if not metadata_file.exists():
            # Try alternative names
            alt_files = list(self.input_folder.glob('*metadata*.pkl'))
            if alt_files:
                metadata_file = alt_files[0]
            else:
                raise FileNotFoundError(f"No metadata file in {self.input_folder}")
        
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        # Convert to standard format
        image_info = {}
        
        if 'images' in metadata:
            for img_data in metadata['images']:
                img_name = img_data.get('name') or img_data.get('identifier')
                image_info[img_name] = {
                    'name': img_name,
                    'width': img_data.get('width', 0),
                    'height': img_data.get('height', 0),
                    'channels': img_data.get('channels', 3),
                    'size': (img_data.get('height', 0), img_data.get('width', 0)),
                    'filepath': img_data.get('filepath', '')
                }
        else:
            for img_name, img_data in metadata.items():
                if not isinstance(img_data, dict):
                    continue
                image_info[img_name] = {
                    'name': img_name,
                    'width': img_data.get('width', 0),
                    'height': img_data.get('height', 0),
                    'channels': img_data.get('channels', 3),
                    'size': img_data.get('size', (0, 0)),
                    'filepath': img_data.get('filepath', '')
                }
        
        return image_info
    
    def _index_pairs(self):
        """Index all pair files"""
        print("  Indexing pairs...")
        
        for pair_file in self.pairs_folder.glob('*.pkl'):
            # Parse filename: (img1_img2).pkl
            pair = self._parse_pair_filename(pair_file)
            if pair is None:
                continue
            
            # Quick metadata check without loading full data
            try:
                with open(pair_file, 'rb') as f:
                    pair_data = pickle.load(f)
                
                num_matches = pair_data.get('num_matches', 0)
                quality = pair_data.get('quality_score', 
                                       pair_data.get('standardized_pair_quality', 0.5))
                
                # Apply filters
                if self.min_matches and num_matches < self.min_matches:
                    continue
                if self.min_quality and quality < self.min_quality:
                    continue
                
                # Store mapping
                self._pair_to_file[pair] = str(pair_file)
                self._pair_metadata[pair] = {
                    'num_matches': num_matches,
                    'quality': quality,
                    'method': pair_data.get('method', 'unknown')
                }
                
                # Build image->pairs mapping
                for img in pair:
                    if img not in self._image_to_pairs:
                        self._image_to_pairs[img] = []
                    self._image_to_pairs[img].append(pair)
                    
            except Exception as e:
                print(f"    Warning: Failed to index {pair_file.name}: {e}")
                continue
    
    def _parse_pair_filename(self, filepath: Path) -> Optional[Tuple[str, str]]:
        """Parse pair filename to extract image names"""
        filename = filepath.stem
        
        if filename.startswith('(') and filename.endswith(')'):
            filename = filename[1:-1]
        
        # Find where first image name ends (look for .jpg_, .png_, etc.)
        import re
        match = re.match(r'(.+\.(jpg|png|jpeg|JPG|PNG))_(.+\.(jpg|png|jpeg|JPG|PNG))', filename)
        if match:
            return (match.group(1), match.group(3))
            
        return None
    
    def _load_pair_from_file(self, pair: Tuple[str, str]) -> StructuredMatchData:
        """Load and convert pair data from file"""
        # Get file path
        pair_file = self._pair_to_file.get(pair)
        if pair_file is None:
            # Try reverse
            pair_rev = (pair[1], pair[0])
            pair_file = self._pair_to_file.get(pair_rev)
            if pair_file is None:
                raise KeyError(f"Pair {pair} not found")
        
        # Load pickle
        with open(pair_file, 'rb') as f:
            pair_data = pickle.load(f)
        
        # Convert to StructuredMatchData
        return self._convert_to_structured_match_data(pair_data, pair)
    
    def _convert_to_structured_match_data(self, pair_data: Dict, 
                                     pair: Tuple[str, str]) -> StructuredMatchData:
        """Convert pair data to StructuredMatchData format"""
        
        # ADD THIS IMPORT AT THE TOP OF THE FILE (not inside the function)
        # from CameraPoseEstimation2.core.structures import ScoreType as CPEScoreType
        
        # Extract keypoints
        kp1 = self._extract_keypoints(pair_data.get('keypoints1'))
        kp2 = self._extract_keypoints(pair_data.get('keypoints2'))
        
        # Extract matches
        raw_matches = pair_data.get('matches', [])
        correspondences = pair_data.get('correspondences', None)
        
        # Build EnhancedDMatch list
        matches = []
        
        if correspondences is not None and len(correspondences) > 0:
            for i, corr in enumerate(correspondences):
                match = EnhancedDMatch(
                    queryIdx=i,
                    trainIdx=i,
                    score=1.0,
                    score_type=CPEScoreType.CONFIDENCE,  # CHANGE HERE
                    confidence=1.0,
                    standardized_quality=1.0,
                    source_method=pair_data.get('method', 'unknown')
                )
                matches.append(match)
        else:
            for match_item in raw_matches:
                if isinstance(match_item, (tuple, list)):
                    idx1, idx2 = int(match_item[0]), int(match_item[1])
                    score = 1.0
                elif hasattr(match_item, 'queryIdx'):
                    idx1 = match_item.queryIdx
                    idx2 = match_item.trainIdx
                    score = 1.0 - match_item.distance if hasattr(match_item, 'distance') else 1.0
                else:
                    continue
                
                match = EnhancedDMatch(
                    queryIdx=idx1,
                    trainIdx=idx2,
                    score=score,
                    score_type=CPEScoreType.CONFIDENCE,  # CHANGE HERE
                    confidence=score,
                    standardized_quality=score,
                    source_method=pair_data.get('method', 'unknown')
                )
                matches.append(match)
        
        # Quality metrics
        quality = pair_data.get('quality_score', 
                            pair_data.get('standardized_pair_quality', 0.5))
        
        quality_stats = {
            'mean': quality,
            'std': 0.1,
            'min': max(0.0, quality - 0.2),
            'max': min(1.0, quality + 0.2),
            'median': quality
        }
        
        match_data = StructuredMatchData(
            matches=matches,
            keypoints1=kp1,
            keypoints2=kp2,
            method=pair_data.get('method', 'unknown'),
            score_type=CPEScoreType.CONFIDENCE,  # CHANGE HERE
            num_matches=len(matches),
            standardized_pair_quality=quality,
            match_quality_stats=quality_stats,
            matching_time=pair_data.get('matching_time', 0.0)
        )

        img1_info = self._image_info.get(pair[0], {})
        img2_info = self._image_info.get(pair[1], {})
        match_data.image1_size = (img1_info.get('width', 0), img1_info.get('height', 0))
        match_data.image2_size = (img2_info.get('width', 0), img2_info.get('height', 0))
        
        return match_data

    
    def _extract_keypoints(self, keypoints) -> np.ndarray:
        """Extract keypoint coordinates"""
        if keypoints is None:
            return np.empty((0, 2), dtype=np.float32)
        
        if isinstance(keypoints, np.ndarray):
            if keypoints.ndim == 2 and keypoints.shape[1] >= 2:
                return keypoints[:, :2].astype(np.float32)
            return keypoints.astype(np.float32)
        
        if isinstance(keypoints, list):
            if len(keypoints) == 0:
                return np.empty((0, 2), dtype=np.float32)
            
            if isinstance(keypoints[0], cv2.KeyPoint):
                return np.array([kp.pt for kp in keypoints], dtype=np.float32)
            elif isinstance(keypoints[0], (tuple, list)):
                return np.array(keypoints, dtype=np.float32)
        
        return np.array(keypoints, dtype=np.float32)
    
    def _preload_all(self):
        """Preload all pairs into cache"""
        print(f"  Preloading {len(self._pair_to_file)} pairs...")
        for i, pair in enumerate(self._pair_to_file.keys(), 1):
            if i % 100 == 0:
                print(f"    Loaded {i}/{len(self._pair_to_file)} pairs...")
            self.get_match_data(pair)
        print(f"  ✓ Preloaded all pairs")
    
    # =========================================================================
    # IMatchDataProvider Implementation
    # =========================================================================
    
    def get_match_data(self, pair: Tuple[str, str]) -> StructuredMatchData:
        """Get match data for a pair (with caching)"""
        # Check cache first
        cached = self._cache.get(pair)
        if cached is not None:
            return cached
        
        # Load from file
        match_data = self._load_pair_from_file(pair)
        
        # Cache it
        self._cache.put(pair, match_data)
        
        return match_data
    
    def get_all_pairs(self) -> List[Tuple[str, str]]:
        """Get all available pairs"""
        return list(self._pair_to_file.keys())
    
    def get_image_info(self, image_name: str) -> Dict:
        """Get metadata for a specific image"""
        if image_name not in self._image_info:
            raise KeyError(f"Image {image_name} not found")
        return self._image_info[image_name].copy()
    
    def get_all_images(self) -> List[str]:
        """Get all image names"""
        return list(self._image_info.keys())
    
    def get_pairs_with_image(self, image_name: str) -> List[Tuple[str, str]]:
        """Get all pairs containing a specific image"""
        return self._image_to_pairs.get(image_name, []).copy()
    
    def get_pairs_for_image(self, image_name: str) -> List[Tuple[str, str]]:
        """Alias for get_pairs_with_image (required by interface)"""
        return self.get_pairs_with_image(image_name)
    
    def has_pair(self, pair: Tuple[str, str]) -> bool:
        """Check if a pair exists"""
        if pair in self._pair_to_file:
            return True
        # Check reverse
        pair_rev = (pair[1], pair[0])
        return pair_rev in self._pair_to_file
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if not self._pair_metadata:
            return {
                'total_pairs': 0,
                'total_images': len(self._image_info),
                'total_matches': 0
            }
        
        qualities = [m['quality'] for m in self._pair_metadata.values()]
        match_counts = [m['num_matches'] for m in self._pair_metadata.values()]
        
        return {
            'total_pairs': len(self._pair_to_file),
            'total_images': len(self._image_info),
            'total_matches': sum(match_counts),
            'avg_matches_per_pair': float(np.mean(match_counts)) if match_counts else 0,
            'quality_distribution': {
                'mean': float(np.mean(qualities)),
                'std': float(np.std(qualities)),
                'min': float(np.min(qualities)),
                'max': float(np.max(qualities)),
                'median': float(np.median(qualities))
            },
            'match_distribution': {
                'mean': float(np.mean(match_counts)),
                'std': float(np.std(match_counts)),
                'min': int(np.min(match_counts)),
                'max': int(np.max(match_counts)),
                'median': float(np.median(match_counts))
            },
            'cache_stats': {
                'cache_size': self._cache.size(),
                'cache_capacity': self._cache.capacity
            }
        }
    
    def get_best_pairs(self, k: int = 10, criterion: str = 'quality') -> List[Tuple[str, str]]:
        """Get top-k pairs by criterion"""
        if criterion == 'quality':
            sorted_pairs = sorted(
                self._pair_metadata.items(),
                key=lambda x: x[1]['quality'],
                reverse=True
            )
        elif criterion == 'match_count':
            sorted_pairs = sorted(
                self._pair_metadata.items(),
                key=lambda x: x[1]['num_matches'],
                reverse=True
            )
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        return [pair for pair, _ in sorted_pairs[:k]]
    
    def filter_pairs(self,
                    min_quality: Optional[float] = None,
                    min_matches: Optional[int] = None,
                    max_pairs: Optional[int] = None,
                    images: Optional[List[str]] = None) -> List[Tuple[str, str]]:
        """Filter pairs by criteria"""
        filtered = []
        
        for pair, meta in self._pair_metadata.items():
            # Quality filter
            if min_quality and meta['quality'] < min_quality:
                continue
            
            # Match count filter
            if min_matches and meta['num_matches'] < min_matches:
                continue
            
            # Image filter
            if images and not any(img in pair for img in images):
                continue
            
            filtered.append(pair)
        
        # Limit number of pairs
        if max_pairs:
            filtered = filtered[:max_pairs]
        
        return filtered
    
    def validate(self) -> ValidationResult:
        """Validate provider data"""
        errors = []
        warnings = []
        
        # Check image info
        if not self._image_info:
            errors.append("No image metadata found")
        
        # Check pairs
        if not self._pair_to_file:
            errors.append("No pairs found")
        
        # Check for orphaned images
        for img in self._image_info.keys():
            if img not in self._image_to_pairs:
                warnings.append(f"Image {img} has no pairs")
        
        # Sample validation
        if self._pair_to_file:
            try:
                sample_pair = next(iter(self._pair_to_file.keys()))
                self.get_match_data(sample_pair)
            except Exception as e:
                errors.append(f"Failed to load sample pair: {e}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def get_image_count(self) -> int:
        """Get number of images"""
        return len(self._image_info)
    
    def get_pair_count(self) -> int:
        """Get number of pairs"""
        return len(self._pair_to_file)
    
    def print_summary(self):
        """Print provider summary"""
        print(f"\n{'='*70}")
        print(f"FolderMatchDataProvider Summary")
        print(f"{'='*70}")
        print(f"Source: {self.input_folder}")
        print(f"Images: {self.get_image_count()}")
        print(f"Pairs: {self.get_pair_count()}")
        print(f"Cache size: {self._cache.size()}/{self._cache.capacity}")
        
        if self._pair_metadata:
            qualities = [m['quality'] for m in self._pair_metadata.values()]
            match_counts = [m['num_matches'] for m in self._pair_metadata.values()]
            
            print(f"\nQuality: {np.mean(qualities):.3f} ± {np.std(qualities):.3f}")
            print(f"  Range: [{np.min(qualities):.3f}, {np.max(qualities):.3f}]")
            print(f"Matches: {np.mean(match_counts):.1f} ± {np.std(match_counts):.1f}")
            print(f"  Range: [{np.min(match_counts)}, {np.max(match_counts)}]")
        
        print(f"{'='*70}\n")


# =============================================================================
# Factory Integration
# =============================================================================

def is_folder_format(path: Path) -> bool:
    """Check if path is folder-based format"""
    if not path.is_dir():
        return False
    
    pairs_folder = path / 'pairs'
    if not pairs_folder.exists():
        return False
    
    # Check for metadata pickle in root folder
    metadata_file = path / 'image_metadata.pkl'
    if not metadata_file.exists():
        # Try alternative names
        alt_files = list(path.glob('*metadata*.pkl'))
        if not alt_files:
            return False
    
    # Check for pair pickles
    pair_files = list(pairs_folder.glob('*.pkl'))
    if not pair_files:
        return False
    
    return True


# Add this to make it easily importable
__all__ = ['FolderMatchDataProvider', 'is_folder_format']


# Register with factory (add to factory.py)
# ProviderFactory.register('folder', FolderMatchDataProvider, is_folder_format)