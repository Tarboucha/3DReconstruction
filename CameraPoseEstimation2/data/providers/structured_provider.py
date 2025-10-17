"""
Provider for loading StructuredMatchData from FeatureMatchingExtraction batch files.

File: CameraPoseEstimation/data_providers/structured_provider.py
"""

import pickle
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from collections import OrderedDict
import numpy as np
import cv2

from CameraPoseEstimation2.core.interfaces import IMatchDataProvider, ValidationResult
from CameraPoseEstimation2.core.structures import StructuredMatchData, EnhancedDMatch, ScoreType


class LRUCache:
    """Simple LRU cache for match data"""
    
    def __init__(self, capacity: int = 100):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Remove least recently used
            self.cache.popitem(last=False)
    
    def clear(self):
        self.cache.clear()
    
    def size(self):
        return len(self.cache)


class StructuredMatchDataProvider(IMatchDataProvider):
    """
    Provider for loading match data from FeatureMatchingExtraction batch files.
    
    Features:
    - Lazy loading: Only loads data when requested
    - LRU caching: Keeps recently accessed pairs in memory
    - Efficient indexing: Fast pair lookup without loading all data
    - Quality filtering: Built-in filtering by quality/match count
    """
    
    def __init__(self, 
                 results_dir: str,
                 cache_size: int = 100,
                 preload_all: bool = False,
                 min_quality: Optional[float] = None,
                 min_matches: Optional[int] = None):
        """
        Initialize provider.
        
        Args:
            results_dir: Directory containing batch pickle files
            cache_size: Number of pairs to keep in memory (LRU)
            preload_all: If True, load all data immediately
            min_quality: Filter pairs below this quality
            min_matches: Filter pairs with fewer matches
        """
        self.results_dir = Path(results_dir)
        self.cache_size = cache_size
        self.min_quality = min_quality
        self.min_matches = min_matches
        
        # Initialize cache
        self._cache = LRUCache(cache_size)
        
        # Index structures (built during initialization)
        self._pair_to_file: Dict[Tuple[str, str], str] = {}  # pair -> batch file
        self._image_to_pairs: Dict[str, List[Tuple[str, str]]] = {}  # image -> pairs
        self._pair_metadata: Dict[Tuple[str, str], Dict] = {}  # Basic metadata
        self._image_info: Dict[str, Dict] = {}
        
        # Statistics
        self._statistics: Optional[Dict] = None
        
        # Build index
        self._build_index()
        
        # Preload if requested
        if preload_all:
            self._preload_all()
        
        print(f"✓ StructuredDataProvider initialized")
        print(f"  Directory: {results_dir}")
        print(f"  Pairs indexed: {len(self._pair_to_file)}")
        print(f"  Images: {len(self._image_info)}")
        print(f"  Cache size: {cache_size}")
        if min_quality:
            print(f"  Quality filter: ≥{min_quality}")
        if min_matches:
            print(f"  Match count filter: ≥{min_matches}")
    


    def _build_index(self):
        """Build index of pairs to files without loading all data"""
        print(f"Building index from: {self.results_dir}")
        
        # Load image metadata
        metadata_files = list(self.results_dir.glob("*_image_metadata.pkl"))
        if not metadata_files:
            raise FileNotFoundError(f"No image metadata file in {self.results_dir}")
        
        with open(metadata_files[0], 'rb') as f:
            image_metadata = pickle.load(f)
        
        # Build image info dictionary
        for img_data in image_metadata['images']:
            img_name = img_data['name']
            self._image_info[img_name] = {
                'name': img_name,
                'size': tuple(img_data['size']),
                'width': img_data['width'],
                'height': img_data['height'],
                'channels': img_data.get('channels', 3)
            }
        
        # Index batch files
        batch_files = sorted(self.results_dir.glob("*_batch_*.pkl"))
        if not batch_files:
            raise FileNotFoundError(f"No batch files in {self.results_dir}")
        
        print(f"  Found {len(batch_files)} batch files")
        
        for batch_file in batch_files:
            try:
                with open(batch_file, 'rb') as f:
                    batch_data = pickle.load(f)
                
                batch_results = batch_data.get('results', {})
                
                for pair_key in batch_results.keys():
                    # Convert string keys to tuples
                    if isinstance(pair_key, str):
                        try:
                            pair_key = eval(pair_key)
                        except:
                            continue
                    
                    if not isinstance(pair_key, tuple):
                        continue
                    
                    result = batch_results[pair_key]
                    
                    # Skip failed matches
                    if 'error' in result:
                        continue
                    
                    # Apply filters during indexing
                    num_matches = result.get('num_matches', 0)
                    quality = result.get('standardized_pair_quality', 
                                       result.get('quality_score', 0.5))
                    
                    if self.min_matches and num_matches < self.min_matches:
                        continue
                    if self.min_quality and quality < self.min_quality:
                        continue
                    
                    # Store mapping
                    self._pair_to_file[pair_key] = str(batch_file)
                    
                    # Store basic metadata (lightweight)
                    self._pair_metadata[pair_key] = {
                        'num_matches': num_matches,
                        'quality': quality,
                        'method': result.get('method', 'unknown')
                    }
                    
                    # Build image->pairs mapping
                    for img in pair_key:
                        if img not in self._image_to_pairs:
                            self._image_to_pairs[img] = []
                        self._image_to_pairs[img].append(pair_key)
                
            except Exception as e:
                print(f"  Warning: Could not index {batch_file.name}: {e}")
                continue
        
        print(f"  Indexed {len(self._pair_to_file)} valid pairs")
    
    def _load_pair_from_file(self, pair: Tuple[str, str]) -> StructuredMatchData:
        """Load a specific pair from its batch file"""
        if pair not in self._pair_to_file:
            raise KeyError(f"Pair {pair} not found in dataset")
        
        batch_file = self._pair_to_file[pair]
        
        try:
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
            
            batch_results = batch_data.get('results', {})
            
            # Handle both tuple and string keys
            result = None
            if pair in batch_results:
                result = batch_results[pair]
            else:
                # Try string representation
                pair_str = str(pair)
                if pair_str in batch_results:
                    result = batch_results[pair_str]
            
            if result is None or 'error' in result:
                raise ValueError(f"Invalid match data for pair {pair}")
            
            # Reconstruct StructuredMatchData
            structured_data = self._reconstruct_match_data(result)
            
            return structured_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to load pair {pair}: {e}")
    
    def _reconstruct_match_data(self, result: Dict) -> StructuredMatchData:
        """Reconstruct StructuredMatchData from batch result"""
        # Import the reconstruction function from load_matches module
        from ..loaders.match_loader import _reconstruct_structured_match_data, MatchQualityStandardizer
        
        standardizer = MatchQualityStandardizer()
        return _reconstruct_structured_match_data(
            result, 
            standardize=True,
            reconstruct_kp=True,
            standardizer=standardizer
        )
    
    def _preload_all(self):
        """Load all pairs into cache"""
        print("Preloading all pairs...")
        for i, pair in enumerate(self._pair_to_file.keys()):
            if i % 100 == 0:
                print(f"  Loaded {i}/{len(self._pair_to_file)} pairs...")
            self.get_match_data(pair)  # Will cache automatically
        print(f"✓ Preloaded {len(self._pair_to_file)} pairs")
    
    # =========================================================================
    # IMatchDataProvider Implementation
    # =========================================================================
    
    def get_match_data(self, pair: Tuple[str, str]) -> StructuredMatchData:
        """Get match data for a specific pair (with caching)"""
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
                    predicate: Optional[Callable] = None) -> Dict[Tuple[str, str], StructuredMatchData]:
        """Filter pairs based on criteria"""
        result = {}
        
        for pair, metadata in self._pair_metadata.items():
            # Apply simple filters using cached metadata
            if min_quality and metadata['quality'] < min_quality:
                continue
            if min_matches and metadata['num_matches'] < min_matches:
                continue
            
            # Apply custom predicate (requires loading data)
            if predicate:
                match_data = self.get_match_data(pair)
                if not predicate(pair, match_data):
                    continue
                result[pair] = match_data
            else:
                result[pair] = None  # Don't load yet
        
        return result
    
    def get_statistics(self) -> Dict:
        """Get global statistics"""
        if self._statistics is not None:
            return self._statistics.copy()
        
        # Compute statistics
        all_qualities = [m['quality'] for m in self._pair_metadata.values()]
        all_matches = [m['num_matches'] for m in self._pair_metadata.values()]
        
        self._statistics = {
            'total_pairs': len(self._pair_to_file),
            'total_images': len(self._image_info),
            'total_matches': sum(all_matches),
            'avg_matches_per_pair': np.mean(all_matches) if all_matches else 0,
            'quality_distribution': {
                'mean': float(np.mean(all_qualities)) if all_qualities else 0,
                'std': float(np.std(all_qualities)) if all_qualities else 0,
                'min': float(np.min(all_qualities)) if all_qualities else 0,
                'max': float(np.max(all_qualities)) if all_qualities else 0,
                'median': float(np.median(all_qualities)) if all_qualities else 0
            },
            'match_count_distribution': {
                'mean': float(np.mean(all_matches)) if all_matches else 0,
                'std': float(np.std(all_matches)) if all_matches else 0,
                'min': int(np.min(all_matches)) if all_matches else 0,
                'max': int(np.max(all_matches)) if all_matches else 0,
                'median': float(np.median(all_matches)) if all_matches else 0
            },
            'cache_stats': {
                'cache_size': self._cache.size(),
                'cache_capacity': self.cache_size
            }
        }
        
        return self._statistics.copy()
    
    def validate(self) -> ValidationResult:
        """Validate data integrity"""
        errors = []
        warnings = []
        
        # Check directory exists
        if not self.results_dir.exists():
            errors.append(f"Directory does not exist: {self.results_dir}")
            return ValidationResult(False, errors, warnings, {})
        
        # Check we have data
        if not self._pair_to_file:
            errors.append("No valid pairs found in dataset")
        
        if not self._image_info:
            errors.append("No image metadata found")
        
        # Check for orphaned pairs (pairs with missing images)
        for pair in self._pair_to_file.keys():
            for img in pair:
                if img not in self._image_info:
                    warnings.append(f"Pair {pair} references unknown image: {img}")
        
        # Sample check: try loading a few pairs
        sample_pairs = list(self._pair_to_file.keys())[:5]
        for pair in sample_pairs:
            try:
                match_data = self.get_match_data(pair)
                if match_data.num_matches == 0:
                    warnings.append(f"Pair {pair} has zero matches")
            except Exception as e:
                errors.append(f"Failed to load pair {pair}: {e}")
        
        # Statistics
        stats = self.get_statistics()
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, stats)
    
    # =========================================================================
    # Additional utility methods
    # =========================================================================
    
    def clear_cache(self):
        """Clear the LRU cache"""
        self._cache.clear()
        print("Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'current_size': self._cache.size(),
            'capacity': self.cache_size,
            'utilization': self._cache.size() / self.cache_size if self.cache_size > 0 else 0
        }
    
    def close(self):
        """Cleanup resources"""
        self.clear_cache()