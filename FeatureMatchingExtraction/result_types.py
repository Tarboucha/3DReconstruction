"""
Result Types - Modern Multi-Method Design

Clean, simple result types with dictionary-like interface.
All methods are treated equally - no primary/alternative distinction.

Version: 3.0.0 (Complete rewrite)
"""

import cv2
import numpy as np
import pickle
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from .core_data_structures import FeatureData, MatchData, ScoreType


# =============================================================================
# METADATA CLASSES
# =============================================================================

@dataclass
class ImagePairInfo:
    """Information about an image pair"""
    image1_id: str
    image2_id: str
    image1_size: Tuple[int, int]  # (width, height)
    image2_size: Tuple[int, int]
    
    # Optional
    image1: Optional[np.ndarray] = None
    image2: Optional[np.ndarray] = None
    image1_path: Optional[str] = None
    image2_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingMetadata:
    """Pipeline processing metadata"""
    methods_used: List[str]
    total_processing_time: float
    config_used: Dict[str, Any] = field(default_factory=dict)
    pipeline_version: str = "3.0.0"
    timestamp: str = field(default_factory=lambda: time.strftime('%Y-%m-%d %H:%M:%S'))
    method_timings: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# METHOD RESULT
# =============================================================================

@dataclass
class MethodResult:
    """
    Result from a single method
    
    Contains features, matches, and geometry for one detection/matching method.
    """
    method_name: str
    features1: FeatureData
    features2: FeatureData
    match_data: MatchData
    
    # Optional geometry
    homography: Optional[np.ndarray] = None
    fundamental_matrix: Optional[np.ndarray] = None
    inlier_ratio: Optional[float] = None
    reprojection_error: Optional[float] = None
    
    # Timing
    detection_time: float = 0.0
    matching_time: float = 0.0
    
    @property
    def num_features1(self) -> int:
        return len(self.features1.keypoints)
    
    @property
    def num_features2(self) -> int:
        return len(self.features2.keypoints)
    
    @property
    def num_matches(self) -> int:
        """Number of filtered matches (or raw if no filtering)"""
        return len(self.match_data.get_best_matches())
    
    @property
    def num_raw_matches(self) -> int:
        """Number of raw matches before filtering"""
        return len(self.match_data.matches)
    
    @property
    def total_time(self) -> float:
        return self.detection_time + self.matching_time
    
    def get_quality_score(self) -> float:
        """Compute overall quality score for ranking"""
        score = 0.0
        
        # Matches (normalized to 0-1)
        if self.num_matches > 0:
            score += min(self.num_matches / 500.0, 1.0) * 0.4
        
        # Inlier ratio
        if self.inlier_ratio is not None:
            score += self.inlier_ratio * 0.4
        
        # Reprojection error (inverse - lower is better)
        if self.reprojection_error is not None:
            score += max(0, 1.0 - self.reprojection_error / 10.0) * 0.2
        
        return score


# =============================================================================
# MATCHING RESULT - Main Container
# =============================================================================

class MatchingResult:
    """
    Multi-method matching result with dictionary-like interface
    
    All methods are treated equally. Access any method directly:
        result['SIFT']
        result['ALIKED']
    
    No artificial primary/alternative distinction.
    """
    
    def __init__(self,
                 methods: Dict[str, MethodResult],
                 image_pair_info: ImagePairInfo,
                 processing_metadata: ProcessingMetadata):
        self.methods = methods
        self.image_pair_info = image_pair_info
        self.processing_metadata = processing_metadata
    
    # -------------------------------------------------------------------------
    # Dictionary Interface
    # -------------------------------------------------------------------------
    
    def __getstate__(self):
        """Custom pickle serialization"""
        from .utils import keypoints_to_list
        
        state = self.__dict__.copy()
        
        # Convert KeyPoint objects to serializable format
        serialized_methods = {}
        for name, method in self.methods.items():
            serialized_method = {
                'method_name': method.method_name,
                'features1': {
                    'keypoints': keypoints_to_list(method.features1.keypoints),
                    'descriptors': method.features1.descriptors,
                    'method': method.features1.method,
                    'confidence_scores': method.features1.confidence_scores,
                    'detection_time': method.features1.detection_time,
                },
                'features2': {
                    'keypoints': keypoints_to_list(method.features2.keypoints),
                    'descriptors': method.features2.descriptors,
                    'method': method.features2.method,
                    'confidence_scores': method.features2.confidence_scores,
                    'detection_time': method.features2.detection_time,
                },
                'match_data': method.match_data,
                'homography': method.homography,
                'fundamental_matrix': method.fundamental_matrix,
                'inlier_ratio': method.inlier_ratio,
                'reprojection_error': method.reprojection_error,
                'detection_time': method.detection_time,
                'matching_time': method.matching_time,
            }
            serialized_methods[name] = serialized_method
        
        state['methods'] = serialized_methods
        return state
    
    def __setstate__(self, state):
        """Custom pickle deserialization"""
        from .utils import list_to_keypoints
        from .core_data_structures import FeatureData
        
        # Deserialize methods
        deserialized_methods = {}
        for name, method_data in state['methods'].items():
            # Reconstruct FeatureData
            features1 = FeatureData(
                keypoints=list_to_keypoints(method_data['features1']['keypoints']),
                descriptors=method_data['features1']['descriptors'],
                method=method_data['features1']['method'],
                confidence_scores=method_data['features1']['confidence_scores'],
                detection_time=method_data['features1']['detection_time'],
            )
            
            features2 = FeatureData(
                keypoints=list_to_keypoints(method_data['features2']['keypoints']),
                descriptors=method_data['features2']['descriptors'],
                method=method_data['features2']['method'],
                confidence_scores=method_data['features2']['confidence_scores'],
                detection_time=method_data['features2']['detection_time'],
            )
            
            # Reconstruct MethodResult
            method_result = MethodResult(
                method_name=method_data['method_name'],
                features1=features1,
                features2=features2,
                match_data=method_data['match_data'],
                homography=method_data['homography'],
                fundamental_matrix=method_data['fundamental_matrix'],
                inlier_ratio=method_data['inlier_ratio'],
                reprojection_error=method_data['reprojection_error'],
                detection_time=method_data['detection_time'],
                matching_time=method_data['matching_time'],
            )
            
            deserialized_methods[name] = method_result
        
        state['methods'] = deserialized_methods
        self.__dict__.update(state)

    def __getitem__(self, method_name: str) -> MethodResult:
        """Access method: result['SIFT']"""
        return self.methods[method_name]
    
    def __contains__(self, method_name: str) -> bool:
        """Check: 'SIFT' in result"""
        return method_name in self.methods
    
    def __len__(self) -> int:
        """Total matches across all methods"""
        return sum(m.num_matches for m in self.methods.values())
    
    def __iter__(self):
        """Iterate over method names"""
        return iter(self.methods)
    
    def items(self):
        return self.methods.items()
    
    def keys(self):
        return self.methods.keys()
    
    def values(self):
        return self.methods.values()
    
    # -------------------------------------------------------------------------
    # Method Selection
    # -------------------------------------------------------------------------
    
    def get_method(self, name: str) -> Optional[MethodResult]:
        """Get method by name (None if not found)"""
        return self.methods.get(name)
    
    def get_best(self, 
                 criterion: str = 'quality',
                 min_matches: int = 0) -> Optional[MethodResult]:
        """
        Get best method by criterion
        
        Args:
            criterion: 'quality', 'num_matches', 'inlier_ratio', 'speed'
            min_matches: Minimum matches required
        """
        candidates = [m for m in self.methods.values() if m.num_matches >= min_matches]
        
        if not candidates:
            return None
        
        if criterion == 'quality':
            return max(candidates, key=lambda m: m.get_quality_score())
        elif criterion == 'num_matches':
            return max(candidates, key=lambda m: m.num_matches)
        elif criterion == 'inlier_ratio':
            return max(candidates, key=lambda m: m.inlier_ratio if m.inlier_ratio else 0)
        elif criterion == 'speed':
            return min(candidates, key=lambda m: m.total_time)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
    
    def get_best_name(self, criterion: str = 'quality') -> Optional[str]:
        """Get name of best method"""
        best = self.get_best(criterion)
        return best.method_name if best else None
    
    def rank_methods(self, criterion: str = 'quality') -> List[Tuple[str, MethodResult]]:
        """Rank all methods (best first)"""
        if criterion == 'quality':
            key_func = lambda x: x[1].get_quality_score()
        elif criterion == 'num_matches':
            key_func = lambda x: x[1].num_matches
        elif criterion == 'inlier_ratio':
            key_func = lambda x: x[1].inlier_ratio if x[1].inlier_ratio else 0
        elif criterion == 'speed':
            key_func = lambda x: -x[1].total_time  # Negative for ascending
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        return sorted(self.methods.items(), key=key_func, reverse=True)
    
    # -------------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------------
    
    def filter_methods(self,
                      min_matches: Optional[int] = None,
                      min_inlier_ratio: Optional[float] = None,
                      min_quality: Optional[float] = None,
                      method_names: Optional[List[str]] = None) -> 'MatchingResult':
        """Create new result with filtered methods"""
        filtered = {}
        
        for name, method in self.methods.items():
            # Apply filters
            if method_names and name not in method_names:
                continue
            if min_matches and method.num_matches < min_matches:
                continue
            if min_inlier_ratio and (not method.inlier_ratio or method.inlier_ratio < min_inlier_ratio):
                continue
            if min_quality and method.get_quality_score() < min_quality:
                continue
            
            filtered[name] = method
        
        return MatchingResult(
            methods=filtered,
            image_pair_info=self.image_pair_info,
            processing_metadata=self.processing_metadata
        )
    
    def select_methods(self, method_names: List[str]) -> 'MatchingResult':
        """Create result with only specified methods"""
        return self.filter_methods(method_names=method_names)
    
    def get_top_n(self, n: int, criterion: str = 'quality') -> 'MatchingResult':
        """Get result with top N methods"""
        ranked = self.rank_methods(criterion)
        top_names = [name for name, _ in ranked[:n]]
        return self.select_methods(top_names)
    
    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    
    def summary(self) -> str:
        """Human-readable summary"""
        lines = [
            f"MatchingResult: {len(self.methods)} methods",
            f"Images: {self.image_pair_info.image1_id} ↔ {self.image_pair_info.image2_id}",
            f"Total matches: {len(self)}",
            f"Processing time: {self.processing_metadata.total_processing_time:.3f}s",
            ""
        ]
        
        ranked = self.rank_methods('quality')
        
        if not ranked:
            lines.append("⚠️  No methods produced any matches")
            return "\n".join(lines)
        
        for i, (name, method) in enumerate(ranked, 1):
            status = ""
            if method.num_matches == 0:
                status = " ⚠️  NO MATCHES"
            
            lines.append(
                f"{i}. {name:12s}: "
                f"{method.num_matches:4d} matches, "
                f"quality={method.get_quality_score():.3f}, "
                f"time={method.total_time:.3f}s{status}"
            )
        
        return "\n".join(lines)
    
    def compare_methods(self, method_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare methods across metrics"""
        methods_to_compare = (
            {name: self.methods[name] for name in method_names if name in self.methods}
            if method_names else self.methods
        )
        
        comparison = {'methods': {}}
        
        for name, method in methods_to_compare.items():
            comparison['methods'][name] = {
                'num_matches': method.num_matches,
                'inlier_ratio': method.inlier_ratio,
                'reprojection_error': method.reprojection_error,
                'quality_score': method.get_quality_score(),
                'detection_time': method.detection_time,
                'matching_time': method.matching_time,
                'total_time': method.total_time
            }
        
        # Add rankings
        comparison['rankings'] = {
            'quality': [name for name, _ in self.rank_methods('quality')],
            'num_matches': [name for name, _ in self.rank_methods('num_matches')],
            'speed': [name for name, _ in self.rank_methods('speed')]
        }
        
        return comparison
    
    # -------------------------------------------------------------------------
    # Conversion
    # -------------------------------------------------------------------------
    
    def to_visualization(self, include_images: bool = True):
        """Convert to VisualizationData"""
        from .result_converters import ResultConverter
        converter = ResultConverter()
        return converter.to_visualization(self, include_images=include_images)
    
    def to_reconstruction(self, methods: Optional[List[str]] = None, exclude_empty: bool = True):
        """
        Convert to MultiMethodReconstruction
        
        Args:
            methods: Specific methods to include (None = all)
            exclude_empty: Automatically exclude methods with 0 matches (default: True)
        """
        from .result_converters import ResultConverter
        converter = ResultConverter()
        return converter.to_reconstruction(self, methods=methods, exclude_empty=exclude_empty)
    
    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    
    def visualize(self, method: Optional[str] = None):
        """Quick visualization"""
        viz = self.to_visualization(include_images=True)
        if method:
            viz.plot_method(method)
        else:
            viz.plot()
    
    # -------------------------------------------------------------------------
    # Save/Load
    # -------------------------------------------------------------------------
    
    def save(self, filepath: Union[str, Path], format: str = 'pickle'):
        """Save result"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"✅ Saved MatchingResult to {filepath}")
    
    @staticmethod
    def load(filepath: Union[str, Path]) -> 'MatchingResult':
        """Load result"""
        with open(filepath, 'rb') as f:
            result = pickle.load(f)
        print(f"✅ Loaded MatchingResult from {filepath}")
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_method_result(method_name: str,
                        features1: FeatureData,
                        features2: FeatureData,
                        match_data: MatchData,
                        **kwargs) -> MethodResult:
    """Create MethodResult with optional parameters"""
    return MethodResult(
        method_name=method_name,
        features1=features1,
        features2=features2,
        match_data=match_data,
        detection_time=kwargs.get('detection_time', 0.0),
        matching_time=kwargs.get('matching_time', 0.0),
        homography=kwargs.get('homography'),
        fundamental_matrix=kwargs.get('fundamental_matrix'),
        inlier_ratio=kwargs.get('inlier_ratio'),
        reprojection_error=kwargs.get('reprojection_error')
    )


def save_results_batch(results: List[MatchingResult],
                      output_dir: Union[str, Path],
                      prefix: str = 'result'):
    """Save batch of results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, result in enumerate(results):
        filepath = output_dir / f"{prefix}_{i:04d}.pkl"
        result.save(filepath)
    
    print(f"✅ Saved {len(results)} results to {output_dir}")


def load_results_batch(input_dir: Union[str, Path],
                      pattern: str = 'result_*.pkl') -> List[MatchingResult]:
    """Load batch of results"""
    import glob
    
    input_dir = Path(input_dir)
    files = sorted(glob.glob(str(input_dir / pattern)))
    
    results = []
    for filepath in files:
        try:
            results.append(MatchingResult.load(filepath))
        except Exception as e:
            print(f"❌ Failed to load {filepath}: {e}")
    
    print(f"✅ Loaded {len(results)} results")
    return results


def export_summary_csv(results: List[MatchingResult],
                      filepath: Union[str, Path]):
    """Export summary as CSV"""
    import csv
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'pair_id', 'image1', 'image2', 'methods', 'best_method',
            'total_matches', 'inlier_ratio', 'reprojection_error', 'processing_time'
        ])
        
        for i, result in enumerate(results):
            best = result.get_best()
            writer.writerow([
                i,
                result.image_pair_info.image1_id,
                result.image_pair_info.image2_id,
                ','.join(result.methods.keys()),
                best.method_name if best else None,
                len(result),
                best.inlier_ratio if best else None,
                best.reprojection_error if best else None,
                result.processing_metadata.total_processing_time
            ])
    
    print(f"✅ Exported summary to {filepath}")