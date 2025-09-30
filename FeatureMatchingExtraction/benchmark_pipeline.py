"""
Unified Benchmarking Pipeline for Feature Detection Methods
Fixed version that properly compares complete matching pipelines
"""

import cv2
import numpy as np
import time
import os
import glob
import json
import tracemalloc
import psutil
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Iterator, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from enum import Enum

from .benchmarking import (
    BenchmarkConfig, SyntheticImageGenerator, AdvancedQualityMetrics, 
    StatisticalAnalyzer, BenchmarkResult
)
from .traditional_detectors import create_traditional_detector
from .feature_matchers import create_traditional_matcher, auto_select_matcher
from .pipeline import FeatureProcessingPipeline
from .core_data_structures import MatchData
from .utils import (
    ImageSource, ImageInfo, ImageSourceType, 
    FolderImageSource, SingleImageSource,
    save_enhanced_results
)


class BenchmarkType(Enum):
    """Types of benchmarks available"""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy" 
    COMPREHENSIVE = "comprehensive"


@dataclass
class UnifiedBenchmarkConfig:
    """Unified configuration for all benchmark types"""
    # Benchmark settings
    benchmark_types: List[BenchmarkType] = field(default_factory=lambda: [BenchmarkType.PERFORMANCE])
    methods: List[str] = field(default_factory=lambda: ['SIFT', 'ORB', 'AKAZE'])
    num_runs: int = 3
    
    # Image settings
    max_images: Optional[int] = None
    resize_to: Optional[Tuple[int, int]] = None  # (width, height)
    image_extensions: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'])
    
    # Synthetic image settings (for synthetic sources)
    synthetic_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [(480, 640)])
    synthetic_densities: List[str] = field(default_factory=lambda: ['medium'])
    synthetic_textures: List[str] = field(default_factory=lambda: ['medium'])
    
    # Transformation settings (for accuracy benchmarks)
    transformation_params: Dict = field(default_factory=lambda: {
        'perspective_offset_range': (0.05, 0.15),
        'affine_angle_range': (-15, 15),
        'affine_scale_range': (0.8, 1.2),
        'affine_translation_range': 0.1,
        'rotation_angle_range': (-30, 30),
        'scale_factor_range': (0.7, 1.3)
    })
    transformation_types: List[str] = field(default_factory=lambda: ['perspective', 'affine', 'rotation', 'scale'])
    num_transform_pairs: int = 5
    
    # Analysis settings
    statistical_confidence: float = 0.95
    memory_profiling: bool = True
    quality_thresholds: Dict = field(default_factory=lambda: {
        'min_matches': 10,
        'max_reprojection_error': 3.0,
        'min_inlier_ratio': 0.3
    })
    
    # Output settings
    save_results: bool = True
    output_dir: str = "benchmark_results"
    save_plots: bool = True


# =============================================================================
# Synthetic Image Source (Benchmark-Specific)
# =============================================================================

class SyntheticImageSource(ImageSource):
    """Generate synthetic images on-demand for benchmarking"""
    
    def __init__(self, config: UnifiedBenchmarkConfig):
        self.config = config
        self.generator = SyntheticImageGenerator()
        
    def get_images(self) -> Iterator[ImageInfo]:
        count = 0
        max_images = self.config.max_images or len(self.config.synthetic_sizes) * len(self.config.synthetic_densities) * len(self.config.synthetic_textures)
        
        for size in self.config.synthetic_sizes:
            for density in self.config.synthetic_densities:
                for texture in self.config.synthetic_textures:
                    if count >= max_images:
                        return
                    
                    height, width = size
                    image = self.generator.create_realistic_test_image(height, width, density, texture)
                    
                    yield ImageInfo(
                        image=image,
                        identifier=f"synthetic_{width}x{height}_{density}_{texture}_{count}",
                        metadata={
                            'size': (width, height),
                            'density': density,
                            'texture': texture,
                            'generated': True
                        },
                        source_type=ImageSourceType.SYNTHETIC
                    )
                    count += 1
    
    def get_image_pairs(self) -> Iterator[Tuple[ImageInfo, ImageInfo]]:
        """Generate pairs of images for matching benchmarks"""
        images = list(self.get_images())
        
        # Create pairs from consecutive images
        for i in range(len(images) - 1):
            yield (images[i], images[i + 1])
        
        # If only one image, create a transformed version for pairing
        if len(images) == 1:
            img1 = images[0]
            # Apply small transformation to create second image
            h, w = img1.image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), 5, 1.0)  # 5 degree rotation
            img2_array = cv2.warpAffine(img1.image, M, (w, h))
            img2 = ImageInfo(
                image=img2_array,
                identifier=f"{img1.identifier}_transformed",
                metadata=img1.metadata.copy(),
                source_type=img1.source_type
            )
            yield (img1, img2)
    
    def get_source_info(self) -> Dict[str, Any]:
        return {
            'type': 'synthetic',
            'sizes': self.config.synthetic_sizes,
            'densities': self.config.synthetic_densities,
            'textures': self.config.synthetic_textures,
            'total_combinations': len(self.config.synthetic_sizes) * len(self.config.synthetic_densities) * len(self.config.synthetic_textures)
        }


# =============================================================================
# Enhanced Image Sources (Using utils.py classes with config integration)
# =============================================================================

class ConfiguredFolderImageSource(FolderImageSource):
    """FolderImageSource with benchmark config integration"""
    
    def __init__(self, folder_path: str, config: UnifiedBenchmarkConfig):
        super().__init__(
            folder_path=folder_path,
            max_images=config.max_images,
            resize_to=config.resize_to,
            image_extensions=config.image_extensions
        )
        self.config = config


class ConfiguredSingleImageSource(SingleImageSource):
    """SingleImageSource with benchmark config integration"""
    
    def __init__(self, image_path: str, config: UnifiedBenchmarkConfig):
        super().__init__(
            image_path=image_path,
            resize_to=config.resize_to
        )
        self.config = config


# =============================================================================
# Benchmark Task Abstractions - PROPERLY COMPARING MATCHING PIPELINES
# =============================================================================

class BenchmarkTask(ABC):
    """Abstract base class for benchmark tasks"""
    
    def __init__(self, config: UnifiedBenchmarkConfig):
        self.config = config
        self.statistical_analyzer = StatisticalAnalyzer(config.statistical_confidence)
    
    @abstractmethod
    def run_benchmark(self, image_source: ImageSource, methods: List[str]) -> Dict[str, Any]:
        """Run the benchmark task"""
        pass
    
    @abstractmethod
    def get_task_name(self) -> str:
        """Get the name of this task"""
        pass


class PerformanceTask(BenchmarkTask):
    """Performance benchmarking task - compares complete matching pipelines"""
    
    def get_task_name(self) -> str:
        return "performance"
    
    def run_benchmark(self, image_source: ImageSource, methods: List[str]) -> Dict[str, Any]:
        print(f"Running Performance Benchmark on {image_source.get_source_info()['type']} images...")
        print("Comparing complete matching pipelines (detection + matching)...")
        
        all_results = {}
        method_aggregated_data = defaultdict(list)
        
        for method in methods:
            print(f"  Benchmarking {method} matching pipeline...")
            
            # Create detector/matcher ONCE per method
            try:
                detector, matcher = self._initialize_method(method)
                print(f"    Initialized {method} successfully")
            except Exception as e:
                print(f"    Failed to initialize {method}: {e}")
                all_results[method] = [{
                    'method': method,
                    'error': f'Initialization failed: {str(e)}',
                    'success': False
                }]
                continue
            
            method_results = []
            
            # Process all image pairs with the SAME detector/matcher
            pair_count = 0
            for img1_info, img2_info in image_source.get_image_pairs():
                pair_count += 1
                print(f"    Processing pair {pair_count}: {img1_info.identifier} - {img2_info.identifier}")
                
                result = self._benchmark_matching_pipeline(method, img1_info, img2_info, detector, matcher)
                method_results.append(result)
                
                # Collect data for statistical analysis
                if result['success']:
                    for run_data in result['raw_runs']:
                        if run_data['success']:
                            method_aggregated_data[method].append(run_data['total_time'])
            
            all_results[method] = method_results
        
        # Statistical comparisons
        statistical_comparisons = {}
        method_list = list(method_aggregated_data.keys())
        for i, method1 in enumerate(method_list):
            for method2 in method_list[i+1:]:
                if len(method_aggregated_data[method1]) > 0 and len(method_aggregated_data[method2]) > 0:
                    comparison = self.statistical_analyzer.compare_methods(
                        method_aggregated_data[method1],
                        method_aggregated_data[method2],
                        'total_matching_time'
                    )
                    statistical_comparisons[f"{method1}_vs_{method2}"] = comparison
        
        return {
            'task': 'performance',
            'detailed_results': all_results,
            'statistical_comparisons': statistical_comparisons,
            'summary': self._create_summary(all_results, method_aggregated_data)
        }
    
    
    def _initialize_method(self, method: str) -> Tuple[Any, Any]:
        """Initialize detector and matcher once for a method"""
        detector = None
        matcher = None
        
        if method.lower() == 'lightglue':
            # LightGlue acts as both detector and matcher
            from .feature_matchers import create_deep_learning_matcher
            lightglue_matcher = create_deep_learning_matcher('LightGlue', max_num_keypoints=2000)
            return lightglue_matcher, None  # LightGlue is both detector and matcher
            
        elif method in ['SIFT', 'ORB', 'AKAZE', 'BRISK', 'Harris', 'GoodFeatures']:
            # Traditional method
            detector = create_traditional_detector(method, max_features=2000)
            # Matcher will be created on first use based on descriptor type
            return detector, None
            
        elif method in ['SuperPoint', 'DISK', 'ALIKED']:
            # Deep learning detectors with traditional matching
            from .deep_learning_detectors import create_deep_learning_detector
            detector = create_deep_learning_detector(method, max_features=2000)
            from .feature_matchers import EnhancedFLANNMatcher
            matcher = EnhancedFLANNMatcher()
            return detector, matcher
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _benchmark_matching_pipeline(self, method: str, img1_info: ImageInfo, img2_info: ImageInfo, 
                                    detector: Any, matcher: Any) -> Dict[str, Any]:
        """Benchmark the complete matching pipeline for any method using provided detector/matcher"""
        
        # Now run the benchmark multiple times with the SAME detector/matcher
        run_results = []
        
        for run in range(self.config.num_runs):
            try:
                # Memory tracking
                memory_data = {}
                if self.config.memory_profiling:
                    tracemalloc.start()
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024
                
                # Time ONLY the matching pipeline (not initialization)
                start_time = time.time()
                
                if method.lower() == 'lightglue':
                    # LightGlue: detector is the matcher
                    lightglue_matcher = detector  # detector is actually the LightGlue matcher
                    features1, features2, match_data = lightglue_matcher.match_images_directly(
                        img1_info.image, img2_info.image
                    )
                    num_matches = len(match_data.matches)
                    num_features1 = len(features1.keypoints)
                    num_features2 = len(features2.keypoints)
                    
                elif method in ['SIFT', 'ORB', 'AKAZE', 'BRISK', 'Harris', 'GoodFeatures']:
                    # Traditional method: use provided detector
                    features1 = detector.detect(img1_info.image)
                    features2 = detector.detect(img2_info.image)
                    
                    # Match features
                    if len(features1.keypoints) > 0 and len(features2.keypoints) > 0:
                        # Create matcher on first use if not provided
                        if matcher is None:
                            matcher = auto_select_matcher(features1, features2)
                        match_data = matcher.match(features1, features2)
                        num_matches = len(match_data.matches)
                    else:
                        num_matches = 0
                    
                    num_features1 = len(features1.keypoints)
                    num_features2 = len(features2.keypoints)
                    
                elif method in ['SuperPoint', 'DISK', 'ALIKED']:
                    # Deep learning detectors: use provided detector and matcher
                    features1 = detector.detect(img1_info.image)
                    features2 = detector.detect(img2_info.image)
                    
                    # Match features with provided matcher
                    if len(features1.keypoints) > 0 and len(features2.keypoints) > 0:
                        match_data = matcher.match(features1, features2)
                        num_matches = len(match_data.matches)
                    else:
                        num_matches = 0
                    
                    num_features1 = len(features1.keypoints)
                    num_features2 = len(features2.keypoints)
                
                total_time = time.time() - start_time
                
                # Memory tracking
                if self.config.memory_profiling:
                    memory_after = process.memory_info().rss / 1024 / 1024
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    
                    memory_data = {
                        'memory_before_mb': memory_before,
                        'memory_after_mb': memory_after,
                        'memory_increase_mb': memory_after - memory_before,
                        'traced_peak_mb': peak / 1024 / 1024
                    }
                
                if hasattr(match_data, 'get_all_matches'):
                    # MultiMethodMatchData
                    num_matches = len(match_data.get_all_matches())
                elif hasattr(match_data, 'matches'):
                    # MatchData
                    num_matches = len(match_data.matches)
                else:
                    num_matches = 0
                
                run_results.append({
                    'run': run,
                    'total_time': total_time,
                    'num_matches': num_matches,
                    'num_features1': num_features1,
                    'num_features2': num_features2,
                    'avg_features': (num_features1 + num_features2) / 2,
                    'memory': memory_data,
                    'success': True
                })
                
            except Exception as e:
                run_results.append({
                    'run': run,
                    'error': str(e),
                    'success': False
                })
        
        # Analyze results
        successful_runs = [r for r in run_results if r['success']]
        
        if successful_runs:
            times = [r['total_time'] for r in successful_runs]
            matches = [r['num_matches'] for r in successful_runs]
            features_avg = [r['avg_features'] for r in successful_runs]
            
            return {
                'method': method,
                'image_pair': f"{img1_info.identifier}-{img2_info.identifier}",
                'successful_runs': len(successful_runs),
                'total_runs': len(run_results),
                'time_stats': self.statistical_analyzer._compute_descriptive_stats(times),
                'match_stats': self.statistical_analyzer._compute_descriptive_stats(matches),
                'feature_stats': self.statistical_analyzer._compute_descriptive_stats(features_avg),
                'raw_runs': run_results,
                'success': True
            }
        else:
            return {
                'method': method,
                'image_pair': f"{img1_info.identifier}-{img2_info.identifier}",
                'error': 'All runs failed',
                'raw_runs': run_results,
                'success': False
            }
                

    
    def _create_summary(self, all_results: Dict, aggregated_data: Dict) -> Dict:
        """Create summary statistics for matching pipeline performance"""
        summary = {}
        
        for method, method_results in all_results.items():
            successful_results = [r for r in method_results if r['success']]
            
            if successful_results and method in aggregated_data:
                times = aggregated_data[method]
                all_matches = []
                all_features = []
                all_memory = []
                
                for result in successful_results:
                    for run in result['raw_runs']:
                        if run['success']:
                            all_matches.append(run['num_matches'])
                            all_features.append(run['avg_features'])
                            if run.get('memory'):
                                all_memory.append(run['memory']['memory_increase_mb'])
                
                summary[method] = {
                    'success_rate': len(successful_results) / len(method_results),
                    'time_stats': self.statistical_analyzer._compute_descriptive_stats(times),
                    'match_stats': self.statistical_analyzer._compute_descriptive_stats(all_matches),
                    'feature_stats': self.statistical_analyzer._compute_descriptive_stats(all_features),
                    'avg_fps': 1.0 / np.mean(times) if times else 0,
                    'matches_per_second': np.mean(all_matches) / np.mean(times) if times and all_matches else 0
                }
                
                if all_memory:
                    summary[method]['memory_stats'] = self.statistical_analyzer._compute_descriptive_stats(all_memory)
            else:
                summary[method] = {
                    'success_rate': 0.0,
                    'error': 'All tests failed'
                }
        
        return summary


class AccuracyTask(BenchmarkTask):
    """Accuracy benchmarking task - tests matching quality with ground truth"""
    
    def __init__(self, config: UnifiedBenchmarkConfig):
        super().__init__(config)
        self.quality_metrics = AdvancedQualityMetrics(BenchmarkConfig())
        self.generator = SyntheticImageGenerator()
    
    def get_task_name(self) -> str:
        return "accuracy"
    
    def run_benchmark(self, image_source: ImageSource, methods: List[str]) -> Dict[str, Any]:
        print("Running Accuracy Benchmark...")
        print("Testing matching accuracy with synthetic transformations...")
        
        all_results = {}
        
        for method in methods:
            print(f"  Testing {method} accuracy...")
            
            # Initialize detector/matcher ONCE per method
            try:
                detector, matcher = self._initialize_method(method)
                print(f"    Initialized {method} successfully")
            except Exception as e:
                print(f"    Failed to initialize {method}: {e}")
                all_results[method] = [{
                    'method': method,
                    'error': f'Initialization failed: {str(e)}',
                    'success': False
                }]
                continue
            
            method_results = []
            
            pair_count = 0
            for base_image_info in image_source.get_images():
                if pair_count >= self.config.num_transform_pairs:
                    break
                
                for transform_type in self.config.transformation_types:
                    img1, img2, gt_transform = self._create_transform_pair(
                        base_image_info.image, transform_type
                    )
                    
                    result = self._test_matching_accuracy_with_detector(
                        method, img1, img2, gt_transform, transform_type,
                        f"{base_image_info.identifier}_{transform_type}_{pair_count}",
                        detector, matcher
                    )
                    method_results.append(result)
                
                pair_count += 1
            
            all_results[method] = method_results
        
        return {
            'task': 'accuracy',
            'detailed_results': all_results,
            'summary': self._create_accuracy_summary(all_results)
        }
    
    def _initialize_method(self, method: str) -> Tuple[Any, Any]:
        """Initialize detector and matcher once for a method"""
        if method.lower() == 'lightglue':
            from .feature_matchers import create_deep_learning_matcher
            lightglue_matcher = create_deep_learning_matcher('LightGlue', max_num_keypoints=1000)
            return lightglue_matcher, None
        elif method in ['SIFT', 'ORB', 'AKAZE', 'BRISK', 'Harris', 'GoodFeatures']:
            detector = create_traditional_detector(method, max_features=1000)
            return detector, None
        elif method in ['SuperPoint', 'DISK', 'ALIKED']:
            from .deep_learning_detectors import create_deep_learning_detector
            detector = create_deep_learning_detector(method, max_features=1000)
            from .feature_matchers import EnhancedFLANNMatcher
            matcher = EnhancedFLANNMatcher()
            return detector, matcher
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _create_transform_pair(self, base_image: np.ndarray, transform_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a transformed image pair with ground truth"""
        h, w = base_image.shape[:2]
        params = self.config.transformation_params
        
        if transform_type == 'perspective':
            src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            offset_min, offset_max = params['perspective_offset_range']
            offset = min(w, h) * np.random.uniform(offset_min, offset_max)
            
            dst_points = np.float32([
                [np.random.uniform(0, offset), np.random.uniform(0, offset/2)], 
                [w - np.random.uniform(0, offset), np.random.uniform(0, offset)], 
                [w - np.random.uniform(0, offset/2), h - np.random.uniform(0, offset)], 
                [np.random.uniform(0, offset/2), h - np.random.uniform(0, offset/2)]
            ])
            
            transform = cv2.getPerspectiveTransform(src_points, dst_points)
            img2 = cv2.warpPerspective(base_image, transform, (w, h))
            
        elif transform_type == 'affine':
            center = (w/2, h/2)
            angle_min, angle_max = params['affine_angle_range']
            scale_min, scale_max = params['affine_scale_range']
            trans_range = params['affine_translation_range']
            
            angle = np.random.uniform(angle_min, angle_max)
            scale = np.random.uniform(scale_min, scale_max)
            
            transform = cv2.getRotationMatrix2D(center, angle, scale)
            transform[0, 2] += np.random.uniform(-w*trans_range, w*trans_range)
            transform[1, 2] += np.random.uniform(-h*trans_range, h*trans_range)
            
            img2 = cv2.warpAffine(base_image, transform, (w, h))
            transform = np.vstack([transform, [0, 0, 1]])
            
        elif transform_type == 'rotation':
            center = (w/2, h/2)
            angle_min, angle_max = params['rotation_angle_range']
            angle = np.random.uniform(angle_min, angle_max)
            
            transform_2x3 = cv2.getRotationMatrix2D(center, angle, 1.0)
            img2 = cv2.warpAffine(base_image, transform_2x3, (w, h))
            transform = np.vstack([transform_2x3, [0, 0, 1]])
            
        elif transform_type == 'scale':
            scale_min, scale_max = params['scale_factor_range']
            scale_factor = np.random.uniform(scale_min, scale_max)
            
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            img2 = cv2.resize(base_image, (new_w, new_h))
            
            if scale_factor > 1:
                start_x = (new_w - w) // 2
                start_y = (new_h - h) // 2
                img2 = img2[start_y:start_y+h, start_x:start_x+w]
                transform = np.array([[scale_factor, 0, -start_x], [0, scale_factor, -start_y], [0, 0, 1]])
            else:
                pad_x = (w - new_w) // 2
                pad_y = (h - new_h) // 2
                remaining_pad_x = w - new_w - pad_x
                remaining_pad_y = h - new_h - pad_y
                img2 = cv2.copyMakeBorder(img2, pad_y, remaining_pad_y, pad_x, remaining_pad_x, cv2.BORDER_CONSTANT)
                transform = np.array([[scale_factor, 0, pad_x], [0, scale_factor, pad_y], [0, 0, 1]])
        
        return base_image, img2, transform
    
    def _test_matching_accuracy_with_detector(self, method: str, img1: np.ndarray, img2: np.ndarray, 
                               gt_transform: np.ndarray, transform_type: str, pair_id: str,
                               detector: Any, matcher: Any) -> Dict[str, Any]:
        """Test matching accuracy for a single image pair using provided detector/matcher"""
        try:
            start_time = time.time()
            
            # Use the provided detector/matcher
            if method.lower() == 'lightglue':
                lightglue_matcher = detector  # detector is actually the LightGlue matcher
                features1, features2, match_data = lightglue_matcher.match_images_directly(img1, img2)
                
            elif method in ['SIFT', 'ORB', 'AKAZE', 'BRISK', 'Harris', 'GoodFeatures']:
                features1 = detector.detect(img1)
                features2 = detector.detect(img2)
                
                if len(features1.keypoints) > 0 and len(features2.keypoints) > 0:
                    if matcher is None:
                        matcher = auto_select_matcher(features1, features2)
                    match_data = matcher.match(features1, features2)
                else:
                    match_data = MatchData([], method=method)
                    features1 = features1 if 'features1' in locals() else None
                    features2 = features2 if 'features2' in locals() else None
                    
            elif method in ['SuperPoint', 'DISK', 'ALIKED']:
                features1 = detector.detect(img1)
                features2 = detector.detect(img2)
                
                if len(features1.keypoints) > 0 and len(features2.keypoints) > 0:
                    match_data = matcher.match(features1, features2)
                else:
                    match_data = MatchData([], method=method)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            total_time = time.time() - start_time
            
            # Analyze match quality
            matches = match_data.get_best_matches()
            quality_score = 0.0
            quality_metrics = {}
            
            if hasattr(match_data, 'get_filtered_matches'):
                matches = match_data.get_filtered_matches()
            else:
                matches = match_data.get_best_matches()
            
            quality_score = 0.0
            quality_metrics = {}
            
            if len(matches) > 0 and features1 and features2:
                keypoints1 = features1.keypoints
                keypoints2 = features2.keypoints
                match_indices = [(m.queryIdx, m.trainIdx) for m in matches]
                
                quality_metrics = self.quality_metrics.comprehensive_quality_assessment(
                    match_indices, keypoints1, keypoints2, gt_transform
                )
                quality_score = quality_metrics.get('overall_quality', 0.0)
            
            return {
                'pair_id': pair_id,
                'method': method,
                'transformation': transform_type,
                'num_matches': len(matches),
                'quality_score': quality_score,
                'quality_metrics': quality_metrics,
                'total_time': total_time,
                'success': True
            }
            
        except Exception as e:
            import traceback
            print(f"      Error in accuracy test: {e}")
            traceback.print_exc()
            return {
                'pair_id': pair_id,
                'method': method,
                'transformation': transform_type,
                'error': str(e),
                'success': False
            }
    
    def _create_accuracy_summary(self, all_results: Dict) -> Dict:
        """Create accuracy summary"""
        summary = {}
        
        for method, method_results in all_results.items():
            successful_results = [r for r in method_results if r['success']]
            
            if successful_results:
                quality_scores = [r['quality_score'] for r in successful_results]
                match_counts = [r['num_matches'] for r in successful_results]
                
                summary[method] = {
                    'success_rate': len(successful_results) / len(method_results),
                    'avg_quality': np.mean(quality_scores),
                    'avg_matches': np.mean(match_counts),
                    'quality_stats': self.statistical_analyzer._compute_descriptive_stats(quality_scores),
                    'match_stats': self.statistical_analyzer._compute_descriptive_stats(match_counts),
                    'by_transformation': {}
                }
                
                # Break down by transformation type
                for transform_type in self.config.transformation_types:
                    transform_results = [r for r in successful_results if r['transformation'] == transform_type]
                    if transform_results:
                        trans_quality = [r['quality_score'] for r in transform_results]
                        trans_matches = [r['num_matches'] for r in transform_results]
                        
                        summary[method]['by_transformation'][transform_type] = {
                            'avg_quality': np.mean(trans_quality),
                            'avg_matches': np.mean(trans_matches),
                            'success_rate': len(transform_results) / sum(1 for r in method_results if r.get('transformation') == transform_type)
                        }
            else:
                summary[method] = {
                    'success_rate': 0.0,
                    'error': 'All tests failed'
                }
        
        return summary


# =============================================================================
# Unified Pipeline
# =============================================================================

class UnifiedBenchmarkPipeline:
    """Unified pipeline for all types of benchmarking"""
    
    def __init__(self, config: UnifiedBenchmarkConfig = None):
        self.config = config or UnifiedBenchmarkConfig()
        self.tasks = {
            BenchmarkType.PERFORMANCE: PerformanceTask(self.config),
            BenchmarkType.ACCURACY: AccuracyTask(self.config)
        }
    
    def benchmark_synthetic_images(self, methods: List[str] = None, 
                                  benchmark_types: List[BenchmarkType] = None) -> Dict[str, Any]:
        """Benchmark on synthetic images"""
        methods = methods or self.config.methods
        benchmark_types = benchmark_types or self.config.benchmark_types
        
        image_source = SyntheticImageSource(self.config)
        return self._run_benchmarks(image_source, methods, benchmark_types)
    
    def benchmark_folder(self, folder_path: str, methods: List[str] = None,
                        benchmark_types: List[BenchmarkType] = None) -> Dict[str, Any]:
        """Benchmark on images from a folder"""
        methods = methods or self.config.methods
        benchmark_types = benchmark_types or self.config.benchmark_types
        
        image_source = ConfiguredFolderImageSource(folder_path, self.config)
        return self._run_benchmarks(image_source, methods, benchmark_types)
    
    def benchmark_single_image(self, image_path: str, methods: List[str] = None,
                              benchmark_types: List[BenchmarkType] = None) -> Dict[str, Any]:
        """Benchmark on a single image (creates transformed pair)"""
        methods = methods or self.config.methods
        benchmark_types = benchmark_types or self.config.benchmark_types
        
        image_source = ConfiguredSingleImageSource(image_path, self.config)
        return self._run_benchmarks(image_source, methods, benchmark_types)
    
    def _run_benchmarks(self, image_source: ImageSource, methods: List[str], 
                       benchmark_types: List[BenchmarkType]) -> Dict[str, Any]:
        """Run the specified benchmarks"""
        print("=" * 60)
        print("FEATURE MATCHING PIPELINE BENCHMARK")
        print("=" * 60)
        print(f"Comparing complete matching pipelines:")
        print(f"  - Traditional: Detect → Match")
        print(f"  - LightGlue: End-to-end matching")
        print(f"Methods: {methods}")
        print(f"Benchmarks: {[bt.value for bt in benchmark_types]}")
        print(f"Image source: {image_source.get_source_info()}")
        print("=" * 60)
        
        results = {
            'config': self.config,
            'image_source_info': image_source.get_source_info(),
            'methods': methods,
            'benchmark_types': [bt.value for bt in benchmark_types],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'benchmarks': {}
        }
        
        for benchmark_type in benchmark_types:
            if benchmark_type in self.tasks:
                task = self.tasks[benchmark_type]
                task_results = task.run_benchmark(image_source, methods)
                results['benchmarks'][benchmark_type.value] = task_results
            else:
                print(f"Warning: Benchmark type {benchmark_type.value} not implemented")
        
        # Add comprehensive analysis if multiple benchmarks
        if len(benchmark_types) > 1:
            results['comprehensive_analysis'] = self._create_comprehensive_analysis(results)
        
        # Save results if requested
        if self.config.save_results:
            self._save_results(results)
        
        return results
    
    def _create_comprehensive_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive analysis across multiple benchmark types"""
        analysis = {}
        
        # Extract summaries from each benchmark
        benchmarks = results['benchmarks']
        
        if 'performance' in benchmarks and 'accuracy' in benchmarks:
            perf_summary = benchmarks['performance']['summary']
            acc_summary = benchmarks['accuracy']['summary']
            
            # Combine performance and accuracy metrics
            combined_rankings = {}
            
            for method in results['methods']:
                if method in perf_summary and method in acc_summary:
                    # Performance score based on matches per second
                    perf_score = perf_summary[method].get('matches_per_second', 0)
                    # Accuracy score
                    acc_score = acc_summary[method].get('avg_quality', 0)
                    
                    # Normalize and combine scores
                    max_perf = max(s.get('matches_per_second', 0) for s in perf_summary.values() if 'error' not in s)
                    norm_perf = perf_score / max_perf if max_perf > 0 else 0
                    
                    # Combined score (equal weighting)
                    combined_score = (norm_perf + acc_score) / 2
                    
                    combined_rankings[method] = {
                        'performance_score': perf_score,
                        'normalized_performance': norm_perf,
                        'accuracy_score': acc_score,
                        'combined_score': combined_score
                    }
            
            # Sort by combined score
            sorted_methods = sorted(combined_rankings.items(), key=lambda x: x[1]['combined_score'], reverse=True)
            
            analysis['method_rankings'] = {
                'by_combined_score': sorted_methods,
                'best_performance': max(perf_summary.items(), 
                                      key=lambda x: x[1].get('matches_per_second', 0) if 'error' not in x[1] else 0)[0] if perf_summary else None,
                'best_accuracy': max(acc_summary.items(), 
                                   key=lambda x: x[1].get('avg_quality', 0) if 'error' not in x[1] else 0)[0] if acc_summary else None
            }
        
        return analysis
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to files"""
        try:
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Save as JSON
            results_file = os.path.join(self.config.output_dir, f"benchmark_results_{int(time.time())}.json")
            
            # Use the save function from utils
            success = save_enhanced_results(results, results_file, format='json')
            
            if success:
                print(f"\nResults saved to: {results_file}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of results"""
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Image Source: {results['image_source_info']['type']}")
        print(f"Methods: {', '.join(results['methods'])}")
        print(f"Benchmarks: {', '.join(results['benchmark_types'])}")
        
        for benchmark_type, benchmark_results in results['benchmarks'].items():
            print(f"\n{benchmark_type.upper()} RESULTS:")
            print("-" * 40)
            
            summary = benchmark_results['summary']
            
            if benchmark_type == 'performance':
                print(f"{'Method':<12} {'Success':<10} {'Time (s)':<10} {'Matches':<10} {'FPS':<8} {'Match/s':<10}")
                print("-" * 70)
                
                for method, stats in summary.items():
                    if 'error' not in stats:
                        success_rate = stats['success_rate'] * 100
                        avg_time = stats['time_stats']['mean']
                        avg_matches = stats['match_stats']['mean']
                        fps = stats['avg_fps']
                        matches_per_sec = stats['matches_per_second']
                        
                        print(f"{method:<12} {success_rate:<10.1f}% {avg_time:<10.3f} {avg_matches:<10.1f} {fps:<8.1f} {matches_per_sec:<10.1f}")
                    else:
                        print(f"{method:<12} {'ERROR':<10} {'-':<10} {'-':<10} {'-':<8} {'-':<10}")
            
            elif benchmark_type == 'accuracy':
                print(f"{'Method':<12} {'Success':<10} {'Quality':<12} {'Matches':<12}")
                print("-" * 46)
                
                for method, stats in summary.items():
                    if 'error' not in stats:
                        success_rate = stats['success_rate'] * 100
                        avg_quality = stats['avg_quality']
                        avg_matches = stats['avg_matches']
                        
                        print(f"{method:<12} {success_rate:<10.1f}% {avg_quality:<12.3f} {avg_matches:<12.1f}")
                    else:
                        print(f"{method:<12} {'ERROR':<10} {'-':<12} {'-':<12}")
        
        # Print comprehensive analysis if available
        if 'comprehensive_analysis' in results:
            analysis = results['comprehensive_analysis']
            if 'method_rankings' in analysis:
                rankings = analysis['method_rankings']
                
                print(f"\n{'='*60}")
                print("OVERALL RANKINGS:")
                print("-" * 40)
                print(f"Best Performance: {rankings['best_performance']}")
                print(f"Best Accuracy: {rankings['best_accuracy']}")
                
                print(f"\nCombined Rankings (Performance + Accuracy):")
                for i, (method, scores) in enumerate(rankings['by_combined_score'][:5]):
                    print(f"  {i+1}. {method}: {scores['combined_score']:.3f}")
                    print(f"     Performance: {scores['performance_score']:.1f} matches/s")
                    print(f"     Accuracy: {scores['accuracy_score']:.3f}")


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_folder_benchmark(folder_path: str, methods: List[str] = None,
                          benchmark_types: List[str] = None, **kwargs) -> Dict[str, Any]:
    """Quick benchmark on a folder of images"""
    if methods is None:
        methods = ['SIFT', 'ORB', 'lightglue']  # Include LightGlue by default
    
    if benchmark_types is None:
        benchmark_types = ['performance']
    
    # Convert string benchmark types to enum
    bt_enums = []
    for bt in benchmark_types:
        if bt == 'performance':
            bt_enums.append(BenchmarkType.PERFORMANCE)
        elif bt == 'accuracy':
            bt_enums.append(BenchmarkType.ACCURACY)
        elif bt == 'comprehensive':
            bt_enums.extend([BenchmarkType.PERFORMANCE, BenchmarkType.ACCURACY])
    
    config = UnifiedBenchmarkConfig(
        methods=methods,
        benchmark_types=bt_enums,
        max_images=kwargs.get('max_images', 10),
        resize_to=kwargs.get('resize_to', (640, 480)),
        num_runs=kwargs.get('num_runs', 3)
    )
    
    pipeline = UnifiedBenchmarkPipeline(config)
    results = pipeline.benchmark_folder(folder_path)
    pipeline.print_summary(results)
    
    return results


def quick_synthetic_benchmark(methods: List[str] = None, **kwargs) -> Dict[str, Any]:
    """Quick benchmark on synthetic images"""
    if methods is None:
        methods = ['SIFT', 'ORB', 'lightglue']
    
    config = UnifiedBenchmarkConfig(
        methods=methods,
        benchmark_types=[BenchmarkType.PERFORMANCE, BenchmarkType.ACCURACY],
        synthetic_sizes=kwargs.get('sizes', [(480, 640)]),
        num_runs=kwargs.get('num_runs', 3)
    )
    
    pipeline = UnifiedBenchmarkPipeline(config)
    results = pipeline.benchmark_synthetic_images()
    pipeline.print_summary(results)
    
    return results


def quick_single_image_benchmark(image_path: str, methods: List[str] = None, **kwargs) -> Dict[str, Any]:
    """Quick benchmark on a single image"""
    if methods is None:
        methods = ['SIFT', 'ORB', 'lightglue']
    
    config = UnifiedBenchmarkConfig(
        methods=methods,
        benchmark_types=[BenchmarkType.PERFORMANCE],
        num_runs=kwargs.get('num_runs', 5)
    )
    
    pipeline = UnifiedBenchmarkPipeline(config)
    results = pipeline.benchmark_single_image(image_path)
    pipeline.print_summary(results)
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Unified Feature Detection Benchmarking Pipeline")
    print("Testing complete matching pipelines...")
    
    # Quick synthetic test comparing all methods
    synthetic_results = quick_synthetic_benchmark(['SIFT', 'ORB', 'AKAZE', 'lightglue'])