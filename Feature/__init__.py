"""
Multi-Method Feature Detection and Matching System

A comprehensive computer vision library for feature detection and matching
that combines traditional and deep learning approaches with proper score handling
and unified benchmarking capabilities.

Main Components:
- Traditional detectors: SIFT, ORB, AKAZE, BRISK, Harris
- Deep learning detectors: SuperPoint, DISK, ALIKED
- Advanced matchers: LightGlue, enhanced FLANN, BF
- Multi-score support: distance, confidence, similarity
- Comprehensive analysis and visualization tools
- Unified benchmarking pipeline for performance and accuracy testing
- Support for synthetic images, real image folders, and custom datasets

Quick Start:
    >>> import feature_detection_system as fds
    >>> fds.print_capabilities()
    >>> features = fds.detect_features(image, 'SIFT')
    >>> results = fds.match_images(img1, img2, ['SIFT', 'ORB'])
    >>> benchmark = fds.benchmark_folder("/path/to/images", ['SIFT', 'ORB'])
"""

__version__ = "1.0.0"
__author__ = "Feature Detection Team"

# Core data structures
from .core_data_structures import (
    FeatureData,
    MatchData,
    EnhancedDMatch,
    DetectorType,
    ScoreType,
    keypoints_to_serializable,
    keypoints_from_serializable
)

# Base classes
from .base_classes import (
    BaseFeatureDetector,
    BaseFeatureMatcher,
    BasePairMatcher
)

# Traditional detectors
from .traditional_detectors import (
    SIFTDetector,
    ORBDetector,
    AKAZEDetector,
    BRISKDetector,
    HarrisCornerDetector,
    GoodFeaturesToTrackDetector,
    create_traditional_detector
)

# Deep learning detectors (optional, requires PyTorch)
try:
    from .deep_learning_detectors import (
        SuperPointDetector,
        DISKDetector,
        ALIKEDDetector,
        create_deep_learning_detector
    )
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

# Feature matchers
from .feature_matchers import (
    EnhancedFLANNMatcher,
    EnhancedBFMatcher,
    create_traditional_matcher,
    auto_select_matcher
)

# Deep learning matchers (optional, requires PyTorch + LightGlue)
try:
    from .feature_matchers import (
        LightGlueMatcher,
        create_deep_learning_matcher
    )
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False

# Pipeline and multi-method processing
from .pipeline import (
    MultiMethodFeatureDetector,
    FeatureProcessingPipeline,
    create_pipeline
)

# Utilities
from .utils import (
    enhanced_filter_matches_with_homography,
    adaptive_match_filtering,
    remove_duplicate_matches,
    extract_correspondences,
    visualize_matches_with_scores,
    visualize_keypoints,
    MatchQualityAnalyzer,
    save_enhanced_results,
    load_enhanced_results,
    verify_matches_with_fundamental_matrix,
    calculate_reprojection_error,
    benchmark_detector_performance,
    get_images_from_folder
)

# Configuration management
from .config import (
    get_default_config,
    validate_config,
    create_config_from_preset,
    print_available_presets 
)

# Unified Benchmarking Pipeline
from .benchmark_pipeline import (
    UnifiedBenchmarkConfig,
    UnifiedBenchmarkPipeline,
    BenchmarkType,
    ImageSourceType,
    ImageInfo,
    ImageSource,
    SyntheticImageSource,
    FolderImageSource,
    SingleImageSource,
    #ImageListSource,
    BenchmarkTask,
    PerformanceTask,
    AccuracyTask,
    quick_folder_benchmark,
    quick_synthetic_benchmark,
    quick_single_image_benchmark
)

# Legacy benchmarking components (for backward compatibility)
from .benchmarking import (
    BenchmarkConfig,
    BenchmarkResult,
    SyntheticImageGenerator,
    AdvancedQualityMetrics,
    StatisticalAnalyzer,
    PerformanceBenchmark,
    quick_test,
)


# Make commonly used functions easily accessible
def detect_features(image, method='SIFT', **kwargs):
    """
    Quick feature detection with a single method
    
    Args:
        image: Input image (numpy array)
        method: Detection method ('SIFT', 'ORB', 'AKAZE', etc.)
        **kwargs: Additional parameters for the detector
        
    Returns:
        FeatureData object with detected features
    """
    if method in ['SIFT', 'ORB', 'AKAZE', 'BRISK', 'Harris', 'GoodFeatures']:
        detector = create_traditional_detector(method, **kwargs)
    elif DEEP_LEARNING_AVAILABLE and method in ['SuperPoint', 'DISK', 'ALIKED']:
        detector = create_deep_learning_detector(method, **kwargs)
    else:
        available_methods = ['SIFT', 'ORB', 'AKAZE', 'BRISK', 'Harris', 'GoodFeatures']
        if DEEP_LEARNING_AVAILABLE:
            available_methods.extend(['SuperPoint', 'DISK', 'ALIKED'])
        raise ValueError(f"Unknown method: {method}. Available: {available_methods}")
    
    return detector.detect(image)


def match_images(img1, img2, methods=['SIFT'], **kwargs):
    """
    Quick image matching with automatic method selection
    
    Args:
        img1: First image (numpy array)
        img2: Second image (numpy array)
        methods: List of methods to try (or single method)
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary with matching results
    """
    if isinstance(methods, str):
        methods = [methods]
    
    config = {
        'methods': methods,
        'max_features': kwargs.get('max_features', 2000),
        'combine_strategy': kwargs.get('combine_strategy', 'best')
    }
    
    pipeline = FeatureProcessingPipeline(config)
    return pipeline.process_image_pair(img1, img2, 
                                     visualize=kwargs.get('visualize', True))


# Updated benchmark functions using unified pipeline
def benchmark_folder(folder_path, methods=None, **kwargs):
    """
    Benchmark methods on a folder of images using unified pipeline
    
    Args:
        folder_path: Path to folder containing images
        methods: List of methods to benchmark (default: ['SIFT', 'ORB', 'AKAZE'])
        **kwargs: Additional configuration parameters
        
    Returns:
        Comprehensive benchmark results
    """
    return quick_folder_benchmark(
        folder_path=folder_path,
        methods=methods or ['SIFT', 'ORB', 'AKAZE'],
        benchmark_types=kwargs.get('benchmark_types', ['performance']),
        max_images=kwargs.get('max_images', 15),
        resize_to=kwargs.get('resize_to', (640, 480)),
        num_runs=kwargs.get('num_runs', 3)
    )


def benchmark_single_image(image_path, methods=None, **kwargs):
    """
    Benchmark methods on a single image using unified pipeline
    
    Args:
        image_path: Path to image file
        methods: List of methods to benchmark
        **kwargs: Additional configuration parameters
        
    Returns:
        Single image benchmark results
    """
    return quick_single_image_benchmark(
        image_path=image_path,
        methods=methods or ['SIFT', 'ORB', 'AKAZE'],
        num_runs=kwargs.get('num_runs', 5)
    )


def benchmark_synthetic(methods=None, **kwargs):
    """
    Benchmark methods on synthetic images with ground truth transformations
    
    Args:
        methods: List of methods to benchmark
        **kwargs: Additional configuration parameters
        
    Returns:
        Synthetic benchmark results with accuracy metrics
    """
    return quick_synthetic_benchmark(
        methods=methods or ['SIFT', 'ORB', 'AKAZE'],
        sizes=kwargs.get('sizes', [(480, 640), (720, 1280)]),
        num_runs=kwargs.get('num_runs', 3)
    )


def create_benchmark_pipeline(methods=None, **kwargs):
    """
    Create a configured unified benchmark pipeline
    
    Args:
        methods: List of methods to include
        **kwargs: Configuration parameters
        
    Returns:
        Configured UnifiedBenchmarkPipeline instance
    """
    if methods is None:
        methods = ['SIFT', 'ORB', 'AKAZE']
        if DEEP_LEARNING_AVAILABLE:
            methods.append('SuperPoint')
        if LIGHTGLUE_AVAILABLE:
            methods.append('lightglue')
    
    # Convert string benchmark types to enums
    benchmark_types = []
    bt_strings = kwargs.get('benchmark_types', ['performance'])
    for bt in bt_strings:
        if bt == 'performance':
            benchmark_types.append(BenchmarkType.PERFORMANCE)
        elif bt == 'accuracy':
            benchmark_types.append(BenchmarkType.ACCURACY)
        elif bt == 'comprehensive':
            benchmark_types.extend([BenchmarkType.PERFORMANCE, BenchmarkType.ACCURACY])
    
    config = UnifiedBenchmarkConfig(
        methods=methods,
        benchmark_types=benchmark_types,
        num_runs=kwargs.get('num_runs', 3),
        max_images=kwargs.get('max_images', None),
        resize_to=kwargs.get('resize_to', None),
        statistical_confidence=kwargs.get('confidence', 0.95),
        memory_profiling=kwargs.get('memory_profiling', True),
        save_results=kwargs.get('save_results', True),
        output_dir=kwargs.get('output_dir', "benchmark_results")
    )
    
    return UnifiedBenchmarkPipeline(config)


# Legacy function for backward compatibility
def benchmark_methods(img1, img2, methods=None, **kwargs):
    """
    Legacy benchmark function - use benchmark_folder or benchmark_single_image instead
    
    Args:
        img1: First image (numpy array)
        img2: Second image (numpy array)  
        methods: List of methods to benchmark
        **kwargs: Additional parameters
        
    Returns:
        Benchmark results
    """
    print("Warning: benchmark_methods is deprecated. Use benchmark_folder, benchmark_single_image, or benchmark_synthetic instead.")
    
    if methods is None:
        methods = ['SIFT', 'ORB', 'AKAZE']
        if LIGHTGLUE_AVAILABLE:
            methods.append('lightglue')
    
    config = BenchmarkConfig(
        num_runs=kwargs.get('num_runs', 3),
        statistical_confidence=kwargs.get('confidence_level', 0.95),
        memory_profiling=kwargs.get('memory_profiling', True)
    )
    
    benchmark = PerformanceBenchmark(config)
    return benchmark.run_statistical_benchmark(methods)


# Version and capability information
def get_version_info():
    """Get version and capability information"""
    info = {
        'version': __version__,
        'traditional_detectors': True,
        'deep_learning_detectors': DEEP_LEARNING_AVAILABLE,
        'lightglue_matcher': LIGHTGLUE_AVAILABLE,
        'unified_benchmarking': True,  # NEW
        'statistical_analysis': True,   # NEW
        'synthetic_image_generation': True,  # NEW
        'available_traditional': ['SIFT', 'ORB', 'AKAZE', 'BRISK', 'Harris', 'GoodFeatures'],
        'available_deep_learning': ['SuperPoint', 'DISK', 'ALIKED'] if DEEP_LEARNING_AVAILABLE else [],
        'available_matchers': ['FLANN', 'BF'] + (['LightGlue'] if LIGHTGLUE_AVAILABLE else []),
        'benchmark_features': [  # NEW
            'Unified pipeline for all image sources',
            'Performance benchmarking with statistics',
            'Accuracy testing with synthetic transformations', 
            'Memory profiling and resource tracking',
            'Statistical significance testing',
            'Support for folders, single images, and image lists',
            'Configurable synthetic image generation'
        ],
        'image_sources': [  # NEW
            'Synthetic images with transformations',
            'Real image folders',
            'Single image files',
            'Lists of images/paths',
            'Custom image sources'
        ]
    }
    return info


def print_capabilities():
    """Print available capabilities"""
    info = get_version_info()
    print(f"Feature Detection System v{info['version']}")
    print("=" * 70)
    print(f"Traditional Detectors: {', '.join(info['available_traditional'])}")
    
    if info['deep_learning_detectors']:
        print(f"Deep Learning Detectors: {', '.join(info['available_deep_learning'])}")
    else:
        print("Deep Learning Detectors: Not available (install PyTorch)")
    
    print(f"Matchers: {', '.join(info['available_matchers'])}")
    
    print("\nUnified Benchmarking Features:")
    for feature in info['benchmark_features']:
        print(f"  ✓ {feature}")
    
    print("\nSupported Image Sources:")
    for source in info['image_sources']:
        print(f"  • {source}")
    
    print("\nQuick Start Examples:")
    print("  # Basic detection")
    print("  features = detect_features(image, 'SIFT')")
    print("")
    print("  # Quick matching")
    print("  result = match_images(img1, img2, ['SIFT', 'ORB'])")
    print("")
    print("  # Benchmark folder of images")
    print("  results = benchmark_folder('/path/to/images', ['SIFT', 'ORB'])")
    print("")
    print("  # Benchmark single image")
    print("  results = benchmark_single_image('/path/to/image.jpg')")
    print("")
    print("  # Comprehensive synthetic benchmark")
    print("  results = benchmark_synthetic(['SIFT', 'ORB', 'AKAZE'])")
    print("")
    print("  # Custom pipeline")
    print("  pipeline = create_benchmark_pipeline(['SIFT', 'ORB'])")
    print("  results = pipeline.benchmark_folder('/path/to/images')")
    
    if not info['lightglue_matcher']:
        print("\nNote: LightGlue not available (install: pip install lightglue)")


# Export main components for easy access
__all__ = [
    # Core data structures
    'FeatureData', 'MatchData', 'EnhancedDMatch', 'DetectorType', 'ScoreType',
    'keypoints_to_serializable', 'keypoints_from_serializable',
    
    # Base classes
    'BaseFeatureDetector', 'BaseFeatureMatcher', 'BasePairMatcher',
    
    # Traditional detectors
    'SIFTDetector', 'ORBDetector', 'AKAZEDetector', 'BRISKDetector', 
    'HarrisCornerDetector', 'GoodFeaturesToTrackDetector',
    
    # Deep learning detectors (if available)
    'SuperPointDetector', 'DISKDetector', 'ALIKEDDetector',
    
    # Matchers
    'EnhancedFLANNMatcher', 'EnhancedBFMatcher', 'LightGlueMatcher',
    
    # Pipeline
    'MultiMethodFeatureDetector', 'FeatureProcessingPipeline',
    
    # Utilities
    'MatchQualityAnalyzer', 'visualize_matches_with_scores', 'visualize_keypoints',
    'extract_correspondences', 'save_enhanced_results', 'load_enhanced_results',
    'enhanced_filter_matches_with_homography', 'adaptive_match_filtering',
    'remove_duplicate_matches', 'verify_matches_with_fundamental_matrix',
    'calculate_reprojection_error', 'benchmark_detector_performance', 'get_images_from_folder',
    
    # Unified Benchmarking Pipeline (NEW)
    'UnifiedBenchmarkConfig', 'UnifiedBenchmarkPipeline', 'BenchmarkType', 'ImageSourceType',
    'ImageInfo', 'ImageSource', 'SyntheticImageSource', 'FolderImageSource', 
    'SingleImageSource', 'ImageListSource', 'BenchmarkTask', 'PerformanceTask', 'AccuracyTask',
    
    # Legacy Benchmarking (for compatibility)
    'BenchmarkConfig', 'BenchmarkResult', 'SyntheticImageGenerator', 'AdvancedQualityMetrics',
    'StatisticalAnalyzer', 'PerformanceBenchmark',
    
    # Factory functions
    'create_traditional_detector', 'create_deep_learning_detector',
    'create_traditional_matcher', 'create_deep_learning_matcher',
    'create_pipeline', 'auto_select_matcher',
    
    # Quick access functions (UPDATED)
    'detect_features', 'match_images', 'benchmark_folder', 'benchmark_single_image', 
    'benchmark_synthetic', 'create_benchmark_pipeline',
    
    # Quick benchmark functions (NEW)
    'quick_folder_benchmark', 'quick_synthetic_benchmark', 'quick_single_image_benchmark',
    
    # Legacy functions (for compatibility)
    'benchmark_methods', 'quick_test',
    
    # Configuration
    'get_default_config', 'validate_config', 'create_config_from_preset', 'print_available_presets',
    
    # Info functions
    'get_version_info', 'print_capabilities'
]