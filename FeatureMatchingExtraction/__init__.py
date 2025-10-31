"""
FeatureMatchingExtraction - Modern Feature Detection and Matching System

A comprehensive system for feature detection, matching, and 3D reconstruction
with batch processing, smart caching, and fault tolerance.

Key Features:
- Multi-method feature detection (SIFT, ORB, AKAZE, ALIKED, etc.)
- Batch processing with smart image caching
- Checkpointing and resume support
- Memory efficient (bounded RAM usage)
- Automatic result saving
- Export to COLMAP format

Quick Start:
    >>> from FeatureMatchingExtraction import create_pipeline
    >>> 
    >>> # Create pipeline
    >>> pipeline = create_pipeline('balanced')
    >>> 
    >>> # Match single pair
    >>> result = pipeline.match(img1, img2)
    >>> 
    >>> # Batch process folder with auto-save
    >>> results = pipeline.match_folder(
    ...     './images',
    ...     output_dir='./output',
    ...     auto_save=True,
    ...     batch_size=10
    ... )
"""

__version__ = '2.0.0'

# =============================================================================
# CORE PIPELINE
# =============================================================================

from .pipeline import (
    FeatureProcessingPipeline,
    create_pipeline
)

# =============================================================================
# RESULT TYPES
# =============================================================================

from .result_types import (
    MatchingResult,
    MethodResult,
    ImagePairInfo,
    ProcessingMetadata,
    create_method_result,
    save_results_batch,
    load_results_batch,
    export_summary_csv,
)

# =============================================================================
# CONVERTED RESULT TYPES
# =============================================================================

from .result_converters import (
    VisualizationData,
    VisualMatch,
    MultiMethodReconstruction,  # NEW NAME
    MethodReconstructionData,
    ResultConverter,
    save_for_reconstruction,
    load_for_reconstruction,
)

# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

from .core_data_structures import (
    FeatureData,
    MatchData,
    EnhancedDMatch,
    ScoreType,
    MultiMethodMatchData,
)

from .multi_method_detector import MultiMethodFeatureDetector
# =============================================================================
# IMAGE MANAGEMENT (NEW!)
# =============================================================================

from .image_manager import (
    # Core classes
    ImageMetadata,
    ImageInfo,
    ImageSourceType,
    
    # Cache and loading
    ImageCache,
    BatchImageLoader,
    FolderImageSource,
    
    # Helper functions
    create_pairs_from_metadata,
    analyze_batch_reuse,
    estimate_batch_memory,
    scan_folder_quick,
)

# =============================================================================
# BATCH PROCESSING (NEW!)
# =============================================================================

from .batch_processor import (
    BatchProcessor,
    load_progress,
    delete_progress,
    get_remaining_pairs,
)

# =============================================================================
# DETECTORS (from matchers module)
# =============================================================================

# Import from new matchers module
from .matchers.detectors.traditional import (
    SIFTDetector,
    ORBDetector,
    AKAZEDetector,
)

# Deep learning detectors (if available)
try:
    from .matchers.detectors.deep_learning import (
        SuperPointDetector,
        ALIKEDDetector,
    )
    _has_deep_learning = True
except ImportError:
    _has_deep_learning = False

# Multi-method detector (uses matchers module internally)
from .multi_method_detector import MultiMethodFeatureDetector

# =============================================================================
# MATCHERS (from matchers module)
# =============================================================================

from .matchers import (
    create_matcher,
    create_dense_matcher,
    DenseMatcher,
    SparsePipeline,
)

# =============================================================================
# VISUALIZATION
# =============================================================================

from .visualization import (
    plot_visualization_data,
    plot_method_comparison,
    visualize_matches_quick,
    show_matches,
    visualize_matches_with_scores,
    save_visualization,
    visualize_keypoints_only,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

from .config import (
    get_default_config,
    create_config_from_preset,
)

# =============================================================================
# UTILITIES
# =============================================================================

from .utils import (
    # Size helpers
    validate_size,
    image_size_from_shape,
    resize_image,
    print_size_info,

    # Filtering
    enhanced_filter_matches_with_homography,
    adaptive_match_filtering,
    calculate_reprojection_error,

    # Serialization helpers
    keypoint_to_dict,
    dict_to_keypoint,
    keypoints_to_list,
    list_to_keypoints,
)

# =============================================================================
# LOGGING
# =============================================================================

from .logger import (
    setup_logger,
    get_logger,
    configure_root_logger,
    disable_console_logging,
    set_level,
)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Core Pipeline
    'FeatureProcessingPipeline',
    'create_pipeline',
    
    # Result Types
    'MatchingResult',
    'MethodResult',
    'ImagePairInfo',
    'ProcessingMetadata',
    'save_for_reconstruction',
    'load_for_reconstruction',
    'save_results_batch',
    'load_results_batch',
    'export_summary_csv',
    
    # Converted Results
    'VisualizationData',
    'ReconstructionData',
    'MethodReconstructionData',
    'ResultConverter',
    'VisualMatch',
    
    # Core Data Structures
    'FeatureData',
    'MatchData',
    'EnhancedDMatch',
    'ScoreType',
    'MultiMethodFeatureData',
    'MultiMethodMatchData',
    
    # Image Management
    'ImageMetadata',
    'ImageInfo',
    'ImageSourceType',
    'ImageCache',
    'BatchImageLoader',
    'FolderImageSource',
    'create_pairs_from_metadata',
    'analyze_batch_reuse',
    'estimate_batch_memory',
    'scan_folder_quick',
    
    # Batch Processing
    'BatchProcessor',
    'load_progress',
    'delete_progress',
    'get_remaining_pairs',
    
    # Detectors (from matchers module)
    'SIFTDetector',
    'ORBDetector',
    'AKAZEDetector',
    'MultiMethodFeatureDetector',

    # Matchers (from matchers module)
    'create_matcher',
    'create_dense_matcher',
    'DenseMatcher',
    'SparsePipeline',
    
    # Visualization
    'plot_visualization_data',
    'plot_method_comparison',
    'visualize_matches_quick',
    'show_matches',
    'visualize_matches_with_scores',
    'save_visualization',
    'visualize_keypoints_only',
    
    # Configuration
    'get_default_config',
    'create_config_from_preset',
    
    # Utilities
    'validate_size',
    'image_size_from_shape',
    'resize_image',
    'print_size_info',
    'enhanced_filter_matches_with_homography',
    'adaptive_match_filtering',
    'calculate_reprojection_error',
    'keypoint_to_dict',
    'dict_to_keypoint',
    'keypoints_to_list',
    'list_to_keypoints',

    # Logging
    'setup_logger',
    'get_logger',
    'configure_root_logger',
    'disable_console_logging',
    'set_level',
]

# Add deep learning detectors if available
if _has_deep_learning:
    __all__.extend([
        'SuperPointDetector',
        'ALIKEDDetector',
    ])

# =============================================================================
# MODULE INFO
# =============================================================================

def get_version():
    """Get package version"""
    return __version__


def get_available_methods():
    """
    Get list of available feature detection methods
    
    Returns:
        Dictionary with method categories and availability
    """
    methods = {
        'traditional': {
            'SIFT': True,
            'ORB': True,
            'AKAZE': True,
            'BRISK': True,
        },
        'deep_learning': {
            'SuperPoint': _has_deep_learning,
            'ALIKED': _has_deep_learning,
        }
    }
    return methods


def check_dependencies():
    """
    Check which optional dependencies are available
    
    Returns:
        Dictionary with dependency availability
    """
    deps = {
        'opencv': True,  # Required
        'numpy': True,   # Required
        'torch': _has_deep_learning,
        'matplotlib': True,  # Usually available
    }
    
    # Check matplotlib
    try:
        import matplotlib
        deps['matplotlib'] = True
    except ImportError:
        deps['matplotlib'] = False
    
    return deps


def print_info():
    """Print package information"""
    print("="*70)
    print(f"FeatureMatchingExtraction v{__version__}")
    print("="*70)
    
    print("\n📦 Available Methods:")
    methods = get_available_methods()
    
    print("\n  Traditional:")
    for method, available in methods['traditional'].items():
        status = "✓" if available else "✗"
        print(f"    {status} {method}")
    
    print("\n  Deep Learning:")
    for method, available in methods['deep_learning'].items():
        status = "✓" if available else "✗"
        print(f"    {status} {method}")
    
    print("\n🔧 Dependencies:")
    deps = check_dependencies()
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")
    
    print("\n" + "="*70)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_match(
    image1,
    image2,
    method: str = 'SIFT',
    visualize: bool = False
):
    """
    Quick match two images with a single method
    
    Args:
        image1: First image (RGB numpy array or path)
        image2: Second image (RGB numpy array or path)
        method: Method name (default: 'SIFT')
        visualize: Show visualization
    
    Returns:
        MatchingResult
    
    Example:
        >>> import cv2
        >>> from FeatureMatchingExtraction import quick_match
        >>> 
        >>> img1 = cv2.imread('img1.jpg')
        >>> img2 = cv2.imread('img2.jpg')
        >>> result = quick_match(img1, img2, method='SIFT', visualize=True)
    """
    import cv2
    import numpy as np
    
    # Load images if paths provided
    if isinstance(image1, str):
        image1 = cv2.imread(image1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    
    if isinstance(image2, str):
        image2 = cv2.imread(image2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    
    # Create pipeline
    pipeline = create_pipeline('custom', methods=[method])
    
    # Match
    result = pipeline.match(image1, image2, visualize=visualize)
    
    return result


def quick_process_folder(
    folder_path: str,
    output_dir: str,
    method: str = 'SIFT',
    batch_size: int = 10,
    **kwargs
):
    """
    Quick process entire folder with auto-save
    
    Args:
        folder_path: Path to image folder
        output_dir: Output directory
        method: Method to use (default: 'SIFT')
        batch_size: Batch size (default: 10)
        **kwargs: Additional arguments for match_folder
    
    Returns:
        List of MatchingResult
    
    Example:
        >>> from FeatureMatchingExtraction import quick_process_folder
        >>> 
        >>> results = quick_process_folder(
        ...     './my_images',
        ...     './output',
        ...     method='SIFT',
        ...     batch_size=10
        ... )
    """
    # Create pipeline
    pipeline = create_pipeline('custom', methods=[method])
    
    # Process folder with auto-save
    results = pipeline.match_folder(
        folder_path=folder_path,
        output_dir=output_dir,
        auto_save=True,
        batch_size=batch_size,
        **kwargs
    )
    
    return results


# =============================================================================
# MIGRATION HELPERS (for backward compatibility)
# =============================================================================

def migrate_from_v1():
    """
    Helper function to migrate from v1.x to v2.0
    
    Prints migration guide
    """
    guide = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║           Migration Guide: v1.x → v2.0                          ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    BREAKING CHANGES:
    
    1. Image Loading:
       OLD: Uses ImageLoader (no longer exists)
       NEW: Uses FolderImageSource from image_manager
       
       Before:
       >>> from FeatureMatchingExtraction import ImageLoader
       >>> loader = ImageLoader()
       
       After:
       >>> from FeatureMatchingExtraction import FolderImageSource
       >>> source = FolderImageSource(folder_path)
    
    2. Batch Processing:
       OLD: Manual loop + save
       NEW: Built into match_folder()
       
       Before:
       >>> results = pipeline.match_folder('./images')
       >>> for result in results:
       ...     result.save(...)
       
       After:
       >>> results = pipeline.match_folder(
       ...     './images',
       ...     output_dir='./output',
       ...     auto_save=True,
       ...     batch_size=10
       ... )
    
    3. Reconstruction Saving:
       OLD: pickle.dump(recon, f)
       NEW: recon.save(path)
       
       Before:
       >>> with open('recon.pkl', 'wb') as f:
       ...     pickle.dump(recon, f)
       
       After:
       >>> recon.save('recon.pkl')
    
    NEW FEATURES:
    
    ✨ Batch processing with smart caching
    ✨ Checkpointing and resume support
    ✨ Memory efficient (100x less RAM!)
    ✨ Cache hit tracking
    ✨ Better progress reporting
    
    For more details, see documentation.
    """
    print(guide)


# =============================================================================
# STARTUP MESSAGE (optional, can be disabled)
# =============================================================================

def _print_startup_message():
    """Print startup message (only if explicitly called)"""
    import sys
    
    if '--verbose' in sys.argv or '-v' in sys.argv:
        print(f"✓ FeatureMatchingExtraction v{__version__} loaded")
        if _has_deep_learning:
            print("✓ Deep learning methods available")

# Uncomment to show startup message:
# _print_startup_message()