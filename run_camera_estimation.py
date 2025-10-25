#!/usr/bin/env python3
"""
Camera Pose Estimation Pipeline - Main Script
"""
# =============================================================================
# MAIN
# =============================================================================
from pathlib import Path
from typing import Optional
from CameraPoseEstimation2.core.interfaces import IMatchDataProvider
from CameraPoseEstimation2.pipeline.incremental import IncrementalReconstructionPipeline
from CameraPoseEstimation2.data import create_provider

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input/Output
MATCH_RESULTS_DIR = './output/matching_results'
OUTPUT_DIR = './reconstruction_output'

# Provider Settings
CACHE_SIZE = 100
PRELOAD_ALL = False
MIN_QUALITY = None
MIN_MATCHES = None



def load_matching_results(
    matching_output_dir: str,
    cache_size: int = 100,
    min_quality: Optional[float] = None,
    min_matches: Optional[int] = None
) -> IMatchDataProvider:
    """
    Load feature matching results for reconstruction pipeline.
    
    This method reads the output from FeatureMatchingExtraction pipeline
    and creates a data provider for CameraPoseEstimation2 pipeline.
    
    Args:
        matching_output_dir: Path to the 'matching_results' directory 
                           from FeatureMatchingExtraction output
        cache_size: Number of image pairs to keep in memory (default: 100)
        min_quality: Filter pairs below this quality (None = no filter)
        min_matches: Filter pairs with fewer matches (None = no filter)
    
    Returns:
        IMatchDataProvider: Provider ready for reconstruction pipeline
    
    Example:
        >>> # Read matching results
        >>> provider = load_matching_results('./output/matching_results')
        >>> 
        >>> # Use with reconstruction pipeline
        >>> from CameraPoseEstimation2.pipeline.incremental import IncrementalReconstructionPipeline
        >>> pipeline = IncrementalReconstructionPipeline(provider, output_dir='./reconstruction')
        >>> reconstruction = pipeline.run()
    """
    
    # Convert to Path for validation
    matching_dir = Path(matching_output_dir)
    
    # Validate directory exists
    if not matching_dir.exists():
        raise FileNotFoundError(
            f"Matching results directory not found: {matching_output_dir}\n"
            f"Make sure you've run the feature matching pipeline first."
        )
    
    # Look for matching_results subdirectory if it exists
    if (matching_dir / 'matching_results').exists():
        matching_dir = matching_dir / 'matching_results'
    
    # Validate we have the required files
    batch_files = list(matching_dir.glob('*_batch_*.pkl'))
    metadata_files = list(matching_dir.glob('*_image_metadata.pkl'))
    
    if not batch_files:
        raise FileNotFoundError(
            f"No batch result files found in: {matching_dir}\n"
            f"Expected files matching pattern: *_batch_*.pkl"
        )
    
    if not metadata_files:
        raise FileNotFoundError(
            f"No image metadata file found in: {matching_dir}\n"
            f"Expected file matching pattern: *_image_metadata.pkl"
        )
    
    # Create provider
    provider = create_provider(
        path=str(matching_dir),
        cache_size=cache_size,
        preload_all=False,
        min_quality=min_quality,
        min_matches=min_matches
    )
    
    return provider


# =============================================================================
# COMPLETE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    """
    Complete example: Load matching results and run reconstruction
    """
    
    # Configuration
    MATCHING_OUTPUT = './output'  # Output directory from feature matching
    RECONSTRUCTION_OUTPUT = './reconstruction_output'
    
    try:
        # Load matching results
        print("Loading matching results...")
        provider = load_matching_results(
            matching_output_dir=MATCHING_OUTPUT,
            cache_size=100,
            min_quality=None,  # No quality filter
            min_matches=10     # Require at least 10 matches
        )
        
        print(f"✓ Loaded {len(list(provider.get_all_images()))} images")
        print(f"✓ Found {len(list(provider.get_all_pairs()))} image pairs")
        
        # Run reconstruction
        print("\nRunning reconstruction pipeline...")
        from CameraPoseEstimation2.pipeline.incremental import IncrementalReconstructionPipeline
        
        pipeline = IncrementalReconstructionPipeline(
            provider=provider,
            output_dir=RECONSTRUCTION_OUTPUT
        )
        
        reconstruction = pipeline.run()
        
        print("\n" + "="*70)
        print("COMPLETE!")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    #reconstruction = pipeline.run()