#!/usr/bin/env python3
"""
Feature Matching Pipeline - Main Script

Runs feature detection and matching with ORB, SIFT, SuperPoint, and DKM

Requirements:
    - Basic: opencv-python, numpy
    - SuperPoint: torch, lightglue (https://github.com/cvg/LightGlue)
    - DKM: torch,  gim-dkm from source
           https://github.com/xuelunshen/gim
"""

from FeatureMatchingExtraction import create_pipeline
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input/Output
INPUT_FOLDER = './images/statue_of_liberty_images'
OUTPUT_FOLDER = './output'
IMAGE_PATTERN = '*.jpg'

# Methods - Mix of sparse and dense methods
METHODS = ['SIFT', 'ORB', 'SuperPoint', 'DKM']
MAX_FEATURES = 8000

# Logging
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = './output/processing.log'

# Detector Parameters
DETECTOR_PARAMS = {
    'SIFT': {
        'contrast_threshold': 0.04,
        'edge_threshold': 10,
        'sigma': 1.6
    },
    'ORB': {
        'scale_factor': 1.2,
        'n_levels': 8,
        'edge_threshold': 31
    },
    'SuperPoint': {
        'device': 'cuda',  # or 'cpu'
        'max_keypoints': 8000
    },
    'DKM': {
        'device': 'cuda',  # or 'cpu'
        'resize_to': (896, 672),  # Training resolution for best results
        'weights': 'gim_dkm_100h',
        'max_matches': 8000
    }
}

# Matcher Configuration
MATCHER_CONFIG = {
    'SIFT': 'FLANN',
    'ORB': 'BruteForce',      # ORB needs BruteForce, not 'bf'
    'SuperPoint': 'LightGlue',  # SuperPoint works best with LightGlue
    'DKM': None  # DKM is end-to-end (detector + matcher in one)
}

# Filtering
FILTERING = {
    'use_adaptive_filtering': False,
    'ransac_threshold': 4.0,
    'confidence': 0.99,
    'max_iters': 2000
}

# Processing
PAIR_MODE = 'all'  # 'all', 'consecutive', 'first'
BATCH_SIZE = 10
CACHE_SIZE_MB = 500

# Output
AUTO_SAVE = True
MIN_MATCHES_FOR_SAVE = 10
SAVE_VISUALIZATIONS = False
RESUME = False  # Set to True to resume from checkpoint

COMBINE_STRATEGY = 'independent'  # Keep each method independent

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("="*70)
    print("FEATURE MATCHING PIPELINE")
    print("="*70)
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Methods: {METHODS}")
    print(f"Max features: {MAX_FEATURES}")
    print("="*70)

    # Create pipeline with logging
    pipeline = create_pipeline(
        preset='custom',
        methods=METHODS,
        max_features=MAX_FEATURES,
        detector_params=DETECTOR_PARAMS,
        matcher_config=MATCHER_CONFIG,
        filtering=FILTERING,
        combine_strategy=COMBINE_STRATEGY,
        log_level=LOG_LEVEL,
        log_file=LOG_FILE
    )

    # Run batch processing
    try:
        results = pipeline.match_folder(
            folder_path=INPUT_FOLDER,
            pattern=IMAGE_PATTERN,
            pair_mode=PAIR_MODE,
            batch_size=BATCH_SIZE,
            cache_size_mb=CACHE_SIZE_MB,
            output_dir=OUTPUT_FOLDER,
            auto_save=AUTO_SAVE,
            min_matches_for_save=MIN_MATCHES_FOR_SAVE,
            save_visualizations=SAVE_VISUALIZATIONS,
            resume=RESUME
        )

        print(f"\nProcessing complete! Processed {len(results)} pairs")
        print(f"Results saved to: {OUTPUT_FOLDER}")
        print(f"Log file: {LOG_FILE}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)