#!/usr/bin/env python3
"""
Feature Matching Pipeline - Main Script
"""

from FeatureMatchingExtraction import FeatureProcessingPipeline

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input/Output
INPUT_FOLDER = './images/statue_of_liberty_images'
OUTPUT_FOLDER = './output'
IMAGE_PATTERN = '*.jpg'

# Methods
METHODS = ['SIFT', 'ORB', 'AKAZE', 'SuperPoint','ALIKED']
MAX_FEATURES = 30000

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
    'AKAZE': {
        'threshold': 0.001,
        'n_octaves': 4,
    }
}

# Matcher Configuration
MATCHER_CONFIG = {
    'SIFT': 'flann',
    'ORB': 'bf',
    'AKAZE': 'bf'
}

# Filtering
FILTERING = {
    'use_adaptive_filtering': False,
    'ransac_threshold': 4.0,
    'confidence': 0.99,
    'max_iters': 2000
}

# Processing
PAIR_MODE = 'all'
BATCH_SIZE = 10
CACHE_SIZE_MB = 500

# Output
AUTO_SAVE = True
MIN_MATCHES_FOR_SAVE = 5
SAVE_VISUALIZATIONS = False

COMBINE_STRATEGY = 'independent'

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    config = {
        'methods': METHODS,
        'max_features': MAX_FEATURES,
        'detector_params': DETECTOR_PARAMS,
        'matcher_config': MATCHER_CONFIG,
        'filtering': FILTERING,
        'combine_strategy': COMBINE_STRATEGY
    }
    
    pipeline = FeatureProcessingPipeline(config)
    
    results = pipeline.match_folder(
        folder_path=INPUT_FOLDER,
        pattern=IMAGE_PATTERN,
        pair_mode=PAIR_MODE,
        batch_size=BATCH_SIZE,
        cache_size_mb=CACHE_SIZE_MB,
        output_dir=OUTPUT_FOLDER,
        auto_save=AUTO_SAVE,
        min_matches_for_save=MIN_MATCHES_FOR_SAVE,
        save_visualizations=SAVE_VISUALIZATIONS
    )