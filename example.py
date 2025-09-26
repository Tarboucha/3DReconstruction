"""
Usage examples for the multi-method feature detection system.

This module provides comprehensive examples showing how to use various
components of the feature detection and matching system.
"""

import cv2
import numpy as np
import os
from typing import List, Tuple
import time

import Feature as fds
from Feature import config, MatchQualityAnalyzer, create_traditional_detector, create_deep_learning_detector, create_pipeline,  benchmark_detector_performance, verify_matches_with_fundamental_matrix, calculate_reprojection_error, remove_duplicate_matches

# =============================================================================
# Basic Examples
# =============================================================================

def example_01_simple_feature_detection():
    """Example 1: Simple feature detection with a single method"""
    print("Example 1: Simple Feature Detection")
    print("=" * 40)
    
    # Load a sample image (you'll need to provide your own)
    image_path = "sample_image.jpg"  # Replace with your image path
    
    if not os.path.exists(image_path):
        print(f"Please provide a sample image at: {image_path}")
        print("Creating a synthetic image for demonstration...")
        
        # Create a synthetic image with some features
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        # Add some rectangles and circles
        cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.circle(image, (300, 200), 50, (128, 128, 128), -1)
        cv2.rectangle(image, (400, 250), (550, 350), (200, 200, 200), -1)
    else:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
    
    # Method 1: Using the simple interface
    try:
        features = fds.detect_features(image, method='SIFT', max_features=1000)
        print(f"SIFT detected {len(features)} features")
        print(f"Detection time: {features.detection_time:.3f}s")
        
        # Visualize features
        fds.visualize_keypoints(image, features, title="SIFT Features")
        
    except Exception as e:
        print(f"Feature detection failed: {e}")
    
    # Method 2: Using detector classes directly
    try:
        sift_detector = create_traditional_detector('SIFT', max_features=1000)
        orb_detector = create_traditional_detector('ORB', max_features=1000)
        
        sift_features = sift_detector.detect(image)
        orb_features = orb_detector.detect(image)
        
        print(f"\nDirect detection:")
        print(f"SIFT: {len(sift_features)} features in {sift_features.detection_time:.3f}s")
        print(f"ORB: {len(orb_features)} features in {orb_features.detection_time:.3f}s")
        
    except Exception as e:
        print(f"Direct detection failed: {e}")


def example_02_simple_matching():
    """Example 2: Simple feature matching between two images"""
    print("\nExample 2: Simple Feature Matching")
    print("=" * 40)
    
    # For this example, we'll create two similar synthetic images
    img1 = np.zeros((300, 400, 3), dtype=np.uint8)
    img2 = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Add some common features with slight differences
    cv2.rectangle(img1, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.rectangle(img2, (55, 55), (155, 155), (255, 255, 255), -1)  # Slightly shifted
    
    cv2.circle(img1, (250, 150), 40, (128, 128, 128), -1)
    cv2.circle(img2, (245, 155), 40, (128, 128, 128), -1)  # Slightly shifted
    
    try:
        # Simple matching interface
        result = fds.match_images(img1, img2, methods='SIFT', visualize=True)
        
        print(f"Found {len(result['match_data'].get_best_matches())} matches")
        print(f"Method used: {result['method_used']}")
        
        # Extract correspondences
        if isinstance(result['correspondences'], tuple):
            correspondences, scores = result['correspondences']
            print(f"Mean match score: {np.mean(scores):.3f}")
        else:
            correspondences = result['correspondences']
            
        print(f"Correspondences shape: {correspondences.shape}")
        
    except Exception as e:
        print(f"Matching failed: {e}")


def example_03_multi_method_comparison():
    """Example 3: Compare multiple detection methods"""
    print("\nExample 3: Multi-Method Comparison")
    print("=" * 40)
    
    # Create a test image with various features
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Add different types of features
    cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(image, (300, 200), 50, (128, 128, 128), -1)
    cv2.rectangle(image, (400, 250), (550, 350), (200, 200, 200), 2)
    
    # Add some texture
    for i in range(10):
        cv2.line(image, (200 + i*5, 50), (200 + i*5, 150), (100, 100, 100), 1)
    
    # Test multiple methods
    methods_to_test = ['SIFT', 'ORB', 'AKAZE', 'BRISK']
    results = {}
    
    for method in methods_to_test:
        try:
            start_time = time.time()
            features = fds.detect_features(image, method=method, max_features=1000)
            detection_time = time.time() - start_time
            
            results[method] = {
                'num_features': len(features),
                'detection_time': detection_time,
                'avg_response': np.mean([kp.response for kp in features.keypoints]) if features.keypoints else 0
            }
            
            print(f"{method}: {len(features)} features in {detection_time:.3f}s")
            
        except Exception as e:
            print(f"{method} failed: {e}")
            results[method] = {'error': str(e)}
    
    # Print comparison
    print("\nComparison Summary:")
    print("-" * 50)
    print(f"{'Method':<8} {'Features':<8} {'Time(s)':<8} {'Avg Response':<12}")
    print("-" * 50)
    
    for method, result in results.items():
        if 'error' not in result:
            print(f"{method:<8} {result['num_features']:<8} {result['detection_time']:<8.3f} {result['avg_response']:<12.3f}")
        else:
            print(f"{method:<8} {'ERROR':<8}")


# =============================================================================
# Intermediate Examples
# =============================================================================

def example_04_pipeline_with_presets():
    """Example 4: Using pipeline with different presets"""
    print("\nExample 4: Pipeline with Presets")
    print("=" * 40)
    
    # Show available presets
    print("Available presets:")
    fds.print_available_presets()
    
    # Create test images
    img1 = np.zeros((300, 400, 3), dtype=np.uint8)
    img2 = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Add features
    cv2.rectangle(img1, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.rectangle(img2, (60, 55), (160, 155), (255, 255, 255), -1)
    
    cv2.circle(img1, (250, 150), 40, (200, 200, 200), -1)
    cv2.circle(img2, (245, 155), 40, (200, 200, 200), -1)
    
    # Test different presets
    presets_to_test = ['fast', 'balanced', 'accurate']
    
    for preset in presets_to_test:
        try:
            print(f"\nTesting preset: {preset}")
            print("-" * 30)
            
            # Create pipeline with preset
            pipeline = create_pipeline(preset)
            
            # Process image pair
            result = pipeline.process_image_pair(img1, img2, visualize=False)
            
            print(f"Method used: {result['method_used']}")
            print(f"Matches found: {len(result['match_data'].get_best_matches())}")
            
            # Analyze quality
            analyzer = MatchQualityAnalyzer()
            analysis = analyzer.analyze_match_data(result['match_data'])
            print(f"Quality score: {analysis['quality_score']:.3f}")
            
        except Exception as e:
            print(f"Preset {preset} failed: {e}")


def example_05_custom_configuration():
    """Example 5: Creating custom configurations"""
    print("\nExample 5: Custom Configuration")
    print("=" * 40)
    
    # Create a custom configuration

    
    custom_config = config.create_custom_config(
        methods=['SIFT', 'ORB'],
        max_features=1500,
        combine_strategy='weighted',
        filtering={
            'use_adaptive_filtering': True,
            'ransac_threshold': 4.0,
            'top_k': 100
        },
        visualization={
            'max_matches_display': 30,
            'show_histogram': True
        }
    )
    
    # Validate configuration
    validation = config.validate_config(custom_config)
    if validation['errors']:
        print("Configuration errors:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    if validation['warnings']:
        print("Configuration warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    # Print configuration
    config.print_config(custom_config, "Custom Configuration")
    
    # Use the custom configuration
    try:
        pipeline = fds.FeatureProcessingPipeline(custom_config)
        
        # Create test images
        img1 = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        img2 = np.roll(img1, 10, axis=1)  # Shift image slightly
        
        result = pipeline.process_image_pair(img1, img2, visualize=False)
        print(f"\nCustom pipeline result: {len(result['match_data'].get_best_matches())} matches")
        
    except Exception as e:
        print(f"Custom configuration failed: {e}")


def example_06_comprehensive_analysis():
    """Example 6: Comprehensive analysis with multiple methods"""
    print("\nExample 6: Comprehensive Analysis")
    print("=" * 40)
    
    # Create more complex test images
    img1 = np.zeros((400, 600, 3), dtype=np.uint8)
    img2 = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Add various features to both images
    features = [
        ((50, 50), (150, 150)),   # Rectangle
        ((200, 100), (240,140)),         # Circle  
        ((350, 200), (450, 300)), # Another rectangle
        ((500, 50), (580, 130))   # Small rectangle
    ]
    
    for i, feature in enumerate(features):
        color = (255 - i*50, 255 - i*30, 255 - i*40)
        
        if len(feature[0]) == 2 and len(feature[1]) == 2:  # Rectangle
            cv2.rectangle(img1, feature[0], feature[1], color, -1)
            # Add slight transformation for img2
            offset = (5, 5)
            cv2.rectangle(img2, 
                         (feature[0][0] + offset[0], feature[0][1] + offset[1]),
                         (feature[1][0] + offset[0], feature[1][1] + offset[1]), 
                         color, -1)
        else:  # Circle
            cv2.circle(img1, feature[0], feature[1], color, -1)
            cv2.circle(img2, (feature[0][0] + 3, feature[0][1] + 3), feature[1], color, -1)
    
    try:
        # Use comprehensive analysis
        result = fds.benchmark_methods(
            img1, img2, 
            methods=['SIFT', 'ORB', 'AKAZE'], 
            visualize=False
        )
        
        print("Comprehensive Analysis Results:")
        print("-" * 40)
        
        if 'quality_analysis' in result:
            for method, analysis in result['quality_analysis'].items():
                if method not in ['best_method', 'best_quality']:
                    print(f"\n{method}:")
                    print(f"  Matches: {analysis['num_matches']}")
                    print(f"  Quality Score: {analysis['quality_score']:.3f}")
                    print(f"  Score Type: {analysis['score_type']}")
                    print(f"  Mean Score: {analysis['mean_score']:.3f}")
            
            if 'best_method' in result['quality_analysis']:
                print(f"\nBest Method: {result['quality_analysis']['best_method']}")
                print(f"Best Quality: {result['quality_analysis']['best_quality']:.3f}")
        
        # Show detailed match information
        if 'all_results' in result:
            print(f"\nDetailed Results:")
            for method_name, method_result in result['all_results'].items():
                match_data = method_result['match_data']
                print(f"  {method_name}: {len(match_data.get_best_matches())} matches")
                print(f"    Score type: {match_data.score_type.value}")
                print(f"    Processing time: {match_data.matching_time:.3f}s")
        
    except Exception as e:
        print(f"Comprehensive analysis failed: {e}")


# =============================================================================
# Advanced Examples
# =============================================================================

def example_07_deep_learning_methods():
    """Example 7: Using deep learning methods (if available)"""
    print("\nExample 7: Deep Learning Methods")
    print("=" * 40)
    
    # Check if deep learning methods are available
    info = fds.get_version_info()
    
    if not info['deep_learning_detectors']:
        print("Deep learning detectors not available (PyTorch not installed)")
        print("Install with: pip install torch torchvision")
        return
    
    if not info['lightglue_matcher']:
        print("LightGlue matcher not available")
        print("Install with: pip install lightglue")
        print("Falling back to SuperPoint detection only...")
    
    # Create test images
    img1 = np.zeros((300, 400, 3), dtype=np.uint8)
    img2 = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Add some features
    cv2.rectangle(img1, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.rectangle(img2, (55, 55), (155, 155), (255, 255, 255), -1)
    
    cv2.circle(img1, (250, 150), 40, (200, 200, 200), -1)
    cv2.circle(img2, (245, 155), 40, (200, 200, 200), -1)
    
    try:
        if info['lightglue_matcher']:
            # Test LightGlue
            print("Testing LightGlue...")
            result = fds.match_images(img1, img2, methods='lightglue', visualize=False)
            print(f"LightGlue found {len(result['match_data'].get_best_matches())} matches")
            print(f"Score type: {result['match_data'].score_type.value}")
            
            if result['match_data'].match_confidences is not None:
                print(f"Mean confidence: {np.mean(result['match_data'].match_confidences):.3f}")
        
        # Test SuperPoint detection
        if 'SuperPoint' in info['available_deep_learning']:
            print("\nTesting SuperPoint detection...")
            try:
                superpoint = create_deep_learning_detector('SuperPoint', max_features=1000)
                
                features1 = superpoint.detect(img1)
                features2 = superpoint.detect(img2)
                
                print(f"SuperPoint detected {len(features1)} and {len(features2)} features")
                print(f"Detection times: {features1.detection_time:.3f}s, {features2.detection_time:.3f}s")
                
                if features1.confidence_scores:
                    print(f"Mean confidence: {np.mean(features1.confidence_scores):.3f}")
                
            except Exception as e:
                print(f"SuperPoint detection failed: {e}")
        
    except Exception as e:
        print(f"Deep learning methods failed: {e}")


def example_08_dataset_processing():
    """Example 8: Processing a dataset of image pairs"""
    print("\nExample 8: Dataset Processing")
    print("=" * 40)
    
    # Create a small synthetic dataset
    print("Creating synthetic dataset...")
    
    dataset_images = []
    for i in range(5):
        # Create slightly different images
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        
        # Add some random features
        for j in range(3):
            x, y = np.random.randint(20, 280), np.random.randint(20, 180)
            size = np.random.randint(10, 30)
            color = tuple(np.random.randint(100, 255, 3).tolist())
            
            if j % 2 == 0:
                cv2.rectangle(img, (x, y), (x+size, y+size), color, -1)
            else:
                cv2.circle(img, (x, y), size//2, color, -1)
        
        dataset_images.append((img, f"image_{i:02d}.jpg"))
    
    # Create pipeline for dataset processing
    config = fds.create_config_from_preset('balanced')
    pipeline = fds.FeatureProcessingPipeline(config)
    
    try:
        # Process dataset
        print("Processing dataset...")
        results = pipeline.process_dataset(
            dataset_images, 
            output_file="dataset_results.json",
            max_pairs=10,  # Limit pairs for demo
            save_format='json'
        )
        
        # Print summary
        stats = results['stats']
        print(f"\nDataset Processing Summary:")
        print(f"  Total pairs: {stats['total_pairs']}")
        print(f"  Successful pairs: {stats['successful_pairs']}")
        print(f"  Success rate: {stats['successful_pairs']/stats['total_pairs']*100:.1f}%")
        print(f"  Average matches: {stats.get('avg_matches', 0):.1f}")
        print(f"  Total processing time: {stats['processing_time']:.1f}s")
        
        # Show some individual results
        print(f"\nIndividual Pair Results:")
        pair_results = results['results']
        for i, (pair_key, pair_result) in enumerate(list(pair_results.items())[:3]):
            if 'error' not in pair_result:
                print(f"  {pair_key}: {pair_result['num_matches']} matches "
                      f"(quality: {pair_result.get('quality_score', 0):.3f})")
            else:
                print(f"  {pair_key}: ERROR")
        
    except Exception as e:
        print(f"Dataset processing failed: {e}")


def example_09_benchmarking_and_performance():
    """Example 9: Benchmarking detector performance"""
    print("\nExample 9: Benchmarking and Performance")
    print("=" * 40)
    
    # Create test images of different sizes
    test_images = []
    sizes = [(200, 300), (400, 600), (600, 800)]
    
    for i, (h, w) in enumerate(sizes):
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        
        # Add some structured features
        for j in range(5):
            x, y = np.random.randint(20, w-50), np.random.randint(20, h-50)
            cv2.rectangle(img, (x, y), (x+30, y+30), (255, 255, 255), -1)
        
        test_images.append(img)
    
    # Test different detectors
    methods_to_benchmark = ['SIFT', 'ORB', 'AKAZE']
    
    print("Benchmarking detectors on different image sizes...")
    print("-" * 60)
    print(f"{'Method':<8} {'Size':<12} {'Features':<10} {'Time(s)':<10} {'FPS':<10}")
    print("-" * 60)
    
    for method in methods_to_benchmark:
        try:
            
            detector = create_traditional_detector(method, max_features=1000)
            
            for i, img in enumerate(test_images):
                size_str = f"{img.shape[1]}x{img.shape[0]}"
                
                # Benchmark this detector on this image
                benchmark_result = benchmark_detector_performance(detector, [img], num_runs=3)
                
                mean_time = benchmark_result['mean_time']
                mean_features = benchmark_result['mean_features']
                fps = 1.0 / mean_time if mean_time > 0 else 0
                
                print(f"{method:<8} {size_str:<12} {mean_features:<10.0f} {mean_time:<10.3f} {fps:<10.1f}")
                
        except Exception as e:
            print(f"Benchmarking {method} failed: {e}")
    
    print("-" * 60)


def example_10_advanced_filtering_and_analysis():
    """Example 10: Advanced filtering and geometric verification"""
    print("\nExample 10: Advanced Filtering and Analysis")
    print("=" * 40)
    
    # Create images with known geometric transformation
    img1 = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Add a pattern
    for i in range(5):
        for j in range(6):
            x, y = 50 + j*50, 50 + i*40
            cv2.circle(img1, (x, y), 8, (255, 255, 255), -1)
    
    # Create transformed version
    # Apply a simple perspective transformation
    src_points = np.float32([[0, 0], [400, 0], [400, 300], [0, 300]])
    dst_points = np.float32([[20, 10], [380, 15], [390, 290], [10, 295]])
    
    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img2 = cv2.warpPerspective(img1, transform_matrix, (400, 300))
    
    try:
        # Detect and match features
        result = fds.match_images(img1, img2, methods='SIFT', visualize=False)
        
        print(f"Initial matches: {len(result['match_data'].matches)}")
        print(f"Filtered matches: {len(result['match_data'].get_best_matches())}")
        
        # Apply additional geometric verification
        features1 = result['features1']
        features2 = result['features2']
        matches = result['match_data'].get_best_matches()
        
        if len(matches) >= 8:
            # Verify with fundamental matrix
            fund_matches, F = verify_matches_with_fundamental_matrix(
                features1.keypoints, features2.keypoints, matches
            )
            print(f"Fundamental matrix verification: {len(fund_matches)} matches")
            
            # Calculate reprojection errors if homography is available
            if result['match_data'].homography is not None:
                errors = calculate_reprojection_error(
                    features1.keypoints, features2.keypoints, 
                    matches, result['match_data'].homography
                )
                
                print(f"Reprojection errors - Mean: {np.mean(errors):.2f}, "
                      f"Std: {np.std(errors):.2f}")
                print(f"Errors < 5 pixels: {np.sum(errors < 5)}/{len(errors)}")
        
        # Remove duplicates
        unique_matches = remove_duplicate_matches(matches, distance_threshold=2.0)
        print(f"After duplicate removal: {len(unique_matches)} matches")
        
        # Detailed quality analysis
        analyzer = fds.MatchQualityAnalyzer()
        analysis = analyzer.analyze_match_data(result['match_data'])
        
        print(f"\nDetailed Quality Analysis:")
        print(f"  Method: {analysis['method']}")
        print(f"  Score type: {analysis['score_type']}")
        print(f"  Quality score: {analysis['quality_score']:.3f}")
        print(f"  Mean score: {analysis['mean_score']:.3f}")
        print(f"  Score std: {analysis['std_score']:.3f}")
        
        if 'percentiles' in analysis:
            print(f"  Score percentiles:")
            for percentile, value in analysis['percentiles'].items():
                print(f"    {percentile}: {value:.3f}")
        
    except Exception as e:
        print(f"Advanced filtering failed: {e}")


# =============================================================================
# Main Example Runner
# =============================================================================

def run_all_examples():
    """Run all examples"""
    print("Multi-Method Feature Detection System - Examples")
    print("=" * 60)
    
    # Check system capabilities
    print("System Capabilities:")
    fds.print_capabilities()
    print()
    
    examples = [
        example_01_simple_feature_detection,
        example_02_simple_matching,
        example_03_multi_method_comparison,
        example_04_pipeline_with_presets,
        example_05_custom_configuration,
        example_06_comprehensive_analysis,
        example_07_deep_learning_methods,
        example_08_dataset_processing,
        example_09_benchmarking_and_performance,
        example_10_advanced_filtering_and_analysis
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            print(f"\n{'='*60}")
            example_func()
            print(f"{'='*60}")
            
            # Optional: pause between examples
            # input("Press Enter to continue to next example...")
            
        except Exception as e:
            print(f"Example {i} failed: {e}")
            import traceback
            traceback.print_exc()


def run_quick_demo():
    """Run a quick demonstration"""
    print("Quick Demo - Feature Detection and Matching")
    print("=" * 50)
    
    try:
        # Create simple test images
        img1 = np.zeros((200, 300, 3), dtype=np.uint8)
        img2 = np.zeros((200, 300, 3), dtype=np.uint8)
        
        # Add features
        cv2.rectangle(img1, (50, 50), (120, 120), (255, 255, 255), -1)
        cv2.rectangle(img2, (55, 55), (125, 125), (255, 255, 255), -1)
        
        cv2.circle(img1, (200, 100), 30, (128, 128, 128), -1)
        cv2.circle(img2, (195, 105), 30, (128, 128, 128), -1)
        
        # Quick matching
        result = fds.match_images(img1, img2, methods='SIFT', visualize=True)
        
        print(f"Quick demo result:")
        print(f"  Method: {result['method_used']}")
        print(f"  Matches: {len(result['match_data'].get_best_matches())}")
        print(f"  Score type: {result['match_data'].score_type.value}")
        
        # Quality analysis
        analyzer = fds.MatchQualityAnalyzer()
        analysis = analyzer.analyze_match_data(result['match_data'])
        print(f"  Quality score: {analysis['quality_score']:.3f}")
        
        print("\nQuick demo completed successfully!")
        
    except Exception as e:
        print(f"Quick demo failed: {e}")


if __name__ == "__main__":
    # You can run individual examples or all of them
    
    # Option 1: Run quick demo
    #run_quick_demo()
    
    # Option 2: Run specific example
    # example_03_multi_method_comparison()
    
    # Option 3: Run all examples (uncomment to run)
    run_all_examples()
    
    print("\nFor more examples, check the individual example functions in this file.")