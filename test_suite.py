import numpy as np
import os
from scipy.spatial.transform import Rotation
from CameraPoseEstimation import CameraPoseEstimator
from CameraPoseEstimation import TwoViewReconstruction
from CameraPoseEstimation import complete_photogrammetry_pipeline

import sys
import os


def test_synthetic_reconstruction():
    """Test camera pose estimation with synthetic data"""
    np.random.seed(42)
    
    # Generate synthetic correspondences
    num_points = 100
    points_3d_true = np.random.randn(3, num_points) * 3
    points_3d_true[2] += 8  # Move in front of camera
    
    # Camera parameters
    width, height = 1920, 1080
    focal_length = width * 1.2
    camera_matrix_true = np.array([
        [focal_length, 0, width/2],
        [0, focal_length, height/2],
        [0, 0, 1]
    ])
    
    # Camera poses
    R_true = Rotation.from_euler('xyz', [10, 15, 5], degrees=True).as_matrix()
    t_true = np.array([[1.0], [0.2], [0.3]])
    
    # Project to images
    P1 = camera_matrix_true @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = camera_matrix_true @ np.hstack([R_true, t_true])
    
    points_3d_hom = np.vstack([points_3d_true, np.ones(num_points)])
    pts1_hom = P1 @ points_3d_hom
    pts2_hom = P2 @ points_3d_hom
    
    pts1 = (pts1_hom[:2] / pts1_hom[2]).T
    pts2 = (pts2_hom[:2] / pts2_hom[2]).T
    
    # Add realistic noise
    pts1 += np.random.normal(0, 0.5, pts1.shape)
    pts2 += np.random.normal(0, 0.5, pts2.shape)
    
    # Test reconstruction
    two_view = TwoViewReconstruction()
    results = two_view.process_two_view_reconstruction(
        pts1, pts2, (width, height),
        camera_matrix=None,
        perform_bundle_adjustment=True,
        visualize=True
    )
    
    if results.get('success'):
        # Validate quality
        from CameraPoseEstimation import QualityValidator
        validation = QualityValidator.validate_reconstruction_quality(results)
        print(f"Synthetic test - Quality score: {validation['quality_score']:.2f}")
        print(f"Valid reconstruction: {validation['valid']}")
        
        # Compare with ground truth
        R_est = results['rotation']
        t_est = results['translation']
        
        rotation_error = np.degrees(np.arccos((np.trace(R_est.T @ R_true) - 1) / 2))
        translation_error = np.linalg.norm(t_est / np.linalg.norm(t_est) - 
                                         t_true / np.linalg.norm(t_true))
        
        print(f"Rotation error: {rotation_error:.2f}°")
        print(f"Translation direction error: {translation_error:.3f}")
    else:
        print("Synthetic test failed:", results.get('error'))

def test_real_image_pipeline():
    """Test with real images if available"""
    
    # Example image paths - replace with your monument photos
    image_paths = [
        'E:\images\Statue of Liberty\Image_1.jpg',
        'E:\images\Statue of Liberty\Image_5.jpg',
        'E:\images\Statue of Liberty\Image_8.jpg',
        'E:\images\Statue of Liberty\Image_10.jpg'
    ]
    
    # Check if images exist
    available_images = [path for path in image_paths if os.path.exists(path)]
    
    if len(available_images) < 2:
        print("Real image test skipped - need at least 2 monument photos")
        print("Place your monument photos in the current directory with names:")
        for path in image_paths:
            print(f"  - {path}")
        return
    
    print(f"Testing with {len(available_images)} real images")
    
    # Test different feature types
    feature_types = ['LightGlue']
    
    for feature_type in feature_types:
        print(f"\nTesting with {feature_type} features...")
        
        try:
            results = complete_photogrammetry_pipeline(
                available_images,  
                feature_type=feature_type,
                use_deep_learning=False
            )
            
            if results.get('success'):
                stats = results['pipeline_stats']
                print(f"Pipeline successful with {feature_type}!")
                print(f"  Final matches: {stats['final_matches']}")
                print(f"  3D points: {stats['triangulated_points']}")
                print(f"  Quality score: {stats['quality_score']:.2f}")
            else:
                print(f"Pipeline failed with {feature_type}: {results.get('error')}")
                
        except Exception as e:
            print(f"Error testing {feature_type}: {e}")

def main():
    """Example usage of complete pipeline"""

    print("Testing complete photogrammetry pipeline")
    
    # # Test with synthetic data first
    # print("\n1. Testing with synthetic data...")
    # test_synthetic_reconstruction()
    
    # Test with real images (if available)
    print("\n2. Testing with real images...")
    test_real_image_pipeline()

if __name__ == "__main__":
    main()