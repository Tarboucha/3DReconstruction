#!/usr/bin/env python3
"""
Complete Working Pipeline: Feature Matching → Pose Estimation

This script properly handles the FeatureMatchingExtraction output
and converts it to pose estimation format.
"""

import pickle
from pathlib import Path
from typing import List
import sys

# Feature matching
from FeatureMatchingExtraction import create_pipeline, MatchingResult

# Pose estimation  
from CameraPoseEstimation2.pipeline import IncrementalReconstructionPipeline
from CameraPoseEstimation2.data import create_provider


def save_for_pose_estimation(
    results: List[MatchingResult],
    output_dir: str,
    method: str = 'best',
    min_matches: int = 15,
    min_quality: float = 0.3
):
    """
    Convert MatchingResult list to pose estimation format.
    
    Args:
        results: List of MatchingResult from match_folder()
        output_dir: Where to save converted data
        method: Which method to extract ('best' or method name)
        min_matches: Minimum matches threshold
        min_quality: Minimum quality threshold
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pairs_dir = output_path / 'pairs'
    pairs_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Converting {len(results)} results for pose estimation")
    print(f"{'='*70}")
    
    # Collect image metadata
    image_info = {}
    saved_pairs = 0
    
    for result in results:
        # Get image info
        img1 = result.image_pair_info.image1_id
        img2 = result.image_pair_info.image2_id
        
        # Store metadata
        if img1 not in image_info:
            image_info[img1] = {
                'name': img1,
                'identifier': img1,
                'width': result.image_pair_info.image1_size[0] if result.image_pair_info.image1_size else 0,
                'height': result.image_pair_info.image1_size[1] if result.image_pair_info.image1_size else 0,
                'channels': 3,
                'size': result.image_pair_info.image1_size if result.image_pair_info.image1_size else (0, 0)
            }
        
        if img2 not in image_info:
            image_info[img2] = {
                'name': img2,
                'identifier': img2,
                'width': result.image_pair_info.image2_size[0] if result.image_pair_info.image2_size else 0,
                'height': result.image_pair_info.image2_size[1] if result.image_pair_info.image2_size else 0,
                'channels': 3,
                'size': result.image_pair_info.image2_size if result.image_pair_info.image2_size else (0, 0)
            }
        
        # Select method
        if method == 'best':
            method_result = result.get_best('quality')
        else:
            if method not in result.methods:
                print(f"  Warning: {method} not found in {img1}-{img2}, skipping")
                continue
            method_result = result.methods[method]
        
        if method_result is None:
            continue
        
        # Get match data
        match_data = method_result.match_data
        num_matches = len(match_data.get_best_matches())
        
        # Apply filters
        quality = method_result.inlier_ratio if method_result.inlier_ratio else 0.5
        if num_matches < min_matches or quality < min_quality:
            continue
        
        # Extract keypoints as arrays
        import cv2
        import numpy as np
        
        def convert_keypoints(kp):
            """Convert keypoints to numpy array"""
            if kp is None:
                return np.empty((0, 2), dtype=np.float32)
            
            # If it's already a numpy array
            if isinstance(kp, np.ndarray):
                if len(kp) == 0:
                    return np.empty((0, 2), dtype=np.float32)
                if kp.ndim == 2 and kp.shape[1] >= 2:
                    return kp[:, :2].astype(np.float32)
                return kp.astype(np.float32)
            
            # If it's a list or tuple
            if isinstance(kp, (list, tuple)):
                if len(kp) == 0:
                    return np.empty((0, 2), dtype=np.float32)
                
                # Check first element
                first = kp[0]
                if isinstance(first, cv2.KeyPoint):
                    # List of KeyPoint objects
                    return np.array([k.pt for k in kp], dtype=np.float32)
                elif isinstance(first, (list, tuple, np.ndarray)):
                    # List of coordinate pairs
                    return np.array(kp, dtype=np.float32)
                elif hasattr(first, 'pt'):
                    # Has .pt attribute (KeyPoint-like)
                    return np.array([k.pt for k in kp], dtype=np.float32)
                else:
                    # Unknown - try direct conversion
                    try:
                        return np.array([[k[0], k[1]] for k in kp], dtype=np.float32)
                    except:
                        print(f"Warning: Unknown keypoint type: {type(first)}")
                        return np.empty((0, 2), dtype=np.float32)
            
            # Unknown type
            print(f"Warning: Cannot convert keypoints of type: {type(kp)}")
            return np.empty((0, 2), dtype=np.float32)
        
        kp1 = convert_keypoints(method_result.features1.keypoints)
        kp2 = convert_keypoints(method_result.features2.keypoints)
        
        # Extract matches
        matches = []
        for m in match_data.get_best_matches():
            if hasattr(m, 'queryIdx'):
                matches.append((m.queryIdx, m.trainIdx))
            elif isinstance(m, (tuple, list)) and len(m) >= 2:
                matches.append((int(m[0]), int(m[1])))
        
        # Create pair data
        pair_data = {
            'image1': img1,
            'image2': img2,
            'keypoints1': kp1,
            'keypoints2': kp2,
            'descriptors1': method_result.features1.descriptors,
            'descriptors2': method_result.features2.descriptors,
            'matches': matches,
            'num_matches': len(matches),
            'quality_score': float(quality),
            'method': method_result.method_name,
            'matching_time': method_result.matching_time,
            'homography': method_result.homography,
            'fundamental_matrix': method_result.fundamental_matrix
        }
        
        # Save pair file
        pair_filename = f"({img1}_{img2}).pkl"
        with open(pairs_dir / pair_filename, 'wb') as f:
            pickle.dump(pair_data, f)
        
        saved_pairs += 1
    
    # Save metadata
    metadata = {
        'total_images': len(image_info),
        'images': list(image_info.values())
    }
    
    with open(output_path / 'image_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✓ Saved {saved_pairs} pairs")
    print(f"✓ Saved metadata for {len(image_info)} images")
    print(f"✓ Output: {output_dir}")


def run_complete_pipeline(
    images_dir: str,
    output_dir: str,
    preset: str = 'balanced',
    min_matches: int = 15,
    min_quality: float = 0.3
):
    """
    Complete pipeline from images to 3D reconstruction.
    
    Args:
        images_dir: Directory with images
        output_dir: Output directory
        preset: Feature pipeline preset
        min_matches: Minimum matches for export
        min_quality: Minimum quality for export
    """
    output_path = Path(output_dir)
    
    print("="*70)
    print("STEP 1: Feature Matching")
    print("="*70)
    
    # Create pipeline
    pipeline = create_pipeline(preset)
    
    # Match images
    results = pipeline.match_folder(
        folder_path=images_dir,
        pattern='*.jpg',
        pair_mode='all',
        output_dir='./test_v5',
        auto_save='True',
        resize_to=None,
        resume=False,
        filter_matches=False
    )
    
    print(f"\n✓ Matched {len(results)} image pairs")
    
    # Convert for pose estimation
    print("\n" + "="*70)
    print("STEP 2: Converting for Pose Estimation")
    print("="*70)
    
    pose_input_dir = output_path / 'pose_input'
    save_for_pose_estimation(
        results=results,
        output_dir=str(pose_input_dir),
        method='best',
        min_matches=0,
        min_quality=0
    )
    
    # Run pose estimation
    print("\n" + "="*70)
    print("STEP 3: Camera Pose Estimation")
    print("="*70)
    
    provider = create_provider(str(pose_input_dir))
    
    # Validate
    validation = provider.validate()
    if not validation.is_valid:
        print("❌ Validation failed:")
        for error in validation.errors:
            print(f"  - {error}")
        return None
    
    print("✓ Provider validated")
    provider.print_summary()
    
    # Create and run pipeline
    reconstruction_dir = output_path / 'reconstruction'
    pose_pipeline = IncrementalReconstructionPipeline(
        provider=provider,
        output_dir=str(reconstruction_dir)
    )
    
    reconstruction = pose_pipeline.run()
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Images: {provider.get_image_count()}")
    print(f"Cameras: {len(reconstruction.cameras)}")
    print(f"3D Points: {len(reconstruction.points)}")
    print(f"Observations: {len(reconstruction.observations)}")
    print(f"\nOutput: {output_dir}")
    print("="*70)
    
    return reconstruction


def main():
    """Main entry point"""
    
    images_dir = './images/Eiffel Tower copy'
    output_dir = './test_v4'
    preset =  'balanced'
    
    try:
        reconstruction = run_complete_pipeline(
            images_dir=images_dir,
            output_dir=output_dir,
            preset=preset,
            min_matches=15,
            min_quality=0.3
        )
        
        if reconstruction:
            print("\n✓ Pipeline completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Pipeline failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()