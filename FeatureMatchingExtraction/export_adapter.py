"""
Export Adapter for FeatureMatchingExtraction

Converts FeatureMatchingExtraction output to format compatible with
FolderMatchDataProvider (CameraPoseEstimation2).

This adapter extracts the essential data from MatchingResult objects
and saves in the simple folder structure expected by pose estimation.

File: FeatureMatchingExtraction/export_adapter.py
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2


def export_for_pose_estimation(
    input_dir: str,
    output_dir: str,
    method: str = 'best',
    min_matches: int = 10,
    min_quality: float = 0.0
):
    """
    Export FeatureMatchingExtraction results for pose estimation.
    
    Converts from batch format to folder format:
        input_dir/  (FeatureMatchingExtraction output)
        ├── image_metadata.pkl
        ├── batch_001.pkl
        └── batch_002.pkl
        
    To:
        output_dir/  (Pose estimation format)
        ├── image_metadata.pkl
        └── pairs/
            ├── (img1_img2).pkl
            └── ...
    
    Args:
        input_dir: FeatureMatchingExtraction output directory
        output_dir: Where to save pose estimation format
        method: Which method to extract ('best', 'SIFT', 'ORB', etc.)
        min_matches: Minimum matches to export
        min_quality: Minimum quality to export
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    print(f"="*70)
    print(f"Exporting for Pose Estimation")
    print(f"="*70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Method: {method}")
    print(f"Filters: min_matches={min_matches}, min_quality={min_quality}")
    print()
    
    # Create output structure
    output_path.mkdir(parents=True, exist_ok=True)
    pairs_dir = output_path / 'pairs'
    pairs_dir.mkdir(exist_ok=True)
    
    # 1. Copy/convert image metadata
    print("Step 1: Converting image metadata...")
    _convert_metadata(input_path, output_path)
    
    # 2. Load and convert batch files
    print("\nStep 2: Converting match data...")
    
    # Look for batch files in multiple locations
    batch_files = sorted(input_path.glob('*_batch_*.pkl'))
    
    # If not found in root, check subdirectories
    if not batch_files:
        # Check matching_results subdirectory
        matching_results_dir = input_path / 'matching_results'
        if matching_results_dir.exists():
            batch_files = sorted(matching_results_dir.glob('*_batch_*.pkl'))
            print(f"  Found batch files in matching_results/ subdirectory")
        
        # Check reconstruction subdirectory as backup
        if not batch_files:
            recon_dir = input_path / 'reconstruction'
            if recon_dir.exists():
                batch_files = sorted(recon_dir.glob('*_batch_*.pkl'))
                print(f"  Found batch files in reconstruction/ subdirectory")
    
    if not batch_files:
        # List what's actually in the directory
        print(f"\n  Searching in: {input_path}")
        print(f"  Found files:")
        for f in input_path.glob('*'):
            print(f"    - {f.name}")
        if (input_path / 'matching_results').exists():
            print(f"  Found files in matching_results/:")
            for f in (input_path / 'matching_results').glob('*'):
                print(f"    - {f.name}")
        
        raise FileNotFoundError(
            f"No batch files found in {input_dir}\n"
            f"Expected: *_batch_*.pkl files in root or matching_results/ subdirectory"
        )
    
    print(f"Found {len(batch_files)} batch files")
    
    stats = {
        'total_pairs': 0,
        'exported_pairs': 0,
        'filtered_low_matches': 0,
        'filtered_low_quality': 0,
        'failed_pairs': 0
    }
    
    for batch_file in batch_files:
        print(f"  Processing {batch_file.name}...")
        _convert_batch(
            batch_file, 
            pairs_dir, 
            method, 
            min_matches, 
            min_quality,
            stats
        )
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Export Summary")
    print(f"{'='*70}")
    print(f"Total pairs processed: {stats['total_pairs']}")
    print(f"Exported pairs: {stats['exported_pairs']}")
    print(f"Filtered (low matches): {stats['filtered_low_matches']}")
    print(f"Filtered (low quality): {stats['filtered_low_quality']}")
    print(f"Failed pairs: {stats['failed_pairs']}")
    print(f"\n✓ Export complete!")
    print(f"Output ready at: {output_dir}")


def _convert_metadata(input_path: Path, output_path: Path):
    """Convert image metadata to pose estimation format"""
    # Find metadata file
    metadata_files = list(input_path.glob('*_image_metadata.pkl')) + \
                    list(input_path.glob('image_metadata.pkl'))
    
    if not metadata_files:
        raise FileNotFoundError(f"No image metadata file in {input_path}")
    
    with open(metadata_files[0], 'rb') as f:
        metadata = pickle.load(f)
    
    # Convert format
    if 'images' in metadata:
        # Already in correct format
        output_metadata = metadata
        # Ensure total_images exists
        if 'total_images' not in output_metadata:
            output_metadata['total_images'] = len(output_metadata['images'])
    else:
        # Need to convert from dict format
        images = []
        for img_name, img_data in metadata.items():
            if isinstance(img_data, dict):
                images.append({
                    'name': img_name,
                    'identifier': img_name,
                    'width': img_data.get('width', 0),
                    'height': img_data.get('height', 0),
                    'channels': img_data.get('channels', 3),
                    'size': img_data.get('size', (0, 0)),
                    'filepath': img_data.get('filepath', '')
                })
        
        output_metadata = {
            'total_images': len(images),
            'images': images
        }
    
    # Save
    output_file = output_path / 'image_metadata.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(output_metadata, f)
    
    total = output_metadata.get('total_images', len(output_metadata.get('images', [])))
    print(f"  ✓ Converted metadata for {total} images")


def _convert_batch(
    batch_file: Path,
    pairs_dir: Path,
    method: str,
    min_matches: int,
    min_quality: float,
    stats: Dict
):
    """Convert one batch file to individual pair files"""
    with open(batch_file, 'rb') as f:
        batch_data = pickle.load(f)
    
    # Extract results
    results = batch_data.get('results', {})
    
    for pair_key, pair_result in results.items():
        stats['total_pairs'] += 1
        
        # Handle string keys
        if isinstance(pair_key, str):
            try:
                pair_key = eval(pair_key)
            except:
                continue
        
        if not isinstance(pair_key, tuple) or len(pair_key) != 2:
            continue
        
        # Check for errors
        if 'error' in pair_result:
            stats['failed_pairs'] += 1
            continue
        
        # Convert MatchingResult to simple format
        try:
            pair_data = _extract_pair_data(pair_result, method)
        except Exception as e:
            print(f"    Warning: Failed to extract {pair_key}: {e}")
            stats['failed_pairs'] += 1
            continue
        
        # Apply filters
        if pair_data['num_matches'] < min_matches:
            stats['filtered_low_matches'] += 1
            continue
        
        if pair_data['quality_score'] < min_quality:
            stats['filtered_low_quality'] += 1
            continue
        
        # Save pair file
        img1, img2 = pair_key
        pair_filename = f"({img1}_{img2}).pkl"
        pair_filepath = pairs_dir / pair_filename
        
        with open(pair_filepath, 'wb') as f:
            pickle.dump(pair_data, f)
        
        stats['exported_pairs'] += 1


def _extract_pair_data(pair_result, method: str) -> Dict:
    """
    Extract essential data from MatchingResult.
    
    Args:
        pair_result: MatchingResult object or dict
        method: Which method to extract ('best' or method name)
    
    Returns:
        Dict with format expected by FolderMatchDataProvider
    """
    # Handle both MatchingResult objects and dicts
    if hasattr(pair_result, 'methods'):
        # It's a MatchingResult object
        matching_result = pair_result
        
        # Select method
        if method == 'best':
            # Use get_best_method() if available
            if hasattr(matching_result, 'get_best_method'):
                method_result = matching_result.get_best_method()
            else:
                # Fallback: pick first available
                method_result = next(iter(matching_result.methods.values()))
        else:
            # Specific method requested
            if method not in matching_result.methods:
                available = list(matching_result.methods.keys())
                raise ValueError(f"Method {method} not found. Available: {available}")
            method_result = matching_result.methods[method]
        
        # Extract data
        features1 = method_result.features1
        features2 = method_result.features2
        match_data = method_result.match_data
        
        # Get keypoints
        kp1 = _keypoints_to_array(features1.keypoints)
        kp2 = _keypoints_to_array(features2.keypoints)
        
        # Get matches
        matches = _matches_to_list(match_data.get_best_matches())
        
        # Quality score
        quality_score = getattr(match_data, 'standardized_pair_quality', 0.5)
        if quality_score is None:
            quality_score = method_result.inlier_ratio if method_result.inlier_ratio else 0.5
        
        pair_data = {
            'image1': matching_result.pair_info.image1_id,
            'image2': matching_result.pair_info.image2_id,
            'keypoints1': kp1,
            'keypoints2': kp2,
            'descriptors1': features1.descriptors,
            'descriptors2': features2.descriptors,
            'matches': matches,
            'num_matches': len(matches),
            'quality_score': float(quality_score),
            'method': method_result.method_name,
            'matching_time': method_result.matching_time,
            'homography': method_result.homography,
            'fundamental_matrix': method_result.fundamental_matrix,
            'inlier_mask': None  # Could extract if needed
        }
        
    else:
        # It's already a dict (legacy format)
        pair_data = pair_result
    
    return pair_data


def _keypoints_to_array(keypoints) -> np.ndarray:
    """Convert keypoints to numpy array"""
    if keypoints is None:
        return np.empty((0, 2), dtype=np.float32)
    
    if isinstance(keypoints, np.ndarray):
        if keypoints.ndim == 2 and keypoints.shape[1] >= 2:
            return keypoints[:, :2].astype(np.float32)
        return keypoints.astype(np.float32)
    
    if isinstance(keypoints, list):
        if len(keypoints) == 0:
            return np.empty((0, 2), dtype=np.float32)
        
        if isinstance(keypoints[0], cv2.KeyPoint):
            return np.array([kp.pt for kp in keypoints], dtype=np.float32)
        elif isinstance(keypoints[0], (tuple, list)):
            return np.array(keypoints, dtype=np.float32)
    
    return np.array(keypoints, dtype=np.float32)


def _matches_to_list(matches) -> List[Tuple[int, int]]:
    """Convert matches to list of (idx1, idx2) tuples"""
    result = []
    
    for match in matches:
        if hasattr(match, 'queryIdx'):
            # cv2.DMatch or EnhancedDMatch
            result.append((match.queryIdx, match.trainIdx))
        elif isinstance(match, (tuple, list)) and len(match) >= 2:
            result.append((int(match[0]), int(match[1])))
    
    return result


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Export FeatureMatchingExtraction results for pose estimation'
    )
    parser.add_argument('input_dir', help='Input directory (FeatureMatchingExtraction output)')
    parser.add_argument('output_dir', help='Output directory (pose estimation format)')
    parser.add_argument('--method', default='best', help='Method to extract (default: best)')
    parser.add_argument('--min-matches', type=int, default=10, help='Minimum matches (default: 10)')
    parser.add_argument('--min-quality', type=float, default=0.0, help='Minimum quality (default: 0.0)')
    
    args = parser.parse_args()
    
    export_for_pose_estimation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        method=args.method,
        min_matches=args.min_matches,
        min_quality=args.min_quality
    )


if __name__ == '__main__':
    main()