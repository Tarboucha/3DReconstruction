import pickle
import json
import base64
import zlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import hashlib
from FeatureMatchingExtraction import ScoreType

def export_for_pose_estimation(
    input_dir: str,
    output_dir: str,
    method: str = 'best',
    min_matches: int = 10,
    min_quality: float = 0.0
):
    """
    Export FeatureMatchingExtraction results for pose estimation.
    
    Converts from NEW batch format to folder format:
        input_dir/  (FeatureMatchingExtraction output)
        ├── image_metadata.pkl
        └── batches/
            ├── batch_001/
            │   ├── batch_metadata.json
            │   └── reconstruction_data.pkl
            └── batch_002/
                ├── batch_metadata.json
                └── reconstruction_data.pkl
        
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
    
    # 2. Load and convert batch directories
    print("\nStep 2: Converting match data...")
    
    # Look for batches directory
    batches_dir = input_path / 'batches'
    
    if not batches_dir.exists():
        raise FileNotFoundError(
            f"No batches/ directory found in {input_dir}\n"
            f"Expected: {input_dir}/batches/"
        )
    
    # Find all batch directories
    batch_dirs = sorted(batches_dir.glob('batch_*'))
    
    if not batch_dirs:
        print(f"\n  Searching in: {batches_dir}")
        print(f"  Found:")
        for item in batches_dir.iterdir():
            print(f"    - {item.name}")
        
        raise FileNotFoundError(
            f"No batch_XXX directories found in {batches_dir}\n"
            f"Expected: batch_001/, batch_002/, etc."
        )
    
    print(f"Found {len(batch_dirs)} batch directories")
    
    stats = {
        'total_pairs': 0,
        'exported_pairs': 0,
        'filtered_low_matches': 0,
        'filtered_low_quality': 0,
        'failed_pairs': 0
    }
    
    for batch_dir in batch_dirs:
        print(f"  Processing {batch_dir.name}...")
        _convert_batch_directory(
            batch_dir, 
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


def _generate_pair_filename(img1_id: str, img2_id: str) -> str:
    """Compress and encode pair names"""
    pair_str = f"{img1_id}|{img2_id}"
    
    # Compress using zlib
    compressed = zlib.compress(pair_str.encode('utf-8'), level=9)
    
    # Encode to base64 (URL-safe)
    encoded = base64.urlsafe_b64encode(compressed).decode('ascii').rstrip('=')
    
    return f"pair_{encoded}.pkl"


def _decode_pair_filename(filename: str) -> Tuple[str, str]:
    """Decompress and decode back to pair names"""
    encoded = filename.replace('pair_', '').replace('.pkl', '')
    
    # Add padding
    padding = 4 - (len(encoded) % 4)
    if padding != 4:
        encoded += '=' * padding
    
    # Decode from base64
    compressed = base64.urlsafe_b64decode(encoded.encode('ascii'))
    
    # Decompress
    pair_str = zlib.decompress(compressed).decode('utf-8')
    
    # Split
    img1, img2 = pair_str.split('|')
    return (img1, img2)  

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


def _convert_batch_directory(
    batch_dir: Path,
    pairs_dir: Path,
    method: str,
    min_matches: int,
    min_quality: float,
    stats: Dict
):
    """
    Convert one batch directory to individual pair files.
    
    Args:
        batch_dir: Path to batch_XXX/ directory
        pairs_dir: Output pairs/ directory
        method: Which method to extract
        min_matches: Minimum matches filter
        min_quality: Minimum quality filter
        stats: Statistics dictionary to update
    """
    # Load reconstruction data
    recon_file = batch_dir / 'reconstruction_data.pkl'
    
    if not recon_file.exists():
        print(f"    ⚠️  No reconstruction_data.pkl in {batch_dir.name}")
        return
    
    try:
        with open(recon_file, 'rb') as f:
            batch_data = pickle.load(f)
    except Exception as e:
        print(f"    ⚠️  Failed to load {recon_file}: {e}")
        return
    
    if not isinstance(batch_data, list):
        print(f"    ⚠️  Unexpected batch data format in {batch_dir.name}")
        return
    
    for item in batch_data:
        stats['total_pairs'] += 1
        
        try:
            # Extract reconstruction data
            recon_data = item.get('reconstruction')
            if recon_data is None:
                stats['failed_pairs'] += 1
                continue
            
            # Get image IDs
            img1_id = item.get('image1_id')
            img2_id = item.get('image2_id')
            
            if not img1_id or not img2_id:
                # Fallback: try to get from reconstruction
                if hasattr(recon_data, 'image_pair_info'):
                    img1_id = recon_data.image_pair_info.image1_id
                    img2_id = recon_data.image_pair_info.image2_id
                else:
                    stats['failed_pairs'] += 1
                    continue
            
            # Extract pair data
            pair_data = _extract_reconstruction_data(recon_data, method)
            
            # Add image IDs
            pair_data['image1'] = img1_id
            pair_data['image2'] = img2_id
            
            # Apply filters
            # if pair_data['num_matches'] < min_matches:
            #     stats['filtered_low_matches'] += 1
            #     continue
            
            # if pair_data['quality_score'] < min_quality:
            #     stats['filtered_low_quality'] += 1
            #     continue
            
            # Save pair file
            pair_filename = _generate_pair_filename(img1_id, img2_id)
            pair_filepath = pairs_dir / pair_filename
            
            with open(pair_filepath, 'wb') as f:
                pickle.dump(pair_data, f)
            
            stats['exported_pairs'] += 1
            
        except Exception as e:
            print(f"    ⚠️  Failed to process pair: {e}")
            stats['failed_pairs'] += 1
            continue

def _extract_reconstruction_data(recon_data, method: str) -> Dict:
    """
    Extract essential data from MultiMethodReconstruction object.
    
    Args:
        recon_data: MultiMethodReconstruction object
        method: 'best', 'all', or specific method name
    
    Returns:
        Dict with format expected by FolderMatchDataProvider
    """
    if not hasattr(recon_data, 'methods'):
        raise ValueError("Reconstruction data missing 'methods' attribute")
    
    if method == 'all':
        return _extract_all_methods_combined(recon_data)
    elif method == 'best':
        best_method_name = None
        best_num_matches = 0
        
        for method_name, method_data in recon_data.methods.items():
            if method_data.num_matches > best_num_matches:
                best_num_matches = method_data.num_matches
                best_method_name = method_name
        
        if best_method_name is None:
            raise ValueError("No valid method found")
        
        return _extract_single_method(recon_data.methods[best_method_name], best_method_name, recon_data)
    else:
        if method not in recon_data.methods:
            available = list(recon_data.methods.keys())
            raise ValueError(f"Method {method} not found. Available: {available}")
        
        return _extract_single_method(recon_data.methods[method], method, recon_data)


def _extract_single_method(method_data, method_name: str, recon_data) -> Dict:
    """Extract data from a single MethodReconstructionData."""
    from CameraPoseEstimation2.core.structures import EnhancedDMatch, ScoreType as CPEScoreType
    
    # Extract data from MethodReconstructionData
    keypoints1 = method_data.keypoints1  # Tuple of cv2.KeyPoint
    keypoints2 = method_data.keypoints2
    descriptors1 = method_data.descriptors1
    descriptors2 = method_data.descriptors2
    query_indices = method_data.query_indices  # np.ndarray
    train_indices = method_data.train_indices  # np.ndarray
    match_scores = method_data.match_scores    # np.ndarray
    score_type = method_data.score_type        # ScoreType
    correspondences = method_data.correspondences  # np.ndarray (N, 4)
    
    # Convert keypoints to arrays
    kp1 = _keypoints_to_array(keypoints1)
    kp2 = _keypoints_to_array(keypoints2)
    
    # Build EnhancedDMatch list
    enhanced_matches = []
    for i in range(len(query_indices)):
        # Map ScoreType to CPEScoreType
        if score_type == ScoreType.DISTANCE:
            cpe_score_type = CPEScoreType.DISTANCE
        elif score_type == ScoreType.SIMILARITY:
            cpe_score_type = CPEScoreType.SIMILARITY
        else:
            cpe_score_type = CPEScoreType.CONFIDENCE
        
        enhanced = EnhancedDMatch(
            queryIdx=int(query_indices[i]),
            trainIdx=int(train_indices[i]),
            score=float(match_scores[i]),
            score_type=cpe_score_type,
            confidence=method_data.inlier_ratio if method_data.inlier_ratio else 0.5,
            standardized_quality=method_data.get_quality_score(),
            source_method=method_name
        )
        enhanced_matches.append(enhanced)
    
    # Simple matches list for compatibility
    matches_list = [(int(q), int(t)) for q, t in zip(query_indices, train_indices)]
    
    # Get image sizes
    image1_size = None
    image2_size = None
    if hasattr(recon_data, 'image_pair_info'):
        image1_size = recon_data.image_pair_info.image1_size
        image2_size = recon_data.image_pair_info.image2_size
    
    return {
        'keypoints1': kp1,
        'keypoints2': kp2,
        'descriptors1': descriptors1,
        'descriptors2': descriptors2,
        'matches': enhanced_matches,
        'matches_list': matches_list,
        'correspondences': correspondences if correspondences is not None else np.empty((0, 4), dtype=np.float32),
        'num_matches': method_data.num_matches,
        'method': method_name,
        'quality_score': float(method_data.get_quality_score()),
        'image1_size': image1_size,
        'image2_size': image2_size,
        'homography': method_data.homography,
        'fundamental_matrix': method_data.fundamental_matrix,
    }


def _extract_all_methods_combined(recon_data) -> Dict:
    """Extract and combine all methods into single match list with EnhancedDMatch."""
    from CameraPoseEstimation2.core.structures import EnhancedDMatch, ScoreType as CPEScoreType
    
    image1_size = None
    image2_size = None
    if hasattr(recon_data, 'image_pair_info'):
        image1_size = recon_data.image_pair_info.image1_size
        image2_size = recon_data.image_pair_info.image2_size
    
    first_method = next(iter(recon_data.methods.values()))
    all_keypoints1 = _keypoints_to_array(first_method.keypoints1)
    all_keypoints2 = _keypoints_to_array(first_method.keypoints2)
    all_descriptors1 = first_method.descriptors1
    all_descriptors2 = first_method.descriptors2
    
    max_idx1 = len(all_keypoints1)
    max_idx2 = len(all_keypoints2)
    
    all_enhanced_matches = []
    all_correspondences = []
    total_matches = 0
    weighted_quality_sum = 0.0
    
    for method_name, method_data in recon_data.methods.items():
        query_indices = method_data.query_indices
        train_indices = method_data.train_indices
        match_scores = method_data.match_scores
        score_type = method_data.score_type
        method_quality = method_data.get_quality_score()
        
        # Map ScoreType
        if score_type == ScoreType.DISTANCE:
            cpe_score_type = CPEScoreType.DISTANCE
        elif score_type == ScoreType.SIMILARITY:
            cpe_score_type = CPEScoreType.SIMILARITY
        else:
            cpe_score_type = CPEScoreType.CONFIDENCE
        
        for i in range(len(query_indices)):
            idx1 = int(query_indices[i])
            idx2 = int(train_indices[i])
            
            # Skip if indices are out of bounds
            if idx1 >= max_idx1 or idx2 >= max_idx2:
                continue
            
            enhanced = EnhancedDMatch(
                queryIdx=idx1,
                trainIdx=idx2,
                score=float(match_scores[i]),
                score_type=cpe_score_type,
                confidence=method_data.inlier_ratio if method_data.inlier_ratio else 0.5,
                standardized_quality=method_quality,
                source_method=method_name
            )
            all_enhanced_matches.append(enhanced)
        
        # Collect correspondences (also filter)
        if method_data.correspondences is not None:
            all_correspondences.append(method_data.correspondences)
        
        num_matches = len([i for i in range(len(query_indices)) 
                          if query_indices[i] < max_idx1 and train_indices[i] < max_idx2])
        weighted_quality_sum += method_quality * num_matches
        total_matches += num_matches
        
    # Combine correspondences
    if all_correspondences:
        combined_correspondences = np.vstack(all_correspondences)
    else:
        combined_correspondences = np.empty((0, 4), dtype=np.float32)
    
    # Simple matches list
    matches_list = [(m.queryIdx, m.trainIdx) for m in all_enhanced_matches]
    
    # Average quality
    quality_scores = [m.standardized_quality for m in all_enhanced_matches]
    avg_quality = float(np.mean(quality_scores)) if quality_scores else 0.5
    
    return {
        'keypoints1': all_keypoints1,
        'keypoints2': all_keypoints2,
        'descriptors1': all_descriptors1,
        'descriptors2': all_descriptors2,
        'matches': all_enhanced_matches,
        'matches_list': matches_list,
        'correspondences': combined_correspondences,
        'num_matches': len(all_enhanced_matches),
        'method': 'multi_method',
        'methods_used': list(recon_data.methods.keys()),
        'quality_score': avg_quality,
        'image1_size': image1_size,
        'image2_size': image2_size,
    }



def _create_correspondences_array(kp1: np.ndarray, kp2: np.ndarray, 
                                  matches: List[Tuple[int, int]]) -> np.ndarray:
    """
    Create correspondences array from keypoints and matches.
    
    Args:
        kp1: Keypoints from image 1 (N, 2)
        kp2: Keypoints from image 2 (M, 2)
        matches: List of (idx1, idx2) tuples
    
    Returns:
        Correspondences array of shape (K, 4) with [x1, y1, x2, y2]
    """
    if len(matches) == 0:
        return np.empty((0, 4), dtype=np.float32)
    
    correspondences = []
    for idx1, idx2 in matches:
        if idx1 < len(kp1) and idx2 < len(kp2):
            pt1 = kp1[idx1]
            pt2 = kp2[idx2]
            correspondences.append([pt1[0], pt1[1], pt2[0], pt2[1]])
    
    return np.array(correspondences, dtype=np.float32)


def _keypoints_to_array(keypoints) -> np.ndarray:
    """Convert keypoints to numpy array"""
    if keypoints is None:
        return np.empty((0, 2), dtype=np.float32)
    
    # Handle numpy arrays
    if isinstance(keypoints, np.ndarray):
        if keypoints.ndim == 2 and keypoints.shape[1] >= 2:
            return keypoints[:, :2].astype(np.float32)
        return keypoints.astype(np.float32)
    
    # Handle lists and other iterables
    if isinstance(keypoints, (list, tuple)):
        if len(keypoints) == 0:
            return np.empty((0, 2), dtype=np.float32)
        
        # Check first element to determine type
        first_elem = keypoints[0]
        
        if isinstance(first_elem, cv2.KeyPoint):
            # Extract .pt from cv2.KeyPoint objects
            return np.array([kp.pt for kp in keypoints], dtype=np.float32)
        elif isinstance(first_elem, (tuple, list)):
            # Already in tuple/list format
            return np.array(keypoints, dtype=np.float32)
        elif isinstance(first_elem, np.ndarray):
            # List of numpy arrays
            return np.array([kp[:2] if len(kp) >= 2 else kp for kp in keypoints], dtype=np.float32)
        else:
            # Try to convert directly
            return np.array(keypoints, dtype=np.float32)
    
    # Try to iterate and check for KeyPoint objects
    try:
        keypoints_list = list(keypoints)
        if len(keypoints_list) == 0:
            return np.empty((0, 2), dtype=np.float32)
        
        if isinstance(keypoints_list[0], cv2.KeyPoint):
            return np.array([kp.pt for kp in keypoints_list], dtype=np.float32)
        else:
            return np.array(keypoints_list, dtype=np.float32)
    except:
        pass
    
    # Final fallback - shouldn't reach here
    raise TypeError(f"Cannot convert keypoints of type {type(keypoints)} to numpy array")

