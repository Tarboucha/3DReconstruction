import pickle
import glob
from pathlib import Path
from typing import Dict, Tuple
import numpy as np


def load_matches_for_pose_estimation(results_dir: str) -> Dict:
    """
    Load match data from FeatureMatchingExtraction batch files and prepare
    it for CameraPoseEstimation pipeline.
    
    Args:
        results_dir: Directory containing batch pickle files from FeatureMatchingExtraction
        
    Returns:
        Dictionary with structure:
        {
            'matches_data': Dict[Tuple[str, str], Dict],  # Pairwise matches
            'image_info': Dict[str, Dict]                 # Image metadata
        }
        
    Example:
        >>> matches_pickle = load_matches_for_pose_estimation('./results/')
        >>> pipeline = MainPosePipeline()
        >>> pipeline.process_monument_reconstruction(matches_pickle, './output/')
    """
    results_path = Path(results_dir)
    
    print(f"Loading match data from: {results_dir}")
    
    # 1. Load image metadata
    metadata_files = list(results_path.glob("*_image_metadata.pkl"))
    if not metadata_files:
        raise FileNotFoundError(f"No image metadata file found in {results_dir}")
    
    with open(metadata_files[0], 'rb') as f:
        image_metadata = pickle.load(f)
    
    print(f"✓ Loaded metadata for {image_metadata['total_images']} images")
    
    # 2. Convert image metadata to expected format
    image_info = {}
    for img_data in image_metadata['images']:
        img_name = img_data['name']
        image_info[img_name] = {
            'name': img_name,
            'size': tuple(img_data['size']),  # (height, width) or (height, width, channels)
            'width': img_data['width'],
            'height': img_data['height'],
            'channels': img_data.get('channels', 3)
        }
    
    # 3. Load all batch files and combine matches_data
    batch_files = sorted(results_path.glob("*_batch_*.pkl"))
    if not batch_files:
        raise FileNotFoundError(f"No batch files found in {results_dir}")
    
    print(f"✓ Found {len(batch_files)} batch files")
    
    all_matches_data = {}
    total_pairs = 0
    successful_pairs = 0
    
    for batch_file in batch_files:
        try:
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
            
            batch_results = batch_data.get('results', {})
            
            for pair_key, result in batch_results.items():
                # Convert string keys back to tuples if needed
                if isinstance(pair_key, str):
                    try:
                        pair_key = eval(pair_key)
                    except:
                        continue
                
                # Skip failed matches
                if 'error' in result:
                    continue
                
                # Add to combined matches_data
                all_matches_data[pair_key] = result
                successful_pairs += 1
            
            total_pairs += len(batch_results)
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load {batch_file.name}: {e}")
            continue
    
    print(f"✓ Loaded {successful_pairs}/{total_pairs} successful image pairs")
    
    # 4. Validate and return in expected format
    if not all_matches_data:
        raise ValueError("No valid match data found in batch files")
    
    matches_pickle = {
        'matches_data': all_matches_data,
        'image_info': image_info
    }
    
    # Print summary
    print("\n" + "="*60)
    print("MATCH DATA SUMMARY")
    print("="*60)
    print(f"Total images: {len(image_info)}")
    print(f"Total image pairs with matches: {len(all_matches_data)}")
    
    # Calculate match statistics
    match_counts = [data['num_matches'] for data in all_matches_data.values()]
    if match_counts:
        print(f"Matches per pair: {np.mean(match_counts):.1f} ± {np.std(match_counts):.1f}")
        print(f"Match range: [{min(match_counts)}, {max(match_counts)}]")
    
    print("="*60 + "\n")
    
    return matches_pickle


def validate_matches_structure(matches_pickle: Dict) -> bool:
    """
    Validate that matches_pickle has the correct structure for CameraPoseEstimation
    
    Args:
        matches_pickle: Dictionary to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    # Check top-level keys
    if 'matches_data' not in matches_pickle:
        raise ValueError("Missing 'matches_data' key")
    if 'image_info' not in matches_pickle:
        raise ValueError("Missing 'image_info' key")
    
    matches_data = matches_pickle['matches_data']
    image_info = matches_pickle['image_info']
    
    # Check matches_data structure
    if not isinstance(matches_data, dict):
        raise ValueError("'matches_data' must be a dictionary")
    
    if not matches_data:
        raise ValueError("'matches_data' is empty")
    
    # Check a sample pair
    sample_pair_key = next(iter(matches_data))
    sample_pair_data = matches_data[sample_pair_key]
    
    required_fields = ['correspondences', 'num_matches']
    for field in required_fields:
        if field not in sample_pair_data:
            raise ValueError(f"Match data missing required field: '{field}'")
    
    # Check image_info structure
    if not isinstance(image_info, dict):
        raise ValueError("'image_info' must be a dictionary")
    
    if not image_info:
        raise ValueError("'image_info' is empty")
    
    print("✓ Match data structure validated successfully")
    return True