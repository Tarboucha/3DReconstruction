import pickle
import os
from typing import List, Tuple, Dict
import cv2
import numpy as np



# def load_and_validate_pickle(pickle_file: str, convert_keypoints: bool = False) -> Dict:
#     """
#     Load and validate a pickle file created by this class
    
#     Args:
#         pickle_file: Path to pickle file
#         convert_keypoints: Whether to convert serialized keypoints back to cv2.KeyPoint objects
    
#     Returns:
#         Dictionary containing:
#             - image_names: List of image names
#             - matches_data: Dictionary of pairwise matches
#             - processing_stats: Processing statistics
#             - feature_type: Feature extraction method used
#             - total_images: Number of images processed
#     """
#     print(f"Loading and validating pickle file: {pickle_file}")
    
#     if not os.path.exists(pickle_file):
#         raise FileNotFoundError(f"Pickle file not found: {pickle_file}")
    
#     with open(pickle_file, 'rb') as f:
#         data = pickle.load(f)
    
#     # Handle both old format (just matches_data) and new format (complete structure)
#     if isinstance(data, dict) and 'matches_data' in data:
#         # New format with complete structure
#         image_info = data.get('image_info', {})
#         matches_data = data.get('matches_data', {})
#         processing_stats = data.get('processing_stats', {})
#         feature_type = data.get('feature_type', 'Unknown')
#         total_images = data.get('total_images', 0)
        
#         print(f"Loaded complete data structure:")
#         print(f"  Images: {len(image_info)}")
#         print(f"  Image pairs: {len(matches_data)}")
#         print(f"  Feature type: {feature_type}")
#         print(f"  Total images: {total_images}")
        
#     else:
#         # Old format (just matches_data dictionary)
#         matches_data = data
#         image_info = {}
#         processing_stats = {}
#         feature_type = 'Unknown'
#         total_images = 0
        
#         print(f"Loaded legacy format with {len(matches_data)} image pairs")
        
#         total_images = len(image_info)
#         print(f"  Extracted {total_images} unique image names")
    
#     # Convert keypoints back to cv2.KeyPoint objects if requested
#     if convert_keypoints:
#         print("Converting serialized keypoints back to cv2.KeyPoint objects...")
#         for pair_key, pair_data in matches_data.items():
#             if 'keypoints1' in pair_data:
#                 pair_data['keypoints1'] = serializable_to_keypoints(pair_data['keypoints1'])
#             if 'keypoints2' in pair_data:
#                 pair_data['keypoints2'] = serializable_to_keypoints(pair_data['keypoints2'])
    
#     # Validate matches_data format
#     print("Validating matches data format...")
#     for pair_key, pair_data in matches_data.items():
#         # Check pair key format
#         if not isinstance(pair_key, tuple) or len(pair_key) != 2:
#             raise ValueError(f"Invalid pair key format: {pair_key}")
        
#         # Check required fields
#         required_fields = ['correspondences', 'num_matches']
#         for field in required_fields:
#             if field not in pair_data:
#                 raise ValueError(f"Missing field '{field}' for pair {pair_key}")
        
#         # Check correspondences format (if not empty)
#         correspondences = pair_data['correspondences']
#         if correspondences and len(correspondences) > 0 and len(correspondences[0]) != 4:
#             raise ValueError(f"Invalid correspondences format for pair {pair_key}")
    
#     print("✓ Pickle file validation passed")
    
#     # Print summary statistics
#     if matches_data:
#         total_matches = sum(pair_data['num_matches'] for pair_data in matches_data.values())
#         avg_matches = total_matches / len(matches_data)
        
#         print(f"\nSummary:")
#         print(f"  Image pairs: {len(matches_data)}")
#         print(f"  Total matches: {total_matches}")
#         print(f"  Average matches per pair: {avg_matches:.1f}")
        
#         if processing_stats:
#             avg_time = processing_stats.get('avg_processing_time', 0)
#             total_time = processing_stats.get('total_processing_time', 0)
#             print(f"  Average processing time: {avg_time:.2f}s")
#             print(f"  Total processing time: {total_time:.2f}s")
    
#     # Return complete structure
#     return {
#         'image_info': image_info,
#         'matches_data': matches_data,
#         'processing_stats': processing_stats,
#         'feature_type': feature_type,
#         'total_images': total_images
#     }


def load_and_validate_pickle(pickle_file: str, convert_keypoints: bool = False) -> Dict:
    """
    Load and validate pickle file(s) created by save_current_batch.
    Can handle both single batch files and multiple batch files.
    Also loads image metadata from the separate metadata pickle file.
    
    Args:
        pickle_file: Path to a batch pickle file or pattern (e.g., 'results_batch_*.pkl')
                    If a single file is provided, will attempt to find all related batch files
        convert_keypoints: Whether to convert serialized keypoints back to cv2.KeyPoint objects
    
    Returns:
        Dictionary containing:
            - image_names: List of unique image names across all batches
            - image_info: Dictionary with image metadata (name, size, dimensions)
            - matches_data: Combined dictionary of pairwise matches from all batches
            - processing_stats: Aggregated processing statistics
            - feature_type: Feature extraction method used
            - total_images: Number of unique images processed
            - batch_info: Information about loaded batches
            - metadata_file: Path to the metadata file if found
    """
    import glob
    import os
    import re
    import pickle
    
    print(f"Loading pickle file(s): {pickle_file}")
    
    # Find all batch files and determine base pattern
    batch_files = []
    base_pattern = None
    dir_path = '.'
    
    if '*' in pickle_file:
        # Pattern provided (e.g., 'results_batch_*.pkl')
        batch_files = sorted(glob.glob(pickle_file))
        dir_path = os.path.dirname(pickle_file) or '.'
        base_name = os.path.basename(pickle_file)
        
        # Extract base pattern from wildcard
        match = re.match(r'(.+?)_batch_\*(\.\w+)$', base_name)
        if match:
            base_pattern = match.group(1)
    elif os.path.exists(pickle_file):
        # Single file provided
        dir_path = os.path.dirname(pickle_file) or '.'
        base_name = os.path.basename(pickle_file)
        
        # Check if it's a batch file
        match = re.match(r'(.+?)_batch_\d{3}(\.\w+)$', base_name)
        if match:
            # This is a batch file, find all related batches
            base_pattern = match.group(1)
            extension = match.group(2)
            pattern = os.path.join(dir_path, f"{base_pattern}_batch_*{extension}")
            batch_files = sorted(glob.glob(pattern))
        else:
            # Not a batch file pattern, check for summary file
            match = re.match(r'(.+?)_summary(\.\w+)$', base_name)
            if match:
                # This is a summary file, find all batch files
                base_pattern = match.group(1)
                extension = match.group(2)
                pattern = os.path.join(dir_path, f"{base_pattern}_batch_*{extension}")
                batch_files = sorted(glob.glob(pattern))
            else:
                # Just load this single file
                batch_files = [pickle_file]
    else:
        raise FileNotFoundError(f"No files found matching: {pickle_file}")
    
    if not batch_files:
        raise FileNotFoundError(f"No batch files found for pattern: {pickle_file}")
    
    print(f"Found {len(batch_files)} batch file(s) to load")
    
    # Load image metadata file if available
    image_metadata = None
    metadata_file = None
    
    if base_pattern:
        metadata_file = os.path.join(dir_path, f"{base_pattern}_image_metadata.pkl")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'rb') as f:
                    image_metadata = pickle.load(f)
                print(f"\n✓ Loaded image metadata from: {metadata_file}")
                print(f"  Total images in metadata: {image_metadata.get('total_images', 0)}")
                
                # Show sample image info
                if image_metadata.get('images'):
                    first_img = image_metadata['images'][0]
                    print(f"  Sample: {first_img['name']} - Size: {first_img['size']}")
            except Exception as e:
                print(f"⚠ Warning: Could not load image metadata file: {e}")
                image_metadata = None
        else:
            print(f"ℹ Note: Image metadata file not found: {metadata_file}")
            metadata_file = None
    
    # Initialize combined data structures
    all_matches_data = {}
    all_image_names = set()
    combined_stats = {
        'total_batches': len(batch_files),
        'total_pairs': 0,
        'successful_pairs': 0,
        'failed_pairs': 0,
        'total_matches': 0,
        'total_processing_time': 0,
        'batch_details': [],
        'quality_scores': [],
        'methods_used': set(),
        'score_types': set()
    }
    
    feature_type = 'Unknown'
    config = None
    
    # Process each batch file
    for batch_idx, batch_file in enumerate(batch_files, 1):
        print(f"\nProcessing batch {batch_idx}/{len(batch_files)}: {os.path.basename(batch_file)}")
        
        try:
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
            
            # Extract batch components
            results = batch_data.get('results', {})
            batch_stats = batch_data.get('batch_stats', {})
            overall_progress = batch_data.get('overall_progress', {})
            
            # Store config from first batch
            if config is None and 'config' in batch_data:
                config = batch_data['config']
                if config:
                    feature_type = config.get('feature_type', config.get('method', 'Unknown'))
            
            # Process pairs in this batch
            successful_in_batch = 0
            failed_in_batch = 0
            matches_in_batch = 0
            
            for pair_key, pair_data in results.items():
                # Handle string representation of tuple keys (backward compatibility)
                if isinstance(pair_key, str) and pair_key.startswith('(') and pair_key.endswith(')'):
                    try:
                        pair_key = eval(pair_key)
                    except:
                        print(f"  ⚠ Warning: Could not parse pair key: {pair_key}")
                        continue
                
                # Validate pair key format
                if not isinstance(pair_key, tuple) or len(pair_key) != 2:
                    print(f"  ⚠ Warning: Invalid pair key format: {pair_key}")
                    continue
                
                # Add image names to set
                all_image_names.add(pair_key[0])
                all_image_names.add(pair_key[1])
                
                # Check for duplicates
                if pair_key in all_matches_data:
                    print(f"  ⚠ Warning: Duplicate pair {pair_key} found, keeping first occurrence")
                    continue
                
                # Store pair data
                all_matches_data[pair_key] = pair_data
                
                # Update statistics
                if 'error' in pair_data:
                    failed_in_batch += 1
                    combined_stats['failed_pairs'] += 1
                else:
                    successful_in_batch += 1
                    combined_stats['successful_pairs'] += 1
                    
                    num_matches = pair_data.get('num_matches', 0)
                    matches_in_batch += num_matches
                    combined_stats['total_matches'] += num_matches
                    
                    # Collect quality scores
                    if 'quality_score' in pair_data:
                        combined_stats['quality_scores'].append(pair_data['quality_score'])
                    
                    # Track methods and score types
                    if 'method' in pair_data:
                        combined_stats['methods_used'].add(pair_data['method'])
                    if 'score_type' in pair_data:
                        combined_stats['score_types'].add(pair_data['score_type'])
                    
                    # Add processing time
                    if 'processing_time' in pair_data:
                        combined_stats['total_processing_time'] += pair_data['processing_time']
            
            combined_stats['total_pairs'] += len(results)
            
            # Store batch details
            batch_detail = {
                'batch_file': os.path.basename(batch_file),
                'batch_number': batch_stats.get('batch_number', batch_idx),
                'pairs_in_batch': len(results),
                'successful_pairs': successful_in_batch,
                'failed_pairs': failed_in_batch,
                'total_matches': matches_in_batch,
                'batch_processing_time': batch_stats.get('batch_processing_time', 0)
            }
            
            if overall_progress:
                batch_detail['overall_progress'] = overall_progress.get('progress_percent', 0)
                batch_detail['resumed_from'] = overall_progress.get('resumed_from')
            
            combined_stats['batch_details'].append(batch_detail)
            
            print(f"  ✓ Loaded: {successful_in_batch} successful, {failed_in_batch} failed pairs")
            print(f"  Total matches in batch: {matches_in_batch}")
            
        except Exception as e:
            print(f"  ✗ Error loading batch file: {e}")
            continue
    
    # Convert keypoints if requested
    if convert_keypoints:
        print("\nConverting serialized keypoints to cv2.KeyPoint objects...")
        converted_count = 0
        for pair_key, pair_data in all_matches_data.items():
            if 'keypoints1' in pair_data:
                pair_data['keypoints1'] = serializable_to_keypoints(pair_data['keypoints1'])
                converted_count += 1
            if 'keypoints2' in pair_data:
                pair_data['keypoints2'] = serializable_to_keypoints(pair_data['keypoints2'])
        print(f"  ✓ Converted keypoints for {converted_count} pairs")
    
    # Build image_info dictionary with metadata
    image_info = {}
    
    if image_metadata and image_metadata.get('images'):
        # Create lookup dictionary from metadata
        metadata_lookup = {img['name']: img for img in image_metadata['images']}
        
        # Build image_info with actual metadata
        images_with_metadata = 0
        images_without_metadata = []
        
        for name in sorted(all_image_names):
            if name in metadata_lookup:
                image_info[name] = metadata_lookup[name].copy()
                images_with_metadata += 1
            else:
                image_info[name] = {'name': name}
                images_without_metadata.append(name)
        
        print(f"\n✓ Image metadata applied to {images_with_metadata}/{len(all_image_names)} images")
        
        if images_without_metadata:
            print(f"⚠ Warning: {len(images_without_metadata)} images without metadata:")
            for name in images_without_metadata[:3]:
                print(f"    - {name}")
            if len(images_without_metadata) > 3:
                print(f"    ... and {len(images_without_metadata) - 3} more")
        
        # Check for unused images in metadata
        metadata_names = set(metadata_lookup.keys())
        unused_images = metadata_names - all_image_names
        if unused_images:
            print(f"\nℹ Note: {len(unused_images)} images in metadata not used in matches:")
            for name in list(unused_images)[:3]:
                print(f"    - {name}")
            if len(unused_images) > 3:
                print(f"    ... and {len(unused_images) - 3} more")
    else:
        # Create basic image info without metadata
        print("\nℹ Creating basic image info (no metadata available)")
        for name in sorted(all_image_names):
            image_info[name] = {'name': name}
    
    # Validate matches data
    print("\nValidating combined matches data...")
    validation_errors = 0
    validation_details = []
    
    for pair_key, pair_data in all_matches_data.items():
        if 'error' in pair_data:
            continue
        
        # Validate correspondences format
        if 'correspondences' in pair_data:
            correspondences = pair_data['correspondences']
            if correspondences and len(correspondences) > 0:
                if not isinstance(correspondences[0], (list, tuple)) or len(correspondences[0]) != 4:
                    validation_errors += 1
                    if len(validation_details) < 5:
                        validation_details.append(f"Invalid correspondences format for {pair_key}")
    
    if validation_errors > 0:
        print(f"  ⚠ Found {validation_errors} validation issues")
        for detail in validation_details:
            print(f"    - {detail}")
        if validation_errors > len(validation_details):
            print(f"    ... and {validation_errors - len(validation_details)} more")
    else:
        print("  ✓ All data validated successfully")
    
    # Calculate aggregated statistics
    if combined_stats['successful_pairs'] > 0:
        avg_matches = combined_stats['total_matches'] / combined_stats['successful_pairs']
        avg_processing_time = combined_stats['total_processing_time'] / combined_stats['successful_pairs']
    else:
        avg_matches = 0
        avg_processing_time = 0
    
    avg_quality = (sum(combined_stats['quality_scores']) / len(combined_stats['quality_scores']) 
                   if combined_stats['quality_scores'] else 0)
    
    processing_stats = {
        'total_batches': combined_stats['total_batches'],
        'total_pairs': combined_stats['total_pairs'],
        'successful_pairs': combined_stats['successful_pairs'],
        'failed_pairs': combined_stats['failed_pairs'],
        'success_rate': combined_stats['successful_pairs'] / max(combined_stats['total_pairs'], 1),
        'total_matches': combined_stats['total_matches'],
        'avg_matches_per_pair': avg_matches,
        'avg_processing_time': avg_processing_time,
        'total_processing_time': combined_stats['total_processing_time'],
        'avg_quality_score': avg_quality,
        'methods_used': list(combined_stats['methods_used']),
        'score_types': list(combined_stats['score_types'])
    }
    
    # Print final summary
    print("\n" + "="*60)
    print("LOADING COMPLETE - FINAL SUMMARY")
    print("="*60)
    
    print(f"\n📁 Files:")
    print(f"  Total batches loaded: {combined_stats['total_batches']}")
    if metadata_file:
        print(f"  Metadata file: ✓ {os.path.basename(metadata_file)}")
    else:
        print(f"  Metadata file: ✗ Not found")
    
    print(f"\n📷 Images:")
    print(f"  Total unique images: {len(all_image_names)}")
    
    if image_metadata and image_metadata.get('images'):
        # Show sample image dimensions
        sample_images = list(sorted(all_image_names))[:3]
        if sample_images:
            print("  Sample dimensions:")
            for name in sample_images:
                if name in image_info and 'size' in image_info[name]:
                    info = image_info[name]
                    print(f"    • {name}: {info['size']} ({info.get('dtype', 'unknown')})")
    
    print(f"\n📊 Matching Results:")
    print(f"  Total pairs processed: {combined_stats['total_pairs']}")
    print(f"    ✓ Successful: {combined_stats['successful_pairs']} ({processing_stats['success_rate']:.1%})")
    print(f"    ✗ Failed: {combined_stats['failed_pairs']}")
    print(f"  Total matches found: {combined_stats['total_matches']:,}")
    print(f"  Average matches/pair: {avg_matches:.1f}")
    
    if avg_quality > 0:
        print(f"  Average quality score: {avg_quality:.3f}")
    
    print(f"\n⏱️ Performance:")
    print(f"  Total processing time: {combined_stats['total_processing_time']:.2f}s")
    print(f"  Average time/pair: {avg_processing_time:.3f}s")
    
    if combined_stats['methods_used']:
        print(f"\n🔧 Methods used: {', '.join(combined_stats['methods_used'])}")
    
    if len(combined_stats['batch_details']) > 1:
        print(f"\n📦 Batch breakdown:")
        for detail in combined_stats['batch_details'][:5]:
            print(f"  Batch {detail['batch_number']:3d}: "
                  f"{detail['successful_pairs']:4d} successful, "
                  f"{detail['failed_pairs']:3d} failed "
                  f"({detail['total_matches']:5d} matches)")
        if len(combined_stats['batch_details']) > 5:
            print(f"  ... and {len(combined_stats['batch_details']) - 5} more batches")
    
    print("="*60)
    
    # Return complete structure
    return {
        'image_info': image_info,
        'image_names': sorted(all_image_names),
        'matches_data': all_matches_data,
        'processing_stats': processing_stats,
        'feature_type': feature_type,
        'total_images': len(all_image_names),
        'batch_info': combined_stats['batch_details'],
        'config': config,
        'metadata_file': metadata_file,
        'image_metadata': image_metadata  # Include raw metadata if needed
    }







def load_images(image_paths: List[str]) -> List[Tuple[np.ndarray, str]]:
        """
        Load images from file paths - MODIFIED to return filename with image
        
        Returns:
            List of tuples (image, filename)
        """
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not load image {path}")
                continue
            # Convert to RGB for processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            filename = os.path.basename(path)  # Extract filename
            images.append((img_rgb, filename))  # Return tuple
        return images


def serializable_to_keypoints( serializable_kps: List[Dict]) -> List[cv2.KeyPoint]:
    """
    Convert serializable format back to OpenCV KeyPoints
    
    Args:
        serializable_kps: List of dictionaries with keypoint data
        
    Returns:
        List of OpenCV KeyPoint objects
    """
    if not serializable_kps:
        return []
    
    keypoints = []
    for kp_dict in serializable_kps:
        kp = cv2.KeyPoint(
            x=kp_dict['pt'][0],
            y=kp_dict['pt'][1],
            size=kp_dict['size'],
            angle=kp_dict['angle'],
            response=kp_dict['response'],
            octave=kp_dict['octave'],
            class_id=kp_dict['class_id']
        )
        keypoints.append(kp)
   