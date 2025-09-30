"""
Example: Process a folder of images using ORB and LightGlue
Save results to pickle file for later analysis
"""

import os
from pathlib import Path
from FeatureMatchingExtraction.pipeline import create_pipeline
import pickle
import time

def process_folder_with_orb_lightglue(folder_path: str, 
                                     output_dir: str = "results",
                                     max_images: int = None,
                                     resize_images: bool = True):
    """
    Process folder using ORB and LightGlue, save to pickle
    
    Args:
        folder_path: Path to folder containing images
        output_dir: Directory to save results
        max_images: Maximum number of images to process (None = all)
        resize_images: Whether to resize images for faster processing
    """
    
    print("🚀 Starting folder processing with ORB and LightGlue")
    print("=" * 60)
    
    # =============================================================================
    # Method 1: Simple approach using create_pipeline
    # =============================================================================
    
    print("Setting up pipeline...")
    
    # Create pipeline with custom configuration
    pipeline = create_pipeline(
        preset='fast',  # Base preset (we'll override methods)
        methods=['lightglue'], 
        detector_params={
            'ORB': {
                'scale_factor': 1.2,
                'n_levels': 8,
                'edge_threshold': 31
            },
            'lightglue': {
                'features': 'superpoint',
                'confidence_threshold': 0.2,
                'max_num_keypoints': 2000
            }
        }
    )
    
    print(f"📁 Processing folder: {folder_path}")
    print(f"🔧 Methods: ORB, LightGlue")
    print(f"💾 Output directory: {output_dir}")
    
    # =============================================================================
    # Process the folder
    # =============================================================================
    
    # Set up resize configuration
    resize_to = (640, 480) if resize_images else None
    
    # Generate output filename with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = f"folder_processing_orb_lightglue_{timestamp}"
    
    try:
        # Process the folder
        results = pipeline.process_folder(
            folder_path=folder_path,
            max_images=max_images,
            resize_to=resize_to,
            visualize=False,  # Don't show visualizations for batch processing
            output_file=os.path.join(output_dir, output_file),
            save_format='pickle'  # Save as pickle file
        )
        
        print("\n✅ Processing completed successfully!")
        
        # =============================================================================
        # Print summary statistics
        # =============================================================================
        
        print("\n📊 RESULTS SUMMARY:")
        print("-" * 40)
        
        overall_stats = results['overall_stats']
        print(f"Total image pairs: {overall_stats['total_pairs']}")
        print(f"Successful pairs: {overall_stats['successful_pairs']}")
        print(f"Success rate: {overall_stats['success_rate']:.1%}")
        print(f"Average matches per pair: {overall_stats['avg_matches']:.1f}")
        print(f"Processing time: {overall_stats['processing_time']:.1f} seconds")
        print(f"Methods used: {', '.join(overall_stats['methods_used'])}")
        
        # Show file information
        print(f"\n💾 SAVED FILES:")
        print("-" * 20)
        for i, batch_file in enumerate(results['batch_files'][:3]):  # Show first 3 batch files
            file_size = os.path.getsize(batch_file) / 1024 / 1024  # MB
            print(f"Batch {i+1}: {os.path.basename(batch_file)} ({file_size:.1f} MB)")
        
        if len(results['batch_files']) > 3:
            print(f"... and {len(results['batch_files']) - 3} more batch files")
        
        print(f"Summary: {os.path.basename(results['summary_file'])}")
        print(f"Image metadata: {os.path.basename(results['image_metadata_file'])}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_saved_results(results_dir: str, summary_file: str = None):
    """
    Analyze previously saved results from pickle files
    
    Args:
        results_dir: Directory containing saved results
        summary_file: Specific summary file to analyze (auto-detect if None)
    """
    
    print("🔍 Analyzing saved results...")
    print("=" * 40)
    
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Find summary file if not specified
    if summary_file is None:
        summary_files = list(results_path.glob("*_summary.pkl"))
        if not summary_files:
            print(f"❌ No summary files found in {results_dir}")
            return
        summary_file = str(summary_files[-1])  # Use most recent
    
    # Load summary
    try:
        with open(summary_file, 'rb') as f:
            summary_data = pickle.load(f)
        
        print(f"📄 Loaded: {os.path.basename(summary_file)}")
        
        # Extract key information
        overall_stats = summary_data['overall_stats']
        config = summary_data['config']
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"Success rate: {overall_stats['success_rate']:.1%}")
        print(f"Total pairs processed: {overall_stats['total_pairs']}")
        print(f"Average matches: {overall_stats['avg_matches']:.1f}")
        print(f"Total processing time: {overall_stats['processing_time']:.1f}s")
        print(f"Methods: {', '.join(overall_stats['methods_used'])}")
        
        # Resume information
        if overall_stats.get('resumed_from_batch', 0) > 0:
            resume_info = summary_data['resume_info']
            print(f"\n🔄 RESUME INFO:")
            print(f"Was resumed: {resume_info['was_resumed']}")
            print(f"Previously processed pairs: {resume_info['total_existing_pairs']}")
            print(f"New pairs this session: {resume_info['new_pairs_processed']}")
        
        # Load and analyze a sample batch
        batch_files = overall_stats['batch_files']
        if batch_files:
            print(f"\n🗂️  BATCH FILES: {len(batch_files)} total")
            
            # Load first batch for detailed analysis
            first_batch = batch_files[0]
            try:
                with open(first_batch, 'rb') as f:
                    batch_data = pickle.load(f)
                
                batch_results = batch_data['results']
                successful_pairs = [r for r in batch_results.values() if 'error' not in r]
                
                if successful_pairs:
                    # Analyze methods used
                    methods_count = {}
                    quality_scores = []
                    match_counts = []
                    
                    for result in successful_pairs:
                        method = result.get('method', 'unknown')
                        methods_count[method] = methods_count.get(method, 0) + 1
                        quality_scores.append(result.get('quality_score', 0))
                        match_counts.append(result.get('num_matches', 0))
                    
                    print(f"\n📊 SAMPLE ANALYSIS (First batch):")
                    print(f"Methods used: {methods_count}")
                    print(f"Average quality score: {sum(quality_scores)/len(quality_scores):.3f}")
                    print(f"Average matches: {sum(match_counts)/len(match_counts):.1f}")
                    
                    # Show a few example pairs
                    print(f"\n🔍 EXAMPLE PAIRS:")
                    for i, (pair_key, result) in enumerate(list(batch_results.items())[:3]):
                        if 'error' not in result:
                            print(f"  {pair_key}: {result['num_matches']} matches "
                                  f"({result['method']}, quality: {result['quality_score']:.3f})")
                        else:
                            print(f"  {pair_key}: ERROR - {result['error']}")
                
            except Exception as e:
                print(f"⚠️  Could not analyze batch file: {e}")
        
        return summary_data
        
    except Exception as e:
        print(f"❌ Error loading summary file: {e}")
        return None


def extract_correspondences_from_results(results_dir: str, pair_name: str = None):
    """
    Extract correspondence data from saved results for further analysis
    
    Args:
        results_dir: Directory containing saved results
        pair_name: Specific pair to extract (e.g., "('image1.jpg', 'image2.jpg')")
    """
    
    print("📐 Extracting correspondence data...")
    
    results_path = Path(results_dir)
    batch_files = list(results_path.glob("*_batch_*.pkl"))
    
    all_correspondences = {}
    
    for batch_file in batch_files:
        try:
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
            
            batch_results = batch_data['results']
            
            for pair_key, result in batch_results.items():
                if 'error' not in result and 'correspondences' in result:
                    correspondences = result['correspondences']
                    if correspondences:  # Not empty
                        all_correspondences[str(pair_key)] = {
                            'correspondences': correspondences,
                            'num_matches': result['num_matches'],
                            'method': result['method'],
                            'quality_score': result['quality_score']
                        }
        
        except Exception as e:
            print(f"⚠️  Error reading {batch_file}: {e}")
    
    print(f"📊 Found correspondences for {len(all_correspondences)} pairs")
    
    # If specific pair requested, return just that
    if pair_name and pair_name in all_correspondences:
        return all_correspondences[pair_name]
    
    return all_correspondences


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    
    # Example 1: Process a folder
    folder_path = "E:\\project\\3Dreconstruction\\images\\Eiffel Tower copy"  # Replace with your folder path

    print("Example 1: Processing folder with ORB and LightGlue")
    results = process_folder_with_orb_lightglue(
        folder_path=folder_path,
        output_dir="./results",
        max_images=20,  # Process max 20 images for testing
        resize_images=True
    )
    
    if results:
        print(f"\n✅ Results saved to: {results['summary_file']}")
        
        # Example 2: Analyze the saved results
        print("\n" + "="*60)
        print("Example 2: Analyzing saved results")
        
        summary_data = analyze_saved_results("./results")
        
        # Example 3: Extract correspondence data
        if summary_data:
            print("\n" + "="*60)
            print("Example 3: Extracting correspondences")
            
            correspondences = extract_correspondences_from_results("./results")
            
            if correspondences:
                # Show first few pairs
                for i, (pair_name, data) in enumerate(list(correspondences.items())[:3]):
                    print(f"Pair {i+1}: {pair_name}")
                    print(f"  Method: {data['method']}")
                    print(f"  Matches: {data['num_matches']}")
                    print(f"  Quality: {data['quality_score']:.3f}")
                    print(f"  Correspondences shape: {len(data['correspondences'])} x 4")


# =============================================================================
# Quick one-liner for simple use cases
# =============================================================================

def quick_folder_process(folder_path: str):
    """One-liner to quickly process a folder with ORB and LightGlue"""
    
    pipeline = create_pipeline(
        methods=['ORB', 'lightglue'],
        max_features=1500
    )
    
    return pipeline.process_folder(
        folder_path=folder_path,
        resize_to=(640, 480),
        save_format='pickle'
    )

# Usage: 
# results = quick_folder_process("/path/to/images")