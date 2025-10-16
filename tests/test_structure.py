"""
Simple Test Script for Incremental Pipeline

A minimal test script for quick pipeline testing.

Usage:
    python simple_test.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent  # Go up from tests/ to project root
sys.path.insert(0, str(project_root))

from CameraPoseEstimation2.pipeline.incremental import IncrementalReconstructionPipeline
from CameraPoseEstimation2.data import create_provider


def simple_test():
    """Run a simple two-view reconstruction test"""
    
    print("="*70)
    print("SIMPLE PIPELINE TEST")
    print("="*70)
    
    # Configuration
    DATA_DIR = "./test_output_v3/matching_results/matching_results"  # Change this to your match results directory
    OUTPUT_DIR = "./simple_test_output"
    
    print(f"\nData: {DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    try:
        # 1. Create provider
        print("\n[1/3] Creating provider...")
        provider = create_provider("structured", results_dir=DATA_DIR, cache_size=50)
        
        images = list(provider.get_all_images())
        print(f"✓ Found {len(images)} images")
        
        if len(images) < 2:
            print("❌ Need at least 2 images!")
            return False
        
        # 2. Create pipeline
        print("\n[2/3] Creating pipeline...")
        pipeline = IncrementalReconstructionPipeline(provider, output_dir=OUTPUT_DIR)
        print("✓ Pipeline ready")
        
        # 3. Run two-view initialization only (quick test)
        print("\n[3/3] Running two-view initialization...")
        print("-"*70)
        
        pipeline._initialize_two_view()
        
        print("-"*70)
        print("\n✅ TEST COMPLETE!")
        print("\nResults:")
        print(f"  Cameras: {len(pipeline.reconstruction.cameras)}")
        print(f"  Points: {len(pipeline.reconstruction.points)}")
        print(f"  Observations: {len(pipeline.reconstruction.observations)}")
        
        # Quick quality check
        if len(pipeline.reconstruction.points) < 10:
            print("\n⚠️  Warning: Very few points triangulated!")
            return False
        
        print(f"\n✓ Output saved to: {OUTPUT_DIR}")
        return True
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\nMake sure match results exist in: {DATA_DIR}")
        return False
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = simple_test()
    sys.exit(0 if success else 1)