#!/usr/bin/env python3
"""
Test script for two-view initialization with bundle adjustment
"""

import numpy as np
import cv2
import pickle
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import your bundle adjustment classes
from CameraPoseEstimation2 import IncrementalBundleAdjuster
from CameraPoseEstimation2.pipeline2 import MainPosePipeline
from others import load_and_validate_pickle


def test_reconstruction_state_structure(reconstruction_state):
    """Test that the reconstruction state has the expected structure"""
    print("\n=== Testing Reconstruction State Structure ===")
    
    required_keys = ['cameras', 'points_3d', 'observations', 'camera_matrix', 'initialization_info']
    
    for key in required_keys:
        assert key in reconstruction_state, f"Missing key: {key}"
        print(f"✓ {key} present")
    
    # Test cameras structure
    cameras = reconstruction_state['cameras']
    assert len(cameras) == 2, f"Expected 2 cameras, got {len(cameras)}"
    
    for cam_id, cam_data in cameras.items():
        assert 'R' in cam_data and 't' in cam_data, f"Camera {cam_id} missing R or t"
        assert cam_data['R'].shape == (3, 3), f"Wrong R shape for {cam_id}"
        assert cam_data['t'].shape == (3, 1), f"Wrong t shape for {cam_id}"
    
    # Test points_3d structure
    points_3d = reconstruction_state['points_3d']['points_3d']
    assert points_3d.shape[0] == 3, f"Points should have 3 coordinates, got {points_3d.shape[0]}"
    assert points_3d.shape[1] > 0, "Should have some 3D points"
    
    # Test observations structure
    observations = reconstruction_state['observations']
    assert len(observations) == 2, f"Expected observations for 2 cameras, got {len(observations)}"
    
    print("✓ All structure tests passed!")

def test_bundle_adjustment_integration(reconstruction_state):
    """Test that bundle adjustment was properly integrated"""
    print("\n=== Testing Bundle Adjustment Integration ===")
    
    # Check if optimization history was added
    if 'optimization_history' in reconstruction_state:
        opt_history = reconstruction_state['optimization_history']
        print(f"✓ Optimization history present with {len(opt_history)} entries")
        
        if len(opt_history) > 0:
            last_opt = opt_history[-1]
            required_opt_keys = ['type', 'initial_cost', 'final_cost', 'iterations']
            
            for key in required_opt_keys:
                assert key in last_opt, f"Missing optimization key: {key}"
            
            print(f"✓ Last optimization: {last_opt['type']}")
            print(f"✓ Cost improvement: {last_opt['initial_cost']:.3f} → {last_opt['final_cost']:.3f}")
    else:
        print("! No optimization history found (BA might have been skipped)")

def main():
    """Main test function"""
    print("=== Testing Two-View Initialization with Bundle Adjustment ===")
    
    try:
        # Create mock pipeline
        pipeline = MainPosePipeline()
        
        pickle_file= './output/sol2/dataset_results_batch_001.pkl'
        matches_pickle = load_and_validate_pickle(pickle_file=pickle_file, convert_keypoints=True)
        
        print("Starting two-view initialization test...")
        
        # Run the method we're testing
        reconstruction_state = pipeline.process_monument_reconstruction(matches_pickle, chosen_images=(
            'g01f09eae8523020ec8f9d6b1452305c1c02b066eda4ddbd5135158740033e45302806e322816a39cc80f6dbb123bf1fb33f9f2d3e7e4f4b2ea28030c5efd731f_1280.jpg',
            'ge9f6ae7f4cad8547289327e648fa051a162c6d68027d76c8d6e4f0450e187681febb3ba0bab05be18330f3d27fb2c878bf3d3982d297d49e39f848c7647505c0_1280.jpg'))
        
        # Test the results
        test_reconstruction_state_structure(reconstruction_state)
        test_bundle_adjustment_integration(reconstruction_state)
        
        # Print summary
        print("\n=== Test Results Summary ===")
        print(f"✓ Cameras initialized: {list(reconstruction_state['cameras'].keys())}")
        print(f"✓ 3D points: {reconstruction_state['points_3d']['points_3d'].shape[1]}")
        print(f"✓ Camera matrix shape: {reconstruction_state['camera_matrix'].shape}")
        print(f"✓ Initialization method: {reconstruction_state['initialization_info']['method']}")
        print(f"✓ Quality assessment: {reconstruction_state['initialization_info']['quality_assessment']}")
        
        if 'optimization_history' in reconstruction_state:
            print(f"✓ Bundle adjustment completed: {len(reconstruction_state['optimization_history'])} optimization(s)")
        
        print("\n🎉 All tests passed! Two-view initialization working correctly.")
        
        return reconstruction_state
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
    
    if result is not None:
        print("\n=== Optional: Inspect Results ===")
        print("You can now inspect 'result' to see the full reconstruction state")
        
        # Optional: Print some detailed results
        print(f"\nCamera 1 pose (R):\n{result['cameras']['Image_1.jpg']['R']}")
        print(f"\nCamera 2 pose (R):\n{result['cameras']['Image_5.jpg']['R']}")
        print(f"\nFirst few 3D points:\n{result['points_3d']['points_3d'][:, :5]}")