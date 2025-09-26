"""
Test script for Feature2 package using LightGlue matcher
"""
import cv2
import numpy as np
import sys
import os
import Feature as fds
import glob

# Ensure Feature2 is in the path
sys.path.insert(0, os.path.dirname(__file__))

from Feature import create_deep_learning_matcher, create_deep_learning_detector,create_pipeline

if __name__ == "__main__":

    folder_path='E://project//3Dreconstruction//images//statue_of_liberty'
    images_paths= fds.get_images_from_folder(folder_path)
    existing_paths = [path for path in images_paths if os.path.exists(path)]
    dataset_images = []
    for path in existing_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not load image {path}")
            continue
        # Convert to RGB for processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        filename = os.path.basename(path)  # Extract filename
        dataset_images.append((img_rgb, filename))  # Return tuple


    # Create pipeline for dataset processing
    config = fds.create_config_from_preset('deep_learning')
    pipeline = fds.FeatureProcessingPipeline(config)
    
    try:
        # Process dataset
        print("Processing dataset...")
        results = pipeline.process_dataset(
            dataset_images, 
            output_file="./output/sol/dataset_results.pkl",
            save_format='pickle'
        )
        
        
    except Exception as e:
        print(f"Dataset processing failed: {e}")