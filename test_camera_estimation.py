from FeatureMatchingExtraction import create_pipeline
from FeatureMatchingExtraction.export_adapter import export_for_pose_estimation
from CameraPoseEstimation.pipeline import IncrementalReconstructionPipeline
from CameraPoseEstimation.data import create_provider

# ============================================================================
# STEP 1: Feature Detection & Matching (Already Done)
# ============================================================================
# Your run_feature_detection.py has already created:
# - ./output/batches/
# - ./output/matching_results/
# - ./output/image_metadata.pkl

# ============================================================================
# STEP 2: Convert to Pose Estimation Format
# ============================================================================
print("Converting batch data to pose estimation format...")

# export_for_pose_estimation(
#     input_dir='./output',                    # Your batch output
#     output_dir='./pose_estimation_input',    # Converted format
#     method='all',                           # Use best method from multi-method
#     min_matches=10,                          # Filter threshold
#     min_quality=0.3                          # Quality threshold
# )

# ============================================================================
# STEP 3: Run Pose Estimation
# ============================================================================
print("\nRunning pose estimation...")

# Create data provider
provider = create_provider(
    './pose_estimation_input',
    cache_size=100,
    min_matches=10
)

# Create reconstruction pipeline
pipeline = IncrementalReconstructionPipeline(
    provider=provider,
    output_dir='./reconstruction_output'
)

# Run reconstruction
reconstruction = pipeline.run()

# ============================================================================
# STEP 4: Results
# ============================================================================
print("\n" + "="*70)
print("RECONSTRUCTION COMPLETE")
print("="*70)
print(f"Cameras reconstructed: {len(reconstruction.cameras)}")
print(f"3D points: {reconstruction.get_num_points()}")
print(f"Observations: {len(reconstruction.observations)}")

# Save reconstruction
reconstruction.save('./reconstruction_output/final_reconstruction.pkl')