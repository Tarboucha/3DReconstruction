# Copilot Instructions for 3D Reconstruction Pipeline

## Project Overview
This repository implements a modular 3D reconstruction pipeline, focusing on multi-view geometry, camera pose estimation, and dense reconstruction. The codebase is organized into several submodules, each responsible for a distinct stage of the pipeline.

## Key Components
- `CameraPoseEstimation2/`: Main pipeline logic for camera pose estimation, including:
  - `pipeline.py`: Orchestrates the full pose estimation workflow (pair selection, essential matrix, triangulation, bundle adjustment, export).
  - `essential_estimation.py`, `triangulation.py`, `bundle_adjusment.py`, `pair_selector.py`, `pose_recovery.py`, `intrinsics_estimator.py`, `quality_assessment.py`: Modular components for each geometric/computational step.
- `DenseReconstruction/`: Dense point cloud and mesh generation from estimated poses.
- `Feature/LightGlue/`: Feature extraction and matching using LightGlue and other local feature methods.
- `images/`: Input images, organized by scene/monument.
- `output/`: Output files (point clouds, meshes, reports).

## Data Flow
1. **Feature Extraction/Matching**: Features are extracted and matched (see `Feature/LightGlue/`).
2. **Pose Estimation**: `CameraPoseEstimation2/pipeline.py` loads matches, selects pairs, estimates essential matrices, recovers poses, triangulates points, and performs bundle adjustment.
3. **Dense Reconstruction**: Results are exported for dense reconstruction (see `DenseReconstruction/`).

## Patterns & Conventions
- **Reconstruction State**: Passed as a dict, containing cameras, points, observations, and intrinsics. Updated at each stage.
- **Observations**: 2D-3D correspondences are tracked per image for bundle adjustment and PnP.
- **Per-Camera Intrinsics**: Intrinsics (`K`) are stored per camera in the reconstruction state after initial estimation.
- **File Formats**: Pickle is used for intermediate results; JSON and COLMAP formats are supported for export.
- **Bootstrap Triangulation**: Uses both initial cameras to maximize early 3D points.
- **Quality Assessment**: Each major step logs quality metrics (inlier ratios, reprojection error, etc.).

## Developer Workflows
- **Run the pipeline**: Use `MainPosePipeline.process_monument_reconstruction()` with a matches pickle and output directory.
- **Intermediate Results**: Saved as `saved_variable.pkl` and in the output directory.
- **Testing**: See `test_suite.py`, `testCameraPose.py`, and `testFeature.py` for test entry points.
- **Feature Matching**: See `Feature/LightGlue/README.md` for LightGlue usage and configuration.

## Integration Points
- **LightGlue**: Integrated for feature matching; see `Feature/LightGlue/README.md` for details.
- **External Dependencies**: Numpy, OpenCV, and PyTorch are required. Install LightGlue as described in its README.

## Project-Specific Notes
- **Match Data Format**: Supports both LightGlue (`correspondences`) and OpenCV-style (`pts1`, `pts2`, `keypoints1`, `keypoints2`) match formats.
- **Image Naming**: Image pairs are referenced as tuples or `img1_vs_img2` strings in match dictionaries.
- **Extensibility**: New feature matchers or triangulation strategies can be added by extending the relevant modules.

## Example: Adding a New View
- Use `select_best_next_image()` to choose the next image.
- Use `find_correspondences_with_existing_3d()` to get 2D-3D matches.
- Use `pnp_solver.solve_pnp()` to estimate pose.
- Update the reconstruction state and run incremental bundle adjustment.

---

For more details, see `CameraPoseEstimation2/pipeline.py` and the LightGlue README.
