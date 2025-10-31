"""
Dense Reconstruction - Complete Workflow Example
=================================================

This script demonstrates the complete dense reconstruction pipeline:
1. Load sparse reconstruction from CameraPoseEstimation
2. Run dense MVS (COLMAP/OpenMVS/Neural)
3. TSDF fusion
4. Mesh extraction and processing
5. Export results

Usage:
    python run_dense_reconstruction.py --sparse ./reconstruction_output --images ./images --output ./dense_output
"""

import argparse
from pathlib import Path
import pickle
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from DenseReconstruction import DenseReconstructionPipeline, DenseReconstructionConfig


def load_sparse_reconstruction(reconstruction_dir: Path):
    """
    Load sparse reconstruction from pickle file.

    Args:
        reconstruction_dir: Directory containing reconstruction.pkl

    Returns:
        Sparse reconstruction object
    """
    reconstruction_file = reconstruction_dir / "reconstruction.pkl"

    if not reconstruction_file.exists():
        raise FileNotFoundError(f"Reconstruction file not found: {reconstruction_file}")

    print(f"Loading sparse reconstruction from {reconstruction_file}...")

    with open(reconstruction_file, 'rb') as f:
        reconstruction = pickle.load(f)

    # Print reconstruction info
    if hasattr(reconstruction, 'cameras'):
        num_cameras = len(reconstruction.cameras)
        num_points = len(reconstruction.points) if hasattr(reconstruction, 'points') else 0
    elif isinstance(reconstruction, dict):
        num_cameras = len(reconstruction.get('cameras', {}))
        points_3d = reconstruction.get('points_3d', [])
        if isinstance(points_3d, dict):
            num_points = len(points_3d)
        else:
            num_points = points_3d.shape[1] if hasattr(points_3d, 'shape') else len(points_3d)
    else:
        num_cameras = 0
        num_points = 0

    print(f"✓ Loaded: {num_cameras} cameras, {num_points} sparse points")

    return reconstruction


def main():
    parser = argparse.ArgumentParser(description="Dense Reconstruction Pipeline")

    # Input/Output
    parser.add_argument('--sparse', type=str, required=True,
                       help='Path to sparse reconstruction directory (containing reconstruction.pkl)')
    parser.add_argument('--images', type=str, required=True,
                       help='Path to image folder')
    parser.add_argument('--output', type=str, default='./dense_output',
                       help='Output directory for dense reconstruction')

    # Method selection
    parser.add_argument('--method', type=str, default='auto',
                       choices=['auto', 'colmap', 'openmvs', 'neural'],
                       help='Dense reconstruction method (auto=best available)')

    # Quality settings
    parser.add_argument('--quality', type=str, default='medium',
                       choices=['low', 'medium', 'high', 'ultra'],
                       help='Depth map quality preset')

    # TSDF parameters
    parser.add_argument('--voxel-size', type=float, default=0.01,
                       help='TSDF voxel size in meters (default: 0.01 = 1cm)')
    parser.add_argument('--sdf-trunc', type=float, default=0.04,
                       help='TSDF truncation distance in meters (default: 0.04 = 4cm)')

    # Mesh options
    parser.add_argument('--mesh-method', type=str, default='marching_cubes',
                       choices=['marching_cubes', 'poisson', 'ball_pivoting'],
                       help='Mesh extraction method')
    parser.add_argument('--simplify', action='store_true',
                       help='Simplify mesh')
    parser.add_argument('--target-triangles', type=int, default=100000,
                       help='Target triangle count for simplification')
    parser.add_argument('--smooth', action='store_true',
                       help='Apply mesh smoothing')

    # Output options
    parser.add_argument('--no-point-cloud', action='store_true',
                       help='Skip point cloud export')
    parser.add_argument('--no-mesh', action='store_true',
                       help='Skip mesh export')

    # Logging
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log to file')

    args = parser.parse_args()

    # Setup paths
    sparse_dir = Path(args.sparse)
    image_folder = Path(args.images)
    output_dir = Path(args.output)

    # Validate inputs
    if not sparse_dir.exists():
        print(f"Error: Sparse reconstruction directory not found: {sparse_dir}")
        return 1

    if not image_folder.exists():
        print(f"Error: Image folder not found: {image_folder}")
        return 1

    # Load sparse reconstruction
    try:
        sparse_reconstruction = load_sparse_reconstruction(sparse_dir)
    except Exception as e:
        print(f"Error loading sparse reconstruction: {e}")
        return 1

    # Create configuration
    config = DenseReconstructionConfig(
        method=args.method,
        voxel_size=args.voxel_size,
        sdf_trunc=args.sdf_trunc,
        extract_mesh=not args.no_mesh,
        mesh_method=args.mesh_method,
        simplify_mesh=args.simplify,
        target_triangles=args.target_triangles,
        smooth_mesh=args.smooth,
        export_point_cloud=not args.no_point_cloud,
        export_mesh=not args.no_mesh,
        depth_map_quality=args.quality,
        verbose=args.verbose,
        log_file=args.log_file
    )

    # Create pipeline
    print("\n" + "="*70)
    print("DENSE RECONSTRUCTION PIPELINE")
    print("="*70)
    print(f"Method: {config.method}")
    print(f"Quality: {config.depth_map_quality}")
    print(f"Voxel size: {config.voxel_size}m")
    print(f"Output: {output_dir}")
    print("="*70 + "\n")

    try:
        pipeline = DenseReconstructionPipeline(config)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("\nTroubleshooting:")
        print("  - Install pycolmap: pip install pycolmap")
        print("  - Install Open3D: pip install open3d")
        print("  - Install PyTorch for neural depth: pip install torch")
        return 1

    # Run pipeline
    try:
        result = pipeline.run(
            sparse_reconstruction=sparse_reconstruction,
            image_folder=image_folder,
            output_dir=output_dir
        )

        if result['success']:
            print("\n" + "="*70)
            print("RECONSTRUCTION SUCCESSFUL")
            print("="*70)
            print(f"Point cloud: {result['point_cloud_path']}")
            print(f"Mesh: {result['mesh_path']}")
            print("\nStatistics:")
            for key, value in result['statistics'].items():
                print(f"  {key}: {value}")
            print("="*70)
            return 0
        else:
            print(f"\nReconstruction failed: {result.get('error', 'Unknown error')}")
            return 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during reconstruction: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
