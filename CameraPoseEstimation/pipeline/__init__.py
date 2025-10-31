"""
Camera Pose Estimation Pipeline

Main pipeline module for incremental 3D reconstruction from image matches.

Components:
- IncrementalReconstructionPipeline: Main pipeline class
- Reconstruction: Data structure for managing cameras, points, and observations
- Camera, Point3D, Observation: Core data classes

Usage:
    from CameraPoseEstimation2.pipeline import IncrementalReconstructionPipeline
    from CameraPoseEstimation2.data import create_provider
    
    # Create data provider
    provider = create_provider('./feature_output')
    
    # Create and run pipeline
    pipeline = IncrementalReconstructionPipeline(
        provider=provider,
        output_dir='./reconstruction_output'
    )
    
    reconstruction = pipeline.run()
    
    logger.info(f"Cameras: {len(reconstruction.cameras)}")
    logger.info(f"Points: {len(reconstruction.points)}")

File: CameraPoseEstimation2/pipeline/__init__.py
"""

# Main pipeline class
from .incremental import (
from CameraPoseEstimation2.logger import get_logger

logger = get_logger("pipeline")
    IncrementalReconstructionPipeline,
    Reconstruction,
    Camera,
    Point3D,
    Observation
)


__all__ = [
    # Main pipeline
    'IncrementalReconstructionPipeline',
    
    # Data structures
    'Reconstruction',
    'Camera',
    'Point3D',
    'Observation',
]


__version__ = '2.0.0'


# Module metadata
__author__ = '3D Reconstruction Team'
__description__ = 'Incremental 3D reconstruction pipeline from image matches'


# Convenience function for backward compatibility
def create_pipeline(provider, output_dir: str = "./output"):
    """
    Create a reconstruction pipeline.
    
    Args:
        provider: IMatchDataProvider instance
        output_dir: Output directory
        
    Returns:
        IncrementalReconstructionPipeline instance
        
    Example:
        >>> from CameraPoseEstimation2.pipeline import create_pipeline
        >>> from CameraPoseEstimation2.data import create_provider
        >>> 
        >>> provider = create_provider('./feature_output')
        >>> pipeline = create_pipeline(provider, './output')
        >>> reconstruction = pipeline.run()
    """
    return IncrementalReconstructionPipeline(provider, output_dir)


# Legacy alias for backward compatibility
MainPosePipeline = IncrementalReconstructionPipeline