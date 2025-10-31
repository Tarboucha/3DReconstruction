"""
CameraPoseEstimation2 - Incremental 3D Reconstruction Pipeline

A modular framework for incremental Structure from Motion (SfM) reconstruction.
"""

from .logger import (
    setup_logger,
    get_logger,
    configure_root_logger,
    disable_console_logging,
    set_level
)

__version__ = "2.0.0"
__all__ = [
    "setup_logger",
    "get_logger",
    "configure_root_logger",
    "disable_console_logging",
    "set_level",
]
