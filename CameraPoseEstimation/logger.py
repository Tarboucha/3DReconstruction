"""
Logging utility for CameraPoseEstimation2

Provides centralized logging configuration with file and console output support.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "CameraPoseEstimation2",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    force: bool = False
) -> logging.Logger:
    """
    Setup logger with file and/or console output

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        console: Whether to output to console
        force: Force reconfiguration even if already configured

    Returns:
        Configured logger

    Example:
        >>> logger = setup_logger('CameraPoseEstimation2', level='DEBUG', log_file='reconstruction.log')
        >>> logger.info("Reconstruction started")
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured (unless force=True)
    if logger.handlers and not force:
        return logger

    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []  # Clear existing handlers

    # Format: [2025-10-31 10:15:30] [INFO] [reconstruction] Message
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger for a specific module

    Args:
        name: Module name (e.g., 'pipeline', 'triangulation', 'bundle_adjustment')

    Returns:
        Logger instance

    Example:
        >>> # In incremental.py
        >>> from .logger import get_logger
        >>> logger = get_logger("pipeline")
        >>> logger.info("Starting reconstruction")
    """
    return logging.getLogger(f"CameraPoseEstimation2.{name}")


def configure_root_logger(level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure the root CameraPoseEstimation2 logger

    This should be called once at the start of reconstruction.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file

    Example:
        >>> from CameraPoseEstimation2.logger import configure_root_logger
        >>> configure_root_logger(level='DEBUG', log_file='./output/reconstruction.log')
    """
    setup_logger(
        name="CameraPoseEstimation2",
        level=level,
        log_file=log_file,
        console=True,
        force=True
    )


def disable_console_logging():
    """Disable console output, keep only file logging"""
    logger = logging.getLogger("CameraPoseEstimation2")
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            logger.removeHandler(handler)


def set_level(level: str):
    """
    Change logging level dynamically

    Args:
        level: New logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logger = logging.getLogger("CameraPoseEstimation2")
    logger.setLevel(getattr(logging, level.upper()))
