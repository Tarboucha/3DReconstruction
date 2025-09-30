"""
Setup script for the Multi-Method Feature Detection and Matching System.
"""

from setuptools import setup, find_packages
import os


def read_requirements(filename):
    """Read requirements from file, filtering out comments and empty lines."""
    requirements = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements


def read_readme():
    """Read README file for long description."""
    if os.path.exists('README.md'):
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    return "Multi-Method Feature Detection and Matching System"


# Core requirements (always installed)
install_requires = [
    'opencv-python>=4.5.0',
    'numpy>=1.19.0',
    'matplotlib>=3.3.0',
    'pandas>=1.2.0',
    'psutil>=5.8.0'
]

# Optional dependencies for different use cases
extras_require = {
    'deep_learning': [
        'torch>=1.8.0',
        'torchvision>=0.9.0',
        'lightglue>=0.1.0'
    ],
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.12.0',
        'black>=21.0.0',
        'isort>=5.9.0',
        'flake8>=3.9.0'
    ],
    'docs': [
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.0',
        'myst-parser>=0.15.0'
    ],
    'all': [
        # Include all optional dependencies
        'torch>=1.8.0',
        'torchvision>=0.9.0',
        'lightglue>=0.1.0',
        'pytest>=6.0.0',
        'pytest-cov>=2.12.0',
        'black>=21.0.0',
        'isort>=5.9.0',
        'flake8>=3.9.0',
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.0',
        'myst-parser>=0.15.0'
    ]
}

setup(
    name="feature-detection-system",
    version="1.0.0",
    author="Feature Detection Team",
    author_email="support@example.com",
    description="A comprehensive multi-method feature detection and matching system",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/feature-detection-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        'feature_detection_system': [
            '*.yml',
            '*.yaml',
            '*.json'
        ]
    },
    entry_points={
        'console_scripts': [
            'fds-benchmark=feature_detection_system.benchmarking:main',
            'fds-demo=feature_detection_system.examples:run_quick_demo',
        ],
    },
    keywords=[
        "computer vision",
        "feature detection",
        "feature matching", 
        "SIFT",
        "ORB",
        "SuperPoint",
        "LightGlue",
        "image processing",
        "deep learning",
        "opencv"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/feature-detection-system/issues",
        "Source": "https://github.com/your-username/feature-detection-system",
        "Documentation": "https://feature-detection-system.readthedocs.io/",
    }
)


# Installation message
print("""
🚀 Feature Detection System Installation
=========================================

Installation Options:
  pip install feature-detection-system                    # Basic installation
  pip install feature-detection-system[deep_learning]     # With PyTorch support
  pip install feature-detection-system[all]               # Full installation
  pip install -e .[dev]                                   # Development mode

Quick Start:
  import feature_detection_system as fds
  fds.print_capabilities()

For examples and documentation, visit:
  https://github.com/your-username/feature-detection-system
""")