# Dockerfile for 3D Reconstruction Pipeline
# Base image with Python, OpenCV, PyTorch, and system dependencies
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set non-interactive frontend for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        build-essential git wget unzip libgl1-mesa-glx libglib2.0-0 \
        libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install numpy opencv-python-headless torch torchvision

# Install LightGlue (editable mode)
RUN pip3 install -e Feature/LightGlue

# Install any additional requirements
RUN if [ -f CameraPoseEstimation/requirements.txt ]; then pip3 install -r CameraPoseEstimation/requirements.txt; fi

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

# Default command (can be overridden)
CMD ["python3"]
