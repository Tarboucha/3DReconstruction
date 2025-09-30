FROM python:3.9

WORKDIR /app

# Install core packages
RUN pip install --upgrade pip && pip install \
    numpy>=1.19.0 \
    matplotlib>=3.3.0 \
    pandas>=1.2.0 \
    psutil>=5.8.0 \
    scipy>=1.7.0

# Install PyTorch
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Install LightGlue dependencies
RUN pip install kornia einops

# STEP 1: Install headless OpenCV FIRST
RUN pip install opencv-python-headless==4.12.0.88

# STEP 2: Install LightGlue WITHOUT dependencies (to avoid opencv-python conflict)
RUN pip install --no-deps git+https://github.com/cvg/LightGlue.git || echo "LightGlue installation failed - continuing without it"

# STEP 3: Remove any GUI OpenCV that might have snuck in
RUN pip uninstall -y opencv-python || true

# Verify only headless is installed
RUN pip list | grep opencv && echo "OpenCV packages installed above"

# Set up package
RUN mkdir -p /app/feature_detection_system
COPY *.py ./feature_detection_system/
RUN touch /app/feature_detection_system/__init__.py
ENV PYTHONPATH=/app:/app/feature_detection_system
ENV QT_QPA_PLATFORM=offscreen

CMD ["python", "-c", "import cv2; print('OpenCV headless ready:', cv2.__version__)"]