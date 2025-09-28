#!/bin/bash

# Docker build script for Feature Detection System
# Usage: ./build.sh [basic|full|dev|all]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check if required files exist
check_files() {
    local required_files=("Dockerfile" "requirements.txt" "docker-compose.yml")
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            print_error "Required file '$file' not found in current directory"
            exit 1
        fi
    done
    print_success "All required files found"
}

# Function to create necessary directories
create_directories() {
    local dirs=("images" "results" "benchmark_results" "notebooks")
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        fi
    done
}

# Function to build basic version
build_basic() {
    print_status "Building basic version (OpenCV + traditional methods)..."
    docker build -t feature-detection:basic --target base . || {
        print_error "Failed to build basic version"
        return 1
    }
    print_success "Basic version built successfully"
}

# Function to build full version
build_full() {
    print_status "Building full version (+ PyTorch/deep learning)..."
    docker build -t feature-detection:full --target full . || {
        print_error "Failed to build full version"
        return 1
    }
    print_success "Full version built successfully"
}

# Function to build development version
build_dev() {
    print_status "Building development version (+ Jupyter/dev tools)..."
    docker build -t feature-detection:dev --target development . || {
        print_error "Failed to build development version"
        return 1
    }
    print_success "Development version built successfully"
}

# Function to test basic installation
test_basic() {
    print_status "Testing basic installation..."
    docker run --rm feature-detection:basic python -c "
import cv2
import numpy as np
import matplotlib
print('✓ OpenCV version:', cv2.__version__)
print('✓ NumPy version:', np.__version__)
print('✓ Matplotlib version:', matplotlib.__version__)
print('✓ Basic installation working!')
" || {
        print_error "Basic installation test failed"
        return 1
    }
    print_success "Basic installation test passed"
}

# Function to test full installation
test_full() {
    print_status "Testing full installation..."
    docker run --rm feature-detection:full python -c "
import cv2
import numpy as np
import torch
print('✓ OpenCV version:', cv2.__version__)
print('✓ NumPy version:', np.__version__)
print('✓ PyTorch version:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
print('✓ Full installation working!')
" || {
        print_error "Full installation test failed"
        return 1
    }
    print_success "Full installation test passed"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [basic|full|dev|all|test|clean]"
    echo ""
    echo "Options:"
    echo "  basic  - Build basic version (OpenCV + traditional methods)"
    echo "  full   - Build full version (+ PyTorch/deep learning)"
    echo "  dev    - Build development version (+ Jupyter/dev tools)"
    echo "  all    - Build all versions"
    echo "  test   - Test installations"
    echo "  clean  - Clean up Docker images"
    echo "  help   - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 basic        # Build basic version only"
    echo "  $0 all          # Build all versions"
    echo "  $0 full test    # Build full version and test"
}

# Function to clean up Docker images
clean_docker() {
    print_status "Cleaning up Docker images..."
    
    # Remove feature detection images
    docker images | grep feature-detection | awk '{print $3}' | xargs -r docker rmi -f
    
    # Clean up dangling images
    docker image prune -f
    
    print_success "Docker cleanup completed"
}

# Function to show quick start instructions
show_quick_start() {
    echo ""
    echo -e "${GREEN}🎉 Build completed successfully!${NC}"
    echo ""
    echo "Quick start commands:"
    echo ""
    echo "1. Run basic version:"
    echo "   docker-compose up -d feature-detection-basic"
    echo "   docker-compose exec feature-detection-basic bash"
    echo ""
    echo "2. Run full version:"
    echo "   docker-compose up -d feature-detection-full"
    echo "   docker-compose exec feature-detection-full bash"
    echo ""
    echo "3. Start Jupyter development:"
    echo "   docker-compose up feature-detection-dev"
    echo "   # Then open http://localhost:8888"
    echo ""
    echo "4. Run a quick test:"
    echo "   # Copy some images to ./images/ directory first"
    echo "   docker-compose exec feature-detection-full python -c \\"
    echo "   \"import feature_detection_system as fds; \\"
    echo "   results = fds.quick_folder_benchmark('/app/images', ['SIFT', 'ORB'])\""
    echo ""
}

# Main script logic
main() {
    print_status "Feature Detection System - Docker Build Script"
    print_status "============================================="
    
    # Check prerequisites
    check_docker
    check_files
    create_directories
    
    # Parse command line arguments
    if [[ $# -eq 0 ]]; then
        show_usage
        exit 1
    fi
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            basic)
                build_basic
                shift
                ;;
            full)
                build_full
                shift
                ;;
            dev)
                build_dev
                shift
                ;;
            all)
                build_basic
                build_full  
                build_dev
                shift
                ;;
            test)
                if docker image inspect feature-detection:basic > /dev/null 2>&1; then
                    test_basic
                fi
                if docker image inspect feature-detection:full > /dev/null 2>&1; then
                    test_full
                fi
                shift
                ;;
            clean)
                clean_docker
                shift
                ;;
            help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Show quick start guide
    show_quick_start
}

# Run main function
main "$@"