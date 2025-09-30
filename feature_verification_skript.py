#!/usr/bin/env python3
"""
Setup Verification Script for Feature Detection Pipeline

This script checks if all dependencies are properly installed and 
the pipeline modules are working correctly.
"""

import sys
import os
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def check_python_version():
    """Check Python version"""
    print("🐍 Python Version Check")
    print("-" * 25)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7+ required")
        return False
    else:
        print("✅ Python version OK")
        return True


def check_core_dependencies():
    """Check core dependencies"""
    print("\n📦 Core Dependencies Check")
    print("-" * 30)
    
    dependencies = {
        'numpy': 'numpy',
        'opencv-python': 'cv2',
        'matplotlib': 'matplotlib',
        'pandas': 'pandas',
        'psutil': 'psutil',
        'scipy': 'scipy'
    }
    
    all_good = True
    
    for package_name, import_name in dependencies.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package_name}: {version}")
        except ImportError:
            print(f"❌ {package_name}: NOT INSTALLED")
            all_good = False
        except Exception as e:
            print(f"⚠️  {package_name}: ERROR - {str(e)}")
            all_good = False
    
    return all_good


def check_optional_dependencies():
    """Check optional dependencies"""
    print("\n🔬 Optional Dependencies Check")
    print("-" * 35)
    
    optional_deps = {
        'PyTorch': 'torch',
        'Torchvision': 'torchvision',
    }
    
    results = {}
    
    for package_name, import_name in optional_deps.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package_name}: {version}")
            results[package_name.lower()] = True
            
            # Check CUDA for PyTorch
            if import_name == 'torch':
                cuda_available = module.cuda.is_available()
                device_count = module.cuda.device_count() if cuda_available else 0
                print(f"   CUDA available: {cuda_available}")
                if cuda_available:
                    print(f"   CUDA devices: {device_count}")
                    
        except ImportError:
            print(f"⚠️  {package_name}: NOT INSTALLED")
            results[package_name.lower()] = False
        except Exception as e:
            print(f"❌ {package_name}: ERROR - {str(e)}")
            results[package_name.lower()] = False
    
    return results


def check_lightglue():
    """Check LightGlue availability"""
    print("\n🔦 LightGlue Check")
    print("-" * 20)
    
    # Check if lightglue folder exists
    lightglue_path = Path("lightglue")
    if lightglue_path.exists():
        print(f"✅ LightGlue folder found: {lightglue_path}")
    else:
        print("⚠️  LightGlue folder not found")
        return False
    
    # Try importing LightGlue
    try:
        sys.path.insert(0, str(lightglue_path))
        from FeatureMatchingExtraction.LightGlue.lightglue import LightGlue, SuperPoint
        print("✅ LightGlue imports successful")
        
        # Try creating a simple model
        try:
            extractor = SuperPoint(max_num_keypoints=100)
            matcher = LightGlue(features='superpoint')
            print("✅ LightGlue models can be created")
            return True
        except Exception as e:
            print(f"⚠️  LightGlue model creation failed: {str(e)}")
            return False
            
    except ImportError as e:
        print(f"❌ LightGlue import failed: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ LightGlue error: {str(e)}")
        return False



def check_deep_learning_modules():
    """Check deep learning specific modules"""
    print("\n🧠 Deep Learning Modules Check")
    print("-" * 35)
    
    try:
        from FeatureMatchingExtraction.deep_learning_detectors import create_deep_learning_detector
        print("✅ deep_learning_detectors")
        
        # Test if we can create detectors
        try:
            import torch
            detector = create_deep_learning_detector('SuperPoint', max_features=100)
            print("✅ SuperPoint detector creation")
        except Exception as e:
            print(f"⚠️  SuperPoint detector: {str(e)}")
            
    except ImportError as e:
        print(f"⚠️  deep_learning_detectors: {str(e)}")
    except Exception as e:
        print(f"❌ deep_learning_detectors: {str(e)}")


def test_basic_functionality():
    """Test basic functionality"""
    print("\n🧪 Basic Functionality Test")
    print("-" * 30)
    
    try:
        # Test traditional detector
        from FeatureMatchingExtraction.traditional_detectors import create_traditional_detector
        detector = create_traditional_detector('SIFT', max_features=100)
        print("✅ Traditional detector creation")
        
        # Test synthetic image generation
        from FeatureMatchingExtraction.benchmarking import SyntheticImageGenerator
        generator = SyntheticImageGenerator()
        test_img = generator.create_realistic_test_image(240, 320, 'medium', 'medium')
        print("✅ Synthetic image generation")
        
        # Test feature detection
        features = detector.detect(test_img)
        print(f"✅ Feature detection: {len(features.keypoints)} features")
        
        # Test pipeline creation
        from FeatureMatchingExtraction.pipeline import create_pipeline
        pipeline = create_pipeline('fast')
        print("✅ Pipeline creation")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {str(e)}")
        return False


def test_advanced_functionality():
    """Test advanced functionality if dependencies are available"""
    print("\n🚀 Advanced Functionality Test")
    print("-" * 35)
    
    try:
        # Test if we can run a quick benchmark
        from FeatureMatchingExtraction.benchmark_pipeline import quick_synthetic_benchmark
        
        print("Running quick benchmark...")
        results = quick_synthetic_benchmark(
            methods=['SIFT'],
            sizes=[(240, 320)],
            num_runs=1
        )
        
        if 'benchmarks' in results and 'performance' in results['benchmarks']:
            print("✅ Benchmarking system working")
            return True
        else:
            print("⚠️  Benchmarking completed but results incomplete")
            return False
            
    except Exception as e:
        print(f"❌ Advanced functionality test failed: {str(e)}")
        return False


def check_file_structure():
    """Check if all expected files are present"""
    print("\n📁 File Structure Check")
    print("-" * 25)
    
    expected_files = [
        'core_data_structures.py',
        'traditional_detectors.py',
        'feature_matchers.py',
        'pipeline.py',
        'benchmarking.py',
        'benchmark_pipeline.py',
        'config.py',
        'utils.py',
        'base_classes.py',
        'requirements.txt'
    ]
    
    optional_files = [
        'deep_learning_detectors.py',
        'setup.py',
        'Dockerfile'
    ]
    
    all_present = True
    
    for file in expected_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            all_present = False
    
    print("\nOptional files:")
    for file in optional_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"⚠️  {file}")
    
    # Check lightglue folder
    if Path("lightglue").exists():
        print("✅ lightglue/ (folder)")
    else:
        print("⚠️  lightglue/ (folder)")
    
    return all_present


def generate_install_recommendations(results):
    """Generate installation recommendations based on check results"""
    print("\n💡 Installation Recommendations")
    print("-" * 35)
    
    if not results.get('core_dependencies', True):
        print("Core dependencies missing. Install with:")
        print("  pip install opencv-python numpy matplotlib pandas psutil scipy")
    
    if not results.get('pytorch', False):
        print("\nFor deep learning features, install PyTorch:")
        print("  pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu")
        print("  # Or for CUDA: pip install torch torchvision")
    
    if not results.get('lightglue', False):
        print("\nFor LightGlue support:")
        print("  1. Make sure PyTorch is installed")
        print("  2. Clone LightGlue: git clone https://github.com/cvg/LightGlue.git lightglue")
        print("  3. Or install: pip install lightglue")
    
    if not results.get('modules', True):
        print("\nProject modules missing. Make sure you're in the correct directory.")
    
    print("\nFor full installation:")
    print("  pip install -r requirements.txt")


def main():
    """Run complete setup verification"""
    print("🔧 Feature Detection Pipeline Setup Verification")
    print("=" * 60)
    
    results = {}
    
    # Run all checks
    results['python'] = check_python_version()
    results['core_dependencies'] = check_core_dependencies()
    optional_results = check_optional_dependencies()
    results.update(optional_results)
    results['lightglue'] = check_lightglue()
    check_deep_learning_modules()
    results['files'] = check_file_structure()
    results['basic_test'] = test_basic_functionality()
    
    if results['basic_test']:
        results['advanced_test'] = test_advanced_functionality()
    
    # Summary
    print("\n📊 Setup Summary")
    print("-" * 20)
    
    essential_checks = ['python', 'core_dependencies', 'modules', 'files', 'basic_test']
    essential_passed = all(results.get(check, False) for check in essential_checks)
    
    optional_checks = ['pytorch', 'lightglue', 'advanced_test']
    optional_passed = sum(results.get(check, False) for check in optional_checks)
    
    print(f"Essential components: {'✅ PASSED' if essential_passed else '❌ FAILED'}")
    print(f"Optional components: {optional_passed}/{len(optional_checks)} available")
    
    if essential_passed:
        print("\n🎉 Basic setup is complete! You can use:")
        print("   - Traditional detectors (SIFT, ORB, AKAZE, BRISK)")
        print("   - Feature matching and pipelines")
        print("   - Benchmarking system")
        
        if results.get('pytorch', False):
            print("   - Deep learning detectors (SuperPoint, DISK, ALIKED)")
        
        if results.get('lightglue', False):
            print("   - LightGlue end-to-end matching")
        
        print("\nNext steps:")
        print("   - Run: python test_pipeline.py --quick")
        print("   - Run: python demo_pipeline.py")
        
    else:
        print("\n⚠️  Setup incomplete. See recommendations below.")
        generate_install_recommendations(results)
    
    return essential_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)