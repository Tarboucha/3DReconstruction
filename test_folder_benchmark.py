#!/usr/bin/env python3
"""
Test Script for Feature Matching Pipeline Benchmarking on Image Folder

This script demonstrates how to benchmark complete matching pipelines
(SIFT, ORB, LightGlue, etc.) on a folder of images using the unified 
benchmarking pipeline.

The benchmark compares complete matching pipelines:
- Traditional methods: Detection + Matching
- LightGlue: End-to-end matching

Usage:
    python test_folder_benchmark.py /path/to/image/folder
    python test_folder_benchmark.py /path/to/image/folder --methods SIFT ORB lightglue --max-images 20
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Import your feature detection system
try:
    import FeatureMatchingExtraction as fds  # Fixed import
except ImportError:
    print("Error: Cannot import feature_detection_system")
    print("Make sure the module is in your Python path")
    sys.exit(1)

# Check if required packages are available
try:
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Install with: pip install opencv-python numpy matplotlib")
    sys.exit(1)


def check_folder_exists(folder_path):
    """Check if the folder exists and contains images"""
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return False
    
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory")
        return False
    
    # Check for image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f"*{ext}"))
        image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"Warning: No image files found in '{folder_path}'")
        print(f"Looking for extensions: {image_extensions}")
        return False
    
    print(f"Found {len(image_files)} image files in '{folder_path}'")
    
    # Note about pairs
    if len(image_files) < 2:
        print("Warning: Need at least 2 images for matching benchmark")
        return False
    
    print(f"Will create {len(image_files)-1} consecutive image pairs for matching")
    return True


def test_method_availability(methods):
    """Test if the requested methods are available"""
    info = fds.get_version_info()
    available_methods = info['available_traditional'].copy()
    
    if info['deep_learning_detectors']:
        available_methods.extend(info['available_deep_learning'])
    
    if info['lightglue_matcher']:
        available_methods.append('lightglue')
    
    unavailable_methods = []
    for method in methods:
        if method not in available_methods and method.lower() != 'lightglue':
            unavailable_methods.append(method)
    
    if unavailable_methods:
        print(f"Warning: The following methods are not available: {unavailable_methods}")
        print(f"Available methods: {available_methods}")
        
        # Remove unavailable methods
        methods = [m for m in methods if m not in unavailable_methods]
        
        if not methods:
            print("Error: No valid methods remaining")
            return None
    
    return methods


def run_performance_benchmark(folder_path, methods, max_images=15, num_runs=3):
    """Run performance benchmark on folder - comparing matching pipelines"""
    print("\n" + "="*60)
    print("MATCHING PIPELINE PERFORMANCE BENCHMARK")
    print("="*60)
    print("Comparing complete matching pipelines:")
    print("  - Traditional: Feature Detection → Matching")
    print("  - LightGlue: End-to-end matching")
    print("-"*60)
    
    try:
        # Use the quick folder benchmark function
        results = fds.quick_folder_benchmark(
            folder_path=folder_path,
            methods=methods,
            benchmark_types=['performance'],
            max_images=max_images,
            resize_to=(640, 480),  # Consistent size for fair comparison
            num_runs=num_runs
        )
        
        return results
        
    except Exception as e:
        print(f"Error during performance benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_advanced_benchmark(folder_path, methods, max_images=10):
    """Run advanced benchmark with accuracy testing"""
    print("\n" + "="*60)
    print("ADVANCED BENCHMARK WITH ACCURACY TESTING")
    print("="*60)
    
    try:
        # Run both performance and accuracy benchmarks
        results = fds.quick_folder_benchmark(
            folder_path=folder_path,
            methods=methods,
            benchmark_types=['performance', 'accuracy'],  # Both benchmarks
            max_images=max_images,
            resize_to=(640, 480),
            num_runs=5
        )
        
        return results
        
    except Exception as e:
        print(f"Error during advanced benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_results(results):
    """Analyze and display detailed results for matching pipelines"""
    if not results or 'benchmarks' not in results:
        print("No valid results to analyze")
        return
    
    print("\n" + "="*60)
    print("DETAILED MATCHING PIPELINE RESULTS ANALYSIS")
    print("="*60)
    
    # Performance results
    performance_results = results['benchmarks'].get('performance')
    if performance_results:
        summary = performance_results['summary']
        
        # Performance comparison table
        print(f"\nPERFORMANCE METRICS:")
        print(f"{'Method':<12} {'Success':<10} {'Time(s)':<10} {'Matches':<10} {'Features':<10} {'FPS':<8} {'Match/s':<10}")
        print("-" * 80)
        
        method_data = []
        for method, stats in summary.items():
            if 'error' not in stats:
                success_rate = stats['success_rate'] * 100
                avg_time = stats['time_stats']['mean']
                avg_matches = stats.get('match_stats', {}).get('mean', 0)
                avg_features = stats.get('feature_stats', {}).get('mean', 0)
                fps = stats['avg_fps']
                matches_per_sec = stats.get('matches_per_second', 0)
                
                print(f"{method:<12} {success_rate:<10.1f}% {avg_time:<10.3f} {avg_matches:<10.1f} {avg_features:<10.1f} {fps:<8.1f} {matches_per_sec:<10.1f}")
                
                method_data.append({
                    'method': method,
                    'avg_time': avg_time,
                    'fps': fps,
                    'matches': avg_matches,
                    'features': avg_features,
                    'matches_per_sec': matches_per_sec
                })
            else:
                print(f"{method:<12} {'ERROR':<10} {'-':<10} {'-':<10} {'-':<10} {'-':<8} {'-':<10}")
        
        # Rankings
        if method_data:
            print(f"\nRANKINGS:")
            print("-" * 30)
            
            # Fastest matching pipeline
            fastest = sorted(method_data, key=lambda x: x['avg_time'])
            print(f"Fastest Pipeline (total time):")
            for i, data in enumerate(fastest[:3]):
                print(f"  {i+1}. {data['method']}: {data['avg_time']:.3f}s")
            
            # Most matches per second (efficiency)
            most_efficient = sorted(method_data, key=lambda x: x['matches_per_sec'], reverse=True)
            print(f"\nMost Efficient (matches/second):")
            for i, data in enumerate(most_efficient[:3]):
                print(f"  {i+1}. {data['method']}: {data['matches_per_sec']:.1f} matches/s")
            
            # Most matches found
            most_matches = sorted(method_data, key=lambda x: x['matches'], reverse=True)
            print(f"\nMost Matches Found:")
            for i, data in enumerate(most_matches[:3]):
                print(f"  {i+1}. {data['method']}: {data['matches']:.0f} matches")
        
        # Statistical comparisons
        statistical_comparisons = performance_results.get('statistical_comparisons', {})
        if statistical_comparisons:
            print(f"\nSTATISTICAL SIGNIFICANCE TESTS:")
            print("-" * 40)
            
            for comparison_name, comparison in statistical_comparisons.items():
                if 'statistical_test' in comparison:
                    test = comparison['statistical_test']
                    significance = "significant" if test['significant'] else "not significant"
                    p_value = test['p_value']
                    
                    effect_size_info = ""
                    if 'effect_size' in comparison:
                        effect_size_info = f" (effect: {comparison['effect_size']['magnitude']})"
                    
                    print(f"{comparison_name}: p={p_value:.4f} ({significance}){effect_size_info}")
    
    # Accuracy results (if available)
    accuracy_results = results['benchmarks'].get('accuracy')
    if accuracy_results:
        print(f"\n\nACCURACY METRICS:")
        print("-" * 40)
        
        summary = accuracy_results['summary']
        for method, stats in summary.items():
            if 'error' not in stats:
                print(f"\n{method}:")
                print(f"  Average Quality Score: {stats['avg_quality']:.3f}")
                print(f"  Average Matches: {stats['avg_matches']:.1f}")
                
                # Per transformation type
                if 'by_transformation' in stats:
                    print(f"  By Transformation:")
                    for transform, transform_stats in stats['by_transformation'].items():
                        print(f"    {transform}: quality={transform_stats['avg_quality']:.3f}, matches={transform_stats['avg_matches']:.1f}")


def save_results_summary(results, output_file="benchmark_summary.txt"):
    """Save a summary of matching pipeline results to a text file"""
    try:
        with open(output_file, 'w') as f:
            f.write("Feature Matching Pipeline Benchmark Summary\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Timestamp: {results.get('timestamp', 'Unknown')}\n")
            f.write(f"Image Source: {results['image_source_info']['type']}\n")
            
            if 'folder_path' in results['image_source_info']:
                f.write(f"Folder: {results['image_source_info']['folder_path']}\n")
            
            f.write(f"Methods: {', '.join(results['methods'])}\n")
            f.write(f"Benchmark Types: {', '.join(results['benchmark_types'])}\n\n")
            
            if 'benchmarks' in results and 'performance' in results['benchmarks']:
                performance = results['benchmarks']['performance']
                summary = performance['summary']
                
                f.write("Matching Pipeline Performance Results:\n")
                f.write("-" * 30 + "\n")
                
                for method, stats in summary.items():
                    if 'error' not in stats:
                        f.write(f"\n{method}:\n")
                        f.write(f"  Success Rate: {stats['success_rate']*100:.1f}%\n")
                        f.write(f"  Avg Pipeline Time: {stats['time_stats']['mean']:.3f}s\n")
                        if 'match_stats' in stats:
                            f.write(f"  Avg Matches: {stats['match_stats']['mean']:.1f}\n")
                        if 'feature_stats' in stats:
                            f.write(f"  Avg Features: {stats['feature_stats']['mean']:.0f}\n")
                        f.write(f"  FPS: {stats['avg_fps']:.1f}\n")
                        if 'matches_per_second' in stats:
                            f.write(f"  Matches/Second: {stats['matches_per_second']:.1f}\n")
                    else:
                        f.write(f"\n{method}: ERROR\n")
            
            if 'benchmarks' in results and 'accuracy' in results['benchmarks']:
                accuracy = results['benchmarks']['accuracy']
                summary = accuracy['summary']
                
                f.write("\n\nMatching Accuracy Results:\n")
                f.write("-" * 30 + "\n")
                
                for method, stats in summary.items():
                    if 'error' not in stats:
                        f.write(f"\n{method}:\n")
                        f.write(f"  Avg Quality Score: {stats['avg_quality']:.3f}\n")
                        f.write(f"  Avg Matches: {stats['avg_matches']:.1f}\n")
        
        print(f"\nSummary saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving summary: {e}")


def plot_results(results, save_plot=True):
    """Create plots of matching pipeline benchmark results"""
    try:
        import matplotlib.pyplot as plt
        
        performance_results = results['benchmarks'].get('performance')
        if not performance_results:
            return
        
        summary = performance_results['summary']
        
        # Extract data for plotting
        methods = []
        times = []
        matches = []
        features = []
        matches_per_sec = []
        
        for method, stats in summary.items():
            if 'error' not in stats:
                methods.append(method)
                times.append(stats['time_stats']['mean'])
                matches.append(stats.get('match_stats', {}).get('mean', 0))
                features.append(stats.get('feature_stats', {}).get('mean', 0))
                matches_per_sec.append(stats.get('matches_per_second', 0))
        
        if not methods:
            return
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        
        # Total pipeline time
        bars1 = ax1.bar(methods, times, color=colors)
        ax1.set_title('Total Matching Pipeline Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, time_val in zip(bars1, times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        # Matches found
        bars2 = ax2.bar(methods, matches, color=colors)
        ax2.set_title('Average Matches Found', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Matches')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, match_val in zip(bars2, matches):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(matches)*0.01, 
                    f'{match_val:.0f}', ha='center', va='bottom')
        
        # Matches per second (efficiency)
        bars3 = ax3.bar(methods, matches_per_sec, color=colors)
        ax3.set_title('Matching Efficiency', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Matches per Second')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, mps_val in zip(bars3, matches_per_sec):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(matches_per_sec)*0.01, 
                    f'{mps_val:.1f}', ha='center', va='bottom')
        
        # Time vs Matches scatter (efficiency visualization)
        ax4.scatter(times, matches, s=200, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
        for i, method in enumerate(methods):
            ax4.annotate(method, (times[i], matches[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=11)
        ax4.set_xlabel('Pipeline Time (s)')
        ax4.set_ylabel('Matches Found')
        ax4.set_title('Speed vs Quality Trade-off', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add ideal line (more matches in less time)
        ax4.axhline(y=np.mean(matches), color='gray', linestyle='--', alpha=0.5, label='Avg Matches')
        ax4.axvline(x=np.mean(times), color='gray', linestyle='--', alpha=0.5, label='Avg Time')
        ax4.legend(loc='upper right')
        
        plt.suptitle('Feature Matching Pipeline Benchmark Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('matching_benchmark_results.png', dpi=300, bbox_inches='tight')
            print("Plot saved as: matching_benchmark_results.png")
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")
    except Exception as e:
        print(f"Error creating plots: {e}")


def main():
    # parser = argparse.ArgumentParser(description="Benchmark feature matching pipelines on image folder")
    # parser.add_argument("folder", help="Path to folder containing images")
    # parser.add_argument("--methods", nargs="+", default=["SIFT", "ORB", "lightglue"], 
    #                    help="Methods to benchmark (default: SIFT ORB lightglue)")
    # parser.add_argument("--max-images", type=int, default=15,
    #                    help="Maximum number of images to process (default: 15)")
    # parser.add_argument("--num-runs", type=int, default=3,
    #                    help="Number of runs per image pair for statistics (default: 3)")
    # parser.add_argument("--no-plot", action="store_true",
    #                    help="Don't generate plots")
    # parser.add_argument("--advanced", action="store_true",
    #                    help="Run advanced benchmark with accuracy testing")
    
    # args = parser.parse_args()

    #Alternative: hardcoded values for testing
    methods = ['SIFT', 'ORB', 'Harris', 'AKAZE', 'lightglue']
    folder = 'E://project//3Dreconstruction//images//statue_of_liberty_images'
    max_images = 50
    num_runs = 3
    no_plot = False
    advanced = True
    
    # Use argparse values
    # methods = args.methods
    # folder = args.folder
    # max_images = args.max_images
    # num_runs = args.num_runs
    # no_plot = args.no_plot
    # advanced = args.advanced

    # Print system info
    print("\nFeature Matching Pipeline Benchmark Test")
    print("="*60)
    print("This benchmark compares complete matching pipelines:")
    print("  • Traditional methods: Detection → Matching")
    print("  • LightGlue: End-to-end matching")
    print("\nSystem Capabilities:")
    print("-"*60)
    fds.print_capabilities()
    
    # Check folder
    if not check_folder_exists(folder):
        sys.exit(1)
    
    # Check method availability
    methods = test_method_availability(methods)
    if not methods:
        sys.exit(1)
    
    print(f"\nBenchmark Configuration:")
    print(f"  Methods: {methods}")
    print(f"  Max images: {max_images}")
    print(f"  Runs per pair: {num_runs}")
    print(f"  Mode: {'Advanced (with accuracy)' if advanced else 'Performance only'}")
    
    # Run benchmarks
    start_time = time.time()
    
    if advanced:
        results = run_advanced_benchmark(folder, methods, max_images)
    else:
        results = run_performance_benchmark(folder, methods, max_images, num_runs)
    
    total_time = time.time() - start_time
    
    if results:
        # Analyze results
        analyze_results(results)
        
        # Save summary
        save_results_summary(results)
        
        # Create plots
        if not no_plot:
            plot_results(results,save_plot=True)
        
        print(f"\n" + "="*60)
        print(f"Benchmark completed successfully!")
        print(f"Total benchmark time: {total_time:.2f} seconds")
        
        if 'config' in results:
            output_dir = results['config'].output_dir
            print(f"Results saved in: {output_dir}/")
        
    else:
        print("Benchmark failed - no results to display")
        sys.exit(1)


if __name__ == "__main__":
    main()