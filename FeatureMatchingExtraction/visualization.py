"""
Visualization Functions for Feature Detection System

Complete visualization module for the new API.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Dict, Any
from pathlib import Path
from .result_types import MatchingResult
from .result_converters import VisualizationData

# =============================================================================
# NEW API VISUALIZATION FUNCTIONS
# =============================================================================

def plot_visualization_data(viz_data,
                           method: Optional[str] = None,
                           figsize: tuple = (15, 8),
                           show_scores: bool = True,
                           title_override: Optional[str] = None):
    """
    Plot matches from VisualizationData using matplotlib
    
    Args:
        viz_data: VisualizationData object
        method: Show only this method (None = show all methods)
        figsize: Figure size (width, height)
        show_scores: Whether to show match scores
        title_override: Override default title
    """
    
    # Filter by method if specified
    if method is not None:
        if method not in viz_data.method_info:
            print(f"âŒ Method '{method}' not found in visualization data")
            print(f"   Available: {list(viz_data.method_info.keys())}")
            return
        viz_data = viz_data.filter_by_method(method)
    
    # Get images
    img1 = viz_data.image_pair_info.image1
    img2 = viz_data.image_pair_info.image2
    
    if img1 is None or img2 is None:
        print("âŒ Images not stored in VisualizationData. Cannot plot.")
        print("   Try: result.to_visualization(include_images=True)")
        return
    
    # Convert to RGB if needed
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    
    # Create side-by-side image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    
    combined = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    combined[:h1, :w1] = img1
    combined[:h2, w1:w1+w2] = img2
    
    # Setup plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(combined)
    
    # Draw keypoints
    for kp in viz_data.keypoints1:
        circle = plt.Circle((kp.pt[0], kp.pt[1]), 3, 
                          color='yellow', fill=False, linewidth=1)
        ax.add_patch(circle)
    
    for kp in viz_data.keypoints2:
        circle = plt.Circle((kp.pt[0] + w1, kp.pt[1]), 3, 
                          color='yellow', fill=False, linewidth=1)
        ax.add_patch(circle)
    
    # Draw matches
    for match in viz_data.matches:
        kp1 = viz_data.keypoints1[match.query_idx]
        kp2 = viz_data.keypoints2[match.train_idx]
        
        # Convert RGB color to matplotlib format [0-1]
        color = tuple(c/255.0 for c in match.color)
        
        x1, y1 = kp1.pt
        x2, y2 = kp2.pt[0] + w1, kp2.pt[1]
        
        ax.plot([x1, x2], [y1, y2], 
               color=color, linewidth=match.line_width, alpha=0.7)
    
    # Title
    if title_override:
        title = title_override
    else:
        title = viz_data.title
        if viz_data.subtitle:
            title += f"\n{viz_data.subtitle}"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend for methods
    if len(viz_data.method_info) > 1:
        from matplotlib.patches import Patch
        legend_elements = []
        for method_name, info in viz_data.method_info.items():
            color = tuple(c/255.0 for c in info['color'])
            label = f"{info['label']} ({info['num_matches']})"
            legend_elements.append(Patch(facecolor=color, label=label))
        
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()


def plot_method_comparison(viz_data,
                          figsize: tuple = (20, 12),
                          max_cols: int = 3):
    """
    Plot side-by-side comparison of all methods
    
    Args:
        viz_data: VisualizationData with multiple methods
        figsize: Figure size
        max_cols: Maximum columns in grid
    """
    methods = list(viz_data.method_info.keys())
    n_methods = len(methods)
    
    if n_methods == 0:
        print("âŒ No methods to compare")
        return
    
    if n_methods == 1:
        print("âš ï¸  Only one method, using regular plot")
        plot_visualization_data(viz_data, figsize=figsize)
        return
    
    # Calculate grid layout
    n_cols = min(n_methods, max_cols)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_methods == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_methods > 1 else [axes]
    
    # Plot each method
    for i, method in enumerate(methods):
        viz_method = viz_data.filter_by_method(method)
        
        # Get images
        img1 = viz_method.image_pair_info.image1_array
        img2 = viz_method.image_pair_info.image2_array
        
        if img1 is None or img2 is None:
            continue
        
        # Create side-by-side
        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h = max(h1, h2)
        
        combined = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
        combined[:h1, :w1] = img1
        combined[:h2, w1:w1+w2] = img2
        
        axes[i].imshow(combined)
        
        # Draw matches
        for match in viz_method.matches:
            kp1 = viz_method.keypoints1[match.query_idx]
            kp2 = viz_method.keypoints2[match.train_idx]
            
            color = tuple(c/255.0 for c in match.color)
            
            x1, y1 = kp1.pt
            x2, y2 = kp2.pt[0] + w1, kp2.pt[1]
            
            axes[i].plot([x1, x2], [y1, y2], 
                       color=color, linewidth=1, alpha=0.7)
        
        # Title
        info = viz_method.method_info[method]
        axes[i].set_title(f"{info['label']}\n{info['num_matches']} matches",
                         fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_methods, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(viz_data.title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_matches_quick(result, method: Optional[str] = None, **kwargs):
    """
    Quick visualization helper - works with both old and new result types
    
    Args:
        result: Can be MatchingResult, VisualizationData, or old dict
        method: Optional method filter
        **kwargs: Additional plotting parameters
    """
    from result_types import MatchingResult
    from result_converters import VisualizationData
    
    if isinstance(result, MatchingResult):
        # New result type
        viz = result.to_visualization(include_images=True)
        plot_visualization_data(viz, method=method, **kwargs)
        
    elif isinstance(result, VisualizationData):
        # Already visualization data
        plot_visualization_data(result, method=method, **kwargs)
        
    elif isinstance(result, dict):
        # Old dict format - use legacy visualization
        print("âš ï¸  Using legacy visualization format")
        
        visualize_matches_with_scores(
            result.get('image_pair_info').image1_array if 'image_pair_info' in result else None,
            result.get('image_pair_info').image2_array if 'image_pair_info' in result else None,
            result['features1'].keypoints,
            result['features2'].keypoints,
            result['match_data'],
            title=kwargs.get('title', 'Feature Matches')
        )
    else:
        print(f"âŒ Unknown result type: {type(result)}")


def show_matches(result, method: Optional[str] = None):
    """
    Ultra-simple visualization - just call this!
    
    Usage:
        result = pipeline.match(img1, img2)
        show_matches(result)
        
        # Or show specific method
        show_matches(result, method='SIFT')
    """
    visualize_matches_quick(result, method=method)


# =============================================================================
# LEGACY VISUALIZATION FUNCTION (Keep for backward compatibility)
# =============================================================================

def visualize_matches_with_scores(img1, img2, keypoints1, keypoints2, 
                                 match_data, title="Feature Matches",
                                 max_matches=None, figsize=(15, 8)):
    """
    Legacy visualization function - kept for backward compatibility
    
    Args:
        img1: First image
        img2: Second image
        keypoints1: Keypoints from image 1
        keypoints2: Keypoints from image 2
        match_data: MatchData object
        title: Plot title
        max_matches: Maximum matches to display
        figsize: Figure size
    """
    # Get matches
    matches = match_data.get_best_matches()
    
    if max_matches is not None and len(matches) > max_matches:
        matches = matches[:max_matches]
    
    if len(matches) == 0:
        print("âš ï¸  No matches to visualize")
        return
    
    # Convert to RGB if grayscale
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    
    # Create side-by-side image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    
    combined = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    combined[:h1, :w1] = img1
    combined[:h2, w1:w1+w2] = img2
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(combined)
    
    # Draw matches
    for match in matches:
        kp1 = keypoints1[match.queryIdx]
        kp2 = keypoints2[match.trainIdx]
        
        x1, y1 = kp1.pt
        x2, y2 = kp2.pt[0] + w1, kp2.pt[1]
        
        # Color based on score
        if hasattr(match, 'score'):
            # Normalize score to color
            color = plt.cm.viridis(match.score)
        else:
            color = 'green'
        
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=1, alpha=0.7)
    
    ax.set_title(f"{title}\n{len(matches)} matches", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_visualization(result, filepath: str, method: Optional[str] = None):
    """
    Save visualization to file
    
    Args:
        result: MatchingResult or VisualizationData
        filepath: Output file path (e.g., 'output.png')
        method: Optional method to visualize
    """

    
    # Convert to VisualizationData if needed
    if isinstance(result, MatchingResult):
        viz_data = result.to_visualization(include_images=True)
    elif isinstance(result, VisualizationData):
        viz_data = result
    else:
        print(f"âŒ Cannot save visualization for type: {type(result)}")
        return
    
    # Filter by method if specified
    if method is not None:
        viz_data = viz_data.filter_by_method(method)
    
    # Create figure (don't show)
    plt.ioff()  # Turn off interactive mode
    
    # Create visualization
    img1 = viz_data.image_pair_info.image1
    img2 = viz_data.image_pair_info.image2
    
    if img1 is None or img2 is None:
        print("âŒ Images not available")
        return
    
    # Convert to RGB if needed
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    
    # Create side-by-side
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    
    combined = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    combined[:h1, :w1] = img1
    combined[:h2, w1:w1+w2] = img2
    
    # Draw matches on combined image
    for match in viz_data.matches:
        kp1 = viz_data.keypoints1[match.query_idx]
        kp2 = viz_data.keypoints2[match.train_idx]
        
        pt1 = (int(kp1.pt[0]), int(kp1.pt[1]))
        pt2 = (int(kp2.pt[0] + w1), int(kp2.pt[1]))
        
        cv2.line(combined, pt1, pt2, match.color, 1)
        cv2.circle(combined, pt1, 3, (255, 255, 0), 1)
        cv2.circle(combined, pt2, 3, (255, 255, 0), 1)
    
    # Save
    cv2.imwrite(filepath, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"âœ… Visualization saved to: {filepath}")
    
    plt.ion()  # Turn interactive mode back on


def visualize_keypoints_only(image, keypoints, title="Keypoints", figsize=(12, 8)):
    """
    Visualize keypoints on a single image
    
    Args:
        image: Input image
        keypoints: List of cv2.KeyPoint
        title: Plot title
        figsize: Figure size
    """
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        img_display = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        img_display = image.copy()
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img_display)
    
    # Draw keypoints
    for kp in keypoints:
        circle = plt.Circle((kp.pt[0], kp.pt[1]), kp.size/2, 
                          color='red', fill=False, linewidth=2)
        ax.add_patch(circle)
        
        # Draw orientation
        x, y = kp.pt
        angle = np.deg2rad(kp.angle)
        dx = np.cos(angle) * kp.size
        dy = np.sin(angle) * kp.size
        ax.arrow(x, y, dx, dy, head_width=3, head_length=3, fc='yellow', ec='yellow')
    
    ax.set_title(f"{title}\n{len(keypoints)} keypoints", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'plot_visualization_data',
    'plot_method_comparison',
    'visualize_matches_quick',
    'show_matches',
    'visualize_matches_with_scores',
    'save_visualization',
    'visualize_keypoints_only',
]