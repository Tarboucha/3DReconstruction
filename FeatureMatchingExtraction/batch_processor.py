"""
Batch processing checkpoint manager

Handles progress tracking for resumable batch processing.
"""

import json
import time
from pathlib import Path
from typing import Set, Dict, Any
import pickle

class BatchProcessor:
    """
    Manages batch processing with automatic batch file creation
    """
    def __init__(self, output_dir: Path, batch_size: int = 10):
        self.output_dir = Path(output_dir)
        self.progress_file = self.output_dir / 'progress.json'
        self.batch_size = batch_size
        
        self.completed_pairs = self._load_progress()
        self.metadata = self._load_metadata()
        
        if not self.metadata:
            self.metadata = {
                'batch_size': batch_size,
                'started': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # Batch accumulator
        self.current_batch = []
        self.current_batch_number = 0
    
    def _load_progress(self) -> Set[str]:
        if not self.progress_file.exists():
            return set()
        
        try:
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                return set(data.get('completed_pairs', []))
        except:
            return set()
    
    def _load_metadata(self) -> Dict[str, Any]:
        if not self.progress_file.exists():
            return {}
        
        try:
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                return data.get('metadata', {})
        except:
            return {}
    
    def add_result(self, pair_id: str, result: Any):
        """Add result to current batch"""
        self.current_batch.append((pair_id, result))
        
        if len(self.current_batch) >= self.batch_size:
            self._save_and_clear_batch()


    def _save_and_clear_batch(self):
        """Save current batch to file and clear from memory"""
        if not self.current_batch:
            return
        
        self.current_batch_number += 1
        batch_file = self.output_dir / 'matching_results' / f'matching_results_batch_{self.current_batch_number:03d}.pkl'
        batch_file.parent.mkdir(parents=True, exist_ok=True)
        
        batch_data = {
            'results': {},
            'batch_stats': {
                'batch_number': self.current_batch_number,
                'num_pairs': len(self.current_batch),
                'batch_processing_time': 0
            }
        }
        
        for pair_id, recon_data in self.current_batch:
            pair_key = (recon_data.image_pair_info.image1_id, recon_data.image_pair_info.image2_id)
            
            total_matches = 0
            best_method = None
            best_num_matches = 0
            for method_name, method_data in recon_data.methods.items():
                num_matches = method_data.num_matches
                total_matches += num_matches
                
                if num_matches > best_num_matches:
                    best_num_matches = num_matches
                    best_method = method_name

            batch_data['results'][pair_key] = {
                'total_matches': total_matches,
                'best_method': best_method,
                'best_num_matches': best_num_matches,
                'num_methods': len(recon_data.methods),
                'recon_data': recon_data
            }
        
        with open(batch_file, 'wb') as f:
            pickle.dump(batch_data, f)
        
        # ✅ Batch update progress for all pairs
        for pair_id, _ in self.current_batch:
            self.completed_pairs.add(pair_id)
        
        # Write progress file once for the entire batch
        progress_data = {
            'completed_pairs': sorted(list(self.completed_pairs)),
            'total_completed': len(self.completed_pairs),
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metadata': self.metadata
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        self.current_batch.clear()


    def finalize(self):
        """Save any remaining results in current batch"""
        if self.current_batch:
            self._save_and_clear_batch()
    
    def save_progress(self, pair_id: str):
        """Save progress checkpoint"""
        self.completed_pairs.add(pair_id)
        
        progress_data = {
            'completed_pairs': sorted(list(self.completed_pairs)),
            'total_completed': len(self.completed_pairs),
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metadata': self.metadata
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def is_completed(self, pair_id: str) -> bool:
        return pair_id in self.completed_pairs
    
    def reset(self):
        self.completed_pairs.clear()
        self.current_batch.clear()
        self.current_batch_number = 0
        
        if self.progress_file.exists():
            self.progress_file.unlink()
        
        self.metadata = {
            'batch_size': self.batch_size,
            'started': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_progress_info(self) -> Dict[str, Any]:
        """
        Get current progress information
        
        Returns:
            Dictionary with progress statistics
        """
        return {
            'completed_pairs': len(self.completed_pairs),
            'pairs_list': sorted(list(self.completed_pairs)),
            'last_updated': self.metadata.get('last_updated', 'Never'),
            'batch_size': self.metadata.get('batch_size', self.batch_size),
            'started': self.metadata.get('started', 'Unknown')
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"BatchProcessor("
            f"completed={len(self.completed_pairs)}, "
            f"output_dir={self.output_dir})"
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_progress(output_dir: Path) -> Set[str]:
    """
    Load completed pairs from progress file without creating processor
    
    Args:
        output_dir: Directory containing progress.json
    
    Returns:
        Set of completed pair IDs
        
    Example:
        >>> completed = load_progress(Path('./output'))
        >>> print(f"Already completed: {len(completed)} pairs")
    """
    progress_file = Path(output_dir) / 'progress.json'
    
    if not progress_file.exists():
        return set()
    
    try:
        with open(progress_file, 'r') as f:
            data = json.load(f)
            return set(data.get('completed_pairs', []))
    except Exception:
        return set()


def delete_progress(output_dir: Path):
    """
    Delete progress file to force reprocessing
    
    Args:
        output_dir: Directory containing progress.json
        
    Example:
        >>> delete_progress(Path('./output'))
        >>> # Now running pipeline will reprocess everything
    """
    progress_file = Path(output_dir) / 'progress.json'
    
    if progress_file.exists():
        try:
            progress_file.unlink()
            print(f"✓ Deleted progress file: {progress_file}")
        except Exception as e:
            print(f"⚠️  Could not delete progress file: {e}")
    else:
        print(f"No progress file found at: {progress_file}")


def get_remaining_pairs(
    total_pairs: int,
    output_dir: Path,
    prefix: str = 'pair'
) -> list:
    """
    Get list of pairs that still need to be processed
    
    Args:
        total_pairs: Total number of pairs
        output_dir: Directory containing progress.json
        prefix: Pair ID prefix (default: 'pair')
    
    Returns:
        List of pair IDs that haven't been processed yet
        
    Example:
        >>> remaining = get_remaining_pairs(100, Path('./output'))
        >>> print(f"Need to process: {remaining}")
        ['pair_042', 'pair_043', 'pair_044']
    """
    completed = load_progress(output_dir)
    
    all_pairs = [f"{prefix}_{i:03d}" for i in range(total_pairs)]
    remaining = [p for p in all_pairs if p not in completed]
    
    return remaining


