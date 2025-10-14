"""
Batch processing checkpoint manager

Handles progress tracking for resumable batch processing.
"""

import json
import time
from pathlib import Path
from typing import Set, Dict, Any


class BatchProcessor:
    """
    Manages batch processing checkpoints for fault-tolerant pipeline execution
    
    Tracks which image pairs have been successfully processed to enable:
    - Resume after crash/interruption
    - Skip already-processed pairs
    - Progress monitoring
    
    The checkpoint file (progress.json) contains:
    {
        "completed_pairs": ["pair_000", "pair_001", ...],
        "total_completed": 42,
        "last_updated": "2025-01-10 14:30:15",
        "metadata": {
            "batch_size": 10,
            "started": "2025-01-10 14:00:00"
        }
    }
    
    Usage:
        >>> processor = BatchProcessor(output_dir)
        >>> 
        >>> # Check if pair was already processed
        >>> if not processor.is_completed('pair_042'):
        ...     result = pipeline.match(img1, img2)
        ...     processor.save_progress('pair_042')
        >>> 
        >>> # Reset to start fresh
        >>> processor.reset()
    """
    
    def __init__(self, output_dir: Path, batch_size: int = 10):
        """
        Initialize batch processor
        
        Args:
            output_dir: Directory where progress.json will be stored
            batch_size: Batch size (stored in metadata for reference)
        """
        self.output_dir = Path(output_dir)
        self.progress_file = self.output_dir / 'progress.json'
        self.batch_size = batch_size
        
        # Load existing progress
        self.completed_pairs = self._load_progress()
        self.metadata = self._load_metadata()
        
        # If starting fresh, initialize metadata
        if not self.metadata:
            self.metadata = {
                'batch_size': batch_size,
                'started': time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def _load_progress(self) -> Set[str]:
        """
        Load which pairs have been completed from progress file
        
        Returns:
            Set of completed pair IDs (e.g., {"pair_000", "pair_001"})
        """
        if not self.progress_file.exists():
            return set()
        
        try:
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                return set(data.get('completed_pairs', []))
        except json.JSONDecodeError as e:
            print(f"⚠️  Warning: Corrupted progress file, starting fresh: {e}")
            return set()
        except Exception as e:
            print(f"⚠️  Warning: Could not load progress file: {e}")
            return set()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from progress file"""
        if not self.progress_file.exists():
            return {}
        
        try:
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                return data.get('metadata', {})
        except Exception:
            return {}
    
    def save_progress(self, pair_id: str):
        """
        Save progress after completing a pair
        
        This is called immediately after each pair is successfully processed
        and saved to enable resume from exact point of failure.
        
        Args:
            pair_id: Pair identifier (e.g., "pair_042")
        """
        # Add to completed set
        self.completed_pairs.add(pair_id)
        
        # Prepare data to save
        progress_data = {
            'completed_pairs': sorted(list(self.completed_pairs)),
            'total_completed': len(self.completed_pairs),
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metadata': self.metadata
        }
        
        # Save to file
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Could not save progress: {e}")
            # Don't raise - continue processing even if checkpoint fails
    
    def is_completed(self, pair_id: str) -> bool:
        """
        Check if a pair has already been processed
        
        Args:
            pair_id: Pair identifier (e.g., "pair_042")
        
        Returns:
            True if pair was already processed and saved
        """
        return pair_id in self.completed_pairs
    
    def reset(self):
        """
        Reset progress (start from scratch)
        
        Deletes the checkpoint file and clears completed pairs.
        Use this to force reprocessing of all pairs.
        """
        self.completed_pairs.clear()
        
        if self.progress_file.exists():
            try:
                self.progress_file.unlink()
                print(f"✓ Reset progress: Deleted {self.progress_file}")
            except Exception as e:
                print(f"⚠️  Warning: Could not delete progress file: {e}")
        
        # Reset metadata
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


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test the BatchProcessor"""
    import tempfile
    import shutil
    
    print("Testing BatchProcessor...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test 1: Create processor
        print("\n1. Creating processor...")
        processor = BatchProcessor(tmpdir, batch_size=5)
        assert len(processor.completed_pairs) == 0
        print("✓ Empty processor created")
        
        # Test 2: Save progress
        print("\n2. Saving progress...")
        processor.save_progress('pair_000')
        processor.save_progress('pair_001')
        processor.save_progress('pair_002')
        assert len(processor.completed_pairs) == 3
        print("✓ Saved 3 pairs")
        
        # Test 3: Check completion
        print("\n3. Checking completion...")
        assert processor.is_completed('pair_000')
        assert processor.is_completed('pair_001')
        assert not processor.is_completed('pair_999')
        print("✓ Completion check works")
        
        # Test 4: Load existing progress
        print("\n4. Testing resume...")
        processor2 = BatchProcessor(tmpdir)
        assert len(processor2.completed_pairs) == 3
        assert processor2.is_completed('pair_001')
        print("✓ Resume works")
        
        # Test 5: Reset
        print("\n5. Testing reset...")
        processor2.reset()
        assert len(processor2.completed_pairs) == 0
        print("✓ Reset works")
        
        # Test 6: Utility functions
        print("\n6. Testing utility functions...")
        processor.save_progress('pair_010')
        processor.save_progress('pair_011')
        
        completed = load_progress(tmpdir)
        assert len(completed) == 2
        
        remaining = get_remaining_pairs(15, tmpdir)
        assert len(remaining) == 13
        assert 'pair_010' not in remaining
        print("✓ Utility functions work")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED")
    print("="*70)