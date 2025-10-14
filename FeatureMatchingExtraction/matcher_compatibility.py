"""
Matcher compatibility configuration system.
Loads compatibility rules from matcher_compatibility.json file.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class MatcherCompatibilityManager:
    """Manages detector-matcher compatibility from JSON configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize compatibility manager
        
        Args:
            config_path: Path to compatibility JSON file. 
                        If None, looks for matcher_compatibility.json in package directory.
        """
        if config_path is None:
            # Look for config in package directory
            config_path = Path(__file__).parent / "matcher_compatibility.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Compatibility config not found at {self.config_path}\n"
                f"Please ensure matcher_compatibility.json exists in the package directory."
            )
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            print(f"Loaded compatibility config v{config.get('version', 'unknown')}")
            return config
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading compatibility config: {e}")
    
    def is_compatible(self, detector: str, matcher: str) -> bool:
        """
        Check if a detector-matcher combination is compatible
        
        Args:
            detector: Detector name (e.g., 'SIFT', 'DISK')
            matcher: Matcher name (e.g., 'flann', 'lightglue')
        
        Returns:
            True if compatible, False otherwise
        """
        detector_info = self.config.get('detectors', {}).get(detector)
        
        if not detector_info:
            print(f" Unknown detector: {detector}")
            return False
        
        compatible_matchers = detector_info.get('compatible_matchers', [])
        return matcher.lower() in [m.lower() for m in compatible_matchers]
    
    def get_default_matcher(self, detector: str) -> Optional[str]:
        """Get the default matcher for a detector"""
        detector_info = self.config.get('detectors', {}).get(detector)
        
        if detector_info:
            return detector_info.get('default_matcher')
        
        return None
    
    def get_recommended_matcher(self, detector: str) -> Optional[str]:
        """Get the recommended matcher for a detector"""
        detector_info = self.config.get('detectors', {}).get(detector)
        
        if detector_info:
            return detector_info.get('recommended_matcher')
        
        return None
    
    def get_compatible_matchers(self, detector: str) -> List[str]:
        """Get all compatible matchers for a detector"""
        detector_info = self.config.get('detectors', {}).get(detector)
        
        if detector_info:
            return detector_info.get('compatible_matchers', [])
        
        return []
    
    def get_matcher_params(self, detector: str, matcher: str) -> Dict:
        """Get specific matcher parameters for a detector"""
        detector_info = self.config.get('detectors', {}).get(detector)
        
        if detector_info and 'matcher_params' in detector_info:
            matcher_params = detector_info['matcher_params'].get(matcher, {})
            return matcher_params
        
        return {}
    
    def validate_configuration(self, detector: str, matcher: str) -> Tuple[bool, List[str]]:
        """
        Validate a detector-matcher configuration
        
        Args:
            detector: Detector name
            matcher: Matcher name
        
        Returns:
            (is_valid, list_of_warnings/errors)
        """
        messages = []
        
        # Check if detector exists
        detector_info = self.config.get('detectors', {}).get(detector)
        if not detector_info:
            messages.append(f"âŒ Unknown detector: {detector}")
            return False, messages
        
        # Check if matcher exists
        matcher_info = self.config.get('matchers', {}).get(matcher)
        if not matcher_info:
            messages.append(f"âŒ Unknown matcher: {matcher}")
            return False, messages
        
        # Check compatibility
        if not self.is_compatible(detector, matcher):
            compatible = self.get_compatible_matchers(detector)
            messages.append(f"âŒ {detector} is not compatible with {matcher}")
            messages.append(f"   Compatible matchers: {', '.join(compatible)}")
            return False, messages
        
        # Check if it's the recommended matcher
        recommended = self.get_recommended_matcher(detector)
        if recommended and matcher.lower() != recommended.lower():
            messages.append(f"âš ï¸  {detector} recommends '{recommended}' matcher, you're using '{matcher}'")
        
        # Check descriptor type compatibility
        descriptor_type = detector_info.get('descriptor_type')
        supported_types = matcher_info.get('supports_descriptor_types', [])
        
        if descriptor_type not in supported_types and descriptor_type != 'none':
            messages.append(f"âš ï¸  {matcher} may not work optimally with {descriptor_type} descriptors")
        
        # Check if raw image is required
        if detector_info.get('requires_raw_image') and matcher_info.get('requires_raw_images'):
            messages.append(f"â„¹ï¸  Note: Both {detector} and {matcher} require raw images")
        
        messages.append(f"âœ“ {detector} + {matcher} is compatible")
        return True, messages
    
    def get_detector_info(self, detector: str) -> Dict:
        """Get full information about a detector"""
        return self.config.get('detectors', {}).get(detector, {})
    
    def get_matcher_info(self, matcher: str) -> Dict:
        """Get full information about a matcher"""
        return self.config.get('matchers', {}).get(matcher, {})
    
    def print_compatibility_matrix(self):
        """Print a compatibility matrix for all detectors and matchers"""
        detectors = list(self.config.get('detectors', {}).keys())
        matchers = list(self.config.get('matchers', {}).keys())
        
        print("\n" + "="*60)
        print("DETECTOR-MATCHER COMPATIBILITY MATRIX")
        print("="*60)
        
        # Header
        header = f"{'Detector':<15}"
        for matcher in matchers:
            header += f"{matcher:<12}"
        print(header)
        print("-" * 60)
        
        # Rows
        for detector in detectors:
            row = f"{detector:<15}"
            for matcher in matchers:
                if self.is_compatible(detector, matcher):
                    is_recommended = (matcher == self.get_recommended_matcher(detector))
                    symbol = "✔✔" if is_recommended else "✔"
                else:
                    symbol = "✖"
                row += f"{symbol:<12}"
            print(row)

        print("\n✔✔ = Recommended | ✔ = Compatible | ✖ = Incompatible")

    def list_all_detectors(self) -> List[str]:
        """Get list of all configured detectors"""
        return list(self.config.get('detectors', {}).keys())
    
    def list_all_matchers(self) -> List[str]:
        """Get list of all configured matchers"""
        return list(self.config.get('matchers', {}).keys())


# =============================================================================
# Usage example
# =============================================================================

if __name__ == "__main__":
    # Load configuration
    compat_mgr = MatcherCompatibilityManager()
    
    # Print compatibility matrix
    compat_mgr.print_compatibility_matrix()
    
    # Validate specific combinations
    print("\n\nValidating specific configurations:")
    
    test_combinations = [
        ('DISK', 'lightglue'),
        ('SIFT', 'flann'),
        ('ORB', 'bf'),
        ('DISK', 'bf'),  # Valid but not recommended
    ]
    
    for detector, matcher in test_combinations:
        is_valid, messages = compat_mgr.validate_configuration(detector, matcher)
        print(f"\n{detector} + {matcher}:")
        for msg in messages:
            print(f"  {msg}")