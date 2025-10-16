"""
Base Selection Strategy

Abstract base class for all selection strategies (pair selection, view selection, etc.).
Defines the common interface that all selectors must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np


class SelectionMode(Enum):
    """Selection mode for different reconstruction stages"""
    INITIALIZATION = "initialization"  # Initial two-view pair
    INCREMENTAL = "incremental"        # Next camera to add
    REFINEMENT = "refinement"          # Additional views for refinement


@dataclass
class SelectionResult:
    """
    Result of a selection operation.
    
    Attributes:
        selected_item: The selected item (pair, camera, etc.)
        score: Selection score
        rank: Rank among candidates (1 = best)
        metadata: Additional selection information
        alternatives: List of alternative candidates
    """
    selected_item: Any
    score: float
    rank: int
    metadata: Dict[str, Any]
    alternatives: Optional[List[Tuple[Any, float]]] = None
    
    def __repr__(self) -> str:
        return f"SelectionResult(item={self.selected_item}, score={self.score:.3f}, rank={self.rank})"


class BaseSelector(ABC):
    """
    Abstract base class for selection strategies.
    
    All selectors (pair selectors, view selectors, etc.) should extend this class
    and implement the required methods.
    
    This provides a common interface for:
    - Scoring candidates
    - Selecting the best candidate(s)
    - Validating selections
    - Providing alternatives
    """
    
    def __init__(self, **config):
        """
        Initialize selector with configuration.
        
        Args:
            **config: Configuration parameters
        """
        self.config = config
        self.candidates = []
        self.scores = {}
    
    @abstractmethod
    def score_candidate(self, candidate: Any, context: Optional[Dict] = None) -> float:
        """
        Score a single candidate.
        
        Args:
            candidate: The candidate to score
            context: Optional context information for scoring
            
        Returns:
            Score (higher is better)
        """
        pass
    
    @abstractmethod
    def select_best(self, 
                    candidates: List[Any],
                    context: Optional[Dict] = None,
                    num_selections: int = 1) -> SelectionResult:
        """
        Select the best candidate(s) from a list.
        
        Args:
            candidates: List of candidates to choose from
            context: Optional context for selection
            num_selections: Number of candidates to select
            
        Returns:
            SelectionResult with best candidate(s)
        """
        pass
    
    def score_all_candidates(self, 
                            candidates: List[Any],
                            context: Optional[Dict] = None) -> Dict[Any, float]:
        """
        Score all candidates.
        
        Args:
            candidates: List of candidates
            context: Optional context information
            
        Returns:
            Dictionary mapping candidates to scores
        """
        scores = {}
        for candidate in candidates:
            try:
                score = self.score_candidate(candidate, context)
                scores[candidate] = score
            except Exception as e:
                print(f"Warning: Failed to score candidate {candidate}: {e}")
                scores[candidate] = 0.0
        
        return scores
    
    def rank_candidates(self,
                       candidates: List[Any],
                       context: Optional[Dict] = None) -> List[Tuple[Any, float, int]]:
        """
        Rank all candidates by score.
        
        Args:
            candidates: List of candidates
            context: Optional context
            
        Returns:
            List of (candidate, score, rank) tuples, sorted by rank
        """
        # Score all candidates
        scores = self.score_all_candidates(candidates, context)
        
        # Sort by score (descending)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Add ranks
        ranked_with_rank = [(item, score, rank + 1) 
                           for rank, (item, score) in enumerate(ranked)]
        
        return ranked_with_rank
    
    def validate_selection(self, 
                          selection: Any,
                          context: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Validate a selection.
        
        Args:
            selection: The selected item
            context: Optional context
            
        Returns:
            (is_valid, message)
        """
        # Default: always valid
        # Override in subclasses for specific validation
        return True, ""
    
    def get_alternatives(self,
                        candidates: List[Any],
                        context: Optional[Dict] = None,
                        num_alternatives: int = 3) -> List[Tuple[Any, float]]:
        """
        Get alternative candidates.
        
        Args:
            candidates: List of all candidates
            context: Optional context
            num_alternatives: Number of alternatives to return
            
        Returns:
            List of (candidate, score) tuples
        """
        ranked = self.rank_candidates(candidates, context)
        
        # Return top N alternatives (excluding the best)
        alternatives = [(item, score) 
                       for item, score, rank in ranked[1:num_alternatives + 1]]
        
        return alternatives
    
    def explain_selection(self, 
                         selection: Any,
                         score: float,
                         context: Optional[Dict] = None) -> str:
        """
        Provide human-readable explanation for selection.
        
        Args:
            selection: The selected item
            score: Selection score
            context: Optional context
            
        Returns:
            Explanation string
        """
        return f"Selected: {selection} (score: {score:.3f})"


class ScoreComponent:
    """
    A single component of a composite score.
    
    Used for multi-criteria scoring where the total score
    is a weighted combination of multiple components.
    """
    
    def __init__(self,
                 name: str,
                 value: float,
                 weight: float = 1.0,
                 description: str = ""):
        """
        Initialize score component.
        
        Args:
            name: Component name
            value: Raw score value
            weight: Weight in total score (0-1)
            description: Human-readable description
        """
        self.name = name
        self.value = value
        self.weight = weight
        self.description = description
        self.weighted_value = value * weight
    
    def __repr__(self) -> str:
        return f"{self.name}: {self.value:.3f} (weight={self.weight:.2f})"


class CompositeScore:
    """
    Composite score made up of multiple weighted components.
    
    Example:
        score = CompositeScore()
        score.add_component("matches", 0.8, weight=0.4)
        score.add_component("coverage", 0.6, weight=0.3)
        score.add_component("distribution", 0.7, weight=0.3)
        total = score.get_total_score()  # Weighted sum
    """
    
    def __init__(self):
        """Initialize composite score."""
        self.components: List[ScoreComponent] = []
    
    def add_component(self,
                     name: str,
                     value: float,
                     weight: float = 1.0,
                     description: str = ""):
        """
        Add a score component.
        
        Args:
            name: Component name
            value: Score value (typically 0-1)
            weight: Component weight
            description: Description
        """
        component = ScoreComponent(name, value, weight, description)
        self.components.append(component)
    
    def get_total_score(self) -> float:
        """
        Get total weighted score.
        
        Returns:
            Sum of weighted component values
        """
        return sum(c.weighted_value for c in self.components)
    
    def get_component_scores(self) -> Dict[str, float]:
        """
        Get individual component scores.
        
        Returns:
            Dictionary of component names to values
        """
        return {c.name: c.value for c in self.components}
    
    def get_weighted_scores(self) -> Dict[str, float]:
        """
        Get weighted component scores.
        
        Returns:
            Dictionary of component names to weighted values
        """
        return {c.name: c.weighted_value for c in self.components}
    
    def normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total_weight = sum(c.weight for c in self.components)
        
        if total_weight > 0:
            for component in self.components:
                component.weight /= total_weight
                component.weighted_value = component.value * component.weight
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary with score information
        """
        return {
            'total_score': self.get_total_score(),
            'components': {
                c.name: {
                    'value': c.value,
                    'weight': c.weight,
                    'weighted_value': c.weighted_value,
                    'description': c.description
                }
                for c in self.components
            }
        }
    
    def __repr__(self) -> str:
        components_str = ", ".join([f"{c.name}={c.value:.2f}" for c in self.components])
        return f"CompositeScore(total={self.get_total_score():.3f}, {components_str})"


# Utility functions

def normalize_score(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a value to [0, 1] range.
    
    Args:
        value: Value to normalize
        min_val: Minimum value in range
        max_val: Maximum value in range
        
    Returns:
        Normalized value in [0, 1]
    """
    if max_val == min_val:
        return 1.0
    
    normalized = (value - min_val) / (max_val - min_val)
    return np.clip(normalized, 0.0, 1.0)


def apply_threshold(score: float, threshold: float, mode: str = 'hard') -> float:
    """
    Apply threshold to score.
    
    Args:
        score: Input score
        threshold: Threshold value
        mode: 'hard' (0 or 1) or 'soft' (smooth transition)
        
    Returns:
        Thresholded score
    """
    if mode == 'hard':
        return 1.0 if score >= threshold else 0.0
    elif mode == 'soft':
        # Sigmoid-like smooth transition
        k = 10  # Steepness
        return 1.0 / (1.0 + np.exp(-k * (score - threshold)))
    else:
        return score