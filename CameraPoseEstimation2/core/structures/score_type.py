from enum import Enum


class ScoreType(Enum):
    """
    Type of matching score.
    
    Different matching methods produce different types of scores:
    - DISTANCE: Lower values indicate better matches (e.g., L2 distance)
    - CONFIDENCE: Higher values indicate better matches (e.g., neural network output)
    - SIMILARITY: Higher values indicate better matches (e.g., cosine similarity)
    """
    DISTANCE = "distance"
    CONFIDENCE = "confidence"
    SIMILARITY = "similarity"
    
    def __str__(self):
        return self.value
    
    @classmethod
    def from_string(cls, s: str) -> 'ScoreType':
        """Create ScoreType from string"""
        s_lower = s.lower()
        for score_type in cls:
            if score_type.value == s_lower:
                return score_type
        # Default fallback
        return cls.CONFIDENCE
