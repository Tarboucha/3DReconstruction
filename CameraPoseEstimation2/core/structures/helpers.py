import cv2
from typing import List, Dict, Optional

def keypoints_to_serializable(keypoints: List[cv2.KeyPoint]) -> List[Dict]:
    """
    Convert keypoints to JSON-serializable format.
    
    Args:
        keypoints: List of cv2.KeyPoint objects
        
    Returns:
        List of dictionaries with keypoint data
    """
    return [
        {
            'pt': (float(kp.pt[0]), float(kp.pt[1])),
            'angle': float(kp.angle),
            'class_id': int(kp.class_id),
            'octave': int(kp.octave),
            'response': float(kp.response),
            'size': float(kp.size)
        }
        for kp in keypoints
    ]


def keypoints_from_serializable(keypoints_data: List[Dict]) -> List[cv2.KeyPoint]:
    """
    Reconstruct cv2.KeyPoint objects from serialized format.
    
    Args:
        keypoints_data: List of keypoint dictionaries
        
    Returns:
        List of cv2.KeyPoint objects
    """
    keypoints = []
    for kp_data in keypoints_data:
        kp = cv2.KeyPoint(
            x=float(kp_data['pt'][0]),
            y=float(kp_data['pt'][1]),
            size=float(kp_data.get('size', 1.0)),
            angle=float(kp_data.get('angle', -1.0)),
            response=float(kp_data.get('response', 0.0)),
            octave=int(kp_data.get('octave', 0)),
            class_id=int(kp_data.get('class_id', -1))
        )
        keypoints.append(kp)
    return keypoints


