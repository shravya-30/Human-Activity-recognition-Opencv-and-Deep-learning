"""
Utility Functions
Helper functions for video processing, visualization, and evaluation.
"""

import numpy as np
import os
from typing import Dict, List, Tuple


def load_activity_classes(kinetic_400_file: str = None) -> Dict[int, str]:
    """
    Load activity class mappings.
    
    Args:
        kinetic_400_file: Optional path to file containing Kinetic-400 class names
    
    Returns:
        Dictionary mapping class indices to activity names
    """
    # If no file provided, return sample classes
    if kinetic_400_file is None:
        return {
            0: 'walking',
            1: 'running',
            2: 'sitting',
            3: 'standing',
            4: 'jumping',
            5: 'waving',
            6: 'clapping',
            7: 'eating',
            8: 'drinking',
            9: 'dancing'
        }
    
    # Load from file
    classes = {}
    if os.path.exists(kinetic_400_file):
        with open(kinetic_400_file, 'r') as f:
            for idx, line in enumerate(f):
                classes[idx] = line.strip()
    
    return classes


def smooth_predictions(predictions: List[Tuple[int, float]], 
                      window_size: int = 3) -> List[int]:
    """
    Smooth activity predictions over time using sliding window.
    
    Useful for reducing noise in frame-by-frame predictions.
    
    Args:
        predictions: List of (class_index, confidence) tuples
        window_size: Size of smoothing window
    
    Returns:
        Smoothed class predictions
    """
    smoothed = []
    
    for i in range(len(predictions)):
        start = max(0, i - window_size // 2)
        end = min(len(predictions), i + window_size // 2 + 1)
        
        window_classes = [pred[0] for pred in predictions[start:end]]
        # Most common class in window
        most_common = max(set(window_classes), 
                         key=window_classes.count)
        smoothed.append(most_common)
    
    return smoothed


def calculate_metrics(predictions: np.ndarray, 
                     ground_truth: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for activity recognition.
    
    Args:
        predictions: Array of predicted class indices
        ground_truth: Array of true class indices
    
    Returns:
        Dictionary containing accuracy, precision, recall metrics
    """
    correct = (predictions == ground_truth).sum()
    total = len(ground_truth)
    accuracy = correct / total
    
    return {
        'accuracy': accuracy,
        'correct_predictions': int(correct),
        'total_samples': int(total)
    }


def visualize_predictions(activity_names: List[str], 
                         confidences: List[float]) -> str:
    """
    Create a simple text-based visualization of predictions.
    
    Args:
        activity_names: List of activity names
        confidences: List of confidence scores (0-1)
    
    Returns:
        Formatted string visualization
    """
    result = "\n=== Activity Predictions ===\n"
    
    for activity, conf in zip(activity_names, confidences):
        bar_length = int(conf * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        result += f"{activity:20s} │{bar}│ {conf:.1%}\n"
    
    return result


def create_frame_batch(frames: List[np.ndarray], 
                      batch_size: int = 32) -> List[np.ndarray]:
    """
    Create batches of frames for efficient batch processing.
    
    Args:
        frames: List of frame arrays
        batch_size: Number of frames per batch
    
    Returns:
        List of batched frame arrays
    """
    batches = []
    
    for i in range(0, len(frames), batch_size):
        batch = np.array(frames[i:i + batch_size])
        batches.append(batch)
    
    return batches


def get_video_info(video_path: str) -> Dict:
    """
    Extract metadata about a video file.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary containing video properties
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration_seconds': cap.get(cv2.CAP_PROP_FRAME_COUNT) / 
                               cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    
    except Exception as e:
        print(f"Error reading video: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Load classes
    classes = load_activity_classes()
    print(f"Loaded {len(classes)} activity classes")
    
    # Example visualization
    activities = ['walking', 'running', 'sitting']
    scores = [0.85, 0.12, 0.03]
    print(visualize_predictions(activities, scores))
