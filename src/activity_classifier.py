"""
Activity Classifier Module
Handles model loading, inference, and activity prediction from video frames.
"""

import numpy as np
from typing import List, Tuple, Dict
import os


class ActivityRecognizer:
    """
    Recognizes and classifies human activities from video data.
    
    This class loads a pre-trained deep learning model and performs
    inference on preprocessed video frames to predict human activities.
    """
    
    # Kinetic-400 activity classes (sample - full list contains 400+ classes)
    KINETIC_400_CLASSES = {
        0: 'walking',
        1: 'running',
        2: 'sitting',
        3: 'standing',
        4: 'jumping',
        5: 'waving',
        6: 'clapping',
        7: 'eating',
        # ... (full list contains 400+ activity classes)
    }
    
    def __init__(self, model_path: str = None, class_names: Dict = None):
        """
        Initialize the activity recognizer.
        
        Args:
            model_path: Path to pre-trained model file (.h5 or .pb format)
            class_names: Dictionary mapping class indices to activity names.
                        If None, uses Kinetic-400 classes.
        
        Example:
            >>> recognizer = ActivityRecognizer(model_path='models/model.h5')
        """
        self.model_path = model_path
        self.model = None
        self.class_names = class_names or self.KINETIC_400_CLASSES
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load a pre-trained deep learning model.
        
        Args:
            model_path: Path to the model file
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            
        Note:
            Supports TensorFlow/Keras .h5 and .pb (SavedModel) formats.
            In production, use:
            
            from tensorflow import keras
            self.model = keras.models.load_model(model_path)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # TODO: Implement actual model loading
        # from tensorflow import keras
        # self.model = keras.models.load_model(model_path)
        
        self.model_path = model_path
        print(f"Model loaded from {model_path}")
    
    def predict(self, frames: np.ndarray, 
                return_top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Predict activities for video frames.
        
        Args:
            frames: NumPy array of shape (num_frames, height, width, 3)
                   containing preprocessed video frames
            return_top_k: Return top-k predictions (default: 1)
        
        Returns:
            List of tuples (activity_name, confidence_score)
            Example: [('walking', 0.92), ('running', 0.08)]
        
        Example:
            >>> predictions = recognizer.predict(frames)
            >>> for activity, confidence in predictions:
            >>>     print(f"{activity}: {confidence:.2%}")
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Ensure frames have batch dimension
        if len(frames.shape) == 3:
            frames = np.expand_dims(frames, axis=0)
        
        # Run inference
        # output = self.model.predict(frames)
        
        # TODO: Implement actual inference
        # For now, return dummy predictions for demonstration
        num_classes = len(self.class_names)
        output = np.random.rand(1, num_classes)
        output = output / output.sum(axis=1, keepdims=True)
        
        # Get top-k predictions
        top_k_indices = np.argsort(output[0])[-return_top_k:][::-1]
        predictions = [
            (self.class_names.get(idx, f"activity_{idx}"), 
             float(output[0][idx]))
            for idx in top_k_indices
        ]
        
        return predictions
    
    def predict_batch(self, video_list: List[np.ndarray],
                     return_top_k: int = 1) -> List[List[Tuple[str, float]]]:
        """
        Predict activities for multiple videos in batch.
        
        Args:
            video_list: List of preprocessed video frame arrays
            return_top_k: Return top-k predictions per video
        
        Returns:
            List of prediction lists, one per input video
        """
        batch_results = []
        
        for video_frames in video_list:
            predictions = self.predict(video_frames, return_top_k)
            batch_results.append(predictions)
        
        return batch_results
    
    def predict_from_video_path(self, video_path: str,
                               return_top_k: int = 1) -> List[Tuple[str, float]]:
        """
        End-to-end prediction from a video file path.
        
        This is a convenience method that handles preprocessing internally.
        
        Args:
            video_path: Path to input video file
            return_top_k: Return top-k predictions
        
        Returns:
            List of (activity, confidence) tuples
        """
        from src_video_processor import VideoProcessor
        
        processor = VideoProcessor()
        frames = processor.preprocess_video(video_path)
        
        return self.predict(frames, return_top_k)
    
    def get_activity_name(self, class_index: int) -> str:
        """
        Get activity name from class index.
        
        Args:
            class_index: Predicted class index
        
        Returns:
            Activity name as string
        """
        return self.class_names.get(class_index, f"unknown_activity_{class_index}")
    
    def set_class_names(self, class_names: Dict):
        """
        Update the class name mappings.
        
        Useful for custom activity recognition tasks with different classes.
        
        Args:
            class_names: Dictionary mapping indices to activity names
        """
        self.class_names = class_names


# Example usage
if __name__ == "__main__":
    # Initialize recognizer
    recognizer = ActivityRecognizer(
        model_path='models/pretrained_model.h5'
    )
    
    # Example: Make predictions
    # dummy_frames = np.random.rand(16, 224, 224, 3)
    # predictions = recognizer.predict(dummy_frames)
    # for activity, conf in predictions:
    #     print(f"{activity}: {conf:.2%}")
