"""
Video Processor Module
Handles video preprocessing, frame extraction, and normalization for activity recognition.
"""

import cv2
import numpy as np
from typing import List, Tuple
import os


class VideoProcessor:
    """
    Processes video files for activity recognition.
    
    Handles frame extraction, resizing, normalization, and preprocessing
    required for feeding video data into the deep learning model.
    """
    
    def __init__(self, target_frame_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the video processor.
        
        Args:
            target_frame_size: Tuple of (height, width) for frame resizing.
                              Default is (224, 224) for standard CNN input.
        """
        self.target_frame_size = target_frame_size
    
    def extract_frames(self, video_path: str, num_frames: int = 16, 
                      sample_rate: int = 2) -> np.ndarray:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the input video file
            num_frames: Number of frames to extract from the video
            sample_rate: Sample every nth frame (helps with temporal subsampling)
        
        Returns:
            NumPy array of shape (num_frames, height, width, 3) containing
            preprocessed frames. Returns None if video cannot be read.
        
        Example:
            >>> processor = VideoProcessor()
            >>> frames = processor.extract_frames('video.mp4', num_frames=16)
            >>> print(frames.shape)  # (16, 224, 224, 3)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        frames = []
        frame_count = 0
        
        while len(frames) < num_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Sample frames at specified rate
            if frame_count % sample_rate == 0:
                # Resize frame to target size
                frame = cv2.resize(frame, 
                                  (self.target_frame_size[1], 
                                   self.target_frame_size[0]))
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
        # Pad frames if fewer than required
        if len(frames) < num_frames:
            frames = self._pad_frames(frames, num_frames)
        
        return np.array(frames, dtype=np.float32)
    
    def normalize_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Normalize frames to [0, 1] range.
        
        Args:
            frames: NumPy array of video frames with values in [0, 255]
        
        Returns:
            Normalized frames with values in [0, 1]
        """
        return frames / 255.0
    
    def apply_imagenet_normalization(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply ImageNet normalization (used by pre-trained models).
        
        Subtracts ImageNet mean and divides by standard deviation.
        
        Args:
            frames: Normalized frames in [0, 1] range
        
        Returns:
            ImageNet-normalized frames
        """
        # ImageNet mean and std (RGB format)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        frames = (frames - mean) / std
        return frames
    
    def preprocess_video(self, video_path: str, num_frames: int = 16,
                        normalize: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline for a video file.
        
        Includes: frame extraction, resizing, normalization, and ImageNet
        standardization.
        
        Args:
            video_path: Path to input video
            num_frames: Number of frames to extract
            normalize: Whether to apply ImageNet normalization
        
        Returns:
            Preprocessed frames ready for model input
        """
        # Extract frames
        frames = self.extract_frames(video_path, num_frames)
        
        # Normalize to [0, 1]
        frames = self.normalize_frames(frames)
        
        # Apply ImageNet normalization if specified
        if normalize:
            frames = self.apply_imagenet_normalization(frames)
        
        return frames
    
    @staticmethod
    def _pad_frames(frames: List[np.ndarray], target_num: int) -> List[np.ndarray]:
        """
        Pad frame list by repeating the last frame.
        
        Args:
            frames: List of extracted frames
            target_num: Target number of frames
        
        Returns:
            Padded list of frames
        """
        while len(frames) < target_num:
            frames.append(frames[-1].copy())
        
        return frames


# Example usage
if __name__ == "__main__":
    processor = VideoProcessor(target_frame_size=(224, 224))
    
    # Example: Process a video file
    # frames = processor.preprocess_video('sample_video.mp4', num_frames=16)
    # print(f"Processed frames shape: {frames.shape}")
