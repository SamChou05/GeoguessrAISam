"""
GeoCNN-Edge: CNN with edge-augmented input channels.

Side Experiment A: Edge-augmented input (Sobel or Canny)
- Compute Sobel Gx, Gy at load time
- Form two channels: cos(θ) = Gx/√(Gx²+Gy²+ε), sin(θ) = Gy/√(Gx²+Gy²+ε)
- Concatenate to RGB to get 5-channel input
- Change first conv to 5→32

Motivation: Countries differ in road markings, lane boundaries, and skyline geometry;
edges/gradients may amplify these cues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple

from .geocnn_base import GeoCNNBase


class SobelEdgeExtractor:
    """Extract Sobel edge orientation channels from images."""
    
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
    
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Sobel edge orientation channels.
        
        Args:
            image: RGB image as numpy array (H, W, 3), uint8 or float
            
        Returns:
            cos_theta: cos of gradient orientation (H, W)
            sin_theta: sin of gradient orientation (H, W)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Ensure float for precision
        gray = gray.astype(np.float32)
        
        # Compute Sobel gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Compute magnitude for normalization
        magnitude = np.sqrt(gx**2 + gy**2 + self.epsilon)
        
        # Compute normalized orientation channels
        cos_theta = gx / magnitude
        sin_theta = gy / magnitude
        
        return cos_theta, sin_theta


class CannyEdgeExtractor:
    """Extract Canny edge map from images."""
    
    def __init__(self, low_threshold: int = 50, high_threshold: int = 150):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Compute Canny edge map.
        
        Args:
            image: RGB image as numpy array (H, W, 3), uint8
            
        Returns:
            edges: Binary edge map (H, W), float32 in [0, 1]
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Ensure uint8 for Canny
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        
        # Normalize to [0, 1]
        edges = edges.astype(np.float32) / 255.0
        
        return edges


class GeoCNNEdge(GeoCNNBase):
    """
    GeoCNN with edge-augmented input.
    
    Supports two modes:
    - 'sobel': RGB + cos(θ) + sin(θ) = 5 channels
    - 'canny': RGB + edge_map = 4 channels
    """
    
    def __init__(self, num_classes: int, edge_type: str = 'sobel', dropout_p: float = 0.3):
        # Determine input channels based on edge type
        if edge_type == 'sobel':
            in_channels = 5  # RGB + cos(θ) + sin(θ)
        elif edge_type == 'canny':
            in_channels = 4  # RGB + edge
        else:
            raise ValueError(f"Unknown edge_type: {edge_type}. Use 'sobel' or 'canny'.")
        
        super().__init__(num_classes=num_classes, in_channels=in_channels, dropout_p=dropout_p)
        self.edge_type = edge_type


class EdgeAugmentedTransform:
    """
    Transform that adds edge channels to RGB images.
    
    Use this in the data loading pipeline to augment images with edge information.
    """
    
    def __init__(self, edge_type: str = 'sobel'):
        self.edge_type = edge_type
        if edge_type == 'sobel':
            self.extractor = SobelEdgeExtractor()
        elif edge_type == 'canny':
            self.extractor = CannyEdgeExtractor()
        else:
            raise ValueError(f"Unknown edge_type: {edge_type}")
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Add edge channels to image.
        
        Args:
            image: RGB image (H, W, 3) as numpy array
            
        Returns:
            augmented: Image with edge channels (H, W, 4 or 5)
        """
        if self.edge_type == 'sobel':
            cos_theta, sin_theta = self.extractor(image)
            # Stack: RGB + cos(θ) + sin(θ)
            augmented = np.dstack([image, cos_theta, sin_theta])
        else:  # canny
            edges = self.extractor(image)
            # Stack: RGB + edge
            augmented = np.dstack([image, edges])
        
        return augmented


if __name__ == "__main__":
    # Test edge extraction
    print("Testing edge extraction...")
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test Sobel
    sobel_extractor = SobelEdgeExtractor()
    cos_theta, sin_theta = sobel_extractor(dummy_image)
    print(f"Sobel cos_theta shape: {cos_theta.shape}, range: [{cos_theta.min():.2f}, {cos_theta.max():.2f}]")
    print(f"Sobel sin_theta shape: {sin_theta.shape}, range: [{sin_theta.min():.2f}, {sin_theta.max():.2f}]")
    
    # Test Canny
    canny_extractor = CannyEdgeExtractor()
    edges = canny_extractor(dummy_image)
    print(f"Canny edges shape: {edges.shape}, range: [{edges.min():.2f}, {edges.max():.2f}]")
    
    # Test transform
    transform = EdgeAugmentedTransform(edge_type='sobel')
    augmented = transform(dummy_image)
    print(f"Sobel augmented shape: {augmented.shape}")
    
    transform = EdgeAugmentedTransform(edge_type='canny')
    augmented = transform(dummy_image)
    print(f"Canny augmented shape: {augmented.shape}")
    
    # Test model
    print("\nTesting GeoCNN-Edge models...")
    
    model_sobel = GeoCNNEdge(num_classes=50, edge_type='sobel')
    print(f"GeoCNN-Edge (Sobel) Parameters: {model_sobel.count_parameters():,}")
    x_sobel = torch.randn(4, 5, 224, 224)
    out_sobel = model_sobel(x_sobel)
    print(f"Sobel input: {x_sobel.shape} → output: {out_sobel.shape}")
    
    model_canny = GeoCNNEdge(num_classes=50, edge_type='canny')
    print(f"GeoCNN-Edge (Canny) Parameters: {model_canny.count_parameters():,}")
    x_canny = torch.randn(4, 4, 224, 224)
    out_canny = model_canny(x_canny)
    print(f"Canny input: {x_canny.shape} → output: {out_canny.shape}")

