"""
GeoCNN-Lines: CNN with line orientation histogram fusion.

Side Experiment D: Line orientation histogram + fusion
- Run Canny + Probabilistic Hough Transform
- Build a 16-bin histogram over line orientations θ in [0, π)
- Add summary stats (mean θ, circular variance, fraction of long lines)
- Feed the 18-20D vector through a small MLP (64→64)
- Concatenate with the 256-d CNN feature before the final classifier

Motivation: Lane geometry, power lines, and skyline structure can be region-specific.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Tuple

from .geocnn_base import GeoCNNBase, ConvBlock


class LineFeatureExtractor:
    """
    Extract line orientation histogram from images using Hough Transform.
    """
    
    def __init__(
        self,
        num_bins: int = 16,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 50,
        min_line_length: int = 30,
        max_line_gap: int = 10,
        long_line_threshold: int = 100
    ):
        self.num_bins = num_bins
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.long_line_threshold = long_line_threshold
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Extract line features from image.
        
        Args:
            image: RGB image (H, W, 3) as numpy array
            
        Returns:
            features: Line feature vector (num_bins + 4,)
                - num_bins orientation histogram values
                - mean orientation (normalized)
                - circular variance
                - fraction of long lines
                - total line count (normalized)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Ensure uint8
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        
        # Apply Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        # Initialize feature vector
        histogram = np.zeros(self.num_bins)
        mean_theta = 0.0
        circular_variance = 0.0
        frac_long_lines = 0.0
        line_count_norm = 0.0
        
        if lines is not None and len(lines) > 0:
            # Compute orientations and lengths
            orientations = []
            lengths = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1
                
                # Orientation in [0, π)
                theta = np.arctan2(abs(dy), abs(dx))
                orientations.append(theta)
                
                # Line length
                length = np.sqrt(dx**2 + dy**2)
                lengths.append(length)
            
            orientations = np.array(orientations)
            lengths = np.array(lengths)
            
            # Build histogram (weighted by line length)
            bin_edges = np.linspace(0, np.pi, self.num_bins + 1)
            for theta, length in zip(orientations, lengths):
                bin_idx = min(int(theta / np.pi * self.num_bins), self.num_bins - 1)
                histogram[bin_idx] += length
            
            # Normalize histogram
            if histogram.sum() > 0:
                histogram = histogram / histogram.sum()
            
            # Mean orientation (weighted by length)
            weights = lengths / lengths.sum() if lengths.sum() > 0 else np.ones(len(lengths)) / len(lengths)
            mean_theta = np.average(orientations, weights=weights) / np.pi  # Normalize to [0, 1]
            
            # Circular variance
            # Using R = |mean of unit vectors|
            cos_sum = np.sum(weights * np.cos(2 * orientations))  # 2θ for orientation
            sin_sum = np.sum(weights * np.sin(2 * orientations))
            R = np.sqrt(cos_sum**2 + sin_sum**2)
            circular_variance = 1 - R  # Higher = more spread
            
            # Fraction of long lines
            long_mask = lengths > self.long_line_threshold
            frac_long_lines = long_mask.sum() / len(lengths)
            
            # Normalized line count
            line_count_norm = min(len(lines) / 200.0, 1.0)  # Cap at 200 lines
        
        # Concatenate features
        features = np.concatenate([
            histogram,
            [mean_theta, circular_variance, frac_long_lines, line_count_norm]
        ])
        
        return features.astype(np.float32)


class LineMLP(nn.Module):
    """Small MLP to process line features."""
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GeoCNNLines(nn.Module):
    """
    GeoCNN with line feature fusion.
    
    Architecture:
    - GeoCNN backbone extracts 256-d visual features
    - Line MLP processes 20-d line features to 64-d
    - Concatenate to get 320-d
    - Final classifier: 320 → num_classes
    """
    
    def __init__(
        self,
        num_classes: int,
        line_feature_dim: int = 20,
        line_hidden_dim: int = 64,
        dropout_p: float = 0.3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Visual backbone (GeoCNN blocks without final classifier)
        self.block1 = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3),
            ConvBlock(32, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3),
            ConvBlock(64, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3),
            ConvBlock(128, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3),
            ConvBlock(256, 256, kernel_size=1)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Line feature MLP
        self.line_mlp = LineMLP(input_dim=line_feature_dim, hidden_dim=line_hidden_dim)
        
        # Fused classifier
        fused_dim = 256 + line_hidden_dim  # 320
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(fused_dim, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def extract_visual_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 256-d visual features."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x
    
    def forward(self, image: torch.Tensor, line_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with image and line features.
        
        Args:
            image: Input images (B, 3, 224, 224)
            line_features: Pre-computed line features (B, 20)
            
        Returns:
            Logits (B, num_classes)
        """
        # Extract visual features
        visual_features = self.extract_visual_features(image)  # (B, 256)
        
        # Process line features
        line_features = self.line_mlp(line_features)  # (B, 64)
        
        # Fuse features
        fused = torch.cat([visual_features, line_features], dim=1)  # (B, 320)
        fused = self.dropout(fused)
        
        # Classify
        logits = self.fc(fused)
        
        return logits
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test line feature extraction
    print("Testing line feature extraction...")
    
    # Create dummy image with some lines
    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
    # Draw some lines
    cv2.line(dummy_image, (10, 10), (200, 50), (255, 255, 255), 2)
    cv2.line(dummy_image, (20, 100), (180, 120), (255, 255, 255), 2)
    cv2.line(dummy_image, (50, 200), (200, 200), (255, 255, 255), 2)
    
    extractor = LineFeatureExtractor()
    features = extractor(dummy_image)
    print(f"Line features shape: {features.shape}")
    print(f"Histogram (first 16): {features[:16]}")
    print(f"Stats (last 4): mean_θ={features[-4]:.3f}, circ_var={features[-3]:.3f}, "
          f"frac_long={features[-2]:.3f}, count_norm={features[-1]:.3f}")
    
    # Test model
    print("\nTesting GeoCNN-Lines model...")
    model = GeoCNNLines(num_classes=50)
    print(f"GeoCNN-Lines Parameters: {model.count_parameters():,}")
    
    # Forward pass
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    line_feats = torch.randn(batch_size, 20)
    out = model(images, line_feats)
    print(f"Input: images {images.shape}, lines {line_feats.shape}")
    print(f"Output: {out.shape}")

