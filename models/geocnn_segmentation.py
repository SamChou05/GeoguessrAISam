"""
GeoCNN-Segmentation: CNN with superpixel-based sky/ground gating.

Side Experiment C: Superpixel gating of sky/ground
- Compute SLIC superpixels (e.g., 200 superpixels)
- Classify superpixels into "sky-like" vs "ground-like" using simple color/position rules:
  - Sky-like: top-half + high blue channel + low saturation
  - Ground-like: bottom-half + higher saturation + brown/green channels
- Create two masked images: sky-only and ground-only
- Feed each to a shared-weight GeoCNN tower
- Fuse by concatenating the 256-d GAP features then a small MLP before classifier

Motivation: Many country cues are in the skyline (architecture, signage) or ground
(lane markings, soil/vegetation).
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.color import rgb2hsv
from typing import Tuple

from .geocnn_base import ConvBlock


class SkyGroundSegmenter:
    """
    Segment image into sky and ground regions using SLIC superpixels.
    """
    
    def __init__(
        self,
        n_segments: int = 200,
        compactness: float = 10.0,
        sky_blue_threshold: float = 0.5,
        sky_saturation_threshold: float = 0.3,
        sky_position_threshold: float = 0.5,  # Fraction of image height
        ground_saturation_threshold: float = 0.2,
        ground_position_threshold: float = 0.4
    ):
        self.n_segments = n_segments
        self.compactness = compactness
        self.sky_blue_threshold = sky_blue_threshold
        self.sky_saturation_threshold = sky_saturation_threshold
        self.sky_position_threshold = sky_position_threshold
        self.ground_saturation_threshold = ground_saturation_threshold
        self.ground_position_threshold = ground_position_threshold
    
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment image into sky and ground regions.
        
        Args:
            image: RGB image (H, W, 3) as numpy array, uint8 or float [0, 1]
            
        Returns:
            sky_mask: Binary mask for sky regions (H, W)
            ground_mask: Binary mask for ground regions (H, W)
        """
        H, W = image.shape[:2]
        
        # Normalize to [0, 1] for processing
        if image.dtype == np.uint8:
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.astype(np.float32)
        
        # Compute SLIC superpixels
        segments = slic(image_float, n_segments=self.n_segments, compactness=self.compactness, 
                       channel_axis=2, start_label=0)
        
        # Convert to HSV for saturation analysis
        hsv = rgb2hsv(image_float)
        
        # Initialize masks
        sky_mask = np.zeros((H, W), dtype=np.float32)
        ground_mask = np.zeros((H, W), dtype=np.float32)
        
        # Analyze each superpixel
        for seg_id in np.unique(segments):
            mask = segments == seg_id
            
            # Get superpixel properties
            y_coords, x_coords = np.where(mask)
            mean_y = y_coords.mean() / H  # Normalized position
            
            # Mean color values
            mean_rgb = image_float[mask].mean(axis=0)
            mean_hsv = hsv[mask].mean(axis=0)
            
            mean_blue = mean_rgb[2]  # Blue channel
            mean_saturation = mean_hsv[1]  # Saturation
            
            # Sky classification: top half + high blue + low saturation
            is_sky = (
                mean_y < self.sky_position_threshold and
                mean_blue > self.sky_blue_threshold and
                mean_saturation < self.sky_saturation_threshold
            )
            
            # Ground classification: bottom half + moderate to high saturation
            # Also consider brown/green hues (hue in [0.05, 0.45] roughly)
            hue = mean_hsv[0]
            is_ground = (
                mean_y > self.ground_position_threshold and
                (mean_saturation > self.ground_saturation_threshold or
                 (0.05 < hue < 0.45))  # Brown to green hue range
            )
            
            if is_sky:
                sky_mask[mask] = 1.0
            if is_ground:
                ground_mask[mask] = 1.0
        
        return sky_mask, ground_mask
    
    def apply_masks(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply sky and ground masks to create masked images.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            sky_image: Image with only sky regions (rest is black)
            ground_image: Image with only ground regions (rest is black)
        """
        sky_mask, ground_mask = self(image)
        
        # Apply masks
        sky_image = image.copy()
        ground_image = image.copy()
        
        sky_image[sky_mask == 0] = 0
        ground_image[ground_mask == 0] = 0
        
        return sky_image, ground_image


class SharedEncoder(nn.Module):
    """Shared CNN encoder for sky and ground images."""
    
    def __init__(self):
        super().__init__()
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x


class GeoCNNSegmentation(nn.Module):
    """
    GeoCNN with sky/ground two-tower fusion.
    
    Architecture:
    - Shared encoder processes sky-masked and ground-masked images
    - Each tower outputs 256-d features
    - Concatenate to get 512-d
    - MLP fusion: 512 → 256
    - Final classifier: 256 → num_classes
    """
    
    def __init__(self, num_classes: int, dropout_p: float = 0.3):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Shared encoder for both sky and ground
        self.encoder = SharedEncoder()
        
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )
        
        # Classifier
        self.fc = nn.Linear(256, num_classes)
        
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
    
    def forward(self, sky_image: torch.Tensor, ground_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sky and ground masked images.
        
        Args:
            sky_image: Sky-masked images (B, 3, 224, 224)
            ground_image: Ground-masked images (B, 3, 224, 224)
            
        Returns:
            Logits (B, num_classes)
        """
        # Extract features from both views
        sky_features = self.encoder(sky_image)  # (B, 256)
        ground_features = self.encoder(ground_image)  # (B, 256)
        
        # Fuse features
        fused = torch.cat([sky_features, ground_features], dim=1)  # (B, 512)
        fused = self.fusion(fused)  # (B, 256)
        
        # Classify
        logits = self.fc(fused)
        
        return logits
    
    def forward_single(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with a single image (no segmentation).
        Useful for comparison or when segmentation is not available.
        """
        features = self.encoder(image)
        # Duplicate features as if sky and ground were the same
        fused = torch.cat([features, features], dim=1)
        fused = self.fusion(fused)
        logits = self.fc(fused)
        return logits
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test segmentation
    print("Testing sky/ground segmentation...")
    
    # Create a simple test image (sky at top, ground at bottom)
    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
    # Sky (blue, top half)
    dummy_image[:100, :, 2] = 200  # Blue channel
    dummy_image[:100, :, 1] = 150  # Some green
    dummy_image[:100, :, 0] = 150  # Some red
    # Ground (brown/green, bottom half)
    dummy_image[124:, :, 0] = 100  # Brown-ish
    dummy_image[124:, :, 1] = 80
    dummy_image[124:, :, 2] = 50
    
    segmenter = SkyGroundSegmenter()
    sky_mask, ground_mask = segmenter(dummy_image)
    print(f"Sky mask: {sky_mask.sum():.0f} pixels ({sky_mask.mean()*100:.1f}%)")
    print(f"Ground mask: {ground_mask.sum():.0f} pixels ({ground_mask.mean()*100:.1f}%)")
    
    sky_image, ground_image = segmenter.apply_masks(dummy_image)
    print(f"Sky image non-zero: {(sky_image > 0).any(axis=2).sum()} pixels")
    print(f"Ground image non-zero: {(ground_image > 0).any(axis=2).sum()} pixels")
    
    # Test model
    print("\nTesting GeoCNN-Segmentation model...")
    model = GeoCNNSegmentation(num_classes=50)
    print(f"GeoCNN-Segmentation Parameters: {model.count_parameters():,}")
    
    # Forward pass
    batch_size = 4
    sky_imgs = torch.randn(batch_size, 3, 224, 224)
    ground_imgs = torch.randn(batch_size, 3, 224, 224)
    out = model(sky_imgs, ground_imgs)
    print(f"Input: sky {sky_imgs.shape}, ground {ground_imgs.shape}")
    print(f"Output: {out.shape}")
    
    # Test single image forward
    single_img = torch.randn(batch_size, 3, 224, 224)
    out_single = model.forward_single(single_img)
    print(f"Single image input: {single_img.shape} → output: {out_single.shape}")

