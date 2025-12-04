"""
Custom augmentations for side experiments.

Includes:
- EdgeAugmentation: Adds Sobel/Canny edge channels to images
- LineFeatureAugmentation: Extracts line orientation histogram features
- SkyGroundAugmentation: Creates sky and ground masked images
"""

import numpy as np
import cv2
import torch
from PIL import Image
from typing import Tuple, Union


class EdgeAugmentation:
    """
    Augmentation that adds edge channels to RGB images.
    
    For use with GeoCNN-Edge model (Side Experiment A).
    
    Modes:
    - 'sobel': Adds cos(θ) and sin(θ) channels (5 total channels)
    - 'canny': Adds binary edge map (4 total channels)
    """
    
    def __init__(self, edge_type: str = 'sobel', epsilon: float = 1e-6):
        self.edge_type = edge_type
        self.epsilon = epsilon
        
        # Canny parameters
        self.canny_low = 50
        self.canny_high = 150
    
    def _sobel_edges(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Sobel edge orientation channels."""
        gray = gray.astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(gx**2 + gy**2 + self.epsilon)
        cos_theta = gx / magnitude
        sin_theta = gy / magnitude
        
        return cos_theta, sin_theta
    
    def _canny_edges(self, gray: np.ndarray) -> np.ndarray:
        """Compute Canny edge map."""
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        return edges.astype(np.float32) / 255.0
    
    def __call__(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Add edge channels to image.
        
        Args:
            image: PIL Image or numpy array (H, W, 3)
            
        Returns:
            augmented: numpy array (H, W, 4 or 5) depending on edge_type
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Normalize RGB to [0, 1] for consistency
        rgb = image.astype(np.float32) / 255.0
        
        if self.edge_type == 'sobel':
            cos_theta, sin_theta = self._sobel_edges(gray)
            augmented = np.dstack([rgb, cos_theta, sin_theta])
        else:  # canny
            edges = self._canny_edges(gray)
            augmented = np.dstack([rgb, edges])
        
        return augmented


class EdgeToTensor:
    """
    Convert edge-augmented numpy array to tensor.
    
    Use after EdgeAugmentation and before feeding to model.
    """
    
    def __init__(self, mean: list = None, std: list = None):
        # Default: ImageNet stats for RGB, 0/1 for edge channels
        if mean is None:
            self.mean = [0.485, 0.456, 0.406, 0.0, 0.0]  # For 5-channel Sobel
        else:
            self.mean = mean
        
        if std is None:
            self.std = [0.229, 0.224, 0.225, 1.0, 1.0]
        else:
            self.std = std
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert to tensor and normalize.
        
        Args:
            image: numpy array (H, W, C)
            
        Returns:
            tensor: torch tensor (C, H, W)
        """
        # Transpose to (C, H, W)
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # Normalize each channel
        for c in range(tensor.shape[0]):
            if c < len(self.mean):
                tensor[c] = (tensor[c] - self.mean[c]) / self.std[c]
        
        return tensor


class LineFeatureAugmentation:
    """
    Extract line orientation histogram features from images.
    
    For use with GeoCNN-Lines model (Side Experiment D).
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
    
    def __call__(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Extract line features from image.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            features: numpy array (num_bins + 4,)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)
        
        # Canny edge detection
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        
        # Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        # Initialize features
        histogram = np.zeros(self.num_bins, dtype=np.float32)
        mean_theta = 0.0
        circular_variance = 0.0
        frac_long_lines = 0.0
        line_count_norm = 0.0
        
        if lines is not None and len(lines) > 0:
            orientations = []
            lengths = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1
                
                theta = np.arctan2(abs(dy), abs(dx))
                orientations.append(theta)
                lengths.append(np.sqrt(dx**2 + dy**2))
            
            orientations = np.array(orientations)
            lengths = np.array(lengths)
            
            # Build length-weighted histogram
            for theta, length in zip(orientations, lengths):
                bin_idx = min(int(theta / np.pi * self.num_bins), self.num_bins - 1)
                histogram[bin_idx] += length
            
            if histogram.sum() > 0:
                histogram = histogram / histogram.sum()
            
            # Statistics
            weights = lengths / lengths.sum() if lengths.sum() > 0 else np.ones(len(lengths)) / len(lengths)
            mean_theta = np.average(orientations, weights=weights) / np.pi
            
            cos_sum = np.sum(weights * np.cos(2 * orientations))
            sin_sum = np.sum(weights * np.sin(2 * orientations))
            circular_variance = 1 - np.sqrt(cos_sum**2 + sin_sum**2)
            
            frac_long_lines = (lengths > self.long_line_threshold).sum() / len(lengths)
            line_count_norm = min(len(lines) / 200.0, 1.0)
        
        features = np.concatenate([
            histogram,
            [mean_theta, circular_variance, frac_long_lines, line_count_norm]
        ])
        
        return features.astype(np.float32)


class SkyGroundAugmentation:
    """
    Create sky and ground masked images using superpixel segmentation.
    
    For use with GeoCNN-Segmentation model (Side Experiment C).
    """
    
    def __init__(
        self,
        n_segments: int = 200,
        compactness: float = 10.0,
        sky_blue_threshold: float = 0.5,
        sky_saturation_threshold: float = 0.3,
        sky_position_threshold: float = 0.5,
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
    
    def __call__(self, image: Union[Image.Image, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sky and ground masked images.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            sky_image: Image with only sky regions
            ground_image: Image with only ground regions
        """
        from skimage.segmentation import slic
        from skimage.color import rgb2hsv
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        H, W = image.shape[:2]
        image_float = image.astype(np.float32) / 255.0
        
        # SLIC superpixels
        segments = slic(image_float, n_segments=self.n_segments, 
                       compactness=self.compactness, channel_axis=2, start_label=0)
        
        # HSV for color analysis
        hsv = rgb2hsv(image_float)
        
        sky_mask = np.zeros((H, W), dtype=np.float32)
        ground_mask = np.zeros((H, W), dtype=np.float32)
        
        for seg_id in np.unique(segments):
            mask = segments == seg_id
            y_coords, _ = np.where(mask)
            mean_y = y_coords.mean() / H
            
            mean_rgb = image_float[mask].mean(axis=0)
            mean_hsv = hsv[mask].mean(axis=0)
            
            mean_blue = mean_rgb[2]
            mean_saturation = mean_hsv[1]
            hue = mean_hsv[0]
            
            is_sky = (
                mean_y < self.sky_position_threshold and
                mean_blue > self.sky_blue_threshold and
                mean_saturation < self.sky_saturation_threshold
            )
            
            is_ground = (
                mean_y > self.ground_position_threshold and
                (mean_saturation > self.ground_saturation_threshold or
                 (0.05 < hue < 0.45))
            )
            
            if is_sky:
                sky_mask[mask] = 1.0
            if is_ground:
                ground_mask[mask] = 1.0
        
        sky_image = image.copy()
        ground_image = image.copy()
        
        sky_image[sky_mask == 0] = 0
        ground_image[ground_mask == 0] = 0
        
        return sky_image, ground_image


if __name__ == "__main__":
    # Test augmentations
    print("Testing augmentations...")
    
    # Create dummy image
    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test EdgeAugmentation
    edge_aug_sobel = EdgeAugmentation(edge_type='sobel')
    edge_aug_canny = EdgeAugmentation(edge_type='canny')
    
    sobel_out = edge_aug_sobel(dummy)
    canny_out = edge_aug_canny(dummy)
    print(f"Sobel augmentation output shape: {sobel_out.shape}")
    print(f"Canny augmentation output shape: {canny_out.shape}")
    
    # Test EdgeToTensor
    edge_to_tensor = EdgeToTensor()
    tensor_out = edge_to_tensor(sobel_out)
    print(f"Edge tensor output shape: {tensor_out.shape}")
    
    # Test LineFeatureAugmentation
    line_aug = LineFeatureAugmentation()
    line_features = line_aug(dummy)
    print(f"Line features shape: {line_features.shape}")
    
    # Test SkyGroundAugmentation
    sky_ground_aug = SkyGroundAugmentation()
    sky_img, ground_img = sky_ground_aug(dummy)
    print(f"Sky image shape: {sky_img.shape}")
    print(f"Ground image shape: {ground_img.shape}")
    
    print("\nAll augmentations working correctly!")

