"""
Image transforms and augmentations for GeoGuessr classification.

Key design decisions:
- DO NOT horizontally flip (it flips driving side and sign orientation)
- Avoid large rotations (sun and shadows encode hemisphere)
- Color jitter is moderate (soil/vegetation color is a strong cue)
- Simulate camera FoV changes with RandomResizedCrop
"""

import torch
from torchvision import transforms


# ImageNet normalization (used for convenience)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    input_size: int = 224,
    color_jitter: float = 0.2,
    blur_prob: float = 0.2,
    use_horizontal_flip: bool = False,  # Default OFF for geolocation
    normalize_mean: list = None,
    normalize_std: list = None
) -> transforms.Compose:
    """
    Get training transforms.
    
    Args:
        input_size: Target image size
        color_jitter: Strength of color augmentation
        blur_prob: Probability of Gaussian blur
        use_horizontal_flip: Whether to use horizontal flip (default: False)
        normalize_mean: Normalization mean (default: ImageNet)
        normalize_std: Normalization std (default: ImageNet)
        
    Returns:
        Composed transforms
    """
    if normalize_mean is None:
        normalize_mean = IMAGENET_MEAN
    if normalize_std is None:
        normalize_std = IMAGENET_STD
    
    transform_list = [
        # Resize to 256 on short side first
        transforms.Resize(256),
        # Random crop to simulate camera FoV changes
        transforms.RandomResizedCrop(
            input_size,
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1)
        ),
    ]
    
    # Optional horizontal flip (default off for geolocation)
    if use_horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    
    # Color augmentation
    transform_list.append(
        transforms.ColorJitter(
            brightness=color_jitter,
            contrast=color_jitter,
            saturation=color_jitter,
            hue=0.02  # Small hue change
        )
    )
    
    # Random Gaussian blur
    if blur_prob > 0:
        transform_list.append(
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=blur_prob)
        )
    
    # Convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    
    return transforms.Compose(transform_list)


def get_val_transforms(
    input_size: int = 224,
    normalize_mean: list = None,
    normalize_std: list = None
) -> transforms.Compose:
    """
    Get validation transforms (center crop, no augmentation).
    
    Args:
        input_size: Target image size
        normalize_mean: Normalization mean (default: ImageNet)
        normalize_std: Normalization std (default: ImageNet)
        
    Returns:
        Composed transforms
    """
    if normalize_mean is None:
        normalize_mean = IMAGENET_MEAN
    if normalize_std is None:
        normalize_std = IMAGENET_STD
    
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])


def get_test_transforms(
    input_size: int = 224,
    normalize_mean: list = None,
    normalize_std: list = None
) -> transforms.Compose:
    """
    Get test transforms (same as validation).
    
    Args:
        input_size: Target image size
        normalize_mean: Normalization mean (default: ImageNet)
        normalize_std: Normalization std (default: ImageNet)
        
    Returns:
        Composed transforms
    """
    return get_val_transforms(input_size, normalize_mean, normalize_std)


class FiveCropTTA:
    """
    Five-crop Test Time Augmentation.
    
    Extracts center crop and four corner crops, then averages predictions.
    """
    
    def __init__(
        self,
        input_size: int = 224,
        normalize_mean: list = None,
        normalize_std: list = None
    ):
        if normalize_mean is None:
            normalize_mean = IMAGENET_MEAN
        if normalize_std is None:
            normalize_std = IMAGENET_STD
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.FiveCrop(input_size),
            transforms.Lambda(lambda crops: torch.stack([
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=normalize_mean, std=normalize_std)
                ])(crop) for crop in crops
            ]))
        ])
    
    def __call__(self, image):
        return self.transform(image)


class MultiScaleTTA:
    """
    Multi-scale Test Time Augmentation.
    
    Processes image at multiple scales and averages predictions.
    """
    
    def __init__(
        self,
        input_size: int = 224,
        scales: list = None,
        normalize_mean: list = None,
        normalize_std: list = None
    ):
        if normalize_mean is None:
            normalize_mean = IMAGENET_MEAN
        if normalize_std is None:
            normalize_std = IMAGENET_STD
        if scales is None:
            scales = [224, 256, 288]
        
        self.input_size = input_size
        self.scales = scales
        self.normalize = transforms.Normalize(mean=normalize_mean, std=normalize_std)
        self.to_tensor = transforms.ToTensor()
    
    def __call__(self, image):
        crops = []
        for scale in self.scales:
            transform = transforms.Compose([
                transforms.Resize(scale),
                transforms.CenterCrop(self.input_size),
                self.to_tensor,
                self.normalize
            ])
            crops.append(transform(image))
        return torch.stack(crops)


if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    
    # Create dummy image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
    
    # Test transforms
    train_tf = get_train_transforms()
    val_tf = get_val_transforms()
    test_tf = get_test_transforms()
    
    train_out = train_tf(dummy_img)
    val_out = val_tf(dummy_img)
    test_out = test_tf(dummy_img)
    
    print(f"Train transform output shape: {train_out.shape}")
    print(f"Val transform output shape: {val_out.shape}")
    print(f"Test transform output shape: {test_out.shape}")
    
    # Test TTA
    five_crop = FiveCropTTA()
    five_out = five_crop(dummy_img)
    print(f"FiveCrop TTA output shape: {five_out.shape}")
    
    multi_scale = MultiScaleTTA()
    multi_out = multi_scale(dummy_img)
    print(f"MultiScale TTA output shape: {multi_out.shape}")

