"""
Dataset loading for GeoGuessr country classification.

Dataset structure expected:
    dataset/
        CountryA/
            image1.jpg
            image2.jpg
            ...
        CountryB/
            ...
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Optional, Tuple, Callable
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split


class GeoDataset(Dataset):
    """
    Dataset for GeoGuessr country classification.
    
    Loads images from a directory structure where each subdirectory
    represents a country/class.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        class_names: List[str],
        transform: Optional[Callable] = None,
        return_path: bool = False
    ):
        """
        Args:
            image_paths: List of paths to images
            labels: List of integer labels
            class_names: List of class names (countries)
            transform: Optional transform to apply to images
            return_path: Whether to return image path along with image and label
        """
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform
        self.return_path = return_path
        self.num_classes = len(class_names)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple:
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        if self.return_path:
            return image, label, img_path
        return image, label
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get count of images per class."""
        counts = Counter(self.labels)
        return {self.class_names[k]: v for k, v in sorted(counts.items())}
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalanced data.
        Weight formula: w_c = 1 / log(1 + n_c)
        """
        counts = Counter(self.labels)
        weights = []
        for i in range(self.num_classes):
            n_c = counts.get(i, 1)
            w = 1.0 / np.log(1 + n_c)
            weights.append(w)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum() * self.num_classes
        
        return torch.FloatTensor(weights)


def load_dataset_from_directory(
    root_dir: str,
    extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.webp'),
    min_samples_per_class: int = 10
) -> Tuple[List[str], List[int], List[str]]:
    """
    Load image paths and labels from directory structure.
    
    Args:
        root_dir: Root directory containing country subdirectories
        extensions: Valid image file extensions
        min_samples_per_class: Minimum samples required per class (for stratified split)
        
    Returns:
        image_paths: List of image file paths
        labels: List of integer labels
        class_names: List of class names (sorted alphabetically)
    """
    # First pass: count images per class
    all_dirs = sorted([
        d for d in os.listdir(root_dir) 
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    
    class_counts = {}
    for class_name in all_dirs:
        class_dir = os.path.join(root_dir, class_name)
        count = len([f for f in os.listdir(class_dir) if f.lower().endswith(extensions)])
        class_counts[class_name] = count
    
    # Filter out classes with too few samples
    valid_classes = [name for name, count in class_counts.items() if count >= min_samples_per_class]
    excluded_classes = [name for name, count in class_counts.items() if count < min_samples_per_class]
    
    if excluded_classes:
        print(f"Excluding {len(excluded_classes)} classes with < {min_samples_per_class} samples:")
        for name in excluded_classes[:10]:
            print(f"  - {name}: {class_counts[name]} samples")
        if len(excluded_classes) > 10:
            print(f"  ... and {len(excluded_classes) - 10} more")
    
    # Build final dataset
    class_names = sorted(valid_classes)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    image_paths = []
    labels = []
    
    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        class_idx = class_to_idx[class_name]
        
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(extensions):
                image_paths.append(os.path.join(class_dir, filename))
                labels.append(class_idx)
    
    print(f"Loaded {len(image_paths)} images from {len(class_names)} classes")
    return image_paths, labels, class_names


def create_splits(
    image_paths: List[str],
    labels: List[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[Tuple[List, List], Tuple[List, List], Tuple[List, List]]:
    """
    Create stratified train/val/test splits.
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_state: Random seed for reproducibility
        
    Returns:
        (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # First split: train vs (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        train_size=train_ratio,
        stratify=labels,
        random_state=random_state
    )
    
    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        train_size=val_ratio_adjusted,
        stratify=temp_labels,
        random_state=random_state
    )
    
    print(f"Split sizes - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def create_data_loaders(
    root_dir: str,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    test_transform: Optional[Callable] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], torch.Tensor]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        root_dir: Root directory containing dataset
        train_transform: Transform for training images
        val_transform: Transform for validation images
        test_transform: Transform for test images
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        use_weighted_sampler: Whether to use weighted sampling for class imbalance
        train_ratio, val_ratio, test_ratio: Split ratios
        random_state: Random seed
        
    Returns:
        train_loader, val_loader, test_loader, class_names, class_weights
    """
    # Load all data
    image_paths, labels, class_names = load_dataset_from_directory(root_dir)
    
    # Create splits
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = create_splits(
        image_paths, labels, train_ratio, val_ratio, test_ratio, random_state
    )
    
    # Create datasets
    train_dataset = GeoDataset(train_paths, train_labels, class_names, transform=train_transform)
    val_dataset = GeoDataset(val_paths, val_labels, class_names, transform=val_transform)
    test_dataset = GeoDataset(test_paths, test_labels, class_names, transform=test_transform, return_path=True)
    
    # Get class weights
    class_weights = train_dataset.get_class_weights()
    
    # Create weighted sampler for training
    if use_weighted_sampler:
        sample_weights = [class_weights[label].item() for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_names, class_weights


if __name__ == "__main__":
    # Test with a mock directory structure
    print("GeoDataset module loaded successfully.")
    print("Usage: create_data_loaders('path/to/dataset')")

