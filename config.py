"""
Default configuration for GeoGuessr experiments.

This file contains all hyperparameters and settings used in the paper.
"""

# Dataset settings
DATA_CONFIG = {
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'image_extensions': ('.jpg', '.jpeg', '.png', '.webp'),
}

# Model settings
MODEL_CONFIG = {
    'input_size': 224,
    'dropout_p': 0.3,
    'num_blocks': 4,
    'channels': [32, 64, 128, 256],
}

# Training settings (from paper recipe)
TRAIN_CONFIG = {
    'optimizer': 'adamw',
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'epochs': 80,
    'batch_size': 64,
    'warmup_epochs': 3,
    'patience': 10,
    'label_smoothing': 0.1,
    'use_amp': True,
}

# Augmentation settings
AUGMENT_CONFIG = {
    'resize_size': 256,
    'crop_scale': (0.7, 1.0),
    'crop_ratio': (0.9, 1.1),
    'color_jitter_brightness': 0.2,
    'color_jitter_contrast': 0.2,
    'color_jitter_saturation': 0.2,
    'color_jitter_hue': 0.02,
    'blur_prob': 0.2,
    'horizontal_flip': False,  # Important: OFF for geolocation
}

# ImageNet normalization
NORMALIZE_CONFIG = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}

# BoVW settings
BOVW_CONFIG = {
    'n_clusters': 256,
    'max_descriptors_per_image': 500,
    'use_tfidf': True,
    'descriptor_type': 'orb',  # 'orb' or 'sift'
}

# Line feature settings
LINE_CONFIG = {
    'num_bins': 16,
    'canny_low': 50,
    'canny_high': 150,
    'hough_threshold': 50,
    'min_line_length': 30,
    'max_line_gap': 10,
    'long_line_threshold': 100,
}

# Segmentation settings
SEGMENTATION_CONFIG = {
    'n_segments': 200,
    'compactness': 10.0,
    'sky_blue_threshold': 0.5,
    'sky_saturation_threshold': 0.3,
    'sky_position_threshold': 0.5,
    'ground_saturation_threshold': 0.2,
    'ground_position_threshold': 0.4,
}

# Country to continent mapping (example - extend as needed)
COUNTRY_TO_CONTINENT = {
    # North America
    'United States': 'North America',
    'Canada': 'North America',
    'Mexico': 'North America',
    
    # South America
    'Argentina': 'South America',
    'Brazil': 'South America',
    'Chile': 'South America',
    'Colombia': 'South America',
    'Peru': 'South America',
    
    # Europe
    'United Kingdom': 'Europe',
    'France': 'Europe',
    'Germany': 'Europe',
    'Italy': 'Europe',
    'Spain': 'Europe',
    'Portugal': 'Europe',
    'Netherlands': 'Europe',
    'Belgium': 'Europe',
    'Sweden': 'Europe',
    'Norway': 'Europe',
    'Finland': 'Europe',
    'Denmark': 'Europe',
    'Poland': 'Europe',
    'Czech Republic': 'Europe',
    'Austria': 'Europe',
    'Switzerland': 'Europe',
    'Greece': 'Europe',
    'Russia': 'Europe',
    
    # Asia
    'Japan': 'Asia',
    'South Korea': 'Asia',
    'Taiwan': 'Asia',
    'Thailand': 'Asia',
    'Indonesia': 'Asia',
    'Malaysia': 'Asia',
    'Singapore': 'Asia',
    'Philippines': 'Asia',
    'India': 'Asia',
    
    # Oceania
    'Australia': 'Oceania',
    'New Zealand': 'Oceania',
    
    # Africa
    'South Africa': 'Africa',
    'Kenya': 'Africa',
    'Nigeria': 'Africa',
    'Ghana': 'Africa',
    'Senegal': 'Africa',
    'Uganda': 'Africa',
    'Botswana': 'Africa',
}

