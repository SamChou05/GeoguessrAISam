# Learning to Geolocate: CNN with Classic CV Augmentations for Country-Level GeoGuessr

A computer vision research project implementing CNNs and classical CV techniques for country-level image geolocation.

## Project Overview

This project addresses the problem of predicting the country from a street scene image. It implements:

1. **Main Model**: GeoCNN-Base - A simple 4-block CNN (~1.2M parameters)
2. **Side Experiment A**: Edge-augmented CNN (Sobel/Canny channels)
3. **Side Experiment B**: Bag of Visual Words baseline (ORB/SIFT + k-means + SVM)
4. **Side Experiment C**: Sky/Ground segmentation with two-tower fusion
5. **Side Experiment D**: Line orientation histogram fusion

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dataset Structure

Organize your dataset as:
```
dataset/
├── Argentina/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Brazil/
│   └── ...
├── Japan/
│   └── ...
└── ...
```

## Quick Start

### Train the main model (GeoCNN-Base)
```bash
python train.py --model base --data_dir ./dataset --epochs 80
```

### Train with edge augmentation
```bash
python train.py --model edge --edge_type sobel --data_dir ./dataset
```

### Evaluate a trained model
```bash
python evaluate.py --checkpoint outputs/model/best_model.pth --data_dir ./dataset
```

### Run all experiments
```bash
python run_all_experiments.py --data_dir ./dataset --output_dir ./experiments
```

## Models

### GeoCNN-Base (Main Model)

Architecture for 224×224 RGB input:
- **Block 1**: Conv3×3 3→32, BN, ReLU; Conv3×3 32→32, BN, ReLU; MaxPool 2×2
- **Block 2**: Conv3×3 32→64, BN, ReLU; Conv3×3 64→64, BN, ReLU; MaxPool 2×2
- **Block 3**: Conv3×3 64→128, BN, ReLU; Conv3×3 128→128, BN, ReLU; MaxPool 2×2
- **Block 4**: Conv3×3 128→256, BN, ReLU; Conv1×1 256→256, BN, ReLU
- **Head**: Global Average Pool → Dropout(0.3) → FC(256→C)

Parameters: ~1.2M + 256×C

### GeoCNN-Edge

Same architecture but with 5-channel input (RGB + Sobel cos/sin) or 4-channel (RGB + Canny).

### GeoCNN-Lines

GeoCNN-Base backbone fused with 20-D line orientation features via MLP.

### GeoCNN-Segmentation

Two-tower architecture with shared weights processing sky-masked and ground-masked images.

## Training Configuration

Default hyperparameters (following the paper plan):
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4)
- **Scheduler**: Cosine annealing over 80 epochs
- **Batch size**: 64
- **Loss**: Cross-entropy with label smoothing (ε=0.1)
- **Augmentation**: RandomResizedCrop(224), ColorJitter(0.2), GaussianBlur(p=0.2)
- **No horizontal flip** (preserves driving side information)

## Augmentation Philosophy

Key design decisions for geolocation:
1. **NO horizontal flips** - Flipping changes driving side and sign orientation
2. **Minimal rotation** - Sun/shadow positions encode hemisphere
3. **Moderate color jitter** - Soil/vegetation color is a location cue
4. **RandomResizedCrop** - Simulates camera FoV changes

## Baselines

### 1. Majority Class
Simply predicts the most common class - sanity check baseline.

### 2. Linear Pixels + PCA
- Flatten 224×224×3 images
- PCA to 256 dimensions
- Logistic regression classifier

### 3. Bag of Visual Words (BoVW)
- ORB/SIFT keypoint detection
- Descriptor extraction and k-means clustering (K=256)
- TF-IDF weighted histograms
- Linear SVM classifier

## Experiments and Metrics

### Metrics
- Top-1 Accuracy
- Top-5 Accuracy
- Macro-averaged F1 Score
- Per-class Recall
- Confusion Matrix

### Ablation Studies
- Input resolution: 160 vs 224 vs 320
- Loss: CE vs Focal vs Label Smoothing
- With/without horizontal flips
- RGB vs RGB+Sobel vs RGB+Canny

## Project Structure

```
455/
├── models/
│   ├── __init__.py
│   ├── geocnn_base.py      # Main CNN model
│   ├── geocnn_edge.py      # Edge-augmented model
│   ├── geocnn_lines.py     # Line features fusion
│   └── geocnn_segmentation.py  # Sky/ground two-tower
├── data/
│   ├── __init__.py
│   ├── dataset.py          # Data loading utilities
│   ├── transforms.py       # Image transforms
│   └── augmentations.py    # Custom augmentations
├── baselines/
│   ├── __init__.py
│   ├── bovw.py            # Bag of Visual Words
│   └── linear_baseline.py # Linear pixel classifier
├── utils/
│   ├── __init__.py
│   └── visualization.py   # Plotting utilities
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── run_all_experiments.py # Run all experiments
├── requirements.txt
└── README.md
```

## Course Topics Covered

This project demonstrates:

| Topic | Implementation |
|-------|---------------|
| **Pixels & Filters** | Image preprocessing, normalization |
| **Derivatives & Edges** | Sobel gradients, Canny edge detection |
| **Detecting Lines** | Hough transform, line orientation histograms |
| **Keypoints & Corners** | ORB/SIFT detection |
| **Descriptors** | ORB (32-D binary), SIFT (128-D) |
| **Segmentation** | SLIC superpixels, sky/ground masking |
| **Clustering** | k-means for visual vocabulary |
| **Linear Classifiers** | SVM, Logistic Regression |
| **Backpropagation** | CNN training with PyTorch |
| **Recognition with CNNs** | GeoCNN architecture |
| **Object Detection** | Implicit via attention visualization |

## Expected Results

Based on the paper plan, expect:
- **GeoCNN-Base >> Linear baselines** in Top-1 accuracy
- **Edge augmentation**: +1-3 percentage points
- **Horizontal flips likely hurt** (driving side changes)
- **Pretrained models** (if used) will outperform from-scratch training

## Visualization Examples

Generate visualizations:
```python
from utils.visualization import (
    plot_confusion_matrix,
    plot_training_history,
    plot_per_class_recall,
    create_gradcam_visualization
)

# Plot confusion matrix
plot_confusion_matrix(cm, class_names, 'confusion_matrix.png')

# Plot training curves
plot_training_history(history, 'training_history.png')

# Grad-CAM visualization
create_gradcam_visualization(model, 'test_image.jpg', 'block4', save_path='gradcam.png')
```

## Citation

If you use this code for your research, please cite:
```
@misc{geoguesser_cnn_2024,
  title={Learning to Geolocate: A Simple CNN with Classic CV Augmentations for Country-Level GeoGuessr},
  author={Your Name},
  year={2024}
}
```

## License

MIT License

# GeoguessrAISam
