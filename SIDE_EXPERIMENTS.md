# Side Experiments Setup Guide

This guide explains how to run **Side Experiment A** (Edge-Augmented CNN) and **Side Experiment D** (Line Features) - two experiments that use classic computer vision techniques from your course.

---

## Quick Start

### Run Both Experiments
```bash
python run_side_experiments.py --data_dir ./compressed_dataset --epochs 20
```

### Run Individual Experiments

**Side A1: Sobel Edge Channels**
```bash
python train.py --model edge --edge_type sobel --data_dir ./compressed_dataset \
  --epochs 20 --batch_size 64 --weighted_sampler --use_class_weights --num_workers 0
```

**Side A2: Canny Edge Channels**
```bash
python train.py --model edge --edge_type canny --data_dir ./compressed_dataset \
  --epochs 20 --batch_size 64 --weighted_sampler --use_class_weights --num_workers 0
```

**Side D: Line Features**
```bash
python train.py --model lines --data_dir ./compressed_dataset \
  --epochs 20 --batch_size 64 --weighted_sampler --use_class_weights --num_workers 0
```

---

## Side Experiment A: Edge-Augmented CNN

### What It Does
Adds edge information as additional input channels to help the model identify country-specific features like:
- Road markings and lane boundaries
- Skyline geometry (architecture)
- Signage edges

### Computer Vision Techniques
- **Sobel Operator:** Computes image gradients (Gx, Gy) and encodes orientation as cos(θ) and sin(θ)
- **Canny Edge Detection:** Produces a binary edge map

### Architecture Changes
- **Sobel:** Input changes from 3 channels (RGB) → 5 channels (RGB + cos(θ) + sin(θ))
- **Canny:** Input changes from 3 channels (RGB) → 4 channels (RGB + edge map)
- First convolutional layer adjusted to accept the new channel count

### Expected Results
- **Hypothesis:** Edge channels should provide +1-3 percentage points improvement
- **Why:** Countries differ in road markings, lane geometry, and architectural edges
- **Baseline to Beat:** GeoCNN-Base (43.06% Top-1)

---

## Side Experiment D: Line Features

### What It Does
Extracts line orientation histograms using Hough Transform and fuses them with CNN features to capture:
- Lane geometry patterns
- Power line orientations
- Architectural line structures
- Skyline edges

### Computer Vision Techniques
1. **Canny Edge Detection:** Find edges in the image
2. **Probabilistic Hough Transform:** Detect lines and their orientations
3. **Histogram:** 16-bin histogram of line orientations θ ∈ [0, π)
4. **Statistics:** Mean orientation, circular variance, fraction of long lines

### Architecture Changes
- CNN backbone extracts 256-d visual features
- Line features (20-d: 16 bins + 4 stats) processed through MLP → 64-d
- Features concatenated: 256 + 64 = 320-d
- Final classifier: 320 → num_classes

### Expected Results
- **Hypothesis:** Line features may help countries with distinctive lane geometry or architecture
- **Why:** Different regions have different road layouts and building styles
- **Baseline to Beat:** GeoCNN-Base (43.06% Top-1)

---

## Course Topics Covered

| Experiment | CV Techniques | Course Topics |
|------------|--------------|---------------|
| **Side A** | Sobel gradients, Canny edges | Filters, Derivatives, Edges |
| **Side D** | Canny + Hough Transform | Line Detection, Hough Transform |

---

## Outputs

Each experiment will create:
- Model checkpoints: `outputs/geocnn_edge_{type}_TIMESTAMP/` or `outputs/geocnn_lines_TIMESTAMP/`
- Training plots: Loss curves, accuracy curves, F1 scores
- Confusion matrices
- Per-class recall charts
- Summary statistics

---

## Comparison with Baseline

After running experiments, compare results:

| Model | Top-1 Acc | Top-5 Acc | Macro F1 | Improvement |
|-------|-----------|-----------|----------|-------------|
| GeoCNN-Base | 43.06% | 74.09% | 0.1218 | Baseline |
| GeoCNN-Edge (Sobel) | ? | ? | ? | vs Base |
| GeoCNN-Edge (Canny) | ? | ? | ? | vs Base |
| GeoCNN-Lines | ? | ? | ? | vs Base |

---

## Troubleshooting

### Edge Model Issues
- **Slow data loading:** Edge extraction happens on-the-fly. Consider pre-computing if too slow.
- **Memory issues:** Reduce batch size if running out of memory.

### Lines Model Issues
- **Very slow:** Line feature extraction (Canny + Hough) is computationally expensive.
  - Consider reducing batch size
  - Or pre-compute line features and save to disk
- **No lines detected:** Some images may have few/no lines. The model handles this with zero features.

---

## Next Steps

1. **Run experiments** (use commands above)
2. **Compare results** with GeoCNN-Base
3. **Analyze:** Did edges help? Did lines help?
4. **Update research_progress.md** with findings
5. **Generate comparison plots** for your paper

---

## For Your Paper

These experiments demonstrate:
- **Classical CV techniques** (edges, lines) can complement deep learning
- **Feature engineering** still has value in modern pipelines
- **Multi-modal fusion** (visual + geometric features) can improve performance

Perfect for your "Experiments and Results" section!

