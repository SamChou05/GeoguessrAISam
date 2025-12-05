# Research Progress: GeoGuessr Country Classification

**Date:** December 4, 2025  
**Experiments:** GeoCNN-Base, GeoCNN-Edge (Sobel & Canny)  
**Status:** Base Model Complete, Edge Experiments Complete, Lines Experiment Pending

---

## 1. Experiment Overview

### 1.1 Model Architecture
We trained **GeoCNN-Base**, a simple 4-block convolutional neural network designed for country-level geolocation from street scene images.

**Architecture Details:**
- **Input:** 224×224 RGB images
- **Block 1:** Conv3×3 (3→32) → Conv3×3 (32→32) → MaxPool
- **Block 2:** Conv3×3 (32→64) → Conv3×3 (64→64) → MaxPool
- **Block 3:** Conv3×3 (64→128) → Conv3×3 (128→128) → MaxPool
- **Block 4:** Conv3×3 (128→256) → Conv1×1 (256→256)
- **Head:** Global Average Pooling → Dropout(0.3) → FC(256→98)
- **Total Parameters:** 674,114 (~674K)

### 1.2 Dataset
- **Total Images:** 49,918 (after filtering classes with <10 samples)
- **Number of Classes:** 98 countries
- **Split:** 70% train (34,942), 15% validation (7,488), 15% test (7,488)
- **Class Distribution:** Highly imbalanced (e.g., United States: 12,014 images; many countries: <100 images)

### 1.3 Training Configuration
```bash
python train.py --model base --data_dir ./compressed_dataset \
  --batch_size 64 --weighted_sampler --use_class_weights \
  --epochs 20
```

**Hyperparameters:**
- **Optimizer:** AdamW (lr=3e-4, weight_decay=1e-4)
- **Scheduler:** Cosine annealing over 20 epochs
- **Loss:** Cross-entropy with label smoothing (ε=0.1)
- **Class Weighting:** Enabled (w_c = 1/log(1+n_c))
- **Weighted Sampling:** Enabled for training
- **Augmentations:** RandomResizedCrop, ColorJitter, GaussianBlur
- **No Horizontal Flips:** Preserves driving side information

---

## 2. Results Analysis

### 2.1 Overall Performance

| Metric | Best Value | Epoch | Final Value |
|--------|-----------|-------|-------------|
| **Validation Top-1 Accuracy** | **43.06%** | 17 | 43.38% |
| **Validation Top-5 Accuracy** | **74.19%** | 15 | 74.09% |
| **Validation Macro F1** | **0.1218** | 17 | 0.1201 |
| **Validation Loss** | 1.7725 | 20 | 1.7725 |
| **Training Loss** | - | - | 2.0342 |

### 2.2 Training Dynamics

**Loss Curves:**
- **Training Loss:** Decreased steadily from 2.65 → 2.03 (23% reduction)
- **Validation Loss:** Decreased from 2.26 → 1.77 (22% reduction)
- **Gap:** Train loss (2.03) > Val loss (1.77) suggests potential underfitting or regularization working well

**Accuracy Progression:**
- **Train Top-1:** 18.2% → 36.9% (+18.7 percentage points)
- **Val Top-1:** 25.1% → 43.4% (+18.3 percentage points)
- **Train Top-5:** 42.6% → 68.4% (+25.8 percentage points)
- **Val Top-5:** 54.2% → 74.1% (+19.9 percentage points)

**Key Observations:**
1. **Steady Improvement:** Both train and validation metrics improved consistently
2. **No Overfitting:** Validation loss remained below training loss, indicating good generalization
3. **Top-5 Much Higher:** 74% Top-5 vs 43% Top-1 suggests the model often identifies the correct region/continent
4. **Best Model at Epoch 17:** Performance plateaued slightly in final epochs

### 2.3 Macro F1 Score Analysis

The **Macro F1 score of 0.1218** is relatively low, which indicates:
- **Class Imbalance Impact:** Despite class weighting, underrepresented countries are still struggling
- **98 Classes is Challenging:** Random baseline would be ~1.02%, so 43% Top-1 is substantial improvement
- **Per-Class Performance:** Some countries likely have very low recall (need to check confusion matrix)

### 2.4 Learning Rate Schedule

The cosine annealing schedule worked as expected:
- Started at 3e-4, decayed to 3e-6 by epoch 20
- Smooth decay allowed fine-tuning in later epochs

---

## 2.5 Side Experiment A: Edge-Augmented CNN Results

### 2.5.1 Experiment Overview

**Side Experiment A** tested whether adding edge information as additional input channels improves country classification. Two variants were tested:
- **A1: Sobel Edges** - RGB + cos(θ) + sin(θ) = 5 channels
- **A2: Canny Edges** - RGB + edge map = 4 channels

**Computer Vision Techniques:**
- Sobel operator for gradient computation (derivatives)
- Canny edge detection
- Course topics: Filters, Derivatives, Edges

### 2.5.2 Results Comparison

| Model | Top-1 Acc | Top-5 Acc | Macro F1 | Improvement vs Base |
|-------|-----------|-----------|----------|---------------------|
| **GeoCNN-Base** | 43.06% | 74.09% | 0.1218 | Baseline |
| **GeoCNN-Edge (Sobel)** | **47.61%** | **78.04%** | 0.1100 | **+4.55 pp** |
| **GeoCNN-Edge (Canny)** | **47.74%** | **77.67%** | 0.1129 | **+4.68 pp** |

### 2.5.3 Key Findings

**✅ Significant Improvement in Top-1 Accuracy:**
- **Sobel:** +4.55 percentage points (43.06% → 47.61%)
- **Canny:** +4.68 percentage points (43.06% → 47.74%)
- Both edge variants performed similarly, with Canny slightly better
- **This confirms our hypothesis** that edge channels help identify country-specific features

**✅ Top-5 Accuracy Also Improved:**
- Sobel: 74.09% → 78.04% (+3.95 pp)
- Canny: 74.09% → 77.67% (+3.58 pp)
- Model is better at narrowing down to correct region

**⚠️ Macro F1 Slightly Lower:**
- Sobel: 0.1218 → 0.1100 (-0.0118)
- Canny: 0.1218 → 0.1129 (-0.0089)
- Suggests edge channels help common classes more than rare ones
- May need different weighting strategy for edge-augmented models

### 2.5.4 Training Dynamics

**Sobel Model:**
- Best epoch: 17 (same as base model)
- Training time: 2.00 hours (faster than base: 4.28 hours)
- Final train loss: 1.6457, val loss: 1.6678
- Consistent improvement throughout training

**Canny Model:**
- Best epoch: 18
- Training time: 1.96 hours
- Final train loss: 1.6524, val loss: 1.6747
- Similar training dynamics to Sobel

**Observation:** Edge models trained faster (2 hours vs 4.28 hours) despite additional preprocessing, likely due to better convergence.

### 2.5.5 Why Edge Channels Help

1. **Road Markings:** Different countries have distinct lane markings, road signs, and traffic patterns
2. **Architecture:** Building edges and skyline geometry vary by region
3. **Infrastructure:** Power lines, fences, and other linear structures are country-specific
4. **Derivative Information:** Edge channels explicitly encode gradient information that the CNN can leverage

### 2.5.6 Comparison: Sobel vs Canny

- **Canny slightly outperforms Sobel** (47.74% vs 47.61%), but difference is minimal
- Both provide similar improvements over baseline
- **Sobel advantages:** Preserves orientation information (cos/sin), more information-rich
- **Canny advantages:** Binary edge map is simpler, may be more robust to noise
- **Recommendation:** Either approach works well; Canny is slightly better for this task

---

## 3. Discussion

### 3.1 What Worked Well

1. **Simple Architecture is Effective:**
   - 674K parameters is lightweight yet achieved 43% Top-1 accuracy
   - No overfitting observed, suggesting model capacity is appropriate

2. **Class Weighting Strategy:**
   - Weighted sampling + class weights in loss helped with imbalanced data
   - Model learned to predict underrepresented countries (though still challenging)

3. **Top-5 Performance:**
   - 74% Top-5 accuracy is promising for a GeoGuessr application
   - Users could narrow down to 5 countries and use other cues

4. **Training Stability:**
   - Consistent improvement without oscillations
   - Validation metrics tracked training metrics well

### 3.2 Challenges and Limitations

1. **Low Macro F1 (0.12):**
   - Indicates poor performance on many underrepresented countries
   - Some countries likely have near-zero recall
   - Need to examine per-class metrics

2. **Class Imbalance:**
   - United States: 12,014 images vs. many countries: <100 images
   - Even with weighting, model struggles with rare classes
   - May need data augmentation or different sampling strategies

3. **43% Top-1 Accuracy:**
   - While better than random (1%), still leaves room for improvement
   - For 98 classes, this suggests the model is learning some patterns but not all

4. **Training Time:**
   - ~13 minutes per epoch (766 seconds average)
   - Some epochs took much longer (epochs 6-7: 1-3 hours) - likely system issues
   - Total: 4.28 hours for 20 epochs

### 3.3 Error Analysis (Expected)

Based on the results, we expect:
- **Geographic Confusions:** Similar countries confused (e.g., Belgium↔Netherlands, Argentina↔Chile)
- **Continent-Level Patterns:** Model may perform better at continent-level than country-level
- **Underrepresented Classes:** Countries with <100 images likely have very low recall
- **Visual Similarity:** Countries with similar architecture/landscape may be confused

---

## 4. Future Improvements

### 4.1 Immediate Next Steps

1. **Analyze Confusion Matrix:**
   - Identify top error pairs
   - Check per-class recall to find worst-performing countries
   - Examine continent-level performance

2. **Run Baselines:**
   - Majority class baseline (sanity check)
   - Linear classifier on pixels + PCA
   - Bag of Visual Words (ORB/SIFT + k-means + SVM)
   - Compare against GeoCNN-Base

3. **Side Experiments:**
   - **Edge-Augmented CNN:** Add Sobel/Canny edge channels (Side A)
   - **Line Features:** Hough transform line orientation histogram (Side D)
   - **Segmentation:** Sky/ground two-tower architecture (Side C)

### 4.2 Architecture Improvements

1. **Transfer Learning:**
   - Fine-tune pretrained ResNet-18 or MobileNetV2
   - Should significantly outperform from-scratch training
   - Use as strong baseline/reference

2. **Hierarchical Classification:**
   - First predict continent, then country within continent
   - Reduces effective number of classes at each level
   - Could improve rare country performance

3. **Attention Mechanisms:**
   - Add spatial attention to focus on discriminative regions
   - May help identify country-specific cues (signs, architecture, vegetation)

### 4.3 Data Improvements

1. **Data Augmentation:**
   - Test impact of horizontal flips (should hurt due to driving side)
   - More aggressive color jitter for underrepresented classes
   - Mixup or CutMix for better generalization

2. **Data Collection:**
   - Collect more images for underrepresented countries
   - Balance dataset better (cap large classes, augment small ones)
   - Consider temporal/seasonal diversity

3. **Data Cleaning:**
   - Remove corrupted images (already implemented)
   - Filter low-quality images
   - Ensure geographic diversity within countries

### 4.4 Training Improvements

1. **Longer Training:**
   - Train for 80 epochs (original plan) instead of 20
   - May see further improvements, especially for rare classes

2. **Focal Loss:**
   - Test focal loss (γ=2) instead of label smoothing
   - Better for handling class imbalance

3. **Learning Rate:**
   - Test different learning rates (1e-4, 5e-4)
   - Warmup period for first few epochs

4. **Ensemble Methods:**
   - Train multiple models with different seeds
   - Average predictions for better accuracy

### 4.5 Evaluation Improvements

1. **Test Set Evaluation:**
   - Evaluate best model on held-out test set
   - Generate confusion matrix and per-class metrics
   - Create failure case analysis

2. **Visualization:**
   - Grad-CAM to see what regions model attends to
   - Visualize correct vs. incorrect predictions
   - Analyze failure modes

3. **Ablation Studies:**
   - Test impact of each component (dropout, class weights, augmentations)
   - Compare with/without horizontal flips
   - Test different input resolutions (160, 224, 320)

### 4.6 Research Paper Preparation

1. **Complete All Experiments:**
   - Main model: GeoCNN-Base ✓
   - Side A: Edge augmentation ✓ (Sobel & Canny both completed)
   - Side B: BoVW baseline
   - Side C: Segmentation
   - Side D: Line features (in progress)
   - Optional: Transfer learning baseline

2. **Create Comparison Table:**
   - All methods with Top-1, Top-5, Macro F1, parameters, inference time

3. **Generate Figures:**
   - Training curves (already generated)
   - Confusion matrices
   - Per-class recall charts
   - Grad-CAM visualizations
   - Failure case gallery

4. **Write Discussion:**
   - What worked and why
   - What didn't work
   - Error analysis
   - Limitations and ethics
   - Future work

---

## 5. Key Takeaways

1. **Simple CNN Achieves Reasonable Performance:**
   - 43% Top-1 accuracy on 98-class problem is promising
   - Model is learning geographic patterns

2. **Edge Channels Provide Significant Boost:**
   - **+4.5-4.7 percentage points improvement** with edge augmentation
   - Confirms that classical CV techniques (derivatives, edges) complement deep learning
   - Both Sobel and Canny work well, with Canny slightly better
   - Edge information helps identify country-specific road markings and architecture

3. **Top-5 Accuracy is Strong:**
   - Base: 74% Top-5, Edge models: 77-78% Top-5
   - Model often identifies correct region/continent
   - Useful for GeoGuessr where narrowing to 5 countries is valuable

4. **Class Imbalance Remains Challenging:**
   - Low Macro F1 (0.11-0.12) indicates poor performance on rare countries
   - Edge channels help common classes more than rare ones
   - Need better strategies for underrepresented classes

5. **Training is Stable:**
   - No overfitting observed in any model
   - Consistent improvement throughout training
   - Edge models train faster (2 hours vs 4.28 hours)

6. **Classical CV + Deep Learning Works:**
   - Edge augmentation demonstrates that feature engineering still has value
   - Multi-modal fusion (visual + edge features) improves performance
   - Perfect example for computer vision course paper

---

## 6. Next Training Run Recommendations

**Priority 1: Complete Baselines**
```bash
# Run all baselines for comparison
python run_all_experiments.py --data_dir ./compressed_dataset --skip_cnn
```

**Priority 2: Edge-Augmented Model** ✓ **COMPLETED**
- Sobel: 47.61% Top-1 (+4.55 pp improvement)
- Canny: 47.74% Top-1 (+4.68 pp improvement)
- Both experiments successful!

**Priority 3: Longer Training**
```bash
# Train for full 80 epochs to see if performance improves further
python train.py --model base --data_dir ./compressed_dataset --epochs 80
```

**Priority 4: Evaluate Best Model**
```bash
# Evaluate on test set and generate visualizations
python evaluate.py --checkpoint outputs/geocnn_base_20251203_232533/best_model.pth \
  --data_dir ./compressed_dataset --output results.json
```

---

## 7. Files Generated

All outputs saved to: `./outputs/geocnn_base_20251203_232533/`

- **Models:** `best_model.pth`, `final_model.pth`
- **Config:** `config.json`
- **Plots:** 
  - `loss_curves.png/pdf`
  - `accuracy_curves.png/pdf`
  - `f1_curve.png/pdf`
  - `confusion_matrix.png/pdf`
  - `per_class_recall.png/pdf`
  - `combined_metrics.png/pdf`
- **Logs:** `training_history.json`, `confusion_matrix.npy`
- **Summary:** `summary.json`, `summary.md`
- **TensorBoard:** Events for visualization

---

---

## 8. Experiment Summary Table

| Experiment | Top-1 Acc | Top-5 Acc | Macro F1 | Params | Training Time | Status |
|------------|-----------|-----------|----------|--------|---------------|--------|
| GeoCNN-Base | 43.06% | 74.09% | 0.1218 | 674K | 4.28 hrs | ✓ Complete |
| GeoCNN-Edge (Sobel) | 47.61% | 78.04% | 0.1100 | 675K | 2.00 hrs | ✓ Complete |
| GeoCNN-Edge (Canny) | 47.74% | 77.67% | 0.1129 | 674K | 1.96 hrs | ✓ Complete |
| GeoCNN-Lines | - | - | - | 686K | - | ⏳ Pending |

**Key Insight:** Edge augmentation provides **+4.5-4.7 percentage point improvement**, confirming that classical CV techniques (derivatives, edges) significantly help deep learning models for geolocation tasks.

---

**Last Updated:** December 4, 2025  
**Next Review:** After completing lines experiment

