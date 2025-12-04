"""
Visualization utilities for GeoGuessr experiments.

Includes:
- Confusion matrix plots
- Training history plots
- Per-class recall bar charts
- Grad-CAM visualizations
- Failure case analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import json


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str = None,
    figsize: tuple = (12, 10),
    normalize: bool = True,
    title: str = 'Confusion Matrix'
):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix (N x N)
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
        normalize: Whether to normalize by row (true class)
        title: Plot title
    """
    if normalize:
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)
        fmt = '.2f'
    else:
        cm_norm = cm
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # For many classes, don't show annotations
    annot = len(class_names) <= 20
    
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()
    return fig


def plot_continent_confusion(
    cm: np.ndarray,
    class_names: List[str],
    country_to_continent: Dict[str, str],
    save_path: str = None,
    figsize: tuple = (10, 8)
):
    """
    Plot confusion matrix collapsed by continent.
    
    Args:
        cm: Country-level confusion matrix
        class_names: Country names
        country_to_continent: Mapping from country to continent
        save_path: Path to save figure
    """
    # Get unique continents
    continents = sorted(set(country_to_continent.values()))
    n_continents = len(continents)
    
    # Create continent-level confusion matrix
    continent_cm = np.zeros((n_continents, n_continents))
    
    for i, true_country in enumerate(class_names):
        for j, pred_country in enumerate(class_names):
            true_cont = country_to_continent.get(true_country, 'Unknown')
            pred_cont = country_to_continent.get(pred_country, 'Unknown')
            
            if true_cont in continents and pred_cont in continents:
                true_idx = continents.index(true_cont)
                pred_idx = continents.index(pred_cont)
                continent_cm[true_idx, pred_idx] += cm[i, j]
    
    # Plot
    plot_confusion_matrix(
        continent_cm.astype(int),
        continents,
        save_path=save_path,
        figsize=figsize,
        normalize=True,
        title='Continent-Level Confusion Matrix'
    )


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str = None,
    figsize: tuple = (14, 5)
):
    """
    Plot training history (loss, accuracy, F1).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'val_f1'
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, [a*100 for a in history['train_acc']], 'b-', label='Train')
    axes[1].plot(epochs, [a*100 for a in history['val_acc']], 'r-', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1 plot
    if 'val_f1' in history:
        axes[2].plot(epochs, history['val_f1'], 'g-', label='Val F1')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Macro F1')
        axes[2].set_title('Validation Macro F1')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.close()
    return fig


def plot_per_class_recall(
    per_class_recall: Dict[str, float],
    save_path: str = None,
    figsize: tuple = None,
    top_n: int = None
):
    """
    Plot per-class recall as horizontal bar chart.
    
    Args:
        per_class_recall: Dictionary mapping class name to recall
        save_path: Path to save figure
        figsize: Figure size (auto-calculated if None)
        top_n: Only show top and bottom N classes
    """
    # Sort by recall
    sorted_items = sorted(per_class_recall.items(), key=lambda x: x[1])
    
    if top_n and len(sorted_items) > 2 * top_n:
        # Show worst and best
        items_to_show = sorted_items[:top_n] + [('...', None)] + sorted_items[-top_n:]
    else:
        items_to_show = sorted_items
    
    # Filter out placeholder
    names = [item[0] for item in items_to_show if item[1] is not None]
    recalls = [item[1] * 100 for item in items_to_show if item[1] is not None]
    
    if figsize is None:
        figsize = (10, max(6, len(names) * 0.3))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color bars by recall value
    colors = plt.cm.RdYlGn(np.array(recalls) / 100)
    
    y_pos = np.arange(len(names))
    ax.barh(y_pos, recalls, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Recall (%)')
    ax.set_title('Per-Class Recall')
    ax.set_xlim(0, 100)
    
    # Add value labels
    for i, v in enumerate(recalls):
        ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-class recall saved to {save_path}")
    
    plt.close()
    return fig


def visualize_predictions(
    image_paths: List[str],
    true_labels: List[int],
    pred_labels: List[int],
    class_names: List[str],
    save_dir: str,
    n_correct: int = 10,
    n_incorrect: int = 10
):
    """
    Visualize correct and incorrect predictions.
    
    Args:
        image_paths: List of image paths
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: Class names
        save_dir: Directory to save visualizations
        n_correct: Number of correct predictions to show
        n_incorrect: Number of incorrect predictions to show
    """
    from PIL import Image
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Find correct and incorrect predictions
    correct_idx = [i for i in range(len(true_labels)) if true_labels[i] == pred_labels[i]]
    incorrect_idx = [i for i in range(len(true_labels)) if true_labels[i] != pred_labels[i]]
    
    # Sample
    if len(correct_idx) > n_correct:
        correct_idx = np.random.choice(correct_idx, n_correct, replace=False)
    if len(incorrect_idx) > n_incorrect:
        incorrect_idx = np.random.choice(incorrect_idx, n_incorrect, replace=False)
    
    # Plot correct predictions
    if len(correct_idx) > 0:
        n_cols = min(5, len(correct_idx))
        n_rows = (len(correct_idx) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        axes = np.array(axes).flatten() if n_rows * n_cols > 1 else [axes]
        
        for ax, idx in zip(axes, correct_idx):
            img = Image.open(image_paths[idx])
            ax.imshow(img)
            ax.set_title(f'True: {class_names[true_labels[idx]]}', fontsize=8, color='green')
            ax.axis('off')
        
        for ax in axes[len(correct_idx):]:
            ax.axis('off')
        
        plt.suptitle('Correct Predictions', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'correct_predictions.png'), dpi=150)
        plt.close()
    
    # Plot incorrect predictions
    if len(incorrect_idx) > 0:
        n_cols = min(5, len(incorrect_idx))
        n_rows = (len(incorrect_idx) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        axes = np.array(axes).flatten() if n_rows * n_cols > 1 else [axes]
        
        for ax, idx in zip(axes, incorrect_idx):
            img = Image.open(image_paths[idx])
            ax.imshow(img)
            ax.set_title(
                f'True: {class_names[true_labels[idx]]}\nPred: {class_names[pred_labels[idx]]}',
                fontsize=7, color='red'
            )
            ax.axis('off')
        
        for ax in axes[len(incorrect_idx):]:
            ax.axis('off')
        
        plt.suptitle('Incorrect Predictions (Failure Cases)', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'incorrect_predictions.png'), dpi=150)
        plt.close()
    
    print(f"Prediction visualizations saved to {save_dir}")


def create_gradcam_visualization(
    model,
    image_path: str,
    target_layer: str,
    class_idx: int = None,
    save_path: str = None
):
    """
    Create Grad-CAM visualization for a single image.
    
    Args:
        model: PyTorch model
        image_path: Path to image
        target_layer: Name of target layer for Grad-CAM
        class_idx: Target class (None for predicted class)
        save_path: Path to save visualization
    """
    import torch
    from PIL import Image
    from torchvision import transforms
    
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("Please install grad-cam: pip install grad-cam")
        return None
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img).unsqueeze(0)
    
    # Get the target layer
    target_layers = [dict(model.named_modules())[target_layer]]
    
    # Create Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Generate CAM
    if class_idx is not None:
        targets = [class_idx]
    else:
        targets = None
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Prepare original image for overlay
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    
    # Create visualization
    visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img_resized)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(grayscale_cam, cmap='jet')
    axes[1].set_title('Grad-CAM')
    axes[1].axis('off')
    
    axes[2].imshow(visualization)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grad-CAM visualization saved to {save_path}")
    
    plt.close()
    return fig


def create_results_table(
    results: Dict[str, Dict],
    save_path: str = None
):
    """
    Create a comparison table of all experiment results.
    
    Args:
        results: Dictionary mapping model name to results dict
        save_path: Path to save as CSV
    """
    import pandas as pd
    
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            'Model': model_name,
            'Top-1 Acc': f"{metrics.get('top1_accuracy', 0)*100:.2f}%",
            'Top-5 Acc': f"{metrics.get('top5_accuracy', 0)*100:.2f}%",
            'Macro F1': f"{metrics.get('macro_f1', 0):.4f}",
            'Params': metrics.get('params', 'N/A')
        })
    
    df = pd.DataFrame(rows)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Results table saved to {save_path}")
    
    return df


if __name__ == "__main__":
    # Test visualization functions
    print("Visualization module loaded successfully.")
    
    # Create dummy data for testing
    n_classes = 10
    class_names = [f'Class_{i}' for i in range(n_classes)]
    
    # Dummy confusion matrix
    cm = np.random.randint(0, 50, (n_classes, n_classes))
    np.fill_diagonal(cm, np.random.randint(50, 100, n_classes))
    
    # Dummy history
    history = {
        'train_loss': [2.0 - i*0.1 for i in range(20)],
        'val_loss': [2.1 - i*0.08 for i in range(20)],
        'train_acc': [0.2 + i*0.03 for i in range(20)],
        'val_acc': [0.15 + i*0.025 for i in range(20)],
        'val_f1': [0.15 + i*0.025 for i in range(20)]
    }
    
    # Dummy per-class recall
    per_class_recall = {name: np.random.uniform(0.3, 0.95) for name in class_names}
    
    print("Test plots can be created with:")
    print("  plot_confusion_matrix(cm, class_names, 'test_cm.png')")
    print("  plot_training_history(history, 'test_history.png')")
    print("  plot_per_class_recall(per_class_recall, 'test_recall.png')")

