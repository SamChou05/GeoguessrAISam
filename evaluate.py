"""
Evaluation script for GeoGuessr classification models.

Computes:
- Top-1 and Top-5 accuracy
- Macro-averaged F1 score
- Per-class recall
- Confusion matrix
- Continent-level confusion (if mapping provided)

Usage:
    python evaluate.py --checkpoint outputs/model/best_model.pth --data_dir ./dataset
"""

import os
import argparse
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, top_k_accuracy_score
)

from models import GeoCNNBase, GeoCNNEdge, GeoCNNLines, GeoCNNSegmentation
from data.dataset import load_dataset_from_directory, create_splits, GeoDataset
from data.transforms import get_val_transforms, FiveCropTTA, MultiScaleTTA


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(checkpoint_path: str, config_path: str, device: torch.device) -> nn.Module:
    """Load model from checkpoint."""
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    num_classes = config['num_classes']
    model_type = config.get('model', 'base')
    
    # Create model
    if model_type == 'base':
        model = GeoCNNBase(num_classes=num_classes, dropout_p=config.get('dropout', 0.3))
    elif model_type == 'edge':
        edge_type = config.get('edge_type', 'sobel')
        model = GeoCNNEdge(num_classes=num_classes, edge_type=edge_type)
    else:
        model = GeoCNNBase(num_classes=num_classes)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list,
    use_tta: bool = False
) -> dict:
    """
    Evaluate model on test set.
    
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    all_paths = []
    
    for batch in tqdm(test_loader, desc='Evaluating'):
        if len(batch) == 3:
            images, labels, paths = batch
            all_paths.extend(paths)
        else:
            images, labels = batch
        
        images = images.to(device)
        
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    top1_acc = accuracy_score(all_labels, all_preds)
    top5_acc = top_k_accuracy_score(all_labels, all_probs, k=5)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Per-class metrics
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Per-class recall
    per_class_recall = {}
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class_recall[name] = (all_preds[mask] == i).mean()
        else:
            per_class_recall[name] = 0.0
    
    # Find top error pairs
    error_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                error_pairs.append((class_names[i], class_names[j], int(cm[i, j])))
    error_pairs.sort(key=lambda x: x[2], reverse=True)
    
    results = {
        'top1_accuracy': float(top1_acc),
        'top5_accuracy': float(top5_acc),
        'macro_f1': float(macro_f1),
        'per_class_recall': per_class_recall,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'top_error_pairs': error_pairs[:20],  # Top 20 error pairs
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist()
    }
    
    return results


def evaluate_with_tta(
    model: nn.Module,
    test_paths: list,
    test_labels: list,
    device: torch.device,
    class_names: list,
    tta_type: str = 'five_crop'
) -> dict:
    """
    Evaluate with Test Time Augmentation.
    
    Args:
        model: Trained model
        test_paths: List of test image paths
        test_labels: List of test labels
        device: Device to use
        class_names: List of class names
        tta_type: 'five_crop' or 'multi_scale'
    """
    model.eval()
    
    if tta_type == 'five_crop':
        tta_transform = FiveCropTTA()
    else:
        tta_transform = MultiScaleTTA()
    
    all_preds = []
    all_probs = []
    
    from PIL import Image
    
    for path in tqdm(test_paths, desc=f'Evaluating with {tta_type} TTA'):
        image = Image.open(path).convert('RGB')
        crops = tta_transform(image)  # Shape: (N, C, H, W)
        
        # Move to device
        crops = crops.to(device)
        
        # Get predictions for all crops
        with torch.no_grad():
            outputs = model(crops)
            probs = torch.softmax(outputs, dim=1)
        
        # Average predictions
        avg_probs = probs.mean(dim=0)
        pred = avg_probs.argmax().item()
        
        all_preds.append(pred)
        all_probs.append(avg_probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(test_labels)
    
    # Compute metrics
    top1_acc = accuracy_score(all_labels, all_preds)
    top5_acc = top_k_accuracy_score(all_labels, all_probs, k=5)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return {
        'top1_accuracy': float(top1_acc),
        'top5_accuracy': float(top5_acc),
        'macro_f1': float(macro_f1),
        'tta_type': tta_type
    }


def print_results(results: dict, class_names: list):
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nTop-1 Accuracy: {results['top1_accuracy']*100:.2f}%")
    print(f"Top-5 Accuracy: {results['top5_accuracy']*100:.2f}%")
    print(f"Macro F1 Score: {results['macro_f1']:.4f}")
    
    print("\n" + "-" * 40)
    print("Per-Class Recall (sorted by recall):")
    print("-" * 40)
    
    sorted_recall = sorted(results['per_class_recall'].items(), key=lambda x: x[1])
    for name, recall in sorted_recall[:10]:
        print(f"  {name}: {recall*100:.1f}%")
    print("  ...")
    for name, recall in sorted_recall[-5:]:
        print(f"  {name}: {recall*100:.1f}%")
    
    print("\n" + "-" * 40)
    print("Top Error Pairs (True → Predicted):")
    print("-" * 40)
    for true_cls, pred_cls, count in results['top_error_pairs'][:10]:
        print(f"  {true_cls} → {pred_cls}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate GeoGuessr models')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (JSON)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--use_tta', action='store_true',
                       help='Use test-time augmentation')
    parser.add_argument('--tta_type', type=str, default='five_crop',
                       choices=['five_crop', 'multi_scale'],
                       help='TTA type')
    
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load config
    checkpoint_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(checkpoint_dir, 'config.json')
    
    # Load model
    model, config = load_model(args.checkpoint, config_path, device)
    class_names = config['class_names']
    print(f"Loaded model with {len(class_names)} classes")
    
    # Load test data
    image_paths, labels, _ = load_dataset_from_directory(args.data_dir)
    _, _, (test_paths, test_labels) = create_splits(image_paths, labels)
    
    if args.use_tta:
        # Evaluate with TTA
        results = evaluate_with_tta(
            model, test_paths, test_labels, device, class_names, args.tta_type
        )
        print(f"\nResults with {args.tta_type} TTA:")
        print(f"Top-1 Accuracy: {results['top1_accuracy']*100:.2f}%")
        print(f"Top-5 Accuracy: {results['top5_accuracy']*100:.2f}%")
        print(f"Macro F1 Score: {results['macro_f1']:.4f}")
    else:
        # Standard evaluation
        test_transform = get_val_transforms(input_size=config.get('input_size', 224))
        test_dataset = GeoDataset(test_paths, test_labels, class_names, 
                                  transform=test_transform, return_path=True)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers
        )
        
        results = evaluate_model(model, test_loader, device, class_names)
        print_results(results, class_names)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

