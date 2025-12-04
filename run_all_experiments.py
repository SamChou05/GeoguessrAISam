"""
Master script to run all experiments for GeoGuessr classification.

This script runs:
1. Baselines: Majority class, Linear pixels, BoVW
2. Main model: GeoCNN-Base
3. Side experiments: Edge, Lines, Segmentation
4. Ablations: Flip, loss variants, resolution

Usage:
    python run_all_experiments.py --data_dir ./dataset --output_dir ./experiments
"""

import os
import sys
import argparse
import json
import subprocess
from datetime import datetime
import pandas as pd
import numpy as np
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_command(cmd: list, description: str):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def run_majority_baseline(data_dir: str, output_dir: str):
    """Run majority class baseline."""
    from data.dataset import load_dataset_from_directory, create_splits
    from sklearn.metrics import accuracy_score, f1_score
    
    print("\n" + "="*60)
    print("Running Majority Class Baseline")
    print("="*60)
    
    # Load data
    image_paths, labels, class_names = load_dataset_from_directory(data_dir)
    _, _, (test_paths, test_labels) = create_splits(image_paths, labels)
    _, (_, train_labels), _ = create_splits(image_paths, labels)
    
    # Find majority class
    counter = Counter(train_labels)
    majority_class = counter.most_common(1)[0][0]
    
    # Predict majority class for all
    predictions = [majority_class] * len(test_labels)
    
    # Compute metrics
    top1_acc = accuracy_score(test_labels, predictions)
    macro_f1 = f1_score(test_labels, predictions, average='macro')
    
    results = {
        'model': 'Majority Class',
        'top1_accuracy': float(top1_acc),
        'top5_accuracy': float(top1_acc),  # Same for majority
        'macro_f1': float(macro_f1),
        'params': 0
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'majority_baseline.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Majority Class Baseline: Top-1 = {top1_acc*100:.2f}%, F1 = {macro_f1:.4f}")
    return results


def run_linear_baseline(data_dir: str, output_dir: str):
    """Run linear pixel baseline."""
    from data.dataset import load_dataset_from_directory, create_splits
    from baselines.linear_baseline import LinearPixelClassifier
    
    print("\n" + "="*60)
    print("Running Linear Pixel Baseline")
    print("="*60)
    
    # Load data
    image_paths, labels, class_names = load_dataset_from_directory(data_dir)
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = create_splits(
        image_paths, labels
    )
    
    # Train classifier
    classifier = LinearPixelClassifier(n_components=256)
    classifier.fit(train_paths, train_labels, class_names)
    
    # Evaluate
    results = classifier.evaluate(test_paths, test_labels)
    results['model'] = 'Linear Pixels + PCA + LogReg'
    results['params'] = 256 * len(class_names)  # Approximate
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    classifier.save(os.path.join(output_dir, 'linear_baseline.pkl'))
    
    # Save results (remove non-serializable items)
    save_results = {k: v for k, v in results.items() if k not in ['predictions', 'report']}
    with open(os.path.join(output_dir, 'linear_baseline.json'), 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"Linear Baseline: Top-1 = {results['top1_accuracy']*100:.2f}%, "
          f"Top-5 = {results['top5_accuracy']*100:.2f}%, F1 = {results['macro_f1']:.4f}")
    
    return save_results


def run_bovw_baseline(data_dir: str, output_dir: str, descriptor_type: str = 'orb'):
    """Run Bag of Visual Words baseline."""
    from data.dataset import load_dataset_from_directory, create_splits
    from baselines.bovw import BOVWClassifier
    
    print("\n" + "="*60)
    print(f"Running BoVW Baseline ({descriptor_type.upper()})")
    print("="*60)
    
    # Load data
    image_paths, labels, class_names = load_dataset_from_directory(data_dir)
    (train_paths, train_labels), _, (test_paths, test_labels) = create_splits(image_paths, labels)
    
    # Train classifier
    classifier = BOVWClassifier(
        descriptor_type=descriptor_type,
        n_clusters=256,
        classifier_type='svm'
    )
    classifier.fit(train_paths, train_labels, class_names)
    
    # Evaluate
    results = classifier.evaluate(test_paths, test_labels)
    results['model'] = f'BoVW ({descriptor_type.upper()}) + SVM'
    results['params'] = 256 * len(class_names)
    
    # Save
    bovw_dir = os.path.join(output_dir, f'bovw_{descriptor_type}')
    classifier.save(bovw_dir)
    
    save_results = {k: v for k, v in results.items() if k not in ['predictions', 'report']}
    with open(os.path.join(bovw_dir, 'results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"BoVW Baseline: Top-1 = {results['top1_accuracy']*100:.2f}%, F1 = {results['macro_f1']:.4f}")
    
    return save_results


def run_cnn_experiment(
    data_dir: str,
    output_dir: str,
    model_type: str = 'base',
    **kwargs
):
    """Run CNN experiment using train.py."""
    cmd = [
        sys.executable, 'train.py',
        '--model', model_type,
        '--data_dir', data_dir,
        '--output_dir', output_dir,
        '--epochs', str(kwargs.get('epochs', 80)),
        '--batch_size', str(kwargs.get('batch_size', 64)),
        '--lr', str(kwargs.get('lr', 3e-4)),
        '--loss', kwargs.get('loss', 'smooth'),
        '--patience', str(kwargs.get('patience', 10))
    ]
    
    if kwargs.get('use_flip', False):
        cmd.append('--use_flip')
    if kwargs.get('weighted_sampler', True):
        cmd.append('--weighted_sampler')
    if kwargs.get('use_class_weights', True):
        cmd.append('--use_class_weights')
    if kwargs.get('use_amp', True):
        cmd.append('--use_amp')
    
    # Edge-specific
    if model_type == 'edge':
        cmd.extend(['--edge_type', kwargs.get('edge_type', 'sobel')])
    
    description = f"GeoCNN-{model_type.capitalize()}"
    if model_type == 'edge':
        description += f" ({kwargs.get('edge_type', 'sobel')})"
    
    success = run_command(cmd, description)
    return success


def create_summary_table(results_dir: str, output_path: str):
    """Create summary table from all experiment results."""
    all_results = []
    
    # Walk through results directory
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.json') and 'results' in file.lower() or 'baseline' in file.lower():
                path = os.path.join(root, file)
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        if 'top1_accuracy' in data:
                            all_results.append(data)
                except:
                    continue
    
    if not all_results:
        print("No results found to summarize.")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Format columns
    if 'top1_accuracy' in df.columns:
        df['Top-1 (%)'] = df['top1_accuracy'].apply(lambda x: f"{x*100:.2f}")
    if 'top5_accuracy' in df.columns:
        df['Top-5 (%)'] = df['top5_accuracy'].apply(lambda x: f"{x*100:.2f}")
    if 'macro_f1' in df.columns:
        df['Macro F1'] = df['macro_f1'].apply(lambda x: f"{x:.4f}")
    
    # Select and order columns
    cols = ['model', 'Top-1 (%)', 'Top-5 (%)', 'Macro F1', 'params']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"\nResults summary saved to {output_path}")
    print("\n" + df.to_string(index=False))
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Run all GeoGuessr experiments')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./experiments',
                       help='Output directory for all experiments')
    parser.add_argument('--epochs', type=int, default=80,
                       help='Number of training epochs for CNN models')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--skip_baselines', action='store_true',
                       help='Skip baseline experiments')
    parser.add_argument('--skip_cnn', action='store_true',
                       help='Skip CNN experiments')
    parser.add_argument('--skip_ablations', action='store_true',
                       help='Skip ablation studies')
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    all_results = {}
    
    # ===== BASELINES =====
    if not args.skip_baselines:
        print("\n" + "#"*60)
        print("# RUNNING BASELINES")
        print("#"*60)
        
        # Majority class
        try:
            results = run_majority_baseline(args.data_dir, os.path.join(output_dir, 'baselines'))
            all_results['majority'] = results
        except Exception as e:
            print(f"Majority baseline failed: {e}")
        
        # Linear pixels
        try:
            results = run_linear_baseline(args.data_dir, os.path.join(output_dir, 'baselines'))
            all_results['linear'] = results
        except Exception as e:
            print(f"Linear baseline failed: {e}")
        
        # BoVW ORB
        try:
            results = run_bovw_baseline(args.data_dir, os.path.join(output_dir, 'baselines'), 'orb')
            all_results['bovw_orb'] = results
        except Exception as e:
            print(f"BoVW ORB baseline failed: {e}")
    
    # ===== MAIN CNN EXPERIMENTS =====
    if not args.skip_cnn:
        print("\n" + "#"*60)
        print("# RUNNING CNN EXPERIMENTS")
        print("#"*60)
        
        # GeoCNN-Base (main model)
        run_cnn_experiment(
            args.data_dir,
            os.path.join(output_dir, 'cnn'),
            model_type='base',
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # GeoCNN-Edge (Sobel)
        run_cnn_experiment(
            args.data_dir,
            os.path.join(output_dir, 'cnn'),
            model_type='edge',
            edge_type='sobel',
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # GeoCNN-Edge (Canny)
        run_cnn_experiment(
            args.data_dir,
            os.path.join(output_dir, 'cnn'),
            model_type='edge',
            edge_type='canny',
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    # ===== ABLATION STUDIES =====
    if not args.skip_ablations:
        print("\n" + "#"*60)
        print("# RUNNING ABLATION STUDIES")
        print("#"*60)
        
        # With horizontal flip (should hurt)
        run_cnn_experiment(
            args.data_dir,
            os.path.join(output_dir, 'ablations'),
            model_type='base',
            use_flip=True,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Focal loss
        run_cnn_experiment(
            args.data_dir,
            os.path.join(output_dir, 'ablations'),
            model_type='base',
            loss='focal',
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Plain cross-entropy (no smoothing)
        run_cnn_experiment(
            args.data_dir,
            os.path.join(output_dir, 'ablations'),
            model_type='base',
            loss='ce',
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    # ===== CREATE SUMMARY =====
    print("\n" + "#"*60)
    print("# CREATING SUMMARY")
    print("#"*60)
    
    create_summary_table(output_dir, os.path.join(output_dir, 'results_summary.csv'))
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

