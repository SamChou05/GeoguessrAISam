"""
Script to run Side Experiments A and D:
- Side A: Edge-Augmented CNN (Sobel and Canny)
- Side D: Line Features with Hough Transform

These experiments test classic computer vision techniques:
- Derivatives & Edges (Sobel/Canny)
- Line Detection (Hough Transform)

Usage:
    python run_side_experiments.py --data_dir ./compressed_dataset --epochs 20
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime


def run_experiment(model_type, edge_type=None, data_dir=None, epochs=20, batch_size=64):
    """Run a single experiment."""
    cmd = [
        sys.executable, 'train.py',
        '--model', model_type,
        '--data_dir', data_dir,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--weighted_sampler',
        '--use_class_weights',
        '--num_workers', '0'
    ]
    
    if edge_type:
        cmd.extend(['--edge_type', edge_type])
    
    print(f"\n{'='*70}")
    print(f"Running: {model_type.upper()}" + (f" ({edge_type})" if edge_type else ""))
    print(f"Command: {' '.join(cmd)}")
    print('='*70)
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run Side Experiments A and D')
    
    parser.add_argument('--data_dir', type=str, default='./compressed_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--skip_edge', action='store_true',
                       help='Skip edge experiments')
    parser.add_argument('--skip_lines', action='store_true',
                       help='Skip lines experiment')
    parser.add_argument('--edge_only', action='store_true',
                       help='Only run edge experiments')
    parser.add_argument('--lines_only', action='store_true',
                       help='Only run lines experiment')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SIDE EXPERIMENTS: Edge-Augmented CNN & Line Features")
    print("="*70)
    print(f"Dataset: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print("="*70)
    
    results = {}
    
    # Side Experiment A: Edge-Augmented CNN
    if not args.skip_edge and not args.lines_only:
        print("\n" + "#"*70)
        print("# SIDE EXPERIMENT A: Edge-Augmented CNN")
        print("#"*70)
        print("\nThis experiment tests if edge information (derivatives) helps")
        print("by adding Sobel or Canny edge channels to the input.")
        print("\nTechniques: Sobel gradients, Canny edge detection")
        print("Course Topics: Filters, Derivatives, Edges")
        
        # Sobel edges
        print("\n--- Experiment A1: Sobel Edge Channels (RGB + cos(θ) + sin(θ)) ---")
        success_sobel = run_experiment(
            'edge', 'sobel', args.data_dir, args.epochs, args.batch_size
        )
        results['edge_sobel'] = success_sobel
        
        # Canny edges
        print("\n--- Experiment A2: Canny Edge Channels (RGB + edge map) ---")
        success_canny = run_experiment(
            'edge', 'canny', args.data_dir, args.epochs, args.batch_size
        )
        results['edge_canny'] = success_canny
    
    # Side Experiment D: Line Features
    if not args.skip_lines and not args.edge_only:
        print("\n" + "#"*70)
        print("# SIDE EXPERIMENT D: Line Orientation Histogram")
        print("#"*70)
        print("\nThis experiment tests if line geometry helps by extracting")
        print("line orientation histograms using Hough Transform.")
        print("\nTechniques: Canny edge detection, Hough Transform, Line orientation")
        print("Course Topics: Line Detection, Hough Transform")
        
        print("\n--- Experiment D: Line Features Fusion ---")
        success_lines = run_experiment(
            'lines', None, args.data_dir, args.epochs, args.batch_size
        )
        results['lines'] = success_lines
    
    # Summary
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print("="*70)
    
    for exp_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{exp_name:20s}: {status}")
    
    print("\n" + "="*70)
    print("Next Steps:")
    print("1. Compare results with GeoCNN-Base (43.06% Top-1)")
    print("2. Check if edge channels improved accuracy")
    print("3. Check if line features helped")
    print("4. Generate comparison plots")
    print("="*70)


if __name__ == '__main__':
    main()

