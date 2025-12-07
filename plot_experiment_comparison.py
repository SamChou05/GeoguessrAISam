"""
Generate comparison plots of top-1 accuracy curves across all experiments.

This script finds all completed experiments in the outputs directory,
loads their training histories, and creates comparison plots showing
top-1 accuracy curves labeled by experiment name.
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
plt.style.use('seaborn-v0_8-whitegrid')


def find_experiments(outputs_dir='./outputs', prefer_complete=True):
    """Find all experiment directories with training history files.
    
    Args:
        outputs_dir: Directory containing experiment outputs
        prefer_complete: If True, prefer experiments with summary.json (completed runs)
    """
    experiments = []
    
    # Find all directories in outputs
    for exp_dir in glob.glob(os.path.join(outputs_dir, '*')):
        if not os.path.isdir(exp_dir):
            continue
            
        history_path = os.path.join(exp_dir, 'logs', 'training_history.json')
        config_path = os.path.join(exp_dir, 'config.json')
        summary_path = os.path.join(exp_dir, 'summary.json')
        
        if os.path.exists(history_path) and os.path.exists(config_path):
            has_summary = os.path.exists(summary_path)
            experiments.append({
                'dir': exp_dir,
                'history_path': history_path,
                'config_path': config_path,
                'has_summary': has_summary
            })
    
    # If prefer_complete, sort so completed experiments come first
    # Then we can filter duplicates by keeping first occurrence
    if prefer_complete:
        experiments.sort(key=lambda x: (not x['has_summary'], x['dir']))
    
    # Filter duplicates: keep only one experiment per name
    seen_names = set()
    unique_experiments = []
    for exp in experiments:
        # Quick check of name without full load
        try:
            with open(exp['config_path'], 'r') as f:
                config = json.load(f)
            exp_name = config.get('experiment_name', '')
            if not exp_name:
                dir_name = os.path.basename(exp['dir'])
                if 'geocnn_base' in dir_name:
                    exp_name = 'GeoCNN-Base'
                elif 'geocnn_edge_sobel' in dir_name:
                    exp_name = 'GeoCNN-Edge (Sobel)'
                elif 'geocnn_edge_canny' in dir_name:
                    exp_name = 'GeoCNN-Edge (Canny)'
                elif 'geocnn_lines' in dir_name:
                    exp_name = 'GeoCNN-Lines'
                else:
                    exp_name = dir_name
            
            exp_name = exp_name.replace('GeoCNN-Edge-canny', 'GeoCNN-Edge (Canny)')
            exp_name = exp_name.replace('GeoCNN-Edge-sobel', 'GeoCNN-Edge (Sobel)')
            
            if exp_name not in seen_names:
                seen_names.add(exp_name)
                unique_experiments.append(exp)
        except:
            # If we can't read config, include it anyway
            unique_experiments.append(exp)
    
    return unique_experiments


def load_experiment_data(exp_info):
    """Load experiment data from history and config files."""
    # Load config to get experiment name
    with open(exp_info['config_path'], 'r') as f:
        config = json.load(f)
    
    # Load training history
    with open(exp_info['history_path'], 'r') as f:
        history = json.load(f)
    
    # Get experiment name from config
    exp_name = config.get('experiment_name', '')
    if not exp_name:
        # Derive from directory name (e.g., geocnn_base_20251203_232533 -> GeoCNN-Base)
        dir_name = os.path.basename(exp_info['dir'])
        if 'geocnn_base' in dir_name:
            exp_name = 'GeoCNN-Base'
        elif 'geocnn_edge_sobel' in dir_name:
            exp_name = 'GeoCNN-Edge (Sobel)'
        elif 'geocnn_edge_canny' in dir_name:
            exp_name = 'GeoCNN-Edge (Canny)'
        elif 'geocnn_lines' in dir_name:
            exp_name = 'GeoCNN-Lines'
        else:
            exp_name = dir_name
    
    # Clean up experiment name formatting
    exp_name = exp_name.replace('GeoCNN-Edge-canny', 'GeoCNN-Edge (Canny)')
    exp_name = exp_name.replace('GeoCNN-Edge-sobel', 'GeoCNN-Edge (Sobel)')
    
    return {
        'name': exp_name,
        'epochs': history['epoch'],
        'train_acc_top1': [a * 100 for a in history['train_acc_top1']],
        'val_acc_top1': [a * 100 for a in history['val_acc_top1']]
    }


def plot_comparison(experiments_data, save_dir='./outputs'):
    """Create comparison plots of accuracy curves."""
    
    # Create figure with two subplots: train and validation
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color palette for different experiments
    colors = plt.cm.tab10(range(len(experiments_data)))
    
    # Plot training accuracy
    ax_train = axes[0]
    for i, exp_data in enumerate(experiments_data):
        ax_train.plot(
            exp_data['epochs'],
            exp_data['train_acc_top1'],
            linewidth=2,
            label=exp_data['name'],
            color=colors[i]
        )
    
    ax_train.set_xlabel('Epoch', fontsize=12)
    ax_train.set_ylabel('Accuracy (%)', fontsize=12)
    ax_train.set_title('Training Accuracy Comparison across experiments', fontsize=14, fontweight='bold')
    ax_train.legend(fontsize=10, loc='best')
    ax_train.grid(True, alpha=0.3)
    
    # Plot validation accuracy
    ax_val = axes[1]
    for i, exp_data in enumerate(experiments_data):
        ax_val.plot(
            exp_data['epochs'],
            exp_data['val_acc_top1'],
            linewidth=2,
            label=exp_data['name'],
            color=colors[i]
        )
    
    ax_val.set_xlabel('Epoch', fontsize=12)
    ax_val.set_ylabel('Accuracy (%)', fontsize=12)
    ax_val.set_title('Validation Accuracy Comparison across experiments', fontsize=14, fontweight='bold')
    ax_val.legend(fontsize=10, loc='best')
    ax_val.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, 'experiment_comparison_top1.png')
    pdf_path = os.path.join(save_dir, 'experiment_comparison_top1.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    
    # Also create a single combined plot with both train and val
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    for i, exp_data in enumerate(experiments_data):
        # Plot validation (solid line)
        ax.plot(
            exp_data['epochs'],
            exp_data['val_acc_top1'],
            linewidth=2.5,
            label=f"{exp_data['name']} (Val)",
            color=colors[i],
            linestyle='-'
        )
        # Plot training (dashed line, same color)
        ax.plot(
            exp_data['epochs'],
            exp_data['train_acc_top1'],
            linewidth=2,
            label=f"{exp_data['name']} (Train)",
            color=colors[i],
            linestyle='--',
            alpha=0.7
        )
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy Comparison across experiments', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    png_path_combined = os.path.join(save_dir, 'experiment_comparison_top1_combined.png')
    pdf_path_combined = os.path.join(save_dir, 'experiment_comparison_top1_combined.pdf')
    
    plt.savefig(png_path_combined, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path_combined, bbox_inches='tight')
    plt.close()
    
    print(f"  {png_path_combined}")
    print(f"  {pdf_path_combined}")


def main():
    """Main function to generate comparison plots."""
    outputs_dir = './outputs'
    
    print("Finding experiments...")
    experiments = find_experiments(outputs_dir)
    
    if not experiments:
        print(f"No experiments found in {outputs_dir}")
        return
    
    print(f"Found {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {os.path.basename(exp['dir'])}")
    
    print("\nLoading experiment data...")
    experiments_data = []
    for exp_info in experiments:
        try:
            data = load_experiment_data(exp_info)
            experiments_data.append(data)
            print(f"  ✓ Loaded: {data['name']}")
        except Exception as e:
            print(f"  ✗ Failed to load {exp_info['dir']}: {e}")
    
    if not experiments_data:
        print("No valid experiment data found.")
        return
    
    print(f"\nGenerating comparison plots for {len(experiments_data)} experiments...")
    plot_comparison(experiments_data, save_dir=outputs_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
