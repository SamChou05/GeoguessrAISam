"""
Training Tracker - Comprehensive logging and visualization for research papers.

Tracks:
- Loss curves (train/val)
- Accuracy (Top-1, Top-5)
- Macro F1 score
- Per-class metrics
- Learning rate schedule
- Training time
- Best model checkpoints

Generates publication-ready plots automatically.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
plt.style.use('seaborn-v0_8-whitegrid')  # Clean style for papers


class TrainingTracker:
    """
    Comprehensive training tracker for research experiments.
    
    Automatically logs metrics, generates plots, and creates summary tables.
    """
    
    def __init__(
        self,
        output_dir: str,
        experiment_name: str,
        config: dict,
        class_names: List[str],
        use_tensorboard: bool = True
    ):
        """
        Args:
            output_dir: Directory to save all outputs
            experiment_name: Name of experiment (e.g., 'geocnn_base')
            config: Training configuration dictionary
            class_names: List of class names
            use_tensorboard: Whether to log to TensorBoard
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.config = config
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Create directories
        self.plots_dir = os.path.join(output_dir, 'plots')
        self.logs_dir = os.path.join(output_dir, 'logs')
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
        
        # Metrics history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc_top1': [],
            'train_acc_top5': [],
            'val_loss': [],
            'val_acc_top1': [],
            'val_acc_top5': [],
            'val_macro_f1': [],
            'learning_rate': [],
            'epoch_time': [],
            'timestamp': []
        }
        
        # Per-class metrics (stored periodically)
        self.per_class_history = []
        
        # Best model tracking
        self.best_val_f1 = 0.0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # Timing
        self.start_time = time.time()
        self.epoch_start_time = None
        
        # Save config
        self._save_config()
    
    def _save_config(self):
        """Save experiment configuration."""
        config_path = os.path.join(self.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            # Convert non-serializable items
            config_save = {}
            for k, v in self.config.items():
                if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                    config_save[k] = v
                else:
                    config_save[k] = str(v)
            config_save['class_names'] = self.class_names
            config_save['num_classes'] = self.num_classes
            config_save['start_time'] = datetime.now().isoformat()
            json.dump(config_save, f, indent=2)
    
    def start_epoch(self, epoch: int):
        """Call at the start of each epoch."""
        self.epoch_start_time = time.time()
        self.current_epoch = epoch
    
    def end_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc_top1: float,
        train_acc_top5: float,
        val_loss: float,
        val_acc_top1: float,
        val_acc_top5: float,
        val_macro_f1: float,
        learning_rate: float,
        per_class_metrics: Optional[Dict] = None
    ):
        """
        Log metrics at end of epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_acc_top1: Training top-1 accuracy
            train_acc_top5: Training top-5 accuracy  
            val_loss: Validation loss
            val_acc_top1: Validation top-1 accuracy
            val_acc_top5: Validation top-5 accuracy
            val_macro_f1: Validation macro F1 score
            learning_rate: Current learning rate
            per_class_metrics: Optional dict with per-class recall/precision
        """
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        # Store in history
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc_top1'].append(train_acc_top1)
        self.history['train_acc_top5'].append(train_acc_top5)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc_top1'].append(val_acc_top1)
        self.history['val_acc_top5'].append(val_acc_top5)
        self.history['val_macro_f1'].append(val_macro_f1)
        self.history['learning_rate'].append(learning_rate)
        self.history['epoch_time'].append(epoch_time)
        self.history['timestamp'].append(datetime.now().isoformat())
        
        # Store per-class metrics
        if per_class_metrics:
            self.per_class_history.append({
                'epoch': epoch,
                'metrics': per_class_metrics
            })
        
        # TensorBoard logging
        if self.use_tensorboard:
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train_top1', train_acc_top1, epoch)
            self.writer.add_scalar('Accuracy/train_top5', train_acc_top5, epoch)
            self.writer.add_scalar('Accuracy/val_top1', val_acc_top1, epoch)
            self.writer.add_scalar('Accuracy/val_top5', val_acc_top5, epoch)
            self.writer.add_scalar('F1/val_macro', val_macro_f1, epoch)
            self.writer.add_scalar('LearningRate', learning_rate, epoch)
            self.writer.add_scalar('Time/epoch_seconds', epoch_time, epoch)
        
        # Track best model
        is_best = False
        if val_macro_f1 > self.best_val_f1:
            self.best_val_f1 = val_macro_f1
            self.best_val_acc = val_acc_top1
            self.best_epoch = epoch
            is_best = True
        
        # Save history after each epoch
        self._save_history()
        
        # Generate plots every 5 epochs or at the end
        if epoch % 5 == 0 or epoch == self.config.get('epochs', 80):
            self.generate_plots()
        
        return is_best
    
    def _save_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.logs_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def generate_plots(self):
        """Generate all publication-ready plots."""
        self._plot_loss_curves()
        self._plot_accuracy_curves()
        self._plot_f1_curve()
        self._plot_learning_rate()
        self._plot_combined_metrics()
    
    def _plot_loss_curves(self):
        """Plot training and validation loss."""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        epochs = self.history['epoch']
        ax.plot(epochs, self.history['train_loss'], 'b-', linewidth=2, label='Train Loss')
        ax.plot(epochs, self.history['val_loss'], 'r-', linewidth=2, label='Val Loss')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Mark best epoch
        ax.axvline(x=self.best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best (epoch {self.best_epoch})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.plots_dir, 'loss_curves.pdf'), bbox_inches='tight')
        plt.close()
    
    def _plot_accuracy_curves(self):
        """Plot accuracy curves (Top-1 and Top-5)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = self.history['epoch']
        
        # Top-1 Accuracy
        axes[0].plot(epochs, [a*100 for a in self.history['train_acc_top1']], 'b-', linewidth=2, label='Train')
        axes[0].plot(epochs, [a*100 for a in self.history['val_acc_top1']], 'r-', linewidth=2, label='Val')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        axes[0].set_title('Top-1 Accuracy', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(x=self.best_epoch, color='g', linestyle='--', alpha=0.7)
        
        # Top-5 Accuracy
        axes[1].plot(epochs, [a*100 for a in self.history['train_acc_top5']], 'b-', linewidth=2, label='Train')
        axes[1].plot(epochs, [a*100 for a in self.history['val_acc_top5']], 'r-', linewidth=2, label='Val')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Top-5 Accuracy (%)', fontsize=12)
        axes[1].set_title('Top-5 Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(x=self.best_epoch, color='g', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'accuracy_curves.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.plots_dir, 'accuracy_curves.pdf'), bbox_inches='tight')
        plt.close()
    
    def _plot_f1_curve(self):
        """Plot macro F1 score."""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        epochs = self.history['epoch']
        ax.plot(epochs, self.history['val_macro_f1'], 'g-', linewidth=2, label='Val Macro F1')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Macro F1 Score', fontsize=12)
        ax.set_title('Validation Macro F1 Score', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=self.best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best: {self.best_val_f1:.4f}')
        
        # Add annotation for best F1
        ax.annotate(f'Best: {self.best_val_f1:.4f}', 
                   xy=(self.best_epoch, self.best_val_f1),
                   xytext=(self.best_epoch + 5, self.best_val_f1 + 0.02),
                   fontsize=10,
                   arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'f1_curve.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.plots_dir, 'f1_curve.pdf'), bbox_inches='tight')
        plt.close()
    
    def _plot_learning_rate(self):
        """Plot learning rate schedule."""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        epochs = self.history['epoch']
        ax.plot(epochs, self.history['learning_rate'], 'purple', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.plots_dir, 'learning_rate.pdf'), bbox_inches='tight')
        plt.close()
    
    def _plot_combined_metrics(self):
        """Create a combined figure with all key metrics - ideal for papers."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = self.history['epoch']
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', linewidth=2, label='Train')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', linewidth=2, label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('(a) Training Loss', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Top-1 Accuracy
        axes[0, 1].plot(epochs, [a*100 for a in self.history['train_acc_top1']], 'b-', linewidth=2, label='Train')
        axes[0, 1].plot(epochs, [a*100 for a in self.history['val_acc_top1']], 'r-', linewidth=2, label='Val')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('(b) Top-1 Accuracy', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-5 Accuracy
        axes[1, 0].plot(epochs, [a*100 for a in self.history['train_acc_top5']], 'b-', linewidth=2, label='Train')
        axes[1, 0].plot(epochs, [a*100 for a in self.history['val_acc_top5']], 'r-', linewidth=2, label='Val')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('(c) Top-5 Accuracy', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Macro F1
        axes[1, 1].plot(epochs, self.history['val_macro_f1'], 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Macro F1')
        axes[1, 1].set_title('(d) Validation Macro F1', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Mark best epoch on all plots
        for ax in axes.flat:
            ax.axvline(x=self.best_epoch, color='gray', linestyle='--', alpha=0.5)
        
        plt.suptitle(f'{self.experiment_name} Training Metrics', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'combined_metrics.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.plots_dir, 'combined_metrics.pdf'), bbox_inches='tight')
        plt.close()
    
    def log_confusion_matrix(self, cm: np.ndarray, epoch: int = None):
        """Log confusion matrix."""
        import seaborn as sns
        
        # Save raw confusion matrix
        cm_path = os.path.join(self.logs_dir, 'confusion_matrix.npy')
        np.save(cm_path, cm)
        
        # Plot normalized confusion matrix
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)
        
        fig, ax = plt.subplots(figsize=(max(12, self.num_classes * 0.3), max(10, self.num_classes * 0.25)))
        
        # Only show annotations if not too many classes
        annot = self.num_classes <= 30
        
        sns.heatmap(cm_norm, annot=annot, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names if self.num_classes <= 50 else False,
                   yticklabels=self.class_names if self.num_classes <= 50 else False,
                   ax=ax)
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix.pdf'), bbox_inches='tight')
        plt.close()
        
        # Also log to TensorBoard
        if self.use_tensorboard and epoch:
            self.writer.add_figure('ConfusionMatrix', fig, epoch)
    
    def log_per_class_recall(self, recall_dict: Dict[str, float]):
        """Plot per-class recall bar chart."""
        # Sort by recall
        sorted_items = sorted(recall_dict.items(), key=lambda x: x[1])
        
        # Show top and bottom classes if too many
        if len(sorted_items) > 40:
            items_to_show = sorted_items[:15] + sorted_items[-15:]
            title_suffix = ' (Bottom 15 & Top 15)'
        else:
            items_to_show = sorted_items
            title_suffix = ''
        
        names = [item[0] for item in items_to_show]
        recalls = [item[1] * 100 for item in items_to_show]
        
        fig, ax = plt.subplots(figsize=(10, max(8, len(names) * 0.25)))
        
        colors = plt.cm.RdYlGn(np.array(recalls) / 100)
        y_pos = np.arange(len(names))
        
        ax.barh(y_pos, recalls, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Recall (%)', fontsize=12)
        ax.set_title(f'Per-Class Recall{title_suffix}', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'per_class_recall.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.plots_dir, 'per_class_recall.pdf'), bbox_inches='tight')
        plt.close()
    
    def generate_summary_table(self) -> str:
        """Generate a summary table for the paper."""
        total_time = time.time() - self.start_time
        
        summary = {
            'Experiment': self.experiment_name,
            'Total Epochs': len(self.history['epoch']),
            'Best Epoch': self.best_epoch,
            'Best Val Top-1 Acc': f"{self.best_val_acc * 100:.2f}%",
            'Best Val Macro F1': f"{self.best_val_f1:.4f}",
            'Final Train Loss': f"{self.history['train_loss'][-1]:.4f}",
            'Final Val Loss': f"{self.history['val_loss'][-1]:.4f}",
            'Total Training Time': f"{total_time / 3600:.2f} hours",
            'Avg Time/Epoch': f"{np.mean(self.history['epoch_time']):.1f} seconds",
            'Num Classes': self.num_classes,
        }
        
        # Save as JSON
        summary_path = os.path.join(self.output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create markdown table
        md_table = "| Metric | Value |\n|--------|-------|\n"
        for k, v in summary.items():
            md_table += f"| {k} | {v} |\n"
        
        md_path = os.path.join(self.output_dir, 'summary.md')
        with open(md_path, 'w') as f:
            f.write(f"# {self.experiment_name} Results\n\n")
            f.write(md_table)
        
        return md_table
    
    def finalize(self):
        """Call at end of training to generate final outputs."""
        # Generate all plots
        self.generate_plots()
        
        # Generate summary
        summary = self.generate_summary_table()
        
        # Close TensorBoard writer
        if self.use_tensorboard:
            self.writer.close()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE - Summary")
        print("="*60)
        print(summary)
        print(f"\nAll outputs saved to: {self.output_dir}")
        print(f"  - Plots: {self.plots_dir}")
        print(f"  - Logs: {self.logs_dir}")
        if self.use_tensorboard:
            print(f"  - TensorBoard: {os.path.join(self.output_dir, 'tensorboard')}")


def compute_topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> List[float]:
    """
    Compute top-k accuracy.
    
    Args:
        output: Model logits (B, C)
        target: Ground truth labels (B,)
        topk: Tuple of k values
        
    Returns:
        List of accuracies for each k
    """
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k / batch_size).item())
    
    return res


if __name__ == "__main__":
    print("TrainingTracker module loaded successfully.")
    print("Usage:")
    print("  tracker = TrainingTracker(output_dir, 'experiment_name', config, class_names)")
    print("  tracker.start_epoch(epoch)")
    print("  tracker.end_epoch(epoch, train_loss, ...)")
    print("  tracker.finalize()")

