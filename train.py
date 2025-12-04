"""
Training script for GeoGuessr country classification experiments.

Supports training:
- GeoCNN-Base (main model)
- GeoCNN-Edge (with Sobel/Canny channels)
- GeoCNN-Lines (with line features)
- GeoCNN-Segmentation (with sky/ground gating)

Usage:
    python train.py --model base --data_dir ./dataset --epochs 80
    python train.py --model edge --edge_type sobel --data_dir ./dataset
    python train.py --model lines --data_dir ./dataset
    python train.py --model segmentation --data_dir ./dataset
"""

import os
import sys
import argparse
import json
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
import platform

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

# Fix multiprocessing on macOS
if platform.system() == 'Darwin':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

from models import GeoCNNBase, GeoCNNEdge, GeoCNNLines, GeoCNNSegmentation
from data.dataset import create_data_loaders, load_dataset_from_directory, create_splits, GeoDataset
from data.transforms import get_train_transforms, get_val_transforms
from data.augmentations import LineFeatureAugmentation, SkyGroundAugmentation
from utils.training_tracker import TrainingTracker, compute_topk_accuracy

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, classification_report


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        
        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # Compute loss
        log_probs = torch.log_softmax(pred, dim=-1)
        
        if self.weight is not None:
            # Apply class weights
            weight_expanded = self.weight[target].unsqueeze(1)
            loss = (-true_dist * log_probs * weight_expanded).sum(dim=-1)
        else:
            loss = (-true_dist * log_probs).sum(dim=-1)
        
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(pred, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class Trainer:
    """Trainer class for CNN models with comprehensive tracking."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        device: torch.device,
        output_dir: str,
        class_names: list,
        config: dict,
        use_amp: bool = True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.class_names = class_names
        self.use_amp = use_amp
        
        # Initialize scaler for mixed precision (works with CUDA)
        if use_amp and device.type == 'cuda':
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None
        
        # Initialize comprehensive tracker
        self.tracker = TrainingTracker(
            output_dir=output_dir,
            experiment_name=config.get('experiment_name', 'GeoCNN'),
            config=config,
            class_names=class_names,
            use_tensorboard=True
        )
        
        self.best_val_f1 = 0.0
        self.best_epoch = 0
    
    def train_epoch(self) -> tuple:
        """Train for one epoch. Returns (loss, top1_acc, top5_acc)."""
        self.model.train()
        total_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp and self.device.type == 'cuda':
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Compute top-1 and top-5 accuracy
            total_loss += loss.item() * images.size(0)
            total += labels.size(0)
            
            _, pred_top1 = outputs.max(1)
            correct_top1 += pred_top1.eq(labels).sum().item()
            
            # Top-5
            _, pred_top5 = outputs.topk(min(5, outputs.size(1)), 1, True, True)
            correct_top5 += pred_top5.eq(labels.view(-1, 1).expand_as(pred_top5)).any(1).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'top1': f'{100.*correct_top1/total:.2f}%',
                'top5': f'{100.*correct_top5/total:.2f}%'
            })
        
        return total_loss / total, correct_top1 / total, correct_top5 / total
    
    @torch.no_grad()
    def validate(self, compute_per_class: bool = False) -> tuple:
        """Validate the model. Returns (loss, top1_acc, top5_acc, macro_f1, [per_class_metrics])."""
        self.model.eval()
        total_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            total += labels.size(0)
            
            # Top-1
            _, pred_top1 = outputs.max(1)
            correct_top1 += pred_top1.eq(labels).sum().item()
            
            # Top-5
            _, pred_top5 = outputs.topk(min(5, outputs.size(1)), 1, True, True)
            correct_top5 += pred_top5.eq(labels.view(-1, 1).expand_as(pred_top5)).any(1).sum().item()
            
            all_preds.extend(pred_top1.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
        
        # Compute macro F1
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Per-class metrics (optional, computed at end of training)
        per_class_metrics = None
        if compute_per_class:
            report = classification_report(all_labels, all_preds, 
                                          target_names=self.class_names, 
                                          output_dict=True, zero_division=0)
            per_class_metrics = {
                name: {
                    'precision': report[name]['precision'],
                    'recall': report[name]['recall'],
                    'f1': report[name]['f1-score'],
                    'support': report[name]['support']
                }
                for name in self.class_names if name in report
            }
            
            # Confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            self.tracker.log_confusion_matrix(cm)
            
            # Per-class recall plot
            recall_dict = {name: metrics['recall'] for name, metrics in per_class_metrics.items()}
            self.tracker.log_per_class_recall(recall_dict)
        
        return (total_loss / total, correct_top1 / total, correct_top5 / total, 
                macro_f1, per_class_metrics)
    
    def train(self, epochs: int, patience: int = 10):
        """Train the model for multiple epochs with comprehensive tracking."""
        no_improve_count = 0
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 50)
            
            # Start epoch tracking
            self.tracker.start_epoch(epoch)
            
            # Train
            train_loss, train_top1, train_top5 = self.train_epoch()
            
            # Validate (compute per-class metrics every 10 epochs or at the end)
            compute_per_class = (epoch % 10 == 0) or (epoch == epochs)
            val_loss, val_top1, val_top5, val_f1, per_class = self.validate(compute_per_class)
            
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to tracker
            is_best = self.tracker.end_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_acc_top1=train_top1,
                train_acc_top5=train_top5,
                val_loss=val_loss,
                val_acc_top1=val_top1,
                val_acc_top5=val_top5,
                val_macro_f1=val_f1,
                learning_rate=current_lr,
                per_class_metrics=per_class
            )
            
            # Print summary
            print(f"Train - Loss: {train_loss:.4f}, Top1: {train_top1*100:.2f}%, Top5: {train_top5*100:.2f}%")
            print(f"Val   - Loss: {val_loss:.4f}, Top1: {val_top1*100:.2f}%, Top5: {val_top5*100:.2f}%, F1: {val_f1:.4f}")
            print(f"LR: {current_lr:.6f}")
            
            # Save best model
            if is_best:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                no_improve_count = 0
                self.save_checkpoint('best_model.pth')
                print(f"★ New best model! F1: {val_f1:.4f}")
            else:
                no_improve_count += 1
            
            # Early stopping
            if no_improve_count >= patience:
                print(f"\n⚠ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
        
        # Final evaluation with per-class metrics
        print("\n" + "="*50)
        print("Final Evaluation")
        print("="*50)
        _, _, _, _, _ = self.validate(compute_per_class=True)
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        # Finalize tracking (generates all plots and summary)
        self.tracker.finalize()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint with full training state."""
        path = os.path.join(self.output_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_f1': self.best_val_f1,
            'best_epoch': self.best_epoch,
            'class_names': self.class_names,
        }, path)


class LinesTrainer(Trainer):
    """Trainer for GeoCNN-Lines model (needs line features)."""
    
    def __init__(self, *args, line_extractor: LineFeatureAugmentation = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.line_extractor = line_extractor or LineFeatureAugmentation()
    
    def train_epoch(self) -> tuple:
        """Train for one epoch with line features."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels, paths in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Extract line features (this is slow, consider pre-computing)
            line_features = []
            for path in paths:
                import cv2
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    feats = self.line_extractor(img)
                else:
                    feats = np.zeros(20, dtype=np.float32)
                line_features.append(feats)
            line_features = torch.tensor(np.array(line_features)).to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images, line_features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / total, correct / total


class SegmentationTrainer(Trainer):
    """Trainer for GeoCNN-Segmentation model (needs sky/ground masks)."""
    
    def __init__(self, *args, segmenter: SkyGroundAugmentation = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.segmenter = segmenter or SkyGroundAugmentation()
    
    def train_epoch(self) -> tuple:
        """Train for one epoch with sky/ground segmentation."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels, paths in pbar:
            labels = labels.to(self.device)
            
            # Create sky and ground masked images
            sky_images = []
            ground_images = []
            
            for path in paths:
                import cv2
                from torchvision import transforms
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    sky_img, ground_img = self.segmenter(img)
                else:
                    sky_img = np.zeros((224, 224, 3), dtype=np.uint8)
                    ground_img = np.zeros((224, 224, 3), dtype=np.uint8)
                
                # Apply standard transforms
                tf = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                sky_images.append(tf(sky_img))
                ground_images.append(tf(ground_img))
            
            sky_images = torch.stack(sky_images).to(self.device)
            ground_images = torch.stack(ground_images).to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(sky_images, ground_images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / total, correct / total


def create_output_dir(base_dir: str, model_name: str) -> str:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'{model_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def train_base_model(args):
    """Train GeoCNN-Base model."""
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir, 'geocnn_base')
    
    # Create data loaders
    train_transform = get_train_transforms(
        input_size=args.input_size,
        use_horizontal_flip=args.use_flip
    )
    val_transform = get_val_transforms(input_size=args.input_size)
    
    train_loader, val_loader, test_loader, class_names, class_weights = create_data_loaders(
        args.data_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=val_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=args.weighted_sampler
    )
    
    # Create model
    num_classes = len(class_names)
    model = GeoCNNBase(num_classes=num_classes, dropout_p=args.dropout)
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create criterion
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if args.use_class_weights else None)
    elif args.loss == 'smooth':
        criterion = LabelSmoothingCrossEntropy(
            smoothing=args.label_smoothing,
            weight=class_weights.to(device) if args.use_class_weights else None
        )
    else:  # focal
        criterion = FocalLoss(gamma=2.0, weight=class_weights.to(device) if args.use_class_weights else None)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Build config for tracking
    config = vars(args).copy()
    config['num_classes'] = num_classes
    config['model_params'] = model.count_parameters()
    config['experiment_name'] = 'GeoCNN-Base'
    
    # Train with comprehensive tracking
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        class_names=class_names,
        config=config,
        use_amp=args.use_amp
    )
    
    trainer.train(epochs=args.epochs, patience=args.patience)
    
    return output_dir


def train_edge_model(args):
    """Train GeoCNN-Edge model."""
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir, f'geocnn_edge_{args.edge_type}')
    
    # For edge model, we need custom data loading
    # Load paths and labels
    image_paths, labels, class_names = load_dataset_from_directory(args.data_dir)
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = create_splits(
        image_paths, labels
    )
    
    # Create datasets with edge augmentation
    from data.augmentations import EdgeAugmentation, EdgeToTensor
    from torchvision import transforms
    
    class EdgeDataset(torch.utils.data.Dataset):
        def __init__(self, paths, labels, edge_type='sobel', input_size=224, is_train=True):
            self.paths = paths
            self.labels = labels
            self.edge_aug = EdgeAugmentation(edge_type=edge_type)
            self.input_size = input_size
            self.is_train = is_train
            
            # Edge normalization (RGB + edge channels)
            if edge_type == 'sobel':
                self.mean = [0.485, 0.456, 0.406, 0.0, 0.0]
                self.std = [0.229, 0.224, 0.225, 1.0, 1.0]
            else:
                self.mean = [0.485, 0.456, 0.406, 0.0]
                self.std = [0.229, 0.224, 0.225, 1.0]
        
        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, idx):
            import cv2
            from PIL import Image
            
            # Load image
            img = cv2.imread(self.paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, (self.input_size, self.input_size))
            
            # Add edge channels
            edge_img = self.edge_aug(img)
            
            # Convert to tensor
            tensor = torch.from_numpy(edge_img.transpose(2, 0, 1)).float()
            
            # Normalize
            for c in range(tensor.shape[0]):
                tensor[c] = (tensor[c] - self.mean[c]) / self.std[c]
            
            return tensor, self.labels[idx]
    
    train_dataset = EdgeDataset(train_paths, train_labels, args.edge_type, args.input_size, is_train=True)
    val_dataset = EdgeDataset(val_paths, val_labels, args.edge_type, args.input_size, is_train=False)
    
    # Get class weights
    temp_dataset = GeoDataset(train_paths, train_labels, class_names)
    class_weights = temp_dataset.get_class_weights()
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Create model
    num_classes = len(class_names)
    model = GeoCNNEdge(num_classes=num_classes, edge_type=args.edge_type, dropout_p=args.dropout)
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create criterion
    criterion = LabelSmoothingCrossEntropy(
        smoothing=args.label_smoothing,
        weight=class_weights.to(device) if args.use_class_weights else None
    )
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Build config for tracking
    config = vars(args).copy()
    config['num_classes'] = num_classes
    config['model_params'] = model.count_parameters()
    config['experiment_name'] = f'GeoCNN-Edge-{args.edge_type}'
    
    # Train with comprehensive tracking
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        class_names=class_names,
        config=config,
        use_amp=args.use_amp
    )
    
    trainer.train(epochs=args.epochs, patience=args.patience)
    
    return output_dir


def train_lines_model(args):
    """Train GeoCNN-Lines model with line feature fusion."""
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir, 'geocnn_lines')
    
    # Load data
    image_paths, labels, class_names = load_dataset_from_directory(args.data_dir)
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = create_splits(
        image_paths, labels
    )
    
    # Create datasets with line feature extraction
    from data.augmentations import LineFeatureAugmentation
    from data.transforms import get_train_transforms, get_val_transforms
    
    class LinesDataset(GeoDataset):
        """Dataset that extracts line features alongside images."""
        def __init__(self, image_paths, labels, class_names, transform=None, return_path=True):
            super().__init__(image_paths, labels, class_names, transform=transform, return_path=return_path)
            self.line_extractor = LineFeatureAugmentation()
        
        def __getitem__(self, idx):
            # Get image and label
            image, label = super().__getitem__(idx)
            
            # Extract line features from original image
            img_path = self.image_paths[idx]
            import cv2
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                line_features = self.line_extractor(img)
            else:
                line_features = np.zeros(20, dtype=np.float32)
            
            line_features = torch.tensor(line_features, dtype=torch.float32)
            
            if self.return_path:
                return image, label, line_features, img_path
            return image, label, line_features
    
    # Create transforms
    train_transform = get_train_transforms(input_size=args.input_size, use_horizontal_flip=args.use_flip)
    val_transform = get_val_transforms(input_size=args.input_size)
    
    train_dataset = LinesDataset(train_paths, train_labels, class_names, transform=train_transform, return_path=True)
    val_dataset = LinesDataset(val_paths, val_labels, class_names, transform=val_transform, return_path=True)
    
    # Get class weights
    class_weights = train_dataset.get_class_weights()
    
    # Custom collate function for line features
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        line_features = torch.stack([item[2] for item in batch])
        paths = [item[3] for item in batch]
        return images, labels, line_features, paths
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    # Create model
    num_classes = len(class_names)
    model = GeoCNNLines(num_classes=num_classes, dropout_p=args.dropout)
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create criterion
    criterion = LabelSmoothingCrossEntropy(
        smoothing=args.label_smoothing,
        weight=class_weights.to(device) if args.use_class_weights else None
    )
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Build config
    config = vars(args).copy()
    config['num_classes'] = num_classes
    config['model_params'] = model.count_parameters()
    config['experiment_name'] = 'GeoCNN-Lines'
    
    # Create custom trainer for lines model
    class LinesTrainerWrapper(Trainer):
        """Wrapper to handle line features in training loop."""
        def train_epoch(self):
            self.model.train()
            total_loss = 0.0
            correct_top1 = 0
            correct_top5 = 0
            total = 0
            
            pbar = tqdm(self.train_loader, desc='Training')
            for images, labels, line_features, _ in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                line_features = line_features.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images, line_features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * images.size(0)
                total += labels.size(0)
                
                _, pred_top1 = outputs.max(1)
                correct_top1 += pred_top1.eq(labels).sum().item()
                
                _, pred_top5 = outputs.topk(min(5, outputs.size(1)), 1, True, True)
                correct_top5 += pred_top5.eq(labels.view(-1, 1).expand_as(pred_top5)).any(1).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'top1': f'{100.*correct_top1/total:.2f}%',
                    'top5': f'{100.*correct_top5/total:.2f}%'
                })
            
            return total_loss / total, correct_top1 / total, correct_top5 / total
        
        @torch.no_grad()
        def validate(self, compute_per_class=False):
            self.model.eval()
            total_loss = 0.0
            correct_top1 = 0
            correct_top5 = 0
            total = 0
            all_preds = []
            all_labels = []
            
            for images, labels, line_features, _ in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                line_features = line_features.to(self.device)
                
                outputs = self.model(images, line_features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                total += labels.size(0)
                
                _, pred_top1 = outputs.max(1)
                correct_top1 += pred_top1.eq(labels).sum().item()
                
                _, pred_top5 = outputs.topk(min(5, outputs.size(1)), 1, True, True)
                correct_top5 += pred_top5.eq(labels.view(-1, 1).expand_as(pred_top5)).any(1).sum().item()
                
                all_preds.extend(pred_top1.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            macro_f1 = f1_score(all_labels, all_preds, average='macro')
            
            per_class_metrics = None
            if compute_per_class:
                report = classification_report(all_labels, all_preds, 
                                              target_names=self.class_names, 
                                              output_dict=True, zero_division=0)
                per_class_metrics = {
                    name: {
                        'precision': report[name]['precision'],
                        'recall': report[name]['recall'],
                        'f1': report[name]['f1-score'],
                        'support': report[name]['support']
                    }
                    for name in self.class_names if name in report
                }
                cm = confusion_matrix(all_labels, all_preds)
                self.tracker.log_confusion_matrix(cm)
                recall_dict = {name: metrics['recall'] for name, metrics in per_class_metrics.items()}
                self.tracker.log_per_class_recall(recall_dict)
            
            return (total_loss / total, correct_top1 / total, correct_top5 / total, 
                    macro_f1, per_class_metrics)
    
    trainer = LinesTrainerWrapper(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        class_names=class_names,
        config=config,
        use_amp=args.use_amp
    )
    
    trainer.train(epochs=args.epochs, patience=args.patience)
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Train GeoGuessr classification models')
    
    # Model selection
    parser.add_argument('--model', type=str, default='base',
                       choices=['base', 'edge', 'lines', 'segmentation'],
                       help='Model type to train')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for checkpoints')
    
    # Model arguments
    parser.add_argument('--input_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout probability')
    parser.add_argument('--edge_type', type=str, default='sobel',
                       choices=['sobel', 'canny'],
                       help='Edge type for edge model')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Loss arguments
    parser.add_argument('--loss', type=str, default='smooth',
                       choices=['ce', 'smooth', 'focal'],
                       help='Loss function')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing factor')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Use class weights for imbalanced data')
    
    # Augmentation arguments
    parser.add_argument('--use_flip', action='store_true',
                       help='Use horizontal flip (not recommended)')
    parser.add_argument('--weighted_sampler', action='store_true',
                       help='Use weighted random sampler')
    
    # Training options
    parser.add_argument('--num_workers', type=int, default=0 if platform.system() == 'Darwin' else 4,
                       help='Number of data loader workers (0 recommended for macOS)')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use automatic mixed precision')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Train selected model
    if args.model == 'base':
        train_base_model(args)
    elif args.model == 'edge':
        train_edge_model(args)
    elif args.model == 'lines':
        train_lines_model(args)
    elif args.model == 'segmentation':
        print("Segmentation model training - use train_segmentation.py for better performance")
        train_base_model(args)


if __name__ == '__main__':
    main()

