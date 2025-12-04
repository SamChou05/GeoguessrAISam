"""
Test script to verify the project setup is working correctly.

Run this after installation to ensure all components are functional.
"""

import sys
import torch
import numpy as np


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from models import GeoCNNBase, GeoCNNEdge, GeoCNNLines, GeoCNNSegmentation
        print("  ✓ Models imported successfully")
    except ImportError as e:
        print(f"  ✗ Models import failed: {e}")
        return False
    
    try:
        from data.dataset import GeoDataset, create_data_loaders
        from data.transforms import get_train_transforms, get_val_transforms
        from data.augmentations import EdgeAugmentation, LineFeatureAugmentation
        print("  ✓ Data modules imported successfully")
    except ImportError as e:
        print(f"  ✗ Data modules import failed: {e}")
        return False
    
    try:
        from baselines.bovw import BOVWClassifier
        from baselines.linear_baseline import LinearPixelClassifier
        print("  ✓ Baselines imported successfully")
    except ImportError as e:
        print(f"  ✗ Baselines import failed: {e}")
        return False
    
    try:
        from utils.visualization import plot_confusion_matrix, plot_training_history
        print("  ✓ Utils imported successfully")
    except ImportError as e:
        print(f"  ✗ Utils import failed: {e}")
        return False
    
    return True


def test_models():
    """Test model forward passes."""
    print("\nTesting models...")
    
    batch_size = 2
    num_classes = 10
    
    # Test GeoCNN-Base
    try:
        from models import GeoCNNBase
        model = GeoCNNBase(num_classes=num_classes)
        x = torch.randn(batch_size, 3, 224, 224)
        out = model(x)
        assert out.shape == (batch_size, num_classes)
        print(f"  ✓ GeoCNN-Base: {model.count_parameters():,} params")
    except Exception as e:
        print(f"  ✗ GeoCNN-Base failed: {e}")
        return False
    
    # Test GeoCNN-Edge (Sobel)
    try:
        from models import GeoCNNEdge
        model = GeoCNNEdge(num_classes=num_classes, edge_type='sobel')
        x = torch.randn(batch_size, 5, 224, 224)
        out = model(x)
        assert out.shape == (batch_size, num_classes)
        print(f"  ✓ GeoCNN-Edge (Sobel): {model.count_parameters():,} params")
    except Exception as e:
        print(f"  ✗ GeoCNN-Edge failed: {e}")
        return False
    
    # Test GeoCNN-Lines
    try:
        from models import GeoCNNLines
        model = GeoCNNLines(num_classes=num_classes)
        images = torch.randn(batch_size, 3, 224, 224)
        line_features = torch.randn(batch_size, 20)
        out = model(images, line_features)
        assert out.shape == (batch_size, num_classes)
        print(f"  ✓ GeoCNN-Lines: {model.count_parameters():,} params")
    except Exception as e:
        print(f"  ✗ GeoCNN-Lines failed: {e}")
        return False
    
    # Test GeoCNN-Segmentation
    try:
        from models import GeoCNNSegmentation
        model = GeoCNNSegmentation(num_classes=num_classes)
        sky_images = torch.randn(batch_size, 3, 224, 224)
        ground_images = torch.randn(batch_size, 3, 224, 224)
        out = model(sky_images, ground_images)
        assert out.shape == (batch_size, num_classes)
        print(f"  ✓ GeoCNN-Segmentation: {model.count_parameters():,} params")
    except Exception as e:
        print(f"  ✗ GeoCNN-Segmentation failed: {e}")
        return False
    
    return True


def test_augmentations():
    """Test augmentation functions."""
    print("\nTesting augmentations...")
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test Edge Augmentation
    try:
        from data.augmentations import EdgeAugmentation
        
        sobel_aug = EdgeAugmentation(edge_type='sobel')
        sobel_out = sobel_aug(dummy_image)
        assert sobel_out.shape == (224, 224, 5), f"Expected (224, 224, 5), got {sobel_out.shape}"
        print("  ✓ Sobel edge augmentation working")
        
        canny_aug = EdgeAugmentation(edge_type='canny')
        canny_out = canny_aug(dummy_image)
        assert canny_out.shape == (224, 224, 4), f"Expected (224, 224, 4), got {canny_out.shape}"
        print("  ✓ Canny edge augmentation working")
    except Exception as e:
        print(f"  ✗ Edge augmentation failed: {e}")
        return False
    
    # Test Line Feature Augmentation
    try:
        from data.augmentations import LineFeatureAugmentation
        line_aug = LineFeatureAugmentation()
        line_features = line_aug(dummy_image)
        assert line_features.shape == (20,), f"Expected (20,), got {line_features.shape}"
        print("  ✓ Line feature extraction working")
    except Exception as e:
        print(f"  ✗ Line feature extraction failed: {e}")
        return False
    
    # Test Sky/Ground Segmentation
    try:
        from data.augmentations import SkyGroundAugmentation
        seg_aug = SkyGroundAugmentation()
        sky_img, ground_img = seg_aug(dummy_image)
        assert sky_img.shape == (224, 224, 3)
        assert ground_img.shape == (224, 224, 3)
        print("  ✓ Sky/Ground segmentation working")
    except Exception as e:
        print(f"  ✗ Sky/Ground segmentation failed: {e}")
        return False
    
    return True


def test_transforms():
    """Test data transforms."""
    print("\nTesting transforms...")
    
    try:
        from data.transforms import get_train_transforms, get_val_transforms, FiveCropTTA
        from PIL import Image
        
        # Create dummy PIL image
        dummy_img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
        
        train_tf = get_train_transforms()
        val_tf = get_val_transforms()
        
        train_out = train_tf(dummy_img)
        val_out = val_tf(dummy_img)
        
        assert train_out.shape == (3, 224, 224)
        assert val_out.shape == (3, 224, 224)
        print("  ✓ Train/Val transforms working")
        
        # Test TTA
        tta = FiveCropTTA()
        tta_out = tta(dummy_img)
        assert tta_out.shape == (5, 3, 224, 224)
        print("  ✓ Five-crop TTA working")
        
    except Exception as e:
        print(f"  ✗ Transforms failed: {e}")
        return False
    
    return True


def test_device():
    """Test device availability."""
    print("\nTesting device...")
    
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print("  ✓ MPS (Apple Silicon) available")
    else:
        print("  ⚠ No GPU available, will use CPU")
    
    return True


def main():
    print("=" * 60)
    print("GeoGuessr Project Setup Test")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_models()
    all_passed &= test_augmentations()
    all_passed &= test_transforms()
    all_passed &= test_device()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Project is ready to use.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

