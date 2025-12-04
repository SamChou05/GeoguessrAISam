"""
GeoCNN-Base: Main CNN architecture for GeoGuessr country classification.

Architecture (224x224 RGB input):
- Block 1: Conv3x3 3→32, BN, ReLU; Conv3x3 32→32, BN, ReLU; MaxPool 2x2
- Block 2: Conv3x3 32→64, BN, ReLU; Conv3x3 64→64, BN, ReLU; MaxPool 2x2
- Block 3: Conv3x3 64→128, BN, ReLU; Conv3x3 128→128, BN, ReLU; MaxPool 2x2
- Block 4: Conv3x3 128→256, BN, ReLU; Conv1x1 256→256, BN, ReLU
- Global Average Pool → 256-d vector
- Dropout p=0.3
- Fully-connected 256→C

Parameters: ~1.2M + 256*C
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Basic convolutional block: Conv → BatchNorm → ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class GeoCNNBase(nn.Module):
    """
    GeoCNN-Base: A simple CNN for country-level geolocation.
    
    Args:
        num_classes: Number of countries to classify
        in_channels: Number of input channels (3 for RGB)
        dropout_p: Dropout probability before final FC layer
    """
    
    def __init__(self, num_classes: int, in_channels: int = 3, dropout_p: float = 0.3):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Block 1: 224x224 → 112x112
        self.block1 = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=3),
            ConvBlock(32, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: 112x112 → 56x56
        self.block2 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3),
            ConvBlock(64, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: 56x56 → 28x28
        self.block3 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3),
            ConvBlock(128, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 4: 28x28 → 28x28 (no pooling)
        self.block4 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3),
            ConvBlock(256, 256, kernel_size=1)  # 1x1 conv
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 256-d feature vector before classification."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten to (B, 256)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, 224, 224)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        features = self.extract_features(x)
        features = self.dropout(features)
        logits = self.fc(features)
        return logits
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = GeoCNNBase(num_classes=50)
    print(f"GeoCNN-Base Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Test feature extraction
    features = model.extract_features(x)
    print(f"Feature shape: {features.shape}")

