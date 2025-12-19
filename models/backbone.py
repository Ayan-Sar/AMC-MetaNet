"""
Backbone networks for feature extraction.
Includes ResNet-12 and Conv4 architectures commonly used in few-shot learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-12."""
    
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        drop_rate: float = 0.0
    ):
        super().__init__()
        
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.maxpool = nn.MaxPool2d(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        out = self.maxpool(out)
        
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        
        return out


class ResNet12(nn.Module):
    """
    ResNet-12 backbone for few-shot learning.
    
    A 12-layer residual network optimized for meta-learning,
    producing 640-dimensional feature embeddings.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 640,
        drop_rate: float = 0.1,
        dropblock_size: int = 5
    ):
        super().__init__()
        
        self.in_channels = 64
        self.drop_rate = drop_rate
        
        # Four stages with increasing channels: 64 -> 128 -> 256 -> 512
        channels = [64, 128, 256, 512]
        
        self.layer1 = self._make_layer(channels[0], stride=1)
        self.layer2 = self._make_layer(channels[1], stride=1)
        self.layer3 = self._make_layer(channels[2], stride=1)
        self.layer4 = self._make_layer(channels[3], stride=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = channels[3]
        
        # Optional projection head
        if feature_dim != channels[3]:
            self.projection = nn.Sequential(
                nn.Linear(channels[3], feature_dim),
                nn.ReLU(inplace=True)
            )
            self.feature_dim = feature_dim
        else:
            self.projection = None
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(self, out_channels: int, stride: int = 1) -> BasicBlock:
        """Create a residual block."""
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels,
                    kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        
        block = BasicBlock(
            self.in_channels, out_channels,
            stride=stride, downsample=downsample,
            drop_rate=self.drop_rate
        )
        self.in_channels = out_channels
        
        return block
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        if self.projection is not None:
            x = self.projection(x)
        
        return x


class Conv4Block(nn.Module):
    """Basic convolutional block for Conv4."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class Conv4(nn.Module):
    """
    4-layer ConvNet backbone.
    
    A simple but effective backbone for few-shot learning,
    commonly used as a baseline.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        feature_dim: int = 64
    ):
        super().__init__()
        
        self.layer1 = Conv4Block(in_channels, hidden_dim)
        self.layer2 = Conv4Block(hidden_dim, hidden_dim)
        self.layer3 = Conv4Block(hidden_dim, hidden_dim)
        self.layer4 = Conv4Block(hidden_dim, hidden_dim)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = hidden_dim
        
        # Optional projection
        if feature_dim != hidden_dim:
            self.projection = nn.Linear(hidden_dim, feature_dim)
            self.feature_dim = feature_dim
        else:
            self.projection = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        if self.projection is not None:
            x = self.projection(x)
        
        return x


def get_backbone(
    name: str = 'resnet12',
    feature_dim: int = 640,
    **kwargs
) -> nn.Module:
    """
    Factory function to create backbone networks.
    
    Args:
        name: Backbone name ('resnet12' or 'conv4')
        feature_dim: Output feature dimension
        
    Returns:
        Backbone network module
    """
    backbones = {
        'resnet12': ResNet12,
        'conv4': Conv4
    }
    
    if name not in backbones:
        raise ValueError(f"Unknown backbone: {name}")
    
    return backbones[name](feature_dim=feature_dim, **kwargs)
