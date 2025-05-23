import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import List, Dict, Any

class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature extraction."""
    
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels[i], out_channels, 1)
            for i in range(len(in_channels))
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(len(in_channels))
        ])
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        laterals = [
            lateral_conv(features[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode='nearest'
            )
        
        # Build output
        outs = [
            fpn_conv(laterals[i])
            for i, fpn_conv in enumerate(self.fpn_convs)
        ]
        
        return outs

class ResNetFPN(nn.Module):
    """ResNet50 backbone with FPN for multi-task perception."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Load pretrained ResNet50
        weights = ResNet50_Weights.DEFAULT if config['model']['backbone']['pretrained'] else None
        resnet = resnet50(weights=weights)
        
        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
        # Freeze batch norm if specified
        if config['model']['backbone']['freeze_bn']:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        
        # Initialize FPN
        self.fpn = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=config['model']['fpn_channels']
        )
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Initial features
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet stages
        c1 = self.layer1(x)   # 1/4
        c2 = self.layer2(c1)  # 1/8
        c3 = self.layer3(c2)  # 1/16
        c4 = self.layer4(c3)  # 1/32
        
        # FPN features
        features = self.fpn([c1, c2, c3, c4])
        
        return features

    @torch.jit.export
    def get_strides(self) -> List[int]:
        """Get the strides of the feature maps."""
        return [4, 8, 16, 32]
