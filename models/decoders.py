import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module for semantic segmentation."""
    
    def __init__(self, in_channels: int, out_channels: int, rates: List[int]):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # 1x1 conv
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )
        
        # Atrous convs
        for rate in rates:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3,
                             padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Global average pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 1,
                     bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res.append(F.interpolate(
            self.global_avg_pool(x),
            size=x.shape[-2:],
            mode='bilinear',
            align_corners=False
        ))
        return self.final_conv(torch.cat(res, dim=1))


class SemanticDecoder(nn.Module):
    """Decoder for semantic segmentation task."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        in_channels = config['model']['fpn_channels']
        num_classes = config['model']['num_classes']['semantic']
        
        self.aspp = ASPP(in_channels, 256, rates=[6, 12, 18])
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        x = features[-1]  # Use the last FPN feature
        x = self.aspp(x)
        x = self.decoder(x)
        return x


class DetectionDecoder(nn.Module):
    """Decoder for object detection task using anchor-based approach."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        in_channels = config['model']['fpn_channels']
        num_classes = config['model']['num_classes']['detection']
        num_anchors = config['model']['detection']['num_anchors']
        
        self.cls_head = nn.ModuleList([
            nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
            for _ in range(len(config['model']['detection']['anchor_sizes']))
        ])
        
        self.reg_head = nn.ModuleList([
            nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)
            for _ in range(len(config['model']['detection']['anchor_sizes']))
        ])
        
    def forward(self, features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        cls_preds = []
        reg_preds = []
        
        for i, feature in enumerate(features):
            cls_preds.append(self.cls_head[i](feature))
            reg_preds.append(self.reg_head[i](feature))
            
        return cls_preds, reg_preds


class ClassificationDecoder(nn.Module):
    """Decoder for stain classification task."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        in_channels = config['model']['fpn_channels']
        num_classes = config['model']['num_classes']['classification']
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        x = features[-1]  # Use the last FPN feature
        x = self.pool(x).flatten(1)
        return self.fc(x)


class DepthDecoder(nn.Module):
    """Decoder for depth estimation task."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        in_channels = config['model']['fpn_channels']
        
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()  # Normalize depth to [0, 1]
        )
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        x = features[-1]  # Use the last FPN feature
        return self.decoder(x)
