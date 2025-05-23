import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

from .backbone import ResNetFPN
from .decoders import (
    SemanticDecoder,
    DetectionDecoder,
    ClassificationDecoder,
    DepthDecoder
)


class MultiTaskModel(nn.Module):
    """Multi-task perception model for autonomous vehicles."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Initialize backbone
        self.backbone = ResNetFPN(config)
        
        # Initialize task-specific decoders
        self.semantic_decoder = SemanticDecoder(config)
        self.detection_decoder = DetectionDecoder(config)
        self.classification_decoder = ClassificationDecoder(config)
        self.depth_decoder = DepthDecoder(config)
        
        # Initialize uncertainty weights if enabled
        if config['training']['loss']['uncertainty_weighting']:
            self.log_vars = nn.Parameter(torch.zeros(4))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features from backbone
        features = self.backbone(x)
        
        # Get predictions from each decoder
        semantic_pred = self.semantic_decoder(features)
        detection_cls, detection_reg = self.detection_decoder(features)
        classification_pred = self.classification_decoder(features)
        depth_pred = self.depth_decoder(features)
        
        # Return all predictions
        return {
            'semantic': semantic_pred,
            'detection_cls': detection_cls,
            'detection_reg': detection_reg,
            'classification': classification_pred,
            'depth': depth_pred
        }
    
    @torch.jit.export
    def get_uncertainty_weights(self) -> torch.Tensor:
        """Get the uncertainty weights for loss computation."""
        if self.config['training']['loss']['uncertainty_weighting']:
            return torch.exp(-self.log_vars)
        else:
            return torch.ones(4, device=next(self.parameters()).device)
    
    def export_onnx(self, save_path: str) -> None:
        """Export model to ONNX format."""
        self.eval()
        dummy_input = torch.randn(
            1, 3,
            self.config['model']['input_size'][0],
            self.config['model']['input_size'][1]
        )
        
        torch.onnx.export(
            self,
            dummy_input,
            save_path,
            opset_version=self.config['deployment']['onnx']['opset_version'],
            input_names=['input'],
            output_names=[
                'semantic',
                'detection_cls',
                'detection_reg',
                'classification',
                'depth'
            ],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'semantic': {0: 'batch_size'},
                'detection_cls': {0: 'batch_size'},
                'detection_reg': {0: 'batch_size'},
                'classification': {0: 'batch_size'},
                'depth': {0: 'batch_size'}
            } if self.config['deployment']['onnx']['dynamic_axes'] else None
        )
