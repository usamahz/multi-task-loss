import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple


class FocalLoss(nn.Module):
    """Focal Loss for object detection classification."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self,
        pred: List[torch.Tensor],
        target: torch.Tensor
    ) -> torch.Tensor:
        # Concatenate predictions from all FPN levels
        # Each pred tensor has shape [batch_size, num_anchors * num_classes, height, width]
        # We need to reshape to [batch_size, num_anchors, num_classes, height * width]
        batch_size = pred[0].size(0)
        num_classes = pred[0].size(1) // 9  # 9 is num_anchors from config
        
        reshaped_preds = []
        for p in pred:
            # Reshape to [batch_size, num_anchors, num_classes, height * width]
            p = p.view(batch_size, 9, num_classes, -1)
            # Move num_classes to last dimension for easier indexing
            p = p.permute(0, 1, 3, 2)
            reshaped_preds.append(p)
        
        # Concatenate along height * width dimension
        pred = torch.cat(reshaped_preds, dim=2)
        
        # Create one-hot target tensor
        # target has shape [batch_size, num_objects]
        # We need to expand it to match pred shape
        target = target.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, num_objects]
        target = target.expand(-1, 9, pred.size(2), -1)  # [batch_size, num_anchors, height*width, num_objects]
        
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(-1, target, 1)
        
        pred_sigmoid = pred.sigmoid()
        pt = (1 - pred_sigmoid) * target_one_hot + pred_sigmoid * (1 - target_one_hot)
        focal_weight = (
            self.alpha * target_one_hot + (1 - self.alpha) * (1 - target_one_hot)
        ) * pt.pow(self.gamma)
        
        loss = F.binary_cross_entropy_with_logits(
            pred, target_one_hot, reduction='none'
        ) * focal_weight
        
        return loss.mean()


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes.
    
    Args:
        boxes1: [..., 4] (x1, y1, x2, y2)
        boxes2: [..., 4] (x1, y1, x2, y2)
    
    Returns:
        IoU: [...]
    """
    # Get coordinates
    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
    
    # Compute intersection area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Compute union area
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - intersection
    
    # Compute IoU
    iou = intersection / (union + 1e-6)
    
    return iou


class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss for object detection regression."""
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(
        self,
        pred: List[torch.Tensor],
        target: torch.Tensor
    ) -> torch.Tensor:
        # Concatenate predictions from all FPN levels
        # Each pred tensor has shape [batch_size, num_anchors * 4, height, width]
        # We need to reshape to [batch_size, num_anchors, 4, height * width]
        batch_size = pred[0].size(0)
        
        reshaped_preds = []
        for p in pred:
            # Reshape to [batch_size, num_anchors, 4, height * width]
            p = p.view(batch_size, 9, 4, -1)
            # Move 4 to last dimension for easier indexing
            p = p.permute(0, 1, 3, 2)
            reshaped_preds.append(p)
        
        # Concatenate along height * width dimension
        pred = torch.cat(reshaped_preds, dim=2)  # [batch_size, num_anchors, height*width, 4]
        
        # Reshape target to match pred shape
        # target has shape [batch_size, num_objects, 4]
        # We need to expand it to [batch_size, num_anchors, height*width, num_objects, 4]
        target = target.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, num_objects, 4]
        target = target.expand(batch_size, 9, pred.size(2), -1, 4)  # [batch_size, num_anchors, height*width, num_objects, 4]
        
        # Reshape pred for IoU computation
        pred = pred.unsqueeze(3)  # [batch_size, num_anchors, height*width, 1, 4]
        pred = pred.expand(batch_size, 9, pred.size(2), target.size(3), 4)  # [batch_size, num_anchors, height*width, num_objects, 4]
        
        # Compute IoU between each anchor and target box
        ious = compute_iou(pred, target)  # [batch_size, num_anchors, height*width, num_objects]
        
        # For each spatial location, find the target with highest IoU
        best_ious, best_target_idx = ious.max(dim=-1)  # [batch_size, num_anchors, height*width]
        
        # Create a mask for positive anchors (IoU > 0.5)
        pos_mask = best_ious > 0.5
        
        # Get the target boxes for positive anchors
        best_target_idx = best_target_idx.unsqueeze(-1).unsqueeze(-1)  # [batch_size, num_anchors, height*width, 1, 1]
        best_target_idx = best_target_idx.expand(batch_size, 9, pred.size(2), 1, 4)  # [batch_size, num_anchors, height*width, 1, 4]
        target_boxes = target.gather(3, best_target_idx).squeeze(3)  # [batch_size, num_anchors, height*width, 4]
        
        # Only compute loss for positive anchors
        if pos_mask.sum() > 0:
            # Reshape predictions and targets to match
            pred_boxes = pred[pos_mask].view(-1, 4)  # [num_pos, 4]
            target_boxes = target_boxes[pos_mask].view(-1, 4)  # [num_pos, 4]
            
            loss = F.smooth_l1_loss(
                pred_boxes,
                target_boxes,
                beta=self.beta,
                reduction='sum'
            )
            
            # Normalize by number of positive anchors
            num_pos = pos_mask.sum().clamp(min=1)
            return loss / num_pos
        else:
            # Return zero loss if no positive anchors
            return torch.tensor(0.0, device=pred.device)


class DepthLoss(nn.Module):
    """L1 Loss for depth estimation with size matching."""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        # Resize prediction to match target size
        pred = F.interpolate(
            pred,
            size=target.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        return F.l1_loss(pred, target)


class MultiTaskLoss(nn.Module):
    """Multi-task loss with uncertainty weighting."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Initialize task-specific losses
        self.semantic_loss = nn.CrossEntropyLoss()
        self.detection_cls_loss = FocalLoss()
        self.detection_reg_loss = SmoothL1Loss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.depth_loss = DepthLoss()  # Use custom depth loss
        
        # Get loss weights from config
        self.weights = {
            'semantic': config['training']['loss']['semantic_weight'],
            'detection': config['training']['loss']['detection_weight'],
            'classification': config['training']['loss']['classification_weight'],
            'depth': config['training']['loss']['depth_weight']
        }
    
    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        uncertainty_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-task loss with uncertainty weighting."""
        
        # Compute individual task losses
        semantic_loss = self.semantic_loss(
            preds['semantic'],
            targets['semantic']
        )
        
        detection_cls_loss = self.detection_cls_loss(
            preds['detection_cls'],
            targets['detection_cls']
        )
        
        detection_reg_loss = self.detection_reg_loss(
            preds['detection_reg'],
            targets['detection_boxes']
        )
        
        detection_loss = detection_cls_loss + detection_reg_loss
        
        classification_loss = self.classification_loss(
            preds['classification'],
            targets['classification']
        )
        
        depth_loss = self.depth_loss(
            preds['depth'],
            targets['depth']
        )
        
        # Apply uncertainty weighting if enabled
        if self.config['training']['loss']['uncertainty_weighting']:
            losses = torch.stack([
                semantic_loss,
                detection_loss,
                classification_loss,
                depth_loss
            ])
            
            # Apply uncertainty weights
            weighted_losses = losses * uncertainty_weights
            
            # Add regularization term
            total_loss = weighted_losses.sum() + torch.log(
                uncertainty_weights.prod()
            )
        else:
            # Apply fixed weights
            total_loss = (
                self.weights['semantic'] * semantic_loss +
                self.weights['detection'] * detection_loss +
                self.weights['classification'] * classification_loss +
                self.weights['depth'] * depth_loss
            )
        
        # Return total loss and individual losses for logging
        loss_dict = {
            'total': total_loss.item(),
            'semantic': semantic_loss.item(),
            'detection': detection_loss.item(),
            'classification': classification_loss.item(),
            'depth': depth_loss.item()
        }
        
        return total_loss, loss_dict
