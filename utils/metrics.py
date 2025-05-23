import torch
import numpy as np
from typing import Dict, Any, List, Tuple
import torch.nn.functional as F


def compute_semantic_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int
) -> Dict[str, float]:
    """Compute semantic segmentation metrics."""
    # Convert to numpy
    pred = pred.argmax(dim=1).cpu().numpy()
    target = target.cpu().numpy()
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # Compute confusion matrix
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = np.sum((pred == i) & (target == j))
    
    # Compute metrics
    iou = []
    for i in range(num_classes):
        if confusion_matrix[i, i] == 0:
            iou.append(0)
        else:
            iou.append(
                confusion_matrix[i, i] / (
                    np.sum(confusion_matrix[i, :]) +
                    np.sum(confusion_matrix[:, i]) -
                    confusion_matrix[i, i]
                )
            )
    
    # Compute mean IoU
    mean_iou = np.mean(iou)
    
    # Compute pixel accuracy
    pixel_acc = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    
    return {
        'mean_iou': mean_iou,
        'pixel_acc': pixel_acc
    }


def compute_detection_metrics(
    pred_cls: List[torch.Tensor],
    pred_reg: List[torch.Tensor],
    target_cls: List[torch.Tensor],
    target_boxes: List[torch.Tensor],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """Compute object detection metrics."""
    # Convert predictions to boxes
    pred_boxes = []
    pred_scores = []
    for cls, reg in zip(pred_cls, pred_reg):
        # Get class scores
        scores, labels = torch.max(cls, dim=1)
        
        # Convert regression to boxes
        boxes = convert_reg_to_boxes(reg)
        
        pred_boxes.append(boxes)
        pred_scores.append(scores)
    
    # Compute mAP
    mean_ap = compute_map(
        pred_boxes,
        pred_scores,
        target_boxes,
        target_cls,
        iou_threshold
    )
    
    return {
        'mAP': mean_ap
    }


def compute_classification_metrics(
    pred: torch.Tensor,
    target: torch.Tensor
) -> Dict[str, float]:
    """Compute classification metrics."""
    # Convert to numpy
    pred = pred.argmax(dim=1).cpu().numpy()
    target = target.cpu().numpy()
    
    # Compute accuracy
    accuracy = np.mean(pred == target)
    
    return {
        'accuracy': accuracy
    }


def compute_depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor
) -> Dict[str, float]:
    """Compute depth estimation metrics."""
    # Resize prediction to match target size
    pred = F.interpolate(
        pred,
        size=target.shape[-2:],
        mode='bilinear',
        align_corners=False
    )
    
    # Convert to numpy
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    # Compute absolute relative error
    abs_rel = np.mean(np.abs(pred - target) / target)
    
    # Compute squared relative error
    sq_rel = np.mean(((pred - target) ** 2) / target)
    
    # Compute RMSE
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    
    # Compute RMSE log
    rmse_log = np.sqrt(np.mean((np.log(pred) - np.log(target)) ** 2))
    
    # Compute threshold accuracy
    thresh = np.maximum((target / pred), (pred / target))
    a1 = np.mean(thresh < 1.25)
    a2 = np.mean(thresh < 1.25 ** 2)
    a3 = np.mean(thresh < 1.25 ** 3)
    
    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'a1': a1,
        'a2': a2,
        'a3': a3
    }


def compute_metrics(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """Compute all task-specific metrics."""
    metrics = {}
    
    # Compute semantic segmentation metrics
    semantic_metrics = compute_semantic_metrics(
        preds['semantic'],
        targets['semantic'],
        num_classes=4  # From config
    )
    metrics.update({f'semantic_{k}': v for k, v in semantic_metrics.items()})
    
    # Compute detection metrics
    detection_metrics = compute_detection_metrics(
        preds['detection_cls'],
        preds['detection_reg'],
        targets['detection_cls'],
        targets['detection_boxes']
    )
    metrics.update({f'detection_{k}': v for k, v in detection_metrics.items()})
    
    # Compute classification metrics
    classification_metrics = compute_classification_metrics(
        preds['classification'],
        targets['classification']
    )
    metrics.update({
        f'classification_{k}': v
        for k, v in classification_metrics.items()
    })
    
    # Compute depth metrics
    depth_metrics = compute_depth_metrics(
        preds['depth'],
        targets['depth']
    )
    metrics.update({f'depth_{k}': v for k, v in depth_metrics.items()})
    
    return metrics


def convert_reg_to_boxes(reg: torch.Tensor) -> torch.Tensor:
    """Convert regression values to bounding boxes."""
    # This is a simplified version - you'll need to adapt this
    # based on your actual box encoding format
    return reg


def compute_map(
    pred_boxes: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    target_boxes: List[torch.Tensor],
    target_labels: List[torch.Tensor],
    iou_threshold: float
) -> float:
    """Compute mean Average Precision."""
    # This is a simplified version - you'll need to adapt this
    # based on your actual evaluation protocol
    return 0.0
