import os
import cv2
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from typing import Dict, Any, Tuple, List
from PIL import Image
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf


class MultiTaskDataset(Dataset):
    """Dataset for multi-task perception with semantic segmentation,
    object detection, classification, and depth estimation."""
    
    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        super().__init__()
        self.config = config
        self.split = split
        
        # Set data path based on split
        self.data_path = to_absolute_path(config['data'][f'{split}_path'])
        
        # Load data list
        self.data_list = self._load_data_list()
        
        # Initialize augmentations
        self.transform = self._get_transforms()
    
    def _load_data_list(self) -> List[Dict[str, str]]:
        """Load list of data samples with their paths."""
        data_list = []
        
        # Get images directory
        images_dir = os.path.join(self.data_path, 'images')
        
        # Walk through images directory
        for file in os.listdir(images_dir):
            if file.endswith('.jpg') or file.endswith('.png'):
                # Get image path
                img_path = os.path.join(images_dir, file)
                
                # Get corresponding annotation paths
                base_name = os.path.splitext(file)[0]
                semantic_path = os.path.join(
                    self.data_path, 'semantic', f'{base_name}.png'
                )
                detection_path = os.path.join(
                    self.data_path, 'detection', f'{base_name}.json'
                )
                classification_path = os.path.join(
                    self.data_path, 'classification', f'{base_name}.json'
                )
                depth_path = os.path.join(
                    self.data_path, 'depth', f'{base_name}.png'
                )
                
                # Add to data list if all files exist
                if all(os.path.exists(p) for p in [
                    img_path, semantic_path, detection_path,
                    classification_path, depth_path
                ]):
                    data_list.append({
                        'image': img_path,
                        'semantic': semantic_path,
                        'detection': detection_path,
                        'classification': classification_path,
                        'depth': depth_path
                    })
        
        return data_list
    
    def _get_transforms(self) -> A.Compose:
        """Get data augmentation transforms."""
        if self.split == 'train' and self.config['data']['augmentation']['enabled']:
            return A.Compose([
                A.RandomResizedCrop(
                    size=(self.config['model']['input_size'][0],
                          self.config['model']['input_size'][1]),
                    scale=(0.8, 1.0)
                ),
                A.HorizontalFlip(
                    p=self.config['data']['augmentation']['horizontal_flip']
                ),
                A.VerticalFlip(
                    p=self.config['data']['augmentation']['vertical_flip']
                ),
                A.Rotate(
                    limit=self.config['data']['augmentation']['rotation']
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=self.config['data']['augmentation']['brightness_contrast'],
                    contrast_limit=self.config['data']['augmentation']['brightness_contrast']
                ),
                A.GaussianBlur(
                    p=self.config['data']['augmentation']['blur']
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ], is_check_shapes=False)
        else:
            return A.Compose([
                A.Resize(*self.config['model']['input_size']),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def _load_detection_targets(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load detection targets from JSON file."""
        import json
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert annotations to tensors
        # This is a simplified version - you'll need to adapt this
        # based on your actual annotation format
        boxes = torch.tensor(data['boxes'], dtype=torch.float32)
        labels = torch.tensor(data['labels'], dtype=torch.long)
        
        return boxes, labels
    
    def _load_classification_target(self, path: str) -> torch.Tensor:
        """Load classification target from JSON file."""
        import json
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return torch.tensor(data['label'], dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get data paths
        data = self.data_list[idx]
        
        # Load image
        image = cv2.imread(data['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load semantic segmentation mask
        semantic = cv2.imread(data['semantic'], cv2.IMREAD_GRAYSCALE)
        
        # Load depth map
        depth = cv2.imread(data['depth'], cv2.IMREAD_GRAYSCALE)
        depth = depth.astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Apply augmentations
        if self.transform:
            transformed = self.transform(
                image=image,
                mask=semantic,
                depth=depth
            )
            image = transformed['image']
            semantic = transformed['mask']
            depth = transformed['depth']
        
        # Resize semantic mask to 16×16 AFTER augmentation
        semantic = cv2.resize(semantic, (16, 16), interpolation=cv2.INTER_NEAREST)
        
        # Load detection targets
        detection_boxes, detection_labels = self._load_detection_targets(
            data['detection']
        )
        
        # Load classification target
        classification = self._load_classification_target(data['classification'])
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)
        semantic = torch.from_numpy(semantic).long()
        depth = torch.from_numpy(depth).unsqueeze(0)
        
        return {
            'image': image,
            'targets': {
                'semantic': semantic,
                'detection_boxes': detection_boxes,
                'detection_cls': detection_labels,
                'classification': classification,
                'depth': depth
            }
        }
