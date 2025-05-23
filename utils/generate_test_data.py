import os
import json
import numpy as np
import cv2
from PIL import Image

def create_sample_data():
    # Create sample image
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Create sample semantic mask (4 classes)
    semantic = np.random.randint(0, 4, (512, 512), dtype=np.uint8)
    
    # Create sample depth map
    depth = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    
    # Create sample detection annotations
    detection = {
        'boxes': [[100, 100, 200, 200], [300, 300, 400, 400]],
        'labels': [0, 1]  # 0: vehicle, 1: pedestrian
    }
    
    # Create sample classification annotation
    classification = {
        'label': np.random.randint(0, 2)  # 0: no_stain, 1: stain
    }
    
    return img, semantic, depth, detection, classification

def save_sample_data(split='train', num_samples=10):
    base_dir = f'data/{split}'
    
    for i in range(num_samples):
        # Generate sample data
        img, semantic, depth, detection, classification = create_sample_data()
        
        # Save image
        img_path = os.path.join(base_dir, 'images', f'sample_{i}.jpg')
        cv2.imwrite(img_path, img)
        
        # Save semantic mask
        semantic_path = os.path.join(base_dir, 'semantic', f'sample_{i}.png')
        cv2.imwrite(semantic_path, semantic)
        
        # Save depth map
        depth_path = os.path.join(base_dir, 'depth', f'sample_{i}.png')
        cv2.imwrite(depth_path, depth)
        
        # Save detection annotations
        detection_path = os.path.join(base_dir, 'detection', f'sample_{i}.json')
        with open(detection_path, 'w') as f:
            json.dump(detection, f)
        
        # Save classification annotation
        classification_path = os.path.join(base_dir, 'classification', f'sample_{i}.json')
        with open(classification_path, 'w') as f:
            json.dump(classification, f)

if __name__ == '__main__':
    # Create sample data for train, val, and test sets
    save_sample_data('train', num_samples=10)
    save_sample_data('val', num_samples=5)
    save_sample_data('test', num_samples=5) 