#!/usr/bin/env python3
"""
COCO Dataset Loader for Spill Segmentation
SCOPE Smart Building System - BiSeNet V2 Training

Loads and processes COCO format annotations for spill detection
Categories: 'liquid' (id=0), 'spill' (id=1)
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
import cv2
from pathlib import Path


class COCOSpillDataset(Dataset):
    """
    COCO format dataset for spill segmentation
    
    Expected directory structure:
    data/
    ├── train/
    │   ├── images/
    │   └── _annotations.coco.json
    ├── valid/
    │   ├── images/
    │   └── _annotations.coco.json
    └── test/
        ├── images/
        └── _annotations.coco.json
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (512, 512),
        transforms: Optional[A.Compose] = None,
        binary_segmentation: bool = True
    ):
        """
        Initialize COCO Spill Dataset
        
        Args:
            data_dir: Path to data directory
            split: Dataset split ('train', 'valid', 'test')
            image_size: Target image size (H, W)
            transforms: Albumentations transforms
            binary_segmentation: Convert to binary segmentation (background/spill)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.binary_segmentation = binary_segmentation
        
        # Setup paths
        self.split_dir = self.data_dir / split
        self.images_dir = self.split_dir
        self.annotations_file = self.split_dir / '_annotations.coco.json'
        
        # Verify paths exist
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")
        
        # Load COCO annotations
        self.coco_data = self._load_coco_annotations()
        self.images_info = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.categories = self.coco_data['categories']
        
        # Create mappings
        self.image_id_to_info = {img['id']: img for img in self.images_info}
        self.image_id_to_annotations = self._group_annotations_by_image()
        
        # Setup category mapping
        self.category_mapping = self._setup_category_mapping()
        
        # Setup transforms
        self.transforms = transforms or self._get_default_transforms()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loaded {len(self.images_info)} images for {split} split")
        self.logger.info(f"Categories: {[cat['name'] for cat in self.categories]}")
    
    def _load_coco_annotations(self) -> Dict[str, Any]:
        """Load COCO format annotations"""
        with open(self.annotations_file, 'r') as f:
            coco_data = json.load(f)
        return coco_data
    
    def _group_annotations_by_image(self) -> Dict[int, List[Dict]]:
        """Group annotations by image ID"""
        grouped = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            if image_id not in grouped:
                grouped[image_id] = []
            grouped[image_id].append(ann)
        return grouped
    
    def _setup_category_mapping(self) -> Dict[int, int]:
        """
        Setup category ID mapping
        Original: liquid=0, spill=1
        For binary: background=0, spill=1 (combine liquid and spill)
        """
        if self.binary_segmentation:
            # Map both liquid and spill to class 1 (spill)
            mapping = {}
            for cat in self.categories:
                if cat['name'] in ['liquid', 'spill']:
                    mapping[cat['id']] = 1  # Spill class
                else:
                    mapping[cat['id']] = 0  # Background
            return mapping
        else:
            # Keep original mapping
            return {cat['id']: cat['id'] for cat in self.categories}
    
    def _get_default_transforms(self) -> A.Compose:
        """Get default transforms based on split"""
        if self.split == 'train':
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5
                ),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def _decode_rle(self, rle: Dict[str, Any], shape: Tuple[int, int]) -> np.ndarray:
        """
        Decode COCO RLE format to binary mask
        
        Args:
            rle: RLE format annotation
            shape: Image shape (H, W)
            
        Returns:
            Binary mask
        """
        if isinstance(rle, dict) and 'counts' in rle:
            # RLE format
            from pycocotools import mask as coco_mask
            mask = coco_mask.decode(rle)
        else:
            # Polygon format - convert to mask
            mask = self._polygon_to_mask(rle, shape)
        
        return mask
    
    def _polygon_to_mask(self, polygons: List[float], shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert polygon coordinates to binary mask
        
        Args:
            polygons: List of polygon coordinates [x1, y1, x2, y2, ...]
            shape: Image shape (H, W)
            
        Returns:
            Binary mask
        """
        mask = np.zeros(shape, dtype=np.uint8)
        
        # Reshape coordinates to (N, 2)
        coords = np.array(polygons).reshape(-1, 2)
        coords = coords.astype(np.int32)
        
        # Fill polygon
        cv2.fillPoly(mask, [coords], 1)
        
        return mask
    
    def _create_segmentation_mask(self, image_info: Dict, annotations: List[Dict]) -> np.ndarray:
        """
        Create segmentation mask from annotations
        
        Args:
            image_info: Image information from COCO
            annotations: List of annotations for the image
            
        Returns:
            Segmentation mask (H, W) with class IDs
        """
        height, width = image_info['height'], image_info['width']
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for ann in annotations:
            category_id = ann['category_id']
            mapped_id = self.category_mapping.get(category_id, 0)
            
            if 'segmentation' in ann and ann['segmentation']:
                seg = ann['segmentation']
                
                if isinstance(seg, dict):
                    # RLE format
                    obj_mask = self._decode_rle(seg, (height, width))
                elif isinstance(seg, list) and len(seg) > 0:
                    # Polygon format
                    if isinstance(seg[0], list):
                        # Multiple polygons
                        obj_mask = np.zeros((height, width), dtype=np.uint8)
                        for polygon in seg:
                            poly_mask = self._polygon_to_mask(polygon, (height, width))
                            obj_mask = np.maximum(obj_mask, poly_mask)
                    else:
                        # Single polygon
                        obj_mask = self._polygon_to_mask(seg, (height, width))
                else:
                    continue
                
                # Apply category mapping
                mask[obj_mask == 1] = mapped_id
        
        return mask
    
    def __len__(self) -> int:
        return len(self.images_info)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary with 'image' and 'mask' tensors
        """
        # Get image info
        image_info = self.images_info[idx]
        image_id = image_info['id']
        image_filename = image_info['file_name']
        
        # Load image
        image_path = self.images_dir / image_filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        annotations = self.image_id_to_annotations.get(image_id, [])
        
        # Create segmentation mask
        mask = self._create_segmentation_mask(image_info, annotations)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert mask to long tensor for cross-entropy loss
        if isinstance(mask, torch.Tensor):
            mask = mask.long()
        else:
            mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'image_id': image_id,
            'filename': image_filename
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for balanced training
        
        Returns:
            Class weights tensor
        """
        self.logger.info("Calculating class weights...")
        
        class_counts = np.zeros(2 if self.binary_segmentation else len(self.categories))
        total_pixels = 0
        
        for idx in range(len(self)):
            item = self.__getitem__(idx)
            mask = item['mask'].numpy() if isinstance(item['mask'], torch.Tensor) else item['mask']
            
            unique, counts = np.unique(mask, return_counts=True)
            for class_id, count in zip(unique, counts):
                if class_id < len(class_counts):
                    class_counts[class_id] += count
                    total_pixels += count
        
        # Calculate weights (inverse frequency)
        class_weights = total_pixels / (len(class_counts) * class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        
        self.logger.info(f"Class weights: {class_weights}")
        return torch.from_numpy(class_weights).float()


def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 512),
    binary_segmentation: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for training
        num_workers: Number of worker processes
        image_size: Target image size
        binary_segmentation: Use binary segmentation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = COCOSpillDataset(
        data_dir=data_dir,
        split='train',
        image_size=image_size,
        binary_segmentation=binary_segmentation
    )
    
    val_dataset = COCOSpillDataset(
        data_dir=data_dir,
        split='valid',
        image_size=image_size,
        binary_segmentation=binary_segmentation
    )
    
    test_dataset = COCOSpillDataset(
        data_dir=data_dir,
        split='test',
        image_size=image_size,
        binary_segmentation=binary_segmentation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def test_dataset_loading(data_dir: str = "data"):
    """Test dataset loading functionality"""
    import matplotlib.pyplot as plt
    
    print("Testing COCO Spill Dataset loading...")
    
    try:
        # Create dataset
        dataset = COCOSpillDataset(
            data_dir=data_dir,
            split='train',
            image_size=(512, 512)
        )
        
        print(f"Dataset loaded successfully!")
        print(f"Number of images: {len(dataset)}")
        print(f"Categories: {[cat['name'] for cat in dataset.categories]}")
        
        # Test loading a few samples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            image = sample['image']
            mask = sample['mask']
            
            print(f"Sample {i}:")
            print(f"  Image shape: {image.shape}")
            print(f"  Mask shape: {mask.shape}")
            print(f"  Mask unique values: {torch.unique(mask)}")
            print(f"  Filename: {sample['filename']}")
        
        # Test class weights
        class_weights = dataset.get_class_weights()
        print(f"Class weights: {class_weights}")
        
        print("\n✅ Dataset test passed!")
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        raise


if __name__ == "__main__":
    # Test the dataset
    test_dataset_loading()