# DeepLabV3+ Module

This folder contains DeepLabV3+ segmentation components for precise spill detection.

## Files:
- `deeplabv3_spill_detector.py` - DeepLabV3+ with MobileNetV3 backbone for spill segmentation
- `coco_training.py` - Training script adapted for segmentation
- `test_image_inference.py` - Single image testing and visualization
- `model.py` - Model architecture definitions

## Purpose:
Provides pixel-level spill detection using semantic segmentation instead of object detection for better accuracy and precise spill boundaries.

## Features:
- Pre-trained DeepLabV3+ with MobileNetV3 backbone
- 512x512 input resolution for detailed segmentation
- Fast mode (384x384) for real-time Stage 1 processing
- Precise spill boundary detection with contour analysis