# YOLO Module

This folder contains YOLO-based object detection components for building safety monitoring.

## Files:
- `improved_object_detector.py` - High-accuracy YOLOv8m/YOLOv9c object detector
- `detector.py` - Legacy YOLO-World detector (reference)
- `main_config.yaml` - Detection configuration
- `stage1_config.yaml` - Stage 1 specific settings
- Model files: `yolov8*.pt` - Pre-trained YOLO models

## Purpose:
Provides high-accuracy object detection for regular building items (person, bottles, chairs, laptops, etc.) using COCO-trained models with building-relevant filtering.

## Features:
- YOLOv8m for better accuracy vs YOLOv8n
- COCO 80 classes with building-relevant filtering
- Priority-based detection (security items, valuables, safety equipment)
- TensorRT optimization support
- Cross-platform compatibility (CUDA/MPS/CPU)