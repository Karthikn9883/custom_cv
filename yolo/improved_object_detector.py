#!/usr/bin/env python3
"""
Improved Object Detector - SCOPE Smart Building System
High-accuracy object detection using YOLOv8m or YOLOv9c
Optimized for detecting regular items (person, bottles, chairs, etc.)
"""

import cv2
import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics")
    exit(1)

from cuda_memory_manager import get_memory_manager

class ImprovedObjectDetector:
    """
    High-accuracy object detector using YOLOv8m/YOLOv9c
    Optimized for common building objects with better accuracy than YOLOv8n
    """
    
    def __init__(self, model_name: str = "yolov8m.pt", device: Optional[torch.device] = None):
        self.device = device or self._get_optimal_device()
        self.model = None
        self.model_name = model_name
        self.memory_manager = get_memory_manager()
        
        # Performance tracking
        self.inference_times = []
        self.detection_count = 0
        
        # COCO class names for building objects
        self.building_classes = self._get_building_classes()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self._initialize_model()
        
        self.logger.info(f"Improved Object Detector ({model_name}) initialized on {self.device}")
    
    def _get_optimal_device(self) -> torch.device:
        """Get optimal device for inference"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _get_building_classes(self) -> Dict[int, str]:
        """Get COCO classes relevant for building environments"""
        # COCO class mappings for building-relevant objects
        coco_building_classes = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            6: 'train',
            7: 'truck',
            8: 'boat',
            15: 'cat',
            16: 'dog',
            24: 'backpack',
            25: 'umbrella',
            26: 'handbag',
            27: 'tie',
            28: 'suitcase',
            31: 'skis',
            32: 'snowboard',
            33: 'sports ball',
            34: 'kite',
            35: 'baseball bat',
            36: 'baseball glove',
            37: 'skateboard',
            38: 'surfboard',
            39: 'tennis racket',
            40: 'bottle',
            41: 'wine glass',
            42: 'cup',
            43: 'fork',
            44: 'knife',
            45: 'spoon',
            46: 'bowl',
            56: 'chair',
            57: 'couch',
            58: 'potted plant',
            59: 'bed',
            60: 'dining table',
            61: 'toilet',
            62: 'tv',
            63: 'laptop',
            64: 'mouse',
            65: 'remote',
            66: 'keyboard',
            67: 'cell phone',
            68: 'microwave',
            69: 'oven',
            70: 'toaster',
            71: 'sink',
            72: 'refrigerator',
            73: 'book',
            74: 'clock',
            75: 'vase',
            76: 'scissors',
            77: 'teddy bear',
            78: 'hair drier',
            79: 'toothbrush'
        }
        return coco_building_classes
    
    def _initialize_model(self):
        """Initialize YOLO model"""
        try:
            # Check for TensorRT engine first
            engine_path = self.model_name.replace('.pt', '.engine')
            
            if Path(engine_path).exists() and self.device.type == 'cuda':
                self.logger.info(f"Loading TensorRT engine: {engine_path}")
                self.model = YOLO(engine_path)
            else:
                self.logger.info(f"Loading YOLO model: {self.model_name}")
                self.model = YOLO(self.model_name)
            
            # Optimize model
            self._optimize_model()
            
            # Warmup
            self._warmup_model()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize object detector: {e}")
            raise
    
    def _optimize_model(self):
        """Apply device-specific optimizations"""
        try:
            # Move model to device
            if hasattr(self.model.model, 'to'):
                self.model.model.to(self.device)
            
            # Enable half precision for CUDA/MPS
            if self.device.type in ['cuda', 'mps']:
                try:
                    if hasattr(self.model.model, 'half'):
                        self.model.model.half()
                        self.logger.info("Half precision (FP16) enabled")
                except Exception as e:
                    self.logger.warning(f"Half precision failed: {e}")
            
            # Set to evaluation mode
            if hasattr(self.model.model, 'eval'):
                self.model.model.eval()
            
            # CUDA-specific optimizations
            if self.device.type == 'cuda':
                try:
                    # Enable optimizations
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    
                    # Torch compile (if supported and not TensorRT)
                    if (hasattr(torch, 'compile') and 
                        not self.model_name.endswith('.engine') and
                        hasattr(self.model.model, 'forward')):
                        self.model.model = torch.compile(self.model.model, mode='reduce-overhead')
                        self.logger.info("Model compiled with torch.compile")
                        
                except Exception as e:
                    self.logger.warning(f"CUDA optimizations failed: {e}")
            
        except Exception as e:
            self.logger.warning(f"Some optimizations failed: {e}")
    
    def _warmup_model(self, num_iterations: int = 3):
        """Warmup model with dummy inputs"""
        self.logger.info(f"Warming up model ({num_iterations} iterations)...")
        
        dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        for i in range(num_iterations):
            try:
                _ = self.model(dummy_frame, verbose=False, conf=0.5, imgsz=640)
            except Exception as e:
                self.logger.warning(f"Warmup iteration {i} failed: {e}")
        
        self.logger.info("Model warmup completed")
    
    def detect_objects(self, frame: np.ndarray, 
                      confidence_threshold: float = 0.4,
                      iou_threshold: float = 0.5,
                      max_detections: int = 50,
                      image_size: int = 640) -> Dict[str, Any]:
        """
        Detect objects in frame
        
        Args:
            frame: Input image (H, W, 3) in BGR format
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections
            image_size: Input size for model
            
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        try:
            # Run YOLO inference
            results = self.model(
                frame,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=image_size,
                max_det=max_detections,
                verbose=False,
                device=self.device
            )
            
            # Process results
            detections = []
            annotated_frame = frame.copy()
            
            if results and len(results) > 0:
                result = results[0]
                
                # Get annotated frame
                annotated_frame = result.plot()
                
                # Extract detection information
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    # Batch convert to CPU for efficiency
                    bboxes = boxes.xyxy.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy().astype(int)
                    
                    # Process each detection
                    for i in range(len(boxes)):
                        class_id = classes[i]
                        confidence = float(confidences[i])
                        bbox = bboxes[i].astype(int)
                        
                        # Get class name
                        class_name = self.model.names.get(class_id, f"class_{class_id}")
                        
                        # Check if it's a building-relevant class
                        is_building_relevant = class_id in self.building_classes
                        
                        detection = {
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': tuple(bbox),  # (x1, y1, x2, y2)
                            'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                            'center': ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
                            'building_relevant': is_building_relevant
                        }
                        detections.append(detection)
            
            # Filter for building-relevant objects
            building_detections = [d for d in detections if d['building_relevant']]
            
            # Calculate statistics
            inference_time = (time.time() - start_time) * 1000
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            
            self.detection_count += 1
            
            # Group detections by class
            class_counts = {}
            for detection in building_detections:
                class_name = detection['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            return {
                'detections': detections,
                'building_detections': building_detections,
                'annotated_frame': annotated_frame,
                'detection_count': len(detections),
                'building_detection_count': len(building_detections),
                'class_counts': class_counts,
                'inference_time_ms': inference_time,
                'input_size': image_size,
                'confidence_threshold': confidence_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Object detection failed: {e}")
            return {
                'detections': [],
                'building_detections': [],
                'annotated_frame': frame,
                'detection_count': 0,
                'building_detection_count': 0,
                'class_counts': {},
                'inference_time_ms': (time.time() - start_time) * 1000,
                'error': str(e)
            }
    
    def filter_high_priority_objects(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter detections for high-priority objects in building context
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Filtered list of high-priority detections
        """
        # High priority classes for building safety
        high_priority_classes = {
            'person', 'laptop', 'cell phone', 'backpack', 'handbag', 
            'suitcase', 'bottle', 'chair', 'tv', 'book', 'scissors'
        }
        
        # Very high priority (security concerns)
        very_high_priority = {'person', 'laptop', 'cell phone', 'backpack', 'suitcase'}
        
        high_priority_detections = []
        
        for detection in detections:
            class_name = detection['class_name']
            
            # Add priority score
            if class_name in very_high_priority:
                detection['priority'] = 'very_high'
                high_priority_detections.append(detection)
            elif class_name in high_priority_classes:
                detection['priority'] = 'high'
                high_priority_detections.append(detection)
            elif detection['confidence'] > 0.7:  # High confidence other objects
                detection['priority'] = 'medium'
                high_priority_detections.append(detection)
        
        # Sort by priority and confidence
        priority_order = {'very_high': 3, 'high': 2, 'medium': 1}
        high_priority_detections.sort(
            key=lambda x: (priority_order[x['priority']], x['confidence']), 
            reverse=True
        )
        
        return high_priority_detections
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            'total_detections': self.detection_count,
            'device': str(self.device),
            'model_name': self.model_name
        }
        
        if self.inference_times:
            stats.update({
                'avg_inference_ms': np.mean(self.inference_times),
                'p95_inference_ms': np.percentile(self.inference_times, 95),
                'max_inference_ms': np.max(self.inference_times),
                'min_inference_ms': np.min(self.inference_times)
            })
        
        return stats
    
    def visualize_detections(self, frame: np.ndarray, results: Dict[str, Any], 
                           show_all: bool = False) -> np.ndarray:
        """
        Create detailed visualization of object detection results
        
        Args:
            frame: Original frame
            results: Detection results
            show_all: Show all detections or only building-relevant ones
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        detections = results['detections'] if show_all else results['building_detections']
        
        if not detections:
            cv2.putText(vis_frame, "No Objects Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return vis_frame
        
        # Color mapping for different priorities
        priority_colors = {
            'very_high': (0, 0, 255),    # Red
            'high': (0, 165, 255),       # Orange
            'medium': (0, 255, 255),     # Yellow
            'low': (255, 255, 255)       # White
        }
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            priority = detection.get('priority', 'low')
            
            # Get color based on priority
            color = priority_colors.get(priority, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{class_name}: {confidence:.2f}"
            if priority != 'low':
                label += f" ({priority})"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add summary information
        info_y = 30
        summary_texts = [
            f"Building Objects: {results['building_detection_count']}/{results['detection_count']}",
            f"Inference: {results['inference_time_ms']:.1f}ms",
            f"Model: {self.model_name}"
        ]
        
        for text in summary_texts:
            cv2.putText(vis_frame, text, (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info_y += 30
        
        # Add class counts
        if results['class_counts']:
            counts_text = "Detected: " + ", ".join([f"{k}({v})" for k, v in results['class_counts'].items()])
            cv2.putText(vis_frame, counts_text, (10, vis_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_frame
    
    def export_to_tensorrt(self, output_path: Optional[str] = None,
                          image_size: int = 640, 
                          precision: str = "fp16") -> bool:
        """Export model to TensorRT engine"""
        try:
            if self.device.type != 'cuda':
                self.logger.warning("TensorRT export requires CUDA device")
                return False
            
            if output_path is None:
                output_path = self.model_name.replace('.pt', '.engine')
            
            self.logger.info(f"Exporting to TensorRT: {output_path}")
            
            # Export parameters
            export_params = {
                'format': 'engine',
                'imgsz': image_size,
                'half': precision == 'fp16',
                'int8': precision == 'int8',
                'device': self.device,
                'verbose': True
            }
            
            success = self.model.export(**export_params)
            
            if success:
                self.logger.info(f"TensorRT export successful: {output_path}")
                return True
            else:
                self.logger.error("TensorRT export failed")
                return False
                
        except Exception as e:
            self.logger.error(f"TensorRT export error: {e}")
            return False

def main():
    """Test the improved object detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved Object Detector Test')
    parser.add_argument('--model', type=str, default='yolov8m.pt',
                       choices=['yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov9c.pt', 'yolov9e.pt'],
                       help='YOLO model to use')
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--webcam', action='store_true', help='Test with webcam')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold')
    parser.add_argument('--export-tensorrt', action='store_true', help='Export to TensorRT')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ImprovedObjectDetector(args.model)
    
    if args.export_tensorrt:
        success = detector.export_to_tensorrt()
        print(f"TensorRT export: {'✅ Success' if success else '❌ Failed'}")
        return
    
    if args.benchmark:
        print("🚀 Running performance benchmark...")
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Benchmark different input sizes
        for img_size in [320, 480, 640]:
            times = []
            for i in range(30):
                start = time.time()
                _ = detector.detect_objects(test_image, 
                                          confidence_threshold=0.5,
                                          image_size=img_size)
                times.append((time.time() - start) * 1000)
            
            print(f"Image size {img_size}: {np.mean(times):.1f}ms ± {np.std(times):.1f}ms")
        
        stats = detector.get_performance_stats()
        print(f"Device: {stats['device']}")
        print(f"Model: {stats['model_name']}")
        return
    
    if args.image:
        # Test with single image
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image {args.image}")
            return
        
        print("🔍 Detecting objects in image...")
        results = detector.detect_objects(image, confidence_threshold=args.conf)
        
        print(f"Total detections: {results['detection_count']}")
        print(f"Building objects: {results['building_detection_count']}")
        print(f"Inference time: {results['inference_time_ms']:.1f}ms")
        print(f"Class counts: {results['class_counts']}")
        
        # Filter high priority
        high_priority = detector.filter_high_priority_objects(results['building_detections'])
        print(f"High priority objects: {len(high_priority)}")
        
        # Show visualization
        vis_frame = detector.visualize_detections(image, results)
        cv2.imshow('Object Detection Results', vis_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    elif args.webcam:
        # Test with webcam
        cap = cv2.VideoCapture(0)
        
        print("🎥 Starting webcam object detection (press 'q' to quit)...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            results = detector.detect_objects(frame, confidence_threshold=args.conf)
            
            # Visualize
            vis_frame = detector.visualize_detections(frame, results)
            
            cv2.imshow('Improved Object Detection', vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        stats = detector.get_performance_stats()
        print(f"\n📊 Final Performance Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    else:
        print("✅ Improved Object Detector initialized successfully")
        print(f"Model: {detector.model_name}")
        print(f"Device: {detector.device}")
        print(f"Building classes: {len(detector.building_classes)}")
        print("Use --image, --webcam, or --benchmark for testing")

if __name__ == "__main__":
    main()