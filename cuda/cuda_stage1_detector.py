#!/usr/bin/env python3
"""
CUDA Stage 1 Detector - SCOPE Smart Building System
Ultra-fast token generation using YOLO-World + lightweight spill detection
Optimized for <20ms latency with CUDA/TensorRT support
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time
import uuid
import json
import yaml
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics")
    exit(1)

from cuda_memory_manager import get_memory_manager
from bisenet.bisenetv2_spill_detector import BiSeNetV2SpillDetector
from yolo.improved_object_detector import ImprovedObjectDetector

@dataclass
class DetectionToken:
    """Lightweight token for Stage 1 detections"""
    uuid: str
    timestamp: float
    category: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    camera_id: str
    frame_id: int
    detector_type: str  # 'object' or 'spill'

class CUDAStage1Detector:
    """
    Ultra-fast Stage 1 detector optimized for CUDA
    Combines object detection + spill detection in single pipeline
    Target: <20ms total latency
    """
    
    def __init__(self, config_path: str = "configs/cuda_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Models
        self.object_detector = None
        self.spill_detector = None
        
        # Memory manager
        self.memory_manager = get_memory_manager()
        self.device = self.memory_manager.device
        
        # Performance tracking
        self.frame_count = 0
        self.inference_times = []
        self.start_time = time.time()
        self.warmup_done = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_models()
        self._setup_optimization()
        
        self.logger.info(f"CUDA Stage 1 Detector initialized on {self.device}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load CUDA configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default CUDA configuration"""
        return {
            'device': {
                'auto_detect': True,
                'preferred': 'cuda',
                'cuda': {
                    'memory_fraction': 0.8,
                    'enable_tensorrt': True,
                    'precision': 'fp16'
                }
            },
            'stage1': {
                'models': {
                    'object_detector': 'yolov8n-world.pt',
                    'spill_detector': 'yolov8n-seg.pt'
                },
                'quantization': 'int8',
                'target_latency_ms': 20,
                'confidence_threshold': 0.35,
                'iou_threshold': 0.5,
                'max_detections': 30,
                'image_size': 640,  # Higher resolution for accuracy
                'detection_prompts': [
                    'backpack', 'suitcase', 'laptop', 'phone', 'wallet',
                    'trash bin', 'recycling bin', 'litter on floor',
                    'fire extinguisher', 'fire exit', 'fire hydrant',
                    'wet floor sign', 'safety cone', 'spill on floor'
                ]
            }
        }
    
    def _initialize_models(self):
        """Initialize CUDA-optimized models"""
        stage1_config = self.config['stage1']
        
        # Initialize object detector (YOLO-World)
        self._load_object_detector(stage1_config)
        
        # Initialize spill detector (YOLOv8-seg)
        self._load_spill_detector(stage1_config)
        
        # Warm up models
        self.memory_manager.warm_up_device()
        self._warmup_models()
    
    def _load_object_detector(self, config: Dict[str, Any]):
        """Load and optimize object detection model"""
        try:
            model_path = config['models']['object_detector']
            self.logger.info(f"Loading object detector: {model_path}")
            
            # Check for TensorRT engine first
            engine_path = model_path.replace('.pt', '.engine')
            if Path(engine_path).exists() and self.device.type == 'cuda':
                self.logger.info(f"Loading TensorRT engine: {engine_path}")
                self.object_detector = YOLO(engine_path)
            else:
                self.object_detector = YOLO(model_path)
            
            # Set detection prompts
            if hasattr(self.object_detector, 'set_classes'):
                prompts = [p for p in config['detection_prompts'] if 'spill' not in p.lower()]
                self.object_detector.set_classes(prompts)
            
            # Optimize for device
            self._optimize_model(self.object_detector, config)
            
            self.logger.info("Object detector loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load object detector: {e}")
            raise
    
    def _load_spill_detector(self, config: Dict[str, Any]):
        """Load and optimize BiSeNet V2 spill detection model"""
        try:
            model_path = config['models'].get('spill_detector')
            if model_path and Path(model_path).exists():
                self.logger.info(f"Loading BiSeNet V2 spill detector with custom weights: {model_path}")
            else:
                self.logger.info("Loading BiSeNet V2 spill detector with pre-trained weights")
                model_path = None
            
            # Initialize BiSeNet V2 SpillDetector
            self.spill_detector = BiSeNetV2SpillDetector(
                model_path=model_path,
                device=self.device
            )
            
            # Warmup the spill detector
            self.spill_detector.warmup()
            
            self.logger.info("BiSeNet V2 spill detector loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to load BiSeNet V2 spill detector: {e}")
            self.spill_detector = None
    
    def _optimize_model(self, model: YOLO, config: Dict[str, Any]):
        """Apply CUDA optimizations to model"""
        try:
            # Move to device
            if hasattr(model.model, 'to'):
                model.model.to(self.device)
            
            # Enable half precision for CUDA/MPS
            if (self.device.type in ['cuda', 'mps'] and 
                config.get('quantization') in ['fp16', 'half']):
                if hasattr(model.model, 'half'):
                    model.model.half()
                    self.logger.info("Half precision enabled")
            
            # Set to evaluation mode
            if hasattr(model.model, 'eval'):
                model.model.eval()
            
            # CUDA-specific optimizations
            if self.device.type == 'cuda':
                # Enable TensorRT if available
                if config.get('quantization') == 'int8':
                    self.logger.info("INT8 quantization requested (requires TensorRT)")
                
                # Torch compile (if supported)
                if (hasattr(torch, 'compile') and 
                    hasattr(model.model, 'forward')):
                    try:
                        model.model = torch.compile(model.model, mode='reduce-overhead')
                        self.logger.info("Model compiled with torch.compile")
                    except Exception as e:
                        self.logger.warning(f"torch.compile failed: {e}")
            
        except Exception as e:
            self.logger.warning(f"Some model optimizations failed: {e}")
    
    def _setup_optimization(self):
        """Setup inference optimizations"""
        # Optimize memory manager for inference
        self.memory_manager.optimize_for_inference()
        
        # Pre-allocate common tensors for new model sizes
        common_shapes = [
            (1, 3, 640, 640),   # Object detection input (YOLOv8m)
            (1, 3, 512, 512),   # Spill segmentation input (BiSeNet V2)
            (50, 4),            # Bounding boxes (more detections with YOLOv8m)
            (50,),              # Confidences
            (50,),              # Classes
            (512, 512),         # Segmentation masks
        ]
        
        dtype = torch.float16 if self.device.type in ['cuda', 'mps'] else torch.float32
        self.tensor_cache = self.memory_manager.batch_allocate(
            [(shape, dtype) for shape in common_shapes]
        )
        
        self.logger.info("Optimization setup completed")
    
    def _warmup_models(self):
        """Warm up models with dummy inputs"""
        self.logger.info("Warming up models...")
        
        # Create dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Warmup object detector
        if self.object_detector:
            try:
                with self.memory_manager.cuda_stream('stage1_detection'):
                    _ = self.object_detector(dummy_frame, verbose=False, conf=0.5, imgsz=320)
                self.logger.info("Object detector warmed up")
            except Exception as e:
                self.logger.warning(f"Object detector warmup failed: {e}")
        
        # BiSeNet V2 spill detector is already warmed up in _load_spill_detector
        if self.spill_detector:
            self.logger.info("BiSeNet V2 spill detector already warmed up")
        
        self.warmup_done = True
        self.logger.info("Model warmup completed")
    
    def detect_frame(self, frame: np.ndarray, camera_id: str = "cam_01") -> List[DetectionToken]:
        """
        Ultra-fast Stage 1 detection on single frame
        
        Args:
            frame: Input image
            camera_id: Camera identifier
            
        Returns:
            List of detection tokens
        """
        start_time = time.time()
        tokens = []
        
        # Memory management
        if self.frame_count % 50 == 0:
            self.memory_manager.clear_cache()
        
        config = self.config['stage1']
        
        # Parallel detection using CUDA streams
        object_results = None
        spill_results = None
        
        try:
            # Object detection in parallel stream
            if self.object_detector:
                with self.memory_manager.cuda_stream('stage1_detection'):
                    object_results = self.object_detector(
                        frame,
                        conf=config['confidence_threshold'],
                        iou=config['iou_threshold'],
                        imgsz=config['image_size'],
                        max_det=config['max_detections'],
                        verbose=False,
                        device=self.device
                    )
            
            # Spill detection in parallel stream
            if self.spill_detector:
                with self.memory_manager.cuda_stream('preprocessing'):
                    spill_results = self.spill_detector(
                        frame,
                        conf=config['confidence_threshold'],
                        iou=config['iou_threshold'],
                        imgsz=config['image_size'],
                        max_det=10,  # Fewer spills expected
                        verbose=False,
                        device=self.device
                    )
            
            # Synchronize streams if using CUDA
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Process object detection results
            if object_results and len(object_results) > 0:
                tokens.extend(self._process_detection_results(
                    object_results[0], camera_id, 'object'
                ))
            
            # Process spill detection results
            if spill_results and len(spill_results) > 0:
                tokens.extend(self._process_detection_results(
                    spill_results[0], camera_id, 'spill'
                ))
        
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
        
        # Performance tracking
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        
        # Keep rolling window of performance data
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
        
        self.frame_count += 1
        
        return tokens
    
    def _process_detection_results(self, result, camera_id: str, detector_type: str) -> List[DetectionToken]:
        """Process YOLO detection results into tokens"""
        tokens = []
        
        if result.boxes is None:
            return tokens
        
        try:
            # Batch process all detections
            boxes_data = result.boxes
            num_boxes = len(boxes_data)
            
            if num_boxes == 0:
                return tokens
            
            # Efficient batch conversion to CPU
            bboxes = boxes_data.xyxy.cpu().numpy().astype(int)
            confidences = boxes_data.conf.cpu().numpy()
            classes = boxes_data.cls.cpu().numpy().astype(int)
            
            # Get class names
            names = getattr(result, 'names', {}) or getattr(self.object_detector, 'names', {})
            
            # Create tokens for each detection
            current_time = time.time()
            for i in range(num_boxes):
                bbox = tuple(bboxes[i])
                confidence = float(confidences[i])
                class_id = int(classes[i])
                
                # Get category name
                category = names.get(class_id, f"class_{class_id}")
                
                # Create detection token
                token = DetectionToken(
                    uuid=str(uuid.uuid4()),
                    timestamp=current_time,
                    category=category,
                    confidence=confidence,
                    bbox=bbox,
                    camera_id=camera_id,
                    frame_id=self.frame_count,
                    detector_type=detector_type
                )
                tokens.append(token)
        
        except Exception as e:
            self.logger.error(f"Failed to process {detector_type} results: {e}")
        
        return tokens
    
    def filter_tokens_for_stage2(self, tokens: List[DetectionToken]) -> List[DetectionToken]:
        """
        Filter tokens that should be sent to Stage 2 verification
        
        Args:
            tokens: List of detection tokens
            
        Returns:
            Filtered tokens for Stage 2
        """
        stage2_tokens = []
        min_confidence = self.config['stage1']['confidence_threshold']
        
        for token in tokens:
            # Send to Stage 2 if confidence is above threshold
            if token.confidence >= min_confidence:
                # High-value items always go to Stage 2
                if any(keyword in token.category.lower() for keyword in 
                       ['laptop', 'phone', 'wallet', 'bag', 'fire', 'spill']):
                    stage2_tokens.append(token)
                
                # Other items go to Stage 2 if confidence is high enough
                elif token.confidence >= min_confidence + 0.1:
                    stage2_tokens.append(token)
        
        return stage2_tokens
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        if not self.inference_times:
            return {}
        
        avg_latency = np.mean(self.inference_times)
        p95_latency = np.percentile(self.inference_times, 95)
        max_latency = np.max(self.inference_times)
        current_fps = self.frame_count / (time.time() - self.start_time) if self.frame_count > 0 else 0
        
        stats = {
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
            'max_latency_ms': max_latency,
            'current_fps': current_fps,
            'total_frames': self.frame_count,
            'target_latency_ms': self.config['stage1']['target_latency_ms'],
            'device': str(self.device)
        }
        
        # Add memory stats
        memory_stats = self.memory_manager.get_memory_stats()
        stats.update({f"memory_{k}": v for k, v in memory_stats.items()})
        
        return stats
    
    def draw_detections(self, frame: np.ndarray, tokens: List[DetectionToken]) -> np.ndarray:
        """Draw detection tokens on frame for visualization"""
        annotated_frame = frame.copy()
        
        for token in tokens:
            x1, y1, x2, y2 = token.bbox
            
            # Color coding by detector type and category
            if token.detector_type == 'spill':
                color = (0, 100, 255)  # Orange for spills
            elif 'fire' in token.category.lower():
                color = (0, 0, 255)    # Red for fire safety
            elif any(val in token.category.lower() for val in ['bag', 'laptop', 'phone', 'wallet']):
                color = (255, 0, 0)    # Blue for valuables
            elif 'bin' in token.category.lower() or 'trash' in token.category.lower():
                color = (0, 255, 0)    # Green for waste
            else:
                color = (0, 255, 255)  # Yellow for other
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence and detector type
            label = f"{token.category}: {token.confidence:.2f} ({token.detector_type})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for text
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame
    
    def convert_to_tensorrt(self, model_name: str = 'both') -> bool:
        """
        Convert models to TensorRT engines for maximum performance
        
        Args:
            model_name: 'object', 'spill', or 'both'
            
        Returns:
            Success status
        """
        if self.device.type != 'cuda':
            self.logger.warning("TensorRT conversion requires CUDA device")
            return False
        
        success = True
        
        try:
            if model_name in ['object', 'both'] and self.object_detector:
                self.logger.info("Converting object detector to TensorRT...")
                model_path = self.config['stage1']['models']['object_detector']
                engine_path = model_path.replace('.pt', '.engine')
                
                # Export to TensorRT
                self.object_detector.export(
                    format='engine',
                    half=True,
                    int8=self.config['stage1']['quantization'] == 'int8',
                    imgsz=self.config['stage1']['image_size'],
                    device=self.device
                )
                
                self.logger.info(f"Object detector TensorRT engine saved: {engine_path}")
            
            if model_name in ['spill', 'both'] and self.spill_detector:
                self.logger.info("Converting spill detector to TensorRT...")
                model_path = self.config['stage1']['models']['spill_detector']
                engine_path = model_path.replace('.pt', '.engine')
                
                # Export to TensorRT
                self.spill_detector.export(
                    format='engine',
                    half=True,
                    int8=self.config['stage1']['quantization'] == 'int8',
                    imgsz=self.config['stage1']['image_size'],
                    device=self.device
                )
                
                self.logger.info(f"Spill detector TensorRT engine saved: {engine_path}")
        
        except Exception as e:
            self.logger.error(f"TensorRT conversion failed: {e}")
            success = False
        
        return success

def main():
    """Test the CUDA Stage 1 detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CUDA Stage 1 Detector Test')
    parser.add_argument('--config', type=str, default='configs/cuda_config.yaml')
    parser.add_argument('--convert-tensorrt', action='store_true', 
                       help='Convert models to TensorRT engines')
    parser.add_argument('--test', action='store_true', help='Test model loading only')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = CUDAStage1Detector(args.config)
    
    if args.convert_tensorrt:
        success = detector.convert_to_tensorrt('both')
        print(f"TensorRT conversion {'succeeded' if success else 'failed'}")
        return
    
    if args.test:
        stats = detector.get_performance_stats()
        print("✅ CUDA Stage 1 Detector initialized successfully")
        print(f"Device: {detector.device}")
        print(f"Target latency: {detector.config['stage1']['target_latency_ms']}ms")
        return
    
    # Test with dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Warm up
    for i in range(5):
        _ = detector.detect_frame(dummy_frame)
    
    # Performance test
    num_tests = 100
    start_time = time.time()
    
    for i in range(num_tests):
        tokens = detector.detect_frame(dummy_frame)
    
    total_time = time.time() - start_time
    avg_time = (total_time / num_tests) * 1000  # ms
    
    stats = detector.get_performance_stats()
    print(f"\n🚀 Performance Test Results:")
    print(f"Average latency: {avg_time:.2f}ms")
    print(f"Target latency: {stats['target_latency_ms']}ms")
    print(f"Status: {'✅ PASS' if avg_time < stats['target_latency_ms'] else '❌ FAIL'}")
    print(f"FPS: {1000/avg_time:.1f}")

if __name__ == "__main__":
    main()