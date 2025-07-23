#!/usr/bin/env python3
"""
CUDA Stage 2 Verifier - SCOPE Smart Building System
Heavy verification model using RT-DETR-M for high-accuracy confirmation
Called on-demand for Stage 1 tokens, implements confidence fusion & temporal filtering
"""

import cv2
import time
import json
import yaml
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

try:
    from ultralytics import RTDETR
except ImportError:
    try:
        from ultralytics import YOLO as RTDETR
        print("Warning: Using YOLO as fallback for RT-DETR")
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        exit(1)

from cuda_memory_manager import get_memory_manager
from cuda_stage1_detector import DetectionToken

@dataclass
class VerificationResult:
    """Result of Stage 2 verification"""
    token_uuid: str
    stage1_confidence: float
    stage2_confidence: float
    fused_confidence: float
    verified: bool
    temporal_count: int
    roi_bbox: Tuple[int, int, int, int]
    verification_time_ms: float
    category: str

@dataclass
class ValidatedEvent:
    """Final validated detection event"""
    uuid: str
    timestamp: float
    category: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    camera_id: str
    frame_id: int
    verification_result: VerificationResult

class CUDAStage2Verifier:
    """
    Stage 2 heavy verifier using RT-DETR-M
    Implements confidence fusion and temporal filtering
    Target: <500ms latency, 99%+ accuracy
    """
    
    def __init__(self, config_path: str = "configs/cuda_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Models
        self.verifier_model = None
        
        # Memory manager
        self.memory_manager = get_memory_manager()
        self.device = self.memory_manager.device
        
        # Temporal tracking
        self.detection_history = defaultdict(deque)  # Track detections over time
        self.spatial_tracker = {}  # Track spatial consistency
        
        # Performance tracking
        self.verification_count = 0
        self.verification_times = []
        self.validation_success_rate = 0.0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_verifier()
        self._setup_optimization()
        
        self.logger.info(f"CUDA Stage 2 Verifier initialized on {self.device}")
    
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
            'stage2': {
                'model': 'rtdetr-l.pt',  # RT-DETR Large for better accuracy
                'precision': 'fp16',
                'confidence_threshold': 0.65,
                'iou_threshold': 0.5,
                'temporal_frames': 3,
                'roi_expansion': 1.5,  # Expand ROI by 50%
                'max_verification_time_ms': 500,
                'confidence_fusion': {
                    'stage1_weight': 0.3,
                    'stage2_weight': 0.7,
                    'minimum_fused': 0.6
                },
                'spatial_consistency': {
                    'max_displacement': 50,  # pixels
                    'consistency_window': 5   # frames
                }
            }
        }
    
    def _initialize_verifier(self):
        """Initialize RT-DETR verifier model"""
        try:
            stage2_config = self.config['stage2']
            model_path = stage2_config['model']
            
            self.logger.info(f"Loading Stage 2 verifier: {model_path}")
            
            # Check for TensorRT engine first
            engine_path = model_path.replace('.pt', '.engine')
            if Path(engine_path).exists() and self.device.type == 'cuda':
                self.logger.info(f"Loading TensorRT engine: {engine_path}")
                self.verifier_model = RTDETR(engine_path)
            else:
                self.verifier_model = RTDETR(model_path)
            
            # Optimize model
            self._optimize_verifier(stage2_config)
            
            # Warm up model
            self._warmup_verifier()
            
            self.logger.info("Stage 2 verifier loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load Stage 2 verifier: {e}")
            raise
    
    def _optimize_verifier(self, config: Dict[str, Any]):
        """Apply CUDA optimizations to verifier model"""
        try:
            # Move to device
            if hasattr(self.verifier_model.model, 'to'):
                self.verifier_model.model.to(self.device)
            
            # Enable half precision for CUDA/MPS
            if (self.device.type in ['cuda', 'mps'] and 
                config.get('precision') == 'fp16'):
                if hasattr(self.verifier_model.model, 'half'):
                    self.verifier_model.model.half()
                    self.logger.info("Half precision enabled for Stage 2")
            
            # Set to evaluation mode
            if hasattr(self.verifier_model.model, 'eval'):
                self.verifier_model.model.eval()
            
            # CUDA-specific optimizations
            if self.device.type == 'cuda':
                # Torch compile for RT-DETR (if supported)
                if (hasattr(torch, 'compile') and 
                    hasattr(self.verifier_model.model, 'forward')):
                    try:
                        self.verifier_model.model = torch.compile(
                            self.verifier_model.model, mode='reduce-overhead'
                        )
                        self.logger.info("Stage 2 model compiled with torch.compile")
                    except Exception as e:
                        self.logger.warning(f"Stage 2 torch.compile failed: {e}")
            
        except Exception as e:
            self.logger.warning(f"Some Stage 2 optimizations failed: {e}")
    
    def _setup_optimization(self):
        """Setup verification optimizations"""
        # Pre-allocate ROI processing tensors
        common_roi_shapes = [
            (1, 3, 224, 224),   # Small ROI
            (1, 3, 320, 320),   # Medium ROI  
            (1, 3, 448, 448),   # Large ROI
        ]
        
        dtype = torch.float16 if self.device.type in ['cuda', 'mps'] else torch.float32
        self.roi_tensors = self.memory_manager.batch_allocate(
            [(shape, dtype) for shape in common_roi_shapes]
        )
        
        self.logger.info("Stage 2 optimization setup completed")
    
    def _warmup_verifier(self):
        """Warm up verifier model"""
        self.logger.info("Warming up Stage 2 verifier...")
        
        # Create dummy ROI
        dummy_roi = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        
        try:
            with self.memory_manager.cuda_stream('stage2_verification'):
                _ = self.verifier_model(dummy_roi, verbose=False, conf=0.5)
            self.logger.info("Stage 2 verifier warmed up")
        except Exception as e:
            self.logger.warning(f"Stage 2 warmup failed: {e}")
    
    def extract_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                   expansion_factor: float = 1.5) -> np.ndarray:
        """
        Extract and expand ROI from frame
        
        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            expansion_factor: ROI expansion factor
            
        Returns:
            Extracted ROI
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Calculate expanded ROI
        roi_w = x2 - x1
        roi_h = y2 - y1
        
        expand_w = int(roi_w * (expansion_factor - 1) / 2)
        expand_h = int(roi_h * (expansion_factor - 1) / 2)
        
        # Expanded coordinates with bounds checking
        x1_exp = max(0, x1 - expand_w)
        y1_exp = max(0, y1 - expand_h)
        x2_exp = min(w, x2 + expand_w)
        y2_exp = min(h, y2 + expand_h)
        
        # Extract ROI
        roi = frame[y1_exp:y2_exp, x1_exp:x2_exp]
        
        return roi
    
    def verify_token(self, token: DetectionToken, frame: np.ndarray) -> VerificationResult:
        """
        Verify a single Stage 1 token using Stage 2 model
        
        Args:
            token: Detection token from Stage 1
            frame: Current frame
            
        Returns:
            Verification result
        """
        start_time = time.time()
        stage2_config = self.config['stage2']
        
        try:
            # Extract ROI
            roi = self.extract_roi(
                frame, token.bbox, 
                stage2_config['roi_expansion']
            )
            
            # Run Stage 2 inference
            with self.memory_manager.cuda_stream('stage2_verification'):
                results = self.verifier_model(
                    roi,
                    conf=stage2_config['confidence_threshold'],
                    iou=stage2_config['iou_threshold'],
                    verbose=False,
                    device=self.device
                )
            
            # Synchronize if using CUDA
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Process results
            stage2_confidence = 0.0
            verified = False
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    # Find best matching detection
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    # Get the highest confidence detection
                    best_idx = np.argmax(confidences)
                    stage2_confidence = float(confidences[best_idx])
                    
                    # Check if category matches or is compatible
                    class_names = getattr(result, 'names', {})
                    detected_class = class_names.get(int(classes[best_idx]), '')
                    
                    # Category compatibility check
                    if self._categories_compatible(token.category, detected_class):
                        verified = stage2_confidence >= stage2_config['confidence_threshold']
            
            # Confidence fusion
            fusion_config = stage2_config['confidence_fusion']
            fused_confidence = (
                token.confidence * fusion_config['stage1_weight'] +
                stage2_confidence * fusion_config['stage2_weight']
            )
            
            # Final verification decision
            final_verified = (
                verified and 
                fused_confidence >= fusion_config['minimum_fused']
            )
            
            verification_time = (time.time() - start_time) * 1000  # ms
            
            # Create verification result
            result = VerificationResult(
                token_uuid=token.uuid,
                stage1_confidence=token.confidence,
                stage2_confidence=stage2_confidence,
                fused_confidence=fused_confidence,
                verified=final_verified,
                temporal_count=1,  # Will be updated by temporal filtering
                roi_bbox=token.bbox,
                verification_time_ms=verification_time,
                category=token.category
            )
            
            # Update performance tracking
            self.verification_times.append(verification_time)
            if len(self.verification_times) > 100:
                self.verification_times.pop(0)
            
            self.verification_count += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Token verification failed: {e}")
            
            # Return failed verification
            return VerificationResult(
                token_uuid=token.uuid,
                stage1_confidence=token.confidence,
                stage2_confidence=0.0,
                fused_confidence=token.confidence * 0.3,  # Heavily penalize
                verified=False,
                temporal_count=0,
                roi_bbox=token.bbox,
                verification_time_ms=(time.time() - start_time) * 1000,
                category=token.category
            )
    
    def _categories_compatible(self, stage1_category: str, stage2_category: str) -> bool:
        """Check if categories from Stage 1 and Stage 2 are compatible"""
        # Exact match
        if stage1_category.lower() == stage2_category.lower():
            return True
        
        # Category groups for compatibility
        category_groups = {
            'valuables': ['laptop', 'phone', 'wallet', 'bag', 'backpack', 'suitcase'],
            'waste': ['bin', 'trash', 'recycling', 'garbage'],
            'fire_safety': ['fire', 'extinguisher', 'exit', 'hydrant', 'alarm'],
            'spills': ['spill', 'liquid', 'puddle', 'wet', 'water'],
            'safety': ['cone', 'sign', 'barrier', 'caution']
        }
        
        # Find groups for both categories
        stage1_groups = []
        stage2_groups = []
        
        for group, keywords in category_groups.items():
            if any(keyword in stage1_category.lower() for keyword in keywords):
                stage1_groups.append(group)
            if any(keyword in stage2_category.lower() for keyword in keywords):
                stage2_groups.append(group)
        
        # Check for group overlap
        return bool(set(stage1_groups) & set(stage2_groups))
    
    def apply_temporal_filtering(self, verification_results: List[VerificationResult], 
                                camera_id: str) -> List[VerificationResult]:
        """
        Apply temporal filtering to verification results
        
        Args:
            verification_results: List of verification results
            camera_id: Camera identifier
            
        Returns:
            Temporally filtered results
        """
        stage2_config = self.config['stage2']
        temporal_frames = stage2_config['temporal_frames']
        
        filtered_results = []
        current_time = time.time()
        
        for result in verification_results:
            if not result.verified:
                continue
            
            # Create detection key for temporal tracking
            detection_key = f"{camera_id}_{result.category}"
            
            # Add to history
            self.detection_history[detection_key].append({
                'result': result,
                'timestamp': current_time,
                'bbox': result.roi_bbox
            })
            
            # Keep only recent detections
            while (self.detection_history[detection_key] and 
                   current_time - self.detection_history[detection_key][0]['timestamp'] > 5.0):
                self.detection_history[detection_key].popleft()
            
            # Check temporal consistency
            recent_detections = list(self.detection_history[detection_key])
            
            if len(recent_detections) >= temporal_frames:
                # Check spatial consistency
                if self._check_spatial_consistency(recent_detections, stage2_config):
                    # Update temporal count
                    result.temporal_count = len(recent_detections)
                    filtered_results.append(result)
        
        return filtered_results
    
    def _check_spatial_consistency(self, detections: List[Dict], config: Dict) -> bool:
        """Check spatial consistency across temporal detections"""
        if len(detections) < 2:
            return True
        
        max_displacement = config['spatial_consistency']['max_displacement']
        
        # Calculate centroids
        centroids = []
        for det in detections:
            bbox = det['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            centroids.append((cx, cy))
        
        # Check if all centroids are within max displacement
        base_centroid = centroids[0]
        for centroid in centroids[1:]:
            displacement = np.sqrt(
                (centroid[0] - base_centroid[0])**2 + 
                (centroid[1] - base_centroid[1])**2
            )
            
            if displacement > max_displacement:
                return False
        
        return True
    
    def create_validated_event(self, token: DetectionToken, 
                             verification_result: VerificationResult) -> ValidatedEvent:
        """
        Create final validated event from token and verification
        
        Args:
            token: Original detection token
            verification_result: Verification result
            
        Returns:
            Validated event for MQTT publishing
        """
        return ValidatedEvent(
            uuid=token.uuid,
            timestamp=token.timestamp,
            category=token.category,
            confidence=verification_result.fused_confidence,
            bbox=token.bbox,
            camera_id=token.camera_id,
            frame_id=token.frame_id,
            verification_result=verification_result
        )
    
    def batch_verify_tokens(self, tokens: List[DetectionToken], 
                           frame: np.ndarray, camera_id: str) -> List[ValidatedEvent]:
        """
        Batch verify multiple tokens and apply temporal filtering
        
        Args:
            tokens: List of detection tokens
            frame: Current frame
            camera_id: Camera identifier
            
        Returns:
            List of validated events
        """
        if not tokens:
            return []
        
        # Verify each token
        verification_results = []
        for token in tokens:
            result = self.verify_token(token, frame)
            verification_results.append(result)
        
        # Apply temporal filtering
        filtered_results = self.apply_temporal_filtering(verification_results, camera_id)
        
        # Create validated events
        validated_events = []
        for token, result in zip(tokens, verification_results):
            if result in filtered_results:
                event = self.create_validated_event(token, result)
                validated_events.append(event)
        
        # Update success rate
        if verification_results:
            success_count = len(filtered_results)
            self.validation_success_rate = (
                self.validation_success_rate * 0.9 + 
                (success_count / len(verification_results)) * 0.1
            )
        
        return validated_events
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get Stage 2 performance statistics"""
        stats = {
            'verification_count': self.verification_count,
            'validation_success_rate': self.validation_success_rate,
            'device': str(self.device)
        }
        
        if self.verification_times:
            stats.update({
                'avg_verification_time_ms': np.mean(self.verification_times),
                'p95_verification_time_ms': np.percentile(self.verification_times, 95),
                'max_verification_time_ms': np.max(self.verification_times),
                'target_time_ms': self.config['stage2']['max_verification_time_ms']
            })
        
        # Add memory stats
        memory_stats = self.memory_manager.get_memory_stats()
        stats.update({f"memory_{k}": v for k, v in memory_stats.items()})
        
        return stats
    
    def convert_to_tensorrt(self) -> bool:
        """Convert RT-DETR model to TensorRT engine"""
        if self.device.type != 'cuda':
            self.logger.warning("TensorRT conversion requires CUDA device")
            return False
        
        try:
            self.logger.info("Converting Stage 2 model to TensorRT...")
            
            # Export to TensorRT
            self.verifier_model.export(
                format='engine',
                half=True,
                imgsz=640,  # Standard size for RT-DETR
                device=self.device
            )
            
            model_path = self.config['stage2']['model']
            engine_path = model_path.replace('.pt', '.engine')
            self.logger.info(f"Stage 2 TensorRT engine saved: {engine_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Stage 2 TensorRT conversion failed: {e}")
            return False

def main():
    """Test the CUDA Stage 2 verifier"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CUDA Stage 2 Verifier Test')
    parser.add_argument('--config', type=str, default='configs/cuda_config.yaml')
    parser.add_argument('--convert-tensorrt', action='store_true')
    parser.add_argument('--test', action='store_true')
    
    args = parser.parse_args()
    
    # Initialize verifier
    verifier = CUDAStage2Verifier(args.config)
    
    if args.convert_tensorrt:
        success = verifier.convert_to_tensorrt()
        print(f"TensorRT conversion {'succeeded' if success else 'failed'}")
        return
    
    if args.test:
        stats = verifier.get_performance_stats()
        print("✅ CUDA Stage 2 Verifier initialized successfully")
        print(f"Device: {verifier.device}")
        print(f"Target verification time: {verifier.config['stage2']['max_verification_time_ms']}ms")
        return
    
    # Test with dummy data
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_token = DetectionToken(
        uuid="test-uuid",
        timestamp=time.time(),
        category="laptop",
        confidence=0.8,
        bbox=(100, 100, 200, 200),
        camera_id="cam_01",
        frame_id=1,
        detector_type="object"
    )
    
    # Performance test
    num_tests = 50
    start_time = time.time()
    
    for i in range(num_tests):
        result = verifier.verify_token(dummy_token, dummy_frame)
    
    total_time = time.time() - start_time
    avg_time = (total_time / num_tests) * 1000  # ms
    
    stats = verifier.get_performance_stats()
    print(f"\n🚀 Stage 2 Performance Test Results:")
    print(f"Average verification time: {avg_time:.2f}ms")
    print(f"Target time: {verifier.config['stage2']['max_verification_time_ms']}ms")
    print(f"Status: {'✅ PASS' if avg_time < verifier.config['stage2']['max_verification_time_ms'] else '❌ FAIL'}")

if __name__ == "__main__":
    main()