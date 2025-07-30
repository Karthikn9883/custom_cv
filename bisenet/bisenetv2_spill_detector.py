#!/usr/bin/env python3
"""
BiSeNet V2 Spill Detector - SCOPE Smart Building System
Real-time spill segmentation using BiSeNet V2 bilateral segmentation network
Optimized for precise pixel-level spill boundary detection with CUDA acceleration
"""

import cv2
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import yaml

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("Error: albumentations not installed. Run: pip install albumentations")
    exit(1)

from .bisenetv2_model import create_bisenetv2
try:
    from cuda.cuda_memory_manager import get_memory_manager
except ImportError:
    # Fallback if CUDA memory manager not available
    class DummyMemoryManager:
        def get_tensor(self, *args, **kwargs):
            return None
        def return_tensor(self, *args, **kwargs):
            pass
    
    def get_memory_manager():
        return DummyMemoryManager()


class BiSeNetV2SpillDetector:
    """
    BiSeNet V2 based spill segmentation detector
    Provides real-time pixel-level spill detection with superior boundary accuracy
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None, 
                 config_path: Optional[str] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize BiSeNet V2 Spill Detector
        
        Args:
            model_path: Path to trained model weights
            config_path: Path to configuration file
            device: Device for inference
        """
        # Setup logging first
        self.logger = logging.getLogger(__name__)
        
        self.device = device or self._get_optimal_device()
        self.model = None
        self.transform = None
        self.config = self._load_config(config_path)
        
        # Memory manager for optimization
        self.memory_manager = get_memory_manager()
        
        # Performance tracking
        self.inference_times = []
        self.detection_count = 0
        
        # Initialize model
        self._initialize_model(model_path)
        self._setup_transforms()
        
        self.logger.info(f"BiSeNet V2 Spill Detector initialized on {self.device}")
    
    def _get_optimal_device(self) -> torch.device:
        """Get optimal device for inference"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration file"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded configuration from: {config_path}")
        else:
            # Default configuration
            config = {
                'model': {'num_classes': 2, 'aux_mode': 'eval'},
                'dataset': {'image_size': [512, 512]},
                'inference': {
                    'confidence_threshold': 0.5,
                    'min_region_area': 100,
                    'mixed_precision': True
                }
            }
            self.logger.info("Using default configuration")
        
        return config
    
    def _initialize_model(self, model_path: Optional[str] = None):
        """Initialize BiSeNet V2 model"""
        try:
            # Create model
            num_classes = self.config['model']['num_classes']
            aux_mode = self.config['model'].get('aux_mode', 'eval')
            
            self.model = create_bisenetv2(
                num_classes=num_classes,
                aux_mode=aux_mode,
                pretrained=False
            )
            
            # Load custom weights if provided
            if model_path and Path(model_path).exists():
                self.logger.info(f"Loading trained BiSeNet V2 model: {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                self.model.load_state_dict(state_dict, strict=False)
                self.logger.info("BiSeNet V2 model loaded successfully")
            else:
                self.logger.warning("No model weights provided - using random initialization")
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            # Optimize model
            self._optimize_model()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize BiSeNet V2 model: {e}")
            raise
    
    def _optimize_model(self):
        """Apply device-specific optimizations"""
        try:
            # Enable mixed precision if configured
            use_mixed_precision = self.config.get('inference', {}).get('mixed_precision', True)
            
            if use_mixed_precision and self.device.type in ['cuda', 'mps']:
                try:
                    self.model.half()
                    self.use_half_precision = True
                    self.logger.info("Half precision (FP16) enabled")
                except Exception as e:
                    self.logger.warning(f"Half precision failed: {e}")
                    self.use_half_precision = False
            else:
                self.use_half_precision = False
            
            # Torch compile optimization (PyTorch 2.0+) - Disabled for Windows compatibility
            if False and hasattr(torch, 'compile') and self.device.type == 'cuda':
                try:
                    self.model = torch.compile(self.model, mode='reduce-overhead')
                    self.logger.info("Model compiled with torch.compile")
                except Exception as e:
                    self.logger.warning(f"Torch compile failed: {e}")
            
            # Set to evaluation mode with optimizations
            self.model.eval()
            
            # Disable gradient computation
            for param in self.model.parameters():
                param.requires_grad = False
            
        except Exception as e:
            self.logger.warning(f"Some optimizations failed: {e}")
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        image_size = self.config['dataset']['image_size']
        
        # Standard preprocessing for inference
        self.transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]    # ImageNet stds
            ),
            ToTensorV2()
        ])
        
        # Fast transform with smaller resolution for speed testing
        self.fast_transform = A.Compose([
            A.Resize(384, 384),  # Smaller for speed
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def detect_spills(self, frame: np.ndarray, use_fast_mode: bool = False) -> Dict[str, Any]:
        """
        Detect spills in frame using BiSeNet V2 segmentation
        
        Args:
            frame: Input image (H, W, 3) in BGR format
            use_fast_mode: Use smaller input size for speed
            
        Returns:
            Dictionary with spill detection results
        """
        start_time = time.time()
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_shape = frame.shape[:2]
            
            # Apply transforms
            transform = self.fast_transform if use_fast_mode else self.transform
            transformed = transform(image=rgb_frame)
            input_tensor = transformed['image'].unsqueeze(0)
            
            # Move to device with proper dtype
            if self.use_half_precision:
                input_tensor = input_tensor.half().to(self.device)
            else:
                input_tensor = input_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
                
                # Handle different output formats
                if isinstance(output, tuple):
                    # Training mode output (main_pred, aux_preds)
                    logits = output[0]
                else:
                    # Evaluation mode output
                    logits = output
                
                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=1)
                
                # Get spill prediction (class 1)
                spill_prob = probs[0, 1].cpu().float().numpy()
                
                # Create binary mask with configurable threshold
                confidence_threshold = self.config.get('inference', {}).get('confidence_threshold', 0.5)
                spill_mask = (spill_prob > confidence_threshold).astype(np.uint8)
            
            # Resize mask to original frame size
            spill_mask_resized = cv2.resize(
                spill_mask, 
                (original_shape[1], original_shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Post-processing: remove small regions
            min_region_area = self.config.get('inference', {}).get('min_region_area', 100)
            spill_mask_resized = self._post_process_mask(spill_mask_resized, min_region_area)
            
            # Calculate spill metrics
            spill_pixels = np.sum(spill_mask_resized)
            total_pixels = original_shape[0] * original_shape[1]
            coverage_percent = (spill_pixels / total_pixels) * 100
            
            # Create colored overlay
            overlay = frame.copy()
            spill_indices = np.where(spill_mask_resized == 1)
            overlay[spill_indices] = [0, 0, 255]  # Red for spills
            
            # Find contours for boundary detection
            contours, _ = cv2.findContours(
                spill_mask_resized, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Calculate spill regions and bounding boxes
            spill_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_region_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate additional geometric properties
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    spill_regions.append({
                        'bbox': (x, y, x + w, y + h),
                        'area': area,
                        'perimeter': perimeter,
                        'circularity': circularity,
                        'contour': contour.tolist()
                    })
            
            # Sort regions by area (largest first)
            spill_regions.sort(key=lambda x: x['area'], reverse=True)
            
            # Performance tracking
            inference_time = (time.time() - start_time) * 1000
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            
            self.detection_count += 1
            
            # Determine if spill is detected
            spill_detected = spill_pixels > min_region_area
            
            return {
                'spill_detected': spill_detected,
                'mask': spill_mask_resized,
                'overlay': overlay,
                'coverage_percent': coverage_percent,
                'spill_pixels': int(spill_pixels),
                'spill_regions': spill_regions,
                'num_spills': len(spill_regions),
                'inference_time_ms': inference_time,
                'input_size': input_tensor.shape[-2:],
                'confidence_map': spill_prob if not use_fast_mode else None,
                'model_name': 'BiSeNet V2'
            }
            
        except Exception as e:
            self.logger.error(f"Spill detection failed: {e}")
            return {
                'spill_detected': False,
                'mask': np.zeros(frame.shape[:2], dtype=np.uint8),
                'overlay': frame,
                'coverage_percent': 0.0,
                'spill_pixels': 0,
                'spill_regions': [],
                'num_spills': 0,
                'inference_time_ms': (time.time() - start_time) * 1000,
                'error': str(e),
                'model_name': 'BiSeNet V2'
            }
    
    def _post_process_mask(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """
        Post-process segmentation mask to remove noise
        
        Args:
            mask: Binary mask
            min_area: Minimum area to keep
            
        Returns:
            Cleaned mask
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Create clean mask
        clean_mask = np.zeros_like(mask)
        
        # Keep components larger than minimum area
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean_mask[labels == i] = 1
        
        # Apply morphological operations to smooth boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
        
        return clean_mask
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_ms': np.mean(self.inference_times),
            'p95_inference_ms': np.percentile(self.inference_times, 95),
            'max_inference_ms': np.max(self.inference_times),
            'min_inference_ms': np.min(self.inference_times),
            'total_detections': self.detection_count,
            'device': str(self.device),
            'model_name': 'BiSeNet V2'
        }
    
    def warmup(self, num_iterations: int = 5):
        """Warmup the model with dummy inputs"""
        self.logger.info(f"Warming up BiSeNet V2 model ({num_iterations} iterations)...")
        
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        for i in range(num_iterations):
            _ = self.detect_spills(dummy_frame, use_fast_mode=(i < 2))
        
        self.logger.info("Warmup completed")
    
    def visualize_results(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Create comprehensive visualization of spill detection results
        
        Args:
            frame: Original frame
            results: Detection results from detect_spills()
            
        Returns:
            Annotated frame with spill visualization
        """
        vis_frame = frame.copy()
        
        if not results['spill_detected']:
            # Add "No Spills" text
            cv2.putText(vis_frame, "No Spills Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return vis_frame
        
        # Draw spill overlay with transparency
        overlay = results['overlay']
        alpha = 0.6
        vis_frame = cv2.addWeighted(vis_frame, 1 - alpha, overlay, alpha, 0)
        
        # Draw spill contours and information
        for i, spill_region in enumerate(results['spill_regions']):
            contour = np.array(spill_region['contour'], dtype=np.int32)
            cv2.drawContours(vis_frame, [contour], -1, (0, 0, 255), 3)
            
            # Draw bounding box
            x1, y1, x2, y2 = spill_region['bbox']
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Add region information
            info_texts = [
                f"Spill {i+1}",
                f"Area: {spill_region['area']:.0f}px",
                f"Circ: {spill_region['circularity']:.2f}"
            ]
            
            for j, text in enumerate(info_texts):
                y_offset = y1 - 10 - (j * 20)
                if y_offset > 0:
                    cv2.putText(vis_frame, text, (x1, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add summary information
        info_y = 30
        info_texts = [
            f"BiSeNet V2 - Spills: {results['num_spills']}",
            f"Coverage: {results['coverage_percent']:.1f}%",
            f"Inference: {results['inference_time_ms']:.1f}ms"
        ]
        
        for text in info_texts:
            cv2.putText(vis_frame, text, (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            info_y += 25
        
        return vis_frame
    
    def save_model(self, save_path: str):
        """Save the current model state"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'device': str(self.device),
                'model_type': 'bisenetv2_spill_detector'
            }, save_path)
            self.logger.info(f"Model saved to: {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")


def main():
    """Test the BiSeNet V2 spill detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BiSeNet V2 Spill Detector Test')
    parser.add_argument('--model', type=str, help='Path to trained model weights')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--webcam', action='store_true', help='Test with webcam')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--fast', action='store_true', help='Use fast mode (smaller input)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = BiSeNetV2SpillDetector(
        model_path=args.model,
        config_path=args.config
    )
    
    # Warmup
    detector.warmup()
    
    if args.benchmark:
        print("🚀 Running BiSeNet V2 performance benchmark...")
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Benchmark normal mode
        times_normal = []
        for i in range(50):
            start = time.time()
            _ = detector.detect_spills(test_image, use_fast_mode=False)
            times_normal.append((time.time() - start) * 1000)
        
        # Benchmark fast mode
        times_fast = []
        for i in range(50):
            start = time.time()
            _ = detector.detect_spills(test_image, use_fast_mode=True)
            times_fast.append((time.time() - start) * 1000)
        
        print(f"Normal Mode: {np.mean(times_normal):.1f}ms ± {np.std(times_normal):.1f}ms")
        print(f"Fast Mode: {np.mean(times_fast):.1f}ms ± {np.std(times_fast):.1f}ms")
        
        stats = detector.get_performance_stats()
        print(f"Device: {stats['device']}")
        return
    
    if args.image:
        # Test with single image
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image {args.image}")
            return
        
        print("🔍 Detecting spills with BiSeNet V2...")
        results = detector.detect_spills(image, use_fast_mode=args.fast)
        
        print(f"Spill detected: {results['spill_detected']}")
        print(f"Coverage: {results['coverage_percent']:.1f}%")
        print(f"Number of spills: {results['num_spills']}")
        print(f"Inference time: {results['inference_time_ms']:.1f}ms")
        
        # Show visualization
        vis_frame = detector.visualize_results(image, results)
        cv2.imshow('BiSeNet V2 Spill Detection Results', vis_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    elif args.webcam:
        # Test with webcam
        cap = cv2.VideoCapture(0)
        
        print("🎥 Starting BiSeNet V2 webcam spill detection (press 'q' to quit)...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect spills
            results = detector.detect_spills(frame, use_fast_mode=args.fast)
            
            # Visualize
            vis_frame = detector.visualize_results(frame, results)
            
            cv2.imshow('BiSeNet V2 Spill Detection', vis_frame)
            
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
        print("✅ BiSeNet V2 Spill Detector initialized successfully")
        print(f"Device: {detector.device}")
        print("Use --image, --webcam, or --benchmark for testing")


if __name__ == "__main__":
    main()