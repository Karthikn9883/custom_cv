#!/usr/bin/env python3
"""
DeepLabV3+ Spill Detector - SCOPE Smart Building System
High-accuracy spill segmentation using DeepLabV3+ with MobileNetV3 backbone
Optimized for precise pixel-level spill boundary detection
"""

import cv2
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

try:
    import torchvision.transforms as transforms
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
    from torchvision.models.segmentation.deeplabv3 import DeepLabHead
except ImportError:
    print("Error: torchvision not installed. Run: pip install torchvision")
    exit(1)

from cuda_memory_manager import get_memory_manager

class DeepLabV3SpillDetector:
    """
    DeepLabV3+ based spill segmentation detector
    Provides pixel-level spill detection with precise boundaries
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[torch.device] = None):
        self.device = device or self._get_optimal_device()
        self.model = None
        self.transform = None
        self.memory_manager = get_memory_manager()
        
        # Performance tracking
        self.inference_times = []
        self.detection_count = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self._initialize_model(model_path)
        self._setup_transforms()
        
        self.logger.info(f"DeepLabV3+ Spill Detector initialized on {self.device}")
    
    def _get_optimal_device(self) -> torch.device:
        """Get optimal device for inference"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _initialize_model(self, model_path: Optional[str] = None):
        """Initialize DeepLabV3+ model"""
        try:
            # Create DeepLabV3+ with MobileNetV3 backbone
            self.model = deeplabv3_mobilenet_v3_large(pretrained=True)
            
            # Modify classifier for spill detection (background + spill = 2 classes)
            self.model.classifier = DeepLabHead(960, 2)  # MobileNetV3 feature channels = 960
            
            # Load custom weights if provided
            if model_path and Path(model_path).exists():
                self.logger.info(f"Loading custom spill model: {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                self.model.load_state_dict(state_dict, strict=False)
                self.logger.info("Custom spill model loaded successfully")
            else:
                self.logger.info("Using pre-trained DeepLabV3+ (will need fine-tuning for spills)")
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            # Optimize model
            self._optimize_model()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DeepLabV3+ model: {e}")
            raise
    
    def _optimize_model(self):
        """Apply device-specific optimizations"""
        try:
            # Enable half precision for CUDA/MPS
            if self.device.type in ['cuda', 'mps']:
                try:
                    self.model.half()
                    self.logger.info("Half precision (FP16) enabled")
                except Exception as e:
                    self.logger.warning(f"Half precision failed: {e}")
            
            # Torch compile optimization (skip on MPS due to compatibility issues)
            if (hasattr(torch, 'compile') and self.device.type == 'cuda'):
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
        # Standard ImageNet normalization for pre-trained models
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),  # Higher resolution for better segmentation
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]    # ImageNet stds
            )
        ])
        
        # Also create a simpler transform for speed testing
        self.fast_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((384, 384)),  # Smaller for speed
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def detect_spills(self, frame: np.ndarray, use_fast_mode: bool = False) -> Dict[str, Any]:
        """
        Detect spills in frame using segmentation
        
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
            input_tensor = transform(rgb_frame).unsqueeze(0)
            
            # Move to device with proper dtype
            if self.device.type in ['cuda', 'mps']:
                input_tensor = input_tensor.half().to(self.device)
            else:
                input_tensor = input_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
                
                # Get segmentation logits
                if isinstance(output, dict):
                    logits = output['out']
                else:
                    logits = output
                
                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=1)
                
                # Get spill prediction (class 1)
                spill_prob = probs[0, 1].cpu().numpy()
                
                # Create binary mask (threshold at 0.5)
                spill_mask = (spill_prob > 0.5).astype(np.uint8)
            
            # Resize mask to original frame size
            spill_mask_resized = cv2.resize(
                spill_mask, 
                (original_shape[1], original_shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
            
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
            
            # Calculate spill areas and bounding boxes
            spill_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)
                    spill_regions.append({
                        'bbox': (x, y, x + w, y + h),
                        'area': area,
                        'contour': contour.tolist()
                    })
            
            # Performance tracking
            inference_time = (time.time() - start_time) * 1000
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            
            self.detection_count += 1
            
            # Determine if spill is detected
            spill_detected = spill_pixels > 100  # Minimum pixel threshold
            
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
                'confidence_map': spill_prob if not use_fast_mode else None
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
                'error': str(e)
            }
    
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
            'device': str(self.device)
        }
    
    def warmup(self, num_iterations: int = 5):
        """Warmup the model with dummy inputs"""
        self.logger.info(f"Warming up DeepLabV3+ model ({num_iterations} iterations)...")
        
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
        
        # Draw spill contours
        for spill_region in results['spill_regions']:
            contour = np.array(spill_region['contour'], dtype=np.int32)
            cv2.drawContours(vis_frame, [contour], -1, (0, 0, 255), 3)
            
            # Draw bounding box
            x1, y1, x2, y2 = spill_region['bbox']
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Add area text
            area_text = f"Area: {spill_region['area']:.0f}px"
            cv2.putText(vis_frame, area_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add summary information
        info_y = 30
        info_texts = [
            f"Spills Detected: {results['num_spills']}",
            f"Coverage: {results['coverage_percent']:.1f}%",
            f"Inference: {results['inference_time_ms']:.1f}ms"
        ]
        
        for text in info_texts:
            cv2.putText(vis_frame, text, (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            info_y += 30
        
        return vis_frame
    
    def save_model(self, save_path: str):
        """Save the current model state"""
        try:
            torch.save({
                'state_dict': self.model.state_dict(),
                'device': str(self.device),
                'model_type': 'deeplabv3_spill_detector'
            }, save_path)
            self.logger.info(f"Model saved to: {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")

def main():
    """Test the DeepLabV3+ spill detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepLabV3+ Spill Detector Test')
    parser.add_argument('--model', type=str, help='Path to custom model weights')
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--webcam', action='store_true', help='Test with webcam')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--fast', action='store_true', help='Use fast mode (smaller input)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = DeepLabV3SpillDetector(args.model)
    
    # Warmup
    detector.warmup()
    
    if args.benchmark:
        print("🚀 Running performance benchmark...")
        
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
        
        print("🔍 Detecting spills in image...")
        results = detector.detect_spills(image, use_fast_mode=args.fast)
        
        print(f"Spill detected: {results['spill_detected']}")
        print(f"Coverage: {results['coverage_percent']:.1f}%")
        print(f"Number of spills: {results['num_spills']}")
        print(f"Inference time: {results['inference_time_ms']:.1f}ms")
        
        # Show visualization
        vis_frame = detector.visualize_results(image, results)
        cv2.imshow('Spill Detection Results', vis_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    elif args.webcam:
        # Test with webcam
        cap = cv2.VideoCapture(0)
        
        print("🎥 Starting webcam spill detection (press 'q' to quit)...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect spills
            results = detector.detect_spills(frame, use_fast_mode=args.fast)
            
            # Visualize
            vis_frame = detector.visualize_results(frame, results)
            
            cv2.imshow('DeepLabV3+ Spill Detection', vis_frame)
            
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
        print("✅ DeepLabV3+ Spill Detector initialized successfully")
        print(f"Device: {detector.device}")
        print("Use --image, --webcam, or --benchmark for testing")

if __name__ == "__main__":
    main()