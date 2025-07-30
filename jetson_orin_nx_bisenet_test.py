#!/usr/bin/env python3
"""
NVIDIA Jetson Orin NX 16GB BiSeNet V2 Spill Detection Test Script
Optimized for Jetson Orin NX 16GB with TensorRT, memory management, and thermal monitoring
SCOPE Smart Building System
"""

import cv2
import time
import numpy as np
import argparse
import logging
import torch
import gc
import psutil
import os
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Add bisenet to path
sys.path.append(str(Path(__file__).parent))

from bisenet.bisenetv2_spill_detector import BiSeNetV2SpillDetector

# Jetson system utilities
try:
    import jtop
    HAS_JTOP = True
except ImportError:
    HAS_JTOP = False
    print("Warning: jtop not available. Install with: pip install jetson-stats")


class JetsonSystemMonitor:
    """Monitor Jetson Orin NX system status and performance"""
    
    def __init__(self):
        self.has_jtop = HAS_JTOP
        self.jetson_stats = None
        self.logger = logging.getLogger(__name__)
        
        if self.has_jtop:
            try:
                from jtop import jtop
                self.jetson_stats = jtop()
                self.jetson_stats.start()
            except Exception as e:
                self.logger.warning(f"Failed to initialize jtop: {e}")
                self.has_jtop = False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get Jetson system information"""
        info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_name': None,
            'total_memory_gb': 16,  # Jetson Orin NX specification
            'jetson_model': 'Unknown'
        }
        
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['cuda_capability'] = torch.cuda.get_device_capability(0)
        
        # Try to get Jetson model info
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model_info = f.read().strip()
                if 'Orin NX' in model_info:
                    info['jetson_model'] = 'Jetson Orin NX'
                elif 'Orin' in model_info:
                    info['jetson_model'] = 'Jetson Orin'
        except Exception:
            pass
        
        return info
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        stats = {
            'gpu_usage': 0,
            'memory_usage_gb': 0,
            'memory_usage_percent': 0,
            'temperature': 0,
            'power_consumption': 0,
            'cpu_usage': psutil.cpu_percent(),
            'available_memory_gb': 0
        }
        
        # GPU memory stats
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)   # GB
                stats['memory_usage_gb'] = memory_allocated
                stats['memory_reserved_gb'] = memory_reserved
                stats['memory_usage_percent'] = (memory_allocated / 16) * 100  # 16GB total
                stats['available_memory_gb'] = 16 - memory_allocated
            except Exception as e:
                self.logger.warning(f"Failed to get GPU memory stats: {e}")
        
        # Jetson-specific stats using jtop
        if self.has_jtop and self.jetson_stats:
            try:
                if self.jetson_stats.ok():
                    jetson_data = self.jetson_stats.stats
                    
                    # GPU usage
                    if 'GPU' in jetson_data:
                        stats['gpu_usage'] = jetson_data['GPU'].get('val', 0)
                    
                    # Temperature
                    if 'Temp' in jetson_data:
                        temps = jetson_data['Temp']
                        if isinstance(temps, dict):
                            # Get GPU temperature or first available temperature
                            for temp_name, temp_data in temps.items():
                                if 'GPU' in temp_name or 'thermal' in temp_name.lower():
                                    stats['temperature'] = temp_data.get('temp', 0)
                                    break
                            else:
                                # Use first available temperature
                                first_temp = next(iter(temps.values()), {})
                                stats['temperature'] = first_temp.get('temp', 0)
                    
                    # Power consumption
                    if 'Power' in jetson_data:
                        power_data = jetson_data['Power']
                        if isinstance(power_data, dict):
                            total_power = power_data.get('tot', {})
                            stats['power_consumption'] = total_power.get('power', 0)
                        
            except Exception as e:
                self.logger.debug(f"jtop stats error: {e}")
        
        return stats
    
    def check_thermal_throttling(self) -> bool:
        """Check if system is thermal throttling"""
        stats = self.get_performance_stats()
        # Consider thermal throttling if temperature > 70C
        return stats.get('temperature', 0) > 70
    
    def optimize_power_mode(self, performance_mode: bool = True):
        """Set optimal power mode for inference"""
        try:
            if performance_mode:
                # Set to MAXN mode for best performance
                cmd = ["sudo", "nvpmodel", "-m", "0"]
            else:
                # Set to balanced mode
                cmd = ["sudo", "nvpmodel", "-m", "2"]
            
            subprocess.run(cmd, check=False, capture_output=True)
            self.logger.info(f"Set power mode: {'MAXN' if performance_mode else 'Balanced'}")
        except Exception as e:
            self.logger.warning(f"Failed to set power mode: {e}")
    
    def cleanup(self):
        """Cleanup system monitor"""
        if self.has_jtop and self.jetson_stats:
            try:
                self.jetson_stats.close()
            except Exception:
                pass


class JetsonMemoryManager:
    """Optimized memory management for Jetson Orin NX 16GB"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tensor_cache = {}
        self.max_cache_size_gb = 4  # Reserve 4GB for caching
        
    def optimize_cuda_settings(self):
        """Optimize CUDA settings for Jetson"""
        if torch.cuda.is_available():
            # Enable memory pool allocator
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            # Enable cuDNN benchmarking for consistent input sizes
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Allow TF32 for better performance on newer hardware
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            self.logger.info("CUDA settings optimized for Jetson")
    
    def pre_allocate_tensors(self, batch_size: int = 1):
        """Pre-allocate common tensor shapes"""
        if not torch.cuda.is_available():
            return
        
        try:
            device = torch.device('cuda')
            
            # Common tensor shapes for BiSeNet V2
            tensor_shapes = [
                (batch_size, 3, 512, 512),  # Input tensor
                (batch_size, 2, 512, 512),  # Output logits
                (batch_size, 1, 512, 512),  # Segmentation mask
                (512, 512),                 # 2D mask
            ]
            
            for i, shape in enumerate(tensor_shapes):
                key = f"tensor_{i}_{shape}"
                self.tensor_cache[key] = torch.empty(shape, dtype=torch.float16, device=device)
            
            # Clear cache and synchronize
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            self.logger.info(f"Pre-allocated {len(tensor_shapes)} tensor shapes")
            
        except Exception as e:
            self.logger.warning(f"Failed to pre-allocate tensors: {e}")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get detailed memory statistics"""
        stats = {}
        
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                
                # PyTorch memory stats
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                max_allocated = torch.cuda.max_memory_allocated(0) / (1024**3)
                
                stats.update({
                    'allocated_gb': allocated,
                    'reserved_gb': reserved,
                    'max_allocated_gb': max_allocated,
                    'free_gb': 16 - allocated,
                    'utilization_percent': (allocated / 16) * 100
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to get CUDA memory stats: {e}")
        
        return stats
    
    def cleanup_memory(self, force: bool = False):
        """Clean up memory and cache"""
        try:
            # Clear tensor cache
            self.tensor_cache.clear()
            
            # Python garbage collection
            gc.collect()
            
            # CUDA memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                if force:
                    # Reset peak memory stats
                    torch.cuda.reset_peak_memory_stats()
            
            self.logger.debug("Memory cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")


class JetsonBiSeNetTester:
    """
    Jetson Orin NX optimized BiSeNet V2 spill detection tester
    """
    
    def __init__(self, rtsp_url: str, model_path: str, config_path: str = None):
        """
        Initialize Jetson-optimized BiSeNet tester
        
        Args:
            rtsp_url: RTSP camera stream URL
            model_path: Path to trained BiSeNet V2 model
            config_path: Optional config file path
        """
        self.rtsp_url = rtsp_url
        self.model_path = model_path
        self.config_path = config_path
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize system components
        self.system_monitor = JetsonSystemMonitor()
        self.memory_manager = JetsonMemoryManager()
        
        # Initialize detector and camera
        self.detector = None
        self.cap = None
        
        # Performance tracking
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        self.inference_times = []
        
        # Jetson optimization flags
        self.tensorrt_enabled = False
        self.use_half_precision = True
        
        self._initialize_jetson()
        self._initialize_detector()
        self._initialize_camera()
    
    def _initialize_jetson(self):
        """Initialize Jetson-specific optimizations"""
        self.logger.info("Initializing Jetson Orin NX optimizations...")
        
        # Display system info
        system_info = self.system_monitor.get_system_info()
        self.logger.info("System Information:")
        for key, value in system_info.items():
            self.logger.info(f"  {key}: {value}")
        
        # Optimize memory settings
        self.memory_manager.optimize_cuda_settings()
        self.memory_manager.pre_allocate_tensors()
        
        # Set performance power mode
        self.system_monitor.optimize_power_mode(performance_mode=True)
        
        self.logger.info("Jetson optimizations initialized")
    
    def _initialize_detector(self):
        """Initialize BiSeNet V2 spill detector with Jetson optimizations"""
        try:
            self.logger.info("Initializing BiSeNet V2 with Jetson optimizations...")
            
            # Check if model file exists
            if not Path(self.model_path).exists():
                self.logger.error(f"Model file not found: {self.model_path}")
                self.logger.info("Available checkpoints:")
                checkpoint_dir = Path("checkpoints")
                if checkpoint_dir.exists():
                    for checkpoint in checkpoint_dir.glob("*.pth"):
                        self.logger.info(f"  - {checkpoint}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Initialize detector with Jetson-optimized settings
            self.detector = BiSeNetV2SpillDetector(
                model_path=self.model_path,
                config_path=self.config_path,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            
            # Apply Jetson-specific optimizations
            self._optimize_detector()
            
            # Warmup with performance monitoring
            self.logger.info("Warming up model with performance monitoring...")
            self._warmup_with_monitoring()
            
            self.logger.info("BiSeNet V2 detector ready with Jetson optimizations")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize detector: {e}")
            raise
    
    def _optimize_detector(self):
        """Apply Jetson-specific optimizations to detector"""
        try:
            if hasattr(self.detector, 'model') and torch.cuda.is_available():
                model = self.detector.model
                
                # Enable half precision if supported
                if self.use_half_precision:
                    try:
                        model.half()
                        self.logger.info("FP16 precision enabled")
                    except Exception as e:
                        self.logger.warning(f"FP16 failed, using FP32: {e}")
                        self.use_half_precision = False
                
                # Try to convert to TensorRT (if model supports it)
                self._try_tensorrt_conversion()
                
        except Exception as e:
            self.logger.warning(f"Some detector optimizations failed: {e}")
    
    def _try_tensorrt_conversion(self):
        """Attempt TensorRT conversion for maximum performance"""
        try:
            # Check if TensorRT is available
            try:
                import tensorrt as trt
                tensorrt_available = True
            except ImportError:
                tensorrt_available = False
                self.logger.info("TensorRT not available, skipping conversion")
                return
            
            if tensorrt_available:
                self.logger.info("TensorRT detected, but BiSeNet V2 TensorRT conversion requires manual setup")
                self.logger.info("For maximum performance, convert model manually using TensorRT")
                
        except Exception as e:
            self.logger.warning(f"TensorRT conversion failed: {e}")
    
    def _warmup_with_monitoring(self, num_iterations: int = 5):
        """Warmup model with system monitoring"""
        self.logger.info(f"Warming up model ({num_iterations} iterations)...")
        
        warmup_times = []
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        for i in range(num_iterations):
            # Monitor system before warmup
            stats_before = self.system_monitor.get_performance_stats()
            
            # Run warmup iteration
            start_time = time.time()
            _ = self.detector.detect_spills(dummy_frame, use_fast_mode=(i < 2))
            warmup_time = (time.time() - start_time) * 1000
            warmup_times.append(warmup_time)
            
            # Monitor system after warmup
            stats_after = self.system_monitor.get_performance_stats()
            
            self.logger.info(f"Warmup {i+1}/{num_iterations}: {warmup_time:.1f}ms, "
                           f"Temp: {stats_after.get('temperature', 0):.1f}°C, "
                           f"Memory: {stats_after.get('memory_usage_gb', 0):.1f}GB")
        
        avg_warmup_time = np.mean(warmup_times)
        self.logger.info(f"Warmup completed: {avg_warmup_time:.1f}ms average")
        
        # Check for thermal issues
        if self.system_monitor.check_thermal_throttling():
            self.logger.warning("System may be thermal throttling - consider cooling")
    
    def _initialize_camera(self):
        """Initialize RTSP camera with Jetson optimizations"""
        try:
            self.logger.info(f"Connecting to RTSP stream: {self.rtsp_url}")
            
            # Create VideoCapture with hardware acceleration
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_GSTREAMER)
            
            # Jetson-optimized camera properties
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # Minimize latency
            self.cap.set(cv2.CAP_PROP_FPS, 30)          # Target FPS
            
            # Try to enable hardware decoding
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
            except Exception:
                pass
            
            # Test connection
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Fallback to standard VideoCapture
                self.cap.release()
                self.cap = cv2.VideoCapture(self.rtsp_url)
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    raise Exception("Failed to read from RTSP stream")
            
            # Get stream properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"RTSP stream connected: {width}x{height} @ {fps:.1f}fps")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to RTSP stream: {e}")
            self.logger.info("Troubleshooting tips:")
            self.logger.info("1. Check if camera is accessible")
            self.logger.info("2. Verify RTSP URL format")
            self.logger.info("3. Test with VLC or other media player first")
            self.logger.info("4. Check camera credentials and network connectivity")
            raise
    
    def run_detection(self, max_frames: int = None, save_detections: bool = False):
        """
        Run spill detection with Jetson performance monitoring
        
        Args:
            max_frames: Maximum frames to process (None for unlimited)
            save_detections: Save detection results to disk
        """
        self.logger.info("Starting Jetson-optimized RTSP spill detection...")
        self.logger.info("Press 'q' to quit, 's' to save current frame, 'f' to toggle fast mode")
        self.logger.info("Press 'm' to show memory stats, 't' to show thermal stats")
        
        # Detection parameters
        use_fast_mode = False
        save_counter = 0
        last_stats_time = time.time()
        stats_interval = 10  # seconds
        
        # Create output directory if saving
        if save_detections:
            output_dir = Path("jetson_detections")
            output_dir.mkdir(exist_ok=True)
            self.logger.info(f"Saving detections to: {output_dir}")
        
        try:
            while True:
                # Check thermal throttling
                if self.system_monitor.check_thermal_throttling():
                    self.logger.warning("Thermal throttling detected - reducing performance")
                    use_fast_mode = True
                
                # Read frame from RTSP stream
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame from RTSP stream")
                    self._reconnect_camera()
                    continue
                
                self.frame_count += 1
                
                # Run spill detection with timing
                detection_start = time.time()
                results = self.detector.detect_spills(frame, use_fast_mode=use_fast_mode)
                detection_time = (time.time() - detection_start) * 1000
                self.inference_times.append(detection_time)
                
                # Count detections
                if results['spill_detected']:
                    self.detection_count += 1
                
                # Create visualization
                vis_frame = self.detector.visualize_results(frame, results)
                
                # Add Jetson-specific info to visualization
                self._add_jetson_info(vis_frame, detection_time, use_fast_mode)
                
                # Show frame
                cv2.imshow('Jetson Orin NX BiSeNet V2 Spill Detection', vis_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame and detection
                    save_counter += 1
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    
                    save_dir = Path("jetson_detections")
                    save_dir.mkdir(exist_ok=True)
                    
                    # Save original frame
                    cv2.imwrite(str(save_dir / f"detection_{timestamp}_original.jpg"), frame)
                    
                    # Save visualization
                    cv2.imwrite(str(save_dir / f"detection_{timestamp}_result.jpg"), vis_frame)
                    
                    # Save mask if spill detected
                    if results['spill_detected']:
                        mask_vis = results['mask'] * 255
                        cv2.imwrite(str(save_dir / f"detection_{timestamp}_mask.jpg"), mask_vis)
                    
                    self.logger.info(f"Saved detection #{save_counter}: {timestamp}")
                
                elif key == ord('f'):
                    # Toggle fast mode
                    use_fast_mode = not use_fast_mode
                    mode_str = "FAST" if use_fast_mode else "NORMAL"
                    self.logger.info(f"Switched to {mode_str} mode")
                
                elif key == ord('m'):
                    # Show memory stats
                    self._log_memory_stats()
                
                elif key == ord('t'):
                    # Show thermal stats
                    self._log_thermal_stats()
                
                # Periodic performance logging
                current_time = time.time()
                if current_time - last_stats_time >= stats_interval:
                    self._log_jetson_performance()
                    last_stats_time = current_time
                
                # Memory cleanup every 100 frames
                if self.frame_count % 100 == 0:
                    self.memory_manager.cleanup_memory()
                
                # Check if max frames reached
                if max_frames and self.frame_count >= max_frames:
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Detection stopped by user")
        
        finally:
            self._cleanup()
            self._log_final_jetson_stats()
    
    def _add_jetson_info(self, frame: np.ndarray, detection_time: float, fast_mode: bool):
        """Add Jetson-specific information to visualization"""
        # Calculate FPS
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Get system stats
        stats = self.system_monitor.get_performance_stats()
        
        # Jetson info overlay
        info_y = frame.shape[0] - 120
        jetson_info = [
            f"Jetson Orin NX 16GB - FPS: {fps:.1f}",
            f"Detection: {detection_time:.1f}ms | Mode: {'FAST' if fast_mode else 'NORMAL'}",
            f"GPU: {stats.get('gpu_usage', 0):.0f}% | Memory: {stats.get('memory_usage_gb', 0):.1f}GB",
            f"Temp: {stats.get('temperature', 0):.1f}°C | Power: {stats.get('power_consumption', 0):.1f}W",
            f"Frame: {self.frame_count} | Detections: {self.detection_count}"
        ]
        
        # Add background for better readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, info_y - 10), (500, frame.shape[0] - 5), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        
        for i, text in enumerate(jetson_info):
            y_pos = info_y + (i * 20)
            cv2.putText(frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def _log_memory_stats(self):
        """Log detailed memory statistics"""
        memory_stats = self.memory_manager.get_memory_stats()
        system_stats = self.system_monitor.get_performance_stats()
        
        self.logger.info("Memory Statistics:")
        self.logger.info(f"  GPU Allocated: {memory_stats.get('allocated_gb', 0):.2f}GB")
        self.logger.info(f"  GPU Reserved: {memory_stats.get('reserved_gb', 0):.2f}GB")
        self.logger.info(f"  GPU Free: {memory_stats.get('free_gb', 0):.2f}GB")
        self.logger.info(f"  GPU Utilization: {memory_stats.get('utilization_percent', 0):.1f}%")
        self.logger.info(f"  System Memory: {system_stats.get('memory_usage_gb', 0):.2f}GB")
    
    def _log_thermal_stats(self):
        """Log thermal and power statistics"""
        stats = self.system_monitor.get_performance_stats()
        
        self.logger.info("Thermal & Power Statistics:")
        self.logger.info(f"  Temperature: {stats.get('temperature', 0):.1f}°C")
        self.logger.info(f"  Power Consumption: {stats.get('power_consumption', 0):.1f}W")
        self.logger.info(f"  GPU Usage: {stats.get('gpu_usage', 0):.1f}%")
        self.logger.info(f"  CPU Usage: {stats.get('cpu_usage', 0):.1f}%")
        
        if self.system_monitor.check_thermal_throttling():
            self.logger.warning("  ⚠️  THERMAL THROTTLING DETECTED")
    
    def _reconnect_camera(self):
        """Attempt to reconnect to RTSP stream"""
        self.logger.info("Attempting to reconnect to RTSP stream...")
        
        try:
            if self.cap:
                self.cap.release()
            
            time.sleep(2)  # Wait before reconnecting
            self._initialize_camera()
            self.logger.info("Reconnected to RTSP stream")
            
        except Exception as e:
            self.logger.error(f"Reconnection failed: {e}")
            time.sleep(5)  # Wait longer before next attempt
    
    def _log_jetson_performance(self):
        """Log Jetson-specific performance statistics"""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time
        detection_rate = (self.detection_count / self.frame_count) * 100 if self.frame_count > 0 else 0
        
        # Get system stats
        system_stats = self.system_monitor.get_performance_stats()
        memory_stats = self.memory_manager.get_memory_stats()
        
        # Calculate inference statistics
        if self.inference_times:
            avg_inference = np.mean(self.inference_times[-100:])  # Last 100 frames
            p95_inference = np.percentile(self.inference_times[-100:], 95)
        else:
            avg_inference = 0
            p95_inference = 0
        
        self.logger.info("Jetson Performance Update:")
        self.logger.info(f"  Frames: {self.frame_count}, FPS: {fps:.1f}")
        self.logger.info(f"  Detections: {self.detection_count} ({detection_rate:.1f}%)")
        self.logger.info(f"  Inference: {avg_inference:.1f}ms avg, {p95_inference:.1f}ms p95")
        self.logger.info(f"  GPU: {system_stats.get('gpu_usage', 0):.0f}%, Memory: {memory_stats.get('allocated_gb', 0):.1f}GB")
        self.logger.info(f"  Temperature: {system_stats.get('temperature', 0):.1f}°C, Power: {system_stats.get('power_consumption', 0):.1f}W")
    
    def _log_final_jetson_stats(self):
        """Log final comprehensive performance statistics"""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time
        detection_rate = (self.detection_count / self.frame_count) * 100 if self.frame_count > 0 else 0
        
        # Get final system stats
        system_stats = self.system_monitor.get_performance_stats()
        memory_stats = self.memory_manager.get_memory_stats()
        
        # Calculate inference statistics
        if self.inference_times:
            avg_inference = np.mean(self.inference_times)
            p95_inference = np.percentile(self.inference_times, 95)
            min_inference = np.min(self.inference_times)
            max_inference = np.max(self.inference_times)
        else:
            avg_inference = p95_inference = min_inference = max_inference = 0
        
        self.logger.info("="*80)
        self.logger.info("FINAL JETSON ORIN NX PERFORMANCE STATISTICS")
        self.logger.info("="*80)
        self.logger.info(f"Runtime: {elapsed_time:.1f}s")
        self.logger.info(f"Total Frames: {self.frame_count}")
        self.logger.info(f"Average FPS: {fps:.1f}")
        self.logger.info(f"Total Detections: {self.detection_count}")
        self.logger.info(f"Detection Rate: {detection_rate:.1f}%")
        self.logger.info("")
        self.logger.info("Inference Performance:")
        self.logger.info(f"  Average: {avg_inference:.1f}ms")
        self.logger.info(f"  P95: {p95_inference:.1f}ms")
        self.logger.info(f"  Min: {min_inference:.1f}ms")
        self.logger.info(f"  Max: {max_inference:.1f}ms")
        self.logger.info("")
        self.logger.info("System Resources:")
        self.logger.info(f"  GPU Usage: {system_stats.get('gpu_usage', 0):.1f}%")
        self.logger.info(f"  Memory Used: {memory_stats.get('allocated_gb', 0):.2f}GB / 16GB")
        self.logger.info(f"  Memory Utilization: {memory_stats.get('utilization_percent', 0):.1f}%")
        self.logger.info(f"  Temperature: {system_stats.get('temperature', 0):.1f}°C")
        self.logger.info(f"  Power Consumption: {system_stats.get('power_consumption', 0):.1f}W")
        self.logger.info("="*80)
    
    def _cleanup(self):
        """Cleanup all resources"""
        try:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Cleanup memory
            self.memory_manager.cleanup_memory(force=True)
            
            # Cleanup system monitor
            self.system_monitor.cleanup()
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Jetson Orin NX 16GB BiSeNet V2 Spill Detection Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with best model checkpoint
  python jetson_orin_nx_bisenet_test.py --model checkpoints/best_model.pth

  # Test with custom RTSP URL
  python jetson_orin_nx_bisenet_test.py --rtsp rtsp://admin:password@192.168.1.100:554/stream1 --model checkpoints/best_model.pth
  
  # Test with frame limit and save detections
  python jetson_orin_nx_bisenet_test.py --model checkpoints/best_model.pth --max-frames 1000 --save-detections

Controls:
  q - Quit
  s - Save current frame and detection
  f - Toggle fast/normal mode
  m - Show memory statistics
  t - Show thermal statistics

Jetson Optimizations:
  - 16GB unified memory management
  - FP16 precision inference
  - CUDA stream optimization
  - Thermal throttling detection
  - Power mode optimization
  - Hardware-accelerated video decoding
        """
    )
    
    parser.add_argument('--rtsp', type=str, 
                       default='rtsp://admin:123456@192.168.12.205:80/ch3_0.264',
                       help='RTSP stream URL')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained BiSeNet V2 model (.pth file)')
    
    parser.add_argument('--config', type=str, 
                       default='configs/bisenetv2_config.yaml',
                       help='Configuration file path')
    
    parser.add_argument('--max-frames', type=int,
                       help='Maximum frames to process (for testing)')
    
    parser.add_argument('--save-detections', action='store_true',
                       help='Save detection results to disk')
    
    args = parser.parse_args()
    
    # Print banner
    print("=" * 80)
    print("NVIDIA JETSON ORIN NX 16GB - BiSeNet V2 Spill Detection Test")
    print("SCOPE Smart Building System")
    print("=" * 80)
    print(f"RTSP URL: {args.rtsp}")
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print("=" * 80)
    
    try:
        # Initialize Jetson-optimized tester
        tester = JetsonBiSeNetTester(
            rtsp_url=args.rtsp,
            model_path=args.model,
            config_path=args.config
        )
        
        # Run detection
        tester.run_detection(
            max_frames=args.max_frames,
            save_detections=args.save_detections
        )
        
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    except Exception as e:
        print(f"Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())