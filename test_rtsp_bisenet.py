#!/usr/bin/env python3
"""
RTSP BiSeNet V2 Spill Detection Test Script
Real-time spill detection using trained BiSeNet V2 model with RTSP camera stream
SCOPE Smart Building System
"""

import cv2
import time
import numpy as np
import argparse
import logging
from pathlib import Path
import sys

# Add bisenet to path
sys.path.append(str(Path(__file__).parent))

from bisenet.bisenetv2_spill_detector import BiSeNetV2SpillDetector


class RTSPSpillTester:
    """
    RTSP stream tester for BiSeNet V2 spill detection
    """
    
    def __init__(self, rtsp_url: str, model_path: str, config_path: str = None):
        """
        Initialize RTSP tester
        
        Args:
            rtsp_url: RTSP camera stream URL
            model_path: Path to trained BiSeNet V2 model
            config_path: Optional config file path
        """
        self.rtsp_url = rtsp_url
        self.model_path = model_path
        self.config_path = config_path
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize detector
        self.detector = None
        self.cap = None
        
        # Performance tracking
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        
        self._initialize_detector()
        self._initialize_camera()
    
    def _initialize_detector(self):
        """Initialize BiSeNet V2 spill detector"""
        try:
            self.logger.info("Initializing BiSeNet V2 spill detector...")
            
            # Check if model file exists
            if not Path(self.model_path).exists():
                self.logger.error(f"Model file not found: {self.model_path}")
                self.logger.info("Available checkpoints:")
                checkpoint_dir = Path("checkpoints")
                if checkpoint_dir.exists():
                    for checkpoint in checkpoint_dir.glob("*.pth"):
                        self.logger.info(f"  - {checkpoint}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Initialize detector with trained model
            self.detector = BiSeNetV2SpillDetector(
                model_path=self.model_path,
                config_path=self.config_path
            )
            
            # Warmup the model
            self.logger.info("Warming up model...")
            self.detector.warmup(num_iterations=3)
            
            self.logger.info("BiSeNet V2 detector ready")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize detector: {e}")
            raise
    
    def _initialize_camera(self):
        """Initialize RTSP camera connection"""
        try:
            self.logger.info(f"Connecting to RTSP stream: {self.rtsp_url}")
            
            # Create VideoCapture with RTSP URL
            self.cap = cv2.VideoCapture(self.rtsp_url)
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
            self.cap.set(cv2.CAP_PROP_FPS, 30)       # Set desired FPS
            
            # Test connection
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
            self.logger.info("1. Check if camera is accessible: ping 192.168.1.204")
            self.logger.info("2. Verify RTSP URL format: rtsp://username:password@ip:port/path")
            self.logger.info("3. Test with VLC or other media player first")
            self.logger.info("4. Check camera credentials and network connectivity")
            raise
    
    def run_detection(self, max_frames: int = None, save_detections: bool = False):
        """
        Run spill detection on RTSP stream
        
        Args:
            max_frames: Maximum frames to process (None for unlimited)
            save_detections: Save detection results to disk
        """
        self.logger.info("Starting RTSP spill detection...")
        self.logger.info("Press 'q' to quit, 's' to save current frame, 'f' to toggle fast mode")
        
        # Detection parameters
        use_fast_mode = False
        save_counter = 0
        
        # Create output directory if saving
        if save_detections:
            output_dir = Path("rtsp_detections")
            output_dir.mkdir(exist_ok=True)
            self.logger.info(f"Saving detections to: {output_dir}")
        
        try:
            while True:
                # Read frame from RTSP stream
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame from RTSP stream")
                    # Try to reconnect
                    self._reconnect_camera()
                    continue
                
                self.frame_count += 1
                
                # Run spill detection
                detection_start = time.time()
                results = self.detector.detect_spills(frame, use_fast_mode=use_fast_mode)
                detection_time = (time.time() - detection_start) * 1000
                
                # Count detections
                if results['spill_detected']:
                    self.detection_count += 1
                
                # Create visualization
                vis_frame = self.detector.visualize_results(frame, results)
                
                # Add additional info to visualization
                self._add_stream_info(vis_frame, detection_time, use_fast_mode)
                
                # Show frame
                cv2.imshow('RTSP BiSeNet V2 Spill Detection', vis_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame and detection
                    if save_detections or True:  # Always allow manual save
                        save_counter += 1
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        
                        # Save original frame
                        cv2.imwrite(f"detection_{timestamp}_original.jpg", frame)
                        
                        # Save visualization
                        cv2.imwrite(f"detection_{timestamp}_result.jpg", vis_frame)
                        
                        # Save mask if spill detected
                        if results['spill_detected']:
                            mask_vis = results['mask'] * 255
                            cv2.imwrite(f"detection_{timestamp}_mask.jpg", mask_vis)
                        
                        self.logger.info(f"Saved detection #{save_counter}: {timestamp}")
                
                elif key == ord('f'):
                    # Toggle fast mode
                    use_fast_mode = not use_fast_mode
                    mode_str = "FAST" if use_fast_mode else "NORMAL"
                    self.logger.info(f"Switched to {mode_str} mode")
                
                # Check if max frames reached
                if max_frames and self.frame_count >= max_frames:
                    break
                
                # Periodic performance logging
                if self.frame_count % 100 == 0:
                    self._log_performance()
        
        except KeyboardInterrupt:
            self.logger.info("Detection stopped by user")
        
        finally:
            self._cleanup()
            self._log_final_stats()
    
    def _add_stream_info(self, frame: np.ndarray, detection_time: float, fast_mode: bool):
        """Add stream information to visualization"""
        # Calculate FPS
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Stream info
        info_y = frame.shape[0] - 60
        stream_info = [
            f"RTSP Stream FPS: {fps:.1f}",
            f"Detection: {detection_time:.1f}ms",
            f"Mode: {'FAST' if fast_mode else 'NORMAL'}",
            f"Frame: {self.frame_count}"
        ]
        
        for i, text in enumerate(stream_info):
            y_pos = info_y + (i * 15)
            cv2.putText(frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
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
    
    def _log_performance(self):
        """Log performance statistics"""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time
        detection_rate = (self.detection_count / self.frame_count) * 100
        
        stats = self.detector.get_performance_stats()
        
        self.logger.info(f"Performance Update:")
        self.logger.info(f"  Frames: {self.frame_count}, FPS: {fps:.1f}")
        self.logger.info(f"  Detections: {self.detection_count} ({detection_rate:.1f}%)")
        self.logger.info(f"  Avg inference: {stats.get('avg_inference_ms', 0):.1f}ms")
    
    def _log_final_stats(self):
        """Log final performance statistics"""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time
        detection_rate = (self.detection_count / self.frame_count) * 100
        
        stats = self.detector.get_performance_stats()
        
        self.logger.info("Final Performance Statistics:")
        self.logger.info(f"  Total runtime: {elapsed_time:.1f}s")
        self.logger.info(f"  Total frames: {self.frame_count}")
        self.logger.info(f"  Average FPS: {fps:.1f}")
        self.logger.info(f"  Total detections: {self.detection_count}")
        self.logger.info(f"  Detection rate: {detection_rate:.1f}%")
        self.logger.info(f"  Avg inference time: {stats.get('avg_inference_ms', 0):.1f}ms")
        self.logger.info(f"  P95 inference time: {stats.get('p95_inference_ms', 0):.1f}ms")
    
    def _cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Cleanup completed")


def main():
    parser = argparse.ArgumentParser(
        description='RTSP BiSeNet V2 Spill Detection Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with best model checkpoint
  python test_rtsp_bisenet.py --model checkpoints/best_model.pth

  # Test with custom RTSP URL
  python test_rtsp_bisenet.py --rtsp rtsp://admin:password@192.168.1.100:554/stream1 --model checkpoints/best_model.pth
  
  # Test with frame limit and save detections
  python test_rtsp_bisenet.py --model checkpoints/best_model.pth --max-frames 1000 --save-detections

Controls:
  q - Quit
  s - Save current frame and detection
  f - Toggle fast/normal mode
        """
    )
    
    parser.add_argument('--rtsp', type=str, 
                       default='rtsp://admin:123456@192.168.1.204:80/ch3_0.264',
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
    print("SCOPE Smart Building - RTSP BiSeNet V2 Spill Detection Test")
    print("=" * 80)
    print(f"RTSP URL: {args.rtsp}")
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print("=" * 80)
    
    try:
        # Initialize tester
        tester = RTSPSpillTester(
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