#!/usr/bin/env python3
"""
GStreamer Utilities for SCOPE Smart Building
Support for RTSP/ONVIF camera feeds with frame skip and ROI cropping
Optimized for <30ms latency requirement
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GObject
    GST_AVAILABLE = True
except ImportError:
    GST_AVAILABLE = False
    logging.warning("GStreamer Python bindings not available. Install with: sudo apt-get install python3-gst-1.0")

class RTSPCamera:
    """
    RTSP Camera handler with GStreamer pipeline optimization
    Supports frame skip and ROI cropping for <30ms latency
    """
    
    def __init__(self, rtsp_url: str, target_fps: int = 30, buffer_size: int = 1):
        self.rtsp_url = rtsp_url
        self.target_fps = target_fps
        self.buffer_size = buffer_size
        self.cap = None
        self.frame_skip_count = 0
        self.total_frames = 0
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize GStreamer if available
        if GST_AVAILABLE:
            Gst.init(None)
        
        self._initialize_capture()
    
    def _get_gstreamer_pipeline(self) -> str:
        """
        Build optimized GStreamer pipeline for RTSP with frame skip
        """
        pipeline = (
            f"rtspsrc location={self.rtsp_url} "
            f"buffer-mode=1 "  # Low latency mode
            f"latency=0 "
            f"protocols=tcp "
            f"! rtph264depay "
            f"! h264parse "
            f"! avdec_h264 "
            f"max-threads=2 "  # Limit decoder threads for power efficiency
            f"! videorate "
            f"! video/x-raw,framerate={self.target_fps}/1 "
            f"! videoconvert "
            f"! videoscale "
            f"! video/x-raw,format=BGR "
            f"! appsink emit-signals=true sync=false max-buffers={self.buffer_size} drop=true"
        )
        return pipeline
    
    def _initialize_capture(self):
        """Initialize video capture with optimized settings"""
        if GST_AVAILABLE:
            # Use GStreamer pipeline for better RTSP handling
            pipeline = self._get_gstreamer_pipeline()
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            self.logger.info(f"Initialized GStreamer pipeline: {pipeline}")
        else:
            # Fallback to OpenCV RTSP
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            # Optimize OpenCV RTSP settings
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open RTSP stream: {self.rtsp_url}")
        
        self.logger.info(f"RTSP camera initialized: {self.rtsp_url}")
    
    def read_frame(self, roi: Optional[Tuple[int, int, int, int]] = None) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame with optional ROI cropping
        Args:
            roi: (x, y, width, height) for region of interest
        Returns:
            (success, frame) tuple
        """
        ret, frame = self.cap.read()
        
        if not ret:
            return False, None
        
        self.total_frames += 1
        
        # Apply ROI cropping if specified
        if roi is not None and frame is not None:
            x, y, w, h = roi
            frame = frame[y:y+h, x:x+w]
        
        return True, frame
    
    def skip_frames(self, count: int = 1):
        """Skip frames to reduce latency"""
        for _ in range(count):
            ret, _ = self.cap.read()
            if ret:
                self.frame_skip_count += 1
    
    def get_properties(self) -> Dict[str, Any]:
        """Get camera properties"""
        if not self.cap:
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'total_frames': self.total_frames,
            'skipped_frames': self.frame_skip_count
        }
    
    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            self.logger.info("RTSP camera released")

class ONVIFCamera:
    """
    ONVIF Camera handler with optimized settings
    Extends RTSP functionality with ONVIF-specific features
    """
    
    def __init__(self, onvif_url: str, username: str = "", password: str = "", **kwargs):
        # Build RTSP URL from ONVIF parameters
        if username and password:
            rtsp_url = onvif_url.replace('://', f'://{username}:{password}@')
        else:
            rtsp_url = onvif_url
        
        # Initialize as RTSP camera
        super().__init__(rtsp_url, **kwargs)
        
        self.onvif_url = onvif_url
        self.username = username
        self.password = password

class CameraManager:
    """
    Manage multiple camera feeds for Stage 1 detection
    Handles RTSP/ONVIF streams with load balancing
    """
    
    def __init__(self):
        self.cameras: Dict[str, RTSPCamera] = {}
        self.active_camera = None
        self.logger = logging.getLogger(__name__)
    
    def add_camera(self, camera_id: str, rtsp_url: str, **kwargs) -> bool:
        """Add a new camera feed"""
        try:
            camera = RTSPCamera(rtsp_url, **kwargs)
            self.cameras[camera_id] = camera
            
            if self.active_camera is None:
                self.active_camera = camera_id
            
            self.logger.info(f"Added camera {camera_id}: {rtsp_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add camera {camera_id}: {e}")
            return False
    
    def get_frame(self, camera_id: Optional[str] = None, roi: Optional[Tuple[int, int, int, int]] = None):
        """Get frame from specified camera or active camera"""
        target_id = camera_id or self.active_camera
        
        if target_id not in self.cameras:
            return False, None
        
        return self.cameras[target_id].read_frame(roi)
    
    def switch_camera(self, camera_id: str) -> bool:
        """Switch active camera"""
        if camera_id in self.cameras:
            self.active_camera = camera_id
            self.logger.info(f"Switched to camera: {camera_id}")
            return True
        return False
    
    def get_camera_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all cameras"""
        status = {}
        for cam_id, camera in self.cameras.items():
            status[cam_id] = {
                'properties': camera.get_properties(),
                'is_active': cam_id == self.active_camera
            }
        return status
    
    def release_all(self):
        """Release all camera resources"""
        for camera in self.cameras.values():
            camera.release()
        self.cameras.clear()
        self.active_camera = None
        self.logger.info("All cameras released")

def test_rtsp_connection(rtsp_url: str, timeout: int = 5) -> bool:
    """Test RTSP connection"""
    try:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        ret, frame = cap.read()
        cap.release()
        
        return ret and frame is not None
        
    except Exception:
        return False

def main():
    """Test GStreamer utilities"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test GStreamer RTSP utilities')
    parser.add_argument('--rtsp', type=str, required=True, help='RTSP URL to test')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS')
    parser.add_argument('--roi', type=str, help='ROI as "x,y,w,h"')
    
    args = parser.parse_args()
    
    # Test connection first
    print(f"Testing RTSP connection: {args.rtsp}")
    if not test_rtsp_connection(args.rtsp):
        print("❌ RTSP connection failed")
        return
    
    print("✅ RTSP connection successful")
    
    # Parse ROI if provided
    roi = None
    if args.roi:
        try:
            roi = tuple(map(int, args.roi.split(',')))
            print(f"Using ROI: {roi}")
        except ValueError:
            print("Invalid ROI format. Use: x,y,w,h")
            return
    
    # Test camera
    try:
        camera = RTSPCamera(args.rtsp, target_fps=args.fps)
        
        print("Press 'q' to quit, 's' to skip frames")
        
        while True:
            ret, frame = camera.read_frame(roi)
            
            if not ret:
                print("Failed to read frame")
                break
            
            # Show frame properties
            props = camera.get_properties()
            cv2.putText(frame, f"FPS: {props['fps']:.1f} | Frames: {props['total_frames']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('RTSP Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                camera.skip_frames(5)
                print("Skipped 5 frames")
        
        camera.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()