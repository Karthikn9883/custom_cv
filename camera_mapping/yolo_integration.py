"""
YOLO Integration for Camera Mapping
===================================

This module integrates the camera calibration system with YOLOv5 object detection
for real-world coordinate mapping of detected objects.
"""

import cv2
import numpy as np
import torch
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add the parent directory to path to import camera_calibrator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_mapping.camera_calibrator import CameraMapper

class YOLOCameraIntegration:
    """
    Integrates YOLOv5 detection with camera calibration for real-world mapping
    """
    
    def __init__(self, 
                 camera_id: str,
                 calibration_file: str,
                 yolo_weights: str = "yolov5s.pt",
                 confidence_threshold: float = 0.5):
        """
        Initialize the YOLO-Camera integration
        
        Args:
            camera_id: Unique identifier for the camera
            calibration_file: Path to the camera calibration file
            yolo_weights: Path to YOLO weights file
            confidence_threshold: Minimum confidence for detections
        """
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        
        # Initialize camera mapper
        self.mapper = CameraMapper(camera_id)
        if not self.mapper.load_calibration(calibration_file):
            raise ValueError(f"Failed to load calibration from {calibration_file}")
        
        # Initialize YOLO model
        self.model = self._load_yolo_model(yolo_weights)
        
        # Class names for COCO dataset (YOLOv5 default)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
    def _load_yolo_model(self, weights_path: str):
        """Load YOLOv5 model"""
        try:
            # Try to load from YOLOv5 directory
            yolo_path = Path(__file__).parent.parent / "yolov5-master"
            if yolo_path.exists():
                sys.path.insert(0, str(yolo_path))
            
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
            model.conf = self.confidence_threshold
            return model
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            print("Falling back to local implementation...")
            return None
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in the frame using YOLO
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detection dictionaries with world coordinates
        """
        detections = []
        
        if self.model is None:
            # Fallback: simulate detections for demo
            return self._simulate_detections(frame)
        
        # Run YOLO inference
        results = self.model(frame)
        
        # Parse results
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if conf >= self.confidence_threshold:
                x1, y1, x2, y2 = map(int, box)
                
                # Calculate object center
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Map to world coordinates
                world_coords = self.mapper.image_to_world((center_x, center_y))
                
                if world_coords:
                    detection = {
                        'class_id': int(cls),
                        'class_name': self.class_names[int(cls)] if int(cls) < len(self.class_names) else 'unknown',
                        'confidence': float(conf),
                        'bbox': [x1, y1, x2, y2],
                        'center_image': (center_x, center_y),
                        'center_world': world_coords,
                        'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
                    }
                    detections.append(detection)
        
        return detections
    
    def _simulate_detections(self, frame: np.ndarray) -> List[Dict]:
        """Simulate object detections for demo purposes"""
        h, w = frame.shape[:2]
        
        # Create some fake detections
        fake_detections = [
            {'bbox': [w//4, h//4, w//2, h//2], 'class': 'bottle', 'conf': 0.85},
            {'bbox': [w//2, h//3, 3*w//4, 2*h//3], 'class': 'person', 'conf': 0.92},
            {'bbox': [w//8, 3*h//4, w//3, h-10], 'class': 'chair', 'conf': 0.76}
        ]
        
        detections = []
        for det in fake_detections:
            x1, y1, x2, y2 = det['bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            world_coords = self.mapper.image_to_world((center_x, center_y))
            
            if world_coords:
                detection = {
                    'class_id': -1,  # Fake ID
                    'class_name': det['class'],
                    'confidence': det['conf'],
                    'bbox': [x1, y1, x2, y2],
                    'center_image': (center_x, center_y),
                    'center_world': world_coords,
                    'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
                }
                detections.append(detection)
        
        return detections
    
    def filter_detections(self, detections: List[Dict], 
                         target_classes: List[str] = None,
                         min_confidence: float = None,
                         world_bounds: Tuple[float, float, float, float] = None) -> List[Dict]:
        """
        Filter detections based on various criteria
        
        Args:
            detections: List of detection dictionaries
            target_classes: List of class names to keep (None = keep all)
            min_confidence: Minimum confidence threshold (None = use default)
            world_bounds: (min_x, min_y, max_x, max_y) world coordinate bounds
            
        Returns:
            Filtered list of detections
        """
        filtered = detections.copy()
        
        # Filter by class
        if target_classes:
            filtered = [d for d in filtered if d['class_name'] in target_classes]
        
        # Filter by confidence
        if min_confidence:
            filtered = [d for d in filtered if d['confidence'] >= min_confidence]
        
        # Filter by world bounds
        if world_bounds:
            min_x, min_y, max_x, max_y = world_bounds
            filtered = [d for d in filtered 
                       if (min_x <= d['center_world'][0] <= max_x and 
                           min_y <= d['center_world'][1] <= max_y)]
        
        return filtered
    
    def generate_robot_commands(self, detections: List[Dict]) -> List[Dict]:
        """
        Generate robot navigation commands from detections
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of robot command dictionaries
        """
        commands = []
        
        # Priority mapping for different object types
        priority_map = {
            'person': 1,      # High priority - safety
            'bottle': 2,      # Medium priority - cleanup
            'cup': 2,
            'chair': 3,       # Low priority - obstacle
            'couch': 3,
            'default': 4      # Lowest priority
        }
        
        # Action mapping
        action_map = {
            'bottle': 'PICKUP',
            'cup': 'PICKUP',
            'person': 'AVOID',
            'chair': 'NAVIGATE_AROUND',
            'couch': 'NAVIGATE_AROUND',
            'default': 'INVESTIGATE'
        }
        
        for detection in detections:
            class_name = detection['class_name']
            world_x, world_y = detection['center_world']
            
            priority = priority_map.get(class_name, priority_map['default'])
            action = action_map.get(class_name, action_map['default'])
            
            command = {
                'target_id': f"{class_name}_{detection['timestamp']:.0f}",
                'action': action,
                'priority': priority,
                'target_world': (world_x, world_y),
                'target_image': detection['center_image'],
                'confidence': detection['confidence'],
                'class_name': class_name,
                'timestamp': detection['timestamp']
            }
            
            commands.append(command)
        
        # Sort by priority (lower number = higher priority)
        commands.sort(key=lambda x: x['priority'])
        
        return commands
    
    def visualize_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Visualize detections with world coordinates on the frame
        
        Args:
            frame: Input image frame
            detections: List of detection dictionaries
            
        Returns:
            Frame with detection overlays
        """
        vis_frame = frame.copy()
        
        # Draw calibration grid
        vis_frame = self.mapper.visualize_calibration(vis_frame, grid_spacing=1.0)
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x, center_y = detection['center_image']
            world_x, world_y = detection['center_world']
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(vis_frame, (center_x, center_y), 5, (0, 255, 0), -1)
            
            # Draw label with world coordinates
            label = f"{detection['class_name']} ({world_x:.1f}, {world_y:.1f})m"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Background for label
            cv2.rectangle(vis_frame, (x1, y1-label_size[1]-10), 
                         (x1+label_size[0], y1), (0, 255, 0), -1)
            
            # Label text
            cv2.putText(vis_frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Confidence
            conf_text = f"{detection['confidence']:.2f}"
            cv2.putText(vis_frame, conf_text, (x2-30, y1+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return vis_frame
    
    def process_video_stream(self, source: int = 0, output_file: str = None):
        """
        Process live video stream with object detection and coordinate mapping
        
        Args:
            source: Video source (0 for webcam, path for video file)
            output_file: Optional output video file path
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        
        # Setup video writer if output requested
        writer = None
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        print("Processing video stream...")
        print("Press 'q' to quit, 's' to save frame, 'r' to show robot commands")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect objects
                detections = self.detect_objects(frame)
                
                # Visualize
                vis_frame = self.visualize_detections(frame, detections)
                
                # Add frame info
                info_text = f"Frame: {frame_count} | Detections: {len(detections)}"
                cv2.putText(vis_frame, info_text, (10, vis_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow("YOLO Camera Mapping", vis_frame)
                
                # Save frame if requested
                if writer:
                    writer.write(vis_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(f"detection_frame_{frame_count}.jpg", vis_frame)
                    print(f"Saved frame {frame_count}")
                elif key == ord('r'):
                    # Show robot commands
                    commands = self.generate_robot_commands(detections)
                    print(f"\nRobot Commands (Frame {frame_count}):")
                    for cmd in commands:
                        print(f"  {cmd['action']}: {cmd['class_name']} at "
                              f"({cmd['target_world'][0]:.2f}, {cmd['target_world'][1]:.2f})")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

def main():
    """Demo the YOLO-Camera integration"""
    
    # Check for calibration file
    calibration_files = [
        "demo_calibration.json",
        "aruco_calibration.json", 
        "../camera_mapping/demo_calibration.json"
    ]
    
    calibration_file = None
    for cal_file in calibration_files:
        if os.path.exists(cal_file):
            calibration_file = cal_file
            break
    
    if not calibration_file:
        print("No calibration file found. Please run camera calibration first.")
        print("Expected files: demo_calibration.json or aruco_calibration.json")
        return
    
    try:
        # Initialize integration
        integration = YOLOCameraIntegration(
            camera_id="yolo_camera",
            calibration_file=calibration_file,
            confidence_threshold=0.5
        )
        
        print(f"Loaded calibration from: {calibration_file}")
        print("Starting YOLO detection with camera mapping...")
        
        # Process video stream
        integration.process_video_stream(source=0)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. A valid calibration file")
        print("2. YOLOv5 installed")
        print("3. Camera connected")

if __name__ == "__main__":
    main()
