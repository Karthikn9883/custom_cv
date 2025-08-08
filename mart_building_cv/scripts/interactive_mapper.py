import cv2
import numpy as np
import threading
import time
import yaml
import os
import sys
from typing import List, Tuple, Dict, Any, Optional
import json
from datetime import datetime

# Add the src directory to the path to import projection_utils
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from projection_utils import ProjectionSystem

class Marker:
    def __init__(self, id: int, pixel_coords: Tuple[float, float], world_coords: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)):
        self.id = id
        self.pixel_coords = pixel_coords
        self.world_coords = world_coords
        self.color = color
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'pixel_coords': self.pixel_coords,
            'world_coords': self.world_coords.tolist(),
            'color': self.color,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Marker':
        marker = cls(
            id=data['id'],
            pixel_coords=tuple(data['pixel_coords']),
            world_coords=np.array(data['world_coords']),
            color=tuple(data['color'])
        )
        marker.timestamp = datetime.fromisoformat(data['timestamp'])
        return marker

class InteractiveMapper:
    def __init__(self, rtsp_url: str = None, config_dir: str = None):
        # Initialize projection system
        try:
            self.projection_system = ProjectionSystem(config_dir)
            if not self.projection_system.is_calibrated():
                print("WARNING: Projection system not fully calibrated!")
                print("Some features may not work correctly.")
        except Exception as e:
            print(f"Error initializing projection system: {e}")
            self.projection_system = None
        
        # Load RTSP URL from config if not provided
        if rtsp_url is None:
            if config_dir is None:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                config_dir = os.path.join(script_dir, '..', 'configs')
            
            config_path = os.path.join(config_dir, 'main_config.yaml')
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                rtsp_url = config.get('input_source', '0')
            except Exception as e:
                print(f"Error loading config: {e}")
                rtsp_url = '0'
        
        self.rtsp_url = rtsp_url
        self.markers: List[Marker] = []
        self.marker_lock = threading.Lock()
        self.next_marker_id = 1
        
        # Video capture
        self.cap = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # UI state
        self.running = True
        self.video_thread = None
        
        # Colors for markers (cycling through them)
        self.marker_colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]
    
    def initialize_camera(self) -> bool:
        """Initialize the camera/video source."""
        try:
            # Handle different source types
            if self.rtsp_url.isdigit():
                source = int(self.rtsp_url)
            else:
                source = self.rtsp_url
            
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open video source: {source}")
                return False
            
            # Set some properties for better performance
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            print(f"✓ Video source opened successfully: {source}")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def video_capture_thread(self):
        """Thread function for continuous video capture."""
        while self.running:
            if self.cap is None:
                time.sleep(0.1)
                continue
                
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame.copy()
            else:
                print("Warning: Could not read frame")
                time.sleep(0.1)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks on the video window."""
        if event == cv2.EVENT_LBUTTONDOWN and self.projection_system is not None:
            # Project 2D pixel to 3D world coordinates
            world_coords = self.projection_system.project_2d_to_3d((x, y))
            
            if world_coords is not None:
                # Create new marker
                color = self.marker_colors[(self.next_marker_id - 1) % len(self.marker_colors)]
                marker = Marker(self.next_marker_id, (x, y), world_coords, color)
                
                with self.marker_lock:
                    self.markers.append(marker)
                    self.next_marker_id += 1
                
                print(f"Marker {marker.id} placed:")
                print(f"  Pixel: ({x}, {y})")
                print(f"  World: ({world_coords[0]:.3f}, {world_coords[1]:.3f}, {world_coords[2]:.3f})")
            else:
                print(f"Could not project pixel ({x}, {y}) to 3D coordinates")
    
    def draw_markers(self, frame: np.ndarray) -> np.ndarray:
        """Draw all markers on the frame."""
        annotated_frame = frame.copy()
        
        with self.marker_lock:
            for marker in self.markers:
                # Draw marker circle
                cv2.circle(annotated_frame, 
                          (int(marker.pixel_coords[0]), int(marker.pixel_coords[1])), 
                          8, marker.color, -1)
                
                # Draw marker ID
                cv2.putText(annotated_frame, str(marker.id),
                           (int(marker.pixel_coords[0]) - 15, int(marker.pixel_coords[1]) - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, marker.color, 2)
                
                # Draw world coordinates
                world_text = f"({marker.world_coords[0]:.2f}, {marker.world_coords[1]:.2f}, {marker.world_coords[2]:.2f})"
                cv2.putText(annotated_frame, world_text,
                           (int(marker.pixel_coords[0]) + 15, int(marker.pixel_coords[1])),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, marker.color, 1)
        
        return annotated_frame
    
    def print_instructions(self):
        """Print usage instructions."""
        print("\n" + "="*60)
        print("INTERACTIVE 2D-to-3D MAPPER")
        print("="*60)
        print("Instructions:")
        print("  • Click on the video to place 3D markers")
        print("  • Press 'c' to clear all markers")
        print("  • Press 's' to save markers to file")
        print("  • Press 'l' to load markers from file") 
        print("  • Press 'i' to show calibration info")
        print("  • Press 'h' to show this help")
        print("  • Press 'q' to quit")
        print("="*60)
        
        if self.projection_system is None:
            print("⚠️  WARNING: Projection system not initialized!")
        elif not self.projection_system.is_calibrated():
            print("⚠️  WARNING: System not fully calibrated!")
            info = self.projection_system.get_calibration_info()
            for key, value in info.items():
                print(f"     {key}: {'✓' if value else '✗'}")
        else:
            print("✓ System fully calibrated and ready")
        print()
    
    def clear_markers(self):
        """Clear all markers."""
        with self.marker_lock:
            self.markers.clear()
            self.next_marker_id = 1
        print("All markers cleared")
    
    def save_markers(self, filename: str = None):
        """Save markers to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"markers_{timestamp}.json"
        
        try:
            with self.marker_lock:
                marker_data = {
                    'timestamp': datetime.now().isoformat(),
                    'markers': [marker.to_dict() for marker in self.markers]
                }
            
            with open(filename, 'w') as f:
                json.dump(marker_data, f, indent=2)
            
            print(f"✓ Saved {len(self.markers)} markers to {filename}")
        except Exception as e:
            print(f"Error saving markers: {e}")
    
    def load_markers(self, filename: str = "markers.json"):
        """Load markers from a JSON file."""
        try:
            if not os.path.exists(filename):
                print(f"File {filename} not found")
                return
                
            with open(filename, 'r') as f:
                marker_data = json.load(f)
            
            with self.marker_lock:
                self.markers.clear()
                for marker_dict in marker_data['markers']:
                    marker = Marker.from_dict(marker_dict)
                    self.markers.append(marker)
                
                # Update next_marker_id to avoid conflicts
                if self.markers:
                    self.next_marker_id = max(marker.id for marker in self.markers) + 1
                else:
                    self.next_marker_id = 1
            
            print(f"✓ Loaded {len(self.markers)} markers from {filename}")
        except Exception as e:
            print(f"Error loading markers: {e}")
    
    def show_calibration_info(self):
        """Display calibration information."""
        if self.projection_system is None:
            print("Projection system not initialized")
            return
            
        print("\nCalibration Information:")
        print("-" * 30)
        info = self.projection_system.get_calibration_info()
        for key, value in info.items():
            status = "✓" if value else "✗"
            print(f"{status} {key.replace('_', ' ').title()}")
        print()
    
    def run(self):
        """Run the interactive mapper application."""
        if not self.initialize_camera():
            return
        
        self.print_instructions()
        
        # Start video capture thread
        self.video_thread = threading.Thread(target=self.video_capture_thread)
        self.video_thread.start()
        
        # Set up mouse callback
        cv2.namedWindow('Interactive 2D-to-3D Mapper', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Interactive 2D-to-3D Mapper', self.mouse_callback)
        
        try:
            while self.running:
                # Get current frame
                with self.frame_lock:
                    if self.current_frame is not None:
                        display_frame = self.current_frame.copy()
                    else:
                        # Create a blank frame if no video yet
                        display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(display_frame, "Waiting for video...", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Draw markers
                if self.current_frame is not None:
                    display_frame = self.draw_markers(display_frame)
                
                # Add status info
                status_text = f"Markers: {len(self.markers)}"
                cv2.putText(display_frame, status_text, (10, display_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Interactive 2D-to-3D Mapper', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.clear_markers()
                elif key == ord('s'):
                    self.save_markers()
                elif key == ord('l'):
                    self.load_markers()
                elif key == ord('i'):
                    self.show_calibration_info()
                elif key == ord('h'):
                    self.print_instructions()
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            self.running = False
            
            if self.video_thread:
                self.video_thread.join(timeout=1.0)
            
            if self.cap:
                self.cap.release()
            
            cv2.destroyAllWindows()
            print("Interactive mapper closed")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive 2D-to-3D Mapper")
    parser.add_argument('--source', type=str, help='Video source (RTSP URL, camera index, or video file)')
    parser.add_argument('--config-dir', type=str, help='Directory containing calibration config files')
    args = parser.parse_args()
    
    mapper = InteractiveMapper(rtsp_url=args.source, config_dir=args.config_dir)
    mapper.run()

if __name__ == "__main__":
    main()