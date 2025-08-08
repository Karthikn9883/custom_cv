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
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from camera_visualizer import CameraVisualizer
from performance_optimizer import PerformanceOptimizer

class CameraPlacementTool:
    """
    Interactive tool for placing and adjusting camera position marker in 3D space
    before beginning the 2D-to-3D mapping calibration process.
    """
    
    def __init__(self, rtsp_url: str = None, config_dir: str = None):
        if config_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(script_dir, '..', 'configs')
        
        self.config_dir = config_dir
        self.rtsp_url = rtsp_url
        
        # Camera position and orientation (manual placement)
        self.camera_position = np.array([0.0, 0.0, 2.0])  # Default 2m above origin
        self.camera_rotation = np.array([0.0, 0.0, 0.0])  # Euler angles in degrees
        self.camera_fov_h = 60.0  # Horizontal field of view in degrees
        self.camera_fov_v = 45.0  # Vertical field of view in degrees
        
        # Adjustment parameters
        self.position_step = 0.1  # 10cm steps
        self.rotation_step = 5.0  # 5 degree steps
        self.fov_step = 2.0  # 2 degree FOV steps
        
        # 3D visualization
        self.point_cloud = None
        self.fig = None
        self.ax = None
        
        # Video feed
        self.cap = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.running = True
        
        # Performance optimizer
        self.optimizer = PerformanceOptimizer()
        self.optimizer.apply_all_optimizations()
        
        self.load_lidar_scan()
        self.load_rtsp_config()
    
    def load_rtsp_config(self):
        """Load RTSP URL from config if not provided."""
        if self.rtsp_url is None:
            config_path = os.path.join(self.config_dir, 'main_config.yaml')
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                self.rtsp_url = config.get('input_source', '0')
            except Exception as e:
                print(f"Warning: Could not load RTSP config: {e}")
                self.rtsp_url = '0'
    
    def load_lidar_scan(self):
        """Load the LiDAR scan for 3D visualization."""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            lidar_path = os.path.join(project_root, "8_6_2025.glb")
            
            print(f"Loading LiDAR scan from {lidar_path}...")
            mesh = trimesh.load(lidar_path, force='mesh')
            self.point_cloud = np.array(mesh.vertices)
            print(f"✓ Loaded {len(self.point_cloud)} points for 3D visualization")
        except Exception as e:
            print(f"Error loading LiDAR scan: {e}")
            return False
        return True
    
    def setup_3d_plot(self):
        """Setup the 3D matplotlib plot for camera placement."""
        try:
            plt.ion()  # Interactive mode
            self.fig = plt.figure(figsize=(15, 10))
            self.ax = self.fig.add_subplot(111, projection='3d')
            
            # Plot downsampled point cloud
            downsample_factor = max(1, len(self.point_cloud) // 5000)
            downsampled_pc = self.point_cloud[::downsample_factor]
            
            # Color points based on height
            colors = downsampled_pc[:, 2]
            self.ax.scatter(downsampled_pc[:, 0], downsampled_pc[:, 1], downsampled_pc[:, 2], 
                           c=colors, s=1.5, alpha=0.4, cmap='viridis', label='LiDAR Scan')
            
            print(f"✓ Visualizing {len(downsampled_pc)} points (downsampled from {len(self.point_cloud)})") 
            
            # Set labels and title
            self.ax.set_xlabel('X (meters)')
            self.ax.set_ylabel('Y (meters)')
            self.ax.set_zlabel('Z (meters)')
            self.ax.set_title('Camera Placement Tool - Position Your Camera')
            
            # Set axis limits based on point cloud
            if len(downsampled_pc) > 0:
                margin = 0.5
                self.ax.set_xlim(downsampled_pc[:, 0].min() - margin, downsampled_pc[:, 0].max() + margin)
                self.ax.set_ylim(downsampled_pc[:, 1].min() - margin, downsampled_pc[:, 1].max() + margin)
                self.ax.set_zlim(downsampled_pc[:, 2].min() - margin, downsampled_pc[:, 2].max() + margin)
            
            # Good viewing angle
            self.ax.view_init(elev=20, azim=45)
            
            # Update camera visualization
            self.update_camera_visualization()
            
            plt.show(block=False)
            return True
            
        except Exception as e:
            print(f"Error setting up 3D plot: {e}")
            return False
    
    def euler_to_rotation_matrix(self, euler_angles):
        """Convert Euler angles (in degrees) to rotation matrix."""
        angles_rad = np.radians(euler_angles)
        rx, ry, rz = angles_rad
        
        # Rotation matrices for each axis
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        
        # Combined rotation (ZYX order)
        R = Rz @ Ry @ Rx
        return R
    
    def generate_camera_frustum(self):
        """Generate camera frustum based on current position and orientation."""
        R = self.euler_to_rotation_matrix(self.camera_rotation)
        
        # Camera coordinate system directions
        forward = R @ np.array([0, 0, 1])  # Camera looks along +Z
        up = R @ np.array([0, -1, 0])     # Camera up is -Y
        right = R @ np.array([1, 0, 0])   # Camera right is +X
        
        # FOV calculations
        h_fov_rad = np.radians(self.camera_fov_h)
        v_fov_rad = np.radians(self.camera_fov_v)
        
        # Frustum parameters
        near_dist = 0.1
        far_dist = 3.0
        
        # Calculate corner offsets
        near_half_width = near_dist * np.tan(h_fov_rad / 2)
        near_half_height = near_dist * np.tan(v_fov_rad / 2)
        far_half_width = far_dist * np.tan(h_fov_rad / 2)
        far_half_height = far_dist * np.tan(v_fov_rad / 2)
        
        # Generate frustum corners
        corners = []
        
        # Near plane
        near_center = self.camera_position + forward * near_dist
        corners.extend([
            near_center + right * near_half_width + up * near_half_height,
            near_center - right * near_half_width + up * near_half_height,
            near_center - right * near_half_width - up * near_half_height,
            near_center + right * near_half_width - up * near_half_height,
        ])
        
        # Far plane
        far_center = self.camera_position + forward * far_dist
        corners.extend([
            far_center + right * far_half_width + up * far_half_height,
            far_center - right * far_half_width + up * far_half_height,
            far_center - right * far_half_width - up * far_half_height,
            far_center + right * far_half_width - up * far_half_height,
        ])
        
        return np.array(corners), forward, up, right
    
    def update_camera_visualization(self):
        """Update the camera visualization in the 3D plot."""
        if self.ax is None:
            return
        
        # Remove previous camera visualization
        artists_to_remove = []
        for artist in self.ax.collections:
            if hasattr(artist, '_camera_placement'):
                artists_to_remove.append(artist)
        for artist in artists_to_remove:
            artist.remove()
        
        # Remove previous lines
        lines_to_remove = []
        for line in self.ax.lines:
            if hasattr(line, '_camera_placement'):
                lines_to_remove.append(line)
        for line in lines_to_remove:
            line.remove()
        
        # Remove text
        texts_to_remove = []
        for text in self.ax.texts:
            if hasattr(text, '_camera_placement'):
                texts_to_remove.append(text)
        for text in texts_to_remove:
            text.remove()
        
        # Add camera position
        scatter = self.ax.scatter(self.camera_position[0], self.camera_position[1], self.camera_position[2], 
                                c='red', s=200, marker='o', edgecolors='black', linewidth=2, 
                                label='Camera Position')
        scatter._camera_placement = True
        
        # Add camera orientation and frustum
        corners, forward, up, right = self.generate_camera_frustum()
        
        # Draw direction vector
        direction_end = self.camera_position + forward * 1.0
        line = self.ax.plot([self.camera_position[0], direction_end[0]], 
                           [self.camera_position[1], direction_end[1]], 
                           [self.camera_position[2], direction_end[2]], 
                           'r-', linewidth=3, label='Camera Direction')[0]
        line._camera_placement = True
        
        # Draw frustum edges
        frustum_color = 'orange'
        alpha = 0.6
        
        # Near plane edges
        near_edges = [(0,1), (1,2), (2,3), (3,0)]
        for i, j in near_edges:
            line = self.ax.plot([corners[i][0], corners[j][0]], 
                               [corners[i][1], corners[j][1]], 
                               [corners[i][2], corners[j][2]], 
                               color=frustum_color, alpha=alpha)[0]
            line._camera_placement = True
        
        # Far plane edges
        far_edges = [(4,5), (5,6), (6,7), (7,4)]
        for i, j in far_edges:
            line = self.ax.plot([corners[i][0], corners[j][0]], 
                               [corners[i][1], corners[j][1]], 
                               [corners[i][2], corners[j][2]], 
                               color=frustum_color, alpha=alpha)[0]
            line._camera_placement = True
        
        # Connecting edges (from near to far)
        connecting_edges = [(0,4), (1,5), (2,6), (3,7)]
        for i, j in connecting_edges:
            line = self.ax.plot([corners[i][0], corners[j][0]], 
                               [corners[i][1], corners[j][1]], 
                               [corners[i][2], corners[j][2]], 
                               color=frustum_color, alpha=alpha)[0]
            line._camera_placement = True
        
        # Add position and rotation text
        info_text = (f"Pos: ({self.camera_position[0]:.2f}, {self.camera_position[1]:.2f}, {self.camera_position[2]:.2f})\\n"
                    f"Rot: ({self.camera_rotation[0]:.1f}°, {self.camera_rotation[1]:.1f}°, {self.camera_rotation[2]:.1f}°)\\n"
                    f"FOV: {self.camera_fov_h:.1f}° × {self.camera_fov_v:.1f}°")
        
        text = self.ax.text2D(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                             fontsize=10, verticalalignment='top', 
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        text._camera_placement = True
        
        # Refresh the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def initialize_video_capture(self):
        """Initialize video capture with optimization."""
        try:
            if self.rtsp_url.isdigit():
                self.cap = cv2.VideoCapture(int(self.rtsp_url))
            else:
                self.cap = self.optimizer.create_optimized_video_capture(self.rtsp_url)
            
            return self.cap.isOpened()
        except Exception as e:
            print(f"Error initializing video: {e}")
            return False
    
    def video_capture_thread(self):
        """Thread for continuous video capture."""
        while self.running:
            if self.cap is None:
                time.sleep(0.1)
                continue
            
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame.copy()
            else:
                time.sleep(0.1)
    
    def save_camera_placement(self, filename: str = None):
        """Save current camera placement settings."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_placement_{timestamp}.yaml"
        
        placement_data = {
            'camera_placement': {
                'position': self.camera_position.tolist(),
                'rotation_euler_degrees': self.camera_rotation.tolist(),
                'field_of_view': {
                    'horizontal_degrees': float(self.camera_fov_h),
                    'vertical_degrees': float(self.camera_fov_v)
                }
            },
            'timestamp': datetime.now().isoformat(),
            'notes': 'Manual camera placement for 2D-3D mapping calibration'
        }
        
        try:
            with open(filename, 'w') as f:
                yaml.dump(placement_data, f, default_flow_style=False, indent=2)
            print(f"\\n✓ Camera placement saved to: {filename}")
        except Exception as e:
            print(f"Error saving camera placement: {e}")
    
    def load_camera_placement(self, filename: str = "camera_placement.yaml"):
        """Load camera placement settings."""
        try:
            if not os.path.exists(filename):
                print(f"File {filename} not found")
                return False
            
            with open(filename, 'r') as f:
                data = yaml.safe_load(f)
            
            placement = data['camera_placement']
            self.camera_position = np.array(placement['position'])
            self.camera_rotation = np.array(placement['rotation_euler_degrees'])
            self.camera_fov_h = placement['field_of_view']['horizontal_degrees']
            self.camera_fov_v = placement['field_of_view']['vertical_degrees']
            
            print(f"\\n✓ Camera placement loaded from: {filename}")
            self.update_camera_visualization()
            return True
            
        except Exception as e:
            print(f"Error loading camera placement: {e}")
            return False
    
    def print_instructions(self):
        """Print usage instructions."""
        print(f"\\n{'='*70}")
        print("CAMERA PLACEMENT TOOL")
        print(f"{'='*70}")
        print("POSITION CONTROLS:")
        print("  W/S - Move forward/backward")
        print("  A/D - Move left/right") 
        print("  Q/E - Move up/down")
        print("\\nROTATION CONTROLS:")
        print("  I/K - Pitch up/down")
        print("  J/L - Yaw left/right")
        print("  U/O - Roll left/right")
        print("\\nFIELD OF VIEW:")
        print("  +/- - Increase/decrease horizontal FOV")
        print("  [/] - Increase/decrease vertical FOV")
        print("\\nOTHER CONTROLS:")
        print("  R - Reset to default position")
        print("  F - Fine adjustment mode (smaller steps)")
        print("  C - Coarse adjustment mode (larger steps)")
        print("  V - Save camera placement")
        print("  B - Load camera placement")
        print("  H - Show this help")
        print("  ESC - Quit")
        print(f"{'='*70}\\n")
    
    def run(self):
        """Run the camera placement tool."""
        print("\\nInitializing Camera Placement Tool...")
        
        # Setup 3D visualization
        if not self.setup_3d_plot():
            print("Failed to setup 3D visualization")
            return
        
        # Initialize video capture
        if not self.initialize_video_capture():
            print("Warning: Could not initialize video capture")
        else:
            # Start video thread
            video_thread = threading.Thread(target=self.video_capture_thread)
            video_thread.start()
        
        self.print_instructions()
        
        print("Camera placement tool ready!")
        print("Use keyboard controls to position the camera, then save when satisfied.")
        
        try:
            while self.running:
                # Show video feed if available
                if self.current_frame is not None:
                    with self.frame_lock:
                        display_frame = self.current_frame.copy()
                    
                    # Add overlay info
                    cv2.putText(display_frame, "Camera Placement Mode", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Pos: {self.camera_position}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(display_frame, f"Rot: {self.camera_rotation}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow('Camera Feed - Placement Mode', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(50) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord('w'):  # Forward
                    R = self.euler_to_rotation_matrix(self.camera_rotation)
                    self.camera_position += (R @ np.array([0, 0, self.position_step]))
                elif key == ord('s'):  # Backward
                    R = self.euler_to_rotation_matrix(self.camera_rotation)
                    self.camera_position -= (R @ np.array([0, 0, self.position_step]))
                elif key == ord('a'):  # Left
                    R = self.euler_to_rotation_matrix(self.camera_rotation)
                    self.camera_position -= (R @ np.array([self.position_step, 0, 0]))
                elif key == ord('d'):  # Right
                    R = self.euler_to_rotation_matrix(self.camera_rotation)
                    self.camera_position += (R @ np.array([self.position_step, 0, 0]))
                elif key == ord('q'):  # Up
                    self.camera_position[2] += self.position_step
                elif key == ord('e'):  # Down
                    self.camera_position[2] -= self.position_step
                elif key == ord('i'):  # Pitch up
                    self.camera_rotation[0] += self.rotation_step
                elif key == ord('k'):  # Pitch down
                    self.camera_rotation[0] -= self.rotation_step
                elif key == ord('j'):  # Yaw left
                    self.camera_rotation[1] -= self.rotation_step
                elif key == ord('l'):  # Yaw right
                    self.camera_rotation[1] += self.rotation_step
                elif key == ord('u'):  # Roll left
                    self.camera_rotation[2] -= self.rotation_step
                elif key == ord('o'):  # Roll right
                    self.camera_rotation[2] += self.rotation_step
                elif key == ord('+') or key == ord('='):  # Increase H FOV
                    self.camera_fov_h = min(120, self.camera_fov_h + self.fov_step)
                elif key == ord('-'):  # Decrease H FOV
                    self.camera_fov_h = max(10, self.camera_fov_h - self.fov_step)
                elif key == ord('['):  # Increase V FOV
                    self.camera_fov_v = min(90, self.camera_fov_v + self.fov_step)
                elif key == ord(']'):  # Decrease V FOV
                    self.camera_fov_v = max(10, self.camera_fov_v - self.fov_step)
                elif key == ord('r'):  # Reset
                    self.camera_position = np.array([0.0, 0.0, 2.0])
                    self.camera_rotation = np.array([0.0, 0.0, 0.0])
                    self.camera_fov_h = 60.0
                    self.camera_fov_v = 45.0
                elif key == ord('f'):  # Fine mode
                    self.position_step = 0.05
                    self.rotation_step = 1.0
                    self.fov_step = 1.0
                    print("Fine adjustment mode")
                elif key == ord('c'):  # Coarse mode
                    self.position_step = 0.2
                    self.rotation_step = 10.0
                    self.fov_step = 5.0
                    print("Coarse adjustment mode")
                elif key == ord('v'):  # Save
                    self.save_camera_placement()
                elif key == ord('b'):  # Load
                    self.load_camera_placement()
                elif key == ord('h'):  # Help
                    self.print_instructions()
                
                # Update visualization if any key was pressed
                if key != 255:
                    self.update_camera_visualization()
                
        except KeyboardInterrupt:
            print("\\nInterrupted by user")
        
        finally:
            self.running = False
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            if self.fig:
                plt.close(self.fig)
            print("\\nCamera placement tool closed")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Camera Placement Tool for 2D-3D Mapping")
    parser.add_argument('--source', type=str, help='Video source (RTSP URL, camera index, or video file)')
    parser.add_argument('--config-dir', type=str, help='Directory containing calibration config files')
    args = parser.parse_args()
    
    tool = CameraPlacementTool(rtsp_url=args.source, config_dir=args.config_dir)
    tool.run()

if __name__ == "__main__":
    main()