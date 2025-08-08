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
import queue
import trimesh

# Configure matplotlib backend before importing pyplot
import matplotlib

# Try different backends in order of preference with actual figure creation test
backends_to_try = ['Qt5Agg', 'MacOSX', 'TkAgg', 'Agg']
backend_set = False

for backend in backends_to_try:
    try:
        matplotlib.use(backend, force=True)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Test actual figure creation - this is where some backends fail
        test_fig = plt.figure(figsize=(1, 1))
        test_fig.add_subplot(111, projection='3d')
        plt.close(test_fig)
        
        print(f"Using matplotlib backend: {backend}")
        backend_set = True
        break
    except ImportError as e:
        print(f"Backend {backend} not available: {e}")
        continue
    except Exception as e:
        print(f"Backend {backend} figure creation failed: {e}")
        continue

if not backend_set:
    print("Warning: No interactive matplotlib backend available. 3D visualization will be disabled.")
    # Use Agg backend as fallback (non-interactive)
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

# Add the src directory to the path to import projection_utils
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from projection_utils import ProjectionSystem
from camera_visualizer import CameraVisualizer
from performance_optimizer import PerformanceOptimizer

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

class InteractiveMapper3D:
    def __init__(self, rtsp_url: str = None, config_dir: str = None, enable_3d: bool = True, 
                 show_camera: bool = True, enable_optimizations: bool = True):
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
        self.enable_3d = enable_3d
        self.show_camera = show_camera
        self.enable_optimizations = enable_optimizations
        
        # Initialize performance optimizer
        self.optimizer = None
        if self.enable_optimizations:
            try:
                self.optimizer = PerformanceOptimizer()
                self.optimizer.apply_all_optimizations()
                print("✓ Performance optimizations applied")
            except Exception as e:
                print(f"Warning: Performance optimizer failed: {e}")
        
        # Initialize camera visualizer
        self.camera_visualizer = None
        if self.show_camera:
            try:
                self.camera_visualizer = CameraVisualizer(config_dir)
                print("✓ Camera visualizer initialized")
            except Exception as e:
                print(f"Warning: Camera visualizer failed to initialize: {e}")
                self.show_camera = False
        
        # Video capture
        self.cap = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # 3D Visualization
        self.point_cloud = None
        self.fig = None
        self.ax = None
        self.plot_update_queue = queue.Queue()
        
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
        
        # Load LiDAR scan for 3D visualization
        if self.enable_3d:
            self.load_lidar_scan()
    
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
            print(f"Warning: Could not load LiDAR scan: {e}")
            self.enable_3d = False
    
    def initialize_camera(self) -> bool:
        """Initialize the camera/video source with optimizations."""
        try:
            # Handle different source types
            if self.rtsp_url.isdigit():
                source = int(self.rtsp_url)
                # Standard VideoCapture for camera indices
                self.cap = cv2.VideoCapture(source)
            else:
                # Use optimized RTSP capture if optimizer is available
                if self.optimizer and 'rtsp://' in self.rtsp_url.lower():
                    self.cap = self.optimizer.create_optimized_video_capture(self.rtsp_url)
                else:
                    self.cap = cv2.VideoCapture(self.rtsp_url)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open video source: {self.rtsp_url}")
                return False
            
            # Apply additional optimizations if available
            if not self.optimizer:
                # Fallback optimizations
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Get video properties for information
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"✓ Video source opened successfully: {self.rtsp_url}")
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {fps:.1f}" if fps > 0 else "  FPS: Unknown")
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
                
                # Queue 3D plot update
                self.plot_update_queue.put('update')
            else:
                print(f"Could not project pixel ({x}, {y}) to 3D coordinates")
    
    def setup_3d_plot(self):
        """Setup the 3D matplotlib plot - must be called from main thread."""
        if not self.enable_3d or self.point_cloud is None:
            return False
            
        try:
            # Check if we have an interactive backend
            current_backend = matplotlib.get_backend()
            self.is_interactive_backend = current_backend not in ['Agg', 'svg', 'pdf', 'ps']
            
            if self.is_interactive_backend:
                plt.ion()  # Enable interactive mode
            
            self.fig = plt.figure(figsize=(12, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            
            # Plot downsampled point cloud with better visibility
            downsample_factor = max(1, len(self.point_cloud) // 15000)  # Less aggressive downsampling
            self.downsampled_pc = self.point_cloud[::downsample_factor]
            
            # Color points based on height (Z coordinate) for better visualization
            colors = self.downsampled_pc[:, 2]  # Use Z coordinate for coloring
            
            self.ax.scatter(self.downsampled_pc[:, 0], self.downsampled_pc[:, 1], self.downsampled_pc[:, 2], 
                           c=colors, s=2.0, alpha=0.6, cmap='viridis', label='LiDAR Scan')
            
            # Add camera visualization
            if self.show_camera and self.camera_visualizer:
                try:
                    self.camera_visualizer.add_camera_to_plot(self.ax, show_frustum=True, show_body=True, 
                                                             show_direction=True, alpha=0.2)
                    print("✓ Camera visualization added to 3D plot")
                except Exception as e:
                    print(f"Warning: Could not add camera visualization: {e}")
            
            print(f"✓ Visualizing {len(self.downsampled_pc)} points (downsampled from {len(self.point_cloud)})")
            
            self.ax.set_xlabel('X (meters)')
            self.ax.set_ylabel('Y (meters)')
            self.ax.set_zlabel('Z (meters)')
            self.ax.set_title('3D LiDAR Scan with Real-time Markers')
            
            # Set better axis limits based on actual data
            if len(self.downsampled_pc) > 0:
                margin = 0.1  # 10cm margin
                self.ax.set_xlim(self.downsampled_pc[:, 0].min() - margin, self.downsampled_pc[:, 0].max() + margin)
                self.ax.set_ylim(self.downsampled_pc[:, 1].min() - margin, self.downsampled_pc[:, 1].max() + margin)
                self.ax.set_zlim(self.downsampled_pc[:, 2].min() - margin, self.downsampled_pc[:, 2].max() + margin)
            
            # Set a good initial viewing angle (looking down at slight angle)
            self.ax.view_init(elev=20, azim=45)
            
            if self.is_interactive_backend:
                plt.show(block=False)
                print("✓ Interactive 3D visualization window opened")
            else:
                print("✓ Static 3D visualization mode (images will be saved)")
                self.save_3d_plot()  # Save initial plot
                
            return True
        except Exception as e:
            print(f"Error setting up 3D plot: {e}")
            self.enable_3d = False
            return False
    
    def save_3d_plot(self, filename: str = None):
        """Save the current 3D plot as an image."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"3d_markers_{timestamp}.png"
        
        try:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"3D plot saved as: {filename}")
        except Exception as e:
            print(f"Error saving 3D plot: {e}")
    
    def update_3d_plot(self):
        """Update the 3D plot with current markers and camera visualization."""
        if not self.enable_3d or self.ax is None:
            return
            
        try:
            # Clear previous markers and camera visualization (keep point cloud)
            # Find and remove marker artists and camera visualization
            artists_to_remove = []
            for artist in self.ax.collections:
                if (hasattr(artist, '_3d_marker') or 
                    artist.get_label().startswith('Marker') or
                    artist.get_label().startswith('Camera')):
                    artists_to_remove.append(artist)
            for artist in artists_to_remove:
                artist.remove()
            
            # Clear text labels and camera-related elements
            texts_to_remove = []
            for text in self.ax.texts:
                if hasattr(text, '_3d_marker') or hasattr(text, '_camera_viz'):
                    texts_to_remove.append(text)
            for text in texts_to_remove:
                text.remove()
            
            # Remove camera-related line plots
            lines_to_remove = []
            for line in self.ax.lines:
                if hasattr(line, '_camera_viz'):
                    lines_to_remove.append(line)
            for line in lines_to_remove:
                line.remove()
            
            # Add camera visualization if enabled
            if self.show_camera and self.camera_visualizer:
                try:
                    # Temporarily store current alpha for better performance
                    temp_camera_viz = CameraVisualizer(self.camera_visualizer.config_dir)
                    temp_camera_viz.add_camera_to_plot(self.ax, show_frustum=True, show_body=True, 
                                                     show_direction=True, alpha=0.2)
                    # Mark camera elements for easy removal
                    for artist in self.ax.collections[-3:]:
                        if not hasattr(artist, '_3d_marker'):
                            artist._camera_viz = True
                    for line in self.ax.lines[-1:]:
                        line._camera_viz = True
                except Exception as e:
                    print(f"Warning: Could not update camera visualization: {e}")
            
            # Add current markers
            with self.marker_lock:
                for marker in self.markers:
                    world_coords = marker.world_coords
                    color = [c/255.0 for c in marker.color]
                    
                    scatter = self.ax.scatter(world_coords[0], world_coords[1], world_coords[2], 
                              c=[color], s=200, alpha=0.9, marker='o', 
                              edgecolors='black', linewidth=2,
                              label=f'Marker {marker.id}')
                    scatter._3d_marker = True  # Mark for easy removal
                    
                    # Add text label
                    text = self.ax.text(world_coords[0], world_coords[1], world_coords[2] + 0.1, 
                               f'M{marker.id}', fontsize=8, ha='center')
                    text._3d_marker = True  # Mark for easy removal
            
            # Update display based on backend type
            if hasattr(self, 'is_interactive_backend') and self.is_interactive_backend:
                # Interactive mode - update the window
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            else:
                # Static mode - save updated image
                self.save_3d_plot()
            
        except Exception as e:
            print(f"Error updating 3D plot: {e}")
    
    def check_3d_updates(self):
        """Check for 3D plot update requests from main thread."""
        if not self.enable_3d or self.ax is None:
            return
            
        # Process all pending updates
        updates_processed = 0
        try:
            while not self.plot_update_queue.empty():
                self.plot_update_queue.get_nowait()
                updates_processed += 1
        except queue.Empty:
            pass
            
        # Update plot if there were any updates
        if updates_processed > 0:
            self.update_3d_plot()
    
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
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, marker.color, 1)
        
        return annotated_frame
    
    def print_instructions(self):
        """Print usage instructions."""
        print("\n" + "="*60)
        print("INTERACTIVE 2D-to-3D MAPPER WITH 3D VISUALIZATION")
        print("="*60)
        print("Instructions:")
        print("  • Click on the video to place 3D markers")
        print("  • Markers appear instantly in both 2D video and 3D LiDAR view")
        print("  • Press 'c' to clear all markers")
        print("  • Press 's' to save markers to file")
        print("  • Press 'p' to save 3D plot as image")
        print("  • Press 'l' to load markers from file") 
        print("  • Press 'i' to show calibration info")
        print("  • Press 'v' to show validation statistics")
        print("  • Press 'k' to show camera position info")
        print("  • Press 't' to toggle camera visualization")
        print("  • Press 'd' to measure distance between last two markers")
        print("  • Press 'o' to show performance/system info")
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
        
        if self.enable_3d:
            print("✓ 3D visualization enabled")
            if self.show_camera and self.camera_visualizer:
                print("✓ Camera visualization enabled")
            else:
                print("⚠️  Camera visualization disabled")
        else:
            print("⚠️  3D visualization disabled")
        print()
    
    def clear_markers(self):
        """Clear all markers."""
        with self.marker_lock:
            self.markers.clear()
            self.next_marker_id = 1
        self.plot_update_queue.put('update')  # Update 3D plot
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
            
            self.plot_update_queue.put('update')  # Update 3D plot
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
    
    def show_validation_stats(self):
        """Show marker validation statistics."""
        if not self.markers:
            print("No markers to analyze")
            return
        
        print(f"\n{'='*40}")
        print("MARKER VALIDATION STATISTICS")
        print(f"{'='*40}")
        print(f"Total markers: {len(self.markers)}")
        
        # Extract world coordinates
        world_coords = np.array([marker.world_coords for marker in self.markers])
        
        # Calculate bounds
        bounds = {
            'x': (world_coords[:, 0].min(), world_coords[:, 0].max()),
            'y': (world_coords[:, 1].min(), world_coords[:, 1].max()),
            'z': (world_coords[:, 2].min(), world_coords[:, 2].max())
        }
        
        print("\nSpatial distribution:")
        for axis, (min_val, max_val) in bounds.items():
            range_val = max_val - min_val
            print(f"  {axis}: {min_val:.3f} to {max_val:.3f} meters (range: {range_val:.3f}m)")
        
        # Calculate center point and spread
        center = world_coords.mean(axis=0)
        std_dev = world_coords.std(axis=0)
        print(f"\nCentroid: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) meters")
        print(f"Std dev:  ({std_dev[0]:.3f}, {std_dev[1]:.3f}, {std_dev[2]:.3f}) meters")
        
        # Individual marker details
        print(f"\nMarker details:")
        for marker in self.markers:
            coords = marker.world_coords
            pixel = marker.pixel_coords
            print(f"  M{marker.id:2d}: Pixel({pixel[0]:4.0f},{pixel[1]:4.0f}) → World({coords[0]:6.3f},{coords[1]:6.3f},{coords[2]:6.3f})")
        print()
    
    def show_camera_info(self):
        """Display detailed camera position and orientation information."""
        if not self.show_camera or not self.camera_visualizer:
            print("Camera visualization not available")
            return
            
        self.camera_visualizer.print_camera_info()
    
    def toggle_camera_visualization(self):
        """Toggle camera visualization on/off."""
        if not self.enable_3d:
            print("3D visualization is disabled")
            return
            
        if not self.camera_visualizer:
            print("Camera visualizer not initialized")
            return
            
        self.show_camera = not self.show_camera
        self.plot_update_queue.put('update')  # Update 3D plot
        print(f"Camera visualization: {'ON' if self.show_camera else 'OFF'}")
    
    def measure_distance_between_markers(self):
        """Measure distance between the last two markers placed."""
        with self.marker_lock:
            if len(self.markers) < 2:
                print("Need at least 2 markers to measure distance")
                return
            
            marker1 = self.markers[-2]  # Second to last
            marker2 = self.markers[-1]  # Last
            
            # Calculate 3D distance
            distance_3d = np.linalg.norm(marker2.world_coords - marker1.world_coords)
            
            # Calculate 2D pixel distance 
            pixel_dist = np.sqrt((marker2.pixel_coords[0] - marker1.pixel_coords[0])**2 + 
                               (marker2.pixel_coords[1] - marker1.pixel_coords[1])**2)
            
            print(f"\n{'='*50}")
            print("DISTANCE MEASUREMENT")
            print(f"{'='*50}")
            print(f"Between Marker {marker1.id} and Marker {marker2.id}:")
            print(f"  3D World Distance: {distance_3d:.3f} meters")
            print(f"  2D Pixel Distance: {pixel_dist:.1f} pixels")
            print(f"  → Scale: {distance_3d/pixel_dist*1000:.2f} mm per pixel")
            print(f"{'='*50}\n")
    
    def show_performance_info(self):
        """Display performance and system information."""
        if self.optimizer:
            self.optimizer.print_system_summary()
        else:
            print("\n⚠️  Performance optimizer not initialized")
            print("Use --enable-optimizations flag to enable performance monitoring")
    
    def run(self):
        """Run the interactive mapper application."""
        if not self.initialize_camera():
            return
        
        self.print_instructions()
        
        # Start video capture thread
        self.video_thread = threading.Thread(target=self.video_capture_thread)
        self.video_thread.start()
        
        # Setup 3D plot if enabled (must be done in main thread)
        if self.enable_3d:
            if not self.setup_3d_plot():
                print("3D visualization disabled due to setup failure")
        
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
                status_text = f"Markers: {len(self.markers)} | 3D View: {'ON' if self.enable_3d else 'OFF'}"
                cv2.putText(display_frame, status_text, (10, display_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Interactive 2D-to-3D Mapper', display_frame)
                
                # Check for 3D plot updates (must be done in main thread)
                self.check_3d_updates()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.clear_markers()
                elif key == ord('s'):
                    self.save_markers()
                elif key == ord('p'):
                    if self.enable_3d and self.fig is not None:
                        self.save_3d_plot()
                    else:
                        print("3D visualization not available")
                elif key == ord('l'):
                    self.load_markers()
                elif key == ord('i'):
                    self.show_calibration_info()
                elif key == ord('v'):
                    self.show_validation_stats()
                elif key == ord('k'):
                    self.show_camera_info()
                elif key == ord('t'):
                    self.toggle_camera_visualization()
                elif key == ord('d'):
                    self.measure_distance_between_markers()
                elif key == ord('o'):
                    self.show_performance_info()
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
            
            if self.fig:
                plt.close(self.fig)
            
            print("Interactive mapper closed")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive 2D-to-3D Mapper with 3D Visualization")
    parser.add_argument('--source', type=str, help='Video source (RTSP URL, camera index, or video file)')
    parser.add_argument('--config-dir', type=str, help='Directory containing calibration config files')
    parser.add_argument('--no-3d', action='store_true', help='Disable 3D visualization')
    parser.add_argument('--no-camera', action='store_true', help='Disable camera position visualization')
    parser.add_argument('--no-optimizations', action='store_true', help='Disable performance optimizations')
    args = parser.parse_args()
    
    mapper = InteractiveMapper3D(rtsp_url=args.source, config_dir=args.config_dir, 
                                enable_3d=not args.no_3d, show_camera=not args.no_camera,
                                enable_optimizations=not args.no_optimizations)
    mapper.run()

if __name__ == "__main__":
    main()