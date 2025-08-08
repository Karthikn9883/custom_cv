import numpy as np
import json
import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Import trimesh for GLB loading (without open3d dependency for now)
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add the src directory to import projection_utils
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

class Marker3DVisualizer:
    def __init__(self, lidar_scan_path: str = None, marker_file: str = None):
        # Set default paths
        if lidar_scan_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            lidar_scan_path = os.path.join(project_root, "8_6_2025.glb")
        
        self.lidar_scan_path = lidar_scan_path
        self.marker_file = marker_file
        
        # Load data
        self.point_cloud = None
        self.markers = []
        self.validation_points = []  # Ground truth points for validation
        
        self.load_lidar_scan()
        if marker_file and os.path.exists(marker_file):
            self.load_markers()
    
    def load_lidar_scan(self):
        """Load the LiDAR scan from GLB file."""
        try:
            print(f"Loading LiDAR scan from {self.lidar_scan_path}...")
            mesh = trimesh.load(self.lidar_scan_path, force='mesh')
            
            # Extract vertices as point cloud
            self.point_cloud = np.array(mesh.vertices)
            print(f"✓ Loaded {len(self.point_cloud)} points from LiDAR scan")
            
            # Print point cloud bounds for reference
            bounds = {
                'x': (self.point_cloud[:, 0].min(), self.point_cloud[:, 0].max()),
                'y': (self.point_cloud[:, 1].min(), self.point_cloud[:, 1].max()),
                'z': (self.point_cloud[:, 2].min(), self.point_cloud[:, 2].max())
            }
            print("Point cloud bounds:")
            for axis, (min_val, max_val) in bounds.items():
                print(f"  {axis}: {min_val:.3f} to {max_val:.3f} meters")
                
        except Exception as e:
            print(f"Error loading LiDAR scan: {e}")
            self.point_cloud = None
    
    def load_markers(self):
        """Load markers from JSON file."""
        try:
            with open(self.marker_file, 'r') as f:
                marker_data = json.load(f)
            
            self.markers = []
            for marker_dict in marker_data['markers']:
                self.markers.append({
                    'id': marker_dict['id'],
                    'pixel_coords': tuple(marker_dict['pixel_coords']),
                    'world_coords': np.array(marker_dict['world_coords']),
                    'color': tuple(marker_dict['color']),
                    'timestamp': marker_dict['timestamp']
                })
            
            print(f"✓ Loaded {len(self.markers)} markers from {self.marker_file}")
            
        except Exception as e:
            print(f"Error loading markers: {e}")
            self.markers = []
    
    def visualize_matplotlib(self):
        """Visualize using matplotlib (fallback when open3d not available)."""
        if self.point_cloud is None:
            print("No point cloud data to display")
            return
        
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Downsample point cloud for performance (every 10th point)
        downsample_factor = max(1, len(self.point_cloud) // 50000)
        downsampled_pc = self.point_cloud[::downsample_factor]
        
        # Plot point cloud
        ax.scatter(downsampled_pc[:, 0], downsampled_pc[:, 1], downsampled_pc[:, 2], 
                  c='lightgray', s=0.1, alpha=0.3, label='LiDAR Scan')
        
        # Plot markers
        if self.markers:
            for marker in self.markers:
                world_coords = marker['world_coords']
                color = [c/255.0 for c in marker['color']]  # Convert to matplotlib color format
                
                ax.scatter(world_coords[0], world_coords[1], world_coords[2], 
                          c=[color], s=200, alpha=0.9, marker='o', 
                          edgecolors='black', linewidth=2,
                          label=f'Marker {marker["id"]}' if marker["id"] <= 3 else "")
                
                # Add text label
                ax.text(world_coords[0], world_coords[1], world_coords[2] + 0.1, 
                       f'M{marker["id"]}', fontsize=10, ha='center')
        
        # Plot validation points if any
        if self.validation_points:
            for i, point in enumerate(self.validation_points):
                ax.scatter(point[0], point[1], point[2], 
                          c='red', s=300, alpha=0.9, marker='s',
                          edgecolors='darkred', linewidth=2,
                          label='Ground Truth' if i == 0 else "")
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('3D LiDAR Scan with Projected Markers')
        
        # Set equal aspect ratio
        max_range = np.array([
            downsampled_pc[:, 0].max() - downsampled_pc[:, 0].min(),
            downsampled_pc[:, 1].max() - downsampled_pc[:, 1].min(),
            downsampled_pc[:, 2].max() - downsampled_pc[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (downsampled_pc[:, 0].max() + downsampled_pc[:, 0].min()) * 0.5
        mid_y = (downsampled_pc[:, 1].max() + downsampled_pc[:, 1].min()) * 0.5
        mid_z = (downsampled_pc[:, 2].max() + downsampled_pc[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if self.markers or self.validation_points:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def try_visualize_open3d(self):
        """Try to visualize using open3d if available."""
        try:
            import open3d as o3d
            
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
            pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
            
            geometries = [pcd]
            
            # Add markers as spheres
            if self.markers:
                for marker in self.markers:
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                    color = [c/255.0 for c in marker['color']]
                    sphere.paint_uniform_color(color)
                    sphere.translate(marker['world_coords'])
                    geometries.append(sphere)
            
            # Add validation points as cubes
            if self.validation_points:
                for point in self.validation_points:
                    cube = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.1, depth=0.1)
                    cube.paint_uniform_color([1.0, 0.0, 0.0])  # Red
                    cube.translate(point - 0.05)  # Center the cube
                    geometries.append(cube)
            
            print("\nOpen3D Visualization Controls:")
            print("  • Mouse: Rotate view")
            print("  • Wheel: Zoom in/out") 
            print("  • Shift+Mouse: Pan")
            print("  • Press 'H' for more help")
            print("  • Close window to continue")
            
            o3d.visualization.draw_geometries(geometries, 
                                            window_name="3D LiDAR Scan with Projected Markers",
                                            width=1200, height=800)
            return True
            
        except ImportError:
            print("Open3D not available, falling back to matplotlib...")
            return False
        except Exception as e:
            print(f"Error with Open3D visualization: {e}")
            return False
    
    def calculate_marker_statistics(self):
        """Calculate and display marker statistics."""
        if not self.markers:
            print("No markers to analyze")
            return
        
        print(f"\n{'='*50}")
        print("MARKER ANALYSIS")
        print(f"{'='*50}")
        print(f"Total markers: {len(self.markers)}")
        
        # Extract world coordinates
        world_coords = np.array([marker['world_coords'] for marker in self.markers])
        
        # Calculate bounds
        bounds = {
            'x': (world_coords[:, 0].min(), world_coords[:, 0].max()),
            'y': (world_coords[:, 1].min(), world_coords[:, 1].max()),
            'z': (world_coords[:, 2].min(), world_coords[:, 2].max())
        }
        
        print("\nMarker distribution:")
        for axis, (min_val, max_val) in bounds.items():
            range_val = max_val - min_val
            print(f"  {axis}: {min_val:.3f} to {max_val:.3f} meters (range: {range_val:.3f}m)")
        
        # Calculate center point
        center = world_coords.mean(axis=0)
        print(f"\nMarker centroid: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) meters")
        
        # Calculate distances from origin
        distances = np.linalg.norm(world_coords, axis=1)
        print(f"\nDistance from origin:")
        print(f"  Mean: {distances.mean():.3f} meters")
        print(f"  Min:  {distances.min():.3f} meters (Marker {self.markers[distances.argmin()]['id']})")
        print(f"  Max:  {distances.max():.3f} meters (Marker {self.markers[distances.argmax()]['id']})")
        
        # Show individual markers
        print(f"\nIndividual markers:")
        for marker in self.markers:
            coords = marker['world_coords']
            pixel = marker['pixel_coords']
            print(f"  Marker {marker['id']:2d}: 2D({pixel[0]:4.0f}, {pixel[1]:4.0f}) → 3D({coords[0]:6.3f}, {coords[1]:6.3f}, {coords[2]:6.3f})")
    
    def add_validation_point(self, x: float, y: float, z: float, label: str = None):
        """Add a ground truth validation point."""
        point = np.array([x, y, z])
        self.validation_points.append(point)
        print(f"Added validation point: ({x:.3f}, {y:.3f}, {z:.3f})")
        if label:
            print(f"  Label: {label}")
    
    def calculate_validation_errors(self):
        """Calculate errors between markers and validation points."""
        if not self.markers or not self.validation_points:
            print("Need both markers and validation points for error calculation")
            return
        
        print(f"\n{'='*50}")
        print("VALIDATION ERROR ANALYSIS")
        print(f"{'='*50}")
        
        marker_coords = np.array([marker['world_coords'] for marker in self.markers])
        
        errors = []
        for i, marker in enumerate(self.markers):
            if i < len(self.validation_points):
                error_vec = marker['world_coords'] - self.validation_points[i]
                error_distance = np.linalg.norm(error_vec)
                errors.append(error_distance)
                
                print(f"Marker {marker['id']:2d}:")
                print(f"  Projected: ({marker['world_coords'][0]:6.3f}, {marker['world_coords'][1]:6.3f}, {marker['world_coords'][2]:6.3f})")
                print(f"  Ground truth: ({self.validation_points[i][0]:6.3f}, {self.validation_points[i][1]:6.3f}, {self.validation_points[i][2]:6.3f})")
                print(f"  Error: {error_distance:.3f} meters")
                print(f"  Error vector: ({error_vec[0]:6.3f}, {error_vec[1]:6.3f}, {error_vec[2]:6.3f})")
                print()
        
        if errors:
            errors = np.array(errors)
            print(f"Error Statistics:")
            print(f"  Mean error: {errors.mean():.3f} meters")
            print(f"  Std error:  {errors.std():.3f} meters")
            print(f"  Min error:  {errors.min():.3f} meters")
            print(f"  Max error:  {errors.max():.3f} meters")
            print(f"  RMS error:  {np.sqrt((errors**2).mean()):.3f} meters")
    
    def visualize(self):
        """Main visualization function."""
        print(f"\n{'='*60}")
        print("3D MARKER VISUALIZATION")
        print(f"{'='*60}")
        
        if self.point_cloud is None:
            print("Error: No LiDAR scan data loaded")
            return
        
        # Calculate statistics
        self.calculate_marker_statistics()
        
        # Try Open3D first, fall back to matplotlib
        if not self.try_visualize_open3d():
            print("Using matplotlib visualization...")
            self.visualize_matplotlib()

def main():
    parser = argparse.ArgumentParser(description="Visualize 3D markers on LiDAR scan")
    parser.add_argument('--lidar', type=str, help='Path to LiDAR scan file (.glb)')
    parser.add_argument('--markers', type=str, help='Path to markers JSON file')
    parser.add_argument('--add-validation', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                       help='Add a ground truth validation point')
    args = parser.parse_args()
    
    # Look for latest marker file if not specified
    marker_file = args.markers
    if marker_file is None:
        # Look for marker files in current directory
        marker_files = [f for f in os.listdir('.') if f.startswith('markers_') and f.endswith('.json')]
        if marker_files:
            # Get the most recent one
            marker_files.sort()
            marker_file = marker_files[-1]
            print(f"Using latest marker file: {marker_file}")
    
    visualizer = Marker3DVisualizer(lidar_scan_path=args.lidar, marker_file=marker_file)
    
    # Add validation point if provided
    if args.add_validation:
        visualizer.add_validation_point(args.add_validation[0], args.add_validation[1], args.add_validation[2])
        visualizer.calculate_validation_errors()
    
    visualizer.visualize()

if __name__ == "__main__":
    main()