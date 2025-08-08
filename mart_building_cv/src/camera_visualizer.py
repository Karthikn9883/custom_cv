import numpy as np
import cv2
import yaml
import os
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class CameraVisualizer:
    """
    Utility class for visualizing camera position, orientation, and field of view
    in 3D space overlaid with LiDAR point cloud data.
    """
    
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(script_dir, '..', 'configs')
        
        self.config_dir = config_dir
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvec = None
        self.tvec = None
        self.image_width = None
        self.image_height = None
        
        # Camera visualization parameters
        self.camera_size = 0.1  # Size of camera body in meters
        self.frustum_length = 2.0  # Length of viewing frustum in meters
        self.frustum_color = 'red'
        self.camera_color = 'blue'
        
        self._load_camera_parameters()
    
    def _load_camera_parameters(self):
        """Load camera intrinsic and extrinsic parameters."""
        try:
            # Load camera intrinsics
            intrinsics_path = os.path.join(self.config_dir, 'camera_intrinsics.yaml')
            with open(intrinsics_path, 'r') as f:
                intrinsics_data = yaml.safe_load(f)
            
            self.camera_matrix = np.array(intrinsics_data['camera_matrix']['data'], dtype=np.float64)
            self.dist_coeffs = np.array(intrinsics_data['distortion_coefficients']['data'], dtype=np.float64)
            
            # Get image dimensions (assuming standard resolution if not specified)
            self.image_width = intrinsics_data.get('image_width', 1920)
            self.image_height = intrinsics_data.get('image_height', 1080)
            
            print("✓ Camera intrinsics loaded successfully")
            
            # Load camera extrinsics
            extrinsics_path = os.path.join(self.config_dir, 'camera_extrinsics.yaml')
            with open(extrinsics_path, 'r') as f:
                extrinsics_data = yaml.safe_load(f)
            
            self.rvec = np.array(extrinsics_data['rotation_vector']['data'], dtype=np.float64)
            self.tvec = np.array(extrinsics_data['translation_vector']['data'], dtype=np.float64)
            
            print("✓ Camera extrinsics loaded successfully")
            
        except Exception as e:
            print(f"Warning: Could not load camera parameters: {e}")
    
    def get_camera_position(self) -> Optional[np.ndarray]:
        """Get camera position in world coordinates."""
        if self.rvec is None or self.tvec is None:
            return None
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(self.rvec)
        
        # Camera position in world coordinates
        camera_position = -R.T @ self.tvec.reshape(3)
        return camera_position
    
    def get_camera_direction(self) -> Optional[np.ndarray]:
        """Get camera looking direction in world coordinates."""
        if self.rvec is None:
            return None
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(self.rvec)
        
        # Camera forward direction (negative Z in camera coordinates)
        camera_forward = np.array([0, 0, 1])  # Positive Z is forward in OpenCV
        world_direction = R @ camera_forward
        return world_direction
    
    def get_camera_up_vector(self) -> Optional[np.ndarray]:
        """Get camera up direction in world coordinates."""
        if self.rvec is None:
            return None
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(self.rvec)
        
        # Camera up direction (negative Y in camera coordinates)
        camera_up = np.array([0, -1, 0])  # Negative Y is up in OpenCV
        world_up = R @ camera_up
        return world_up
    
    def get_camera_right_vector(self) -> Optional[np.ndarray]:
        """Get camera right direction in world coordinates."""
        if self.rvec is None:
            return None
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(self.rvec)
        
        # Camera right direction (positive X in camera coordinates)
        camera_right = np.array([1, 0, 0])
        world_right = R @ camera_right
        return world_right
    
    def get_field_of_view(self) -> Tuple[float, float]:
        """Calculate horizontal and vertical field of view in degrees."""
        if self.camera_matrix is None:
            return 60.0, 45.0  # Default values
        
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        
        # FOV = 2 * arctan(sensor_size / (2 * focal_length))
        h_fov = 2 * np.arctan(self.image_width / (2 * fx))
        v_fov = 2 * np.arctan(self.image_height / (2 * fy))
        
        return np.degrees(h_fov), np.degrees(v_fov)
    
    def generate_frustum_corners(self) -> Optional[np.ndarray]:
        """Generate the 8 corners of the camera frustum."""
        camera_pos = self.get_camera_position()
        camera_dir = self.get_camera_direction()
        camera_up = self.get_camera_up_vector()
        camera_right = self.get_camera_right_vector()
        
        if any(x is None for x in [camera_pos, camera_dir, camera_up, camera_right]):
            return None
        
        h_fov, v_fov = self.get_field_of_view()
        h_fov_rad = np.radians(h_fov)
        v_fov_rad = np.radians(v_fov)
        
        # Near and far plane distances
        near_dist = 0.1
        far_dist = self.frustum_length
        
        # Calculate half-widths and half-heights at near and far planes
        near_half_width = near_dist * np.tan(h_fov_rad / 2)
        near_half_height = near_dist * np.tan(v_fov_rad / 2)
        far_half_width = far_dist * np.tan(h_fov_rad / 2)
        far_half_height = far_dist * np.tan(v_fov_rad / 2)
        
        # Generate frustum corners
        corners = []
        
        # Near plane corners
        near_center = camera_pos + camera_dir * near_dist
        corners.extend([
            near_center + camera_right * near_half_width + camera_up * near_half_height,  # Top-right
            near_center - camera_right * near_half_width + camera_up * near_half_height,  # Top-left
            near_center - camera_right * near_half_width - camera_up * near_half_height,  # Bottom-left
            near_center + camera_right * near_half_width - camera_up * near_half_height,  # Bottom-right
        ])
        
        # Far plane corners
        far_center = camera_pos + camera_dir * far_dist
        corners.extend([
            far_center + camera_right * far_half_width + camera_up * far_half_height,    # Top-right
            far_center - camera_right * far_half_width + camera_up * far_half_height,    # Top-left
            far_center - camera_right * far_half_width - camera_up * far_half_height,    # Bottom-left
            far_center + camera_right * far_half_width - camera_up * far_half_height,    # Bottom-right
        ])
        
        return np.array(corners)
    
    def generate_camera_body(self) -> Optional[np.ndarray]:
        """Generate vertices for a simple camera body representation."""
        camera_pos = self.get_camera_position()
        camera_dir = self.get_camera_direction()
        camera_up = self.get_camera_up_vector()
        camera_right = self.get_camera_right_vector()
        
        if any(x is None for x in [camera_pos, camera_dir, camera_up, camera_right]):
            return None
        
        # Simple camera body as a rectangular box
        size = self.camera_size
        
        # Define camera body vertices relative to camera position
        body_vertices = [
            camera_pos - camera_right * size/2 - camera_up * size/4 - camera_dir * size/4,  # Back-bottom-left
            camera_pos + camera_right * size/2 - camera_up * size/4 - camera_dir * size/4,  # Back-bottom-right
            camera_pos + camera_right * size/2 + camera_up * size/4 - camera_dir * size/4,  # Back-top-right
            camera_pos - camera_right * size/2 + camera_up * size/4 - camera_dir * size/4,  # Back-top-left
            camera_pos - camera_right * size/2 - camera_up * size/4 + camera_dir * size/4,  # Front-bottom-left
            camera_pos + camera_right * size/2 - camera_up * size/4 + camera_dir * size/4,  # Front-bottom-right
            camera_pos + camera_right * size/2 + camera_up * size/4 + camera_dir * size/4,  # Front-top-right
            camera_pos - camera_right * size/2 + camera_up * size/4 + camera_dir * size/4,  # Front-top-left
        ]
        
        return np.array(body_vertices)
    
    def add_camera_to_plot(self, ax, show_frustum: bool = True, show_body: bool = True, 
                          show_direction: bool = True, alpha: float = 0.3):
        """
        Add camera visualization to a matplotlib 3D plot.
        
        Args:
            ax: matplotlib 3D axes object
            show_frustum: Whether to show the viewing frustum
            show_body: Whether to show the camera body
            show_direction: Whether to show the direction vector
            alpha: Transparency of the frustum faces
        """
        camera_pos = self.get_camera_position()
        camera_dir = self.get_camera_direction()
        
        if camera_pos is None:
            print("Warning: Camera position not available")
            return
        
        # Plot camera position as a point
        ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], 
                  c=self.camera_color, s=100, marker='o', label='Camera Position')
        
        # Show direction vector
        if show_direction and camera_dir is not None:
            direction_end = camera_pos + camera_dir * 0.5  # 0.5 meter direction arrow
            ax.plot([camera_pos[0], direction_end[0]], 
                   [camera_pos[1], direction_end[1]], 
                   [camera_pos[2], direction_end[2]], 
                   c=self.camera_color, linewidth=3, label='Camera Direction')
        
        # Show camera body
        if show_body:
            body_vertices = self.generate_camera_body()
            if body_vertices is not None:
                # Define the faces of the camera body box
                faces = [
                    [body_vertices[0], body_vertices[1], body_vertices[2], body_vertices[3]],  # Back
                    [body_vertices[4], body_vertices[5], body_vertices[6], body_vertices[7]],  # Front
                    [body_vertices[0], body_vertices[1], body_vertices[5], body_vertices[4]],  # Bottom
                    [body_vertices[2], body_vertices[3], body_vertices[7], body_vertices[6]],  # Top
                    [body_vertices[0], body_vertices[3], body_vertices[7], body_vertices[4]],  # Left
                    [body_vertices[1], body_vertices[2], body_vertices[6], body_vertices[5]],  # Right
                ]
                
                # Create 3D collection for camera body
                camera_body = Poly3DCollection(faces, alpha=0.6, facecolor=self.camera_color, 
                                             edgecolor='black', linewidth=1)
                ax.add_collection3d(camera_body)
        
        # Show viewing frustum
        if show_frustum:
            frustum_corners = self.generate_frustum_corners()
            if frustum_corners is not None:
                # Define frustum faces (connecting near and far plane corners)
                faces = [
                    # Near plane (quad)
                    [frustum_corners[0], frustum_corners[1], frustum_corners[2], frustum_corners[3]],
                    # Far plane (quad)
                    [frustum_corners[4], frustum_corners[5], frustum_corners[6], frustum_corners[7]],
                    # Side faces (triangles from camera to far plane)
                    [camera_pos, frustum_corners[4], frustum_corners[5]],  # Top face
                    [camera_pos, frustum_corners[5], frustum_corners[6]],  # Left face
                    [camera_pos, frustum_corners[6], frustum_corners[7]],  # Bottom face
                    [camera_pos, frustum_corners[7], frustum_corners[4]],  # Right face
                ]
                
                # Create 3D collection for frustum
                frustum_collection = Poly3DCollection(faces, alpha=alpha, facecolor=self.frustum_color, 
                                                   edgecolor=self.frustum_color, linewidth=1)
                ax.add_collection3d(frustum_collection)
                
                # Draw frustum edges more prominently
                edges = [
                    # Near plane edges
                    [frustum_corners[0], frustum_corners[1]],
                    [frustum_corners[1], frustum_corners[2]],
                    [frustum_corners[2], frustum_corners[3]],
                    [frustum_corners[3], frustum_corners[0]],
                    # Far plane edges  
                    [frustum_corners[4], frustum_corners[5]],
                    [frustum_corners[5], frustum_corners[6]],
                    [frustum_corners[6], frustum_corners[7]],
                    [frustum_corners[7], frustum_corners[4]],
                    # Connecting edges
                    [frustum_corners[0], frustum_corners[4]],
                    [frustum_corners[1], frustum_corners[5]],
                    [frustum_corners[2], frustum_corners[6]],
                    [frustum_corners[3], frustum_corners[7]],
                ]
                
                for edge in edges:
                    ax.plot([edge[0][0], edge[1][0]], 
                           [edge[0][1], edge[1][1]], 
                           [edge[0][2], edge[1][2]], 
                           c=self.frustum_color, linewidth=1, alpha=0.8)
    
    def print_camera_info(self):
        """Print detailed camera information."""
        print("\n" + "="*50)
        print("CAMERA VISUALIZATION INFO")
        print("="*50)
        
        camera_pos = self.get_camera_position()
        camera_dir = self.get_camera_direction()
        h_fov, v_fov = self.get_field_of_view()
        
        if camera_pos is not None:
            print(f"Camera Position: ({camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}) meters")
        else:
            print("Camera Position: Not available")
        
        if camera_dir is not None:
            print(f"Camera Direction: ({camera_dir[0]:.3f}, {camera_dir[1]:.3f}, {camera_dir[2]:.3f})")
        else:
            print("Camera Direction: Not available")
        
        print(f"Field of View: {h_fov:.1f}° (horizontal) × {v_fov:.1f}° (vertical)")
        print(f"Image Resolution: {self.image_width} × {self.image_height}")
        print(f"Frustum Length: {self.frustum_length:.1f} meters")
        
        if self.camera_matrix is not None:
            fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
            cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
            print(f"Focal Length: fx={fx:.1f}, fy={fy:.1f}")
            print(f"Principal Point: cx={cx:.1f}, cy={cy:.1f}")
        
        print("="*50 + "\n")

def test_camera_visualizer():
    """Test function for the camera visualizer."""
    try:
        viz = CameraVisualizer()
        viz.print_camera_info()
        
        # Create a simple test plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Add camera to plot
        viz.add_camera_to_plot(ax)
        
        # Set up the plot
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('Camera Position and Field of View Visualization')
        ax.legend()
        
        # Set equal aspect ratio
        camera_pos = viz.get_camera_position()
        if camera_pos is not None:
            # Set plot limits based on camera position
            range_size = 2.0
            ax.set_xlim(camera_pos[0] - range_size, camera_pos[0] + range_size)
            ax.set_ylim(camera_pos[1] - range_size, camera_pos[1] + range_size)
            ax.set_zlim(camera_pos[2] - range_size, camera_pos[2] + range_size)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error testing camera visualizer: {e}")

if __name__ == "__main__":
    test_camera_visualizer()