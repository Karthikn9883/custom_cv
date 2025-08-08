import numpy as np
import cv2
import yaml
import os
from typing import Tuple, Optional, Dict, Any

class ProjectionSystem:
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(script_dir, '..', 'configs')
        
        self.config_dir = config_dir
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvec = None
        self.tvec = None
        self.floor_plane_normal = None
        self.floor_plane_point = None
        
        self._load_calibration_data()
    
    def _load_calibration_data(self):
        try:
            # Load camera intrinsics
            intrinsics_path = os.path.join(self.config_dir, 'camera_intrinsics.yaml')
            with open(intrinsics_path, 'r') as f:
                intrinsics_data = yaml.safe_load(f)
            
            self.camera_matrix = np.array(intrinsics_data['camera_matrix']['data'], dtype=np.float64)
            self.dist_coeffs = np.array(intrinsics_data['distortion_coefficients']['data'], dtype=np.float64)
            print("✓ Camera intrinsics loaded successfully")
            
            # Load camera extrinsics
            extrinsics_path = os.path.join(self.config_dir, 'camera_extrinsics.yaml')
            with open(extrinsics_path, 'r') as f:
                extrinsics_data = yaml.safe_load(f)
            
            self.rvec = np.array(extrinsics_data['rotation_vector']['data'], dtype=np.float64)
            self.tvec = np.array(extrinsics_data['translation_vector']['data'], dtype=np.float64)
            print("✓ Camera extrinsics loaded successfully")
            
            # Load floor plane
            floor_plane_path = os.path.join(self.config_dir, 'floor_plane.yaml')
            with open(floor_plane_path, 'r') as f:
                floor_data = yaml.safe_load(f)
            
            self.floor_plane_normal = np.array(floor_data['floor_plane']['normal'], dtype=np.float64)
            # Handle the complex floor plane point format
            point_data = floor_data['floor_plane']['point_on_plane']
            if len(point_data) >= 3:
                # Extract the numeric values, handling the numpy scalar object
                self.floor_plane_point = np.array([
                    float(point_data[0]) if isinstance(point_data[0], (int, float)) else 0.0,
                    float(point_data[1]) if isinstance(point_data[1], (int, float)) else 0.0,
                    float(point_data[2]) if hasattr(point_data[2], 'item') else 0.0
                ], dtype=np.float64)
            else:
                self.floor_plane_point = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            
            print("✓ Floor plane data loaded successfully")
            
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            raise
    
    def pixel_to_3d_ray(self, pixel_coords: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert 2D pixel coordinates to a 3D ray in world coordinates.
        
        Args:
            pixel_coords: (x, y) pixel coordinates
            
        Returns:
            Tuple of (ray_origin, ray_direction) in world coordinates
        """
        if self.camera_matrix is None or self.rvec is None or self.tvec is None:
            raise RuntimeError("Calibration data not loaded")
        
        # Convert pixel to normalized camera coordinates
        pixel_point = np.array([[pixel_coords]], dtype=np.float32)
        undistorted_point = cv2.undistortPoints(pixel_point, self.camera_matrix, self.dist_coeffs)
        
        # Create 3D point in camera coordinates (at unit distance)
        camera_point = np.array([undistorted_point[0][0][0], undistorted_point[0][0][1], 1.0])
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(self.rvec)
        
        # Transform ray to world coordinates
        ray_direction_world = R @ camera_point
        ray_direction_world = ray_direction_world / np.linalg.norm(ray_direction_world)  # Normalize
        
        # Camera position in world coordinates
        camera_position_world = -R.T @ self.tvec.reshape(3)
        
        return camera_position_world, ray_direction_world
    
    def ray_plane_intersection(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> Optional[np.ndarray]:
        """
        Find intersection of a 3D ray with the floor plane.
        
        Args:
            ray_origin: 3D point where ray starts
            ray_direction: 3D direction vector of ray
            
        Returns:
            3D intersection point or None if no intersection
        """
        if self.floor_plane_normal is None or self.floor_plane_point is None:
            raise RuntimeError("Floor plane data not loaded")
        
        # Plane equation: (P - P0) · N = 0
        # Ray equation: P = O + t * D
        # Substituting: (O + t * D - P0) · N = 0
        # Solving for t: t = (P0 - O) · N / (D · N)
        
        denominator = np.dot(ray_direction, self.floor_plane_normal)
        
        # Check if ray is parallel to plane
        if abs(denominator) < 1e-6:
            return None
        
        t = np.dot(self.floor_plane_point - ray_origin, self.floor_plane_normal) / denominator
        
        # Check if intersection is behind the ray origin
        if t < 0:
            return None
        
        # Calculate intersection point
        intersection = ray_origin + t * ray_direction
        return intersection
    
    def project_2d_to_3d(self, pixel_coords: Tuple[float, float]) -> Optional[np.ndarray]:
        """
        Main function to convert 2D pixel coordinates to 3D world coordinates.
        
        Args:
            pixel_coords: (x, y) pixel coordinates
            
        Returns:
            3D world coordinates [x, y, z] or None if projection fails
        """
        try:
            # Get 3D ray from pixel
            ray_origin, ray_direction = self.pixel_to_3d_ray(pixel_coords)
            
            # Find intersection with floor plane
            world_point = self.ray_plane_intersection(ray_origin, ray_direction)
            
            return world_point
            
        except Exception as e:
            print(f"Error in 2D to 3D projection: {e}")
            return None
    
    def project_3d_to_2d(self, world_point: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Project 3D world coordinates back to 2D pixel coordinates.
        
        Args:
            world_point: 3D point in world coordinates [x, y, z]
            
        Returns:
            (x, y) pixel coordinates or None if projection fails
        """
        try:
            if self.camera_matrix is None or self.rvec is None or self.tvec is None:
                raise RuntimeError("Calibration data not loaded")
            
            # Project 3D point to 2D pixel coordinates
            pixel_coords, _ = cv2.projectPoints(
                world_point.reshape(1, 1, 3), 
                self.rvec, 
                self.tvec, 
                self.camera_matrix, 
                self.dist_coeffs
            )
            
            return (float(pixel_coords[0][0][0]), float(pixel_coords[0][0][1]))
            
        except Exception as e:
            print(f"Error in 3D to 2D projection: {e}")
            return None
    
    def is_calibrated(self) -> bool:
        """Check if all calibration data is properly loaded."""
        return all([
            self.camera_matrix is not None,
            self.dist_coeffs is not None,
            self.rvec is not None,
            self.tvec is not None,
            self.floor_plane_normal is not None,
            self.floor_plane_point is not None
        ])
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """Get summary of loaded calibration data."""
        return {
            'camera_matrix_loaded': self.camera_matrix is not None,
            'distortion_coeffs_loaded': self.dist_coeffs is not None,
            'extrinsics_loaded': self.rvec is not None and self.tvec is not None,
            'floor_plane_loaded': self.floor_plane_normal is not None and self.floor_plane_point is not None,
            'fully_calibrated': self.is_calibrated()
        }

def test_projection_system():
    """Test function to validate the projection system."""
    try:
        proj_system = ProjectionSystem()
        
        print("Calibration Status:")
        info = proj_system.get_calibration_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        if proj_system.is_calibrated():
            # Test with a sample pixel coordinate
            test_pixel = (320, 240)  # Center of a 640x480 image
            world_point = proj_system.project_2d_to_3d(test_pixel)
            
            if world_point is not None:
                print(f"\nTest projection:")
                print(f"  Pixel {test_pixel} -> World {world_point}")
                
                # Test reverse projection
                back_to_pixel = proj_system.project_3d_to_2d(world_point)
                if back_to_pixel is not None:
                    print(f"  World {world_point} -> Pixel {back_to_pixel}")
                    error = np.sqrt((test_pixel[0] - back_to_pixel[0])**2 + (test_pixel[1] - back_to_pixel[1])**2)
                    print(f"  Reprojection error: {error:.2f} pixels")
            else:
                print("\nTest projection failed - could not project to 3D")
        else:
            print("\nProjection system not fully calibrated - skipping tests")
            
    except Exception as e:
        print(f"Error testing projection system: {e}")

if __name__ == "__main__":
    test_projection_system()