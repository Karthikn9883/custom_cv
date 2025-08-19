"""
2D to 3D Mapping System for CCTV Robot Navigation
==================================================

This module provides a comprehensive solution for mapping CCTV camera coordinates 
to real-world coordinates for robot navigation. It supports multiple calibration 
methods including homography estimation, fiducial markers, and manual calibration.

Features:
- Planar homography mapping for floor plane detection
- ArUco/AprilTag marker-based calibration
- Manual calibration with known geometry
- Floor plan registration
- Camera distortion handling
- Real-time coordinate transformation
"""

import cv2
import numpy as np
import json
import os
import logging
from typing import List, Tuple, Dict, Optional, Union
import time
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CalibrationMethod(Enum):
    """Supported calibration methods"""
    MANUAL_POINTS = "manual_points"
    ARUCO_MARKERS = "aruco_markers"
    FLOOR_PLAN = "floor_plan"
    CHECKERBOARD = "checkerboard"

@dataclass
class CalibrationPoint:
    """Represents a point correspondence between image and world coordinates"""
    image_point: Tuple[float, float]  # (u, v) in pixels
    world_point: Tuple[float, float]  # (X, Y) in meters
    confidence: float = 1.0

@dataclass
class CameraCalibration:
    """Stores camera calibration data"""
    homography_matrix: np.ndarray
    calibration_points: List[CalibrationPoint]
    method: CalibrationMethod
    timestamp: float
    camera_id: str
    coordinate_frame: str = "floor_plane"
    units: str = "meters"
    reprojection_error: float = 0.0

class CameraMapper:
    """
    Main class for handling 2D to 3D coordinate mapping from CCTV cameras
    """
    
    def __init__(self, camera_id: str = "default"):
        self.camera_id = camera_id
        self.calibration: Optional[CameraCalibration] = None
        self.is_calibrated = False
        
        # ArUco detector setup
        try:
            # Try new OpenCV 4.7+ API
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            # Fallback to older API
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # Mouse callback data for manual point selection
        self.mouse_points = []
        self.current_frame = None
        
    def load_calibration(self, calibration_file: str) -> bool:
        """Load calibration from file"""
        try:
            with open(calibration_file, 'r') as f:
                data = json.load(f)
            
            # Reconstruct calibration object
            points = [CalibrationPoint(
                image_point=tuple(p['image_point']),
                world_point=tuple(p['world_point']),
                confidence=p.get('confidence', 1.0)
            ) for p in data['calibration_points']]
            
            self.calibration = CameraCalibration(
                homography_matrix=np.array(data['homography_matrix']),
                calibration_points=points,
                method=CalibrationMethod(data['method']),
                timestamp=data['timestamp'],
                camera_id=data['camera_id'],
                coordinate_frame=data.get('coordinate_frame', 'floor_plane'),
                units=data.get('units', 'meters'),
                reprojection_error=data.get('reprojection_error', 0.0)
            )
            
            self.is_calibrated = True
            logger.info(f"Loaded calibration for camera {self.camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False
    
    def save_calibration(self, calibration_file: str) -> bool:
        """Save calibration to file"""
        if not self.is_calibrated:
            logger.error("No calibration to save")
            return False
        
        try:
            data = {
                'camera_id': self.calibration.camera_id,
                'method': self.calibration.method.value,
                'timestamp': self.calibration.timestamp,
                'coordinate_frame': self.calibration.coordinate_frame,
                'units': self.calibration.units,
                'reprojection_error': self.calibration.reprojection_error,
                'homography_matrix': self.calibration.homography_matrix.tolist(),
                'calibration_points': [
                    {
                        'image_point': list(p.image_point),
                        'world_point': list(p.world_point),
                        'confidence': p.confidence
                    } for p in self.calibration.calibration_points
                ]
            }
            
            with open(calibration_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved calibration to {calibration_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for manual point selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_points.append((x, y))
            cv2.circle(self.current_frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.current_frame, f"P{len(self.mouse_points)}", 
                       (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Calibration", self.current_frame)
            print(f"Point {len(self.mouse_points)}: ({x}, {y})")
    
    def manual_calibration(self, frame: np.ndarray, world_points: List[Tuple[float, float]]) -> bool:
        """
        Manual calibration by clicking points on the image
        
        Args:
            frame: Input image frame
            world_points: Corresponding world coordinates for each point to be clicked
        """
        self.mouse_points = []
        self.current_frame = frame.copy()
        
        print(f"Manual Calibration Mode")
        print(f"Click {len(world_points)} points on the image in order:")
        for i, (x, y) in enumerate(world_points):
            print(f"Point {i+1}: World coordinates ({x}, {y})")
        print("Press 'q' when done, 'r' to reset")
        
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.mouse_callback)
        cv2.imshow("Calibration", self.current_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') and len(self.mouse_points) >= len(world_points):
                break
            elif key == ord('r'):
                self.mouse_points = []
                self.current_frame = frame.copy()
                cv2.imshow("Calibration", self.current_frame)
                print("Reset points. Click again.")
        
        cv2.destroyAllWindows()
        
        if len(self.mouse_points) < len(world_points):
            logger.error("Insufficient points selected")
            return False
        
        # Create calibration points
        calibration_points = []
        for i in range(len(world_points)):
            calibration_points.append(CalibrationPoint(
                image_point=self.mouse_points[i],
                world_point=world_points[i]
            ))
        
        return self._compute_homography(calibration_points, CalibrationMethod.MANUAL_POINTS)
    
    def aruco_calibration(self, frame: np.ndarray, marker_world_coords: Dict[int, Tuple[float, float]]) -> bool:
        """
        Calibration using ArUco markers
        
        Args:
            frame: Input image frame
            marker_world_coords: Dictionary mapping marker IDs to world coordinates
        """
        # Detect ArUco markers
        try:
            # Try new OpenCV 4.7+ API
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, _ = detector.detectMarkers(frame)
        except AttributeError:
            # Fallback to older API
            corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is None or len(ids) < 4:
            logger.error("Need at least 4 ArUco markers for calibration")
            return False
        
        calibration_points = []
        
        # Visualize detected markers
        frame_vis = frame.copy()
        try:
            # Try new API
            cv2.aruco.drawDetectedMarkers(frame_vis, corners, ids)
        except:
            # Fallback - manual drawing if needed
            if ids is not None:
                for i, corner_set in enumerate(corners):
                    corner_points = corner_set[0].astype(int)
                    cv2.polylines(frame_vis, [corner_points], True, (0, 255, 0), 2)
                    center = corner_points.mean(axis=0).astype(int)
                    cv2.putText(frame_vis, str(ids[i][0]), tuple(center), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        for i, corner_set in enumerate(corners):
            marker_id = ids[i][0]
            
            if marker_id in marker_world_coords:
                # Calculate marker center
                corner_points = corner_set[0]
                center_x = np.mean(corner_points[:, 0])
                center_y = np.mean(corner_points[:, 1])
                
                world_coord = marker_world_coords[marker_id]
                
                calibration_points.append(CalibrationPoint(
                    image_point=(center_x, center_y),
                    world_point=world_coord
                ))
                
                # Draw marker info
                cv2.putText(frame_vis, f"ID:{marker_id} World:{world_coord}", 
                           (int(center_x), int(center_y-20)), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 1)
        
        cv2.imshow("ArUco Detection", frame_vis)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        
        if len(calibration_points) < 4:
            logger.error("Insufficient markers detected with known coordinates")
            return False
        
        return self._compute_homography(calibration_points, CalibrationMethod.ARUCO_MARKERS)
    
    def checkerboard_calibration(self, frame: np.ndarray, pattern_size: Tuple[int, int], 
                                square_size: float, world_origin: Tuple[float, float] = (0, 0)) -> bool:
        """
        Calibration using checkerboard pattern
        
        Args:
            frame: Input image frame
            pattern_size: (cols, rows) of internal corners
            square_size: Size of each square in meters
            world_origin: World coordinate of the top-left corner
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size)
        
        if not ret:
            logger.error("Checkerboard pattern not found")
            return False
        
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Generate world coordinates for corners
        calibration_points = []
        for i in range(pattern_size[1]):  # rows
            for j in range(pattern_size[0]):  # cols
                world_x = world_origin[0] + j * square_size
                world_y = world_origin[1] + i * square_size
                
                corner_idx = i * pattern_size[0] + j
                image_point = corners[corner_idx][0]
                
                calibration_points.append(CalibrationPoint(
                    image_point=(image_point[0], image_point[1]),
                    world_point=(world_x, world_y)
                ))
        
        # Visualize
        frame_vis = frame.copy()
        cv2.drawChessboardCorners(frame_vis, pattern_size, corners, ret)
        cv2.imshow("Checkerboard Detection", frame_vis)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        
        return self._compute_homography(calibration_points, CalibrationMethod.CHECKERBOARD)
    
    def _compute_homography(self, calibration_points: List[CalibrationPoint], 
                           method: CalibrationMethod) -> bool:
        """Compute homography matrix from calibration points"""
        if len(calibration_points) < 4:
            logger.error("Need at least 4 points for homography computation")
            return False
        
        # Prepare point arrays
        image_points = np.array([p.image_point for p in calibration_points], dtype=np.float32)
        world_points = np.array([p.world_point for p in calibration_points], dtype=np.float32)
        
        # Compute homography
        if len(calibration_points) == 4:
            H = cv2.getPerspectiveTransform(image_points, world_points)
            reprojection_error = 0.0
        else:
            H, mask = cv2.findHomography(image_points, world_points, 
                                       cv2.RANSAC, ransacReprojThreshold=5.0)
            
            # Calculate reprojection error
            projected_points = cv2.perspectiveTransform(
                image_points.reshape(-1, 1, 2), H
            ).reshape(-1, 2)
            reprojection_error = np.mean(np.linalg.norm(projected_points - world_points, axis=1))
        
        # Create calibration object
        self.calibration = CameraCalibration(
            homography_matrix=H,
            calibration_points=calibration_points,
            method=method,
            timestamp=time.time(),
            camera_id=self.camera_id,
            reprojection_error=reprojection_error
        )
        
        self.is_calibrated = True
        
        logger.info(f"Calibration successful. Reprojection error: {reprojection_error:.3f}")
        logger.info(f"Homography matrix:\n{H}")
        
        return True
    
    def image_to_world(self, image_point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Convert image coordinates to world coordinates
        
        Args:
            image_point: (u, v) pixel coordinates
            
        Returns:
            (X, Y) world coordinates in meters, or None if not calibrated
        """
        if not self.is_calibrated:
            logger.error("Camera not calibrated")
            return None
        
        # Convert to homogeneous coordinates
        point = np.array([[[image_point[0], image_point[1]]]], dtype=np.float32)
        
        # Apply homography transformation
        world_point = cv2.perspectiveTransform(point, self.calibration.homography_matrix)
        
        return (float(world_point[0, 0, 0]), float(world_point[0, 0, 1]))
    
    def world_to_image(self, world_point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Convert world coordinates to image coordinates
        
        Args:
            world_point: (X, Y) world coordinates in meters
            
        Returns:
            (u, v) pixel coordinates, or None if not calibrated
        """
        if not self.is_calibrated:
            logger.error("Camera not calibrated")
            return None
        
        # Invert homography
        H_inv = np.linalg.inv(self.calibration.homography_matrix)
        
        # Convert to homogeneous coordinates
        point = np.array([[[world_point[0], world_point[1]]]], dtype=np.float32)
        
        # Apply inverse transformation
        image_point = cv2.perspectiveTransform(point, H_inv)
        
        return (float(image_point[0, 0, 0]), float(image_point[0, 0, 1]))
    
    def batch_transform(self, image_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Transform multiple image points to world coordinates
        
        Args:
            image_points: List of (u, v) pixel coordinates
            
        Returns:
            List of (X, Y) world coordinates
        """
        if not self.is_calibrated:
            logger.error("Camera not calibrated")
            return []
        
        # Convert to numpy array
        points = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
        
        # Apply transformation
        world_points = cv2.perspectiveTransform(points, self.calibration.homography_matrix)
        
        # Convert back to list of tuples
        return [(float(p[0, 0]), float(p[0, 1])) for p in world_points]
    
    def validate_calibration(self) -> Dict[str, float]:
        """
        Validate calibration by checking reprojection accuracy
        
        Returns:
            Dictionary with validation metrics
        """
        if not self.is_calibrated:
            return {"error": "Not calibrated"}
        
        # Test forward and backward transformation on calibration points
        errors = []
        for point in self.calibration.calibration_points:
            # Forward transform
            world_pred = self.image_to_world(point.image_point)
            world_error = np.linalg.norm(np.array(world_pred) - np.array(point.world_point))
            
            # Backward transform
            image_pred = self.world_to_image(point.world_point)
            image_error = np.linalg.norm(np.array(image_pred) - np.array(point.image_point))
            
            errors.append({
                'world_error': world_error,
                'image_error': image_error
            })
        
        world_errors = [e['world_error'] for e in errors]
        image_errors = [e['image_error'] for e in errors]
        
        return {
            'mean_world_error': np.mean(world_errors),
            'max_world_error': np.max(world_errors),
            'mean_image_error': np.mean(image_errors),
            'max_image_error': np.max(image_errors),
            'num_points': len(errors)
        }
    
    def visualize_calibration(self, frame: np.ndarray, grid_spacing: float = 1.0) -> np.ndarray:
        """
        Visualize calibration by overlaying a world coordinate grid on the image
        
        Args:
            frame: Input image frame
            grid_spacing: Spacing between grid lines in world units
            
        Returns:
            Frame with grid overlay
        """
        if not self.is_calibrated:
            return frame
        
        frame_vis = frame.copy()
        
        # Draw calibration points
        for i, point in enumerate(self.calibration.calibration_points):
            image_pt = (int(point.image_point[0]), int(point.image_point[1]))
            cv2.circle(frame_vis, image_pt, 5, (0, 255, 0), -1)
            cv2.putText(frame_vis, f"P{i+1}", (image_pt[0]+10, image_pt[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Estimate world bounds from calibration points
        world_points = [p.world_point for p in self.calibration.calibration_points]
        min_x = min(p[0] for p in world_points) - grid_spacing
        max_x = max(p[0] for p in world_points) + grid_spacing
        min_y = min(p[1] for p in world_points) - grid_spacing
        max_y = max(p[1] for p in world_points) + grid_spacing
        
        # Draw grid lines
        # Vertical lines
        x = min_x
        while x <= max_x:
            start_world = (x, min_y)
            end_world = (x, max_y)
            
            start_image = self.world_to_image(start_world)
            end_image = self.world_to_image(end_world)
            
            if start_image and end_image:
                start_pt = (int(start_image[0]), int(start_image[1]))
                end_pt = (int(end_image[0]), int(end_image[1]))
                
                # Check if points are within image bounds
                h, w = frame.shape[:2]
                if (0 <= start_pt[0] < w and 0 <= start_pt[1] < h and
                    0 <= end_pt[0] < w and 0 <= end_pt[1] < h):
                    cv2.line(frame_vis, start_pt, end_pt, (255, 0, 0), 1)
                    
                    # Label every meter
                    if x % 1.0 == 0:
                        cv2.putText(frame_vis, f"{x:.0f}m", (start_pt[0], start_pt[1]-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            
            x += grid_spacing
        
        # Horizontal lines
        y = min_y
        while y <= max_y:
            start_world = (min_x, y)
            end_world = (max_x, y)
            
            start_image = self.world_to_image(start_world)
            end_image = self.world_to_image(end_world)
            
            if start_image and end_image:
                start_pt = (int(start_image[0]), int(start_image[1]))
                end_pt = (int(end_image[0]), int(end_image[1]))
                
                # Check if points are within image bounds
                h, w = frame.shape[:2]
                if (0 <= start_pt[0] < w and 0 <= start_pt[1] < h and
                    0 <= end_pt[0] < w and 0 <= end_pt[1] < h):
                    cv2.line(frame_vis, start_pt, end_pt, (255, 0, 0), 1)
                    
                    # Label every meter
                    if y % 1.0 == 0:
                        cv2.putText(frame_vis, f"{y:.0f}m", (start_pt[0]+5, start_pt[1]),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            
            y += grid_spacing
        
        # Add calibration info
        info_text = [
            f"Camera: {self.camera_id}",
            f"Method: {self.calibration.method.value}",
            f"Points: {len(self.calibration.calibration_points)}",
            f"Error: {self.calibration.reprojection_error:.3f}m"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame_vis, text, (10, 30 + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame_vis