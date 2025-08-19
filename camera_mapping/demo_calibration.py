"""
Demo script for Camera Calibration System
==========================================

This script demonstrates various calibration methods and real-time coordinate mapping.
"""

import cv2
import numpy as np
import os
import sys
from camera_calibrator import CameraMapper, CalibrationMethod

def demo_manual_calibration():
    """Demonstrate manual calibration with known room dimensions"""
    print("=== Manual Calibration Demo ===")
    
    # Initialize camera mapper
    mapper = CameraMapper("demo_camera")
    
    # Simulate a room: 5m x 4m with origin at one corner
    world_points = [
        (0.0, 0.0),    # Corner 1
        (5.0, 0.0),    # Corner 2
        (5.0, 4.0),    # Corner 3
        (0.0, 4.0)     # Corner 4
    ]
    
    # Try to open camera or use a sample image
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera, using sample image")
        # Create a sample image for demo
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (50, 50), (590, 430), (100, 100, 100), 2)
        cv2.putText(frame, "Click 4 corners of the room in order", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera")
            return
    
    # Perform manual calibration
    success = mapper.manual_calibration(frame, world_points)
    
    if success:
        print("Calibration successful!")
        
        # Save calibration
        mapper.save_calibration("demo_calibration.json")
        
        # Test some transformations
        test_points = [(320, 240), (100, 100), (500, 400)]
        print("\nTesting coordinate transformations:")
        for img_pt in test_points:
            world_pt = mapper.image_to_world(img_pt)
            if world_pt:
                print(f"Image {img_pt} -> World ({world_pt[0]:.2f}, {world_pt[1]:.2f}) meters")
        
        # Validate calibration
        validation = mapper.validate_calibration()
        print(f"\nValidation results:")
        for key, value in validation.items():
            print(f"  {key}: {value:.3f}")
        
        # Visualize calibration
        if 'frame' in locals():
            vis_frame = mapper.visualize_calibration(frame)
            cv2.imshow("Calibration Visualization", vis_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    if cap.isOpened():
        cap.release()

def demo_aruco_calibration():
    """Demonstrate ArUco marker calibration"""
    print("\n=== ArUco Calibration Demo ===")
    
    # Initialize camera mapper
    mapper = CameraMapper("aruco_camera")
    
    # Define marker positions (in real world coordinates)
    marker_positions = {
        0: (0.0, 0.0),    # Marker ID 0 at origin
        1: (3.0, 0.0),    # Marker ID 1 at 3m along X
        2: (3.0, 2.0),    # Marker ID 2 at (3,2)
        3: (0.0, 2.0)     # Marker ID 3 at (0,2)
    }
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera for ArUco demo")
        return
    
    print("Position ArUco markers with IDs 0,1,2,3 at the specified locations")
    print("Marker positions:")
    for marker_id, pos in marker_positions.items():
        print(f"  Marker {marker_id}: ({pos[0]}, {pos[1]}) meters")
    print("Press 'c' to calibrate, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect markers for preview
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame, mapper.aruco_dict, parameters=mapper.aruco_params)
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Show which markers we have positions for
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in marker_positions:
                    corner_set = corners[i][0]
                    center_x = int(np.mean(corner_set[:, 0]))
                    center_y = int(np.mean(corner_set[:, 1]))
                    pos = marker_positions[marker_id]
                    cv2.putText(frame, f"World: {pos}", 
                               (center_x, center_y-30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 255, 0), 1)
        
        cv2.imshow("ArUco Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            success = mapper.aruco_calibration(frame, marker_positions)
            if success:
                print("ArUco calibration successful!")
                mapper.save_calibration("aruco_calibration.json")
                
                # Show validation
                validation = mapper.validate_calibration()
                print("Validation results:")
                for key, value in validation.items():
                    print(f"  {key}: {value:.3f}")
                
                break
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def demo_real_time_mapping():
    """Demonstrate real-time coordinate mapping"""
    print("\n=== Real-Time Mapping Demo ===")
    
    # Try to load existing calibration
    mapper = CameraMapper("realtime_camera")
    
    if os.path.exists("demo_calibration.json"):
        if mapper.load_calibration("demo_calibration.json"):
            print("Loaded existing calibration")
        else:
            print("Failed to load calibration. Run manual calibration first.")
            return
    elif os.path.exists("aruco_calibration.json"):
        if mapper.load_calibration("aruco_calibration.json"):
            print("Loaded ArUco calibration")
        else:
            print("Failed to load calibration. Run ArUco calibration first.")
            return
    else:
        print("No calibration found. Run calibration first.")
        return
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    print("Click on the video to see world coordinates")
    print("Press 'q' to quit")
    
    # Mouse callback for real-time mapping
    current_world_coords = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_world_coords
        if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:
            world_coords = mapper.image_to_world((x, y))
            if world_coords:
                current_world_coords = (x, y, world_coords[0], world_coords[1])
    
    cv2.namedWindow("Real-Time Mapping")
    cv2.setMouseCallback("Real-Time Mapping", mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Visualize calibration grid
        vis_frame = mapper.visualize_calibration(frame, grid_spacing=0.5)
        
        # Show current mouse position and world coordinates
        if current_world_coords:
            x, y, world_x, world_y = current_world_coords
            cv2.circle(vis_frame, (x, y), 8, (0, 255, 255), 2)
            text = f"Image: ({x}, {y}) World: ({world_x:.2f}, {world_y:.2f})m"
            cv2.putText(vis_frame, text, (10, vis_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Real-Time Mapping", vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def demo_robot_navigation():
    """Simulate robot navigation using detected objects"""
    print("\n=== Robot Navigation Simulation ===")
    
    # Load calibration
    mapper = CameraMapper("robot_camera")
    
    calibration_files = ["demo_calibration.json", "aruco_calibration.json"]
    loaded = False
    for cal_file in calibration_files:
        if os.path.exists(cal_file):
            if mapper.load_calibration(cal_file):
                print(f"Loaded calibration from {cal_file}")
                loaded = True
                break
    
    if not loaded:
        print("No calibration found. Run calibration first.")
        return
    
    # Simulate object detection results (normally from YOLO or other detector)
    detected_objects = [
        {"name": "spill", "bbox": [200, 150, 250, 200], "confidence": 0.85},
        {"name": "object", "bbox": [400, 300, 450, 350], "confidence": 0.92},
        {"name": "person", "bbox": [100, 100, 180, 280], "confidence": 0.78}
    ]
    
    print("Simulated object detections:")
    robot_targets = []
    
    for obj in detected_objects:
        # Calculate object center
        bbox = obj["bbox"]
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        
        # Convert to world coordinates
        world_coords = mapper.image_to_world((center_x, center_y))
        
        if world_coords:
            target = {
                "name": obj["name"],
                "image_coords": (center_x, center_y),
                "world_coords": world_coords,
                "confidence": obj["confidence"]
            }
            robot_targets.append(target)
            
            print(f"  {obj['name']}: Image({center_x}, {center_y}) -> "
                  f"World({world_coords[0]:.2f}, {world_coords[1]:.2f})m "
                  f"[Confidence: {obj['confidence']:.2f}]")
    
    # Simulate robot navigation commands
    print("\nRobot Navigation Commands:")
    for target in robot_targets:
        if target["name"] in ["spill", "object"]:  # Navigate to these
            x, y = target["world_coords"]
            print(f"  NAVIGATE_TO: ({x:.2f}, {y:.2f}) # Clean {target['name']}")
        else:  # Just track these
            x, y = target["world_coords"]
            print(f"  TRACK: ({x:.2f}, {y:.2f}) # Monitor {target['name']}")
    
    return robot_targets

def main():
    """Main demo function"""
    print("Camera Calibration and Mapping Demo")
    print("===================================")
    
    while True:
        print("\nSelect demo option:")
        print("1. Manual Calibration (click 4 room corners)")
        print("2. ArUco Marker Calibration")
        print("3. Real-time Coordinate Mapping")
        print("4. Robot Navigation Simulation")
        print("5. Exit")
        
        try:
            choice = input("Enter choice (1-5): ").strip()
            
            if choice == "1":
                demo_manual_calibration()
            elif choice == "2":
                demo_aruco_calibration()
            elif choice == "3":
                demo_real_time_mapping()
            elif choice == "4":
                demo_robot_navigation()
            elif choice == "5":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
