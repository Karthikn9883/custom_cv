#!/usr/bin/env python3
"""
Test Script for Camera Mapping System
======================================

This script tests the basic functionality of the camera mapping system.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
        print(f"   Version: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
        print(f"   Version: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
        print(f"   Version: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        from camera_calibrator import CameraMapper
        print("✅ Camera calibrator imported successfully")
    except ImportError as e:
        print(f"❌ Camera calibrator import failed: {e}")
        return False
    
    return True

def test_camera_mapper():
    """Test basic camera mapper functionality"""
    print("\nTesting Camera Mapper...")
    
    try:
        from camera_calibrator import CameraMapper, CalibrationPoint, CalibrationMethod
        
        # Create mapper
        mapper = CameraMapper("test_camera")
        print("✅ CameraMapper created")
        
        # Test with sample calibration points
        calibration_points = [
            CalibrationPoint((100, 100), (0.0, 0.0)),
            CalibrationPoint((500, 100), (4.0, 0.0)),
            CalibrationPoint((500, 400), (4.0, 3.0)),
            CalibrationPoint((100, 400), (0.0, 3.0))
        ]
        
        # Test homography computation
        success = mapper._compute_homography(calibration_points, CalibrationMethod.MANUAL_POINTS)
        
        if success:
            print("✅ Homography computation successful")
            
            # Test coordinate transformation
            test_point = (300, 250)
            world_coords = mapper.image_to_world(test_point)
            
            if world_coords:
                print(f"✅ Coordinate transformation: {test_point} -> {world_coords}")
                return True
            else:
                print("❌ Coordinate transformation failed")
        else:
            print("❌ Homography computation failed")
            
    except Exception as e:
        print(f"❌ Camera mapper test failed: {e}")
        return False
    
    return False

def test_aruco_detection():
    """Test ArUco marker detection"""
    print("\nTesting ArUco Detection...")
    
    try:
        import cv2
        import numpy as np
        
        # Check if ArUco is available
        if hasattr(cv2, 'aruco'):
            try:
                # Try new OpenCV 4.7+ API
                aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
                aruco_params = cv2.aruco.DetectorParameters()
                detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
                print("✅ ArUco module available (new API)")
            except AttributeError:
                # Fallback to older API
                aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
                aruco_params = cv2.aruco.DetectorParameters_create()
                detector = None
                print("✅ ArUco module available (old API)")
            
            # Create a test image with ArUco marker
            test_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
            
            # Generate a marker
            marker_size = 200
            marker_id = 0
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
            
            # Place marker in test image
            y_pos = 200
            x_pos = 300
            test_image[y_pos:y_pos+marker_size, x_pos:x_pos+marker_size] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
            
            # Detect markers
            if detector:
                corners, ids, _ = detector.detectMarkers(test_image)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(test_image, aruco_dict, parameters=aruco_params)
            
            if ids is not None and len(ids) > 0:
                print(f"✅ ArUco detection successful: detected {len(ids)} markers")
                return True
            else:
                print("❌ ArUco detection failed: no markers detected")
        else:
            print("❌ ArUco module not available")
            
    except Exception as e:
        print(f"❌ ArUco test failed: {e}")
    
    return False

def test_yolo_integration():
    """Test YOLO integration (basic)"""
    print("\nTesting YOLO Integration...")
    
    try:
        # Create a sample calibration file for testing
        import json
        sample_calibration = {
            "camera_id": "test_camera",
            "method": "manual_points", 
            "timestamp": 1692360000.0,
            "coordinate_frame": "floor_plane",
            "units": "meters",
            "reprojection_error": 0.05,
            "homography_matrix": [
                [0.01, 0.0, -3.2],
                [0.0, 0.01, -2.4], 
                [0.0, 0.0, 1.0]
            ],
            "calibration_points": [
                {"image_point": [100, 100], "world_point": [0.0, 0.0], "confidence": 1.0},
                {"image_point": [500, 100], "world_point": [4.0, 0.0], "confidence": 1.0},
                {"image_point": [500, 400], "world_point": [4.0, 3.0], "confidence": 1.0},
                {"image_point": [100, 400], "world_point": [0.0, 3.0], "confidence": 1.0}
            ]
        }
        
        test_cal_file = "test_calibration.json"
        with open(test_cal_file, 'w') as f:
            json.dump(sample_calibration, f)
        
        print("✅ Test calibration file created")
        
        # Test YOLO integration import
        from yolo_integration import YOLOCameraIntegration
        
        integration = YOLOCameraIntegration(
            camera_id="test_cam",
            calibration_file=test_cal_file,
            confidence_threshold=0.5
        )
        
        print("✅ YOLO integration initialized")
        
        # Test with dummy frame
        import numpy as np
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # This will use simulation mode since YOLO model might not be available
        detections = integration.detect_objects(dummy_frame)
        print(f"✅ Object detection test: {len(detections)} simulated detections")
        
        # Clean up
        if os.path.exists(test_cal_file):
            os.remove(test_cal_file)
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO integration test failed: {e}")
        return False

def test_robot_interface():
    """Test robot interface"""
    print("\nTesting Robot Interface...")
    
    try:
        from robot_interface import SmartBuildingRobot, RobotCommand, Priority
        
        # Create robot
        robot = SmartBuildingRobot("test_robot")
        print("✅ Robot interface created")
        
        # Test command creation
        command = RobotCommand(
            command_id="test_cmd_1",
            action="NAVIGATE_TO",
            target_world=(2.0, 3.0),
            priority=Priority.MEDIUM
        )
        
        print("✅ Robot command created")
        
        # Test command queuing (don't start processing for test)
        success = robot.send_command(command)
        
        if success:
            print("✅ Command queuing successful")
            
            # Test status
            status = robot.get_status()
            print(f"✅ Robot status: {status['robot_id']}")
            
            return True
        else:
            print("❌ Command queuing failed")
            
    except Exception as e:
        print(f"❌ Robot interface test failed: {e}")
    
    return False

def main():
    """Run all tests"""
    print("Camera Mapping System - Test Suite")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_imports),
        ("Camera Mapper", test_camera_mapper),
        ("ArUco Detection", test_aruco_detection),
        ("YOLO Integration", test_yolo_integration),
        ("Robot Interface", test_robot_interface)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run camera calibration: python demo_calibration.py")
        print("2. Test full integration: python complete_example.py")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Check error messages above.")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check Python version: Python 3.8+ required")
        print("3. Verify OpenCV installation includes contrib modules")

if __name__ == "__main__":
    main()
