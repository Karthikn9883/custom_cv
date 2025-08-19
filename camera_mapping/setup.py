#!/usr/bin/env python3
"""
Setup and Installation Script for Camera Mapping System
========================================================

This script sets up the camera mapping environment and installs dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and handle errors"""
    print(f"{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Failed!")
        print("Error:", e.stderr)
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please use Python 3.8 or higher")
        return False

def install_dependencies():
    """Install required Python packages"""
    print("Installing dependencies...")
    
    # Install from requirements.txt
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        cmd = f"pip install -r {requirements_file}"
        if run_command(cmd, "Installing packages from requirements.txt"):
            print("✅ All dependencies installed successfully")
            return True
    else:
        print("❌ requirements.txt not found")
    
    # Fallback: install core packages individually
    print("Installing core packages individually...")
    packages = [
        "opencv-contrib-python",
        "numpy",
        "torch",
        "torchvision", 
        "ultralytics",
        "matplotlib",
        "Pillow"
    ]
    
    all_success = True
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            all_success = False
    
    return all_success

def setup_yolo_integration():
    """Setup YOLOv5 integration"""
    print("Setting up YOLOv5 integration...")
    
    yolo_dir = Path(__file__).parent.parent / "yolov5-master"
    if yolo_dir.exists():
        print(f"✅ YOLOv5 directory found at {yolo_dir}")
        
        # Install YOLOv5 requirements if available
        yolo_requirements = yolo_dir / "requirements.txt"
        if yolo_requirements.exists():
            cmd = f"pip install -r {yolo_requirements}"
            run_command(cmd, "Installing YOLOv5 requirements")
        
        return True
    else:
        print(f"ℹ️  YOLOv5 directory not found at {yolo_dir}")
        print("You can still use the system, but YOLO integration will use simulation mode")
        return False

def create_sample_calibration():
    """Create sample calibration files for testing"""
    print("Creating sample calibration data...")
    
    sample_calibration = {
        "camera_id": "sample_camera",
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
            {
                "image_point": [100, 100],
                "world_point": [0.0, 0.0],
                "confidence": 1.0
            },
            {
                "image_point": [500, 100],
                "world_point": [4.0, 0.0],
                "confidence": 1.0
            },
            {
                "image_point": [500, 400],
                "world_point": [4.0, 3.0],
                "confidence": 1.0
            },
            {
                "image_point": [100, 400],
                "world_point": [0.0, 3.0],
                "confidence": 1.0
            }
        ]
    }
    
    import json
    sample_file = Path(__file__).parent / "sample_calibration.json"
    
    try:
        with open(sample_file, 'w') as f:
            json.dump(sample_calibration, f, indent=2)
        print(f"✅ Sample calibration saved to {sample_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to create sample calibration: {e}")
        return False

def test_installation():
    """Test if the installation works"""
    print("Testing installation...")
    
    try:
        # Test imports
        import cv2
        import numpy as np
        print("✅ OpenCV and NumPy import successful")
        
        # Test camera access
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera access successful")
            cap.release()
        else:
            print("ℹ️  Camera not available (this is OK for headless systems)")
        
        # Test camera calibrator import
        sys.path.append(str(Path(__file__).parent))
        from camera_calibrator import CameraMapper
        mapper = CameraMapper("test")
        print("✅ Camera calibrator import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "="*60)
    print("INSTALLATION COMPLETE - USAGE INSTRUCTIONS")
    print("="*60)
    
    print("\n1. CAMERA CALIBRATION:")
    print("   Run the demo script to calibrate your camera:")
    print("   python demo_calibration.py")
    
    print("\n2. MANUAL CALIBRATION:")
    print("   - Select option 1 in the demo")
    print("   - Click 4 corners of a known rectangle on the floor")
    print("   - Enter the real-world dimensions")
    
    print("\n3. ARUCO CALIBRATION:")
    print("   - Print ArUco markers (IDs 0, 1, 2, 3)")
    print("   - Place them at known positions")
    print("   - Select option 2 in the demo")
    
    print("\n4. YOLO INTEGRATION:")
    print("   After calibration, run:")
    print("   python yolo_integration.py")
    
    print("\n5. ROBOT INTEGRATION:")
    print("   Use the generated world coordinates to send navigation")
    print("   commands to your robot system")
    
    print("\n6. FILES CREATED:")
    print("   - demo_calibration.json (manual calibration)")
    print("   - aruco_calibration.json (ArUco calibration)")
    print("   - sample_calibration.json (test data)")
    
    print(f"\n7. TROUBLESHOOTING:")
    print("   - Ensure camera is connected and accessible")
    print("   - Check calibration file paths in integration scripts")
    print("   - Verify coordinate system matches your robot's frame")
    
    print("\n" + "="*60)

def main():
    """Main setup function"""
    print("Camera Mapping System Setup")
    print("="*40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup YOLO integration
    setup_yolo_integration()
    
    # Create sample data
    create_sample_calibration()
    
    # Test installation
    if test_installation():
        print("✅ Installation test passed")
    else:
        print("❌ Installation test failed")
        print("You may need to troubleshoot dependency issues")
    
    # Print usage instructions
    print_usage_instructions()

if __name__ == "__main__":
    main()
