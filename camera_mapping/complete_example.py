"""
Complete Integration Example: CCTV to Robot Navigation
======================================================

This example demonstrates the complete pipeline from camera calibration
to object detection to robot navigation commands.
"""

import cv2
import numpy as np
import time
import os
import sys
from pathlib import Path
from typing import List, Dict

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from camera_calibrator import CameraMapper, CalibrationMethod
from yolo_integration import YOLOCameraIntegration
from robot_interface import SmartBuildingRobot, create_detection_commands, Priority

class SmartBuildingSystem:
    """
    Complete smart building system integrating camera mapping with robot control
    """
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.cameras = {}
        self.robots = {}
        self.detections_history = []
        self.is_running = False
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'commands_sent': 0,
            'successful_pickups': 0,
            'failed_commands': 0
        }
        
        print("Smart Building System initialized")
    
    def _load_config(self, config_file: str) -> dict:
        """Load system configuration"""
        default_config = {
            'cameras': {
                'lobby_cam': {
                    'id': 'lobby_cam',
                    'calibration_file': 'lobby_calibration.json',
                    'source': 0,
                    'detection_classes': ['bottle', 'cup', 'person', 'chair'],
                    'confidence_threshold': 0.5
                }
            },
            'robots': {
                'cleaner_bot': {
                    'id': 'cleaner_bot',
                    'type': 'SmartBuildingRobot',
                    'restricted_areas': [[1.0, 1.0, 2.0, 2.0]],
                    'home_position': [0.5, 0.5]
                }
            },
            'detection_settings': {
                'min_confidence': 0.6,
                'max_detections_per_frame': 10,
                'detection_interval': 1.0  # seconds
            }
        }
        
        if config_file and os.path.exists(config_file):
            import json
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
            # Merge with defaults
            default_config.update(loaded_config)
        
        return default_config
    
    def setup_cameras(self):
        """Initialize all cameras"""
        print("Setting up cameras...")
        
        for cam_id, cam_config in self.config['cameras'].items():
            try:
                # Check if calibration exists
                cal_file = cam_config['calibration_file']
                if not os.path.exists(cal_file):
                    print(f"⚠️  Calibration file not found for {cam_id}: {cal_file}")
                    print(f"   Please calibrate camera first using demo_calibration.py")
                    continue
                
                # Initialize YOLO integration
                integration = YOLOCameraIntegration(
                    camera_id=cam_id,
                    calibration_file=cal_file,
                    confidence_threshold=cam_config['confidence_threshold']
                )
                
                self.cameras[cam_id] = {
                    'integration': integration,
                    'config': cam_config,
                    'last_detection': 0.0
                }
                
                print(f"✅ Camera {cam_id} initialized")
                
            except Exception as e:
                print(f"❌ Failed to initialize camera {cam_id}: {e}")
    
    def setup_robots(self):
        """Initialize all robots"""
        print("Setting up robots...")
        
        for robot_id, robot_config in self.config['robots'].items():
            try:
                # Create robot based on type
                if robot_config['type'] == 'SmartBuildingRobot':
                    robot = SmartBuildingRobot(robot_id)
                    
                    # Configure restricted areas
                    for area in robot_config.get('restricted_areas', []):
                        robot.add_restricted_area(tuple(area))
                    
                    # Set home position
                    home_pos = robot_config.get('home_position', [0, 0])
                    robot.current_position = tuple(home_pos)
                
                else:
                    print(f"Unknown robot type: {robot_config['type']}")
                    continue
                
                # Add event callbacks
                robot.add_callback('command_completed', self._on_command_completed)
                robot.add_callback('command_failed', self._on_command_failed)
                
                # Start robot
                robot.start_processing()
                
                self.robots[robot_id] = robot
                print(f"✅ Robot {robot_id} initialized at {robot.current_position}")
                
            except Exception as e:
                print(f"❌ Failed to initialize robot {robot_id}: {e}")
    
    def _on_command_completed(self, command, message=""):
        """Handle successful command completion"""
        self.stats['successful_pickups'] += 1
        print(f"✅ {command.robot_id if hasattr(command, 'robot_id') else 'Robot'} completed: {command.action}")
    
    def _on_command_failed(self, command, message=""):
        """Handle failed commands"""
        self.stats['failed_commands'] += 1
        print(f"❌ Command failed: {command.action} - {message}")
    
    def process_single_frame(self, cam_id: str) -> List[dict]:
        """Process a single frame from specified camera"""
        if cam_id not in self.cameras:
            return []
        
        camera = self.cameras[cam_id]
        integration = camera['integration']
        
        # Open video source
        source = camera['config']['source']
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Cannot open camera {cam_id} source: {source}")
            return []
        
        try:
            ret, frame = cap.read()
            if not ret:
                return []
            
            # Detect objects
            detections = integration.detect_objects(frame)
            
            # Filter detections
            filtered_detections = integration.filter_detections(
                detections,
                target_classes=camera['config'].get('detection_classes'),
                min_confidence=self.config['detection_settings']['min_confidence']
            )
            
            # Update statistics
            self.stats['total_detections'] += len(filtered_detections)
            camera['last_detection'] = time.time()
            
            return filtered_detections
            
        finally:
            cap.release()
    
    def send_robot_commands(self, detections: List[dict]):
        """Convert detections to robot commands and send them"""
        if not detections or not self.robots:
            return
        
        # Create commands from detections
        commands = create_detection_commands(detections)
        
        # Send commands to available robots
        robot_ids = list(self.robots.keys())
        
        for i, command in enumerate(commands):
            # Simple round-robin assignment
            robot_id = robot_ids[i % len(robot_ids)]
            robot = self.robots[robot_id]
            
            if robot.send_command(command):
                self.stats['commands_sent'] += 1
                print(f"📤 Sent command to {robot_id}: {command.action} at {command.target_world}")
    
    def run_continuous_monitoring(self, duration: float = 60.0):
        """Run continuous monitoring for specified duration"""
        print(f"Starting continuous monitoring for {duration} seconds...")
        
        start_time = time.time()
        self.is_running = True
        
        try:
            while self.is_running and (time.time() - start_time) < duration:
                # Process each camera
                for cam_id in self.cameras.keys():
                    camera = self.cameras[cam_id]
                    
                    # Check if it's time for detection
                    interval = self.config['detection_settings']['detection_interval']
                    if time.time() - camera['last_detection'] >= interval:
                        
                        detections = self.process_single_frame(cam_id)
                        
                        if detections:
                            print(f"📷 {cam_id}: {len(detections)} objects detected")
                            
                            # Store detection history
                            self.detections_history.extend(detections)
                            
                            # Send robot commands
                            self.send_robot_commands(detections)
                
                # Brief pause
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
        
        finally:
            self.is_running = False
            print("Monitoring stopped")
    
    def display_statistics(self):
        """Display system statistics"""
        print("\n" + "="*50)
        print("SYSTEM STATISTICS")
        print("="*50)
        
        print(f"Total Detections: {self.stats['total_detections']}")
        print(f"Commands Sent: {self.stats['commands_sent']}")
        print(f"Successful Pickups: {self.stats['successful_pickups']}")
        print(f"Failed Commands: {self.stats['failed_commands']}")
        
        # Robot status
        print(f"\nRobot Status:")
        for robot_id, robot in self.robots.items():
            status = robot.get_status()
            print(f"  {robot_id}:")
            print(f"    Position: {status['current_position']}")
            print(f"    Busy: {status['is_busy']}")
            print(f"    Queue Size: {status['queue_size']}")
        
        # Camera status
        print(f"\nCamera Status:")
        for cam_id, camera in self.cameras.items():
            last_detection = camera['last_detection']
            time_since = time.time() - last_detection if last_detection > 0 else float('inf')
            print(f"  {cam_id}: Last detection {time_since:.1f}s ago")
    
    def shutdown(self):
        """Shutdown the system"""
        print("Shutting down system...")
        
        self.is_running = False
        
        # Stop all robots
        for robot in self.robots.values():
            robot.stop_processing()
        
        print("System shutdown complete")

def quick_demo():
    """Quick demonstration without camera setup"""
    print("Quick Demo: Smart Building System")
    print("=" * 40)
    
    # Create system with default config
    system = SmartBuildingSystem()
    
    # Setup robots only (skip cameras for demo)
    system.setup_robots()
    
    if not system.robots:
        print("No robots available for demo")
        return
    
    # Simulate detections
    fake_detections = [
        {
            'class_name': 'bottle',
            'center_world': (2.0, 1.5),
            'confidence': 0.85,
            'center_image': (320, 240),
            'timestamp': time.time()
        },
        {
            'class_name': 'cup', 
            'center_world': (3.5, 2.8),
            'confidence': 0.92,
            'center_image': (450, 300),
            'timestamp': time.time()
        },
        {
            'class_name': 'person',
            'center_world': (1.5, 1.5),  # In restricted area
            'confidence': 0.78,
            'center_image': (200, 350),
            'timestamp': time.time()
        }
    ]
    
    print(f"Simulating {len(fake_detections)} detections...")
    
    # Send commands
    system.send_robot_commands(fake_detections)
    
    # Wait for processing
    print("Processing commands...")
    time.sleep(8)
    
    # Show results
    system.display_statistics()
    
    # Shutdown
    system.shutdown()

def full_demo():
    """Full demonstration with camera setup"""
    print("Full Demo: Smart Building System with Cameras")
    print("=" * 50)
    
    # Check for calibration files
    required_files = ["demo_calibration.json", "aruco_calibration.json"]
    available_files = [f for f in required_files if os.path.exists(f)]
    
    if not available_files:
        print("⚠️  No calibration files found!")
        print("Please run camera calibration first:")
        print("  python demo_calibration.py")
        print("\nRunning quick demo instead...")
        quick_demo()
        return
    
    # Update config to use available calibration
    config = {
        'cameras': {
            'main_cam': {
                'id': 'main_cam',
                'calibration_file': available_files[0],
                'source': 0,
                'detection_classes': ['bottle', 'cup', 'person'],
                'confidence_threshold': 0.5
            }
        }
    }
    
    # Create system
    system = SmartBuildingSystem()
    system.config.update(config)
    
    # Setup components
    system.setup_cameras()
    system.setup_robots()
    
    if not system.cameras:
        print("No cameras available, running quick demo...")
        quick_demo()
        return
    
    # Run monitoring
    try:
        system.run_continuous_monitoring(duration=30.0)  # 30 seconds
    except KeyboardInterrupt:
        pass
    
    # Show results
    system.display_statistics()
    
    # Shutdown
    system.shutdown()

def main():
    """Main demo function"""
    print("Smart Building System - Complete Integration Demo")
    print("=" * 55)
    
    while True:
        print("\nSelect demo option:")
        print("1. Quick Demo (simulated detections)")
        print("2. Full Demo (requires camera calibration)")
        print("3. Setup Instructions")
        print("4. Exit")
        
        try:
            choice = input("Enter choice (1-4): ").strip()
            
            if choice == "1":
                quick_demo()
            elif choice == "2":
                full_demo()
            elif choice == "3":
                print_setup_instructions()
            elif choice == "4":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def print_setup_instructions():
    """Print setup instructions"""
    print("\n" + "="*60)
    print("SETUP INSTRUCTIONS")
    print("="*60)
    
    print("\n1. INSTALL DEPENDENCIES:")
    print("   python setup.py")
    print("   # OR manually:")
    print("   pip install -r requirements.txt")
    
    print("\n2. CALIBRATE CAMERA:")
    print("   python demo_calibration.py")
    print("   # Choose option 1 (Manual) or 2 (ArUco)")
    
    print("\n3. TEST INTEGRATION:")
    print("   python complete_example.py")
    print("   # Choose option 2 (Full Demo)")
    
    print("\n4. CUSTOMIZE CONFIGURATION:")
    print("   # Edit camera and robot settings in code")
    print("   # Add more cameras, robots, detection classes")
    
    print("\n5. PRODUCTION DEPLOYMENT:")
    print("   # Integrate with your robot's API")
    print("   # Set up proper coordinate frames")
    print("   # Configure detection thresholds")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
