"""
Robot Interface for Camera Mapping System
==========================================

This module provides interfaces for integrating the camera mapping system
with various robot platforms and navigation systems.
"""

import json
import time
import threading
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import queue
import logging

logger = logging.getLogger(__name__)

class CommandStatus(Enum):
    """Status of robot commands"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Priority(Enum):
    """Command priorities"""
    EMERGENCY = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class RobotCommand:
    """Represents a robot navigation/action command"""
    command_id: str
    action: str
    target_world: Tuple[float, float]
    priority: Priority
    object_type: str = ""
    confidence: float = 1.0
    timeout: float = 60.0  # seconds
    created_at: float = 0.0
    status: CommandStatus = CommandStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

class RobotInterface:
    """
    Base interface for robot communication and command execution
    """
    
    def __init__(self, robot_id: str = "default_robot"):
        self.robot_id = robot_id
        self.command_queue = queue.PriorityQueue()
        self.active_commands = {}
        self.command_history = []
        self.is_running = False
        self.worker_thread = None
        
        # Callbacks for different events
        self.command_callbacks = {
            'command_started': [],
            'command_completed': [],
            'command_failed': [],
            'robot_status_changed': []
        }
        
        # Robot state
        self.current_position = (0.0, 0.0)
        self.is_busy = False
        self.last_update = time.time()
    
    def add_callback(self, event: str, callback: Callable):
        """Add callback for robot events"""
        if event in self.command_callbacks:
            self.command_callbacks[event].append(callback)
    
    def send_command(self, command: RobotCommand) -> bool:
        """
        Send command to robot
        
        Args:
            command: RobotCommand object
            
        Returns:
            True if command was queued successfully
        """
        try:
            # Add to queue with priority (lower number = higher priority)
            priority_value = command.priority.value
            self.command_queue.put((priority_value, time.time(), command))
            
            logger.info(f"Command {command.command_id} queued: {command.action} at {command.target_world}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue command {command.command_id}: {e}")
            return False
    
    def cancel_command(self, command_id: str) -> bool:
        """Cancel a pending or active command"""
        # Check active commands
        if command_id in self.active_commands:
            command = self.active_commands[command_id]
            command.status = CommandStatus.CANCELLED
            self._notify_callbacks('command_failed', command, "Cancelled by user")
            del self.active_commands[command_id]
            return True
        
        # For pending commands in queue, we'd need to implement queue filtering
        # This is a simplified implementation
        logger.warning(f"Command {command_id} not found or already executed")
        return False
    
    def start_processing(self):
        """Start processing command queue"""
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_commands)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        logger.info(f"Robot {self.robot_id} started processing commands")
    
    def stop_processing(self):
        """Stop processing commands"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info(f"Robot {self.robot_id} stopped processing commands")
    
    def _process_commands(self):
        """Worker thread to process command queue"""
        while self.is_running:
            try:
                # Get command with timeout
                try:
                    priority, timestamp, command = self.command_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check if command is still valid
                if time.time() - command.created_at > command.timeout:
                    command.status = CommandStatus.FAILED
                    self._notify_callbacks('command_failed', command, "Timeout")
                    continue
                
                # Execute command
                self._execute_command(command)
                
            except Exception as e:
                logger.error(f"Error in command processing: {e}")
    
    def _execute_command(self, command: RobotCommand):
        """Execute a single command"""
        command.status = CommandStatus.IN_PROGRESS
        self.active_commands[command.command_id] = command
        self.is_busy = True
        
        self._notify_callbacks('command_started', command)
        
        try:
            # Dispatch to specific action handler
            success = False
            
            if command.action == "NAVIGATE_TO":
                success = self._navigate_to(command.target_world)
            elif command.action == "PICKUP":
                success = self._pickup_object(command.target_world, command.object_type)
            elif command.action == "AVOID":
                success = self._avoid_area(command.target_world)
            elif command.action == "INVESTIGATE":
                success = self._investigate_location(command.target_world)
            elif command.action == "PATROL":
                success = self._patrol_area(command.target_world)
            else:
                logger.warning(f"Unknown action: {command.action}")
                success = False
            
            # Update command status
            if success:
                command.status = CommandStatus.COMPLETED
                self._notify_callbacks('command_completed', command)
            else:
                command.status = CommandStatus.FAILED
                self._notify_callbacks('command_failed', command, "Execution failed")
                
                # Retry logic
                if command.retry_count < command.max_retries:
                    command.retry_count += 1
                    command.status = CommandStatus.PENDING
                    self.send_command(command)  # Re-queue
                    
        except Exception as e:
            command.status = CommandStatus.FAILED
            self._notify_callbacks('command_failed', command, str(e))
            logger.error(f"Command execution failed: {e}")
        
        finally:
            # Clean up
            if command.command_id in self.active_commands:
                del self.active_commands[command.command_id]
            self.command_history.append(command)
            self.is_busy = False
            self.last_update = time.time()
    
    def _navigate_to(self, target: Tuple[float, float]) -> bool:
        """Navigate to target location (to be implemented by subclass)"""
        logger.info(f"Navigating to {target}")
        # Simulate navigation
        time.sleep(2.0)
        self.current_position = target
        return True
    
    def _pickup_object(self, location: Tuple[float, float], object_type: str) -> bool:
        """Pick up object at location (to be implemented by subclass)"""
        logger.info(f"Picking up {object_type} at {location}")
        # Simulate pickup
        time.sleep(3.0)
        return True
    
    def _avoid_area(self, location: Tuple[float, float]) -> bool:
        """Avoid area around location (to be implemented by subclass)"""
        logger.info(f"Avoiding area around {location}")
        # Simulate avoidance
        time.sleep(1.0)
        return True
    
    def _investigate_location(self, location: Tuple[float, float]) -> bool:
        """Investigate location (to be implemented by subclass)"""
        logger.info(f"Investigating location {location}")
        # Simulate investigation
        time.sleep(2.5)
        return True
    
    def _patrol_area(self, center: Tuple[float, float]) -> bool:
        """Patrol around area (to be implemented by subclass)"""
        logger.info(f"Patrolling around {center}")
        # Simulate patrol
        time.sleep(5.0)
        return True
    
    def _notify_callbacks(self, event: str, command: RobotCommand, message: str = ""):
        """Notify registered callbacks"""
        for callback in self.command_callbacks.get(event, []):
            try:
                callback(command, message)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def get_status(self) -> Dict:
        """Get current robot status"""
        return {
            'robot_id': self.robot_id,
            'current_position': self.current_position,
            'is_busy': self.is_busy,
            'queue_size': self.command_queue.qsize(),
            'active_commands': len(self.active_commands),
            'last_update': self.last_update
        }

class ROSRobotInterface(RobotInterface):
    """
    ROS-specific robot interface
    """
    
    def __init__(self, robot_id: str = "ros_robot"):
        super().__init__(robot_id)
        self.ros_available = False
        
        try:
            import rospy
            from geometry_msgs.msg import PoseStamped, Twist
            from nav_msgs.msg import OccupancyGrid
            from actionlib_msgs.msg import GoalStatus
            
            self.rospy = rospy
            self.PoseStamped = PoseStamped
            self.Twist = Twist
            self.GoalStatus = GoalStatus
            
            # Initialize ROS publishers/subscribers
            self._init_ros()
            self.ros_available = True
            
        except ImportError:
            logger.warning("ROS not available, using simulation mode")
    
    def _init_ros(self):
        """Initialize ROS communication"""
        if not self.ros_available:
            return
        
        # Navigation goal publisher
        self.goal_pub = self.rospy.Publisher(
            '/move_base_simple/goal', 
            self.PoseStamped, 
            queue_size=1
        )
        
        # Velocity publisher for emergency stops
        self.cmd_vel_pub = self.rospy.Publisher(
            '/cmd_vel', 
            self.Twist, 
            queue_size=1
        )
    
    def _navigate_to(self, target: Tuple[float, float]) -> bool:
        """Navigate using ROS navigation stack"""
        if not self.ros_available:
            return super()._navigate_to(target)
        
        try:
            # Create navigation goal
            goal = self.PoseStamped()
            goal.header.frame_id = "map"
            goal.header.stamp = self.rospy.Time.now()
            goal.pose.position.x = target[0]
            goal.pose.position.y = target[1]
            goal.pose.position.z = 0.0
            goal.pose.orientation.w = 1.0  # No rotation
            
            # Send goal
            self.goal_pub.publish(goal)
            logger.info(f"ROS navigation goal sent: {target}")
            
            # Wait for completion (simplified)
            time.sleep(3.0)  # In real implementation, monitor action server
            self.current_position = target
            
            return True
            
        except Exception as e:
            logger.error(f"ROS navigation failed: {e}")
            return False

class SmartBuildingRobot(RobotInterface):
    """
    Specialized interface for smart building robots
    """
    
    def __init__(self, robot_id: str, building_map: Dict = None):
        super().__init__(robot_id)
        self.building_map = building_map or {}
        self.restricted_areas = []
        self.cleaning_tools = ['vacuum', 'mop', 'pickup_arm']
        
    def add_restricted_area(self, area: Tuple[float, float, float, float]):
        """Add restricted area (min_x, min_y, max_x, max_y)"""
        self.restricted_areas.append(area)
    
    def is_in_restricted_area(self, position: Tuple[float, float]) -> bool:
        """Check if position is in restricted area"""
        x, y = position
        for min_x, min_y, max_x, max_y in self.restricted_areas:
            if min_x <= x <= max_x and min_y <= y <= max_y:
                return True
        return False
    
    def _navigate_to(self, target: Tuple[float, float]) -> bool:
        """Navigate with building-specific logic"""
        # Check restricted areas
        if self.is_in_restricted_area(target):
            logger.warning(f"Target {target} is in restricted area")
            return False
        
        # Simulate smart building navigation
        logger.info(f"Smart building robot navigating to {target}")
        
        # Calculate path (simplified)
        current_x, current_y = self.current_position
        target_x, target_y = target
        
        distance = ((target_x - current_x)**2 + (target_y - current_y)**2)**0.5
        travel_time = distance * 2.0  # 2 seconds per meter
        
        time.sleep(min(travel_time, 10.0))  # Cap at 10 seconds for demo
        self.current_position = target
        
        return True
    
    def _pickup_object(self, location: Tuple[float, float], object_type: str) -> bool:
        """Specialized pickup for building objects"""
        logger.info(f"Building robot picking up {object_type} at {location}")
        
        # Navigate to object
        if not self._navigate_to(location):
            return False
        
        # Simulate different pickup strategies
        pickup_time = {
            'bottle': 2.0,
            'cup': 1.5,
            'paper': 1.0,
            'default': 3.0
        }
        
        time.sleep(pickup_time.get(object_type, pickup_time['default']))
        
        logger.info(f"Successfully picked up {object_type}")
        return True

def create_detection_commands(detections: List[Dict], 
                            priority_map: Dict[str, Priority] = None) -> List[RobotCommand]:
    """
    Create robot commands from camera detections
    
    Args:
        detections: List of detection dictionaries from YOLO integration
        priority_map: Mapping of object types to priorities
        
    Returns:
        List of RobotCommand objects
    """
    if priority_map is None:
        priority_map = {
            'person': Priority.HIGH,
            'bottle': Priority.MEDIUM,
            'cup': Priority.MEDIUM,
            'chair': Priority.LOW,
            'default': Priority.LOW
        }
    
    action_map = {
        'bottle': 'PICKUP',
        'cup': 'PICKUP',
        'person': 'AVOID',
        'chair': 'NAVIGATE_AROUND',
        'default': 'INVESTIGATE'
    }
    
    commands = []
    
    for i, detection in enumerate(detections):
        object_type = detection['class_name']
        world_coords = detection['center_world']
        confidence = detection['confidence']
        
        priority = priority_map.get(object_type, priority_map['default'])
        action = action_map.get(object_type, action_map['default'])
        
        command = RobotCommand(
            command_id=f"{object_type}_{int(time.time())}_{i}",
            action=action,
            target_world=world_coords,
            priority=priority,
            object_type=object_type,
            confidence=confidence
        )
        
        commands.append(command)
    
    return commands

def demo_robot_interface():
    """Demo the robot interface"""
    print("Robot Interface Demo")
    print("===================")
    
    # Create robot
    robot = SmartBuildingRobot("demo_robot")
    
    # Add restricted area
    robot.add_restricted_area((1.0, 1.0, 2.0, 2.0))
    
    # Add callbacks for monitoring
    def on_command_started(command, message=""):
        print(f"✅ Started: {command.action} to {command.target_world}")
    
    def on_command_completed(command, message=""):
        print(f"✅ Completed: {command.action} at {command.target_world}")
    
    def on_command_failed(command, message=""):
        print(f"❌ Failed: {command.action} - {message}")
    
    robot.add_callback('command_started', on_command_started)
    robot.add_callback('command_completed', on_command_completed)
    robot.add_callback('command_failed', on_command_failed)
    
    # Start processing
    robot.start_processing()
    
    # Simulate detections and create commands
    fake_detections = [
        {
            'class_name': 'bottle',
            'center_world': (3.0, 2.0),
            'confidence': 0.85
        },
        {
            'class_name': 'person',
            'center_world': (1.5, 1.5),  # In restricted area
            'confidence': 0.92
        },
        {
            'class_name': 'cup',
            'center_world': (4.0, 3.0),
            'confidence': 0.78
        }
    ]
    
    # Create and send commands
    commands = create_detection_commands(fake_detections)
    
    print(f"\nSending {len(commands)} commands...")
    for command in commands:
        robot.send_command(command)
    
    # Wait for processing
    print("Processing commands...")
    time.sleep(15)
    
    # Show status
    status = robot.get_status()
    print(f"\nRobot Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Stop robot
    robot.stop_processing()
    print("\nDemo completed!")

if __name__ == "__main__":
    demo_robot_interface()
