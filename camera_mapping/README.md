# Camera Mapping System for Robot Navigation

A comprehensive 2D to 3D coordinate mapping system that transforms CCTV camera pixel coordinates to real-world coordinates for robot navigation.

## Overview

This system solves the challenge of mapping object detections from CCTV cameras to real-world coordinates that robots can navigate to. It supports multiple calibration methods and integrates seamlessly with object detection systems like YOLOv5.

### Key Features

- **Multiple Calibration Methods**: Manual points, ArUco markers, checkerboard patterns
- **Real-time Coordinate Mapping**: Convert pixel coordinates to world coordinates instantly
- **YOLO Integration**: Direct integration with YOLOv5 object detection
- **Robot Command Generation**: Generate navigation commands for detected objects
- **Validation Tools**: Built-in calibration accuracy validation
- **Visualization**: Real-time grid overlay and detection visualization

## Installation

### Quick Setup

1. **Run the setup script:**
   ```bash
   cd camera_mapping
   python setup.py
   ```

2. **Manual Installation:**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

- OpenCV (with contrib modules)
- NumPy
- PyTorch (for YOLO integration)
- Matplotlib
- Pillow

## Quick Start

### 1. Camera Calibration

Run the demo script to calibrate your camera:

```bash
python demo_calibration.py
```

Choose from:
- **Manual Calibration**: Click 4 corners of a known rectangle
- **ArUco Calibration**: Use printed ArUco markers
- **Real-time Mapping**: Test coordinate transformation

### 2. Basic Usage

```python
from camera_calibrator import CameraMapper

# Initialize mapper
mapper = CameraMapper("my_camera")

# Load existing calibration
mapper.load_calibration("my_calibration.json")

# Convert pixel to world coordinates
world_coords = mapper.image_to_world((320, 240))
print(f"World coordinates: {world_coords}")  # (2.5, 1.8) meters
```

### 3. YOLO Integration

```python
from yolo_integration import YOLOCameraIntegration

# Initialize with calibration
integration = YOLOCameraIntegration(
    camera_id="main_camera",
    calibration_file="demo_calibration.json"
)

# Process video stream
integration.process_video_stream(source=0)
```

## Calibration Methods

### Manual Calibration

Best for: Known room dimensions, permanent installations

1. Identify 4 points on the floor with known real-world coordinates
2. Click these points in the camera view
3. System computes homography transformation

```python
# Example: 5m x 4m room
world_points = [
    (0.0, 0.0),    # Corner 1
    (5.0, 0.0),    # Corner 2  
    (5.0, 4.0),    # Corner 3
    (0.0, 4.0)     # Corner 4
]

mapper.manual_calibration(frame, world_points)
```

### ArUco Marker Calibration

Best for: Automated calibration, temporary setups

1. Print ArUco markers (IDs 0, 1, 2, 3)
2. Place at known positions
3. System automatically detects and calibrates

```python
# Define marker positions
marker_positions = {
    0: (0.0, 0.0),    # Marker 0 at origin
    1: (3.0, 0.0),    # Marker 1 at 3m along X
    2: (3.0, 2.0),    # Marker 2 at (3,2)
    3: (0.0, 2.0)     # Marker 3 at (0,2)
}

mapper.aruco_calibration(frame, marker_positions)
```

### Checkerboard Calibration

Best for: High accuracy requirements

```python
# 8x6 checkerboard with 2.5cm squares
mapper.checkerboard_calibration(
    frame, 
    pattern_size=(8, 6), 
    square_size=0.025
)
```

## Robot Integration

### Basic Coordinate Transformation

```python
# Object detected at pixel (x, y)
detection_pixel = (450, 300)

# Convert to world coordinates
world_coords = mapper.image_to_world(detection_pixel)

# Send to robot
robot_command = {
    'action': 'NAVIGATE_TO',
    'target': world_coords,  # (3.2, 2.1) meters
    'object_type': 'spill'
}
```

### YOLO + Robot Pipeline

```python
# Full detection and navigation pipeline
integration = YOLOCameraIntegration("camera1", "calibration.json")

# Detect objects
detections = integration.detect_objects(frame)

# Generate robot commands
commands = integration.generate_robot_commands(detections)

# Execute commands
for cmd in commands:
    if cmd['action'] == 'PICKUP':
        robot.navigate_to(cmd['target_world'])
        robot.pickup_object()
```

## Advanced Features

### Batch Processing

```python
# Transform multiple points at once
image_points = [(100, 150), (200, 250), (300, 350)]
world_points = mapper.batch_transform(image_points)
```

### Validation

```python
# Check calibration accuracy
validation = mapper.validate_calibration()
print(f"Mean error: {validation['mean_world_error']:.3f} meters")
```

### Filtering Detections

```python
# Filter by object type and confidence
filtered = integration.filter_detections(
    detections,
    target_classes=['bottle', 'cup'],
    min_confidence=0.7,
    world_bounds=(0, 0, 5, 4)  # Room boundaries
)
```

## Coordinate Systems

### Image Coordinates
- Origin: Top-left corner (0, 0)
- Units: Pixels
- Format: (u, v) where u is horizontal, v is vertical

### World Coordinates
- Origin: User-defined (typically room corner)
- Units: Meters (default)
- Format: (X, Y) where X and Y are real-world distances

### Transformation
```
[X]   [h11  h12  h13] [u]
[Y] = [h21  h22  h23] [v]
[1]   [h31  h32  h33] [1]
```

## Configuration

### Camera Settings

```python
mapper = CameraMapper(
    camera_id="main_camera",
    coordinate_frame="floor_plane",
    units="meters"
)
```

### YOLO Settings

```python
integration = YOLOCameraIntegration(
    camera_id="detection_cam",
    calibration_file="cal.json",
    confidence_threshold=0.5,
    yolo_weights="yolov5s.pt"
)
```

## File Structure

```
camera_mapping/
├── camera_calibrator.py      # Core calibration system
├── demo_calibration.py       # Interactive demos
├── yolo_integration.py       # YOLO detection integration
├── robot_interface.py        # Robot command interface
├── setup.py                  # Installation script
├── requirements.txt          # Dependencies
└── README.md                 # This file

Generated Files:
├── demo_calibration.json     # Manual calibration data
├── aruco_calibration.json    # ArUco calibration data
└── sample_calibration.json   # Test data
```

## Troubleshooting

### Common Issues

1. **"Camera not calibrated" error**
   - Run calibration first: `python demo_calibration.py`
   - Check calibration file path exists

2. **Poor mapping accuracy**
   - Use more calibration points (>4)
   - Ensure points are spread across the image
   - Check for camera lens distortion

3. **YOLO integration fails**
   - Install YOLOv5: `pip install ultralytics`
   - Check weights file path
   - Verify camera access

4. **Coordinate system misalignment**
   - Verify world coordinate definition
   - Check robot coordinate frame
   - Validate with known test points

### Calibration Tips

- **Point Selection**: Choose points at corners of your area of interest
- **Accuracy**: Use precise measurements for world coordinates
- **Distribution**: Spread calibration points across the entire camera view
- **Validation**: Test with known objects before deployment

## Performance

- **Coordinate Transform**: ~0.1ms per point
- **YOLO Detection**: ~50-100ms per frame (GPU dependent)
- **Calibration**: ~1-5 seconds (depending on method)
- **Memory Usage**: <100MB for typical configurations

## Examples

### Smart Building Cleanup Robot

```python
# Setup
integration = YOLOCameraIntegration("ceiling_cam", "lobby_cal.json")

# Detect spills and objects
detections = integration.detect_objects(frame)
spills = integration.filter_detections(detections, ['bottle', 'cup'])

# Navigate robot to clean
for spill in spills:
    world_pos = spill['center_world']
    robot.navigate_to(world_pos)
    robot.clean_area(radius=0.5)  # Clean 50cm radius
```

### Security Monitoring

```python
# Track people in restricted areas
detections = integration.detect_objects(frame)
people = integration.filter_detections(
    detections, 
    ['person'],
    world_bounds=(2.0, 3.0, 5.0, 6.0)  # Restricted zone
)

for person in people:
    alert = {
        'type': 'SECURITY_BREACH',
        'location': person['center_world'],
        'confidence': person['confidence'],
        'timestamp': time.time()
    }
    security_system.send_alert(alert)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review example code
3. Open an issue on GitHub
