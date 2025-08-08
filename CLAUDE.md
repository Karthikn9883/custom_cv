# SCOPE Project Context

This file summarizes the development context for the Enhanced 2D-3D Mapping System. It was last updated on August 8, 2025.

## Project Goal

The project goal is to create a high-fidelity, 3D localization system. The system uses a **3D LiDAR scan** of the environment as a primary ground truth to calculate the precise real-world `(X, Y, Z)` coordinates for any object detected by a CCTV camera. This compensates for camera distortion and provides a complete map of the room, not just the visible area.

## Current Status: Enhanced System - Calibration Required ⚠️

**MAJOR UPDATE:** The complete enhanced 3D localization system is now operational with advanced features, but requires calibration improvement for high precision mapping.

1.  **World Coordinate System Defined:** ✅ The LiDAR scan has been processed, and the floor plane is mathematically defined in `floor_plane.yaml`.
2.  **Camera Intrinsics Calibrated:** ✅ The camera's internal optical properties have been successfully calculated and saved to `camera_intrinsics.yaml`.
3.  **Camera Extrinsics Status:** ⚠️ Calibration files present but accuracy is poor (9,367px reprojection error - requires recalibration).
4.  **Enhanced Interactive Visualization:** ✅ Real-time 2D-to-3D mapping with camera position visualization, distance measurement, and performance monitoring.
5.  **System Validation:** ✅ Comprehensive testing suite confirms 89.5% system health with excellent RTSP connectivity.

**Current State:** The enhanced system is operational with advanced visualization and optimization features. However, **camera extrinsic calibration requires improvement** for high-precision mapping. All tools are ready for recalibration workflow.

## Development Summary

We have successfully completed an enhanced 3D localization pipeline with advanced visualization and optimization features.
1.  **Adopted 3D Localization:** ✅ Designed a new algorithm based on unprojecting 2D detections into 3D rays and finding their intersection with a LiDAR-defined floor plane.
2.  **Added 3D Libraries:** ✅ Installed `trimesh`, `scikit-learn`, `open3d`, and `PyQt5` to handle 3D data processing, plane fitting, and visualization.
3.  **Processed LiDAR Scan:** ✅ Created and ran `extract_floor_plane.py` to generate `floor_plane.yaml`.
4.  **Performed Intrinsic Calibration:** ✅ Overhauled `calibrate_camera.py` to use a chessboard pattern, successfully generating `camera_intrinsics.yaml` from the live CCTV feed.
5.  **Enhanced Extrinsic Calibration:** ✅ Enhanced `calibrate_extrinsics.py` with visual feedback, quality assessment, and automatic error analysis.
6.  **Created Camera Visualization System:** ✅ Built `camera_visualizer.py` for real-time camera position, direction, and FOV display in 3D space.
7.  **Added Performance Optimization:** ✅ Implemented Windows/NVIDIA GPU optimizations with `performance_optimizer.py` for RTX 4070 and 32GB RAM systems.
8.  **Built Camera Placement Tool:** ✅ Created `camera_placement_tool.py` for interactive camera positioning before calibration.
9.  **Implemented System Validation:** ✅ Developed `system_test_validator.py` for comprehensive testing and health monitoring.
10. **Enhanced Interactive Mapper:** ✅ `interactive_mapper_3d.py` now includes camera visualization, distance measurement, and performance monitoring.

## Key Files

### Configuration Files ✅
-   `mart_building_cv/configs/main_config.yaml`: Contains the `input_source` for the camera feed.
-   `mart_building_cv/configs/floor_plane.yaml`: **Generated.** Stores the mathematical definition of the room's floor plane.
-   `mart_building_cv/configs/camera_intrinsics.yaml`: **Generated.** Camera intrinsic matrix and distortion coefficients.
-   `mart_building_cv/configs/camera_extrinsics.yaml`: **Generated.** Camera pose (rotation and translation vectors) - **Needs recalibration for accuracy.**

### Enhanced Operational Scripts ✅
-   `mart_building_cv/scripts/interactive_mapper_3d.py`: **Primary tool.** Enhanced 2D-to-3D mapping with camera visualization, distance measurement, and performance monitoring.
-   `mart_building_cv/scripts/calibrate_extrinsics.py`: **Enhanced.** Interactive calibration with visual feedback, quality assessment, and error analysis.
-   `mart_building_cv/scripts/camera_placement_tool.py`: **NEW.** Interactive camera positioning tool with real-time 3D visualization.
-   `mart_building_cv/scripts/system_test_validator.py`: **NEW.** Comprehensive system testing and validation suite.
-   `mart_building_cv/scripts/calibrate_camera.py`: Intrinsic camera calibration utility.
-   `mart_building_cv/scripts/extract_floor_plane.py`: Floor plane extraction from LiDAR data.

### Core System Files ✅
-   `mart_building_cv/src/projection_utils.py`: **Core system.** Handles all 2D-to-3D projections and calibration data.
-   `mart_building_cv/src/camera_visualizer.py`: **NEW.** Camera position, direction, and FOV visualization in 3D space.
-   `mart_building_cv/src/performance_optimizer.py`: **NEW.** Windows/NVIDIA GPU optimizations for high-performance operation.
-   `8_6_2025.glb`: **LiDAR scan.** Complete 3D point cloud of the room (18,779 points).

### Documentation ✅
-   `ENHANCED_SYSTEM_GUIDE.md`: **NEW.** Comprehensive usage guide for all enhanced features.

### Legacy Files
-   `mart_building_cv/src/detection/detector.py`: Contains old homography logic - superseded by projection_utils.py.

## System Usage

The enhanced 3D localization system requires a specific workflow for optimal results. **Current calibration needs improvement for high precision.**

### System Health Status (Latest Test Results)
- **Overall Health**: GOOD (89.5%)
- **RTSP Connectivity**: ✅ Excellent (1280x1944 resolution, 100% frame capture success)
- **Hardware Performance**: ✅ Excellent (32GB RAM, 32-thread CPU)
- **Critical Issue**: ⚠️ Poor calibration accuracy (9,367px reprojection error)
- **CUDA Status**: ⚠️ Not available (CPU-only processing)

### Recommended Workflow

#### Step 1: System Validation ✅
```bash
# Navigate to project directory
cd C:\Users\Karth\Desktop\projects\custom_cv

# Test complete system health
python mart_building_cv\scripts\system_test_validator.py --source rtsp://192.168.1.234:554/avstream/channel=1/stream=0.sdp --save-report
```

#### Step 2: Camera Positioning (NEW) 🎯
```bash
# Interactive camera placement tool
python mart_building_cv\scripts\camera_placement_tool.py --source rtsp://192.168.1.234:554/avstream/channel=1/stream=0.sdp
```

#### Step 3: **CRITICAL - Recalibrate Extrinsics** ⚠️
```bash
# Enhanced calibration with quality feedback
python mart_building_cv\scripts\calibrate_extrinsics.py
```
**Target**: Reduce reprojection error from 9,367px to <2px for excellent accuracy.

#### Step 4: Enhanced Interactive Mapping ✅
```bash
# Run the enhanced interactive mapper
python mart_building_cv\scripts\interactive_mapper_3d.py --source rtsp://192.168.1.234:554/avstream/channel=1/stream=0.sdp
```

### Enhanced System Features ✅
- **Real-time 2D-to-3D mapping**: Click on video feed to place 3D markers with enhanced accuracy validation
- **Camera Position Visualization**: Real-time camera position, direction, and field-of-view display in 3D space
- **Distance Measurement**: Precise 3D coordinate distance calculation between any two markers
- **Performance Monitoring**: System health, GPU utilization, and RTSP quality monitoring
- **Full LiDAR visualization**: Rich point cloud showing room geometry with height-based coloring
- **Enhanced Interactive Controls**: 
  - Save/load markers with persistence
  - Real-time validation statistics
  - Camera visualization toggle
  - Performance information display
  - Quality assessment and recommendations
- **Visual validation**: Markers appear in both video feed and 3D room model with accuracy feedback

### Technical Specifications ✅
- **Coordinate System**: LiDAR-based world coordinates (meters)
- **Video Input**: RTSP stream at 1280x1944 resolution, 6.0 FPS (tested and verified)
- **Calibration Status**: 
  - Intrinsics: ✅ Valid camera matrix and distortion coefficients
  - Extrinsics: ⚠️ Poor quality (9,367px error) - **requires recalibration**
  - Floor Plane: ✅ Valid mathematical definition
- **Point Cloud**: 18,779 LiDAR points, intelligent downsampling based on performance
- **Performance Optimization**: Windows/NVIDIA optimized, 8-thread processing, high-priority scheduling
- **Memory Management**: Optimized for 32GB RAM systems with large point cloud support
- **Backend**: Adaptive matplotlib backend selection (TkAgg on Windows)
- **Hardware Support**: CPU-only processing (CUDA not available but system performs excellently)

### Camera Visualization Specifications 🎯
- **Camera Position**: Real-time 3D position marker (0.122, -0.251, -2.484) meters
- **Field of View**: 49.9° horizontal × 5.6° vertical (calculated from camera matrix)
- **Frustum Display**: Interactive 3D viewing cone with adjustable length (2.0m default)
- **Direction Vector**: Visual indicator showing camera pointing direction

## Immediate Action Items (Priority Order)

### CRITICAL - Calibration Improvement ⚠️
1.  **Recalibrate Camera Extrinsics**: Use enhanced calibration tool to reduce 9,367px error to <2px for excellent accuracy.
2.  **Use Camera Placement Tool**: Visually verify camera position before calibration for better point selection.
3.  **Validation Testing**: Re-run system validator after recalibration to confirm improvement.

### Recommended Enhancements
4.  **CUDA Installation**: Install CUDA support for 2-3x performance improvement (optional - system works excellently without).
5.  **Calibration Documentation**: Document optimal calibration points and procedures for future reference.

## Future Development

With the enhanced 3D localization system operational and properly calibrated, potential next steps include:

1.  **Object Detection Integration:** Integrate with YOLO or similar for automated object detection + 3D localization.
2.  **ROS 2 Integration:** Send 3D coordinates to ROS 2 nodes for robot navigation.
3.  **Multiple Camera Support:** Extend system to handle multiple RTSP camera feeds.
4.  **Machine Learning Integration:** Use the high-quality 2D-3D correspondences for training spatial AI models.
5.  **Real-time Tracking:** Extend from static marker placement to dynamic object tracking.

## System Health Monitoring

The system now includes comprehensive health monitoring:
- **Automatic Testing**: `system_test_validator.py` provides complete system diagnostics
- **Performance Metrics**: Real-time monitoring of RTSP quality, processing speed, and accuracy
- **Quality Assessment**: Automatic calibration quality scoring and recommendations
- **Hardware Optimization**: Intelligent resource utilization for Windows/NVIDIA systems