# SCOPE Project Context

This file summarizes the development context for the Gemini CLI agent. It was last updated on August 8, 2025.

## Project Goal

The project goal is to create a high-fidelity, 3D localization system. The system uses a **3D LiDAR scan** of the environment as a primary ground truth to calculate the precise real-world `(X, Y, Z)` coordinates for any object detected by a CCTV camera. This compensates for camera distortion and provides a complete map of the room, not just the visible area.

## Current Status: 3D Pipeline Fully Operational ✅

**MAJOR UPDATE:** The complete 3D localization system is now fully functional and calibrated. All components are working together successfully.

1.  **World Coordinate System Defined:** ✅ The LiDAR scan has been processed, and the floor plane is mathematically defined in `floor_plane.yaml`.
2.  **Camera Intrinsics Calibrated:** ✅ The camera's internal optical properties have been successfully calculated and saved to `camera_intrinsics.yaml`.
3.  **Camera Extrinsics Calibrated:** ✅ The camera's 3D position and orientation have been determined and saved to `camera_extrinsics.yaml`.
4.  **Interactive 3D Visualization:** ✅ Real-time 2D-to-3D mapping is working with full LiDAR point cloud visualization.

**Current State:** The system is completely calibrated and operational. The `interactive_mapper_3d.py` script provides real-time 2D-to-3D coordinate mapping with full visual validation using the actual room's LiDAR scan.

## Development Summary

We have successfully completed a full 3D localization pipeline from initial concept to operational system.
1.  **Adopted 3D Localization:** ✅ Designed a new algorithm based on unprojecting 2D detections into 3D rays and finding their intersection with a LiDAR-defined floor plane.
2.  **Added 3D Libraries:** ✅ Installed `trimesh`, `scikit-learn`, `open3d`, and `PyQt5` to handle 3D data processing, plane fitting, and visualization.
3.  **Processed LiDAR Scan:** ✅ Created and ran `extract_floor_plane.py` to generate `floor_plane.yaml`.
4.  **Performed Intrinsic Calibration:** ✅ Overhauled `calibrate_camera.py` to use a chessboard pattern, successfully generating `camera_intrinsics.yaml` from the live CCTV feed.
5.  **Created Extrinsic Calibration Tool:** ✅ Built an interactive script, `calibrate_extrinsics.py`, that allows visual mapping between 3D LiDAR scan and 2D camera feed.
6.  **Fixed 3D Visualization Issues:** ✅ Resolved matplotlib backend compatibility issues and enhanced LiDAR point cloud visualization.
7.  **Completed Interactive Mapper:** ✅ `interactive_mapper_3d.py` now provides real-time 2D-to-3D mapping with rich visual feedback.

## Key Files

### Configuration Files ✅
-   `mart_building_cv/configs/main_config.yaml`: Contains the `input_source` for the camera feed.
-   `mart_building_cv/configs/floor_plane.yaml`: **Generated.** Stores the mathematical definition of the room's floor plane.
-   `mart_building_cv/configs/camera_intrinsics.yaml`: **Generated.** Camera intrinsic matrix and distortion coefficients.
-   `mart_building_cv/configs/camera_extrinsics.yaml`: **Generated.** Camera pose (rotation and translation vectors) in 3D space.

### Operational Scripts ✅
-   `mart_building_cv/scripts/interactive_mapper_3d.py`: **Primary tool.** Real-time 2D-to-3D mapping with full LiDAR visualization.
-   `mart_building_cv/scripts/calibrate_extrinsics.py`: Interactive extrinsic calibration tool.
-   `mart_building_cv/scripts/calibrate_camera.py`: Intrinsic camera calibration utility.
-   `mart_building_cv/scripts/extract_floor_plane.py`: Floor plane extraction from LiDAR data.

### Core System Files ✅
-   `mart_building_cv/src/projection_utils.py`: **Core system.** Handles all 2D-to-3D projections and calibration data.
-   `8_6_2025.glb`: **LiDAR scan.** Complete 3D point cloud of the room (18,779 points).

### Legacy Files
-   `mart_building_cv/src/detection/detector.py`: Contains old homography logic - superseded by projection_utils.py.

## System Usage

The complete 3D localization system is now ready for use. Here's how to operate it:

### Running the Interactive 3D Mapper ✅

```bash
# Navigate to project directory
cd /Users/karthiknutulapati/Desktop/smartbuilding/custom_cv-1

# Activate virtual environment
source venv/bin/activate

# Run the interactive mapper
python3 mart_building_cv/scripts/interactive_mapper_3d.py --source rtsp://192.168.1.234:554/avstream/channel=1/stream=0.sdp
```

### System Features ✅
- **Real-time 2D-to-3D mapping**: Click on video feed to place 3D markers
- **Full LiDAR visualization**: Rich point cloud showing room geometry with height-based coloring
- **Interactive controls**: Save/load markers, validation statistics, clearing
- **Visual validation**: Markers appear in both video feed and 3D room model simultaneously

### Technical Specifications ✅
- **Coordinate System**: LiDAR-based world coordinates (meters)
- **Calibration Data**: Complete intrinsic + extrinsic + floor plane
- **Point Cloud**: 18,779 LiDAR points, downsampled to ~1,250 for visualization
- **Backend**: Qt5Agg matplotlib backend for cross-platform compatibility

## Future Development

With the core 3D localization system operational, potential next steps include:

1.  **Object Detection Integration:** Integrate with YOLO or similar for automated object detection + 3D localization.
2.  **ROS 2 Integration:** Send 3D coordinates to ROS 2 nodes for robot navigation.
3.  **Multiple Camera Support:** Extend system to handle multiple RTSP camera feeds.