# Enhanced 2D-3D Mapping System Guide

## Overview

This enhanced system provides a complete workflow for high-precision 2D-to-3D mapping using RTSP camera feeds and LiDAR point clouds, optimized for Windows with NVIDIA GPU acceleration.

## System Requirements

### Hardware Requirements
- **Windows 10/11** (optimized for Windows)
- **NVIDIA GPU** (RTX 4070 recommended) with CUDA support
- **32GB RAM** (for large point cloud processing)
- **DVR with RTSP capability** for camera feeds

### Software Dependencies
```bash
pip install opencv-python numpy matplotlib trimesh scikit-learn pyyaml open3d psutil
pip install nvidia-ml-py3  # For GPU monitoring (optional)
```

## Enhanced Features ✨

### 1. Camera Position Visualization
- **Real-time camera frustum display** in 3D space
- **Field of view visualization** with adjustable parameters
- **Camera direction vectors** overlaid on LiDAR scan
- **Interactive camera positioning tool**

### 2. Performance Optimization
- **Windows/NVIDIA GPU optimizations** for RTX 4070
- **RTSP stream optimization** for low latency
- **Memory management** for 32GB RAM systems
- **Multi-threaded processing** with CPU core optimization

### 3. Enhanced Calibration
- **Visual feedback** during extrinsic calibration
- **Quality assessment** with reprojection error analysis
- **Automatic calibration validation**
- **Detailed calibration reports**

### 4. Advanced Marker Management
- **Distance measurements** between markers
- **Persistent marker storage** and loading
- **Enhanced 3D visualization** with height-based coloring
- **Real-time validation statistics**

## Complete Workflow

### Step 1: System Validation
Test your complete system setup:

```bash
cd C:\Users\Karth\Desktop\projects\custom_cv
python mart_building_cv\scripts\system_test_validator.py --source rtsp://192.168.1.234:554/avstream/channel=1/stream=0.sdp --save-report
```

**Expected Output:**
- ✅ System requirements check
- ✅ RTSP connectivity validation  
- ✅ Calibration file verification
- ✅ Performance optimization status
- 📊 Comprehensive health report

### Step 2: Camera Placement (New!)
Interactively position your camera marker in 3D space:

```bash
python mart_building_cv\scripts\camera_placement_tool.py --source rtsp://192.168.1.234:554/avstream/channel=1/stream=0.sdp
```

**Interactive Controls:**
- `W/S/A/D` - Move camera position
- `Q/E` - Move up/down
- `I/K/J/L` - Adjust camera rotation
- `+/-/[]` - Adjust field of view
- `V` - Save camera placement
- `F/C` - Fine/coarse adjustment modes

### Step 3: Camera Intrinsic Calibration
Calibrate camera's internal parameters:

```bash
python mart_building_cv\scripts\calibrate_camera.py
```

### Step 4: Camera Extrinsic Calibration (Enhanced!)
Calibrate camera position with enhanced feedback:

```bash
python mart_building_cv\scripts\calibrate_extrinsics.py
```

**Enhanced Features:**
- 🎯 Visual point selection feedback
- 📊 Real-time reprojection error analysis  
- ✅ Automatic quality assessment
- 📝 Detailed calibration reports
- 🔄 Undo/redo functionality (`u`, `r`)

### Step 5: Interactive 2D-3D Mapping (Enhanced!)
Run the complete interactive mapping system:

```bash
python mart_building_cv\scripts\interactive_mapper_3d.py --source rtsp://192.168.1.234:554/avstream/channel=1/stream=0.sdp
```

## New Keyboard Controls 🎮

### Interactive Mapper Controls
- **Click video** - Place 3D markers
- `c` - Clear all markers  
- `s` - Save markers to file
- `l` - Load markers from file
- `p` - Save 3D plot as image
- `v` - Show validation statistics
- `i` - Show calibration info
- **`k` - Show camera position info** ✨
- **`t` - Toggle camera visualization** ✨
- **`d` - Measure distance between last two markers** ✨
- **`o` - Show performance/system info** ✨
- `h` - Show help
- `q` - Quit

### Camera Placement Tool Controls
- `W/S/A/D` - Move camera position
- `Q/E` - Move up/down
- `I/K/J/L/U/O` - Rotate camera (pitch/yaw/roll)
- `+/-` - Adjust horizontal FOV
- `[/]` - Adjust vertical FOV
- `R` - Reset to default position
- `F` - Fine adjustment mode
- `C` - Coarse adjustment mode
- `V` - Save camera placement
- `B` - Load camera placement
- `ESC` - Exit

## Configuration Options

### Main Configuration (main_config.yaml)
```yaml
# RTSP Stream Configuration
input_source: "rtsp://192.168.1.234:554/avstream/channel=1/stream=0.sdp"

# Performance Settings
performance_optimization:
  enable_gpu_acceleration: true
  max_ram_usage_gb: 16
  cpu_threads: 8
  rtsp_buffer_size: 1

# Camera Visualization
camera_visualization:
  show_frustum: true
  show_direction: true
  frustum_length: 2.0
  frustum_alpha: 0.2
```

### Command Line Options

#### Interactive Mapper
```bash
python interactive_mapper_3d.py [OPTIONS]
--source TEXT           # Video source (RTSP URL, camera index, or file)
--config-dir TEXT       # Directory containing calibration config files  
--no-3d                 # Disable 3D visualization
--no-camera             # Disable camera position visualization
--no-optimizations      # Disable performance optimizations
```

#### System Test Validator
```bash
python system_test_validator.py [OPTIONS]
--source TEXT           # RTSP URL to test
--config-dir TEXT       # Configuration directory
--save-report           # Save detailed test report to file
```

## Performance Optimization Features 🚀

### Windows & NVIDIA RTX 4070 Optimizations
- **CUDA acceleration** for OpenCV operations
- **GPU memory pooling** for efficient processing
- **High-priority process** scheduling
- **Optimized RTSP decoding** with FFmpeg backend
- **Multi-threaded operations** utilizing all CPU cores

### Memory Management (32GB RAM)
- **Intelligent point cloud downsampling** based on available RAM
- **Optimized video buffering** for low latency
- **Memory usage monitoring** and automatic adjustment
- **Large dataset support** for detailed LiDAR scans

### Real-time Performance
- **Frame-rate optimization** for smooth video playback
- **Background processing** for 3D calculations
- **Minimal UI latency** with threaded operations
- **Automatic quality scaling** based on performance

## Troubleshooting 🔧

### RTSP Connection Issues
1. **Test connectivity:** Run system validator first
2. **Check network:** Ensure DVR is accessible
3. **Firewall settings:** Allow Python/OpenCV through firewall
4. **RTSP URL format:** Verify correct URL format

### Calibration Problems
1. **Point selection:** Choose well-distributed, clearly visible points
2. **Lighting conditions:** Ensure good lighting for both 2D and 3D views
3. **Camera stability:** Keep camera position fixed during calibration
4. **Reprojection errors:** Use quality assessment to verify accuracy

### Performance Issues
1. **GPU drivers:** Ensure latest NVIDIA drivers installed
2. **CUDA version:** Verify CUDA compatibility with OpenCV
3. **RAM usage:** Monitor memory usage for large point clouds
4. **Background processes:** Close unnecessary applications

### 3D Visualization Problems
1. **Matplotlib backend:** Try different backends (Qt5Agg, TkAgg)
2. **OpenGL support:** Ensure graphics drivers support OpenGL
3. **Display scaling:** Check Windows display scaling settings

## Quality Assurance 📊

### Calibration Quality Metrics
- **Reprojection error < 2.0 pixels:** Excellent
- **Reprojection error < 5.0 pixels:** Good  
- **Reprojection error > 5.0 pixels:** Poor (recalibrate)

### System Health Monitoring
- **Automatic validation** on startup
- **Performance metrics** monitoring
- **Error detection** and reporting
- **Quality recommendations**

## Advanced Features 🔬

### Distance Measurements
- **3D world distance** between any two markers
- **2D pixel distance** correlation
- **Real-world scale** calculation (mm per pixel)
- **Accuracy validation** with known measurements

### Camera Positioning
- **Interactive 3D placement** before calibration
- **Field of view adjustment** for different lenses
- **Position validation** against LiDAR scan
- **Saved configurations** for different setups

### Performance Monitoring
- **Real-time FPS** monitoring
- **GPU utilization** tracking
- **Memory usage** optimization
- **Network latency** measurement

## Best Practices 💡

### For Best Accuracy
1. **Use 6-8 calibration points** distributed across the view
2. **Choose points at different depths** for better 3D calibration
3. **Ensure stable lighting** during calibration process
4. **Verify camera intrinsics** before extrinsic calibration

### For Best Performance  
1. **Enable all optimizations** (default)
2. **Use wired network** for RTSP streams
3. **Close unnecessary applications** during operation
4. **Monitor GPU temperature** during extended use

### For Reliable Operation
1. **Test system** before each calibration session
2. **Save calibration backups** regularly
3. **Document camera positions** and settings
4. **Validate accuracy** with known measurements

## Support & Validation 📞

Use the system test validator to diagnose issues:
```bash
python system_test_validator.py --save-report
```

The validator provides:
- ✅ **Comprehensive system health check**
- 📊 **Performance benchmarks**  
- 🔧 **Specific troubleshooting recommendations**
- 📝 **Detailed diagnostic reports**

---

## Quick Start Summary

1. **Test System:** `python system_test_validator.py`
2. **Place Camera:** `python camera_placement_tool.py`  
3. **Calibrate:** `python calibrate_extrinsics.py`
4. **Map:** `python interactive_mapper_3d.py`

Your enhanced 2D-3D mapping system is now ready for high-precision operation! 🎯