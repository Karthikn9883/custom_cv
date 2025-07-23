# SCOPE SegFormer Spill Detection Testing Guide

## Overview

This guide provides comprehensive instructions for testing the SegFormer spill detection system with your Mac camera. The system can detect liquid spills in real-time and is designed for smart building applications.

## Quick Start

### 1. Run Individual SegFormer Testing
```bash
# Basic Mac camera spill detection
python mac_segformer_tester.py

# With debug mode and custom threshold
python mac_segformer_tester.py --debug --threshold 50

# Using custom model
python mac_segformer_tester.py --model path/to/your/model.pt
```

### 2. Run Dual Model Testing (SegFormer + YOLO-World)
```bash
# Combined spill and object detection
python dual_model_mac_tester.py

# With debug mode for comprehensive logging
python dual_model_mac_tester.py --debug
```

### 3. Run Unified Detection System
```bash
# Production-ready unified system
python unified_detection.py

# With MQTT event publishing
python unified_detection.py --mqtt

# Test model loading only
python unified_detection.py --test
```

## Spill Simulation Methods

### Method 1: Real Liquid Spills (Recommended)
**Best for realistic testing and validation**

**Materials Needed:**
- Water in small containers
- Food coloring (red, blue, green)
- Paper towels for cleanup
- Various liquid containers (cups, bottles)

**Procedure:**
1. **Water Spills**: Pour small amounts of water on flat surfaces
2. **Colored Water**: Add food coloring for better contrast
3. **Different Shapes**: Create circular, elongated, and irregular spills
4. **Volume Testing**: Test small (few drops) to large (cup-sized) spills

**Safety Notes:**
- Use waterproof surfaces
- Keep electronics away from liquids
- Have cleaning materials ready

### Method 2: Digital Spill Images
**Safe alternative for initial testing**

**Materials Needed:**
- Tablet, laptop, or large phone screen
- Spill images (download from internet)
- Various lighting conditions

**Procedure:**
1. **Image Sources**: 
   - Search "oil spill", "water spill", "liquid spill" images
   - Use COCO dataset sample images
   - Create custom spill graphics
2. **Display Testing**:
   - Show images on bright screens
   - Test different screen sizes
   - Vary screen brightness and contrast
3. **Movement Testing**:
   - Slowly move device with spill image
   - Test different viewing angles

### Method 3: Drawn/Painted Spills
**Good for controlled testing**

**Materials Needed:**
- White paper or cardboard
- Colored markers, paints, or crayons
- Various drawing tools

**Procedure:**
1. **Shape Variety**: Draw irregular blob shapes
2. **Color Testing**: Use different colors (dark blue, red, brown)
3. **Size Range**: Small spots to large areas
4. **Texture Simulation**: Add shading and highlights

### Method 4: Material-Based Simulation
**Creative alternatives**

**Materials:**
- Aluminum foil (crumpled and flattened)
- Colored paper cutouts
- Reflective materials
- Plastic wrap with colored backing

## Testing Scenarios

### Scenario 1: Basic Detection Validation
**Goal**: Verify the model can detect obvious spills

1. Start `mac_segformer_tester.py`
2. Create a medium-sized water spill (2-3 inches diameter)
3. Position spill in camera center
4. Verify:
   - Red overlay appears on spill
   - Yellow contours outline the spill
   - Detection percentage > 1%
   - Console shows "SPILL DETECTED!"

### Scenario 2: Sensitivity Testing
**Goal**: Find detection thresholds and limits

1. Start with debug mode: `python mac_segformer_tester.py --debug`
2. Test progressively smaller spills
3. Use +/- keys to adjust threshold
4. Document minimum detectable spill size
5. Check debug files for analysis

### Scenario 3: False Positive Testing
**Goal**: Ensure model doesn't detect non-spills

1. Test common objects that might confuse the model:
   - Dark colored objects (black clothing, shoes)
   - Reflective surfaces (mirrors, metal)
   - Shadows and lighting changes
   - Water bottles, cups (without spills)

### Scenario 4: Real-World Conditions
**Goal**: Test in realistic environments

1. **Lighting Variations**:
   - Bright daylight
   - Indoor lighting
   - Low light conditions
   - Mixed lighting (window + lamp)

2. **Background Complexity**:
   - Clean floors
   - Patterned surfaces
   - Cluttered backgrounds
   - Multiple objects in view

3. **Movement Testing**:
   - Static spills
   - Moving camera
   - Changing viewing angles

### Scenario 5: Performance Validation
**Goal**: Verify real-time performance

1. Monitor FPS in the interface
2. Check inference times (should be < 100ms on Mac)
3. Test continuous operation (10+ minutes)
4. Verify no memory leaks or performance degradation

## Interactive Controls Reference

### Mac SegFormer Tester Controls
- **SPACE**: Pause/resume detection
- **+/-**: Adjust spill detection threshold (sensitivity)
- **m**: Toggle mask visualization window
- **o**: Toggle spill overlay on/off
- **c**: Toggle spill contours on/off
- **s**: Save current frame and analysis (debug mode)
- **ESC**: Quit application

### Dual Model Tester Controls
- **SPACE**: Pause/resume detection
- **s**: Toggle SegFormer spill detection on/off
- **y**: Toggle YOLO-World object detection on/off
- **+/-**: Adjust spill overlay opacity
- **d**: Save debug information manually
- **ESC**: Quit application

### Unified Detection System Controls
- **SPACE**: Toggle detection on/off
- **S**: Toggle SegFormer spill overlay
- **Y**: Toggle YOLO-World bounding boxes
- **+/-**: Adjust spill overlay transparency
- **ESC**: Quit application

## Understanding Detection Output

### SegFormer Spill Detection
- **Red Overlay**: Areas detected as spills
- **Yellow Contours**: Spill boundaries
- **Coverage Percentage**: Portion of frame covered by spills
- **Pixel Count**: Number of pixels classified as spill
- **Threshold**: Minimum pixels needed to trigger detection

### Performance Metrics
- **Inference Time**: Processing time per frame (ms)
- **FPS**: Frames processed per second
- **Detection Rate**: Percentage of frames with spills detected

## Troubleshooting

### Common Issues

#### 1. No Detection on Obvious Spills
**Solutions:**
- Lower threshold with `-` key
- Ensure good lighting
- Try colored liquids for better contrast
- Check if model loaded correctly

#### 2. Too Many False Positives
**Solutions:**
- Increase threshold with `+` key
- Improve lighting conditions
- Remove reflective objects from view
- Use cleaner backgrounds

#### 3. Poor Performance
**Solutions:**
- Close other applications
- Ensure camera resolution is appropriate
- Check device temperature (thermal throttling)
- Restart application

#### 4. Camera Issues
**Solutions:**
- Try different camera index: `--camera 1`
- Check camera permissions in macOS
- Restart camera application
- Verify camera works in other apps

### Debug Information

When running with `--debug` flag, the system saves:
- **Original frames**: `frame_*.jpg`
- **Spill masks**: `mask_*.jpg`
- **Analysis data**: `analysis_*.json`
- **Annotated frames**: `dual_annotated_*.jpg` (dual model)

Check these files to understand detection behavior.

## Expected Results

### Good Detection Performance
- Detects spills > 1 inch diameter consistently
- Minimal false positives on clean backgrounds
- Inference time < 100ms on Mac M1/M2
- Stable performance over extended periods

### Typical Metrics
- **Accuracy**: 90%+ on obvious spills
- **FPS**: 10-15 FPS on Mac camera
- **Latency**: 50-80ms inference time
- **Sensitivity**: Adjustable from 10-10000 pixels

## Advanced Testing

### Custom Model Testing
```bash
# Test with your own trained model
python mac_segformer_tester.py --model path/to/custom_model.pt

# Test on static images
python seg-former/test_image_inference.py --image spill_test.jpg
```

### Integration Testing
```bash
# Test MQTT event publishing
python unified_detection.py --mqtt

# Test with custom YOLO configuration
python unified_detection.py --yolo-config custom_config.yaml
```

### Performance Benchmarking
```bash
# Run for specific duration with statistics
python dual_model_mac_tester.py --debug
# Let run for 5+ minutes, then check statistics output
```

## Next Steps

After successful testing:
1. **Deploy to edge devices** (Jetson Orin Nano)
2. **Integrate with building management systems**
3. **Configure MQTT for real-time alerting**
4. **Set up multi-camera monitoring**
5. **Implement temporal filtering for production use**

For production deployment, see the main README.md and CLAUDE.md documentation.