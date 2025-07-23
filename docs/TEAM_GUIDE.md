# SCOPE Smart Building - Team Development Guide

## Quick Start for New Team Members

### 1. Setup Environment
```bash
# Clone and navigate to project
cd custom_cv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
# Test model loading
python main.py unified --test

# Show system information
python main.py info
```

### 3. Run Detection Systems
```bash
# Primary system - both models
python main.py unified

# Individual models for testing
python main.py stage1      # YOLO-World only
python main.py segformer   # SegFormer only
```

## Project Structure

```
custom_cv/
├── main.py                     # 🚀 PRIMARY ENTRY POINT
├── unified_detection.py        # Combined SegFormer + YOLO-World
├── stage1_detector.py          # YOLO-World object detection
├── requirements.txt            # All dependencies
├── README.md                   # Project overview
├── CLAUDE.md                   # Development guidelines
│
├── models/                     # Trained models
│   ├── coco_spill_detector.pt  # SegFormer model
│   └── yolov8s-world.pt        # YOLO-World model
│
├── configs/                    # Configuration files
│   ├── main_config.yaml        # Main YOLO-World config
│   └── stage1_config.yaml      # Stage 1 detection config
│
├── utils/                      # Utility modules
│   ├── mac_camera_utils.py     # Camera management
│   ├── gstreamer_utils.py      # GStreamer integration
│   └── redis_token_monitor.py  # Redis monitoring
│
├── docs/                       # Documentation
│   ├── TEAM_GUIDE.md          # This file
│   ├── SPILL_TESTING_GUIDE.md # Testing procedures
│   ├── STAGE1_STATUS.md       # Implementation status
│   └── SYNC_IMPROVEMENT.md    # Performance optimization
│
├── seg-former/                 # SegFormer module
│   ├── coco_training.py        # Model training
│   ├── webcam_spill_detection.py
│   └── test_image_inference.py
│
└── yolo-world/                 # YOLO-World module
    └── mart_building_cv/
        ├── configs/
        └── src/detection/
```

## Performance Metrics (Latest Test Results)

### Dual Model Performance
- **Total Frames Processed**: 172
- **Dual Detection Rate**: 65.1% (both models detecting simultaneously)
- **Estimated FPS**: 4.0
- **Total Processing Time**: 247.47ms per frame

### Individual Model Performance
- **SegFormer** (Spill Detection):
  - Detection Rate: 72.7%
  - Average Time: 166.59ms
  
- **YOLO-World** (Object Detection):
  - Detection Rate: 71.5%
  - Average Time: 80.88ms

## Development Workflow

### 1. Understanding the System
- **SegFormer**: Semantic segmentation for precise spill detection
- **YOLO-World**: Open-vocabulary object detection for general building safety
- **Unified System**: Runs both models on the same frame for comprehensive monitoring

### 2. Making Changes
1. **For SegFormer changes**: Work in `seg-former/` directory
2. **For YOLO-World changes**: Work in `yolo-world/` directory  
3. **For integration changes**: Modify `unified_detection.py`
4. **For utilities**: Add to `utils/` directory

### 3. Testing Changes
```bash
# Test individual components
python main.py segformer --test
python main.py stage1 --test

# Test full system
python main.py unified --test

# Run with camera for real testing
python main.py unified --camera 0
```

### 4. Configuration
- **Model paths**: Update in `configs/*.yaml` files
- **Detection prompts**: Modify in configuration files
- **MQTT settings**: Configure in YOLO-World config
- **Performance tuning**: Adjust thresholds and batch sizes

## Common Development Tasks

### Adding New Detection Categories
1. Edit `configs/main_config.yaml` or `configs/stage1_config.yaml`
2. Add new prompts to `detection_prompts` list
3. Test with `python main.py stage1`

### Improving Performance
1. Check current metrics: `python main.py info`
2. Adjust batch sizes in model configurations
3. Modify confidence thresholds
4. Profile with debug mode: `python unified_detection.py --debug`

### Adding New Models
1. Place model file in `models/` directory
2. Update configuration to reference new model
3. Test loading: `python main.py unified --test`

### Debugging Issues
1. **Enable verbose logging**: Set log level in scripts
2. **Check dependencies**: `pip install -r requirements.txt`
3. **Verify models exist**: `python main.py info`
4. **Test individual components**: Use separate mode commands

## Integration Points

### MQTT Events
- **Topic**: `smart-building/detections`
- **Format**: JSON with detection metadata
- **Enable**: `python main.py unified --mqtt`

### Redis Token Queue
- **Purpose**: Lightweight detection tokens for downstream processing
- **Configuration**: Enable in `configs/stage1_config.yaml`
- **Monitor**: Use `utils/redis_token_monitor.py`

### Camera Inputs
- **Webcam**: `python main.py unified --camera 0`
- **RTSP Streams**: Configure in stage1_detector.py
- **Multiple cameras**: Deploy multiple instances

## Deployment Considerations

### Edge Device Optimization
- **Target Hardware**: NVIDIA Jetson Orin Nano
- **Power Budget**: 20W maximum
- **Latency Target**: <2 seconds for detection
- **Accuracy Target**: 99.9% with temporal filtering

### Production Deployment
1. **Model Optimization**: Use TensorRT for edge deployment
2. **Batch Processing**: Configure optimal batch sizes
3. **Memory Management**: Monitor and optimize memory usage
4. **Error Handling**: Implement robust error recovery

## Getting Help

### Common Issues
1. **Import errors**: Check `requirements.txt` and virtual environment
2. **Model not found**: Verify models are in `models/` directory
3. **Poor performance**: Check hardware capabilities and model settings
4. **Camera issues**: Test with different camera indices

### Resources
- **Main Documentation**: `README.md` and `CLAUDE.md`
- **Testing Guide**: `docs/SPILL_TESTING_GUIDE.md`
- **Performance Analysis**: Check latest test results in code comments
- **Configuration Examples**: See `configs/*.yaml` files

### Contact
- Check git commit history for recent contributors
- Refer to CLAUDE.md for development guidelines
- Review test results and performance metrics for baseline expectations