# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is part of the **SCOPE (Smart Campus Operations & Predictive Environment)** smart building computer vision system. The project implements a **CUDA-optimized multi-model architecture** for comprehensive spill detection and building safety monitoring:

1. **BiSeNet V2 Spill Segmentation** (`bisenet/`): Real-time bilateral segmentation for precise spill boundary detection
2. **YOLOv8m Object Detection** (`yolo/`): High-accuracy building object detection for safety and maintenance

### System Context

This module serves as a **SCOPE 4.1 Layer 1 Edge Computing** component in the broader smart building architecture:
- **Edge Computing**: Optimized for NVIDIA Jetson Orin Nano deployment (20W power budget)
- **Real-time Processing**: 2s detection latency target, 99.9% accuracy goal
- **MQTT Integration**: Event publishing to building management systems
- **Two-Stage Architecture**: Stage 1 detection + Stage 2 RT-DETR verification
- **Redis Token Queue**: High-performance Stage 1→Stage 2 coordination

**Repository Status**: Production-ready CUDA architecture with specialized models for precise spill segmentation and improved building object detection.

## Architecture

The system employs a **CUDA-optimized dual-model architecture** with two-stage verification:

### Stage 1: Dual Detection (`cuda/`)
**Purpose**: Primary detection with complementary models
- `cuda_stage1_detector.py`: CUDA-optimized parallel detection
- `cuda_pipeline_coordinator.py`: Complete pipeline orchestration
- `cuda_memory_manager.py`: Memory optimization with tensor pooling

**Model Components**:
- **BiSeNet V2**: Bilateral segmentation network, 512x512 input, real-time spill segmentation with dual-path architecture
- **YOLOv8m**: 640x640 input, building-relevant object detection (COCO filtered)
- **Parallel Processing**: Simultaneous inference with CUDA streams

**Data Flow**:
- Raw camera input → Preprocessing → Dual model inference → Token generation → Redis queue
- Optimized tensor memory pooling for continuous operation
- Priority-based detection scoring for critical vs maintenance alerts

### Stage 2: Verification (`cuda/`)
**Purpose**: High-precision verification of Stage 1 detections
- `cuda_stage2_verifier.py`: RT-DETR-based verification
- `cuda_confidence_fusion.py`: Multi-model confidence scoring
- Redis token consumption with temporal filtering

**Implementation**:
- RT-DETR verification for Stage 1 object detections
- Spill region validation with geometric analysis
- Combined confidence scoring for 99.9% accuracy target

### Cross-Platform Support (`mac/`, `cuda/`)
**Development**: Mac MPS optimization for Apple Silicon development
- `mac/mac_stage1_detector.py`: MPS-optimized development version
- Device auto-detection (CUDA → MPS → CPU fallback)
- Cross-platform model compatibility

## Common Commands

### Environment Setup
```bash
# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Verify CUDA/MPS device availability
python scope_main.py info
```

### Primary Entry Point (Production)
```bash
# Complete CUDA pipeline (Stage 1 + Stage 2 + MQTT)
python scope_main.py pipeline

# Stage 1 detection only (DeepLabV3+ + YOLOv8m)
python scope_main.py stage1

# Stage 2 verification only (RT-DETR)
python scope_main.py stage2

# Test all components
python scope_main.py test

# Convert models to TensorRT
python scope_main.py convert

# System information
python scope_main.py info
```

**Production Options**:
```bash
# Custom camera source
python scope_main.py pipeline --camera "rtsp://admin:password@192.168.1.100:554/stream1"

# Custom configuration
python scope_main.py pipeline --config configs/production_config.yaml

# Test mode (load only)
python scope_main.py stage1 --test
```

### Legacy Compatibility (Maintained)
```bash
# Legacy unified detection (maintained for compatibility)
python unified_detection.py

# Individual legacy components
python stage1_detector.py --test
```

### Model Development
```bash
# Train BiSeNet V2 spill segmentation
python bisenet/train_bisenetv2.py

# Test improved object detection
python yolo/test_improved_detector.py

# Performance benchmarking
python utils/performance_benchmark.py
```

## Device Optimization

### CUDA (Production - Jetson Orin Nano)
- Target device for production deployment
- TensorRT engine conversion for maximum performance
- Memory optimization: FP16 precision, tensor pooling
- Parallel processing: CUDA streams for dual models

### MPS (Development - Apple Silicon)
- Optimized development on Mac M1/M2
- Device detection: `torch.backends.mps.is_available()`
- Automatic fallback to CPU if MPS unavailable

### CPU (Fallback)
- Reduced performance but full compatibility
- Automatic detection and configuration

## Key File Locations

### Primary Entry Points
- `scope_main.py`: **PRIMARY** - Modern CUDA-optimized entry point
- `unified_detection.py`: Legacy entry point (maintained for compatibility)
- `stage1_detector.py`: Legacy Stage 1 entry point

### CUDA Architecture (`cuda/`)
- `cuda_pipeline_coordinator.py`: Complete pipeline orchestration
- `cuda_stage1_detector.py`: Dual-model Stage 1 detection
- `cuda_stage2_verifier.py`: RT-DETR verification
- `cuda_memory_manager.py`: Memory optimization and tensor pooling
- `cuda_config.yaml`: CUDA-specific configuration

### Model Modules
- `bisenet/bisenetv2_spill_detector.py`: **NEW** - BiSeNet V2 real-time spill segmentation
- `bisenet/bisenetv2_model.py`: **NEW** - Core BiSeNet V2 architecture implementation
- `yolo/improved_object_detector.py`: **NEW** - YOLOv8m building object detection
- `yolo/yolov8m.pt`: High-accuracy object detection model
- `deeplab/deeplabv3_spill_detector.py`: Legacy DeepLabV3+ model (archived)
- `models/coco_spill_detector.pt`: Legacy SegFormer model (archived)

### Cross-Platform Development (`mac/`)
- `mac/mac_stage1_detector.py`: MPS-optimized development version
- `mac/mac_memory_manager.py`: MPS memory management
- `mac/mac_config.yaml`: Mac-specific configuration

### System Integration
- `mqtt_event_publisher.py`: Production MQTT integration with QoS and retry logic
- `redis_token_manager.py`: High-performance Redis queue management
- `utils/performance_monitor.py`: Real-time performance tracking

### Configuration
- `configs/cuda_config.yaml`: Production CUDA configuration
- `configs/mac_config.yaml`: Development Mac configuration
- `configs/stage1_config.yaml`: Legacy configuration (maintained)

### Documentation
- `CLAUDE.md`: Development guidance (this file)
- `docs/DEPLOYMENT_GUIDE.md`: Production deployment procedures

### Directory Structure
```
custom_cv/
├── scope_main.py                      # 🚀 PRIMARY: Modern CUDA entry point
├── unified_detection.py               # Legacy: Maintained for compatibility
├── stage1_detector.py                 # Legacy: Original Stage 1
├── requirements.txt                   # All dependencies
│
├── cuda/                              # CUDA-optimized production
│   ├── cuda_pipeline_coordinator.py   # Complete pipeline
│   ├── cuda_stage1_detector.py        # Dual-model Stage 1
│   ├── cuda_stage2_verifier.py        # RT-DETR verification
│   ├── cuda_memory_manager.py         # Memory optimization
│   └── cuda_config.yaml              # CUDA configuration
│
├── mac/                               # MPS-optimized development
│   ├── mac_stage1_detector.py         # MPS Stage 1
│   ├── mac_memory_manager.py          # MPS memory management
│   └── mac_config.yaml               # Mac configuration
│
├── bisenet/                           # BiSeNet V2 spill segmentation
│   ├── bisenetv2_spill_detector.py    # NEW: Real-time segmentation model
│   ├── bisenetv2_model.py             # NEW: Core BiSeNet V2 architecture
│   ├── train_bisenetv2.py             # CUDA-optimized training script
│   ├── dataset.py                     # COCO dataset loader
│   └── losses.py                      # Loss functions (CrossEntropy + Dice)
│
├── deeplab/                           # Legacy DeepLabV3+ (archived)
│   ├── deeplabv3_spill_detector.py    # Archived: DeepLabV3+ implementation
│   └── train_deeplabv3.py            # Archived: Training script
│
├── yolo/                              # Improved object detection
│   ├── improved_object_detector.py    # NEW: YOLOv8m detector
│   ├── yolov8m.pt                    # High-accuracy model
│   └── test_improved_detector.py     # Testing utilities
│
├── configs/                           # Configuration files
│   ├── cuda_config.yaml              # Production CUDA
│   ├── mac_config.yaml               # Development Mac
│   └── stage1_config.yaml            # Legacy configuration
│
├── models/                            # Model files
│   ├── coco_spill_detector.pt         # Legacy SegFormer (archived)
│   └── yolov8s-world.pt              # Legacy YOLO-World (archived)
│
├── utils/                             # Utility modules
│   ├── performance_monitor.py         # Performance tracking
│   ├── camera_utils.py               # Camera management
│   └── redis_token_monitor.py        # Redis monitoring
│
├── docs/                              # Documentation
│   └── DEPLOYMENT_GUIDE.md           # Production deployment
│
├── mqtt_event_publisher.py           # MQTT integration
├── redis_token_manager.py            # Redis management
└── seg-former/                       # Legacy: Archived SegFormer module
```

## Model Configuration

### BiSeNet V2 Spill Segmentation (Current)
- Base model: BiSeNet V2 with bilateral segmentation architecture
- Input size: 512x512
- Output: Binary segmentation (background/spill)
- Architecture: Dual-path design for speed and accuracy balance
- Optimization: Custom trained on spill dataset with CUDA acceleration
- Performance: Real-time inference with superior boundary detection

### YOLOv8m Object Detection (Current)
- Base model: YOLOv8m (medium variant for accuracy/speed balance)
- Input size: 640x640
- Classes: Filtered COCO classes relevant to building environment
- Priority scoring: Critical alerts (fire safety) vs maintenance (waste management)
- Performance: Superior accuracy compared to YOLOv8n-world

### Legacy Models (Archived)
- DeepLabV3+: Replaced by BiSeNet V2 for faster real-time performance
- SegFormer-b2: Replaced by BiSeNet V2 for better segmentation accuracy
- YOLOv8s-world: Replaced by YOLOv8m for improved accuracy

## Training Configuration

### BiSeNet V2 Training
- Architecture: Bilateral segmentation with detail and semantic paths
- Task: 2-class segmentation (background, spill)
- Loss function: Combined CrossEntropyLoss + Dice loss for better boundary detection
- Optimization: AdamW with cosine annealing schedule
- Data augmentation: Horizontal flip, rotation, scaling, color jitter, cutout
- Hardware: CUDA-optimized for NVIDIA RTX 4070 (8GB VRAM)
- Mixed precision: FP16 for memory efficiency and speed

### YOLOv8m Configuration
- Pre-trained: COCO 80-class detection
- Class filtering: Building-relevant subset (person, bottles, chairs, safety equipment)
- Priority mapping: Critical safety alerts vs routine maintenance
- Confidence thresholds: Adaptive based on object category

## Inference Capabilities

### CUDA Pipeline (Production)
**Complete Two-Stage Processing**:
- Input: Camera streams (RTSP, webcam, file)
- Stage 1: BiSeNet V2 spill segmentation + YOLOv8m object detection
- Stage 2: RT-DETR verification of Stage 1 detections
- Output: MQTT events with confidence fusion scores
- Performance: <2s latency, 99.9% accuracy target

**Optimizations**:
- CUDA streams for parallel processing
- Tensor memory pooling for continuous operation
- TensorRT engine conversion for maximum speed
- Half precision (FP16) for memory efficiency

### Individual Model Testing
**BiSeNet V2 Spill Detection**:
- Input: Any image format
- Processing: 512x512 bilateral segmentation with dual-path inference
- Output: Real-time spill regions with confidence scores and precise boundary detection
- Visualization: Red overlay on detected spill areas with improved edge accuracy

**YOLOv8m Object Detection**:
- Input: Images or video streams
- Processing: 640x640 inference with building-specific filtering
- Output: Bounding boxes with priority-based scoring
- Categories: Safety equipment, waste management, personal items

### Cross-Platform Development
**Mac MPS Development**:
- Optimized inference on Apple Silicon
- Same model architectures with MPS acceleration
- Perfect compatibility for development and testing
- Auto-fallback to CPU if MPS unavailable


## SCOPE Smart Building Integration

### System Architecture Position
This module serves as **SCOPE 4.1 Layer 1 Edge Computing** in the smart building ecosystem:
- **Input**: NVR camera streams and building surveillance systems
- **Processing**: Real-time AI inference on Jetson Orin Nano edge devices
- **Output**: MQTT events and Redis tokens for building management
- **Integration**: Two-stage verification with autonomous robotic response

### Camera Integration
**Production Camera Support**:
- **RTSP/ONVIF**: Standard IP camera protocols
- **Multi-camera**: Scalable to multiple camera feeds per edge device
- **Resolution**: 720p+ for optimal accuracy/performance balance
- **Network Resilience**: Auto-reconnection and failover capabilities

### MQTT Communication
- **Topic Structure**: `smart-building/detections/{camera_id}`
- **Event Types**: Spill detection, safety hazards, maintenance alerts
- **Format**: JSON with detection metadata, confidence scores, timestamps
- **QoS**: Configurable quality of service with retry logic
- **Broker**: Eclipse Mosquitto (production deployment)

### Redis Token Management
- **Purpose**: High-performance Stage 1→Stage 2 coordination
- **Queue Management**: Priority-based token distribution
- **Performance**: Sub-millisecond token operations
- **Scalability**: Multi-camera token stream handling
- **Persistence**: Configurable data persistence for reliability

### Performance Targets (Production)
- **Detection Latency**: <2 seconds end-to-end
- **Accuracy**: 99.9% with two-stage verification
- **Throughput**: 5 cameras per Jetson Orin Nano
- **Power Consumption**: <20W per edge device
- **Uptime**: 99.9% availability with automatic recovery

## Development Notes

### Architecture Evolution (Completed)
- **Phase 1**: Original SegFormer + YOLO-World architecture
- **Phase 2**: User feedback identified spill detection issues
- **Phase 3**: Replacement with DeepLabV3+ segmentation approach
- **Phase 4**: Upgrade to YOLOv8m for improved object detection accuracy
- **Phase 5**: Migration to BiSeNet V2 for real-time performance
- **Current**: CUDA-optimized multi-model with two-stage verification

### Model Selection Rationale
- **BiSeNet V2 over DeepLabV3+**: Superior real-time performance with bilateral architecture
- **YOLOv8m over YOLOv8n-world**: Superior accuracy for building object detection
- **Two-stage verification**: Meets 99.9% accuracy requirement with RT-DETR confirmation
- **CUDA optimization**: Required for real-time edge deployment
- **Bilateral segmentation**: Optimal balance between speed and boundary accuracy

### Cross-Platform Development Strategy
- **Development**: Mac MPS for rapid iteration and testing
- **Production**: CUDA for Jetson Orin Nano deployment
- **Compatibility**: Shared model architectures with device-specific optimization
- **Testing**: Comprehensive cross-platform validation

### Memory Optimization
- **Tensor Pooling**: Pre-allocated memory pools for continuous operation
- **Half Precision**: FP16 for memory efficiency without accuracy loss
- **CUDA Streams**: Parallel processing for maximum GPU utilization
- **Garbage Collection**: Optimized memory cleanup for long-running operation

## Common Issues and Solutions

### Jetson CUDA Setup & Troubleshooting

#### **Issue: CUDA Not Available on Jetson (`cuda_available: False`)**

**Symptoms:**
- PyTorch reports `torch.cuda.is_available() = False`
- Very poor performance (2 FPS, 300+ ms inference time)
- GPU usage shows 0%
- System uses CPU for inference

**Diagnosis Commands:**
```bash
# 1. Check if NVIDIA GPU is detected
nvidia-smi

# 2. Check CUDA installation
nvcc --version

# 3. Check PyTorch CUDA support
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# 4. Check JetPack version
sudo apt show nvidia-jetpack

# 5. Check device tree model
cat /proc/device-tree/model
```

**Solutions:**

**1. Fix PyTorch Installation (Most Common Issue)**
```bash
# Uninstall current PyTorch (likely CPU-only version)
pip uninstall torch torchvision torchaudio

# For JetPack 5.x (Jetson Orin series):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Alternative: Use NVIDIA's pre-built wheels for Jetson
# Check https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 for latest wheels
wget https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
pip install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
```

**2. Set CUDA Environment Variables**
```bash
# Add to ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64' >> ~/.bashrc
source ~/.bashrc
```

**3. Install Jetson Stats for Monitoring**
```bash
pip install jetson-stats
sudo reboot  # Required after jetson-stats installation
```

**4. Verify Installation**
```bash
# After reboot, test CUDA availability
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Count:', torch.cuda.device_count()); print('Current Device:', torch.cuda.current_device() if torch.cuda.is_available() else 'None')"

# Test jtop monitoring
jtop
```

**Performance Expectations After Fix:**
- **FPS**: Should improve from 2-3 FPS to 15-25+ FPS
- **Inference Time**: Should drop from 300-400ms to 30-50ms
- **GPU Usage**: Should show 70-90% GPU utilization
- **Memory**: Should use GPU memory instead of system RAM

**5. Jetson Power Mode Optimization**
```bash
# Set to MAXN mode for maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Check current power mode
sudo nvpmodel -q
```

#### **Issue: Thermal Throttling on Jetson**

**Symptoms:**
- Performance degrades over time
- Temperature >70°C
- GPU frequency scaling down

**Solutions:**
```bash
# Monitor thermal status
jtop  # Look for temperature and throttling indicators

# Improve cooling
# - Ensure proper ventilation
# - Consider active cooling fan
# - Check thermal paste on heatsink

# Reduce workload if necessary
# - Use fast mode more frequently
# - Lower input resolution
# - Reduce frame rate
```

#### **Issue: Memory Issues on Jetson**

**Symptoms:**
- Out of memory errors
- System becomes unresponsive
- Swap usage high

**Solutions:**
```bash
# Check memory usage
free -h
nvidia-smi  # For GPU memory

# Optimize memory settings in code:
# - Enable mixed precision (FP16)
# - Reduce batch sizes
# - Clear CUDA cache regularly
```

### Model Loading
- **Issue**: Model compatibility errors
- **Solution**: Check model architecture matches expected format
- **Debug**: Use `scope_main.py test` to validate all components

### Performance Optimization
- **Issue**: High latency or low FPS
- **Solution**: Enable GPU acceleration, adjust input resolution
- **Debug**: Monitor memory usage and device utilization

### Camera Integration
- **Issue**: RTSP connection failures
- **Solution**: Verify network connectivity and camera credentials
- **Debug**: Test camera directly with GStreamer or VLC

### MQTT/Redis Connectivity
- **Issue**: Event publishing failures
- **Solution**: Check broker/server availability and configuration
- **Debug**: Use mosquitto_sub and redis-cli for testing

## Production Deployment

### Jetson Orin Nano Setup
1. **System Preparation**: Ubuntu 20.04, CUDA toolkit, TensorRT
2. **Python Environment**: Python 3.8+, virtual environment setup
3. **Model Optimization**: TensorRT engine conversion for maximum performance
4. **Service Configuration**: systemd service for automatic startup
5. **Monitoring**: Performance monitoring and log management

### Configuration Management
- **Environment Variables**: Device-specific settings
- **YAML Configuration**: Model parameters and network settings
- **Secrets Management**: Secure camera credentials and MQTT keys
- **Version Control**: Configuration versioning for deployment tracking

### Scaling Strategy
- **Multi-camera**: Multiple camera feeds per edge device
- **Multi-device**: Load balancing across multiple Jetson units
- **Cloud Integration**: Optional cloud analytics and model updates
- **Edge-Cloud Hybrid**: Local processing with cloud coordination

## Team Development Guidelines

### Setup for New Team Members
1. Clone repository and create virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Verify installation: `python scope_main.py info`
4. Test components: `python scope_main.py test`
5. Run development version: `python scope_main.py stage1 --test`

### Development Workflow
1. **Mac Development**: Use MPS-optimized components for rapid iteration
2. **Testing**: Cross-platform testing before production deployment
3. **Integration**: Test MQTT and Redis integration locally
4. **Deployment**: CUDA optimization and Jetson deployment

### Code Organization
- **Platform-Specific**: Separate cuda/ and mac/ implementations
- **Model-Specific**: Dedicated deeplab/ and yolo/ modules
- **Shared Utilities**: Common components in utils/
- **Configuration**: Environment-specific configs in configs/

### Performance Monitoring
- **Metrics**: Latency, accuracy, throughput, memory usage
- **Logging**: Structured logging with appropriate levels
- **Profiling**: Performance profiling for optimization
- **Alerting**: Automated alerts for performance degradation

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.