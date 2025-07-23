# SCOPE Smart Building - Deployment Guide

## Overview

This guide covers deployment of the SCOPE unified detection system from development to production edge computing environments.

## Performance Baseline

**Validated Performance (172 frames tested):**
- **Processing Speed**: 4.0 FPS (247ms per frame)
- **Dual Detection Rate**: 65.1%
- **Individual Performance**:
  - SegFormer: 72.7% detection rate, 166ms avg
  - YOLO-World: 71.5% detection rate, 80ms avg

## Development Environment

### Local Development Setup
```bash
# 1. Clone and setup
git clone <repository>
cd custom_cv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Verify installation
python main.py info

# 3. Test individual components
python main.py segformer --test
python main.py stage1 --test
python main.py unified --test

# 4. Run with camera
python main.py unified --camera 0
```

### Development Testing
```bash
# Performance testing
python main.py unified --camera 0  # Monitor FPS and latency

# MQTT integration testing
python main.py unified --mqtt

# Configuration testing
python main.py stage1 --config configs/stage1_config.yaml
```

## Production Deployment

### Target Hardware: NVIDIA Jetson Orin Nano

**Hardware Specifications:**
- **Power Budget**: 20W maximum
- **Memory**: 8GB shared GPU/CPU
- **Storage**: 128GB+ NVMe SSD recommended
- **Compute**: 1024-core NVIDIA Ampere GPU

### Installation on Jetson

#### 1. System Preparation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y python3-pip python3-venv git cmake

# Install GStreamer (for RTSP streams)
sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-*

# Install OpenCV system dependencies
sudo apt install -y libopencv-dev python3-opencv
```

#### 2. Python Environment
```bash
# Create production environment
python3 -m venv /opt/scope-cv
source /opt/scope-cv/bin/activate

# Install PyTorch for Jetson (use appropriate version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install application dependencies
pip install -r requirements.txt
```

#### 3. Model Optimization (Optional)
```bash
# Convert models to TensorRT for maximum performance
# This step is optional but recommended for production

# Install TensorRT tools
sudo apt install -y tensorrt

# Convert SegFormer model (if available)
python -c "
import torch
from transformers import SegformerForSemanticSegmentation
model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b2-finetuned-ade-512-512')
model.load_state_dict(torch.load('models/coco_spill_detector.pt'))
# Convert to TensorRT (implementation depends on specific needs)
"
```

### Configuration for Production

#### 1. System Configuration
```bash
# Set up system service
sudo tee /etc/systemd/system/scope-detection.service << EOF
[Unit]
Description=SCOPE Smart Building Detection System
After=network.target

[Service]
Type=simple
User=scope
WorkingDirectory=/opt/scope-cv
Environment=PATH=/opt/scope-cv/bin
ExecStart=/opt/scope-cv/bin/python main.py unified --mqtt
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable scope-detection.service
sudo systemctl start scope-detection.service
```

#### 2. Performance Tuning
```bash
# Set maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Configure memory management
echo 'vm.swappiness=1' | sudo tee -a /etc/sysctl.conf

# GPU memory fraction (adjust based on requirements)
export CUDA_VISIBLE_DEVICES=0
```

#### 3. Camera Configuration
Update `configs/main_config.yaml` for production cameras:
```yaml
# Production camera configuration
cameras:
  main_entrance:
    input: "rtsp://admin:password@192.168.1.100:554/stream1"
    resolution: [1280, 720]
    fps: 15
  
  cafeteria:
    input: "rtsp://admin:password@192.168.1.101:554/stream1"
    resolution: [1280, 720]
    fps: 15

# Performance settings
performance:
  batch_size: 1  # For real-time processing
  confidence_threshold: 0.3
  max_latency_ms: 2000
```

## Network Integration

### MQTT Broker Setup
```bash
# Install and configure Mosquitto
sudo apt install -y mosquitto mosquitto-clients

# Configure MQTT
sudo tee /etc/mosquitto/conf.d/scope.conf << EOF
listener 1883
allow_anonymous true
log_type all
log_dest file /var/log/mosquitto/mosquitto.log
EOF

# Start MQTT broker
sudo systemctl enable mosquitto
sudo systemctl start mosquitto
```

### Redis Setup (Optional)
```bash
# Install Redis for token queuing
sudo apt install -y redis-server

# Configure Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Test Redis connection
redis-cli ping
```

## Multi-Camera Deployment

### Docker Deployment (Recommended)
```dockerfile
# Dockerfile for SCOPE detection
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

CMD ["python", "main.py", "unified", "--mqtt"]
```

```bash
# Build and run
docker build -t scope-detection .
docker run --runtime nvidia --device /dev/video0 \
  -v /opt/scope-config:/app/configs \
  scope-detection
```

### Multiple Instance Deployment
```bash
# Run multiple instances for different cameras
python main.py unified --camera 0 --mqtt &  # Camera 0
python main.py unified --camera 1 --mqtt &  # Camera 1
python main.py unified --camera 2 --mqtt &  # Camera 2
```

## Monitoring and Maintenance

### System Monitoring
```bash
# Monitor system resources
htop
nvidia-smi

# Monitor detection service
sudo systemctl status scope-detection.service
sudo journalctl -u scope-detection.service -f

# Monitor MQTT messages
mosquitto_sub -t 'smart-building/detections' -v
```

### Performance Monitoring
```bash
# Create monitoring script
cat << EOF > monitor_performance.py
import time
import psutil
import json
from utils.redis_token_monitor import RedisTokenMonitor

def log_performance():
    stats = {
        'timestamp': time.time(),
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'gpu_temp': # Get GPU temperature
    }
    print(json.dumps(stats))

if __name__ == "__main__":
    while True:
        log_performance()
        time.sleep(60)
EOF

python monitor_performance.py >> /var/log/scope-performance.log &
```

### Log Management
```bash
# Set up log rotation
sudo tee /etc/logrotate.d/scope << EOF
/var/log/scope-detection.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 scope scope
}
EOF
```

## Troubleshooting

### Common Issues

#### 1. Memory Issues
```bash
# Check memory usage
free -h
nvidia-smi

# Reduce batch size in configs
# Restart services
sudo systemctl restart scope-detection.service
```

#### 2. Camera Connection Issues
```bash
# Test camera directly
gst-launch-1.0 rtspsrc location=rtsp://... ! autovideosink

# Check network connectivity
ping <camera-ip>
```

#### 3. Model Loading Issues
```bash
# Verify models exist
ls -la models/

# Test model loading
python main.py unified --test

# Check disk space
df -h
```

### Performance Optimization

#### 1. Reduce Latency
- Lower input resolution (720p → 480p)
- Reduce confidence thresholds
- Use TensorRT optimization
- Increase GPU memory allocation

#### 2. Improve Accuracy
- Use higher resolution inputs
- Lower confidence thresholds
- Enable temporal filtering
- Use ensemble methods

#### 3. Scale for Multiple Cameras
- Use container orchestration (Docker Swarm/K8s)
- Load balance across multiple Jetson units
- Implement camera failover
- Use edge-cloud hybrid processing

## Security Considerations

### Network Security
```bash
# Configure firewall
sudo ufw enable
sudo ufw allow 22    # SSH
sudo ufw allow 1883  # MQTT
sudo ufw allow 6379  # Redis (if needed)
```

### Camera Security
- Use strong RTSP credentials
- Enable HTTPS for camera management
- Regular firmware updates
- Network segmentation

### System Hardening
- Regular security updates
- Disable unused services
- Monitor system logs
- Use VPN for remote access

## Backup and Recovery

### Model Backup
```bash
# Backup trained models
tar -czf scope-models-$(date +%Y%m%d).tar.gz models/
```

### Configuration Backup
```bash
# Backup configurations
tar -czf scope-configs-$(date +%Y%m%d).tar.gz configs/
```

### System Recovery
```bash
# Restore from backup
sudo systemctl stop scope-detection.service
tar -xzf scope-models-backup.tar.gz
tar -xzf scope-configs-backup.tar.gz
sudo systemctl start scope-detection.service
```

## Support and Maintenance

### Regular Maintenance Tasks
- **Weekly**: Check system logs and performance
- **Monthly**: Update system packages and security patches
- **Quarterly**: Review and update detection configurations
- **Annually**: Hardware health check and model retraining

### Escalation Procedures
1. **Level 1**: Check system logs and restart services
2. **Level 2**: Verify network connectivity and camera status
3. **Level 3**: Review model performance and retrain if needed
4. **Level 4**: Hardware replacement or architectural changes