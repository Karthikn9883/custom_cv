# BiSeNet V2 - Real-time Spill Segmentation

This module implements BiSeNet V2 (Bilateral Segmentation Network) for real-time spill detection in the SCOPE smart building system.

## Architecture

BiSeNet V2 uses a bilateral structure with two paths:
- **Detail Path**: Captures low-level spatial details
- **Semantic Path**: Captures high-level semantic information
- **Aggregation Layer**: Fuses information from both paths

## Key Features

- **Real-time Performance**: Optimized for speed without sacrificing accuracy
- **CUDA Acceleration**: Full GPU optimization for NVIDIA RTX 4070
- **Mixed Precision**: FP16 training for memory efficiency
- **Superior Boundaries**: Better edge detection than DeepLabV3+

## Files

- `bisenetv2_model.py`: Core BiSeNet V2 architecture
- `bisenetv2_spill_detector.py`: Inference wrapper for spill detection
- `train_bisenetv2.py`: CUDA-optimized training script
- `dataset.py`: COCO dataset loader for spill data
- `losses.py`: Loss functions (CrossEntropy + Dice)

## Usage

```python
from bisenet import BiSeNetV2SpillDetector

# Initialize detector
detector = BiSeNetV2SpillDetector()

# Detect spills
results = detector.detect_spills(frame)
```

## Training

```bash
# Train BiSeNet V2 on spill dataset
python bisenet/train_bisenetv2.py --config configs/bisenetv2_config.yaml
```