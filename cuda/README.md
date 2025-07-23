# CUDA Module

This folder contains CUDA-optimized components for the SCOPE smart building system.

## Files:
- `cuda_memory_manager.py` - Advanced memory management for CUDA/MPS/CPU
- `cuda_stage1_detector.py` - Ultra-fast Stage 1 detection with dual models
- `cuda_stage2_verifier.py` - RT-DETR-based verification with temporal filtering  
- `cuda_pipeline_coordinator.py` - Complete pipeline orchestration
- `tensorrt_converter.py` - TensorRT engine conversion utilities
- `test_cuda_compatibility.py` - CUDA compatibility testing
- `test_stage1_performance.py` - Performance benchmarking

## Purpose:
Provides CUDA-accelerated inference for maximum performance on NVIDIA GPUs with fallback support for MPS (Apple Silicon) and CPU devices.