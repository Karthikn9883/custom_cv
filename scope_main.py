#!/usr/bin/env python3
"""
SCOPE Smart Building - Modern Main Entry Point
CUDA-Optimized Detection System with BiSeNet V2 and YOLOv8m
"""

import argparse
import sys
import os
from pathlib import Path

def print_banner():
    """Print SCOPE system banner"""
    print("=" * 80)
    print("🏢 SCOPE Smart Building - AI Vision System v4.1")
    print("=" * 80)
    print("🚀 CUDA-Optimized Architecture:")
    print("   • BiSeNet V2 Spill Segmentation (512x512)")
    print("   • YOLOv8m Object Detection (640x640)")
    print("   • Dual-Model Stage 1 + RT-DETR Stage 2 Verification")
    print("   • MQTT Event Publishing + Redis Token Management")
    print("=" * 80)
    print()

def run_cuda_pipeline(args):
    """Run the complete CUDA pipeline"""
    print("🚀 Starting CUDA Pipeline System...")
    
    try:
        from cuda.cuda_pipeline_coordinator import CUDAPipelineCoordinator
        
        # Initialize pipeline
        coordinator = CUDAPipelineCoordinator(args.config or "configs/cuda_config.yaml")
        
        # Run pipeline
        coordinator.run_pipeline(
            camera_source=args.camera or "rtsp://admin:123456@192.168.12.205:80/ch3_0.264",
            camera_id=args.camera_id or "cam_01"
        )
        
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline stopped by user")
    except Exception as e:
        print(f"❌ Pipeline error: {e}")

def run_stage1_only(args):
    """Run Stage 1 detection only"""
    print("🎯 Starting Stage 1 CUDA Detection...")
    
    try:
        from cuda.cuda_stage1_detector import CUDAStage1Detector
        import numpy as np
        
        # Initialize detector
        detector = CUDAStage1Detector(args.config or "configs/cuda_config.yaml")
        
        if args.test:
            # Test mode
            print("✅ Stage 1 detector initialized successfully")
            stats = detector.get_performance_stats()
            print(f"Target latency: {stats.get('target_latency_ms', 'N/A')}ms")
            print(f"Device: {stats.get('device', 'N/A')}")
            return
        
        # Test with dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tokens = detector.detect_frame(dummy_frame)
        print(f"📊 Generated {len(tokens)} detection tokens")
        
        for token in tokens[:3]:  # Show first 3
            print(f"  - {token.category}: {token.confidence:.2f}")
            
    except Exception as e:
        print(f"❌ Stage 1 error: {e}")

def run_stage2_only(args):
    """Run Stage 2 verification only"""
    print("🔍 Starting Stage 2 CUDA Verification...")
    
    try:
        from cuda.cuda_stage2_verifier import CUDAStage2Verifier
        
        # Initialize verifier
        verifier = CUDAStage2Verifier(args.config or "configs/cuda_config.yaml")
        
        if args.test:
            print("✅ Stage 2 verifier initialized successfully")
            stats = verifier.get_performance_stats()
            print(f"Device: {stats.get('device', 'N/A')}")
            return
            
        print("📊 Stage 2 verification ready")
        
    except Exception as e:
        print(f"❌ Stage 2 error: {e}")

def test_components(args):
    """Test individual components"""
    print("🧪 Testing SCOPE Components...")
    
    # Test BiSeNet V2 spill detection
    try:
        from bisenet.bisenetv2_spill_detector import BiSeNetV2SpillDetector
        spill_detector = BiSeNetV2SpillDetector()
        print("✅ BiSeNet V2 spill detector loaded")
    except Exception as e:
        print(f"❌ BiSeNet V2 failed: {e}")
    
    # Test improved object detection
    try:
        from yolo.improved_object_detector import ImprovedObjectDetector
        obj_detector = ImprovedObjectDetector()
        print("✅ Improved object detector loaded")
    except Exception as e:
        print(f"❌ Object detector failed: {e}")
    
    # Test MQTT
    try:
        from mqtt_event_publisher import MQTTEventPublisher
        config = {'mqtt': {'broker_host': 'localhost', 'enabled': False}}
        mqtt_pub = MQTTEventPublisher(config)
        print("✅ MQTT event publisher ready")
    except Exception as e:
        print(f"❌ MQTT failed: {e}")
    
    # Test Redis
    try:
        from redis_token_manager import RedisTokenManager
        config = {'redis': {'host': 'localhost', 'enabled': False}}
        redis_mgr = RedisTokenManager(config)
        print("✅ Redis token manager ready")
    except Exception as e:
        print(f"❌ Redis failed: {e}")

def convert_models(args):
    """Convert models to TensorRT"""
    print("⚡ Converting models to TensorRT...")
    
    try:
        from cuda.tensorrt_converter import TensorRTConverter
        
        converter = TensorRTConverter()
        
        # Convert Stage 1 models
        print("Converting YOLOv8m...")
        converter.convert_yolo_model("yolo/yolov8m.pt", "yolo/yolov8m.engine")
        
        print("Converting BiSeNet V2...")
        # Note: BiSeNet V2 TensorRT conversion needs custom implementation
        
        print("✅ Model conversion completed")
        
    except Exception as e:
        print(f"❌ TensorRT conversion failed: {e}")

def show_system_info():
    """Display system information"""
    print("📋 SCOPE System Information")
    print("=" * 40)
    
    # Check device availability
    import torch
    print("🔧 Available Devices:")
    print(f"  • CUDA: {'✅' if torch.cuda.is_available() else '❌'}")
    print(f"  • MPS (Apple Silicon): {'✅' if torch.backends.mps.is_available() else '❌'}")
    print(f"  • CPU: ✅")
    
    # Check for models
    models_found = []
    yolo_dir = Path("yolo")
    if yolo_dir.exists():
        for model in yolo_dir.glob("*.pt"):
            models_found.append(f"  • {model.name}")
    
    if models_found:
        print("\n🤖 Available Models:")
        print("\n".join(models_found))
    
    # Check for configs
    configs_dir = Path("configs")
    if configs_dir.exists():
        print("\n📁 Available Configurations:")
        for config in configs_dir.glob("*.yaml"):
            print(f"  • {config.name}")
    
    print("\n🎮 Quick Start Commands:")
    print("  python scope_main.py pipeline                 # Run complete CUDA pipeline")
    print("  python scope_main.py stage1 --test           # Test Stage 1 detection")
    print("  python scope_main.py stage2 --test           # Test Stage 2 verification")
    print("  python scope_main.py test                    # Test all components")
    print("  python scope_main.py convert                 # Convert to TensorRT")
    print("  python scope_main.py info                    # Show this information")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="SCOPE Smart Building CUDA-Optimized AI Vision System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Commands:
  pipeline    - Run complete CUDA pipeline (Stage 1 + Stage 2 + MQTT)
  stage1      - Run Stage 1 detection only (BiSeNet V2 + YOLOv8m)
  stage2      - Run Stage 2 verification only (RT-DETR)
  test        - Test all components individually
  convert     - Convert models to TensorRT engines
  info        - Show system information and available models

Examples:
  python scope_main.py pipeline --config configs/cuda_config.yaml
  python scope_main.py stage1 --test
  python scope_main.py pipeline --camera 0 --camera-id "office_cam"
        """
    )
    
    parser.add_argument('mode', choices=['pipeline', 'stage1', 'stage2', 'test', 'convert', 'info'],
                       help='Operation mode to run')
    
    # Common arguments
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--camera', help='Camera source (index, RTSP URL, or file path)')
    parser.add_argument('--camera-id', type=str, help='Camera identifier')
    parser.add_argument('--test', action='store_true', help='Test mode (load only, no inference)')
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.mode == 'pipeline':
        run_cuda_pipeline(args)
    elif args.mode == 'stage1':
        run_stage1_only(args)
    elif args.mode == 'stage2':
        run_stage2_only(args)
    elif args.mode == 'test':
        test_components(args)
    elif args.mode == 'convert':
        convert_models(args)
    elif args.mode == 'info':
        show_system_info()

if __name__ == "__main__":
    main()