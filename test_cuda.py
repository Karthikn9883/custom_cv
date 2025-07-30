#!/usr/bin/env python3
"""
Quick CUDA Test Script
Verify that PyTorch can detect and use your RTX 4070
"""

import torch
import sys

print("🔧 CUDA Setup Verification")
print("=" * 50)

# Basic CUDA info
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    # GPU details
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"GPU {i}: {gpu_name}")
        print(f"Memory: {gpu_memory:.1f} GB")
    
    # Test GPU computation
    print("\n🧪 Testing GPU computation...")
    try:
        # Create test tensors on GPU
        a = torch.randn(1000, 1000).cuda()
        b = torch.randn(1000, 1000).cuda()
        c = torch.matmul(a, b)
        print("✅ GPU computation test PASSED")
        print(f"Result tensor shape: {c.shape}")
        print(f"Result tensor device: {c.device}")
    except Exception as e:
        print(f"❌ GPU computation test FAILED: {e}")
        
else:
    print("❌ CUDA not available")
    print("\nPossible solutions:")
    print("1. Reinstall PyTorch with CUDA support:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("2. Check NVIDIA drivers are up to date")
    print("3. Verify CUDA toolkit installation")

print("\n" + "=" * 50)

# Test BiSeNet V2 training compatibility
print("🤖 Testing BiSeNet V2 Training Compatibility...")
try:
    from bisenet.bisenetv2_model import create_bisenetv2
    
    # Create model
    model = create_bisenetv2(num_classes=2, aux_mode='eval')
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("✅ BiSeNet V2 model loaded on GPU")
        
        # Test inference
        dummy_input = torch.randn(1, 3, 512, 512).cuda()
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ GPU inference test PASSED - Output shape: {output.shape}")
    else:
        print("⚠️ BiSeNet V2 model loaded on CPU (GPU not available)")
        
except Exception as e:
    print(f"❌ BiSeNet V2 test FAILED: {e}")

print("\n🎯 Ready for CUDA-accelerated training!" if torch.cuda.is_available() else "\n⚠️ CPU training mode")