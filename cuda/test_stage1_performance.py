#!/usr/bin/env python3
"""
Stage 1 Performance Testing Script
Tests latency and throughput of optimized Stage 1 detector
Includes proper virtual environment activation and memory profiling
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from stage1_detector import Stage1Detector
    from unified_detection import UnifiedDetectionSystem, SegFormerSpillDetector, YOLOWorldDetector
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you've activated the virtual environment:")
    print("source venv/bin/activate")
    sys.exit(1)

class PerformanceTester:
    """Comprehensive performance testing for Stage 1 detector"""
    
    def __init__(self):
        self.setup_logging()
        self.results = {
            'stage1_detector': {},
            'unified_system': {},
            'segformer_only': {},
            'yolo_only': {}
        }
        
    def setup_logging(self):
        """Setup performance logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def check_environment(self):
        """Check system environment and capabilities"""
        self.logger.info("🔍 Checking system environment...")
        
        # Check virtual environment
        venv_active = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
        env_info = {
            'virtual_env': venv_active,
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            'cuda_available': torch.cuda.is_available(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2)
        }
        
        for key, value in env_info.items():
            self.logger.info(f"  {key}: {value}")
            
        if not venv_active:
            self.logger.warning("⚠️  Virtual environment not detected!")
            
        return env_info
    
    def create_test_frames(self, count: int = 100) -> List[np.ndarray]:
        """Create synthetic test frames for consistent testing"""
        self.logger.info(f"📸 Creating {count} test frames...")
        
        frames = []
        for i in range(count):
            # Create synthetic frame with random objects
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add some geometric shapes to simulate objects
            cv2.rectangle(frame, (50 + i, 50), (150 + i, 150), (255, 0, 0), -1)
            cv2.circle(frame, (300 + i % 50, 200), 30, (0, 255, 0), -1)
            
            frames.append(frame)
            
        return frames
    
    def test_stage1_detector(self, frames: List[np.ndarray]) -> Dict:
        """Test Stage 1 detector performance"""
        self.logger.info("🧪 Testing Stage 1 Detector...")
        
        try:
            detector = Stage1Detector("configs/stage1_config.yaml")
            
            # Warmup
            for i in range(5):
                detector.detect_frame(frames[0])
                
            # Performance test
            latencies = []
            start_time = time.time()
            
            for i, frame in enumerate(frames[:50]):  # Test with 50 frames
                frame_start = time.time()
                tokens = detector.detect_frame(frame, f"test_cam_{i}")
                frame_end = time.time()
                
                latency_ms = (frame_end - frame_start) * 1000
                latencies.append(latency_ms)
                
                if i % 10 == 0:
                    self.logger.info(f"  Frame {i}: {latency_ms:.1f}ms, {len(tokens)} detections")
            
            total_time = time.time() - start_time
            
            results = {
                'avg_latency_ms': np.mean(latencies),
                'min_latency_ms': np.min(latencies),
                'max_latency_ms': np.max(latencies),
                'p95_latency_ms': np.percentile(latencies, 95),
                'fps': len(latencies) / total_time,
                'total_frames': len(latencies),
                'target_met': np.mean(latencies) < 30.0
            }
            
            self.logger.info(f"✅ Stage 1 Results: {results['avg_latency_ms']:.1f}ms avg, {results['fps']:.1f} FPS")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Stage 1 test failed: {e}")
            return {'error': str(e)}
    
    def test_unified_system(self, frames: List[np.ndarray]) -> Dict:
        """Test unified detection system performance"""
        self.logger.info("🧪 Testing Unified Detection System...")
        
        try:
            system = UnifiedDetectionSystem(
                enable_mqtt=False  # Disable MQTT for testing
            )
            
            # Warmup
            for i in range(3):
                system._process_frame(frames[0])
                
            # Performance test
            latencies = []
            start_time = time.time()
            
            for i, frame in enumerate(frames[:30]):  # Test with 30 frames (more intensive)
                frame_start = time.time()
                results = system._process_frame(frame)
                frame_end = time.time()
                
                latency_ms = (frame_end - frame_start) * 1000
                latencies.append(latency_ms)
                
                if i % 5 == 0:
                    seg_detected = results['segformer'].get('spill_detected', False)
                    yolo_count = results['yolo'].get('detection_count', 0)
                    self.logger.info(f"  Frame {i}: {latency_ms:.1f}ms, Spill: {seg_detected}, Objects: {yolo_count}")
            
            total_time = time.time() - start_time
            
            results = {
                'avg_latency_ms': np.mean(latencies),
                'min_latency_ms': np.min(latencies),
                'max_latency_ms': np.max(latencies),
                'p95_latency_ms': np.percentile(latencies, 95),
                'fps': len(latencies) / total_time,
                'total_frames': len(latencies),
                'target_met': np.mean(latencies) < 30.0
            }
            
            self.logger.info(f"✅ Unified Results: {results['avg_latency_ms']:.1f}ms avg, {results['fps']:.1f} FPS")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Unified test failed: {e}")
            return {'error': str(e)}
    
    def test_individual_models(self, frames: List[np.ndarray]) -> Tuple[Dict, Dict]:
        """Test individual model performance"""
        self.logger.info("🧪 Testing Individual Models...")
        
        # Test SegFormer only
        segformer_results = {'error': 'Not tested'}
        try:
            segformer = SegFormerSpillDetector()
            
            # Warmup
            for i in range(3):
                segformer.detect(frames[0])
                
            latencies = []
            for frame in frames[:20]:
                start = time.time()
                result = segformer.detect(frame)
                end = time.time()
                latencies.append((end - start) * 1000)
            
            segformer_results = {
                'avg_latency_ms': np.mean(latencies),
                'fps': len(latencies) / (sum(latencies) / 1000),
                'target_met': np.mean(latencies) < 15.0  # Half of total budget
            }
            self.logger.info(f"  SegFormer: {segformer_results['avg_latency_ms']:.1f}ms avg")
            
        except Exception as e:
            self.logger.error(f"❌ SegFormer test failed: {e}")
            segformer_results = {'error': str(e)}
        
        # Test YOLO only  
        yolo_results = {'error': 'Not tested'}
        try:
            yolo = YOLOWorldDetector()
            
            # Warmup
            for i in range(3):
                yolo.detect(frames[0])
                
            latencies = []
            for frame in frames[:20]:
                start = time.time()
                result = yolo.detect(frame)
                end = time.time()
                latencies.append((end - start) * 1000)
            
            yolo_results = {
                'avg_latency_ms': np.mean(latencies),
                'fps': len(latencies) / (sum(latencies) / 1000),
                'target_met': np.mean(latencies) < 15.0  # Half of total budget
            }
            self.logger.info(f"  YOLO-World: {yolo_results['avg_latency_ms']:.1f}ms avg")
            
        except Exception as e:
            self.logger.error(f"❌ YOLO test failed: {e}")
            yolo_results = {'error': str(e)}
        
        return segformer_results, yolo_results
    
    def memory_profiling(self):
        """Profile memory usage during testing"""
        self.logger.info("🧠 Memory profiling...")
        
        process = psutil.Process()
        
        # Before test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        
        # Run a quick test
        frames = self.create_test_frames(10)
        try:
            detector = Stage1Detector("configs/stage1_config.yaml")
            for frame in frames:
                detector.detect_frame(frame)
        except Exception as e:
            self.logger.error(f"Memory profiling test failed: {e}")
            
        # After test
        final_memory = process.memory_info().rss / (1024**2)  # MB
        
        memory_info = {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': final_memory - initial_memory
        }
        
        self.logger.info(f"  Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB (+{memory_info['memory_increase_mb']:.1f}MB)")
        return memory_info
    
    def run_comprehensive_test(self):
        """Run all performance tests"""
        self.logger.info("🚀 Starting comprehensive performance test...")
        
        # Environment check
        env_info = self.check_environment()
        
        # Create test data
        frames = self.create_test_frames(100)
        
        # Memory profiling
        memory_info = self.memory_profiling()
        
        # Run tests
        stage1_results = self.test_stage1_detector(frames)
        unified_results = self.test_unified_system(frames)
        segformer_results, yolo_results = self.test_individual_models(frames)
        
        # Compile results
        final_results = {
            'environment': env_info,
            'memory': memory_info,
            'stage1_detector': stage1_results,
            'unified_system': unified_results,
            'segformer_only': segformer_results,
            'yolo_only': yolo_results,
            'timestamp': time.time()
        }
        
        # Generate report
        self.generate_report(final_results)
        
        return final_results
    
    def generate_report(self, results: Dict):
        """Generate performance report"""
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 PERFORMANCE REPORT")
        self.logger.info("="*60)
        
        # Environment summary
        env = results.get('environment', {})
        self.logger.info(f"Environment: Python {env.get('python_version', 'Unknown')}")
        self.logger.info(f"Device: {'MPS' if env.get('mps_available') else 'CUDA' if env.get('cuda_available') else 'CPU'}")
        self.logger.info(f"Memory: {env.get('memory_gb', 'Unknown')}GB")
        
        # Performance summary
        stage1 = results.get('stage1_detector', {})
        unified = results.get('unified_system', {})
        
        if 'avg_latency_ms' in stage1:
            target_met = "✅" if stage1.get('target_met', False) else "❌"
            self.logger.info(f"\nStage 1 Detector: {stage1['avg_latency_ms']:.1f}ms avg {target_met}")
            self.logger.info(f"  Range: {stage1['min_latency_ms']:.1f}-{stage1['max_latency_ms']:.1f}ms")
            self.logger.info(f"  P95: {stage1['p95_latency_ms']:.1f}ms, FPS: {stage1['fps']:.1f}")
        
        if 'avg_latency_ms' in unified:
            target_met = "✅" if unified.get('target_met', False) else "❌"
            self.logger.info(f"\nUnified System: {unified['avg_latency_ms']:.1f}ms avg {target_met}")
            self.logger.info(f"  Range: {unified['min_latency_ms']:.1f}-{unified['max_latency_ms']:.1f}ms")
            self.logger.info(f"  P95: {unified['p95_latency_ms']:.1f}ms, FPS: {unified['fps']:.1f}")
        
        # Individual models
        segformer = results.get('segformer_only', {})
        yolo = results.get('yolo_only', {})
        
        if 'avg_latency_ms' in segformer:
            self.logger.info(f"\nSegFormer Only: {segformer['avg_latency_ms']:.1f}ms avg")
        if 'avg_latency_ms' in yolo:
            self.logger.info(f"YOLO-World Only: {yolo['avg_latency_ms']:.1f}ms avg")
        
        # Memory usage
        memory = results.get('memory', {})
        if memory:
            self.logger.info(f"\nMemory Impact: +{memory.get('memory_increase_mb', 0):.1f}MB")
        
        self.logger.info("="*60)

def main():
    """Main testing function with proper venv activation check"""
    
    # Check virtual environment
    venv_active = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if not venv_active:
        print("❌ Virtual environment not detected!")
        print("Please activate the virtual environment first:")
        print("  source venv/bin/activate")
        print("Then run this script again.")
        sys.exit(1)
    
    print("✅ Virtual environment detected")
    print("🚀 Starting Stage 1 Performance Testing...")
    
    tester = PerformanceTester()
    results = tester.run_comprehensive_test()
    
    # Save results to file
    import json
    results_file = Path("performance_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("🎯 Target: <30ms average latency")
    
    # Final recommendation
    stage1_latency = results.get('stage1_detector', {}).get('avg_latency_ms', float('inf'))
    if stage1_latency < 30:
        print(f"🎉 SUCCESS: {stage1_latency:.1f}ms meets <30ms target!")
    else:
        print(f"⚠️  NEEDS OPTIMIZATION: {stage1_latency:.1f}ms exceeds 30ms target")
        print("   Consider implementing additional optimizations:")
        print("   - Model quantization (INT8)")
        print("   - TensorRT optimization") 
        print("   - Hardware-specific tuning")

if __name__ == "__main__":
    main()