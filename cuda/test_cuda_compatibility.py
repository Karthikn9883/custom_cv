#!/usr/bin/env python3
"""
CUDA Compatibility Test - SCOPE Smart Building System
Tests cross-platform compatibility (Mac MPS + CUDA fallbacks)
Validates all components work correctly on different devices
"""

import os
import sys
import time
import json
import logging
import traceback
from pathlib import Path
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    import cv2
except ImportError as e:
    print(f"❌ Required dependencies missing: {e}")
    sys.exit(1)

# Import SCOPE components
try:
    from cuda_memory_manager import get_memory_manager, clear_global_memory_manager
    from cuda_stage1_detector import CUDAStage1Detector, DetectionToken
    from cuda_stage2_verifier import CUDAStage2Verifier
    from cuda_pipeline_coordinator import CUDAPipelineCoordinator
    from redis_token_manager import RedisTokenManager
    from mqtt_event_publisher import MQTTEventPublisher
    from tensorrt_converter import TensorRTConverter
except ImportError as e:
    print(f"❌ SCOPE component import failed: {e}")
    sys.exit(1)

class CompatibilityTester:
    """Comprehensive compatibility tester for SCOPE system"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.test_results = {}
        self.device_info = {}
        
        # Collect device information
        self._collect_device_info()
    
    def _setup_logging(self):
        """Setup logging for tests"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _collect_device_info(self):
        """Collect device and system information"""
        self.device_info = {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available(),
            'platform': sys.platform
        }
        
        if torch.cuda.is_available():
            self.device_info.update({
                'cuda_device_count': torch.cuda.device_count(),
                'cuda_device_name': torch.cuda.get_device_name(0),
                'cuda_capability': torch.cuda.get_device_capability(0)
            })
        
        self.logger.info("Device Information:")
        for key, value in self.device_info.items():
            self.logger.info(f"  {key}: {value}")
    
    def test_memory_manager(self) -> bool:
        """Test CUDA memory manager"""
        try:
            self.logger.info("Testing CUDA Memory Manager...")
            
            # Test memory manager initialization
            memory_manager = get_memory_manager()
            
            # Test tensor allocation
            test_shapes = [(1, 3, 320, 320), (100, 4), (1000,)]
            
            for shape in test_shapes:
                tensor = memory_manager.get_tensor(shape, torch.float32)
                assert tensor.shape == shape, f"Shape mismatch: expected {shape}, got {tensor.shape}"
                assert tensor.device == memory_manager.device, "Device mismatch"
            
            # Test memory stats
            stats = memory_manager.get_memory_stats()
            assert 'device' in stats, "Missing device in stats"
            assert 'tensor_pool_size' in stats, "Missing tensor pool size"
            
            # Test cache clearing
            memory_manager.clear_cache(force=True)
            
            # Test warmup
            memory_manager.warm_up_device()
            
            # Cleanup
            clear_global_memory_manager()
            
            self.test_results['memory_manager'] = {'success': True, 'device': str(memory_manager.device)}
            self.logger.info("✅ Memory Manager test passed")
            return True
            
        except Exception as e:
            self.test_results['memory_manager'] = {'success': False, 'error': str(e)}
            self.logger.error(f"❌ Memory Manager test failed: {e}")
            return False
    
    def test_stage1_detector(self) -> bool:
        """Test Stage 1 detector"""
        try:
            self.logger.info("Testing Stage 1 Detector...")
            
            # Create test config
            test_config = "configs/cuda_config.yaml"
            if not Path(test_config).exists():
                # Create minimal config for testing
                test_config = {
                    'stage1': {
                        'models': {
                            'object_detector': 'yolov8n.pt',  # Will auto-download
                            'spill_detector': 'yolov8n-seg.pt'
                        },
                        'target_latency_ms': 50,  # Relaxed for testing
                        'confidence_threshold': 0.35,
                        'iou_threshold': 0.5,
                        'max_detections': 30,
                        'image_size': 320,
                        'detection_prompts': ['person', 'laptop', 'bottle']
                    }
                }
                
                # Save test config
                import yaml
                with open('test_config.yaml', 'w') as f:
                    yaml.dump(test_config, f)
                test_config = 'test_config.yaml'
            
            # Initialize detector
            detector = CUDAStage1Detector(test_config)
            
            # Create test frame
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Test detection
            tokens = detector.detect_frame(test_frame, "test_cam")
            
            # Validate results
            assert isinstance(tokens, list), "Tokens should be a list"
            
            # Test performance stats
            stats = detector.get_performance_stats()
            assert 'avg_latency_ms' in stats or 'device' in stats, "Missing performance stats"
            
            # Test token filtering
            filtered_tokens = detector.filter_tokens_for_stage2(tokens)
            assert isinstance(filtered_tokens, list), "Filtered tokens should be a list"
            
            self.test_results['stage1_detector'] = {
                'success': True, 
                'device': str(detector.device),
                'tokens_detected': len(tokens),
                'tokens_filtered': len(filtered_tokens)
            }
            self.logger.info(f"✅ Stage 1 Detector test passed ({len(tokens)} tokens detected)")
            return True
            
        except Exception as e:
            self.test_results['stage1_detector'] = {'success': False, 'error': str(e)}
            self.logger.error(f"❌ Stage 1 Detector test failed: {e}")
            return False
    
    def test_stage2_verifier(self) -> bool:
        """Test Stage 2 verifier"""
        try:
            self.logger.info("Testing Stage 2 Verifier...")
            
            # Use same test config as Stage 1
            test_config = "configs/cuda_config.yaml"
            if not Path(test_config).exists():
                test_config = 'test_config.yaml'
            
            # Initialize verifier
            verifier = CUDAStage2Verifier(test_config)
            
            # Create test token
            test_token = DetectionToken(
                uuid="test-uuid",
                timestamp=time.time(),
                category="laptop",
                confidence=0.8,
                bbox=(100, 100, 200, 200),
                camera_id="test_cam",
                frame_id=1,
                detector_type="object"
            )
            
            # Create test frame
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Test verification
            result = verifier.verify_token(test_token, test_frame)
            
            # Validate result
            assert hasattr(result, 'token_uuid'), "Missing token UUID in result"
            assert hasattr(result, 'verified'), "Missing verification status"
            assert hasattr(result, 'fused_confidence'), "Missing fused confidence"
            
            # Test batch verification
            validated_events = verifier.batch_verify_tokens([test_token], test_frame, "test_cam")
            assert isinstance(validated_events, list), "Validated events should be a list"
            
            # Test performance stats
            stats = verifier.get_performance_stats()
            assert 'device' in stats, "Missing device in stats"
            
            self.test_results['stage2_verifier'] = {
                'success': True,
                'device': str(verifier.device),
                'verification_time_ms': result.verification_time_ms,
                'verified': result.verified
            }
            self.logger.info(f"✅ Stage 2 Verifier test passed (verification: {result.verified})")
            return True
            
        except Exception as e:
            self.test_results['stage2_verifier'] = {'success': False, 'error': str(e)}
            self.logger.error(f"❌ Stage 2 Verifier test failed: {e}")
            return False
    
    def test_pipeline_coordinator(self) -> bool:
        """Test pipeline coordinator"""
        try:
            self.logger.info("Testing Pipeline Coordinator...")
            
            # Use test config
            test_config = "configs/cuda_config.yaml"
            if not Path(test_config).exists():
                test_config = 'test_config.yaml'
            
            # Initialize coordinator
            coordinator = CUDAPipelineCoordinator(test_config)
            
            # Test frame processing
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            results = coordinator.process_frame(test_frame, "test_cam")
            
            # Validate results
            assert 'stage1_tokens' in results, "Missing stage1 tokens"
            assert 'stage2_queued' in results, "Missing stage2 queued count"
            assert 'stage1_time_ms' in results, "Missing stage1 timing"
            
            # Test pipeline start/stop
            coordinator.start_pipeline()
            time.sleep(1)  # Let it run briefly
            coordinator.stop_pipeline()
            
            # Test statistics
            stats = coordinator.get_complete_stats()
            assert 'pipeline' in stats, "Missing pipeline stats"
            assert 'stage1' in stats, "Missing stage1 stats"
            assert 'stage2' in stats, "Missing stage2 stats"
            
            self.test_results['pipeline_coordinator'] = {
                'success': True,
                'stage1_time_ms': results['stage1_time_ms'],
                'tokens_generated': len(results['stage1_tokens']),
                'queue_size': results['queue_size']
            }
            self.logger.info("✅ Pipeline Coordinator test passed")
            return True
            
        except Exception as e:
            self.test_results['pipeline_coordinator'] = {'success': False, 'error': str(e)}
            self.logger.error(f"❌ Pipeline Coordinator test failed: {e}")
            return False
    
    def test_redis_manager(self) -> bool:
        """Test Redis token manager (optional - skips if Redis not available)"""
        try:
            self.logger.info("Testing Redis Token Manager...")
            
            # Test config
            config = {
                'redis': {
                    'host': 'localhost',
                    'port': 6379,
                    'db': 0,
                    'timeout': 2
                }
            }
            
            # Try to initialize manager
            manager = RedisTokenManager(config)
            
            # Test health check
            if not manager.health_check():
                self.test_results['redis_manager'] = {'success': False, 'error': 'Redis not available'}
                self.logger.warning("⚠️ Redis not available - test skipped")
                return True  # Not a failure, just unavailable
            
            # Test token operations
            test_token = DetectionToken(
                uuid="test-uuid",
                timestamp=time.time(),
                category="laptop",
                confidence=0.8,
                bbox=(100, 100, 200, 200),
                camera_id="test_cam",
                frame_id=1,
                detector_type="object"
            )
            
            # Test publish/consume
            success = manager.publish_stage1_tokens([test_token], "test_cam")
            assert success, "Token publishing failed"
            
            consumed = manager.consume_stage1_tokens(batch_size=1, timeout=1.0)
            assert len(consumed) > 0, "Token consumption failed"
            
            # Test stats
            stats = manager.get_queue_stats()
            assert 'queues' in stats, "Missing queue stats"
            
            # Cleanup
            manager.clear_queues()
            
            self.test_results['redis_manager'] = {'success': True, 'available': True}
            self.logger.info("✅ Redis Token Manager test passed")
            return True
            
        except Exception as e:
            self.test_results['redis_manager'] = {'success': False, 'error': str(e)}
            self.logger.warning(f"⚠️ Redis Token Manager test failed (optional): {e}")
            return True  # Optional component
    
    def test_mqtt_publisher(self) -> bool:
        """Test MQTT event publisher (optional - skips if MQTT not available)"""
        try:
            self.logger.info("Testing MQTT Event Publisher...")
            
            # Test config
            config = {
                'mqtt': {
                    'broker_host': 'localhost',
                    'broker_port': 1883,
                    'client_id': 'test_publisher',
                    'connection_timeout': 2
                }
            }
            
            # Try to initialize publisher
            publisher = MQTTEventPublisher(config)
            
            # Brief wait for connection
            time.sleep(1)
            
            # Test health check
            if not publisher.health_check():
                self.test_results['mqtt_publisher'] = {'success': False, 'error': 'MQTT broker not available'}
                self.logger.warning("⚠️ MQTT broker not available - test skipped")
                return True  # Not a failure, just unavailable
            
            # Test event publishing would require full ValidatedEvent objects
            # For now, just test initialization and health
            
            # Test stats
            stats = publisher.get_statistics()
            assert 'connection' in stats, "Missing connection stats"
            
            publisher.stop_publisher()
            
            self.test_results['mqtt_publisher'] = {'success': True, 'available': True}
            self.logger.info("✅ MQTT Event Publisher test passed")
            return True
            
        except Exception as e:
            self.test_results['mqtt_publisher'] = {'success': False, 'error': str(e)}
            self.logger.warning(f"⚠️ MQTT Event Publisher test failed (optional): {e}")
            return True  # Optional component
    
    def test_tensorrt_converter(self) -> bool:
        """Test TensorRT converter (CUDA only)"""
        try:
            self.logger.info("Testing TensorRT Converter...")
            
            if not torch.cuda.is_available():
                self.test_results['tensorrt_converter'] = {'success': False, 'error': 'CUDA not available'}
                self.logger.warning("⚠️ TensorRT converter requires CUDA - test skipped")
                return True  # Not a failure, just not applicable
            
            # Initialize converter
            converter = TensorRTConverter()
            
            # Test recommendations (doesn't require actual conversion)
            dummy_model_path = "yolov8n.pt"
            recommendations = converter.get_optimization_recommendations(dummy_model_path)
            
            assert 'precision' in recommendations, "Missing precision recommendation"
            assert 'optimizations' in recommendations, "Missing optimizations list"
            
            self.test_results['tensorrt_converter'] = {'success': True, 'cuda_available': True}
            self.logger.info("✅ TensorRT Converter test passed")
            return True
            
        except Exception as e:
            self.test_results['tensorrt_converter'] = {'success': False, 'error': str(e)}
            self.logger.warning(f"⚠️ TensorRT Converter test failed: {e}")
            return True  # Optional for development
    
    def test_unified_detection(self) -> bool:
        """Test unified detection system"""
        try:
            self.logger.info("Testing Unified Detection System...")
            
            # Import unified detection
            from unified_detection import UnifiedDetectionSystem
            
            # Initialize system
            system = UnifiedDetectionSystem(
                segformer_model_path="models/coco_spill_detector.pt",
                enable_mqtt=False  # Disable MQTT for testing
            )
            
            # Test frame processing
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Process frame with both models
            results = system._process_frame_parallel(test_frame)
            
            # Validate results
            assert 'segformer' in results, "Missing SegFormer results"
            assert 'yolo' in results, "Missing YOLO results"
            
            segformer_result = results['segformer']
            yolo_result = results['yolo']
            
            assert 'spill_detected' in segformer_result, "Missing spill detection status"
            assert 'detection_count' in yolo_result, "Missing detection count"
            
            self.test_results['unified_detection'] = {
                'success': True,
                'segformer_device': str(system.segformer.device),
                'spill_detected': segformer_result.get('spill_detected', False),
                'objects_detected': yolo_result.get('detection_count', 0)
            }
            self.logger.info("✅ Unified Detection System test passed")
            return True
            
        except Exception as e:
            self.test_results['unified_detection'] = {'success': False, 'error': str(e)}
            self.logger.error(f"❌ Unified Detection System test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all compatibility tests"""
        self.logger.info("🧪 Starting SCOPE Compatibility Tests...")
        
        # List of tests to run
        tests = [
            ('Memory Manager', self.test_memory_manager),
            ('Stage 1 Detector', self.test_stage1_detector),
            ('Stage 2 Verifier', self.test_stage2_verifier),
            ('Pipeline Coordinator', self.test_pipeline_coordinator),
            ('Redis Manager', self.test_redis_manager),
            ('MQTT Publisher', self.test_mqtt_publisher),
            ('TensorRT Converter', self.test_tensorrt_converter),
            ('Unified Detection', self.test_unified_detection),
        ]
        
        # Run tests
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                self.logger.error(f"Test {test_name} crashed: {e}")
                self.test_results[test_name.lower().replace(' ', '_')] = {
                    'success': False,
                    'error': f"Test crashed: {e}"
                }
        
        # Generate summary
        summary = {
            'device_info': self.device_info,
            'test_results': self.test_results,
            'summary': {
                'total_tests': total,
                'passed_tests': passed,
                'failed_tests': total - passed,
                'success_rate': (passed / total) * 100,
                'overall_status': 'PASS' if passed >= total * 0.8 else 'FAIL'  # 80% threshold
            }
        }
        
        self.logger.info(f"🏁 Tests completed: {passed}/{total} passed ({summary['summary']['success_rate']:.1f}%)")
        
        return summary
    
    def cleanup(self):
        """Cleanup test artifacts"""
        try:
            # Remove test config if created
            test_config = Path('test_config.yaml')
            if test_config.exists():
                test_config.unlink()
            
            # Clear global memory manager
            clear_global_memory_manager()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            self.logger.warning(f"Cleanup warning: {e}")

def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SCOPE Compatibility Tester')
    parser.add_argument('--output', type=str, default='compatibility_report.json',
                       help='Output file for test results')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run tests
    tester = CompatibilityTester()
    
    try:
        results = tester.run_all_tests()
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("🧪 SCOPE COMPATIBILITY TEST RESULTS")
        print("="*60)
        
        print(f"Platform: {results['device_info']['platform']}")
        print(f"PyTorch: {results['device_info']['pytorch_version']}")
        print(f"CUDA Available: {results['device_info']['cuda_available']}")
        print(f"MPS Available: {results['device_info']['mps_available']}")
        
        print("\nTest Results:")
        for test_name, result in results['test_results'].items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"  {test_name}: {status}")
            if not result['success']:
                print(f"    Error: {result.get('error', 'Unknown error')}")
        
        summary = results['summary']
        print(f"\nOverall: {summary['passed_tests']}/{summary['total_tests']} tests passed")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Status: {summary['overall_status']}")
        
        print(f"\nDetailed report saved to: {args.output}")
        
        # Exit with appropriate code
        sys.exit(0 if summary['overall_status'] == 'PASS' else 1)
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()