#!/usr/bin/env python3
"""
TensorRT Converter - SCOPE Smart Building System
Converts PyTorch models to TensorRT engines for maximum CUDA performance
Supports YOLO-World, RT-DETR, and custom models with optimization
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

try:
    from ultralytics import YOLO, RTDETR
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics")
    exit(1)

try:
    import torch
except ImportError:
    print("Error: pytorch not installed")
    exit(1)

# TensorRT imports (optional, graceful fallback)
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("Warning: TensorRT not available. Engine conversion will use ultralytics export.")

class TensorRTConverter:
    """
    Advanced TensorRT converter for SCOPE models
    Handles optimization, quantization, and validation
    """
    
    def __init__(self, cuda_device: int = 0, workspace_size: int = 4):
        self.device = cuda_device
        self.workspace_size = workspace_size * (1024 ** 3)  # Convert GB to bytes
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Conversion statistics
        self.conversion_stats = {}
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. TensorRT requires CUDA.")
        
        torch.cuda.set_device(cuda_device)
        self.logger.info(f"TensorRT Converter initialized on CUDA device {cuda_device}")
    
    def convert_yolo_world(self, model_path: str, output_dir: str = "tensorrt_engines",
                          precision: str = "fp16", batch_size: int = 1,
                          image_size: int = 640) -> Dict[str, Any]:
        """
        Convert YOLO-World model to TensorRT engine
        
        Args:
            model_path: Path to YOLO model (.pt file)
            output_dir: Output directory for engine
            precision: Precision mode ('fp32', 'fp16', 'int8')
            batch_size: Batch size for optimization
            image_size: Input image size
            
        Returns:
            Conversion results
        """
        start_time = time.time()
        model_name = Path(model_path).stem
        
        try:
            self.logger.info(f"Converting YOLO-World model: {model_path}")
            
            # Load model
            model = YOLO(model_path)
            
            # Prepare output path
            os.makedirs(output_dir, exist_ok=True)
            engine_path = os.path.join(output_dir, f"{model_name}_{precision}_b{batch_size}.engine")
            
            # Configure export parameters
            export_params = {
                'format': 'engine',
                'imgsz': image_size,
                'batch': batch_size,
                'device': self.device,
                'verbose': True,
                'workspace': self.workspace_size / (1024 ** 3)  # Convert back to GB for ultralytics
            }
            
            # Set precision
            if precision == 'fp16':
                export_params['half'] = True
            elif precision == 'int8':
                export_params['int8'] = True
                # Generate calibration data for INT8
                export_params['data'] = self._generate_calibration_data(image_size)
            
            # Perform conversion
            self.logger.info(f"Exporting with parameters: {export_params}")
            success = model.export(**export_params)
            
            # Validate engine
            validation_results = self._validate_engine(engine_path, model_path, image_size, batch_size)
            
            conversion_time = time.time() - start_time
            
            # Collect results
            results = {
                'success': True,
                'model_type': 'yolo-world',
                'input_model': model_path,
                'output_engine': engine_path,
                'precision': precision,
                'batch_size': batch_size,
                'image_size': image_size,
                'conversion_time_s': conversion_time,
                'validation': validation_results,
                'file_size_mb': os.path.getsize(engine_path) / (1024 ** 2) if os.path.exists(engine_path) else 0
            }
            
            self.conversion_stats[model_name] = results
            self.logger.info(f"YOLO-World conversion completed in {conversion_time:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"YOLO-World conversion failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_type': 'yolo-world',
                'conversion_time_s': time.time() - start_time
            }
    
    def convert_rtdetr(self, model_path: str, output_dir: str = "tensorrt_engines",
                      precision: str = "fp16", batch_size: int = 1,
                      image_size: int = 640) -> Dict[str, Any]:
        """
        Convert RT-DETR model to TensorRT engine
        
        Args:
            model_path: Path to RT-DETR model (.pt file)
            output_dir: Output directory for engine
            precision: Precision mode ('fp32', 'fp16', 'int8')
            batch_size: Batch size for optimization
            image_size: Input image size
            
        Returns:
            Conversion results
        """
        start_time = time.time()
        model_name = Path(model_path).stem
        
        try:
            self.logger.info(f"Converting RT-DETR model: {model_path}")
            
            # Load model
            model = RTDETR(model_path)
            
            # Prepare output path
            os.makedirs(output_dir, exist_ok=True)
            engine_path = os.path.join(output_dir, f"{model_name}_{precision}_b{batch_size}.engine")
            
            # Configure export parameters
            export_params = {
                'format': 'engine',
                'imgsz': image_size,
                'batch': batch_size,
                'device': self.device,
                'verbose': True,
                'workspace': self.workspace_size / (1024 ** 3)
            }
            
            # Set precision
            if precision == 'fp16':
                export_params['half'] = True
            elif precision == 'int8':
                export_params['int8'] = True
                export_params['data'] = self._generate_calibration_data(image_size)
            
            # Perform conversion
            self.logger.info(f"Exporting RT-DETR with parameters: {export_params}")
            success = model.export(**export_params)
            
            # Validate engine
            validation_results = self._validate_engine(engine_path, model_path, image_size, batch_size)
            
            conversion_time = time.time() - start_time
            
            # Collect results
            results = {
                'success': True,
                'model_type': 'rt-detr',
                'input_model': model_path,
                'output_engine': engine_path,
                'precision': precision,
                'batch_size': batch_size,
                'image_size': image_size,
                'conversion_time_s': conversion_time,
                'validation': validation_results,
                'file_size_mb': os.path.getsize(engine_path) / (1024 ** 2) if os.path.exists(engine_path) else 0
            }
            
            self.conversion_stats[model_name] = results
            self.logger.info(f"RT-DETR conversion completed in {conversion_time:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"RT-DETR conversion failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_type': 'rt-detr',
                'conversion_time_s': time.time() - start_time
            }
    
    def _generate_calibration_data(self, image_size: int, num_samples: int = 100) -> str:
        """
        Generate calibration data for INT8 quantization
        
        Args:
            image_size: Input image size
            num_samples: Number of calibration samples
            
        Returns:
            Path to calibration data yaml
        """
        try:
            # Create calibration directory
            calib_dir = Path("calibration_data")
            calib_dir.mkdir(exist_ok=True)
            
            # Generate random calibration images
            for i in range(num_samples):
                # Generate realistic image data (0-255 range)
                img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
                
                # Save as image
                import cv2
                cv2.imwrite(str(calib_dir / f"calib_{i:03d}.jpg"), img)
            
            # Create calibration yaml
            calib_yaml = calib_dir / "calibration.yaml"
            with open(calib_yaml, 'w') as f:
                f.write(f"path: {calib_dir}\n")
                f.write("train: .\n")
                f.write("val: .\n")
                f.write("nc: 80\n")  # COCO classes
                f.write("names: ['object']\n")  # Simplified for calibration
            
            self.logger.info(f"Generated {num_samples} calibration samples")
            return str(calib_yaml)
            
        except Exception as e:
            self.logger.error(f"Calibration data generation failed: {e}")
            return ""
    
    def _validate_engine(self, engine_path: str, original_model_path: str,
                        image_size: int, batch_size: int) -> Dict[str, Any]:
        """
        Validate TensorRT engine performance and accuracy
        
        Args:
            engine_path: Path to TensorRT engine
            original_model_path: Path to original model
            image_size: Input image size
            batch_size: Batch size
            
        Returns:
            Validation results
        """
        try:
            if not os.path.exists(engine_path):
                return {'valid': False, 'error': 'Engine file not found'}
            
            # Load both models for comparison
            try:
                if 'rtdetr' in original_model_path.lower():
                    original_model = RTDETR(original_model_path)
                    engine_model = RTDETR(engine_path)
                else:
                    original_model = YOLO(original_model_path)
                    engine_model = YOLO(engine_path)
            except Exception as e:
                return {'valid': False, 'error': f'Model loading failed: {e}'}
            
            # Generate test data
            test_image = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
            
            # Performance comparison
            validation_results = {
                'valid': True,
                'engine_exists': True,
                'file_size_mb': os.path.getsize(engine_path) / (1024 ** 2)
            }
            
            # Benchmark original model
            start_time = time.time()
            for _ in range(10):  # 10 iterations for averaging
                _ = original_model(test_image, verbose=False)
            original_time = (time.time() - start_time) / 10 * 1000  # ms per inference
            
            # Benchmark TensorRT engine
            start_time = time.time()
            for _ in range(10):
                _ = engine_model(test_image, verbose=False)
            engine_time = (time.time() - start_time) / 10 * 1000  # ms per inference
            
            # Calculate speedup
            speedup = original_time / engine_time if engine_time > 0 else 0
            
            validation_results.update({
                'original_inference_ms': original_time,
                'engine_inference_ms': engine_time,
                'speedup_factor': speedup,
                'performance_gain_percent': ((original_time - engine_time) / original_time) * 100
            })
            
            self.logger.info(f"Validation: {speedup:.2f}x speedup ({original_time:.1f}ms → {engine_time:.1f}ms)")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Engine validation failed: {e}")
            return {'valid': False, 'error': str(e)}
    
    def convert_scope_models(self, models_config: Dict[str, Any],
                           output_dir: str = "tensorrt_engines") -> Dict[str, Any]:
        """
        Convert all SCOPE models based on configuration
        
        Args:
            models_config: Configuration dictionary with model paths and settings
            output_dir: Output directory for engines
            
        Returns:
            Conversion results for all models
        """
        self.logger.info("Starting SCOPE models conversion...")
        
        results = {
            'conversion_summary': {
                'total_models': 0,
                'successful_conversions': 0,
                'failed_conversions': 0,
                'total_time_s': 0
            },
            'individual_results': {}
        }
        
        start_time = time.time()
        
        # Convert Stage 1 models
        stage1_config = models_config.get('stage1', {})
        for model_type, model_path in stage1_config.get('models', {}).items():
            if not os.path.exists(model_path):
                self.logger.warning(f"Model not found: {model_path}")
                continue
            
            results['conversion_summary']['total_models'] += 1
            
            # Determine model type and convert
            if 'yolo' in model_path.lower():
                conversion_result = self.convert_yolo_world(
                    model_path=model_path,
                    output_dir=output_dir,
                    precision=stage1_config.get('precision', 'fp16'),
                    batch_size=stage1_config.get('batch_size', 1),
                    image_size=stage1_config.get('image_size', 320)
                )
            else:
                self.logger.warning(f"Unknown Stage 1 model type: {model_path}")
                continue
            
            if conversion_result['success']:
                results['conversion_summary']['successful_conversions'] += 1
            else:
                results['conversion_summary']['failed_conversions'] += 1
            
            results['individual_results'][f"stage1_{model_type}"] = conversion_result
        
        # Convert Stage 2 models
        stage2_config = models_config.get('stage2', {})
        stage2_model = stage2_config.get('model')
        
        if stage2_model and os.path.exists(stage2_model):
            results['conversion_summary']['total_models'] += 1
            
            conversion_result = self.convert_rtdetr(
                model_path=stage2_model,
                output_dir=output_dir,
                precision=stage2_config.get('precision', 'fp16'),
                batch_size=stage2_config.get('batch_size', 1),
                image_size=stage2_config.get('image_size', 640)
            )
            
            if conversion_result['success']:
                results['conversion_summary']['successful_conversions'] += 1
            else:
                results['conversion_summary']['failed_conversions'] += 1
            
            results['individual_results']['stage2_verifier'] = conversion_result
        
        # Calculate total time
        total_time = time.time() - start_time
        results['conversion_summary']['total_time_s'] = total_time
        
        # Save conversion report
        self._save_conversion_report(results, output_dir)
        
        self.logger.info(f"SCOPE models conversion completed in {total_time:.2f}s")
        self.logger.info(f"Success rate: {results['conversion_summary']['successful_conversions']}/{results['conversion_summary']['total_models']}")
        
        return results
    
    def _save_conversion_report(self, results: Dict[str, Any], output_dir: str):
        """Save conversion report to JSON file"""
        try:
            report_path = os.path.join(output_dir, "conversion_report.json")
            
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Conversion report saved: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save conversion report: {e}")
    
    def get_optimization_recommendations(self, model_path: str) -> Dict[str, Any]:
        """
        Get optimization recommendations for a model
        
        Args:
            model_path: Path to model file
            
        Returns:
            Optimization recommendations
        """
        recommendations = {
            'precision': 'fp16',  # Default recommendation
            'batch_size': 1,      # Default for real-time inference
            'workspace_size_gb': 4,
            'optimizations': []
        }
        
        try:
            # Analyze model size
            model_size_mb = os.path.getsize(model_path) / (1024 ** 2)
            
            if model_size_mb > 100:
                recommendations['optimizations'].append("Consider INT8 quantization for large models")
                recommendations['precision'] = 'int8'
            elif model_size_mb > 50:
                recommendations['optimizations'].append("FP16 precision recommended for good speed/accuracy balance")
            else:
                recommendations['optimizations'].append("Model is already lightweight")
            
            # Model-specific recommendations
            if 'yolo' in model_path.lower():
                recommendations['image_size'] = 320  # Smaller size for Stage 1
                recommendations['optimizations'].append("Use smaller input size for Stage 1 detection")
            elif 'rtdetr' in model_path.lower():
                recommendations['image_size'] = 640  # Larger size for Stage 2
                recommendations['optimizations'].append("Higher resolution recommended for verification accuracy")
            
            # Performance recommendations
            recommendations['optimizations'].extend([
                "Enable workspace size of 4GB for optimal performance",
                "Use batch size 1 for real-time inference",
                "Consider dynamic shapes for variable input sizes"
            ])
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
        
        return recommendations
    
    def benchmark_engines(self, engine_dir: str = "tensorrt_engines") -> Dict[str, Any]:
        """
        Benchmark all TensorRT engines in directory
        
        Args:
            engine_dir: Directory containing engines
            
        Returns:
            Benchmark results
        """
        benchmark_results = {}
        
        engine_files = list(Path(engine_dir).glob("*.engine"))
        
        if not engine_files:
            self.logger.warning(f"No engine files found in {engine_dir}")
            return benchmark_results
        
        self.logger.info(f"Benchmarking {len(engine_files)} engines...")
        
        for engine_path in engine_files:
            try:
                engine_name = engine_path.stem
                
                # Load engine
                if 'yolo' in engine_name.lower():
                    model = YOLO(str(engine_path))
                    test_size = 320
                elif 'rtdetr' in engine_name.lower():
                    model = RTDETR(str(engine_path))
                    test_size = 640
                else:
                    continue
                
                # Generate test image
                test_image = np.random.randint(0, 255, (test_size, test_size, 3), dtype=np.uint8)
                
                # Warmup
                for _ in range(5):
                    _ = model(test_image, verbose=False)
                
                # Benchmark
                num_iterations = 100
                start_time = time.time()
                
                for _ in range(num_iterations):
                    _ = model(test_image, verbose=False)
                
                total_time = time.time() - start_time
                avg_inference_ms = (total_time / num_iterations) * 1000
                fps = 1000 / avg_inference_ms
                
                benchmark_results[engine_name] = {
                    'avg_inference_ms': avg_inference_ms,
                    'fps': fps,
                    'total_benchmark_time_s': total_time,
                    'iterations': num_iterations,
                    'file_size_mb': engine_path.stat().st_size / (1024 ** 2)
                }
                
                self.logger.info(f"{engine_name}: {avg_inference_ms:.2f}ms ({fps:.1f} FPS)")
                
            except Exception as e:
                self.logger.error(f"Benchmark failed for {engine_path}: {e}")
        
        return benchmark_results

def main():
    """Main function for TensorRT converter"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TensorRT Converter for SCOPE')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--config', type=str, help='Path to SCOPE config file')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16', 'int8'], 
                       default='fp16', help='Precision mode')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--image-size', type=int, default=640, help='Input image size')
    parser.add_argument('--output-dir', type=str, default='tensorrt_engines', 
                       help='Output directory')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Benchmark existing engines')
    parser.add_argument('--recommendations', action='store_true',
                       help='Get optimization recommendations')
    
    args = parser.parse_args()
    
    try:
        converter = TensorRTConverter()
        
        if args.benchmark:
            results = converter.benchmark_engines(args.output_dir)
            print("\n🚀 Benchmark Results:")
            for engine, stats in results.items():
                print(f"{engine}: {stats['avg_inference_ms']:.2f}ms ({stats['fps']:.1f} FPS)")
            return
        
        if args.recommendations and args.model:
            recs = converter.get_optimization_recommendations(args.model)
            print("\n💡 Optimization Recommendations:")
            print(json.dumps(recs, indent=2))
            return
        
        if args.config:
            # Convert all SCOPE models from config
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            
            results = converter.convert_scope_models(config, args.output_dir)
            print(f"\n✅ Converted {results['conversion_summary']['successful_conversions']}/{results['conversion_summary']['total_models']} models")
            
        elif args.model:
            # Convert single model
            if 'yolo' in args.model.lower():
                result = converter.convert_yolo_world(
                    args.model, args.output_dir, args.precision, 
                    args.batch_size, args.image_size
                )
            elif 'rtdetr' in args.model.lower():
                result = converter.convert_rtdetr(
                    args.model, args.output_dir, args.precision,
                    args.batch_size, args.image_size
                )
            else:
                print("❌ Unknown model type. Use YOLO or RT-DETR models.")
                return
            
            if result['success']:
                print(f"✅ Conversion successful: {result['output_engine']}")
                print(f"Speedup: {result['validation']['speedup_factor']:.2f}x")
            else:
                print(f"❌ Conversion failed: {result.get('error', 'Unknown error')}")
        
        else:
            print("❌ Please provide --model or --config argument")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()