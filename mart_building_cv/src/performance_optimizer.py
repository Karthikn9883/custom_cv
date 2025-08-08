import cv2
import os
import sys
import psutil
import platform
import subprocess
from typing import Dict, List, Tuple, Optional
import numpy as np

class PerformanceOptimizer:
    """
    Windows and NVIDIA GPU performance optimization utility for the 2D-3D mapping system.
    Optimizes OpenCV, CUDA operations, and system settings for RTX 4070 and 32GB RAM.
    """
    
    def __init__(self):
        self.system_info = {}
        self.cuda_available = False
        self.gpu_info = {}
        self.optimizations_applied = []
        
        self._detect_system_specs()
        self._detect_gpu_capabilities()
    
    def _detect_system_specs(self):
        """Detect system specifications."""
        self.system_info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'processor': platform.processor(),
            'architecture': platform.architecture()[0],
            'total_ram_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'available_ram_gb': round(psutil.virtual_memory().available / (1024**3), 1),
            'cpu_cores': psutil.cpu_count(logical=False),
            'cpu_threads': psutil.cpu_count(logical=True),
            'python_version': sys.version.split()[0]
        }
        
        print(f"System detected: {self.system_info['platform']} {self.system_info['architecture']}")
        print(f"RAM: {self.system_info['total_ram_gb']:.1f}GB total, {self.system_info['available_ram_gb']:.1f}GB available")
        print(f"CPU: {self.system_info['cpu_cores']} cores, {self.system_info['cpu_threads']} threads")
    
    def _detect_gpu_capabilities(self):
        """Detect NVIDIA GPU capabilities and CUDA support."""
        try:
            # Check OpenCV CUDA support
            self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
            
            if self.cuda_available:
                device_count = cv2.cuda.getCudaEnabledDeviceCount()
                print(f"✓ CUDA enabled devices detected: {device_count}")
                
                # Get GPU memory info if possible
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_name = pynvml.nvmlDeviceGetName(handle).decode()
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    self.gpu_info = {
                        'name': gpu_name,
                        'memory_total_gb': round(meminfo.total / 1024**3, 1),
                        'memory_free_gb': round(meminfo.free / 1024**3, 1),
                        'memory_used_gb': round(meminfo.used / 1024**3, 1)
                    }
                    
                    print(f"✓ GPU: {gpu_name}")
                    print(f"✓ GPU Memory: {self.gpu_info['memory_total_gb']:.1f}GB total, {self.gpu_info['memory_free_gb']:.1f}GB free")
                    
                except ImportError:
                    print("ℹ  Install nvidia-ml-py3 for detailed GPU monitoring")
                except Exception as e:
                    print(f"Warning: Could not get detailed GPU info: {e}")
            else:
                print("⚠️  CUDA not available in OpenCV")
                
        except Exception as e:
            print(f"Warning: Could not detect CUDA capabilities: {e}")
    
    def optimize_opencv_settings(self):
        """Optimize OpenCV settings for Windows and NVIDIA hardware."""
        optimizations = []
        
        try:
            # Enable optimizations
            cv2.setUseOptimized(True)
            if cv2.useOptimized():
                optimizations.append("OpenCV optimizations enabled")
            
            # Set number of threads for OpenCV operations
            optimal_threads = min(self.system_info['cpu_threads'], 8)  # Don't use all threads
            cv2.setNumThreads(optimal_threads)
            optimizations.append(f"OpenCV threads set to {optimal_threads}")
            
            # CUDA optimizations if available
            if self.cuda_available:
                # Set CUDA device
                cv2.cuda.setDevice(0)
                optimizations.append("CUDA device 0 selected")
                
                # Enable fast math for CUDA operations
                try:
                    cv2.cuda.setBufferPoolUsage(True)
                    optimizations.append("CUDA buffer pooling enabled")
                except:
                    pass
            
            self.optimizations_applied.extend(optimizations)
            return optimizations
            
        except Exception as e:
            print(f"Warning: Some OpenCV optimizations failed: {e}")
            return optimizations
    
    def optimize_memory_settings(self):
        """Optimize memory usage for large point clouds and video processing."""
        optimizations = []
        
        try:
            # Calculate optimal buffer sizes based on available RAM
            available_gb = self.system_info['available_ram_gb']
            
            # Use up to 25% of available RAM for point cloud processing
            max_pointcloud_memory_gb = min(available_gb * 0.25, 8.0)  # Cap at 8GB
            max_pointcloud_points = int(max_pointcloud_memory_gb * 1024**3 / (3 * 4))  # 3 floats per point
            
            optimizations.append(f"Max point cloud size: {max_pointcloud_points:,} points ({max_pointcloud_memory_gb:.1f}GB)")
            
            # Video buffer optimization
            optimal_video_buffers = min(available_gb // 4, 8)  # 1 buffer per 4GB RAM, max 8
            optimizations.append(f"Optimal video buffer count: {optimal_video_buffers}")
            
            self.optimizations_applied.extend(optimizations)
            return optimizations
            
        except Exception as e:
            print(f"Warning: Memory optimization failed: {e}")
            return optimizations
    
    def optimize_rtsp_settings(self) -> Dict[str, any]:
        """Get optimized RTSP connection settings for Windows."""
        return {
            'buffer_size': 1,  # Minimize latency
            'timeout_ms': 5000,  # 5 second timeout
            'backend': cv2.CAP_FFMPEG,  # Use FFmpeg backend on Windows
            'thread_count': 2,  # Dedicated threads for RTSP
            'gpu_decode': self.cuda_available,  # Use GPU decoding if available
        }
    
    def create_optimized_video_capture(self, rtsp_url: str) -> cv2.VideoCapture:
        """Create optimized VideoCapture object for RTSP streams."""
        cap = cv2.VideoCapture()
        
        try:
            # Set backend to FFmpeg for better RTSP support on Windows
            if not cap.open(rtsp_url, cv2.CAP_FFMPEG):
                # Fallback to default backend
                cap = cv2.VideoCapture(rtsp_url)
            
            if cap.isOpened():
                # Apply optimizations
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
                cap.set(cv2.CAP_PROP_FPS, 30)  # Try to set 30fps
                
                # Windows-specific optimizations
                if platform.system() == 'Windows':
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
                
                print("✓ Optimized RTSP connection established")
                self.optimizations_applied.append("RTSP optimizations applied")
            
            return cap
            
        except Exception as e:
            print(f"Warning: Could not apply RTSP optimizations: {e}")
            return cv2.VideoCapture(rtsp_url)
    
    def optimize_matplotlib_backend(self):
        """Optimize matplotlib backend for Windows with NVIDIA GPU."""
        optimizations = []
        
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            
            # Try Qt5Agg first for Windows with GPU acceleration
            current_backend = matplotlib.get_backend()
            
            if platform.system() == 'Windows':
                preferred_backends = ['Qt5Agg', 'TkAgg']
                
                for backend in preferred_backends:
                    try:
                        matplotlib.use(backend, force=True)
                        # Test if backend works
                        fig = plt.figure(figsize=(1, 1))
                        plt.close(fig)
                        
                        if matplotlib.get_backend() == backend:
                            optimizations.append(f"Matplotlib backend set to {backend}")
                            break
                    except:
                        continue
                
                # Enable interactive mode for better performance
                plt.ion()
                optimizations.append("Interactive plotting enabled")
            
            self.optimizations_applied.extend(optimizations)
            return optimizations
            
        except Exception as e:
            print(f"Warning: Matplotlib optimization failed: {e}")
            return optimizations
    
    def set_process_priority(self):
        """Set high priority for the current process on Windows."""
        optimizations = []
        
        try:
            if platform.system() == 'Windows':
                import psutil
                
                current_process = psutil.Process()
                current_process.nice(psutil.HIGH_PRIORITY_CLASS)
                optimizations.append("Process priority set to HIGH")
                
                self.optimizations_applied.extend(optimizations)
            
        except Exception as e:
            print(f"Warning: Could not set process priority: {e}")
        
        return optimizations
    
    def optimize_numpy_settings(self):
        """Optimize NumPy for multi-core processing."""
        optimizations = []
        
        try:
            # Set BLAS thread count
            optimal_blas_threads = min(self.system_info['cpu_cores'], 8)
            
            os.environ['OMP_NUM_THREADS'] = str(optimal_blas_threads)
            os.environ['MKL_NUM_THREADS'] = str(optimal_blas_threads)
            os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_blas_threads)
            
            optimizations.append(f"NumPy/BLAS threads set to {optimal_blas_threads}")
            
            # Enable fast math if possible
            try:
                import numba
                os.environ['NUMBA_NUM_THREADS'] = str(optimal_blas_threads)
                optimizations.append("Numba optimizations enabled")
            except ImportError:
                pass
            
            self.optimizations_applied.extend(optimizations)
            return optimizations
            
        except Exception as e:
            print(f"Warning: NumPy optimization failed: {e}")
            return optimizations
    
    def apply_all_optimizations(self) -> List[str]:
        """Apply all available optimizations."""
        print(f"\\n{'='*60}")
        print("APPLYING PERFORMANCE OPTIMIZATIONS")
        print(f"{'='*60}")
        
        all_optimizations = []
        
        # Apply optimizations
        all_optimizations.extend(self.optimize_opencv_settings())
        all_optimizations.extend(self.optimize_memory_settings())
        all_optimizations.extend(self.optimize_matplotlib_backend())
        all_optimizations.extend(self.set_process_priority())
        all_optimizations.extend(self.optimize_numpy_settings())
        
        print("\\nOptimizations applied:")
        for i, opt in enumerate(all_optimizations, 1):
            print(f"  {i}. ✓ {opt}")
        
        print(f"\\n{'='*60}")
        print(f"SYSTEM READY FOR HIGH-PERFORMANCE OPERATION")
        print(f"{'='*60}\\n")
        
        return all_optimizations
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance recommendations based on system specs."""
        recommendations = []
        
        # RAM recommendations
        if self.system_info['total_ram_gb'] >= 32:
            recommendations.append("✓ Excellent RAM capacity for large point clouds")
        elif self.system_info['total_ram_gb'] >= 16:
            recommendations.append("⚠️  Consider point cloud downsampling for very large datasets")
        else:
            recommendations.append("⚠️  Limited RAM - enable aggressive point cloud downsampling")
        
        # GPU recommendations
        if self.cuda_available and 'RTX' in str(self.gpu_info.get('name', '')):
            recommendations.append("✓ NVIDIA RTX GPU detected - excellent for accelerated processing")
        elif self.cuda_available:
            recommendations.append("✓ CUDA-capable GPU detected - good for acceleration")
        else:
            recommendations.append("⚠️  No CUDA support - CPU-only processing will be slower")
        
        # CPU recommendations
        if self.system_info['cpu_threads'] >= 16:
            recommendations.append("✓ High-performance CPU for multi-threaded operations")
        elif self.system_info['cpu_threads'] >= 8:
            recommendations.append("✓ Good CPU performance for real-time processing")
        else:
            recommendations.append("⚠️  Limited CPU cores - may affect real-time performance")
        
        # Platform recommendations
        if platform.system() == 'Windows':
            recommendations.append("✓ Windows optimizations enabled")
        
        return recommendations
    
    def print_system_summary(self):
        """Print comprehensive system and optimization summary."""
        print(f"\\n{'='*70}")
        print("PERFORMANCE OPTIMIZATION SUMMARY")
        print(f"{'='*70}")
        
        print("\\nSYSTEM SPECIFICATIONS:")
        print(f"  Platform: {self.system_info['platform']} {self.system_info['platform_version']}")
        print(f"  CPU: {self.system_info['processor']}")
        print(f"  Cores/Threads: {self.system_info['cpu_cores']}/{self.system_info['cpu_threads']}")
        print(f"  RAM: {self.system_info['total_ram_gb']:.1f}GB total")
        
        if self.gpu_info:
            print(f"  GPU: {self.gpu_info['name']}")
            print(f"  GPU Memory: {self.gpu_info['memory_total_gb']:.1f}GB")
        
        print(f"\\nCUDA Support: {'✓ Enabled' if self.cuda_available else '✗ Not Available'}")
        
        print("\\nRECOMMENDATIONS:")
        for rec in self.get_performance_recommendations():
            print(f"  {rec}")
        
        if self.optimizations_applied:
            print("\\nAPPLIED OPTIMIZATIONS:")
            for i, opt in enumerate(self.optimizations_applied, 1):
                print(f"  {i}. {opt}")
        
        print(f"\\n{'='*70}\\n")

def test_performance_optimizer():
    """Test the performance optimizer."""
    optimizer = PerformanceOptimizer()
    optimizer.apply_all_optimizations()
    optimizer.print_system_summary()
    
    # Test RTSP optimization
    rtsp_settings = optimizer.optimize_rtsp_settings()
    print("RTSP Optimization Settings:")
    for key, value in rtsp_settings.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_performance_optimizer()