#!/usr/bin/env python3
"""
CUDA Memory Manager - SCOPE Smart Building System
Optimizes memory allocation and management for CUDA/MPS/CPU devices
Provides tensor pooling, stream management, and memory optimization
"""

import torch
import gc
import logging
from typing import Dict, Tuple, Optional, List, Any
from contextlib import contextmanager
import threading
import time

class CUDAMemoryManager:
    """
    Advanced memory manager for CUDA/MPS/CPU devices
    Provides tensor pooling, stream management, and optimization
    """
    
    def __init__(self, device: Optional[torch.device] = None, memory_fraction: float = 0.8):
        self.logger = logging.getLogger(__name__)
        self.device = device or self._get_optimal_device()
        self.memory_fraction = memory_fraction
        self.tensor_pool = {}
        self.streams = {}
        
        # Memory tracking
        self.allocated_tensors = {}
        self.peak_memory_usage = 0
        self.current_memory_usage = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Initialize device-specific optimizations
        self._initialize_device_optimizations()
        self._setup_tensor_pools()
        self._setup_cuda_streams()
        
        self.logger.info(f"CUDA Memory Manager initialized for device: {self.device}")
    
    def _get_optimal_device(self) -> torch.device:
        """Detect and configure optimal device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            return device
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            self.logger.info("Using MPS (Apple Silicon)")
            return device
        else:
            device = torch.device('cpu')
            self.logger.info("Using CPU (consider upgrading to GPU)")
            return device
    
    def _initialize_device_optimizations(self):
        """Initialize device-specific optimizations"""
        if self.device.type == 'cuda':
            self._setup_cuda_optimizations()
        elif self.device.type == 'mps':
            self._setup_mps_optimizations()
        else:
            self._setup_cpu_optimizations()
    
    def _setup_cuda_optimizations(self):
        """CUDA-specific optimizations"""
        try:
            # Memory management
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
            torch.cuda.empty_cache()
            
            # Performance optimizations
            torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
            torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster matmul
            torch.set_float32_matmul_precision('medium')  # Balance speed/precision
            
            # Memory allocation strategy
            torch.cuda.memory._set_allocator_settings("garbage_collection_threshold:0.6,max_split_size_mb:128")
            
            self.logger.info("CUDA optimizations enabled")
            
        except Exception as e:
            self.logger.warning(f"Some CUDA optimizations failed: {e}")
    
    def _setup_mps_optimizations(self):
        """MPS-specific optimizations"""
        try:
            # MPS memory optimization
            torch.mps.set_per_process_memory_fraction(self.memory_fraction)
            
            # Disable problematic features for MPS
            torch.backends.mps.enable_fallback(True)
            
            self.logger.info("MPS optimizations enabled")
            
        except Exception as e:
            self.logger.warning(f"Some MPS optimizations failed: {e}")
    
    def _setup_cpu_optimizations(self):
        """CPU-specific optimizations"""
        try:
            # CPU threading
            torch.set_num_threads(torch.get_num_threads())
            
            # Memory optimization
            torch.backends.mkldnn.enabled = True
            
            self.logger.info("CPU optimizations enabled")
            
        except Exception as e:
            self.logger.warning(f"Some CPU optimizations failed: {e}")
    
    def _setup_tensor_pools(self):
        """Pre-allocate common tensor sizes for reuse"""
        common_shapes = [
            # Detection tensors
            (100, 4),   # Bounding boxes
            (100, 1),   # Confidence scores
            (100,),     # Class indices
            
            # Image processing tensors
            (1, 3, 320, 320),   # Small images
            (1, 3, 640, 640),   # Medium images
            (1, 3, 1280, 1280), # Large images
            
            # Feature tensors
            (1, 256, 40, 40),   # Feature maps
            (1, 512, 20, 20),   # Deep features
            (1, 1024, 10, 10),  # Final features
        ]
        
        for shape in common_shapes:
            try:
                # Create tensors for different data types
                self.tensor_pool[f"float32_{shape}"] = torch.zeros(
                    shape, dtype=torch.float32, device=self.device
                )
                
                if self.device.type in ['cuda', 'mps']:
                    self.tensor_pool[f"float16_{shape}"] = torch.zeros(
                        shape, dtype=torch.float16, device=self.device
                    )
                
                self.tensor_pool[f"int32_{shape}"] = torch.zeros(
                    shape, dtype=torch.int32, device=self.device
                )
                
            except Exception as e:
                self.logger.warning(f"Failed to allocate tensor pool for shape {shape}: {e}")
        
        self.logger.info(f"Tensor pools created with {len(self.tensor_pool)} pre-allocated tensors")
    
    def _setup_cuda_streams(self):
        """Setup CUDA streams for parallel processing"""
        if self.device.type == 'cuda':
            try:
                self.streams = {
                    'stage1_detection': torch.cuda.Stream(),
                    'stage2_verification': torch.cuda.Stream(),
                    'preprocessing': torch.cuda.Stream(),
                    'postprocessing': torch.cuda.Stream(),
                }
                self.logger.info("CUDA streams created for parallel processing")
            except Exception as e:
                self.logger.warning(f"Failed to create CUDA streams: {e}")
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, 
                   zero_init: bool = True) -> torch.Tensor:
        """
        Get tensor from pool or create new one
        
        Args:
            shape: Tensor shape
            dtype: Data type
            zero_init: Whether to zero-initialize
            
        Returns:
            Tensor on the managed device
        """
        with self.lock:
            pool_key = f"{str(dtype).split('.')[-1]}_{shape}"
            
            if pool_key in self.tensor_pool:
                tensor = self.tensor_pool[pool_key]
                if zero_init:
                    tensor.zero_()
                return tensor
            else:
                # Create new tensor if not in pool
                tensor = torch.zeros(shape, dtype=dtype, device=self.device)
                
                # Add to pool if it's a common size
                if len(self.tensor_pool) < 50:  # Limit pool size
                    self.tensor_pool[pool_key] = tensor
                
                return tensor
    
    def get_stream(self, name: str) -> Optional[torch.cuda.Stream]:
        """Get CUDA stream by name"""
        return self.streams.get(name)
    
    @contextmanager
    def cuda_stream(self, stream_name: str):
        """Context manager for CUDA stream usage"""
        if self.device.type == 'cuda' and stream_name in self.streams:
            with torch.cuda.stream(self.streams[stream_name]):
                yield self.streams[stream_name]
        else:
            yield None
    
    def clear_cache(self, force: bool = False):
        """Clear memory cache"""
        if force or self.current_memory_usage > self.memory_fraction * 0.9:
            # Clear tensor pool if it gets too large
            if len(self.tensor_pool) > 100:
                # Keep only the most common tensors
                common_keys = list(self.tensor_pool.keys())[:20]
                new_pool = {k: self.tensor_pool[k] for k in common_keys}
                self.tensor_pool = new_pool
            
            # Device-specific cache clearing
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif self.device.type == 'mps':
                try:
                    torch.mps.empty_cache()
                except:
                    pass  # Not available in all PyTorch versions
            
            # Python garbage collection
            gc.collect()
            
            self.logger.debug("Memory cache cleared")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        stats = {
            'device': str(self.device),
            'tensor_pool_size': len(self.tensor_pool),
            'peak_memory_mb': self.peak_memory_usage,
        }
        
        if self.device.type == 'cuda':
            stats.update({
                'cuda_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'cuda_reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                'cuda_max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
            })
        elif self.device.type == 'mps':
            try:
                stats.update({
                    'mps_allocated_mb': torch.mps.current_allocated_memory() / 1024**2,
                    'mps_driver_allocated_mb': torch.mps.driver_allocated_memory() / 1024**2,
                })
            except:
                pass  # Not available in all PyTorch versions
        
        return stats
    
    def optimize_for_inference(self):
        """Optimize settings for inference workload"""
        if self.device.type == 'cuda':
            # Disable autograd for inference
            torch.set_grad_enabled(False)
            
            # Enable cudnn for consistent input sizes
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Optimize memory allocation
            torch.cuda.empty_cache()
        
        # Set thread count for CPU operations
        torch.set_num_threads(min(4, torch.get_num_threads()))
        
        self.logger.info("Memory manager optimized for inference")
    
    def batch_allocate(self, shapes_and_dtypes: List[Tuple[Tuple[int, ...], torch.dtype]]) -> List[torch.Tensor]:
        """
        Efficiently allocate multiple tensors in batch
        
        Args:
            shapes_and_dtypes: List of (shape, dtype) tuples
            
        Returns:
            List of allocated tensors
        """
        tensors = []
        
        with self.lock:
            for shape, dtype in shapes_and_dtypes:
                tensor = self.get_tensor(shape, dtype, zero_init=True)
                tensors.append(tensor)
        
        return tensors
    
    def warm_up_device(self, warmup_shapes: Optional[List[Tuple[int, ...]]] = None):
        """Warm up device with dummy operations"""
        if warmup_shapes is None:
            warmup_shapes = [(1, 3, 320, 320), (1, 3, 640, 640)]
        
        self.logger.info("Warming up device...")
        
        for shape in warmup_shapes:
            try:
                tensor = self.get_tensor(shape, torch.float16 if self.device.type in ['cuda', 'mps'] else torch.float32)
                
                # Dummy operations to warm up kernels
                if self.device.type == 'cuda':
                    with torch.cuda.stream(self.streams.get('preprocessing', torch.cuda.default_stream())):
                        _ = tensor * 2.0
                        _ = torch.relu(tensor)
                        _ = torch.nn.functional.adaptive_avg_pool2d(tensor, (1, 1))
                else:
                    _ = tensor * 2.0
                    _ = torch.relu(tensor)
                
            except Exception as e:
                self.logger.warning(f"Warmup failed for shape {shape}: {e}")
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        self.logger.info("Device warmup completed")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.clear_cache(force=True)
            if hasattr(self, 'streams'):
                # CUDA streams are automatically cleaned up
                pass
        except:
            pass

# Global memory manager instance
_global_memory_manager = None

def get_memory_manager(device: Optional[torch.device] = None, memory_fraction: float = 0.8) -> CUDAMemoryManager:
    """Get global memory manager instance"""
    global _global_memory_manager
    
    if _global_memory_manager is None:
        _global_memory_manager = CUDAMemoryManager(device, memory_fraction)
    
    return _global_memory_manager

def clear_global_memory_manager():
    """Clear global memory manager"""
    global _global_memory_manager
    
    if _global_memory_manager is not None:
        _global_memory_manager.clear_cache(force=True)
        del _global_memory_manager
        _global_memory_manager = None