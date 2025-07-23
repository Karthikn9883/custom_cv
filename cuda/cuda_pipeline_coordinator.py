#!/usr/bin/env python3
"""
CUDA Pipeline Coordinator - SCOPE Smart Building System
Orchestrates Stage 1 → Stage 2 flow with smart token management
Implements the complete 4.1 architecture with confidence fusion and temporal filtering
"""

import cv2
import time
import json
import yaml
import logging
import threading
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict, deque
from dataclasses import asdict
import numpy as np

from cuda_memory_manager import get_memory_manager
from cuda_stage1_detector import CUDAStage1Detector, DetectionToken
from cuda_stage2_verifier import CUDAStage2Verifier, ValidatedEvent

class TokenQueue:
    """Thread-safe token queue for Stage 1 → Stage 2 communication"""
    
    def __init__(self, max_size: int = 1000):
        self.queue = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
    
    def put(self, token: DetectionToken, frame: np.ndarray):
        """Add token and frame to queue"""
        with self.condition:
            self.queue.append((token, frame, time.time()))
            self.condition.notify()
    
    def get(self, timeout: Optional[float] = None) -> Optional[Tuple[DetectionToken, np.ndarray, float]]:
        """Get token and frame from queue"""
        with self.condition:
            if not self.queue and timeout:
                self.condition.wait(timeout)
            
            if self.queue:
                return self.queue.popleft()
            return None
    
    def size(self) -> int:
        """Get current queue size"""
        with self.lock:
            return len(self.queue)

class PerformanceMonitor:
    """Monitor pipeline performance and detect bottlenecks"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.stage1_times = deque(maxlen=window_size)
        self.stage2_times = deque(maxlen=window_size)
        self.end_to_end_times = deque(maxlen=window_size)
        self.token_counts = deque(maxlen=window_size)
        self.validation_rates = deque(maxlen=window_size)
        
        self.lock = threading.Lock()
    
    def record_stage1(self, processing_time_ms: float, token_count: int):
        """Record Stage 1 performance"""
        with self.lock:
            self.stage1_times.append(processing_time_ms)
            self.token_counts.append(token_count)
    
    def record_stage2(self, processing_time_ms: float, validation_rate: float):
        """Record Stage 2 performance"""
        with self.lock:
            self.stage2_times.append(processing_time_ms)
            self.validation_rates.append(validation_rate)
    
    def record_end_to_end(self, total_time_ms: float):
        """Record end-to-end performance"""
        with self.lock:
            self.end_to_end_times.append(total_time_ms)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self.lock:
            stats = {}
            
            if self.stage1_times:
                stats['stage1'] = {
                    'avg_time_ms': np.mean(self.stage1_times),
                    'p95_time_ms': np.percentile(self.stage1_times, 95),
                    'max_time_ms': np.max(self.stage1_times)
                }
            
            if self.stage2_times:
                stats['stage2'] = {
                    'avg_time_ms': np.mean(self.stage2_times),
                    'p95_time_ms': np.percentile(self.stage2_times, 95),
                    'max_time_ms': np.max(self.stage2_times)
                }
            
            if self.end_to_end_times:
                stats['end_to_end'] = {
                    'avg_time_ms': np.mean(self.end_to_end_times),
                    'p95_time_ms': np.percentile(self.end_to_end_times, 95),
                    'max_time_ms': np.max(self.end_to_end_times)
                }
            
            if self.token_counts:
                stats['tokens'] = {
                    'avg_per_frame': np.mean(self.token_counts),
                    'max_per_frame': np.max(self.token_counts)
                }
            
            if self.validation_rates:
                stats['validation'] = {
                    'avg_rate': np.mean(self.validation_rates),
                    'min_rate': np.min(self.validation_rates)
                }
            
            return stats

class CUDAPipelineCoordinator:
    """
    Main pipeline coordinator implementing SCOPE 4.1 architecture
    Manages Stage 1 → Stage 2 flow with smart optimization
    """
    
    def __init__(self, config_path: str = "configs/cuda_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Components
        self.stage1_detector = None
        self.stage2_verifier = None
        self.memory_manager = get_memory_manager()
        
        # Communication
        self.token_queue = TokenQueue(max_size=self.config.get('pipeline', {}).get('queue_size', 1000))
        self.performance_monitor = PerformanceMonitor()
        
        # Threading
        self.stage2_thread = None
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Callbacks
        self.event_callbacks = []  # For validated events
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize pipeline
        self._initialize_pipeline()
        
        self.logger.info("CUDA Pipeline Coordinator initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default pipeline configuration"""
        return {
            'pipeline': {
                'queue_size': 1000,
                'stage2_batch_size': 5,
                'stage2_timeout_ms': 100,
                'frame_skip': 1,  # Process every Nth frame
                'adaptive_processing': True,
                'max_end_to_end_ms': 2000
            },
            'device': {
                'auto_detect': True,
                'preferred': 'cuda'
            },
            'stage1': {
                'target_latency_ms': 20,
                'confidence_threshold': 0.35
            },
            'stage2': {
                'max_verification_time_ms': 500,
                'confidence_threshold': 0.65,
                'temporal_frames': 3
            }
        }
    
    def _initialize_pipeline(self):
        """Initialize pipeline components"""
        try:
            # Initialize Stage 1 detector
            self.logger.info("Initializing Stage 1 detector...")
            self.stage1_detector = CUDAStage1Detector(self.config_path)
            
            # Initialize Stage 2 verifier
            self.logger.info("Initializing Stage 2 verifier...")
            self.stage2_verifier = CUDAStage2Verifier(self.config_path)
            
            self.logger.info("Pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def add_event_callback(self, callback: Callable[[List[ValidatedEvent]], None]):
        """Add callback for validated events"""
        self.event_callbacks.append(callback)
    
    def _start_stage2_worker(self):
        """Start Stage 2 worker thread"""
        self.stage2_thread = threading.Thread(target=self._stage2_worker, daemon=True)
        self.stage2_thread.start()
        self.logger.info("Stage 2 worker thread started")
    
    def _stage2_worker(self):
        """Stage 2 worker thread for token verification"""
        batch_size = self.config['pipeline']['stage2_batch_size']
        timeout_ms = self.config['pipeline']['stage2_timeout_ms'] / 1000.0
        
        token_batch = []
        frame_batch = []
        camera_ids = []
        
        while not self.shutdown_event.is_set():
            try:
                # Collect batch of tokens
                while len(token_batch) < batch_size:
                    result = self.token_queue.get(timeout=timeout_ms)
                    if result is None:
                        break  # Timeout reached
                    
                    token, frame, timestamp = result
                    
                    # Check if token is still fresh (not too old)
                    if time.time() - timestamp < 1.0:  # 1 second timeout
                        token_batch.append(token)
                        frame_batch.append(frame)
                        camera_ids.append(token.camera_id)
                
                # Process batch if we have tokens
                if token_batch:
                    self._process_stage2_batch(token_batch, frame_batch, camera_ids)
                    
                    # Clear batch for next iteration
                    token_batch.clear()
                    frame_batch.clear()
                    camera_ids.clear()
                
            except Exception as e:
                self.logger.error(f"Stage 2 worker error: {e}")
                # Continue processing
    
    def _process_stage2_batch(self, tokens: List[DetectionToken], 
                             frames: List[np.ndarray], camera_ids: List[str]):
        """Process batch of tokens through Stage 2"""
        start_time = time.time()
        
        try:
            # Group tokens by camera for temporal filtering
            camera_groups = defaultdict(list)
            for i, token in enumerate(tokens):
                camera_groups[camera_ids[i]].append((token, frames[i]))
            
            all_validated_events = []
            
            # Process each camera group
            for camera_id, camera_tokens in camera_groups.items():
                if not camera_tokens:
                    continue
                
                # Use the most recent frame for this camera
                tokens_list = [ct[0] for ct in camera_tokens]
                latest_frame = camera_tokens[-1][1]  # Most recent frame
                
                # Batch verify tokens
                validated_events = self.stage2_verifier.batch_verify_tokens(
                    tokens_list, latest_frame, camera_id
                )
                
                all_validated_events.extend(validated_events)
            
            # Record performance
            processing_time = (time.time() - start_time) * 1000
            validation_rate = len(all_validated_events) / len(tokens) if tokens else 0
            
            self.performance_monitor.record_stage2(processing_time, validation_rate)
            
            # Trigger callbacks for validated events
            if all_validated_events:
                for callback in self.event_callbacks:
                    try:
                        callback(all_validated_events)
                    except Exception as e:
                        self.logger.error(f"Event callback error: {e}")
                
                self.logger.info(f"Validated {len(all_validated_events)} events from {len(tokens)} tokens")
        
        except Exception as e:
            self.logger.error(f"Stage 2 batch processing error: {e}")
    
    def process_frame(self, frame: np.ndarray, camera_id: str = "cam_01") -> Dict[str, Any]:
        """
        Process single frame through complete pipeline
        
        Args:
            frame: Input frame
            camera_id: Camera identifier
            
        Returns:
            Processing results and performance stats
        """
        pipeline_start = time.time()
        
        # Stage 1: Fast token detection
        stage1_start = time.time()
        tokens = self.stage1_detector.detect_frame(frame, camera_id)
        stage1_time = (time.time() - stage1_start) * 1000
        
        # Record Stage 1 performance
        self.performance_monitor.record_stage1(stage1_time, len(tokens))
        
        # Filter tokens for Stage 2
        stage2_tokens = self.stage1_detector.filter_tokens_for_stage2(tokens)
        
        # Queue tokens for Stage 2 processing
        for token in stage2_tokens:
            self.token_queue.put(token, frame.copy())
        
        # Record end-to-end time for this frame
        end_to_end_time = (time.time() - pipeline_start) * 1000
        self.performance_monitor.record_end_to_end(end_to_end_time)
        
        return {
            'stage1_tokens': tokens,
            'stage2_queued': len(stage2_tokens),
            'stage1_time_ms': stage1_time,
            'queue_size': self.token_queue.size(),
            'performance': self.performance_monitor.get_stats()
        }
    
    def start_pipeline(self):
        """Start the complete pipeline"""
        if self.running:
            self.logger.warning("Pipeline is already running")
            return
        
        self.running = True
        self.shutdown_event.clear()
        
        # Start Stage 2 worker
        self._start_stage2_worker()
        
        self.logger.info("Pipeline started successfully")
    
    def stop_pipeline(self):
        """Stop the pipeline gracefully"""
        if not self.running:
            return
        
        self.logger.info("Stopping pipeline...")
        
        # Signal shutdown
        self.shutdown_event.set()
        self.running = False
        
        # Wait for Stage 2 worker to finish
        if self.stage2_thread and self.stage2_thread.is_alive():
            self.stage2_thread.join(timeout=5.0)
        
        # Clear memory
        self.memory_manager.clear_cache(force=True)
        
        self.logger.info("Pipeline stopped")
    
    def run_realtime(self, input_source=0, camera_id: str = "cam_01"):
        """
        Run real-time pipeline on video source
        
        Args:
            input_source: Video source (camera index or RTSP URL)
            camera_id: Camera identifier
        """
        self.logger.info(f"Starting real-time pipeline on source: {input_source}")
        
        # Initialize video capture
        if isinstance(input_source, str) and input_source.startswith('rtsp'):
            cap = cv2.VideoCapture(input_source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(input_source)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to open video source: {input_source}")
            return
        
        # Set camera properties for optimal performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Start pipeline
        self.start_pipeline()
        
        frame_count = 0
        frame_skip = self.config['pipeline']['frame_skip']
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame")
                    break
                
                frame_count += 1
                
                # Frame skipping for performance
                if frame_count % frame_skip != 0:
                    continue
                
                # Process frame
                results = self.process_frame(frame, camera_id)
                
                # Visualize results
                annotated_frame = self._draw_pipeline_info(frame, results)
                
                # Display frame
                cv2.imshow(f'SCOPE Pipeline - {camera_id}', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('s'):  # Print stats
                    stats = self.get_complete_stats()
                    self.logger.info(f"Pipeline Stats: {json.dumps(stats, indent=2)}")
                elif key == ord('r'):  # Reset stats
                    self.performance_monitor = PerformanceMonitor()
                    self.logger.info("Performance stats reset")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.stop_pipeline()
            
            # Final performance report
            final_stats = self.get_complete_stats()
            self.logger.info(f"Session complete. Final stats: {json.dumps(final_stats, indent=2)}")
    
    def _draw_pipeline_info(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Draw pipeline information on frame"""
        annotated_frame = frame.copy()
        
        # Draw Stage 1 detections
        tokens = results.get('stage1_tokens', [])
        annotated_frame = self.stage1_detector.draw_detections(annotated_frame, tokens)
        
        # Add pipeline info
        info_y = 30
        
        # Stage 1 info
        stage1_text = f"Stage 1: {results['stage1_time_ms']:.1f}ms | Tokens: {len(tokens)}"
        cv2.putText(annotated_frame, stage1_text, (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        info_y += 25
        
        # Stage 2 info
        stage2_text = f"Stage 2 Queue: {results['queue_size']} | Queued: {results['stage2_queued']}"
        cv2.putText(annotated_frame, stage2_text, (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        info_y += 25
        
        # Performance info
        perf = results.get('performance', {})
        if 'end_to_end' in perf:
            e2e_text = f"End-to-End: {perf['end_to_end']['avg_time_ms']:.1f}ms"
            cv2.putText(annotated_frame, e2e_text, (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return annotated_frame
    
    def get_complete_stats(self) -> Dict[str, Any]:
        """Get complete pipeline statistics"""
        stats = {
            'pipeline': self.performance_monitor.get_stats(),
            'stage1': self.stage1_detector.get_performance_stats(),
            'stage2': self.stage2_verifier.get_performance_stats(),
            'memory': self.memory_manager.get_memory_stats(),
            'queue_size': self.token_queue.size()
        }
        
        return stats
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.stop_pipeline()
        except:
            pass

def main():
    """Test the CUDA pipeline coordinator"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CUDA Pipeline Coordinator')
    parser.add_argument('--config', type=str, default='configs/cuda_config.yaml')
    parser.add_argument('--input', type=str, default=0, 
                       help='Input source: webcam (0) or RTSP URL')
    parser.add_argument('--camera-id', type=str, default='cam_01')
    parser.add_argument('--test', action='store_true', help='Test initialization only')
    
    args = parser.parse_args()
    
    # Event callback for testing
    def log_validated_events(events: List[ValidatedEvent]):
        for event in events:
            print(f"✅ Validated: {event.category} (confidence: {event.confidence:.2f})")
    
    # Initialize pipeline
    pipeline = CUDAPipelineCoordinator(args.config)
    pipeline.add_event_callback(log_validated_events)
    
    if args.test:
        stats = pipeline.get_complete_stats()
        print("✅ CUDA Pipeline Coordinator initialized successfully")
        print(f"Stage 1 device: {pipeline.stage1_detector.device}")
        print(f"Stage 2 device: {pipeline.stage2_verifier.device}")
        return
    
    # Convert input argument
    input_source = args.input
    if isinstance(input_source, str) and input_source.isdigit():
        input_source = int(input_source)
    
    # Run real-time pipeline
    pipeline.run_realtime(input_source, args.camera_id)

if __name__ == "__main__":
    main()