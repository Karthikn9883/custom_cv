#!/usr/bin/env python3
"""
Redis Token Manager - SCOPE Smart Building System
Manages Stage 1 tokens in Redis queue for distributed processing
Provides high-performance token storage and retrieval
"""

import json
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import asdict
from collections import defaultdict

try:
    import redis
    from redis.exceptions import ConnectionError, TimeoutError
except ImportError:
    print("Error: redis not installed. Run: pip install redis")
    exit(1)

from cuda.cuda_stage1_detector import DetectionToken

class RedisTokenManager:
    """
    High-performance Redis-based token management
    Handles Stage 1 token queuing and Stage 2 coordination
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_config = config.get('redis', {})
        
        # Redis connection
        self.redis_client = None
        self.connection_pool = None
        
        # Queues
        self.stage1_queue = self.redis_config.get('stage1_queue', 'scope:tokens:stage1')
        self.stage2_queue = self.redis_config.get('stage2_queue', 'scope:tokens:stage2')
        self.events_queue = self.redis_config.get('events_queue', 'scope:events:validated')
        
        # Performance tracking
        self.tokens_published = 0
        self.tokens_consumed = 0
        self.publish_errors = 0
        self.consume_errors = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize Redis connection
        self._initialize_redis()
        
        self.logger.info("Redis Token Manager initialized")
    
    def _initialize_redis(self):
        """Initialize Redis connection with error handling"""
        try:
            redis_host = self.redis_config.get('host', 'localhost')
            redis_port = self.redis_config.get('port', 6379)
            redis_db = self.redis_config.get('db', 0)
            redis_password = self.redis_config.get('password', None)
            
            # Create connection pool for better performance
            self.connection_pool = redis.ConnectionPool(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                max_connections=self.redis_config.get('max_connections', 20),
                retry_on_timeout=True,
                socket_timeout=self.redis_config.get('timeout', 5),
                socket_connect_timeout=self.redis_config.get('connect_timeout', 5)
            )
            
            # Create Redis client
            self.redis_client = redis.Redis(
                connection_pool=self.connection_pool,
                decode_responses=True
            )
            
            # Test connection
            self.redis_client.ping()
            
            # Set up Redis optimizations
            self._setup_redis_optimizations()
            
            self.logger.info(f"Redis connected: {redis_host}:{redis_port}/{redis_db}")
            
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}")
            self.redis_client = None
            raise
    
    def _setup_redis_optimizations(self):
        """Setup Redis optimizations for high throughput"""
        try:
            # Configure Redis for optimal performance
            pipeline = self.redis_client.pipeline()
            
            # Set memory policies for token queues (LRU eviction)
            pipeline.config_set('maxmemory-policy', 'allkeys-lru')
            
            # Optimize for write performance
            pipeline.config_set('save', '300 10')  # Save every 5 min if 10+ keys changed
            
            # Execute optimizations
            pipeline.execute()
            
            self.logger.info("Redis optimizations applied")
            
        except Exception as e:
            self.logger.warning(f"Some Redis optimizations failed: {e}")
    
    def publish_stage1_tokens(self, tokens: List[DetectionToken], camera_id: str) -> bool:
        """
        Publish Stage 1 tokens to Redis queue
        
        Args:
            tokens: List of detection tokens
            camera_id: Camera identifier
            
        Returns:
            Success status
        """
        if not self.redis_client or not tokens:
            return False
        
        try:
            pipeline = self.redis_client.pipeline()
            
            for token in tokens:
                # Create token data with metadata
                token_data = {
                    'token': asdict(token),
                    'camera_id': camera_id,
                    'published_at': time.time(),
                    'ttl': self.redis_config.get('token_ttl', 60)  # 60 second TTL
                }
                
                # Serialize and push to queue
                token_json = json.dumps(token_data, separators=(',', ':'))
                pipeline.lpush(self.stage1_queue, token_json)
            
            # Set TTL on queue to prevent infinite growth
            queue_ttl = self.redis_config.get('queue_ttl', 3600)  # 1 hour
            pipeline.expire(self.stage1_queue, queue_ttl)
            
            # Execute batch operation
            pipeline.execute()
            
            with self.lock:
                self.tokens_published += len(tokens)
            
            self.logger.debug(f"Published {len(tokens)} tokens to Redis")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish tokens: {e}")
            with self.lock:
                self.publish_errors += 1
            return False
    
    def consume_stage1_tokens(self, batch_size: int = 10, 
                             timeout: float = 1.0) -> List[Tuple[DetectionToken, str, float]]:
        """
        Consume Stage 1 tokens from Redis queue
        
        Args:
            batch_size: Maximum tokens to consume
            timeout: Timeout in seconds
            
        Returns:
            List of (token, camera_id, timestamp) tuples
        """
        if not self.redis_client:
            return []
        
        try:
            tokens = []
            
            # Use pipeline for efficient batch operations
            for _ in range(batch_size):
                # Pop token from queue with timeout
                result = self.redis_client.brpop(self.stage1_queue, timeout=timeout)
                if not result:
                    break  # Timeout or empty queue
                
                queue_name, token_json = result
                
                try:
                    # Parse token data
                    token_data = json.loads(token_json)
                    
                    # Check TTL
                    published_at = token_data.get('published_at', 0)
                    ttl = token_data.get('ttl', 60)
                    
                    if time.time() - published_at > ttl:
                        self.logger.debug("Discarded expired token")
                        continue
                    
                    # Reconstruct token
                    token_dict = token_data['token']
                    token = DetectionToken(**token_dict)
                    
                    camera_id = token_data['camera_id']
                    timestamp = published_at
                    
                    tokens.append((token, camera_id, timestamp))
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse token JSON: {e}")
                    continue
            
            with self.lock:
                self.tokens_consumed += len(tokens)
            
            if tokens:
                self.logger.debug(f"Consumed {len(tokens)} tokens from Redis")
            
            return tokens
            
        except Exception as e:
            self.logger.error(f"Failed to consume tokens: {e}")
            with self.lock:
                self.consume_errors += 1
            return []
    
    def publish_stage2_results(self, verification_results: List[Dict[str, Any]], 
                              camera_id: str) -> bool:
        """
        Publish Stage 2 verification results
        
        Args:
            verification_results: List of verification results
            camera_id: Camera identifier
            
        Returns:
            Success status
        """
        if not self.redis_client or not verification_results:
            return False
        
        try:
            pipeline = self.redis_client.pipeline()
            
            for result in verification_results:
                # Add metadata
                result_data = {
                    'result': result,
                    'camera_id': camera_id,
                    'verified_at': time.time()
                }
                
                result_json = json.dumps(result_data, separators=(',', ':'))
                pipeline.lpush(self.stage2_queue, result_json)
            
            # Set TTL
            pipeline.expire(self.stage2_queue, self.redis_config.get('queue_ttl', 3600))
            pipeline.execute()
            
            self.logger.debug(f"Published {len(verification_results)} verification results")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish verification results: {e}")
            return False
    
    def publish_validated_events(self, events: List[Dict[str, Any]]) -> bool:
        """
        Publish final validated events
        
        Args:
            events: List of validated events
            
        Returns:
            Success status
        """
        if not self.redis_client or not events:
            return False
        
        try:
            pipeline = self.redis_client.pipeline()
            
            for event in events:
                # Add final validation metadata
                event_data = {
                    'event': event,
                    'validated_at': time.time(),
                    'pipeline_version': '4.1'
                }
                
                event_json = json.dumps(event_data, separators=(',', ':'))
                
                # Publish to events queue
                pipeline.lpush(self.events_queue, event_json)
                
                # Also publish to Redis pub/sub for real-time notifications
                channel = f"scope:events:{event.get('camera_id', 'unknown')}"
                pipeline.publish(channel, event_json)
            
            # Set TTL on events queue
            pipeline.expire(self.events_queue, self.redis_config.get('events_ttl', 86400))  # 24 hours
            pipeline.execute()
            
            self.logger.info(f"Published {len(events)} validated events")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish validated events: {e}")
            return False
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        if not self.redis_client:
            return {}
        
        try:
            pipeline = self.redis_client.pipeline()
            
            # Get queue lengths
            pipeline.llen(self.stage1_queue)
            pipeline.llen(self.stage2_queue)
            pipeline.llen(self.events_queue)
            
            # Get memory info
            pipeline.info('memory')
            
            results = pipeline.execute()
            
            stage1_len = results[0] if results[0] is not None else 0
            stage2_len = results[1] if results[1] is not None else 0
            events_len = results[2] if results[2] is not None else 0
            memory_info = results[3] if len(results) > 3 else {}
            
            with self.lock:
                stats = {
                    'queues': {
                        'stage1_tokens': stage1_len,
                        'stage2_results': stage2_len,
                        'validated_events': events_len
                    },
                    'performance': {
                        'tokens_published': self.tokens_published,
                        'tokens_consumed': self.tokens_consumed,
                        'publish_errors': self.publish_errors,
                        'consume_errors': self.consume_errors
                    },
                    'memory': {
                        'used_memory_mb': memory_info.get('used_memory', 0) / 1024 / 1024,
                        'max_memory_mb': memory_info.get('maxmemory', 0) / 1024 / 1024
                    }
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get queue stats: {e}")
            return {}
    
    def clear_queues(self, queue_names: Optional[List[str]] = None):
        """
        Clear specified queues or all queues
        
        Args:
            queue_names: List of queue names to clear, or None for all
        """
        if not self.redis_client:
            return
        
        try:
            if queue_names is None:
                queue_names = [self.stage1_queue, self.stage2_queue, self.events_queue]
            
            pipeline = self.redis_client.pipeline()
            
            for queue_name in queue_names:
                pipeline.delete(queue_name)
            
            pipeline.execute()
            
            self.logger.info(f"Cleared queues: {queue_names}")
            
        except Exception as e:
            self.logger.error(f"Failed to clear queues: {e}")
    
    def setup_monitoring(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Setup monitoring callback for queue statistics
        
        Args:
            callback: Function to call with stats
        """
        def monitor_worker():
            while True:
                try:
                    stats = self.get_queue_stats()
                    callback(stats)
                    time.sleep(10)  # Update every 10 seconds
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    time.sleep(30)  # Wait longer on error
        
        monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
        monitor_thread.start()
        
        self.logger.info("Queue monitoring started")
    
    def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            if not self.redis_client:
                return False
            
            self.redis_client.ping()
            return True
            
        except Exception as e:
            self.logger.error(f"Redis health check failed: {e}")
            return False
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            if self.connection_pool:
                self.connection_pool.disconnect()
        except:
            pass

def main():
    """Test the Redis token manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Redis Token Manager Test')
    parser.add_argument('--test', action='store_true', help='Run test mode')
    parser.add_argument('--clear', action='store_true', help='Clear all queues')
    parser.add_argument('--stats', action='store_true', help='Show queue stats')
    
    args = parser.parse_args()
    
    # Default config for testing
    config = {
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'timeout': 5,
            'max_connections': 10
        }
    }
    
    try:
        manager = RedisTokenManager(config)
        
        if args.clear:
            manager.clear_queues()
            print("✅ All queues cleared")
            return
        
        if args.stats:
            stats = manager.get_queue_stats()
            print("📊 Queue Statistics:")
            print(json.dumps(stats, indent=2))
            return
        
        if args.test:
            print("🧪 Running Redis Token Manager test...")
            
            # Test health check
            if manager.health_check():
                print("✅ Redis connection healthy")
            else:
                print("❌ Redis connection failed")
                return
            
            # Test token publishing
            from cuda.cuda_stage1_detector import DetectionToken
            import time
            
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
            
            # Publish test token
            success = manager.publish_stage1_tokens([test_token], "test_cam")
            print(f"Token publish: {'✅ Success' if success else '❌ Failed'}")
            
            # Consume test token
            consumed = manager.consume_stage1_tokens(batch_size=1, timeout=1.0)
            print(f"Token consume: {'✅ Success' if consumed else '❌ Failed'}")
            
            # Show final stats
            final_stats = manager.get_queue_stats()
            print("\n📊 Final Statistics:")
            print(json.dumps(final_stats, indent=2))
    
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()