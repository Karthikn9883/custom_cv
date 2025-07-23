#!/usr/bin/env python3
"""
Redis Token Monitor for SCOPE Smart Building
Real-time monitoring of Stage 1 detection tokens
Visualizes token flow and provides analytics
"""

import redis
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TokenMonitor:
    """
    Monitor and analyze Stage 1 detection tokens from Redis queue
    Provides real-time statistics and visualization
    """
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, 
                 queue_name: str = "stage1_tokens", db: int = 0):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.queue_name = queue_name
        self.db = db
        
        # Connect to Redis
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=db,
            decode_responses=True
        )
        
        # Token storage and analytics
        self.tokens_received = []
        self.category_counts = defaultdict(int)
        self.camera_counts = defaultdict(int)
        self.recent_tokens = deque(maxlen=100)  # Keep last 100 tokens
        
        # Performance tracking
        self.start_time = time.time()
        self.tokens_per_second = deque(maxlen=60)  # Track TPS over 60 seconds
        self.last_token_time = time.time()
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread = None
        
        # Test Redis connection
        try:
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis at {redis_host}:{redis_port}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    def start_monitoring(self):
        """Start token monitoring in background thread"""
        if self.monitoring:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"🔍 Started monitoring queue: {self.queue_name}")
    
    def stop_monitoring(self):
        """Stop token monitoring"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        logger.info("⏹️ Stopped monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        logger.info("🔄 Token monitoring loop started")
        
        while self.monitoring:
            try:
                # Block for up to 1 second waiting for tokens
                result = self.redis_client.brpop(self.queue_name, timeout=1)
                
                if result:
                    queue_name, token_json = result
                    self._process_token(token_json)
                
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                time.sleep(1)  # Brief pause before retrying
    
    def _process_token(self, token_json: str):
        """Process a received token"""
        try:
            token_data = json.loads(token_json)
            
            # Validate token structure
            required_fields = ['uuid', 'timestamp', 'category', 'confidence', 'bbox', 'camera_id']
            if not all(field in token_data for field in required_fields):
                logger.warning(f"⚠️ Invalid token structure: {token_data}")
                return
            
            # Store token
            self.tokens_received.append(token_data)
            self.recent_tokens.append(token_data)
            
            # Update analytics
            self.category_counts[token_data['category']] += 1
            self.camera_counts[token_data['camera_id']] += 1
            
            # Update performance metrics
            current_time = time.time()
            time_since_last = current_time - self.last_token_time
            if time_since_last > 0:
                tps = 1.0 / time_since_last
                self.tokens_per_second.append(tps)
            self.last_token_time = current_time
            
            # Log token receipt
            logger.info(f"📦 Token received: {token_data['category']} "
                       f"(confidence: {token_data['confidence']:.2f}, "
                       f"camera: {token_data['camera_id']})")
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse token JSON: {e}")
        except Exception as e:
            logger.error(f"❌ Error processing token: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current monitoring statistics"""
        current_time = time.time()
        runtime = current_time - self.start_time
        
        # Calculate tokens per second
        avg_tps = len(self.tokens_per_second) / 60 if self.tokens_per_second else 0
        current_tps = self.tokens_per_second[-1] if self.tokens_per_second else 0
        
        # Get top categories and cameras
        top_categories = dict(sorted(self.category_counts.items(), 
                                   key=lambda x: x[1], reverse=True)[:10])
        top_cameras = dict(sorted(self.camera_counts.items(), 
                                key=lambda x: x[1], reverse=True)[:5])
        
        # Recent activity
        recent_activity = []
        for token in list(self.recent_tokens)[-10:]:  # Last 10 tokens
            recent_activity.append({
                'category': token['category'],
                'confidence': token['confidence'],
                'camera_id': token['camera_id'],
                'timestamp': token['timestamp']
            })
        
        return {
            'runtime_seconds': runtime,
            'total_tokens': len(self.tokens_received),
            'tokens_per_second': {
                'current': current_tps,
                'average': avg_tps
            },
            'categories': {
                'total_unique': len(self.category_counts),
                'top_categories': top_categories
            },
            'cameras': {
                'total_unique': len(self.camera_counts),
                'top_cameras': top_cameras
            },
            'recent_activity': recent_activity,
            'queue_status': {
                'queue_name': self.queue_name,
                'current_queue_length': self.redis_client.llen(self.queue_name)
            }
        }
    
    def print_statistics(self):
        """Print formatted statistics to console"""
        stats = self.get_statistics()
        
        print("\\n" + "="*60)
        print("📊 REDIS TOKEN MONITOR STATISTICS")
        print("="*60)
        
        print(f"⏱️  Runtime: {stats['runtime_seconds']:.1f} seconds")
        print(f"📦 Total tokens: {stats['total_tokens']}")
        print(f"⚡ Tokens/sec: {stats['tokens_per_second']['current']:.2f} (avg: {stats['tokens_per_second']['average']:.2f})")
        print(f"🏷️  Unique categories: {stats['categories']['total_unique']}")
        print(f"📹 Unique cameras: {stats['cameras']['total_unique']}")
        print(f"📋 Queue length: {stats['queue_status']['current_queue_length']}")
        
        if stats['categories']['top_categories']:
            print("\\n🔝 Top Categories:")
            for category, count in stats['categories']['top_categories'].items():
                print(f"   - {category}: {count}")
        
        if stats['cameras']['top_cameras']:
            print("\\n📹 Camera Activity:")
            for camera, count in stats['cameras']['top_cameras'].items():
                print(f"   - {camera}: {count}")
        
        if stats['recent_activity']:
            print("\\n🕐 Recent Activity (last 10 tokens):")
            for activity in stats['recent_activity'][-5:]:  # Show last 5
                timestamp = datetime.fromtimestamp(activity['timestamp']).strftime('%H:%M:%S')
                print(f"   [{timestamp}] {activity['category']} "
                     f"(conf: {activity['confidence']:.2f}, cam: {activity['camera_id']})")
    
    def clear_statistics(self):
        """Clear all statistics"""
        self.tokens_received.clear()
        self.category_counts.clear()
        self.camera_counts.clear()
        self.recent_tokens.clear()
        self.tokens_per_second.clear()
        self.start_time = time.time()
        logger.info("🧹 Statistics cleared")
    
    def export_tokens(self, filename: str = None) -> str:
        """Export all received tokens to JSON file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"stage1_tokens_export_{timestamp}.json"
        
        export_data = {
            'export_timestamp': time.time(),
            'total_tokens': len(self.tokens_received),
            'statistics': self.get_statistics(),
            'tokens': self.tokens_received
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            logger.info(f"📄 Tokens exported to: {filename}")
            return filename
        except Exception as e:
            logger.error(f"❌ Failed to export tokens: {e}")
            raise

def test_redis_connection():
    """Test Redis connection and basic operations"""
    print("🔍 Testing Redis connection...")
    
    try:
        # Connect to Redis
        client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        client.ping()
        print("✅ Redis connection successful")
        
        # Test basic operations
        test_key = "stage1_test"
        client.set(test_key, "test_value", ex=10)  # Expire in 10 seconds
        
        value = client.get(test_key)
        if value == "test_value":
            print("✅ Redis read/write operations working")
        else:
            print("❌ Redis read/write test failed")
            return False
        
        # Test list operations (queue simulation)
        queue_test = "test_queue"
        client.lpush(queue_test, "test_token_1", "test_token_2")
        
        result = client.rpop(queue_test)
        if result == "test_token_1":
            print("✅ Redis queue operations working")
        else:
            print("❌ Redis queue operations failed")
            return False
        
        # Cleanup
        client.delete(queue_test)
        
        return True
        
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        return False

def simulate_tokens(monitor: TokenMonitor, num_tokens: int = 10):
    """Simulate tokens for testing"""
    import uuid
    import random
    
    print(f"🎯 Simulating {num_tokens} tokens...")
    
    categories = ['backpack', 'laptop', 'phone', 'trash bin', 'fire extinguisher', 'water bottle']
    cameras = ['mac_cam_01', 'mac_cam_02', 'test_cam_01']
    
    for i in range(num_tokens):
        # Create simulated token
        token = {
            'uuid': str(uuid.uuid4()),
            'timestamp': time.time(),
            'category': random.choice(categories),
            'confidence': random.uniform(0.5, 0.95),
            'bbox': [
                random.randint(0, 200),
                random.randint(0, 200),
                random.randint(200, 400),
                random.randint(200, 400)
            ],
            'camera_id': random.choice(cameras),
            'frame_id': i
        }
        
        # Push to Redis queue
        token_json = json.dumps(token)
        monitor.redis_client.lpush(monitor.queue_name, token_json)
        
        print(f"   Sent: {token['category']} from {token['camera_id']}")
        time.sleep(0.5)  # Simulate realistic timing

def main():
    """Main token monitoring application"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Redis Token Monitor for Stage 1 Detector')
    parser.add_argument('--host', type=str, default='localhost', help='Redis host')
    parser.add_argument('--port', type=int, default=6379, help='Redis port')
    parser.add_argument('--queue', type=str, default='stage1_tokens', help='Queue name')
    parser.add_argument('--test-redis', action='store_true', help='Test Redis connection only')
    parser.add_argument('--simulate', type=int, help='Simulate N tokens for testing')
    parser.add_argument('--export', type=str, help='Export tokens to file')
    
    args = parser.parse_args()
    
    if args.test_redis:
        success = test_redis_connection()
        return success
    
    print("🚀 Starting Redis Token Monitor")
    print("="*50)
    
    try:
        # Initialize monitor
        monitor = TokenMonitor(args.host, args.port, args.queue)
        
        # Simulate tokens if requested
        if args.simulate:
            simulate_tokens(monitor, args.simulate)
            print("\\n⏳ Waiting for simulated tokens to be processed...")
            time.sleep(2)
        
        # Start monitoring
        monitor.start_monitoring()
        
        print("\\n📋 Commands:")
        print("  's' - Show statistics")
        print("  'c' - Clear statistics")
        print("  'e' - Export tokens")
        print("  'q' - Quit")
        print("\\nMonitoring tokens... (press Enter for command prompt)")
        
        try:
            while True:
                command = input().strip().lower()
                
                if command == 'q':
                    break
                elif command == 's':
                    monitor.print_statistics()
                elif command == 'c':
                    monitor.clear_statistics()
                    print("✅ Statistics cleared")
                elif command == 'e':
                    filename = args.export
                    exported_file = monitor.export_tokens(filename)
                    print(f"✅ Tokens exported to: {exported_file}")
                else:
                    print("Unknown command. Use 's', 'c', 'e', or 'q'")
        
        except KeyboardInterrupt:
            print("\\n⏹️ Stopping monitor...")
        
        finally:
            monitor.stop_monitoring()
            
            # Final statistics
            print("\\n📊 Final Statistics:")
            monitor.print_statistics()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Monitor failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)