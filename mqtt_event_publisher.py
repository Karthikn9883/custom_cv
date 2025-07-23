#!/usr/bin/env python3
"""
MQTT Event Publisher - SCOPE Smart Building System
Publishes validated detection events to MQTT broker for building management
Provides reliable event delivery with QoS and retry mechanisms
"""

import json
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import asdict
from collections import deque
import ssl

try:
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion
except ImportError:
    print("Error: paho-mqtt not installed. Run: pip install paho-mqtt")
    exit(1)

from cuda.cuda_stage2_verifier import ValidatedEvent

class MQTTEventPublisher:
    """
    High-reliability MQTT publisher for validated detection events
    Implements QoS, retry logic, and event batching for optimal performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mqtt_config = config.get('mqtt', {})
        
        # MQTT client
        self.client = None
        self.connected = False
        self.connection_lock = threading.Lock()
        
        # Event queuing and retry
        self.event_queue = deque()
        self.failed_events = deque()
        self.max_queue_size = self.mqtt_config.get('max_queue_size', 10000)
        self.retry_attempts = self.mqtt_config.get('retry_attempts', 3)
        
        # Performance tracking
        self.events_published = 0
        self.events_failed = 0
        self.events_retried = 0
        self.last_publish_time = 0
        
        # Threading
        self.publisher_thread = None
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Topics
        self.base_topic = self.mqtt_config.get('base_topic', 'scope/detection')
        self.events_topic = f"{self.base_topic}/events"
        self.status_topic = f"{self.base_topic}/status"
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize MQTT
        self._initialize_mqtt()
        
        self.logger.info("MQTT Event Publisher initialized")
    
    def _initialize_mqtt(self):
        """Initialize MQTT client with configuration"""
        try:
            # Create MQTT client
            client_id = self.mqtt_config.get('client_id', f"scope_publisher_{int(time.time())}")
            
            self.client = mqtt.Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                client_id=client_id,
                clean_session=self.mqtt_config.get('clean_session', True)
            )
            
            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_publish = self._on_publish
            self.client.on_log = self._on_log
            
            # Authentication
            username = self.mqtt_config.get('username')
            password = self.mqtt_config.get('password')
            if username and password:
                self.client.username_pw_set(username, password)
            
            # SSL/TLS configuration
            if self.mqtt_config.get('use_ssl', False):
                self._setup_ssl()
            
            # Connection settings
            self.client.reconnect_delay_set(
                min_delay=self.mqtt_config.get('reconnect_min_delay', 1),
                max_delay=self.mqtt_config.get('reconnect_max_delay', 120)
            )
            
            # Connect to broker
            self._connect_to_broker()
            
        except Exception as e:
            self.logger.error(f"MQTT initialization failed: {e}")
            raise
    
    def _setup_ssl(self):
        """Setup SSL/TLS configuration"""
        try:
            ssl_config = self.mqtt_config.get('ssl', {})
            
            # Create SSL context
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            
            # Configure SSL settings
            if ssl_config.get('cert_required', True):
                context.check_hostname = ssl_config.get('check_hostname', True)
                context.verify_mode = ssl.CERT_REQUIRED
            else:
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
            
            # Load certificates if provided
            ca_certs = ssl_config.get('ca_certs')
            certfile = ssl_config.get('certfile')
            keyfile = ssl_config.get('keyfile')
            
            if ca_certs:
                context.load_verify_locations(ca_certs)
            
            if certfile and keyfile:
                context.load_cert_chain(certfile, keyfile)
            
            self.client.tls_set_context(context)
            
            self.logger.info("SSL/TLS configured")
            
        except Exception as e:
            self.logger.error(f"SSL setup failed: {e}")
            raise
    
    def _connect_to_broker(self):
        """Connect to MQTT broker"""
        try:
            broker_host = self.mqtt_config.get('broker_host', 'localhost')
            broker_port = self.mqtt_config.get('broker_port', 1883)
            keepalive = self.mqtt_config.get('keepalive', 60)
            
            self.logger.info(f"Connecting to MQTT broker: {broker_host}:{broker_port}")
            
            # Connect asynchronously
            result = self.client.connect_async(broker_host, broker_port, keepalive)
            
            # Start network loop
            self.client.loop_start()
            
            # Wait for connection
            connection_timeout = self.mqtt_config.get('connection_timeout', 10)
            start_time = time.time()
            
            while not self.connected and (time.time() - start_time) < connection_timeout:
                time.sleep(0.1)
            
            if not self.connected:
                raise TimeoutError("MQTT connection timeout")
            
        except Exception as e:
            self.logger.error(f"MQTT connection failed: {e}")
            raise
    
    def _on_connect(self, client, userdata, flags, reason_code, properties):
        """MQTT connection callback"""
        if reason_code == 0:
            with self.connection_lock:
                self.connected = True
            
            self.logger.info("MQTT connected successfully")
            
            # Publish status message
            self._publish_status("connected")
            
        else:
            self.logger.error(f"MQTT connection failed with code: {reason_code}")
    
    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        """MQTT disconnection callback"""
        with self.connection_lock:
            self.connected = False
        
        if reason_code != 0:
            self.logger.warning(f"MQTT unexpected disconnection: {reason_code}")
        else:
            self.logger.info("MQTT disconnected")
    
    def _on_publish(self, client, userdata, mid, reason_codes, properties):
        """MQTT publish callback"""
        self.logger.debug(f"Message published: {mid}")
    
    def _on_log(self, client, userdata, level, buf):
        """MQTT logging callback"""
        if self.mqtt_config.get('debug_logging', False):
            self.logger.debug(f"MQTT: {buf}")
    
    def _publish_status(self, status: str):
        """Publish status message"""
        try:
            status_payload = {
                'status': status,
                'timestamp': time.time(),
                'publisher_id': self.client._client_id.decode(),
                'events_published': self.events_published,
                'events_failed': self.events_failed
            }
            
            self._publish_message(
                self.status_topic,
                json.dumps(status_payload),
                qos=1,
                retain=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to publish status: {e}")
    
    def _publish_message(self, topic: str, payload: str, qos: int = 1, retain: bool = False) -> bool:
        """
        Publish message to MQTT broker
        
        Args:
            topic: MQTT topic
            payload: Message payload
            qos: Quality of Service level
            retain: Retain flag
            
        Returns:
            Success status
        """
        if not self.connected:
            return False
        
        try:
            result = self.client.publish(topic, payload, qos=qos, retain=retain)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.last_publish_time = time.time()
                return True
            else:
                self.logger.error(f"MQTT publish failed with code: {result.rc}")
                return False
                
        except Exception as e:
            self.logger.error(f"MQTT publish error: {e}")
            return False
    
    def publish_validated_event(self, event: ValidatedEvent) -> bool:
        """
        Publish single validated event
        
        Args:
            event: Validated detection event
            
        Returns:
            Success status
        """
        try:
            # Create event payload
            event_payload = {
                'uuid': event.uuid,
                'timestamp': event.timestamp,
                'category': event.category,
                'confidence': event.confidence,
                'bbox': event.bbox,
                'camera_id': event.camera_id,
                'frame_id': event.frame_id,
                'verification': {
                    'stage1_confidence': event.verification_result.stage1_confidence,
                    'stage2_confidence': event.verification_result.stage2_confidence,
                    'fused_confidence': event.verification_result.fused_confidence,
                    'temporal_count': event.verification_result.temporal_count,
                    'verification_time_ms': event.verification_result.verification_time_ms
                },
                'metadata': {
                    'pipeline_version': '4.1',
                    'published_at': time.time(),
                    'publisher_id': self.client._client_id.decode() if self.client else 'unknown'
                }
            }
            
            # Create topic with camera and category
            event_topic = f"{self.events_topic}/{event.camera_id}/{event.category}"
            
            # Serialize payload
            payload = json.dumps(event_payload, separators=(',', ':'))
            
            # Publish with QoS 1 for guaranteed delivery
            success = self._publish_message(event_topic, payload, qos=1)
            
            if success:
                self.events_published += 1
                self.logger.info(f"Published event: {event.category} (confidence: {event.confidence:.2f})")
            else:
                self.events_failed += 1
                # Queue for retry
                self.failed_events.append((event_topic, payload, time.time(), 0))
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to publish event: {e}")
            self.events_failed += 1
            return False
    
    def publish_batch_events(self, events: List[ValidatedEvent]) -> int:
        """
        Publish batch of validated events
        
        Args:
            events: List of validated events
            
        Returns:
            Number of successfully published events
        """
        if not events:
            return 0
        
        published_count = 0
        
        # Group events by camera for efficient publishing
        camera_groups = {}
        for event in events:
            camera_id = event.camera_id
            if camera_id not in camera_groups:
                camera_groups[camera_id] = []
            camera_groups[camera_id].append(event)
        
        # Publish events for each camera
        for camera_id, camera_events in camera_groups.items():
            # Create batch payload for this camera
            batch_payload = {
                'camera_id': camera_id,
                'timestamp': time.time(),
                'event_count': len(camera_events),
                'events': []
            }
            
            # Add individual events
            for event in camera_events:
                event_data = {
                    'uuid': event.uuid,
                    'timestamp': event.timestamp,
                    'category': event.category,
                    'confidence': event.confidence,
                    'bbox': event.bbox,
                    'frame_id': event.frame_id,
                    'verification': asdict(event.verification_result)
                }
                batch_payload['events'].append(event_data)
            
            # Publish batch
            batch_topic = f"{self.events_topic}/{camera_id}/batch"
            batch_json = json.dumps(batch_payload, separators=(',', ':'))
            
            if self._publish_message(batch_topic, batch_json, qos=1):
                published_count += len(camera_events)
                
                # Also publish individual events for specific subscribers
                for event in camera_events:
                    self.publish_validated_event(event)
            else:
                # Queue entire batch for retry
                self.failed_events.append((batch_topic, batch_json, time.time(), 0))
        
        self.logger.info(f"Published batch: {published_count}/{len(events)} events")
        return published_count
    
    def start_publisher(self):
        """Start background publisher thread"""
        if self.running:
            return
        
        self.running = True
        self.shutdown_event.clear()
        
        self.publisher_thread = threading.Thread(target=self._publisher_worker, daemon=True)
        self.publisher_thread.start()
        
        self.logger.info("MQTT publisher thread started")
    
    def _publisher_worker(self):
        """Background worker for processing queued events and retries"""
        retry_interval = self.mqtt_config.get('retry_interval', 30)
        
        while not self.shutdown_event.is_set():
            try:
                # Process failed events for retry
                self._process_retry_queue()
                
                # Publish status update
                if self.connected and (time.time() - self.last_publish_time) > 60:
                    self._publish_status("active")
                
                # Wait before next iteration
                self.shutdown_event.wait(retry_interval)
                
            except Exception as e:
                self.logger.error(f"Publisher worker error: {e}")
                time.sleep(10)
    
    def _process_retry_queue(self):
        """Process failed events for retry"""
        current_time = time.time()
        retry_delay = self.mqtt_config.get('retry_delay', 30)
        
        # Process failed events
        failed_events_to_process = []
        
        while self.failed_events:
            topic, payload, failed_time, attempt_count = self.failed_events.popleft()
            
            # Check if enough time has passed for retry
            if current_time - failed_time >= retry_delay:
                failed_events_to_process.append((topic, payload, failed_time, attempt_count))
            else:
                # Put back in queue - not ready for retry
                self.failed_events.appendleft((topic, payload, failed_time, attempt_count))
                break
        
        # Retry failed events
        for topic, payload, failed_time, attempt_count in failed_events_to_process:
            if attempt_count >= self.retry_attempts:
                self.logger.error(f"Dropping event after {self.retry_attempts} failed attempts")
                continue
            
            if self._publish_message(topic, payload, qos=1):
                self.events_retried += 1
                self.logger.info(f"Retry successful for event (attempt {attempt_count + 1})")
            else:
                # Queue for another retry
                self.failed_events.append((topic, payload, failed_time, attempt_count + 1))
    
    def stop_publisher(self):
        """Stop publisher gracefully"""
        if not self.running:
            return
        
        self.logger.info("Stopping MQTT publisher...")
        
        # Signal shutdown
        self.shutdown_event.set()
        self.running = False
        
        # Wait for worker thread
        if self.publisher_thread and self.publisher_thread.is_alive():
            self.publisher_thread.join(timeout=10)
        
        # Publish final status
        if self.connected:
            self._publish_status("disconnecting")
        
        # Disconnect MQTT
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        
        self.logger.info("MQTT publisher stopped")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get publisher statistics"""
        return {
            'connection': {
                'connected': self.connected,
                'client_id': self.client._client_id.decode() if self.client else None
            },
            'events': {
                'published': self.events_published,
                'failed': self.events_failed,
                'retried': self.events_retried,
                'queued_for_retry': len(self.failed_events)
            },
            'performance': {
                'last_publish_time': self.last_publish_time,
                'publish_rate_per_min': self.events_published / max(1, (time.time() - (self.last_publish_time or time.time())) / 60)
            }
        }
    
    def health_check(self) -> bool:
        """Check publisher health"""
        return self.connected and (self.client is not None)
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.stop_publisher()
        except:
            pass

def main():
    """Test the MQTT event publisher"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MQTT Event Publisher Test')
    parser.add_argument('--test', action='store_true', help='Run test mode')
    parser.add_argument('--config', type=str, help='Config file path')
    
    args = parser.parse_args()
    
    # Default config for testing
    config = {
        'mqtt': {
            'broker_host': 'localhost',
            'broker_port': 1883,
            'client_id': 'test_publisher',
            'base_topic': 'scope/test',
            'username': None,
            'password': None,
            'keepalive': 60,
            'clean_session': True,
            'retry_attempts': 3,
            'retry_delay': 5
        }
    }
    
    try:
        publisher = MQTTEventPublisher(config)
        
        if args.test:
            print("🧪 Running MQTT Event Publisher test...")
            
            # Test health check
            if publisher.health_check():
                print("✅ MQTT connection healthy")
            else:
                print("❌ MQTT connection failed")
                return
            
            # Create test event
            from cuda.cuda_stage2_verifier import ValidatedEvent, VerificationResult
            
            test_verification = VerificationResult(
                token_uuid="test-uuid",
                stage1_confidence=0.8,
                stage2_confidence=0.9,
                fused_confidence=0.85,
                verified=True,
                temporal_count=3,
                roi_bbox=(100, 100, 200, 200),
                verification_time_ms=150.0,
                category="laptop"
            )
            
            test_event = ValidatedEvent(
                uuid="test-event-uuid",
                timestamp=time.time(),
                category="laptop",
                confidence=0.85,
                bbox=(100, 100, 200, 200),
                camera_id="test_cam",
                frame_id=1,
                verification_result=test_verification
            )
            
            # Test single event publish
            success = publisher.publish_validated_event(test_event)
            print(f"Single event publish: {'✅ Success' if success else '❌ Failed'}")
            
            # Test batch publish
            batch_success = publisher.publish_batch_events([test_event, test_event])
            print(f"Batch event publish: {'✅ Success' if batch_success == 2 else '❌ Failed'}")
            
            # Show statistics
            stats = publisher.get_statistics()
            print("\n📊 Publisher Statistics:")
            print(json.dumps(stats, indent=2))
            
            # Wait a bit for messages to be delivered
            time.sleep(2)
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        try:
            publisher.stop_publisher()
        except:
            pass

if __name__ == "__main__":
    main()