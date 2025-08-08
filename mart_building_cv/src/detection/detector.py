# src/detection/detector.py

import cv2
import yaml
import numpy as np
from ultralytics import YOLO
import argparse
import json
import time
import paho.mqtt.client as mqtt
import os
import sys

# Add the parent src directory to the path to import projection_utils
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from projection_utils import ProjectionSystem

# --- HELPER FUNCTION DEFINITIONS ---

def load_config(config_path=None):
    """
    Loads the configuration from a YAML file.
    It robustly finds the config relative to this script's location.
    """
    if config_path is None:
        script_dir = os.path.dirname(__file__)
        config_path = os.path.join(script_dir, '..', '..', 'configs', 'main_config.yaml')
        
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at the expected path '{config_path}'.")
        exit()
    except Exception as e:
        print(f"FATAL: Error reading the configuration file: {e}")
        exit()

def load_homography_matrix(config):
    """Loads the homography matrix from the path specified in the config."""
    matrix_path = config.get('homography_matrix_path')
    if not matrix_path:
        print("WARNING: 'homography_matrix_path' not found in config. Cannot perform coordinate transformation.")
        return None

    # Make path absolute relative to the main config file path
    script_dir = os.path.dirname(__file__)
    # This assumes the homography file path is relative to the project root or is absolute
    # Let's construct the path relative to the config file.
    config_dir = os.path.join(script_dir, '..', '..', 'configs')
    absolute_path = os.path.join(config_dir, os.path.basename(matrix_path))

    try:
        with open(absolute_path, 'r') as file:
            data = yaml.safe_load(file)
            matrix = np.array(data['homography_matrix'])
            print("INFO: Homography matrix loaded successfully.")
            return matrix
    except FileNotFoundError:
        print(f"WARNING: Homography matrix file not found at '{absolute_path}'.")
        return None
    except Exception as e:
        print(f"WARNING: Error loading homography matrix: {e}")
        return None

def setup_mqtt_client(config):
    """Sets up and connects the MQTT client if enabled in the config."""
    if not config.get('mqtt', {}).get('enabled', False):
        print("INFO: MQTT is disabled in the configuration.")
        return None
        
    broker = config['mqtt']['broker_address']
    port = config['mqtt']['port']
    # Use the latest callback API version to avoid deprecation warnings
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    try:
        client.connect(broker, port, 60)
        client.loop_start()
        print(f"INFO: Successfully connected to MQTT Broker at {broker}:{port}")
        return client
    except Exception as e:
        print(f"WARNING: Could not connect to MQTT Broker: {e}")
        return None

def publish_detection(mqtt_client, topic, detection_data):
    """Publishes detection data as a JSON payload to the MQTT broker."""
    if not mqtt_client:
        return
    try:
        payload = json.dumps(detection_data)
        mqtt_client.publish(topic, payload)
        print(f"Published to MQTT topic '{topic}': {payload}")
    except Exception as e:
        print(f"WARNING: Failed to publish to MQTT: {e}")

def is_image_file(filename):
    """Checks if a filename has a common image extension."""
    return str(filename).lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))

def transform_pixel_to_world(pixel_coords, homography_matrix=None, projection_system=None):
    """Transforms a single (x, y) pixel coordinate to world coordinates using either homography or 3D projection."""
    # Use new 3D projection system if available
    if projection_system is not None and projection_system.is_calibrated():
        world_point = projection_system.project_2d_to_3d(pixel_coords)
        if world_point is not None:
            return {
                "x": float(world_point[0]),
                "y": float(world_point[1]),
                "z": float(world_point[2])
            }
        return None
    
    # Fall back to homography method
    if homography_matrix is None:
        return None

    pixel_point = np.array([[pixel_coords]], dtype=np.float32)
    world_point = cv2.perspectiveTransform(pixel_point, homography_matrix)
    
    # The result is inside a nested array, so we extract it.
    # Also, we cast from float32 to a standard Python float for JSON serialization.
    return {
        "x": float(world_point[0][0][0]),
        "y": float(world_point[0][0][1])
    }

def process_detections(results, mqtt_client, config, last_detection_times, detection_topic, homography_matrix=None, projection_system=None):
    """
    Processes detection results, handles cooldown logic, and publishes to MQTT.
    Returns the annotated frame for display.
    """
    frame = results.plot()
    
    if not mqtt_client or not detection_topic:
        # Still return the annotated frame even if MQTT is off
        if homography_matrix is not None or projection_system is not None:
            # If we have coordinate transformation capability, annotate with world coordinates
            for box in results.boxes:
                x_center, y_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2), int(box.xyxy[0][3]) # Bottom center
                world_coords = transform_pixel_to_world((x_center, y_center), homography_matrix, projection_system)
                if world_coords:
                    if 'z' in world_coords:  # 3D coordinates
                        label = f"({world_coords['x']:.2f}, {world_coords['y']:.2f}, {world_coords['z']:.2f})m"
                    else:  # 2D coordinates
                        label = f"({world_coords['x']:.2f}, {world_coords['y']:.2f})in"
                    cv2.putText(frame, label, (x_center, y_center + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

    current_time = time.time()
    cooldown = config['mqtt'].get('message_interval_seconds', 10)
    
    boxes = results.boxes
    for i in range(len(boxes)):
        confidence = boxes.conf[i].item()
        class_id = int(boxes.cls[i].item())
        detected_class = results.names[class_id]

        last_time = last_detection_times.get(detected_class, 0)
        if current_time - last_time > cooldown:
            # Basic detection data
            detection_data = {
                'timestamp': int(current_time),
                'building': config.get('device_context', {}).get('building', 'unknown'),
                'zone': config.get('device_context', {}).get('zone', 'unknown'),
                'detection_class': detected_class,
                'confidence': round(confidence, 4),
                'source': config.get('input_source', 'unknown')
            }

            # Add world coordinates if coordinate transformation is available
            if homography_matrix is not None or projection_system is not None:
                # Use the bottom-center of the bounding box as the object's location
                x_center = (boxes.xyxy[i][0].item() + boxes.xyxy[i][2].item()) / 2
                y_center = boxes.xyxy[i][3].item() # Bottom edge
                world_coords = transform_pixel_to_world((x_center, y_center), homography_matrix, projection_system)
                if world_coords:
                    detection_data['world_coordinates'] = world_coords
                    # Also annotate the frame with these coordinates
                    if 'z' in world_coords:  # 3D coordinates
                        label = f"({world_coords['x']:.2f}, {world_coords['y']:.2f}, {world_coords['z']:.2f})m"
                    else:  # 2D coordinates
                        label = f"({world_coords['x']:.2f}, {world_coords['y']:.2f})in"
                    cv2.putText(frame, label, (int(x_center), int(y_center) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Publish to MQTT and update last detection time
            publish_detection(mqtt_client, detection_topic, detection_data)
            last_detection_times[detected_class] = current_time
            
    return frame

def process_video_stream(model, source, conf, mqtt_client, config, detection_topic, homography_matrix=None, projection_system=None):
    """Handles processing of a live camera feed or a video file."""
    last_detection_times = {}
    results_stream = model.predict(source=source, conf=conf, stream=True, verbose=False)
    
    for results in results_stream:
        # Pass both homography matrix and projection system to the processing function
        annotated_frame = process_detections(results, mqtt_client, config, last_detection_times, detection_topic, homography_matrix, projection_system)
        cv2.imshow("SCOPE Detector", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def process_single_image(model, source, conf, mqtt_client, config, detection_topic, homography_matrix=None, projection_system=None):
    """Handles processing of a single static image."""
    last_detection_times = {}
    results = model.predict(source=source, conf=conf, verbose=False)
    
    result = results[0]
    
    # Pass both homography matrix and projection system to the processing function
    annotated_frame = process_detections(result, mqtt_client, config, last_detection_times, detection_topic, homography_matrix, projection_system)
    cv2.imshow("SCOPE Detector", annotated_frame)

    print("INFO: Detection complete on image. Press any key to exit.")
    cv2.waitKey(0)

# --- MAIN EXECUTION LOGIC ---

def main():
    parser = argparse.ArgumentParser(description="SCOPE: Smart Building Computer Vision Detector")
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detection (e.g., 0.5).')
    args = parser.parse_args()

    config = load_config()
    input_source = "rtsp://192.168.1.228:554/avstream/channel=1/stream=0.sdp"
    config['input_source'] = input_source

    print(f"INFO: Processing source: {input_source}")
    print(f"INFO: Using confidence threshold: {args.conf}")

    # --- Coordinate Transformation System Loading ---
    # Try to load the new 3D projection system first
    projection_system = None
    try:
        projection_system = ProjectionSystem()
        if projection_system.is_calibrated():
            print("INFO: 3D projection system loaded successfully")
        else:
            print("WARNING: 3D projection system not fully calibrated, falling back to homography")
            projection_system = None
    except Exception as e:
        print(f"WARNING: Could not load 3D projection system: {e}")
        projection_system = None
    
    # Load homography matrix as fallback
    homography_matrix = load_homography_matrix(config)

    # --- MQTT Setup ---
    mqtt_client = setup_mqtt_client(config)
    detection_topic = None
    if mqtt_client:
        context = config.get('device_context', {})
        building = context.get('building', 'unknown_building')
        zone = context.get('zone', 'unknown_zone')
        detection_topic = f"scope/detection/{building}/{zone}/event"
        print(f"INFO: Publishing to MQTT Topic: {detection_topic}")

    # --- Model Loading ---
    print("INFO: Loading YOLO-World model...")
    model = YOLO(config['yolo_model'])
    model.set_classes(config['detection_prompts'])
    print("INFO: Model loaded and classes set.")

    try:
        if is_image_file(input_source):
            process_single_image(model, input_source, args.conf, mqtt_client, config, detection_topic, homography_matrix, projection_system)
        else:
            source_for_stream = int(input_source) if str(input_source).isdigit() else input_source
            process_video_stream(model, source_for_stream, args.conf, mqtt_client, config, detection_topic, homography_matrix, projection_system)

    except Exception as e:
        print(f"FATAL: An unexpected error occurred during detection: {e}")
    finally:
        cv2.destroyAllWindows()
        if mqtt_client:
            mqtt_client.loop_stop()
            print("INFO: MQTT client disconnected.")
        print("INFO: Detection finished and resources released.")

if __name__ == "__main__":
    main()
