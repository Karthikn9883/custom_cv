# src/detection/detector.py

import cv2
import yaml
from ultralytics import YOLO
import argparse
import json
import time
import paho.mqtt.client as mqtt

# --- HELPER FUNCTION DEFINITIONS ---
# These need to be defined before they are called in main().

def load_config(config_path='/Users/karthiknutulapati/Desktop/smartbuilding/version2/custom_cv/yolo-world/mart_building_cv/configs/main_config.yaml'):
    """Loads the configuration from a YAML file."""
    # Note: If you run this script from the root directory, the path is correct.
    # If you run from src/detection/, the path might need to be '../../configs/main_config.yaml'
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at '{config_path}'. Make sure you are running the script from the project's root directory.")
        exit()


def setup_mqtt_client(config):
    """Sets up and connects the MQTT client."""
    if not config.get('mqtt', {}).get('enabled', False):
        print("MQTT is disabled in the config.")
        return None
        
    broker = config['mqtt']['broker_address']
    port = config['mqtt']['port']
    client = mqtt.Client()
    try:
        client.connect(broker, port, 60)
        client.loop_start() # Start a background thread for publishing
        print(f"Connected to MQTT Broker at {broker}:{port}")
        return client
    except Exception as e:
        print(f"Could not connect to MQTT Broker: {e}")
        return None


def is_image_file(filename):
    """Checks if a filename has a common image extension."""
    return str(filename).lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))


# --- MAIN EXECUTION LOGIC ---

def main():
    parser = argparse.ArgumentParser(description="YOLO-World Object Detection")
    parser.add_argument('--input', type=str, default=None, help='Path to input file or camera index. Overrides config.')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detection (e.g., 0.5 for 50%).')
    args = parser.parse_args()

    config = load_config()
    input_source = args.input if args.input is not None else config['input_source']
    
    print(f"Processing source: {input_source}")
    print(f"Using confidence threshold: {args.conf}")

    # Initialize MQTT client based on config
    mqtt_client = setup_mqtt_client(config)

    # Load model and set prompts
    model = YOLO(config['yolo_model'])
    model.set_classes(config['detection_prompts'])

    try:
        source_is_camera = str(input_source).isdigit()

        if source_is_camera:
            # --- LIVE FEED LOGIC ---
            cap = cv2.VideoCapture(int(input_source))
            if not cap.isOpened():
                print(f"Error: Could not open camera with index {input_source}")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to grab frame from camera.")
                    break

                results = model.predict(frame, conf=args.conf, verbose=False)
                annotated_frame = results[0].plot()
                
                cv2.imshow("YOLO-World Live Feed", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()

        else:
            # --- STATIC FILE (IMAGE OR VIDEO) LOGIC ---
            if is_image_file(input_source):
                # It's a single image file
                results = model.predict(source=input_source, conf=args.conf, verbose=False)
                annotated_frame = results[0].plot()
                cv2.imshow("YOLO-World Detection", annotated_frame)
                print("Detection complete on image. Press any key to exit.")
                cv2.waitKey(0)
            else:
                # It's a video file, process with streaming
                results = model.predict(source=input_source, conf=args.conf, stream=True, verbose=False)
                for result in results:
                    annotated_frame = result.plot()
                    cv2.imshow("YOLO-World Detection", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    finally:
        # Ensure all windows are closed on exit or error
        cv2.destroyAllWindows()
        if mqtt_client:
            mqtt_client.loop_stop()
        print("Detection finished and windows closed.")


if __name__ == "__main__":
    main()