# mart_building_cv/scripts/calibrate_camera.py

import cv2
import numpy as np
import yaml
import os
import sys
import time

# Add the project root to the Python path to allow importing project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from mart_building_cv.src.detection.detector import load_config

# --- Global Variables ---
image_points = []
world_points = []
FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- Function Definitions ---

def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks to select points in the image."""
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(image_points) < 4:
            image_points.append((x, y))
            print(f"Selected image point #{len(image_points)}: ({x}, {y})")
        else:
            print("You have already selected 4 points. Press 's' to save or 'q' to quit.")

def get_a4_world_coordinates():
    """Returns the standardized real-world coordinates for an A4 paper."""
    # Dimensions of A4 paper in inches: 8.27" x 11.69"
    # We'll use a simplified 8" x 11" for this calibration
    a4_dims = {"width": 8, "height": 11}
    
    # Define the four corners in a specific order:
    # 1. Top-Left (0, 0)
    # 2. Top-Right (width, 0)
    # 3. Bottom-Right (width, height)
    # 4. Bottom-Left (0, height)
    world_points.extend([
        [0, 0],
        [a4_dims["width"], 0],
        [a4_dims["width"], a4_dims["height"]],
        [0, a4_dims["height"]]
    ])
    
    print("\nUsing standardized A4 paper dimensions (8\" x 11\") for world coordinates:")
    for i, p in enumerate(world_points):
        print(f"  Point #{i+1}: {p}")
    return np.array(world_points, dtype=np.float32)

def save_homography_matrix(matrix, path):
    """Saves the calculated homography matrix to a YAML file."""
    try:
        with open(path, 'w') as file:
            yaml.dump({'homography_matrix': matrix.tolist()}, file)
        print(f"\nSuccessfully saved homography matrix to: {path}")
    except Exception as e:
        print(f"Error saving matrix: {e}")

# --- Main Execution ---

def main():
    # For calibration, we typically want a direct camera feed, not the RTSP stream.
    # We will default to camera index 0.
    # For calibration, we typically want a direct camera feed.
    # On macOS, explicitly using the AVFOUNDATION backend can be more reliable.
    calibration_camera_index = 0
    cap = cv2.VideoCapture(calibration_camera_index, cv2.CAP_AVFOUNDATION)

    # Add a short delay to allow the camera to initialize, especially for Continuity Camera
    time.sleep(2.0)

    # If the default camera fails, then we can try the source from the config file.
    if not cap.isOpened():
        print(f"INFO: Could not open default camera index {calibration_camera_index}. Trying config file source...")
        config = load_config()
        input_source = config.get('input_source', 1) # Default to 1 if not in config
        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            print(f"Error: Could not open camera with index {input_source} from config.")
            return

    window_name = "Camera Calibration - Click 4 points, then press 's'"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("--- Camera Calibration with A4 Paper ---")
    print("1. Place a standard A4 sheet of paper on the floor.")
    print("2. Click on the 4 corners of the A4 paper in the following order:")
    print("   - Top-Left")
    print("   - Top-Right")
    print("   - Bottom-Right")
    print("   - Bottom-Left")
    print("3. After selecting 4 points, press the 's' key to calculate and save.")
    print("4. Press 'q' to quit at any time.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Draw selected points on the frame
        for i, point in enumerate(image_points):
            cv2.circle(frame, point, 5, (0, 255, 0), -1)
            cv2.putText(frame, str(i+1), (point[0]+10, point[1]-10), FONT, 0.7, (0, 255, 0), 2)

        # Display instructions
        cv2.putText(frame, "Click 4 points on the floor. Press 's' to save, 'q' to quit.", (10, 30), FONT, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{len(image_points)}/4 points selected", (10, 60), FONT, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        if key == ord('s'):
            if len(image_points) == 4:
                # Get the standardized world coordinates for A4 paper
                world_coords = get_a4_world_coordinates()
                
                # Ensure image_points is a numpy array
                image_coords = np.array(image_points, dtype=np.float32)

                # Calculate the homography matrix
                h_matrix, _ = cv2.findHomography(image_coords, world_coords)
                print("\nCalculated Homography Matrix:")
                print(h_matrix)
                
                # Define where to save the matrix
                script_dir = os.path.dirname(__file__)
                # Go up one level to mart_building_cv, then into configs
                config_dir = os.path.abspath(os.path.join(script_dir, '..', 'configs'))
                save_path = os.path.join(config_dir, 'homography_matrix.yaml')
                
                save_homography_matrix(h_matrix, save_path)
                break # Exit after saving
            else:
                print("\nPlease select exactly 4 points before saving.")

    cap.release()
    cv2.destroyAllWindows()
    print("Calibration script finished.")

if __name__ == "__main__":
    main()
