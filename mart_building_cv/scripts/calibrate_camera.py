import cv2
import numpy as np
import yaml
import os

# --- Configuration ---
CHESSBOARD_SIZE = (9, 6)  # Number of inner corners (width, height)
SQUARE_SIZE_MM = 30       # Real-world size of a chessboard square in millimeters
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CALIBRATION_FRAME_COUNT = 20 # Number of frames to capture for calibration

# --- File Paths ---
# Path to the main config file to get the camera source
CONFIG_FILE = "/Users/Arshad_1/Desktop/projects/custom_cv_new/mart_building_cv/configs/main_config.yaml"
# Path to save the output calibration data
OUTPUT_FILE = "/Users/Arshad_1/Desktop/projects/custom_cv_new/mart_building_cv/configs/camera_intrinsics.yaml"

def load_camera_source_from_config(config_path):
    """Loads the camera input source from the main YAML config file."""
    if not os.path.exists(config_path):
        print(f"Error: Main config file not found at {config_path}")
        return None
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        source = config.get('input_source')
        if source is None:
            print(f"Error: 'input_source' not found in {config_path}")
            return None
        # Convert to integer if it's a simple number for local webcams
        if isinstance(source, str) and source.isdigit():
            return int(source)
        return source
    except Exception as e:
        print(f"Error reading or parsing config file: {e}")
        return None

def calibrate_camera():
    """
    Performs intrinsic camera calibration using a chessboard pattern.
    Saves the camera matrix and distortion coefficients to a YAML file.
    """
    # Load camera source from the main config file
    camera_source = load_camera_source_from_config(CONFIG_FILE)
    if camera_source is None:
        return

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_MM

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # Initialize camera
    print(f"Attempting to open camera source: {camera_source}")
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("\nStarting camera calibration...")
    print(f"Show a {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} chessboard to the camera.")
    print(f"We need to capture {CALIBRATION_FRAME_COUNT} good frames.")
    print("Press 'c' to capture a frame. Press 'q' to quit.")

    captured_frames = 0
    while captured_frames < CALIBRATION_FRAME_COUNT:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        # If found, add object points, image points (after refining them)
        if ret:
            cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, ret)
            display_text = f"Frames captured: {captured_frames}/{CALIBRATION_FRAME_COUNT}"
        else:
            display_text = "Chessboard not detected."

        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Calibration', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Calibration cancelled by user.")
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == ord('c') and ret:
            print(f"Frame {captured_frames + 1} captured!")
            objpoints.append(objp)
            imgpoints.append(corners)
            captured_frames += 1

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 5:
        print("Calibration failed: Not enough valid frames captured.")
        return

    print("\nAll frames captured. Calculating camera matrix...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if not ret:
        print("Calibration failed. Could not compute camera matrix.")
        return

    # Save the calibration result
    calibration_data = {
        'camera_matrix': {
            'rows': mtx.shape[0],
            'cols': mtx.shape[1],
            'data': mtx.tolist()
        },
        'distortion_coefficients': {
            'rows': dist.shape[0],
            'cols': dist.shape[1],
            'data': dist.tolist()
        }
    }

    try:
        with open(OUTPUT_FILE, 'w') as f:
            yaml.dump(calibration_data, f, default_flow_style=False)
        print(f"\nSuccessfully saved calibration data to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving YAML file: {e}")

if __name__ == "__main__":
    # You will need a physical 9x6 chessboard pattern.
    # You can generate and print one from many websites online.
    # Example: https://markhedleyjones.com/projects/calibration-checkerboard-collection
    calibrate_camera()