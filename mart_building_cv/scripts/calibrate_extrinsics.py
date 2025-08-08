import cv2
import numpy as np
import yaml
import os
import open3d as o3d
import trimesh

# --- File Paths ---
# Get the script directory and build relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
config_dir = os.path.join(project_root, "mart_building_cv", "configs")

CONFIG_FILE = os.path.join(config_dir, "main_config.yaml")
INTRINSICS_FILE = os.path.join(config_dir, "camera_intrinsics.yaml")
LIDAR_SCAN_FILE = os.path.join(project_root, "8_6_2025.glb")
OUTPUT_FILE = os.path.join(config_dir, "camera_extrinsics.yaml")

# --- Global Variables ---
points_2d = []
points_3d = []
camera_feed_frame = None

def load_config(file_path):
    """Loads a YAML configuration file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found at {file_path}")
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def mouse_callback_2d(event, x, y, flags, param):
    """Handles mouse clicks on the 2D camera feed window."""
    global points_2d, camera_feed_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        points_2d.append((x, y))
        print(f"2D point selected: ({x}, {y})")
        # Draw a circle on the frame to mark the selected point
        cv2.circle(camera_feed_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("2D Camera Feed - Click to Select Points", camera_feed_frame)

def pick_points_3d(pcd):
    """Handles point picking in the 3D visualizer."""
    print("\n--- 3D Point Selection ---")
    print("1. Hold [SHIFT] and click a point in the 3D window.")
    print("2. A pink sphere will mark your selection.")
    print("3. Press [SHIFT] + [Q] to close the window and save the point.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="3D LiDAR Scan - Pick a Point")
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    picked_indices = vis.get_picked_points()
    if not picked_indices:
        return None
    return pcd.points[picked_indices[0]]

def main():
    global points_2d, points_3d, camera_feed_frame

    # Load camera intrinsics
    print("Loading camera intrinsics...")
    intrinsics_data = load_config(INTRINSICS_FILE)
    camera_matrix = np.array(intrinsics_data['camera_matrix']['data'])
    dist_coeffs = np.array(intrinsics_data['distortion_coefficients']['data'])

    # Load LiDAR scan
    print(f"Loading LiDAR scan from {LIDAR_SCAN_FILE}...")
    mesh = trimesh.load(LIDAR_SCAN_FILE, force='mesh')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
    pcd.paint_uniform_color([0.5, 0.5, 0.5]) # Gray color for the point cloud

    # Load camera source
    main_config = load_config(CONFIG_FILE)
    camera_source = main_config.get('input_source')
    if isinstance(camera_source, str) and camera_source.isdigit():
        camera_source = int(camera_source)

    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"Error: Could not open camera source {camera_source}")
        return

    cv2.namedWindow("2D Camera Feed - Click to Select Points")
    cv2.setMouseCallback("2D Camera Feed - Click to Select Points", mouse_callback_2d)

    print("\n--- Starting Extrinsic Calibration ---")
    print("We need to select at least 4 corresponding points.")
    print("For each pair:")
    print("  1. First, select the point in the 3D window.")
    print("  2. Then, select the *exact same point* in the 2D camera feed.")
    print("Press 's' to start selecting a new point pair.")
    print("Press 'd' when you have enough points and are done.")
    print("Press 'q' to quit at any time.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break
        
        # Keep the latest frame available for the mouse callback
        camera_feed_frame = frame.copy()

        # Display the 2D feed
        cv2.imshow("2D Camera Feed - Click to Select Points", camera_feed_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('s'):
            # Step 1: Pick a point in the 3D cloud
            selected_3d_point = pick_points_3d(pcd)
            if selected_3d_point is None:
                print("No 3D point was selected. Please try again.")
                continue
            
            points_3d.append(selected_3d_point)
            print(f"3D point selected: {selected_3d_point}")
            print("Now, please click the corresponding point in the 2D camera feed.")

        if key == ord('d'):
            if len(points_2d) < 4 or len(points_3d) < 4:
                print("Error: You need at least 4 point pairs to calculate extrinsics.")
                continue
            if len(points_2d) != len(points_3d):
                print("Error: The number of 2D and 3D points must be equal.")
                continue
            
            print("\nCalculating camera extrinsics...")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(points_2d) >= 4 and len(points_2d) == len(points_3d):
        # Convert lists to numpy arrays
        np_points_3d = np.array(points_3d, dtype=np.float32)
        np_points_2d = np.array(points_2d, dtype=np.float32)

        # Use solvePnP to find the camera's pose
        success, rvec, tvec = cv2.solvePnP(np_points_3d, np_points_2d, camera_matrix, dist_coeffs)

        if success:
            print("Extrinsic calibration successful!")
            # Save the results
            extrinsics_data = {
                'rotation_vector': {
                    'rows': rvec.shape[0],
                    'cols': rvec.shape[1],
                    'data': rvec.tolist()
                },
                'translation_vector': {
                    'rows': tvec.shape[0],
                    'cols': tvec.shape[1],
                    'data': tvec.tolist()
                }
            }
            with open(OUTPUT_FILE, 'w') as f:
                yaml.dump(extrinsics_data, f, default_flow_style=False)
            print(f"Successfully saved extrinsic data to: {OUTPUT_FILE}")
        else:
            print("Error: solvePnP failed to find a solution.")
    else:
        print("Calibration cancelled or not enough points were selected.")

if __name__ == "__main__":
    main()
