import cv2
import numpy as np
import yaml
import os
import open3d as o3d
import trimesh

# --- File Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
config_dir = os.path.join(project_root, "mart_building_cv", "configs")

CONFIG_FILE = os.path.join(config_dir, "main_config.yaml")
INTRINSICS_FILE = os.path.join(config_dir, "camera_intrinsics.yaml")
LIDAR_SCAN_FILE = os.path.join(project_root, "8_11_2025.glb")
OUTPUT_FILE = os.path.join(config_dir, "camera_extrinsics.yaml")

def load_config(file_path):
    """Loads a YAML configuration file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found at {file_path}")
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def pick_all_3d_points(pcd):
    """Opens a 3D visualizer with standard, reliable controls to let the user pick multiple points."""
    
    print(f"\n{'='*60}")
    print("STEP 1: SELECT ALL 3D POINTS (RELIABLE CONTROLS)")
    print(f"{'='*60}")
    print("--- 3D NAVIGATION CONTROLS ---")
    print("  ROTATE:     Left-click + Drag")
    print("  ZOOM:       Right-click + Drag or Scroll Wheel")
    print("  PAN:        Middle-click + Drag")
    print("--- POINT SELECTION CONTROLS ---")
    print("  PICK:       [SHIFT] + Left-click")
    print("  UNDO:       [SHIFT] + Right-click")
    print("  DONE:       [ESC] or [Q] (close the window)")
    print("--------------------------------------------------")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="3D Point Picker", width=1280, height=720)
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
    
    print("\nStarting 3D selection... Close the window when you are finished.")
    vis.run() # This is a blocking call
    vis.destroy_window()
    
    picked_indices = vis.get_picked_points()
    
    if not picked_indices or len(picked_indices) < 6:
        return None
        
    selected_points = [pcd.points[i] for i in picked_indices]
    print(f"\n✓ {len(selected_points)} 3D points selected successfully.")
    return selected_points

def pick_all_2d_points(image, num_points, points_3d):
    """Opens a 2D window to let the user pick points corresponding to the 3D list."""
    points_2d = []
    window_name = "STEP 2: Select 2D Points in Order"
    cv2.namedWindow(window_name)
    
    current_point = 0
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_point
        if event == cv2.EVENT_LBUTTONDOWN and current_point < num_points:
            points_2d.append((x, y))
            color = (0, 255, 0)
            cv2.circle(image, (x, y), 8, color, -1)
            cv2.putText(image, str(current_point + 1), (x + 15, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            print(f"✓ 2D point {current_point + 1} selected.")
            current_point += 1

    cv2.setMouseCallback(window_name, mouse_callback)

    while current_point < num_points:
        display_frame = image.copy()
        prompt = f"Click on the point corresponding to 3D point #{current_point + 1}"
        cv2.putText(display_frame, prompt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None

    cv2.destroyAllWindows()
    return points_2d

def main():
    print("Loading initial calibration data and LiDAR scan...")
    intrinsics_data = load_config(INTRINSICS_FILE)
    camera_matrix = np.array(intrinsics_data['camera_matrix']['data'])
    dist_coeffs = np.array(intrinsics_data['distortion_coefficients']['data'])
    mesh = trimesh.load(LIDAR_SCAN_FILE, force='mesh')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    points_3d = pick_all_3d_points(pcd)
    if points_3d is None:
        print("\n❌ Calibration cancelled: Not enough 3D points were selected (minimum 6).")
        return

    print("\nInitializing camera to capture a static frame for 2D selection...")
    main_config = load_config(CONFIG_FILE)
    camera_source = main_config.get('input_source')
    if isinstance(camera_source, str) and camera_source.isdigit():
        camera_source = int(camera_source)

    cap = cv2.VideoCapture(camera_source)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not capture a frame from the camera.")
        return
    
    points_2d = pick_all_2d_points(frame, len(points_3d), points_3d)
    if points_2d is None or len(points_2d) != len(points_3d):
        print("\n❌ Calibration cancelled during 2D point selection.")
        return

    print(f"\nCalculating final calibration with {len(points_2d)} point pairs...")
    np_points_3d = np.array([points_3d], dtype=np.float32)
    np_points_2d = np.array([points_2d], dtype=np.float32)

    # --- THE FIX: Use calibrateCamera for a global solution ---
    # This function will refine the camera matrix and distortion coefficients
    # WHILE ALSO calculating the rotation and translation vectors.
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        np_points_3d, np_points_2d, frame.shape[:2][::-1], camera_matrix, dist_coeffs, flags=flags
    )

    if ret:
        rvec, tvec = rvecs[0], tvecs[0]
        print("✓ Global calibration successful!")
        projected_points, _ = cv2.projectPoints(np_points_3d, rvec, tvec, mtx, dist)
        total_error = 0
        print(f"\n{'='*60}")
        print("FINAL CALIBRATION QUALITY ASSESSMENT")
        print(f"{'='*60}")
        for i, (orig_2d, proj_2d) in enumerate(zip(np_points_2d.reshape(-1, 2), projected_points.reshape(-1, 2))):
            error = np.linalg.norm(orig_2d - proj_2d)
            status = "✓" if error < 1.0 else "⚠️" if error < 2.0 else "❌"
            print(f"  Point {i+1}: {error:.2f} px {status}")
            total_error += error
        mean_error = total_error / len(points_2d)
        print(f"\n  Mean reprojection error: {mean_error:.2f} pixels")
        quality = "POOR ❌" 
        if mean_error < 0.5: quality = "EXCELLENT ✓"
        elif mean_error < 1.0: quality = "GOOD ✓"
        elif mean_error < 2.0: quality = "ACCEPTABLE ⚠️"
        print(f"  Calibration quality: {quality}")
        print(f"{'='*60}\n")

        # Save the REFINED intrinsics and the calculated extrinsics
        intrinsics_data = {'camera_matrix': {'rows': mtx.shape[0], 'cols': mtx.shape[1], 'data': mtx.tolist()}, 'distortion_coefficients': {'rows': dist.shape[0], 'cols': dist.shape[1], 'data': dist.tolist()}}
        with open(INTRINSICS_FILE, 'w') as f: yaml.dump(intrinsics_data, f, default_flow_style=False)
        extrinsics_data = {'rotation_vector': {'rows': rvec.shape[0], 'cols': rvec.shape[1], 'data': rvec.tolist()}, 'translation_vector': {'rows': tvec.shape[0], 'cols': tvec.shape[1], 'data': tvec.tolist()}}
        with open(OUTPUT_FILE, 'w') as f: yaml.dump(extrinsics_data, f, default_flow_style=False)
        
        print(f"\n✅ CALIBRATION COMPLETE!")
        print(f"  REFINED intrinsics saved to: {INTRINSICS_FILE}")
        print(f"  Extrinsics saved to: {OUTPUT_FILE}")
    else:
        print("\n❌ ERROR: Global calibration failed!")

if __name__ == "__main__":
    main()