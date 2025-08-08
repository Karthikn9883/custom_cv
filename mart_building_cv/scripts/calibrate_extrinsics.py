import cv2
import numpy as np
import yaml
import os
import open3d as o3d
import trimesh
import time
from typing import List, Tuple, Optional

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
selection_history = []  # Track selection quality
reprojection_errors = []

def load_config(file_path):
    """Loads a YAML configuration file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found at {file_path}")
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def mouse_callback_2d(event, x, y, flags, param):
    """Handles mouse clicks on the 2D camera feed window with enhanced feedback."""
    global points_2d, camera_feed_frame, selection_history
    if event == cv2.EVENT_LBUTTONDOWN:
        points_2d.append((x, y))
        timestamp = time.time()
        selection_history.append({
            'type': '2d',
            'point': (x, y),
            'timestamp': timestamp,
            'pair_id': len(points_2d)
        })
        
        print(f"\n✓ 2D point {len(points_2d)} selected: ({x}, {y})")
        
        # Enhanced visual feedback with different colors for different points
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), 
                  (255, 0, 255), (255, 255, 0), (128, 0, 128), (255, 165, 0)]
        color = colors[(len(points_2d) - 1) % len(colors)]
        
        # Draw larger circle for visibility
        cv2.circle(camera_feed_frame, (x, y), 8, color, -1)
        cv2.circle(camera_feed_frame, (x, y), 12, (255, 255, 255), 2)
        
        # Add point number
        cv2.putText(camera_feed_frame, str(len(points_2d)), 
                   (x + 15, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("2D Camera Feed - Click to Select Points", camera_feed_frame)
        
        # Check if we have a matching 3D point
        if len(points_3d) == len(points_2d):
            print(f"✓ Point pair {len(points_2d)} completed!")
            print(f"  3D: {points_3d[-1]}")
            print(f"  2D: {points_2d[-1]}")
            print(f"  Total pairs collected: {len(points_2d)}")

def pick_points_3d(pcd):
    """Handles point picking in the 3D visualizer with enhanced feedback."""
    print(f"\n{'='*50}")
    print(f"3D POINT SELECTION #{len(points_3d) + 1}")
    print(f"{'='*50}")
    print("Instructions:")
    print("1. Hold [SHIFT] and click a point in the 3D window")
    print("2. Choose a point that's clearly visible in the 2D camera")
    print("3. A pink sphere will mark your selection")
    print("4. Press [SHIFT] + [Q] to close and confirm selection")
    print("5. Press [ESC] to cancel without selecting")
    print(f"{'='*50}")
    
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=f"3D LiDAR Scan - Pick Point #{len(points_3d) + 1}")
    vis.add_geometry(pcd)
    
    # Add coordinate axes for reference
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(axes)
    
    # Run visualization
    vis.run()
    vis.destroy_window()
    
    picked_indices = vis.get_picked_points()
    if not picked_indices:
        print("⚠️  No 3D point was selected")
        return None
        
    selected_point = pcd.points[picked_indices[0]]
    
    # Store selection in history
    selection_history.append({
        'type': '3d',
        'point': selected_point,
        'timestamp': time.time(),
        'pair_id': len(points_3d) + 1
    })
    
    print(f"✓ 3D point {len(points_3d) + 1} selected: ({selected_point[0]:.3f}, {selected_point[1]:.3f}, {selected_point[2]:.3f})")
    return selected_point

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

    print(f"\n{'='*70}")
    print("ENHANCED EXTRINSIC CALIBRATION TOOL")
    print(f"{'='*70}")
    print("INSTRUCTIONS:")
    print("  We need at least 4 corresponding point pairs for accurate calibration.")
    print("  For best results, select 6-8 well-distributed points.")
    print("\n  FOR EACH POINT PAIR:")
    print("    1. Press 's' to start selecting a new point pair")
    print("    2. First, select the point in the 3D LiDAR window")
    print("    3. Then, select the EXACT same point in the 2D camera feed")
    print("    4. Choose points that are clearly visible in both views")
    print("\n  CONTROLS:")
    print("    's' - Start selecting new point pair")
    print("    'u' - Undo last point pair")
    print("    'r' - Reset/clear all points")
    print("    'd' - Done (calculate extrinsics)")
    print("    'q' - Quit")
    print(f"{'='*70}\n")

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
                print(f"\n⚠️  Error: You need at least 4 point pairs to calculate extrinsics.")
                print(f"Current pairs: {min(len(points_2d), len(points_3d))}")
                continue
            if len(points_2d) != len(points_3d):
                print(f"\n⚠️  Error: Mismatched points - 2D: {len(points_2d)}, 3D: {len(points_3d)}")
                continue
            
            print(f"\n{'='*60}")
            print("STARTING EXTRINSIC CALIBRATION")
            print(f"{'='*60}")
            print(f"Using {len(points_2d)} point pairs")
            break
            
        if key == ord('r'):  # Reset/clear all points
            points_2d.clear()
            points_3d.clear()
            selection_history.clear()
            print("\n✓ All points cleared. Starting fresh...")
            
        if key == ord('u'):  # Undo last point pair
            if points_2d and points_3d:
                points_2d.pop()
                points_3d.pop()
                print(f"\n✓ Last point pair removed. Remaining pairs: {len(points_2d)}")

    cap.release()
    cv2.destroyAllWindows()

    if len(points_2d) >= 4 and len(points_2d) == len(points_3d):
        # Convert lists to numpy arrays
        np_points_3d = np.array(points_3d, dtype=np.float32)
        np_points_2d = np.array(points_2d, dtype=np.float32)

        # Use solvePnP to find the camera's pose
        # Calculate extrinsics with enhanced error analysis
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            np_points_3d, np_points_2d, camera_matrix, dist_coeffs,
            reprojectionError=2.0, confidence=0.99, iterationsCount=1000
        )
        
        if success:
            print(f"✓ Extrinsic calibration successful!")
            print(f"  Inliers: {len(inliers) if inliers is not None else 'N/A'} out of {len(points_2d)} points")
            
            # Calculate reprojection errors for quality assessment
            projected_points, _ = cv2.projectPoints(np_points_3d, rvec, tvec, camera_matrix, dist_coeffs)
            reprojection_errors = []
            
            print(f"\n{'='*60}")
            print("CALIBRATION QUALITY ASSESSMENT")
            print(f"{'='*60}")
            
            total_error = 0
            for i, (orig_2d, proj_2d) in enumerate(zip(np_points_2d, projected_points.reshape(-1, 2))):
                error = np.linalg.norm(orig_2d - proj_2d)
                reprojection_errors.append(error)
                total_error += error
                
                status = "✓" if error < 2.0 else "⚠️" if error < 5.0 else "❌"
                print(f"  Point {i+1}: {error:.2f} px {status}")
            
            mean_error = total_error / len(np_points_2d)
            max_error = max(reprojection_errors)
            
            print(f"\n  Mean reprojection error: {mean_error:.2f} pixels")
            print(f"  Max reprojection error: {max_error:.2f} pixels")
            
            # Quality assessment
            if mean_error < 1.0:
                quality = "EXCELLENT ✓"
            elif mean_error < 2.0:
                quality = "GOOD ✓"
            elif mean_error < 5.0:
                quality = "ACCEPTABLE ⚠️"
            else:
                quality = "POOR ❌ - Consider recalibrating"
            
            print(f"  Calibration quality: {quality}")
            print(f"{'='*60}\n")
            
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
            
            # Also save calibration quality report
            quality_report = {
                'calibration_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'num_points_used': len(points_2d),
                'mean_reprojection_error': float(mean_error),
                'max_reprojection_error': float(max_error),
                'individual_errors': [float(e) for e in reprojection_errors],
                'quality_rating': quality
            }
            
            quality_file = OUTPUT_FILE.replace('.yaml', '_quality_report.yaml')
            with open(quality_file, 'w') as f:
                yaml.dump(quality_report, f, default_flow_style=False)
            
            print(f"\n✅ CALIBRATION COMPLETE!")
            print(f"  Extrinsics saved to: {OUTPUT_FILE}")
            print(f"  Quality report saved to: {quality_file}")
            print(f"  Ready to use with interactive_mapper_3d.py!")
        else:
            print("\n❌ ERROR: Calibration failed!")
            print("  Possible issues:")
            print("  - Points are coplanar")
            print("  - Incorrect point correspondences")
            print("  - Not enough variation in point positions")
            print("  Try selecting points with more spatial distribution.")
    else:
        print(f"\n⚠️  Calibration cancelled or insufficient points.")
        print(f"  Points collected: 2D={len(points_2d)}, 3D={len(points_3d)}")
        print(f"  Need at least 4 matching point pairs for calibration.")
        if selection_history:
            print(f"  Selection history: {len(selection_history)} total selections")

if __name__ == "__main__":
    main()
