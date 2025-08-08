
import trimesh
import numpy as np
from sklearn.linear_model import RANSACRegressor
import yaml
import os

def find_and_save_floor_plane(gltf_path, output_path):
    """
    Loads a 3D mesh, identifies the dominant horizontal plane (the floor)
    using RANSAC, and saves its parameters to a YAML file.

    Args:
        gltf_path (str): Absolute path to the GLTF/GLB file.
        output_path (str): Absolute path to save the output YAML file.
    """
    print(f"Loading mesh from: {gltf_path}")
    try:
        mesh = trimesh.load(gltf_path, force='mesh')
        vertices = mesh.vertices
        print(f"Successfully loaded mesh with {len(vertices)} vertices.")
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    # We assume the floor is mostly aligned with the XY plane, so we model Z = f(X, Y).
    # This means we are looking for a plane equation of the form Ax + By + C = Z.
    xy = vertices[:, :2]
    z = vertices[:, 2]

    # Use RANSAC to find the best-fit plane
    # This is robust to outliers (walls, furniture, etc.)
    ransac = RANSACRegressor(
        min_samples=3,  # A plane is defined by 3 points
        residual_threshold=0.05,  # 5cm tolerance for a point to be an inlier
        max_trials=1000
    )
    ransac.fit(xy, z)

    # The model coefficients give us the plane equation: z = a*x + b*y + d
    # or a*x + b*y - z + d = 0. The normal vector is (a, b, -1).
    a, b = ransac.estimator_.coef_
    d = ransac.estimator_.intercept_

    # Normalize the normal vector
    normal_vector = np.array([a, b, -1])
    normal_vector /= np.linalg.norm(normal_vector)

    # Ensure the normal vector points "up" (positive Z component)
    if normal_vector[2] < 0:
        normal_vector = -normal_vector

    # The plane equation is defined by the normal vector and the distance from the origin (d)
    plane_data = {
        'floor_plane': {
            'normal': normal_vector.tolist(),
            'point_on_plane': [0, 0, d] # A point on the plane
        }
    }

    print(f"Identified floor plane with normal: {plane_data['floor_plane']['normal']}")

    # Save the plane data to the specified YAML file
    try:
        with open(output_path, 'w') as f:
            yaml.dump(plane_data, f, default_flow_style=False)
        print(f"Successfully saved floor plane data to: {output_path}")
    except Exception as e:
        print(f"Error saving YAML file: {e}")

if __name__ == "__main__":
    # --- IMPORTANT ---
    # Please verify these paths are correct.

    # 1. The path to your GLB file from the previous step.
    lidar_scan_path = "/Users/Arshad_1/Desktop/projects/custom_cv_new/8_6_2025.glb"

    # 2. The desired output path for the new config file.
    output_config_path = "/Users/Arshad_1/Desktop/projects/custom_cv_new/mart_building_cv/configs/floor_plane.yaml"

    if not os.path.exists(lidar_scan_path):
        print(f"Error: LiDAR scan file not found at {lidar_scan_path}")
        print("Please update the 'lidar_scan_path' variable in the script.")
    else:
        find_and_save_floor_plane(lidar_scan_path, output_config_path)
