
import trimesh
import os

def inspect_gltf(file_path):
    """
    Loads a GLTF file and prints the names of all geometries (meshes) it contains.

    Args:
        file_path (str): The absolute path to the .gltf file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        # Load the scene from the GLTF file
        # The 'force='mesh'' option ensures that if the file contains point clouds,
        # they are loaded as meshes with vertices.
        scene = trimesh.load(file_path, force='mesh')

        # The 'geometry' attribute of a scene is a dictionary containing all meshes
        if isinstance(scene, trimesh.Scene):
            if not scene.geometry:
                print("The scene is empty or contains no identifiable geometries.")
                return
            
            print("Found the following geometries (meshes) in the scene:")
            for geometry_name in scene.geometry.keys():
                print(f"- {geometry_name}")
        
        elif isinstance(scene, trimesh.Trimesh):
            # If the file contains only a single, unnamed mesh, it's loaded directly
            print("Loaded a single, unnamed mesh.")
            print(f"Number of vertices: {len(scene.vertices)}")

    except Exception as e:
        print(f"An error occurred while loading or processing the file: {e}")

if __name__ == "__main__":
    # --- IMPORTANT ---
    # Please update this path to the actual location of your GLTF file.
    lidar_scan_path = "/Users/Arshad_1/Desktop/projects/custom_cv_new/8_6_2025.glb"
    
    print(f"Inspecting GLTF file: {lidar_scan_path}")
    inspect_gltf(lidar_scan_path)
