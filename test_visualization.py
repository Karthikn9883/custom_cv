#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mart_building_cv', 'src'))

# Configure matplotlib backend
import matplotlib
matplotlib.use('Qt5Agg', force=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Test LiDAR loading and visualization
import trimesh
import numpy as np

print("Testing improved LiDAR visualization...")

# Load LiDAR scan
lidar_path = "8_6_2025.glb"
print(f"Loading LiDAR scan: {lidar_path}")
mesh = trimesh.load(lidar_path, force='mesh')
point_cloud = np.array(mesh.vertices)
print(f"✓ Loaded {len(point_cloud)} points")

# Create 3D plot with improved settings
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Apply improved visualization
downsample_factor = max(1, len(point_cloud) // 15000)
downsampled_pc = point_cloud[::downsample_factor]
colors = downsampled_pc[:, 2]  # Color by height

print(f"✓ Visualizing {len(downsampled_pc)} points (downsampled from {len(point_cloud)})")

# Plot with improved settings
scatter = ax.scatter(downsampled_pc[:, 0], downsampled_pc[:, 1], downsampled_pc[:, 2], 
                    c=colors, s=2.0, alpha=0.8, cmap='viridis', label='LiDAR Scan')

# Set labels and title
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_zlabel('Z (meters)')
ax.set_title('3D LiDAR Scan - Improved Visualization')

# Set proper axis limits
margin = 0.1
ax.set_xlim(downsampled_pc[:, 0].min() - margin, downsampled_pc[:, 0].max() + margin)
ax.set_ylim(downsampled_pc[:, 1].min() - margin, downsampled_pc[:, 1].max() + margin)
ax.set_zlim(downsampled_pc[:, 2].min() - margin, downsampled_pc[:, 2].max() + margin)

# Set good viewing angle
ax.view_init(elev=20, azim=45)

# Add colorbar
plt.colorbar(scatter, ax=ax, label='Height (Z meters)')

print("\n✅ Improved visualization created!")
print("You should now see:")
print("  - Colored point cloud based on height")
print("  - Much more visible points (larger, less transparent)")
print("  - Proper axis limits fitted to your room")
print("  - Better viewing angle")
print("\nClose the window to continue...")

plt.show()