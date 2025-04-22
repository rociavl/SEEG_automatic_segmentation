import slicer
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from skimage.measure import marching_cubes
import logging
import os
from Outermost_centroids_coordinates.outermost_centroids_vol_slicer import (
    get_ras_coordinates_from_ijk,get_array_from_volume, calculate_centroids_numpy,
    get_centroids_ras, get_surface_from_volume, convert_surface_vertices_to_ras )


def find_center_centroids(volume_mask, volume_electrodes, output_dir, min_distance=10, max_distance=30):
    os.makedirs(output_dir, exist_ok=True)
    
    # Get surface vertices
    surface_vertices, _ = get_surface_from_volume(volume_mask)
    surface_points_ras = np.array([
        get_ras_coordinates_from_ijk(volume_mask, [v[2], v[1], v[0]]) 
        for v in surface_vertices
    ])
    
    # Get electrode centroids
    electrodes_array = get_array_from_volume(volume_electrodes)
    centroids_df = calculate_centroids_numpy(electrodes_array)
    centroids_ras = get_centroids_ras(volume_mask, centroids_df)
    
    # Calculate distances to surface
    centroid_points = np.array(list(centroids_ras.values()))
    distances = cdist(centroid_points, surface_points_ras, 'euclidean')
    min_distances = np.min(distances, axis=1)
    
    # Filter centroids based on distance from surface
    center_centroids = {
        label: coords
        for label, coords, dist in zip(centroids_ras.keys(), centroid_points, min_distances)
        if min_distance <= dist <= max_distance
    }
    
    # Plotting
    plt.figure(figsize=(15, 10))
    ax = plt.subplot(111, projection='3d')
    
    # Plot surface points
    ax.scatter(
        surface_points_ras[:, 0], 
        surface_points_ras[:, 1], 
        surface_points_ras[:, 2], 
        alpha=0.1, 
        c='blue', 
        s=1, 
        label='Surface'
    )
    
    # Plot center centroids
    center_points = np.array(list(center_centroids.values()))
    scatter = ax.scatter(
        center_points[:, 0], 
        center_points[:, 1], 
        center_points[:, 2], 
        c=list(range(len(center_points))),  # Color gradient
        cmap='viridis', 
        s=100, 
        edgecolor='black', 
        label='Center Centroids'
    )
    
    plt.colorbar(scatter, ax=ax, label='Centroid Index')
    ax.set_title('Brain Centroids Near Center')
    ax.set_xlabel('X (RAS)')
    ax.set_ylabel('Y (RAS)')
    ax.set_zlabel('Z (RAS)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'center_centroids.png'))
    plt.close()
    
    return center_centroids

def main(output_dir="output_plots"):
    volume_mask = slicer.util.getNode("patient1_mask_5")
    volume_electrodes = slicer.util.getNode('validated_electrode_mask')
    
    center_centroids = find_center_centroids(
        volume_mask, 
        volume_electrodes, 
        output_dir, 
        min_distance=9, 
        max_distance=20  )
    
    print(f"Found {len(center_centroids)} centroids near the brain center")
    print(f"Plot saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main(output_dir=r"C:\Users\rocia\Downloads\TFG\Cohort\Centroids\P1\P1_colab\output_plots")