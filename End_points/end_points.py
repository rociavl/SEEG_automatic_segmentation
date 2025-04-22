import numpy as np
import vtk
import slicer
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Outermost_centroids_coordinates.outermost_centroids_vol_slicer import (
    get_ras_coordinates_from_ijk, 
    calculate_centroids_numpy, 
    get_array_from_volume, 
    binarize_array, 
    get_centroids_ras, 
    get_surface_from_volume,
    filter_centroids_by_surface_distance, 
    convert_surface_vertices_to_ras
)

def create_surface_midplane(volume_mask):

    surface_vertices, surface_faces = get_surface_from_volume(volume_mask)
    surface_points_ras = convert_surface_vertices_to_ras(volume_mask, surface_vertices)
    surface_centroid = np.mean(surface_points_ras, axis=0)
    midplane = vtk.vtkPlane()
    midplane.SetOrigin(surface_centroid)
    midplane.SetNormal(1, 0, 0)  # Normal along X-axis, creating YZ plane
    
    return midplane, surface_centroid, surface_points_ras

def find_deepest_contacts(volume_mask, volume_electrodes, max_distance=2.0):
    """
    Find deepest electrode contacts considering:
    1. Distance from midplane
    2. Z-coordinate depth
    3. Distance from brain mask surface
    
    Returns sorted list of contacts with depth scores
    """
    # Use existing surface distance calculation method
    filtered_centroids, surface_points, surface_distances = filter_centroids_by_surface_distance(
        volume_mask, volume_electrodes, "output_plots", max_distance
    )
    
    # Create midplane
    midplane, midplane_center, _ = create_surface_midplane(volume_mask)
    
    # Score contacts based on multiple criteria
    scored_contacts = []
    for label, coords in filtered_centroids.items():
        # Distance from midplane
        midplane_distance = abs(midplane.DistanceToPlane(coords))
        
        # Z-coordinate depth (assuming RAS coordinate system)
        x_depth = abs(coords[0] - midplane_center[0])
        
        # Surface distance (from existing method)
        surface_dist = surface_distances[list(filtered_centroids.keys()).index(label)]
        
        # Combined score (adjustable weights)
        combined_score = (
            0.3 * midplane_distance + 
            0.3 * x_depth + 
            0.4 * surface_dist
        )
        
        scored_contacts.append({
            'label': label,
            'coordinates': coords,
            'midplane_distance': midplane_distance,
            'x_depth': x_depth,
            'surface_distance': surface_dist,
            'score': combined_score
        })
    
    # Sort contacts by score (lower score means deeper/more central)
    deepest_contacts = sorted(scored_contacts, key=lambda x: x['score'])
    
    return deepest_contacts

def create_comprehensive_3d_visualization(volume_mask, volume_electrodes, output_dir):
    """
    Create comprehensive 3D visualizations of brain mask, midplane, and electrode centroids
    Ensures all visualizations use RAS coordinates
    """
    # Get surface from brain mask
    surface_vertices, surface_faces = get_surface_from_volume(volume_mask)
    
    # Get centroids and midplane
    deepest_contacts = find_deepest_contacts(volume_mask, volume_electrodes)
    midplane, midplane_center, surface_points_ras = create_surface_midplane(volume_mask)
    
    # Prepare data for plotting
    centroid_coords = np.array([contact['coordinates'] for contact in deepest_contacts])
    centroid_scores = np.array([contact['score'] for contact in deepest_contacts])
    
    # Create multiple 3D visualization approaches
    plt.figure(figsize=(20, 15))
    
    # 1. Comprehensive 3D Scatter Plot
    ax1 = plt.subplot(221, projection='3d')
    ax1.scatter(surface_points_ras[:, 0], surface_points_ras[:, 1], surface_points_ras[:, 2], 
                alpha=0.1, color='lightblue', label='Brain Surface')
    
    # Centroids
    scatter = ax1.scatter(centroid_coords[:, 0], centroid_coords[:, 1], centroid_coords[:, 2], 
                          c=centroid_scores, cmap='viridis', s=50, label='Centroids')
    ax1.set_title('Brain Surface and Centroids (RAS)')
    plt.colorbar(scatter, ax=ax1, label='Depth Score')
    ax1.legend()
    
    # 2. Centroids with Depth Color Gradient
    ax2 = plt.subplot(222, projection='3d')
    depth_scatter = ax2.scatter(centroid_coords[:, 0], centroid_coords[:, 1], centroid_coords[:, 2], 
                                c=centroid_scores, cmap='coolwarm', s=100)
    ax2.set_title('Centroids Colored by Depth (RAS)')
    plt.colorbar(depth_scatter, ax=ax2, label='Depth Score')
    
    # 3. Surface with Centroid Projection
    ax3 = plt.subplot(223, projection='3d')
    ax3.scatter(surface_points_ras[:, 0], surface_points_ras[:, 1], surface_points_ras[:, 2], 
                alpha=0.1, color='lightblue', label='Brain Surface')
    
    # Project centroids onto surface
    for point in centroid_coords:
        ax3.plot([point[0], point[0]], [point[1], point[1]], 
                 [point[2], surface_points_ras[:, 2].min()], 
                 color='red', linestyle='--', alpha=0.3)
    
    ax3.scatter(centroid_coords[:, 0], centroid_coords[:, 1], centroid_coords[:, 2], 
                color='red', s=50, label='Centroids')
    ax3.set_title('Centroids with Surface Projection (RAS)')
    ax3.legend()
    
    # 4. Midplane Visualization
    ax4 = plt.subplot(224, projection='3d')
    ax4.scatter(surface_points_ras[:, 0], surface_points_ras[:, 1], surface_points_ras[:, 2], 
                alpha=0.1, color='lightblue', label='Brain Surface')
    
    # Visualize midplane
    xx, yy = np.meshgrid(np.linspace(surface_points_ras[:, 0].min(), surface_points_ras[:, 0].max(), 10),
                         np.linspace(surface_points_ras[:, 1].min(), surface_points_ras[:, 1].max(), 10))
    z = np.ones_like(xx) * midplane_center[2]
    ax4.plot_surface(xx, yy, z, alpha=0.3, color='green')
    
    ax4.scatter(centroid_coords[:, 0], centroid_coords[:, 1], centroid_coords[:, 2], 
                color='red', s=50, label='Centroids')
    ax4.set_title('Midplane and Centroids (RAS)')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_3d_visualization.png')
    plt.close()

def visualize_midplane_from_multiple_angles(volume_mask, volume_electrodes, output_dir):
    """
    Create comprehensive midplane visualizations from different perspectives
    Ensures all visualizations use RAS coordinates
    """
    # Get surface from brain mask
    surface_vertices, surface_faces = get_surface_from_volume(volume_mask)
    
    # Get centroids and midplane
    deepest_contacts = find_deepest_contacts(volume_mask, volume_electrodes)
    midplane, midplane_center, surface_points_ras = create_surface_midplane(volume_mask)
    
    # Prepare centroid data
    centroid_coords = np.array([contact['coordinates'] for contact in deepest_contacts])
    centroid_scores = np.array([contact['score'] for contact in deepest_contacts])
    
    # Create figure with multiple subplots
    plt.figure(figsize=(20, 15))
    
    # 1. Sagittal View (YZ Plane)
    ax1 = plt.subplot(221, projection='3d')
    ax1.scatter(surface_points_ras[:, 1], surface_points_ras[:, 2], surface_points_ras[:, 0], 
                alpha=0.1, color='lightblue', label='Brain Surface')
    
    # Create midplane grid for sagittal view
    y_mid = np.linspace(surface_points_ras[:, 1].min(), surface_points_ras[:, 1].max(), 20)
    z_mid = np.linspace(surface_points_ras[:, 2].min(), surface_points_ras[:, 2].max(), 20)
    Y, Z = np.meshgrid(y_mid, z_mid)
    X = np.ones_like(Y) * midplane_center[0]
    
    ax1.plot_surface(X, Y, Z, alpha=0.3, color='red')
    ax1.set_title('Sagittal View (YZ Plane, RAS)')
    ax1.set_xlabel('Y')
    ax1.set_ylabel('Z')
    ax1.set_zlabel('X')
    
    # 2. Coronal View (XZ Plane)
    ax2 = plt.subplot(222, projection='3d')
    ax2.scatter(surface_points_ras[:, 0], surface_points_ras[:, 2], surface_points_ras[:, 1], 
                alpha=0.1, color='lightblue', label='Brain Surface')
    
    # Create midplane grid for coronal view
    x_mid = np.linspace(surface_points_ras[:, 0].min(), surface_points_ras[:, 0].max(), 20)
    z_mid = np.linspace(surface_points_ras[:, 2].min(), surface_points_ras[:, 2].max(), 20)
    X, Z = np.meshgrid(x_mid, z_mid)
    Y = np.ones_like(X) * midplane_center[1]
    
    ax2.plot_surface(X, Z, Y, alpha=0.3, color='green')
    ax2.set_title('Coronal View (XZ Plane, RAS)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_zlabel('Y')
    
    # 3. Axial View (XY Plane)
    ax3 = plt.subplot(223, projection='3d')
    ax3.scatter(surface_points_ras[:, 0], surface_points_ras[:, 1], surface_points_ras[:, 2], 
                alpha=0.1, color='lightblue', label='Brain Surface')
    
    # Create midplane grid for axial view
    x_mid = np.linspace(surface_points_ras[:, 0].min(), surface_points_ras[:, 0].max(), 20)
    y_mid = np.linspace(surface_points_ras[:, 1].min(), surface_points_ras[:, 1].max(), 20)
    X, Y = np.meshgrid(x_mid, y_mid)
    Z = np.ones_like(X) * midplane_center[2]
    
    ax3.plot_surface(X, Y, Z, alpha=0.3, color='blue')
    ax3.set_title('Axial View (XY Plane, RAS)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    # 4. Comprehensive 3D View with All Planes
    ax4 = plt.subplot(224, projection='3d')
    ax4.scatter(surface_points_ras[:, 0], surface_points_ras[:, 1], surface_points_ras[:, 2], 
                alpha=0.1, color='lightblue', label='Brain Surface')
    
    # Scatter centroids
    scatter = ax4.scatter(centroid_coords[:, 0], centroid_coords[:, 1], centroid_coords[:, 2], 
                          c=centroid_scores, cmap='viridis', s=50, label='Centroids')
    
    # Plot all three orthogonal planes
    ax4.plot_surface(X, Y, Z, alpha=0.2, color='blue')   # Axial
    ax4.plot_surface(X, Z, Y, alpha=0.2, color='green')  # Coronal
    ax4.plot_surface(X, Y, Z, alpha=0.2, color='red')    # Sagittal
    
    ax4.set_title('Comprehensive 3D Midplane View (RAS)')
    plt.colorbar(scatter, ax=ax4, label='Depth Score')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/midplane_multiple_views.png')
    plt.close()

def main():
    # Reuse the previous function to get deepest contacts
    volume_mask = slicer.util.getNode("patient1_mask_5")
    volume_electrodes = slicer.util.getNode('validated_electrode_mask')
    
    deepest_contacts = find_deepest_contacts(volume_mask, volume_electrodes)
    
    # Create visualization plots
    create_comprehensive_3d_visualization(
        volume_mask, 
        volume_electrodes, 
        output_dir=r"C:\Users\rocia\Downloads\TFG\Cohort\End_points"
    )
    visualize_midplane_from_multiple_angles(
        volume_mask, 
        volume_electrodes, 
        output_dir=r"C:\Users\rocia\Downloads\TFG\Cohort\End_points"
    )
    
    print("Depth visualization plots have been saved.")

if __name__ == "__main__":
    main()

#exec(open('C:/Users/rocia/AppData/Local/slicer.org/Slicer 5.6.2/SEEG_module/SEEG_masking/End_points/end_points.py').read())