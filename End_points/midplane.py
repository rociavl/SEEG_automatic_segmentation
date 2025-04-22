import slicer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
import os
import seaborn as sns
from skimage.measure import label, regionprops_table
import vtk
from scipy.spatial.distance import cdist
from skimage.measure import marching_cubes
from Outermost_centroids_coordinates.outermost_centroids_vol_slicer import (
    get_ras_coordinates_from_ijk, get_array_from_volume, calculate_centroids_numpy,
    get_centroids_ras, get_surface_from_volume, convert_surface_vertices_to_ras
)

logging.basicConfig(level=logging.INFO)


# -------------------- Midplane Analysis --------------------
def compute_midsagittal_plane(volume_node):
    """Calculate midplane using surface points or volume bounds"""
    try:
        surface_vertices, _ = get_surface_from_volume(volume_node)
        surface_ras = convert_surface_vertices_to_ras(volume_node, surface_vertices)
        
        if surface_ras.size > 0:
            mid_x = np.median(surface_ras[:, 0])
            logging.info(f"Surface-based midplane: X = {mid_x:.2f} mm")
            return mid_x
    except Exception as e:
        logging.warning(f"Surface midplane failed: {str(e)}")

    # Fallback to volume bounds
    bounds = np.zeros(6)
    volume_node.GetRASBounds(bounds)
    mid_x = (bounds[0] + bounds[1]) / 2
    logging.info(f"Volume bounds midplane: X = {mid_x:.2f} mm")
    return mid_x

def create_midsagittal_plane_node(mid_x, volume_node):
    """Create visible midplane in 3D Slicer"""
    plane_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode")
    plane_node.SetName("MidSagittalPlane")
    
    # Set plane properties
    bounds = np.zeros(6)
    volume_node.GetRASBounds(bounds)
    plane_origin = [mid_x, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
    plane_node.SetOrigin(plane_origin)
    plane_node.SetNormal([1, 0, 0])  # X-normal
    
    # Set plane size with margin
    y_size = (bounds[3] - bounds[2]) * 1.2
    z_size = (bounds[5] - bounds[4]) * 1.2
    plane_node.SetSize(y_size, z_size)
    
    # Configure display
    display_node = plane_node.GetDisplayNode()
    display_node.SetColor(0, 1, 0)  # Green
    display_node.SetOpacity(0.3)
    display_node.SetVisibility(True)
    
    return plane_node

# -------------------- Visualization & Analysis --------------------
def calculate_distances(centroids_ras, mid_x):
    """Calculate distances to midplane"""
    return {label: abs(coords[0] - mid_x) for label, coords in centroids_ras.items()}

def generate_heatmap_visualizations(centroids, distances, mid_x, output_dir, max_distance):
    """Generate enhanced heatmap-style visualizations"""
    coords = np.array(list(centroids.values()))
    dist_values = np.array(list(distances.values()))
    
    # Main figure
    fig = plt.figure(figsize=(20, 15))
    
    # 3D Scatter Plot
    ax1 = fig.add_subplot(231, projection='3d')
    sc1 = ax1.scatter(coords[:,0], coords[:,1], coords[:,2], c=dist_values, 
                     cmap='inferno', s=50, alpha=0.8, edgecolor='w')
    ax1.set_title('3D Electrode Distribution')
    plt.colorbar(sc1, ax=ax1, label='Distance from Midplane (mm)')
    
    # Hexbin Heatmaps
    projections = [('X-Y', 0, 1), ('X-Z', 0, 2), ('Y-Z', 1, 2)]
    for idx, (title, x, y) in enumerate(projections):
        ax = fig.add_subplot(234 + idx)
        hb = ax.hexbin(coords[:,x], coords[:,y], C=dist_values, 
                      gridsize=25, cmap='inferno', reduce_C_function=np.mean)
        ax.set_title(f'{title} Heatmap')
        plt.colorbar(hb, ax=ax, label='Mean Distance (mm)')
    
    # Distance Distribution
    ax_dist = fig.add_subplot(236)
    sns.kdeplot(dist_values, fill=True, ax=ax_dist, color='#d62728')
    ax_dist.axvline(max_distance, color='#2ca02c', linestyle='--')
    ax_dist.set_title('Distance Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'midplane_analysis.png'), dpi=300)
    plt.close()

def plot_surface_with_contacts(surface_ras, surface_faces, centroids, distances, mid_x, output_dir, max_distance):
    """Create 3D plot of brain surface with colored contacts"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot brain surface
    if len(surface_ras) > 0 and len(surface_faces) > 0:
        ax.plot_trisurf(
            surface_ras[:, 0], surface_ras[:, 1], surface_ras[:, 2],
            triangles=surface_faces, alpha=0.1, color='gray', label='Brain Surface'
        )
    
    # Plot contacts with distance coloring
    if len(centroids) > 0:
        sc = ax.scatter(
            centroids[:, 0], centroids[:, 1], centroids[:, 2],
            c=distances, cmap='inferno', s=50, edgecolor='black',
            label='Electrode Contacts'
        )
        plt.colorbar(sc, label='Distance to Midplane (mm)')
    
    # Add midplane visualization
    y_range = surface_ras[:, 1].min()-10, surface_ras[:, 1].max()+10
    z_range = surface_ras[:, 2].min()-10, surface_ras[:, 2].max()+10
    Y, Z = np.meshgrid(np.linspace(*y_range, 10), np.linspace(*z_range, 10))
    X = np.full_like(Y, mid_x)
    ax.plot_surface(X, Y, Z, color='green', alpha=0.2, label='Midplane')
    
    ax.set_xlabel('X (RAS)')
    ax.set_ylabel('Y (RAS)')
    ax.set_zlabel('Z (RAS)')
    ax.set_title(f'Brain Surface with Contacts (Threshold: {max_distance}mm)')
    ax.legend()
    
    plt.savefig(os.path.join(output_dir, 'surface_contacts_plot.png'), dpi=300)
    plt.close()

# -------------------- Main Workflow --------------------
def run_analysis(mask_volume, electrode_volume, output_dir, max_distance=5.0):
    """Complete analysis pipeline"""
    try:
        logging.info("Starting analysis pipeline")
        

        mid_x = compute_midsagittal_plane(mask_volume)
        plane_node = create_midsagittal_plane_node(mid_x, mask_volume)
        
        # 3. Get electrode centroids and surface data
        electrodes_array = get_array_from_volume(electrode_volume)
        centroids_df = calculate_centroids_numpy(electrodes_array)
        centroids_ras = get_centroids_ras(electrode_volume, centroids_df)
        
        # Get surface data for visualization
        surface_vertices, surface_faces = get_surface_from_volume(mask_volume)
        surface_ras = convert_surface_vertices_to_ras(mask_volume, surface_vertices)
        
        if not centroids_ras:
            raise ValueError("No valid centroids found")
        
        # 4. Calculate distances
        distances = calculate_distances(centroids_ras, mid_x)
        
        # 5. Generate visualizations
        os.makedirs(output_dir, exist_ok=True)
        
        # Existing heatmap visualizations
        generate_heatmap_visualizations(centroids_ras, distances, mid_x, output_dir, max_distance)
        
        # New surface+contacts plot
        centroid_points = np.array(list(centroids_ras.values()))
        distance_values = np.array(list(distances.values()))
        plot_surface_with_contacts(surface_ras, surface_faces, centroid_points, 
                                 distance_values, mid_x, output_dir, max_distance)
        
        # 6. Create midplane contacts markups
        filtered = {k:v for k,v in centroids_ras.items() if distances[k] <= max_distance}
        markups = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "MidplaneContacts")
        for label, coords in filtered.items():
            markups.AddControlPoint(coords, f"{label}\n{distances[label]:.1f}mm")
        
        # Configure markups display
        display_node = markups.GetDisplayNode()
        display_node.SetSelectedColor(1, 0, 0)  # Red
        display_node.SetGlyphScale(1.5)
        display_node.SetTextScale(2.0)
        
        print(f"Analysis complete. Results saved to {output_dir}")
        return True
        
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        return False
    
def main():
    """Main entry point"""
    try:
        mask_volume = slicer.util.getNode("patient1_mask_5")
        electrode_volume = slicer.util.getNode("validated_electrode_mask")
        output_dir = r"C:\Users\rocia\Downloads\TFG\Cohort\End_points\output_plots"
        
        success = run_analysis(mask_volume, electrode_volume, output_dir, max_distance=15)
        
    except Exception as e:
        logging.error(f"Execution failed: {str(e)}")

if __name__ == "__main__":
    main()