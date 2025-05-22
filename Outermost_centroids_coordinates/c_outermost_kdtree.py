import slicer
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import vtk
from scipy.spatial import KDTree
from skimage.measure import marching_cubes
import logging
import os
import seaborn as sns
from Bolt_head.bolt_head_concensus import VolumeHelper
import time


logging.basicConfig(level=logging.INFO)

def get_array_from_volume(volume_node):
    if volume_node is None:
        logging.error("Volume node is None")
        return None
    return slicer.util.arrayFromVolume(volume_node)

def binarize_array(array, threshold=0):
    return (array > threshold).astype(np.uint8) if array is not None else None

def calculate_centroids_numpy(electrodes_array):
    if electrodes_array is None:
        return pd.DataFrame(columns=['label', 'centroid-0', 'centroid-1', 'centroid-2'])
    
    labeled_array = label(electrodes_array)
    props = regionprops_table(labeled_array, properties=['label', 'centroid'])
    return pd.DataFrame(props)

def get_ras_coordinates_from_ijk(volume_node, ijk):
    ijk_to_ras = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(ijk_to_ras)
    
    homogeneous_ijk = [ijk[0], ijk[1], ijk[2], 1]
    ras = [
        sum(ijk_to_ras.GetElement(i, j) * homogeneous_ijk[j] for j in range(4))
        for i in range(4)
    ]
    return ras[:3]

def get_centroids_ras(volume_node, centroids_df):
    return {
        
        int(row['label']): tuple(get_ras_coordinates_from_ijk(volume_node, (row['centroid-2'], row['centroid-1'], row['centroid-0'])))
        
        for _, row in centroids_df.iterrows()
    }

def get_surface_from_volume(volume_node, threshold=0):
    array = get_array_from_volume(volume_node)
    if array is None:
        return np.array([]), np.array([])
    
    binary_array = binarize_array(array, threshold)
    if binary_array.sum() == 0:
        logging.warning("Binary array is all zeros; no surface to extract.")
        return np.array([]), np.array([])
    
    try:
        vertices, faces, _, _ = marching_cubes(binary_array, level=0)
        return vertices, faces
    except ValueError as e:
        logging.error(f"Marching cubes error: {str(e)}")
        return np.array([]), np.array([])

def convert_surface_vertices_to_ras(volume_node, surface_vertices):
    surface_points_ras = []
    for vertex in surface_vertices:
        ijk = [vertex[2], vertex[1], vertex[0]]  # Convert array (z,y,x) to IJK (x,y,z)
        ras = get_ras_coordinates_from_ijk(volume_node, ijk)
        surface_points_ras.append(ras)
    return np.array(surface_points_ras)

def create_centroids_volume(volume_mask, centroids_ras, output_dir, volume_name="outermost_centroids"):
    """Create a NRRD volume containing only the outermost centroids."""
    spacing = volume_mask.GetSpacing()
    origin = volume_mask.GetOrigin()
    direction_matrix = vtk.vtkMatrix4x4()
    volume_mask.GetIJKToRASDirectionMatrix(direction_matrix)
    
    helper = VolumeHelper(spacing, origin, direction_matrix, output_dir)
    dims = volume_mask.GetImageData().GetDimensions()
    empty_array = np.zeros(dims[::-1], dtype=np.uint8)  # Note: z,y,x order
    
    ras_to_ijk = vtk.vtkMatrix4x4()
    volume_mask.GetRASToIJKMatrix(ras_to_ijk)
    
    for label, ras in centroids_ras.items():
        # Convert RAS to IJK correctly
        homogeneous_ras = [ras[0], ras[1], ras[2], 1]
        ijk = [0, 0, 0, 0]
        ras_to_ijk.MultiplyPoint(homogeneous_ras, ijk)
        
        # Round to nearest voxel and ensure within bounds
        x, y, z = [int(round(coord)) for coord in ijk[:3]]
        if 0 <= x < dims[0] and 0 <= y < dims[1] and 0 <= z < dims[2]:
            empty_array[z, y, x] = label
    
    output_filename = f"{volume_name}.nrrd"
    centroids_volume = helper.create_volume(
        empty_array, 
        volume_name, 
        save_filename=output_filename
    )
    return centroids_volume

def filter_centroids_by_surface_distance(volume_mask, volume_electrodes, output_dir, max_distance=2.0):
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract surface mesh
    surface_vertices, surface_faces = get_surface_from_volume(volume_mask)
    if len(surface_vertices) == 0:
        return {}, np.array([]), []
    
    # Convert surface to RAS coordinates
    surface_points_ras = convert_surface_vertices_to_ras(volume_mask, surface_vertices)
    
    # Build KDTree for fast nearest-neighbor search
    surface_kdtree = KDTree(surface_points_ras)  # O(M log M) time
    
    # Get electrode centroids
    electrodes_array = get_array_from_volume(volume_electrodes)
    if electrodes_array is None:
        return {}, surface_points_ras, []
    
    centroids_df = calculate_centroids_numpy(electrodes_array)
    if centroids_df.empty:
        return {}, surface_points_ras, []
    
    centroids_ras = get_centroids_ras(volume_mask, centroids_df)
    if not centroids_ras:
        return {}, surface_points_ras, []
    
    # Convert centroids to array and query KDTree
    centroid_points = np.array(list(centroids_ras.values()))
    min_distances, _ = surface_kdtree.query(centroid_points)  # O(N log M) time
    
    # Filter by distance
    filtered_centroids = {
        label: coords
        for label, coords, dist in zip(centroids_ras.keys(), centroid_points, min_distances)
        if dist <= max_distance
    }
    
    # Visualization
    plot_3d_surface_and_centroids(
        surface_points_ras,
        surface_faces,
        centroid_points,
        min_distances,
        output_dir,
        max_distance
    )
    
    return filtered_centroids, surface_points_ras, min_distances

def plot_3d_surface_and_centroids(surface_vertices_ras, surface_faces, centroids_ras, distances, output_dir, max_distance=2.0):
    fig = plt.figure(figsize=(18, 10))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('3D Surface and Electrode Centroids')
    
    if len(surface_vertices_ras) > 0 and len(surface_faces) > 0:
        ax1.plot_trisurf(
            surface_vertices_ras[:, 0], 
            surface_vertices_ras[:, 1], 
            surface_vertices_ras[:, 2],
            triangles=surface_faces,
            alpha=0.2, 
            color='blue'
        )
    
    if len(centroids_ras) > 0:
        sc = ax1.scatter(
            centroids_ras[:, 0], 
            centroids_ras[:, 1], 
            centroids_ras[:, 2], 
            c=distances,
            cmap='viridis', 
            s=50,
            edgecolor='black'
        )
        plt.colorbar(sc, ax=ax1, label='Distance to Surface (mm)')
    
    ax1.set_xlabel('X (RAS)')
    ax1.set_ylabel('Y (RAS)')
    ax1.set_zlabel('Z (RAS)')

    # Save the plot
    plt.savefig(os.path.join(output_dir, 'surface_and_centroids.png'))
    plt.close()

def create_markups_from_centroids(centroids):
    markups = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
    markups.SetName("Filtered Centroids")
    
    for label, coords in centroids.items():
        markups.AddControlPoint(coords[0], coords[1], coords[2], f"Centroid_{label}")
    
    return markups

def main(output_dir="output_plots"):
    volume_mask = slicer.util.getNode("patient1_mask_5")
    volume_electrodes = slicer.util.getNode('electrode_mask_success_1')

    start_time = time.time()
    
    filtered_centroids, surface, distances = filter_centroids_by_surface_distance(
        volume_mask, volume_electrodes, output_dir, max_distance=2.0
    )
    
    markups = create_markups_from_centroids(filtered_centroids)
    centroids_volume = create_centroids_volume(volume_mask, filtered_centroids, output_dir)
    
    print(f"Created {len(filtered_centroids)} markups and volume")

    finish_time = time.time()
    minutes = (finish_time - start_time) / 60
    seconds = (finish_time - start_time) % 60
    print(f"Execution time: {int(minutes)} minutes and {int(seconds)} seconds")

    print(f"Results saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main(output_dir=r"C:\Users\rocia\Downloads\TFG\Cohort\Centroids\P1_kdtree\P1_colab\output_plots")

#exec(open(r"C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Outermost_centroids_coordinates\c_outermost_kdtree.py").read())