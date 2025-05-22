import slicer
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import vtk
from scipy.spatial.distance import cdist
from skimage.measure import marching_cubes
import logging
import os
import seaborn as sns
from scipy.ndimage import distance_transform_edt
from Bolt_head.bolt_head_concensus import VolumeHelper

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
        int(row['label']): tuple(get_ras_coordinates_from_ijk(volume_node, [row['centroid-2'], row['centroid-1'], row['centroid-0']]))
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
    """
    Create a NRRD volume containing only the outermost centroids.
    
    Args:
        volume_mask: Reference volume node (for spacing/orientation)
        centroids_ras: Dictionary of {label: RAS coordinates}
        output_dir: Directory to save the output
        volume_name: Base name for the output volume
    
    Returns:
        vtkMRMLScalarVolumeNode: The created volume node
    """
    # Get reference volume properties
    spacing = volume_mask.GetSpacing()
    origin = volume_mask.GetOrigin()
    direction_matrix = vtk.vtkMatrix4x4()
    volume_mask.GetIJKToRASDirectionMatrix(direction_matrix)
    
    # Create helper instance
    helper = VolumeHelper(spacing, origin, direction_matrix, output_dir)
    
    # Get array dimensions from reference volume
    dims = volume_mask.GetImageData().GetDimensions()
    empty_array = np.zeros(dims[::-1], dtype=np.uint8)  # Note: z,y,x order
    
    # Convert RAS centroids to IJK indices
    ras_to_ijk = vtk.vtkMatrix4x4()
    volume_mask.GetRASToIJKMatrix(ras_to_ijk)
    
    # Mark centroid positions in the array
    for label, ras in centroids_ras.items():
        # Convert RAS to IJK
        homogeneous_ras = [ras[0], ras[1], ras[2], 1]
        ijk = [
            sum(ras_to_ijk.GetElement(i, j) * homogeneous_ras[j] for j in range(4)
            for i in range(3))
        ]
        
        # Round to nearest voxel and ensure within bounds
        x, y, z = [int(round(coord)) for coord in ijk]
        if 0 <= x < dims[0] and 0 <= y < dims[1] and 0 <= z < dims[2]:
            empty_array[z, y, x] = label  # Use label as intensity value
    
    # Create and save volume
    output_filename = f"{volume_name}.nrrd"
    centroids_volume = helper.create_volume(
        empty_array, 
        volume_name, 
        save_filename=output_filename
    )
    return centroids_volume

def plot_3d_surface_and_centroids(surface_vertices_ras, surface_faces, centroids_ras, distances, output_dir, max_distance=2.0):
    fig = plt.figure(figsize=(18, 10))
    
    # 3D Surface and Centroids Plot
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

    # 2D Projection Plots
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(324)
    ax4 = fig.add_subplot(326)
    
    projection_plots = [
        (ax2, 'Axial (X-Y)', 0, 1),
        (ax3, 'Coronal (X-Z)', 0, 2),
        (ax4, 'Sagittal (Y-Z)', 1, 2)
    ]
    
    for ax, title, x_dim, y_dim in projection_plots:
        if len(surface_vertices_ras) > 0:
            ax.scatter(
                surface_vertices_ras[:, x_dim], 
                surface_vertices_ras[:, y_dim],
                alpha=0.1, 
                c='blue', 
                s=1,
                label='Surface'
            )
        
        if len(centroids_ras) > 0:
            im = ax.scatter(
                centroids_ras[:, x_dim],
                centroids_ras[:, y_dim],
                c=distances,
                cmap='viridis',
                s=40,
                edgecolor='black',
                label='Centroids'
            )
        
        ax.set_title(title)
        ax.set_xlabel(['X (RAS)', 'X (RAS)', 'Y (RAS)'][x_dim])
        ax.set_ylabel(['Y (RAS)', 'Z (RAS)', 'Z (RAS)'][y_dim])
        ax.grid(True)
        ax.legend()

    plt.colorbar(im, ax=projection_plots[-1][0], label='Distance to Surface (mm)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_analysis.png'))
    plt.close()

def filter_centroids_by_surface_distance(volume_mask, volume_electrodes, output_dir, max_distance=2.0):
    os.makedirs(output_dir, exist_ok=True)
    
    surface_vertices, surface_faces = get_surface_from_volume(volume_mask)
    if len(surface_vertices) == 0:
        return {}, np.array([]), []
    
    surface_points_ras = convert_surface_vertices_to_ras(volume_mask, surface_vertices)
    
    electrodes_array = get_array_from_volume(volume_electrodes)
    if electrodes_array is None:
        return {}, surface_points_ras, []
    
    centroids_df = calculate_centroids_numpy(electrodes_array)
    if centroids_df.empty:
        return {}, surface_points_ras, []
    
    centroids_ras = get_centroids_ras(volume_mask, centroids_df)
    if not centroids_ras:
        return {}, surface_points_ras, []
    
    centroid_points = np.array(list(centroids_ras.values()))
    if centroid_points.size == 0:
        return {}, surface_points_ras, []
    
    distances = cdist(centroid_points, surface_points_ras, 'euclidean')
    min_distances = np.min(distances, axis=1)
    
    filtered_centroids = {
        label: coords
        for label, coords, dist in zip(centroids_ras.keys(), centroid_points, min_distances)
        if dist <= max_distance
    }
    
    # Plot distance distribution
    plt.figure(figsize=(10, 5))
    plt.hist(min_distances, bins=20, edgecolor='black')
    plt.title('Distance to Surface Distribution')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'distance_distribution.png'))
    plt.close()
    
    # Create comprehensive plots
    plot_3d_surface_and_centroids(
        surface_points_ras,
        surface_faces,
        centroid_points,
        min_distances,
        output_dir,
        max_distance
    )
    
    return filtered_centroids, surface_points_ras, min_distances

def create_markups_from_centroids(centroids):
    """Create markups node using updated control point API"""
    markups = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
    markups.SetName("Filtered Centroids")
    
    for label, coords in centroids.items():
        index = markups.AddControlPoint(
            coords[0],  
            coords[1],  
            coords[2],  
            f"Centroid_{label}"  
        )
    
    return markups

def create_anatomical_layers(volume_mask_node, layer_thickness_mm=2.0, num_layers=4):
    """
    Creates nested anatomical layers from brain surface inward.
    Returns list of vtkMRMLScalarVolumeNodes representing each layer.
    """
    # Get binary mask array and spacing
    brain_array = slicer.util.arrayFromVolume(volume_mask_node)
    binary_mask = (brain_array > 0).astype(np.uint8)
    spacing = volume_mask_node.GetSpacing()
    
    # Compute distance from surface (in mm)
    distance_map = distance_transform_edt(binary_mask, sampling=spacing)
    
    # Create layered volumes
    layer_nodes = []
    for i in range(num_layers):
        min_dist = i * layer_thickness_mm
        max_dist = (i+1) * layer_thickness_mm
        
        layer_mask = ((distance_map >= min_dist) & (distance_map < max_dist)).astype(np.uint8)
        
        layer_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        layer_node.SetName(f"Brain_Layer_{i+1}_{min_dist:.1f}-{max_dist:.1f}mm")
        slicer.util.updateVolumeFromArray(layer_node, layer_mask)
        layer_node.Copy(volume_mask_node)  # Copy geometry
        layer_nodes.append(layer_node)
    
    return layer_nodes

def assign_contacts_to_layers(electrode_node, layer_nodes):
    """
    Assigns each contact to its corresponding anatomical layer.
    Returns dictionary: {contact_name: layer_name}
    """
    contacts_layer = {}
    
    # Get all contact positions
    markups = electrode_node.GetMarkups()
    for i in range(markups.GetNumberOfControlPoints()):
        contact_ras = [0,0,0]
        markups.GetNthControlPointPositionWorld(i, contact_ras)
        contact_name = markups.GetNthControlPointLabel(i)
        
        # Convert RAS to IJK for each layer
        for layer in layer_nodes:
            ijk = [0,0,0]
            layer.GetRASToIJKMatrix().MultiplyPoint(contact_ras + [1], ijk)
            ijk = [int(round(x)) for x in ijk[:3]]
            
            # Check if in layer volume bounds
            dims = layer.GetImageData().GetDimensions()
            if all(0 <= ijk[d] < dims[d] for d in range(3)):
                layer_array = slicer.util.arrayFromVolume(layer)
                if layer_array[ijk[2], ijk[1], ijk[0]]:  # z,y,x order
                    contacts_layer[contact_name] = layer.GetName()
                    break
    
    return contacts_layer

def main(output_dir="output_plots"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get input volumes
    volume_mask = slicer.util.getNode("patient1_mask_5")
    volume_electrodes = slicer.util.getNode('electrode_mask_success')
    
    # 1. Filter centroids by surface distance
    filtered_centroids, surface, distances = filter_centroids_by_surface_distance(
        volume_mask, volume_electrodes, output_dir, max_distance=2.0
    )
    
    # 2. Create markups for filtered centroids
    markups = create_markups_from_centroids(filtered_centroids)
    
    # 3. Create centroids volume
    centroids_volume = create_centroids_volume(
        volume_mask, 
        filtered_centroids, 
        output_dir
    )
    
    # 4. Create anatomical layers and assign contacts
    anatomical_layers = create_anatomical_layers(
        volume_mask_node=volume_mask,
        layer_thickness_mm=1.0,
        num_layers=4
    )
    
    # Assign electrodes to layers (assuming electrode_list contains markup fiducial nodes)
    electrode_list = [markups]  # Using our filtered centroids markups
    contact_assignments = {}
    for electrode_node in electrode_list:
        contact_assignments.update(assign_contacts_to_layers(electrode_node, anatomical_layers))
    
    # Save contact assignments to file
    with open(os.path.join(output_dir, 'contact_assignments.txt'), 'w') as f:
        f.write("Contact Layer Assignments:\n")
        for contact, layer in contact_assignments.items():
            f.write(f"{contact}: {layer}\n")
    
    # Visualize layers with rainbow colormap and 50% opacity
    for layer in anatomical_layers:
        # Create display node if doesn't exist
        if not layer.GetDisplayNode():
            layer.CreateDefaultDisplayNodes()
        
        display_node = layer.GetDisplayNode()
        display_node.SetAndObserveColorNodeID("vtkMRMLColorTableNodeRainbow")
        display_node.SetOpacity(0.5)
        
        # Auto-contrast for visibility
        display_node.AutoWindowLevelOn()
    
    print(f"Created {len(filtered_centroids)} centroids markups and volume")
    print(f"Created {len(anatomical_layers)} anatomical layers")
    print(f"Output saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main(output_dir=r"C:\Users\rocia\Downloads\TFG\Cohort\Centroids\P1\P1_colab_layering\output_plots")

#exec(open(r"C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Outermost_centroids_coordinates\layering.py").read())


