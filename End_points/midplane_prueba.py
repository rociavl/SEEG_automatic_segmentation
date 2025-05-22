import slicer
import numpy as np
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import label, regionprops_table
from scipy.spatial.distance import cdist
from skimage.measure import marching_cubes
from Outermost_centroids_coordinates.outermost_centroids_vol_slicer import (
    get_array_from_volume, binarize_array, calculate_centroids_numpy, 
    get_centroids_ras, get_surface_from_volume, convert_surface_vertices_to_ras, 
    get_ras_coordinates_from_ijk)

# Configure logging
logging.basicConfig(level=logging.INFO)

# -------------------- Core Functions --------------------
def get_all_centroids(volume_node):
    """Get all electrode centroids in RAS coordinates"""
    electrodes_array = get_array_from_volume(volume_node)
    if electrodes_array is None:
        logging.error("No electrode array found")
        return {}
    
    centroids_df = calculate_centroids_numpy(electrodes_array)
    if centroids_df.empty:
        logging.error("No centroids calculated")
        return {}
    
    return get_centroids_ras(volume_node, centroids_df)

def compute_midsagittal_plane(volume_node):
    """Calculate mid-sagittal plane using surface points with improved fitting"""
    surface_vertices, _ = get_surface_from_volume(volume_node)
    surface_ras = convert_surface_vertices_to_ras(volume_node, surface_vertices)
    
    if surface_ras.size > 0:

        mid_x = np.median(surface_ras[:, 0])  
        normal = np.array([1, 0, 0])  
        point_on_plane = np.array([mid_x, np.median(surface_ras[:, 1]), np.median(surface_ras[:, 2])])
        
        return mid_x, point_on_plane, normal
    
    bounds = np.zeros(6)
    volume_node.GetRASBounds(bounds)
    mid_x = (bounds[0] + bounds[1]) / 2
    point_on_plane = np.array([mid_x, (bounds[2] + bounds[3])/2, (bounds[4] + bounds[5])/2])
    normal = np.array([1, 0, 0])
    
    return mid_x, point_on_plane, normal

# -------------------- Distance Calculation --------------------
def calculate_orthogonal_distance(point, plane_point, plane_normal):
    """Robust distance calculation with input validation"""
    try:
        point = np.asarray(point, dtype=np.float64)
        plane_point = np.asarray(plane_point, dtype=np.float64)
        plane_normal = np.asarray(plane_normal, dtype=np.float64)
        
        # Normalize plane normal
        plane_normal /= np.linalg.norm(plane_normal)
        
        return abs(np.dot(point - plane_point, plane_normal))
    
    except Exception as e:
        logging.error(f"Distance calculation error: {str(e)}")
        raise

def find_closest_to_midplane(centroids_ras, plane_point, plane_normal, max_distance=2.0):
    """Distance calculation with detailed logging"""
    close_contacts = {}
    
    for label, coords in centroids_ras.items():
        try:
            distance = calculate_orthogonal_distance(coords, plane_point, plane_normal)
            if distance <= max_distance:
                close_contacts[label] = {
                    'coords': coords,
                    'distance': distance,
                    'group': label.split('-')[0] if '-' in label else 'U'
                }
                logging.debug(f"Contact {label} distance: {distance:.2f}mm")
        
        except Exception as e:
            logging.error(f"Error processing {label}: {str(e)}")
    
    logging.info(f"Found {len(close_contacts)} contacts within {max_distance}mm")
    return close_contacts
# -------------------- Visualization --------------------
def plot_full_analysis(volume_mask, electrodes_ras, mid_x, plane_origin, normal, midplane_contacts, output_dir, max_distance):
    """Create comprehensive 3D visualization with improved plane representation"""
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot brain surface
    surface_vertices, surface_faces = get_surface_from_volume(volume_mask)
    surface_ras = convert_surface_vertices_to_ras(volume_mask, surface_vertices)
    if surface_ras.size > 0:
        ax.plot_trisurf(
            surface_ras[:, 0], surface_ras[:, 1], surface_ras[:, 2],
            triangles=surface_faces, alpha=0.1, color='lightblue', label='Brain Surface'
        )
    
    # Plot midplane
    y_range = surface_ras[:, 1].min()-10, surface_ras[:, 1].max()+10 if surface_ras.size > 0 else [-100, 100]
    z_range = surface_ras[:, 2].min()-10, surface_ras[:, 2].max()+10 if surface_ras.size > 0 else [-100, 100]
    Y, Z = np.meshgrid(np.linspace(*y_range, 50), np.linspace(*z_range, 50))
    X = np.full_like(Y, mid_x)
    ax.plot_surface(X, Y, Z, alpha=0.2, color='green', label='Midline Plane')
    
    # Plot electrodes
    if electrodes_ras:
        centroids = np.array(list(electrodes_ras.values()))
        labels = list(electrodes_ras.keys())
        
        distances = np.array([
            calculate_orthogonal_distance(c, plane_origin, normal) 
            for c in centroids
        ])
        
        scatter = ax.scatter(
            centroids[:, 0], centroids[:, 1], centroids[:, 2],
            c=distances, cmap='viridis', s=50, edgecolor='black', label='All Electrodes'
        )
        plt.colorbar(scatter, ax=ax, label='Distance from Midplane (mm)')
        
        if midplane_contacts:
            mid_coords = np.array([c['coords'] for c in midplane_contacts.values()])
            ax.scatter(
                mid_coords[:, 0], mid_coords[:, 1], mid_coords[:, 2],
                s=100, edgecolor='red', facecolor='none', linewidth=2, 
                label=f'Midplane Contacts (<{max_distance}mm)'
            )
    
    ax.set_xlabel('X (RAS)\nLeft → Right', fontsize=12)
    ax.set_ylabel('Y (RAS)\nPosterior → Anterior', fontsize=12)
    ax.set_zlabel('Z (RAS)\nInferior → Superior', fontsize=12)
    ax.set_title(f'Midplane Contact Analysis (Threshold: {max_distance}mm)', fontsize=14)
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'midplane_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

# -------------------- Improved Markups Creation --------------------
def create_midplane_markups(midplane_contacts):
    """Create enhanced 3D Slicer markups node for midplane contacts with grouping"""
    if not midplane_contacts:
        print("No midplane contacts to visualize")
        return None
    
    # Create main fiducial node
    markups = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
    markups.SetName("Midplane_Contacts")
    
    # Enhance display properties
    display_node = markups.GetDisplayNode()
    display_node.SetSelectedColor(1, 0, 0)  # Red color
    display_node.SetGlyphScale(1.5)  # Slightly larger than default
    display_node.SetTextScale(2.5)  # Ensure labels are visible
    display_node.SetVisibility(True)
    display_node.SetOpacity(1.0)
    
    # Group contacts by electrode group for better organization
    groups = {}
    for label, data in midplane_contacts.items():
        group = data['electrode_group']
        if group not in groups:
            groups[group] = []
        groups[group].append((label, data))
    
    # Add points with improved labeling
    for group, contacts in groups.items():
        # Sort contacts within each group by distance
        contacts.sort(key=lambda x: x[1]['distance'])
        
        for label, data in contacts:
            coords = data['coords']
            distance = data['distance']
            description = f"{label} ({distance:.2f}mm)"
            markup_id = markups.AddControlPoint(coords, description)
            
            # Set different colors for different electrode groups for better visualization
            # This cycles through predefined colors for each group
            group_idx = list(groups.keys()).index(group)
            colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1)]
            color = colors[group_idx % len(colors)]
            markups.SetNthControlPointSelected(markup_id, False)
            markups.SetNthControlPointLocked(markup_id, True)  # Lock to prevent accidental movement
    
    # Create a table node with the same information
    table_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
    table_node.SetName("Midplane_Contacts_Table")
    table_node.AddColumn().SetName("Label")
    table_node.AddColumn().SetName("Electrode_Group")
    table_node.AddColumn().SetName("X")
    table_node.AddColumn().SetName("Y")
    table_node.AddColumn().SetName("Z") 
    table_node.AddColumn().SetName("Distance(mm)")
    
    # Fill table
    for label, data in midplane_contacts.items():
        row = table_node.AddEmptyRow()
        table_node.SetCellText(row, 0, label)
        table_node.SetCellText(row, 1, data['electrode_group'])
        for i, coord in enumerate(['X', 'Y', 'Z']):
            table_node.SetCellText(row, i+2, f"{data['coords'][i]:.2f}")
        table_node.SetCellText(row, 5, f"{data['distance']:.2f}")
    
    logging.info(f"Created markups node with {markups.GetNumberOfControlPoints()} contacts")
    logging.info(f"Created table with {table_node.GetNumberOfRows()} contacts")
    
    return markups, table_node

def create_midsagittal_plane_node(volume_node, plane_origin, plane_normal=[1,0,0]):
    """Create a visible plane node in 3D Slicer with correct display properties"""
    plane_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode")
    plane_node.SetName("MidSagittalPlane")
    
    # Set plane properties
    plane_node.SetOrigin(plane_origin)
    plane_node.SetNormal(plane_normal)
    
    # Configure display properties
    display_node = plane_node.GetDisplayNode()
    display_node.SetColor(0, 1, 0)  # Green
    display_node.SetOpacity(0.3)     # 30% opacity
    display_node.SetVisibility(True) # Ensure plane is visible
    
    return plane_node

# -------------------- Main Workflow --------------------
def run_full_analysis(volume_mask, volume_electrodes, output_dir, max_distance=2.0):
    """Complete analysis pipeline with improved distance calculations"""
    logging.info("Starting analysis pipeline")

    mid_x, plane_origin, normal = compute_midsagittal_plane(volume_mask)
    print(f"Computed midplane at X = {mid_x:.2f} mm")

    plane_node = create_midsagittal_plane_node(volume_mask, plane_origin, normal)

    electrodes_ras = get_all_centroids(volume_electrodes)
    if not electrodes_ras:
        logging.error("No electrodes found - check input volumes")
        return None

    midplane_contacts = find_closest_to_midplane(
        electrodes_ras, plane_origin, normal, max_distance
    )
    print(f"Found {len(midplane_contacts)} contacts within {max_distance}mm of midplane")

    os.makedirs(output_dir, exist_ok=True)

    plot_full_analysis(
        volume_mask, electrodes_ras, mid_x, plane_origin, normal, 
        midplane_contacts, output_dir, max_distance
    )

    if midplane_contacts:
        df = pd.DataFrame([
            {
                'Label': label, 
                'Electrode_Group': data['electrode_group'],
                'X': data['coords'][0], 
                'Y': data['coords'][1], 
                'Z': data['coords'][2], 
                'Distance(mm)': data['distance']
            }
            for label, data in midplane_contacts.items()
        ])
        # Sort by electrode group and distance for better organization
        df = df.sort_values(['Electrode_Group', 'Distance(mm)'])
        
        csv_path = os.path.join(output_dir, 'midplane_contacts.csv')
        df.to_csv(csv_path, index=False)
        print(f"Exported contacts to {csv_path}")
    
    return midplane_contacts

# def main():
#     try:
#         # Initialize with validation
#         mask_volume = slicer.util.getNode("patient1_mask_5")
#         electrodes_volume = slicer.util.getNode('validated_electrode_mask')
#         output_dir = r"C:\Users\rocia\Downloads\TFG\Cohort\Centroids\End_points\output_plots"
#         logging.info("Starting analysis with verification:")
        
#         # Step 1: Verify plane position
#         mid_x, plane_point, normal = compute_midsagittal_plane(mask_volume)
#         logging.info(f"Plane position: X={mid_x:.2f}, Y={plane_point[1]:.2f}, Z={plane_point[2]:.2f}")
        
#         # Step 2: Verify electrode coordinates
#         electrodes = get_all_centroids(electrodes_volume)
#         sample_label = next(iter(electrodes)) if electrodes else None
#         if sample_label:
#             logging.info(f"Sample electrode {sample_label} at {electrodes[sample_label]}")

#         midplane_contacts = run_full_analysis(
#             mask_volume,
#             electrodes_volume,
#             output_dir,
#             max_distance=5.0 
#         )

#         if midplane_contacts:
#             create_midplane_markups(midplane_contacts)
#             logging.info(f"Created {len(midplane_contacts)} markups")
        
#         logging.info("Process completed successfully")
        
#     except Exception as e:
#         logging.error(f"Main execution failed: {str(e)}")
# main()
#exec(open('C:/Users/rocia/AppData/Local/slicer.org/Slicer 5.6.2/SEEG_module/SEEG_masking/End_points/midplane.py').read())