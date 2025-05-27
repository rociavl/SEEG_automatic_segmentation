"""
Electrode Trajectory Analysis Module

This module provides functionality for analyzing SEEG electrode trajectories in 3D space.
It performs clustering, community detection, and trajectory analysis on electrode coordinates.

The module is structured into several components:
1. Data processing - Functions for extracting and processing electrode data
2. Analysis - Core analysis algorithms (DBSCAN, Louvain, PCA)
3. Visualization - Functions for creating visualizations and reports
4. Utilities - Helper functions and classes

Author: Rocío Ávalos

"""

import slicer
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import networkx as nx
from scipy.spatial.distance import cdist
from collections import defaultdict
from scipy.interpolate import splprep, splev
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from skimage.measure import label, regionprops_table
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import pandas as pd
import time

# Import utility functions from external modules
from Outermost_centroids_coordinates.outermost_centroids_vol_slicer import (
    get_ras_coordinates_from_ijk, get_array_from_volume, calculate_centroids_numpy,
    get_centroids_ras, get_surface_from_volume, convert_surface_vertices_to_ras, 
    filter_centroids_by_surface_distance
)
from End_points.midplane_prueba import get_all_centroids
from Electrode_path.construction_4 import (create_summary_page, create_3d_visualization,
    create_trajectory_details_page, create_noise_points_page)
#------------------------------------------------------------------------------
# PART 1: UTILITY CLASSES AND FUNCTIONS
#------------------------------------------------------------------------------

class Arrow3D(FancyArrowPatch):
    """
    A custom 3D arrow patch for visualization in matplotlib.
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return min(zs)


def calculate_angles(direction):
    """
    Calculate angles between a direction vector and principal axes.
    
    Args:
        direction (numpy.ndarray): A 3D unit vector representing direction
        
    Returns:
        dict: Dictionary containing angles with X, Y, and Z axes in degrees
    """
    angles = {}
    axes = {
        'X': np.array([1, 0, 0]),
        'Y': np.array([0, 1, 0]),
        'Z': np.array([0, 0, 1])
    }
    
    for name, axis in axes.items():
        dot_product = np.dot(direction, axis)
        angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        angles[name] = angle
        
    return angles

#------------------------------------------------------------------------------
# PART 2: CORE ANALYSIS FUNCTIONS
#------------------------------------------------------------------------------
def extract_trajectories_from_combined_mask(combined_volume, brain_volume=None):
    """
    Extract trajectories directly from the combined mask volume that contains
    bolt heads (value=1), entry points (value=2), and trajectory lines (value=3).
    
    This function:
    1. Identifies bolt heads and entry points in the volume
    2. Identifies trajectory lines connecting them
    3. Calculates direction vectors from bolt heads toward the brain
    
    Args:
        combined_volume: Slicer volume node containing the combined mask
        brain_volume: Optional brain mask volume for validation
        
    Returns:
        dict: Dictionary with bolt IDs as keys and dictionaries of 
             {'start_point', 'end_point', 'direction', 'length', 'trajectory_points'} as values
    """
    # Get array from combined volume
    combined_array = get_array_from_volume(combined_volume)
    if combined_array is None or np.sum(combined_array) == 0:
        print("No data found in combined mask.")
        return {}
    
    # Create separate masks for each component
    bolt_mask = (combined_array == 1)
    entry_mask = (combined_array == 2)
    trajectory_mask = (combined_array == 3)
    
    # Label each component
    bolt_labeled = label(bolt_mask, connectivity=3)
    entry_labeled = label(entry_mask, connectivity=3)
    
    # Get region properties for bolts and entry points
    bolt_props = regionprops_table(bolt_labeled, properties=['label', 'centroid', 'area'])
    entry_props = regionprops_table(entry_labeled, properties=['label', 'centroid', 'area'])
    
    # Get bolt head centroids in RAS
    bolt_centroids_ras = {}
    for i in range(len(bolt_props['label'])):
        bolt_id = bolt_props['label'][i]
        centroid = [bolt_props[f'centroid-{j}'][i] for j in range(3)]
        
        # Convert to RAS
        bolt_ras = get_ras_coordinates_from_ijk(
            combined_volume, 
            [centroid[2], centroid[1], centroid[0]]
        )
        
        bolt_centroids_ras[bolt_id] = {
            'centroid': bolt_ras,
            'area': bolt_props['area'][i],
            'ijk_centroid': centroid
        }
    
    # Get entry point centroids in RAS
    entry_centroids_ras = {}
    for i in range(len(entry_props['label'])):
        entry_id = entry_props['label'][i]
        centroid = [entry_props[f'centroid-{j}'][i] for j in range(3)]
        
        # Convert to RAS
        entry_ras = get_ras_coordinates_from_ijk(
            combined_volume, 
            [centroid[2], centroid[1], centroid[0]]
        )
        
        entry_centroids_ras[entry_id] = {
            'centroid': entry_ras,
            'area': entry_props['area'][i],
            'ijk_centroid': centroid
        }
    
    print(f"Found {len(bolt_centroids_ras)} bolt heads and {len(entry_centroids_ras)} entry points")
    
    # Process trajectories
    trajectories = {}
    
    # For each bolt, find connected entry point and trajectory
    for bolt_id, bolt_info in bolt_centroids_ras.items():
        bolt_centroid_ijk = bolt_info['ijk_centroid']
        bolt_point_ras = bolt_info['centroid']
        
        # For each entry point, check if there's a trajectory connecting to this bolt
        closest_entry = None
        min_distance = float('inf')
        
        for entry_id, entry_info in entry_centroids_ras.items():
            entry_centroid_ijk = entry_info['ijk_centroid']
            entry_point_ras = entry_info['centroid']
            
            # Calculate Euclidean distance between bolt and entry in RAS space
            # Ensure we're working with numpy arrays
            bolt_point_np = np.array(bolt_point_ras)
            entry_point_np = np.array(entry_point_ras)
            distance = np.linalg.norm(bolt_point_np - entry_point_np)
            
            # Check if there's a trajectory path between them by finding connected components
            # Create a temporary mask combining bolt, entry and trajectory
            temp_mask = np.zeros_like(combined_array, dtype=bool)
            temp_mask[bolt_labeled == bolt_id] = True
            temp_mask[entry_labeled == entry_id] = True
            temp_mask[trajectory_mask] = True
            
            # Label the connected components
            connected_labeled = label(temp_mask, connectivity=1)
            
            # Get the label at bolt centroid position
            x, y, z = np.round(bolt_centroid_ijk).astype(int)
            bolt_component = connected_labeled[x, y, z]
            
            # Get the label at entry centroid position
            x, y, z = np.round(entry_centroid_ijk).astype(int)
            entry_component = connected_labeled[x, y, z]
            
            # If both are in the same connected component, they're connected by a trajectory
            if bolt_component == entry_component and bolt_component != 0:
                if distance < min_distance:
                    min_distance = distance
                    closest_entry = {
                        'entry_id': entry_id,
                        'entry_point': entry_point_ras,
                        'distance': distance,
                        'connected_component': bolt_component
                    }
        
        # If we found a connected entry point, extract the trajectory
        if closest_entry:
            # Calculate direction from bolt to entry (pointing toward brain)
            bolt_point_np = np.array(bolt_point_ras)
            entry_point_np = np.array(closest_entry['entry_point'])
            bolt_to_entry = entry_point_np - bolt_point_np
            
            length = np.linalg.norm(bolt_to_entry)
            direction = bolt_to_entry / length if length > 0 else np.array([0, 0, 0])
            
            # Extract the trajectory points from the connected component
            connected_component = closest_entry['connected_component']
            component_mask = (connected_labeled == connected_component)
            
            # Extract only the trajectory part (value=3)
            trajectory_points_mask = component_mask & trajectory_mask
            trajectory_coords = np.argwhere(trajectory_points_mask)
            
            # Convert trajectory points to RAS
            trajectory_points_ras = []
            for coord in trajectory_coords:
                ras = get_ras_coordinates_from_ijk(combined_volume, [coord[2], coord[1], coord[0]])
                trajectory_points_ras.append(ras)
            
            # Store trajectory information
            trajectories[int(bolt_id)] = {
                'start_point': bolt_point_ras,     # Store as original data type (list)
                'end_point': closest_entry['entry_point'],  # Store as original data type (list)
                'direction': direction.tolist(),   # Convert numpy array to list
                'length': float(length),
                'entry_id': int(closest_entry['entry_id']),
                'trajectory_points': trajectory_points_ras,
                'method': 'combined_mask_direct'
            }
    
    print(f"Extracted {len(trajectories)} bolt-to-entry trajectories from combined mask")
    
    # If brain volume is provided, verify directions point toward brain
    if brain_volume and trajectories:
        print("Verifying directions with brain volume...")
        verify_directions_with_brain(trajectories, brain_volume)
    
    return trajectories

def create_trajectory_lines_volume(bolt_directions, volume_template, output_dir):
    """
    Create a volume visualizing bolt-to-brain trajectories as lines.
    
    Args:
        bolt_directions (dict): Direction info from extract_bolt_to_entry_directions
        volume_template: Template volume node to get dimensions and spacing
        output_dir (str): Directory to save the output volume
        
    Returns:
        slicer.vtkMRMLScalarVolumeNode: Volume node containing trajectory lines
    """
    # Get dimensions and properties from template volume
    dims = volume_template.GetImageData().GetDimensions()
    spacing = volume_template.GetSpacing()
    origin = volume_template.GetOrigin()
    
    # Create a new volume with same dimensions
    volume_array = np.zeros(dims[::-1], dtype=np.uint8)
    
    # For each bolt direction, draw a line from bolt to entry point
    for bolt_id, bolt_info in bolt_directions.items():
        start_point = np.array(bolt_info['start_point'])
        end_point = np.array(bolt_info['end_point'])
        
        # Convert RAS coordinates to IJK
        start_ijk = np.round(
            (start_point - np.array(origin)) / np.array(spacing)
        ).astype(int)
        start_ijk = start_ijk[::-1]  # Reverse order for NumPy indexing
        
        end_ijk = np.round(
            (end_point - np.array(origin)) / np.array(spacing)
        ).astype(int)
        end_ijk = end_ijk[::-1]  # Reverse order for NumPy indexing
        
        # Draw line using Bresenham's algorithm
        line_points = _bresenham_line_3d(
            start_ijk[0], start_ijk[1], start_ijk[2],
            end_ijk[0], end_ijk[1], end_ijk[2]
        )
        
        # Set line points in the volume
        for point in line_points:
            x, y, z = point
            if (0 <= x < volume_array.shape[0] and 
                0 <= y < volume_array.shape[1] and 
                0 <= z < volume_array.shape[2]):
                volume_array[x, y, z] = bolt_id  # Use bolt ID as voxel value
    
    # Create volume node
    volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "BoltTrajectoryLines")
    volume_node.SetSpacing(spacing)
    volume_node.SetOrigin(origin)
    
    # Set direction matrix 
    direction_matrix = vtk.vtkMatrix4x4()
    volume_template.GetIJKToRASDirectionMatrix(direction_matrix)
    volume_node.SetIJKToRASDirectionMatrix(direction_matrix)
    
    # Update volume from array
    slicer.util.updateVolumeFromArray(volume_node, volume_array)
    
    # Save the volume
    save_path = os.path.join(output_dir, "bolt_trajectory_lines.nrrd")
    slicer.util.saveNode(volume_node, save_path)
    print(f"✅ Saved bolt trajectory lines volume to {save_path}")
    
    return volume_node

def _bresenham_line_3d(x0, y0, z0, x1, y1, z1):
    """
    Implementation of 3D Bresenham's line algorithm to create a line between two points in a 3D volume.
    Returns a list of points (voxel coordinates) along the line.
    
    Args:
        x0, y0, z0: Start point coordinates
        x1, y1, z1: End point coordinates
        
    Returns:
        list: List of tuples containing voxel coordinates along the line
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    sz = 1 if z1 > z0 else -1
    
    if dx >= dy and dx >= dz:
        err_y = dx // 2
        err_z = dx // 2
        
        x, y, z = x0, y0, z0
        while x != x1:
            points.append((x, y, z))
            err_y -= dy
            if err_y < 0:
                y += sy
                err_y += dx
            
            err_z -= dz
            if err_z < 0:
                z += sz
                err_z += dx
            
            x += sx
        
    elif dy >= dx and dy >= dz:
        err_x = dy // 2
        err_z = dy // 2
        
        x, y, z = x0, y0, z0
        while y != y1:
            points.append((x, y, z))
            err_x -= dx
            if err_x < 0:
                x += sx
                err_x += dy
            
            err_z -= dz
            if err_z < 0:
                z += sz
                err_z += dy
            
            y += sy
    
    else:
        err_x = dz // 2
        err_y = dz // 2
        
        x, y, z = x0, y0, z0
        while z != z1:
            points.append((x, y, z))
            err_x -= dx
            if err_x < 0:
                x += sx
                err_x += dz
            
            err_y -= dy
            if err_y < 0:
                y += sy
                err_y += dz
            
            z += sz
    
    # Add the last point
    points.append((x1, y1, z1))
    
    # Ensure we don't have duplicate points
    return list(dict.fromkeys(map(tuple, points)))

def verify_directions_with_brain(directions, brain_volume):
    """
    Verify that bolt entry directions point toward the brain and validate entry angles.
    Added validation for entry angles relative to the brain surface normal (30-60 degrees is ideal).
    
    Args:
        directions: Dictionary of direction information
        brain_volume: Slicer volume node containing brain mask
    """
    # Calculate brain centroid
    brain_array = get_array_from_volume(brain_volume)
    if brain_array is None or np.sum(brain_array) == 0:
        print("No brain volume data found.")
        return
    
    brain_coords = np.argwhere(brain_array > 0)
    brain_centroid_ijk = np.mean(brain_coords, axis=0)
    brain_centroid = get_ras_coordinates_from_ijk(brain_volume, [
        brain_centroid_ijk[2], brain_centroid_ijk[1], brain_centroid_ijk[0]
    ])
    
    print(f"Brain centroid: {brain_centroid}")
    
    # Extract brain surface
    vertices, _ = get_surface_from_volume(brain_volume)
    if len(vertices) == 0:
        print("Could not extract brain surface.")
        return
        
    brain_surface_points = convert_surface_vertices_to_ras(brain_volume, vertices)
    
    # For each direction, check that it points toward the brain
    for bolt_id, bolt_info in directions.items():
        # Ensure we're working with numpy arrays, not lists
        bolt_point = np.array(bolt_info['start_point'])
        entry_point = np.array(bolt_info['end_point'])
        
        # Handle direction which might be a list
        if isinstance(bolt_info['direction'], list):
            current_direction = np.array(bolt_info['direction'])
        else:
            current_direction = bolt_info['direction']
        
        # Check 1: Direction to brain centroid
        to_brain_center = np.array(brain_centroid) - bolt_point
        to_brain_center = to_brain_center / np.linalg.norm(to_brain_center)
        
        # Dot product between current direction and direction to brain center
        # Higher values mean more aligned directions
        brain_alignment = np.dot(current_direction, to_brain_center)
        
        # Check 2: Compare distances to brain surface
        bolt_to_surface = cdist([bolt_point], brain_surface_points).min()
        entry_to_surface = cdist([entry_point], brain_surface_points).min()
        
        # NEW - Check 3: Calculate angle relative to surface normal at entry point
        # First find the closest point on the brain surface to the entry point
        closest_idx = np.argmin(cdist([entry_point], brain_surface_points))
        closest_surface_point = brain_surface_points[closest_idx]
        
        # Estimate the surface normal at this point (pointing outward from brain)
        # Use k-nearest neighbors to estimate a local plane, then get normal
        k = min(20, len(brain_surface_points))  # Use up to 20 nearest neighbors
        dists = cdist([closest_surface_point], brain_surface_points)[0]
        nearest_idxs = np.argsort(dists)[:k]
        nearest_points = brain_surface_points[nearest_idxs]
        
        # Use PCA to estimate the local surface plane
        pca = PCA(n_components=3)
        pca.fit(nearest_points)
        
        # The third component (least variance) approximates the surface normal
        surface_normal = pca.components_[2]
        
        # Make sure the normal points outward from the brain (away from centroid)
        to_centroid = brain_centroid - closest_surface_point
        if np.dot(surface_normal, to_centroid) > 0:
            surface_normal = -surface_normal
        
        # Calculate angle between trajectory direction and surface normal
        dot_product = np.dot(current_direction, surface_normal)
        angle_with_normal = np.degrees(np.arccos(np.clip(np.abs(dot_product), -1.0, 1.0)))
        
        # Angle with surface is complementary to angle with normal
        angle_with_surface = 90 - angle_with_normal
        
        # Check if angle is in the ideal surgical range (30-60 degrees)
        is_angle_valid = 30 <= angle_with_surface <= 60
        
        print(f"Bolt {bolt_id}: Brain alignment: {brain_alignment:.2f}, "
              f"Bolt-to-surface: {bolt_to_surface:.2f}mm, "
              f"Entry-to-surface: {entry_to_surface:.2f}mm, "
              f"Angle with surface: {angle_with_surface:.2f}° ({'VALID' if is_angle_valid else 'INVALID'})")
        
        # Entry point should be closer to brain surface than bolt point
        if entry_to_surface > bolt_to_surface:
            print(f"Warning: Bolt {bolt_id} - Entry point ({entry_to_surface:.2f}mm) is "
                  f"farther from brain surface than bolt point ({bolt_to_surface:.2f}mm)")
        
        # Direction should roughly point toward brain
        if brain_alignment < 0.5:  # Less than 60° angle
            print(f"Warning: Bolt {bolt_id} - Direction may not be pointing toward brain "
                  f"(alignment: {brain_alignment:.2f})")
        
        # Add validation metrics to direction info
        bolt_info['brain_alignment'] = float(brain_alignment)
        bolt_info['bolt_to_surface_dist'] = float(bolt_to_surface)
        bolt_info['entry_to_surface_dist'] = float(entry_to_surface)
        bolt_info['angle_with_surface'] = float(angle_with_surface)
        bolt_info['is_angle_valid'] = bool(is_angle_valid)

#visualization
def visualize_entry_angle_validation(bolt_directions, brain_volume, output_dir=None):
    """
    Create visualization showing the validation of entry angles relative to brain surface.
    
    Args:
        bolt_directions (dict): Direction info from extract_bolt_entry_directions
        brain_volume: Brain volume to get surface points
        output_dir (str, optional): Directory to save visualization
        
    Returns:
        matplotlib.figure.Figure: Figure containing validation visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Entry Angle Validation (Ideal: 30-60° with surface)', fontsize=16)
    
    # Create grid layout
    gs = GridSpec(2, 2, figure=fig)
    
    # Extract brain surface points for 3D visualization
    vertices, _ = get_surface_from_volume(brain_volume)
    brain_surface_points = convert_surface_vertices_to_ras(brain_volume, vertices)
    
    # Downsample surface points for better visualization
    if len(brain_surface_points) > 5000:
        step = len(brain_surface_points) // 5000
        brain_surface_points = brain_surface_points[::step]
    
    # 3D visualization with brain and trajectories
    ax1 = fig.add_subplot(gs[0, :], projection='3d')
    
    # Plot brain surface
    ax1.scatter(brain_surface_points[:, 0], brain_surface_points[:, 1], brain_surface_points[:, 2],
                c='lightgray', s=1, alpha=0.3, label='Brain Surface')
    
    # Plot trajectories with color coding based on angle validity
    valid_count = 0
    total_count = 0
    angles = []
    
    for bolt_id, bolt_info in bolt_directions.items():
        total_count += 1
        is_valid = bolt_info.get('is_angle_valid', False)
        angle = bolt_info.get('angle_with_surface', 0)
        angles.append(angle)
        
        if is_valid:
            valid_count += 1
            color = 'green'
        else:
            color = 'red'
        
        # Plot entry point
        entry_point = np.array(bolt_info['end_point'])
        ax1.scatter(entry_point[0], entry_point[1], entry_point[2],
                   c=color, marker='*', s=100, alpha=0.9)
        
        # Plot bolt point
        bolt_point = np.array(bolt_info['start_point'])
        ax1.scatter(bolt_point[0], bolt_point[1], bolt_point[2],
                   c=color, marker='o', s=50, alpha=0.7)
        
        # Plot trajectory line
        ax1.plot([bolt_point[0], entry_point[0]],
                [bolt_point[1], entry_point[1]],
                [bolt_point[2], entry_point[2]],
                c=color, linewidth=2, alpha=0.8)
        
        # Add label with angle
        midpoint = (bolt_point + entry_point) / 2
        ax1.text(midpoint[0], midpoint[1], midpoint[2],
                f"{bolt_id}: {angle:.1f}°",
                color=color, fontsize=8)
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'Entry Angle Validation: {valid_count}/{total_count} valid ({valid_count/total_count*100:.1f}%)')
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Valid Angle (30-60°)'),
        Line2D([0], [0], color='red', lw=2, label='Invalid Angle')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Histogram of angles
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(angles, bins=20, color='lightblue', edgecolor='black')
    
    # Add vertical lines for valid range
    ax2.axvline(x=30, color='green', linestyle='--', linewidth=2, label='Min Valid (30°)')
    ax2.axvline(x=60, color='green', linestyle='--', linewidth=2, label='Max Valid (60°)')
    
    ax2.set_xlabel('Angle with Surface (°)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Entry Angles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Table with angle data
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    # Create table data
    table_data = []
    for bolt_id, bolt_info in bolt_directions.items():
        angle = bolt_info.get('angle_with_surface', 0)
        is_valid = bolt_info.get('is_angle_valid', False)
        status = 'VALID' if is_valid else 'INVALID'
        table_data.append([bolt_id, f"{angle:.2f}°", status])
    
    # Sort by bolt ID
    table_data.sort(key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0])
    
    # Create table
    table = ax3.table(cellText=table_data, 
                     colLabels=['Bolt ID', 'Angle with Surface', 'Status'],
                     loc='center', cellLoc='center')
    
    # Set table properties
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Color code cells based on status
    for i, row in enumerate(table_data):
        status = row[2]
        cell = table[(i+1, 2)]  # +1 for header row
        if status == 'VALID':
            cell.set_facecolor('lightgreen')
        else:
            cell.set_facecolor('lightcoral')
    
    ax3.set_title('Entry Angle Measurements')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output directory provided
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'entry_angle_validation.png'), dpi=300)
        
        # Also save as PDF
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(os.path.join(output_dir, 'entry_angle_validation.pdf')) as pdf:
            pdf.savefig(fig)
            
        print(f"✅ Entry angle validation saved to {os.path.join(output_dir, 'entry_angle_validation.pdf')}")
    
    return fig


def match_bolt_directions_to_trajectories(bolt_directions, trajectories, max_distance=20, max_angle=40):
    """
    Match bolt+entry directions to electrode trajectories.
    
    Args:
        bolt_directions (dict): Direction info from extract_bolt_entry_directions
        trajectories (list): Trajectories from integrated_trajectory_analysis
        max_distance (float): Maximum distance between bolt start and trajectory endpoint
        max_angle (float): Maximum angle (degrees) between directions
        
    Returns:
        dict: Dictionary mapping trajectory IDs to matched bolt directions
    """
    matches = {}
    
    for traj in trajectories:
        traj_id = traj['cluster_id']
        traj_endpoints = np.array(traj['endpoints'])
        traj_first_contact = traj_endpoints[0]  # Assuming this is the first contact point
        traj_direction = np.array(traj['direction'])
        
        best_match = None
        best_score = float('inf')
        
        for bolt_id, bolt_info in bolt_directions.items():
            bolt_start = bolt_info['start_point']
            bolt_direction = bolt_info['direction']
            
            # Calculate distance between bolt start and trajectory first contact
            distance = np.linalg.norm(bolt_start - traj_first_contact)
            
            # Calculate angle between directions (in degrees)
            cos_angle = np.abs(np.dot(bolt_direction, traj_direction))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure within valid range
            angle = np.degrees(np.arccos(cos_angle))
            
            # If angle is > 90 degrees, directions are opposite, so take 180-angle
            if angle > 90:
                angle = 180 - angle
            
            # Create a weighted score (lower is better)
            score = distance + angle * 2
            
            # Check if this is a valid match and better than current best
            if distance <= max_distance and angle <= max_angle and score < best_score:
                best_match = {
                    'bolt_id': bolt_id,
                    'distance': distance,
                    'angle': angle,
                    'score': score,
                    'bolt_info': bolt_info
                }
                best_score = score
        
        if best_match:
            matches[traj_id] = best_match
    
    return matches

def integrated_trajectory_analysis(coords_array, entry_points=None, max_neighbor_distance=8, min_neighbors=3, 
                                  expected_spacing_range=(3.0, 5.0)):
    """
    Perform integrated trajectory analysis on electrode coordinates.
    
    This function combines DBSCAN clustering, Louvain community detection,
    and PCA-based trajectory analysis to identify and characterize electrode trajectories.
    Added spacing validation for trajectories.
    
    Args:
        coords_array (numpy.ndarray): Array of electrode coordinates (shape: [n_electrodes, 3])
        entry_points (numpy.ndarray, optional): Array of entry point coordinates (shape: [n_entry_points, 3])
        max_neighbor_distance (float): Maximum distance between neighbors for DBSCAN clustering
        min_neighbors (int): Minimum number of neighbors for DBSCAN clustering
        expected_spacing_range (tuple): Expected range of spacing (min, max) in mm
        
    Returns:
        dict: Results dictionary containing clustering, community detection, and trajectory information
    """
    results = {
        'dbscan': {},
        'louvain': {},
        'combined': {},
        'parameters': {
            'max_neighbor_distance': max_neighbor_distance,
            'min_neighbors': min_neighbors,
            'n_electrodes': len(coords_array),
            'expected_spacing_range': expected_spacing_range
        },
        'pca_stats': []
    }
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=max_neighbor_distance, min_samples=min_neighbors)
    clusters = dbscan.fit_predict(coords_array)
    
    unique_clusters = set(clusters)
    results['dbscan']['n_clusters'] = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    results['dbscan']['noise_points'] = np.sum(clusters == -1)
    results['dbscan']['cluster_sizes'] = [np.sum(clusters == c) for c in unique_clusters if c != -1]
    
    # Create graph for Louvain
    G = nx.Graph()
    results['graph'] = G
    
    for i, coord in enumerate(coords_array):
        G.add_node(i, pos=coord, dbscan_cluster=int(clusters[i]))

    # Add edges based on distance
    distances = cdist(coords_array, coords_array)
    for i in range(len(coords_array)):
        for j in range(i + 1, len(coords_array)):
            dist = distances[i,j]
            if dist <= max_neighbor_distance:
                G.add_edge(i, j, weight=1.0 / (dist + 1e-6))  

    # Louvain community detection
    try:
        louvain_partition = nx.community.louvain_communities(G, weight='weight', resolution=1.0)
        modularity = nx.community.modularity(G, louvain_partition, weight='weight')
        
        results['louvain']['n_communities'] = len(louvain_partition)
        results['louvain']['modularity'] = modularity
        results['louvain']['community_sizes'] = [len(c) for c in louvain_partition]
        
        node_to_community = {}
        for comm_id, comm_nodes in enumerate(louvain_partition):
            for node in comm_nodes:
                node_to_community[node] = comm_id
                
        for node in G.nodes:
            G.nodes[node]['louvain_community'] = node_to_community.get(node, -1)
            
    except Exception as e:
        logging.warning(f"Louvain community detection failed: {e}")
        results['louvain']['error'] = str(e)
    
    # Combined analysis (mapping between DBSCAN clusters and Louvain communities)
    if 'error' not in results['louvain']:
        cluster_community_mapping = defaultdict(set)
        for node in G.nodes:
            dbscan_cluster = G.nodes[node]['dbscan_cluster']
            louvain_community = G.nodes[node]['louvain_community']
            if dbscan_cluster != -1:  
                cluster_community_mapping[dbscan_cluster].add(louvain_community)
        
        # Calculate purity scores (how well clusters map to communities)
        purity_scores = []
        for cluster, communities in cluster_community_mapping.items():
            if len(communities) > 0:
                comm_counts = defaultdict(int)
                for node in G.nodes:
                    if G.nodes[node]['dbscan_cluster'] == cluster:
                        comm_counts[G.nodes[node]['louvain_community']] += 1
                
                if comm_counts:
                    max_count = max(comm_counts.values())
                    total = sum(comm_counts.values())
                    purity_scores.append(max_count / total)
        
        results['combined']['avg_cluster_purity'] = np.mean(purity_scores) if purity_scores else 0

        # Map each DBSCAN cluster to its dominant Louvain community
        dbscan_to_louvain = {}
        for cluster in cluster_community_mapping:
            comm_counts = defaultdict(int)
            for node in G.nodes:
                if G.nodes[node]['dbscan_cluster'] == cluster:
                    comm_counts[G.nodes[node]['louvain_community']] += 1
            
            if comm_counts:
                dominant_comm = max(comm_counts.items(), key=lambda x: x[1])[0]
                dbscan_to_louvain[cluster] = dominant_comm
        
        results['combined']['dbscan_to_louvain_mapping'] = dbscan_to_louvain
    
    # Trajectory analysis with enhanced PCA, angle calculations, and spacing validation
    trajectories = []
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue
            
        cluster_mask = clusters == cluster_id
        cluster_coords = coords_array[cluster_mask]
        
        if len(cluster_coords) < 2:
            continue
        
        louvain_community = None
        if 'dbscan_to_louvain_mapping' in results['combined']:
            louvain_community = results['combined']['dbscan_to_louvain_mapping'].get(cluster_id, None)
        
        try:
            # Apply PCA to find the principal direction of the trajectory
            pca = PCA(n_components=3)
            pca.fit(cluster_coords)
            
            # Store PCA statistics for pattern analysis
            pca_stats = {
                'cluster_id': cluster_id,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'components': pca.components_.tolist(),
                'singular_values': pca.singular_values_.tolist(),
                'mean': pca.mean_.tolist()
            }
            results['pca_stats'].append(pca_stats)
            
            linearity = pca.explained_variance_ratio_[0]
            direction = pca.components_[0]
            center = np.mean(cluster_coords, axis=0)
            
            # Calculate angles with principal axes
            angles = calculate_angles(direction)
            
            projected = np.dot(cluster_coords - center, direction)
            
            # Enhanced entry point handling
            start_entry_point = None
            if entry_points is not None:
                min_dist = float('inf')
                for entry in entry_points:
                    dists = cdist([entry], cluster_coords)
                    min_cluster_dist = np.min(dists)
                    if min_cluster_dist < min_dist:
                        min_dist = min_cluster_dist
                        start_entry_point = entry
                
                if start_entry_point is not None:
                    entry_projection = np.dot(start_entry_point - center, direction)
                    sorted_indices = np.argsort(projected)
                    sorted_coords = cluster_coords[sorted_indices]
                    
                    # Ensure direction points from entry to electrodes
                    entry_vector = sorted_coords[0] - start_entry_point
                    if np.dot(entry_vector, direction) < 0:
                        direction = -direction
                        projected = -projected
                        sorted_indices = sorted_indices[::-1]
                        sorted_coords = cluster_coords[sorted_indices]
            else:
                sorted_indices = np.argsort(projected)
                sorted_coords = cluster_coords[sorted_indices]
            
            # Calculate trajectory metrics
            distances = np.linalg.norm(np.diff(sorted_coords, axis=0), axis=1)
            spacing_regularity = np.std(distances) / np.mean(distances) if len(distances) > 1 else np.nan
            trajectory_length = np.sum(distances)
            
            # Add spacing validation
            spacing_validation = None
            if expected_spacing_range:
                spacing_validation = validate_electrode_spacing(sorted_coords, expected_spacing_range)
            
            # Spline fitting
            spline_points = None
            if len(sorted_coords) > 2:
                try:
                    tck, u = splprep(sorted_coords.T, s=0)
                    u_new = np.linspace(0, 1, 50)
                    spline_points = np.array(splev(u_new, tck)).T
                except:
                    pass
            
            trajectory_dict = {
                "cluster_id": int(cluster_id),
                "louvain_community": louvain_community,
                "electrode_count": int(len(cluster_coords)),
                "linearity": float(linearity),
                "direction": direction.tolist(),
                "center": center.tolist(),
                "length_mm": float(trajectory_length),
                "spacing_regularity": float(spacing_regularity) if not np.isnan(spacing_regularity) else None,
                "avg_spacing_mm": float(np.mean(distances)) if len(distances) > 0 else None,
                "endpoints": [sorted_coords[0].tolist(), sorted_coords[-1].tolist()],
                "entry_point": start_entry_point.tolist() if start_entry_point is not None else None,
                "spline_points": spline_points.tolist() if spline_points is not None else None,
                "angles_with_axes": angles,
                "pca_variance": pca.explained_variance_ratio_.tolist()
            }
            
            # Add spacing validation if available
            if spacing_validation:
                trajectory_dict["spacing_validation"] = spacing_validation
            
            trajectories.append(trajectory_dict)
            
        except Exception as e:
            logging.warning(f"PCA failed for cluster {cluster_id}: {e}")
            continue
    
    results['trajectories'] = trajectories
    results['n_trajectories'] = len(trajectories)
    
    # Calculate overall spacing validation statistics
    if expected_spacing_range and trajectories:
        all_spacings = []
        valid_trajectories = 0
        
        for traj in trajectories:
            if 'spacing_validation' in traj and 'distances' in traj['spacing_validation']:
                all_spacings.extend(traj['spacing_validation']['distances'])
                if traj['spacing_validation'].get('is_valid', False):
                    valid_trajectories += 1
        
        results['spacing_validation_summary'] = {
            'total_trajectories': len(trajectories),
            'valid_trajectories': valid_trajectories,
            'valid_percentage': (valid_trajectories / len(trajectories) * 100) if trajectories else 0,
            'all_spacings': all_spacings,
            'mean_spacing': np.mean(all_spacings) if all_spacings else np.nan,
            'min_spacing': np.min(all_spacings) if all_spacings else np.nan,
            'max_spacing': np.max(all_spacings) if all_spacings else np.nan,
            'std_spacing': np.std(all_spacings) if all_spacings else np.nan,
            'expected_spacing_range': expected_spacing_range
        }
    
    # Add noise points information
    noise_mask = clusters == -1
    results['dbscan']['noise_points_coords'] = coords_array[noise_mask].tolist()
    results['dbscan']['noise_points_indices'] = np.where(noise_mask)[0].tolist()

    # Add to the trajectory dictionary - new field for entry angle validation
    for traj in trajectories:
        # Initialize entry angle fields
        traj["entry_angle_validation"] = {
            "angle_with_surface": None,
            "is_valid": None,
            "status": "unknown"
        }
        
        # If we have an entry point, calculate surface angle
        if traj['entry_point'] is not None and 'brain_surface_points' in results:
            entry_point = np.array(traj['entry_point'])
            direction = np.array(traj['direction'])
            
            # Get closest surface point
            surface_points = results['brain_surface_points']
            if len(surface_points) > 0:
                closest_idx = np.argmin(cdist([entry_point], surface_points))
                closest_surface_point = surface_points[closest_idx]
                
                # Estimate the surface normal (as in verify_directions_with_brain)
                k = min(20, len(surface_points))
                dists = cdist([closest_surface_point], surface_points)[0]
                nearest_idxs = np.argsort(dists)[:k]
                nearest_points = surface_points[nearest_idxs]
                
                pca = PCA(n_components=3)
                pca.fit(nearest_points)
                surface_normal = pca.components_[2]
                
                # Make sure normal points outward
                brain_centroid = results.get('brain_centroid')
                if brain_centroid is not None:
                    to_centroid = brain_centroid - closest_surface_point
                    if np.dot(surface_normal, to_centroid) > 0:
                        surface_normal = -surface_normal
                
                # Calculate angle
                dot_product = np.dot(direction, surface_normal)
                angle_with_normal = np.degrees(np.arccos(np.clip(np.abs(dot_product), -1.0, 1.0)))
                angle_with_surface = 90 - angle_with_normal
                
                # Validate angle
                is_valid = 30 <= angle_with_surface <= 60
                
                traj["entry_angle_validation"] = {
                    "angle_with_surface": float(angle_with_surface),
                    "is_valid": bool(is_valid),
                    "status": "valid" if is_valid else "invalid"
                }
    results['trajectories'] = trajectories
    results['n_trajectories'] = len(trajectories)

    return results

#------------------------------------------------------------------------------
# PART 2.1: Validation paths
#------------------------------------------------------------------------------
def validate_electrode_clusters(results, expected_contact_counts=[5, 8, 10, 12, 15, 18]):
    """
    Validate the identified electrode clusters against expected contact counts.
    
    Args:
        results (dict): Results from integrated_trajectory_analysis
        expected_contact_counts (list): List of expected electrode contact counts
        
    Returns:
        dict: Dictionary with validation results
    """
    validation = {
        'clusters': {},
        'summary': {
            'total_clusters': 0,
            'valid_clusters': 0,
            'invalid_clusters': 0,
            'match_percentage': 0,
            'by_size': {count: 0 for count in expected_contact_counts}
        }
    }
    
    # Get clusters from DBSCAN
    clusters = None
    if 'graph' in results:
        clusters = [node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)]
    
    if not clusters:
        return validation
    
    # Count the number of points in each cluster
    unique_clusters = set(clusters)
    if -1 in unique_clusters:
        unique_clusters.remove(-1)  # Remove noise points
    
    cluster_sizes = {}
    for cluster_id in unique_clusters:
        size = sum(1 for c in clusters if c == cluster_id)
        cluster_sizes[cluster_id] = size
        
    validation['summary']['total_clusters'] = len(cluster_sizes)
    
    # Validate each cluster
    for cluster_id, size in cluster_sizes.items():
        # Find the closest expected size
        closest_size = min(expected_contact_counts, key=lambda x: abs(x - size))
        difference = abs(closest_size - size)
        
        # Determine if this is a valid cluster (exact match or within tolerance)
        is_valid = size in expected_contact_counts
        is_close = difference <= 2  # Allow small deviations (±2)
        
        # Find trajectory info for this cluster if available
        trajectory_info = None
        if 'trajectories' in results:
            for traj in results['trajectories']:
                if traj['cluster_id'] == cluster_id:
                    trajectory_info = traj
                    break
        
        validation['clusters'][cluster_id] = {
            'size': size,
            'closest_expected': closest_size,
            'difference': difference,
            'valid': is_valid,
            'close': is_close,
            'pca_linearity': trajectory_info['linearity'] if trajectory_info else None,
            'electrode_type': f"{closest_size}-contact" if is_close else "Unknown"
        }
        
        # Update summary statistics
        if is_valid:
            validation['summary']['valid_clusters'] += 1
            validation['summary']['by_size'][size] += 1
        elif is_close:
            validation['summary']['by_size'][closest_size] += 1
        else:
            validation['summary']['invalid_clusters'] += 1
    
    # Calculate match percentage
    total = validation['summary']['total_clusters']
    if total > 0:
        valid = validation['summary']['valid_clusters']
        close = sum(1 for c in validation['clusters'].values() if c['close'] and not c['valid'])
        validation['summary']['match_percentage'] = (valid / total) * 100
        validation['summary']['close_percentage'] = ((valid + close) / total) * 100
    
    return validation

def create_electrode_validation_page(results, validation):
    """
    Create a visualization page for electrode cluster validation.
    
    Args:
        results (dict): Results from integrated_trajectory_analysis
        validation (dict): Results from validate_electrode_clusters
        
    Returns:
        matplotlib.figure.Figure: Figure containing validation results
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np
    
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle('Electrode Cluster Validation', fontsize=16)
    
    # Create grid layout
    gs = GridSpec(2, 2, figure=fig)
    
    # Summary statistics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    summary = validation['summary']
    
    # Create summary table
    summary_data = []
    summary_columns = [
        'Total Clusters', 
        'Valid Clusters', 
        'Close Clusters',
        'Invalid Clusters', 
        'Match %'
    ]
    
    close_clusters = sum(1 for c in validation['clusters'].values() 
                        if c['close'] and not c['valid'])
    
    summary_data.append([
        str(summary['total_clusters']),
        f"{summary['valid_clusters']} ({summary['match_percentage']:.1f}%)",
        str(close_clusters),
        str(summary['invalid_clusters']),
        f"{summary.get('close_percentage', 0):.1f}%"
    ])
    
    summary_table = ax1.table(cellText=summary_data, colLabels=summary_columns,
                             loc='center', cellLoc='center')
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(10)
    summary_table.scale(1, 1.5)
    ax1.set_title('Validation Summary')
    
    # Distribution by expected size
    ax2 = fig.add_subplot(gs[0, 1])
    expected_sizes = sorted(summary['by_size'].keys())
    counts = [summary['by_size'][size] for size in expected_sizes]
    
    bars = ax2.bar(expected_sizes, counts)
    ax2.set_xlabel('Number of Contacts')
    ax2.set_ylabel('Number of Clusters')
    ax2.set_title('Electrode Distribution by Contact Count')
    ax2.set_xticks(expected_sizes)
    
    # Add count labels above bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom')
    
    # Detailed cluster validation table
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    # Create detailed validation table
    detail_data = []
    detail_columns = [
        'Cluster ID', 
        'Size', 
        'Expected Size', 
        'Difference', 
        'Valid', 
        'Close',
        'Linearity',
        'Electrode Type'
    ]
    
    for cluster_id, cluster_info in validation['clusters'].items():
        row = [
            cluster_id,
            cluster_info['size'],
            cluster_info['closest_expected'],
            cluster_info['difference'],
            "Yes" if cluster_info['valid'] else "No",
            "Yes" if cluster_info['close'] else "No",
            f"{cluster_info['pca_linearity']:.3f}" if cluster_info['pca_linearity'] is not None else "N/A",
            cluster_info['electrode_type']
        ]
        detail_data.append(row)
    
    # Sort by cluster ID
    detail_data.sort(key=lambda x: int(x[0]) if isinstance(x[0], (int, str)) and str(x[0]).isdigit() else x[0])
    
    detail_table = ax3.table(cellText=detail_data, colLabels=detail_columns,
                           loc='center', cellLoc='center')
    detail_table.auto_set_font_size(False)
    detail_table.set_fontsize(10)
    detail_table.scale(1, 1.5)
    ax3.set_title('Detailed Cluster Validation')
    
    plt.tight_layout()
    return fig

def enhance_integrated_trajectory_analysis(coords_array, entry_points=None, max_neighbor_distance=10, 
                                          min_neighbors=3, expected_contact_counts=[5, 8, 10, 12, 15, 18],
                                          expected_spacing_range=(3.0, 5.0)):
    """
    Enhanced version of integrated_trajectory_analysis with electrode validation.
    
    This function extends the original integrated_trajectory_analysis by adding
    validation against expected electrode contact counts and spacing validation.
    
    Args:
        coords_array (numpy.ndarray): Array of electrode coordinates
        entry_points (numpy.ndarray, optional): Array of entry point coordinates
        max_neighbor_distance (float): Maximum distance between neighbors for DBSCAN
        min_neighbors (int): Minimum number of neighbors for DBSCAN
        expected_contact_counts (list): List of expected electrode contact counts
        expected_spacing_range (tuple): Expected range for contact spacing (min, max) in mm
        
    Returns:
        dict: Results dictionary with added validation information
    """
    # Run the original analysis with spacing validation
    results = integrated_trajectory_analysis(
        coords_array=coords_array,
        entry_points=entry_points,
        max_neighbor_distance=max_neighbor_distance,
        min_neighbors=min_neighbors,
        expected_spacing_range=expected_spacing_range
    )
    
    # Add validation
    validation = validate_electrode_clusters(results, expected_contact_counts)
    results['electrode_validation'] = validation
    
    # Create validation visualization and add to results
    if 'figures' not in results:
        results['figures'] = {}
    
    results['figures']['electrode_validation'] = create_electrode_validation_page(results, validation)
    
    return results

#------------------------------------------------------------------------------
# PART 2.2: MATCHING TRAJECTORIES TO BOLT DIRECTIONS
#-------------------------------------------------------------------------------
def compare_trajectories_with_combined_data(integrated_results, combined_trajectories):
    """
    Compare trajectories detected through clustering with those from the combined volume.
    This function doesn't use the comparison for validation but provides statistics
    on the matching between the two methods.
    
    Args:
        integrated_results (dict): Results from integrated_trajectory_analysis
        combined_trajectories (dict): Trajectories extracted from combined mask
        
    Returns:
        dict: Comparison statistics and matching information
    """
    comparison = {
        'summary': {
            'integrated_trajectories': 0,
            'combined_trajectories': 0,
            'matching_trajectories': 0,
            'matching_percentage': 0,
            'spatial_alignment_stats': {}
        },
        'matches': {},
        'unmatched_integrated': [],
        'unmatched_combined': []
    }
    
    # Get trajectories from integrated analysis
    integrated_trajectories = integrated_results.get('trajectories', [])
    comparison['summary']['integrated_trajectories'] = len(integrated_trajectories)
    comparison['summary']['combined_trajectories'] = len(combined_trajectories)
    
    if not integrated_trajectories or not combined_trajectories:
        return comparison
    
    # For each integrated trajectory, find potential matches in combined trajectories
    for traj in integrated_trajectories:
        traj_id = traj['cluster_id']
        traj_endpoints = np.array(traj['endpoints'])
        traj_direction = np.array(traj['direction'])
        
        best_match = None
        best_score = float('inf')
        
        # Compare with each combined trajectory
        for bolt_id, combined_traj in combined_trajectories.items():
            combined_start = np.array(combined_traj['start_point'])
            combined_end = np.array(combined_traj['end_point'])
            combined_direction = np.array(combined_traj['direction'])
            
            # Calculate distance between endpoints
            # Find the closest pair of endpoints
            distances = [
                np.linalg.norm(traj_endpoints[0] - combined_start),
                np.linalg.norm(traj_endpoints[0] - combined_end),
                np.linalg.norm(traj_endpoints[1] - combined_start),
                np.linalg.norm(traj_endpoints[1] - combined_end)
            ]
            min_distance = min(distances)
            
            # Calculate angle between directions
            cos_angle = np.abs(np.dot(traj_direction, combined_direction))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            # If angle is > 90 degrees, consider opposite directions
            if angle > 90:
                angle = 180 - angle
            
            # Create a weighted score (lower is better)
            score = min_distance + angle * 2
            
            # Check if this is a better match than current best
            if score < best_score:
                best_match = {
                    'bolt_id': bolt_id,
                    'min_distance': min_distance,
                    'angle': angle,
                    'score': score,
                    'combined_traj': combined_traj
                }
                best_score = score
        
        # Use a threshold to determine if it's a valid match
        if best_match and best_match['score'] < 30:  # Adjustable threshold
            comparison['matches'][traj_id] = best_match
        else:
            comparison['unmatched_integrated'].append(traj_id)
    
    # Find unmatched combined trajectories
    matched_bolt_ids = {match['bolt_id'] for match in comparison['matches'].values()}
    comparison['unmatched_combined'] = [
        bolt_id for bolt_id in combined_trajectories.keys() 
        if bolt_id not in matched_bolt_ids
    ]
    
    # Calculate summary statistics
    matching_count = len(comparison['matches'])
    comparison['summary']['matching_trajectories'] = matching_count
    
    if comparison['summary']['integrated_trajectories'] > 0:
        comparison['summary']['matching_percentage'] = (
            matching_count / comparison['summary']['integrated_trajectories'] * 100
        )
    
    # Calculate spatial alignment statistics if there are matches
    if matching_count > 0:
        distances = [match['min_distance'] for match in comparison['matches'].values()]
        angles = [match['angle'] for match in comparison['matches'].values()]
        
        comparison['summary']['spatial_alignment_stats'] = {
            'min_distance': {
                'mean': np.mean(distances),
                'median': np.median(distances),
                'std': np.std(distances),
                'min': min(distances),
                'max': max(distances)
            },
            'angle': {
                'mean': np.mean(angles),
                'median': np.median(angles),
                'std': np.std(angles),
                'min': min(angles),
                'max': max(angles)
            }
        }
    
    return comparison

#------------------------------------------------------------------------------
# PART 2.3: Dealing with duplicates points of contacts 
#------------------------------------------------------------------------------
def identify_potential_duplicates(centroids, threshold=0.5):
    """
    Identify potential duplicate centroids that are within threshold distance of each other.
    
    Args:
        centroids: List or array of centroid coordinates [(x1,y1,z1), (x2,y2,z2), ...]
        threshold: Distance threshold in mm for considering centroids as duplicates
        
    Returns:
        A dictionary with:
        - 'all_centroids': Original list of centroids
        - 'potential_duplicates': List of tuples (i, j) where centroids[i] and centroids[j] are potential duplicates
        - 'duplicate_groups': List of lists, where each inner list contains indices of centroids in a duplicate group
        - 'stats': Basic statistics about the potential duplicates
    """
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    import matplotlib.pyplot as plt
    
    # Convert input to numpy array if it's not already
    centroids_array = np.array(centroids)
    
    # Calculate pairwise distances between all centroids
    distances = squareform(pdist(centroids_array))
    
    # Find pairs of centroids closer than the threshold (excluding self-comparisons)
    potential_duplicate_pairs = []
    for i in range(len(centroids_array)):
        for j in range(i+1, len(centroids_array)):
            if distances[i,j] < threshold:
                potential_duplicate_pairs.append((i, j, distances[i,j]))
    
    # Group duplicates that form clusters
    duplicate_groups = []
    used_indices = set()
    
    for i, j, _ in potential_duplicate_pairs:
        # Check if either index is already in a group
        found_group = False
        for group in duplicate_groups:
            if i in group or j in group:
                # Add both to this group if not already present
                if i not in group:
                    group.append(i)
                if j not in group:
                    group.append(j)
                found_group = True
                break
        
        if not found_group:
            # Create a new group
            duplicate_groups.append([i, j])
        
        used_indices.add(i)
        used_indices.add(j)
    
    # Create statistics
    stats = {
        'total_centroids': len(centroids_array),
        'potential_duplicate_pairs': len(potential_duplicate_pairs),
        'duplicate_groups': len(duplicate_groups),
        'centroids_in_duplicates': len(used_indices),
        'min_duplicate_distance': min([d for _, _, d in potential_duplicate_pairs]) if potential_duplicate_pairs else None,
        'max_duplicate_distance': max([d for _, _, d in potential_duplicate_pairs]) if potential_duplicate_pairs else None,
        'avg_duplicate_distance': np.mean([d for _, _, d in potential_duplicate_pairs]) if potential_duplicate_pairs else None
    }
    
    # Create a simple visualization of distances
    if len(potential_duplicate_pairs) > 0:
        plt.figure(figsize=(10, 6))
        duplicate_distances = [d for _, _, d in potential_duplicate_pairs]
        plt.hist(duplicate_distances, bins=20)
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold}mm)')
        plt.xlabel('Distance between potential duplicate centroids (mm)')
        plt.ylabel('Count')
        plt.title('Distribution of distances between potential duplicate centroids')
        plt.legend()
        plt.grid(True, alpha=0.3)
        stats['distance_histogram'] = plt.gcf()
        plt.close()
    
    return {
        'all_centroids': centroids_array,
        'potential_duplicate_pairs': potential_duplicate_pairs,
        'duplicate_groups': duplicate_groups,
        'stats': stats
    }

def visualize_potential_duplicates(centroids, duplicate_result, trajectory_direction=None):
    """
    Visualize the centroids with potential duplicates highlighted.
    
    Args:
        centroids: Original list or array of centroid coordinates
        duplicate_result: Result dictionary from identify_potential_duplicates function
        trajectory_direction: Optional trajectory direction vector for sorting points
        
    Returns:
        matplotlib figure with visualization
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    centroids_array = np.array(centroids)
    
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sort centroids along trajectory if direction is provided
    sorted_indices = None
    if trajectory_direction is not None:
        center = np.mean(centroids_array, axis=0)
        projected = np.dot(centroids_array - center, trajectory_direction)
        sorted_indices = np.argsort(projected)
        centroids_array = centroids_array[sorted_indices]
    
    # Plot all centroids
    ax.scatter(centroids_array[:, 0], centroids_array[:, 1], centroids_array[:, 2], 
              c='blue', marker='o', s=50, alpha=0.7, label='All centroids')
    
    # Mark centroids that are in duplicate groups
    duplicate_groups = duplicate_result['duplicate_groups']
    
    if sorted_indices is not None:
        # Convert original indices to sorted indices
        sorted_idx_map = {original: sorted_i for sorted_i, original in enumerate(sorted_indices)}
        converted_groups = []
        for group in duplicate_groups:
            converted_groups.append([sorted_idx_map[idx] for idx in group])
        duplicate_groups = converted_groups
    
    # Plot each duplicate group with a different color
    colors = plt.cm.tab10(np.linspace(0, 1, len(duplicate_groups)))
    
    for i, group in enumerate(duplicate_groups):
        group_centroids = centroids_array[group]
        color = colors[i % len(colors)]
        
        # Plot the group
        ax.scatter(group_centroids[:, 0], group_centroids[:, 1], group_centroids[:, 2],
                  c=[color], marker='*', s=150, label=f'Duplicate group {i+1}')
        
        # Connect duplicate points with lines
        for idx1 in range(len(group)):
            for idx2 in range(idx1+1, len(group)):
                ax.plot([group_centroids[idx1, 0], group_centroids[idx2, 0]],
                       [group_centroids[idx1, 1], group_centroids[idx2, 1]],
                       [group_centroids[idx1, 2], group_centroids[idx2, 2]],
                       c=color, linestyle='--', alpha=0.7)
    
    # If we have a trajectory direction, draw the main trajectory line
    if trajectory_direction is not None:
        # Extend the line a bit beyond the endpoints
        min_proj = np.dot(centroids_array[0] - np.mean(centroids_array, axis=0), trajectory_direction)
        max_proj = np.dot(centroids_array[-1] - np.mean(centroids_array, axis=0), trajectory_direction)
        
        # Extend by 10% on each end
        extension = (max_proj - min_proj) * 0.1
        min_proj -= extension
        max_proj += extension
        
        center = np.mean(centroids_array, axis=0)
        start_point = center + trajectory_direction * min_proj
        end_point = center + trajectory_direction * max_proj
        
        ax.plot([start_point[0], end_point[0]],
               [start_point[1], end_point[1]],
               [start_point[2], end_point[2]],
               c='red', linestyle='-', linewidth=2, alpha=0.7, label='Trajectory')
    
    # Add labels and legend
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Potential Duplicate Centroids (threshold = 0.5mm)')
    
    # Add text with statistics
    stats = duplicate_result['stats']
    stat_text = (
        f"Total centroids: {stats['total_centroids']}\n"
        f"Duplicate pairs: {stats['potential_duplicate_pairs']}\n"
        f"Duplicate groups: {stats['duplicate_groups']}\n"
        f"Centroids in duplicates: {stats['centroids_in_duplicates']}"
    )
    if stats['min_duplicate_distance'] is not None:
        stat_text += f"\nDuplicate distances: {stats['min_duplicate_distance']:.2f}-{stats['max_duplicate_distance']:.2f}mm"
    
    ax.text2D(0.05, 0.95, stat_text, transform=ax.transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.legend()
    plt.tight_layout()
    
    return fig

def analyze_duplicates_on_trajectory(centroids, expected_count, threshold=0.5):
    """
    Analyze potential duplicate centroids on a single electrode trajectory.
    
    Args:
        centroids: List or array of centroid coordinates for a single trajectory
        expected_count: Expected number of contacts for this electrode
        threshold: Distance threshold in mm for considering centroids as duplicates
        
    Returns:
        Dictionary with analysis results and visualizations
    """
    import numpy as np
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    centroids_array = np.array(centroids)
    
    # Get trajectory direction using PCA
    pca = PCA(n_components=3)
    pca.fit(centroids_array)
    direction = pca.components_[0]
    
    # Identify potential duplicates
    duplicate_result = identify_potential_duplicates(centroids_array, threshold=threshold)
    
    # Create visualizations
    vis_fig = visualize_potential_duplicates(centroids_array, duplicate_result, trajectory_direction=direction)
    
    # Provide recommendations based on the analysis
    stats = duplicate_result['stats']
    total = stats['total_centroids']
    in_duplicates = stats['centroids_in_duplicates']
    
    recommendations = []
    
    if total > expected_count:
        excess = total - expected_count
        if in_duplicates >= excess:
            recommendations.append(f"Found {excess} excess centroids. Can remove from identified {in_duplicates} potential duplicate centroids.")
        else:
            recommendations.append(f"Found {excess} excess centroids but only {in_duplicates} in duplicate groups. May need additional criteria to remove {excess - in_duplicates} more centroids.")
    elif total == expected_count:
        if in_duplicates > 0:
            recommendations.append(f"Centroid count matches expected count ({expected_count}), but found {in_duplicates} centroids in potential duplicate groups. Consider reviewing trajectory for noise.")
    else:
        recommendations.append(f"Found fewer centroids ({total}) than expected ({expected_count}). Missing {expected_count - total} contact centroids.")
    
    # For each duplicate group, recommend which one to keep
    duplicate_groups = duplicate_result['duplicate_groups']
    
    if duplicate_groups:
        # Project centroids onto trajectory
        center = np.mean(centroids_array, axis=0)
        projected = np.dot(centroids_array - center, direction)
        
        for i, group in enumerate(duplicate_groups):
            group_centroids = centroids_array[group]
            
            # Check if these points create irregular spacing
            if len(group) > 1:
                # Sort by projection along trajectory
                group_projected = projected[group]
                sorted_indices = np.argsort(group_projected)
                sorted_group = [group[i] for i in sorted_indices]
                
                # Calculate distances to adjacent non-duplicate centroids
                group_recommendations = []
                
                for j, idx in enumerate(sorted_group):
                    # Find nearest non-duplicate centroids before and after
                    before_centroids = [k for k in range(len(centroids_array)) if k not in group and projected[k] < projected[idx]]
                    after_centroids = [k for k in range(len(centroids_array)) if k not in group and projected[k] > projected[idx]]
                    
                    before_idx = max(before_centroids, key=lambda k: projected[k]) if before_centroids else None
                    after_idx = min(after_centroids, key=lambda k: projected[k]) if after_centroids else None
                    
                    # Compute spacings
                    before_spacing = np.linalg.norm(centroids_array[before_idx] - centroids_array[idx]) if before_idx is not None else None
                    after_spacing = np.linalg.norm(centroids_array[after_idx] - centroids_array[idx]) if after_idx is not None else None
                    
                    group_recommendations.append({
                        'centroid_idx': idx,
                        'position_in_group': j+1,
                        'spacing_before': before_spacing,
                        'spacing_after': after_spacing,
                        'score': (before_spacing if before_spacing is not None else 0) + 
                                (after_spacing if after_spacing is not None else 0)
                    })
                
                # Determine which centroids might be best to keep/remove based on spacing
                if group_recommendations:
                    # Sort by score (higher score = more regular spacing)
                    sorted_recommendations = sorted(group_recommendations, key=lambda x: x['score'], reverse=True)
                    
                    keep_idx = sorted_recommendations[0]['centroid_idx']
                    keep_info = sorted_recommendations[0]
                    
                    recommendations.append(f"Duplicate group {i+1}: Recommend keeping centroid {keep_idx} "
                                         f"(position {keep_info['position_in_group']}/{len(group)} in group) "
                                         f"for more regular spacing.")
    
    return {
        'centroids': centroids_array,
        'duplicate_result': duplicate_result,
        'expected_count': expected_count,
        'actual_count': stats['total_centroids'],
        'excess_count': stats['total_centroids'] - expected_count,
        'recommendations': recommendations,
        'visualization': vis_fig,
        'distance_histogram': duplicate_result['stats'].get('distance_histogram')
    }

def analyze_all_trajectories(results, coords_array, expected_contact_counts=[5, 8, 10, 12, 15, 18], threshold=0.5):
    """
    Analyze all trajectories for potential duplicate centroids.
    
    Args:
        results: Results from integrated_trajectory_analysis
        coords_array: Array of all electrode coordinates
        expected_contact_counts: List of expected electrode contact counts
        threshold: Distance threshold for considering centroids as duplicates
        
    Returns:
        Dictionary mapping trajectory IDs to duplicate analysis results
    """
    # Get all trajectory IDs from DBSCAN clustering
    clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
    unique_clusters = set(clusters)
    
    if -1 in unique_clusters:
        unique_clusters.remove(-1)  # Remove noise points
    
    # Analyze each trajectory
    all_analyses = {}
    
    for trajectory_idx in unique_clusters:
        # Get centroids for this trajectory
        mask = clusters == trajectory_idx
        trajectory_centroids = coords_array[mask]
        
        # Get expected count for this electrode type
        expected_count = None
        if 'electrode_validation' in results and 'clusters' in results['electrode_validation']:
            if trajectory_idx in results['electrode_validation']['clusters']:
                cluster_info = results['electrode_validation']['clusters'][trajectory_idx]
                if cluster_info['close']:
                    expected_count = cluster_info['closest_expected']
        
        if expected_count is None:
            # If no validation info, use most common electrode type or default
            expected_count = 8  # Default expected count
        
        # Analyze duplicates
        print(f"Analyzing trajectory {trajectory_idx} (expected contacts: {expected_count})...")
        analysis = analyze_duplicates_on_trajectory(trajectory_centroids, expected_count, threshold)
        all_analyses[trajectory_idx] = analysis
        
        # Print brief summary for this trajectory
        duplicate_groups = analysis['duplicate_result']['duplicate_groups']
        if duplicate_groups:
            print(f"- Found {len(duplicate_groups)} duplicate groups with {analysis['duplicate_result']['stats']['centroids_in_duplicates']} centroids")
            if analysis['excess_count'] > 0:
                print(f"- Excess centroids: {analysis['excess_count']} (expected: {expected_count}, actual: {analysis['actual_count']})")
            for rec in analysis['recommendations']:
                print(f"  • {rec}")
        else:
            print(f"- No duplicates found. Centroids: {analysis['actual_count']}, Expected: {expected_count}")
    
    return all_analyses

def create_duplicate_analysis_report(duplicate_analyses, output_dir):
    """
    Create a PDF report of duplicate centroid analysis results.
    
    Args:
        duplicate_analyses: Dictionary mapping trajectory IDs to duplicate analysis results
        output_dir: Directory to save the report
    """
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    pdf_path = os.path.join(output_dir, 'duplicate_centroid_analysis.pdf')
    
    with PdfPages(pdf_path) as pdf:
        # Create summary page
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle('Duplicate Centroid Analysis Summary', fontsize=16)
        
        # Summary statistics
        trajectories_with_duplicates = sum(1 for a in duplicate_analyses.values() 
                                        if a['duplicate_result']['duplicate_groups'])
        total_duplicate_groups = sum(len(a['duplicate_result']['duplicate_groups']) 
                                    for a in duplicate_analyses.values())
        total_centroids = sum(a['actual_count'] for a in duplicate_analyses.values())
        total_in_duplicates = sum(a['duplicate_result']['stats']['centroids_in_duplicates'] 
                                for a in duplicate_analyses.values())
        
        # Create a pie chart of trajectories with/without duplicates
        ax1 = fig.add_subplot(221)
        labels = ['With duplicates', 'Without duplicates']
        sizes = [trajectories_with_duplicates, len(duplicate_analyses) - trajectories_with_duplicates]
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Trajectories with Duplicate Centroids')
        
        # Create a bar chart of duplicate groups by trajectory
        ax2 = fig.add_subplot(222)
        traj_ids = []
        group_counts = []
        
        for traj_id, analysis in duplicate_analyses.items():
            if analysis['duplicate_result']['duplicate_groups']:
                traj_ids.append(traj_id)
                group_counts.append(len(analysis['duplicate_result']['duplicate_groups']))
        
        if traj_ids:
            # Sort by number of duplicate groups
            sorted_indices = np.argsort(group_counts)[::-1]
            sorted_traj_ids = [traj_ids[i] for i in sorted_indices]
            sorted_group_counts = [group_counts[i] for i in sorted_indices]
            
            # Limit to top 10 trajectories
            if len(sorted_traj_ids) > 10:
                sorted_traj_ids = sorted_traj_ids[:10]
                sorted_group_counts = sorted_group_counts[:10]
            
            ax2.bar(range(len(sorted_traj_ids)), sorted_group_counts)
            ax2.set_xticks(range(len(sorted_traj_ids)))
            ax2.set_xticklabels([f"Traj {id}" for id in sorted_traj_ids], rotation=45)
            ax2.set_title('Number of Duplicate Groups by Trajectory')
            ax2.set_xlabel('Trajectory ID')
            ax2.set_ylabel('Number of Duplicate Groups')
        else:
            ax2.text(0.5, 0.5, 'No duplicate groups found', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # Create a table with summary statistics
        ax3 = fig.add_subplot(212)
        ax3.axis('off')
        
        table_data = [
            ['Total trajectories', str(len(duplicate_analyses))],
            ['Trajectories with duplicates', f"{trajectories_with_duplicates} ({trajectories_with_duplicates/len(duplicate_analyses)*100:.1f}%)"],
            ['Total duplicate groups', str(total_duplicate_groups)],
            ['Total centroids', str(total_centroids)],
            ['Centroids in duplicates', f"{total_in_duplicates} ({total_in_duplicates/total_centroids*100:.1f}%)"]
        ]
        
        table = ax3.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # Add visualization for each trajectory with duplicates
        for traj_id, analysis in duplicate_analyses.items():
            if analysis['duplicate_result']['duplicate_groups']:
                # Add the visualization figure
                if 'visualization' in analysis:
                    fig = analysis['visualization']
                    # Add trajectory ID to title
                    ax = fig.axes[0]
                    current_title = ax.get_title()
                    ax.set_title(f"Trajectory {traj_id}: {current_title}")
                    
                    pdf.savefig(fig)
                    plt.close(fig)
                
                # Add the distance histogram if available
                if 'distance_histogram' in analysis and analysis['distance_histogram'] is not None:
                    fig = analysis['distance_histogram']
                    ax = fig.axes[0]
                    current_title = ax.get_title()
                    ax.set_title(f"Trajectory {traj_id}: {current_title}")
                    
                    pdf.savefig(fig)
                    plt.close(fig)
                
                # Create a recommendations page
                if analysis['recommendations']:
                    fig = plt.figure(figsize=(10, 8))
                    fig.suptitle(f'Recommendations for Trajectory {traj_id}', fontsize=16)
                    
                    ax = fig.add_subplot(111)
                    ax.axis('off')
                    
                    text = "\n\n".join([f"{i+1}. {rec}" for i, rec in enumerate(analysis['recommendations'])])
                    ax.text(0.05, 0.95, text, transform=ax.transAxes, 
                           fontsize=12, verticalalignment='top')
                    
                    pdf.savefig(fig)
                    plt.close(fig)
    
    print(f"✅ Duplicate centroid analysis report saved to {pdf_path}")

#------------------------------------------------------------------------------
# PART 2.4: ADAPTIVE CLUSTERING
#------------------------------------------------------------------------------

def adaptive_clustering_parameters(coords_array, initial_eps=8, initial_min_neighbors=3, 
                                   expected_contact_counts=[5, 8, 10, 12, 15, 18],
                                   max_iterations=10, eps_step=0.5, verbose=True):
    """
    Adaptively find optimal eps and min_neighbors parameters for DBSCAN clustering
    of electrode contacts.
    
    Args:
        coords_array (numpy.ndarray): Array of electrode coordinates (shape: [n_electrodes, 3])
        initial_eps (float): Initial value for max neighbor distance (eps) in DBSCAN
        initial_min_neighbors (int): Initial value for min_samples in DBSCAN
        expected_contact_counts (list): List of expected electrode contact counts
        max_iterations (int): Maximum number of iterations to try
        eps_step (float): Step size for adjusting eps
        verbose (bool): Whether to print progress details
        
    Returns:
        dict: Results dictionary with optimal parameters and visualization
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    from matplotlib.gridspec import GridSpec
    from collections import Counter
    
    # Initialize parameters
    current_eps = initial_eps
    current_min_neighbors = initial_min_neighbors
    best_score = 0
    best_params = {'eps': current_eps, 'min_neighbors': current_min_neighbors}
    best_clusters = None
    iterations_data = []
    
    # Function to evaluate clustering quality
    def evaluate_clustering(clusters, n_points):
        # Count points in each cluster (excluding noise points)
        cluster_sizes = Counter([c for c in clusters if c != -1])
        
        # If no clusters found, return 0
        if not cluster_sizes:
            return 0, 0, 0, {}
        
        # Calculate how many clusters have sizes close to expected
        valid_clusters = 0
        cluster_quality = {}
        
        for cluster_id, size in cluster_sizes.items():
            # Find closest expected size
            closest_expected = min(expected_contact_counts, key=lambda x: abs(x - size))
            difference = abs(closest_expected - size)
            
            # Consider valid if exact match or very close (within 2)
            is_valid = size in expected_contact_counts
            is_close = difference <= 2
            
            cluster_quality[cluster_id] = {
                'size': size,
                'closest_expected': closest_expected,
                'difference': difference,
                'valid': is_valid,
                'close': is_close
            }
            
            if is_valid:
                valid_clusters += 1
            
        # Calculate percentage of clustered points (non-noise)
        clustered_percentage = sum(clusters != -1) / n_points * 100
        
        # Calculate percentage of valid clusters
        n_clusters = len(cluster_sizes)
        valid_percentage = (valid_clusters / n_clusters * 100) if n_clusters > 0 else 0
        
        # Overall score is a weighted combination of valid clusters and clustered points
        score = (0.7 * valid_percentage) + (0.3 * clustered_percentage)
        
        return score, valid_percentage, clustered_percentage, cluster_quality
    
    if verbose:
        print(f"Starting adaptive parameter search with eps={current_eps}, min_neighbors={current_min_neighbors}")
        print(f"Expected contact counts: {expected_contact_counts}")
    
    for iteration in range(max_iterations):
        # Apply DBSCAN with current parameters
        dbscan = DBSCAN(eps=current_eps, min_samples=current_min_neighbors)
        clusters = dbscan.fit_predict(coords_array)
        
        # Evaluate clustering quality
        score, valid_percentage, clustered_percentage, cluster_quality = evaluate_clustering(clusters, len(coords_array))
        
        # Count clusters and noise points
        unique_clusters = set(clusters)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        n_noise = np.sum(clusters == -1)
        
        # Store iteration data for visualization
        iterations_data.append({
            'iteration': iteration,
            'eps': current_eps,
            'min_neighbors': current_min_neighbors,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'score': score,
            'valid_percentage': valid_percentage,
            'clustered_percentage': clustered_percentage,
            'clusters': clusters.copy(),
            'cluster_quality': cluster_quality
        })
        
        if verbose:
            print(f"Iteration {iteration+1}: eps={current_eps}, min_neighbors={current_min_neighbors}, "
                  f"clusters={n_clusters}, noise={n_noise}, score={score:.2f}")
        
        # Check if this is the best score so far
        if score > best_score:
            best_score = score
            best_params = {'eps': current_eps, 'min_neighbors': current_min_neighbors}
            best_clusters = clusters.copy()
            
            if verbose:
                print(f"  → New best parameters found!")
        
        # Adaptive strategy: adjust parameters based on results
        if n_clusters == 0 or n_noise > 0.5 * len(coords_array):
            # Too many noise points or no clusters - increase eps or decrease min_neighbors
            if current_min_neighbors > 2:
                current_min_neighbors -= 1
                if verbose:
                    print(f"  → Too many noise points, decreasing min_neighbors to {current_min_neighbors}")
            else:
                current_eps += eps_step
                if verbose:
                    print(f"  → Too many noise points, increasing eps to {current_eps}")
        elif n_clusters > 2 * len(expected_contact_counts):
            # Too many small clusters - increase eps
            current_eps += eps_step
            if verbose:
                print(f"  → Too many small clusters, increasing eps to {current_eps}")
        elif valid_percentage < 50 and clustered_percentage > 80:
            # Most points are clustered but clusters don't match expected sizes
            # Try decreasing eps slightly to split merged clusters
            current_eps -= eps_step * 0.5
            if verbose:
                print(f"  → Clusters don't match expected sizes, slightly decreasing eps to {current_eps}")
        else:
            # Try small adjustments in both directions
            if iteration % 2 == 0:
                current_eps += eps_step * 0.5
                if verbose:
                    print(f"  → Fine-tuning, slightly increasing eps to {current_eps}")
            else:
                current_eps -= eps_step * 0.3
                if verbose:
                    print(f"  → Fine-tuning, slightly decreasing eps to {current_eps}")
        
        # Ensure eps doesn't go below a minimum threshold
        current_eps = max(current_eps, 1.0)
    
    # Create visualization of the parameter search
    fig = create_parameter_search_visualization(coords_array, iterations_data, expected_contact_counts)
    
    return {
        'optimal_eps': best_params['eps'],
        'optimal_min_neighbors': best_params['min_neighbors'],
        'score': best_score,
        'iterations_data': iterations_data,
        'best_clusters': best_clusters,
        'visualization': fig
    }

def create_parameter_search_visualization(coords_array, iterations_data, expected_contact_counts):
    """
    Create visualizations showing the clustering process across iterations.
    
    Args:
        coords_array (numpy.ndarray): Array of electrode coordinates
        iterations_data (list): List of dictionaries with iteration results
        expected_contact_counts (list): List of expected electrode contact counts
        
    Returns:
        matplotlib.figure.Figure: Figure with visualization panels
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Adaptive Clustering Parameter Search', fontsize=18)
    
    # Create grid layout
    gs = GridSpec(3, 3, figure=fig)
    
    # 1. Parameter trajectory plot
    ax1 = fig.add_subplot(gs[0, 0])
    eps_values = [data['eps'] for data in iterations_data]
    min_neighbors_values = [data['min_neighbors'] for data in iterations_data]
    iterations = [data['iteration'] for data in iterations_data]
    
    ax1.plot(iterations, eps_values, 'o-', label='eps')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('eps (max distance)')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Parameter Values by Iteration')
    
    # Add min_neighbors as a secondary y-axis
    ax1_twin = ax1.twinx()
    ax1_twin.plot(iterations, min_neighbors_values, 'x--', color='red', label='min_neighbors')
    ax1_twin.set_ylabel('min_neighbors')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # 2. Score plot
    ax2 = fig.add_subplot(gs[0, 1])
    scores = [data['score'] for data in iterations_data]
    valid_percentages = [data['valid_percentage'] for data in iterations_data]
    clustered_percentages = [data['clustered_percentage'] for data in iterations_data]
    
    ax2.plot(iterations, scores, 'o-', label='Overall Score')
    ax2.plot(iterations, valid_percentages, 's--', label='Valid Clusters %')
    ax2.plot(iterations, clustered_percentages, '^-.', label='Clustered Points %')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Score / Percentage')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('Clustering Quality Metrics')
    
    # 3. Cluster count plot
    ax3 = fig.add_subplot(gs[0, 2])
    n_clusters = [data['n_clusters'] for data in iterations_data]
    n_noise = [data['n_noise'] for data in iterations_data]
    
    ax3.plot(iterations, n_clusters, 'o-', label='Number of Clusters')
    ax3.plot(iterations, n_noise, 'x--', label='Number of Noise Points')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_title('Cluster and Noise Point Counts')
    
    # 4. 3D visualization of best clustering result
    ax4 = fig.add_subplot(gs[1, :], projection='3d')
    
    # Get best iteration (highest score)
    best_iteration = iterations_data[np.argmax([data['score'] for data in iterations_data])]
    clusters = best_iteration['clusters']
    
    # Get unique clusters (excluding noise)
    unique_clusters = sorted(set(clusters))
    if -1 in unique_clusters:
        unique_clusters.remove(-1)
    
    # Plot each cluster with a different color
    colormap = plt.cm.tab20(np.linspace(0, 1, max(20, len(unique_clusters))))
    
    for i, cluster_id in enumerate(unique_clusters):
        mask = clusters == cluster_id
        cluster_size = np.sum(mask)
        
        # Get quality info for this cluster
        quality_info = best_iteration['cluster_quality'].get(cluster_id, {})
        expected_size = quality_info.get('closest_expected', 'unknown')
        is_valid = quality_info.get('valid', False)
        
        # Choose color and marker based on validity
        color = colormap[i % len(colormap)]
        marker = 'o' if is_valid else '^'
        
        ax4.scatter(
            coords_array[mask, 0], 
            coords_array[mask, 1], 
            coords_array[mask, 2],
            c=[color], 
            marker=marker,
            s=80, 
            alpha=0.8, 
            label=f'Cluster {cluster_id} (n={cluster_size}, exp={expected_size})'
        )
    
    # Plot noise points if any
    noise_mask = clusters == -1
    if np.any(noise_mask):
        ax4.scatter(
            coords_array[noise_mask, 0], 
            coords_array[noise_mask, 1], 
            coords_array[noise_mask, 2],
            c='black', 
            marker='x', 
            s=50, 
            alpha=0.5, 
            label=f'Noise points (n={np.sum(noise_mask)})'
        )
    
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Y (mm)')
    ax4.set_zlabel('Z (mm)')
    ax4.set_title(f'Best Clustering Result (eps={best_iteration["eps"]:.2f}, min_neighbors={best_iteration["min_neighbors"]})')
    
    # Create a simplified legend (limit to 15 items max)
    handles, labels = ax4.get_legend_handles_labels()
    if len(handles) > 15:
        handles = handles[:14] + [handles[-1]]
        labels = labels[:14] + [labels[-1]]
    
    ax4.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1))
    
    # 5. Cluster size distribution
    ax5 = fig.add_subplot(gs[2, 0:2])
    
    # Collect cluster sizes from best iteration
    cluster_sizes = [np.sum(clusters == c) for c in unique_clusters]
    
    # Create histogram of cluster sizes
    bins = np.arange(min(expected_contact_counts) - 3, max(expected_contact_counts) + 4)
    hist, edges = np.histogram(cluster_sizes, bins=bins)
    
    # Plot the histogram
    bars = ax5.bar(edges[:-1], hist, width=0.8, align='edge', alpha=0.7)
    
    # Highlight expected contact counts
    for i, size in enumerate(expected_contact_counts):
        nearest_edge_idx = np.argmin(np.abs(edges - size))
        if nearest_edge_idx < len(bars):
            bars[nearest_edge_idx].set_color('green')
            bars[nearest_edge_idx].set_alpha(0.9)
    
    # Add count labels above bars
    for bar, count in zip(bars, hist):
        if count > 0:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom')
    
    # Mark expected contact counts with vertical lines
    for size in expected_contact_counts:
        ax5.axvline(x=size, color='red', linestyle='--', alpha=0.5)
        ax5.text(size, max(hist) + 0.5, str(size), ha='center', va='bottom', color='red')
    
    ax5.set_xlabel('Cluster Size (Number of Contacts)')
    ax5.set_ylabel('Number of Clusters')
    ax5.set_title('Distribution of Cluster Sizes')
    ax5.set_xticks(range(min(expected_contact_counts) - 2, max(expected_contact_counts) + 3))
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary table with parameter recommendations
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    # Get best parameters
    best_iteration = iterations_data[np.argmax([data['score'] for data in iterations_data])]
    best_eps = best_iteration['eps']
    best_min_neighbors = best_iteration['min_neighbors']
    best_score = best_iteration['score']
    best_valid_percent = best_iteration['valid_percentage']
    best_n_clusters = best_iteration['n_clusters']
    best_n_noise = best_iteration['n_noise']
    
    # Calculate percentage of expected size matches
    n_valid_clusters = sum(1 for info in best_iteration['cluster_quality'].values() if info.get('valid', False))
    n_close_clusters = sum(1 for info in best_iteration['cluster_quality'].values() if info.get('close', False))
    
    # Create summary text
    summary_text = [
        f"Optimal Parameters:",
        f"→ eps = {best_eps:.2f}",
        f"→ min_neighbors = {best_min_neighbors}",
        f"",
        f"Clustering Results:",
        f"→ Total clusters: {best_n_clusters}",
        f"→ Valid clusters: {n_valid_clusters} ({n_valid_clusters/best_n_clusters*100:.1f}% if >0)",
        f"→ Close clusters: {n_close_clusters} ({n_close_clusters/best_n_clusters*100:.1f}% if >0)",
        f"→ Noise points: {best_n_noise} ({best_n_noise/len(coords_array)*100:.1f}%)",
        f"",
        f"Quality Metrics:",
        f"→ Overall score: {best_score:.2f}",
        f"→ Valid cluster %: {best_valid_percent:.1f}%",
        f"→ Clustered points %: {best_iteration['clustered_percentage']:.1f}%",
        f"",
        f"Expected Contact Counts:",
        f"→ {', '.join(map(str, expected_contact_counts))}"
    ]
    
    ax6.text(0.05, 0.95, '\n'.join(summary_text), 
             transform=ax6.transAxes, 
             fontsize=12,
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax6.set_title('Parameter Recommendations')
    
    plt.tight_layout()
    
    return fig

def perform_adaptive_trajectory_analysis(coords_array, entry_points=None, 
                                         initial_eps=7.5, initial_min_neighbors=3,
                                         expected_contact_counts=[5, 8, 10, 12, 15, 18],
                                         output_dir=None):
    """
    Perform trajectory analysis with adaptive parameter selection for DBSCAN.
    
    Args:
        coords_array (numpy.ndarray): Array of electrode coordinates
        entry_points (numpy.ndarray, optional): Array of entry point coordinates
        initial_eps (float): Initial value for max neighbor distance (eps) in DBSCAN
        initial_min_neighbors (int): Initial value for min_samples in DBSCAN
        expected_contact_counts (list): List of expected electrode contact counts
        output_dir (str, optional): Directory to save visualizations
        
    Returns:
        dict: Results dictionary with trajectory analysis and parameter search results
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    import networkx as nx
    from scipy.spatial.distance import cdist
    from matplotlib.backends.backend_pdf import PdfPages
    
    print(f"Starting adaptive trajectory analysis...")
    
    # Step 1: Find optimal clustering parameters
    print(f"Finding optimal clustering parameters...")
    parameter_search = adaptive_clustering_parameters(
        coords_array, 
        initial_eps=initial_eps,
        initial_min_neighbors=initial_min_neighbors,
        expected_contact_counts=expected_contact_counts,
        max_iterations=10,
        verbose=True
    )
    
    optimal_eps = parameter_search['optimal_eps']
    optimal_min_neighbors = parameter_search['optimal_min_neighbors']
    
    print(f"Found optimal parameters: eps={optimal_eps}, min_neighbors={optimal_min_neighbors}")
    
    # Step 2: Run integrated trajectory analysis with optimal parameters
    print(f"Running trajectory analysis with optimal parameters...")
    results = integrated_trajectory_analysis(
        coords_array=coords_array,
        entry_points=entry_points,
        max_neighbor_distance=optimal_eps,
        min_neighbors=optimal_min_neighbors
    )
    
    # Step 3: Add validation
    validation = validate_electrode_clusters(results, expected_contact_counts)
    results['electrode_validation'] = validation
    
    # Create validation visualization and add to results
    if 'figures' not in results:
        results['figures'] = {}
    
    results['figures']['electrode_validation'] = create_electrode_validation_page(results, validation)
    results['parameter_search'] = parameter_search
    
    # Save parameter search visualization to PDF if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save parameter search visualization
        plt.figure(parameter_search['visualization'].number)
        plt.savefig(os.path.join(output_dir, 'adaptive_parameter_search.png'), dpi=300)
        
        # Save parameter search data to PDF
        with PdfPages(os.path.join(output_dir, 'adaptive_parameter_search.pdf')) as pdf:
            pdf.savefig(parameter_search['visualization'])
            
            # Add comparison of initial vs optimal clustering
            fig = plt.figure(figsize=(15, 12))
            fig.suptitle('Comparison of Initial vs. Optimal Clustering', fontsize=16)
            
            # Run DBSCAN with initial parameters for comparison
            initial_dbscan = DBSCAN(eps=initial_eps, min_samples=initial_min_neighbors)
            initial_clusters = initial_dbscan.fit_predict(coords_array)
            
            # Get optimal clusters
            optimal_clusters = parameter_search['best_clusters']
            
            # Create 3D plots side by side
            # Initial parameters plot
            ax1 = fig.add_subplot(121, projection='3d')
            
            # Get unique clusters (excluding noise)
            initial_unique_clusters = sorted(set(initial_clusters))
            if -1 in initial_unique_clusters:
                initial_unique_clusters.remove(-1)
            
            # Plot each cluster with a different color
            colormap = plt.cm.tab20(np.linspace(0, 1, max(20, len(initial_unique_clusters))))
            
            for i, cluster_id in enumerate(initial_unique_clusters):
                mask = initial_clusters == cluster_id
                cluster_size = np.sum(mask)
                
                color = colormap[i % len(colormap)]
                
                ax1.scatter(
                    coords_array[mask, 0], 
                    coords_array[mask, 1], 
                    coords_array[mask, 2],
                    c=[color], 
                    marker='o',
                    s=80, 
                    alpha=0.8, 
                    label=f'Cluster {cluster_id} (n={cluster_size})'
                )
            
            # Plot noise points if any
            noise_mask = initial_clusters == -1
            if np.any(noise_mask):
                ax1.scatter(
                    coords_array[noise_mask, 0], 
                    coords_array[noise_mask, 1], 
                    coords_array[noise_mask, 2],
                    c='black', 
                    marker='x', 
                    s=50, 
                    alpha=0.5, 
                    label=f'Noise points (n={np.sum(noise_mask)})'
                )
            
            ax1.set_xlabel('X (mm)')
            ax1.set_ylabel('Y (mm)')
            ax1.set_zlabel('Z (mm)')
            ax1.set_title(f'Initial Clustering (eps={initial_eps}, min_neighbors={initial_min_neighbors})\n'
                         f'Clusters: {len(initial_unique_clusters)}, Noise: {np.sum(noise_mask)}')
            
            # Optimal parameters plot
            ax2 = fig.add_subplot(122, projection='3d')
            
            # Get unique clusters (excluding noise)
            optimal_unique_clusters = sorted(set(optimal_clusters))
            if -1 in optimal_unique_clusters:
                optimal_unique_clusters.remove(-1)
            
            # Plot each cluster with a different color
            colormap = plt.cm.tab20(np.linspace(0, 1, max(20, len(optimal_unique_clusters))))
            
            for i, cluster_id in enumerate(optimal_unique_clusters):
                mask = optimal_clusters == cluster_id
                cluster_size = np.sum(mask)
                
                color = colormap[i % len(colormap)]
                
                ax2.scatter(
                    coords_array[mask, 0], 
                    coords_array[mask, 1], 
                    coords_array[mask, 2],
                    c=[color], 
                    marker='o',
                    s=80, 
                    alpha=0.8, 
                    label=f'Cluster {cluster_id} (n={cluster_size})'
                )
            
            # Plot noise points if any
            noise_mask = optimal_clusters == -1
            if np.any(noise_mask):
                ax2.scatter(
                    coords_array[noise_mask, 0], 
                    coords_array[noise_mask, 1], 
                    coords_array[noise_mask, 2],
                    c='black', 
                    marker='x', 
                    s=50, 
                    alpha=0.5, 
                    label=f'Noise points (n={np.sum(noise_mask)})'
                )
            
            ax2.set_xlabel('X (mm)')
            ax2.set_ylabel('Y (mm)')
            ax2.set_zlabel('Z (mm)')
            ax2.set_title(f'Optimal Clustering (eps={optimal_eps:.2f}, min_neighbors={optimal_min_neighbors})\n'
                         f'Clusters: {len(optimal_unique_clusters)}, Noise: {np.sum(noise_mask)}')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        
        print(f"✅ Adaptive parameter search report saved to {os.path.join(output_dir, 'adaptive_parameter_search.pdf')}")
    
    return results

def visualize_adaptive_clustering(coords_array, iterations_data, expected_contact_counts, output_dir=None):
    """
    Create an animated or multi-panel visualization showing the evolution of 
    clustering across iterations of the adaptive parameter search.
    
    Args:
        coords_array (numpy.ndarray): Array of electrode coordinates
        iterations_data (list): List of dictionaries with iteration results
        expected_contact_counts (list): List of expected electrode contact counts
        output_dir (str, optional): Directory to save visualizations
        
    Returns:
        dict: Information about the visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.gridspec import GridSpec
    import os
    
    # Create a multi-panel figure showing the evolution
    n_iterations = len(iterations_data)
    
    # Calculate grid dimensions
    if n_iterations <= 6:
        n_rows, n_cols = 2, 3
    elif n_iterations <= 9:
        n_rows, n_cols = 3, 3
    elif n_iterations <= 12:
        n_rows, n_cols = 3, 4
    else:
        n_rows, n_cols = 4, 4
    
    # Ensure we don't have more panels than iterations
    n_plots = min(n_rows * n_cols, n_iterations)
    
    # Calculate which iterations to show (distribute evenly)
    if n_iterations > n_plots:
        plot_indices = np.linspace(0, n_iterations-1, n_plots, dtype=int)
    else:
        plot_indices = np.arange(n_iterations)
    
    # Create figure
    fig = plt.figure(figsize=(5*n_cols, 4*n_rows))
    fig.suptitle('Evolution of Clustering Parameters', fontsize=18)
    
    # Add a colormap for consistency across plots
    max_clusters = max([data['n_clusters'] for data in iterations_data])
    colormap = plt.cm.tab20(np.linspace(0, 1, max(20, max_clusters)))
    
    # Create a plot for each selected iteration
    for i, idx in enumerate(plot_indices):
        data = iterations_data[idx]
        
        # Create 3D subplot
        ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
        
        # Get clusters for this iteration
        clusters = data['clusters']
        unique_clusters = sorted(set(clusters))
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
        
        # Plot each cluster
        for j, cluster_id in enumerate(unique_clusters):
            mask = clusters == cluster_id
            size = np.sum(mask)
            
            # Check if size matches expected counts
            color = colormap[j % len(colormap)]
            marker = 'o'
            
            # Mark clusters that match expected sizes
            if size in expected_contact_counts:
                marker = '*'
            
            ax.scatter(
                coords_array[mask, 0], 
                coords_array[mask, 1], 
                coords_array[mask, 2],
                c=[color], 
                marker=marker,
                s=50, 
                alpha=0.8
            )
        
        # Plot noise points
        noise_mask = clusters == -1
        if np.any(noise_mask):
            ax.scatter(
                coords_array[noise_mask, 0], 
                coords_array[noise_mask, 1], 
                coords_array[noise_mask, 2],
                c='black', 
                marker='x', 
                s=30, 
                alpha=0.5
            )
        
        # Set axis labels and title
        if i >= (n_rows-1) * n_cols:  # Only bottom row gets x labels
            ax.set_xlabel('X (mm)')
        if i % n_cols == 0:  # Only leftmost column gets y labels
            ax.set_ylabel('Y (mm)')
        
        ax.set_title(f"Iteration {data['iteration']+1}\neps={data['eps']:.2f}, min_n={data['min_neighbors']}\nClusters: {data['n_clusters']}, Noise: {data['n_noise']}")
        
        # Adjust view angle for better visibility
        ax.view_init(elev=20, azim=45+i*5)
    
    plt.tight_layout()
    
    # Save visualization if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'adaptive_clustering_evolution.png'), dpi=300)
        plt.close(fig)
        
        # Also create individual frames for possible animation
        print("Creating individual frames...")
        frames_dir = os.path.join(output_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        for i, data in enumerate(iterations_data):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get clusters for this iteration
            clusters = data['clusters']
            unique_clusters = sorted(set(clusters))
            if -1 in unique_clusters:
                unique_clusters.remove(-1)
            
            # Plot each cluster
            for j, cluster_id in enumerate(unique_clusters):
                mask = clusters == cluster_id
                size = np.sum(mask)
                
                # Check if size matches expected counts
                color = colormap[j % len(colormap)]
                marker = 'o'
                
                # Mark clusters that match expected sizes
                if size in expected_contact_counts:
                    marker = '*'
                
                ax.scatter(
                    coords_array[mask, 0], 
                    coords_array[mask, 1], 
                    coords_array[mask, 2],
                    c=[color], 
                    marker=marker,
                    s=50, 
                    alpha=0.8
                )
            
            # Plot noise points
            noise_mask = clusters == -1
            if np.any(noise_mask):
                ax.scatter(
                    coords_array[noise_mask, 0], 
                    coords_array[noise_mask, 1], 
                    coords_array[noise_mask, 2],
                    c='black', 
                    marker='x', 
                    s=30, 
                    alpha=0.5
                )
            
            # Set axis labels and title
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            
            # Add parameter info
            ax.set_title(
                f"Iteration {data['iteration']+1}: eps={data['eps']:.2f}, min_neighbors={data['min_neighbors']}\n"
                f"Clusters: {data['n_clusters']}, Noise: {data['n_noise']}, Score: {data['score']:.2f}"
            )
            
            # Adjust view angle for better visibility
            ax.view_init(elev=20, azim=45)
            
            # Save frame
            plt.savefig(os.path.join(frames_dir, f'frame_{i:02d}.png'), dpi=300)
            plt.close(fig)
        
        print(f"✅ Created {len(iterations_data)} frames in {frames_dir}")
        
        # Attempt to create an animated GIF if PIL is available
        try:
            from PIL import Image
            import glob
            
            print("Creating animated GIF...")
            frames = []
            frame_files = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.png')))
            
            for frame_file in frame_files:
                frame = Image.open(frame_file)
                frames.append(frame)
            
            gif_path = os.path.join(output_dir, 'adaptive_clustering_animation.gif')
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000,  # 1 second per frame
                loop=0  # Loop indefinitely
            )
            
            print(f"✅ Created animation: {gif_path}")
            
        except ImportError:
            print("PIL not available, skipping animated GIF creation.")
    
    return {
        'figure': fig,
        'iterations': n_iterations,
        'plot_indices': plot_indices.tolist()
    }

# ------------------------------------------------------------------------------
# PART 2.5: SPACING 
# ------------------------------------------------------------------------------

def validate_electrode_spacing(trajectory_points, expected_spacing_range=(3.0, 5.0)):
    """
    Validate the spacing between electrode contacts in a trajectory.
    
    Args:
        trajectory_points (numpy.ndarray): Array of contact coordinates along a trajectory
        expected_spacing_range (tuple): Expected range of spacing (min, max) in mm
        
    Returns:
        dict: Dictionary with spacing validation results
    """
    import numpy as np
    
    # Make sure points are sorted along the trajectory
    # This assumes trajectory_points are already sorted along the main axis
    
    # Calculate pairwise distances between adjacent points
    distances = []
    for i in range(1, len(trajectory_points)):
        dist = np.linalg.norm(trajectory_points[i] - trajectory_points[i-1])
        distances.append(dist)
    
    # Calculate spacing statistics
    min_spacing = np.min(distances) if distances else np.nan
    max_spacing = np.max(distances) if distances else np.nan
    mean_spacing = np.mean(distances) if distances else np.nan
    std_spacing = np.std(distances) if distances else np.nan
    
    # Check if spacings are within expected range
    min_expected, max_expected = expected_spacing_range
    valid_spacings = [min_expected <= d <= max_expected for d in distances]
    
    # Identify problematic spacings (too close or too far)
    too_close = [i for i, d in enumerate(distances) if d < min_expected]
    too_far = [i for i, d in enumerate(distances) if d > max_expected]
    
    # Calculate percentage of valid spacings
    valid_percentage = np.mean(valid_spacings) * 100 if valid_spacings else 0
    
    return {
        'distances': distances,
        'min_spacing': min_spacing,
        'max_spacing': max_spacing,
        'mean_spacing': mean_spacing,
        'std_spacing': std_spacing,
        'cv_spacing': std_spacing / mean_spacing if mean_spacing > 0 else np.nan,  # Coefficient of variation
        'valid_percentage': valid_percentage,
        'valid_spacings': valid_spacings,
        'too_close_indices': too_close,
        'too_far_indices': too_far,
        'expected_range': expected_spacing_range,
        'is_valid': valid_percentage >= 75,  # Consider valid if at least 75% of spacings are valid
        'status': 'valid' if valid_percentage >= 75 else 'invalid'
    }

# 2.5.1: SPACING VALIDATION PAGE
def create_spacing_validation_page(results):
    """
    Create a visualization page for electrode spacing validation results.
    
    Args:
        results (dict): Results from integrated_trajectory_analysis with spacing validation
        
    Returns:
        matplotlib.figure.Figure: Figure containing spacing validation results
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np
    
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle('Electrode Contact Spacing Validation (Expected: 3-5mm)', fontsize=16)
    
    # Create grid layout
    gs = GridSpec(2, 2, figure=fig)
    
    # Summary statistics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    # Calculate overall statistics
    trajectories = results.get('trajectories', [])
    
    if not trajectories:
        ax1.text(0.5, 0.5, "No trajectory data available", ha='center', va='center', fontsize=14)
        return fig
    
    # Count trajectories with valid/invalid spacing
    valid_trajectories = sum(1 for t in trajectories if t.get('spacing_validation', {}).get('is_valid', False))
    invalid_trajectories = len(trajectories) - valid_trajectories
    
    # Calculate average spacing statistics across all trajectories
    all_spacings = []
    for traj in trajectories:
        if 'spacing_validation' in traj and 'distances' in traj['spacing_validation']:
            all_spacings.extend(traj['spacing_validation']['distances'])
    
    mean_spacing = np.mean(all_spacings) if all_spacings else np.nan
    std_spacing = np.std(all_spacings) if all_spacings else np.nan
    min_spacing = np.min(all_spacings) if all_spacings else np.nan
    max_spacing = np.max(all_spacings) if all_spacings else np.nan
    
    # Create summary table
    summary_data = []
    summary_columns = [
        'Total Trajectories', 
        'Valid Spacing', 
        'Invalid Spacing',
        'Mean Spacing (mm)',
        'Min-Max Spacing (mm)'
    ]
    
    summary_data.append([
        str(len(trajectories)),
        f"{valid_trajectories} ({valid_trajectories/len(trajectories)*100:.1f}%)",
        f"{invalid_trajectories} ({invalid_trajectories/len(trajectories)*100:.1f}%)",
        f"{mean_spacing:.2f} ± {std_spacing:.2f}",
        f"{min_spacing:.2f} - {max_spacing:.2f}"
    ])
    
    summary_table = ax1.table(cellText=summary_data, colLabels=summary_columns,
                             loc='center', cellLoc='center')
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(10)
    summary_table.scale(1, 1.5)
    ax1.set_title('Spacing Validation Summary')
    
    # Histogram of all spacings
    ax2 = fig.add_subplot(gs[0, 1])
    if all_spacings:
        ax2.hist(all_spacings, bins=20, alpha=0.7)
        ax2.axvline(x=3.0, color='r', linestyle='--', label='Min Expected (3mm)')
        ax2.axvline(x=5.0, color='r', linestyle='--', label='Max Expected (5mm)')
        ax2.set_xlabel('Spacing (mm)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Contact Spacings')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No spacing data available", ha='center', va='center', fontsize=14)
    
    # Detailed trajectory spacing table
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    # Create detailed spacing table
    detail_data = []
    detail_columns = [
        'Trajectory ID', 
        'Contacts', 
        'Mean Spacing (mm)', 
        'Min Spacing (mm)', 
        'Max Spacing (mm)',
        'CV (%)',
        'Valid Percentage',
        'Status'
    ]
    
    for traj in trajectories:
        traj_id = traj['cluster_id']
        contact_count = traj['electrode_count']
        
        spacing_validation = traj.get('spacing_validation', {})
        if not spacing_validation:
            continue
            
        mean_spacing = spacing_validation.get('mean_spacing', np.nan)
        min_spacing = spacing_validation.get('min_spacing', np.nan)
        max_spacing = spacing_validation.get('max_spacing', np.nan)
        cv_spacing = spacing_validation.get('cv_spacing', np.nan) * 100  # Convert to percentage
        valid_percentage = spacing_validation.get('valid_percentage', 0)
        status = spacing_validation.get('status', 'unknown')
        
        row = [
            traj_id,
            contact_count,
            f"{mean_spacing:.2f}" if not np.isnan(mean_spacing) else "N/A",
            f"{min_spacing:.2f}" if not np.isnan(min_spacing) else "N/A",
            f"{max_spacing:.2f}" if not np.isnan(max_spacing) else "N/A",
            f"{cv_spacing:.1f}%" if not np.isnan(cv_spacing) else "N/A",
            f"{valid_percentage:.1f}%" if not np.isnan(valid_percentage) else "N/A",
            status.upper()
        ]
        detail_data.append(row)
    
    # Sort by trajectory ID - FIXED SORTING FUNCTION
    def safe_sort_key(x):
        if isinstance(x[0], int):
            return (0, x[0], "")  # Integer IDs come first
        elif isinstance(x[0], str) and x[0].isdigit():
            return (0, int(x[0]), "")  # String representations of integers
        else:
            # For complex IDs like "M_1_2", extract any numeric parts
            try:
                # Try to extract a primary numeric component
                if isinstance(x[0], str) and "_" in x[0]:
                    parts = x[0].split("_")
                    if len(parts) > 1 and parts[1].isdigit():
                        return (1, int(parts[1]), x[0])  # Sort by first numeric part after prefix
                # If that fails, just use the string itself
                return (2, 0, x[0])
            except:
                return (3, 0, str(x[0]))  # Last resort
    
    detail_data.sort(key=safe_sort_key)
    
    if detail_data:
        detail_table = ax3.table(cellText=detail_data, colLabels=detail_columns,
                               loc='center', cellLoc='center')
        detail_table.auto_set_font_size(False)
        detail_table.set_fontsize(10)
        detail_table.scale(1, 1.5)
        
        # Color code status cells
        for i, row in enumerate(detail_data):
            status = row[-1]
            cell = detail_table[(i+1, len(detail_columns)-1)]  # +1 for header row
            if status == 'VALID':
                cell.set_facecolor('lightgreen')
            elif status == 'INVALID':
                cell.set_facecolor('lightcoral')
    else:
        ax3.text(0.5, 0.5, "No detailed spacing data available", ha='center', va='center', fontsize=14)
    
    ax3.set_title('Detailed Trajectory Spacing Analysis')
    
    plt.tight_layout()
    return fig

# 2.5.2: ENHANCED 3D VISUALIZATION

def enhance_3d_visualization_with_spacing(coords_array, results):
    """
    Create an enhanced 3D visualization highlighting electrode spacing issues.
    
    Args:
        coords_array (numpy.ndarray): Array of electrode coordinates
        results (dict): Results from integrated_trajectory_analysis
        
    Returns:
        matplotlib.figure.Figure: Figure containing the enhanced 3D visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get data for plotting
    clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
    unique_clusters = sorted(set(clusters))
    if -1 in unique_clusters:
        unique_clusters.remove(-1)  # Remove noise points
    
    # Create colormap for clusters
    cluster_cmap = plt.cm.tab20(np.linspace(0, 1, max(20, len(unique_clusters))))
    
    # Plot trajectories with spacing validation
    for traj in results.get('trajectories', []):
        if 'spacing_validation' not in traj:
            continue
            
        cluster_id = traj['cluster_id']
        cluster_mask = clusters == cluster_id
        cluster_coords = coords_array[cluster_mask]
        
        # Skip if not enough points
        if len(cluster_coords) < 2:
            continue
        
        # Get trajectory direction and sort points
        direction = np.array(traj['direction'])
        center = np.mean(cluster_coords, axis=0)
        projected = np.dot(cluster_coords - center, direction)
        sorted_indices = np.argsort(projected)
        sorted_coords = cluster_coords[sorted_indices]
        
        # Get color for this cluster
        color_idx = unique_clusters.index(cluster_id) if cluster_id in unique_clusters else 0
        color = cluster_cmap[color_idx % len(cluster_cmap)]
        
        # Plot electrode contacts
        ax.scatter(sorted_coords[:, 0], sorted_coords[:, 1], sorted_coords[:, 2], 
                  color=color, marker='o', s=80, alpha=0.7, label=f'Cluster {cluster_id}')
        
        # Highlight spacing issues
        spacing_validation = traj['spacing_validation']
        too_close = spacing_validation.get('too_close_indices', [])
        too_far = spacing_validation.get('too_far_indices', [])
        
        # For each problematic spacing, highlight the pair of contacts
        for idx in too_close:
            # These are indices in the distances array, so idx and idx+1 in sorted_coords
            p1, p2 = sorted_coords[idx], sorted_coords[idx+1]
            
            # Plot these contacts with special marker
            ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                      color='red', marker='*', s=150, alpha=1.0)
            
            # Connect them with a red line to highlight
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                   '-', color='red', linewidth=3, alpha=0.8)
        
        for idx in too_far:
            # These are indices in the distances array, so idx and idx+1 in sorted_coords
            p1, p2 = sorted_coords[idx], sorted_coords[idx+1]
            
            # Plot these contacts with special marker
            ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                      color='orange', marker='*', s=150, alpha=1.0)
            
            # Connect them with an orange line to highlight
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                   '-', color='orange', linewidth=3, alpha=0.8)
        
        # Plot the main trajectory line
        if 'spline_points' in traj and traj['spline_points']:
            spline_points = np.array(traj['spline_points'])
            ax.plot(spline_points[:, 0], spline_points[:, 1], spline_points[:, 2], 
                   '-', color=color, linewidth=2, alpha=0.5)
        else:
            ax.plot([sorted_coords[0, 0], sorted_coords[-1, 0]],
                   [sorted_coords[0, 1], sorted_coords[-1, 1]],
                   [sorted_coords[0, 2], sorted_coords[-1, 2]],
                   '-', color=color, linewidth=2, alpha=0.5)
    
    # Plot noise points
    noise_mask = clusters == -1
    if np.any(noise_mask):
        ax.scatter(coords_array[noise_mask, 0], coords_array[noise_mask, 1], coords_array[noise_mask, 2],
                  c='black', marker='x', s=30, alpha=0.5, label='Noise points')
    
    # Add legend and labels
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    
    # Add spacing validation info to title
    total_trajectories = len(results.get('trajectories', []))
    valid_trajectories = sum(1 for t in results.get('trajectories', []) 
                         if t.get('spacing_validation', {}).get('is_valid', False))
    
    title = (f'3D Electrode Trajectory Analysis with Spacing Validation\n'
            f'{valid_trajectories} of {total_trajectories} trajectories have valid spacing (3-5mm)\n'
            f'Red stars: contacts too close (<3mm), Orange stars: contacts too far (>5mm)')
    ax.set_title(title)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Electrode contact'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, label='Too close (<3mm)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='orange', markersize=15, label='Too far (>5mm)'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=10, label='Noise point'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig

#--------------------------------------------------------------------------------
# PART 2.6: Trajectories problems: merging and splitting
#--------------------------------------------------------------------------------

def targeted_trajectory_refinement(trajectories, expected_contact_counts=[5, 8, 10, 12, 15, 18], 
                                 max_expected=18, tolerance=3):
    """
    Apply splitting and merging operations only to trajectories that need it.
    
    Args:
        trajectories: List of trajectory dictionaries
        expected_contact_counts: List of expected electrode contact counts
        max_expected: Maximum reasonable number of contacts in a single trajectory
        tolerance: How close to expected counts is considered valid
        
    Returns:
        dict: Results with refined trajectories and statistics
    """
    # Step 1: Flag trajectories that need attention
    merge_candidates = []  # Trajectories that might need to be merged (too few contacts)
    split_candidates = []  # Trajectories that might need to be split (too many contacts)
    valid_trajectories = []  # Trajectories that match expected counts
    
    # Group trajectories by contact count validity
    for traj in trajectories:
        contact_count = traj['electrode_count']
        
        # Find closest expected count
        closest_expected = min(expected_contact_counts, key=lambda x: abs(x - contact_count))
        difference = abs(closest_expected - contact_count)
        
        # Flag based on contact count
        if contact_count > max_expected:
            # Too many contacts - likely needs splitting
            traj['closest_expected'] = closest_expected
            traj['count_difference'] = difference
            split_candidates.append(traj)
        elif difference > tolerance and contact_count < closest_expected:
            # Too few contacts compared to expected - potential merge candidate
            traj['closest_expected'] = closest_expected
            traj['count_difference'] = difference
            traj['missing_contacts'] = closest_expected - contact_count
            merge_candidates.append(traj)
        else:
            # Contact count is valid
            valid_trajectories.append(traj)
    
    print(f"Initial classification:")
    print(f"- Valid trajectories: {len(valid_trajectories)}")
    print(f"- Potential merge candidates: {len(merge_candidates)}")
    print(f"- Potential split candidates: {len(split_candidates)}")
    
    # Step 2: Process merge candidates
    # Sort merge candidates by missing contact count (ascending)
    merge_candidates.sort(key=lambda x: x['missing_contacts'])
    
    merged_trajectories = []
    used_in_merge = set()  # Track which trajectories have been used in merges
    
    # For each merge candidate, look for other candidates to merge with
    for i, traj1 in enumerate(merge_candidates):
        if traj1['cluster_id'] in used_in_merge:
            continue
            
        best_match = None
        best_score = float('inf')
        
        for j, traj2 in enumerate(merge_candidates):
            if i == j or traj2['cluster_id'] in used_in_merge:
                continue
                
            # Check if merging would result in a valid count
            combined_count = traj1['electrode_count'] + traj2['electrode_count']
            closest_expected = min(expected_contact_counts, key=lambda x: abs(x - combined_count))
            combined_difference = abs(closest_expected - combined_count)
            
            # Only consider merging if it improves the count validity
            if combined_difference < min(traj1['count_difference'], traj2['count_difference']):
                # Check spatial compatibility
                score = check_merge_compatibility(traj1, traj2)
                if score is not None and score < best_score:
                    best_match = (j, traj2, score)
                    best_score = score
        
        # If we found a good match, merge them
        if best_match:
            j, traj2, score = best_match
            merged_traj = merge_trajectories(traj1, traj2)
            merged_trajectories.append(merged_traj)
            used_in_merge.add(traj1['cluster_id'])
            used_in_merge.add(traj2['cluster_id'])
            print(f"Merged: {traj1['cluster_id']} + {traj2['cluster_id']} = {merged_traj['cluster_id']}")
        else:
            # No good match, keep original
            merged_trajectories.append(traj1)
    
    # Add any merge candidates that weren't used
    for traj in merge_candidates:
        if traj['cluster_id'] not in used_in_merge:
            merged_trajectories.append(traj)
    
    # Step 3: Process split candidates
    final_trajectories = []
    
    # Process each split candidate
    for traj in split_candidates:
        # Try to split the trajectory
        split_result = split_trajectory(traj, expected_contact_counts)
        
        if split_result['success']:
            # Add the split trajectories
            final_trajectories.extend(split_result['trajectories'])
            print(f"Split {traj['cluster_id']} into {len(split_result['trajectories'])} trajectories")
        else:
            # Couldn't split effectively, keep original
            final_trajectories.append(traj)
    
    # Add all merged trajectories and valid trajectories
    final_trajectories.extend(merged_trajectories)
    final_trajectories.extend(valid_trajectories)
    
    # Step 4: Final validation
    validation_results = validate_trajectories(final_trajectories, expected_contact_counts, tolerance)
    
    return {
        'trajectories': final_trajectories,
        'n_trajectories': len(final_trajectories),
        'original_count': len(trajectories),
        'valid_count': len(valid_trajectories),
        'merge_candidates': len(merge_candidates),
        'split_candidates': len(split_candidates),
        'merged_count': len([t for t in final_trajectories if 'merged_from' in t]),
        'split_count': len([t for t in final_trajectories if 'split_from' in t]),
        'validation': validation_results
    }

def check_merge_compatibility(traj1, traj2, max_distance=15, max_angle_diff=20):
    """
    Check if two trajectories can be merged by examining their spatial relationship.
    
    Args:
        traj1, traj2: Trajectory dictionaries
        max_distance: Maximum distance between endpoints to consider merging
        max_angle_diff: Maximum angle difference between directions (degrees)
        
    Returns:
        float: Compatibility score (lower is better) or None if incompatible
    """
    # Get endpoints
    endpoints1 = np.array(traj1['endpoints'])
    endpoints2 = np.array(traj2['endpoints'])
    
    # Calculate distances between all endpoint combinations
    distances = [
        np.linalg.norm(endpoints1[0] - endpoints2[0]),
        np.linalg.norm(endpoints1[0] - endpoints2[1]),
        np.linalg.norm(endpoints1[1] - endpoints2[0]),
        np.linalg.norm(endpoints1[1] - endpoints2[1])
    ]
    
    min_distance = min(distances)
    
    # If endpoints are too far apart, not compatible
    if min_distance > max_distance:
        return None
    
    # Check angle between trajectory directions
    dir1 = np.array(traj1['direction'])
    dir2 = np.array(traj2['direction'])
    
    angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(dir1, dir2)), -1.0, 1.0)))
    
    # If direction vectors point in opposite directions, we need 180-angle
    if np.dot(dir1, dir2) < 0:
        angle = 180 - angle
    
    # If trajectories have very different directions, not compatible
    if angle > max_angle_diff:
        return None
    
    # Compute compatibility score (lower is better)
    score = min_distance + angle * 0.5
    
    return score

def merge_trajectories(traj1, traj2):
    """
    Merge two trajectories into one.
    
    Args:
        traj1, traj2: Trajectory dictionaries
        
    Returns:
        dict: Merged trajectory
    """
    # Create a new trajectory 
    merged_traj = traj1.copy()
    
    # Set new ID
    merged_traj['cluster_id'] = traj1['cluster_id']  # Keep using the first trajectory's ID
    
    # Store original IDs in metadata instead of in the ID itself
    merged_traj['merged_from'] = [traj1['cluster_id'], traj2['cluster_id']]
    merged_traj['is_merged'] = True
    
    # Get endpoints
    endpoints1 = np.array(traj1['endpoints'])
    endpoints2 = np.array(traj2['endpoints'])
    
    # Calculate distances between all endpoint combinations
    distances = [
        (0, 0, np.linalg.norm(endpoints1[0] - endpoints2[0])),
        (0, 1, np.linalg.norm(endpoints1[0] - endpoints2[1])),
        (1, 0, np.linalg.norm(endpoints1[1] - endpoints2[0])),
        (1, 1, np.linalg.norm(endpoints1[1] - endpoints2[1]))
    ]
    
    # Find which endpoints are closest
    closest_pair = min(distances, key=lambda x: x[2])
    idx1, idx2, _ = closest_pair
    
    # Update endpoints to span the full merged trajectory
    if idx1 == 0 and idx2 == 0:
        new_endpoints = [endpoints1[1], endpoints2[1]]
    elif idx1 == 0 and idx2 == 1:
        new_endpoints = [endpoints1[1], endpoints2[0]]
    elif idx1 == 1 and idx2 == 0:
        new_endpoints = [endpoints1[0], endpoints2[1]]
    else:  # idx1 == 1 and idx2 == 1
        new_endpoints = [endpoints1[0], endpoints2[0]]
    
    merged_traj['endpoints'] = [new_endpoints[0].tolist(), new_endpoints[1].tolist()]
    
    # Recalculate direction and length
    new_direction = new_endpoints[1] - new_endpoints[0]
    new_length = np.linalg.norm(new_direction)
    
    merged_traj['direction'] = (new_direction / new_length).tolist()
    merged_traj['length_mm'] = float(new_length)
    merged_traj['electrode_count'] = traj1['electrode_count'] + traj2['electrode_count']
    merged_traj['center'] = ((new_endpoints[0] + new_endpoints[1]) / 2).tolist()
    
    # Mark as merged and record original trajectories
    merged_traj['merged_from'] = [traj1['cluster_id'], traj2['cluster_id']]
    
    return merged_traj

def split_trajectory(traj, expected_contact_counts=[5, 8, 10, 12, 15, 18]):
    """
    Split a trajectory that may contain multiple electrodes.
    
    Args:
        traj: Trajectory dictionary
        expected_contact_counts: List of expected electrode contact counts
        
    Returns:
        dict: Split results with success flag and trajectories
    """
    # Need coordinates for this trajectory
    if 'sorted_coords' not in traj:
        return {'success': False, 'reason': 'No coordinates available', 'trajectories': [traj]}
    
    coords = np.array(traj['sorted_coords'])
    
    # Try to determine how many sub-trajectories to create
    contact_count = traj['electrode_count']
    
    # Find potential combinations of expected counts that would sum close to our count
    best_combination = None
    min_difference = float('inf')
    
    # Try combinations of 2 trajectories
    for count1 in expected_contact_counts:
        for count2 in expected_contact_counts:
            if abs((count1 + count2) - contact_count) < min_difference:
                min_difference = abs((count1 + count2) - contact_count)
                best_combination = [count1, count2]
    
    # If needed, try combinations of 3 trajectories
    if min_difference > 3:  # If 2-trajectory combo isn't close enough
        for count1 in expected_contact_counts:
            for count2 in expected_contact_counts:
                for count3 in expected_contact_counts:
                    diff = abs((count1 + count2 + count3) - contact_count)
                    if diff < min_difference:
                        min_difference = diff
                        best_combination = [count1, count2, count3]
    
    # No good combination found
    if best_combination is None or min_difference > 5:
        return {'success': False, 'reason': 'No good contact count combination found', 'trajectories': [traj]}
    
    # Determine if we should use DBSCAN or K-means for splitting
    if len(best_combination) <= 2:
        # For 2 clusters, try DBSCAN with adjusted parameters
        from sklearn.cluster import DBSCAN
        
        # Estimate good eps value based on contact spacing
        if len(coords) > 1:
            distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            median_spacing = np.median(distances)
            splitting_eps = median_spacing * 1.5  # A bit larger than typical spacing
        else:
            splitting_eps = 5.0  # Default if we can't estimate
        
        # Apply DBSCAN
        splitter = DBSCAN(eps=splitting_eps, min_samples=3)
        labels = splitter.fit_predict(coords)
        
        # Check if splitting produced meaningful results
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
            
        if len(unique_labels) < 2:
            # DBSCAN failed, try K-means as backup
            use_kmeans = True
        else:
            use_kmeans = False
    else:
        # For 3+ clusters, use K-means directly
        use_kmeans = True
    
    # Apply K-means if needed
    if use_kmeans:
        from sklearn.cluster import KMeans
        
        n_clusters = len(best_combination)
        splitter = KMeans(n_clusters=n_clusters)
        labels = splitter.fit_predict(coords)
        unique_labels = set(labels)
    
    # Create sub-trajectories
    split_trajectories = []
    
    # Get maximum existing cluster ID to ensure we create unique IDs
    # This will work even if we don't have access to all trajectories
    base_id = traj['cluster_id']
    id_offset = 1000  # Large offset to avoid conflicts with existing IDs
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        sub_coords = coords[mask]
        
        if len(sub_coords) < 3:  # Need at least 3 points
            continue
            
        # Create new trajectory
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca.fit(sub_coords)
        
        sub_traj = traj.copy()
        
        # Generate a new integer ID for the split trajectory
        if isinstance(base_id, (int, np.integer)):
            # If base_id is already an integer, use it as a base
            new_id = base_id + id_offset * (i + 1)
        else:
            # If base_id is a string, try to extract a numeric part
            try:
                # Try to convert to int if it's a digit string
                if isinstance(base_id, str) and base_id.isdigit():
                    new_id = int(base_id) + id_offset * (i + 1)
                else:
                    # For complex string IDs, create a completely new ID
                    new_id = 90000 + (i + 1)  # Use a high number range to avoid collisions
            except:
                # Fallback ID generation
                new_id = 90000 + (i + 1)
        
        sub_traj['cluster_id'] = new_id
        sub_traj['electrode_count'] = len(sub_coords)
        sub_traj['sorted_coords'] = sub_coords.tolist()
        
        # Store split information in metadata rather than in the ID
        sub_traj['is_split'] = True  # Flag to indicate this is a split trajectory
        sub_traj['split_from'] = base_id  # Store original trajectory ID
        sub_traj['split_index'] = i + 1  # Index of this split (1-based)
        sub_traj['split_label'] = f"S{i+1}_{base_id}"  # Descriptive label (not used for computation)
        
        # Calculate direction and endpoints
        direction = pca.components_[0]
        center = np.mean(sub_coords, axis=0)
        projected = np.dot(sub_coords - center, direction)
        sorted_indices = np.argsort(projected)
        
        sub_traj['endpoints'] = [
            sub_coords[sorted_indices[0]].tolist(),
            sub_coords[sorted_indices[-1]].tolist()
        ]
        
        # Update other properties
        sub_traj['direction'] = direction.tolist()
        sub_traj['center'] = center.tolist()
        sub_traj['length_mm'] = float(np.linalg.norm(
            sub_coords[sorted_indices[-1]] - sub_coords[sorted_indices[0]]
        ))
        
        split_trajectories.append(sub_traj)
    
    # Only consider the split successful if we created at least 2 sub-trajectories
    success = len(split_trajectories) >= 2
    
    return {
        'success': success,
        'reason': 'Split successful' if success else 'Failed to create multiple valid sub-trajectories',
        'trajectories': split_trajectories if success else [traj],
        'n_trajectories': len(split_trajectories) if success else 1
    }

def validate_trajectories(trajectories, expected_contact_counts, tolerance=2):
    """
    Validate trajectories against expected contact counts.
    
    Args:
        trajectories: List of trajectories
        expected_contact_counts: List of expected electrode contact counts
        tolerance: Maximum allowed deviation from expected counts
        
    Returns:
        dict: Validation results
    """
    validation = {
        'total': len(trajectories),
        'valid': 0,
        'invalid': 0,
        'valid_ids': [],
        'invalid_details': []
    }
    
    for traj in trajectories:
        count = traj['electrode_count']
        
        # Check if count is close to any expected count
        is_valid = any(abs(count - expected) <= tolerance for expected in expected_contact_counts)
        
        if is_valid:
            validation['valid'] += 1
            validation['valid_ids'].append(traj['cluster_id'])
        else:
            validation['invalid'] += 1
            closest = min(expected_contact_counts, key=lambda x: abs(x - count))
            validation['invalid_details'].append({
                'id': traj['cluster_id'],
                'count': count,
                'closest_expected': closest,
                'difference': abs(closest - count)
            })
    
    validation['valid_percentage'] = (validation['valid'] / validation['total'] * 100) if validation['total'] > 0 else 0
    
    return validation

#---------------------------------------------------------------------------------
# PART 2.7: HELPERS 
#---------------------------------------------------------------------------------
# Helper functions for trajectory refinement

def targeted_trajectory_refinement(trajectories, expected_contact_counts=[5, 8, 10, 12, 15, 18], 
                                 max_expected=20, tolerance=2):
    """
    Apply splitting and merging operations only to trajectories that need it.
    
    Args:
        trajectories: List of trajectory dictionaries
        expected_contact_counts: List of expected electrode contact counts
        max_expected: Maximum reasonable number of contacts in a single trajectory
        tolerance: How close to expected counts is considered valid
        
    Returns:
        dict: Results with refined trajectories and statistics
    """
    # Step 1: Flag trajectories that need attention
    merge_candidates = []  # Trajectories that might need to be merged (too few contacts)
    split_candidates = []  # Trajectories that might need to be split (too many contacts)
    valid_trajectories = []  # Trajectories that match expected counts
    
    # Group trajectories by contact count validity
    for traj in trajectories:
        contact_count = traj['electrode_count']
        
        # Find closest expected count
        closest_expected = min(expected_contact_counts, key=lambda x: abs(x - contact_count))
        difference = abs(closest_expected - contact_count)
        
        # Flag based on contact count
        if contact_count > max_expected:
            # Too many contacts - likely needs splitting
            traj['closest_expected'] = closest_expected
            traj['count_difference'] = difference
            split_candidates.append(traj)
        elif difference > tolerance and contact_count < closest_expected:
            # Too few contacts compared to expected - potential merge candidate
            traj['closest_expected'] = closest_expected
            traj['count_difference'] = difference
            traj['missing_contacts'] = closest_expected - contact_count
            merge_candidates.append(traj)
        else:
            # Contact count is valid
            valid_trajectories.append(traj)
    
    print(f"Initial classification:")
    print(f"- Valid trajectories: {len(valid_trajectories)}")
    print(f"- Potential merge candidates: {len(merge_candidates)}")
    print(f"- Potential split candidates: {len(split_candidates)}")
    
    # Step 2: Process merge candidates
    # Sort merge candidates by missing contact count (ascending)
    merge_candidates.sort(key=lambda x: x['missing_contacts'])
    
    merged_trajectories = []
    used_in_merge = set()  # Track which trajectories have been used in merges
    
    # For each merge candidate, look for other candidates to merge with
    for i, traj1 in enumerate(merge_candidates):
        if traj1['cluster_id'] in used_in_merge:
            continue
            
        best_match = None
        best_score = float('inf')
        
        for j, traj2 in enumerate(merge_candidates):
            if i == j or traj2['cluster_id'] in used_in_merge:
                continue
                
            # Check if merging would result in a valid count
            combined_count = traj1['electrode_count'] + traj2['electrode_count']
            closest_expected = min(expected_contact_counts, key=lambda x: abs(x - combined_count))
            combined_difference = abs(closest_expected - combined_count)
            
            # Only consider merging if it improves the count validity
            if combined_difference < min(traj1['count_difference'], traj2['count_difference']):
                # Check spatial compatibility
                score = check_merge_compatibility(traj1, traj2)
                if score is not None and score < best_score:
                    best_match = (j, traj2, score)
                    best_score = score
        
        # If we found a good match, merge them
        if best_match:
            j, traj2, score = best_match
            merged_traj = merge_trajectories(traj1, traj2)
            merged_trajectories.append(merged_traj)
            used_in_merge.add(traj1['cluster_id'])
            used_in_merge.add(traj2['cluster_id'])
            print(f"Merged: {traj1['cluster_id']} + {traj2['cluster_id']} = {merged_traj['cluster_id']}")
        else:
            # No good match, keep original
            merged_trajectories.append(traj1)
    
    # Add any merge candidates that weren't used
    for traj in merge_candidates:
        if traj['cluster_id'] not in used_in_merge:
            merged_trajectories.append(traj)
    
    # Step 3: Process split candidates
    final_trajectories = []
    
    # Process each split candidate
    for traj in split_candidates:
        # Try to split the trajectory
        split_result = split_trajectory(traj, expected_contact_counts)
        
        if split_result['success']:
            # Add the split trajectories
            final_trajectories.extend(split_result['trajectories'])
            print(f"Split {traj['cluster_id']} into {len(split_result['trajectories'])} trajectories")
        else:
            # Couldn't split effectively, keep original
            final_trajectories.append(traj)
    
    # Add all merged trajectories and valid trajectories
    final_trajectories.extend(merged_trajectories)
    final_trajectories.extend(valid_trajectories)
    
    # Step 4: Final validation
    validation_results = validate_trajectories(final_trajectories, expected_contact_counts, tolerance)
    
    return {
        'trajectories': final_trajectories,
        'n_trajectories': len(final_trajectories),
        'original_count': len(trajectories),
        'valid_count': len(valid_trajectories),
        'merge_candidates': len(merge_candidates),
        'split_candidates': len(split_candidates),
        'merged_count': len([t for t in final_trajectories if 'merged_from' in t]),
        'split_count': len([t for t in final_trajectories if 'split_from' in t]),
        'validation': validation_results
    }

def check_merge_compatibility(traj1, traj2, max_distance=15, max_angle_diff=20):
    """
    Check if two trajectories can be merged by examining their spatial relationship.
    
    Args:
        traj1, traj2: Trajectory dictionaries
        max_distance: Maximum distance between endpoints to consider merging
        max_angle_diff: Maximum angle difference between directions (degrees)
        
    Returns:
        float: Compatibility score (lower is better) or None if incompatible
    """
    # Get endpoints
    endpoints1 = np.array(traj1['endpoints'])
    endpoints2 = np.array(traj2['endpoints'])
    
    # Calculate distances between all endpoint combinations
    distances = [
        np.linalg.norm(endpoints1[0] - endpoints2[0]),
        np.linalg.norm(endpoints1[0] - endpoints2[1]),
        np.linalg.norm(endpoints1[1] - endpoints2[0]),
        np.linalg.norm(endpoints1[1] - endpoints2[1])
    ]
    
    min_distance = min(distances)
    
    # If endpoints are too far apart, not compatible
    if min_distance > max_distance:
        return None
    
    # Check angle between trajectory directions
    dir1 = np.array(traj1['direction'])
    dir2 = np.array(traj2['direction'])
    
    angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(dir1, dir2)), -1.0, 1.0)))
    
    # If direction vectors point in opposite directions, we need 180-angle
    if np.dot(dir1, dir2) < 0:
        angle = 180 - angle
    
    # If trajectories have very different directions, not compatible
    if angle > max_angle_diff:
        return None
    
    # Compute compatibility score (lower is better)
    score = min_distance + angle * 0.5
    
    return score

def merge_trajectories(traj1, traj2):
    """
    Merge two trajectories into one.
    
    Args:
        traj1, traj2: Trajectory dictionaries
        
    Returns:
        dict: Merged trajectory
    """
    # Create a new trajectory 
    merged_traj = traj1.copy()
    
    # Set new ID
    merged_traj['cluster_id'] = f"M_{traj1['cluster_id']}_{traj2['cluster_id']}"
    
    # Get endpoints
    endpoints1 = np.array(traj1['endpoints'])
    endpoints2 = np.array(traj2['endpoints'])
    
    # Calculate distances between all endpoint combinations
    distances = [
        (0, 0, np.linalg.norm(endpoints1[0] - endpoints2[0])),
        (0, 1, np.linalg.norm(endpoints1[0] - endpoints2[1])),
        (1, 0, np.linalg.norm(endpoints1[1] - endpoints2[0])),
        (1, 1, np.linalg.norm(endpoints1[1] - endpoints2[1]))
    ]
    
    # Find which endpoints are closest
    closest_pair = min(distances, key=lambda x: x[2])
    idx1, idx2, _ = closest_pair
    
    # Update endpoints to span the full merged trajectory
    if idx1 == 0 and idx2 == 0:
        new_endpoints = [endpoints1[1], endpoints2[1]]
    elif idx1 == 0 and idx2 == 1:
        new_endpoints = [endpoints1[1], endpoints2[0]]
    elif idx1 == 1 and idx2 == 0:
        new_endpoints = [endpoints1[0], endpoints2[1]]
    else:  # idx1 == 1 and idx2 == 1
        new_endpoints = [endpoints1[0], endpoints2[0]]
    
    merged_traj['endpoints'] = [new_endpoints[0].tolist(), new_endpoints[1].tolist()]
    
    # Recalculate direction and length
    new_direction = new_endpoints[1] - new_endpoints[0]
    new_length = np.linalg.norm(new_direction)
    
    merged_traj['direction'] = (new_direction / new_length).tolist()
    merged_traj['length_mm'] = float(new_length)
    merged_traj['electrode_count'] = traj1['electrode_count'] + traj2['electrode_count']
    merged_traj['center'] = ((new_endpoints[0] + new_endpoints[1]) / 2).tolist()
    
    # Combine sorted coordinates if available
    if 'sorted_coords' in traj1 and 'sorted_coords' in traj2:
        # This is a simplification - you'd need to ensure proper ordering here
        if idx1 == 0 and idx2 == 0:
            sorted_coords = np.array(traj1['sorted_coords'])[::-1].tolist() + np.array(traj2['sorted_coords']).tolist()
        elif idx1 == 0 and idx2 == 1:
            sorted_coords = np.array(traj1['sorted_coords'])[::-1].tolist() + np.array(traj2['sorted_coords'])[::-1].tolist()
        elif idx1 == 1 and idx2 == 0:
            sorted_coords = np.array(traj1['sorted_coords']).tolist() + np.array(traj2['sorted_coords']).tolist()
        else:  # idx1 == 1 and idx2 == 1
            sorted_coords = np.array(traj1['sorted_coords']).tolist() + np.array(traj2['sorted_coords'])[::-1].tolist()
            
        merged_traj['sorted_coords'] = sorted_coords
    
    # Mark as merged and record original trajectories
    merged_traj['merged_from'] = [traj1['cluster_id'], traj2['cluster_id']]
    
    return merged_traj

def split_trajectory(traj, expected_contact_counts=[5, 8, 10, 12, 15, 18]):
    """
    Split a trajectory that may contain multiple electrodes.
    
    Args:
        traj: Trajectory dictionary
        expected_contact_counts: List of expected electrode contact counts
        
    Returns:
        dict: Split results with success flag and trajectories
    """
    # Need coordinates for this trajectory
    if 'sorted_coords' not in traj:
        return {'success': False, 'reason': 'No coordinates available', 'trajectories': [traj]}
    
    coords = np.array(traj['sorted_coords'])
    
    # Try to determine how many sub-trajectories to create
    contact_count = traj['electrode_count']
    
    # Find potential combinations of expected counts that would sum close to our count
    best_combination = None
    min_difference = float('inf')
    
    # Try combinations of 2 trajectories
    for count1 in expected_contact_counts:
        for count2 in expected_contact_counts:
            if abs((count1 + count2) - contact_count) < min_difference:
                min_difference = abs((count1 + count2) - contact_count)
                best_combination = [count1, count2]
    
    # If needed, try combinations of 3 trajectories
    if min_difference > 3:  # If 2-trajectory combo isn't close enough
        for count1 in expected_contact_counts:
            for count2 in expected_contact_counts:
                for count3 in expected_contact_counts:
                    diff = abs((count1 + count2 + count3) - contact_count)
                    if diff < min_difference:
                        min_difference = diff
                        best_combination = [count1, count2, count3]
    
    # No good combination found
    if best_combination is None or min_difference > 5:
        return {'success': False, 'reason': 'No good contact count combination found', 'trajectories': [traj]}
    
    # Determine if we should use DBSCAN or K-means for splitting
    if len(best_combination) <= 2:
        # For 2 clusters, try DBSCAN with adjusted parameters
        from sklearn.cluster import DBSCAN
        
        # Estimate good eps value based on contact spacing
        if len(coords) > 1:
            distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            median_spacing = np.median(distances)
            splitting_eps = median_spacing * 1.5  # A bit larger than typical spacing
        else:
            splitting_eps = 5.0  # Default if we can't estimate
        
        # Apply DBSCAN
        splitter = DBSCAN(eps=splitting_eps, min_samples=3)
        labels = splitter.fit_predict(coords)
        
        # Check if splitting produced meaningful results
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
            
        if len(unique_labels) < 2:
            # DBSCAN failed, try K-means as backup
            use_kmeans = True
        else:
            use_kmeans = False
    else:
        # For 3+ clusters, use K-means directly
        use_kmeans = True
    
    # Apply K-means if needed
    if use_kmeans:
        from sklearn.cluster import KMeans
        
        n_clusters = len(best_combination)
        splitter = KMeans(n_clusters=n_clusters)
        labels = splitter.fit_predict(coords)
        unique_labels = set(labels)
    
    # Create sub-trajectories
    split_trajectories = []
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        sub_coords = coords[mask]
        
        if len(sub_coords) < 3:  # Need at least 3 points
            continue
            
        # Create new trajectory
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca.fit(sub_coords)
        
        sub_traj = traj.copy()
        sub_traj['cluster_id'] = f"S{i+1}_{traj['cluster_id']}"
        sub_traj['electrode_count'] = len(sub_coords)
        sub_traj['sorted_coords'] = sub_coords.tolist()
        
        # Calculate direction and endpoints
        direction = pca.components_[0]
        center = np.mean(sub_coords, axis=0)
        projected = np.dot(sub_coords - center, direction)
        sorted_indices = np.argsort(projected)
        
        sub_traj['endpoints'] = [
            sub_coords[sorted_indices[0]].tolist(),
            sub_coords[sorted_indices[-1]].tolist()
        ]
        
        # Update other properties
        sub_traj['direction'] = direction.tolist()
        sub_traj['center'] = center.tolist()
        sub_traj['length_mm'] = float(np.linalg.norm(
            sub_coords[sorted_indices[-1]] - sub_coords[sorted_indices[0]]
        ))
        
        # Mark as split and record original trajectory
        sub_traj['split_from'] = traj['cluster_id']
        
        split_trajectories.append(sub_traj)
    
    # Only consider the split successful if we created at least 2 sub-trajectories
    success = len(split_trajectories) >= 2
    
    return {
        'success': success,
        'reason': 'Split successful' if success else 'Failed to create multiple valid sub-trajectories',
        'trajectories': split_trajectories if success else [traj],
        'n_trajectories': len(split_trajectories) if success else 1
    }

def validate_trajectories(trajectories, expected_contact_counts, tolerance=2):
    """
    Validate trajectories against expected contact counts.
    
    Args:
        trajectories: List of trajectories
        expected_contact_counts: List of expected electrode contact counts
        tolerance: Maximum allowed deviation from expected counts
        
    Returns:
        dict: Validation results
    """
    validation = {
        'total': len(trajectories),
        'valid': 0,
        'invalid': 0,
        'valid_ids': [],
        'invalid_details': []
    }
    
    for traj in trajectories:
        count = traj['electrode_count']
        
        # Check if count is close to any expected count
        is_valid = any(abs(count - expected) <= tolerance for expected in expected_contact_counts)
        
        if is_valid:
            validation['valid'] += 1
            validation['valid_ids'].append(traj['cluster_id'])
        else:
            validation['invalid'] += 1
            closest = min(expected_contact_counts, key=lambda x: abs(x - count))
            validation['invalid_details'].append({
                'id': traj['cluster_id'],
                'count': count,
                'closest_expected': closest,
                'difference': abs(closest - count)
            })
    
    validation['valid_percentage'] = (validation['valid'] / validation['total'] * 100) if validation['total'] > 0 else 0
    
    return validation

def visualize_trajectory_refinement(coords_array, original_trajectories, refined_trajectories, refinement_results):
    """
    Create visualization showing the results of trajectory refinement.
    
    Args:
        coords_array: Array of all electrode coordinates
        original_trajectories: List of trajectories before refinement
        refined_trajectories: List of trajectories after refinement
        refinement_results: Results from targeted_trajectory_refinement
        
    Returns:
        matplotlib.figure.Figure: Visualization showing before and after refinement
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('Trajectory Refinement Results', fontsize=16)
    
    # Create before/after 3D plots
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot electrodes as background in both plots
    ax1.scatter(coords_array[:, 0], coords_array[:, 1], coords_array[:, 2], 
               c='lightgray', marker='.', s=5, alpha=0.2)
    ax2.scatter(coords_array[:, 0], coords_array[:, 1], coords_array[:, 2], 
               c='lightgray', marker='.', s=5, alpha=0.2)
    
    # Plot original trajectories
    for i, traj in enumerate(original_trajectories):
        color = plt.cm.tab20(i % 20)
        endpoints = np.array(traj['endpoints'])
        
        # Highlight trajectories that were candidates for refinement
        is_split_candidate = traj.get('electrode_count', 0) > refinement_results.get('max_expected', 20)
        is_merge_candidate = 'missing_contacts' in traj
        
        if is_split_candidate:
            marker_style = '*'
            linewidth = 3
            alpha = 0.9
            s = 100
        elif is_merge_candidate:
            marker_style = 's'
            linewidth = 2
            alpha = 0.8
            s = 80
        else:
            marker_style = 'o'
            linewidth = 1
            alpha = 0.7
            s = 50
        
        # Plot endpoints
        ax1.scatter(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], 
                   color=color, marker=marker_style, s=s, alpha=alpha)
        
        # Plot trajectory line
        ax1.plot(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], 
               '-', color=color, linewidth=linewidth, alpha=alpha, 
               label=f"ID {traj['cluster_id']} ({traj['electrode_count']} contacts)")
        
        # Add label with contact count
        midpoint = np.mean(endpoints, axis=0)
        ax1.text(midpoint[0], midpoint[1], midpoint[2], 
               f"{traj['electrode_count']}", color=color, fontsize=8)
    
    # Plot refined trajectories
    for i, traj in enumerate(refined_trajectories):
        color = plt.cm.tab20(i % 20)
        endpoints = np.array(traj['endpoints'])
        
        # Use different markers for merged or split trajectories
        if 'merged_from' in traj:
            marker_style = '^'
            linewidth = 3
            alpha = 0.9
            s = 100
            label = f"Merged: {traj['cluster_id']} ({traj['electrode_count']} contacts)"
        elif 'split_from' in traj:
            marker_style = '*'
            linewidth = 3
            alpha = 0.9
            s = 100
            label = f"Split: {traj['cluster_id']} ({traj['electrode_count']} contacts)"
        else:
            marker_style = 'o'
            linewidth = 1
            alpha = 0.7
            s = 50
            label = f"ID {traj['cluster_id']} ({traj['electrode_count']} contacts)"
        
        # Plot endpoints
        ax2.scatter(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], 
                   color=color, marker=marker_style, s=s, alpha=alpha)
        
        # Plot trajectory line
        ax2.plot(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], 
               '-', color=color, linewidth=linewidth, alpha=alpha, label=label)
        
        # Add label with contact count
        midpoint = np.mean(endpoints, axis=0)
        ax2.text(midpoint[0], midpoint[1], midpoint[2], 
               f"{traj['electrode_count']}", color=color, fontsize=8)
    
    # Add titles and labels
    ax1.set_title(f"Before Refinement\n({len(original_trajectories)} trajectories)")
    ax2.set_title(f"After Refinement\n({len(refined_trajectories)} trajectories, "
                 f"{refinement_results['merged_count']} merged, {refinement_results['split_count']} split)")
    
    for ax in [ax1, ax2]:
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        # Set view angle
        ax.view_init(elev=20, azim=30)
    
    # Add summary statistics as text
    stats_text = (
        f"Refinement Summary:\n"
        f"- Original: {len(original_trajectories)} trajectories\n"
        f"- Final: {len(refined_trajectories)} trajectories\n"
        f"- Merged: {refinement_results['merged_count']}\n"
        f"- Split: {refinement_results['split_count']}\n"
        f"- Valid before: {refinement_results['valid_count']}\n"
        f"- Valid after: {refinement_results['validation']['valid']}"
    )
    
    fig.text(0.01, 0.05, stats_text, fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    return fig

#------------------------------------------------------------------------------
# PART 2.8: VALIDATION ENTRY 
#------------------------------------------------------------------------------
def validate_entry_angles(bolt_directions, min_angle=25, max_angle=60):
    """
    Validate entry angles against the standard surgical planning range.
    
    In SEEG surgery planning, the bolt head and entry point typically form
    an angle of 30-60 degrees with the skull surface normal. This function
    validates the calculated entry angles against this expected range.
    
    Args:
        bolt_directions (dict): Direction info from extract_bolt_entry_directions
        min_angle (float): Minimum expected angle in degrees (default: 30)
        max_angle (float): Maximum expected angle in degrees (default: 60)
        
    Returns:
        dict: Dictionary with validation results for each bolt
    """
    import numpy as np
    
    validation_results = {
        'bolts': {},
        'summary': {
            'total_bolts': len(bolt_directions),
            'valid_count': 0,
            'invalid_count': 0,
            'below_min': 0,
            'above_max': 0
        }
    }
    
    if not bolt_directions:
        return validation_results
    
    # Validate each bolt direction
    for bolt_id, bolt_info in bolt_directions.items():
        # Extract direction vector
        direction = np.array(bolt_info['direction'])
        
        # Approximate the skull normal as perpendicular to bolt direction
        # In a more accurate implementation, we would use the actual skull surface normal
        # from a segmented skull mesh, but for this validation we'll approximate
        
        # For simplicity, we'll find vectors perpendicular to the trajectory
        # and use the average as an approximate normal
        # Create two orthogonal vectors to the direction
        v1 = np.array([1, 0, 0])
        if np.abs(np.dot(direction, v1)) > 0.9:
            # If direction is close to x-axis, use y-axis as reference
            v1 = np.array([0, 1, 0])
        
        v2 = np.cross(direction, v1)
        v2 = v2 / np.linalg.norm(v2)  # Normalize
        v1 = np.cross(v2, direction)
        v1 = v1 / np.linalg.norm(v1)  # Normalize
        
        # Use average of multiple normals to approximate skull normal
        normals = []
        for theta in np.linspace(0, 2*np.pi, 8):
            # Generate normals around the trajectory
            normal = v1 * np.cos(theta) + v2 * np.sin(theta)
            normals.append(normal)
        
        # Calculate angles with each normal
        angles = []
        for normal in normals:
            # Calculate angle between direction and normal
            # For a perpendicular normal, this would be 90°
            # Entry angles of 30-60° would make this 30-60° away from 90°
            cos_angle = np.abs(np.dot(direction, normal))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            angles.append(angle)
        
        # Use the smallest angle for validation
        # This is most conservative approach - if any normal is within range
        min_normal_angle = min(angles)
        
        # For a trajectory perpendicular to skull, angle with normal would be 90°
        # Adjust to get entry angle relative to skull surface
        entry_angle = 90 - min_normal_angle if min_normal_angle < 90 else min_normal_angle - 90
        
        # Validate against expected range
        is_valid = min_angle <= entry_angle <= max_angle
        
        validation_results['bolts'][bolt_id] = {
            'entry_angle': float(entry_angle),
            'valid': is_valid,
            'status': 'valid' if is_valid else 'invalid',
            'direction': direction.tolist()
        }
        
        # Update summary statistics
        if is_valid:
            validation_results['summary']['valid_count'] += 1
        else:
            validation_results['summary']['invalid_count'] += 1
            if entry_angle < min_angle:
                validation_results['summary']['below_min'] += 1
            else:
                validation_results['summary']['above_max'] += 1
    
    # Calculate percentage
    total = validation_results['summary']['total_bolts']
    if total > 0:
        valid = validation_results['summary']['valid_count']
        validation_results['summary']['valid_percentage'] = (valid / total) * 100
    else:
        validation_results['summary']['valid_percentage'] = 0
    
    return validation_results

#------------------------------------------------------------------------------
# PART 2.9: HEMISPHERE FILTERING FUNCTIONS
#------------------------------------------------------------------------------

def filter_coordinates_by_hemisphere(coords_array, hemisphere='left', verbose=True):
    """
    Filter electrode coordinates by hemisphere.
    
    Args:
        coords_array (numpy.ndarray): Array of coordinates in RAS format [N, 3]
        hemisphere (str): 'left' (x < 0), 'right' (x > 0), or 'both' (no filtering)
        verbose (bool): Whether to print filtering results
        
    Returns:
        tuple: (filtered_coords, hemisphere_mask, filtered_indices)
            - filtered_coords: Coordinates in the specified hemisphere
            - hemisphere_mask: Boolean mask indicating which points are in hemisphere
            - filtered_indices: Original indices of the filtered points
    """
    import numpy as np
    
    if hemisphere.lower() == 'both':
        if verbose:
            print(f"No hemisphere filtering applied. Keeping all {len(coords_array)} coordinates.")
        return coords_array, np.ones(len(coords_array), dtype=bool), np.arange(len(coords_array))
    
    # Create hemisphere mask based on RAS x-coordinate
    if hemisphere.lower() == 'left':
        hemisphere_mask = coords_array[:, 0] < 0  # RAS_x < 0 is left
        hemisphere_name = "left"
    elif hemisphere.lower() == 'right':
        hemisphere_mask = coords_array[:, 0] > 0  # RAS_x > 0 is right
        hemisphere_name = "right"
    else:
        raise ValueError("hemisphere must be 'left', 'right', or 'both'")
    
    # Apply filter
    filtered_coords = coords_array[hemisphere_mask]
    filtered_indices = np.where(hemisphere_mask)[0]
    
    if verbose:
        original_count = len(coords_array)
        filtered_count = len(filtered_coords)
        discarded_count = original_count - filtered_count
        
        print(f"Hemisphere filtering results ({hemisphere_name}):")
        print(f"- Original coordinates: {original_count}")
        print(f"- Coordinates in {hemisphere_name} hemisphere: {filtered_count}")
        print(f"- Discarded coordinates: {discarded_count}")
        print(f"- Filtering efficiency: {filtered_count/original_count*100:.1f}%")
        
        if discarded_count > 0:
            discarded_coords = coords_array[~hemisphere_mask]
            x_range = f"[{discarded_coords[:, 0].min():.1f}, {discarded_coords[:, 0].max():.1f}]"
            print(f"- Discarded coordinates x-range: {x_range}")
    
    return filtered_coords, hemisphere_mask, filtered_indices

def filter_trajectories_by_hemisphere(trajectories, hemisphere='left', verbose=True):
    """
    Filter trajectories by hemisphere based on their center points.
    
    Args:
        trajectories (list): List of trajectory dictionaries
        hemisphere (str): 'left' (x < 0), 'right' (x > 0), or 'both' (no filtering)
        verbose (bool): Whether to print filtering results
        
    Returns:
        tuple: (filtered_trajectories, hemisphere_mask)
    """
    import numpy as np
    
    if not trajectories:
        return [], np.array([])
    
    if hemisphere.lower() == 'both':
        if verbose:
            print(f"No hemisphere filtering applied to trajectories. Keeping all {len(trajectories)}.")
        return trajectories, np.ones(len(trajectories), dtype=bool)
    
    hemisphere_mask = []
    
    for traj in trajectories:
        # Use trajectory center for hemisphere determination
        center = np.array(traj['center'])
        
        if hemisphere.lower() == 'left':
            in_hemisphere = center[0] < 0  # RAS_x < 0 is left
        elif hemisphere.lower() == 'right':
            in_hemisphere = center[0] > 0  # RAS_x > 0 is right
        else:
            raise ValueError("hemisphere must be 'left', 'right', or 'both'")
        
        hemisphere_mask.append(in_hemisphere)
    
    hemisphere_mask = np.array(hemisphere_mask)
    filtered_trajectories = [traj for i, traj in enumerate(trajectories) if hemisphere_mask[i]]
    
    if verbose:
        original_count = len(trajectories)
        filtered_count = len(filtered_trajectories)
        hemisphere_name = hemisphere.lower()
        
        print(f"Trajectory hemisphere filtering results ({hemisphere_name}):")
        print(f"- Original trajectories: {original_count}")
        print(f"- Trajectories in {hemisphere_name} hemisphere: {filtered_count}")
        print(f"- Discarded trajectories: {original_count - filtered_count}")
    
    return filtered_trajectories, hemisphere_mask

def filter_bolt_directions_by_hemisphere(bolt_directions, hemisphere='left', verbose=True):
    """
    Filter bolt directions by hemisphere based on their start points.
    
    Args:
        bolt_directions (dict): Dictionary of bolt direction info
        hemisphere (str): 'left' (x < 0), 'right' (x > 0), or 'both' (no filtering)
        verbose (bool): Whether to print filtering results
        
    Returns:
        dict: Filtered bolt directions dictionary
    """
    import numpy as np
    
    if not bolt_directions:
        return {}
    
    if hemisphere.lower() == 'both':
        if verbose:
            print(f"No hemisphere filtering applied to bolt directions. Keeping all {len(bolt_directions)}.")
        return bolt_directions
    
    filtered_bolt_directions = {}
    
    for bolt_id, bolt_info in bolt_directions.items():
        start_point = np.array(bolt_info['start_point'])
        
        if hemisphere.lower() == 'left':
            in_hemisphere = start_point[0] < 0  # RAS_x < 0 is left
        elif hemisphere.lower() == 'right':
            in_hemisphere = start_point[0] > 0  # RAS_x > 0 is right
        else:
            raise ValueError("hemisphere must be 'left', 'right', or 'both'")
        
        if in_hemisphere:
            filtered_bolt_directions[bolt_id] = bolt_info
    
    if verbose:
        original_count = len(bolt_directions)
        filtered_count = len(filtered_bolt_directions)
        hemisphere_name = hemisphere.lower()
        
        print(f"Bolt directions hemisphere filtering results ({hemisphere_name}):")
        print(f"- Original bolt directions: {original_count}")
        print(f"- Bolt directions in {hemisphere_name} hemisphere: {filtered_count}")
        print(f"- Discarded bolt directions: {original_count - filtered_count}")
    
    return filtered_bolt_directions

def apply_hemisphere_filtering_to_results(results, coords_array, hemisphere='left', verbose=True):
    """
    Apply hemisphere filtering to all analysis results.
    
    Args:
        results (dict): Results dictionary from trajectory analysis
        coords_array (numpy.ndarray): Original coordinate array
        hemisphere (str): 'left', 'right', or 'both'
        verbose (bool): Whether to print filtering results
        
    Returns:
        tuple: (filtered_results, filtered_coords, hemisphere_info)
    """
    import numpy as np
    import copy
    
    if hemisphere.lower() == 'both':
        if verbose:
            print("No hemisphere filtering requested.")
        return results, coords_array, {'hemisphere': 'both', 'filtering_applied': False}
    
    print(f"\n=== Applying {hemisphere.upper()} Hemisphere Filtering ===")
    
    # Filter coordinates
    filtered_coords, coord_mask, filtered_indices = filter_coordinates_by_hemisphere(
        coords_array, hemisphere, verbose
    )
    
    # Create a deep copy of results to avoid modifying original
    filtered_results = copy.deepcopy(results)
    
    # Update coordinate-dependent results
    if 'dbscan' in filtered_results:
        # Update noise points coordinates
        if 'noise_points_coords' in filtered_results['dbscan']:
            original_noise = np.array(filtered_results['dbscan']['noise_points_coords'])
            if len(original_noise) > 0:
                # Filter noise points by hemisphere
                if hemisphere.lower() == 'left':
                    noise_mask = original_noise[:, 0] < 0
                else:  # right
                    noise_mask = original_noise[:, 0] > 0
                
                filtered_noise = original_noise[noise_mask]
                filtered_results['dbscan']['noise_points_coords'] = filtered_noise.tolist()
                filtered_results['dbscan']['noise_points'] = len(filtered_noise)
    
    # Filter trajectories
    if 'trajectories' in filtered_results:
        filtered_trajectories, traj_mask = filter_trajectories_by_hemisphere(
            filtered_results['trajectories'], hemisphere, verbose
        )
        filtered_results['trajectories'] = filtered_trajectories
        filtered_results['n_trajectories'] = len(filtered_trajectories)
    
    # Filter bolt directions if present
    if 'bolt_directions' in results:
        filtered_bolt_directions = filter_bolt_directions_by_hemisphere(
            results['bolt_directions'], hemisphere, verbose
        )
        filtered_results['bolt_directions'] = filtered_bolt_directions
    
    # Filter combined volume trajectories if present
    if 'combined_volume' in results and 'trajectories' in results['combined_volume']:
        original_combined = results['combined_volume']['trajectories']
        filtered_combined = {}
        
        for bolt_id, traj_info in original_combined.items():
            start_point = np.array(traj_info['start_point'])
            
            if hemisphere.lower() == 'left':
                in_hemisphere = start_point[0] < 0
            else:  # right
                in_hemisphere = start_point[0] > 0
            
            if in_hemisphere:
                filtered_combined[bolt_id] = traj_info
        
        filtered_results['combined_volume']['trajectories'] = filtered_combined
        filtered_results['combined_volume']['trajectory_count'] = len(filtered_combined)
        
        if verbose:
            print(f"Combined volume hemisphere filtering:")
            print(f"- Original combined trajectories: {len(original_combined)}")
            print(f"- Filtered combined trajectories: {len(filtered_combined)}")
    
    # Update electrode validation if present
    if 'electrode_validation' in filtered_results:
        # Recalculate validation for filtered trajectories
        if 'trajectories' in filtered_results:
            expected_counts = results.get('parameters', {}).get('expected_contact_counts', [5, 8, 10, 12, 15, 18])
            validation = validate_electrode_clusters(filtered_results, expected_counts)
            filtered_results['electrode_validation'] = validation
    
    # Create hemisphere info
    hemisphere_info = {
        'hemisphere': hemisphere,
        'filtering_applied': True,
        'original_coords': len(coords_array),
        'filtered_coords': len(filtered_coords),
        'filtering_efficiency': len(filtered_coords) / len(coords_array) * 100,
        'coord_mask': coord_mask,
        'filtered_indices': filtered_indices
    }
    
    # Add hemisphere info to filtered results
    filtered_results['hemisphere_filtering'] = hemisphere_info
    
    print(f"✅ Hemisphere filtering complete. Results updated for {hemisphere} hemisphere.")
    
    return filtered_results, filtered_coords, hemisphere_info

def create_hemisphere_comparison_visualization(coords_array, results, hemisphere_results, hemisphere='left'):
    """
    Create a visualization comparing original vs hemisphere-filtered results.
    
    Args:
        coords_array (numpy.ndarray): Original coordinates
        results (dict): Original analysis results
        hemisphere_results (dict): Hemisphere-filtered results
        hemisphere (str): Which hemisphere was filtered
        
    Returns:
        matplotlib.figure.Figure: Comparison visualization
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f'Hemisphere Filtering Comparison: {hemisphere.upper()} Hemisphere Only', fontsize=16)
    
    # Original results (left plot)
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot all original coordinates
    ax1.scatter(coords_array[:, 0], coords_array[:, 1], coords_array[:, 2], 
               c='lightgray', marker='.', s=10, alpha=0.3)
    
    # Highlight hemisphere boundary
    if hemisphere.lower() == 'left':
        hemisphere_coords = coords_array[coords_array[:, 0] < 0]
        other_coords = coords_array[coords_array[:, 0] >= 0]
        boundary_x = 0
    else:
        hemisphere_coords = coords_array[coords_array[:, 0] > 0]
        other_coords = coords_array[coords_array[:, 0] <= 0]
        boundary_x = 0
    
    # Plot hemisphere coordinates in color
    ax1.scatter(hemisphere_coords[:, 0], hemisphere_coords[:, 1], hemisphere_coords[:, 2], 
               c='blue', marker='o', s=20, alpha=0.7, label=f'{hemisphere.title()} hemisphere')
    
    # Plot other hemisphere coordinates in gray
    if len(other_coords) > 0:
        ax1.scatter(other_coords[:, 0], other_coords[:, 1], other_coords[:, 2], 
                   c='red', marker='x', s=15, alpha=0.5, label='Other hemisphere (discarded)')
    
    # Add hemisphere boundary plane
    y_range = [coords_array[:, 1].min(), coords_array[:, 1].max()]
    z_range = [coords_array[:, 2].min(), coords_array[:, 2].max()]
    Y, Z = np.meshgrid(y_range, z_range)
    X = np.full_like(Y, boundary_x)
    ax1.plot_surface(X, Y, Z, alpha=0.2, color='yellow')
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'Original Analysis\n({len(coords_array)} coordinates)')
    ax1.legend()
    
    # Filtered results (right plot)
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Get filtered coordinates and trajectories
    hemisphere_info = hemisphere_results.get('hemisphere_filtering', {})
    filtered_coords = coords_array[hemisphere_info.get('coord_mask', np.ones(len(coords_array), dtype=bool))]
    
    # Plot filtered coordinates
    ax2.scatter(filtered_coords[:, 0], filtered_coords[:, 1], filtered_coords[:, 2], 
               c='blue', marker='o', s=20, alpha=0.7)
    
    # Plot filtered trajectories if available
    if 'trajectories' in hemisphere_results:
        for i, traj in enumerate(hemisphere_results['trajectories']):
            endpoints = np.array(traj['endpoints'])
            color = plt.cm.tab20(i % 20)
            
            # Plot trajectory line
            ax2.plot([endpoints[0][0], endpoints[1][0]],
                    [endpoints[0][1], endpoints[1][1]],
                    [endpoints[0][2], endpoints[1][2]],
                    '-', color=color, linewidth=2, alpha=0.8)
            
            # Plot endpoints
            ax2.scatter(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2],
                       color=color, marker='*', s=100, alpha=0.9)
    
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    ax2.set_title(f'Filtered Analysis ({hemisphere.title()} Hemisphere)\n'
                 f'({len(filtered_coords)} coordinates, '
                 f'{hemisphere_results.get("n_trajectories", 0)} trajectories)')
    
    # Add summary statistics
    original_trajectories = len(results.get('trajectories', []))
    filtered_trajectories = len(hemisphere_results.get('trajectories', []))
    
    stats_text = (
        f"Filtering Results:\n"
        f"Coordinates: {len(coords_array)} → {len(filtered_coords)} "
        f"({hemisphere_info.get('filtering_efficiency', 0):.1f}%)\n"
        f"Trajectories: {original_trajectories} → {filtered_trajectories}\n"
        f"Hemisphere: {hemisphere.title()} (x {'< 0' if hemisphere.lower() == 'left' else '> 0'})"
    )
    
    fig.text(0.02, 0.02, stats_text, fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


#------------------------------------------------------------------------------
# PART 3: VISUALIZATION FUNCTIONS
#------------------------------------------------------------------------------

def visualize_bolt_entry_directions(ax, bolt_directions, matches=None, arrow_length=10):
    """
    Add bolt+entry directions to a 3D plot.
    
    Args:
        ax: Matplotlib 3D axis
        bolt_directions (dict): Direction info from extract_bolt_entry_directions
        matches (dict, optional): Mapping from match_bolt_directions_to_trajectories
        arrow_length (float): Length of the direction arrows
    """
    # Create a colormap for unmatched bolt directions
    n_bolts = len(bolt_directions)
    if n_bolts == 0:
        return
        
    bolt_cmap = plt.cm.Paired(np.linspace(0, 1, max(n_bolts, 2)))
    
    for i, (bolt_id, bolt_info) in enumerate(bolt_directions.items()):
        start_point = np.array(bolt_info['start_point'])
        direction = np.array(bolt_info['direction'])
        
        # Use matched color if this bolt is matched to a trajectory
        color = bolt_cmap[i % len(bolt_cmap)]
        is_matched = False
        if matches:
            for traj_id, match in matches.items():
                # Handle comparison properly for different types
                try:
                    if str(match['bolt_id']) == str(bolt_id):
                        color = 'crimson'  # Use a distinct color for matched bolts
                        is_matched = True
                        break
                except:
                    continue
        
        # Plot the bolt+entry points
        if 'points' in bolt_info:
            bolt_points = np.array(bolt_info['points'])
            ax.scatter(bolt_points[:, 0], bolt_points[:, 1], bolt_points[:, 2], 
                      color=color, marker='.', s=30, alpha=0.5)
        
        # Plot the direction arrow
        arrow = Arrow3D(
            [start_point[0], start_point[0] + direction[0] * arrow_length],
            [start_point[1], start_point[1] + direction[1] * arrow_length],
            [start_point[2], start_point[2] + direction[2] * arrow_length],
            mutation_scale=15, lw=2, arrowstyle="-|>", color=color)
        ax.add_artist(arrow)
        
        # Add a label to the start point
        label = f"Bolt {bolt_id}"
        if is_matched:
            label += " (matched)"
        ax.text(start_point[0], start_point[1], start_point[2], label, 
               color=color, fontsize=8)
        
        # If we have end point, draw line from start to end
        if 'end_point' in bolt_info:
            end_point = np.array(bolt_info['end_point'])
            ax.plot([start_point[0], end_point[0]],
                   [start_point[1], end_point[1]],
                   [start_point[2], end_point[2]],
                   '--', color=color, linewidth=2, alpha=0.7)


def visualize_combined_volume_trajectories(combined_trajectories, coords_array=None, brain_volume=None, output_dir=None):
    """
    Create 3D visualization of trajectories extracted from the combined volume.
    
    Args:
        combined_trajectories (dict): Trajectories extracted from combined mask
        coords_array (numpy.ndarray, optional): Electrode coordinates for context
        brain_volume (vtkMRMLScalarVolumeNode, optional): Brain volume for surface context
        output_dir (str, optional): Directory to save visualization
        
    Returns:
        matplotlib.figure.Figure: Figure containing visualization
    """
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot brain surface if available
    if brain_volume:
        print("Extracting brain surface...")
        vertices, faces = get_surface_from_volume(brain_volume)
        
        if len(vertices) > 0 and len(faces) > 0:
            # Convert surface vertices to RAS coordinates
            surface_points_ras = convert_surface_vertices_to_ras(brain_volume, vertices)
            
            # Downsample surface points for better performance
            if len(surface_points_ras) > 10000:
                step = len(surface_points_ras) // 10000
                surface_points_ras = surface_points_ras[::step]
            
            print(f"Rendering {len(surface_points_ras)} surface points...")
            
            # Plot brain surface as scattered points with alpha transparency
            ax.scatter(
                surface_points_ras[:, 0], 
                surface_points_ras[:, 1], 
                surface_points_ras[:, 2],
                c='gray', s=1, alpha=0.1, label='Brain Surface'
            )
    
    # Plot electrode coordinates if available
    if coords_array is not None and len(coords_array) > 0:
        ax.scatter(coords_array[:, 0], coords_array[:, 1], coords_array[:, 2],
                  c='lightgray', marker='.', s=10, alpha=0.3, label='Electrodes')
    
    # Create colormap for trajectories
    trajectory_count = len(combined_trajectories)
    if trajectory_count == 0:
        ax.text(0, 0, 0, "No trajectories found", color='red', fontsize=14)
        
    trajectory_cmap = plt.cm.tab20(np.linspace(0, 1, max(trajectory_count, 1)))
    
    # Plot each trajectory
    for i, (bolt_id, traj_info) in enumerate(combined_trajectories.items()):
        color = trajectory_cmap[i % len(trajectory_cmap)]
        
        start_point = np.array(traj_info['start_point'])
        end_point = np.array(traj_info['end_point'])
        
        # Plot bolt head
        ax.scatter(start_point[0], start_point[1], start_point[2],
                  c=[color], marker='o', s=150, edgecolor='black',
                  label=f'Bolt {bolt_id}')
        
        # Plot entry point
        ax.scatter(end_point[0], end_point[1], end_point[2],
                  c='red', marker='*', s=150, edgecolor='black',
                  label=f'Entry {traj_info["entry_id"]}')
        
        # Plot direction arrow
        direction = np.array(traj_info['direction'])
        arrow_length = min(traj_info['length'] * 0.3, 15)  # 30% of length or max 15mm
        
        arrow = Arrow3D(
            [start_point[0], start_point[0] + direction[0]*arrow_length],
            [start_point[1], start_point[1] + direction[1]*arrow_length],
            [start_point[2], start_point[2] + direction[2]*arrow_length],
            mutation_scale=15, lw=2, arrowstyle="-|>", color=color)
        ax.add_artist(arrow)
        
        # Plot trajectory line
        ax.plot([start_point[0], end_point[0]],
               [start_point[1], end_point[1]],
               [start_point[2], end_point[2]], 
               '-', color=color, linewidth=2, alpha=0.8)
        
        # Label bolt and entry
        ax.text(start_point[0], start_point[1], start_point[2], 
               f"Bolt {bolt_id}", color=color, fontsize=8)
        ax.text(end_point[0], end_point[1], end_point[2], 
               f"Entry {traj_info['entry_id']}", color='red', fontsize=8)
        
        # Plot trajectory points if available
        if 'trajectory_points' in traj_info and len(traj_info['trajectory_points']) > 0:
            traj_points = np.array(traj_info['trajectory_points'])
            ax.scatter(traj_points[:, 0], traj_points[:, 1], traj_points[:, 2],
                      color=color, marker='.', s=5, alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Combined Volume Trajectory Analysis\n({trajectory_count} trajectories detected)')
    
    # Create a clean legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    
    if output_dir:
        save_path = os.path.join(output_dir, 'combined_volume_trajectories.png')
        plt.savefig(save_path, dpi=300)
        print(f"✅ Saved combined volume trajectory visualization to {save_path}")
    
    return fig

def create_3d_visualization(coords_array, results, bolt_directions=None):
    """
    Create a 3D visualization of electrodes and trajectories.
    
    Args:
        coords_array (numpy.ndarray): Array of electrode coordinates
        results (dict): Results from integrated_trajectory_analysis
        bolt_directions (dict, optional): Direction info from extract_bolt_entry_directions
        
    Returns:
        matplotlib.figure.Figure: Figure containing the 3D visualization
    """
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get data for plotting
    clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
    
    # FIXED: Ensure unique_clusters is a list of integers or strings, not a mix
    unique_clusters = []
    for c in set(clusters):
        # Skip noise points (typically -1)
        if c == -1:
            continue
        unique_clusters.append(c)
    
    # Create colormaps - FIXED: Use a list instead of a set for unique_clusters
    n_clusters = len(unique_clusters)
    cluster_cmap = plt.colormaps['tab20'].resampled(max(1, n_clusters))
    community_cmap = plt.colormaps['gist_ncar'].resampled(results['louvain']['n_communities'])
    
    # Plot electrodes with cluster colors
    for i, cluster_id in enumerate(unique_clusters):
        mask = clusters == cluster_id
        ax.scatter(coords_array[mask, 0], coords_array[mask, 1], coords_array[mask, 2], 
                  c=[cluster_cmap(i)], label=f'Cluster {cluster_id}', s=80, alpha=0.8)
    
    # Plot trajectories with enhanced features
    for traj in results.get('trajectories', []):
        # Find the index of this trajectory's cluster_id in unique_clusters
        try:
            # Handle both integer and string cluster IDs
            if isinstance(traj['cluster_id'], (int, np.integer)):
                # For integer IDs, find matching integer in unique_clusters
                color_idx = [i for i, c in enumerate(unique_clusters) if c == traj['cluster_id']]
                if color_idx:
                    color_idx = color_idx[0]
                else:
                    color_idx = 0
            else:
                # For string IDs (from refinement), use a predictable color
                # Use a hash function to generate a consistent index
                color_idx = hash(str(traj['cluster_id'])) % len(cluster_cmap)
        except:
            # Fallback to a default color
            color_idx = 0
        
        color = cluster_cmap(color_idx)
        
        # Plot spline if available, otherwise line
        if traj.get('spline_points') is not None:
            sp = np.array(traj['spline_points'])
            ax.plot(sp[:,0], sp[:,1], sp[:,2], '-', color=color, linewidth=3, alpha=0.7)
        else:
            endpoints = traj['endpoints']
            ax.plot([endpoints[0][0], endpoints[1][0]],
                   [endpoints[0][1], endpoints[1][1]],
                   [endpoints[0][2], endpoints[1][2]], 
                   '-', color=color, linewidth=3, alpha=0.7)
        
        # Add direction arrow
        center = np.array(traj['center'])
        direction = np.array(traj['direction'])
        arrow_length = traj['length_mm'] * 0.3  # Scale arrow to trajectory length
        
        arrow = Arrow3D(
            [center[0], center[0] + direction[0]*arrow_length],
            [center[1], center[1] + direction[1]*arrow_length],
            [center[2], center[2] + direction[2]*arrow_length],
            mutation_scale=15, lw=2, arrowstyle="-|>", color=color)
        ax.add_artist(arrow)
        
        # Mark entry point if available
        if traj.get('entry_point') is not None:
            entry = np.array(traj['entry_point'])
            ax.scatter(entry[0], entry[1], entry[2], 
                      c='red', marker='*', s=300, edgecolor='black', 
                      label=f'Entry {traj["cluster_id"]}')
            
            # Draw line from entry point to first contact
            first_contact = np.array(traj['endpoints'][0])
            ax.plot([entry[0], first_contact[0]],
                   [entry[1], first_contact[1]],
                   [entry[2], first_contact[2]], 
                   '--', color='red', linewidth=2, alpha=0.7)
    
    # Plot bolt+entry directions if available
    if bolt_directions:
        # Check if there are trajectory matches
        matches = None
        if 'trajectories' in results:
            matches = match_bolt_directions_to_trajectories(
                bolt_directions, results['trajectories'])
            results['bolt_trajectory_matches'] = matches
        
        visualize_bolt_entry_directions(ax, bolt_directions, matches)
    
    # Plot noise points
    if 'noise_points_coords' in results['dbscan'] and len(results['dbscan']['noise_points_coords']) > 0:
        noise_coords = np.array(results['dbscan']['noise_points_coords'])
        ax.scatter(noise_coords[:,0], noise_coords[:,1], noise_coords[:,2],
                  c='black', marker='x', s=100, label='Noise points (DBSCAN -1)')
    
    # Add legend and labels
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    title = 'Electrode Trajectory Analysis with Bolt Head Directions' if bolt_directions else 'Electrode Trajectory Analysis'
    ax.set_title(f'3D {title}\n(Colors=Clusters, Stars=Entry Points, Arrows=Directions, X=Noise)')
    
    # Simplify legend to avoid duplicates
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    ax.legend(unique_handles, unique_labels, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    return fig

def create_bolt_direction_analysis_page(bolt_directions, results):
    """
    Create a visualization page for bolt head directions and their relationship to trajectories.
    
    Args:
        bolt_directions (dict): Direction info from extract_bolt_entry_directions
        results (dict): Results from integrated_trajectory_analysis
        
    Returns:
        matplotlib.figure.Figure: Figure containing bolt direction analysis
    """
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle('Bolt Head Direction Analysis', fontsize=16)
    
    if not bolt_directions:
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5, 'No bolt head directions available', ha='center', va='center')
        return fig
    
    # Create grid layout
    gs = GridSpec(2, 2, figure=fig)
    
    # 3D view of bolt directions
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    
    # Plot all bolt directions
    for bolt_id, bolt_info in bolt_directions.items():
        start_point = bolt_info['start_point']
        direction = bolt_info['direction']
        points = np.array(bolt_info['points'])
        
        # Plot points
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   marker='.', s=30, alpha=0.5)
        
        # Plot direction arrow (10mm long)
        arrow = Arrow3D(
            [start_point[0], start_point[0] + direction[0] * 10],
            [start_point[1], start_point[1] + direction[1] * 10],
            [start_point[2], start_point[2] + direction[2] * 10],
            mutation_scale=15, lw=2, arrowstyle="-|>")
        ax1.add_artist(arrow)
        
        # Label
        ax1.text(start_point[0], start_point[1], start_point[2], 
                f"Bolt {bolt_id}", fontsize=8)
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('Bolt Head Directions')
    
    # Create table of bolt directions
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    table_data = []
    columns = ['ID', 'Length (mm)', 'Angle X', 'Angle Y', 'Angle Z']
    
    for bolt_id, bolt_info in bolt_directions.items():
        direction = bolt_info['direction']
        length = bolt_info['length']
        
        # Calculate angles with principal axes
        angles = calculate_angles(direction)
        
        row = [
            bolt_id,
            f"{length:.1f}",
            f"{angles['X']:.1f}°",
            f"{angles['Y']:.1f}°",
            f"{angles['Z']:.1f}°"
        ]
        table_data.append(row)
    
    table = ax2.table(cellText=table_data, colLabels=columns, 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax2.set_title('Bolt Direction Metrics')
    
    # Create table of matches between bolt directions and trajectories
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    if 'bolt_trajectory_matches' in results and results['bolt_trajectory_matches']:
        matches = results['bolt_trajectory_matches']
        
        match_data = []
        match_columns = ['Trajectory ID', 'Bolt ID', 'Distance (mm)', 'Angle (°)', 'Score']
        
        for traj_id, match in matches.items():
            row = [
                traj_id,
                match['bolt_id'],
                f"{match['distance']:.2f}",
                f"{match['angle']:.2f}",
                f"{match['score']:.2f}"
            ]
            match_data.append(row)
        
        match_table = ax3.table(cellText=match_data, colLabels=match_columns, 
                               loc='center', cellLoc='center')
        match_table.auto_set_font_size(False)
        match_table.set_fontsize(10)
        match_table.scale(1, 1.5)
        ax3.set_title('Bolt-Trajectory Matches')
    else:
        ax3.text(0.5, 0.5, 'No bolt-trajectory matches found or calculated', 
                ha='center', va='center')
    
    plt.tight_layout()
    return fig

def visualize_bolt_trajectory_comparison(coords_array, bolt_directions, trajectories, matches, results, output_dir=None):
    """
    Create visualization comparing bolt directions with electrode trajectories.
    
    Args:
        coords_array (numpy.ndarray): Array of electrode coordinates
        bolt_directions (dict): Direction info from extract_bolt_entry_directions
        trajectories (list): Electrode trajectories
        matches (dict): Matches between bolt directions and trajectories
        results (dict): Results from trajectory analysis
        output_dir (str, optional): Directory to save visualizations
        
    Returns:
        matplotlib.figure.Figure: Figure containing comparison visualization
    """
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle('Bolt-Trajectory Direction Comparison', fontsize=16)
    
    # Create 3D plot
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot electrodes as background
    ax.scatter(coords_array[:, 0], coords_array[:, 1], coords_array[:, 2], 
               c='lightgray', marker='.', s=10, alpha=0.3)
    
    # Create a colormap for matches
    colormap = plt.cm.tab10.resampled(10)
    
    # Plot matched bolt-trajectory pairs
    for i, (traj_id, match_info) in enumerate(matches.items()):
        # Use the loop index for the color instead of traj_id
        color_idx = i % 10
        color = colormap(color_idx)
        
        # Get the bolt ID from the match info
        bolt_id = match_info['bolt_id']
        
        # Check if the bolt ID exists in bolt_directions
        if bolt_id not in bolt_directions:
            continue
            
        bolt_info = bolt_directions[bolt_id]
        
        # Find trajectory
        traj = None
        for t in trajectories:
            # Convert both to strings for comparison to handle different types
            if str(t['cluster_id']) == str(traj_id):
                traj = t
                break
        
        if not traj:
            continue
            
        # Get bolt direction data
        bolt_start = np.array(bolt_info['start_point'])
        bolt_direction = np.array(bolt_info['direction'])
        
        # Safely handle points if they exist
        if 'points' in bolt_info and len(bolt_info['points']) > 0:
            bolt_points = np.array(bolt_info['points'])
            ax.scatter(bolt_points[:, 0], bolt_points[:, 1], bolt_points[:, 2], 
                      color=color, marker='.', s=30, alpha=0.7, label=f'Bolt {bolt_id} points')
        
        # Get trajectory data
        traj_first_contact = np.array(traj['endpoints'][0])
        traj_direction = np.array(traj['direction'])
        traj_points = np.array([traj['endpoints'][0], traj['endpoints'][1]])
        
        # Plot bolt direction arrow
        arrow_length = 15  # mm
        bolt_arrow = Arrow3D(
            [bolt_start[0], bolt_start[0] + bolt_direction[0] * arrow_length],
            [bolt_start[1], bolt_start[1] + bolt_direction[1] * arrow_length],
            [bolt_start[2], bolt_start[2] + bolt_direction[2] * arrow_length],
            mutation_scale=15, lw=2, arrowstyle="-|>", color=color)
        ax.add_artist(bolt_arrow)
        
        # Plot trajectory
        if traj.get('spline_points'):
            spline = np.array(traj['spline_points'])
            ax.plot(spline[:, 0], spline[:, 1], spline[:, 2], 
                   '-', color=color, linewidth=2, alpha=0.7, label=f'Trajectory {traj_id}')
        else:
            ax.plot([traj_points[0][0], traj_points[1][0]],
                   [traj_points[0][1], traj_points[1][1]],
                   [traj_points[0][2], traj_points[1][2]], 
                   '-', color=color, linewidth=2, alpha=0.7, label=f'Trajectory {traj_id}')
        
        # Plot connection between bolt and first contact
        ax.plot([bolt_start[0], traj_first_contact[0]],
               [bolt_start[1], traj_first_contact[1]],
               [bolt_start[2], traj_first_contact[2]],
               '--', color=color, linewidth=1.5, alpha=0.7)
        
        # Add labels for bolt and first contact
        ax.text(bolt_start[0], bolt_start[1], bolt_start[2], f"Bolt {bolt_id}", 
               color=color, fontsize=8)
        ax.text(traj_first_contact[0], traj_first_contact[1], traj_first_contact[2], 
               f"First contact {traj_id}", color=color, fontsize=8)
        
        # Add match details as text
        distance = match_info['distance']
        angle = match_info['angle']
        ax.text(bolt_start[0], bolt_start[1], bolt_start[2] - 5, 
               f"Dist: {distance:.1f}mm, Angle: {angle:.1f}°", 
               color=color, fontsize=8)
    
    # Add unmatched bolts
    unmatched_bolt_ids = []
    for bolt_id in bolt_directions.keys():
        # Check if this bolt_id is in any match
        is_matched = False
        for match in matches.values():
            if str(match['bolt_id']) == str(bolt_id):
                is_matched = True
                break
        if not is_matched:
            unmatched_bolt_ids.append(bolt_id)
    
    for bolt_id in unmatched_bolt_ids:
        bolt_info = bolt_directions[bolt_id]
        bolt_start = np.array(bolt_info['start_point'])
        bolt_direction = np.array(bolt_info['direction'])
        
        # Safely handle points if they exist
        if 'points' in bolt_info and len(bolt_info['points']) > 0:
            bolt_points = np.array(bolt_info['points'])
            ax.scatter(bolt_points[:, 0], bolt_points[:, 1], bolt_points[:, 2], 
                      color='darkgray', marker='.', s=30, alpha=0.7)
        
        # Plot bolt direction arrow
        arrow_length = 15  # mm
        bolt_arrow = Arrow3D(
            [bolt_start[0], bolt_start[0] + bolt_direction[0] * arrow_length],
            [bolt_start[1], bolt_start[1] + bolt_direction[1] * arrow_length],
            [bolt_start[2], bolt_start[2] + bolt_direction[2] * arrow_length],
            mutation_scale=15, lw=2, arrowstyle="-|>", color='darkgray')
        ax.add_artist(bolt_arrow)
        
        # Add label
        ax.text(bolt_start[0], bolt_start[1], bolt_start[2], 
               f"Unmatched Bolt {bolt_id}", color='darkgray', fontsize=8)
    
    # Create a clean legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Bolt-Trajectory Direction Comparison\n({len(matches)} matches found)')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'bolt_trajectory_comparison.png'), dpi=300)
    
    return fig

def create_bolt_trajectory_validation_page(bolt_directions, trajectories, matches, validations, output_dir=None):
    """
    Create a visualization page for bolt-trajectory validation results.
    
    Args:
        bolt_directions (dict): Direction info from extract_bolt_entry_directions
        trajectories (list): Electrode trajectories
        matches (dict): Matches between bolt directions and trajectories
        validations (dict): Validation results for first contacts
        output_dir (str, optional): Directory to save visualizations
        
    Returns:
        matplotlib.figure.Figure: Figure containing validation results
    """
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle('Bolt-Trajectory Validation Results', fontsize=16)
    
    # Create grid layout
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 2])
    
    # Summary table
    ax1 = fig.add_subplot(gs[0])
    ax1.axis('off')
    
    # Create validation summary table
    summary_data = []
    summary_columns = ['Total Trajectories', 'Matched with Bolt', 'Valid First Contacts', 'Invalid First Contacts']
    
    total_trajectories = len(trajectories)
    total_matches = len(matches)
    valid_contacts = sum(1 for v in validations.values() if v['valid'])
    invalid_contacts = sum(1 for v in validations.values() if not v['valid'])
    
    summary_data.append([
        str(total_trajectories),
        f"{total_matches} ({total_matches/total_trajectories*100:.1f}%)",
        f"{valid_contacts} ({valid_contacts/total_matches*100:.1f}%)",
        f"{invalid_contacts} ({invalid_contacts/total_matches*100:.1f}%)"
    ])
    
    summary_table = ax1.table(cellText=summary_data, colLabels=summary_columns,
                             loc='center', cellLoc='center')
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(10)
    summary_table.scale(1, 1.5)
    ax1.set_title('Validation Summary')
    
    # Detailed validation results
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    
    # Create detailed validation table
    detail_data = []
    detail_columns = ['Trajectory ID', 'Bolt ID', 'First Contact Valid', 'Distance Error (mm)', 'Angle Error (°)', 'Reason']
    
    for traj_id, validation in validations.items():
        match = matches.get(traj_id)
        if not match:
            continue
            
        bolt_id = match['bolt_id']
        valid_status = "Yes" if validation['valid'] else "No"
        
        row = [
            traj_id,
            bolt_id,
            valid_status,
            f"{validation.get('distance_error', 'N/A')}" if 'distance_error' in validation else 'N/A',
            f"{validation.get('angle_error', 'N/A')}" if 'angle_error' in validation else 'N/A',
            validation.get('reason', 'N/A')
        ]
        detail_data.append(row)
    
    # Sort by trajectory ID
    detail_data.sort(key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0])
    
    detail_table = ax2.table(cellText=detail_data, colLabels=detail_columns,
                           loc='center', cellLoc='center')
    detail_table.auto_set_font_size(False)
    detail_table.set_fontsize(10)
    detail_table.scale(1, 1.5)
    ax2.set_title('Detailed Validation Results')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'bolt_trajectory_validation.png'), dpi=300)
    
    return fig

def create_pca_angle_analysis_page(results):
    """
    Create a visualization page for PCA and angle analysis results.
    
    Args:
        results (dict): Results from integrated_trajectory_analysis
        
    Returns:
        matplotlib.figure.Figure: Figure containing PCA and angle analysis
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np
    
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle('PCA and Angular Analysis of Electrode Trajectories', fontsize=16)
    
    # Create grid layout
    gs = GridSpec(3, 2, figure=fig)
    
    # Get trajectories
    trajectories = results.get('trajectories', [])
    
    if not trajectories:
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5, 'No trajectory data available for analysis', 
                ha='center', va='center', fontsize=14)
        return fig
    
    # 1. PCA explained variance ratio distribution
    ax1 = fig.add_subplot(gs[0, 0])
    
    explained_variances = []
    linearity_scores = []
    
    for traj in trajectories:
        if 'pca_variance' in traj and len(traj['pca_variance']) > 0:
            explained_variances.append(traj['pca_variance'][0])  # First component
            linearity_scores.append(traj.get('linearity', 0))
    
    if explained_variances:
        ax1.hist(explained_variances, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=np.mean(explained_variances), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(explained_variances):.3f}')
        ax1.set_xlabel('PCA First Component Variance Ratio')
        ax1.set_ylabel('Number of Trajectories')
        ax1.set_title('Distribution of Trajectory Linearity (PCA)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No PCA data available', ha='center', va='center')
    
    # 2. Linearity vs trajectory length scatter plot
    ax2 = fig.add_subplot(gs[0, 1])
    
    lengths = []
    for traj in trajectories:
        if 'length_mm' in traj and 'linearity' in traj:
            lengths.append(traj['length_mm'])
    
    if lengths and linearity_scores:
        scatter = ax2.scatter(lengths, linearity_scores, alpha=0.7, c='green')
        ax2.set_xlabel('Trajectory Length (mm)')
        ax2.set_ylabel('Linearity Score (PCA 1st Component)')
        ax2.set_title('Trajectory Length vs Linearity')
        ax2.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(lengths) > 1:
            correlation = np.corrcoef(lengths, linearity_scores)[0, 1]
            ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'Insufficient data for scatter plot', ha='center', va='center')
    
    # 3. Angular distribution with respect to coordinate axes
    ax3 = fig.add_subplot(gs[1, :])
    
    angles_x = []
    angles_y = []
    angles_z = []
    
    for traj in trajectories:
        if 'angles_with_axes' in traj:
            angles = traj['angles_with_axes']
            angles_x.append(angles.get('X', 0))
            angles_y.append(angles.get('Y', 0))
            angles_z.append(angles.get('Z', 0))
    
    if angles_x:
        bins = np.linspace(0, 180, 19)  # 10-degree bins
        
        ax3.hist(angles_x, bins=bins, alpha=0.7, label='X-axis angles', color='red')
        ax3.hist(angles_y, bins=bins, alpha=0.7, label='Y-axis angles', color='green')
        ax3.hist(angles_z, bins=bins, alpha=0.7, label='Z-axis angles', color='blue')
        
        ax3.set_xlabel('Angle with Coordinate Axis (degrees)')
        ax3.set_ylabel('Number of Trajectories')
        ax3.set_title('Distribution of Trajectory Angles with Coordinate Axes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add vertical lines for common angles
        for angle in [0, 30, 45, 60, 90]:
            ax3.axvline(x=angle, color='gray', linestyle=':', alpha=0.5)
            ax3.text(angle, ax3.get_ylim()[1] * 0.9, f'{angle}°', 
                    ha='center', fontsize=8, color='gray')
    else:
        ax3.text(0.5, 0.5, 'No angle data available', ha='center', va='center')
    
    # 4. Summary statistics table
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    
    # Calculate summary statistics
    if trajectories:
        mean_linearity = np.mean(linearity_scores) if linearity_scores else 0
        std_linearity = np.std(linearity_scores) if linearity_scores else 0
        mean_length = np.mean(lengths) if lengths else 0
        std_length = np.std(lengths) if lengths else 0
        
        # Count highly linear trajectories (linearity > 0.9)
        high_linearity = sum(1 for score in linearity_scores if score > 0.9) if linearity_scores else 0
        
        summary_data = [
            ['Number of Trajectories', str(len(trajectories))],
            ['Mean Linearity', f'{mean_linearity:.3f} ± {std_linearity:.3f}'],
            ['Mean Length (mm)', f'{mean_length:.1f} ± {std_length:.1f}'],
            ['Highly Linear (>0.9)', f'{high_linearity} ({high_linearity/len(trajectories)*100:.1f}%)'],
            ['Min Linearity', f'{min(linearity_scores):.3f}' if linearity_scores else 'N/A'],
            ['Max Linearity', f'{max(linearity_scores):.3f}' if linearity_scores else 'N/A']
        ]
        
        table = ax4.table(cellText=summary_data, colLabels=['Metric', 'Value'],
                         loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Color code linearity values
        for i, (metric, value) in enumerate(summary_data):
            if 'Linearity' in metric and value != 'N/A':
                try:
                    val = float(value.split()[0])
                    if val > 0.9:
                        table[(i+1, 1)].set_facecolor('lightgreen')
                    elif val < 0.7:
                        table[(i+1, 1)].set_facecolor('lightcoral')
                except:
                    pass
    
    ax4.set_title('PCA Analysis Summary')
    
    # 5. Trajectory quality assessment
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Create a quality score based on linearity and other factors
    quality_scores = []
    quality_labels = []
    
    for traj in trajectories:
        linearity = traj.get('linearity', 0)
        length = traj.get('length_mm', 0)
        contact_count = traj.get('electrode_count', 0)
        
        # Simple quality score: high linearity, reasonable length, good contact count
        quality_score = linearity
        
        # Penalize very short or very long trajectories
        if length < 20 or length > 100:
            quality_score *= 0.8
        
        # Boost score for standard electrode sizes
        standard_sizes = [5, 8, 10, 12, 15, 18]
        if contact_count in standard_sizes:
            quality_score *= 1.1
        
        quality_scores.append(quality_score)
        
        # Classify quality
        if quality_score > 0.9:
            quality_labels.append('Excellent')
        elif quality_score > 0.8:
            quality_labels.append('Good')
        elif quality_score > 0.7:
            quality_labels.append('Fair')
        else:
            quality_labels.append('Poor')
    
    if quality_labels:
        # Count each quality level
        quality_counts = {label: quality_labels.count(label) for label in ['Excellent', 'Good', 'Fair', 'Poor']}
        
        # Create pie chart
        labels = []
        sizes = []
        colors = []
        color_map = {'Excellent': 'green', 'Good': 'lightgreen', 'Fair': 'orange', 'Poor': 'red'}
        
        for label, count in quality_counts.items():
            if count > 0:
                labels.append(f'{label}\n({count})')
                sizes.append(count)
                colors.append(color_map[label])
        
        if sizes:
            ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax5.set_title('Trajectory Quality Assessment')
        else:
            ax5.text(0.5, 0.5, 'No quality data available', ha='center', va='center')
    else:
        ax5.text(0.5, 0.5, 'No trajectories for quality assessment', ha='center', va='center')
    
    plt.tight_layout()
    return fig


def visualize_combined_results(coords_array, results, output_dir=None, bolt_directions=None):
    """
    Create and save/display visualizations of trajectory analysis results.
    
    Args:
        coords_array (numpy.ndarray): Array of electrode coordinates
        results (dict): Results from integrated_trajectory_analysis
        output_dir (str, optional): Directory to save PDF report. If None, displays plots interactively.
        bolt_directions (dict, optional): Results from extract_bolt_entry_directions
    """
    if output_dir:
        pdf_path = os.path.join(output_dir, 'trajectory_analysis_report.pdf')
        with PdfPages(pdf_path) as pdf:
            # Create summary page
            fig = create_summary_page(results)
            pdf.savefig(fig)
            plt.close(fig)
            
            # Create 3D visualization page
            fig = create_3d_visualization(coords_array, results, bolt_directions)
            pdf.savefig(fig)
            plt.close(fig)
            
            # Create trajectory details page
            if 'trajectories' in results:
                fig = create_trajectory_details_page(results)
                pdf.savefig(fig)
                plt.close(fig)
                
            # Create PCA and angle analysis page
            fig = create_pca_angle_analysis_page(results)
            pdf.savefig(fig)
            plt.close(fig)
                
            # Create noise points page
            fig = create_noise_points_page(coords_array, results)
            pdf.savefig(fig)
            plt.close(fig)
            
            # Create bolt direction analysis page if applicable
            if bolt_directions:
                fig = create_bolt_direction_analysis_page(bolt_directions, results)
                pdf.savefig(fig)
                plt.close(fig)
                
                # If we have bolt-trajectory matches, add a comparison visualization
                if 'bolt_trajectory_matches' in results and results['bolt_trajectory_matches']:
                    fig = visualize_bolt_trajectory_comparison(
                        coords_array, bolt_directions, results['trajectories'], 
                        results['bolt_trajectory_matches'], results
                    )
                    pdf.savefig(fig)
                    plt.close(fig)
                    
        print(f"Complete analysis report saved to: {pdf_path}")
    else:
        # Interactive mode - show all plots
        fig = create_summary_page(results)
        plt.show()
        
        fig = create_3d_visualization(coords_array, results, bolt_directions)
        plt.show()
        
        if 'trajectories' in results:
            fig = create_trajectory_details_page(results)
            plt.show()
            
        fig = create_pca_angle_analysis_page(results)
        plt.show()
            
        fig = create_noise_points_page(coords_array, results)
        plt.show()
        
        if bolt_directions:
            fig = create_bolt_direction_analysis_page(bolt_directions, results)
            plt.show()
            
            if 'bolt_trajectory_matches' in results and results['bolt_trajectory_matches']:
                fig = visualize_bolt_trajectory_comparison(
                    coords_array, bolt_directions, results['trajectories'], 
                    results['bolt_trajectory_matches'], results
                )
                plt.show()

def visualize_trajectory_comparison(coords_array, integrated_results, combined_trajectories, comparison):
    """
    Create a visualization comparing trajectories from integrated analysis and combined volume.
    
    Args:
        coords_array (numpy.ndarray): Array of electrode coordinates
        integrated_results (dict): Results from integrated_trajectory_analysis
        combined_trajectories (dict): Trajectories extracted from combined mask
        comparison (dict): Results from compare_trajectories_with_combined_data
        
    Returns:
        matplotlib.figure.Figure: Figure containing comparison visualization
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Trajectory Detection Comparison: Clustering vs. Combined Volume', fontsize=16)
    
    # Create 3D plot
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all electrode points as background
    ax.scatter(coords_array[:, 0], coords_array[:, 1], coords_array[:, 2], 
              c='lightgray', marker='.', s=10, alpha=0.3, label='Electrode points')
    
    # Create a colormap for consistent colors
    colormap = plt.cm.tab10.resampled(10)
    
    # Plot matched trajectories
    for i, (traj_id, match_info) in enumerate(comparison['matches'].items()):
        # Use loop index for color assignment
        color_idx = i % 10
        color = colormap(color_idx)
        
        bolt_id = match_info['bolt_id']
        
        # Get integrated trajectory
        integrated_traj = None
        for t in integrated_results['trajectories']:
            # Compare as strings to handle mixed types
            if str(t['cluster_id']) == str(traj_id):
                integrated_traj = t
                break
        
        if not integrated_traj:
            continue
        
        # Check if bolt_id exists in combined_trajectories
        if bolt_id not in combined_trajectories:
            continue
            
        # Get combined trajectory
        combined_traj = combined_trajectories[bolt_id]
        
        # Get trajectory data
        integrated_endpoints = np.array(integrated_traj['endpoints'])
        combined_start = np.array(combined_traj['start_point'])
        combined_end = np.array(combined_traj['end_point'])
        
        # Plot integrated trajectory
        ax.plot([integrated_endpoints[0][0], integrated_endpoints[1][0]],
               [integrated_endpoints[0][1], integrated_endpoints[1][1]],
               [integrated_endpoints[0][2], integrated_endpoints[1][2]],
               '-', color=color, linewidth=2, alpha=0.7)
        
        # Plot combined trajectory
        ax.plot([combined_start[0], combined_end[0]],
               [combined_start[1], combined_end[1]],
               [combined_start[2], combined_end[2]],
               '--', color=color, linewidth=2, alpha=0.7)
        
        # Add labels
        ax.text(integrated_endpoints[0][0], integrated_endpoints[0][1], integrated_endpoints[0][2],
               f"Cluster {traj_id}", fontsize=8, color=color)
        ax.text(combined_start[0], combined_start[1], combined_start[2],
               f"Bolt {bolt_id}", fontsize=8, color=color)
        
        # Add match info
        mid_point = (integrated_endpoints[0] + integrated_endpoints[1]) / 2
        ax.text(mid_point[0], mid_point[1], mid_point[2],
               f"Dist: {match_info['min_distance']:.1f}mm\nAngle: {match_info['angle']:.1f}°", 
               fontsize=8, color=color)
    
    # Plot unmatched integrated trajectories
    for unmatched_id in comparison['unmatched_integrated']:
        # Find the trajectory
        traj = None
        for t in integrated_results['trajectories']:
            # Compare as strings to handle mixed types
            if str(t['cluster_id']) == str(unmatched_id):
                traj = t
                break
        
        if not traj:
            continue
            
        endpoints = np.array(traj['endpoints'])
        
        # Plot in a distinct color
        ax.plot([endpoints[0][0], endpoints[1][0]],
               [endpoints[0][1], endpoints[1][1]],
               [endpoints[0][2], endpoints[1][2]],
               '-', color='blue', linewidth=2, alpha=0.5)
        
        ax.text(endpoints[0][0], endpoints[0][1], endpoints[0][2],
               f"Unmatched Cluster {unmatched_id}", fontsize=8, color='blue')
    
    # Plot unmatched combined trajectories
    for bolt_id in comparison['unmatched_combined']:
        if bolt_id not in combined_trajectories:
            continue
            
        combined_traj = combined_trajectories[bolt_id]
        start = np.array(combined_traj['start_point'])
        end = np.array(combined_traj['end_point'])
        
        # Plot in a distinct color
        ax.plot([start[0], end[0]],
               [start[1], end[1]],
               [start[2], end[2]],
               '--', color='red', linewidth=2, alpha=0.5)
        
        ax.text(start[0], start[1], start[2],
               f"Unmatched Bolt {bolt_id}", fontsize=8, color='red')
    
    # Add summary statistics as text
    summary = comparison['summary']
    stats_text = (
        f"Integrated trajectories: {summary['integrated_trajectories']}\n"
        f"Combined trajectories: {summary['combined_trajectories']}\n"
        f"Matching trajectories: {summary['matching_trajectories']} "
        f"({summary['matching_percentage']:.1f}%)\n"
    )
    
    if 'spatial_alignment_stats' in summary and summary['spatial_alignment_stats']:
        dist_stats = summary['spatial_alignment_stats']['min_distance']
        angle_stats = summary['spatial_alignment_stats']['angle']
        stats_text += (
            f"Mean distance: {dist_stats['mean']:.2f}mm\n"
            f"Mean angle: {angle_stats['mean']:.2f}°"
        )
    
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    
    # Create a legend
    legend_elements = [
        plt.Line2D([0], [0], color='k', lw=2, linestyle='-', label='Integrated (Clustering)'),
        plt.Line2D([0], [0], color='k', lw=2, linestyle='--', label='Combined Volume'),
        plt.Line2D([0], [0], color='blue', lw=2, linestyle='-', label='Unmatched Integrated'),
        plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label='Unmatched Combined')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig
#------------------------------------------------------------------------------
# PART 4: MAIN EXECUTION FUNCTION
#------------------------------------------------------------------------------

def main(use_combined_volume=True, use_original_reports=True, detect_duplicates=True, 
         duplicate_threshold=0.5, use_adaptive_clustering=False, max_iterations=10,
         validate_spacing=True, expected_spacing_range=(3.0, 5.0),
         refine_trajectories=True, max_contacts_per_trajectory=20,
         validate_entry_angles=True, hemisphere='both'):  # NEW PARAMETER
    """
    Enhanced main function for electrode trajectory analysis with hemisphere filtering and flexible options
    including adaptive clustering, spacing validation, and trajectory refinement.
    
    This function provides a unified workflow for both combined volume and traditional
    analysis approaches, with options to generate various reports and use adaptive
    parameter selection for clustering. Now includes hemisphere-based filtering.
    
    Args:
        use_combined_volume (bool): Whether to use the combined volume approach for trajectory extraction
        use_original_reports (bool): Whether to generate the original format reports
        detect_duplicates (bool): Whether to detect duplicate centroids
        duplicate_threshold (float): Threshold for duplicate detection in mm
        use_adaptive_clustering (bool): Whether to use adaptive clustering parameter selection
        max_iterations (int): Maximum number of iterations for adaptive parameter search
        validate_spacing (bool): Whether to validate electrode spacing
        expected_spacing_range (tuple): Expected range for contact spacing (min, max) in mm
        refine_trajectories (bool): Whether to apply trajectory refinement (merging/splitting)
        max_contacts_per_trajectory (int): Maximum number of contacts allowed in a single trajectory
        validate_entry_angles (bool): Whether to validate entry angles against surgical constraints (30-60°)
        hemisphere (str): 'left' (x < 0), 'right' (x > 0), or 'both' (no filtering) - NEW PARAMETER
        
    Returns:
        dict: Results dictionary containing all analysis results
    """
    try:
        start_time = time.time()
        print(f"Starting electrode trajectory analysis...")
        print(f"Options: combined_volume={use_combined_volume}, adaptive_clustering={use_adaptive_clustering}, "
              f"detect_duplicates={detect_duplicates}, duplicate_threshold={duplicate_threshold}, "
              f"validate_spacing={validate_spacing}, spacing_range={expected_spacing_range}, "
              f"refine_trajectories={refine_trajectories}, validate_entry_angles={validate_entry_angles}, "
              f"hemisphere={hemisphere}")  # NEW: Show hemisphere setting
        
        # Step 1: Load required volumes from Slicer
        print("Loading volumes from Slicer...")
        electrodes_volume = slicer.util.getNode('P2_electrode_mask_success_1')
        brain_volume = slicer.util.getNode("patient2_mask_5")
        
        # Create output directories
        base_dir = r"C:\Users\rocia\Downloads\TFG\Cohort\Centroids\P2_BoltHeadandpaths_contactsXpath_SPACING_dealing_with_problems_test_fiducials"
        
        # NEW: Include hemisphere in output directory name
        output_dir_name = "trajectory_analysis_results"
        if hemisphere.lower() != 'both':
            output_dir_name += f"_{hemisphere}_hemisphere"
        
        output_dir = os.path.join(base_dir, output_dir_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create adaptive clustering subdirectory if needed
        if use_adaptive_clustering:
            adaptive_dir = os.path.join(output_dir, "adaptive_clustering")
            os.makedirs(adaptive_dir, exist_ok=True)
        
        # Create entry angle validation subdirectory if needed
        if validate_entry_angles:
            entry_angle_dir = os.path.join(output_dir, "entry_angle_validation")
            os.makedirs(entry_angle_dir, exist_ok=True)
        
        # Define expected electrode contact counts
        expected_contact_counts = [5, 8, 10, 12, 15, 18]
        
        # Combined results dictionary
        all_results = {
            'parameters': {
                'use_combined_volume': use_combined_volume,
                'use_original_reports': use_original_reports,
                'expected_contact_counts': expected_contact_counts,
                'use_adaptive_clustering': use_adaptive_clustering,
                'detect_duplicates': detect_duplicates,
                'duplicate_threshold': duplicate_threshold,
                'validate_spacing': validate_spacing,
                'expected_spacing_range': expected_spacing_range,
                'refine_trajectories': refine_trajectories,
                'max_contacts_per_trajectory': max_contacts_per_trajectory,
                'validate_entry_angles': validate_entry_angles,
                'hemisphere': hemisphere  # NEW: Store hemisphere parameter
            }
        }
        
        # Step 2: Get electrode coordinates
        print("Extracting electrode coordinates...")
        centroids_ras = get_all_centroids(electrodes_volume) if electrodes_volume else None
        original_coords_array = np.array(list(centroids_ras.values())) if centroids_ras else None
        
        if original_coords_array is None or len(original_coords_array) == 0:
            print("No electrode coordinates found. Cannot proceed with analysis.")
            return {}
        
        print(f"Found {len(original_coords_array)} electrode coordinates.")
        
        # NEW: Apply hemisphere filtering to coordinates
        coords_array, hemisphere_mask, filtered_indices = filter_coordinates_by_hemisphere(
            original_coords_array, hemisphere, verbose=True
        )
        
        if len(coords_array) == 0:
            print(f"No coordinates found in {hemisphere} hemisphere. Cannot proceed with analysis.")
            return {'error': f'No coordinates in {hemisphere} hemisphere'}
        
        all_results['electrode_count'] = len(coords_array)
        all_results['original_electrode_count'] = len(original_coords_array)  # NEW: Store original count
        all_results['hemisphere_filtering'] = {  # NEW: Store filtering info
            'hemisphere': hemisphere,
            'original_count': len(original_coords_array),
            'filtered_count': len(coords_array),
            'filtering_efficiency': len(coords_array) / len(original_coords_array) * 100,
            'discarded_count': len(original_coords_array) - len(coords_array)
        }
        
        # Step 3: Combined volume analysis (if requested)
        combined_trajectories = {}
        if use_combined_volume:
            print("Performing combined volume analysis...")
            combined_volume = slicer.util.getNode('P2_CombinedBoltHeadEntryPointsTrajectoryMask')
            
            if combined_volume:
                # Extract trajectories from combined volume
                all_combined_trajectories = extract_trajectories_from_combined_mask(
                    combined_volume,
                    brain_volume=brain_volume
                )
                
                # NEW: Filter combined trajectories by hemisphere
                combined_trajectories = filter_bolt_directions_by_hemisphere(
                    all_combined_trajectories, hemisphere, verbose=True
                )
                
                all_results['combined_volume'] = {
                    'trajectories': combined_trajectories,
                    'trajectory_count': len(combined_trajectories),
                    'original_trajectory_count': len(all_combined_trajectories)  # NEW: Store original count
                }
                
                print(f"Extracted {len(combined_trajectories)} trajectories from combined volume (after hemisphere filtering).")
                
                # Create trajectory lines volume
                if combined_trajectories:
                    trajectory_volume = create_trajectory_lines_volume(
                        combined_trajectories, 
                        combined_volume, 
                        output_dir
                    )
                    all_results['combined_volume']['trajectory_volume'] = trajectory_volume
                
                # Visualize combined volume trajectories
                if combined_trajectories:
                    print(f"Creating combined volume visualizations...")
                    fig = visualize_combined_volume_trajectories(
                        combined_trajectories,
                        coords_array=coords_array,
                        brain_volume=brain_volume,
                        output_dir=output_dir
                    )
                    plt.close(fig)
            else:
                print("Combined volume not found. Skipping combined volume analysis.")
        
        # Step 4: Get entry points if available and filter by hemisphere
        entry_points = None
        entry_points_volume = slicer.util.getNode('P2_brain_entry_points')
        if entry_points_volume:
            all_entry_centroids_ras = get_all_centroids(entry_points_volume)
            if all_entry_centroids_ras:
                all_entry_points = np.array(list(all_entry_centroids_ras.values()))
                
                # NEW: Filter entry points by hemisphere
                entry_points, entry_hemisphere_mask, _ = filter_coordinates_by_hemisphere(
                    all_entry_points, hemisphere, verbose=True
                )
                
                print(f"Found {len(entry_points)} entry points in {hemisphere} hemisphere (original: {len(all_entry_points)}).")
        
        # Step 5: Perform trajectory analysis with regular or adaptive approach
        if use_adaptive_clustering:
            print("Running trajectory analysis with adaptive clustering...")
            
            # Run adaptive parameter search
            parameter_search = adaptive_clustering_parameters(
                coords_array=coords_array,
                initial_eps=8,
                initial_min_neighbors=3,
                expected_contact_counts=expected_contact_counts,
                max_iterations=max_iterations,
                eps_step=0.5,
                verbose=True
            )
            
            optimal_eps = parameter_search['optimal_eps']
            optimal_min_neighbors = parameter_search['optimal_min_neighbors']
            
            print(f"Found optimal parameters: eps={optimal_eps:.2f}, min_neighbors={optimal_min_neighbors}")
            
            # Run integrated trajectory analysis with optimal parameters and spacing validation
            integrated_results = integrated_trajectory_analysis(
                coords_array=coords_array,
                entry_points=entry_points,
                max_neighbor_distance=optimal_eps,
                min_neighbors=optimal_min_neighbors,
                expected_spacing_range=expected_spacing_range if validate_spacing else None
            )
            
            # Store the optimal parameters in the results
            all_results['adaptive_parameters'] = {
                'optimal_eps': optimal_eps,
                'optimal_min_neighbors': optimal_min_neighbors,
                'score': parameter_search['score'],
                'iterations': len(parameter_search['iterations_data'])
            }
            
            # Save parameter search visualization
            if use_original_reports:
                # Create adaptive clustering subdirectory
                adaptive_dir = os.path.join(output_dir, "adaptive_clustering")
                os.makedirs(adaptive_dir, exist_ok=True)
                
                # Save parameter search visualization
                plt.figure(parameter_search['visualization'].number)
                plt.savefig(os.path.join(adaptive_dir, 'adaptive_parameter_search.png'), dpi=300)
                
                # Save parameter search results to PDF
                with PdfPages(os.path.join(adaptive_dir, 'adaptive_parameter_search.pdf')) as pdf:
                    pdf.savefig(parameter_search['visualization'])
                
                # Create evolution visualization
                evolution_vis = visualize_adaptive_clustering(
                    coords_array,
                    parameter_search['iterations_data'],
                    expected_contact_counts,
                    adaptive_dir
                )
                
                print(f"✅ Adaptive clustering results saved to {adaptive_dir}")
                
            integrated_results['parameter_search'] = parameter_search
        else:
            # Run the original enhanced integrated trajectory analysis with fixed parameters
            print("Running integrated trajectory analysis with fixed parameters (eps=7.5, min_neighbors=3)...")
            integrated_results = integrated_trajectory_analysis(
                coords_array=coords_array,
                entry_points=entry_points,
                max_neighbor_distance=7.5,
                min_neighbors=3,
                expected_spacing_range=expected_spacing_range if validate_spacing else None
            )
        
        # Add trajectory sorting by projection along principal direction
        # This is needed for both spacing validation and trajectory refinement
        for traj in integrated_results.get('trajectories', []):
            if 'endpoints' in traj:
                # Get coordinates for this trajectory from the graph
                clusters = np.array([node[1]['dbscan_cluster'] for node in integrated_results['graph'].nodes(data=True)])
                mask = clusters == traj['cluster_id']
                
                if np.sum(mask) > 0:
                    cluster_coords = coords_array[mask]
                    
                    # Sort contacts along trajectory direction
                    direction = np.array(traj['direction'])
                    center = np.mean(cluster_coords, axis=0)
                    projected = np.dot(cluster_coords - center, direction)
                    sorted_indices = np.argsort(projected)
                    sorted_coords = cluster_coords[sorted_indices]
                    
                    # Store sorted coordinates for later use
                    traj['sorted_coords'] = sorted_coords.tolist()
        
        # Step 6: Apply trajectory refinement if requested
        if refine_trajectories:
            print("Applying trajectory refinement (merging and splitting)...")
            
            # First, add validation against expected contact counts
            validation = validate_electrode_clusters(integrated_results, expected_contact_counts)
            integrated_results['electrode_validation'] = validation
            
            # Create validation visualization
            if 'figures' not in integrated_results:
                integrated_results['figures'] = {}
            
            integrated_results['figures']['electrode_validation'] = create_electrode_validation_page(integrated_results, validation)
            
            # Apply targeted trajectory refinement
            refinement_results = targeted_trajectory_refinement(
                integrated_results['trajectories'],
                expected_contact_counts=expected_contact_counts,
                max_expected=max_contacts_per_trajectory,
                tolerance=2
            )
            
            # Update results with refined trajectories
            original_trajectory_count = len(integrated_results.get('trajectories', []))
            final_trajectory_count = len(refinement_results['trajectories'])
            
            print(f"Trajectory refinement results:")
            print(f"- Original trajectories: {original_trajectory_count}")
            print(f"- Final trajectories after refinement: {final_trajectory_count}")
            print(f"- Merged trajectories: {refinement_results['merged_count']}")
            print(f"- Split trajectories: {refinement_results['split_count']}")
            
            # Update integrated results with refined trajectories
            integrated_results['original_trajectories'] = integrated_results.get('trajectories', []).copy()
            integrated_results['trajectories'] = refinement_results['trajectories']
            integrated_results['n_trajectories'] = final_trajectory_count
            integrated_results['trajectory_refinement'] = refinement_results
            
            # Create refinement visualization if reports are enabled
            if use_original_reports:
                refinement_dir = os.path.join(output_dir, "trajectory_refinement")
                os.makedirs(refinement_dir, exist_ok=True)
                
                # Create visualization comparing original and refined trajectories
                refinement_fig = visualize_trajectory_refinement(
                    coords_array,
                    integrated_results['original_trajectories'],
                    integrated_results['trajectories'],
                    refinement_results
                )
                
                plt.savefig(os.path.join(refinement_dir, 'trajectory_refinement.png'), dpi=300)
                
                with PdfPages(os.path.join(refinement_dir, 'trajectory_refinement_report.pdf')) as pdf:
                    pdf.savefig(refinement_fig)
                    plt.close(refinement_fig)
                
                print(f"✅ Trajectory refinement report saved to {refinement_dir}")
        else:
            # Add validation without refinement
            validation = validate_electrode_clusters(integrated_results, expected_contact_counts)
            integrated_results['electrode_validation'] = validation
            
            # Create validation visualization
            if 'figures' not in integrated_results:
                integrated_results['figures'] = {}
            
            integrated_results['figures']['electrode_validation'] = create_electrode_validation_page(integrated_results, validation)
        
        # Store integrated results
        all_results['integrated_analysis'] = integrated_results
        print(f"Identified {integrated_results.get('n_trajectories', 0)} trajectories through clustering.")
        
        # Print electrode validation results
        if 'electrode_validation' in integrated_results and 'summary' in integrated_results['electrode_validation']:
            validation_summary = integrated_results['electrode_validation']['summary']
            print(f"Electrode validation results:")
            print(f"- Total clusters: {validation_summary['total_clusters']}")
            print(f"- Valid clusters: {validation_summary['valid_clusters']} ({validation_summary['match_percentage']:.1f}%)")
            print(f"- Distribution by contact count:")
            for count, num in validation_summary['by_size'].items():
                if num > 0:
                    print(f"  • {count}-contact electrodes: {num}")
        
        # Step 7: Duplicate analysis (if requested)
        if detect_duplicates:
            print("Analyzing trajectories for potential duplicate centroids...")
            duplicate_analyses = analyze_all_trajectories(
                integrated_results, 
                coords_array, 
                expected_contact_counts, 
                threshold=duplicate_threshold
            )
            
            all_results['duplicate_analysis'] = duplicate_analyses
            
            # Generate summary statistics
            if duplicate_analyses:
                trajectories_with_duplicates = sum(1 for a in duplicate_analyses.values() 
                                                if a['duplicate_result']['duplicate_groups'])
                total_duplicate_groups = sum(len(a['duplicate_result']['duplicate_groups']) 
                                            for a in duplicate_analyses.values())
                total_centroids = sum(a['actual_count'] for a in duplicate_analyses.values())
                total_in_duplicates = sum(a['duplicate_result']['stats']['centroids_in_duplicates'] 
                                        for a in duplicate_analyses.values())
                
                all_results['duplicate_summary'] = {
                    'trajectories_analyzed': len(duplicate_analyses),
                    'trajectories_with_duplicates': trajectories_with_duplicates,
                    'percentage_with_duplicates': trajectories_with_duplicates/len(duplicate_analyses)*100 if duplicate_analyses else 0,
                    'total_duplicate_groups': total_duplicate_groups,
                    'total_centroids': total_centroids,
                    'centroids_in_duplicates': total_in_duplicates,
                    'percentage_in_duplicates': total_in_duplicates/total_centroids*100 if total_centroids else 0
                }
                
                print("\n=== Duplicate Analysis Summary ===")
                print(f"Total trajectories analyzed: {len(duplicate_analyses)}")
                print(f"Trajectories with potential duplicates: {trajectories_with_duplicates} "
                     f"({trajectories_with_duplicates/len(duplicate_analyses)*100:.1f}%)")
                print(f"Total duplicate groups: {total_duplicate_groups}")
                print(f"Total centroids: {total_centroids}")
                print(f"Centroids in duplicate groups: {total_in_duplicates} "
                     f"({total_in_duplicates/total_centroids*100:.1f}%)")
                
                # Generate PDF report for duplicate analysis
                if output_dir and use_original_reports:
                    create_duplicate_analysis_report(duplicate_analyses, output_dir)
                    print(f"✅ Duplicate centroid analysis report saved to {os.path.join(output_dir, 'duplicate_centroid_analysis.pdf')}")
        
        # Step 8: Get bolt directions and filter by hemisphere
        bolt_directions = None
        bolt_head_volume = slicer.util.getNode('P2_bolt_heads')
        
        if bolt_head_volume and entry_points_volume:
            print("Extracting bolt-to-entry directions...")
            
            # If we used combined volume, convert trajectories to bolt directions format
            if combined_trajectories and use_combined_volume:
                bolt_directions = {}
                for bolt_id, traj_info in combined_trajectories.items():
                    # Collect points (trajectory points or extract from bolt volume)
                    points = []
                    if 'trajectory_points' in traj_info:
                        points = traj_info['trajectory_points']
                    
                    # If no trajectory points, get bolt head points
                    if not points and bolt_head_volume:
                        bolt_mask = (get_array_from_volume(bolt_head_volume) > 0)
                        bolt_labeled = label(bolt_mask, connectivity=3)
                        bolt_coords = np.argwhere(bolt_labeled == bolt_id)
                        
                        for coord in bolt_coords:
                            ras = get_ras_coordinates_from_ijk(bolt_head_volume, [coord[2], coord[1], coord[0]])
                            points.append(ras)
                    
                    # Add entry point coordinates if available
                    if entry_points_volume and 'end_point' in traj_info:
                        entry_mask = (get_array_from_volume(entry_points_volume) > 0)
                        entry_labeled = label(entry_mask, connectivity=3)
                        entry_coords = np.argwhere(entry_labeled == traj_info['entry_id'])
                        
                        for coord in entry_coords:
                            ras = get_ras_coordinates_from_ijk(entry_points_volume, [coord[2], coord[1], coord[0]])
                            points.append(ras)
                    
                    bolt_directions[bolt_id] = {
                        'start_point': traj_info['start_point'],
                        'end_point': traj_info['end_point'],
                        'direction': traj_info['direction'],
                        'length': traj_info['length'],
                        'points': points,
                        'method': 'combined_volume'
                    }
                
                # NEW: Apply hemisphere filtering to bolt directions
                # (This is already done above when filtering combined_trajectories, 
                # but we ensure consistency here)
                bolt_directions = filter_bolt_directions_by_hemisphere(
                    bolt_directions, hemisphere, verbose=False  # Already printed above
                )
            
            print(f"Found {len(bolt_directions) if bolt_directions else 0} bolt-to-entry directions.")
            all_results['bolt_directions'] = bolt_directions
            
            # Match bolt directions to trajectories
            if bolt_directions and integrated_results.get('trajectories'):
                print("Matching bolt directions to trajectories...")
                matches = match_bolt_directions_to_trajectories(
                    bolt_directions, integrated_results['trajectories'],
                    max_distance=40,
                    max_angle=40.0
                )
                integrated_results['bolt_trajectory_matches'] = matches
                all_results['bolt_trajectory_matches'] = matches
                print(f"Found {len(matches)} matches between bolt directions and trajectories.")
                
            # NEW: Add entry angle validation if requested
            if validate_entry_angles and bolt_directions and brain_volume:
                print("Validating entry angles against surgical constraints (30-60°)...")
                verify_directions_with_brain(bolt_directions, brain_volume)
                
                # Count valid/invalid angles
                valid_angles = sum(1 for info in bolt_directions.values() if info.get('is_angle_valid', False))
                total_angles = len(bolt_directions)
                
                print(f"Entry angle validation: {valid_angles}/{total_angles} valid ({valid_angles/total_angles*100:.1f}%)")
                
                # Create visualization if reports are enabled
                if use_original_reports:
                    entry_validation_fig = visualize_entry_angle_validation(
                        bolt_directions, 
                        brain_volume, 
                        entry_angle_dir if 'entry_angle_dir' in locals() else output_dir
                    )
                    
                    if 'figures' not in integrated_results:
                        integrated_results['figures'] = {}
                    
                    integrated_results['figures']['entry_angle_validation'] = entry_validation_fig
                    
                    plt.close(entry_validation_fig)
                    
                all_results['entry_angle_validation'] = {
                    'valid_count': valid_angles,
                    'total_count': total_angles,
                    'valid_percentage': valid_angles/total_angles*100 if total_angles > 0 else 0
                }
        
        # Step 9: Generate spacing validation reports if enabled
        if validate_spacing and use_original_reports:
            print("Generating spacing validation reports...")
            
            # Create spacing validation page
            spacing_fig = create_spacing_validation_page(integrated_results)
            
            with PdfPages(os.path.join(output_dir, 'spacing_validation_report.pdf')) as pdf:
                pdf.savefig(spacing_fig)
                plt.close(spacing_fig)
            
            # Create enhanced 3D visualization with spacing issues highlighted
            spacing_3d_fig = enhance_3d_visualization_with_spacing(coords_array, integrated_results)
            
            with PdfPages(os.path.join(output_dir, 'spacing_validation_3d.pdf')) as pdf:
                pdf.savefig(spacing_3d_fig)
                plt.close(spacing_3d_fig)
            
            print(f"✅ Spacing validation reports saved to {output_dir}")
            
            # Add the figures to the results
            if 'figures' not in integrated_results:
                integrated_results['figures'] = {}
            integrated_results['figures']['spacing_validation'] = spacing_fig
            integrated_results['figures']['spacing_validation_3d'] = spacing_3d_fig
        
        # NEW: Generate hemisphere comparison visualization if reports are enabled and hemisphere filtering was applied
        if use_original_reports and hemisphere.lower() != 'both':
            print("Generating hemisphere comparison visualization...")
            
            # Create a mock "original" results for comparison
            original_results_mock = {
                'trajectories': [],  # Would need original trajectories for full comparison
                'n_trajectories': len(original_coords_array)  # Simplified for demo
            }
            
            hemisphere_comparison_fig = create_hemisphere_comparison_visualization(
                original_coords_array, original_results_mock, integrated_results, hemisphere
            )
            
            with PdfPages(os.path.join(output_dir, f'hemisphere_filtering_{hemisphere}.pdf')) as pdf:
                pdf.savefig(hemisphere_comparison_fig)
                plt.close(hemisphere_comparison_fig)
            
            print(f"✅ Hemisphere comparison report saved to {os.path.join(output_dir, f'hemisphere_filtering_{hemisphere}.pdf')}")
        
        # Step 10: Generate other reports
        if use_original_reports:
            print("Generating detailed analysis reports...")
            visualize_combined_results(coords_array, integrated_results, output_dir, bolt_directions)
            
            # Add electrode validation report to PDF
            if 'electrode_validation' in integrated_results and 'figures' in integrated_results:
                validation_fig = integrated_results['figures'].get('electrode_validation')
                if validation_fig:
                    with PdfPages(os.path.join(output_dir, 'electrode_validation_report.pdf')) as pdf:
                        pdf.savefig(validation_fig)
                        plt.close(validation_fig)
                    print(f"✅ Electrode validation report saved to {os.path.join(output_dir, 'electrode_validation_report.pdf')}")
        
        # Create combined volume PDF report
        if combined_trajectories and use_combined_volume and use_original_reports:
            print("Generating combined volume report...")
            with PdfPages(os.path.join(output_dir, 'combined_volume_trajectory_report.pdf')) as pdf:
                # Visualization page
                fig = visualize_combined_volume_trajectories(
                    combined_trajectories,
                    coords_array=coords_array,
                    brain_volume=brain_volume
                )
                pdf.savefig(fig)
                plt.close(fig)
                
                # Create trajectory details page
                fig = plt.figure(figsize=(12, 10))
                fig.suptitle('Combined Volume Trajectory Details', fontsize=16)
                
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                table_data = []
                columns = ['Bolt ID', 'Entry ID', 'Length (mm)', 'Angle X (°)', 'Angle Y (°)', 'Angle Z (°)']
                
                for bolt_id, traj_info in combined_trajectories.items():
                    direction = np.array(traj_info['direction'])
                    length = traj_info['length']
                    
                    angles = calculate_angles(direction)
                    
                    row = [
                        bolt_id,
                        traj_info['entry_id'],
                        f"{length:.1f}",
                        f"{angles['X']:.1f}",
                        f"{angles['Y']:.1f}",
                        f"{angles['Z']:.1f}"
                    ]
                    table_data.append(row)
                
                if table_data:
                    table = ax.table(cellText=table_data, colLabels=columns, 
                                     loc='center', cellLoc='center')
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 1.5)
                else:
                    ax.text(0.5, 0.5, "No trajectory data available", 
                           ha='center', va='center', fontsize=14)
                
                pdf.savefig(fig)
                plt.close(fig)
            
            print(f"✅ Combined volume report saved to {os.path.join(output_dir, 'combined_volume_trajectory_report.pdf')}")
        
        # After both analyses are complete, compare trajectories from both methods
        if combined_trajectories and 'trajectories' in integrated_results and use_original_reports:
            print("Comparing trajectories from both methods...")
            comparison = compare_trajectories_with_combined_data(
                integrated_results, combined_trajectories
            )
            all_results['trajectory_comparison'] = comparison
            
            # Create visualization
            comparison_fig = visualize_trajectory_comparison(
                coords_array, integrated_results, combined_trajectories, comparison
            )
            
            # Save to PDF
            with PdfPages(os.path.join(output_dir, 'trajectory_comparison.pdf')) as pdf:
                pdf.savefig(comparison_fig)
                plt.close(comparison_fig)
            
            print(f"✅ Trajectory comparison report saved to {os.path.join(output_dir, 'trajectory_comparison.pdf')}")
            
            # Print summary statistics
            summary = comparison['summary']
            print(f"Trajectory comparison summary:")
            print(f"- Integrated trajectories: {summary['integrated_trajectories']}")
            print(f"- Combined trajectories: {summary['combined_trajectories']}")
            print(f"- Matching trajectories: {summary['matching_trajectories']} ({summary['matching_percentage']:.1f}%)")
        
        # Report execution time
        finish_time = time.time()
        execution_time = finish_time - start_time
        all_results['execution_time'] = execution_time
        
        print(f"\nAnalysis Summary:")
        # NEW: Show hemisphere filtering results
        if hemisphere.lower() != 'both':
            hemisphere_info = all_results['hemisphere_filtering']
            print(f"- Hemisphere filtering ({hemisphere}): {hemisphere_info['filtered_count']} of {hemisphere_info['original_count']} "
                  f"coordinates ({hemisphere_info['filtering_efficiency']:.1f}%)")
            print(f"- Discarded coordinates: {hemisphere_info['discarded_count']}")
        
        print(f"- Analyzed {len(coords_array)} electrode coordinates")
        print(f"- Combined volume trajectories: {len(combined_trajectories) if combined_trajectories else 0}")
        print(f"- Integrated analysis trajectories: {integrated_results.get('n_trajectories', 0)}")
        print(f"- Bolt-trajectory matches: {len(integrated_results.get('bolt_trajectory_matches', {}))}")
        
        # Add electrode validation summary to final report
        if 'electrode_validation' in integrated_results and 'summary' in integrated_results['electrode_validation']:
            validation_summary = integrated_results['electrode_validation']['summary'] 
            print(f"- Electrode validation: {validation_summary['match_percentage']:.1f}% match with expected contact counts")
            print(f"- Valid electrodes: {validation_summary['valid_clusters']} of {validation_summary['total_clusters']}")
        
        # Add adaptive clustering summary if used
        if use_adaptive_clustering and 'adaptive_parameters' in all_results:
            adaptive_params = all_results['adaptive_parameters']
            print(f"- Adaptive clustering parameters: eps={adaptive_params['optimal_eps']:.2f}, "
                  f"min_neighbors={adaptive_params['optimal_min_neighbors']}")
            print(f"- Parameter search score: {adaptive_params['score']:.2f} "
                  f"(from {adaptive_params['iterations']} iterations)")
        
        # Add trajectory refinement summary if performed
        if refine_trajectories and 'trajectory_refinement' in integrated_results:
            refinement = integrated_results['trajectory_refinement']
            print(f"- Trajectory refinement: {refinement['original_count']} original -> {refinement['n_trajectories']} final trajectories")
            print(f"- Merged trajectories: {refinement['merged_count']}")
            print(f"- Split trajectories: {refinement['split_count']}")
        
        # Add duplicate analysis summary if performed
        if detect_duplicates and 'duplicate_summary' in all_results:
            dup_summary = all_results['duplicate_summary']
            print(f"- Duplicate analysis: {dup_summary['trajectories_with_duplicates']} of {dup_summary['trajectories_analyzed']} "
                  f"trajectories ({dup_summary['percentage_with_duplicates']:.1f}%) have potential duplicates")
            print(f"- Total duplicate groups: {dup_summary['total_duplicate_groups']}")
            print(f"- Centroids in duplicates: {dup_summary['centroids_in_duplicates']} of {dup_summary['total_centroids']} "
                  f"({dup_summary['percentage_in_duplicates']:.1f}%)")
        
        # Add spacing validation summary to final report
        if validate_spacing and 'spacing_validation_summary' in integrated_results:
            spacing_summary = integrated_results['spacing_validation_summary']
            print(f"- Spacing validation: {spacing_summary['valid_trajectories']} of {spacing_summary['total_trajectories']} "
                  f"trajectories ({spacing_summary['valid_percentage']:.1f}%) have valid spacing")
            print(f"- Mean contact spacing: {spacing_summary['mean_spacing']:.2f}mm (expected: {expected_spacing_range[0]}-{expected_spacing_range[1]}mm)")
        
        # Add entry angle validation summary to final report
        if validate_entry_angles and 'entry_angle_validation' in all_results:
            angle_validation = all_results['entry_angle_validation']
            print(f"- Entry angle validation: {angle_validation['valid_count']} of {angle_validation['total_count']} "
                  f"trajectories ({angle_validation['valid_percentage']:.1f}%) have valid entry angles (30-60°)")
        
        minutes = int(execution_time // 60)
        seconds = execution_time % 60
        print(f"- Total execution time: {minutes} min {seconds:.2f} sec")
        
        print(f"✅ Results saved to {output_dir}")
        
        return all_results
        
    except Exception as e:
        logging.error(f"Electrode trajectory analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'traceback': traceback.format_exc()}

# Example usage in __main__ block
if __name__ == "__main__":
    # Set parameters here for easy modification
    results = main(
        use_combined_volume=True,            # Whether to analyze combined volume
        use_original_reports=True,           # Whether to generate visualization reports
        detect_duplicates=True,              # Whether to detect duplicate centroids
        duplicate_threshold=2.5,             # Distance threshold for duplicate detection (mm)
        use_adaptive_clustering=True,        # Whether to use adaptive parameter selection
        max_iterations=8,                    # Maximum iterations for parameter search
        validate_spacing=True,               # Enable spacing validation
        expected_spacing_range=(3.0, 5.0),   # Set expected spacing range (3-5mm)
        refine_trajectories=True,            # Enable trajectory refinement (merging/splitting)
        max_contacts_per_trajectory=18,      # Maximum contacts allowed in a single trajectory
        validate_entry_angles=True,          # Enable entry angle validation (30-60°)
        hemisphere='left'                    # NEW: Set hemisphere ('left', 'right', or 'both')
    )
    print("Analysis completed.")

# Additional convenience functions for hemisphere-specific analysis
def analyze_left_hemisphere():
    """Convenience function to analyze only left hemisphere electrodes."""
    return main(hemisphere='left')

def analyze_right_hemisphere():
    """Convenience function to analyze only right hemisphere electrodes.""" 
    return main(hemisphere='right')

def analyze_both_hemispheres():
    """Convenience function to analyze all electrodes (no hemisphere filtering)."""
    return main(hemisphere='both')

def compare_hemispheres():
    """
    Compare analysis results between left and right hemispheres.
    
    Returns:
        dict: Comparison results between hemispheres
    """
    print("Running hemisphere comparison analysis...")
    
    # Analyze left hemisphere
    print("\n" + "="*50)
    print("ANALYZING LEFT HEMISPHERE")
    print("="*50)
    left_results = main(hemisphere='left', use_original_reports=False)
    
    # Analyze right hemisphere  
    print("\n" + "="*50)
    print("ANALYZING RIGHT HEMISPHERE")
    print("="*50)
    right_results = main(hemisphere='right', use_original_reports=False)
    
    # Create comparison
    comparison = {
        'left_hemisphere': left_results,
        'right_hemisphere': right_results,
        'comparison_summary': {
            'left_electrodes': left_results.get('electrode_count', 0),
            'right_electrodes': right_results.get('electrode_count', 0),
            'left_trajectories': left_results.get('integrated_analysis', {}).get('n_trajectories', 0),
            'right_trajectories': right_results.get('integrated_analysis', {}).get('n_trajectories', 0),
            'total_electrodes': left_results.get('electrode_count', 0) + right_results.get('electrode_count', 0),
            'total_trajectories': (left_results.get('integrated_analysis', {}).get('n_trajectories', 0) + 
                                 right_results.get('integrated_analysis', {}).get('n_trajectories', 0))
        }
    }
    
    # Print comparison summary
    print("\n" + "="*50)
    print("HEMISPHERE COMPARISON SUMMARY")
    print("="*50)
    summary = comparison['comparison_summary']
    print(f"Left hemisphere: {summary['left_electrodes']} electrodes, {summary['left_trajectories']} trajectories")
    print(f"Right hemisphere: {summary['right_electrodes']} electrodes, {summary['right_trajectories']} trajectories")
    print(f"Total: {summary['total_electrodes']} electrodes, {summary['total_trajectories']} trajectories")
    
    if summary['total_electrodes'] > 0:
        left_percentage = (summary['left_electrodes'] / summary['total_electrodes']) * 100
        right_percentage = (summary['right_electrodes'] / summary['total_electrodes']) * 100
        print(f"Distribution: {left_percentage:.1f}% left, {right_percentage:.1f}% right")
    
    return comparison

#------------------------------------------------------------------------------
#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Electrode_path\orga.py').read())