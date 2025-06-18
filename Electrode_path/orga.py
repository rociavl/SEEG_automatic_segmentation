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
import plotly.graph_objects as go
import plotly.express as px

# Import utility functions from external modules
from Outermost_centroids_coordinates.outermost_centroids_vol_slicer import (
    get_ras_coordinates_from_ijk, get_array_from_volume, calculate_centroids_numpy,
    get_centroids_ras, get_surface_from_volume, convert_surface_vertices_to_ras, 
    filter_centroids_by_surface_distance
)
from End_points.midplane_prueba import get_all_centroids
from Electrode_path.construction_4 import (create_3d_visualization,
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

def create_summary_page(results):
    fig = plt.figure(figsize=(12, 15))
    fig.suptitle('Trajectory Analysis Summary Report', fontsize=16, y=0.98)
    
    # Create grid layout
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1, 1, 1, 2])
    
    # Parameters section
    ax1 = fig.add_subplot(gs[0])
    ax1.axis('off')
    
    params_text = "Analysis Parameters:\n"
    
    # Handle different parameter structures
    if 'parameters' in results:
        params = results['parameters']
        
        # Check for adaptive clustering parameters
        if 'use_adaptive_clustering' in params and params['use_adaptive_clustering']:
            if 'adaptive_parameters' in results:
                adaptive_params = results['adaptive_parameters']
                params_text += f"- Adaptive clustering used: Yes\n"
                params_text += f"- Optimal eps: {adaptive_params['optimal_eps']:.2f} mm\n"
                params_text += f"- Optimal min neighbors: {adaptive_params['optimal_min_neighbors']}\n"
                params_text += f"- Parameter search score: {adaptive_params['score']:.2f}\n"
            else:
                params_text += f"- Adaptive clustering: Yes (parameters not found)\n"
        else:
            # Fixed parameters - try to get from integrated_analysis
            if 'integrated_analysis' in results and 'parameters' in results['integrated_analysis']:
                int_params = results['integrated_analysis']['parameters']
                params_text += f"- Max neighbor distance: {int_params.get('max_neighbor_distance', 'N/A')} mm\n"
                params_text += f"- Min neighbors: {int_params.get('min_neighbors', 'N/A')}\n"
            else:
                params_text += f"- Fixed parameters used (details not available)\n"
        
        # Other parameters
        params_text += f"- Expected contact counts: {params.get('expected_contact_counts', 'N/A')}\n"
        params_text += f"- Hemisphere: {params.get('hemisphere', 'both')}\n"
        params_text += f"- Validate spacing: {params.get('validate_spacing', False)}\n"
        if params.get('validate_spacing', False):
            spacing_range = params.get('expected_spacing_range', (3.0, 5.0))
            params_text += f"- Expected spacing range: {spacing_range[0]}-{spacing_range[1]} mm\n"
        params_text += f"- Refine trajectories: {params.get('refine_trajectories', False)}\n"
        params_text += f"- Validate entry angles: {params.get('validate_entry_angles', False)}\n"
    
    # Electrode count information
    if 'electrode_count' in results:
        params_text += f"- Total electrodes analyzed: {results['electrode_count']}\n"
    if 'original_electrode_count' in results:
        params_text += f"- Original electrodes: {results['original_electrode_count']}\n"
    
    # Hemisphere filtering info
    if 'hemisphere_filtering' in results:
        hemi_info = results['hemisphere_filtering']
        if hemi_info['hemisphere'] != 'both':
            params_text += f"- Hemisphere filtering: {hemi_info['hemisphere']} ({hemi_info['filtering_efficiency']:.1f}% retained)\n"
    
    params_text += "\n"
    
    # Get DBSCAN results from integrated_analysis
    integrated_results = results.get('integrated_analysis', {})
    
    if 'dbscan' in integrated_results:
        dbscan_results = integrated_results['dbscan']
        params_text += "DBSCAN Results:\n"
        params_text += f"- Number of clusters: {dbscan_results.get('n_clusters', 'N/A')}\n"
        params_text += f"- Noise points: {dbscan_results.get('noise_points', 'N/A')}\n"
        params_text += f"- Cluster sizes: {dbscan_results.get('cluster_sizes', 'N/A')}\n\n"
    
    # Louvain community detection
    if 'louvain' in integrated_results and 'error' not in integrated_results['louvain']:
        louvain_results = integrated_results['louvain']
        params_text += "Louvain Community Detection:\n"
        params_text += f"- Number of communities: {louvain_results.get('n_communities', 'N/A')}\n"
        params_text += f"- Modularity score: {louvain_results.get('modularity', 0):.3f}\n"
        params_text += f"- Community sizes: {louvain_results.get('community_sizes', 'N/A')}\n\n"
    
    # Trajectory count
    n_trajectories = integrated_results.get('n_trajectories', results.get('n_trajectories', 0))
    params_text += f"Trajectories Detected: {n_trajectories}\n"
    
    # Add hemisphere splitting info if available
    if 'hemisphere_splitting' in integrated_results:
        split_info = integrated_results['hemisphere_splitting']
        if split_info['splits_performed'] > 0:
            params_text += f"- Hemisphere splits performed: {split_info['splits_performed']}\n"
            params_text += f"- Final trajectory count: {split_info['final_count']}\n"
    
    # Add trajectory refinement info if available
    if 'trajectory_refinement' in integrated_results:
        refinement = integrated_results['trajectory_refinement']
        params_text += f"- Trajectory refinement: {refinement['merged_count']} merged, {refinement['split_count']} split\n"
    
    ax1.text(0.05, 0.95, params_text, ha='left', va='top', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Trajectory metrics table
    trajectories = integrated_results.get('trajectories', results.get('trajectories', []))
    
    if trajectories and len(trajectories) > 0:
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        
        # Prepare data for table
        table_data = []
        columns = ['ID', 'Electrodes', 'Length (mm)', 'Linearity', 'Avg Spacing (mm)', 'Spacing Var']
        
        for traj in trajectories:
            # Handle different ID formats (including hemisphere splits)
            traj_id = traj['cluster_id']
            if 'is_hemisphere_split' in traj and traj['is_hemisphere_split']:
                traj_id = f"{traj_id} ({traj['hemisphere'][0].upper()})"
            elif 'merged_from' in traj:
                traj_id = f"{traj_id} (M)"
            elif 'split_from' in traj:
                traj_id = f"{traj_id} (S)"
            
            row = [
                traj_id,
                traj['electrode_count'],
                f"{traj.get('length_mm', 0):.1f}",
                f"{traj.get('linearity', 0):.2f}",
                f"{traj.get('avg_spacing_mm', 0):.2f}" if traj.get('avg_spacing_mm') else 'N/A',
                f"{traj.get('spacing_regularity', 0):.2f}" if traj.get('spacing_regularity') else 'N/A'
            ]
            table_data.append(row)
        
        # Sort by trajectory ID for better readability
        table_data.sort(key=lambda x: str(x[0]))
        
        # Limit to first 15 trajectories if too many
        if len(table_data) > 15:
            table_data = table_data[:15]
            # Add a note about truncation
            ax2.text(0.5, 0.02, f"Note: Showing first 15 of {len(trajectories)} trajectories",
                    ha='center', va='bottom', transform=ax2.transAxes, fontsize=10, style='italic')
        
        table = ax2.table(cellText=table_data, colLabels=columns,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)
        ax2.set_title('Trajectory Metrics', pad=20)
    else:
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        ax2.text(0.5, 0.5, 'No trajectory data available', ha='center', va='center', 
                fontsize=14, transform=ax2.transAxes)
        ax2.set_title('Trajectory Metrics', pad=20)
    
    # Cluster-community mapping
    if 'combined' in integrated_results and 'dbscan_to_louvain_mapping' in integrated_results['combined']:
        ax3 = fig.add_subplot(gs[2])
        ax3.axis('off')
        
        mapping_text = "Cluster to Community Mapping:\n\n"
        mapping = integrated_results['combined']['dbscan_to_louvain_mapping']
        
        # Limit the number of mappings shown
        mapping_items = list(mapping.items())
        if len(mapping_items) > 10:
            mapping_items = mapping_items[:10]
            mapping_text += "First 10 mappings shown:\n"
        
        for cluster, community in mapping_items:
            mapping_text += f"Cluster {cluster} → Community {community}\n"
        
        if len(mapping) > 10:
            mapping_text += f"... and {len(mapping) - 10} more mappings\n"
        
        if 'avg_cluster_purity' in integrated_results['combined']:
            mapping_text += f"\nAverage Cluster Purity: {integrated_results['combined']['avg_cluster_purity']:.2f}"
        
        ax3.text(0.05, 0.95, mapping_text, ha='left', va='top', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))
        ax3.set_title('Cluster-Community Relationships', pad=10)
    else:
        ax3 = fig.add_subplot(gs[2])
        ax3.axis('off')
        ax3.text(0.5, 0.5, 'No cluster-community mapping available', ha='center', va='center',
                fontsize=12, transform=ax3.transAxes)
        ax3.set_title('Cluster-Community Relationships', pad=10)
    
    # Additional analysis summary
    ax4 = fig.add_subplot(gs[3])
    ax4.axis('off')
    
    summary_text = "Analysis Summary:\n\n"
    
    # Electrode validation summary
    if 'electrode_validation' in integrated_results:
        validation = integrated_results['electrode_validation']['summary']
        summary_text += f"Electrode Validation:\n"
        summary_text += f"- Valid clusters: {validation['valid_clusters']}/{validation['total_clusters']} ({validation['match_percentage']:.1f}%)\n"
        summary_text += f"- Distribution by contact count: {dict(validation['by_size'])}\n\n"
    
    # Spacing validation summary
    if 'spacing_validation_summary' in integrated_results:
        spacing = integrated_results['spacing_validation_summary']
        summary_text += f"Spacing Validation:\n"
        summary_text += f"- Valid trajectories: {spacing['valid_trajectories']}/{spacing['total_trajectories']} ({spacing['valid_percentage']:.1f}%)\n"
        summary_text += f"- Mean spacing: {spacing['mean_spacing']:.2f}mm\n\n"
    
    # Contact angle analysis summary
    if 'contact_angle_analysis' in results:
        angle_analyses = results['contact_angle_analysis']
        flagged_count = sum(1 for a in angle_analyses.values() if not a['is_linear'])
        total_count = len(angle_analyses)
        summary_text += f"Contact Angle Analysis:\n"
        summary_text += f"- Flagged trajectories: {flagged_count}/{total_count} ({flagged_count/total_count*100:.1f}%)\n"
        if angle_analyses:
            all_max_angles = [a['max_curvature'] for a in angle_analyses.values()]
            summary_text += f"- Max angle deviation: {max(all_max_angles):.1f}°\n\n"
    
    # Entry angle validation summary
    if 'entry_angle_validation' in results:
        entry_val = results['entry_angle_validation']
        summary_text += f"Entry Angle Validation:\n"
        summary_text += f"- Valid angles: {entry_val['valid_count']}/{entry_val['total_count']} ({entry_val['valid_percentage']:.1f}%)\n\n"
    
    # Combined volume analysis summary
    if 'combined_volume' in results:
        combined = results['combined_volume']
        summary_text += f"Combined Volume Analysis:\n"
        summary_text += f"- Trajectories detected: {combined['trajectory_count']}\n\n"
    
    # Duplicate analysis summary
    if 'duplicate_summary' in results:
        dup_summary = results['duplicate_summary']
        summary_text += f"Duplicate Analysis:\n"
        summary_text += f"- Trajectories with duplicates: {dup_summary['trajectories_with_duplicates']}/{dup_summary['trajectories_analyzed']}\n"
        summary_text += f"- Total duplicate groups: {dup_summary['total_duplicate_groups']}\n\n"
    
    # Final scoring summary
    if 'final_scoring' in results:
        scoring = results['final_scoring']['summary']
        summary_text += f"Final Quality Scoring:\n"
        summary_text += f"- High quality (≥80): {scoring['high_quality']}\n"
        summary_text += f"- Medium quality (60-79): {scoring['medium_quality']}\n"
        summary_text += f"- Low quality (<60): {scoring['low_quality']}\n"
        summary_text += f"- Mean score: {scoring['mean_score']:.2f}\n\n"
    
    # Execution time
    if 'execution_time' in results:
        exec_time = results['execution_time']
        minutes = int(exec_time // 60)
        seconds = exec_time % 60
        summary_text += f"Execution Time: {minutes} min {seconds:.2f} sec"
    
    ax4.text(0.05, 0.95, summary_text, ha='left', va='top', fontsize=11,
            bbox=dict(facecolor='white', alpha=0.8))
    ax4.set_title('Analysis Summary', pad=10)
    
    plt.tight_layout()
    return fig

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
    FIXED: Improved entry angle calculation using actual brain surface normals.
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
        brain_alignment = np.dot(current_direction, to_brain_center)
        
        # Check 2: Compare distances to brain surface
        bolt_to_surface = cdist([bolt_point], brain_surface_points).min()
        entry_to_surface = cdist([entry_point], brain_surface_points).min()
        
        # FIXED - Check 3: Calculate angle relative to surface normal at entry point
        # Find the closest point on the brain surface to the entry point
        closest_idx = np.argmin(cdist([entry_point], brain_surface_points))
        closest_surface_point = brain_surface_points[closest_idx]
        
        # IMPROVED: Better surface normal estimation using larger neighborhood
        k = min(50, len(brain_surface_points))  # Use up to 50 nearest neighbors
        dists = cdist([closest_surface_point], brain_surface_points)[0]
        nearest_idxs = np.argsort(dists)[:k]
        nearest_points = brain_surface_points[nearest_idxs]
        
        # FIXED: Use PCA to estimate the local surface plane more accurately
        pca = PCA(n_components=3)
        # Center the points around the closest surface point
        centered_points = nearest_points - closest_surface_point
        pca.fit(centered_points)
        
        # The third component (least variance) approximates the surface normal
        surface_normal = pca.components_[2]
        
        # IMPROVED: Make sure the normal points outward from the brain
        # Use multiple reference points to determine inward/outward direction
        brain_center_local = np.mean(nearest_points, axis=0)
        to_local_center = brain_center_local - closest_surface_point
        
        # If normal points toward local brain center, flip it
        if np.dot(surface_normal, to_local_center) > 0:
            surface_normal = -surface_normal
        
        # FIXED: Calculate angle between trajectory direction and surface normal
        # The trajectory should enter at an angle to the surface normal
        dot_product = np.dot(current_direction, surface_normal)
        
        # CORRECTED: Angle with surface normal (not absolute value)
        angle_with_normal = np.degrees(np.arccos(np.clip(np.abs(dot_product), 0.0, 1.0)))
        
        # FIXED: Convert to angle with surface plane
        # If angle_with_normal is small, trajectory is perpendicular to surface
        # If angle_with_normal is ~90°, trajectory is parallel to surface
        angle_with_surface = 90 - angle_with_normal
        
        # ADDITIONAL: Calculate the actual entry angle (angle between trajectory and surface plane)
        # Project trajectory direction onto surface plane
        trajectory_on_surface = current_direction - np.dot(current_direction, surface_normal) * surface_normal
        trajectory_on_surface_norm = np.linalg.norm(trajectory_on_surface)
        
        if trajectory_on_surface_norm > 1e-6:  # Avoid division by zero
            # Angle between original trajectory and its projection on surface
            cos_entry_angle = np.dot(current_direction, trajectory_on_surface) / trajectory_on_surface_norm
            entry_angle_with_surface = np.degrees(np.arccos(np.clip(np.abs(cos_entry_angle), 0.0, 1.0)))
        else:
            # Trajectory is perpendicular to surface
            entry_angle_with_surface = 90.0
        
        # IMPROVED: Use the more accurate entry angle calculation
        final_entry_angle = entry_angle_with_surface
        
        # Check if angle is in the ideal surgical range (30-60 degrees)
        is_angle_valid = 30 <= final_entry_angle <= 60
        
        print(f"Bolt {bolt_id}: Brain alignment: {brain_alignment:.2f}, "
              f"Bolt-to-surface: {bolt_to_surface:.2f}mm, "
              f"Entry-to-surface: {entry_to_surface:.2f}mm, "
              f"Entry angle: {final_entry_angle:.2f}° ({'VALID' if is_angle_valid else 'INVALID'})")
        
        # Entry point should be closer to brain surface than bolt point
        if entry_to_surface > bolt_to_surface:
            print(f"Warning: Bolt {bolt_id} - Entry point ({entry_to_surface:.2f}mm) is "
                  f"farther from brain surface than bolt point ({bolt_to_surface:.2f}mm)")
        
        # Direction should roughly point toward brain
        if brain_alignment < 0.5:  # Less than 60° angle
            print(f"Warning: Bolt {bolt_id} - Direction may not be pointing toward brain "
                  f"(alignment: {brain_alignment:.2f})")
        
        # FIXED: Add validation metrics to direction info with corrected angle
        bolt_info['brain_alignment'] = float(brain_alignment)
        bolt_info['bolt_to_surface_dist'] = float(bolt_to_surface)
        bolt_info['entry_to_surface_dist'] = float(entry_to_surface)
        bolt_info['angle_with_surface'] = float(final_entry_angle)  # CORRECTED
        bolt_info['is_angle_valid'] = bool(is_angle_valid)
        bolt_info['surface_normal'] = surface_normal.tolist()  # Store for debugging
        bolt_info['closest_surface_point'] = closest_surface_point.tolist()

#visualization
def visualize_entry_angle_validation(bolt_directions, brain_volume, output_dir=None):
    """
    Create visualization showing the validation of entry angles relative to brain surface.
    ENHANCED: Better visualization of corrected entry angles.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Entry Angle Validation - CORRECTED CALCULATION (Ideal: 30-60° with surface)', fontsize=16)
    
    # Create grid layout
    gs = GridSpec(2, 3, figure=fig, height_ratios=[2, 1])
    
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
                c='lightblue', s=1, alpha=0.2, label='Brain Surface')
    
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
            marker_size = 120
            alpha = 1.0
        else:
            color = 'red'
            marker_size = 150
            alpha = 0.9
        
        # Plot entry point
        entry_point = np.array(bolt_info['end_point'])
        ax1.scatter(entry_point[0], entry_point[1], entry_point[2],
                   c=color, marker='*', s=marker_size, alpha=alpha, edgecolor='black')
        
        # Plot bolt point
        bolt_point = np.array(bolt_info['start_point'])
        ax1.scatter(bolt_point[0], bolt_point[1], bolt_point[2],
                   c=color, marker='o', s=80, alpha=0.8, edgecolor='black')
        
        # Plot trajectory line with thickness based on validity
        linewidth = 3 if is_valid else 4
        linestyle = '-' if is_valid else '--'
        ax1.plot([bolt_point[0], entry_point[0]],
                [bolt_point[1], entry_point[1]],
                [bolt_point[2], entry_point[2]],
                color=color, linewidth=linewidth, alpha=0.8, linestyle=linestyle)
        
        # Add label with corrected angle
        midpoint = (bolt_point + entry_point) / 2
        ax1.text(midpoint[0], midpoint[1], midpoint[2],
                f"B{bolt_id}: {angle:.1f}°",
                color=color, fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # ENHANCED: Show surface normal at entry point if available
        if 'surface_normal' in bolt_info and 'closest_surface_point' in bolt_info:
            surface_point = np.array(bolt_info['closest_surface_point'])
            surface_normal = np.array(bolt_info['surface_normal'])
            
            # Draw surface normal vector (short arrow)
            normal_end = surface_point + surface_normal * 10  # 10mm normal vector
            ax1.plot([surface_point[0], normal_end[0]],
                    [surface_point[1], normal_end[1]],
                    [surface_point[2], normal_end[2]],
                    'purple', linewidth=2, alpha=0.6)
            
            # Mark surface contact point
            ax1.scatter(surface_point[0], surface_point[1], surface_point[2],
                       c='purple', marker='^', s=60, alpha=0.8)
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'Entry Angle Validation: {valid_count}/{total_count} valid ({valid_count/total_count*100:.1f}%)\n'
                 f'Green: Valid angles (30-60°), Red: Invalid angles', fontweight='bold')
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=3, label='Valid Angle (30-60°)'),
        Line2D([0], [0], color='red', lw=4, linestyle='--', label='Invalid Angle'),
        Line2D([0], [0], color='purple', lw=2, label='Surface Normal'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='purple', markersize=8, label='Surface Contact')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Enhanced histogram of angles with detailed ranges
    ax2 = fig.add_subplot(gs[1, 0])
    if angles:
        # Create histogram with custom bins
        bins = np.arange(0, max(angles) + 5, 5)  # 5-degree bins
        n, bins_edges, patches = ax2.hist(angles, bins=bins, alpha=0.7, edgecolor='black')
        
        # Color code histogram bars
        for i, (patch, bin_start) in enumerate(zip(patches, bins_edges[:-1])):
            bin_end = bins_edges[i + 1]
            if 30 <= bin_start < 60 or (bin_start < 30 and bin_end > 30):
                patch.set_facecolor('green')
                patch.set_alpha(0.7)
            else:
                patch.set_facecolor('red')
                patch.set_alpha(0.7)
        
        # Add vertical lines for valid range
        ax2.axvline(x=30, color='green', linestyle='--', linewidth=2, alpha=0.8)
        ax2.axvline(x=60, color='green', linestyle='--', linewidth=2, alpha=0.8)
        ax2.axvspan(30, 60, alpha=0.2, color='green', label='Valid Range')
        
        ax2.set_xlabel('Entry Angle with Surface (degrees)')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Entry Angles\n(CORRECTED calculation)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No angle data available", ha='center', va='center', fontsize=14)
    
    # Detailed statistics table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    if angles:
        stats_data = [
            ['Metric', 'Value'],
            ['Total Trajectories', str(total_count)],
            ['Valid Angles (30-60°)', f"{valid_count} ({valid_count/total_count*100:.1f}%)"],
            ['Invalid Angles', f"{total_count-valid_count} ({(total_count-valid_count)/total_count*100:.1f}%)"],
            ['Mean Angle', f"{np.mean(angles):.1f}°"],
            ['Median Angle', f"{np.median(angles):.1f}°"],
            ['Min Angle', f"{np.min(angles):.1f}°"],
            ['Max Angle', f"{np.max(angles):.1f}°"],
            ['Std Deviation', f"{np.std(angles):.1f}°"]
        ]
        
        table = ax3.table(cellText=stats_data[1:], colLabels=stats_data[0],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.3)
        
        # Color code validation rows
        table[(2, 1)].set_facecolor('lightgreen')  # Valid angles
        table[(3, 1)].set_facecolor('lightcoral')  # Invalid angles
        
        ax3.set_title('Entry Angle Statistics\n(Corrected Calculation)')
    
    # Individual trajectory angle details
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    # Create detailed trajectory table
    detail_data = []
    for bolt_id, bolt_info in bolt_directions.items():
        angle = bolt_info.get('angle_with_surface', 0)
        is_valid = bolt_info.get('is_angle_valid', False)
        brain_alignment = bolt_info.get('brain_alignment', 0)
        
        status = 'VALID' if is_valid else 'INVALID'
        detail_data.append([
            f"Bolt {bolt_id}",
            f"{angle:.1f}°",
            status,
            f"{brain_alignment:.2f}"
        ])
    
    # Sort by angle
    detail_data.sort(key=lambda x: float(x[1].replace('°', '')))
    
    if detail_data:
        detail_table = ax4.table(
            cellText=detail_data, 
            colLabels=['Trajectory', 'Entry Angle', 'Status', 'Brain Align'],
            loc='center', cellLoc='center'
        )
        detail_table.auto_set_font_size(False)
        detail_table.set_fontsize(9)
        detail_table.scale(1, 1.2)
        
        # Color code status cells
        for i, row in enumerate(detail_data):
            status = row[2]
            if status == 'VALID':
                detail_table[(i+1, 2)].set_facecolor('lightgreen')
            else:
                detail_table[(i+1, 2)].set_facecolor('lightcoral')
    
    ax4.set_title('Individual Trajectory Details')
    
    plt.tight_layout()
    
    # Save figure if output directory provided
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'corrected_entry_angle_validation.png'), dpi=300)
        
        # Also save as PDF
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(os.path.join(output_dir, 'corrected_entry_angle_validation.pdf')) as pdf:
            pdf.savefig(fig)
            
        print(f"✅ Corrected entry angle validation saved to {os.path.join(output_dir, 'corrected_entry_angle_validation.pdf')}")
    
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
    FIXED VERSION: Merge two trajectories into one with proper coordinate reconstruction.
    """
    merged_traj = traj1.copy()
    
    # Use string-based ID like original code for better tracking
    merged_traj['cluster_id'] = f"M_{traj1['cluster_id']}_{traj2['cluster_id']}"
    merged_traj['merged_from'] = [traj1['cluster_id'], traj2['cluster_id']]
    merged_traj['is_merged'] = True
    
    endpoints1 = np.array(traj1['endpoints'])
    endpoints2 = np.array(traj2['endpoints'])
    
    distances = [
        (0, 0, np.linalg.norm(endpoints1[0] - endpoints2[0])),
        (0, 1, np.linalg.norm(endpoints1[0] - endpoints2[1])),
        (1, 0, np.linalg.norm(endpoints1[1] - endpoints2[0])),
        (1, 1, np.linalg.norm(endpoints1[1] - endpoints2[1]))
    ]
    
    closest_pair = min(distances, key=lambda x: x[2])
    idx1, idx2, _ = closest_pair
    
    # Update endpoints
    if idx1 == 0 and idx2 == 0:
        new_endpoints = [endpoints1[1], endpoints2[1]]
    elif idx1 == 0 and idx2 == 1:
        new_endpoints = [endpoints1[1], endpoints2[0]]
    elif idx1 == 1 and idx2 == 0:
        new_endpoints = [endpoints1[0], endpoints2[1]]
    else:
        new_endpoints = [endpoints1[0], endpoints2[0]]
    
    merged_traj['endpoints'] = [new_endpoints[0].tolist(), new_endpoints[1].tolist()]
    
    # *** CRITICAL FIX: Combine sorted coordinates ***
    if 'sorted_coords' in traj1 and 'sorted_coords' in traj2:
        coords1 = np.array(traj1['sorted_coords'])
        coords2 = np.array(traj2['sorted_coords'])
        
        # Combine coordinates based on closest endpoint pairing
        if idx1 == 0 and idx2 == 0:
            # Connect end1[0] to end2[0], so reverse traj1 and keep traj2 normal
            sorted_coords = np.vstack([coords1[::-1], coords2])
        elif idx1 == 0 and idx2 == 1:
            # Connect end1[0] to end2[1], so reverse both
            sorted_coords = np.vstack([coords1[::-1], coords2[::-1]])
        elif idx1 == 1 and idx2 == 0:
            # Connect end1[1] to end2[0], so keep both normal
            sorted_coords = np.vstack([coords1, coords2])
        else:  # idx1 == 1 and idx2 == 1
            # Connect end1[1] to end2[1], so keep traj1 normal and reverse traj2
            sorted_coords = np.vstack([coords1, coords2[::-1]])
            
        merged_traj['sorted_coords'] = sorted_coords.tolist()
    else:
        # Fallback: try to reconstruct from endpoints
        print(f"Warning: Missing sorted_coords for merge, using endpoint interpolation")
        n_contacts = traj1['electrode_count'] + traj2['electrode_count']
        t = np.linspace(0, 1, n_contacts)
        interpolated_coords = []
        for i in range(n_contacts):
            point = new_endpoints[0] + t[i] * (new_endpoints[1] - new_endpoints[0])
            interpolated_coords.append(point.tolist())
        merged_traj['sorted_coords'] = interpolated_coords
    
    # Update other properties
    new_direction = new_endpoints[1] - new_endpoints[0]
    new_length = np.linalg.norm(new_direction)
    
    merged_traj['direction'] = (new_direction / new_length).tolist()
    merged_traj['length_mm'] = float(new_length)
    merged_traj['electrode_count'] = traj1['electrode_count'] + traj2['electrode_count']
    merged_traj['center'] = ((new_endpoints[0] + new_endpoints[1]) / 2).tolist()
    
    return merged_traj

def get_trajectory_coordinates(traj_id, results, coords_array, clusters):
    """
    IMPROVED VERSION: Get coordinates for any trajectory type with better fallbacks.
    """
    # Method 1: Check if trajectory has sorted_coords (for merged/split trajectories)
    for traj in results.get('trajectories', []):
        if str(traj['cluster_id']) == str(traj_id):
            if 'sorted_coords' in traj and traj['sorted_coords']:
                return np.array(traj['sorted_coords'])
            break
    
    # Method 2: For regular cluster IDs, try cluster mapping
    if isinstance(traj_id, (int, np.integer)) or (isinstance(traj_id, str) and traj_id.isdigit()):
        try:
            cluster_id_int = int(traj_id)
            mask = clusters == cluster_id_int
            if np.any(mask):
                return coords_array[mask]
        except (ValueError, TypeError):
            pass
    
    # Method 3: For merged trajectories (M_X_Y format), try to extract original IDs
    if isinstance(traj_id, str) and traj_id.startswith('M_'):
        try:
            parts = traj_id.replace('M_', '').split('_')
            if len(parts) >= 2:
                orig_id1, orig_id2 = int(parts[0]), int(parts[1])
                mask1 = clusters == orig_id1
                mask2 = clusters == orig_id2
                if np.any(mask1) and np.any(mask2):
                    # Return combined coordinates
                    coords1 = coords_array[mask1]
                    coords2 = coords_array[mask2]
                    return np.vstack([coords1, coords2])
        except (ValueError, TypeError):
            pass
    
    # Method 4: For split trajectories, try to reconstruct from endpoints
    for traj in results.get('trajectories', []):
        if str(traj['cluster_id']) == str(traj_id):
            if 'endpoints' in traj:
                endpoints = np.array(traj['endpoints'])
                if len(endpoints) == 2:
                    # Create a simple line between endpoints
                    n_points = traj.get('electrode_count', 10)
                    t = np.linspace(0, 1, n_points)
                    coords = []
                    for i in range(n_points):
                        point = endpoints[0] + t[i] * (endpoints[1] - endpoints[0])
                        coords.append(point)
                    return np.array(coords)
            break
    
    print(f"Warning: Could not find coordinates for trajectory {traj_id}")
    return None

def split_trajectory(traj, expected_contact_counts=[5, 8, 10, 12, 15, 18]):
    """
    FIXED VERSION: Split a trajectory that may contain multiple electrodes.
    Now properly generates string-based IDs and ensures sorted_coords are available.
    """
    if 'sorted_coords' not in traj or not traj['sorted_coords']:
        return {'success': False, 'reason': 'No coordinates available', 'trajectories': [traj]}
    
    coords = np.array(traj['sorted_coords'])
    contact_count = traj['electrode_count']
    
    # Find potential combinations of expected counts
    best_combination = None
    min_difference = float('inf')
    
    for count1 in expected_contact_counts:
        for count2 in expected_contact_counts:
            if abs((count1 + count2) - contact_count) < min_difference:
                min_difference = abs((count1 + count2) - contact_count)
                best_combination = [count1, count2]
    
    if min_difference > 0 and contact_count > 20:
        for count1 in expected_contact_counts:
            for count2 in expected_contact_counts:
                for count3 in expected_contact_counts:
                    diff = abs((count1 + count2 + count3) - contact_count)
                    if diff < min_difference:
                        min_difference = diff
                        best_combination = [count1, count2, count3]
    
    if best_combination is None or min_difference > 5:
        return {'success': False, 'reason': 'No good contact count combination found', 'trajectories': [traj]}
    
    # Apply clustering to split
    from sklearn.cluster import KMeans
    
    n_clusters = len(best_combination)
    splitter = KMeans(n_clusters=n_clusters, random_state=42)
    labels = splitter.fit_predict(coords)
    unique_labels = set(labels)
    
    split_trajectories = []
    base_id = traj['cluster_id']
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        sub_coords = coords[mask]
        
        if len(sub_coords) < 3:
            continue
            
        pca = PCA(n_components=3)
        pca.fit(sub_coords)
        
        sub_traj = traj.copy()
        
        # *** CRITICAL FIX: Use string-based ID like original code ***
        new_id = f"S{i+1}_{base_id}"
        
        sub_traj['cluster_id'] = new_id
        sub_traj['electrode_count'] = len(sub_coords)
        
        # Store split information in metadata
        sub_traj['is_split'] = True
        sub_traj['split_from'] = base_id
        sub_traj['split_index'] = i + 1
        sub_traj['split_label'] = f"S{i+1}_{base_id}"
        
        # Sort coordinates along trajectory direction
        direction = pca.components_[0]
        center = np.mean(sub_coords, axis=0)
        projected = np.dot(sub_coords - center, direction)
        sorted_indices = np.argsort(projected)
        sorted_coords = sub_coords[sorted_indices]
        
        # *** CRITICAL FIX: Store the sorted coordinates ***
        sub_traj['sorted_coords'] = sorted_coords.tolist()
        
        sub_traj['endpoints'] = [
            sorted_coords[0].tolist(),
            sorted_coords[-1].tolist()
        ]
        
        sub_traj['direction'] = direction.tolist()
        sub_traj['center'] = center.tolist()
        sub_traj['length_mm'] = float(np.linalg.norm(
            sorted_coords[-1] - sorted_coords[0]
        ))
        
        # Update linearity using PCA
        if len(sub_coords) > 2:
            sub_traj['linearity'] = float(pca.explained_variance_ratio_[0])
            sub_traj['pca_variance'] = pca.explained_variance_ratio_.tolist()
        else:
            sub_traj['linearity'] = 1.0
            sub_traj['pca_variance'] = [1.0, 0.0, 0.0]
        
        # Copy spacing validation if it exists
        if 'spacing_validation' in traj:
            spacing_validation = validate_electrode_spacing(
                sorted_coords, 
                traj['spacing_validation'].get('expected_range', (3.0, 5.0))
            )
            sub_traj['spacing_validation'] = spacing_validation
        
        split_trajectories.append(sub_traj)
    
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

############# angles helper 



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
# PART 2.10: VISUALIZATION HELPER FUNCTIONS  
# CONTACT ANGLE ANALYSIS MODULE
# Analyzes angles between consecutive contacts within electrode trajectories
#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

def calculate_contact_angles(trajectory_points, angle_threshold=40.0):
    """
    Calculate curvature angles and direction changes between consecutive contact segments.
    
    This function now calculates two types of angles:
    1. Curvature angle: actual angle between consecutive segments (0° = straight)
    2. Direction change: how much the trajectory direction changes at each point
    
    Args:
        trajectory_points (numpy.ndarray): Array of contact coordinates sorted along trajectory
        angle_threshold (float): Threshold angle in degrees to flag problematic segments (default: 40°)
        
    Returns:
        dict: Dictionary containing enhanced angle analysis results
    """
    if len(trajectory_points) < 3:
        return {
            'curvature_angles': [],
            'direction_changes': [],
            'flagged_segments': [],
            'max_curvature': 0,
            'mean_curvature': 0,
            'max_direction_change': 0,
            'mean_direction_change': 0,
            'std_curvature': 0,
            'is_linear': True,
            'linearity_score': 1.0,
            'total_segments': 0,
            'flagged_count': 0,
            'cumulative_direction_change': 0
        }
    
    trajectory_points = np.array(trajectory_points)
    curvature_angles = []
    direction_changes = []
    flagged_segments = []
    
    # Calculate angles between consecutive segments
    for i in range(1, len(trajectory_points) - 1):
        # Get three consecutive points
        p1 = trajectory_points[i-1]  # Previous point
        p2 = trajectory_points[i]    # Current point (vertex)
        p3 = trajectory_points[i+1]  # Next point
        
        # Calculate vectors
        v1 = p2 - p1  # Vector from p1 to p2
        v2 = p3 - p2  # Vector from p2 to p3
        
        # Skip if either vector is too short (degenerate case)
        v1_length = np.linalg.norm(v1)
        v2_length = np.linalg.norm(v2)
        
        if v1_length < 1e-6 or v2_length < 1e-6:
            curvature_angles.append(0.0)
            direction_changes.append(0.0)
            continue
        
        # Normalize vectors
        v1_norm = v1 / v1_length
        v2_norm = v2 / v2_length
        
        # 1. CURVATURE ANGLE: Angle between the two vectors
        # This is the actual bend at point p2
        dot_product = np.dot(v1_norm, v2_norm)
         # Handle numerical errors
        
        # The angle between vectors (0° = parallel, 180° = opposite)
        angle_between_vectors = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

        
        # For curvature, we want the deviation from straight (180°)
        # 0° = perfectly straight, 180° = complete reversal
        curvature_angle = angle_between_vectors 
        curvature_angles.append(curvature_angle)
        
        # 2. DIRECTION CHANGE: How much the direction vector changes
        # This measures the magnitude of direction change
        direction_diff = v2_norm - v1_norm
        direction_change_magnitude = np.linalg.norm(direction_diff)
        
        # Convert to degrees (0 = no change, 2 = complete reversal)
        # Scale to 0-180° range for consistency
        direction_change = direction_change_magnitude * 90.0  # Scale factor
        direction_changes.append(min(direction_change, 180.0))  # Cap at 180°
        
        # Flag segments that exceed threshold (using curvature angle)
        if curvature_angle > angle_threshold:
            # Calculate additional metrics for flagged segments
            segment_length_1 = v1_length
            segment_length_2 = v2_length
            total_span = np.linalg.norm(p3 - p1)  # Direct distance from p1 to p3
            
            flagged_segments.append({
                'segment_index': i,
                'contact_indices': [i-1, i, i+1],
                'points': [p1.tolist(), p2.tolist(), p3.tolist()],
                'curvature_angle': curvature_angle,
                'direction_change': direction_change,
                'segment_lengths': [segment_length_1, segment_length_2],
                'total_span': total_span,
                'vectors': [v1.tolist(), v2.tolist()],
                'curvature_severity': 'High' if curvature_angle > 60 else 'Medium'
            })
    
    # Calculate statistics
    curvature_angles = np.array(curvature_angles)
    direction_changes = np.array(direction_changes)
    
    max_curvature = np.max(curvature_angles) if len(curvature_angles) > 0 else 0
    mean_curvature = np.mean(curvature_angles) if len(curvature_angles) > 0 else 0
    std_curvature = np.std(curvature_angles) if len(curvature_angles) > 0 else 0
    
    max_direction_change = np.max(direction_changes) if len(direction_changes) > 0 else 0
    mean_direction_change = np.mean(direction_changes) if len(direction_changes) > 0 else 0
    
    # Calculate cumulative direction change (total "wiggle" in the trajectory)
    cumulative_direction_change = np.sum(direction_changes) if len(direction_changes) > 0 else 0
    
    # Calculate improved linearity score based on curvature
    # Good trajectories should have low maximum curvature and low mean curvature
    linearity_score = max(0, 1 - (max_curvature / 180.0) * 1.5 - (mean_curvature / 60.0) * 0.5)
    linearity_score = min(1.0, linearity_score)  # Cap at 1.0
    
    # Determine if trajectory is considered linear
    is_linear = max_curvature <= angle_threshold and mean_curvature <= (angle_threshold / 2)
    
    return {
        'curvature_angles': curvature_angles.tolist(),
        'direction_changes': direction_changes.tolist(),
        'flagged_segments': flagged_segments,
        'max_curvature': float(max_curvature),
        'mean_curvature': float(mean_curvature),
        'std_curvature': float(std_curvature),
        'max_direction_change': float(max_direction_change),
        'mean_direction_change': float(mean_direction_change),
        'cumulative_direction_change': float(cumulative_direction_change),
        'is_linear': bool(is_linear),
        'linearity_score': float(linearity_score),
        'total_segments': len(curvature_angles),
        'flagged_count': len(flagged_segments),
        'angle_threshold': angle_threshold
    }

def analyze_trajectory_angles(trajectories, coords_array, results, angle_threshold=40.0):
    """
    FIXED VERSION: Analyze contact angles for all trajectories including split/merged ones.
    """
    # Get cluster assignments from results
    if 'graph' not in results:
        print("Warning: No graph information available for cluster mapping")
        return {}
    
    clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
    
    trajectory_angle_analyses = {}
    
    for traj in trajectories:
        cluster_id = traj['cluster_id']
        
        # *** CRITICAL FIX: Handle split/merged trajectories ***
        # For split/merged trajectories, use sorted_coords directly instead of cluster mapping
        if 'sorted_coords' in traj and traj['sorted_coords']:
            sorted_coords = np.array(traj['sorted_coords'])
            print(f"Using sorted_coords for trajectory {cluster_id} ({len(sorted_coords)} points)")
        else:
            # Fallback to cluster mapping for regular trajectories
            mask = clusters == cluster_id
            
            if not np.any(mask):
                print(f"Warning: No coordinates found for trajectory {cluster_id} in cluster mapping")
                continue
                
            cluster_coords = coords_array[mask]
            
            # Sort coordinates along trajectory direction if we have the direction
            if 'direction' in traj and len(cluster_coords) > 2:
                direction = np.array(traj['direction'])
                center = np.mean(cluster_coords, axis=0)
                projected = np.dot(cluster_coords - center, direction)
                sorted_indices = np.argsort(projected)
                sorted_coords = cluster_coords[sorted_indices]
            else:
                sorted_coords = cluster_coords
        
        # Skip trajectories with too few points
        if len(sorted_coords) < 3:
            print(f"Skipping trajectory {cluster_id}: only {len(sorted_coords)} points")
            continue
        
        # Analyze angles for this trajectory
        angle_analysis = calculate_contact_angles(sorted_coords, angle_threshold)
        
        # Add trajectory metadata
        angle_analysis['trajectory_id'] = cluster_id
        angle_analysis['contact_count'] = len(sorted_coords)
        angle_analysis['trajectory_length'] = traj.get('length_mm', 0)
        angle_analysis['pca_linearity'] = traj.get('linearity', 0)
        
        # Add trajectory type information for debugging
        if 'is_split' in traj:
            angle_analysis['trajectory_type'] = 'split'
            angle_analysis['split_from'] = traj.get('split_from')
        elif 'is_merged' in traj:
            angle_analysis['trajectory_type'] = 'merged'
            angle_analysis['merged_from'] = traj.get('merged_from')
        elif 'is_hemisphere_split' in traj:
            angle_analysis['trajectory_type'] = 'hemisphere_split'
            angle_analysis['hemisphere'] = traj.get('hemisphere')
        else:
            angle_analysis['trajectory_type'] = 'original'
        
        trajectory_angle_analyses[cluster_id] = angle_analysis
        
        print(f"✅ Analyzed angles for trajectory {cluster_id} ({angle_analysis['trajectory_type']}): "
              f"{len(sorted_coords)} contacts, max curvature {angle_analysis['max_curvature']:.1f}°")
    
    return trajectory_angle_analyses

def create_angle_analysis_visualization(trajectory_angle_analyses, output_dir=None):
    """
    Create enhanced visualizations for contact angle analysis results with better context.
    
    Args:
        trajectory_angle_analyses (dict): Results from analyze_trajectory_angles
        output_dir (str, optional): Directory to save visualizations
        
    Returns:
        matplotlib.figure.Figure: Figure containing enhanced angle analysis visualization
    """
    if not trajectory_angle_analyses:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No trajectory angle data available', 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Enhanced Contact Angle Analysis: Trajectory Curvature Assessment', fontsize=18)
    
    # Create grid layout
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 1])
    
    # Collect data for analysis
    all_curvature_angles = []
    all_direction_changes = []
    trajectory_stats = []
    flagged_trajectories = []
    
    for traj_id, analysis in trajectory_angle_analyses.items():
        all_curvature_angles.extend(analysis.get('curvature_angles', []))
        all_direction_changes.extend(analysis.get('direction_changes', []))
        
        trajectory_stats.append({
            'trajectory_id': traj_id,
            'max_curvature': analysis.get('max_curvature', 0),
            'mean_curvature': analysis.get('mean_curvature', 0),
            'max_direction_change': analysis.get('max_direction_change', 0),
            'cumulative_direction_change': analysis.get('cumulative_direction_change', 0),
            'contact_count': analysis.get('contact_count', 0),
            'flagged_count': analysis.get('flagged_count', 0),
            'is_linear': analysis.get('is_linear', True),
            'linearity_score': analysis.get('linearity_score', 1.0),
            'pca_linearity': analysis.get('pca_linearity', 0)
        })
        
        if not analysis.get('is_linear', True):
            flagged_trajectories.append(traj_id)
    
    # 1. Distribution of curvature angles
    ax1 = fig.add_subplot(gs[0, 0])
    if all_curvature_angles:
        ax1.hist(all_curvature_angles, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=40, color='red', linestyle='--', linewidth=2, label='Threshold (40°)')
        ax1.axvline(x=np.mean(all_curvature_angles), color='orange', linestyle='-', linewidth=2, 
                   label=f'Mean ({np.mean(all_curvature_angles):.1f}°)')
        ax1.set_xlabel('Curvature Angle (degrees)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Curvature Angles\n(0° = straight, 180° = complete bend)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No curvature data available', ha='center', va='center')
    
    # 2. Distribution of direction changes
    ax2 = fig.add_subplot(gs[0, 1])
    if all_direction_changes:
        ax2.hist(all_direction_changes, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.axvline(x=np.mean(all_direction_changes), color='red', linestyle='-', linewidth=2, 
                   label=f'Mean ({np.mean(all_direction_changes):.1f}°)')
        ax2.set_xlabel('Direction Change Magnitude')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Direction Changes\n(0° = no change, higher = more change)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No direction change data available', ha='center', va='center')
    
    # 3. Curvature vs Direction Change scatter plot
    ax3 = fig.add_subplot(gs[0, 2])
    if trajectory_stats:
        max_curvatures = [t['max_curvature'] for t in trajectory_stats]
        max_direction_changes = [t['max_direction_change'] for t in trajectory_stats]
        colors = ['red' if not t['is_linear'] else 'green' for t in trajectory_stats]
        
        scatter = ax3.scatter(max_curvatures, max_direction_changes, c=colors, alpha=0.7, s=60)
        ax3.set_xlabel('Max Curvature Angle (°)')
        ax3.set_ylabel('Max Direction Change')
        ax3.set_title('Curvature vs Direction Change')
        ax3.grid(True, alpha=0.3)
        
        # Add threshold lines
        ax3.axvline(x=40, color='red', linestyle='--', alpha=0.5, label='Curvature Threshold')
        ax3.legend()
        
        # Add correlation if we have enough data
        if len(max_curvatures) > 1:
            correlation = np.corrcoef(max_curvatures, max_direction_changes)[0, 1]
            ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=ax3.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for scatter plot', ha='center', va='center')
    
    # 4. Trajectory quality pie chart
    ax4 = fig.add_subplot(gs[1, 0])
    if trajectory_stats:
        # Categorize trajectories by curvature
        excellent = sum(1 for t in trajectory_stats if t['max_curvature'] < 10)
        good = sum(1 for t in trajectory_stats if 10 <= t['max_curvature'] < 25)
        fair = sum(1 for t in trajectory_stats if 25 <= t['max_curvature'] < 40)
        poor = sum(1 for t in trajectory_stats if t['max_curvature'] >= 40)
        
        categories = ['Excellent\n(<10°)', 'Good\n(10-25°)', 'Fair\n(25-40°)', 'Poor\n(≥40°)']
        counts = [excellent, good, fair, poor]
        colors = ['darkgreen', 'lightgreen', 'orange', 'red']
        
        # Only include non-zero categories
        non_zero_categories, non_zero_counts, non_zero_colors = [], [], []
        for cat, count, color in zip(categories, counts, colors):
            if count > 0:
                non_zero_categories.append(f'{cat}\n({count})')
                non_zero_counts.append(count)
                non_zero_colors.append(color)
        
        if non_zero_counts:
            ax4.pie(non_zero_counts, labels=non_zero_categories, colors=non_zero_colors, 
                   autopct='%1.1f%%', startangle=90)
            ax4.set_title('Trajectory Quality by\nMax Curvature')
        else:
            ax4.text(0.5, 0.5, 'No quality data available', ha='center', va='center')
    
    # 5. Cumulative direction change analysis
    ax5 = fig.add_subplot(gs[1, 1])
    if trajectory_stats:
        cumulative_changes = [t['cumulative_direction_change'] for t in trajectory_stats]
        contact_counts = [t['contact_count'] for t in trajectory_stats]
        
        # Normalize by contact count to get "wiggle per contact"
        normalized_wiggle = [cum/contacts if contacts > 1 else 0 
                           for cum, contacts in zip(cumulative_changes, contact_counts)]
        
        ax5.hist(normalized_wiggle, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax5.set_xlabel('Cumulative Direction Change per Contact')
        ax5.set_ylabel('Number of Trajectories')
        ax5.set_title('Trajectory "Wiggle" Analysis\n(Total direction change / # contacts)')
        ax5.grid(True, alpha=0.3)
        
        if normalized_wiggle:
            mean_wiggle = np.mean(normalized_wiggle)
            ax5.axvline(x=mean_wiggle, color='red', linestyle='-', linewidth=2, 
                       label=f'Mean: {mean_wiggle:.1f}')
            ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'No wiggle data available', ha='center', va='center')
    
    # 6. Summary statistics table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    if trajectory_stats:
        total_trajectories = len(trajectory_stats)
        linear_trajectories = sum(1 for t in trajectory_stats if t['is_linear'])
        flagged_trajectories_count = total_trajectories - linear_trajectories
        
        # Calculate statistics
        max_curvatures = [t['max_curvature'] for t in trajectory_stats]
        mean_curvatures = [t['mean_curvature'] for t in trajectory_stats]
        
        summary_data = [
            ['Total Trajectories', str(total_trajectories)],
            ['Linear Trajectories', f"{linear_trajectories} ({linear_trajectories/total_trajectories*100:.1f}%)"],
            ['Flagged Trajectories', f"{flagged_trajectories_count} ({flagged_trajectories_count/total_trajectories*100:.1f}%)"],
            ['Max Curvature Overall', f"{max(max_curvatures):.1f}°"],
            ['Mean Max Curvature', f"{np.mean(max_curvatures):.1f}°"],
            ['Mean Avg Curvature', f"{np.mean(mean_curvatures):.1f}°"],
            ['Curvature Threshold', '40.0°']
        ]
        
        table = ax6.table(cellText=summary_data, colLabels=['Metric', 'Value'],
                         loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Color code the flagged trajectories row
        if flagged_trajectories_count > 0:
            table[(3, 1)].set_facecolor('lightcoral')  # Flagged trajectories row
        
        ax6.set_title('Curvature Analysis Summary')
    
    # 7-9. Individual trajectory examples (show 3 worst trajectories)
    if trajectory_stats:
        # Sort by max curvature (worst first)
        worst_trajectories = sorted(trajectory_stats, key=lambda x: x['max_curvature'], reverse=True)[:3]
        
        for idx, traj in enumerate(worst_trajectories):
            ax = fig.add_subplot(gs[2, idx])
            
            traj_id = traj['trajectory_id']
            if traj_id in trajectory_angle_analyses:
                analysis = trajectory_angle_analyses[traj_id]
                curvature_angles = analysis.get('curvature_angles', [])
                
                if curvature_angles:
                    # Create a line plot showing curvature along the trajectory
                    x_positions = range(len(curvature_angles))
                    ax.plot(x_positions, curvature_angles, 'o-', linewidth=2, markersize=6, 
                           color='blue', alpha=0.7)
                    
                    # Highlight flagged segments
                    flagged_segments = analysis.get('flagged_segments', [])
                    for segment in flagged_segments:
                        seg_idx = segment['segment_index'] - 1  # Convert to 0-based for plotting
                        if 0 <= seg_idx < len(curvature_angles):
                            ax.plot(seg_idx, curvature_angles[seg_idx], 'ro', markersize=10, 
                                   alpha=0.8, label='Flagged')
                    
                    # Add threshold line
                    ax.axhline(y=40, color='red', linestyle='--', alpha=0.7, label='Threshold')
                    
                    ax.set_xlabel('Contact Position Along Trajectory')
                    ax.set_ylabel('Curvature Angle (°)')
                    ax.set_title(f'Trajectory {traj_id}\nMax: {traj["max_curvature"]:.1f}°, '
                               f'Mean: {traj["mean_curvature"]:.1f}°')
                    ax.grid(True, alpha=0.3)
                    
                    if flagged_segments and idx == 0:  # Only show legend for first plot
                        ax.legend()
                else:
                    ax.text(0.5, 0.5, f'No data for\nTrajectory {traj_id}', 
                           ha='center', va='center')
                    ax.set_title(f'Trajectory {traj_id}')
            
            ax.set_xlim(-0.5, max(10, len(curvature_angles) + 0.5) if curvature_angles else 10)
    
    # 10. Detailed trajectory table (bottom row)
    ax10 = fig.add_subplot(gs[3, :])
    ax10.axis('off')
    
    if trajectory_stats:
        # Sort trajectories by max curvature (worst first)
        sorted_trajectories = sorted(trajectory_stats, key=lambda x: x['max_curvature'], reverse=True)
        
        # Create detailed table (limit to top 12 worst trajectories)
        display_trajectories = sorted_trajectories[:12]
        
        table_data = []
        columns = ['Trajectory ID', 'Contacts', 'Max Curvature (°)', 'Mean Curvature (°)', 
                  'Max Dir Change', 'Cumulative Wiggle', 'Flagged Segments', 'Linear?', 'Linearity Score']
        
        for traj in display_trajectories:
            row = [
                traj['trajectory_id'],
                traj['contact_count'],
                f"{traj['max_curvature']:.1f}",
                f"{traj['mean_curvature']:.1f}",
                f"{traj['max_direction_change']:.1f}",
                f"{traj['cumulative_direction_change']:.1f}",
                traj['flagged_count'],
                'Yes' if traj['is_linear'] else 'No',
                f"{traj['linearity_score']:.3f}"
            ]
            table_data.append(row)
        
        if table_data:
            detail_table = ax10.table(cellText=table_data, colLabels=columns,
                                   loc='center', cellLoc='center')
            detail_table.auto_set_font_size(False)
            detail_table.set_fontsize(9)
            detail_table.scale(1, 1.2)
            
            # Color code rows based on linearity
            for i, traj in enumerate(display_trajectories):
                if not traj['is_linear']:
                    # Color the entire row for flagged trajectories
                    for j in range(len(columns)):
                        detail_table[(i+1, j)].set_facecolor('lightcoral')
                elif traj['max_curvature'] > 25:
                    # Light yellow for high curvature but not flagged
                    for j in range(len(columns)):
                        detail_table[(i+1, j)].set_facecolor('lightyellow')
        
        title_suffix = f" (Top {len(display_trajectories)} by Max Curvature)" if len(sorted_trajectories) > 12 else ""
        ax10.set_title(f'Detailed Curvature Analysis{title_suffix}')
    
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        import os
        save_path = os.path.join(output_dir, 'enhanced_contact_angle_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Enhanced contact angle analysis saved to {save_path}")
    
    return fig

def create_single_trajectory_curvature_plot(trajectory_points, trajectory_id, analysis_results=None):
    """
    Create a detailed visualization of curvature for a single trajectory.
    This shows the actual trajectory path with curvature information overlaid.
    
    Args:
        trajectory_points (numpy.ndarray): Array of contact coordinates
        trajectory_id: ID of the trajectory
        analysis_results (dict, optional): Results from calculate_contact_angles
        
    Returns:
        matplotlib.figure.Figure: Figure showing trajectory curvature details
    """
    if len(trajectory_points) < 3:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Trajectory {trajectory_id}: Too few points for analysis', 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Calculate angles if not provided
    if analysis_results is None:
        analysis_results = calculate_contact_angles(trajectory_points)
    
    trajectory_points = np.array(trajectory_points)
    curvature_angles = analysis_results.get('curvature_angles', [])
    flagged_segments = analysis_results.get('flagged_segments', [])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Detailed Curvature Analysis: Trajectory {trajectory_id}', fontsize=16)
    
    # Create 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Plot the trajectory path
    ax1.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], 
             'b-', linewidth=2, alpha=0.7, label='Trajectory path')
    
    # Color-code contact points by curvature
    if curvature_angles:
        # Pad curvature_angles to match trajectory_points length
        # (curvature_angles has n-2 elements for n points)
        padded_curvatures = [0] + curvature_angles + [0]  # Add 0 for first and last points
        
        # Create colormap based on curvature severity
        norm = plt.Normalize(vmin=0, vmax=max(60, max(curvature_angles) if curvature_angles else 60))
        cmap = plt.cm.RdYlGn_r  # Red for high curvature, green for low
        
        for i, (point, curvature) in enumerate(zip(trajectory_points, padded_curvatures)):
            color = cmap(norm(curvature))
            size = 150 if curvature > 40 else 100  # Larger markers for high curvature
            marker = 'o' if curvature <= 40 else 'X'  # Different marker for flagged points
            
            ax1.scatter(point[0], point[1], point[2], 
                       c=[color], s=size, marker=marker, alpha=0.8, edgecolor='black')
            
            # Label contact points
            ax1.text(point[0], point[1], point[2], f'{i}', fontsize=8)
    else:
        # Fallback: plot all points in blue
        ax1.scatter(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], 
                   c='blue', s=100, alpha=0.8)
    
    # Highlight flagged segments with red lines
    for segment in flagged_segments:
        indices = segment['contact_indices']
        if len(indices) == 3 and all(0 <= idx < len(trajectory_points) for idx in indices):
            segment_points = trajectory_points[indices]
            ax1.plot(segment_points[:, 0], segment_points[:, 1], segment_points[:, 2], 
                    'r-', linewidth=4, alpha=0.8)
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Trajectory with Curvature Color-coding')
    
    # Add colorbar
    if curvature_angles:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, shrink=0.8)
        cbar.set_label('Curvature Angle (°)')
    
    # Plot curvature along trajectory
    ax2 = fig.add_subplot(222)
    
    if curvature_angles:
        x_positions = range(1, len(curvature_angles) + 1)  # Start from 1 (second contact)
        
        # Plot curvature line
        ax2.plot(x_positions, curvature_angles, 'o-', linewidth=2, markersize=8, 
                color='blue', alpha=0.7, label='Curvature angle')
        
        # Highlight flagged points
        for segment in flagged_segments:
            seg_idx = segment['segment_index']
            if 1 <= seg_idx <= len(curvature_angles):
                plot_idx = seg_idx - 1  # Convert to 0-based for list indexing
                ax2.plot(seg_idx, curvature_angles[plot_idx], 'ro', markersize=12, 
                        alpha=0.8, label='Flagged segment' if segment == flagged_segments[0] else "")
        
        # Add threshold line
        ax2.axhline(y=40, color='red', linestyle='--', alpha=0.7, label='Threshold (40°)')
        
        # Add severity zones
        ax2.axhspan(0, 5, alpha=0.1, color='green', label='Excellent (≤5°)')
        ax2.axhspan(5, 15, alpha=0.1, color='yellow', label='Good (5-15°)')
        ax2.axhspan(15, 40, alpha=0.1, color='orange', label='Fair (15-40°)')
        ax2.axhspan(40, max(180, max(curvature_angles) + 10), alpha=0.1, color='red', label='Poor (>40°)')
        
        ax2.set_xlabel('Contact Position Along Trajectory')
        ax2.set_ylabel('Curvature Angle (°)')
        ax2.set_title('Curvature Profile Along Trajectory')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set x-axis to show contact numbers
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels([f'C{i}' for i in x_positions])
    else:
        ax2.text(0.5, 0.5, 'No curvature data available', ha='center', va='center')
    
    # Direction change analysis
    ax3 = fig.add_subplot(223)
    
    direction_changes = analysis_results.get('direction_changes', [])
    if direction_changes:
        x_positions = range(1, len(direction_changes) + 1)
        
        ax3.plot(x_positions, direction_changes, 's-', linewidth=2, markersize=6, 
                color='purple', alpha=0.7, label='Direction change')
        
        ax3.set_xlabel('Contact Position Along Trajectory') 
        ax3.set_ylabel('Direction Change Magnitude')
        ax3.set_title('Direction Changes Along Trajectory')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Set x-axis to show contact numbers
        ax3.set_xticks(x_positions)
        ax3.set_xticklabels([f'C{i}' for i in x_positions])
        
        # Add mean line
        mean_change = np.mean(direction_changes)
        ax3.axhline(y=mean_change, color='red', linestyle='--', alpha=0.7, 
                   label=f'Mean ({mean_change:.1f})')
    else:
        ax3.text(0.5, 0.5, 'No direction change data available', ha='center', va='center')
    
    # Summary statistics table
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    
    # Create summary table
    stats_data = [
        ['Metric', 'Value'],
        ['Total Contacts', str(len(trajectory_points))],
        ['Analyzed Segments', str(len(curvature_angles))],
        ['Max Curvature', f"{analysis_results.get('max_curvature', 0):.2f}°"],
        ['Mean Curvature', f"{analysis_results.get('mean_curvature', 0):.2f}°"],
        ['Std Curvature', f"{analysis_results.get('std_curvature', 0):.2f}°"],
        ['Max Direction Change', f"{analysis_results.get('max_direction_change', 0):.2f}"],
        ['Cumulative Wiggle', f"{analysis_results.get('cumulative_direction_change', 0):.2f}"],
        ['Flagged Segments', str(analysis_results.get('flagged_count', 0))],
        ['Linear?', 'Yes' if analysis_results.get('is_linear', True) else 'No'],
        ['Linearity Score', f"{analysis_results.get('linearity_score', 1.0):.3f}"]
    ]
    
    table = ax4.table(cellText=stats_data[1:], colLabels=stats_data[0],
                     loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Color code the linear status
    is_linear = analysis_results.get('is_linear', True)
    linear_row_idx = next(i for i, row in enumerate(stats_data[1:]) if row[0] == 'Linear?')
    if not is_linear:
        table[(linear_row_idx + 1, 1)].set_facecolor('lightcoral')
    else:
        table[(linear_row_idx + 1, 1)].set_facecolor('lightgreen')
    
    ax4.set_title('Trajectory Summary Statistics')
    
    plt.tight_layout()
    return fig

def visualize_worst_trajectories(trajectory_angle_analyses, coords_array, results, n_worst=3, output_dir=None):
    """
    Create detailed visualizations for the worst (most curved) trajectories.
    
    Args:
        trajectory_angle_analyses (dict): Results from analyze_trajectory_angles
        coords_array (numpy.ndarray): Array of electrode coordinates
        results (dict): Results from trajectory analysis (contains graph for cluster mapping)
        n_worst (int): Number of worst trajectories to visualize
        output_dir (str, optional): Directory to save visualizations
        
    Returns:
        list: List of figures for the worst trajectories
    """
    if not trajectory_angle_analyses:
        return []
    
    # Get cluster assignments from results
    if 'graph' not in results:
        print("Warning: No graph information available for cluster mapping")
        return []
    
    clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
    
    # Sort trajectories by max curvature (worst first)
    trajectory_stats = []
    for traj_id, analysis in trajectory_angle_analyses.items():
        trajectory_stats.append((traj_id, analysis.get('max_curvature', 0), analysis))
    
    trajectory_stats.sort(key=lambda x: x[1], reverse=True)
    worst_trajectories = trajectory_stats[:n_worst]
    
    figures = []
    
    for rank, (traj_id, max_curvature, analysis) in enumerate(worst_trajectories, 1):
        print(f"Creating detailed visualization for trajectory {traj_id} (rank {rank}, max curvature: {max_curvature:.1f}°)")
        
        # Get coordinates for this trajectory
        mask = clusters == traj_id
        
        if not np.any(mask):
            print(f"Warning: No coordinates found for trajectory {traj_id}")
            continue
            
        cluster_coords = coords_array[mask]
        
        # Sort coordinates along trajectory direction if we have it
        trajectory_info = next((t for t in results.get('trajectories', []) if t['cluster_id'] == traj_id), None)
        
        if trajectory_info and 'direction' in trajectory_info and len(cluster_coords) > 2:
            direction = np.array(trajectory_info['direction'])
            center = np.mean(cluster_coords, axis=0)
            projected = np.dot(cluster_coords - center, direction)
            sorted_indices = np.argsort(projected)
            sorted_coords = cluster_coords[sorted_indices]
        else:
            sorted_coords = cluster_coords
        
        # Create detailed visualization
        fig = create_single_trajectory_curvature_plot(sorted_coords, traj_id, analysis)
        figures.append(fig)
        
        # Save if output directory provided
        if output_dir:
            import os
            save_path = os.path.join(output_dir, f'trajectory_{traj_id}_curvature_detail.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Detailed curvature plot saved for trajectory {traj_id}: {save_path}")
    
    return figures

def create_flagged_segments_report(trajectory_angle_analyses, output_dir=None):
    flagged_data = []
    
    for traj_id, analysis in trajectory_angle_analyses.items():
        for segment in analysis['flagged_segments']:
            flagged_data.append({
                'Trajectory_ID': traj_id,
                'Segment_Index': segment['segment_index'],
                'Contact_Indices': f"{segment['contact_indices'][0]}-{segment['contact_indices'][1]}-{segment['contact_indices'][2]}",
                'Curvature_Angle': round(segment['curvature_angle'], 2),  # FIXED: was 'angle'
                'Direction_Change': round(segment['direction_change'], 2),  # NEW: Added direction change
                'Point_1': f"({segment['points'][0][0]:.1f}, {segment['points'][0][1]:.1f}, {segment['points'][0][2]:.1f})",
                'Point_2': f"({segment['points'][1][0]:.1f}, {segment['points'][1][1]:.1f}, {segment['points'][1][2]:.1f})",
                'Point_3': f"({segment['points'][2][0]:.1f}, {segment['points'][2][1]:.1f}, {segment['points'][2][2]:.1f})",
                'Severity': segment.get('curvature_severity', 'High' if segment['curvature_angle'] > 60 else 'Medium')  # FIXED: Use curvature_angle
            })
    
    if not flagged_data:
        print("No flagged segments found - all trajectories are linear!")
        return pd.DataFrame()
    
    df = pd.DataFrame(flagged_data)
    
    # Sort by curvature angle (worst first) - FIXED: Use correct column name
    df = df.sort_values('Curvature_Angle', ascending=False)
    
    # Save to CSV if output directory provided
    if output_dir:
        import os
        csv_path = os.path.join(output_dir, 'flagged_segments_report.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Flagged segments report saved to {csv_path}")
    
    return df

def add_angle_analysis_to_trajectories(trajectories, coords_array, results, angle_threshold=40.0):
    """
    IMPROVED VERSION: Add angle analysis results directly to trajectory dictionaries.
    Now properly handles all trajectory types.
    """
    # Perform angle analysis
    angle_analyses = analyze_trajectory_angles(trajectories, coords_array, results, angle_threshold)
    
    # Add results to each trajectory
    for traj in trajectories:
        traj_id = traj['cluster_id']
        
        if traj_id in angle_analyses:
            analysis = angle_analyses[traj_id]
            
            # Add angle analysis fields
            traj['contact_angles'] = {
                'curvature_angles': analysis['curvature_angles'],
                'direction_changes': analysis['direction_changes'],
                'max_curvature': analysis['max_curvature'],
                'mean_curvature': analysis['mean_curvature'],
                'std_curvature': analysis['std_curvature'],
                'max_direction_change': analysis['max_direction_change'],
                'mean_direction_change': analysis['mean_direction_change'],
                'cumulative_direction_change': analysis['cumulative_direction_change'],
                'is_linear': analysis['is_linear'],
                'linearity_score': analysis['linearity_score'],
                'flagged_segments_count': analysis['flagged_count'],
                'angle_threshold': analysis['angle_threshold'],
                'trajectory_type': analysis.get('trajectory_type', 'unknown')  # For debugging
            }
            
            # Add flagged segments details if any
            if analysis['flagged_segments']:
                traj['contact_angles']['flagged_segments'] = analysis['flagged_segments']
                
            print(f"✅ Added angle analysis to trajectory {traj_id}: "
                  f"max curvature {analysis['max_curvature']:.1f}°, "
                  f"linear: {analysis['is_linear']}")
        else:
            # Add empty analysis for trajectories without data
            traj['contact_angles'] = {
                'curvature_angles': [],
                'direction_changes': [],
                'max_curvature': 0,
                'mean_curvature': 0,
                'std_curvature': 0,
                'max_direction_change': 0,
                'mean_direction_change': 0,
                'cumulative_direction_change': 0,
                'is_linear': True,
                'linearity_score': 1.0,
                'flagged_segments_count': 0,
                'angle_threshold': angle_threshold,
                'trajectory_type': 'missing_analysis'
            }
            
            print(f"⚠️  No angle analysis available for trajectory {traj_id}")

def ensure_angle_analysis_after_refinement(results, coords_array, analyze_contact_angles=True, angle_threshold=40.0):
    """
    Ensure angle analysis is performed AFTER all trajectory refinement operations.
    This should be called at the end of the main workflow.
    """
    if not analyze_contact_angles or 'trajectories' not in results:
        return results
    
    print("Performing final angle analysis on all trajectories (including splits/merges)...")
    
    # Perform contact angle analysis on final trajectory set
    trajectory_angle_analyses = analyze_trajectory_angles(
        results['trajectories'], 
        coords_array, 
        results, 
        angle_threshold=angle_threshold
    )
    
    # Store analysis results
    results['contact_angle_analysis'] = trajectory_angle_analyses
    
    # Add angle analysis directly to trajectory dictionaries
    add_angle_analysis_to_trajectories(
        results['trajectories'], 
        coords_array, 
        results, 
        angle_threshold
    )
    
    # Print summary
    flagged_count = sum(1 for analysis in trajectory_angle_analyses.values() 
                      if not analysis['is_linear'])
    total_count = len(trajectory_angle_analyses)
    
    print(f"Final angle analysis complete: {flagged_count}/{total_count} trajectories flagged for non-linearity")
    
    # Count by trajectory type
    type_counts = {}
    for analysis in trajectory_angle_analyses.values():
        traj_type = analysis.get('trajectory_type', 'unknown')
        type_counts[traj_type] = type_counts.get(traj_type, 0) + 1
    
    print(f"Analyzed trajectories by type: {type_counts}")
    
    return results

def print_angle_analysis_summary(trajectory_angle_analyses):
    """
    Print an enhanced summary of the contact angle analysis results.
    
    Args:
        trajectory_angle_analyses (dict): Results from analyze_trajectory_angles
    """
    if not trajectory_angle_analyses:
        print("No trajectory angle analysis data available.")
        return
    
    print("\n" + "="*60)
    print("ENHANCED CONTACT ANGLE ANALYSIS SUMMARY")
    print("="*60)
    
    total_trajectories = len(trajectory_angle_analyses)
    linear_trajectories = sum(1 for a in trajectory_angle_analyses.values() if a.get('is_linear', True))
    flagged_trajectories = total_trajectories - linear_trajectories
    
    print(f"Total trajectories analyzed: {total_trajectories}")
    print(f"Linear trajectories: {linear_trajectories} ({linear_trajectories/total_trajectories*100:.1f}%)")
    print(f"Non-linear (flagged) trajectories: {flagged_trajectories} ({flagged_trajectories/total_trajectories*100:.1f}%)")
    
    if flagged_trajectories > 0:
        print("\nNon-linear trajectories (sorted by max curvature):")
        flagged_list = []
        for traj_id, analysis in trajectory_angle_analyses.items():
            if not analysis.get('is_linear', True):
                max_curv = analysis.get('max_curvature', 0)
                mean_curv = analysis.get('mean_curvature', 0)
                segment_count = analysis.get('flagged_count', 0)
                flagged_list.append((traj_id, max_curv, mean_curv, segment_count))
        
        # Sort by max curvature
        flagged_list.sort(key=lambda x: x[1], reverse=True)
        
        for traj_id, max_curv, mean_curv, segment_count in flagged_list:
            print(f"- Trajectory {traj_id}: max curvature {max_curv:.1f}°, "
                  f"mean curvature {mean_curv:.1f}°, {segment_count} flagged segments")
    
    # Overall curvature statistics
    all_curvature_angles = []
    all_direction_changes = []
    all_max_curvatures = []
    all_mean_curvatures = []
    
    for analysis in trajectory_angle_analyses.values():
        curvature_angles = analysis.get('curvature_angles', [])
        direction_changes = analysis.get('direction_changes', [])
        
        all_curvature_angles.extend(curvature_angles)
        all_direction_changes.extend(direction_changes)
        all_max_curvatures.append(analysis.get('max_curvature', 0))
        all_mean_curvatures.append(analysis.get('mean_curvature', 0))
    
    if all_curvature_angles:
        print(f"\nOverall curvature statistics:")
        print(f"- Total contact segments analyzed: {len(all_curvature_angles)}")
        print(f"- Mean curvature angle: {np.mean(all_curvature_angles):.2f}°")
        print(f"- Maximum curvature angle: {np.max(all_curvature_angles):.2f}°")
        print(f"- Standard deviation: {np.std(all_curvature_angles):.2f}°")
        
        print(f"\nTrajectory-level statistics:")
        print(f"- Mean of trajectory max curvatures: {np.mean(all_max_curvatures):.2f}°")
        print(f"- Worst trajectory max curvature: {np.max(all_max_curvatures):.2f}°")
        print(f"- Mean of trajectory mean curvatures: {np.mean(all_mean_curvatures):.2f}°")
        
        # Count angles in different severity ranges
        excellent = sum(1 for a in all_curvature_angles if a <= 5)
        good = sum(1 for a in all_curvature_angles if 5 < a <= 15)
        fair = sum(1 for a in all_curvature_angles if 15 < a <= 40)
        poor = sum(1 for a in all_curvature_angles if a > 40)
        
        print(f"\nCurvature severity distribution:")
        print(f"- Excellent (≤5°): {excellent} ({excellent/len(all_curvature_angles)*100:.1f}%)")
        print(f"- Good (5-15°): {good} ({good/len(all_curvature_angles)*100:.1f}%)")
        print(f"- Fair (15-40°): {fair} ({fair/len(all_curvature_angles)*100:.1f}%)")
        print(f"- Poor (>40°): {poor} ({poor/len(all_curvature_angles)*100:.1f}%)")
    
    if all_direction_changes:
        print(f"\nDirection change statistics:")
        print(f"- Mean direction change: {np.mean(all_direction_changes):.2f}")
        print(f"- Maximum direction change: {np.max(all_direction_changes):.2f}")
        print(f"- Standard deviation: {np.std(all_direction_changes):.2f}")
    
    # Calculate overall trajectory "wiggle" statistics
    cumulative_wiggles = []
    for analysis in trajectory_angle_analyses.values():
        cumulative_wiggle = analysis.get('cumulative_direction_change', 0)
        contact_count = analysis.get('contact_count', 1)
        if contact_count > 1:
            cumulative_wiggles.append(cumulative_wiggle / contact_count)
    
    if cumulative_wiggles:
        print(f"\nTrajectory 'wiggle' analysis (cumulative direction change per contact):")
        print(f"- Mean wiggle per contact: {np.mean(cumulative_wiggles):.2f}")
        print(f"- Most wiggly trajectory: {np.max(cumulative_wiggles):.2f}")
        print(f"- Least wiggly trajectory: {np.min(cumulative_wiggles):.2f}")
    
    print("\nInterpretation Guide:")
    print("- Curvature angles measure actual bending at each contact point")
    print("- 0° = perfectly straight, 180° = complete reversal")
    print("- Direction changes measure how much the trajectory direction shifts")
    print("- Higher cumulative 'wiggle' indicates more tortuous trajectories")
    print("- Good SEEG trajectories should have low curvature (<15°) and minimal wiggle")

###### PLOTY 
def create_plotly_interactive_angle_visualization(trajectory_angle_analyses, coords_array, results, output_dir=None):
    """
    Create an interactive 3D Plotly visualization showing all trajectory paths with contact angles.
    Color-codes trajectories by their maximum curvature and allows interactive exploration.
    
    Args:
        trajectory_angle_analyses (dict): Results from analyze_trajectory_angles
        coords_array (numpy.ndarray): Array of electrode coordinates
        results (dict): Results from trajectory analysis (contains graph for cluster mapping)
        output_dir (str, optional): Directory to save the HTML file
        
    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        import numpy as np
    except ImportError:
        print("Warning: Plotly not available. Skipping interactive visualization.")
        return None
    
    if not trajectory_angle_analyses:
        print("No trajectory angle analysis data available for Plotly visualization.")
        return None
    
    # Get cluster assignments from results
    clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
    
    # Create the main 3D plot
    fig = go.Figure()
    
    # Color scale based on curvature severity
    def get_color_and_info(max_curvature):
        if max_curvature <= 5:
            return 'green', 'Excellent'
        elif max_curvature <= 15:
            return 'lightgreen', 'Good'
        elif max_curvature <= 40:
            return 'orange', 'Fair'
        else:
            return 'red', 'Poor'
    
    # Plot all electrode points as background
    fig.add_trace(go.Scatter3d(
        x=coords_array[:, 0],
        y=coords_array[:, 1], 
        z=coords_array[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='lightgray',
            opacity=0.3
        ),
        name='All Electrodes',
        hovertemplate='<b>Electrode</b><br>' +
                      'X: %{x:.1f} mm<br>' +
                      'Y: %{y:.1f} mm<br>' +
                      'Z: %{z:.1f} mm<extra></extra>'
    ))
    
    # Track statistics for summary
    trajectory_stats = []
    
    # Plot each trajectory with contact angle information
    for traj_id, analysis in trajectory_angle_analyses.items():
        # Get coordinates for this trajectory
        mask = clusters == traj_id
        if not np.any(mask):
            continue
            
        cluster_coords = coords_array[mask]
        
        # Sort coordinates along trajectory if we have direction info
        trajectory_info = next((t for t in results.get('trajectories', []) if t['cluster_id'] == traj_id), None)
        
        if trajectory_info and 'direction' in trajectory_info and len(cluster_coords) > 2:
            direction = np.array(trajectory_info['direction'])
            center = np.mean(cluster_coords, axis=0)
            projected = np.dot(cluster_coords - center, direction)
            sorted_indices = np.argsort(projected)
            sorted_coords = cluster_coords[sorted_indices]
        else:
            sorted_coords = cluster_coords
        
        # Get angle analysis data
        max_curvature = analysis.get('max_curvature', 0)
        mean_curvature = analysis.get('mean_curvature', 0)
        curvature_angles = analysis.get('curvature_angles', [])
        flagged_segments = analysis.get('flagged_segments', [])
        is_linear = analysis.get('is_linear', True)
        linearity_score = analysis.get('linearity_score', 1.0)
        contact_count = analysis.get('contact_count', len(sorted_coords))
        
        # Get color based on curvature
        color, quality = get_color_and_info(max_curvature)
        
        # Create hover text for trajectory
        hover_text = []
        for i, coord in enumerate(sorted_coords):
            # Get curvature angle for this contact (if available)
            if i > 0 and i <= len(curvature_angles):
                curvature_at_contact = curvature_angles[i-1]
                curvature_text = f"<br>Curvature: {curvature_at_contact:.1f}°"
            else:
                curvature_text = "<br>Curvature: N/A (endpoint)"
            
            # Check if this contact is in a flagged segment
            flagged_text = ""
            for segment in flagged_segments:
                if i in segment['contact_indices']:
                    flagged_text = "<br><b>⚠️ FLAGGED SEGMENT</b>"
                    break
            
            hover_text.append(
                f"<b>Trajectory {traj_id}</b><br>" +
                f"Contact {i+1}/{contact_count}<br>" +
                f"Position: ({coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f}) mm" +
                curvature_text +
                flagged_text +
                f"<br><br><b>Trajectory Stats:</b><br>" +
                f"Max Curvature: {max_curvature:.1f}°<br>" +
                f"Mean Curvature: {mean_curvature:.1f}°<br>" +
                f"Quality: {quality}<br>" +
                f"Linear: {'Yes' if is_linear else 'No'}<br>" +
                f"Linearity Score: {linearity_score:.3f}<br>" +
                f"Flagged Segments: {len(flagged_segments)}"
            )
        
        # Plot trajectory line
        fig.add_trace(go.Scatter3d(
            x=sorted_coords[:, 0],
            y=sorted_coords[:, 1],
            z=sorted_coords[:, 2],
            mode='lines+markers',
            line=dict(
                color=color,
                width=6 if not is_linear else 4,
                dash='dash' if not is_linear else 'solid'
            ),
            marker=dict(
                size=8 if not is_linear else 6,
                color=color,
                symbol='diamond' if not is_linear else 'circle',
                line=dict(width=2, color='black')
            ),
            name=f'T{traj_id} - {quality} (Max: {max_curvature:.1f}°)',
            hovertemplate='%{text}<extra></extra>',
            text=hover_text,
            legendgroup=quality,
            showlegend=True
        ))
        
        # Highlight flagged segments with special markers
        for segment in flagged_segments:
            segment_indices = segment['contact_indices']
            # Get the middle contact of the flagged segment (the vertex)
            if len(segment_indices) >= 2:
                vertex_idx = segment_indices[1]  # Middle contact
                if vertex_idx < len(sorted_coords):
                    vertex_coord = sorted_coords[vertex_idx]
                    
                    fig.add_trace(go.Scatter3d(
                        x=[vertex_coord[0]],
                        y=[vertex_coord[1]],
                        z=[vertex_coord[2]],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color='red',
                            symbol='x',
                            line=dict(width=3, color='darkred')
                        ),
                        name=f'Flagged Contact (T{traj_id})',
                        hovertemplate=f'<b>FLAGGED CONTACT</b><br>' +
                                      f'Trajectory {traj_id}<br>' +
                                      f'Contact {vertex_idx+1}<br>' +
                                      f'Curvature: {segment["curvature_angle"]:.1f}°<br>' +
                                      f'Severity: {segment.get("curvature_severity", "High")}<br>' +
                                      f'Position: ({vertex_coord[0]:.1f}, {vertex_coord[1]:.1f}, {vertex_coord[2]:.1f}) mm<extra></extra>',
                        showlegend=False
                    ))
        
        # Store stats for summary
        trajectory_stats.append({
            'id': traj_id,
            'max_curvature': max_curvature,
            'quality': quality,
            'is_linear': is_linear,
            'flagged_segments': len(flagged_segments)
        })
    
    # Update layout for better visualization
    fig.update_layout(
        title={
            'text': 'Interactive 3D Trajectory Analysis with Contact Angles<br>' +
                    '<sub>Color-coded by curvature quality | Flagged segments marked with ✕</sub>',
            'x': 0.5,
            'font': {'size': 16}
        },
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube'
        ),
        width=1200,
        height=800,
        hovermode='closest'
    )
    
    # Add annotation with summary statistics
    excellent_count = sum(1 for t in trajectory_stats if t['quality'] == 'Excellent')
    good_count = sum(1 for t in trajectory_stats if t['quality'] == 'Good')
    fair_count = sum(1 for t in trajectory_stats if t['quality'] == 'Fair')
    poor_count = sum(1 for t in trajectory_stats if t['quality'] == 'Poor')
    total_flagged = sum(t['flagged_segments'] for t in trajectory_stats)
    non_linear_count = sum(1 for t in trajectory_stats if not t['is_linear'])
    
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text=f"<b>Summary:</b><br>" +
             f"Total Trajectories: {len(trajectory_stats)}<br>" +
             f"🟢 Excellent (≤5°): {excellent_count}<br>" +
             f"🟡 Good (5-15°): {good_count}<br>" +
             f"🟠 Fair (15-40°): {fair_count}<br>" +
             f"🔴 Poor (>40°): {poor_count}<br>" +
             f"Non-linear: {non_linear_count}<br>" +
             f"Total Flagged Segments: {total_flagged}",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12),
        align="left"
    )
    
    # Save to HTML file if output directory provided
    if output_dir:
        import os
        html_path = os.path.join(output_dir, 'interactive_trajectory_angles.html')
        fig.write_html(html_path)
        print(f"✅ Interactive Plotly visualization saved to {html_path}")
    
    return fig

###########

def split_trajectories_at_hemisphere_boundary(trajectories, coords_array, results, hemisphere='both'):
    """
    Split trajectories that cross the hemisphere boundary (x=0 in RAS coordinates).
    
    This function identifies trajectories that have contacts in both hemispheres
    and splits them at x=0, creating separate trajectories for each hemisphere.
    
    Args:
        trajectories (list): List of trajectory dictionaries
        coords_array (numpy.ndarray): Array of electrode coordinates
        results (dict): Results from trajectory analysis (contains graph for cluster mapping)
        hemisphere (str): Target hemisphere ('left', 'right', or 'both')
        
    Returns:
        tuple: (updated_trajectories, split_info)
            - updated_trajectories: List of trajectories with splits applied
            - split_info: Dictionary with information about splits performed
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from collections import defaultdict
    
    if hemisphere.lower() != 'both':
        # Hemisphere filtering already applied, no need to split
        return trajectories, {'splits_performed': 0, 'original_count': len(trajectories)}
    
    # Get cluster assignments from results
    clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
    
    updated_trajectories = []
    split_info = {
        'splits_performed': 0,
        'original_count': len(trajectories),
        'split_trajectories': [],
        'hemisphere_target': hemisphere
    }
    
    # Maximum existing cluster ID to ensure we create unique IDs
    existing_ids = [traj['cluster_id'] for traj in trajectories]
    max_id = max([int(id) if isinstance(id, (int, np.integer)) else 0 for id in existing_ids] + [0])
    id_offset = max_id + 1000  # Large offset to avoid conflicts
    
    for traj in trajectories:
        cluster_id = traj['cluster_id']
        
        # Get coordinates for this trajectory
        mask = clusters == cluster_id
        if not np.any(mask):
            # No coordinates found, keep original trajectory
            updated_trajectories.append(traj)
            continue
            
        cluster_coords = coords_array[mask]
        
        # Check if trajectory crosses hemisphere boundary (x=0)
        x_coords = cluster_coords[:, 0]  # RAS x coordinates
        has_left = np.any(x_coords < 0)   # Left hemisphere (x < 0)
        has_right = np.any(x_coords > 0)  # Right hemisphere (x > 0)
        
        if not (has_left and has_right):
            # Trajectory doesn't cross boundary, keep original
            updated_trajectories.append(traj)
            continue
        
        print(f"Splitting trajectory {cluster_id} at hemisphere boundary (x=0)")
        
        # Split coordinates by hemisphere
        left_mask = x_coords < 0
        right_mask = x_coords > 0
        boundary_mask = np.abs(x_coords) < 0.5  # Points very close to boundary
        
        left_coords = cluster_coords[left_mask]
        right_coords = cluster_coords[right_mask]
        boundary_coords = cluster_coords[boundary_mask]
        
        # Add boundary points to both sides if they exist
        if len(boundary_coords) > 0:
            if len(left_coords) > 0:
                left_coords = np.vstack([left_coords, boundary_coords])
            if len(right_coords) > 0:
                right_coords = np.vstack([right_coords, boundary_coords])
        
        # Create split trajectories
        split_trajectories = []
        hemisphere_coords = {'left': left_coords, 'right': right_coords}
        hemisphere_names = {'left': 'L', 'right': 'R'}
        
        for hemi_name, hemi_coords in hemisphere_coords.items():
            if len(hemi_coords) < 3:  # Need at least 3 points for meaningful trajectory
                continue
            
            # Create new trajectory ID
            new_id = id_offset + split_info['splits_performed'] * 10 + (1 if hemi_name == 'left' else 2)
            
            # Create new trajectory dictionary based on original
            split_traj = traj.copy()
            split_traj['cluster_id'] = new_id
            split_traj['electrode_count'] = len(hemi_coords)
            
            # Store split information in metadata
            split_traj['is_hemisphere_split'] = True
            split_traj['split_from'] = cluster_id
            split_traj['hemisphere'] = hemi_name
            split_traj['split_label'] = f"H{hemisphere_names[hemi_name]}_{cluster_id}"
            
            # Sort coordinates along trajectory direction
            if len(hemi_coords) > 2:
                # Use PCA to find principal direction
                pca = PCA(n_components=3)
                pca.fit(hemi_coords)
                direction = pca.components_[0]
                center = np.mean(hemi_coords, axis=0)
                projected = np.dot(hemi_coords - center, direction)
                sorted_indices = np.argsort(projected)
                sorted_coords = hemi_coords[sorted_indices]
            else:
                sorted_coords = hemi_coords
                direction = np.array([1, 0, 0])  # Default direction
                center = np.mean(hemi_coords, axis=0)
            
            # Update trajectory properties
            split_traj['endpoints'] = [
                sorted_coords[0].tolist(),
                sorted_coords[-1].tolist()
            ]
            split_traj['direction'] = direction.tolist()
            split_traj['center'] = center.tolist()
            split_traj['length_mm'] = float(np.linalg.norm(
                sorted_coords[-1] - sorted_coords[0]
            ))
            split_traj['sorted_coords'] = sorted_coords.tolist()
            
            # Update linearity using PCA
            if len(hemi_coords) > 2:
                split_traj['linearity'] = float(pca.explained_variance_ratio_[0])
                split_traj['pca_variance'] = pca.explained_variance_ratio_.tolist()
            else:
                split_traj['linearity'] = 1.0
                split_traj['pca_variance'] = [1.0, 0.0, 0.0]
            
            # Copy spacing validation if it exists
            if 'spacing_validation' in traj:
                # Recalculate spacing for the split trajectory 
                spacing_validation = validate_electrode_spacing(
                    sorted_coords, 
                    traj['spacing_validation'].get('expected_range', (3.0, 5.0))
                )
                split_traj['spacing_validation'] = spacing_validation
            
            # Copy and adjust contact angles if they exist
            if 'contact_angles' in traj:
                # Initialize with default values - would need to recalculate for accuracy
                split_traj['contact_angles'] = {
                    'curvature_angles': [],
                    'direction_changes': [],
                    'max_curvature': 0,
                    'mean_curvature': 0,
                    'std_curvature': 0,
                    'max_direction_change': 0,
                    'mean_direction_change': 0,
                    'cumulative_direction_change': 0,
                    'is_linear': True,
                    'linearity_score': 1.0,
                    'flagged_segments_count': 0,
                    'angle_threshold': traj['contact_angles'].get('angle_threshold', 40.0)
                }
            
            split_trajectories.append(split_traj)
        
        # Add split trajectories to results
        if len(split_trajectories) > 0:
            updated_trajectories.extend(split_trajectories)
            split_info['splits_performed'] += 1
            split_info['split_trajectories'].append({
                'original_id': cluster_id,
                'split_ids': [t['cluster_id'] for t in split_trajectories],
                'hemispheres': [t['hemisphere'] for t in split_trajectories],
                'original_contacts': traj['electrode_count'],
                'split_contacts': [t['electrode_count'] for t in split_trajectories]
            })
            
            print(f"  → Created {len(split_trajectories)} hemisphere trajectories:")
            for st in split_trajectories:
                print(f"    • {st['hemisphere']} hemisphere: ID {st['cluster_id']}, {st['electrode_count']} contacts")
        else:
            # Couldn't create valid splits, keep original
            updated_trajectories.append(traj)
            print(f"  → Could not create valid splits, keeping original trajectory")
    
    split_info['final_count'] = len(updated_trajectories)
    
    if split_info['splits_performed'] > 0:
        print(f"\nHemisphere splitting summary:")
        print(f"- Original trajectories: {split_info['original_count']}")
        print(f"- Trajectories split: {split_info['splits_performed']}")
        print(f"- Final trajectories: {split_info['final_count']}")
    
    return updated_trajectories, split_info


def apply_hemisphere_splitting_to_results(results, coords_array, hemisphere='both'):
    """
    Apply hemisphere splitting to trajectory results and update all related data structures.
    
    Args:
        results (dict): Results dictionary from trajectory analysis
        coords_array (numpy.ndarray): Array of electrode coordinates  
        hemisphere (str): Target hemisphere ('left', 'right', or 'both')
        
    Returns:
        dict: Updated results with hemisphere splitting applied
    """
    if hemisphere.lower() != 'both' or 'trajectories' not in results:
        # No splitting needed - either hemisphere filtering already applied or no trajectories
        return results
    
    # Apply hemisphere splitting
    original_trajectories = results['trajectories']
    updated_trajectories, split_info = split_trajectories_at_hemisphere_boundary(
        original_trajectories, coords_array, results, hemisphere
    )
    
    # Update results
    if split_info['splits_performed'] > 0:
        # Store original trajectories
        results['original_trajectories_before_hemisphere_split'] = original_trajectories
        
        # Update main trajectory list
        results['trajectories'] = updated_trajectories
        results['n_trajectories'] = len(updated_trajectories)
        
        # Store split information
        results['hemisphere_splitting'] = split_info
        
        # Update electrode validation if it exists (trajectories have changed)
        if 'electrode_validation' in results:
            expected_counts = results.get('parameters', {}).get('expected_contact_counts', [5, 8, 10, 12, 15, 18])
            validation = validate_electrode_clusters(results, expected_counts)
            results['electrode_validation'] = validation
        
        print(f"✅ Applied hemisphere splitting to {split_info['splits_performed']} trajectories")
    
    return results

#------------------------------------------------------------------------------
# PART 2.11: FINAL SCORING 
#------------------------------------------------------------------------------
def calculate_trajectory_scores(trajectories, coords_array, results):
    """
    Calculate comprehensive scores for each trajectory
    Returns DataFrame ready for annotation and ML training
    """
    trajectory_scores = []
    
    # Get cluster assignments
    clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
    
    for trajectory in trajectories:
        cluster_id = trajectory['cluster_id']
        mask = clusters == cluster_id
        cluster_coords = coords_array[mask]
        
        # EXTRACT FEATURES
        features = extract_trajectory_features(trajectory, cluster_coords)
        
        # CALCULATE ALGORITHMIC SCORE
        algo_score = calculate_algorithmic_score(features)
        
        # CREATE RECORD
        record = {
            'trajectory_id': cluster_id,
            **features,
            'algorithm_score': algo_score,
            'feedback_label': None,  # To be filled manually
            'notes': ''              # For additional comments
        }
        
        trajectory_scores.append(record)
    
    return pd.DataFrame(trajectory_scores)

def extract_trajectory_features(trajectory, cluster_coords):
    """Extract all relevant features for ML training"""
    
    # Basic features
    features = {
        'n_contacts': trajectory['electrode_count'],
        'length_mm': trajectory.get('length_mm', 0),
        'linearity_pca': trajectory.get('linearity', 0),
        'center_x': trajectory['center'][0],
        'center_y': trajectory['center'][1], 
        'center_z': trajectory['center'][2]
    }
    
    # Contact count validation
    expected_counts = [5, 8, 10, 12, 15, 18]
    closest_expected = min(expected_counts, key=lambda x: abs(x - features['n_contacts']))
    features.update({
        'closest_expected_count': closest_expected,
        'count_difference': abs(closest_expected - features['n_contacts']),
        'count_match_exact': features['n_contacts'] in expected_counts
    })
    
    # Spacing features
    if 'spacing_validation' in trajectory:
        spacing = trajectory['spacing_validation']
        features.update({
            'spacing_mean': spacing.get('mean_spacing', np.nan),
            'spacing_std': spacing.get('std_spacing', np.nan),
            'spacing_cv': spacing.get('cv_spacing', np.nan),
            'spacing_valid_pct': spacing.get('valid_percentage', 0),
            'spacing_min': spacing.get('min_spacing', np.nan),
            'spacing_max': spacing.get('max_spacing', np.nan)
        })
    else:
        features.update({
            'spacing_mean': np.nan, 'spacing_std': np.nan, 'spacing_cv': np.nan,
            'spacing_valid_pct': 0, 'spacing_min': np.nan, 'spacing_max': np.nan
        })
    
    # Angle features  
    if 'contact_angles' in trajectory:
        angles = trajectory['contact_angles']
        features.update({
            'angle_max_curvature': angles.get('max_curvature', 0),
            'angle_mean_curvature': angles.get('mean_curvature', 0),
            'angle_std_curvature': angles.get('std_curvature', 0),
            'angle_flagged_segments': angles.get('flagged_segments_count', 0),
            'angle_linearity_score': angles.get('linearity_score', 1.0),
            'angle_cumulative_change': angles.get('cumulative_direction_change', 0)
        })
    else:
        features.update({
            'angle_max_curvature': 0, 'angle_mean_curvature': 0, 'angle_std_curvature': 0,
            'angle_flagged_segments': 0, 'angle_linearity_score': 1.0, 'angle_cumulative_change': 0
        })
    
    # Entry angle validation (if available)
    if 'entry_angle_validation' in trajectory:
        entry_val = trajectory['entry_angle_validation']
        features.update({
            'entry_angle_degrees': entry_val.get('angle_with_surface', np.nan),
            'entry_angle_valid': entry_val.get('is_valid', False)
        })
    else:
        features.update({
            'entry_angle_degrees': np.nan,
            'entry_angle_valid': False
        })
    
    # Geometric features from actual coordinates
    if len(cluster_coords) > 2:
        # Calculate additional geometric properties
        coord_features = calculate_coordinate_features(cluster_coords)
        features.update(coord_features)
    
    return features

def calculate_coordinate_features(coords):
    """Calculate geometric features from raw coordinates"""
    
    # Bounding box
    bbox_min = np.min(coords, axis=0)
    bbox_max = np.max(coords, axis=0)
    bbox_size = bbox_max - bbox_min
    
    # Contact density
    total_length = np.linalg.norm(bbox_size)
    density = len(coords) / max(total_length, 1.0)
    
    # Compactness (how spread out are the points)
    centroid = np.mean(coords, axis=0)
    distances_to_centroid = np.linalg.norm(coords - centroid, axis=1)
    
    return {
        'bbox_size_x': bbox_size[0],
        'bbox_size_y': bbox_size[1], 
        'bbox_size_z': bbox_size[2],
        'bbox_volume': np.prod(bbox_size),
        'contact_density': density,
        'spread_mean': np.mean(distances_to_centroid),
        'spread_std': np.std(distances_to_centroid),
        'spread_max': np.max(distances_to_centroid)
    }

def calculate_algorithmic_score(features):
    """
    Calculate algorithmic quality score (0-100)
    This will be compared against manual feedback
    """
    score = 0.0
    
    # Contact count score (25 points)
    if features.get('count_match_exact', False):
        score += 25
    elif features.get('count_difference') is not None:
        count_diff = features['count_difference']
        if count_diff <= 2:
            score += 20 - (count_diff * 2.5)
    
    # Linearity score (20 points)
    linearity = features.get('linearity_pca', 0)
    if linearity is not None and not (isinstance(linearity, float) and np.isnan(linearity)):
        if linearity >= 0.95:
            score += 20
        elif linearity >= 0.85:
            score += 20 * (linearity - 0.85) / 0.10 + 10
        else:
            score += 10 * max(0, linearity - 0.70) / 0.15
    
    # Spacing score (20 points)
    spacing_valid_pct = features.get('spacing_valid_pct', 0)
    if spacing_valid_pct is not None and not (isinstance(spacing_valid_pct, float) and np.isnan(spacing_valid_pct)):
        if spacing_valid_pct >= 80:
            score += 20
        elif spacing_valid_pct >= 50:
            score += 20 * (spacing_valid_pct - 50) / 30
    
    # Angle score (15 points)
    max_curvature = features.get('angle_max_curvature', 0)
    if max_curvature is not None and not (isinstance(max_curvature, float) and np.isnan(max_curvature)):
        if max_curvature <= 15:
            score += 15
        elif max_curvature <= 40:
            score += 15 * (40 - max_curvature) / 25
    
    # Length reasonableness (10 points)
    length = features.get('length_mm', 0)
    if length is not None and not (isinstance(length, float) and np.isnan(length)):
        if 20 <= length <= 80:
            score += 10
        elif 15 <= length <= 100:
            score += 5
    
    # Entry angle score (10 points)
    if features.get('entry_angle_valid', False):
        score += 10
    else:
        entry_angle = features.get('entry_angle_degrees')
        if entry_angle is not None and not (isinstance(entry_angle, float) and np.isnan(entry_angle)):
            if 20 <= entry_angle <= 70:  # Slightly wider than ideal range
                score += 5
    
    return min(100, max(0, score))

def safe_isnan(value):
    """
    Safely check if a value is NaN, handling None and other types
    """
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return np.isnan(value) if isinstance(value, float) else False
    return False

def extract_trajectory_features(trajectory, cluster_coords):
    """Extract all relevant features for ML training - FIXED VERSION"""
    
    # Basic features
    features = {
        'n_contacts': trajectory.get('electrode_count', 0),
        'length_mm': trajectory.get('length_mm', 0),
        'linearity_pca': trajectory.get('linearity', 0),
        'center_x': trajectory.get('center', [0, 0, 0])[0],
        'center_y': trajectory.get('center', [0, 0, 0])[1], 
        'center_z': trajectory.get('center', [0, 0, 0])[2]
    }
    
    # Contact count validation
    expected_counts = [5, 8, 10, 12, 15, 18]
    n_contacts = features['n_contacts']
    if n_contacts > 0:
        closest_expected = min(expected_counts, key=lambda x: abs(x - n_contacts))
        features.update({
            'closest_expected_count': closest_expected,
            'count_difference': abs(closest_expected - n_contacts),
            'count_match_exact': n_contacts in expected_counts
        })
    else:
        features.update({
            'closest_expected_count': 0,
            'count_difference': 999,
            'count_match_exact': False
        })
    
    # Spacing features
    if 'spacing_validation' in trajectory:
        spacing = trajectory['spacing_validation']
        features.update({
            'spacing_mean': spacing.get('mean_spacing', 0) or 0,
            'spacing_std': spacing.get('std_spacing', 0) or 0,
            'spacing_cv': spacing.get('cv_spacing', 0) or 0,
            'spacing_valid_pct': spacing.get('valid_percentage', 0) or 0,
            'spacing_min': spacing.get('min_spacing', 0) or 0,
            'spacing_max': spacing.get('max_spacing', 0) or 0
        })
    else:
        features.update({
            'spacing_mean': 0, 'spacing_std': 0, 'spacing_cv': 0,
            'spacing_valid_pct': 0, 'spacing_min': 0, 'spacing_max': 0
        })
    
    # Angle features  
    if 'contact_angles' in trajectory:
        angles = trajectory['contact_angles']
        features.update({
            'angle_max_curvature': angles.get('max_curvature', 0) or 0,
            'angle_mean_curvature': angles.get('mean_curvature', 0) or 0,
            'angle_std_curvature': angles.get('std_curvature', 0) or 0,
            'angle_flagged_segments': angles.get('flagged_segments_count', 0) or 0,
            'angle_linearity_score': angles.get('linearity_score', 1.0) or 1.0,
            'angle_cumulative_change': angles.get('cumulative_direction_change', 0) or 0
        })
    else:
        features.update({
            'angle_max_curvature': 0, 'angle_mean_curvature': 0, 'angle_std_curvature': 0,
            'angle_flagged_segments': 0, 'angle_linearity_score': 1.0, 'angle_cumulative_change': 0
        })
    
    # Entry angle validation (if available)
    if 'entry_angle_validation' in trajectory:
        entry_val = trajectory['entry_angle_validation']
        angle_degrees = entry_val.get('angle_with_surface')
        features.update({
            'entry_angle_degrees': angle_degrees if angle_degrees is not None else 0,
            'entry_angle_valid': entry_val.get('is_valid', False)
        })
    else:
        features.update({
            'entry_angle_degrees': 0,
            'entry_angle_valid': False
        })
    
    # Geometric features from actual coordinates
    if len(cluster_coords) > 2:
        try:
            coord_features = calculate_coordinate_features(cluster_coords)
            features.update(coord_features)
        except Exception as e:
            print(f"Warning: Could not calculate coordinate features: {e}")
            # Add default values
            features.update({
                'bbox_size_x': 0, 'bbox_size_y': 0, 'bbox_size_z': 0,
                'bbox_volume': 0, 'contact_density': 0,
                'spread_mean': 0, 'spread_std': 0, 'spread_max': 0
            })
    else:
        # Add default values for insufficient coordinates
        features.update({
            'bbox_size_x': 0, 'bbox_size_y': 0, 'bbox_size_z': 0,
            'bbox_volume': 0, 'contact_density': 0,
            'spread_mean': 0, 'spread_std': 0, 'spread_max': 0
        })
    
    # Convert any None values to 0 or appropriate defaults
    for key, value in features.items():
        if value is None:
            if 'valid' in key:
                features[key] = False
            elif 'match' in key:
                features[key] = False
            else:
                features[key] = 0
    
    return features



def create_final_trajectory_report(coords_array, results, output_dir, create_interactive=True):
    """
    UPDATED: Create final report with scoring table and 3D visualization
    Now uses the fixed visualization functions
    """
    
    # Calculate scores
    trajectories = results.get('trajectories', [])
    scores_df = calculate_trajectory_scores(trajectories, coords_array, results)
    
    # Save scoring table
    csv_path = os.path.join(output_dir, 'trajectory_scores_for_annotation.csv')
    scores_df.to_csv(csv_path, index=False)
    print(f"✅ Trajectory scores saved to: {csv_path}")
    
    # Create 3D visualization with IDs and scores - USING FIXED FUNCTION
    static_fig = create_scored_3d_visualization(coords_array, results, scores_df)
    
    # Save static visualization
    static_viz_path = os.path.join(output_dir, 'trajectory_scores_3d_visualization.png')
    static_fig.savefig(static_viz_path, dpi=300, bbox_inches='tight')
    plt.close(static_fig)
    
    # Create interactive visualization if requested - USING FIXED FUNCTION
    interactive_viz_path = None
    if create_interactive:
        try:
            # Create interactive 3D plot - USING FIXED FUNCTION
            interactive_fig = create_interactive_scored_3d_visualization(coords_array, results, scores_df)
            
            if interactive_fig is not None:
                interactive_viz_path = os.path.join(output_dir, 'trajectory_scores_3d_interactive.html')
                interactive_fig.write_html(interactive_viz_path)
                print(f"✅ Interactive 3D visualization saved to: {interactive_viz_path}")
            
        except ImportError:
            print("⚠️ Plotly not available. Creating static visualization only.")
            create_interactive = False
        except Exception as e:
            print(f"⚠️ Error creating interactive visualization: {e}")
            create_interactive = False
    
    # Create interactive HTML report with user choice
    html_path = os.path.join(output_dir, 'trajectory_annotation_report.html')
    create_interactive_annotation_report(
        scores_df, 
        static_viz_path, 
        html_path,
        interactive_viz_path if create_interactive else None
    )
    
    print(f"✅ Static 3D visualization saved to: {static_viz_path}")
    print(f"✅ Interactive report saved to: {html_path}")
    
    return scores_df, static_fig


##### interactive visualization
def create_interactive_scored_3d_visualization(coords_array, results, scores_df):
    """
    FIXED VERSION: Create interactive 3D Plotly plot with trajectory IDs and algorithm scores
    Now properly handles split trajectories
    """
    try:
        import plotly.graph_objects as go
        import numpy as np
        
        # Get cluster data
        clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
        
        # Create the main 3D plot
        fig = go.Figure()
        
        # Plot all points as background
        fig.add_trace(go.Scatter3d(
            x=coords_array[:, 0],
            y=coords_array[:, 1], 
            z=coords_array[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='lightgray',
                opacity=0.3
            ),
            name='All Electrodes',
            hovertemplate='<b>Electrode</b><br>' +
                          'X: %{x:.1f} mm<br>' +
                          'Y: %{y:.1f} mm<br>' +
                          'Z: %{z:.1f} mm<extra></extra>'
        ))
        
        # Create color map based on algorithm scores
        scores_dict = dict(zip(scores_df['trajectory_id'], scores_df['algorithm_score']))
        
        # Plot each trajectory with color coding
        trajectories_plotted = 0
        for _, row in scores_df.iterrows():
            traj_id = row['trajectory_id']
            score = row['algorithm_score']
            
            # FIXED: Get coordinates using the new helper function
            cluster_coords = get_trajectory_coordinates(traj_id, results, coords_array, clusters)
            
            if cluster_coords is None or len(cluster_coords) == 0:
                print(f"Warning: No coordinates found for trajectory {traj_id}")
                continue
            
            # Color based on score
            if score >= 80:
                color = 'green'
                marker_symbol = 'circle'
                size = 8
                quality = 'Good'
            elif score >= 60:
                color = 'orange' 
                marker_symbol = 'square'
                size = 7
                quality = 'OK'
            else:
                color = 'red'
                marker_symbol = 'diamond'
                size = 9
                quality = 'Bad'
            
            # Create hover text for trajectory
            hover_text = []
            for i, coord in enumerate(cluster_coords):
                hover_text.append(
                    f"<b>Trajectory {traj_id}</b><br>" +
                    f"Contact {i+1}/{len(cluster_coords)}<br>" +
                    f"Position: ({coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f}) mm<br>" +
                    f"Algorithm Score: {score:.0f}<br>" +
                    f"Quality: {quality}<br>" +
                    f"Expected Contacts: {row.get('closest_expected_count', 'N/A')}<br>" +
                    f"Actual Contacts: {row.get('n_contacts', len(cluster_coords))}<br>" +
                    f"Linearity: {row.get('linearity_pca', 0):.3f}<br>" +
                    f"Length: {row.get('length_mm', 0):.1f} mm"
                )
            
            # Plot trajectory points
            fig.add_trace(go.Scatter3d(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                z=cluster_coords[:, 2],
                mode='markers+lines',
                line=dict(
                    color=color,
                    width=6 if quality == 'Bad' else 4
                ),
                marker=dict(
                    size=size,
                    color=color,
                    symbol=marker_symbol,
                    line=dict(width=2, color='black')
                ),
                name=f'T{traj_id} - {quality} ({score:.0f})',
                hovertemplate='%{text}<extra></extra>',
                text=hover_text,
                legendgroup=quality,
                showlegend=True
            ))
            
            trajectories_plotted += 1
        
        # Update layout for better visualization
        fig.update_layout(
            title={
                'text': f'Interactive 3D Trajectory Quality Scores ({trajectories_plotted}/{len(scores_df)} plotted)<br>' +
                        '<sub>Green=Good (≥80), Orange=OK (60-79), Red=Bad (<60)</sub>',
                'x': 0.5,
                'font': {'size': 16}
            },
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='cube'
            ),
            width=1000,
            height=700,
            hovermode='closest'
        )
        
        # Add summary annotation
        good_count = len(scores_df[scores_df['algorithm_score'] >= 80])
        ok_count = len(scores_df[(scores_df['algorithm_score'] >= 60) & (scores_df['algorithm_score'] < 80)])
        bad_count = len(scores_df[scores_df['algorithm_score'] < 60])
        
        fig.add_annotation(
            x=0.02, y=0.98,
            xref='paper', yref='paper',
            text=f"<b>Quality Summary:</b><br>" +
                 f"Total: {len(scores_df)} trajectories<br>" +
                 f"Plotted: {trajectories_plotted}<br>" +
                 f"🟢 Good (≥80): {good_count}<br>" +
                 f"🟠 OK (60-79): {ok_count}<br>" +
                 f"🔴 Bad (<60): {bad_count}<br>" +
                 f"Mean Score: {scores_df['algorithm_score'].mean():.1f}",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=12),
            align="left"
        )
        
        return fig
        
    except ImportError:
        print("Plotly not available for interactive visualization")
        return None
    except Exception as e:
        print(f"Error creating interactive visualization: {e}")
        return None


def get_trajectory_coordinates(traj_id, results, coords_array, clusters):
    """
    FIXED: Get coordinates for any trajectory type (regular or split)
    
    Args:
        traj_id: Trajectory ID (can be int, string, or split format like "S1_1")
        results: Analysis results dictionary
        coords_array: Array of all coordinates
        clusters: Array of cluster assignments
        
    Returns:
        numpy.ndarray: Coordinates for the trajectory, or None if not found
    """
    
    # Method 1: Try to find in results.trajectories (works for all types)
    for traj in results.get('trajectories', []):
        if str(traj['cluster_id']) == str(traj_id):
            if 'sorted_coords' in traj:
                return np.array(traj['sorted_coords'])
            break
    
    # Method 2: For integer-like IDs, try cluster mapping
    if isinstance(traj_id, (int, np.integer)) or (isinstance(traj_id, str) and traj_id.isdigit()):
        try:
            cluster_id_int = int(traj_id)
            mask = clusters == cluster_id_int
            if np.any(mask):
                return coords_array[mask]
        except (ValueError, TypeError):
            pass
    
    # Method 3: For split IDs, try to find the original trajectory data
    if isinstance(traj_id, str) and ('S' in traj_id or 'M' in traj_id):
        # This is a split or merged trajectory, coordinates should be in sorted_coords
        for traj in results.get('trajectories', []):
            if str(traj['cluster_id']) == str(traj_id):
                # Try different coordinate fields
                for coord_field in ['sorted_coords', 'coordinates', 'points']:
                    if coord_field in traj and traj[coord_field]:
                        return np.array(traj[coord_field])
                
                # If no coordinate fields, try to reconstruct from endpoints
                if 'endpoints' in traj:
                    endpoints = np.array(traj['endpoints'])
                    if len(endpoints) == 2:
                        # Create a simple line between endpoints
                        n_points = traj.get('electrode_count', 10)
                        t = np.linspace(0, 1, n_points)
                        coords = []
                        for i in range(n_points):
                            point = endpoints[0] + t[i] * (endpoints[1] - endpoints[0])
                            coords.append(point)
                        return np.array(coords)
                break
    
    return None

def create_scored_3d_visualization(coords_array, results, scores_df):
    """
    FIXED VERSION: Create 3D plot with trajectory IDs and algorithm scores
    Now properly handles split trajectories (S1_1, S2_1, etc.)
    """
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get cluster data
    clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
    
    # Plot all points in light gray
    ax.scatter(coords_array[:, 0], coords_array[:, 1], coords_array[:, 2], 
               c='lightgray', marker='.', s=10, alpha=0.3)
    
    # Create color map based on algorithm scores
    scores_dict = dict(zip(scores_df['trajectory_id'], scores_df['algorithm_score']))
    
    # Plot each trajectory with color coding
    trajectories_plotted = 0
    for _, row in scores_df.iterrows():
        traj_id = row['trajectory_id']
        score = row['algorithm_score']
        
        # FIXED: Get coordinates using the correct method for each trajectory type
        cluster_coords = get_trajectory_coordinates(traj_id, results, coords_array, clusters)
        
        if cluster_coords is None or len(cluster_coords) == 0:
            print(f"Warning: No coordinates found for trajectory {traj_id}")
            continue
        
        # Color based on score: red (bad) -> yellow (ok) -> green (good)
        if score >= 80:
            color = 'green'
            marker = 'o'
            size = 80
        elif score >= 60:
            color = 'orange' 
            marker = 's'
            size = 70
        else:
            color = 'red'
            marker = '^'
            size = 90
        
        # Plot trajectory points
        ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1], cluster_coords[:, 2],
                  c=color, marker=marker, s=size, alpha=0.8, edgecolor='black')
        
        # Add trajectory line
        if len(cluster_coords) > 1:
            ax.plot(cluster_coords[:, 0], cluster_coords[:, 1], cluster_coords[:, 2],
                   '-', color=color, linewidth=2, alpha=0.6)
        
        # Add label with ID and score
        centroid = np.mean(cluster_coords, axis=0)
        ax.text(centroid[0], centroid[1], centroid[2], 
               f'ID:{traj_id}\nScore:{score:.0f}', 
               fontsize=8, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        trajectories_plotted += 1
    
    # Styling
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)') 
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Trajectory Quality Scores ({trajectories_plotted}/{len(scores_df)} plotted)\n(Green=Good, Orange=OK, Red=Bad)')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Good (≥80)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=10, label='OK (60-79)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Bad (<60)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig


def create_interactive_annotation_report(scores_df, static_viz_path, html_path, interactive_viz_path=None):
    """
    Create an interactive HTML report for easy annotation with user choice between static and interactive plots
    """
    # Determine if interactive visualization is available
    has_interactive = interactive_viz_path is not None and os.path.exists(interactive_viz_path)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trajectory Annotation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
            .viz-container {{ text-align: center; margin: 20px 0; }}
            .viz-controls {{ 
                text-align: center; 
                margin: 10px 0; 
                padding: 10px; 
                background-color: #e8f4f8; 
                border-radius: 5px; 
            }}
            .viz-controls button {{ 
                padding: 10px 20px; 
                margin: 0 10px; 
                font-size: 14px; 
                border: none; 
                border-radius: 5px; 
                cursor: pointer; 
                transition: background-color 0.3s;
            }}
            .viz-controls button.active {{ 
                background-color: #007bff; 
                color: white; 
            }}
            .viz-controls button:not(.active) {{ 
                background-color: #f8f9fa; 
                color: #007bff; 
                border: 1px solid #007bff; 
            }}
            .viz-controls button:hover {{ 
                opacity: 0.8; 
            }}
            .viz-frame {{ 
                width: 100%; 
                height: 700px; 
                border: 1px solid #ddd; 
                border-radius: 5px; 
            }}
            .table-container {{ margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .good {{ background-color: #d4edda; }}
            .ok {{ background-color: #fff3cd; }}
            .bad {{ background-color: #f8d7da; }}
            .instructions {{ background-color: #d1ecf1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .feature-note {{ 
                background-color: #fff3cd; 
                padding: 10px; 
                border-radius: 5px; 
                margin: 10px 0; 
                border-left: 4px solid #ffc107; 
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Trajectory Quality Annotation</h1>
            <p>Review each trajectory and add your feedback in the CSV file.</p>
        </div>
        
        <div class="viz-container">
            <h2>3D Trajectory Visualization</h2>
            
            {"" if not has_interactive else '''
            <div class="viz-controls">
                <h3>Choose Visualization Type:</h3>
                <button id="staticBtn" class="active" onclick="showStaticViz()">📊 Static Plot</button>
                <button id="interactiveBtn" onclick="showInteractiveViz()">🎮 Interactive Plot</button>
            </div>
            
            <div class="feature-note">
                <strong>💡 Tip:</strong> Use the Interactive Plot for better exploration - you can rotate, zoom, and hover over points for detailed information!
            </div>
            '''}
            
            <div id="staticViz" class="viz-display">
                <img src="{os.path.basename(static_viz_path)}" alt="3D Trajectory Visualization" style="max-width: 100%; height: auto;">
            </div>
            
            {"" if not has_interactive else f'''
            <div id="interactiveViz" class="viz-display" style="display: none;">
                <iframe src="{os.path.basename(interactive_viz_path)}" class="viz-frame"></iframe>
            </div>
            '''}
        </div>
        
        <div class="instructions">
            <h3>Annotation Instructions:</h3>
            <ol>
                <li>Open the CSV file: <code>trajectory_scores_for_annotation.csv</code></li>
                <li>For each trajectory, fill in the <code>feedback_label</code> column with: <strong>GOOD</strong>, <strong>BAD</strong>, or <strong>UNCERTAIN</strong></li>
                <li>Add any notes in the <code>notes</code> column</li>
                <li>Use the visualization above to help identify trajectories by their ID</li>
                <li>{'Switch between static and interactive views using the buttons above' if has_interactive else 'Use the static visualization to examine trajectory quality'}</li>
                <li>Focus on trajectories where algorithm score disagrees with your visual assessment</li>
            </ol>
            
            {'' if not has_interactive else '''
            <h4>Interactive Plot Features:</h4>
            <ul>
                <li><strong>Rotate:</strong> Click and drag to rotate the 3D view</li>
                <li><strong>Zoom:</strong> Use mouse wheel or zoom controls</li>
                <li><strong>Hover:</strong> Move mouse over trajectories to see detailed information</li>
                <li><strong>Legend:</strong> Click items to hide/show trajectory groups</li>
                <li><strong>Reset:</strong> Double-click to reset the view</li>
            </ul>
            '''}
        </div>
        
        <div class="table-container">
            <h2>Trajectory Summary</h2>
            {scores_df.to_html(classes='table', table_id='scores_table', escape=False)}
        </div>
        
        <script>
            // Visualization switching functions
            function showStaticViz() {{
                document.getElementById('staticViz').style.display = 'block';
                {"document.getElementById('interactiveViz').style.display = 'none';" if has_interactive else ""}
                document.getElementById('staticBtn').classList.add('active');
                {"document.getElementById('interactiveBtn').classList.remove('active');" if has_interactive else ""}
            }}
            
            {"" if not has_interactive else '''
            function showInteractiveViz() {
                document.getElementById('staticViz').style.display = 'none';
                document.getElementById('interactiveViz').style.display = 'block';
                document.getElementById('staticBtn').classList.remove('active');
                document.getElementById('interactiveBtn').classList.add('active');
            }
            '''}
            
            // Add color coding to table rows based on algorithm score
            document.addEventListener('DOMContentLoaded', function() {{
                const rows = document.querySelectorAll('#scores_table tbody tr');
                rows.forEach(row => {{
                    const scoreCell = row.cells[row.cells.length - 3]; // algorithm_score column
                    const score = parseFloat(scoreCell.textContent);
                    if (score >= 80) {{
                        row.classList.add('good');
                    }} else if (score >= 60) {{
                        row.classList.add('ok');
                    }} else {{
                        row.classList.add('bad');
                    }}
                }});
                
                // Set initial view to static
                showStaticViz();
            }});
        </script>
    </body>
    </html>
    """
    
    with open(html_path, 'w') as f:
        f.write(html_content)

#------------------------------------------------------------------------------
# PART 2.12: SIMPLE HEMISPHERE ANALYSIS INTEGRATION
#------------------------------------------------------------------------------

def analyze_both_hemispheres_separately(coords_array, entry_points=None, 
                                       max_neighbor_distance=7.5, min_neighbors=3,
                                       expected_spacing_range=(3.0, 5.0),
                                       use_adaptive_clustering=False, 
                                       expected_contact_counts=[5, 8, 10, 12, 15, 18],
                                       max_iterations=10):
    """
    Simple function to analyze both hemispheres separately and combine the results.
    
    This function:
    1. Splits coordinates by hemisphere
    2. Runs trajectory analysis on each hemisphere independently (with adaptive clustering if requested)
    3. Combines the results into a single unified result
    4. Returns the same format as integrated_trajectory_analysis()
    
    Args:
        coords_array: Array of electrode coordinates
        entry_points: Entry point coordinates (optional)
        max_neighbor_distance: DBSCAN eps parameter (used if adaptive=False)
        min_neighbors: DBSCAN min_samples parameter (used if adaptive=False)
        expected_spacing_range: Expected contact spacing range
        use_adaptive_clustering: Whether to use adaptive parameter selection
        expected_contact_counts: Expected electrode contact counts for adaptive clustering
        max_iterations: Max iterations for adaptive parameter search
        
    Returns:
        dict: Combined results in the same format as integrated_trajectory_analysis()
    """
    print("Analyzing hemispheres separately for better trajectory detection...")
    
    # Step 1: Split coordinates by hemisphere
    left_mask = coords_array[:, 0] < 0  # RAS x < 0 is left
    right_mask = coords_array[:, 0] > 0  # RAS x > 0 is right
    
    left_coords = coords_array[left_mask]
    right_coords = coords_array[right_mask]
    
    # Split entry points if provided
    left_entry = None
    right_entry = None
    if entry_points is not None:
        left_entry_mask = entry_points[:, 0] < 0
        right_entry_mask = entry_points[:, 0] > 0
        left_entry = entry_points[left_entry_mask] if np.any(left_entry_mask) else None
        right_entry = entry_points[right_entry_mask] if np.any(right_entry_mask) else None
    
    print(f"Split coordinates: {len(left_coords)} left, {len(right_coords)} right")
    
    # Step 2: Analyze each hemisphere independently
    left_results = None
    right_results = None
    
    if len(left_coords) >= min_neighbors:
        print("Analyzing left hemisphere...")
        if use_adaptive_clustering:
            # Run adaptive clustering for left hemisphere
            left_results = perform_adaptive_trajectory_analysis(
                coords_array=left_coords,
                entry_points=left_entry,
                initial_eps=max_neighbor_distance,
                initial_min_neighbors=min_neighbors,
                expected_contact_counts=expected_contact_counts,
                output_dir=None  # No individual output for hemisphere
            )
        else:
            # Run regular analysis for left hemisphere
            left_results = integrated_trajectory_analysis(
                coords_array=left_coords,
                entry_points=left_entry,
                max_neighbor_distance=max_neighbor_distance,
                min_neighbors=min_neighbors,
                expected_spacing_range=expected_spacing_range
            )
    
    if len(right_coords) >= min_neighbors:
        print("Analyzing right hemisphere...")
        if use_adaptive_clustering:
            # Run adaptive clustering for right hemisphere
            right_results = perform_adaptive_trajectory_analysis(
                coords_array=right_coords,
                entry_points=right_entry,
                initial_eps=max_neighbor_distance,
                initial_min_neighbors=min_neighbors,
                expected_contact_counts=expected_contact_counts,
                output_dir=None  # No individual output for hemisphere
            )
        else:
            # Run regular analysis for right hemisphere
            right_results = integrated_trajectory_analysis(
                coords_array=right_coords,
                entry_points=right_entry,
                max_neighbor_distance=max_neighbor_distance,
                min_neighbors=min_neighbors,
                expected_spacing_range=expected_spacing_range
            )
    
    # Step 3: Combine results into single unified result
    combined_results = combine_hemisphere_analysis_results(
        left_results, right_results, left_coords, right_coords, 
        left_mask, right_mask, coords_array
    )
    
    print(f"Combined analysis: {combined_results['n_trajectories']} total trajectories")
    return combined_results

def combine_hemisphere_analysis_results(left_results, right_results, 
                                      left_coords, right_coords,
                                      left_mask, right_mask, original_coords):
    """
    Combine hemisphere analysis results into unified format.
    
    Returns results in exactly the same format as integrated_trajectory_analysis()
    so it's a drop-in replacement.
    """
    # Initialize combined results with the same structure
    combined_results = {
        'dbscan': {'n_clusters': 0, 'noise_points': 0, 'cluster_sizes': []},
        'louvain': {'n_communities': 0, 'modularity': 0.0, 'community_sizes': []},
        'combined': {},
        'trajectories': [],
        'n_trajectories': 0,
        'parameters': {
            'n_electrodes': len(original_coords),
            'hemisphere_analysis': True
        },
        'pca_stats': []
    }
    
    # Create a unified graph that maps back to original coordinate indices
    import networkx as nx
    G = nx.Graph()
    
    # Add all original coordinates as nodes
    for i, coord in enumerate(original_coords):
        hemisphere = 'left' if original_coords[i, 0] < 0 else 'right'
        G.add_node(i, pos=coord, hemisphere=hemisphere, dbscan_cluster=-1)
    
    trajectory_id_counter = 0
    all_trajectories = []
    
    # Process left hemisphere results
    if left_results and 'trajectories' in left_results:
        left_trajectories = process_hemisphere_trajectories(
            left_results, left_coords, left_mask, original_coords, 
            'left', trajectory_id_counter, G
        )
        all_trajectories.extend(left_trajectories)
        trajectory_id_counter += len(left_trajectories)
        
        # Update combined DBSCAN stats
        combined_results['dbscan']['n_clusters'] += left_results['dbscan']['n_clusters']
        combined_results['dbscan']['noise_points'] += left_results['dbscan']['noise_points']
        combined_results['dbscan']['cluster_sizes'].extend(left_results['dbscan']['cluster_sizes'])
    
    # Process right hemisphere results  
    if right_results and 'trajectories' in right_results:
        right_trajectories = process_hemisphere_trajectories(
            right_results, right_coords, right_mask, original_coords,
            'right', trajectory_id_counter, G
        )
        all_trajectories.extend(right_trajectories)
        
        # Update combined DBSCAN stats
        combined_results['dbscan']['n_clusters'] += right_results['dbscan']['n_clusters']
        combined_results['dbscan']['noise_points'] += right_results['dbscan']['noise_points']
        combined_results['dbscan']['cluster_sizes'].extend(right_results['dbscan']['cluster_sizes'])
    
    # Finalize combined results
    combined_results['trajectories'] = all_trajectories
    combined_results['n_trajectories'] = len(all_trajectories)
    combined_results['graph'] = G
    
    # Combine PCA stats
    if left_results and 'pca_stats' in left_results:
        combined_results['pca_stats'].extend(left_results['pca_stats'])
    if right_results and 'pca_stats' in right_results:
        combined_results['pca_stats'].extend(right_results['pca_stats'])
    
    # Add hemisphere info for debugging/reporting
    combined_results['hemisphere_info'] = {
        'left_trajectories': len([t for t in all_trajectories if t.get('hemisphere') == 'left']),
        'right_trajectories': len([t for t in all_trajectories if t.get('hemisphere') == 'right']),
        'left_electrodes': len(left_coords),
        'right_electrodes': len(right_coords)
    }
    
    return combined_results

def process_hemisphere_trajectories(hemisphere_results, hemisphere_coords, 
                                  hemisphere_mask, original_coords, hemisphere_name,
                                  id_offset, unified_graph):
    """
    Process trajectories from one hemisphere and map them back to original coordinate space.
    """
    trajectories = []
    
    if 'graph' not in hemisphere_results:
        return trajectories
    
    # Get the mapping from hemisphere coordinates back to original indices
    original_indices = np.where(hemisphere_mask)[0]
    
    for traj in hemisphere_results['trajectories']:
        # Create new trajectory with unified ID
        new_traj = traj.copy()
        new_traj['cluster_id'] = id_offset + traj['cluster_id']
        new_traj['hemisphere'] = hemisphere_name
        new_traj['original_hemisphere_id'] = traj['cluster_id']
        
        # Update graph nodes for this trajectory
        hemisphere_graph = hemisphere_results['graph']
        hemisphere_cluster_id = traj['cluster_id']
        
        # Find nodes in this cluster and update the unified graph
        for node_data in hemisphere_graph.nodes(data=True):
            node_id, node_attrs = node_data
            if node_attrs.get('dbscan_cluster') == hemisphere_cluster_id:
                # Map back to original coordinate index
                original_node_id = original_indices[node_id]
                unified_graph.nodes[original_node_id]['dbscan_cluster'] = new_traj['cluster_id']
        
        trajectories.append(new_traj)
    
    return trajectories
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

def create_colorful_trajectory_paths_page(coords_array, results):
    """
    Create a dedicated page showing trajectory paths in different colors.
    NEW: Added to the PDF report generation.
    FIXED: Color array concatenation issue.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create two subplots: 3D view and 2D projections
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1])
    
    # Main 3D visualization
    ax1 = fig.add_subplot(gs[0, :], projection='3d')
    
    # Get data for plotting
    clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
    unique_clusters = [c for c in set(clusters) if c != -1]
    
    # FIXED: Create a robust colormap for trajectories
    n_trajectories = len(unique_clusters)
    
    if n_trajectories == 0:
        # No trajectories to plot
        ax1.text(0.5, 0.5, 0.5, 'No trajectories found', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14)
        plt.tight_layout()
        return fig
    
    # Create colors using multiple colormaps to ensure we have enough colors
    def generate_colors(n):
        """Generate n distinct colors using multiple colormaps"""
        if n <= 10:
            return plt.cm.tab10(np.linspace(0, 1, 10))[:n]
        elif n <= 20:
            colors1 = plt.cm.tab10(np.linspace(0, 1, 10))
            colors2 = plt.cm.tab20b(np.linspace(0, 1, 20))[:n-10]
            return np.vstack([colors1, colors2])
        else:
            # For many trajectories, use continuous colormaps
            colors1 = plt.cm.tab10(np.linspace(0, 1, 10))
            colors2 = plt.cm.Set3(np.linspace(0, 1, min(12, n-10)))
            remaining = max(0, n - 22)
            if remaining > 0:
                colors3 = plt.cm.hsv(np.linspace(0, 1, remaining))
                return np.vstack([colors1, colors2, colors3])
            else:
                return np.vstack([colors1, colors2])
    
    colors = generate_colors(n_trajectories)
    
    # Plot electrode coordinates as small gray points
    ax1.scatter(coords_array[:, 0], coords_array[:, 1], coords_array[:, 2], 
               c='lightgray', marker='.', s=5, alpha=0.4, label='Electrode contacts')
    
    # Plot each trajectory with unique colors and enhanced styling
    trajectory_info = []
    
    for i, cluster_id in enumerate(unique_clusters):
        mask = clusters == cluster_id
        cluster_coords = coords_array[mask]
        
        if len(cluster_coords) == 0:
            continue
            
        color = colors[i]
        
        # Find trajectory info
        traj_info = None
        for traj in results.get('trajectories', []):
            if traj['cluster_id'] == cluster_id:
                traj_info = traj
                break
        
        # Get trajectory details
        contact_count = len(cluster_coords)
        length_mm = traj_info.get('length_mm', 0) if traj_info else 0
        linearity = traj_info.get('linearity', 0) if traj_info else 0
        
        # Sort coordinates along trajectory direction
        if traj_info and 'direction' in traj_info and len(cluster_coords) > 2:
            direction = np.array(traj_info['direction'])
            center = np.mean(cluster_coords, axis=0)
            projected = np.dot(cluster_coords - center, direction)
            sorted_indices = np.argsort(projected)
            sorted_coords = cluster_coords[sorted_indices]
        else:
            sorted_coords = cluster_coords
        
        # Plot trajectory path with thick line
        ax1.plot(sorted_coords[:, 0], sorted_coords[:, 1], sorted_coords[:, 2], 
                '-', color=color, linewidth=4, alpha=0.8, 
                label=f'ID {cluster_id} ({contact_count} contacts)')
        
        # Plot individual contacts with larger markers
        ax1.scatter(sorted_coords[:, 0], sorted_coords[:, 1], sorted_coords[:, 2], 
                   c=[color], s=80, alpha=0.9, edgecolor='black', linewidth=0.5)
        
        # Add trajectory endpoints with special markers
        if len(sorted_coords) > 1:
            ax1.scatter(sorted_coords[0, 0], sorted_coords[0, 1], sorted_coords[0, 2], 
                       c='green', marker='>', s=150, alpha=1.0, edgecolor='black', linewidth=1)
            ax1.scatter(sorted_coords[-1, 0], sorted_coords[-1, 1], sorted_coords[-1, 2], 
                       c='red', marker='s', s=150, alpha=1.0, edgecolor='black', linewidth=1)
        
        # Add trajectory ID label at center
        center_point = np.mean(sorted_coords, axis=0)
        ax1.text(center_point[0], center_point[1], center_point[2], 
                f'T{cluster_id}', fontsize=10, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Store info for summary
        trajectory_info.append({
            'id': cluster_id,
            'contacts': contact_count,
            'length': length_mm,
            'linearity': linearity,
            'color': color
        })
    
    # Plot noise points if any
    noise_mask = clusters == -1
    if np.any(noise_mask):
        noise_coords = coords_array[noise_mask]
        ax1.scatter(noise_coords[:, 0], noise_coords[:, 1], noise_coords[:, 2],
                   c='black', marker='x', s=100, alpha=0.7, label='Noise points')
    
    # Enhance 3D plot
    ax1.set_xlabel('X (mm)', fontsize=12)
    ax1.set_ylabel('Y (mm)', fontsize=12)
    ax1.set_zlabel('Z (mm)', fontsize=12)
    ax1.set_title('Electrode Trajectory Paths - 3D View\n(Green arrows: start, Red squares: end)', 
                 fontsize=14, fontweight='bold')
    
    # Set view angle for better visualization
    ax1.view_init(elev=20, azim=45)
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # 2D projections
    projections = [
        ('X-Y Projection (Axial View)', 0, 1, 'Z'),
        ('X-Z Projection (Coronal View)', 0, 2, 'Y')
    ]
    
    for idx, (title, dim1, dim2, depth_dim) in enumerate(projections):
        ax = fig.add_subplot(gs[1, idx])
        
        # Plot all points lightly
        ax.scatter(coords_array[:, dim1], coords_array[:, dim2], 
                  c='lightgray', marker='.', s=3, alpha=0.3)
        
        # Plot trajectories
        for i, cluster_id in enumerate(unique_clusters):
            mask = clusters == cluster_id
            cluster_coords = coords_array[mask]
            
            if len(cluster_coords) == 0:
                continue
                
            color = colors[i]
            
            # Sort coordinates for better line plotting
            traj_info = next((t for t in results.get('trajectories', []) 
                            if t['cluster_id'] == cluster_id), None)
            
            if traj_info and 'direction' in traj_info and len(cluster_coords) > 2:
                direction = np.array(traj_info['direction'])
                center = np.mean(cluster_coords, axis=0)
                projected = np.dot(cluster_coords - center, direction)
                sorted_indices = np.argsort(projected)
                sorted_coords = cluster_coords[sorted_indices]
            else:
                sorted_coords = cluster_coords
            
            # Plot trajectory line
            ax.plot(sorted_coords[:, dim1], sorted_coords[:, dim2], 
                   '-', color=color, linewidth=3, alpha=0.8)
            
            # Plot contacts
            ax.scatter(sorted_coords[:, dim1], sorted_coords[:, dim2], 
                      c=[color], s=50, alpha=0.9, edgecolor='black', linewidth=0.5)
            
            # Add trajectory ID
            center_2d = np.mean(sorted_coords, axis=0)
            ax.text(center_2d[dim1], center_2d[dim2], f'T{cluster_id}', 
                   fontsize=8, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        ax.set_xlabel(f'{["X", "Y", "Z"][dim1]} (mm)', fontsize=10)
        ax.set_ylabel(f'{["X", "Y", "Z"][dim2]} (mm)', fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Set aspect ratio
        try:
            ax.set_aspect('equal', adjustable='box')
        except:
            # If equal aspect fails, use auto
            ax.set_aspect('auto')
    
    plt.tight_layout()
    return fig


def visualize_combined_results(coords_array, results, output_dir=None, bolt_directions=None):
    """
    Create and save/display visualizations of trajectory analysis results.
    ENHANCED: Added colorful trajectory paths page to PDF report.
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
            
            # NEW: Create colorful trajectory paths page
            fig = create_colorful_trajectory_paths_page(coords_array, results)
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
        # Interactive mode - show all plots including the new colorful paths
        fig = create_summary_page(results)
        plt.show()
        
        fig = create_3d_visualization(coords_array, results, bolt_directions)
        plt.show()
        
        # NEW: Show colorful trajectory paths
        fig = create_colorful_trajectory_paths_page(coords_array, results)
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

def main(use_combined_volume=True, use_original_reports=True, 
         detect_duplicates=True, duplicate_threshold=0.5, 
         use_adaptive_clustering=False, max_iterations=10,
         validate_spacing=True, expected_spacing_range=(3.0, 5.0),
         refine_trajectories=True, max_contacts_per_trajectory=20,
         validate_entry_angles=True, hemisphere='both',
         analyze_contact_angles=True, angle_threshold=60.0,
         create_plotly_visualization=False,
         create_interactive_annotation=True): 
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
        from matplotlib.backends.backend_pdf import PdfPages
        print(f"Starting electrode trajectory analysis...")
        print(f"Options: combined_volume={use_combined_volume}, adaptive_clustering={use_adaptive_clustering}, "
              f"detect_duplicates={detect_duplicates}, duplicate_threshold={duplicate_threshold}, "
              f"validate_spacing={validate_spacing}, spacing_range={expected_spacing_range}, "
              f"refine_trajectories={refine_trajectories}, validate_entry_angles={validate_entry_angles}, "
              f"hemisphere={hemisphere}")  
        
        # Step 1: Load required volumes from Slicer
        print("Loading volumes from Slicer...")
        electrodes_volume = slicer.util.getNode('P7_electrode_mask_success_1')
        brain_volume = slicer.util.getNode("patient7_mask_5")
        
        # Create output directories
        base_dir = r"C:\Users\rocia\Downloads\TFG\Cohort\Centroids\P7_BoltHeadandpaths_trial_success_interactive_split_hemishhere_enhanced_orga"
        
        # Include hemisphere in output directory name
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
            combined_volume = slicer.util.getNode('P7_CombinedBoltHeadEntryPointsTrajectoryMask')
            
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
        entry_points_volume = slicer.util.getNode('P7_brain_entry_points')
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
            
            # Check if we should analyze hemispheres separately
            if hemisphere == 'both':
                # Quick check: do we have significant electrodes in both hemispheres?
                left_coords = coords_array[coords_array[:, 0] < 0]
                right_coords = coords_array[coords_array[:, 0] > 0]
                
                # If we have enough electrodes in both hemispheres, analyze separately
                if len(left_coords) >= 10 and len(right_coords) >= 10:
                    print(f"Detected bilateral electrodes (L:{len(left_coords)}, R:{len(right_coords)})")
                    print("Using hemisphere-separated adaptive analysis...")
                    
                    # Use hemisphere separation WITH adaptive clustering
                    integrated_results = analyze_both_hemispheres_separately(
                        coords_array=coords_array,
                        entry_points=entry_points,
                        max_neighbor_distance=8,  # Initial value for adaptive search
                        min_neighbors=3,
                        expected_spacing_range=expected_spacing_range if validate_spacing else None,
                        use_adaptive_clustering=True,  # Enable adaptive clustering
                        expected_contact_counts=expected_contact_counts,
                        max_iterations=max_iterations
                    )
                else:
                    # Use regular adaptive analysis (your existing code)
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
                    
                    integrated_results = integrated_trajectory_analysis(
                        coords_array=coords_array,
                        entry_points=entry_points,
                        max_neighbor_distance=optimal_eps,
                        min_neighbors=optimal_min_neighbors,
                        expected_spacing_range=expected_spacing_range if validate_spacing else None
                    )
                    
                    integrated_results['parameter_search'] = parameter_search
            else:
                # Single hemisphere adaptive analysis (your existing code)
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
                
                integrated_results = integrated_trajectory_analysis(
                    coords_array=coords_array,
                    entry_points=entry_points,
                    max_neighbor_distance=optimal_eps,
                    min_neighbors=optimal_min_neighbors,
                    expected_spacing_range=expected_spacing_range if validate_spacing else None
                )
                
                integrated_results['parameter_search'] = parameter_search
        else:
            # Regular (non-adaptive) analysis
            print("Running integrated trajectory analysis with fixed parameters...")
            
            # Check if we should analyze hemispheres separately
            if hemisphere == 'both':
                # Quick check: do we have significant electrodes in both hemispheres?
                left_coords = coords_array[coords_array[:, 0] < 0]
                right_coords = coords_array[coords_array[:, 0] > 0]
                
                # If we have enough electrodes in both hemispheres, analyze separately
                if len(left_coords) >= 10 and len(right_coords) >= 10:
                    print(f"Detected bilateral electrodes (L:{len(left_coords)}, R:{len(right_coords)})")
                    print("Using hemisphere-separated analysis...")
                    
                    # Use hemisphere separation with fixed parameters
                    integrated_results = analyze_both_hemispheres_separately(
                        coords_array=coords_array,
                        entry_points=entry_points,
                        max_neighbor_distance=7.5,
                        min_neighbors=3,
                        expected_spacing_range=expected_spacing_range if validate_spacing else None,
                        use_adaptive_clustering=False  # Use fixed parameters
                    )
                else:
                    # Use regular analysis (your existing code)
                    integrated_results = integrated_trajectory_analysis(
                        coords_array=coords_array,
                        entry_points=entry_points,
                        max_neighbor_distance=7.5,
                        min_neighbors=3,
                        expected_spacing_range=expected_spacing_range if validate_spacing else None
                    )
            else:
                # Single hemisphere analysis (your existing code)
                integrated_results = integrated_trajectory_analysis(
                    coords_array=coords_array,
                    entry_points=entry_points,
                    max_neighbor_distance=7.5,
                    min_neighbors=3,
                    expected_spacing_range=expected_spacing_range if validate_spacing else None
                )

        if hemisphere.lower() == 'both':
            print("Applying hemisphere splitting for trajectories that cross the boundary (x=0)...")
            integrated_results = apply_hemisphere_splitting_to_results(
                integrated_results, coords_array, hemisphere
            )

        if analyze_contact_angles and 'trajectories' in integrated_results:
                print("Analyzing contact angles within trajectories...")
                
                # Perform contact angle analysis
                trajectory_angle_analyses = analyze_trajectory_angles(
                    integrated_results['trajectories'], 
                    coords_array, 
                    integrated_results, 
                    angle_threshold=angle_threshold
                )
                
                # Add angle analysis results to the main results
                all_results['contact_angle_analysis'] = trajectory_angle_analyses
                
                # Add angle analysis directly to trajectory dictionaries
                add_angle_analysis_to_trajectories(
                    integrated_results['trajectories'], 
                    coords_array, 
                    integrated_results, 
                    angle_threshold
                )
                
                # Print summary
                print_angle_analysis_summary(trajectory_angle_analyses)
                
                # Count flagged trajectories
                flagged_count = sum(1 for analysis in trajectory_angle_analyses.values() 
                                if not analysis['is_linear'])
                total_count = len(trajectory_angle_analyses)
                
                print(f"Contact angle analysis: {flagged_count} of {total_count} trajectories flagged for non-linearity")
                
                # Generate reports if requested
                if use_original_reports:
                    print("Generating contact angle analysis reports...")
                    
                    # Create angle analysis subdirectory
                    angle_analysis_dir = os.path.join(output_dir, "contact_angle_analysis")
                    os.makedirs(angle_analysis_dir, exist_ok=True)
                    
                    # Create visualization
                    angle_fig = create_angle_analysis_visualization(
                        trajectory_angle_analyses, 
                        angle_analysis_dir
                    )
                    
                    # Save to PDF
                    from matplotlib.backends.backend_pdf import PdfPages
                    with PdfPages(os.path.join(angle_analysis_dir, 'contact_angle_analysis.pdf')) as pdf:
                        pdf.savefig(angle_fig)
                        plt.close(angle_fig)
                    
                    # Create flagged segments report
                    flagged_df = create_flagged_segments_report(
                        trajectory_angle_analyses, 
                        angle_analysis_dir
                    )
                    
                    print(f"✅ Contact angle analysis reports saved to {angle_analysis_dir}")
                    
                    # Add to figures for main report
                    if 'figures' not in integrated_results:
                        integrated_results['figures'] = {}
                    integrated_results['figures']['contact_angle_analysis'] = angle_fig
        
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


        if analyze_contact_angles:
            print("Analyzing contact angles (FINAL - after all refinements)...")
            
            # Use the improved angle analysis function with CORRECT variable name
            integrated_results = ensure_angle_analysis_after_refinement(
                integrated_results,  # ← FIXED: Use correct variable name
                coords_array, 
                analyze_contact_angles=True, 
                angle_threshold=angle_threshold
            )
            
            # Get the analysis results with CORRECT variable name
            trajectory_angle_analyses = integrated_results.get('contact_angle_analysis', {})
            
            # Store in all_results for later use
            all_results['contact_angle_analysis'] = trajectory_angle_analyses
            
            # Print detailed summary
            print_angle_analysis_summary(trajectory_angle_analyses)
            
            flagged_count = sum(1 for analysis in trajectory_angle_analyses.values() 
                              if not analysis['is_linear'])
            total_count = len(trajectory_angle_analyses)
            
            # Store angle summary for later reporting
            angle_summary = {
                'total_trajectories': total_count,
                'flagged_trajectories': flagged_count,
                'flagged_percentage': flagged_count/total_count*100 if total_count > 0 else 0
            }
            
            print(f"Contact angles: {flagged_count}/{total_count} trajectories flagged for non-linearity")
            
            # Create minimal angle visualization if reports enabled
            if use_original_reports:
                angle_fig = create_angle_analysis_visualization_minimal(trajectory_angle_analyses, output_dir)
                if angle_fig:
                    plt.close(angle_fig)
        else:
            angle_summary = None
        
        # Store integrated results (AFTER angle analysis is complete)
        all_results['integrated_analysis'] = integrated_results
        print(f"Identified {integrated_results.get('n_trajectories', 0)} trajectories through clustering.")
        
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
        bolt_head_volume = slicer.util.getNode('P7_bolt_heads')
        
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
        
        if analyze_contact_angles and 'contact_angle_analysis' in all_results:
            angle_analyses = all_results['contact_angle_analysis']
            flagged_count = sum(1 for a in angle_analyses.values() if not a['is_linear'])
            total_count = len(angle_analyses)
            
            if total_count > 0:
                print(f"- Contact angle analysis: {flagged_count} of {total_count} trajectories "
                      f"({flagged_count/total_count*100:.1f}%) flagged for non-linearity (>{angle_threshold}°)")
                
                # Calculate overall statistics
                all_max_angles = [a['max_curvature'] for a in angle_analyses.values()]
                if all_max_angles:
                    print(f"- Maximum angle deviation: {max(all_max_angles):.1f}°")
                    print(f"- Mean maximum angle: {np.mean(all_max_angles):.1f}°")

        if analyze_contact_angles and create_plotly_visualization and 'contact_angle_analysis' in all_results:
            print("Creating interactive Plotly visualization for contact angles...")
            
            try:
                plotly_fig = create_plotly_interactive_angle_visualization(
                    all_results['contact_angle_analysis'],
                    coords_array,
                    integrated_results,
                    output_dir if use_original_reports else None
                )
                
                if plotly_fig is not None:
                    # Store the figure in results
                    if 'figures' not in integrated_results:
                        integrated_results['figures'] = {}
                    integrated_results['figures']['plotly_angles'] = plotly_fig
                    
                    # Display the figure if running interactively (optional)
                    # Uncomment the next line if you want to show the plot immediately
                    plotly_fig.show()
                    
                    print("✅ Interactive Plotly visualization created successfully")
                else:
                    print("⚠️ Plotly visualization could not be created (missing dependencies or data)")
                    
            except Exception as e:
                print(f"Error creating Plotly visualization: {e}")
                import traceback
                traceback.print_exc()


        if use_original_reports:
            print("Creating final trajectory scoring report...")
            try:
                # Add create_interactive parameter based on user preference
                scores_df, viz_fig = create_final_trajectory_report(
                    coords_array, 
                    integrated_results, 
                    output_dir,
                    create_interactive=True  # Set to True to enable interactive option
                )
                
                # Print summary
                print(f"\n=== FINAL SCORING SUMMARY ===")
                print(f"Total trajectories: {len(scores_df)}")
                print(f"High quality (≥80): {len(scores_df[scores_df['algorithm_score'] >= 80])}")
                print(f"Medium quality (60-79): {len(scores_df[(scores_df['algorithm_score'] >= 60) & (scores_df['algorithm_score'] < 80)])}")
                print(f"Low quality (<60): {len(scores_df[scores_df['algorithm_score'] < 60])}")
                print(f"Mean algorithm score: {scores_df['algorithm_score'].mean():.2f}")
                
                plt.close(viz_fig)
                
                all_results['final_scoring'] = {
                    'scores_dataframe': scores_df,
                    'summary': {
                        'total_trajectories': len(scores_df),
                        'high_quality': len(scores_df[scores_df['algorithm_score'] >= 80]),
                        'medium_quality': len(scores_df[(scores_df['algorithm_score'] >= 60) & (scores_df['algorithm_score'] < 80)]),
                        'low_quality': len(scores_df[scores_df['algorithm_score'] < 60]),
                        'mean_score': scores_df['algorithm_score'].mean()
                    }
                }
            except Exception as e:
                print(f"Error creating final scoring report: {e}")
                import traceback
                traceback.print_exc()
        if analyze_contact_angles and angle_summary:  # ← Check angle_summary exists
            print(f"- Contact angle analysis: {angle_summary['flagged_trajectories']} of {angle_summary['total_trajectories']} trajectories "
                  f"({angle_summary['flagged_percentage']:.1f}%) flagged for non-linearity (>{angle_threshold}°)")
            
            # Calculate overall statistics from the stored analysis
            if 'contact_angle_analysis' in all_results:
                angle_analyses = all_results['contact_angle_analysis']
                all_max_angles = [a['max_curvature'] for a in angle_analyses.values()]
                if all_max_angles:
                    print(f"- Maximum angle deviation: {max(all_max_angles):.1f}°")
                    print(f"- Mean maximum angle: {np.mean(all_max_angles):.1f}°")
        
        return all_results
        
    except Exception as e:
        logging.error(f"Electrode trajectory analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'traceback': traceback.format_exc()}

# Example usage in __main__ block
if __name__ == "__main__":
    results = main(
        use_combined_volume=True,
        use_original_reports=True,
        detect_duplicates=True,
        duplicate_threshold=2.5,
        use_adaptive_clustering=True,
        max_iterations=8,
        validate_spacing=True,
        expected_spacing_range=(3.0, 5.0),
        refine_trajectories=True,
        max_contacts_per_trajectory=18,
        validate_entry_angles=True,
        hemisphere = 'both',               # Specify hemisphere for analysis
        analyze_contact_angles=True,     # Enable contact angle analysis
        angle_threshold=40.0,             # Set threshold for flagging non-linear segments
        create_plotly_visualization=True, #  Enable Plotly visualization
        create_interactive_annotation=True, # Enable interactive annotation

    )
    print("Enhanced analysis completed with contact angle analysis.")

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