"""
Electrode Trajectory Analysis Module - Reorganized

This module provides comprehensive functionality for analyzing SEEG electrode trajectories in 3D space.
It performs clustering, community detection, trajectory analysis, and various validation methods.

Author: Rocío Ávalos
"""

#==============================================================================
# IMPORTS AND DEPENDENCIES
#==============================================================================

import slicer
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import networkx as nx
from scipy.spatial.distance import cdist
from collections import defaultdict, Counter
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
from Electrode_path.construction_4 import (
    create_summary_page, create_3d_visualization, create_trajectory_details_page, 
    create_noise_points_page
)

#==============================================================================
# SECTION 1: UTILITY CLASSES AND HELPER FUNCTIONS
#==============================================================================

class Arrow3D(FancyArrowPatch):
    """Custom 3D arrow patch for visualization in matplotlib."""
    
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


def _bresenham_line_3d(x0, y0, z0, x1, y1, z1):
    """
    Implementation of 3D Bresenham's line algorithm to create a line between two points.
    
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

#==============================================================================
# SECTION 2: TRAJECTORY EXTRACTION AND ANALYSIS
#==============================================================================

def extract_trajectories_from_combined_mask(combined_volume, brain_volume=None):
    """
    Extract trajectories directly from the combined mask volume that contains
    bolt heads (value=1), entry points (value=2), and trajectory lines (value=3).
    
    Args:
        combined_volume: Slicer volume node containing the combined mask
        brain_volume: Optional brain mask volume for validation
        
    Returns:
        dict: Dictionary with bolt IDs as keys and trajectory information as values
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
            bolt_point_np = np.array(bolt_point_ras)
            entry_point_np = np.array(entry_point_ras)
            distance = np.linalg.norm(bolt_point_np - entry_point_np)
            
            # Check if there's a trajectory path between them by finding connected components
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
                'start_point': bolt_point_ras,
                'end_point': closest_entry['entry_point'],
                'direction': direction.tolist(),
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


def verify_directions_with_brain(directions, brain_volume):
    """
    Verify that bolt entry directions point toward the brain and validate entry angles.
    
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
        brain_alignment = np.dot(current_direction, to_brain_center)
        
        # Check 2: Compare distances to brain surface
        bolt_to_surface = cdist([bolt_point], brain_surface_points).min()
        entry_to_surface = cdist([entry_point], brain_surface_points).min()
        
        # Check 3: Calculate angle relative to surface normal at entry point
        closest_idx = np.argmin(cdist([entry_point], brain_surface_points))
        closest_surface_point = brain_surface_points[closest_idx]
        
        # Estimate the surface normal at this point
        k = min(20, len(brain_surface_points))
        dists = cdist([closest_surface_point], brain_surface_points)[0]
        nearest_idxs = np.argsort(dists)[:k]
        nearest_points = brain_surface_points[nearest_idxs]
        
        # Use PCA to estimate the local surface plane
        pca = PCA(n_components=3)
        pca.fit(nearest_points)
        
        # The third component (least variance) approximates the surface normal
        surface_normal = pca.components_[2]
        
        # Make sure the normal points outward from the brain
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
        
        # Add validation metrics to direction info
        bolt_info['brain_alignment'] = float(brain_alignment)
        bolt_info['bolt_to_surface_dist'] = float(bolt_to_surface)
        bolt_info['entry_to_surface_dist'] = float(entry_to_surface)
        bolt_info['angle_with_surface'] = float(angle_with_surface)
        bolt_info['is_angle_valid'] = bool(is_angle_valid)

#==============================================================================
# SECTION 3: INTEGRATED TRAJECTORY ANALYSIS
#==============================================================================

def integrated_trajectory_analysis(coords_array, entry_points=None, max_neighbor_distance=8, 
                                 min_neighbors=3, expected_spacing_range=(3.0, 5.0)):
    """
    Perform integrated trajectory analysis on electrode coordinates.
    
    This function combines DBSCAN clustering, Louvain community detection,
    and PCA-based trajectory analysis to identify and characterize electrode trajectories.
    
    Args:
        coords_array (numpy.ndarray): Array of electrode coordinates (shape: [n_electrodes, 3])
        entry_points (numpy.ndarray, optional): Array of entry point coordinates
        max_neighbor_distance (float): Maximum distance between neighbors for DBSCAN
        min_neighbors (int): Minimum number of neighbors for DBSCAN
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
        
        # Calculate purity scores
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
    
    # Trajectory analysis with PCA and spacing validation
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
            
            # Store PCA statistics
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

    return results


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
        traj_first_contact = traj_endpoints[0]
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
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
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


def compare_trajectories_with_combined_data(integrated_results, combined_trajectories):
    """
    Compare trajectories detected through clustering with those from the combined volume.
    
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
        if best_match and best_match['score'] < 30:
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

#==============================================================================
# SECTION 4: SPACING VALIDATION
#==============================================================================

def validate_electrode_spacing(trajectory_points, expected_spacing_range=(3.0, 5.0)):
    """
    Validate the spacing between electrode contacts in a trajectory.
    
    Args:
        trajectory_points (numpy.ndarray): Array of contact coordinates along a trajectory
        expected_spacing_range (tuple): Expected range of spacing (min, max) in mm
        
    Returns:
        dict: Dictionary with spacing validation results
    """
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
    
    # Identify problematic spacings
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
        'cv_spacing': std_spacing / mean_spacing if mean_spacing > 0 else np.nan,
        'valid_percentage': valid_percentage,
        'valid_spacings': valid_spacings,
        'too_close_indices': too_close,
        'too_far_indices': too_far,
        'expected_range': expected_spacing_range,
        'is_valid': valid_percentage >= 75,
        'status': 'valid' if valid_percentage >= 75 else 'invalid'
    }


def create_spacing_validation_page(results):
    """
    Create a visualization page for electrode spacing validation results.
    
    Args:
        results (dict): Results from integrated_trajectory_analysis with spacing validation
        
    Returns:
        matplotlib.figure.Figure: Figure containing spacing validation results
    """
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle('Electrode Contact Spacing Validation (Expected: 3-5mm)', fontsize=16)
    
    # Create grid layout
    gs = GridSpec(2, 2, figure=fig)
    
    # Summary statistics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
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
        cv_spacing = spacing_validation.get('cv_spacing', np.nan) * 100
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
    
    # Sort by trajectory ID
    def safe_sort_key(x):
        if isinstance(x[0], int):
            return (0, x[0], "")
        elif isinstance(x[0], str) and x[0].isdigit():
            return (0, int(x[0]), "")
        else:
            try:
                if isinstance(x[0], str) and "_" in x[0]:
                    parts = x[0].split("_")
                    if len(parts) > 1 and parts[1].isdigit():
                        return (1, int(parts[1]), x[0])
                return (2, 0, x[0])
            except:
                return (3, 0, str(x[0]))
    
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
            cell = detail_table[(i+1, len(detail_columns)-1)]
            if status == 'VALID':
                cell.set_facecolor('lightgreen')
            elif status == 'INVALID':
                cell.set_facecolor('lightcoral')
    else:
        ax3.text(0.5, 0.5, "No detailed spacing data available", ha='center', va='center', fontsize=14)
    
    ax3.set_title('Detailed Trajectory Spacing Analysis')
    
    plt.tight_layout()
    return fig

#==============================================================================
# SECTION 5: ELECTRODE VALIDATION
#==============================================================================

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
        
        # Determine if this is a valid cluster
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

#==============================================================================
# SECTION 6: HEMISPHERE FILTERING
#==============================================================================

def filter_coordinates_by_hemisphere(coords_array, hemisphere='left', verbose=True):
    """
    Filter electrode coordinates by hemisphere.
    
    Args:
        coords_array (numpy.ndarray): Array of coordinates in RAS format [N, 3]
        hemisphere (str): 'left' (x < 0), 'right' (x > 0), or 'both' (no filtering)
        verbose (bool): Whether to print filtering results
        
    Returns:
        tuple: (filtered_coords, hemisphere_mask, filtered_indices)
    """
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

#==============================================================================
# SECTION 7: CONTACT ANGLE ANALYSIS
#==============================================================================

def calculate_contact_angles(trajectory_points, angle_threshold=40.0):
    """
    Calculate curvature angles and direction changes between consecutive contact segments.
    
    Args:
        trajectory_points (numpy.ndarray): Array of contact coordinates sorted along trajectory
        angle_threshold (float): Threshold angle in degrees to flag problematic segments
        
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
        
        # Skip if either vector is too short
        v1_length = np.linalg.norm(v1)
        v2_length = np.linalg.norm(v2)
        
        if v1_length < 1e-6 or v2_length < 1e-6:
            curvature_angles.append(0.0)
            direction_changes.append(0.0)
            continue
        
        # Normalize vectors
        v1_norm = v1 / v1_length
        v2_norm = v2 / v2_length
        
        # Calculate curvature angle
        dot_product = np.dot(v1_norm, v2_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        angle_between_vectors = np.degrees(np.arccos(np.abs(dot_product)))
        curvature_angle = 180.0 - angle_between_vectors
        curvature_angles.append(curvature_angle)
        
        # Calculate direction change
        direction_diff = v2_norm - v1_norm
        direction_change_magnitude = np.linalg.norm(direction_diff)
        direction_change = direction_change_magnitude * 90.0
        direction_changes.append(min(direction_change, 180.0))
        
        # Flag segments that exceed threshold
        if curvature_angle > angle_threshold:
            segment_length_1 = v1_length
            segment_length_2 = v2_length
            total_span = np.linalg.norm(p3 - p1)
            
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
    
    cumulative_direction_change = np.sum(direction_changes) if len(direction_changes) > 0 else 0
    
    # Calculate linearity score
    linearity_score = max(0, 1 - (max_curvature / 180.0) * 1.5 - (mean_curvature / 60.0) * 0.5)
    linearity_score = min(1.0, linearity_score)
    
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
    Analyze contact angles for all trajectories in the results.
    
    Args:
        trajectories (list): List of trajectory dictionaries
        coords_array (numpy.ndarray): Array of all electrode coordinates
        results (dict): Results from trajectory analysis
        angle_threshold (float): Threshold angle to flag problematic segments
        
    Returns:
        dict: Dictionary mapping trajectory IDs to angle analysis results
    """
    # Get cluster assignments from results
    if 'graph' not in results:
        print("Warning: No graph information available for cluster mapping")
        return {}
    
    clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
    
    trajectory_angle_analyses = {}
    
    for traj in trajectories:
        cluster_id = traj['cluster_id']
        
        # Get coordinates for this trajectory
        mask = clusters == cluster_id
        
        if not np.any(mask):
            print(f"Warning: No coordinates found for trajectory {cluster_id}")
            continue
            
        cluster_coords = coords_array[mask]
        
        # Sort coordinates along trajectory direction
        if 'direction' in traj and len(cluster_coords) > 2:
            direction = np.array(traj['direction'])
            center = np.mean(cluster_coords, axis=0)
            projected = np.dot(cluster_coords - center, direction)
            sorted_indices = np.argsort(projected)
            sorted_coords = cluster_coords[sorted_indices]
        else:
            sorted_coords = cluster_coords
        
        # Analyze angles for this trajectory
        angle_analysis = calculate_contact_angles(sorted_coords, angle_threshold)
        
        # Add trajectory metadata
        angle_analysis['trajectory_id'] = cluster_id
        angle_analysis['contact_count'] = len(sorted_coords)
        angle_analysis['trajectory_length'] = traj.get('length_mm', 0)
        angle_analysis['pca_linearity'] = traj.get('linearity', 0)
        
        trajectory_angle_analyses[cluster_id] = angle_analysis
    
    return trajectory_angle_analyses


def create_angle_analysis_visualization(trajectory_angle_analyses, output_dir=None):
    """
    Create enhanced visualizations for contact angle analysis results.
    
    Args:
        trajectory_angle_analyses (dict): Results from analyze_trajectory_angles
        output_dir (str, optional): Directory to save visualizations
        
    Returns:
        matplotlib.figure.Figure: Figure containing angle analysis visualization
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
    
    # Distribution of curvature angles
    ax1 = fig.add_subplot(gs[0, 0])
    if all_curvature_angles:
        ax1.hist(all_curvature_angles, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=40, color='red', linestyle='--', linewidth=2, label='Threshold (40°)')
        ax1.axvline(x=np.mean(all_curvature_angles), color='orange', linestyle='-', linewidth=2, 
                   label=f'Mean ({np.mean(all_curvature_angles):.1f}°)')
        ax1.set_xlabel('Curvature Angle (degrees)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Curvature Angles')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No curvature data available', ha='center', va='center')
    
    # Distribution of direction changes
    ax2 = fig.add_subplot(gs[0, 1])
    if all_direction_changes:
        ax2.hist(all_direction_changes, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.axvline(x=np.mean(all_direction_changes), color='red', linestyle='-', linewidth=2, 
                   label=f'Mean ({np.mean(all_direction_changes):.1f}°)')
        ax2.set_xlabel('Direction Change Magnitude')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Direction Changes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No direction change data available', ha='center', va='center')
    
    # Trajectory quality pie chart
    ax3 = fig.add_subplot(gs[0, 2])
    if trajectory_stats:
        excellent = sum(1 for t in trajectory_stats if t['max_curvature'] < 10)
        good = sum(1 for t in trajectory_stats if 10 <= t['max_curvature'] < 25)
        fair = sum(1 for t in trajectory_stats if 25 <= t['max_curvature'] < 40)
        poor = sum(1 for t in trajectory_stats if t['max_curvature'] >= 40)
        
        categories = ['Excellent\n(<10°)', 'Good\n(10-25°)', 'Fair\n(25-40°)', 'Poor\n(≥40°)']
        counts = [excellent, good, fair, poor]
        colors = ['darkgreen', 'lightgreen', 'orange', 'red']
        
        non_zero_categories, non_zero_counts, non_zero_colors = [], [], []
        for cat, count, color in zip(categories, counts, colors):
            if count > 0:
                non_zero_categories.append(f'{cat}\n({count})')
                non_zero_counts.append(count)
                non_zero_colors.append(color)
        
        if non_zero_counts:
            ax3.pie(non_zero_counts, labels=non_zero_categories, colors=non_zero_colors, 
                   autopct='%1.1f%%', startangle=90)
            ax3.set_title('Trajectory Quality by\nMax Curvature')
        else:
            ax3.text(0.5, 0.5, 'No quality data available', ha='center', va='center')
    
    # Summary statistics table
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    
    if trajectory_stats:
        total_trajectories = len(trajectory_stats)
        linear_trajectories = sum(1 for t in trajectory_stats if t['is_linear'])
        flagged_trajectories_count = total_trajectories - linear_trajectories
        
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
        
        table = ax4.table(cellText=summary_data, colLabels=['Metric', 'Value'],
                         loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        if flagged_trajectories_count > 0:
            table[(3, 1)].set_facecolor('lightcoral')
        
        ax4.set_title('Curvature Analysis Summary')
    
    # Individual trajectory examples (3 worst trajectories)
    if trajectory_stats:
        worst_trajectories = sorted(trajectory_stats, key=lambda x: x['max_curvature'], reverse=True)[:3]
        
        for idx, traj in enumerate(worst_trajectories):
            ax = fig.add_subplot(gs[2, idx])
            
            traj_id = traj['trajectory_id']
            if traj_id in trajectory_angle_analyses:
                analysis = trajectory_angle_analyses[traj_id]
                curvature_angles = analysis.get('curvature_angles', [])
                
                if curvature_angles:
                    x_positions = range(len(curvature_angles))
                    ax.plot(x_positions, curvature_angles, 'o-', linewidth=2, markersize=6, 
                           color='blue', alpha=0.7)
                    
                    # Highlight flagged segments
                    flagged_segments = analysis.get('flagged_segments', [])
                    for segment in flagged_segments:
                        seg_idx = segment['segment_index'] - 1
                        if 0 <= seg_idx < len(curvature_angles):
                            ax.plot(seg_idx, curvature_angles[seg_idx], 'ro', markersize=10, 
                                   alpha=0.8, label='Flagged')
                    
                    ax.axhline(y=40, color='red', linestyle='--', alpha=0.7, label='Threshold')
                    
                    ax.set_xlabel('Contact Position Along Trajectory')
                    ax.set_ylabel('Curvature Angle (°)')
                    ax.set_title(f'Trajectory {traj_id}\nMax: {traj["max_curvature"]:.1f}°')
                    ax.grid(True, alpha=0.3)
                    
                    if flagged_segments and idx == 0:
                        ax.legend()
                else:
                    ax.text(0.5, 0.5, f'No data for\nTrajectory {traj_id}', 
                           ha='center', va='center')
                    ax.set_title(f'Trajectory {traj_id}')
            
            ax.set_xlim(-0.5, max(10, len(curvature_angles) + 0.5) if curvature_angles else 10)
    
    # Detailed trajectory table
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')
    
    if trajectory_stats:
        sorted_trajectories = sorted(trajectory_stats, key=lambda x: x['max_curvature'], reverse=True)
        display_trajectories = sorted_trajectories[:12]
        
        table_data = []
        columns = ['Trajectory ID', 'Contacts', 'Max Curvature (°)', 'Mean Curvature (°)', 
                  'Flagged Segments', 'Linear?', 'Linearity Score']
        
        for traj in display_trajectories:
            row = [
                traj['trajectory_id'],
                traj['contact_count'],
                f"{traj['max_curvature']:.1f}",
                f"{traj['mean_curvature']:.1f}",
                traj['flagged_count'],
                'Yes' if traj['is_linear'] else 'No',
                f"{traj['linearity_score']:.3f}"
            ]
            table_data.append(row)
        
        if table_data:
            detail_table = ax5.table(cellText=table_data, colLabels=columns,
                                   loc='center', cellLoc='center')
            detail_table.auto_set_font_size(False)
            detail_table.set_fontsize(9)
            detail_table.scale(1, 1.2)
            
            # Color code rows based on linearity
            for i, traj in enumerate(display_trajectories):
                if not traj['is_linear']:
                    for j in range(len(columns)):
                        detail_table[(i+1, j)].set_facecolor('lightcoral')
                elif traj['max_curvature'] > 25:
                    for j in range(len(columns)):
                        detail_table[(i+1, j)].set_facecolor('lightyellow')
        
        title_suffix = f" (Top {len(display_trajectories)} by Max Curvature)" if len(sorted_trajectories) > 12 else ""
        ax5.set_title(f'Detailed Curvature Analysis{title_suffix}')
    
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        save_path = os.path.join(output_dir, 'enhanced_contact_angle_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Enhanced contact angle analysis saved to {save_path}")
    
    return fig

#==============================================================================
# SECTION 8: ADAPTIVE CLUSTERING
#==============================================================================

def adaptive_clustering_parameters(coords_array, initial_eps=8, initial_min_neighbors=3, 
                                   expected_contact_counts=[5, 8, 10, 12, 15, 18],
                                   max_iterations=10, eps_step=0.5, verbose=True):
    """
    Adaptively find optimal eps and min_neighbors parameters for DBSCAN clustering.
    
    Args:
        coords_array (numpy.ndarray): Array of electrode coordinates
        initial_eps (float): Initial value for max neighbor distance
        initial_min_neighbors (int): Initial value for min_samples
        expected_contact_counts (list): List of expected electrode contact counts
        max_iterations (int): Maximum number of iterations to try
        eps_step (float): Step size for adjusting eps
        verbose (bool): Whether to print progress details
        
    Returns:
        dict: Results dictionary with optimal parameters and visualization
    """
    # Initialize parameters
    current_eps = initial_eps
    current_min_neighbors = initial_min_neighbors
    best_score = 0
    best_params = {'eps': current_eps, 'min_neighbors': current_min_neighbors}
    best_clusters = None
    iterations_data = []
    
    # Function to evaluate clustering quality
    def evaluate_clustering(clusters, n_points):
        cluster_sizes = Counter([c for c in clusters if c != -1])
        
        if not cluster_sizes:
            return 0, 0, 0, {}
        
        valid_clusters = 0
        cluster_quality = {}
        
        for cluster_id, size in cluster_sizes.items():
            closest_expected = min(expected_contact_counts, key=lambda x: abs(x - size))
            difference = abs(closest_expected - size)
            
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
            
        clustered_percentage = sum(clusters != -1) / n_points * 100
        n_clusters = len(cluster_sizes)
        valid_percentage = (valid_clusters / n_clusters * 100) if n_clusters > 0 else 0
        
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
        
        # Store iteration data
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
            if current_min_neighbors > 2:
                current_min_neighbors -= 1
                if verbose:
                    print(f"  → Too many noise points, decreasing min_neighbors to {current_min_neighbors}")
            else:
                current_eps += eps_step
                if verbose:
                    print(f"  → Too many noise points, increasing eps to {current_eps}")
        elif n_clusters > 2 * len(expected_contact_counts):
            current_eps += eps_step
            if verbose:
                print(f"  → Too many small clusters, increasing eps to {current_eps}")
        elif valid_percentage < 50 and clustered_percentage > 80:
            current_eps -= eps_step * 0.5
            if verbose:
                print(f"  → Clusters don't match expected sizes, slightly decreasing eps to {current_eps}")
        else:
            if iteration % 2 == 0:
                current_eps += eps_step * 0.5
                if verbose:
                    print(f"  → Fine-tuning, slightly increasing eps to {current_eps}")
            else:
                current_eps -= eps_step * 0.3
                if verbose:
                    print(f"  → Fine-tuning, slightly decreasing eps to {current_eps}")
        
        # Ensure eps doesn't go below minimum threshold
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
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Adaptive Clustering Parameter Search', fontsize=18)
    
    # Create grid layout
    gs = GridSpec(3, 3, figure=fig)
    
    # Parameter trajectory plot
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
    
    # Score plot
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
    
    # Cluster count plot
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
    
    # 3D visualization of best clustering result
    ax4 = fig.add_subplot(gs[1, :], projection='3d')
    
    # Get best iteration
    best_iteration = iterations_data[np.argmax([data['score'] for data in iterations_data])]
    clusters = best_iteration['clusters']
    
    # Get unique clusters
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
    
    # Create a simplified legend
    handles, labels = ax4.get_legend_handles_labels()
    if len(handles) > 15:
        handles = handles[:14] + [handles[-1]]
        labels = labels[:14] + [labels[-1]]
    
    ax4.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1))
    
    # Cluster size distribution
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
    
    # Summary table with parameter recommendations
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

#==============================================================================
# SECTION 9: VISUALIZATION FUNCTIONS
#==============================================================================

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
            surface_points_ras = convert_surface_vertices_to_ras(brain_volume, vertices)
            
            # Downsample surface points for better performance
            if len(surface_points_ras) > 10000:
                step = len(surface_points_ras) // 10000
                surface_points_ras = surface_points_ras[::step]
            
            print(f"Rendering {len(surface_points_ras)} surface points...")
            
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
        arrow_length = min(traj_info['length'] * 0.3, 15)
        
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
    
    unique_clusters = []
    for c in set(clusters):
        if c == -1:
            continue
        unique_clusters.append(c)
    
    # Create colormaps
    n_clusters = len(unique_clusters)
    cluster_cmap = plt.colormaps['tab20'].resampled(max(1, n_clusters))
    
    # Plot electrodes with cluster colors
    for i, cluster_id in enumerate(unique_clusters):
        mask = clusters == cluster_id
        ax.scatter(coords_array[mask, 0], coords_array[mask, 1], coords_array[mask, 2], 
                  c=[cluster_cmap(i)], label=f'Cluster {cluster_id}', s=80, alpha=0.8)
    
    # Plot trajectories
    for traj in results.get('trajectories', []):
        try:
            if isinstance(traj['cluster_id'], (int, np.integer)):
                color_idx = [i for i, c in enumerate(unique_clusters) if c == traj['cluster_id']]
                if color_idx:
                    color_idx = color_idx[0]
                else:
                    color_idx = 0
            else:
                color_idx = hash(str(traj['cluster_id'])) % len(cluster_cmap)
        except:
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
        arrow_length = traj['length_mm'] * 0.3
        
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
                try:
                    if str(match['bolt_id']) == str(bolt_id):
                        color = 'crimson'
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


def visualize_combined_results(coords_array, results, output_dir=None, bolt_directions=None):
    """
    Create and save/display visualizations of trajectory analysis results.
    
    Args:
        coords_array (numpy.ndarray): Array of electrode coordinates
        results (dict): Results from integrated_trajectory_analysis
        output_dir (str, optional): Directory to save PDF report
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
                
            # Create noise points page
            fig = create_noise_points_page(coords_array, results)
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
            
        fig = create_noise_points_page(coords_array, results)
        plt.show()

#==============================================================================
# SECTION 10: MAIN EXECUTION FUNCTION
#==============================================================================

def main(use_combined_volume=True, use_original_reports=True, 
         detect_duplicates=False, duplicate_threshold=0.5, 
         use_adaptive_clustering=False, max_iterations=10,
         validate_spacing=True, expected_spacing_range=(3.0, 5.0),
         refine_trajectories=False, max_contacts_per_trajectory=20,
         validate_entry_angles=True, hemisphere='both',
         analyze_contact_angles=True, angle_threshold=40.0):
    """
    Enhanced main function for electrode trajectory analysis with comprehensive options.
    
    Args:
        use_combined_volume (bool): Whether to use the combined volume approach
        use_original_reports (bool): Whether to generate the original format reports
        detect_duplicates (bool): Whether to detect duplicate centroids
        duplicate_threshold (float): Threshold for duplicate detection in mm
        use_adaptive_clustering (bool): Whether to use adaptive clustering parameter selection
        max_iterations (int): Maximum number of iterations for adaptive parameter search
        validate_spacing (bool): Whether to validate electrode spacing
        expected_spacing_range (tuple): Expected range for contact spacing (min, max) in mm
        refine_trajectories (bool): Whether to apply trajectory refinement
        max_contacts_per_trajectory (int): Maximum number of contacts allowed
        validate_entry_angles (bool): Whether to validate entry angles
        hemisphere (str): 'left' (x < 0), 'right' (x > 0), or 'both' (no filtering)
        analyze_contact_angles (bool): Whether to analyze contact angles
        angle_threshold (float): Threshold for flagging non-linear segments
        
    Returns:
        dict: Results dictionary containing all analysis results
    """
    try:
        start_time = time.time()
        print(f"Starting electrode trajectory analysis...")
        print(f"Options: combined_volume={use_combined_volume}, adaptive_clustering={use_adaptive_clustering}, "
              f"hemisphere={hemisphere}, analyze_contact_angles={analyze_contact_angles}")
        
        # Step 1: Load required volumes from Slicer
        print("Loading volumes from Slicer...")
        electrodes_volume = slicer.util.getNode('P2_electrode_mask_success')
        brain_volume = slicer.util.getNode("patient2_mask_5")
        
        # Create output directories
        base_dir = r"C:\Users\rocia\Downloads\TFG\Cohort\Centroids\P2_BoltHeadandpaths_organized"
        
        output_dir_name = "trajectory_analysis_results"
        if hemisphere.lower() != 'both':
            output_dir_name += f"_{hemisphere}_hemisphere"
        
        output_dir = os.path.join(base_dir, output_dir_name)
        os.makedirs(output_dir, exist_ok=True)
        
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
                'hemisphere': hemisphere,
                'analyze_contact_angles': analyze_contact_angles,
                'angle_threshold': angle_threshold
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
        
        # Apply hemisphere filtering to coordinates
        coords_array, hemisphere_mask, filtered_indices = filter_coordinates_by_hemisphere(
            original_coords_array, hemisphere, verbose=True
        )
        
        if len(coords_array) == 0:
            print(f"No coordinates found in {hemisphere} hemisphere. Cannot proceed with analysis.")
            return {'error': f'No coordinates in {hemisphere} hemisphere'}
        
        all_results['electrode_count'] = len(coords_array)
        all_results['original_electrode_count'] = len(original_coords_array)
        all_results['hemisphere_filtering'] = {
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
                    combined_volume, brain_volume=brain_volume
                )
                
                # Filter combined trajectories by hemisphere
                combined_trajectories = filter_bolt_directions_by_hemisphere(
                    all_combined_trajectories, hemisphere, verbose=True
                )
                
                all_results['combined_volume'] = {
                    'trajectories': combined_trajectories,
                    'trajectory_count': len(combined_trajectories),
                    'original_trajectory_count': len(all_combined_trajectories)
                }
                
                print(f"Extracted {len(combined_trajectories)} trajectories from combined volume (after hemisphere filtering).")
                
                # Create trajectory lines volume
                if combined_trajectories:
                    trajectory_volume = create_trajectory_lines_volume(
                        combined_trajectories, combined_volume, output_dir
                    )
                    all_results['combined_volume']['trajectory_volume'] = trajectory_volume
                
                # Visualize combined volume trajectories
                if combined_trajectories:
                    print(f"Creating combined volume visualizations...")
                    fig = visualize_combined_volume_trajectories(
                        combined_trajectories, coords_array=coords_array,
                        brain_volume=brain_volume, output_dir=output_dir
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
                
                # Filter entry points by hemisphere
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
            
            # Run integrated trajectory analysis with optimal parameters
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
                adaptive_dir = os.path.join(output_dir, "adaptive_clustering")
                os.makedirs(adaptive_dir, exist_ok=True)
                
                # Save parameter search visualization
                plt.figure(parameter_search['visualization'].number)
                plt.savefig(os.path.join(adaptive_dir, 'adaptive_parameter_search.png'), dpi=300)
                
                # Save parameter search results to PDF
                with PdfPages(os.path.join(adaptive_dir, 'adaptive_parameter_search.pdf')) as pdf:
                    pdf.savefig(parameter_search['visualization'])
                
                print(f"✅ Adaptive clustering results saved to {adaptive_dir}")
                
            integrated_results['parameter_search'] = parameter_search
        else:
            # Run the original integrated trajectory analysis with fixed parameters
            print("Running integrated trajectory analysis with fixed parameters (eps=7.5, min_neighbors=3)...")
            integrated_results = integrated_trajectory_analysis(
                coords_array=coords_array,
                entry_points=entry_points,
                max_neighbor_distance=7.5,
                min_neighbors=3,
                expected_spacing_range=expected_spacing_range if validate_spacing else None
            )

        # Step 6: Contact angle analysis
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
                with PdfPages(os.path.join(angle_analysis_dir, 'contact_angle_analysis.pdf')) as pdf:
                    pdf.savefig(angle_fig)
                    plt.close(angle_fig)
                
                print(f"✅ Contact angle analysis reports saved to {angle_analysis_dir}")
        
        # Step 7: Add trajectory validation
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
        
        # Step 8: Get bolt directions and filter by hemisphere
        bolt_directions = None
        bolt_head_volume = slicer.util.getNode('P2_bolt_heads')
        
        if bolt_head_volume and entry_points_volume:
            print("Extracting bolt-to-entry directions...")
            
            # If we used combined volume, convert trajectories to bolt directions format
            if combined_trajectories and use_combined_volume:
                bolt_directions = {}
                for bolt_id, traj_info in combined_trajectories.items():
                    # Collect points
                    points = []
                    if 'trajectory_points' in traj_info:
                        points = traj_info['trajectory_points']
                    
                    bolt_directions[bolt_id] = {
                        'start_point': traj_info['start_point'],
                        'end_point': traj_info['end_point'],
                        'direction': traj_info['direction'],
                        'length': traj_info['length'],
                        'points': points,
                        'method': 'combined_volume'
                    }
                
                # Apply hemisphere filtering to bolt directions
                bolt_directions = filter_bolt_directions_by_hemisphere(
                    bolt_directions, hemisphere, verbose=False
                )
            
            print(f"Found {len(bolt_directions) if bolt_directions else 0} bolt-to-entry directions.")
            all_results['bolt_directions'] = bolt_directions
            
            # Match bolt directions to trajectories
            if bolt_directions and integrated_results.get('trajectories'):
                print("Matching bolt directions to trajectories...")
                matches = match_bolt_directions_to_trajectories(
                    bolt_directions, integrated_results['trajectories'],
                    max_distance=40, max_angle=40.0
                )
                integrated_results['bolt_trajectory_matches'] = matches
                all_results['bolt_trajectory_matches'] = matches
                print(f"Found {len(matches)} matches between bolt directions and trajectories.")
                
            # Add entry angle validation if requested
            if validate_entry_angles and bolt_directions and brain_volume:
                print("Validating entry angles against surgical constraints (30-60°)...")
                verify_directions_with_brain(bolt_directions, brain_volume)
                
                # Count valid/invalid angles
                valid_angles = sum(1 for info in bolt_directions.values() if info.get('is_angle_valid', False))
                total_angles = len(bolt_directions)
                
                print(f"Entry angle validation: {valid_angles}/{total_angles} valid ({valid_angles/total_angles*100:.1f}%)")
                
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
            
            print(f"✅ Spacing validation reports saved to {output_dir}")
            
            # Add the figures to the results
            if 'figures' not in integrated_results:
                integrated_results['figures'] = {}
            integrated_results['figures']['spacing_validation'] = spacing_fig
        
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
                    combined_trajectories, coords_array=coords_array, brain_volume=brain_volume
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
        # Show hemisphere filtering results
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
        
        # Add contact angle analysis summary
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

#==============================================================================
# SECTION 11: CONVENIENCE FUNCTIONS AND USAGE EXAMPLES
#==============================================================================

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

#==============================================================================
# SECTION 12: EXAMPLE USAGE AND MAIN EXECUTION
#==============================================================================

if __name__ == "__main__":
    # Example usage with different configurations
    
    # Basic analysis with default parameters
    results_basic = main()
    
    # Comprehensive analysis with all features enabled
    results_comprehensive = main(
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
        hemisphere='left',
        analyze_contact_angles=True,
        angle_threshold=40.0
    )
    
    # Hemisphere-specific analysis
    # left_results = analyze_left_hemisphere()
    # right_results = analyze_right_hemisphere()
    # hemisphere_comparison = compare_hemispheres()
    
    print("Enhanced electrode trajectory analysis completed.")

# Execute the analysis 
# exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Electrode_path\adaptive_clustering.py').read())