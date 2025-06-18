"""
Streamlined Electrode Trajectory Analysis Module for Slicer Extension

This module provides core functionality for analyzing SEEG electrode trajectories
with essential features only: adaptive clustering, hemisphere filtering, merging/splitting,
duplicate detection, and contact angle analysis.

Essential outputs:
1. HTML report with interactive visualization
2. CSV file with trajectory features for annotation
3. Contact angle analysis (if enabled)

Author: Rocío Ávalos (Streamlined for Slicer Extension)
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
import pandas as pd
import time

# Import utility functions from external modules
from Outermost_centroids_coordinates.outermost_centroids_vol_slicer import (
    get_ras_coordinates_from_ijk, get_array_from_volume, calculate_centroids_numpy,
    get_centroids_ras, get_surface_from_volume, convert_surface_vertices_to_ras, 
    filter_centroids_by_surface_distance
)
from End_points.midplane_prueba import get_all_centroids

#------------------------------------------------------------------------------
# CORE ANALYSIS FUNCTIONS (ESSENTIAL)
#------------------------------------------------------------------------------

class Arrow3D(FancyArrowPatch):
    """3D arrow patch for visualization."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return min(zs)

def calculate_angles(direction):
    """Calculate angles between a direction vector and principal axes."""
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

def integrated_trajectory_analysis(coords_array, entry_points=None, max_neighbor_distance=8, min_neighbors=3, 
                                  expected_spacing_range=(3.0, 5.0)):
    """
    Core trajectory analysis combining DBSCAN clustering, Louvain community detection,
    and PCA-based trajectory analysis.
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
    
    # Combined analysis (mapping between clusters and communities)
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

        # Map each cluster to its dominant community
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
            # Apply PCA
            pca = PCA(n_components=3)
            pca.fit(cluster_coords)
            
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
            
            # Sort coordinates along trajectory
            sorted_indices = np.argsort(projected)
            sorted_coords = cluster_coords[sorted_indices]
            
            # Calculate trajectory metrics
            distances = np.linalg.norm(np.diff(sorted_coords, axis=0), axis=1)
            spacing_regularity = np.std(distances) / np.mean(distances) if len(distances) > 1 else np.nan
            trajectory_length = np.sum(distances)
            
            # Spacing validation
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
                "spline_points": spline_points.tolist() if spline_points is not None else None,
                "angles_with_axes": angles,
                "pca_variance": pca.explained_variance_ratio_.tolist(),
                "sorted_coords": sorted_coords.tolist()
            }
            
            if spacing_validation:
                trajectory_dict["spacing_validation"] = spacing_validation
            
            trajectories.append(trajectory_dict)
            
        except Exception as e:
            logging.warning(f"PCA failed for cluster {cluster_id}: {e}")
            continue
    
    results['trajectories'] = trajectories
    results['n_trajectories'] = len(trajectories)
    
    # Add noise points information
    noise_mask = clusters == -1
    results['dbscan']['noise_points_coords'] = coords_array[noise_mask].tolist()
    results['dbscan']['noise_points_indices'] = np.where(noise_mask)[0].tolist()

    return results

#------------------------------------------------------------------------------
# SPACING VALIDATION
#------------------------------------------------------------------------------

def validate_electrode_spacing(trajectory_points, expected_spacing_range=(3.0, 5.0)):
    """Validate the spacing between electrode contacts in a trajectory."""
    distances = []
    for i in range(1, len(trajectory_points)):
        dist = np.linalg.norm(trajectory_points[i] - trajectory_points[i-1])
        distances.append(dist)
    
    min_spacing = np.min(distances) if distances else np.nan
    max_spacing = np.max(distances) if distances else np.nan
    mean_spacing = np.mean(distances) if distances else np.nan
    std_spacing = np.std(distances) if distances else np.nan
    
    min_expected, max_expected = expected_spacing_range
    valid_spacings = [min_expected <= d <= max_expected for d in distances]
    
    too_close = [i for i, d in enumerate(distances) if d < min_expected]
    too_far = [i for i, d in enumerate(distances) if d > max_expected]
    
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

#------------------------------------------------------------------------------
# ADAPTIVE CLUSTERING
#------------------------------------------------------------------------------

def adaptive_clustering_parameters(coords_array, initial_eps=8, initial_min_neighbors=3, 
                                   expected_contact_counts=[5, 8, 10, 12, 15, 18],
                                   max_iterations=10, eps_step=0.5, verbose=True):
    """Adaptively find optimal eps and min_neighbors parameters for DBSCAN clustering."""
    from collections import Counter
    
    current_eps = initial_eps
    current_min_neighbors = initial_min_neighbors
    best_score = 0
    best_params = {'eps': current_eps, 'min_neighbors': current_min_neighbors}
    best_clusters = None
    iterations_data = []
    
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
    
    for iteration in range(max_iterations):
        dbscan = DBSCAN(eps=current_eps, min_samples=current_min_neighbors)
        clusters = dbscan.fit_predict(coords_array)
        
        score, valid_percentage, clustered_percentage, cluster_quality = evaluate_clustering(clusters, len(coords_array))
        
        unique_clusters = set(clusters)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        n_noise = np.sum(clusters == -1)
        
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
            else:
                current_eps += eps_step
        elif n_clusters > 2 * len(expected_contact_counts):
            current_eps += eps_step
        elif valid_percentage < 50 and clustered_percentage > 80:
            current_eps -= eps_step * 0.5
        else:
            if iteration % 2 == 0:
                current_eps += eps_step * 0.5
            else:
                current_eps -= eps_step * 0.3
        
        current_eps = max(current_eps, 1.0)
    
    return {
        'optimal_eps': best_params['eps'],
        'optimal_min_neighbors': best_params['min_neighbors'],
        'score': best_score,
        'iterations_data': iterations_data,
        'best_clusters': best_clusters
    }

#------------------------------------------------------------------------------
# HEMISPHERE FILTERING
#------------------------------------------------------------------------------

def filter_coordinates_by_hemisphere(coords_array, hemisphere='left', verbose=True):
    """Filter electrode coordinates by hemisphere."""
    if hemisphere.lower() == 'both':
        if verbose:
            print(f"No hemisphere filtering applied. Keeping all {len(coords_array)} coordinates.")
        return coords_array, np.ones(len(coords_array), dtype=bool), np.arange(len(coords_array))
    
    if hemisphere.lower() == 'left':
        hemisphere_mask = coords_array[:, 0] < 0
        hemisphere_name = "left"
    elif hemisphere.lower() == 'right':
        hemisphere_mask = coords_array[:, 0] > 0
        hemisphere_name = "right"
    else:
        raise ValueError("hemisphere must be 'left', 'right', or 'both'")
    
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

def print_angle_analysis_summary(trajectory_angle_analyses):
    """
    Print detailed summary of angle analysis results.
    
    Args:
        trajectory_angle_analyses (dict): Results from angle analysis
    """
    if not trajectory_angle_analyses:
        print("No angle analysis results to summarize.")
        return
    
    print(f"\n=== CONTACT ANGLE ANALYSIS SUMMARY ===")
    
    # Overall statistics
    total_trajectories = len(trajectory_angle_analyses)
    linear_trajectories = sum(1 for analysis in trajectory_angle_analyses.values() if analysis.get('is_linear', True))
    flagged_trajectories = total_trajectories - linear_trajectories
    
    print(f"Total trajectories analyzed: {total_trajectories}")
    print(f"Linear trajectories: {linear_trajectories} ({linear_trajectories/total_trajectories*100:.1f}%)")
    print(f"Flagged trajectories: {flagged_trajectories} ({flagged_trajectories/total_trajectories*100:.1f}%)")
    
    # Collect angle statistics
    all_max_curvatures = [analysis.get('max_curvature', 0) for analysis in trajectory_angle_analyses.values()]
    all_mean_curvatures = [analysis.get('mean_curvature', 0) for analysis in trajectory_angle_analyses.values()]
    
    if all_max_curvatures:
        print(f"\nCurvature Statistics:")
        print(f"- Max curvature range: {min(all_max_curvatures):.1f}° - {max(all_max_curvatures):.1f}°")
        print(f"- Average max curvature: {np.mean(all_max_curvatures):.1f}°")
        print(f"- Average mean curvature: {np.mean(all_mean_curvatures):.1f}°")
    
    # Quality categories
    excellent = sum(1 for angle in all_max_curvatures if angle < 10)
    good = sum(1 for angle in all_max_curvatures if 10 <= angle < 25)
    fair = sum(1 for angle in all_max_curvatures if 25 <= angle < 40)
    poor = sum(1 for angle in all_max_curvatures if angle >= 40)
    
    print(f"\nQuality Distribution:")
    print(f"- Excellent (<10°): {excellent}")
    print(f"- Good (10-24°): {good}")
    print(f"- Fair (25-39°): {fair}")
    print(f"- Poor (≥40°): {poor}")
    
    # Show worst trajectories
    if flagged_trajectories > 0:
        print(f"\nTrajectories with highest curvature:")
        sorted_analyses = sorted(trajectory_angle_analyses.items(), 
                               key=lambda x: x[1].get('max_curvature', 0), reverse=True)
        for i, (traj_id, analysis) in enumerate(sorted_analyses[:5]):
            if analysis.get('max_curvature', 0) >= 40:
                print(f"- Trajectory {traj_id}: {analysis.get('max_curvature', 0):.1f}° max curvature, "
                      f"{analysis.get('flagged_segments_count', 0)} flagged segments")

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

def ensure_angle_analysis_after_refinement(results, coords_array, analyze_contact_angles=True, angle_threshold=40.0):
    """
    FIXED VERSION: Ensure contact angle analysis is performed after trajectory refinement.
    This is the function that should be called in your main() function.
    """
    if not analyze_contact_angles or 'trajectories' not in results:
        return results
    
    print("=== PERFORMING ANGLE ANALYSIS AFTER REFINEMENT ===")
    
    # Add angle analysis to all trajectories (including split/merged ones)
    add_angle_analysis_to_trajectories(
        results['trajectories'], 
        coords_array, 
        results, 
        angle_threshold
    )
    
    # Create trajectory angle analyses dictionary
    trajectory_angle_analyses = {}
    for traj in results['trajectories']:
        if 'contact_angles' in traj:
            trajectory_angle_analyses[traj['cluster_id']] = traj['contact_angles']
    
    # Store in results
    results['contact_angle_analysis'] = trajectory_angle_analyses
    
    print(f"=== ANGLE ANALYSIS COMPLETE: {len(trajectory_angle_analyses)} trajectories analyzed ===")
    
    return results
#------------------------------------------------------------------------------
# TRAJECTORY REFINEMENT (MERGING/SPLITTING)
#------------------------------------------------------------------------------

def targeted_trajectory_refinement(trajectories, expected_contact_counts=[5, 8, 10, 12, 15, 18], 
                                 max_expected=18, tolerance=3):
    """Apply splitting and merging operations only to trajectories that need it."""
    merge_candidates = []
    split_candidates = []
    valid_trajectories = []
    
    for traj in trajectories:
        contact_count = traj['electrode_count']
        closest_expected = min(expected_contact_counts, key=lambda x: abs(x - contact_count))
        difference = abs(closest_expected - contact_count)
        
        if contact_count > max_expected:
            traj['closest_expected'] = closest_expected
            traj['count_difference'] = difference
            split_candidates.append(traj)
        elif difference > tolerance and contact_count < closest_expected:
            traj['closest_expected'] = closest_expected
            traj['count_difference'] = difference
            traj['missing_contacts'] = closest_expected - contact_count
            merge_candidates.append(traj)
        else:
            valid_trajectories.append(traj)
    
    print(f"Trajectory refinement: {len(valid_trajectories)} valid, {len(merge_candidates)} merge candidates, {len(split_candidates)} split candidates")
    
    # Process merge candidates
    merge_candidates.sort(key=lambda x: x['missing_contacts'])
    merged_trajectories = []
    used_in_merge = set()
    
    for i, traj1 in enumerate(merge_candidates):
        if traj1['cluster_id'] in used_in_merge:
            continue
            
        best_match = None
        best_score = float('inf')
        
        for j, traj2 in enumerate(merge_candidates):
            if i == j or traj2['cluster_id'] in used_in_merge:
                continue
                
            combined_count = traj1['electrode_count'] + traj2['electrode_count']
            closest_expected = min(expected_contact_counts, key=lambda x: abs(x - combined_count))
            combined_difference = abs(closest_expected - combined_count)
            
            if combined_difference < min(traj1['count_difference'], traj2['count_difference']):
                score = check_merge_compatibility(traj1, traj2)
                if score is not None and score < best_score:
                    best_match = (j, traj2, score)
                    best_score = score
        
        if best_match:
            j, traj2, score = best_match
            merged_traj = merge_trajectories(traj1, traj2)
            merged_trajectories.append(merged_traj)
            used_in_merge.add(traj1['cluster_id'])
            used_in_merge.add(traj2['cluster_id'])
            print(f"Merged: {traj1['cluster_id']} + {traj2['cluster_id']} = {merged_traj['cluster_id']}")
        else:
            merged_trajectories.append(traj1)
    
    for traj in merge_candidates:
        if traj['cluster_id'] not in used_in_merge:
            merged_trajectories.append(traj)
    
    # Process split candidates
    final_trajectories = []
    
    for traj in split_candidates:
        split_result = split_trajectory(traj, expected_contact_counts)
        
        if split_result['success']:
            final_trajectories.extend(split_result['trajectories'])
            print(f"Split {traj['cluster_id']} into {len(split_result['trajectories'])} trajectories")
        else:
            final_trajectories.append(traj)
    
    final_trajectories.extend(merged_trajectories)
    final_trajectories.extend(valid_trajectories)
    
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
    """Check if two trajectories can be merged."""
    endpoints1 = np.array(traj1['endpoints'])
    endpoints2 = np.array(traj2['endpoints'])
    
    distances = [
        np.linalg.norm(endpoints1[0] - endpoints2[0]),
        np.linalg.norm(endpoints1[0] - endpoints2[1]),
        np.linalg.norm(endpoints1[1] - endpoints2[0]),
        np.linalg.norm(endpoints1[1] - endpoints2[1])
    ]
    
    min_distance = min(distances)
    
    if min_distance > max_distance:
        return None
    
    dir1 = np.array(traj1['direction'])
    dir2 = np.array(traj2['direction'])
    
    angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(dir1, dir2)), -1.0, 1.0)))
    
    if np.dot(dir1, dir2) < 0:
        angle = 180 - angle
    
    if angle > max_angle_diff:
        return None
    
    return min_distance + angle * 0.5

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

def split_trajectory(traj, expected_contact_counts=[5, 8, 10, 12, 15, 18]):
    """
    IMPROVED VERSION: Split a trajectory using PCA-guided analysis and DBSCAN refinement.
    
    This function:
    1. Uses PCA to find the main trajectory direction
    2. Projects points along this direction to find natural gaps
    3. Uses DBSCAN on problematic regions for better clustering
    4. Creates properly ordered trajectory segments
    
    Args:
        traj: Trajectory dictionary with 'sorted_coords' and other properties
        expected_contact_counts: List of expected electrode contact counts
        
    Returns:
        dict: Split result with success flag and trajectory list
    """
    if 'sorted_coords' not in traj or not traj['sorted_coords']:
        return {'success': False, 'reason': 'No coordinates available', 'trajectories': [traj]}
    
    coords = np.array(traj['sorted_coords'])
    contact_count = traj['electrode_count']
    
    # Find potential combinations of expected counts
    best_combination = None
    min_difference = float('inf')
    
    # Try 2-way splits first
    for count1 in expected_contact_counts:
        for count2 in expected_contact_counts:
            if abs((count1 + count2) - contact_count) < min_difference:
                min_difference = abs((count1 + count2) - contact_count)
                best_combination = [count1, count2]
    
    # Try 3-way splits for very long trajectories
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
    
    n_expected_trajectories = len(best_combination)
    
    # Step 1: PCA to find main trajectory direction
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    
    pca = PCA(n_components=3)
    pca.fit(coords)
    main_direction = pca.components_[0]
    center = np.mean(coords, axis=0)
    
    # Project points along main direction
    projected_1d = np.dot(coords - center, main_direction)
    
    # Step 2: Find natural gaps using distance analysis
    sorted_indices = np.argsort(projected_1d)
    sorted_coords_1d = projected_1d[sorted_indices]
    sorted_coords_3d = coords[sorted_indices]
    
    # Calculate gaps between consecutive points
    gaps = np.diff(sorted_coords_1d)
    median_gap = np.median(gaps)
    large_gap_threshold = median_gap * 2.5  # Significant gaps
    
    # Find potential split points where gaps are unusually large
    large_gap_indices = np.where(gaps > large_gap_threshold)[0]
    
    # Step 3: Use gap-based splitting or DBSCAN refinement
    if len(large_gap_indices) > 0:
        # Use gap-based splitting
        split_points = large_gap_indices + 1  # Convert to split indices
        
        # Create segments based on gaps
        segments = []
        start_idx = 0
        
        for split_point in split_points:
            if split_point > start_idx:
                segment = sorted_coords_3d[start_idx:split_point]
                if len(segment) >= 3:  # Minimum points for a trajectory
                    segments.append(segment)
                start_idx = split_point
        
        # Add final segment
        if start_idx < len(sorted_coords_3d):
            final_segment = sorted_coords_3d[start_idx:]
            if len(final_segment) >= 3:
                segments.append(final_segment)
        
        # If we don't have the right number of segments, try DBSCAN refinement
        if len(segments) != n_expected_trajectories:
            segments = apply_dbscan_refinement(coords, n_expected_trajectories, main_direction)
    else:
        # No clear gaps found, use DBSCAN refinement
        segments = apply_dbscan_refinement(coords, n_expected_trajectories, main_direction)
    
    # Step 4: Validate and create trajectory objects
    if len(segments) < 2:
        return {'success': False, 'reason': 'Could not create multiple valid segments', 'trajectories': [traj]}
    
    # Create split trajectories
    split_trajectories = []
    base_id = traj['cluster_id']
    
    for i, segment_coords in enumerate(segments):
        if len(segment_coords) < 3:
            continue
        
        # Re-apply PCA to each segment for proper trajectory analysis
        segment_pca = PCA(n_components=3)
        segment_pca.fit(segment_coords)
        
        # Sort segment points along its own principal direction
        segment_direction = segment_pca.components_[0]
        segment_center = np.mean(segment_coords, axis=0)
        segment_projected = np.dot(segment_coords - segment_center, segment_direction)
        segment_sorted_indices = np.argsort(segment_projected)
        segment_sorted_coords = segment_coords[segment_sorted_indices]
        
        # Create new trajectory
        sub_traj = traj.copy()
        new_id = f"S{i+1}_{base_id}"
        
        sub_traj['cluster_id'] = new_id
        sub_traj['electrode_count'] = len(segment_sorted_coords)
        sub_traj['is_split'] = True
        sub_traj['split_from'] = base_id
        sub_traj['split_index'] = i + 1
        sub_traj['split_method'] = 'pca_guided_dbscan'
        
        # Update trajectory properties
        sub_traj['sorted_coords'] = segment_sorted_coords.tolist()
        sub_traj['endpoints'] = [
            segment_sorted_coords[0].tolist(),
            segment_sorted_coords[-1].tolist()
        ]
        sub_traj['direction'] = segment_direction.tolist()
        sub_traj['center'] = segment_center.tolist()
        sub_traj['length_mm'] = float(np.linalg.norm(
            segment_sorted_coords[-1] - segment_sorted_coords[0]
        ))
        
        # Update linearity and PCA variance
        if len(segment_coords) > 2:
            sub_traj['linearity'] = float(segment_pca.explained_variance_ratio_[0])
            sub_traj['pca_variance'] = segment_pca.explained_variance_ratio_.tolist()
        else:
            sub_traj['linearity'] = 1.0
            sub_traj['pca_variance'] = [1.0, 0.0, 0.0]
        
        # Copy spacing validation if it exists
        if 'spacing_validation' in traj:
            spacing_validation = validate_electrode_spacing(
                segment_sorted_coords, 
                traj['spacing_validation'].get('expected_range', (3.0, 5.0))
            )
            sub_traj['spacing_validation'] = spacing_validation
        
        split_trajectories.append(sub_traj)
    
    success = len(split_trajectories) >= 2
    
    return {
        'success': success,
        'reason': f'Split into {len(split_trajectories)} trajectories using PCA-guided approach' if success else 'Failed to create multiple valid trajectories',
        'trajectories': split_trajectories if success else [traj],
        'n_trajectories': len(split_trajectories) if success else 1,
        'split_method': 'pca_guided_dbscan' if success else 'none'
    }
############################
## Improved Trajectory Splitting 
#########################

def apply_dbscan_refinement(coords, n_expected_trajectories, main_direction):
    """
    Apply DBSCAN clustering refined by PCA direction analysis.
    
    Args:
        coords: 3D coordinates array
        n_expected_trajectories: Expected number of trajectories to create
        main_direction: Main PCA direction vector
        
    Returns:
        list: List of coordinate segments
    """
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    
    # Create feature matrix combining spatial and directional information
    center = np.mean(coords, axis=0)
    
    # Project onto main direction (1D position along trajectory)
    projected_main = np.dot(coords - center, main_direction)
    
    # Calculate perpendicular distance from main axis
    projected_coords = center + np.outer(projected_main, main_direction)
    perpendicular_distances = np.linalg.norm(coords - projected_coords, axis=1)
    
    # Create enhanced feature space: [x, y, z, position_along_trajectory, distance_from_axis]
    features = np.column_stack([
        coords,
        projected_main,
        perpendicular_distances
    ])
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Try different DBSCAN parameters to get desired number of clusters
    best_segments = None
    best_score = float('inf')
    
    for eps in [0.3, 0.5, 0.7, 1.0, 1.2]:
        for min_samples in [2, 3, 4]:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(features_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters == n_expected_trajectories:
                # Perfect match - create segments
                segments = []
                for cluster_id in set(labels):
                    if cluster_id == -1:  # Skip noise
                        continue
                    cluster_mask = labels == cluster_id
                    cluster_coords = coords[cluster_mask]
                    if len(cluster_coords) >= 3:
                        segments.append(cluster_coords)
                
                if len(segments) == n_expected_trajectories:
                    return segments
            
            # Score based on how close we are to target
            score = abs(n_clusters - n_expected_trajectories)
            if score < best_score and n_clusters >= 2:
                best_score = score
                segments = []
                for cluster_id in set(labels):
                    if cluster_id == -1:
                        continue
                    cluster_mask = labels == cluster_id
                    cluster_coords = coords[cluster_mask]
                    if len(cluster_coords) >= 3:
                        segments.append(cluster_coords)
                
                if len(segments) >= 2:
                    best_segments = segments
    
    # If DBSCAN didn't work well, fall back to simple spatial splitting
    if best_segments is None or len(best_segments) < 2:
        return simple_spatial_split(coords, n_expected_trajectories, main_direction)
    
    return best_segments


def simple_spatial_split(coords, n_expected_trajectories, main_direction):
    """
    Fallback: Simple spatial splitting along main trajectory direction.
    
    Args:
        coords: 3D coordinates
        n_expected_trajectories: Number of expected splits
        main_direction: Main PCA direction
        
    Returns:
        list: List of coordinate segments
    """
    center = np.mean(coords, axis=0)
    projected_1d = np.dot(coords - center, main_direction)
    
    # Sort coordinates along main direction
    sorted_indices = np.argsort(projected_1d)
    sorted_coords = coords[sorted_indices]
    
    # Split into approximately equal segments
    segment_size = len(sorted_coords) // n_expected_trajectories
    segments = []
    
    for i in range(n_expected_trajectories):
        start_idx = i * segment_size
        if i == n_expected_trajectories - 1:
            # Last segment gets remaining points
            end_idx = len(sorted_coords)
        else:
            end_idx = (i + 1) * segment_size
        
        segment = sorted_coords[start_idx:end_idx]
        if len(segment) >= 3:
            segments.append(segment)
    
    return segments

def validate_trajectories(trajectories, expected_contact_counts, tolerance=2):
    """Validate trajectories against expected contact counts."""
    validation = {
        'total': len(trajectories),
        'valid': 0,
        'invalid': 0,
        'valid_ids': [],
        'invalid_details': []
    }
    
    for traj in trajectories:
        count = traj['electrode_count']
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
    Create visualization comparing original and refined trajectories.
    
    Args:
        coords_array (np.array): Original electrode coordinates
        original_trajectories (list): List of original trajectory dictionaries
        refined_trajectories (list): List of refined trajectory dictionaries  
        refinement_results (dict): Results from trajectory refinement
        
    Returns:
        matplotlib.figure.Figure: Figure showing refinement comparison
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 6))
    
    # Original trajectories (left plot)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(coords_array[:, 0], coords_array[:, 1], coords_array[:, 2], 
               c='black', marker='.', s=30, alpha=0.5, label='All electrodes')
    
    for i, traj in enumerate(original_trajectories):
        cluster_coords = get_trajectory_coordinates(traj['cluster_id'], {'trajectories': original_trajectories}, coords_array, np.arange(len(coords_array)))
        if cluster_coords is not None and len(cluster_coords) > 0:
            ax1.scatter(cluster_coords[:, 0], cluster_coords[:, 1], cluster_coords[:, 2],
                       s=60, alpha=0.8, label=f"T{traj['cluster_id']} ({traj['electrode_count']})")
    
    ax1.set_title(f'Original Trajectories ({len(original_trajectories)})')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    
    # Refined trajectories (right plot)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(coords_array[:, 0], coords_array[:, 1], coords_array[:, 2], 
               c='black', marker='.', s=50, alpha=0.5, label='All electrodes')
    
    for i, traj in enumerate(refined_trajectories):
        cluster_coords = get_trajectory_coordinates(traj['cluster_id'], {'trajectories': refined_trajectories}, coords_array, np.arange(len(coords_array)))
        if cluster_coords is not None and len(cluster_coords) > 0:
            color = 'green' if 'merged_from' in traj else 'blue' if 'split_from' in traj else 'gray'
            marker = 's' if 'merged_from' in traj else '^' if 'split_from' in traj else 'o'
            ax2.scatter(cluster_coords[:, 0], cluster_coords[:, 1], cluster_coords[:, 2],
                       c=color, marker=marker, s=60, alpha=0.8, 
                       label=f"T{traj['cluster_id']} ({traj['electrode_count']})")
    
    ax2.set_title(f'Refined Trajectories ({len(refined_trajectories)})')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    
    # Add legend for refinement types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='Merged'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=10, label='Split'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Original')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig

def create_scored_3d_visualization(coords_array, results, scores_df):
    """
    Create 3D visualization with trajectory scores using matplotlib.
    
    Args:
        coords_array (np.array): Electrode coordinates
        results (dict): Analysis results
        scores_df (pd.DataFrame): Trajectory scores dataframe
        
    Returns:
        matplotlib.figure.Figure: 3D plot with scored trajectories
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
    
    # Plot all points in light gray
    ax.scatter(coords_array[:, 0], coords_array[:, 1], coords_array[:, 2], 
               c='black', marker='.', s=50, alpha=0.3, label='All Electrodes')
    
    # Plot each trajectory with color coding
    for _, row in scores_df.iterrows():
        traj_id = row['trajectory_id']
        score = row['algorithm_score']
        
        cluster_coords = get_trajectory_coordinates(traj_id, results, coords_array, clusters)
        
        if cluster_coords is None or len(cluster_coords) == 0:
            continue
        
        # Color based on score
        if score >= 80:
            color = 'green'
            marker = 'o'
        elif score >= 60:
            color = 'orange' 
            marker = 's'
        else:
            color = 'red'
            marker = '^'
        
        # Plot trajectory points
        ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1], cluster_coords[:, 2],
                  c=color, marker=marker, s=60, alpha=0.8, edgecolor='black')
        
        # Add trajectory line
        if len(cluster_coords) > 1:
            ax.plot(cluster_coords[:, 0], cluster_coords[:, 1], cluster_coords[:, 2],
                   '-', color=color, linewidth=2, alpha=0.6)
        
        # Add label
        centroid = np.mean(cluster_coords, axis=0)
        ax.text(centroid[0], centroid[1], centroid[2], 
               f'T{traj_id}\n{score:.0f}', 
               fontsize=8, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)') 
    ax.set_zlabel('Z (mm)')
    ax.set_title('Trajectory Quality Scores\n(Green=Good ≥80, Orange=OK 60-79, Red=Bad <60)')
    
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

def create_interactive_scored_3d_visualization(coords_array, results, scores_df):
    """
    Create interactive 3D visualization with trajectory scores using plotly.
    
    Args:
        coords_array (np.array): Electrode coordinates
        results (dict): Analysis results
        scores_df (pd.DataFrame): Trajectory scores dataframe
        
    Returns:
        plotly.graph_objects.Figure or None: Interactive 3D plot
    """
    try:
        import plotly.graph_objects as go
        
        clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
        
        fig = go.Figure()
        
        # Plot all points as background
        fig.add_trace(go.Scatter3d(
            x=coords_array[:, 0],
            y=coords_array[:, 1], 
            z=coords_array[:, 2],
            mode='markers',
            marker=dict(size=4, color='black', opacity=0.3),
            name='All Electrodes',
            hovertemplate='<b>Electrode</b><br>X: %{x:.1f} mm<br>Y: %{y:.1f} mm<br>Z: %{z:.1f} mm<extra></extra>'
        ))
        
        # Plot each trajectory
        for _, row in scores_df.iterrows():
            traj_id = row['trajectory_id']
            score = row['algorithm_score']
            
            cluster_coords = get_trajectory_coordinates(traj_id, results, coords_array, clusters)
            
            if cluster_coords is None or len(cluster_coords) == 0:
                continue
            
            # Color based on score
            if score >= 80:
                color = 'green'
                quality = 'Good'
            elif score >= 60:
                color = 'orange' 
                quality = 'OK'
            else:
                color = 'red'
                quality = 'Bad'
            
            # Create hover text
            hover_text = []
            for i, coord in enumerate(cluster_coords):
                hover_text.append(
                    f"<b>Trajectory {traj_id}</b><br>" +
                    f"Contact {i+1}/{len(cluster_coords)}<br>" +
                    f"Position: ({coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f}) mm<br>" +
                    f"Algorithm Score: {score:.0f}<br>" +
                    f"Quality: {quality}<br>" +
                    f"Contacts: {row.get('n_contacts', len(cluster_coords))}<br>" +
                    f"Linearity: {row.get('linearity_pca', 0):.3f}"
                )
            
            # Plot trajectory
            fig.add_trace(go.Scatter3d(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                z=cluster_coords[:, 2],
                mode='markers+lines',
                line=dict(color=color, width=4),
                marker=dict(size=6, color=color),
                name=f'T{traj_id} - {quality} ({score:.0f})',
                hovertemplate='%{text}<extra></extra>',
                text=hover_text,
                showlegend=True
            ))
        
        # Update layout
        fig.update_layout(
            title='Interactive 3D Trajectory Quality Scores',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1000,
            height=700
        )
        
        return fig
        
    except ImportError:
        return None

def create_angle_analysis_visualization(trajectory_angle_analyses, output_dir=None):
    """
    Create visualization for contact angle analysis.
    
    Args:
        trajectory_angle_analyses (dict): Results from angle analysis
        output_dir (str, optional): Output directory for saving
        
    Returns:
        matplotlib.figure.Figure: Angle analysis visualization
    """
    import matplotlib.pyplot as plt
    
    if not trajectory_angle_analyses:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Collect data
    all_curvature_angles = []
    trajectory_stats = []
    
    for traj_id, analysis in trajectory_angle_analyses.items():
        all_curvature_angles.extend(analysis.get('curvature_angles', []))
        trajectory_stats.append({
            'id': traj_id,
            'max_curvature': analysis.get('max_curvature', 0),
            'is_linear': analysis.get('is_linear', True),
            'flagged_segments': analysis.get('flagged_count', 0)
        })
    
    # Distribution of curvature angles
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
    
    # Trajectory quality pie chart
    if trajectory_stats:
        excellent = sum(1 for t in trajectory_stats if t['max_curvature'] < 10)
        good = sum(1 for t in trajectory_stats if 10 <= t['max_curvature'] < 25)
        fair = sum(1 for t in trajectory_stats if 25 <= t['max_curvature'] < 40)
        poor = sum(1 for t in trajectory_stats if t['max_curvature'] >= 40)
        
        labels = []
        sizes = []
        colors = []
        color_map = {'Excellent': 'green', 'Good': 'lightgreen', 'Fair': 'orange', 'Poor': 'red'}
        
        for label, count in [('Excellent', excellent), ('Good', good), ('Fair', fair), ('Poor', poor)]:
            if count > 0:
                labels.append(f'{label}\n({count})')
                sizes.append(count)
                colors.append(color_map[label])
        
        if sizes:
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Trajectory Quality by Max Curvature')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'contact_angle_analysis.png'), dpi=300)
    
    return fig

def validate_electrode_clusters(results, expected_contact_counts):
    """
    Validate electrode clusters against expected contact counts.
    
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

#------------------------------------------------------------------------------
# DUPLICATE DETECTION
#------------------------------------------------------------------------------

def identify_potential_duplicates(centroids, threshold=0.5):
    """Identify potential duplicate centroids within threshold distance."""
    from scipy.spatial.distance import pdist, squareform
    
    centroids_array = np.array(centroids)
    distances = squareform(pdist(centroids_array))
    
    potential_duplicate_pairs = []
    for i in range(len(centroids_array)):
        for j in range(i+1, len(centroids_array)):
            if distances[i,j] < threshold:
                potential_duplicate_pairs.append((i, j, distances[i,j]))
    
    # Group duplicates that form clusters
    duplicate_groups = []
    used_indices = set()
    
    for i, j, _ in potential_duplicate_pairs:
        found_group = False
        for group in duplicate_groups:
            if i in group or j in group:
                if i not in group:
                    group.append(i)
                if j not in group:
                    group.append(j)
                found_group = True
                break
        
        if not found_group:
            duplicate_groups.append([i, j])
        
        used_indices.add(i)
        used_indices.add(j)
    
    stats = {
        'total_centroids': len(centroids_array),
        'potential_duplicate_pairs': len(potential_duplicate_pairs),
        'duplicate_groups': len(duplicate_groups),
        'centroids_in_duplicates': len(used_indices),
        'min_duplicate_distance': min([d for _, _, d in potential_duplicate_pairs]) if potential_duplicate_pairs else None,
        'max_duplicate_distance': max([d for _, _, d in potential_duplicate_pairs]) if potential_duplicate_pairs else None,
        'avg_duplicate_distance': np.mean([d for _, _, d in potential_duplicate_pairs]) if potential_duplicate_pairs else None
    }
    
    return {
        'all_centroids': centroids_array,
        'potential_duplicate_pairs': potential_duplicate_pairs,
        'duplicate_groups': duplicate_groups,
        'stats': stats
    }

def analyze_duplicates_on_trajectory(centroids, expected_count, threshold=0.5):
    """Analyze potential duplicate centroids on a single electrode trajectory."""
    from sklearn.decomposition import PCA
    
    centroids_array = np.array(centroids)
    
    # Get trajectory direction using PCA
    pca = PCA(n_components=3)
    pca.fit(centroids_array)
    direction = pca.components_[0]
    
    # Identify potential duplicates
    duplicate_result = identify_potential_duplicates(centroids_array, threshold=threshold)
    
    # Provide recommendations
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
    
    return {
        'centroids': centroids_array,
        'duplicate_result': duplicate_result,
        'expected_count': expected_count,
        'actual_count': stats['total_centroids'],
        'excess_count': stats['total_centroids'] - expected_count,
        'recommendations': recommendations
    }

def analyze_all_trajectories_for_duplicates(results, coords_array, expected_contact_counts=[5, 8, 10, 12, 15, 18], threshold=0.5):
    """Analyze all trajectories for potential duplicate centroids."""
    clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
    unique_clusters = set(clusters)
    
    if -1 in unique_clusters:
        unique_clusters.remove(-1)
    
    all_analyses = {}
    
    for trajectory_idx in unique_clusters:
        mask = clusters == trajectory_idx
        trajectory_centroids = coords_array[mask]
        
        expected_count = 8  # Default
        if 'electrode_validation' in results and 'clusters' in results['electrode_validation']:
            if trajectory_idx in results['electrode_validation']['clusters']:
                cluster_info = results['electrode_validation']['clusters'][trajectory_idx]
                if cluster_info['close']:
                    expected_count = cluster_info['closest_expected']
        
        analysis = analyze_duplicates_on_trajectory(trajectory_centroids, expected_count, threshold)
        all_analyses[trajectory_idx] = analysis
        
        duplicate_groups = analysis['duplicate_result']['duplicate_groups']
        if duplicate_groups:
            print(f"Trajectory {trajectory_idx}: Found {len(duplicate_groups)} duplicate groups with {analysis['duplicate_result']['stats']['centroids_in_duplicates']} centroids")
        else:
            print(f"Trajectory {trajectory_idx}: No duplicates found. Centroids: {analysis['actual_count']}, Expected: {expected_count}")
    
    return all_analyses

#------------------------------------------------------------------------------
# CONTACT ANGLE ANALYSIS
#------------------------------------------------------------------------------

def calculate_contact_angles(trajectory_points, angle_threshold=40.0):
    """Calculate curvature angles between consecutive contact segments."""
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
    
    for i in range(1, len(trajectory_points) - 1):
        p1 = trajectory_points[i-1]
        p2 = trajectory_points[i]
        p3 = trajectory_points[i+1]
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        v1_length = np.linalg.norm(v1)
        v2_length = np.linalg.norm(v2)
        
        if v1_length < 1e-6 or v2_length < 1e-6:
            curvature_angles.append(0.0)
            direction_changes.append(0.0)
            continue
        
        v1_norm = v1 / v1_length
        v2_norm = v2 / v2_length
        
        dot_product = np.dot(v1_norm, v2_norm)
        angle_between_vectors = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        
        curvature_angle = angle_between_vectors 
        curvature_angles.append(curvature_angle)
        
        direction_diff = v2_norm - v1_norm
        direction_change_magnitude = np.linalg.norm(direction_diff)
        direction_change = direction_change_magnitude * 90.0
        direction_changes.append(min(direction_change, 180.0))
        
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
    
    curvature_angles = np.array(curvature_angles)
    direction_changes = np.array(direction_changes)
    
    max_curvature = np.max(curvature_angles) if len(curvature_angles) > 0 else 0
    mean_curvature = np.mean(curvature_angles) if len(curvature_angles) > 0 else 0
    std_curvature = np.std(curvature_angles) if len(curvature_angles) > 0 else 0
    
    max_direction_change = np.max(direction_changes) if len(direction_changes) > 0 else 0
    mean_direction_change = np.mean(direction_changes) if len(direction_changes) > 0 else 0
    
    cumulative_direction_change = np.sum(direction_changes) if len(direction_changes) > 0 else 0
    
    linearity_score = max(0, 1 - (max_curvature / 180.0) * 1.5 - (mean_curvature / 60.0) * 0.5)
    linearity_score = min(1.0, linearity_score)
    
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
    """Analyze contact angles for all trajectories."""
    clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
    
    trajectory_angle_analyses = {}
    
    for traj in trajectories:
        cluster_id = traj['cluster_id']
        
        mask = clusters == cluster_id
        
        if not np.any(mask):
            print(f"Warning: No coordinates found for trajectory {cluster_id}")
            continue
            
        cluster_coords = coords_array[mask]
        
        if 'direction' in traj and len(cluster_coords) > 2:
            direction = np.array(traj['direction'])
            center = np.mean(cluster_coords, axis=0)
            projected = np.dot(cluster_coords - center, direction)
            sorted_indices = np.argsort(projected)
            sorted_coords = cluster_coords[sorted_indices]
        else:
            sorted_coords = cluster_coords
        
        angle_analysis = calculate_contact_angles(sorted_coords, angle_threshold)
        
        angle_analysis['trajectory_id'] = cluster_id
        angle_analysis['contact_count'] = len(sorted_coords)
        angle_analysis['trajectory_length'] = traj.get('length_mm', 0)
        angle_analysis['pca_linearity'] = traj.get('linearity', 0)
        
        trajectory_angle_analyses[cluster_id] = angle_analysis
    
    return trajectory_angle_analyses

def create_empty_angle_analysis(angle_threshold=40.0):
    """Create empty angle analysis for trajectories without coordinates."""
    return {
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
        'angle_threshold': angle_threshold
    }


def add_angle_analysis_to_trajectories(trajectories, coords_array, results, angle_threshold=40.0):
    """
    FIXED VERSION: Add angle analysis results directly to trajectory dictionaries.
    Now properly handles split/merged trajectories by using their stored coordinates.
    """
    print(f"Starting angle analysis for {len(trajectories)} trajectories...")
    
    for traj in trajectories:
        traj_id = traj['cluster_id']
        
        # *** CRITICAL FIX: Use trajectory's own sorted_coords if available ***
        if 'sorted_coords' in traj and traj['sorted_coords']:
            # Use the trajectory's stored coordinates (works for split/merged trajectories)
            sorted_coords = np.array(traj['sorted_coords'])
            print(f"  Using stored coordinates for trajectory {traj_id} ({len(sorted_coords)} points)")
        else:
            # Fallback: try to get coordinates from original cluster mapping
            print(f"  Trying to get coordinates for trajectory {traj_id} from cluster mapping...")
            sorted_coords = get_trajectory_coordinates(traj_id, results, coords_array, 
                                                     np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)]))
            
            if sorted_coords is None:
                print(f"  Warning: No coordinates found for trajectory {traj_id}, skipping angle analysis")
                # Add empty angle analysis
                traj['contact_angles'] = create_empty_angle_analysis(angle_threshold)
                continue
        
        # Perform angle analysis on the coordinates
        print(f"  Calculating angles for trajectory {traj_id}...")
        angle_analysis = calculate_contact_angles(sorted_coords, angle_threshold)
        
        # Store the results in the trajectory
        traj['contact_angles'] = {
            'curvature_angles': angle_analysis['curvature_angles'],
            'direction_changes': angle_analysis['direction_changes'], 
            'max_curvature': angle_analysis['max_curvature'],
            'mean_curvature': angle_analysis['mean_curvature'],
            'std_curvature': angle_analysis['std_curvature'],
            'max_direction_change': angle_analysis['max_direction_change'],
            'mean_direction_change': angle_analysis['mean_direction_change'],
            'cumulative_direction_change': angle_analysis['cumulative_direction_change'],
            'is_linear': angle_analysis['is_linear'],
            'linearity_score': angle_analysis['linearity_score'],
            'flagged_segments_count': angle_analysis['flagged_count'],
            'angle_threshold': angle_analysis['angle_threshold']
        }
        
        # Add flagged segments details if any
        if angle_analysis['flagged_segments']:
            traj['contact_angles']['flagged_segments'] = angle_analysis['flagged_segments']
        
        print(f"  ✅ Angle analysis complete for trajectory {traj_id}: max_curvature={angle_analysis['max_curvature']:.1f}°")

#------------------------------------------------------------------------------
# TRAJECTORY FEATURE EXTRACTION AND SCORING
#------------------------------------------------------------------------------

def extract_trajectory_features(trajectory, cluster_coords):
    """Extract all relevant features for ML training."""
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
    
    # Geometric features from actual coordinates
    if len(cluster_coords) > 2:
        try:
            coord_features = calculate_coordinate_features(cluster_coords)
            features.update(coord_features)
        except Exception as e:
            print(f"Warning: Could not calculate coordinate features: {e}")
            features.update({
                'bbox_size_x': 0, 'bbox_size_y': 0, 'bbox_size_z': 0,
                'bbox_volume': 0, 'contact_density': 0,
                'spread_mean': 0, 'spread_std': 0, 'spread_max': 0
            })
    else:
        features.update({
            'bbox_size_x': 0, 'bbox_size_y': 0, 'bbox_size_z': 0,
            'bbox_volume': 0, 'contact_density': 0,
            'spread_mean': 0, 'spread_std': 0, 'spread_max': 0
        })
    
    # Convert any None values to appropriate defaults
    for key, value in features.items():
        if value is None:
            if 'valid' in key or 'match' in key:
                features[key] = False
            else:
                features[key] = 0
    
    return features

def calculate_coordinate_features(coords):
    """Calculate geometric features from raw coordinates."""
    bbox_min = np.min(coords, axis=0)
    bbox_max = np.max(coords, axis=0)
    bbox_size = bbox_max - bbox_min
    
    total_length = np.linalg.norm(bbox_size)
    density = len(coords) / max(total_length, 1.0)
    
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
    """Calculate algorithmic quality score (0-100)."""
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
    if linearity is not None:
        if linearity >= 0.95:
            score += 20
        elif linearity >= 0.85:
            score += 20 * (linearity - 0.85) / 0.10 + 10
        else:
            score += 10 * max(0, linearity - 0.70) / 0.15
    
    # Spacing score (20 points)
    spacing_valid_pct = features.get('spacing_valid_pct', 0)
    if spacing_valid_pct is not None:
        if spacing_valid_pct >= 80:
            score += 20
        elif spacing_valid_pct >= 50:
            score += 20 * (spacing_valid_pct - 50) / 30
    
    # Angle score (15 points)
    max_curvature = features.get('angle_max_curvature', 0)
    if max_curvature is not None:
        if max_curvature <= 15:
            score += 15
        elif max_curvature <= 40:
            score += 15 * (40 - max_curvature) / 25
    
    # Length reasonableness (10 points)
    length = features.get('length_mm', 0)
    if length is not None:
        if 20 <= length <= 80:
            score += 10
        elif 15 <= length <= 100:
            score += 5
    
    # Additional quality metrics (10 points)
    linearity_score = features.get('angle_linearity_score', 1.0)
    if linearity_score is not None:
        score += 10 * linearity_score
    
    return min(100, max(0, score))

def calculate_trajectory_scores(trajectories, coords_array, results):
    """
    FIXED VERSION: Calculate comprehensive scores for each trajectory
    Now properly handles split/merged trajectories by using sorted_coords
    """
    trajectory_scores = []
    
    # Get cluster assignments (still needed for original trajectories)
    clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
    
    for trajectory in trajectories:
        cluster_id = trajectory['cluster_id']
        
        # *** CRITICAL FIX: Use the new coordinate retrieval function ***
        cluster_coords = get_trajectory_coordinates(cluster_id, results, coords_array, clusters)
        
        if cluster_coords is None or len(cluster_coords) == 0:
            print(f"Warning: No coordinates found for trajectory {cluster_id}, using fallback")
            # Create minimal fallback coordinates from endpoints if available
            if 'endpoints' in trajectory:
                endpoints = np.array(trajectory['endpoints'])
                if len(endpoints) == 2:
                    n_points = trajectory.get('electrode_count', 5)
                    t = np.linspace(0, 1, n_points)
                    cluster_coords = np.array([
                        endpoints[0] + t[i] * (endpoints[1] - endpoints[0])
                        for i in range(n_points)
                    ])
                else:
                    continue  # Skip this trajectory if no coordinates can be obtained
            else:
                continue
        
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

#------------------------------------------------------------------------------
# ESSENTIAL VISUALIZATION (MINIMAL)
#------------------------------------------------------------------------------


def create_interactive_3d_plot(coords_array, results, scores_df):
    """Create interactive 3D plot with trajectory scores."""
    try:
        import plotly.graph_objects as go
        
        clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
        
        fig = go.Figure()
        
        # Plot all points as background
        fig.add_trace(go.Scatter3d(
            x=coords_array[:, 0],
            y=coords_array[:, 1], 
            z=coords_array[:, 2],
            mode='markers',
            marker=dict(size=4, color='black', opacity=0.3),
            name='All Electrodes',
            hovertemplate='<b>Electrode</b><br>X: %{x:.1f} mm<br>Y: %{y:.1f} mm<br>Z: %{z:.1f} mm<extra></extra>'
        ))
        
        # Create color map based on algorithm scores
        scores_dict = dict(zip(scores_df['trajectory_id'], scores_df['algorithm_score']))
        
        # Plot each trajectory
        for _, row in scores_df.iterrows():
            traj_id = row['trajectory_id']
            score = row['algorithm_score']
            
            # Get coordinates for this trajectory
            cluster_coords = get_trajectory_coordinates(traj_id, results, coords_array, clusters)
            
            if cluster_coords is None or len(cluster_coords) == 0:
                continue
            
            # Color based on score
            if score >= 80:
                color = 'green'
                quality = 'Good'
            elif score >= 60:
                color = 'orange' 
                quality = 'OK'
            else:
                color = 'red'
                quality = 'Bad'
            
            # Create hover text
            hover_text = []
            for i, coord in enumerate(cluster_coords):
                hover_text.append(
                    f"<b>Trajectory {traj_id}</b><br>" +
                    f"Contact {i+1}/{len(cluster_coords)}<br>" +
                    f"Position: ({coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f}) mm<br>" +
                    f"Algorithm Score: {score:.0f}<br>" +
                    f"Quality: {quality}<br>" +
                    f"Contacts: {row.get('n_contacts', len(cluster_coords))}<br>" +
                    f"Linearity: {row.get('linearity_pca', 0):.3f}"
                )
            
            # Plot trajectory
            fig.add_trace(go.Scatter3d(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                z=cluster_coords[:, 2],
                mode='markers+lines',
                line=dict(color=color, width=4),
                marker=dict(size=6, color=color),
                name=f'T{traj_id} - {quality} ({score:.0f})',
                hovertemplate='%{text}<extra></extra>',
                text=hover_text,
                showlegend=True
            ))
        
        # Update layout
        fig.update_layout(
            title='Interactive 3D Trajectory Quality Scores',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1000,
            height=700
        )
        
        return fig
        
    except ImportError:
        return None

def get_trajectory_coordinates(traj_id, results, coords_array, clusters):
    """
    ENHANCED VERSION: Get coordinates for any trajectory type with better handling.
    """
    print(f"    Getting coordinates for trajectory {traj_id}...")
    
    # Method 1: Check if trajectory has sorted_coords (for merged/split trajectories)
    for traj in results.get('trajectories', []):
        if str(traj['cluster_id']) == str(traj_id):
            if 'sorted_coords' in traj and traj['sorted_coords']:
                coords = np.array(traj['sorted_coords'])
                print(f"    Found stored coordinates: {len(coords)} points")
                return coords
            break
    
    # Method 2: For regular cluster IDs, try cluster mapping
    if isinstance(traj_id, (int, np.integer)) or (isinstance(traj_id, str) and traj_id.isdigit()):
        try:
            cluster_id_int = int(traj_id)
            mask = clusters == cluster_id_int
            if np.any(mask):
                coords = coords_array[mask]
                print(f"    Found cluster coordinates: {len(coords)} points")
                return coords
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
                    coords1 = coords_array[mask1]
                    coords2 = coords_array[mask2]
                    coords = np.vstack([coords1, coords2])
                    print(f"    Reconstructed merged coordinates: {len(coords)} points")
                    return coords
        except (ValueError, TypeError):
            pass
    
    # Method 4: For split trajectories, try to reconstruct from endpoints
    for traj in results.get('trajectories', []):
        if str(traj['cluster_id']) == str(traj_id):
            if 'endpoints' in traj:
                endpoints = np.array(traj['endpoints'])
                if len(endpoints) == 2:
                    n_points = traj.get('electrode_count', 10)
                    t = np.linspace(0, 1, n_points)
                    coords = []
                    for i in range(n_points):
                        point = endpoints[0] + t[i] * (endpoints[1] - endpoints[0])
                        coords.append(point)
                    coords = np.array(coords)
                    print(f"    Reconstructed from endpoints: {len(coords)} points")
                    return coords
            break
    
    print(f"    Warning: Could not find coordinates for trajectory {traj_id}")
    return None
#------------------------------------------------------------------------------
# HTML REPORT GENERATION (ESSENTIAL OUTPUT)
#------------------------------------------------------------------------------

def create_interactive_annotation_report(scores_df, static_viz_path, html_path, interactive_viz_path=None):
    """Create an interactive HTML report for trajectory annotation."""
    has_interactive = interactive_viz_path is not None and os.path.exists(interactive_viz_path)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trajectory Analysis Report</title>
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
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Electrode Trajectory Analysis Report</h1>
            <p>Review trajectory quality and provide feedback in the CSV file.</p>
        </div>
        
        <div class="viz-container">
            <h2>3D Trajectory Visualization</h2>
            
            {"" if not has_interactive else '''
            <div class="viz-controls">
                <h3>Choose Visualization:</h3>
                <button id="staticBtn" class="active" onclick="showStaticViz()">📊 Static Plot</button>
                <button id="interactiveBtn" onclick="showInteractiveViz()">🎮 Interactive Plot</button>
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
                <li>Open the CSV file: <code>trajectory_features_for_annotation.csv</code></li>
                <li>For each trajectory, fill in the <code>feedback_label</code> column with: <strong>GOOD</strong>, <strong>BAD</strong>, or <strong>UNCERTAIN</strong></li>
                <li>Add any notes in the <code>notes</code> column</li>
                <li>Use the visualization above to help identify trajectories by their ID</li>
                <li>Focus on trajectories where algorithm score disagrees with your visual assessment</li>
            </ol>
        </div>
        
        <div class="table-container">
            <h2>Trajectory Summary</h2>
            {scores_df.to_html(classes='table', table_id='scores_table', escape=False)}
        </div>
        
        <script>
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
            
            // Add color coding to table rows
            document.addEventListener('DOMContentLoaded', function() {{
                const rows = document.querySelectorAll('#scores_table tbody tr');
                rows.forEach(row => {{
                    const scoreCell = row.cells[row.cells.length - 3];
                    const score = parseFloat(scoreCell.textContent);
                    if (score >= 80) {{
                        row.classList.add('good');
                    }} else if (score >= 60) {{
                        row.classList.add('ok');
                    }} else {{
                        row.classList.add('bad');
                    }}
                }});
                
                showStaticViz();
            }});
        </script>
    </body>
    </html>
    """
    
    with open(html_path, 'w') as f:
        f.write(html_content)

def create_basic_3d_plot(coords_array, results, scores_df, output_path):
    """Create basic 3D plot with trajectory scores."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
    
    # Plot all points in light gray
    ax.scatter(coords_array[:, 0], coords_array[:, 1], coords_array[:, 2], 
               c='lightgray', marker='.', s=10, alpha=0.3)
    
    # Plot each trajectory with color coding
    for _, row in scores_df.iterrows():
        traj_id = row['trajectory_id']
        score = row['algorithm_score']
        
        cluster_coords = get_trajectory_coordinates(traj_id, results, coords_array, clusters)
        
        if cluster_coords is None or len(cluster_coords) == 0:
            continue
        
        # Color based on score
        if score >= 80:
            color = 'green'
            marker = 'o'
        elif score >= 60:
            color = 'orange' 
            marker = 's'
        else:
            color = 'red'
            marker = '^'
        
        # Plot trajectory points
        ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1], cluster_coords[:, 2],
                  c=color, marker=marker, s=60, alpha=0.8, edgecolor='black')
        
        # Add trajectory line
        if len(cluster_coords) > 1:
            ax.plot(cluster_coords[:, 0], cluster_coords[:, 1], cluster_coords[:, 2],
                   '-', color=color, linewidth=2, alpha=0.6)
        
        # Add label
        centroid = np.mean(cluster_coords, axis=0)
        ax.text(centroid[0], centroid[1], centroid[2], 
               f'T{traj_id}\n{score:.0f}', 
               fontsize=8, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)') 
    ax.set_zlabel('Z (mm)')
    ax.set_title('Trajectory Quality Scores\n(Green=Good ≥80, Orange=OK 60-79, Red=Bad <60)')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Good (≥80)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=10, label='OK (60-79)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Bad (<60)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

#------------------------------------------------------------------------------
# HEMISPHERE SPLITTING (ESSENTIAL FEATURE)
#------------------------------------------------------------------------------

def split_trajectories_at_hemisphere_boundary(trajectories, coords_array, results, hemisphere='both'):
    """
    Split trajectories that cross the hemisphere boundary (x=0 in RAS coordinates).
    
    This function identifies trajectories that have contacts in both hemispheres
    and splits them at x=0, creating separate trajectories for each hemisphere.
    """
    import numpy as np
    from sklearn.decomposition import PCA
    
    if hemisphere.lower() != 'both':
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
    
    # Maximum existing cluster ID to ensure unique IDs
    existing_ids = [traj['cluster_id'] for traj in trajectories]
    max_id = max([int(id) if isinstance(id, (int, np.integer)) else 0 for id in existing_ids] + [0])
    id_offset = max_id + 1000
    
    for traj in trajectories:
        cluster_id = traj['cluster_id']
        
        # Get coordinates for this trajectory
        mask = clusters == cluster_id
        if not np.any(mask):
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
                spacing_validation = validate_electrode_spacing(
                    sorted_coords, 
                    traj['spacing_validation'].get('expected_range', (3.0, 5.0))
                )
                split_traj['spacing_validation'] = spacing_validation
            
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
    """Apply hemisphere splitting to trajectory results and update all related data structures."""
    if hemisphere.lower() != 'both' or 'trajectories' not in results:
        return results
    
    # Apply hemisphere splitting
    original_trajectories = results['trajectories']
    updated_trajectories, split_info = split_trajectories_at_hemisphere_boundary(
        original_trajectories, coords_array, results, hemisphere
    )
    
    # Update results
    if split_info['splits_performed'] > 0:
        results['original_trajectories_before_hemisphere_split'] = original_trajectories
        results['trajectories'] = updated_trajectories
        results['n_trajectories'] = len(updated_trajectories)
        results['hemisphere_splitting'] = split_info
        
        print(f"✅ Applied hemisphere splitting to {split_info['splits_performed']} trajectories")
    
    return results

#------------------------------------------------------------------------------
# HEMISPHERE ANALYSIS (SIMPLIFIED)
#------------------------------------------------------------------------------

def analyze_both_hemispheres_separately(coords_array, entry_points=None, 
                                       max_neighbor_distance=7.5, min_neighbors=3,
                                       expected_spacing_range=(3.0, 5.0),
                                       use_adaptive_clustering=False, 
                                       expected_contact_counts=[5, 8, 10, 12, 15, 18]):
    """
    CORRECTED VERSION: Analyze both hemispheres separately and combine results.
    
    This function:
    1. Splits coordinates by hemisphere  
    2. Runs trajectory analysis on each hemisphere independently
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
            parameter_search = adaptive_clustering_parameters(
                left_coords, 
                expected_contact_counts=expected_contact_counts
            )
            left_results = integrated_trajectory_analysis(
                coords_array=left_coords,
                entry_points=left_entry,
                max_neighbor_distance=parameter_search['optimal_eps'],
                min_neighbors=parameter_search['optimal_min_neighbors'],
                expected_spacing_range=expected_spacing_range
            )
            left_results['parameter_search'] = parameter_search
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
            parameter_search = adaptive_clustering_parameters(
                right_coords,
                expected_contact_counts=expected_contact_counts
            )
            right_results = integrated_trajectory_analysis(
                coords_array=right_coords,
                entry_points=right_entry,
                max_neighbor_distance=parameter_search['optimal_eps'],
                min_neighbors=parameter_search['optimal_min_neighbors'],
                expected_spacing_range=expected_spacing_range
            )
            right_results['parameter_search'] = parameter_search
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
    combined_results = combine_hemisphere_results(
        left_results, right_results, left_coords, right_coords, 
        left_mask, right_mask, coords_array
    )
    
    print(f"Combined analysis: {combined_results['n_trajectories']} total trajectories")
    return combined_results

def combine_hemisphere_results(left_results, right_results, 
                              left_coords, right_coords,
                              left_mask, right_mask, original_coords):
    """
    CORRECTED VERSION: Combine hemisphere analysis results into unified format.
    """
    import networkx as nx
    
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
    
    # Create unified graph
    G = nx.Graph()
    
    for i, coord in enumerate(original_coords):
        hemisphere = 'left' if original_coords[i, 0] < 0 else 'right'
        G.add_node(i, pos=coord, hemisphere=hemisphere, dbscan_cluster=-1)
    
    trajectory_id_counter = 0
    all_trajectories = []
    
    # Process hemisphere results
    for hemi_results, hemi_coords, hemi_mask, hemi_name in [
        (left_results, left_coords, left_mask, 'left'),
        (right_results, right_coords, right_mask, 'right')
    ]:
        if hemi_results and 'trajectories' in hemi_results:
            hemi_trajectories = process_hemisphere_trajectories(
                hemi_results, hemi_coords, hemi_mask, original_coords, 
                hemi_name, trajectory_id_counter, G
            )
            all_trajectories.extend(hemi_trajectories)
            trajectory_id_counter += len(hemi_trajectories)
            
            # Update combined stats
            combined_results['dbscan']['n_clusters'] += hemi_results['dbscan']['n_clusters']
            combined_results['dbscan']['noise_points'] += hemi_results['dbscan']['noise_points']
            combined_results['dbscan']['cluster_sizes'].extend(hemi_results['dbscan']['cluster_sizes'])
    
    combined_results['trajectories'] = all_trajectories
    combined_results['n_trajectories'] = len(all_trajectories)
    combined_results['graph'] = G
    
    return combined_results

def process_hemisphere_trajectories(hemisphere_results, hemisphere_coords, 
                                  hemisphere_mask, original_coords, hemisphere_name,
                                  id_offset, unified_graph):
    """
    CORRECTED VERSION: Process trajectories from one hemisphere and map back to original coordinate space.
    """
    trajectories = []
    
    if 'graph' not in hemisphere_results:
        return trajectories
    
    original_indices = np.where(hemisphere_mask)[0]
    
    for traj in hemisphere_results['trajectories']:
        new_traj = traj.copy()
        new_traj['cluster_id'] = id_offset + traj['cluster_id']
        new_traj['hemisphere'] = hemisphere_name
        new_traj['original_hemisphere_id'] = traj['cluster_id']
        
        # Update graph nodes
        hemisphere_graph = hemisphere_results['graph']
        hemisphere_cluster_id = traj['cluster_id']
        
        for node_data in hemisphere_graph.nodes(data=True):
            node_id, node_attrs = node_data
            if node_attrs.get('dbscan_cluster') == hemisphere_cluster_id:
                original_node_id = original_indices[node_id]
                unified_graph.nodes[original_node_id]['dbscan_cluster'] = new_traj['cluster_id']
        
        trajectories.append(new_traj)
    
    return trajectories



#------------------------------------------------------------------------------
# MAIN STREAMLINED FUNCTION
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
    FIXED VERSION: Enhanced main function with correct angle analysis order.
    
    Order: Clustering → Refinement → Angle Analysis → Scoring
    This ensures split/merged trajectories get proper angle analysis before scoring.
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
        electrodes_volume = slicer.util.getNode('P5_electrode_mask_success_1')
        brain_volume = slicer.util.getNode("patient5_mask_5")
        
        # Create output directories
        base_dir = r"C:\Users\rocia\Downloads\TFG\Cohort\Centroids\P5_streamlined_results_4"
        
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
                'hemisphere': hemisphere
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
        
        
        # Step 4: Get entry points if available and filter by hemisphere
        entry_points = None
        entry_points_volume = slicer.util.getNode('P5_brain_entry_points')
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
                        max_neighbor_distance=8,
                        min_neighbors=3,
                        expected_spacing_range=expected_spacing_range if validate_spacing else None,
                        use_adaptive_clustering=True,
                        expected_contact_counts=expected_contact_counts,
                    )
                else:
                    # Use regular adaptive analysis
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
                # Single hemisphere adaptive analysis
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
                        use_adaptive_clustering=False
                    )
                else:
                    # Use regular analysis
                    integrated_results = integrated_trajectory_analysis(
                        coords_array=coords_array,
                        entry_points=entry_points,
                        max_neighbor_distance=7.5,
                        min_neighbors=3,
                        expected_spacing_range=expected_spacing_range if validate_spacing else None
                    )
            else:
                # Single hemisphere analysis
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

        # *** CRITICAL FIX: Ensure all trajectories have sorted_coords BEFORE refinement ***
        print("Ensuring all trajectories have sorted coordinates...")
        clusters = np.array([node[1]['dbscan_cluster'] for node in integrated_results['graph'].nodes(data=True)])

        for traj in integrated_results.get('trajectories', []):
            cluster_id = traj['cluster_id']
            
            # Only process original trajectories (not already split/merged ones)
            if 'sorted_coords' not in traj or not traj['sorted_coords']:
                # Get coordinates for this trajectory from the graph
                mask = clusters == cluster_id
                
                if np.sum(mask) > 0:
                    cluster_coords = coords_array[mask]
                    
                    # Sort contacts along trajectory direction
                    if 'direction' in traj and len(cluster_coords) > 2:
                        direction = np.array(traj['direction'])
                        center = np.mean(cluster_coords, axis=0)
                        projected = np.dot(cluster_coords - center, direction)
                        sorted_indices = np.argsort(projected)
                        sorted_coords = cluster_coords[sorted_indices]
                    else:
                        sorted_coords = cluster_coords
                    
                    # Store sorted coordinates for later use
                    traj['sorted_coords'] = sorted_coords.tolist()
                    print(f"✅ Added sorted_coords to trajectory {cluster_id} ({len(sorted_coords)} points)")
                else:
                    print(f"⚠️  Warning: No coordinates found for trajectory {cluster_id}")

        print(f"Coordinate storage complete. Ready for refinement.")
        
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
            
            # Apply targeted trajectory refinement WITH FIXED FUNCTIONS
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

        # *** STEP 7: CONTACT ANGLE ANALYSIS (AFTER REFINEMENT, BEFORE SCORING) ***
        if analyze_contact_angles:
            print("Analyzing contact angles (after refinement, before scoring)...")
            
            # Use the enhanced function that properly handles split/merged trajectories
            integrated_results = ensure_angle_analysis_after_refinement(
                integrated_results, 
                coords_array, 
                analyze_contact_angles=True, 
                angle_threshold=angle_threshold
            )
            
            # Get the angle analyses for summary
            trajectory_angle_analyses = integrated_results.get('contact_angle_analysis', {})
            
            # Store in all_results for later use
            all_results['contact_angle_analysis'] = trajectory_angle_analyses
            
            # Print detailed summary
            print_angle_analysis_summary(trajectory_angle_analyses)
            
            flagged_count = sum(1 for analysis in trajectory_angle_analyses.values() 
                            if not analysis['is_linear'])
            total_count = len(trajectory_angle_analyses)
            
            print(f"Contact angles: {flagged_count}/{total_count} trajectories flagged for non-linearity")
            
            # Create minimal angle visualization if reports enabled
            if use_original_reports:
                angle_fig = create_angle_analysis_visualization(trajectory_angle_analyses, output_dir)
                if angle_fig:
                    plt.close(angle_fig)
        
        # *** STEP 8: GENERATE FINAL SCORING (AFTER ANGLE ANALYSIS) ***
        if use_original_reports or create_interactive_annotation:
            print("Creating final trajectory scoring report (with complete angle data)...")
            try:
                # Calculate trajectory scores - now has complete angle data including split/merged trajectories
                scores_df = calculate_trajectory_scores(integrated_results['trajectories'], coords_array, integrated_results)
                
                # Save scoring table
                csv_path = os.path.join(output_dir, 'trajectory_scores_for_annotation.csv')
                scores_df.to_csv(csv_path, index=False)
                print(f"✅ Trajectory scores saved to: {csv_path}")
                
                # Create 3D visualization with IDs and scores
                static_fig = create_scored_3d_visualization(coords_array, integrated_results, scores_df)
                
                # Save static visualization
                static_viz_path = os.path.join(output_dir, 'trajectory_scores_3d_visualization.png')
                static_fig.savefig(static_viz_path, dpi=300, bbox_inches='tight')
                plt.close(static_fig)
                
                # Create interactive visualization if requested
                interactive_viz_path = None
                if create_interactive_annotation:
                    try:
                        # Create interactive 3D plot
                        interactive_fig = create_interactive_scored_3d_visualization(coords_array, integrated_results, scores_df)
                        
                        if interactive_fig is not None:
                            interactive_viz_path = os.path.join(output_dir, 'trajectory_scores_3d_interactive.html')
                            interactive_fig.write_html(interactive_viz_path)
                            print(f"✅ Interactive 3D visualization saved to: {interactive_viz_path}")
                        
                    except ImportError:
                        print("⚠️ Plotly not available. Creating static visualization only.")
                        create_interactive_annotation = False
                    except Exception as e:
                        print(f"⚠️ Error creating interactive visualization: {e}")
                        create_interactive_annotation = False
                
                # Create HTML report
                html_path = os.path.join(output_dir, 'trajectory_annotation_report.html')
                create_interactive_annotation_report(
                    scores_df, 
                    static_viz_path, 
                    html_path,
                    interactive_viz_path if create_interactive_annotation else None
                )
                
                print(f"✅ Static 3D visualization saved to: {static_viz_path}")
                print(f"✅ Interactive report saved to: {html_path}")
                
                # Print summary
                print(f"\n=== FINAL SCORING SUMMARY ===")
                print(f"Total trajectories: {len(scores_df)}")
                print(f"High quality (≥80): {len(scores_df[scores_df['algorithm_score'] >= 80])}")
                print(f"Medium quality (60-79): {len(scores_df[(scores_df['algorithm_score'] >= 60) & (scores_df['algorithm_score'] < 80)])}")
                print(f"Low quality (<60): {len(scores_df[scores_df['algorithm_score'] < 60])}")
                print(f"Mean algorithm score: {scores_df['algorithm_score'].mean():.2f}")
                
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
        
        # Report execution time
        finish_time = time.time()
        execution_time = finish_time - start_time
        all_results['execution_time'] = execution_time
        
        print(f"\nAnalysis Summary:")
        if hemisphere.lower() != 'both':
            hemisphere_info = all_results['hemisphere_filtering']
            print(f"- Hemisphere filtering ({hemisphere}): {hemisphere_info['filtered_count']} of {hemisphere_info['original_count']} "
                  f"coordinates ({hemisphere_info['filtering_efficiency']:.1f}%)")
            print(f"- Discarded coordinates: {hemisphere_info['discarded_count']}")
        
        print(f"- Analyzed {len(coords_array)} electrode coordinates")
        
        # Add trajectory refinement summary if performed
        if refine_trajectories and 'trajectory_refinement' in integrated_results:
            refinement = integrated_results['trajectory_refinement']
            print(f"- Trajectory refinement: {refinement['original_count']} original -> {refinement['n_trajectories']} final trajectories")
            print(f"- Merged trajectories: {refinement['merged_count']}")
            print(f"- Split trajectories: {refinement['split_count']}")
        
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

# Example usage with the fixed main function
if __name__ == "__main__":
    results = main(
        use_combined_volume=False,
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
        hemisphere='both',
        analyze_contact_angles=True,
        angle_threshold=40.0,
        create_plotly_visualization=True,
        create_interactive_annotation=True,
    )
    print("Enhanced analysis completed with proper trajectory refinement and feature extraction.")

# Simple wrapper function for your exact use case
def analyze_with_custom_volumes(electrode_vol, brain_vol, entry_vol=None, output_path=None):
    """
    Simple wrapper function to run analysis with custom volume names.
    
    Args:
        electrode_vol (str): Name of electrode volume in Slicer
        brain_vol (str): Name of brain volume in Slicer
        entry_vol (str, optional): Name of entry points volume in Slicer
        output_path (str, optional): Custom output directory
    
    Returns:
        dict: Analysis results
    """
    return main(
        electrode_volume_name=electrode_vol,
        brain_volume_name=brain_vol,
        entry_points_volume_name=entry_vol,
        use_adaptive_clustering=True,
        detect_duplicates=True,
        duplicate_threshold=0.5,
        validate_spacing=True,
        expected_spacing_range=(3.0, 5.0),
        refine_trajectories=True,
        analyze_contact_angles=True,
        angle_threshold=40.0,
        hemisphere='both',
        output_dir=output_path
    )

#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Electrode_path\adaptive_clustering.py').read())