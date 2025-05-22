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
import networkx as nx
from skimage.measure import label, regionprops_table
from scipy.spatial.distance import cdist
from collections import defaultdict
from Outermost_centroids_coordinates.outermost_centroids_vol_slicer import (
    get_ras_coordinates_from_ijk, get_array_from_volume, calculate_centroids_numpy,
    get_centroids_ras, get_surface_from_volume, convert_surface_vertices_to_ras, filter_centroids_by_surface_distance
)
from End_points.midplane_prueba import get_all_centroids
from Electrode_path.construction_4 import (create_summary_page, create_3d_visualization,
    create_trajectory_details_page, create_noise_points_page)
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import pandas as pd
import time

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return min(zs)

def calculate_angles(direction):
    """Calculate angles between direction vector and principal axes"""
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

def integrated_trajectory_analysis(coords_array, entry_points=None, max_neighbor_distance=10, min_neighbors=3):
    results = {
        'dbscan': {},
        'louvain': {},
        'combined': {},
        'parameters': {
            'max_neighbor_distance': max_neighbor_distance,
            'min_neighbors': min_neighbors,
            'n_electrodes': len(coords_array)
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
    
    # Combined analysis
    if 'error' not in results['louvain']:
        cluster_community_mapping = defaultdict(set)
        for node in G.nodes:
            dbscan_cluster = G.nodes[node]['dbscan_cluster']
            louvain_community = G.nodes[node]['louvain_community']
            if dbscan_cluster != -1:  
                cluster_community_mapping[dbscan_cluster].add(louvain_community)
        
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
    
    # Trajectory analysis with enhanced PCA and angle calculations
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
            
            # Spline fitting
            spline_points = None
            if len(sorted_coords) > 2:
                try:
                    tck, u = splprep(sorted_coords.T, s=0)
                    u_new = np.linspace(0, 1, 50)
                    spline_points = np.array(splev(u_new, tck)).T
                except:
                    pass
            
            trajectories.append({
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
                "angles_with_axes": angles,  # Added angle information
                "pca_variance": pca.explained_variance_ratio_.tolist()  # Added PCA variance
            })
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

##############################################################
##### Bolt head and entry points dire ction analysis #########
###############################################################

def analyze_bolt_head_directions(bolt_entry_mask_volume):
    """
    Analyzes the bolt head and entry point combined mask to determine direction vectors.
    
    This function takes a combined mask of bolt heads and entry points, identifies connected
    components, and calculates the direction vector for each bolt-entry pair.
    
    Args:
        bolt_entry_mask_volume: The 3D volume node containing the combined mask of bolt heads and entry points
        
    Returns:
        List of dictionaries containing:
            - start_point: The bolt head position (RAS coordinates)
            - end_point: The entry point position (RAS coordinates)
            - direction: The normalized direction vector
            - length: The length of the vector in mm
    """
    import numpy as np
    from skimage.measure import label, regionprops
    
    # Get array from volume
    bolt_entry_array = get_array_from_volume(bolt_entry_mask_volume)
    
    # Label connected components
    labeled_array = label(bolt_entry_array)
    props = regionprops(labeled_array)
    
    trajectories = []
    
    # For each labeled region, try to determine if it's a bolt-entry path
    for prop in props:
        # Get coordinates of all pixels in this region
        coords = prop.coords  # in IJK space
        
        # Convert coordinates to RAS
        ras_coords = []
        for coord in coords:
            ras = get_ras_coordinates_from_ijk(bolt_entry_mask_volume, coord)
            ras_coords.append(ras)
        
        ras_coords = np.array(ras_coords)
        
        # If we don't have enough points to define a direction, skip
        if len(ras_coords) < 2:
            continue
        
        # Perform PCA to find primary direction of this component
        pca = PCA(n_components=3)
        pca.fit(ras_coords)
        
        # Check if this component is linear enough to be a trajectory
        linearity = pca.explained_variance_ratio_[0]
        
        # Skip if not linear enough (you may want to adjust this threshold)
        if linearity < 0.5:
            continue
        
        # Get direction vector (first principal component)
        direction = pca.components_[0]
        
        # Project points onto the direction vector
        mean_point = pca.mean_
        projected = np.dot(ras_coords - mean_point, direction)
        
        # Get extreme points (bolt head should be outside, entry point inside)
        sorted_indices = np.argsort(projected)
        extremes = ras_coords[sorted_indices]
        
        # By convention, bolt head is outer point, entry point is inner point
        bolt_head = extremes[0]  # outermost point
        entry_point = extremes[-1]  # innermost point
        
        # Calculate length
        path_length = np.linalg.norm(entry_point - bolt_head)
        
        # Calculate angles with principal axes
        angles = calculate_angles(direction)
        
        trajectories.append({
            "bolt_head": bolt_head.tolist(),
            "entry_point": entry_point.tolist(),
            "direction": direction.tolist(),
            "length_mm": float(path_length),
            "linearity": float(linearity),
            "angles_with_axes": angles
        })
    
    return trajectories

def visualize_bolt_trajectories(trajectories, ax=None):
    """
    Visualizes the bolt head to entry point trajectories.
    
    Args:
        trajectories: List of trajectory dictionaries returned by analyze_bolt_head_directions
        ax: Optional matplotlib 3D axis to plot on
        
    Returns:
        Matplotlib figure if ax was not provided
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create figure if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        return_fig = True
    else:
        return_fig = False
    
    # Plot each trajectory
    for i, traj in enumerate(trajectories):
        bolt_head = np.array(traj['bolt_head'])
        entry_point = np.array(traj['entry_point'])
        
        # Plot line from bolt head to entry point
        ax.plot([bolt_head[0], entry_point[0]],
               [bolt_head[1], entry_point[1]],
               [bolt_head[2], entry_point[2]],
               'g-', linewidth=2, alpha=0.7)
        
        # Plot bolt head point (bigger, square marker)
        ax.scatter(bolt_head[0], bolt_head[1], bolt_head[2], 
                  color='blue', marker='s', s=100, label='Bolt Head' if i == 0 else "")
        
        # Plot entry point (smaller, round marker)
        ax.scatter(entry_point[0], entry_point[1], entry_point[2], 
                  color='red', marker='o', s=80, label='Entry Point' if i == 0 else "")
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Bolt Head to Entry Point Trajectories')
    
    if len(trajectories) > 0:
        ax.legend()
    
    if return_fig:
        plt.tight_layout()
        return fig


######################################################################
##### Visualization of combined results in a PDF report ############
#####################################################################


def visualize_combined_results(coords_array, results, output_dir=None):
    if output_dir:
        pdf_path = os.path.join(output_dir, 'trajectory_analysis_report.pdf')
        with PdfPages(pdf_path) as pdf:
            # Create summary page
            fig = create_summary_page(results)
            pdf.savefig(fig)
            plt.close(fig)
            
            # Create 3D visualization page
            fig = create_3d_visualization(coords_array, results)
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
    else:
        # Interactive mode - show all plots
        fig = create_summary_page(results)
        plt.show()
        
        fig = create_3d_visualization(coords_array, results)
        plt.show()
        
        if 'trajectories' in results:
            fig = create_trajectory_details_page(results)
            plt.show()
            
        fig = create_pca_angle_analysis_page(results)
        plt.show()
            
        fig = create_noise_points_page(coords_array, results)
        plt.show()

def create_pca_angle_analysis_page(results):
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('PCA and Direction Angle Analysis', fontsize=16)
    
    if not results.get('trajectories'):
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5, 'No trajectories detected', ha='center', va='center')
        return fig
    
    # Create subplots
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    # Plot PCA variance ratios
    variances = [t['pca_variance'] for t in results['trajectories']]
    labels = [f"Traj {t['cluster_id']}" for t in results['trajectories']]
    
    for i, var in enumerate(variances):
        ax1.bar(i, var[0], color='b', alpha=0.6, label='PC1' if i == 0 else "")
        ax1.bar(i, var[1], bottom=var[0], color='g', alpha=0.6, label='PC2' if i == 0 else "")
        ax1.bar(i, var[2], bottom=var[0]+var[1], color='r', alpha=0.6, label='PC3' if i == 0 else "")
    
    ax1.set_xticks(range(len(variances)))
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('PCA Explained Variance by Trajectory')
    ax1.legend()
    
    # Plot angles with principal axes
    angles_x = [t['angles_with_axes']['X'] for t in results['trajectories']]
    angles_y = [t['angles_with_axes']['Y'] for t in results['trajectories']]
    angles_z = [t['angles_with_axes']['Z'] for t in results['trajectories']]
    
    x_pos = np.arange(len(angles_x))
    width = 0.25
    
    ax2.bar(x_pos - width, angles_x, width, label='X-axis', alpha=0.6)
    ax2.bar(x_pos, angles_y, width, label='Y-axis', alpha=0.6)
    ax2.bar(x_pos + width, angles_z, width, label='Z-axis', alpha=0.6)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_title('Trajectory Angles with Principal Axes')
    ax2.legend()
    
    # Plot linearity vs angle with Z-axis
    linearities = [t['linearity'] for t in results['trajectories']]
    ax3.scatter(angles_z, linearities, c=range(len(angles_z)), cmap='viridis')
    for i, txt in enumerate(labels):
        ax3.annotate(txt, (angles_z[i], linearities[i]), fontsize=8)
    
    ax3.set_xlabel('Angle with Z-axis (degrees)')
    ax3.set_ylabel('Linearity (PC1 variance ratio)')
    ax3.set_title('Linearity vs Z-axis Angle')
    
    # Plot direction vectors in 3D
    ax4 = fig.add_subplot(224, projection='3d')
    for t in results['trajectories']:
        direction = np.array(t['direction'])
        center = np.array(t['center'])
        
        ax4.quiver(center[0], center[1], center[2],
                  direction[0], direction[1], direction[2],
                  length=10, normalize=True,
                  color=plt.cm.viridis(t['linearity']),
                  alpha=0.7)
    
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('Trajectory Directions (color by linearity)')
    
    plt.tight_layout()
    return fig

# [Rest of your existing functions remain unchanged: create_summary_page, create_3d_visualization, 
# create_trajectory_details_page, create_noise_points_page, main]

def create_summary_page(results):
    fig = plt.figure(figsize=(12, 15))
    fig.suptitle('Trajectory Analysis Summary Report', fontsize=16, y=0.98)
    
    # Create grid layout
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1, 1, 1, 2])
    
    # Parameters section
    ax1 = fig.add_subplot(gs[0])
    ax1.axis('off')
    params_text = "Analysis Parameters:\n"
    params_text += f"- Max neighbor distance: {results['parameters']['max_neighbor_distance']} mm\n"
    params_text += f"- Min neighbors: {results['parameters']['min_neighbors']}\n"
    params_text += f"- Total electrodes: {results['parameters']['n_electrodes']}\n\n"
    
    params_text += "DBSCAN Results:\n"
    params_text += f"- Number of clusters: {results['dbscan']['n_clusters']}\n"
    params_text += f"- Noise points: {results['dbscan']['noise_points']}\n"
    params_text += f"- Cluster sizes: {results['dbscan']['cluster_sizes']}\n\n"
    
    if 'error' not in results['louvain']:
        params_text += "Louvain Community Detection:\n"
        params_text += f"- Number of communities: {results['louvain']['n_communities']}\n"
        params_text += f"- Modularity score: {results['louvain']['modularity']:.3f}\n"
        params_text += f"- Community sizes: {results['louvain']['community_sizes']}\n\n"
    
    params_text += f"Trajectories Detected: {results['n_trajectories']}"
    
    ax1.text(0.05, 0.95, params_text, ha='left', va='top', fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Trajectory metrics table
    if 'trajectories' in results and len(results['trajectories']) > 0:
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        
        # Prepare data for table
        table_data = []
        columns = ['ID', 'Electrodes', 'Length (mm)', 'Linearity', 'Avg Spacing (mm)', 'Spacing Var', 'Angle Z']
        
        for traj in results['trajectories']:
            row = [
                traj['cluster_id'],
                traj['electrode_count'],
                f"{traj['length_mm']:.1f}",
                f"{traj['linearity']:.2f}",
                f"{traj['avg_spacing_mm']:.2f}" if traj['avg_spacing_mm'] else 'N/A',
                f"{traj['spacing_regularity']:.2f}" if traj['spacing_regularity'] else 'N/A',
                f"{traj['angles_with_axes']['Z']:.1f}°"
            ]
            table_data.append(row)
        
        table = ax2.table(cellText=table_data, colLabels=columns, 
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax2.set_title('Trajectory Metrics', pad=20)
    
    # Cluster-community mapping
    if 'combined' in results and 'dbscan_to_louvain_mapping' in results['combined']:
        ax3 = fig.add_subplot(gs[2])
        ax3.axis('off')
        
        mapping_text = "Cluster to Community Mapping:\n\n"
        for cluster, community in results['combined']['dbscan_to_louvain_mapping'].items():
            mapping_text += f"Cluster {cluster} → Community {community}\n"
        
        if 'avg_cluster_purity' in results['combined']:
            mapping_text += f"\nAverage Cluster Purity: {results['combined']['avg_cluster_purity']:.2f}"
        
        ax3.text(0.05, 0.95, mapping_text, ha='left', va='top', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))
        ax3.set_title('Cluster-Community Relationships', pad=10)
    
    plt.tight_layout()
    return fig

def create_3d_visualization(coords_array, results):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get data for plotting
    clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
    unique_clusters = set(clusters)
    
    # Create colormaps
    cluster_cmap = plt.colormaps['tab20'].resampled(len(unique_clusters))
    community_cmap = plt.colormaps['gist_ncar'].resampled(results['louvain']['n_communities'])
    
    # Plot electrodes with cluster colors
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue  # Noise points will be plotted separately
        mask = clusters == cluster_id
        ax.scatter(coords_array[mask, 0], coords_array[mask, 1], coords_array[mask, 2], 
                  c=[cluster_cmap(cluster_id)], label=f'Cluster {cluster_id}', s=80, alpha=0.8)
    
    # Plot trajectories with enhanced features
    for traj in results.get('trajectories', []):
        color = cluster_cmap(traj['cluster_id'])
        
        # Plot spline if available, otherwise line
        if traj['spline_points'] is not None:
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
        if traj['entry_point'] is not None:
            entry = np.array(traj['entry_point'])
            ax.scatter(entry[0], entry[1], entry[2], 
                      c='red', marker='*', s=300, edgecolor='black', 
                      label=f'Entry {traj["cluster_id"]}')
            
########################################################################
##################################################################
        if traj['entry_point'] is not None:
            entry = np.array(traj['entry_point'])
            first_contact = np.array(traj['endpoints'][0])  # Assuming endpoints[0] is the first contact
            ax.plot([entry[0], first_contact[0]],
                [entry[1], first_contact[1]],
                [entry[2], first_contact[2]], 
                '--', color='red', linewidth=2, alpha=0.7)

########################################################################
    
    # Plot noise points
    if 'noise_points_coords' in results['dbscan'] and len(results['dbscan']['noise_points_coords']) > 0:
        noise_coords = np.array(results['dbscan']['noise_points_coords'])
        ax.scatter(noise_coords[:,0], noise_coords[:,1], noise_coords[:,2],
                  c='black', marker='x', s=100, label='Noise points (DBSCAN -1)')
    
    # Add legend and labels
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Electrode Trajectory Analysis\n(Colors=Clusters, Stars=Entry Points, Arrows=Directions, X=Noise)')
    
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

def create_trajectory_details_page(results):
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Trajectory Details', fontsize=16)
    
    if not results.get('trajectories'):
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5, 'No trajectories detected', ha='center', va='center')
        return fig
    
    # Create table with detailed trajectory information
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    columns = ['ID', 'Community', 'Electrodes', 'Length', 'Linearity', 
              'Avg Spacing', 'Spacing Var', 'Angle X', 'Angle Y', 'Angle Z', 'Entry']
    
    table_data = []
    for traj in results['trajectories']:
        has_entry = traj['entry_point'] is not None
        entry_text = "Yes" if has_entry else "No"
        
        row = [
            traj['cluster_id'],
            traj['louvain_community'] if traj['louvain_community'] is not None else 'N/A',
            traj['electrode_count'],
            f"{traj['length_mm']:.1f}",
            f"{traj['linearity']:.2f}",
            f"{traj['avg_spacing_mm']:.2f}" if traj['avg_spacing_mm'] else 'N/A',
            f"{traj['spacing_regularity']:.2f}" if traj['spacing_regularity'] else 'N/A',
            f"{traj['angles_with_axes']['X']:.1f}°",
            f"{traj['angles_with_axes']['Y']:.1f}°",
            f"{traj['angles_with_axes']['Z']:.1f}°",
            entry_text
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=columns, 
                    loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.tight_layout()
    return fig

def create_noise_points_page(coords_array, results):
    fig = plt.figure(figsize=(12, 8))
    
    if 'noise_points_coords' not in results['dbscan'] or len(results['dbscan']['noise_points_coords']) == 0:
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5, 'No noise points detected (DBSCAN cluster -1)', ha='center', va='center')
        return fig
    
    noise_coords = np.array(results['dbscan']['noise_points_coords'])
    noise_indices = results['dbscan']['noise_points_indices']
    
    # Create 3D plot of noise points
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(noise_coords[:,0], noise_coords[:,1], noise_coords[:,2], 
               c='red', marker='x', s=100)
    
    # Plot all other points in background for context
    all_indices = set(range(len(coords_array)))
    non_noise_indices = list(all_indices - set(noise_indices))
    if non_noise_indices:
        ax1.scatter(coords_array[non_noise_indices,0], coords_array[non_noise_indices,1], 
                   coords_array[non_noise_indices,2], c='gray', alpha=0.2, s=30)
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'Noise Points (n={len(noise_coords)})\nDBSCAN cluster -1')
    
    # Create table with noise point coordinates
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    
    # Sample some points if there are too many
    sample_size = min(10, len(noise_coords))
    sampled_coords = noise_coords[:sample_size]
    sampled_indices = noise_indices[:sample_size]
    
    table_data = []
    for idx, coord in zip(sampled_indices, sampled_coords):
        table_data.append([idx, f"{coord[0]:.1f}", f"{coord[1]:.1f}", f"{coord[2]:.1f}"])
    
    table = ax2.table(cellText=table_data, 
                     colLabels=['Index', 'X', 'Y', 'Z'], 
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax2.set_title('Noise Point Coordinates (sample)')
    
    if len(noise_coords) > sample_size:
        ax2.text(0.5, 0.05, 
                f"Showing {sample_size} of {len(noise_coords)} noise points",
                ha='center', va='center', transform=ax2.transAxes)
    
    fig.suptitle(f'Noise Points Analysis (DBSCAN cluster -1)\nTotal noise points: {len(noise_coords)}', y=0.98)
    plt.tight_layout()
    return fig

def main():
    try:
        start_time = time.time()
        electrodes_volume = slicer.util.getNode('electrode_mask_success')
        entry_points_volume = slicer.util.getNode('EntryPointsMask')
        # New: Get the combined bolt head and entry points mask
        bolt_entry_mask_volume = slicer.util.getNode('CombinedBoltHeadEntryPointsMask')
        
        output_dir = r"C:\Users\rocia\Downloads\TFG\Cohort\Centroids\p1_Trajectories_16_05_enhan_pdf_17_05\output_plots"
        os.makedirs(output_dir, exist_ok=True)
        print("Starting enhanced trajectory analysis...")
        
        # Get centroids for electrodes
        centroids_ras = get_all_centroids(electrodes_volume)
        
        if not centroids_ras:
            logging.error("No centroids found.")
            return
        
        coords_array = np.array(list(centroids_ras.values()))
        
        # Get entry points if available
        entry_points = None
        if entry_points_volume:
            entry_centroids_ras = get_all_centroids(entry_points_volume)
            if entry_centroids_ras:
                entry_points = np.array(list(entry_centroids_ras.values()))
        
        # NEW: Analyze bolt head to entry point directions
        bolt_trajectories = None
        if bolt_entry_mask_volume:
            try:
                print("Analyzing bolt head to entry point directions...")
                bolt_trajectories = analyze_bolt_head_directions(bolt_entry_mask_volume)
                print(f"Found {len(bolt_trajectories)} bolt-entry trajectories")
                
                # Create a standalone visualization of bolt trajectories
                bolt_fig = visualize_bolt_trajectories(bolt_trajectories)
                bolt_fig_path = os.path.join(output_dir, 'bolt_entry_trajectories.png')
                bolt_fig.savefig(bolt_fig_path)
                plt.close(bolt_fig)
                print(f"Bolt trajectory visualization saved to {bolt_fig_path}")
            except Exception as e:
                logging.error(f"Bolt trajectory analysis failed: {str(e)}")
                bolt_trajectories = None
        
        # Run the standard electrode trajectory analysis
        results = integrated_trajectory_analysis(
            coords_array=coords_array,
            entry_points=entry_points,
            max_neighbor_distance=7.5,
            min_neighbors=3
        )
        
        print(f"Analysis complete: {results['n_trajectories']} trajectories detected.")
        
        # NEW: Add bolt trajectories to the 3D visualization
        if bolt_trajectories:
            # Create a combined visualization
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot the electrode trajectories
            for traj in results.get('trajectories', []):
                color = plt.cm.tab20(traj['cluster_id'])
                
                # Plot the electrode path
                if traj['spline_points'] is not None:
                    sp = np.array(traj['spline_points'])
                    ax.plot(sp[:,0], sp[:,1], sp[:,2], '-', color=color, linewidth=3, alpha=0.7)
                else:
                    endpoints = traj['endpoints']
                    ax.plot([endpoints[0][0], endpoints[1][0]],
                           [endpoints[0][1], endpoints[1][1]],
                           [endpoints[0][2], endpoints[1][2]], 
                           '-', color=color, linewidth=3, alpha=0.7)
                
                # Add electrode points
                cluster_mask = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)]) == traj['cluster_id']
                ax.scatter(coords_array[cluster_mask, 0], coords_array[cluster_mask, 1], coords_array[cluster_mask, 2], 
                           c=[color], s=80, alpha=0.8)
                
            # Now add the bolt trajectories
            for traj in bolt_trajectories:
                bolt_head = np.array(traj['bolt_head'])
                entry_point = np.array(traj['entry_point'])
                
                # Plot line from bolt head to entry point
                ax.plot([bolt_head[0], entry_point[0]],
                       [bolt_head[1], entry_point[1]],
                       [bolt_head[2], entry_point[2]],
                       'g-', linewidth=2, alpha=0.7)
                
                # Plot bolt head point
                ax.scatter(bolt_head[0], bolt_head[1], bolt_head[2], 
                          color='blue', marker='s', s=100)
                
                # Plot entry point
                ax.scatter(entry_point[0], entry_point[1], entry_point[2], 
                          color='red', marker='o', s=80)
            
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title('Combined Electrode and Bolt Trajectories')
            
            combined_fig_path = os.path.join(output_dir, 'combined_trajectories.png')
            fig.savefig(combined_fig_path)
            plt.close(fig)
            print(f"Combined trajectory visualization saved to {combined_fig_path}")
        
        # Create the standard visualizations
        visualize_combined_results(coords_array, results, output_dir)
        
        # Attempt to connect bolt trajectories with electrode clusters
        if bolt_trajectories and 'trajectories' in results:
            print("Attempting to match bolt trajectories with electrode clusters...")
            
            # Create a new report page
            match_fig = create_bolt_electrode_matching_page(bolt_trajectories, results)
            match_fig_path = os.path.join(output_dir, 'bolt_cluster_matching.png')
            match_fig.savefig(match_fig_path)
            plt.close(match_fig)
            print(f"Matching analysis saved to {match_fig_path}")
        
        finish_time = time.time()
        print(f"Total execution time: {finish_time - start_time:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")

def create_bolt_electrode_matching_page(bolt_trajectories, results):
    """
    Creates a visualization showing the matching between bolt trajectories and electrode clusters.
    
    Args:
        bolt_trajectories: List of bolt trajectory dictionaries
        results: Results dictionary from integrated_trajectory_analysis
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Bolt Head to Electrode Cluster Matching', fontsize=16)
    
    # Create 3D visualization
    ax1 = fig.add_subplot(121, projection='3d')
    
    # For each bolt trajectory, find the closest electrode cluster
    matches = []
    
    for bolt_idx, bolt_traj in enumerate(bolt_trajectories):
        bolt_entry = np.array(bolt_traj['entry_point'])
        bolt_direction = np.array(bolt_traj['direction'])
        
        best_match = None
        best_score = float('inf')
        
        # For each electrode cluster
        for elec_traj in results.get('trajectories', []):
            # Calculate metrics for matching:
            # 1. Distance from bolt entry to electrode cluster start
            # 2. Angle between directions
            
            elec_start = np.array(elec_traj['endpoints'][0])
            elec_direction = np.array(elec_traj['direction'])
            
            # Distance between entry point and electrode start
            distance = np.linalg.norm(bolt_entry - elec_start)
            
            # Angle between directions (0 to 180 degrees)
            dot_product = np.dot(bolt_direction, elec_direction)
            angle = np.degrees(np.arccos(np.clip(abs(dot_product), -1.0, 1.0)))
            
            # Combined score (lower is better)
            # You may need to adjust the weights
            score = distance + angle * 0.5
            
            if score < best_score:
                best_score = score
                best_match = {
                    'cluster_id': elec_traj['cluster_id'],
                    'distance': distance,
                    'angle': angle,
                    'score': score,
                    'elec_start': elec_start,
                    'elec_direction': elec_direction
                }
        
        if best_match:
            matches.append({
                'bolt_idx': bolt_idx,
                'bolt_entry': bolt_entry,
                'bolt_direction': bolt_direction,
                'match': best_match
            })
            
            # Plot this match on the 3D visualization
            # Bolt trajectory
            bolt_head = np.array(bolt_traj['bolt_head'])
            ax1.plot([bolt_head[0], bolt_entry[0]],
                    [bolt_head[1], bolt_entry[1]],
                    [bolt_head[2], bolt_entry[2]],
                    'g-', linewidth=2, alpha=0.7)
            
            # Plot bolt head and entry points
            ax1.scatter(bolt_head[0], bolt_head[1], bolt_head[2], 
                       color='blue', marker='s', s=100)
            ax1.scatter(bolt_entry[0], bolt_entry[1], bolt_entry[2], 
                       color='red', marker='o', s=80)
            
            # Plot connection to best matched electrode
            ax1.plot([bolt_entry[0], best_match['elec_start'][0]],
                    [bolt_entry[1], best_match['elec_start'][1]],
                    [bolt_entry[2], best_match['elec_start'][2]],
                    'r--', linewidth=1, alpha=0.5)
            
            # Plot cluster start point
            ax1.scatter(best_match['elec_start'][0], 
                       best_match['elec_start'][1], 
                       best_match['elec_start'][2],
                       color='orange', marker='^', s=80)
            
            # Add text labels for cluster IDs
            ax1.text(best_match['elec_start'][0],
                    best_match['elec_start'][1],
                    best_match['elec_start'][2],
                    f"C{best_match['cluster_id']}",
                    color='black', fontsize=8)
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Visualization of Matches')
    
    # Create table with matching information
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    
    if matches:
        columns = ['Bolt #', 'Electrode Cluster', 'Distance (mm)', 'Angle (°)', 'Score']
        table_data = []
        
        for i, match in enumerate(matches):
            table_data.append([
                i,
                match['match']['cluster_id'],
                f"{match['match']['distance']:.2f}",
                f"{match['match']['angle']:.2f}",
                f"{match['match']['score']:.2f}"
            ])
        
        table = ax2.table(cellText=table_data, colLabels=columns, 
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax2.set_title('Bolt-Electrode Cluster Matching Summary', pad=20)
    else:
        ax2.text(0.5, 0.5, "No matches found", ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    main()

#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Electrode_path\cosntruction_angles.py').read())