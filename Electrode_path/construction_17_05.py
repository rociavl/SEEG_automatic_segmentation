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
import vtk

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

############################################
#### Bolt_head and entry point
########################################
def analyze_bolt_head_directions(combined_volume):
    """
    Analyzes the combined bolt head and entry point mask to determine the direction of each bolt.
    
    Parameters:
    -----------
    combined_volume: vtkMRMLScalarVolumeNode
        The combined mask containing both bolt heads and entry points
        
    Returns:
    --------
    directions: dict
        Dictionary mapping region ID to direction vector, center point, and other metrics
    """
    print("Analyzing bolt head directions...")
    
    # Get all centroids from the combined mask
    centroids_ras = get_all_centroids(combined_volume)
    if not centroids_ras:
        logging.error("No centroids found in combined mask.")
        return {}
    
    # Get the volume array and properties for region analysis
    volume_array = get_array_from_volume(combined_volume)
    ras_to_ijk = vtk.vtkMatrix4x4()
    ras_to_ijk_matrix = combined_volume.GetRASToIJKMatrix(ras_to_ijk)
    ijk_to_ras_matrix = combined_volume.GetIJKToRASMatrix(vtk.vtkMatrix4x4())
    
    # Label the regions in the combined mask
    labeled_array, num_features = label(volume_array, return_num=True)
    print(f"Found {num_features} bolt/entry regions in combined mask")
    
    # Get region properties
    props = regionprops_table(labeled_array, properties=('label', 'coords', 'centroid', 'orientation', 'axis_major_length'))
    
    directions = {}
    
    # Process each region to find the direction
    for i in range(num_features):
        region_id = props['label'][i]
        region_coords = props['coords'][i]  # Coordinates in IJK space
        
        # Convert region coordinates to RAS space
        ras_points = []
        for coord in region_coords:
            ras_point = get_ras_coordinates_from_ijk(coord, ijk_to_ras_matrix)
            ras_points.append(ras_point)
        
        ras_points = np.array(ras_points)
        
        # Use PCA to find the principal direction of the region
        if len(ras_points) >= 2:
            pca = PCA(n_components=3)
            pca.fit(ras_points)
            
            # The first principal component is the main direction of the bolt
            direction = pca.components_[0]
            center = np.mean(ras_points, axis=0)
            
            # Calculate linearity (how well the points fit along the main axis)
            linearity = pca.explained_variance_ratio_[0]
            
            # Calculate angles with principal axes
            angles = calculate_angles(direction)
            
            # Extract endpoints based on projections onto the main axis
            projected = np.dot(ras_points - center, direction)
            sorted_indices = np.argsort(projected)
            
            # Assume the bolt head is at one end and entry point at the other
            bolt_head_point = ras_points[sorted_indices[-1]]
            entry_point = ras_points[sorted_indices[0]]
            
            # Store the results
            directions[region_id] = {
                'region_id': int(region_id),
                'direction': direction.tolist(),
                'center': center.tolist(),
                'angles_with_axes': angles,
                'linearity': float(linearity),
                'bolt_head_point': bolt_head_point.tolist(),
                'entry_point': entry_point.tolist(),
                'length_mm': float(np.linalg.norm(bolt_head_point - entry_point)),
                'pca_variance': pca.explained_variance_ratio_.tolist()
            }
            
    print(f"Analyzed {len(directions)} bolt directions")
    return directions

def visualize_bolt_directions(directions, coords_array=None, trajectories=None, output_dir=None):
    """
    Visualizes the bolt directions in a 3D plot.
    
    Parameters:
    -----------
    directions: dict
        Dictionary of bolt directions as returned by analyze_bolt_head_directions
    coords_array: ndarray, optional
        Array of electrode coordinates to overlay
    trajectories: list, optional
        List of trajectory dictionaries to overlay
    output_dir: str, optional
        Directory to save the output plot
    """
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot bolt directions
    for region_id, data in directions.items():
        bolt_head = np.array(data['bolt_head_point'])
        entry_point = np.array(data['entry_point'])
        direction = np.array(data['direction'])
        center = np.array(data['center'])
        
        # Plot bolt to entry line
        ax.plot([bolt_head[0], entry_point[0]],
               [bolt_head[1], entry_point[1]],
               [bolt_head[2], entry_point[2]], 
               '-', color='blue', linewidth=2, alpha=0.8)
        
        # Plot bolt head point
        ax.scatter(bolt_head[0], bolt_head[1], bolt_head[2], 
                  c='green', marker='o', s=100, label='Bolt Head' if region_id == list(directions.keys())[0] else "")
        
        # Plot entry point
        ax.scatter(entry_point[0], entry_point[1], entry_point[2], 
                  c='red', marker='o', s=100, label='Entry Point' if region_id == list(directions.keys())[0] else "")
        
        # Add direction arrow
        arrow_length = data['length_mm'] * 0.3
        arrow = Arrow3D(
            [center[0], center[0] + direction[0]*arrow_length],
            [center[1], center[1] + direction[1]*arrow_length],
            [center[2], center[2] + direction[2]*arrow_length],
            mutation_scale=15, lw=2, arrowstyle="-|>", color='blue')
        ax.add_artist(arrow)
        
        # Add region ID text
        ax.text(center[0], center[1], center[2], f"Region {region_id}", fontsize=8)
    
    # Plot electrode trajectories if provided
    if coords_array is not None and trajectories is not None:
        # Create a colormap for the trajectories
        traj_cmap = plt.colormaps['tab20'].resampled(len(trajectories))
        
        for i, traj in enumerate(trajectories):
            color = traj_cmap(i)
            cluster_id = traj['cluster_id']
            
            # Get mask for this cluster
            clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
            mask = clusters == cluster_id
            
            # Plot electrode points for this cluster
            ax.scatter(coords_array[mask, 0], coords_array[mask, 1], coords_array[mask, 2],
                      c=[color], s=50, alpha=0.6, label=f'Electrodes {cluster_id}')
            
            # If trajectory has endpoints, plot the line
            if 'endpoints' in traj:
                endpoints = traj['endpoints']
                ax.plot([endpoints[0][0], endpoints[1][0]],
                       [endpoints[0][1], endpoints[1][1]],
                       [endpoints[0][2], endpoints[1][2]],
                       '-', color=color, linewidth=2, alpha=0.6)
    
    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Bolt Head to Entry Point Directions\n(Green=Bolt Head, Red=Entry Point, Blue=Direction)')
    
    # Add legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    ax.legend(unique_handles, unique_labels, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'bolt_head_directions.pdf'))
        plt.savefig(os.path.join(output_dir, 'bolt_head_directions.png'), dpi=300)
    
    return fig

def create_bolt_direction_analysis_page(directions):
    """
    Creates a detailed analysis page for bolt directions.
    
    Parameters:
    -----------
    directions: dict
        Dictionary of bolt directions as returned by analyze_bolt_head_directions
        
    Returns:
    --------
    fig: matplotlib.figure.Figure
        The figure object containing the analysis page
    """
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Bolt Head Direction Analysis', fontsize=16)
    
    if not directions:
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5, 'No bolt directions detected', ha='center', va='center')
        return fig
    
    # Create subplots
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, projection='3d')
    
    # Plot PCA variance ratios
    variances = [data['pca_variance'] for data in directions.values()]
    labels = [f"Region {data['region_id']}" for data in directions.values()]
    
    for i, var in enumerate(variances):
        ax1.bar(i, var[0], color='b', alpha=0.6, label='PC1' if i == 0 else "")
        ax1.bar(i, var[1], bottom=var[0], color='g', alpha=0.6, label='PC2' if i == 0 else "")
        ax1.bar(i, var[2], bottom=var[0]+var[1], color='r', alpha=0.6, label='PC3' if i == 0 else "")
    
    ax1.set_xticks(range(len(variances)))
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('PCA Explained Variance by Bolt Region')
    ax1.legend()
    
    # Plot angles with principal axes
    angles_x = [data['angles_with_axes']['X'] for data in directions.values()]
    angles_y = [data['angles_with_axes']['Y'] for data in directions.values()]
    angles_z = [data['angles_with_axes']['Z'] for data in directions.values()]
    
    x_pos = np.arange(len(angles_x))
    width = 0.25
    
    ax2.bar(x_pos - width, angles_x, width, label='X-axis', alpha=0.6)
    ax2.bar(x_pos, angles_y, width, label='Y-axis', alpha=0.6)
    ax2.bar(x_pos + width, angles_z, width, label='Z-axis', alpha=0.6)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_title('Bolt Direction Angles with Principal Axes')
    ax2.legend()
    
    # Plot linearity vs length
    linearities = [data['linearity'] for data in directions.values()]
    lengths = [data['length_mm'] for data in directions.values()]
    
    ax3.scatter(lengths, linearities, c=range(len(linearities)), cmap='viridis')
    for i, txt in enumerate(labels):
        ax3.annotate(txt, (lengths[i], linearities[i]), fontsize=8)
    
    ax3.set_xlabel('Bolt Length (mm)')
    ax3.set_ylabel('Linearity (PC1 variance ratio)')
    ax3.set_title('Linearity vs Bolt Length')
    
    # Create table with measurements
    table_data = []
    for data in directions.values():
        row = [
            data['region_id'],
            f"{data['length_mm']:.1f}",
            f"{data['linearity']:.2f}",
            f"{data['angles_with_axes']['X']:.1f}°",
            f"{data['angles_with_axes']['Y']:.1f}°",
            f"{data['angles_with_axes']['Z']:.1f}°"
        ]
        table_data.append(row)
    
    # Plot direction vectors in 3D
    for data in directions.values():
        direction = np.array(data['direction'])
        center = np.array(data['center'])
        
        ax4.quiver(center[0], center[1], center[2],
                  direction[0], direction[1], direction[2],
                  length=10, normalize=True,
                  color=plt.cm.viridis(data['linearity']),
                  alpha=0.7)
        
        # Label the arrow
        ax4.text(center[0], center[1], center[2], f"R{data['region_id']}", fontsize=8)
    
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('Bolt Direction Vectors (color by linearity)')
    
    plt.tight_layout()
    return fig

def find_nearest_trajectory_to_bolt(bolt_directions, trajectories):
    """
    Finds the nearest trajectory to each bolt direction.
    
    Parameters:
    -----------
    bolt_directions: dict
        Dictionary of bolt directions from analyze_bolt_head_directions
    trajectories: list
        List of trajectory dictionaries from integrated_trajectory_analysis
        
    Returns:
    --------
    matches: dict
        Dictionary mapping bolt region ID to matching trajectory info
    """
    if not bolt_directions or not trajectories:
        return {}
    
    matches = {}
    
    for bolt_id, bolt_data in bolt_directions.items():
        bolt_entry = np.array(bolt_data['entry_point'])
        bolt_direction = np.array(bolt_data['direction'])
        
        best_match = None
        min_dist = float('inf')
        min_angle = float('inf')
        
        for traj in trajectories:
            if 'entry_point' in traj and traj['entry_point'] is not None:
                traj_entry = np.array(traj['entry_point'])
                traj_direction = np.array(traj['direction'])
                
                # Calculate distance between entry points
                dist = np.linalg.norm(bolt_entry - traj_entry)
                
                # Calculate angle between directions
                dot_product = np.dot(bolt_direction, traj_direction)
                angle = np.degrees(np.arccos(np.clip(abs(dot_product), -1.0, 1.0)))
                
                # Find best match based on combined distance and angle
                combined_score = dist + angle * 0.5  # Weight angle slightly less than distance
                
                if combined_score < min_dist + min_angle * 0.5:
                    min_dist = dist
                    min_angle = angle
                    best_match = {
                        'trajectory_id': traj['cluster_id'],
                        'distance_mm': dist,
                        'angle_degrees': angle,
                        'combined_score': combined_score
                    }
        
        if best_match:
            matches[bolt_id] = best_match
    
    return matches

def create_bolt_trajectory_matching_page(bolt_directions, trajectories, matches):
    """
    Creates a visualization showing the matching between bolt directions and trajectories.
    
    Parameters:
    -----------
    bolt_directions: dict
        Dictionary of bolt directions
    trajectories: list
        List of trajectory dictionaries
    matches: dict
        Dictionary of matches between bolts and trajectories
        
    Returns:
    --------
    fig: matplotlib.figure.Figure
        Figure showing the matches
    """
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Bolt to Trajectory Matching', fontsize=16)
    
    if not matches:
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5, 'No matches found between bolts and trajectories', ha='center', va='center')
        return fig
    
    # Create subplots
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    
    # Create colormap for trajectories
    n_trajectories = len(set(match['trajectory_id'] for match in matches.values()))
    traj_cmap = plt.colormaps['tab20'].resampled(max(n_trajectories, 1))
    
    # Plot 3D visualization
    for bolt_id, match in matches.items():
        traj_id = match['trajectory_id']
        bolt_data = bolt_directions[bolt_id]
        
        # Find matching trajectory
        matching_traj = None
        for traj in trajectories:
            if traj['cluster_id'] == traj_id:
                matching_traj = traj
                break
        
        if not matching_traj:
            continue
            
        # Plot bolt direction
        bolt_entry = np.array(bolt_data['entry_point'])
        bolt_head = np.array(bolt_data['bolt_head_point'])
        
        ax1.plot([bolt_head[0], bolt_entry[0]],
                [bolt_head[1], bolt_entry[1]],
                [bolt_head[2], bolt_entry[2]],
                '-', color='blue', linewidth=2, alpha=0.8)
        
        # Plot trajectory
        traj_color = traj_cmap(traj_id % traj_cmap.N)
        
        if 'endpoints' in matching_traj:
            endpoints = matching_traj['endpoints']
            ax1.plot([endpoints[0][0], endpoints[1][0]],
                    [endpoints[0][1], endpoints[1][1]],
                    [endpoints[0][2], endpoints[1][2]],
                    '-', color=traj_color, linewidth=2, alpha=0.8)
        
        # Plot entry points
        traj_entry = np.array(matching_traj['entry_point']) if 'entry_point' in matching_traj and matching_traj['entry_point'] is not None else None
        
        if traj_entry is not None:
            ax1.scatter(traj_entry[0], traj_entry[1], traj_entry[2],
                       c=[traj_color], marker='*', s=100,
                       label=f'Traj {traj_id} Entry' if bolt_id == list(matches.keys())[0] else "")
            
            # Draw connection between bolt entry and trajectory entry
            ax1.plot([bolt_entry[0], traj_entry[0]],
                    [bolt_entry[1], traj_entry[1]],
                    [bolt_entry[2], traj_entry[2]],
                    '--', color='red', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Visualization of Bolt-Trajectory Matches')
    
    # Create table of matches
    ax2.axis('off')
    
    table_data = []
    columns = ['Bolt ID', 'Traj ID', 'Distance (mm)', 'Angle (°)', 'Score']
    
    for bolt_id, match in matches.items():
        row = [
            bolt_id,
            match['trajectory_id'],
            f"{match['distance_mm']:.1f}",
            f"{match['angle_degrees']:.1f}",
            f"{match['combined_score']:.1f}"
        ]
        table_data.append(row)
    
    table = ax2.table(cellText=table_data, colLabels=columns, 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax2.set_title('Bolt-Trajectory Match Metrics')
    
    plt.tight_layout()
    return fig

def analyze_bolt_trajectory_relationships(bolt_directions, coords_array, results, output_dir=None):
    """
    Comprehensive analysis of relationships between bolt directions and electrode trajectories.
    
    Parameters:
    -----------
    bolt_directions: dict
        Dictionary of bolt directions from analyze_bolt_head_directions
    coords_array: ndarray
        Array of electrode coordinates
    results: dict
        Results dictionary from integrated_trajectory_analysis
    output_dir: str, optional
        Directory to save output plots
        
    Returns:
    --------
    match_results: dict
        Dictionary containing match information
    """
    trajectories = results.get('trajectories', [])
    
    # Find matches between bolts and trajectories
    matches = find_nearest_trajectory_to_bolt(bolt_directions, trajectories)
    
    # Create visualizations
    if output_dir:
        # Bolt direction visualization
        bolt_fig = visualize_bolt_directions(bolt_directions, coords_array, trajectories, output_dir)
        plt.close(bolt_fig)
        
        # Bolt direction analysis page
        analysis_fig = create_bolt_direction_analysis_page(bolt_directions)
        plt.savefig(os.path.join(output_dir, 'bolt_direction_analysis.pdf'))
        plt.savefig(os.path.join(output_dir, 'bolt_direction_analysis.png'), dpi=300)
        plt.close(analysis_fig)
        
        # Bolt-trajectory matching page
        matching_fig = create_bolt_trajectory_matching_page(bolt_directions, trajectories, matches)
        plt.savefig(os.path.join(output_dir, 'bolt_trajectory_matching.pdf'))
        plt.savefig(os.path.join(output_dir, 'bolt_trajectory_matching.png'), dpi=300)
        plt.close(matching_fig)
        
        # Create combined PDF report
        with PdfPages(os.path.join(output_dir, 'bolt_trajectory_analysis_report.pdf')) as pdf:
            # Bolt direction visualization
            bolt_fig = visualize_bolt_directions(bolt_directions, coords_array, trajectories)
            pdf.savefig(bolt_fig)
            plt.close(bolt_fig)
            
            # Bolt direction analysis
            analysis_fig = create_bolt_direction_analysis_page(bolt_directions)
            pdf.savefig(analysis_fig)
            plt.close(analysis_fig)
            
            # Bolt-trajectory matching
            matching_fig = create_bolt_trajectory_matching_page(bolt_directions, trajectories, matches)
            pdf.savefig(matching_fig)
            plt.close(matching_fig)
    
    # Return match results
    match_results = {
        'bolt_directions': bolt_directions,
        'matches': matches,
        'match_stats': {
            'total_bolts': len(bolt_directions),
            'matched_bolts': len(matches),
            'unique_trajectories': len(set(m['trajectory_id'] for m in matches.values())),
            'avg_distance': np.mean([m['distance_mm'] for m in matches.values()]) if matches else 0,
            'avg_angle': np.mean([m['angle_degrees'] for m in matches.values()]) if matches else 0
        }
    }
    
    return match_results



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




def updated_main():
    try:
        start_time = time.time()
        electrodes_volume = slicer.util.getNode('electrode_mask_success_1')
        entry_points_volume = slicer.util.getNode('EntryPointsMask')
        combined_bolt_entry_volume = slicer.util.getNode('CombinedBoltHeadEntryPointsMask')
        
        output_dir = r"C:\Users\rocia\Downloads\TFG\Cohort\Centroids\p1_Trajectories_17_05_enhan_pdf_17_05\output_plots"
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
        
        # Run trajectory analysis
        results = integrated_trajectory_analysis(
            coords_array=coords_array,
            entry_points=entry_points,
            max_neighbor_distance=7.5,
            min_neighbors=3
        )
        
        print(f"Analysis complete: {results['n_trajectories']} trajectories detected.")
        visualize_combined_results(coords_array, results, output_dir)
        
        # Analyze bolt head directions if combined volume is available
        if combined_bolt_entry_volume:
            print("Analyzing bolt head directions...")
            bolt_directions = analyze_bolt_head_directions(combined_bolt_entry_volume)
            
            if bolt_directions:
                # Analyze relationships between bolt directions and trajectories
                match_results = analyze_bolt_trajectory_relationships(
                    bolt_directions, coords_array, results, output_dir
                )
                
                print(f"Bolt analysis complete: {len(bolt_directions)} bolt directions found.")
                print(f"Match stats: {match_results['match_stats']}")
            else:
                print("No bolt directions found in the combined mask.")
        
        finish_time = time.time()
        print(f"Total execution time: {finish_time - start_time:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    updated_main()













# def main():
#     try:
#         start_time = time.time()
#         electrodes_volume = slicer.util.getNode('P2_mask_31')
#         entry_points_volume = slicer.util.getNode('P2_brain_entry_points')
#         #brain_mask_volume = slicer.util.getNode('patient1_mask_5') 
        
#         output_dir = r"C:\Users\rocia\Downloads\TFG\Cohort\Centroids\p2_Trajectories_16_05_enhan_pdf_15_05\output_plots"
#         os.makedirs(output_dir, exist_ok=True)
#         print("Starting enhanced trajectory analysis...")
        
#         # Get centroids for electrodes
#         centroids_ras = get_all_centroids(electrodes_volume)
        
#         if not centroids_ras:
#             logging.error("No centroids found.")
#             return
        
#         coords_array = np.array(list(centroids_ras.values()))
        
#         # Get entry points if available
#         entry_points = None
#         if entry_points_volume:
#             entry_centroids_ras = get_all_centroids(entry_points_volume)
#             if entry_centroids_ras:
#                 entry_points = np.array(list(entry_centroids_ras.values()))
        
#         results = integrated_trajectory_analysis(
#             coords_array=coords_array,
#             entry_points=entry_points,
#             max_neighbor_distance=7.5,
#             min_neighbors=3
#         )
        
#         print(f"Analysis complete: {results['n_trajectories']} trajectories detected.")
#         visualize_combined_results(coords_array, results, output_dir)
#         finish_time = time.time()
#         print(f"Total execution time: {finish_time - start_time:.2f} seconds")
        
#         # # Save results to JSON
#         # import json
#         # results_file = os.path.join(output_dir, 'trajectory_analysis_results.json')
#         # with open(results_file, 'w') as f:
#         #     json.dump(results, f, indent=2)
#         # logging.info(f"Results saved to {results_file}")
        
#     except Exception as e:
#         logging.error(f"Main execution failed: {str(e)}")

# if __name__ == "__main__":
#     main()

#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Electrode_path\construction_17_05.py').read())