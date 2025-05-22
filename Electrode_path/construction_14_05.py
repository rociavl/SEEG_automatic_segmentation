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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import pandas as pd

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return min(zs)

def integrated_trajectory_analysis(coords_array, entry_points=None, max_neighbor_distance=7.5, min_neighbors=3):
    results = {
        'dbscan': {},
        'louvain': {},
        'combined': {},
        'parameters': {
            'max_neighbor_distance': max_neighbor_distance,
            'min_neighbors': min_neighbors,
            'n_electrodes': len(coords_array)
        }
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
    
    # Trajectory analysis with enhanced handling
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
            
            linearity = pca.explained_variance_ratio_[0]
            direction = pca.components_[0]
            center = np.mean(cluster_coords, axis=0)
            
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
                "spline_points": spline_points.tolist() if spline_points is not None else None
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
            
        fig = create_noise_points_page(coords_array, results)
        plt.show()

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
        columns = ['ID', 'Electrodes', 'Length (mm)', 'Linearity', 'Avg Spacing (mm)', 'Spacing Var']
        
        for traj in results['trajectories']:
            row = [
                traj['cluster_id'],
                traj['electrode_count'],
                f"{traj['length_mm']:.1f}",
                f"{traj['linearity']:.2f}",
                f"{traj['avg_spacing_mm']:.2f}" if traj['avg_spacing_mm'] else 'N/A',
                f"{traj['spacing_regularity']:.2f}" if traj['spacing_regularity'] else 'N/A'
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
    
    columns = ['ID', 'Community', 'Electrodes', 'Length (mm)', 'Linearity', 
              'Avg Spacing (mm)', 'Spacing Var', 'Entry Point']
    
    table_data = []
    for traj in results['trajectories']:
        has_entry = traj['entry_point'] is not None
        entry_text = f"({traj['entry_point'][0]:.1f}, {traj['entry_point'][1]:.1f}, {traj['entry_point'][2]:.1f})" if has_entry else "None"
        
        row = [
            traj['cluster_id'],
            traj['louvain_community'] if traj['louvain_community'] is not None else 'N/A',
            traj['electrode_count'],
            f"{traj['length_mm']:.1f}",
            f"{traj['linearity']:.2f}",
            f"{traj['avg_spacing_mm']:.2f}" if traj['avg_spacing_mm'] else 'N/A',
            f"{traj['spacing_regularity']:.2f}" if traj['spacing_regularity'] else 'N/A',
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
        electrodes_volume = slicer.util.getNode('electrode_mask_success')
        entry_points_volume = slicer.util.getNode('P1_brain_entry_points')
        brain_mask_volume = slicer.util.getNode('patient1_mask_5') 
        
        output_dir = r"C:\Users\rocia\Downloads\TFG\Cohort\Centroids\Trajectories_16_05_enhan_pdf_10\output_plots"
        os.makedirs(output_dir, exist_ok=True)
        logging.info("Starting enhanced trajectory analysis...")
        
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
        
        results = integrated_trajectory_analysis(
            coords_array=coords_array,
            entry_points=entry_points,
            max_neighbor_distance=10,
            min_neighbors=3
        )
        
        logging.info(f"Analysis complete: {results['n_trajectories']} trajectories detected.")
        visualize_combined_results(coords_array, results, output_dir)
        
        # # Save results to JSON
        # import json
        # results_file = os.path.join(output_dir, 'trajectory_analysis_results.json')
        # with open(results_file, 'w') as f:
        #     json.dump(results, f, indent=2)
        # logging.info(f"Results saved to {results_file}")
        
    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")

if __name__ == "__main__":
    main()

#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Electrode_path\construction_14_05.py').read())