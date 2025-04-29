import slicer
import numpy as np
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import networkx as nx
from skimage.measure import label, regionprops_table
from scipy.spatial.distance import cdist
from collections import defaultdict
from Outermost_centroids_coordinates.outermost_centroids_vol_slicer import (
    get_ras_coordinates_from_ijk, get_array_from_volume, calculate_centroids_numpy,
    get_centroids_ras, get_surface_from_volume, convert_surface_vertices_to_ras
)
from End_points.midplane_prueba import get_all_centroids


def integrated_trajectory_analysis(coords_array, entry_points=None, max_neighbor_distance=7.5, min_neighbors=3):
    results = {
        'dbscan': {},
        'louvain': {},
        'combined': {}
    }
    
    dbscan = DBSCAN(eps=max_neighbor_distance, min_samples=min_neighbors)
    clusters = dbscan.fit_predict(coords_array)
    
    unique_clusters = set(clusters)
    results['dbscan']['n_clusters'] = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    results['dbscan']['noise_points'] = np.sum(clusters == -1)
    G = nx.Graph()
    results['graph'] = G
    
    for i, coord in enumerate(coords_array):
        G.add_node(i, pos=coord, dbscan_cluster=int(clusters[i]))

    distances = cdist(coords_array, coords_array)
    for i in range(len(coords_array)):
        for j in range(i + 1, len(coords_array)):
            dist = distances[i,j]
            if dist <= max_neighbor_distance:
                G.add_edge(i, j, weight=1.0 / (dist + 1e-6))  

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
                
        # Store community info in graph
        for node in G.nodes:
            G.nodes[node]['louvain_community'] = node_to_community.get(node, -1)
            
    except Exception as e:
        logging.warning(f"Louvain community detection failed: {e}")
        results['louvain']['error'] = str(e)
    
    if 'error' not in results['louvain']:
        cluster_community_mapping = defaultdict(set)
        for node in G.nodes:
            dbscan_cluster = G.nodes[node]['dbscan_cluster']
            louvain_community = G.nodes[node]['louvain_community']
            if dbscan_cluster != -1:  
                cluster_community_mapping[dbscan_cluster].add(louvain_community)
        
        # Calculate purity of DBSCAN clusters with respect to Louvain communities
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
        
        # Perform PCA on the cluster
        try:
            pca = PCA(n_components=3)
            pca.fit(cluster_coords)
            
            linearity = pca.explained_variance_ratio_[0]
            direction = pca.components_[0]
            center = np.mean(cluster_coords, axis=0)
            
            # Project points onto the principal axis
            projected = np.dot(cluster_coords - center, direction)
            
            # If we have entry points, find the closest entry point to determine start of path
            if entry_points is not None:
                # Find closest entry point to any point in the cluster
                min_dist = float('inf')
                start_entry_point = None
                for entry in entry_points:
                    dists = cdist([entry], cluster_coords)
                    min_cluster_dist = np.min(dists)
                    if min_cluster_dist < min_dist:
                        min_dist = min_cluster_dist
                        start_entry_point = entry
                
                if start_entry_point is not None:
                    # Calculate projection of entry point onto principal axis
                    entry_projection = np.dot(start_entry_point - center, direction)
                    # Sort points based on projection relative to entry point
                    sorted_indices = np.argsort(np.abs(projected - entry_projection))
                    sorted_coords = cluster_coords[sorted_indices]
                else:
                    sorted_indices = np.argsort(projected)
                    sorted_coords = cluster_coords[sorted_indices]
            else:
                sorted_indices = np.argsort(projected)
                sorted_coords = cluster_coords[sorted_indices]
            
            # Calculate trajectory metrics
            distances = np.linalg.norm(np.diff(sorted_coords, axis=0), axis=1)
            spacing_regularity = np.std(distances) / np.mean(distances) if len(distances) > 1 else np.nan
            trajectory_length = np.sum(distances)
            
            trajectories.append({
                "cluster_id": int(cluster_id),
                "louvain_community": louvain_community,
                "electrode_count": int(len(cluster_coords)),
                "linearity": float(linearity),
                "direction": direction.tolist(),
                "length_mm": float(trajectory_length),
                "spacing_regularity": float(spacing_regularity) if not np.isnan(spacing_regularity) else None,
                "avg_spacing_mm": float(np.mean(distances)) if len(distances) > 0 else None,
                "endpoints": [sorted_coords[0].tolist(), sorted_coords[-1].tolist()]
            })
        except Exception as e:
            logging.warning(f"PCA failed for cluster {cluster_id}: {e}")
            continue
    
    results['trajectories'] = trajectories
    results['n_trajectories'] = len(trajectories)

    return results

def visualize_combined_results(coords_array, results, output_dir=None):
    fig = plt.figure(figsize=(18, 6))
    
    # DBSCAN clusters
    ax1 = fig.add_subplot(131, projection='3d')
    clusters = np.array([node[1]['dbscan_cluster'] for node in results['graph'].nodes(data=True)])
    
    unique_clusters = set(clusters)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    
    for cluster_id, color in zip(unique_clusters, colors):
        if cluster_id == -1:
            continue  # Skip noise
        mask = clusters == cluster_id
        ax1.scatter(coords_array[mask, 0], coords_array[mask, 1], coords_array[mask, 2], 
                   c=[color], label=f'Cluster {cluster_id}', s=50)
    
    ax1.set_title('DBSCAN Clustering')
    ax1.legend()
    
    # Louvain communities
    ax2 = fig.add_subplot(132, projection='3d')
    if 'louvain_community' in next(iter(results['graph'].nodes(data=True)))[1]:
        communities = np.array([node[1]['louvain_community'] for node in results['graph'].nodes(data=True)])
        unique_communities = set(communities)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_communities)))
        
        for comm_id, color in zip(unique_communities, colors):
            mask = communities == comm_id
            ax2.scatter(coords_array[mask, 0], coords_array[mask, 1], coords_array[mask, 2], 
                       c=[color], label=f'Community {comm_id}', s=50)
        
        ax2.set_title('Louvain Communities')
        ax2.legend()
    
    # Combined trajectories
    ax3 = fig.add_subplot(133, projection='3d')
    for traj in results.get('trajectories', []):
        color = np.random.rand(3,)
        endpoints = traj['endpoints']
        ax3.plot([endpoints[0][0], endpoints[1][0]],
                [endpoints[0][1], endpoints[1][1]],
                [endpoints[0][2], endpoints[1][2]], 
                'o-', color=color, markersize=8, linewidth=2,
                label=f'Traj {traj["cluster_id"]} (L:{traj["length_mm"]:.1f}mm)')
    
    ax3.set_title('Detected Trajectories')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'combined_analysis.png'), dpi=300, bbox_inches='tight')
    else:
        plt.show()

def main():
    try:
        electrodes_volume = slicer.util.getNode('electrode_mask_success')
        entry_points_volume = slicer.util.getNode('P1_brain_entry_points_1')  # This should be your entry points volume
        
        output_dir = r"C:\Users\rocia\Downloads\TFG\Cohort\Centroids\Trajectories_28_04\output_plots"
        logging.info("Starting integrated analysis...")
        
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
            entry_points=entry_points,  # Pass entry points if available
            max_neighbor_distance=7.5,
            min_neighbors=3
        )
        
        logging.info(f"Analysis complete: {results['n_trajectories']} trajectories detected.")
        visualize_combined_results(coords_array, results, output_dir)
        
    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")


main()

#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Electrode_path\construction_3.py').read())