import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy.spatial.distance import cdist
import os
from sklearn.cluster import DBSCAN


def extract_additional_features(results_df, patient_id):
    """Extract additional features from results data"""

    # Group by mask for per-mask analysis
    all_features = {}

    for mask_name in results_df['Mask'].unique():
        mask_data = results_df[results_df['Mask'] == mask_name]
        mask_file_stem = Path(mask_name).stem

        # 1. Directional bias analysis - Are errors more prominent in certain directions?
        error_vectors = []
        for _, row in mask_data.iterrows():
            if not any(np.isnan(c) for c in row['GT RAS Coordinates']):
                pred = np.array(row['RAS Coordinates'])
                gt = np.array(row['GT RAS Coordinates'])
                error_vectors.append(pred - gt)

        error_vectors = np.array(error_vectors)

        if len(error_vectors) > 0:
            mean_error_vector = np.mean(error_vectors, axis=0)
            std_error_vector = np.std(error_vectors, axis=0)
            max_error_direction = np.argmax(np.abs(mean_error_vector))
            directions = ['X', 'Y', 'Z']
            max_error_axis = directions[max_error_direction]
        else:
            mean_error_vector = np.array([np.nan, np.nan, np.nan])
            std_error_vector = np.array([np.nan, np.nan, np.nan])
            max_error_axis = "N/A"

        from sklearn.cluster import DBSCAN

        coords = np.array([p for p in mask_data['RAS Coordinates']])
        if len(coords) > 1:
            try:
                clustering = DBSCAN(eps=5.0, min_samples=2).fit(coords)
                num_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                noise_points = list(clustering.labels_).count(-1)
                cluster_ratio = (len(coords) - noise_points) / len(coords) if len(coords) > 0 else 0
            except:
                num_clusters = 0
                noise_points = 0
                cluster_ratio = 0
        else:
            num_clusters = 0
            noise_points = 0
            cluster_ratio = 0

        # 4. Distance statistics by electrode size
        mask_data['Size_Category'] = pd.cut(
            mask_data['Pixel Count'],
            bins=[0, 10, 20, 50, 100, np.inf],
            labels=['Tiny', 'Small', 'Medium', 'Large', 'Very Large']
        )

        size_stats = mask_data.groupby('Size_Category').agg({
            'Distance (mm)': ['mean', 'std', 'count'],
            'Success': 'mean'
        })

        mask_data['Octant'] = mask_data['RAS Coordinates'].apply(
            lambda p: f"{'P' if p[0] >= 0 else 'A'}"
                     f"{'S' if p[1] >= 0 else 'I'}"
                     f"{'R' if p[2] >= 0 else 'L'}"
        )

        octant_stats = mask_data.groupby('Octant').agg({
            'Distance (mm)': ['mean', 'count'],
            'Success': 'mean'
        })

        # Calculate outlier statistics
        q3 = mask_data['Distance (mm)'].quantile(0.75)
        iqr = mask_data['Distance (mm)'].quantile(0.75) - mask_data['Distance (mm)'].quantile(0.25)
        outlier_threshold = q3 + 1.5 * iqr
        outliers = mask_data[mask_data['Distance (mm)'] > outlier_threshold]
        outlier_count = len(outliers)
        outlier_ratio = outlier_count / len(mask_data) if len(mask_data) > 0 else 0

        # Distance to nearest neighbor (electrode crowding)
        nn_distances = []
        for i, row_i in mask_data.iterrows():
            min_dist = float('inf')
            coord_i = np.array(row_i['RAS Coordinates'])

            for j, row_j in mask_data.iterrows():
                if i != j:
                    coord_j = np.array(row_j['RAS Coordinates'])
                    dist = np.linalg.norm(coord_i - coord_j)
                    min_dist = min(min_dist, dist)

            if min_dist < float('inf'):
                nn_distances.append(min_dist)

        avg_nn_distance = np.mean(nn_distances) if nn_distances else np.nan
        min_nn_distance = np.min(nn_distances) if nn_distances else np.nan

        # Compile all features
        all_features[mask_file_stem] = {
            'patient_id': patient_id,
            'mask_name': mask_name,
            'error_x_mean': mean_error_vector[0],
            'error_y_mean': mean_error_vector[1],
            'error_z_mean': mean_error_vector[2],
            'error_x_std': std_error_vector[0],
            'error_y_std': std_error_vector[1],
            'error_z_std': std_error_vector[2],
            'max_error_axis': max_error_axis,
            'detection_count': len(mask_data),
            'success_count': mask_data['Success'].sum(),
            'success_rate': mask_data['Success'].mean() * 100,
            'mean_distance': mask_data['Distance (mm)'].mean(),
            'median_distance': mask_data['Distance (mm)'].median(),
            'std_distance': mask_data['Distance (mm)'].std(),
            'min_distance': mask_data['Distance (mm)'].min(),
            'max_distance': mask_data['Distance (mm)'].max(),
            'num_clusters': num_clusters,
            'cluster_ratio': cluster_ratio,
            'outlier_count': outlier_count,
            'outlier_ratio': outlier_ratio,
            'avg_nearest_neighbor_dist': avg_nn_distance,
            'min_nearest_neighbor_dist': min_nn_distance,
            'avg_pixel_count': mask_data['Pixel Count'].mean(),
            'median_pixel_count': mask_data['Pixel Count'].median(),
            'std_pixel_count': mask_data['Pixel Count'].std(),
            'size_stats': size_stats.to_dict(),
            'octant_stats': octant_stats.to_dict()
        }

    features_df = pd.DataFrame.from_dict(all_features, orient='index')
    features_df.to_csv(f'patient_{patient_id}_extended_features.csv')

    return features_df, all_features

