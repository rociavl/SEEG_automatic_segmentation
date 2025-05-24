"""
Standalone Feature Extractor for Electrode Prediction

Generates exactly the target features:
- Numerical: RAS_X, RAS_Y, RAS_Z, CT_mean_intensity, CT_std_intensity, PCA1, PCA2, PCA3,
            dist_to_surface, mean_neighbor_dist, kde_density, n_neighbors, Louvain_Community,
            Pixel Count, dist_to_centroid, CT_max_intensity, CT_min_intensity,
            x_relative, y_relative, z_relative
- Categorical: Hemisphere, has_neighbors

Uses existing ct_features.py without modifying electrode_analysis.py
"""

import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import networkx as nx
import SimpleITK as sitk
from skimage.measure import marching_cubes
import ast

# Import from existing ct_features.py
from Centroids_pipeline.Features.ct_features import (
    load_nrrd_file_with_sitk,
    ras_to_voxel_coordinates_with_sitk,
    analyze_electrode_intensities,
)


def setup_logging(output_dir):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")


def load_and_parse_coordinates(results_file, patient_id=None, columns_success='all'):
    """Load CSV and parse coordinates safely."""
    df = pd.read_csv(results_file)
    logging.info(f"Loaded {len(df)} entries from results file")

    # Filter by patient if specified
    if patient_id is not None:
        df = df[df['Patient ID'] == patient_id]
        logging.info(f"Filtered to {len(df)} entries for patient {patient_id}")

    # Filter by success if specified and column exists
    if columns_success == 'yes' and 'Success' in df.columns:
        df = df[df['Success'] == True].copy()
        logging.info(f"Filtered to {len(df)} successful detections")

    if df.empty:
        raise ValueError("No electrode detections found to process")

    # Parse coordinates
    if {'RAS_X', 'RAS_Y', 'RAS_Z'}.issubset(df.columns):
        logging.info("Using existing RAS_X, RAS_Y, RAS_Z columns")
    elif 'RAS Coordinates' in df.columns:
        logging.info("Parsing RAS Coordinates column")
        df['RAS_X'] = df['RAS Coordinates'].apply(lambda x: ast.literal_eval(x)[0])
        df['RAS_Y'] = df['RAS Coordinates'].apply(lambda x: ast.literal_eval(x)[1])
        df['RAS_Z'] = df['RAS Coordinates'].apply(lambda x: ast.literal_eval(x)[2])
    else:
        raise ValueError("No coordinate columns found")

    # Ensure we have Patient ID
    if 'Patient ID' not in df.columns:
        df['Patient ID'] = patient_id if patient_id else 'unknown'

    # Add Pixel Count if missing
    if 'Pixel Count' not in df.columns:
        # Try to use other size-related columns
        if 'Volume' in df.columns:
            df['Pixel Count'] = df['Volume']
            logging.info("Used Volume column as Pixel Count")
        elif 'Size' in df.columns:
            df['Pixel Count'] = df['Size']
            logging.info("Used Size column as Pixel Count")
        else:
            df['Pixel Count'] = 1
            logging.info("Added default Pixel Count = 1")

    return df


def extract_ct_features(electrode_coords, ct_file):
    """Extract CT intensity features."""
    if ct_file is None:
        logging.info("No CT file provided - using NaN for CT features")
        n_electrodes = len(electrode_coords)
        return {
            'CT_mean_intensity': [np.nan] * n_electrodes,
            'CT_std_intensity': [np.nan] * n_electrodes,
            'CT_max_intensity': [np.nan] * n_electrodes,
            'CT_min_intensity': [np.nan] * n_electrodes
        }

    try:
        logging.info("Extracting CT intensity features...")
        ct_data, sitk_image = load_nrrd_file_with_sitk(ct_file)
        if ct_data is None:
            raise Exception("Failed to load CT data")

        voxel_coords = ras_to_voxel_coordinates_with_sitk(electrode_coords, sitk_image)

        intensity_results = analyze_electrode_intensities(
            ct_data, voxel_coords, radius=2,
            metrics=['mean', 'std', 'min', 'max']
        )

        # Handle case where some electrodes are outside CT bounds
        n_electrodes = len(electrode_coords)
        ct_features = {
            'CT_mean_intensity': [np.nan] * n_electrodes,
            'CT_std_intensity': [np.nan] * n_electrodes,
            'CT_max_intensity': [np.nan] * n_electrodes,
            'CT_min_intensity': [np.nan] * n_electrodes
        }

        # Fill in values for electrodes that were successfully analyzed
        for i, electrode_id in enumerate(intensity_results['electrode_id']):
            if electrode_id < n_electrodes:
                ct_features['CT_mean_intensity'][electrode_id] = intensity_results['mean_intensity'][i]
                ct_features['CT_std_intensity'][electrode_id] = intensity_results['std_intensity'][i]
                ct_features['CT_max_intensity'][electrode_id] = intensity_results['max_intensity'][i]
                ct_features['CT_min_intensity'][electrode_id] = intensity_results['min_intensity'][i]

        logging.info("✓ CT intensity features extracted")
        return ct_features

    except Exception as e:
        logging.error(f"CT analysis failed: {e}")
        n_electrodes = len(electrode_coords)
        return {
            'CT_mean_intensity': [np.nan] * n_electrodes,
            'CT_std_intensity': [np.nan] * n_electrodes,
            'CT_max_intensity': [np.nan] * n_electrodes,
            'CT_min_intensity': [np.nan] * n_electrodes
        }


def extract_pca_features(electrode_coords):
    """Extract PCA features from electrode coordinates."""
    try:
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(electrode_coords)

        logging.info("✓ PCA features extracted")
        return {
            'PCA1': pca_result[:, 0],
            'PCA2': pca_result[:, 1],
            'PCA3': pca_result[:, 2]
        }
    except Exception as e:
        logging.error(f"PCA analysis failed: {e}")
        n_electrodes = len(electrode_coords)
        return {
            'PCA1': np.zeros(n_electrodes),
            'PCA2': np.zeros(n_electrodes),
            'PCA3': np.zeros(n_electrodes)
        }


def compute_neighbor_features(electrode_coords, max_neighbor_distance=7):
    """Compute neighbor-based features."""
    tree = KDTree(electrode_coords)
    n_neighbors_list = []
    mean_neighbor_dist_list = []

    for coords in electrode_coords:
        distances, indices = tree.query(
            coords, k=len(electrode_coords),
            distance_upper_bound=max_neighbor_distance
        )

        # Remove self (distance = 0) and invalid distances
        valid_mask = (distances > 0) & (distances <= max_neighbor_distance)
        neighbor_dists = distances[valid_mask]

        n_neighbors = len(neighbor_dists)
        mean_dist = np.mean(neighbor_dists) if n_neighbors > 0 else np.nan

        n_neighbors_list.append(n_neighbors)
        mean_neighbor_dist_list.append(mean_dist)

    logging.info("✓ Neighbor features computed")
    return {
        'n_neighbors': n_neighbors_list,
        'mean_neighbor_dist': mean_neighbor_dist_list
    }


def compute_density_features(electrode_coords):
    """Compute local density features using KDE."""
    try:
        kde = gaussian_kde(electrode_coords.T, bw_method='scott')
        densities = kde(electrode_coords.T)
        logging.info("✓ Density features computed")
        return {'kde_density': densities}
    except Exception as e:
        logging.warning(f"Density computation failed: {e}")
        return {'kde_density': np.full(len(electrode_coords), np.nan)}


def compute_louvain_communities(electrode_coords, max_neighbor_distance=7):
    """Compute Louvain community detection."""
    try:
        G = nx.Graph()

        # Add nodes
        for i in range(len(electrode_coords)):
            G.add_node(i)

        # Add edges based on proximity
        tree = KDTree(electrode_coords)
        for i, coords in enumerate(electrode_coords):
            distances, indices = tree.query(
                coords, k=len(electrode_coords),
                distance_upper_bound=max_neighbor_distance
            )

            valid_neighbors = indices[1:][distances[1:] <= max_neighbor_distance]
            for neighbor in valid_neighbors:
                if not G.has_edge(i, neighbor):
                    dist = np.linalg.norm(electrode_coords[i] - electrode_coords[neighbor])
                    G.add_edge(i, neighbor, weight=1/dist)

        # Louvain community detection
        if len(G.edges) > 0:
            communities = nx.community.louvain_communities(G, weight='weight')
            community_labels = np.zeros(len(electrode_coords), dtype=int)

            for comm_id, community in enumerate(communities):
                for node in community:
                    community_labels[node] = comm_id
        else:
            community_labels = np.zeros(len(electrode_coords), dtype=int)

        logging.info("✓ Louvain communities computed")
        return {'Louvain_Community': community_labels}

    except Exception as e:
        logging.warning(f"Community detection failed: {e}")
        return {'Louvain_Community': np.zeros(len(electrode_coords), dtype=int)}


def compute_position_features(electrode_coords):
    """Compute position-relative features."""
    centroid = np.mean(electrode_coords, axis=0)

    dist_to_centroid = np.linalg.norm(electrode_coords - centroid, axis=1)
    x_relative = electrode_coords[:, 0] - centroid[0]
    y_relative = electrode_coords[:, 1] - centroid[1]
    z_relative = electrode_coords[:, 2] - centroid[2]

    logging.info("✓ Position features computed")
    return {
        'dist_to_centroid': dist_to_centroid,
        'x_relative': x_relative,
        'y_relative': y_relative,
        'z_relative': z_relative
    }


def extract_roi_surface_distance(electrode_coords, roi_file_path):
    """Extract distance to ROI surface."""
    if roi_file_path is None:
        logging.info("No ROI file provided - using NaN for surface distance")
        return {'dist_to_surface': np.full(len(electrode_coords), np.nan)}

    try:
        logging.info("Computing distance to ROI surface...")

        # Load ROI file
        roi_image = sitk.ReadImage(str(roi_file_path))
        roi_array = sitk.GetArrayFromImage(roi_image)

        # Get spacing
        spacing = roi_image.GetSpacing()
        spacing_zyx = (spacing[2], spacing[1], spacing[0])

        # Extract surface using marching cubes
        binary_array = (roi_array > 0).astype(np.uint8)
        if binary_array.sum() == 0:
            logging.warning("ROI array is empty")
            return {'dist_to_surface': np.full(len(electrode_coords), np.nan)}

        vertices, faces, _, _ = marching_cubes(binary_array, level=0.5, spacing=spacing_zyx)

        # Transform vertices to RAS coordinates
        surface_points_ras = []
        for vertex in vertices:
            # Convert from numpy (z,y,x) to SimpleITK (x,y,z) index coordinates
            idx_coord = (float(vertex[2]), float(vertex[1]), float(vertex[0]))

            # Transform to physical space (LPS)
            physical_point = roi_image.TransformContinuousIndexToPhysicalPoint(idx_coord)

            # Convert LPS to RAS
            ras_coord = (-physical_point[0], -physical_point[1], physical_point[2])
            surface_points_ras.append(ras_coord)

        surface_points_ras = np.array(surface_points_ras)

        # Calculate distances
        surface_tree = KDTree(surface_points_ras)
        distances = []
        for coords in electrode_coords:
            dist, _ = surface_tree.query(coords, k=1)
            distances.append(float(dist))

        logging.info("✓ ROI surface distances computed")
        return {'dist_to_surface': np.array(distances)}

    except Exception as e:
        logging.error(f"ROI surface analysis failed: {e}")
        return {'dist_to_surface': np.full(len(electrode_coords), np.nan)}


def compute_hemisphere_features(electrode_coords):
    """Compute hemisphere-related features."""
    hemisphere = ['Right' if x > 0 else 'Left' for x in electrode_coords[:, 0]]

    logging.info("✓ Hemisphere features computed")
    return {'Hemisphere': hemisphere}


def extract_all_target_features(
    results_file,
    ct_file=None,
    roi_file_path=None,
    patient_id=None,
    mask_id=None,
    output_dir=None,
    max_neighbor_distance=7,
    columns_success='all'
):
    """
    Extract all target features and save to CSV.

    Returns DataFrame with exactly these features:
    - Numerical: RAS_X, RAS_Y, RAS_Z, CT_mean_intensity, CT_std_intensity, PCA1, PCA2, PCA3,
                dist_to_surface, mean_neighbor_dist, kde_density, n_neighbors, Louvain_Community,
                Pixel Count, dist_to_centroid, CT_max_intensity, CT_min_intensity,
                x_relative, y_relative, z_relative
    - Categorical: Hemisphere, has_neighbors
    """

    # Setup
    if output_dir is None:
        output_dir = f"features_{patient_id}" if patient_id else "features"
    setup_logging(output_dir)

    # Load and parse data
    df = load_and_parse_coordinates(results_file, patient_id, columns_success)
    electrode_coords = df[['RAS_X', 'RAS_Y', 'RAS_Z']].values

    logging.info(f"Processing {len(electrode_coords)} electrodes")

    # Initialize target features DataFrame
    target_features = pd.DataFrame()

    # ==============================================
    # CORE COORDINATES
    # ==============================================
    target_features['RAS_X'] = df['RAS_X'].values
    target_features['RAS_Y'] = df['RAS_Y'].values
    target_features['RAS_Z'] = df['RAS_Z'].values
    logging.info("✓ Added spatial coordinates")

    # ==============================================
    # CT INTENSITY FEATURES
    # ==============================================
    ct_features = extract_ct_features(electrode_coords, ct_file)
    target_features['CT_mean_intensity'] = ct_features['CT_mean_intensity']
    target_features['CT_std_intensity'] = ct_features['CT_std_intensity']
    target_features['CT_max_intensity'] = ct_features['CT_max_intensity']
    target_features['CT_min_intensity'] = ct_features['CT_min_intensity']

    # ==============================================
    # PCA FEATURES
    # ==============================================
    pca_features = extract_pca_features(electrode_coords)
    target_features['PCA1'] = pca_features['PCA1']
    target_features['PCA2'] = pca_features['PCA2']
    target_features['PCA3'] = pca_features['PCA3']

    # ==============================================
    # ROI SURFACE DISTANCE
    # ==============================================
    surface_features = extract_roi_surface_distance(electrode_coords, roi_file_path)
    target_features['dist_to_surface'] = surface_features['dist_to_surface']

    # ==============================================
    # NEIGHBOR FEATURES
    # ==============================================
    neighbor_features = compute_neighbor_features(electrode_coords, max_neighbor_distance)
    target_features['mean_neighbor_dist'] = neighbor_features['mean_neighbor_dist']
    target_features['n_neighbors'] = neighbor_features['n_neighbors']

    # ==============================================
    # DENSITY FEATURES
    # ==============================================
    density_features = compute_density_features(electrode_coords)
    target_features['kde_density'] = density_features['kde_density']

    # ==============================================
    # COMMUNITY FEATURES
    # ==============================================
    community_features = compute_louvain_communities(electrode_coords, max_neighbor_distance)
    target_features['Louvain_Community'] = community_features['Louvain_Community']

    # ==============================================
    # PIXEL COUNT (from original data)
    # ==============================================
    target_features['Pixel Count'] = df['Pixel Count'].values

    # ==============================================
    # POSITION FEATURES
    # ==============================================
    position_features = compute_position_features(electrode_coords)
    target_features['dist_to_centroid'] = position_features['dist_to_centroid']
    target_features['x_relative'] = position_features['x_relative']
    target_features['y_relative'] = position_features['y_relative']
    target_features['z_relative'] = position_features['z_relative']

    # ==============================================
    # CATEGORICAL FEATURES
    # ==============================================
    hemisphere_features = compute_hemisphere_features(electrode_coords)
    target_features['Hemisphere'] = hemisphere_features['Hemisphere']

    # has_neighbors derived from n_neighbors
    target_features['has_neighbors'] = (target_features['n_neighbors'] > 0)

    # ==============================================
    # METADATA (for tracking)
    # ==============================================
    target_features['Patient ID'] = df['Patient ID'].values
    if mask_id is not None:
        target_features['Mask'] = mask_id
    elif 'Mask' in df.columns:
        target_features['Mask'] = df['Mask'].values
    else:
        target_features['Mask'] = 'default'

    # ==============================================
    # CONVERT CATEGORICAL FEATURES
    # ==============================================
    target_features['Hemisphere'] = target_features['Hemisphere'].astype('category')
    target_features['has_neighbors'] = target_features['has_neighbors'].astype('category')

    # ==============================================
    # SAVE TO CSV
    # ==============================================
    output_file = os.path.join(output_dir, f"target_features_{patient_id}_{mask_id}.csv")
    target_features.to_csv(output_file, index=False)

    logging.info(f"✓ Saved target features to {output_file}")
    logging.info(f"Final dataset shape: {target_features.shape}")

    # Verify we have all expected features
    expected_numerical = [
        'RAS_X', 'RAS_Y', 'RAS_Z', 'CT_mean_intensity', 'CT_std_intensity',
        'PCA1', 'PCA2', 'PCA3', 'dist_to_surface', 'mean_neighbor_dist',
        'kde_density', 'n_neighbors', 'Louvain_Community', 'Pixel Count',
        'dist_to_centroid', 'CT_max_intensity', 'CT_min_intensity',
        'x_relative', 'y_relative', 'z_relative'
    ]

    expected_categorical = ['Hemisphere', 'has_neighbors']

    missing_features = []
    for feature in expected_numerical + expected_categorical:
        if feature not in target_features.columns:
            missing_features.append(feature)

    if missing_features:
        logging.warning(f"Missing expected features: {missing_features}")
    else:
        logging.info("✓ All expected target features present")

    # Print feature summary
    print(f"\n{'='*50}")
    print(f"FEATURE EXTRACTION COMPLETE")
    print(f"{'='*50}")
    print(f"Patient ID: {patient_id}")
    print(f"Mask ID: {mask_id}")
    print(f"Electrodes processed: {len(target_features)}")
    print(f"Output file: {output_file}")
    print(f"Features extracted: {len(target_features.columns)}")
    print(f"\nNumerical features ({len(expected_numerical)}):")
    for feature in expected_numerical:
        if feature in target_features.columns:
            non_null = target_features[feature].notna().sum()
            print(f"  ✓ {feature}: {non_null}/{len(target_features)} non-null")
        else:
            print(f"  ✗ {feature}: MISSING")

    print(f"\nCategorical features ({len(expected_categorical)}):")
    for feature in expected_categorical:
        if feature in target_features.columns:
            print(f"  ✓ {feature}: {target_features[feature].dtype}")
        else:
            print(f"  ✗ {feature}: MISSING")

    return target_features


def extract_features_multiple_files(
    results_file_folders,
    patient_id,
    output_base_dir,
    ct_file=None,
    roi_file_path=None,
    max_neighbor_distance=7,
    columns_success='all'
):
    """
    Extract features from multiple CSV files (matching your original structure).
    """
    os.makedirs(output_base_dir, exist_ok=True)

    # Find all CSV files
    csv_files = []
    for root, dirs, files in os.walk(results_file_folders):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    print(f"Found {len(csv_files)} CSV files to process")

    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in {results_file_folders}")

    all_features = []

    # Process each CSV file
    for i, csv_file in enumerate(csv_files):
        print(f"\nProcessing file {i+1}/{len(csv_files)}: {csv_file}")

        # Create output directory for this mask
        mask_output_dir = os.path.join(output_base_dir, f"{patient_id}_mask_{i}")

        try:
            features_df = extract_all_target_features(
                results_file=csv_file,
                ct_file=ct_file,
                roi_file_path=roi_file_path,
                patient_id=patient_id,
                mask_id=f"mask_{i}",
                output_dir=mask_output_dir,
                max_neighbor_distance=max_neighbor_distance,
                columns_success=columns_success
            )

            # Add source file info
            features_df['source_file'] = os.path.basename(csv_file)
            features_df['mask_id'] = i

            all_features.append(features_df)
            print(f"✓ Successfully processed {len(features_df)} electrodes")

        except Exception as e:
            print(f"✗ Error processing {csv_file}: {e}")
            continue

    # Combine all datasets
    if len(all_features) > 0:
        combined_df = pd.concat(all_features, ignore_index=True)

        # Save combined dataset
        combined_output_path = os.path.join(output_base_dir, f"{patient_id}_combined_target_features.csv")
        combined_df.to_csv(combined_output_path, index=False)

        print(f"\n{'='*60}")
        print(f"MULTI-FILE PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Files processed: {len(all_features)}")
        print(f"Total electrodes: {len(combined_df)}")
        print(f"Combined output: {combined_output_path}")
        print(f"Features per electrode: {len(combined_df.columns)}")

        return all_features, combined_df
    else:
        raise ValueError("No files were successfully processed")

if __name__ == "__main__":
        # Multiple files processing (your original workflow)
        individual_dfs, combined_df = extract_features_multiple_files(
            results_file_folders=r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Centroids_model\DATA\P8\P8_RESULTS\m_g",
            patient_id="P8",
            output_base_dir=r"C:\Users\rocia\Downloads\TFG\Cohort\Extension\Feature_Extraction",
            ct_file=r"C:\Users\rocia\Downloads\TFG\Cohort\Enhance_ctp_tests\p8\TH45_histograms_FILTERED\Filtered_roi_volume_8_CTp.3D.nrrd",
            roi_file_path=r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Model_brain_mask\Dataset\MASK\patient8_mask_5.nrrd"
        )

#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Centroids_pipeline\centroids_feature_extraction.py').read())