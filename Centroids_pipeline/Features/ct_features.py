import os
import numpy as np
import pandas as pd
import nrrd
import nibabel as nib
from skimage.filters import threshold_otsu
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import logging
import SimpleITK as sitk
import ast

def load_electrode_coordinates(results_file, patient_id=None, success_type = None):
    df = pd.read_csv(results_file)

    if patient_id is not None:
        df = df[df['Patient ID'] == patient_id]
    if success_type=='yes':
      success_df = df[df['Success'] == True].copy()
    else:
      success_df = df.copy()

    if success_df.empty:
        raise ValueError("No successful electrode detections found.")

    if {'RAS_X', 'RAS_Y', 'RAS_Z'}.issubset(success_df.columns):
        # Columns already exist, nothing to do
        pass
    elif 'RAS Coordinates' in success_df.columns:
        # Parse coordinates safely
        coords = success_df['RAS Coordinates'].apply(ast.literal_eval)
        success_df['RAS_X'] = coords.apply(lambda x: x[0])
        success_df['RAS_Y'] = coords.apply(lambda x: x[1])
        success_df['RAS_Z'] = coords.apply(lambda x: x[2])
    else:
        raise ValueError("No coordinate columns found (neither RAS Coordinates nor RAS_X/Y/Z).")

    # Extract as NumPy array
    electrode_coords = success_df[['RAS_X', 'RAS_Y', 'RAS_Z']].values

    return electrode_coords, success_df

def load_nrrd_file_with_sitk(file_path):
    """
    Load NRRD file using SimpleITK.

    Parameters:
    -----------
    file_path : str
        Path to the NRRD file

    Returns:
    --------
    tuple
        (numpy.ndarray, SimpleITK.Image) containing the image data and SimpleITK image object
    """
    try:
        # Read the image using SimpleITK
        sitk_image = sitk.ReadImage(str(file_path))

        # Convert to numpy array (note: axes order may be different from nrrd.read)
        image_array = sitk.GetArrayFromImage(sitk_image)

        logging.info(f"Successfully loaded NRRD file: {file_path}")
        logging.info(f"Image size: {sitk_image.GetSize()}")
        logging.info(f"Image spacing: {sitk_image.GetSpacing()}")
        logging.info(f"Image origin: {sitk_image.GetOrigin()}")
        logging.info(f"Image direction: {sitk_image.GetDirection()}")

        return image_array, sitk_image
    except Exception as e:
        logging.error(f"Failed to read NRRD file {file_path}: {e}")
        return None, None

def ras_to_voxel_coordinates_with_sitk(ras_coords, sitk_image):
    """
    Convert RAS coordinates to voxel indices using SimpleITK.

    Parameters:
    -----------
    ras_coords : numpy.ndarray
        Array of RAS coordinates
    sitk_image : SimpleITK.Image
        SimpleITK image object

    Returns:
    --------
    numpy.ndarray
        Array of voxel coordinates
    """
    voxel_coords = []

    for ras in ras_coords:
        try:
            # Convert from RAS to LPS (SimpleITK uses LPS coordinates)
            lps_coord = (-ras[0], -ras[1], ras[2])

            # Transform physical point to index
            idx_coord = sitk_image.TransformPhysicalPointToIndex(lps_coord)

            # Convert SimpleITK (x,y,z) index to numpy (z,y,x) index
            # Note: For display and analysis purposes, we often use numpy ordering
            numpy_idx = (idx_coord[2], idx_coord[1], idx_coord[0])

            voxel_coords.append(numpy_idx)

        except Exception as e:
            logging.error(f"Error converting RAS coordinate {ras} to voxel: {e}")
            # Append a placeholder or None if conversion fails
            voxel_coords.append((0, 0, 0))

    return np.array(voxel_coords)

def fallback_ras_to_voxel_coordinates(ras_coords, nrrd_header):
    """
    Fallback method to convert RAS coordinates to voxel indices using NRRD header.

    Parameters:
    -----------
    ras_coords : numpy.ndarray
        Array of RAS coordinates
    nrrd_header : dict
        NRRD header information

    Returns:
    --------
    numpy.ndarray
        Array of voxel coordinates
    """
    # Extract transformation information from NRRD header
    space_directions = np.array(nrrd_header.get('space directions', np.eye(3)))
    space_origin = np.array(nrrd_header.get('space origin', np.zeros(3)))

    # Create transformation matrix
    if space_directions.shape == (3, 3):
        # Standard 3D transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = space_directions
        transform[:3, 3] = space_origin
    else:
        # Fallback to identity transformation
        transform = np.eye(4)
        transform[:3, 3] = space_origin

    # Invert the transformation matrix
    transform_inv = np.linalg.inv(transform)

    # Convert RAS coordinates to voxel indices
    voxel_coords = []
    for ras in ras_coords:
        ras_point = np.append(ras, 1)  # Homogeneous coordinates
        voxel_point = np.dot(transform_inv, ras_point)[:3]
        voxel_coords.append(voxel_point)

    return np.array(voxel_coords)


def analyze_electrode_intensities(ct_data, coords, radius=3, metrics=None):
    if metrics is None:
        metrics = ['mean', 'median', 'std', 'min', 'max', 'gradient', 'homogeneity']  

    # Initialize dictionary to store results
    results = {
        'electrode_id': [],
        'centroid_x': [],
        'centroid_y': [],
        'centroid_z': [],
        'centroid_intensity': [],
        'mean_intensity': [],
        'median_intensity': [],
        'std_intensity': [],
        'min_intensity': [],
        'max_intensity': [],
        'gradient_magnitude': [],
        'homogeneity_score': [],
    }

    # Create a spherical kernel for the neighborhood analysis
    kernel = np.zeros((2*radius+1, 2*radius+1, 2*radius+1))
    for x in range(2*radius+1):
        for y in range(2*radius+1):
            for z in range(2*radius+1):
                if ((x-radius)**2 + (y-radius)**2 + (z-radius)**2) <= radius**2:
                    kernel[x, y, z] = 1

    # Analyze each electrode
    for i, coord in enumerate(coords):
        x, y, z = np.round(coord).astype(int)

        # Skip if coordinates are outside the CT volume
        if (x < radius or y < radius or z < radius or
            x >= ct_data.shape[0]-radius or
            y >= ct_data.shape[1]-radius or
            z >= ct_data.shape[2]-radius):
            continue

        # Extract the neighborhood around the centroid
        x_min, x_max = max(0, x-radius), min(ct_data.shape[0], x+radius+1)
        y_min, y_max = max(0, y-radius), min(ct_data.shape[1], y+radius+1)
        z_min, z_max = max(0, z-radius), min(ct_data.shape[2], z+radius+1)

        neighborhood = ct_data[x_min:x_max, y_min:y_max, z_min:z_max]

        # Calculate the kernel mask for this neighborhood (in case it's at the edge)
        actual_kernel = kernel[:x_max-x_min, :y_max-y_min, :z_max-z_min]

        # Mask the neighborhood with the kernel
        masked_neighborhood = neighborhood * (actual_kernel > 0)
        valid_voxels = masked_neighborhood[actual_kernel > 0]

        if len(valid_voxels) == 0:
            continue

        # Calculate intensity metrics
        results['electrode_id'].append(i)
        results['centroid_x'].append(x)
        results['centroid_y'].append(y)
        results['centroid_z'].append(z)
        results['centroid_intensity'].append(float(ct_data[x, y, z]))

        if 'mean' in metrics:
            results['mean_intensity'].append(float(np.mean(valid_voxels)))
        else:
            results['mean_intensity'].append(None)

        if 'median' in metrics:
            results['median_intensity'].append(float(np.median(valid_voxels)))
        else:
            results['median_intensity'].append(None)

        if 'std' in metrics:
            results['std_intensity'].append(float(np.std(valid_voxels)))
        else:
            results['std_intensity'].append(None)

        if 'min' in metrics:
            results['min_intensity'].append(float(np.min(valid_voxels)))
        else:
            results['min_intensity'].append(None)

        if 'max' in metrics:
            results['max_intensity'].append(float(np.max(valid_voxels)))
        else:
            results['max_intensity'].append(None)

        # Calculate gradient magnitude (edge detection)
        if 'gradient' in metrics:
            gradient_x = ndimage.sobel(neighborhood, axis=0)
            gradient_y = ndimage.sobel(neighborhood, axis=1)
            gradient_z = ndimage.sobel(neighborhood, axis=2)
            gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2 + gradient_z**2)
            mean_gradient = np.mean(gradient_mag * (actual_kernel > 0))
            results['gradient_magnitude'].append(float(mean_gradient))
        else:
            results['gradient_magnitude'].append(None)

        # Calculate homogeneity (lower std/mean ratio indicates more homogeneous)
        if 'homogeneity' in metrics:
            mean_val = np.mean(valid_voxels)
            if mean_val > 0:
                homogeneity = 1 - (np.std(valid_voxels) / mean_val)
            else:
                homogeneity = 0
            results['homogeneity_score'].append(float(homogeneity))
        else:
            results['homogeneity_score'].append(None)
            


    return results


def visualize_electrode_centroids(ct_data, voxel_coords, classifications=None, slice_indices=None, output_dir=None, patient_id=None):
    """
    Visualize electrode centroids overlaid on CT scan slices.

    Parameters:
    -----------
    ct_data : numpy.ndarray
        3D array containing the CT scan data
    voxel_coords : numpy.ndarray
        Array of shape (n, 3) containing the 3D coordinates of electrode centroids
    classifications : list, optional
        List of classifications for each electrode (e.g., 'high_intensity', 'low_intensity')
    slice_indices : dict, optional
        Dictionary with keys 'axial', 'coronal', 'sagittal' containing indices for slices to display
        If None, will use the middle slice and slices with the most electrodes
    output_dir : str, optional
        Directory to save output files
    patient_id : str, optional
        Patient ID for labeling outputs

    Returns:
    --------
    dict
        Dictionary containing paths to saved visualization images
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Create output directory if needed
    if output_dir is None:
        output_dir = f"electrode_visualizations_{patient_id if patient_id else 'all'}"
    os.makedirs(output_dir, exist_ok=True)

    # Round coordinates to integers for indexing
    coords_int = np.round(voxel_coords).astype(int)

    # Filter out coordinates outside the image dimensions
    valid_mask = (
        (coords_int[:, 0] >= 0) & (coords_int[:, 0] < ct_data.shape[0]) &
        (coords_int[:, 1] >= 0) & (coords_int[:, 1] < ct_data.shape[1]) &
        (coords_int[:, 2] >= 0) & (coords_int[:, 2] < ct_data.shape[2])
    )

    coords_int = coords_int[valid_mask]
    if classifications is not None:
        classifications = [classifications[i] for i in range(len(valid_mask)) if valid_mask[i]]

    # Determine which slices to visualize if not provided
    if slice_indices is None:
        # Find the middle slices
        middle_slices = {
            'axial': ct_data.shape[2] // 2,
            'coronal': ct_data.shape[1] // 2,
            'sagittal': ct_data.shape[0] // 2
        }

        # Find slices with the most electrodes
        axial_counts = np.bincount(coords_int[:, 2], minlength=ct_data.shape[2])
        coronal_counts = np.bincount(coords_int[:, 1], minlength=ct_data.shape[1])
        sagittal_counts = np.bincount(coords_int[:, 0], minlength=ct_data.shape[0])

        max_electrode_slices = {
            'axial': np.argmax(axial_counts),
            'coronal': np.argmax(coronal_counts),
            'sagittal': np.argmax(sagittal_counts)
        }

        slice_indices = {
            'axial': [middle_slices['axial'], max_electrode_slices['axial']],
            'coronal': [middle_slices['coronal'], max_electrode_slices['coronal']],
            'sagittal': [middle_slices['sagittal'], max_electrode_slices['sagittal']]
        }

    # Initialize results dictionary
    visualization_paths = {
        'axial': [],
        'coronal': [],
        'sagittal': []
    }

    # Set up colormaps
    ct_cmap = plt.cm.gray
    if classifications is not None:
        # Create a colormap for classifications
        class_colors = {
            'high_intensity': 'red',
            'low_intensity': 'blue',
            'unknown': 'green'
        }
    else:
        # Default to a single color for all electrodes
        marker_color = 'red'

    # Visualize axial slices (top-down view)
    for z_slice in slice_indices['axial']:
        if z_slice < 0 or z_slice >= ct_data.shape[2]:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(ct_data[:, :, z_slice].T, cmap=ct_cmap, origin='lower')

        # Find electrodes in this slice (or near it)
        slice_tolerance = 3  # Include electrodes within 3 slices
        slice_electrodes = np.where(np.abs(coords_int[:, 2] - z_slice) <= slice_tolerance)[0]

        # Plot electrodes
        for i, idx in enumerate(slice_electrodes):
            x, y, z = coords_int[idx]
            distance_to_slice = abs(z - z_slice)

            # Adjust marker size based on distance to slice
            marker_size = 100 * (1 - distance_to_slice / (slice_tolerance + 1))

            if classifications is not None:
                color = class_colors.get(classifications[idx], 'green')
            else:
                color = marker_color

            # Use different marker style for electrodes that are off-slice
            marker_style = 'o' if distance_to_slice == 0 else '+'
            alpha = 1.0 if distance_to_slice == 0 else 0.5

            ax.scatter(x, y, marker=marker_style, color=color, s=marker_size, alpha=alpha,
                       edgecolors='white', linewidth=0.5)
            ax.text(x+2, y+2, str(idx), fontsize=8, color='white')

        ax.set_title(f"Axial Slice {z_slice}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Add a legend if classifications are provided
        if classifications is not None:
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=color, markersize=8, label=label)
                               for label, color in class_colors.items()]
            ax.legend(handles=legend_elements, loc='upper right')

        # Save the figure
        output_file = os.path.join(output_dir,
                                   f"{'patient_' + patient_id if patient_id else 'all'}_axial_slice_{z_slice}.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        visualization_paths['axial'].append(output_file)

    # Visualize coronal slices (front view)
    for y_slice in slice_indices['coronal']:
        if y_slice < 0 or y_slice >= ct_data.shape[1]:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(ct_data[:, y_slice, :].T, cmap=ct_cmap, origin='lower')

        # Find electrodes in this slice (or near it)
        slice_tolerance = 3
        slice_electrodes = np.where(np.abs(coords_int[:, 1] - y_slice) <= slice_tolerance)[0]

        # Plot electrodes
        for i, idx in enumerate(slice_electrodes):
            x, y, z = coords_int[idx]
            distance_to_slice = abs(y - y_slice)

            # Adjust marker size based on distance to slice
            marker_size = 100 * (1 - distance_to_slice / (slice_tolerance + 1))

            if classifications is not None:
                color = class_colors.get(classifications[idx], 'green')
            else:
                color = marker_color

            # Use different marker style for electrodes that are off-slice
            marker_style = 'o' if distance_to_slice == 0 else '+'
            alpha = 1.0 if distance_to_slice == 0 else 0.5

            ax.scatter(x, z, marker=marker_style, color=color, s=marker_size, alpha=alpha,
                       edgecolors='white', linewidth=0.5)
            ax.text(x+2, z+2, str(idx), fontsize=8, color='white')

        ax.set_title(f"Coronal Slice {y_slice}")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")

        # Add a legend if classifications are provided
        if classifications is not None:
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=color, markersize=8, label=label)
                               for label, color in class_colors.items()]
            ax.legend(handles=legend_elements, loc='upper right')

        # Save the figure
        output_file = os.path.join(output_dir,
                                   f"{'patient_' + patient_id if patient_id else 'all'}_coronal_slice_{y_slice}.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        visualization_paths['coronal'].append(output_file)

    # Visualize sagittal slices (side view)
    for x_slice in slice_indices['sagittal']:
        if x_slice < 0 or x_slice >= ct_data.shape[0]:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(ct_data[x_slice, :, :].T, cmap=ct_cmap, origin='lower')

        # Find electrodes in this slice (or near it)
        slice_tolerance = 3
        slice_electrodes = np.where(np.abs(coords_int[:, 0] - x_slice) <= slice_tolerance)[0]

        # Plot electrodes
        for i, idx in enumerate(slice_electrodes):
            x, y, z = coords_int[idx]
            distance_to_slice = abs(x - x_slice)

            # Adjust marker size based on distance to slice
            marker_size = 100 * (1 - distance_to_slice / (slice_tolerance + 1))

            if classifications is not None:
                color = class_colors.get(classifications[idx], 'green')
            else:
                color = marker_color

            # Use different marker style for electrodes that are off-slice
            marker_style = 'o' if distance_to_slice == 0 else '+'
            alpha = 1.0 if distance_to_slice == 0 else 0.5

            ax.scatter(y, z, marker=marker_style, color=color, s=marker_size, alpha=alpha,
                       edgecolors='white', linewidth=0.5)
            ax.text(y+2, z+2, str(idx), fontsize=8, color='white')

        ax.set_title(f"Sagittal Slice {x_slice}")
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")

        # Add a legend if classifications are provided
        if classifications is not None:
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=color, markersize=8, label=label)
                               for label, color in class_colors.items()]
            ax.legend(handles=legend_elements, loc='upper right')

        # Save the figure
        output_file = os.path.join(output_dir,
                                   f"{'patient_' + patient_id if patient_id else 'all'}_sagittal_slice_{x_slice}.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        visualization_paths['sagittal'].append(output_file)

    # Create a multi-panel visualization with best slices from each plane
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Axial (top-down view)
    best_axial = slice_indices['axial'][1]  # Use the slice with most electrodes
    axes[0].imshow(ct_data[:, :, best_axial].T, cmap=ct_cmap, origin='lower')
    axes[0].set_title(f"Axial Slice {best_axial}")

    # Find electrodes in this slice (or near it)
    slice_tolerance = 3
    slice_electrodes = np.where(np.abs(coords_int[:, 2] - best_axial) <= slice_tolerance)[0]

    # Plot electrodes
    for i, idx in enumerate(slice_electrodes):
        x, y, z = coords_int[idx]
        distance_to_slice = abs(z - best_axial)

        # Adjust marker size based on distance to slice
        marker_size = 100 * (1 - distance_to_slice / (slice_tolerance + 1))

        if classifications is not None:
            color = class_colors.get(classifications[idx], 'green')
        else:
            color = marker_color

        marker_style = 'o' if distance_to_slice == 0 else '+'
        alpha = 1.0 if distance_to_slice == 0 else 0.5

        axes[0].scatter(x, y, marker=marker_style, color=color, s=marker_size, alpha=alpha,
                   edgecolors='white', linewidth=0.5)

    # Coronal (front view)
    best_coronal = slice_indices['coronal'][1]
    axes[1].imshow(ct_data[:, best_coronal, :].T, cmap=ct_cmap, origin='lower')
    axes[1].set_title(f"Coronal Slice {best_coronal}")

    # Find electrodes in this slice
    slice_electrodes = np.where(np.abs(coords_int[:, 1] - best_coronal) <= slice_tolerance)[0]

    # Plot electrodes
    for i, idx in enumerate(slice_electrodes):
        x, y, z = coords_int[idx]
        distance_to_slice = abs(y - best_coronal)

        marker_size = 100 * (1 - distance_to_slice / (slice_tolerance + 1))

        if classifications is not None:
            color = class_colors.get(classifications[idx], 'green')
        else:
            color = marker_color

        marker_style = 'o' if distance_to_slice == 0 else '+'
        alpha = 1.0 if distance_to_slice == 0 else 0.5

        axes[1].scatter(x, z, marker=marker_style, color=color, s=marker_size, alpha=alpha,
                   edgecolors='white', linewidth=0.5)

    # Sagittal (side view)
    best_sagittal = slice_indices['sagittal'][1]
    axes[2].imshow(ct_data[best_sagittal, :, :].T, cmap=ct_cmap, origin='lower')
    axes[2].set_title(f"Sagittal Slice {best_sagittal}")

    # Find electrodes in this slice
    slice_electrodes = np.where(np.abs(coords_int[:, 0] - best_sagittal) <= slice_tolerance)[0]

    # Plot electrodes
    for i, idx in enumerate(slice_electrodes):
        x, y, z = coords_int[idx]
        distance_to_slice = abs(x - best_sagittal)

        marker_size = 100 * (1 - distance_to_slice / (slice_tolerance + 1))

        if classifications is not None:
            color = class_colors.get(classifications[idx], 'green')
        else:
            color = marker_color

        marker_style = 'o' if distance_to_slice == 0 else '+'
        alpha = 1.0 if distance_to_slice == 0 else 0.5

        axes[2].scatter(y, z, marker=marker_style, color=color, s=marker_size, alpha=alpha,
                   edgecolors='white', linewidth=0.5)

    # Add a common legend if classifications are provided
    if classifications is not None:
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=color, markersize=8, label=label)
                           for label, color in class_colors.items()]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3)

    # Save the multi-panel figure
    multi_panel_file = os.path.join(output_dir,
                               f"{'patient_' + patient_id if patient_id else 'all'}_multi_panel_visualization.png")
    plt.tight_layout()
    plt.savefig(multi_panel_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    visualization_paths['multi_panel'] = multi_panel_file

    print(f"Visualization complete. Images saved to {output_dir}")
    return visualization_paths

def classify_electrodes_by_intensity(intensity_results, threshold_method='otsu'):
    """
    Classify electrodes based on their intensity profiles.

    Parameters:
    -----------
    intensity_results : dict
        Results dictionary from analyze_electrode_intensities
    threshold_method : str, default='otsu'
        Method to determine threshold ('otsu', 'percentile', or 'fixed')

    Returns:
    --------
    dict
        Dictionary with electrode classifications and thresholds used
    """
    mean_intensities = np.array(intensity_results['mean_intensity'])

    # Skip electrodes with no data
    valid_indices = ~np.isnan(mean_intensities)
    if np.sum(valid_indices) == 0:
        return {'classifications': [], 'threshold': 0}

    valid_intensities = mean_intensities[valid_indices]

    # Determine threshold
    if threshold_method == 'otsu':
        try:
            threshold = threshold_otsu(valid_intensities)
        except:
            # Fallback to percentile if Otsu fails
            threshold = np.percentile(valid_intensities, 50)
    elif threshold_method == 'percentile':
        threshold = np.percentile(valid_intensities, 60)  # Above 60th percentile is "high intensity"
    else:  # fixed
        threshold = 200  # Example fixed threshold, adjust based on your CT data

    # Classify electrodes
    classifications = []
    for i, intensity in enumerate(mean_intensities):
        if np.isnan(intensity):
            classifications.append('unknown')
        elif intensity > threshold:
            classifications.append('high_intensity')
        else:
            classifications.append('low_intensity')

    return {
        'classifications': classifications,
        'threshold': float(threshold)
    }

def nrrd_electrode_intensity_analysis(ct_file, results_file, patient_id=None, output_dir=None, visualize=True):
    """
    Main function to analyze electrode intensities in NRRD CT data.

    Parameters:
    -----------
    ct_file : str
        Path to the CT scan file (NRRD format)
    results_file : str
        Path to the electrode detection results file (CSV format)
    patient_id : str, optional
        Patient ID for labeling outputs
    output_dir : str, optional
        Directory to save output files
    visualize : bool, default=True
        Whether to generate visualization of electrodes on CT slices

    Returns:
    --------
    dict
        Dictionary containing all analysis results
    """
    # Load electrode coordinates
    electrode_coords, success_df = load_electrode_coordinates(results_file, patient_id)

    # Load CT data using SimpleITK (primary method)
    ct_data_sitk, sitk_image = load_nrrd_file_with_sitk(ct_file)

    # Load CT data using nrrd as fallback
    ct_data_nrrd, nrrd_header = nrrd.read(ct_file)

    # Select which CT data to use based on successful loading
    if ct_data_sitk is not None and sitk_image is not None:
        ct_data = ct_data_sitk
        # Convert RAS coordinates to voxel coordinates using SimpleITK
        voxel_coords = ras_to_voxel_coordinates_with_sitk(electrode_coords, sitk_image)
        logging.info("Using SimpleITK for coordinate transformation")
    else:
        ct_data = ct_data_nrrd
        # Fallback to using the NRRD header method
        voxel_coords = fallback_ras_to_voxel_coordinates(electrode_coords, nrrd_header)
        logging.info("Using NRRD header for coordinate transformation (fallback)")


    if ct_data_sitk is not None and sitk_image is not None:
        # SimpleITK typically returns (z, y, x) order, might need to reorient
        # Check the dimensional ordering and transpose if needed
        if ct_data.shape != ct_data_nrrd.shape:
            logging.info(f"SimpleITK shape {ct_data.shape} differs from NRRD shape {ct_data_nrrd.shape}")
            # You might need to transpose ct_data or adjust voxel_coords accordingly

    # Log some information about the coordinates
    logging.info(f"Number of electrodes: {len(electrode_coords)}")
    logging.info(f"RAS coordinate range: X[{np.min(electrode_coords[:,0]):.1f}, {np.max(electrode_coords[:,0]):.1f}], "
                f"Y[{np.min(electrode_coords[:,1]):.1f}, {np.max(electrode_coords[:,1]):.1f}], "
                f"Z[{np.min(electrode_coords[:,2]):.1f}, {np.max(electrode_coords[:,2]):.1f}]")
    logging.info(f"Voxel coordinate range: X[{np.min(voxel_coords[:,0]):.1f}, {np.max(voxel_coords[:,0]):.1f}], "
                f"Y[{np.min(voxel_coords[:,1]):.1f}, {np.max(voxel_coords[:,1]):.1f}], "
                f"Z[{np.min(voxel_coords[:,2]):.1f}, {np.max(voxel_coords[:,2]):.1f}]")

    # Check if voxel coordinates are within the CT volume dimensions
    ct_dims = ct_data.shape
    in_bounds = ((voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < ct_dims[0]) &
                 (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < ct_dims[1]) &
                 (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < ct_dims[2]))

    logging.info(f"CT data shape: {ct_dims}")
    logging.info(f"Percentage of electrodes within CT bounds: {np.mean(in_bounds)*100:.1f}%")

    if np.mean(in_bounds) < 0.5:
        logging.warning("Less than 50% of electrode coordinates are within CT volume boundaries!")
        logging.warning("This might indicate a coordinate transformation issue.")

        # Try to fix by flipping axes or adjusting the coordinate system
        # This is a simple attempt - you might need a more sophisticated approach
        trial_flips = [
            (False, False, False),  # original
            (True, False, False),   # flip x
            (False, True, False),   # flip y
            (False, False, True),   # flip z
            (True, True, False),    # flip x,y
            (True, False, True),    # flip x,z
            (False, True, True),    # flip y,z
            (True, True, True)      # flip x,y,z
        ]

        best_flip = None
        best_in_bounds_pct = np.mean(in_bounds)

        for flip_x, flip_y, flip_z in trial_flips:
            trial_coords = voxel_coords.copy()

            if flip_x:
                trial_coords[:, 0] = ct_dims[0] - trial_coords[:, 0]
            if flip_y:
                trial_coords[:, 1] = ct_dims[1] - trial_coords[:, 1]
            if flip_z:
                trial_coords[:, 2] = ct_dims[2] - trial_coords[:, 2]

            trial_in_bounds = ((trial_coords[:, 0] >= 0) & (trial_coords[:, 0] < ct_dims[0]) &
                             (trial_coords[:, 1] >= 0) & (trial_coords[:, 1] < ct_dims[1]) &
                             (trial_coords[:, 2] >= 0) & (trial_coords[:, 2] < ct_dims[2]))

            trial_pct = np.mean(trial_in_bounds)
            logging.info(f"Flips {(flip_x, flip_y, flip_z)}: {trial_pct*100:.1f}% in bounds")

            if trial_pct > best_in_bounds_pct:
                best_in_bounds_pct = trial_pct
                best_flip = (flip_x, flip_y, flip_z)

        if best_flip != (False, False, False) and best_in_bounds_pct > np.mean(in_bounds):
            logging.info(f"Using best flip configuration {best_flip} with {best_in_bounds_pct*100:.1f}% in bounds")
            flip_x, flip_y, flip_z = best_flip

            if flip_x:
                voxel_coords[:, 0] = ct_dims[0] - voxel_coords[:, 0]
            if flip_y:
                voxel_coords[:, 1] = ct_dims[1] - voxel_coords[:, 1]
            if flip_z:
                voxel_coords[:, 2] = ct_dims[2] - voxel_coords[:, 2]

            in_bounds = ((voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < ct_dims[0]) &
                         (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < ct_dims[1]) &
                         (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < ct_dims[2]))

    # Analyze electrode intensities - using your existing function
    intensity_results = analyze_electrode_intensities(
        ct_data, voxel_coords, radius=3,
        metrics=['mean', 'median', 'std', 'min', 'max', 'gradient', 'homogeneity']
    )

    # Classify electrodes based on intensity
    classifications = classify_electrodes_by_intensity(
        intensity_results, threshold_method='otsu'
    )

    # Create output directory if needed
    if output_dir is None:
        output_dir = f"electrode_intensity_analysis_{patient_id if patient_id else 'all'}"
    os.makedirs(output_dir, exist_ok=True)

    # Save results to CSV
    csv_path = os.path.join(output_dir, f"{'patient_' + patient_id if patient_id else 'all_patients'}_intensity_results.csv")

    # Create a DataFrame from the results dictionary
    results_df = pd.DataFrame({
        'Electrode_ID': intensity_results['electrode_id'],
        'Original_Row': [success_df.index[i] for i in intensity_results['electrode_id']],
        'Centroid_X': intensity_results['centroid_x'],
        'Centroid_Y': intensity_results['centroid_y'],
        'Centroid_Z': intensity_results['centroid_z'],
        'Centroid_Intensity': intensity_results['centroid_intensity'],
        'Mean_Intensity': intensity_results['mean_intensity'],
        'Median_Intensity': intensity_results['median_intensity'],
        'Std_Intensity': intensity_results['std_intensity'],
        'Min_Intensity': intensity_results['min_intensity'],
        'Max_Intensity': intensity_results['max_intensity'],
        'Gradient_Magnitude': intensity_results['gradient_magnitude'],
        'Homogeneity_Score': intensity_results['homogeneity_score'],
        'Classification': classifications['classifications']
    })

    # Save to CSV
    results_df.to_csv(csv_path, index=False)

    # Generate and save plots
    dist_plot_path = os.path.join(output_dir, f"{'patient_' + patient_id if patient_id else 'all_patients'}_intensity_distributions.png")

    # Plot intensity distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot mean intensity histogram
    axes[0, 0].hist(intensity_results['mean_intensity'], bins=20, alpha=0.7, color='blue')
    axes[0, 0].set_title('Mean Intensity Distribution')
    axes[0, 0].set_xlabel('Mean Intensity (HU)')
    axes[0, 0].set_ylabel('Frequency')

    # Plot standard deviation histogram
    axes[0, 1].hist(intensity_results['std_intensity'], bins=20, alpha=0.7, color='green')
    axes[0, 1].set_title('Intensity Standard Deviation Distribution')
    axes[0, 1].set_xlabel('Standard Deviation')
    axes[0, 1].set_ylabel('Frequency')

    # Plot gradient magnitude histogram
    axes[1, 0].hist(intensity_results['gradient_magnitude'], bins=20, alpha=0.7, color='red')
    axes[1, 0].set_title('Gradient Magnitude Distribution')
    axes[1, 0].set_xlabel('Gradient Magnitude')
    axes[1, 0].set_ylabel('Frequency')

    # Plot homogeneity score histogram
    axes[1, 1].hist(intensity_results['homogeneity_score'], bins=20, alpha=0.7, color='purple')
    axes[1, 1].set_title('Homogeneity Score Distribution')
    axes[1, 1].set_xlabel('Homogeneity Score')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')

    # Plot intensity scatter
    scatter_plot_path = os.path.join(output_dir, f"{'patient_' + patient_id if patient_id else 'all_patients'}_intensity_scatter.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Mean intensity vs. standard deviation
    sc1 = axes[0].scatter(
        intensity_results['mean_intensity'],
        intensity_results['std_intensity'],
        c=intensity_results['homogeneity_score'],
        cmap='viridis',
        alpha=0.7,
        s=50
    )
    axes[0].set_title('Mean Intensity vs. Standard Deviation')
    axes[0].set_xlabel('Mean Intensity (HU)')
    axes[0].set_ylabel('Standard Deviation')
    cbar1 = plt.colorbar(sc1, ax=axes[0])
    cbar1.set_label('Homogeneity Score')

    # Mean intensity vs. gradient magnitude
    sc2 = axes[1].scatter(
        intensity_results['mean_intensity'],
        intensity_results['gradient_magnitude'],
        c=intensity_results['electrode_id'],
        cmap='plasma',
        alpha=0.7,
        s=50
    )
    axes[1].set_title('Mean Intensity vs. Gradient Magnitude')
    axes[1].set_xlabel('Mean Intensity (HU)')
    axes[1].set_ylabel('Gradient Magnitude')
    cbar2 = plt.colorbar(sc2, ax=axes[1])
    cbar2.set_label('Electrode ID')

    plt.tight_layout()
    plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')

    # 3D visualization
    visualization_plot_path = os.path.join(output_dir, f"{'patient_' + patient_id if patient_id else 'all_patients'}_3d_visualization.png")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Get coordinates and mean intensities
    x = np.array(intensity_results['centroid_x'])
    y = np.array(intensity_results['centroid_y'])
    z = np.array(intensity_results['centroid_z'])
    intensities = np.array(intensity_results['mean_intensity'])

    # Create scatter plot
    scatter = ax.scatter(
        x, y, z,
        c=intensities,
        cmap='hot',
        s=50,
        alpha=0.8
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mean Intensity (HU)')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Electrode Positions Colored by Intensity')

    # Set axis limits
    ax.set_xlim(0, ct_data.shape[0])
    ax.set_ylim(0, ct_data.shape[1])
    ax.set_zlim(0, ct_data.shape[2])

    plt.tight_layout()
    plt.savefig(visualization_plot_path, dpi=300, bbox_inches='tight')

    # Generate CT slice visualizations with electrode overlays
    visualization_paths = None
    if visualize:
        visualization_paths = visualize_electrode_centroids(
            ct_data=ct_data,
            voxel_coords=voxel_coords,
            classifications=classifications['classifications'],
            output_dir=output_dir,
            patient_id=patient_id
        )

    # Compile all results
    analysis_results = {
        "patient_id": patient_id,
        "intensity_analysis": intensity_results,
        "electrode_classifications": classifications,
        "results_csv_path": csv_path,
        "distribution_plot_path": dist_plot_path,
        "scatter_plot_path": scatter_plot_path,
        "visualization_plot_path": visualization_plot_path,
        "ct_slice_visualizations": visualization_paths
    }

    print(f"Analysis complete. Results saved to {output_dir}")
    return analysis_results
