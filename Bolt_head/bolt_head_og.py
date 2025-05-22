import SimpleITK as sitk
import numpy as np
import slicer
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, marching_cubes
import vtk
from sklearn.decomposition import PCA
import os
from skimage import morphology
import scipy.spatial.distance as distance
from scipy.ndimage import binary_dilation, binary_erosion
import time

CONFIG = {
    'threshold_value': 2350, #P1: 2240, P4:2340, P5: 2746, P7: 2806, P8> 2416
    'min_region_size': 100,         
    'max_region_size': 800,         
    'morph_kernel_size': 1,         
    'principal_axis_length': 15,    
    'output_dir': r"C:\Users\rocia\Downloads\TFG\Cohort\Bolt_heads\P7_2350"  
}
def main():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    print("Loading volume data...")
    # start time
    start_time = time.time() 
    volume_node = slicer.util.getNode('7_CTp.3D')
    brain_mask_node = slicer.util.getNode('patient7_mask_5')
    volume_array = slicer.util.arrayFromVolume(volume_node)
    brain_mask_array = slicer.util.arrayFromVolume(brain_mask_node)
    spacing = volume_node.GetSpacing()
    origin = volume_node.GetOrigin()
    ijkToRasMatrix = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASDirectionMatrix(ijkToRasMatrix)
    volume_helper = VolumeHelper(spacing, origin, ijkToRasMatrix, CONFIG['output_dir'])
    
    print("Performing initial segmentation...")
    binary_mask = volume_array > CONFIG['threshold_value']
    volume_helper.create_volume(binary_mask.astype(np.uint8), "Threshold_Result", "P1_threshold.nrrd")
    
    print("Removing structures inside brain mask...")
    outside_brain_mask = ~brain_mask_array.astype(bool)  
    bolt_heads_mask = binary_mask & outside_brain_mask   
    volume_helper.create_volume(bolt_heads_mask.astype(np.uint8), "Outside_Brain_Result", "P1_outside_brain.nrrd")
    
    print("Applying morphological operations...")
    kernel = morphology.ball(CONFIG['morph_kernel_size'])
    cleaned_mask = morphology.binary_closing(bolt_heads_mask, kernel)
    volume_helper.create_volume(cleaned_mask.astype(np.uint8), "Cleaned_Result", "P1_cleaned.nrrd")
    if not np.any(cleaned_mask):
        print("No bolt head regions found at the given threshold outside the brain mask.")
        return

    print("Identifying and filtering bolt head components...")
    labeled_image = label(cleaned_mask)
    regions = regionprops(labeled_image)
    # Filter regions by size
    filtered_mask = np.zeros_like(labeled_image, dtype=np.uint16)
    region_info = []
    region_sizes = []
    for region in regions:
        volume = region.area
        region_sizes.append(volume)
        if CONFIG['min_region_size'] < volume < CONFIG['max_region_size']:
            filtered_mask[labeled_image == region.label] = region.label
            centroid_physical = tuple(origin[i] + region.centroid[i] * spacing[i] for i in range(3))
            coords = np.argwhere(labeled_image == region.label)
            principal_axis = calculate_principal_axis(coords, spacing)
            bolt_to_brain_center = estimate_brain_center(brain_mask_array, spacing, origin) - np.array(centroid_physical)
            if np.dot(principal_axis, bolt_to_brain_center) < 0:
                principal_axis = -principal_axis  
            region_info.append({
                'label': region.label,
                'physical_centroid': centroid_physical,
                'volume': volume,
                'principal_axis': principal_axis
            })
    
    print(f"Found {len(region_info)} valid bolt head regions after filtering")
    volume_helper.create_volume(filtered_mask, "Filtered_Bolt_Heads", "P1_filtered_bolt_heads.nrrd")

    # Generate PRE-VALIDATION plots
    print("Generating PRE-VALIDATION visualizations...")
    plot_size_histogram(region_sizes)
    plot_bolt_vectors(region_info, filtered_mask, spacing, origin)
    plot_bolt_brain_context(region_info, filtered_mask, brain_mask_array, spacing, origin, name = 'P1_BRAIN_MASK_CONTEXT.png')
    plot_bolt_distances_and_orientations(
            region_info, 
            brain_mask_array, 
            spacing, 
            origin, 
            CONFIG['output_dir'],
            name= 'P1_bolt_spatial_analysis_wo_validation.png'
        )
    validated_regions, invalidated_regions = validate_bolt_head_in_brain_context(
        region_info, brain_mask_array, spacing, origin
    )
    print("Generating POST-VALIDATION visualizations...")
    plot_brain_context_with_validation(
        validated_regions, 
        invalidated_regions, 
        filtered_mask, 
        brain_mask_array, 
        spacing, 
        origin
    )
    print("Calculating brain entry points for validated bolt heads...")
    for info in validated_regions:
        centroid = np.array(info['physical_centroid'])
        direction = np.array(info['principal_axis'])
        direction = direction / np.linalg.norm(direction)
        entry_point, distance = calculate_brain_intersection(
            centroid, direction, brain_mask_array, spacing, origin
        )
        info['brain_entry_point'] = entry_point
        info['entry_distance'] = distance
    # Plotting entry points for validated regions
    plot_entry_points(validated_regions, filtered_mask, brain_mask_array, spacing, origin)
    plot_multi_view_entry_points(validated_regions, filtered_mask, brain_mask_array, spacing, origin)

    # Outlier analysis
    outliers, outlier_details = comprehensive_outlier_analysis(
        region_info, brain_mask_array, spacing, origin
    )
    generate_report(validated_regions, outliers)
    try:
        plot_bolt_distances_and_orientations(
            validated_regions, 
            brain_mask_array, 
            spacing, 
            origin, 
            CONFIG['output_dir']
        )
        generate_advanced_bolt_report(
            validated_regions, 
            brain_mask_array, 
            spacing, 
            origin, 
            CONFIG['output_dir']
        )
        print("✅ Advanced bolt head analysis completed successfully")
    except Exception as e:
        print(f"Error in advanced bolt analysis: {e}")
        import traceback
        traceback.print_exc()
    # Other plotting and analysis functions
    plot_threshold_distribution(volume_array)
    plot_segmentation_stages(volume_array, brain_mask_array, bolt_heads_mask, cleaned_mask)

    create_entry_points_volume(
        validated_regions, 
        brain_mask_array, 
        spacing, 
        origin, 
        volume_helper
    )

    end_time = time.time()
    elapsed = end_time - start_time

    # Convert to minutes and seconds
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"Duration: {minutes} minutes and {seconds} seconds")
    print("\n✅ Processing complete!")
    print(f"✅ All results saved to: {CONFIG['output_dir']}")

def estimate_brain_center(brain_mask, spacing, origin):
    coords = np.argwhere(brain_mask > 0)
    if len(coords) == 0:
        return np.array([0, 0, 0])
    center_voxel = np.mean(coords, axis=0)
    center_physical = np.array([origin[i] + center_voxel[i] * spacing[i] for i in range(3)])
    return center_physical

class VolumeHelper:
    def __init__(self, spacing, origin, direction_matrix, output_dir):
        self.spacing = spacing
        self.origin = origin
        self.direction_matrix = direction_matrix
        self.output_dir = output_dir
    def create_volume(self, array, name, save_filename=None):
        sitk_image = sitk.GetImageFromArray(array)
        sitk_image.SetSpacing(self.spacing)
        sitk_image.SetOrigin(self.origin)
        new_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
        new_node.SetIJKToRASDirectionMatrix(self.direction_matrix)
        new_node.SetOrigin(self.origin)
        slicer.util.updateVolumeFromArray(new_node, array)
        if save_filename:
            save_path = os.path.join(self.output_dir, save_filename)
            slicer.util.saveNode(new_node, save_path)
            print(f"✅ Saved {name} to {save_path}")
        return new_node
    
def calculate_brain_intersection(centroid, direction, brain_mask, spacing, origin):
    try:
        voxel_centroid = np.array([
            (centroid[i] - origin[i]) / spacing[i] for i in range(3)
        ], dtype=np.float64)
    
        shape = brain_mask.shape
        direction = direction / np.linalg.norm(direction)
        strategies = [
            {'step_size': 0.5, 'max_multiplier': 3},   # Conservative
            {'step_size': 1.0, 'max_multiplier': 5},   # Broader
            {'step_size': 0.25, 'max_multiplier': 10}  # More extensive search
        ]
        
        for strategy in strategies:
            step_size = strategy['step_size']
            max_distance = np.sqrt(sum([(shape[i] * spacing[i])**2 for i in range(3)]))
            max_iterations = int(max_distance * strategy['max_multiplier'] / step_size)
            current_pos = voxel_centroid.copy()
            last_pos = current_pos.copy()
            distance_traveled = 0
            
            for _ in range(max_iterations):
                current_pos += direction * step_size / np.array(spacing)
                distance_traveled += step_size
                
                # Round to nearest integer for mask indexing
                x, y, z = np.round(current_pos).astype(int)
                
                # Out of bounds check
                if (x < 0 or x >= shape[0] or
                    y < 0 or y >= shape[1] or
                    z < 0 or z >= shape[2]):
                    break
                    
                # Brain mask intersection
                if brain_mask[x, y, z] > 0:
                    # Interpolate intersection point
                    intersection_voxel = (current_pos + last_pos) / 2
                    intersection_point = np.array([
                        origin[i] + intersection_voxel[i] * spacing[i] for i in range(3)
                    ])
                    
                    # Add sanity checks
                    if np.linalg.norm(intersection_point - centroid) > max_distance:
                        continue
                    
                    return intersection_point, distance_traveled
                
                last_pos = current_pos.copy()
        
        print(f"No brain intersection found for bolt at {centroid}")
        return None, None
    
    except Exception as e:
        print(f"Error in calculate_brain_intersection: {e}")
        print(f"Details - Centroid: {centroid}, Direction: {direction}")
        import traceback
        traceback.print_exc()
        return None, None
    
def plot_threshold_distribution(volume_array):
    plt.figure(figsize=(10, 6))
    plt.hist(volume_array.ravel(), bins=100, color='skyblue', edgecolor='black')
    plt.title('Voxel Intensity Distribution')
    plt.xlabel('Voxel Intensity')
    plt.ylabel('Frequency')
    plt.axvline(x=CONFIG['threshold_value'], color='red', linestyle='--', 
                label=f'Threshold ({CONFIG["threshold_value"]})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CONFIG['output_dir'], "P1_intensity_distribution.png"), dpi=300)
    plt.close()

def plot_segmentation_stages(volume_array, brain_mask_array, bolt_heads_mask, cleaned_mask):
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Segmentation Process Stages', fontsize=16)
    mid_slice = volume_array.shape[0] // 2
    axs[0, 0].imshow(volume_array[mid_slice], cmap='gray')
    axs[0, 0].set_title('Original Volume')
    axs[0, 1].imshow(brain_mask_array[mid_slice], cmap='viridis')     # Brain Mask
    axs[0, 1].set_title('Brain Mask')
    axs[1, 0].imshow(bolt_heads_mask[mid_slice], cmap='hot')   # Thresholded Bolt Heads
    axs[1, 0].set_title('Bolt Heads (Thresholded)')
    axs[1, 1].imshow(cleaned_mask[mid_slice], cmap='hot') # Cleaned Mask
    axs[1, 1].set_title('Cleaned Bolt Heads Mask')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], "P8_segmentation_stages.png"), dpi=300)
    plt.close()

def plot_region_characteristics(regions):
    volumes = [region.area for region in regions]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Bolt Head Region Characteristics', fontsize=16)
    axs[0].hist(volumes, bins=30, color='skyblue', edgecolor='black') # Volume distribution
    axs[0].set_title('Region Volume Distribution')
    axs[0].set_xlabel('Volume (voxels)')
    axs[0].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], "P8_region_characteristics.png"), dpi=300)
    plt.close()

def plot_entry_points(region_info, filtered_mask, brain_mask, spacing, origin):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    plot_surface(ax, brain_mask, spacing, origin, 'lightblue', 0.3)
    for info in region_info:
        plot_surface(ax, filtered_mask == info['label'], spacing, origin, 'yellow', 0.8)
        centroid = np.array(info['physical_centroid'])
        vector = np.array(info['principal_axis'])
        ax.quiver(*centroid, *vector, color='red', linewidth=2, arrow_length_ratio=0.2)
        if 'brain_entry_point' in info and info['brain_entry_point'] is not None:
            entry_point = info['brain_entry_point']
            ax.scatter(entry_point[0], entry_point[1], entry_point[2], 
                      color='green', s=100, marker='o', label='Entry Points')
 
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Bolt Heads with Brain Entry Points')
    ax.view_init(elev=30, azim=45)
    plt.savefig(os.path.join(CONFIG['output_dir'], "P8_bolt_heads_entry_points.png"), dpi=300)
    plt.close()

def calculate_principal_axis(coords, spacing):
    if len(coords) > 2:
        pca = PCA(n_components=3)
        pca.fit(coords)
        principal_axis = pca.components_[0] * spacing  
        return principal_axis / np.linalg.norm(principal_axis) * CONFIG['principal_axis_length']
    else:
        return np.array([0, 0, 1])  # Default if not enough points

def plot_size_histogram(region_sizes):
    plt.figure(figsize=(8, 6))
    plt.hist(region_sizes, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Bolt Head Region Sizes')
    plt.xlabel('Volume (voxels)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(CONFIG['output_dir'], "P8_size_distribution.png"), dpi=300)
    plt.close()

def compute_distance_to_surface(point, brain_mask, spacing, origin):
    point = np.asarray(point)
    origin = np.asarray(origin)
    spacing = np.asarray(spacing)
    voxel_point = np.round((point - origin) / spacing).astype(int)
    # Ensure point is within mask bounds
    if (np.any(voxel_point < 0) or 
        np.any(voxel_point >= np.array(brain_mask.shape))):
        return np.inf
    surface_mask = compute_surface_mask(brain_mask)
    surface_voxels = np.argwhere(surface_mask)
    # Compute distances
    if len(surface_voxels) > 0:
        surface_points_physical = surface_voxels * spacing + origin
        distances = np.min(np.linalg.norm(surface_points_physical - point, axis=1))
        return distances
    return np.inf

def detect_outliers(data, method='zscore', threshold=3):
    import numpy as np
    from scipy import stats
    data = np.array(data)    
    if method == 'zscore':
        z_scores = np.abs(stats.zscore(data))
        return np.where(z_scores > threshold)[0]
    elif method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return np.where((data < lower_bound) | (data > upper_bound))[0]
    else:
        raise ValueError("Invalid outlier detection method")

def comprehensive_outlier_analysis(region_info, brain_mask, spacing, origin):

    SURFACE_DISTANCE_THRESHOLD = 30.0  # mm
    volumes = [info['volume'] for info in region_info]
    centroids = [info['physical_centroid'] for info in region_info]
    principal_axes = [info['principal_axis'] for info in region_info]
    surface_distances = [
        compute_distance_to_surface(centroid, brain_mask, spacing, origin) 
        for centroid in centroids
    ]
    def compute_axis_angles(axes):
        angles = []
        for i in range(len(axes)):
            for j in range(i+1, len(axes)):
                axis1 = axes[i] / np.linalg.norm(axes[i])
                axis2 = axes[j] / np.linalg.norm(axes[j])
                angle = np.arccos(np.clip(np.dot(axis1, axis2), -1.0, 1.0))
                angles.append(np.degrees(angle))
        return angles
    axis_angles = compute_axis_angles(principal_axes)

    outliers = {
        'volume_outliers_zscore': detect_outliers(volumes, method='zscore'),
        'volume_outliers_iqr': detect_outliers(volumes, method='iqr'),
        'surface_distance_outliers_zscore': detect_outliers(surface_distances, method='zscore'),
        'surface_distance_outliers_iqr': detect_outliers(surface_distances, method='iqr'),
        'surface_distance_threshold_outliers': [
            i for i, dist in enumerate(surface_distances) 
            if dist > SURFACE_DISTANCE_THRESHOLD
        ]
    }
    outlier_details = {
        'volume_outliers': [
            {
                'index': idx, 
                'volume': volumes[idx], 
                'centroid': centroids[idx]
            } for idx in outliers['volume_outliers_zscore']
        ],
        'surface_distance_outliers': [
            {
                'index': idx, 
                'distance': surface_distances[idx], 
                'centroid': centroids[idx]
            } for idx in outliers['surface_distance_outliers_zscore']
        ],
        'surface_distance_threshold_outliers': [
            {
                'index': idx,
                'distance': surface_distances[idx],
                'centroid': centroids[idx]
            } for idx in outliers['surface_distance_threshold_outliers']
        ]
    }
    return outliers, outlier_details

def compute_surface_mask(mask, connectivity=1):
    dilated = binary_dilation(mask, iterations=1)
    eroded = binary_erosion(mask, iterations=1)
    return dilated ^ eroded  # XOR to get surface

def compute_axis_angles(axes):
    angles = []
    for i in range(len(axes)):
        for j in range(i+1, len(axes)):
            axis1 = axes[i] / np.linalg.norm(axes[i])
            axis2 = axes[j] / np.linalg.norm(axes[j])
            angle = np.arccos(np.clip(np.dot(axis1, axis2), -1.0, 1.0))
            angles.append(np.degrees(angle))
    return angles

def advanced_bolt_characterization(region_info, brain_mask, spacing, origin):
    centroids = np.array([info['physical_centroid'] for info in region_info])
    inter_bolt_distances = distance.pdist(centroids)
    surface_distances = []
    for centroid in centroids:
        dist = compute_distance_to_surface(centroid, brain_mask, spacing, origin)
        surface_distances.append(dist)
    
    # Orientation analysis
    principal_axes = np.array([info['principal_axis'] for info in region_info])
    axis_angles = compute_axis_angles(principal_axes)

    # Volume statistics
    volumes = [info['volume'] for info in region_info]
    return {
        'bolt_count': len(region_info),
        'inter_bolt_stats': {
            'mean_distance': np.mean(inter_bolt_distances),
            'max_distance': np.max(inter_bolt_distances),
            'min_distance': np.min(inter_bolt_distances)
        },
        'surface_distance_stats': {
            'mean': np.mean(surface_distances),
            'median': np.median(surface_distances),
            'std': np.std(surface_distances)
        },
        'volume_stats': {
            'mean': np.mean(volumes),
            'median': np.median(volumes),
            'std': np.std(volumes)
        },
        'orientation_stats': {
            'mean_angle': np.mean(axis_angles) if axis_angles else None,
            'std_angle': np.std(axis_angles) if axis_angles else None
        }
    }

def plot_bolt_distances_and_orientations(region_info, brain_mask, spacing, origin, output_dir, name = "P1_bolt_spatial_analysis.png"):
    surface_distances = []
    centroids = []
    for info in region_info:
        centroid = info['physical_centroid']
        dist = compute_distance_to_surface(centroid, brain_mask, spacing, origin)
        surface_distances.append(dist)
        centroids.append(centroid)
    # Convert to numpy arrays
    centroids = np.array(centroids)
    surface_distances = np.array(surface_distances)
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Bolt Head Spatial Analysis', fontsize=16)
    # Surface Distance Histogram
    axs[0, 0].hist(surface_distances, bins=20, color='skyblue', edgecolor='black')
    axs[0, 0].set_title('Distribution of Distances to Brain Surface')
    axs[0, 0].set_xlabel('Distance (mm)')
    axs[0, 0].set_ylabel('Frequency') 
    # 3D Scatter of Centroids colored by surface distance
    ax_3d = fig.add_subplot(2, 2, 2, projection='3d')
    scatter = ax_3d.scatter(
        centroids[:, 0], 
        centroids[:, 1], 
        centroids[:, 2], 
        c=surface_distances, 
        cmap='viridis'
    )
    ax_3d.set_title('Bolt Head Centroids')
    ax_3d.set_xlabel('X (mm)')
    ax_3d.set_ylabel('Y (mm)')
    ax_3d.set_zlabel('Z (mm)')
    plt.colorbar(scatter, ax=ax_3d, label='Distance to Surface (mm)')
    
    # Pairwise Distances Heatmap
    pairwise_distances = distance.squareform(distance.pdist(centroids))
    im = axs[1, 0].imshow(pairwise_distances, cmap='YlOrRd')
    axs[1, 0].set_title('Pairwise Bolt Head Distances')
    axs[1, 0].set_xlabel('Bolt Head Index')
    axs[1, 0].set_ylabel('Bolt Head Index')
    plt.colorbar(im, ax=axs[1, 0], label='Distance (mm)')
    
    # Principal Axis Orientation Analysis
    principal_axes = np.array([info['principal_axis'] for info in region_info])
    axis_angles = compute_axis_angles(principal_axes)
    axs[1, 1].hist(axis_angles, bins=20, color='lightgreen', edgecolor='black')
    axs[1, 1].set_title('Distribution of Principal Axis Angles')
    axs[1, 1].set_xlabel('Angle Between Axes (degrees)')
    axs[1, 1].set_ylabel('Frequency') 
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, name), dpi=300)
    plt.close()

def generate_advanced_bolt_report(region_info, brain_mask, spacing, origin, output_dir):
    advanced_analysis = advanced_bolt_characterization(
        region_info, brain_mask, spacing, origin
    )
    with open(os.path.join(output_dir, "P1_advanced_bolt_analysis.txt"), 'w') as f:
        f.write("Advanced SEEG Bolt Head Spatial Analysis\n")
        f.write("=======================================\n\n")
        
        # Bolt Count
        f.write(f"Total Bolt Heads: {advanced_analysis['bolt_count']}\n\n")
        
        # Inter-Bolt Distance Statistics
        f.write("Inter-Bolt Distance Statistics (mm):\n")
        f.write(f"  Mean Distance: {advanced_analysis['inter_bolt_stats']['mean_distance']:.2f}\n")
        f.write(f"  Max Distance: {advanced_analysis['inter_bolt_stats']['max_distance']:.2f}\n")
        f.write(f"  Min Distance: {advanced_analysis['inter_bolt_stats']['min_distance']:.2f}\n\n")
        
        # Surface Distance Statistics
        f.write("Surface Distance Statistics (mm):\n")
        f.write(f"  Mean Distance: {advanced_analysis['surface_distance_stats']['mean']:.2f}\n")
        f.write(f"  Median Distance: {advanced_analysis['surface_distance_stats']['median']:.2f}\n")
        f.write(f"  Distance Std Dev: {advanced_analysis['surface_distance_stats']['std']:.2f}\n\n")
        
        # Volume Statistics
        f.write("Bolt Head Volume Statistics:\n")
        f.write(f"  Mean Volume: {advanced_analysis['volume_stats']['mean']:.2f} voxels\n")
        f.write(f"  Median Volume: {advanced_analysis['volume_stats']['median']:.2f} voxels\n")
        f.write(f"  Volume Std Dev: {advanced_analysis['volume_stats']['std']:.2f} voxels\n\n")
        
        # Orientation Statistics
        f.write("Principal Axis Orientation Statistics:\n")
        f.write(f"  Mean Axis Angle: {advanced_analysis['orientation_stats']['mean_angle']:.2f}°\n")
        f.write(f"  Axis Angle Std Dev: {advanced_analysis['orientation_stats']['std_angle']:.2f}°\n")

def plot_surface(ax, mask, spacing, origin, color='blue', alpha=0.7):
    try:
        verts, faces, _, _ = marching_cubes(mask, level=0.5, spacing=spacing)
        verts += origin  # Convert to physical coordinates
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                        triangles=faces, color=color, alpha=alpha, shade=True)
    except Exception as e:
        print(f"Surface generation error for {color} surface: {e}")

def plot_bolt_vectors(region_info, filtered_mask, spacing, origin):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Plot bolt regions
    for info in region_info:
        plot_surface(ax, filtered_mask == info['label'], spacing, origin, 'orange', 0.8)
    # Plot direction vectors
    for info in region_info:
        centroid = np.array(info['physical_centroid'])
        vector = np.array(info['principal_axis'])
        ax.quiver(*centroid, *vector, color='red', linewidth=2, arrow_length_ratio=0.2)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Bolt Heads with Direction Vectors')
    ax.view_init(elev=30, azim=45)
    
    plt.savefig(os.path.join(CONFIG['output_dir'], "P8_bolt_heads_with_vectors.png"), dpi=300)
    plt.close()


def plot_multi_view_entry_points(region_info, filtered_mask, brain_mask, spacing, origin):
    # Create a 2x2 grid of plots with different viewing angles
    fig = plt.figure(figsize=(18, 16))
    
    # Define different viewing angles
    view_angles = [
        {'elev': 30, 'azim': 45, 'title': 'Standard View (30°, 45°)'},
        {'elev': 0, 'azim': 0, 'title': 'Front View (0°, 0°)'},
        {'elev': 90, 'azim': 0, 'title': 'Top View (90°, 0°)'},
        {'elev': 0, 'azim': 90, 'title': 'Side View (0°, 90°)'}
    ]
    
    for i, view in enumerate(view_angles):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
        try:
            # Plot brain surface
            plot_surface(ax, brain_mask, spacing, origin, 'lightblue', 0.3)
            
            # Plot bolt heads and vectors
            entry_points_exist = False
            for info in region_info:
                try:
                    # Only plot surfaces for regions with valid labels
                    if 'label' in info and filtered_mask is not None:
                        plot_surface(ax, filtered_mask == info['label'], spacing, origin, 'yellow', 0.8)
                    
                    # Plot centroid and direction vector
                    if 'physical_centroid' in info and 'principal_axis' in info:
                        centroid = np.array(info['physical_centroid'])
                        vector = np.array(info['principal_axis'])
                        ax.quiver(*centroid, *vector, color='red', linewidth=2, arrow_length_ratio=0.2)
                    
                    # Plot entry points if available
                    if 'brain_entry_point' in info and info['brain_entry_point'] is not None:
                        entry_points_exist = True
                        entry_point = info['brain_entry_point']
                        ax.scatter(entry_point[0], entry_point[1], entry_point[2], 
                                  color='green', s=100, marker='o')
                        
                        # Add a line connecting bolt head centroid to entry point
                        if 'physical_centroid' in info:
                            centroid = np.array(info['physical_centroid'])
                            ax.plot([centroid[0], entry_point[0]], 
                                    [centroid[1], entry_point[1]], 
                                    [centroid[2], entry_point[2]], 
                                    'g--', alpha=0.7)
                except Exception as e:
                    print(f"Error plotting region: {e}")
                    continue
            
            # Set the view angle for this subplot
            ax.view_init(elev=view['elev'], azim=view['azim'])
            
            # Set labels and title
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title(view['title'])
            
            # Add a legend (only to the first subplot to avoid redundancy)
            if i == 0:
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color='lightblue', lw=4, alpha=0.3),
                    Line2D([0], [0], color='yellow', lw=4, alpha=0.8),
                    Line2D([0], [0], color='red', lw=2)
                ]
                legend_labels = ['Brain Surface', 'Bolt Head', 'Direction Vector']
                
                # Only add entry point legend elements if entry points exist
                if entry_points_exist:
                    legend_elements.extend([
                        Line2D([0], [0], color='green', lw=2, linestyle='--'),
                        Line2D([0], [0], marker='o', color='green', markersize=10, linestyle='None')
                    ])
                    legend_labels.extend(['Trajectory', 'Entry Point'])
                
                ax.legend(legend_elements, legend_labels, loc='upper right')
        
        except Exception as e:
            print(f"Error in subplot {i+1}: {e}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], "P1_bolt_heads_entry_points_multi_view.png"), dpi=300)
    plt.close()
    print("✅ Multi-view entry points plot created successfully")

# Plot bolts with brain context
def plot_bolt_brain_context(region_info, filtered_mask, brain_mask, spacing, origin, name = "P8_bolt_heads_brain_context.png"):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    plot_surface(ax, brain_mask, spacing, origin, 'lightblue', 0.3)
 
    for info in region_info:
        plot_surface(ax, filtered_mask == info['label'], spacing, origin, 'orange', 0.8)
        centroid = np.array(info['physical_centroid'])
        vector = np.array(info['principal_axis'])
        ax.quiver(*centroid, *vector, color='red', linewidth=2, arrow_length_ratio=0.2)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Bolt Heads with Brain Context')
    ax.view_init(elev=30, azim=45)
    plt.savefig(os.path.join(CONFIG['output_dir'], name), dpi=300)
    plt.close()
    generate_report(region_info)
    
    print("\n✅ Processing complete!")
    print(f"✅ All results saved to: {CONFIG['output_dir']}")

# Generate text report
def generate_report(region_info, outliers=None):
    # Safety check to ensure region_info is not empty
    if not region_info:
        print("Warning: No regions to generate report for.")
        with open(os.path.join(CONFIG['output_dir'], "P1_bolt_heads_report.txt"), 'w') as f:
            f.write("SEEG Bolt Heads Analysis Report\n")
            f.write("==============================\n\n")
            f.write("No valid bolt head regions detected.\n")
        return
        
    with open(os.path.join(CONFIG['output_dir'], "P1_bolt_heads_report.txt"), 'w') as f:
        f.write("SEEG Bolt Heads Analysis Report\n")
        f.write("==============================\n\n")
        f.write(f"Total bolt head regions detected: {len(region_info)}\n\n")
        
        # Warning section for distant bolt heads
        if outliers and 'surface_distance_threshold_outliers' in outliers:
            distant_bolts = outliers['surface_distance_threshold_outliers']
            if distant_bolts:
                f.write("⚠️ WARNING: BOLT HEADS TOO FAR FROM BRAIN SURFACE ⚠️\n")
                f.write("The following bolt heads are more than 30 mm from the brain surface:\n")
                for idx in distant_bolts:
                    if idx < len(region_info):  # Ensure index is in range
                        f.write(f"  - Bolt Head #{idx+1}: {region_info[idx]['physical_centroid']}\n")
                f.write("\n")
        
        for i, info in enumerate(region_info, 1):
            f.write(f"Bolt Head #{i} (Label {info.get('label', 'Unknown')}):\n")
            f.write(f"  - Position: ({info['physical_centroid'][0]:.1f}, {info['physical_centroid'][1]:.1f}, {info['physical_centroid'][2]:.1f}) mm\n")
            f.write(f"  - Size: {info.get('volume', 'Unknown')} voxels\n")
            
            if 'principal_axis' in info:
                f.write(f"  - Direction: [{info['principal_axis'][0]:.2f}, {info['principal_axis'][1]:.2f}, {info['principal_axis'][2]:.2f}]\n")
            else:
                f.write("  - Direction: Unknown\n")
            
            # Add entry point information if available
            if 'brain_entry_point' in info and info['brain_entry_point'] is not None:
                entry = info['brain_entry_point']
                f.write(f"  - Brain Entry Point: ({entry[0]:.1f}, {entry[1]:.1f}, {entry[2]:.1f}) mm\n")
                f.write(f"  - Distance to Entry: {info.get('entry_distance', 'Unknown'):.1f} mm\n")
            else:
                f.write(f"  - Brain Entry Point: Not found\n")
            
            f.write("\n")
            
        # Add a section with just the coordinates for easy copying
        f.write("\nBrain Entry Coordinates Summary:\n")
        f.write("-------------------------------\n")
        for i, info in enumerate(region_info, 1):
            if 'brain_entry_point' in info and info['brain_entry_point'] is not None:
                entry = info['brain_entry_point']
                f.write(f"Bolt #{i}: {entry[0]:.1f}, {entry[1]:.1f}, {entry[2]:.1f}\n")


def validate_bolt_head_in_brain_context(region_info, brain_mask, spacing, origin, max_surface_distance=30.0):
    validated_regions = []
    invalidated_regions = []
    for info in region_info:
        centroid = np.array(info['physical_centroid'])
        surface_distance = compute_distance_to_surface(centroid, brain_mask, spacing, origin)
        info['surface_distance'] = surface_distance
        if surface_distance <= max_surface_distance:
            validated_regions.append(info)
        else:
            invalidated_regions.append(info)
    return validated_regions, invalidated_regions

def plot_brain_context_with_validation(validated_regions, invalidated_regions, filtered_mask, brain_mask, spacing, origin):
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot brain mask with transparency
    plot_surface(ax, brain_mask, spacing, origin, 'lightblue', 0.3)
    # Plot validated bolt regions in green
    for info in validated_regions:
        plot_surface(ax, filtered_mask == info['label'], spacing, origin, 'green', 0.8)
        centroid = np.array(info['physical_centroid'])
        vector = np.array(info['principal_axis'])
        ax.quiver(*centroid, *vector, color='blue', linewidth=2, arrow_length_ratio=0.2)
        ax.text(*centroid, f"{info['surface_distance']:.1f} mm", color='blue')
    
    # Plot invalidated bolt regions in red
    for info in invalidated_regions:
        plot_surface(ax, filtered_mask == info['label'], spacing, origin, 'red', 0.5)
        # Plot invalidated direction vectors
        centroid = np.array(info['physical_centroid'])
        vector = np.array(info['principal_axis'])
        ax.quiver(*centroid, *vector, color='orange', linewidth=1, arrow_length_ratio=0.2)
        # Annotate surface distance
        ax.text(*centroid, f"{info['surface_distance']:.1f} mm", color='red')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Bolt Heads Validation: Brain Surface Distance')
    ax.view_init(elev=30, azim=45)
    
    plt.savefig(os.path.join(CONFIG['output_dir'], "P8_bolt_heads_brain_validation.png"), dpi=300)
    plt.close()

    # Update the text report to include surface distance information
    with open(os.path.join(CONFIG['output_dir'], "P8_bolt_heads_validation_report.txt"), 'w') as f:
        f.write("SEEG Bolt Heads Validation Report\n")
        f.write("==================================\n\n")
        
        f.write(f"Total bolt heads: {len(validated_regions) + len(invalidated_regions)}\n")
        f.write(f"Validated bolt heads: {len(validated_regions)}\n")
        f.write(f"Invalidated bolt heads: {len(invalidated_regions)}\n\n")
        
        f.write("Validated Bolt Heads:\n")
        for i, info in enumerate(validated_regions, 1):
            f.write(f"Bolt Head #{i}:\n")
            f.write(f"  - Position: ({info['physical_centroid'][0]:.1f}, {info['physical_centroid'][1]:.1f}, {info['physical_centroid'][2]:.1f}) mm\n")
            f.write(f"  - Surface Distance: {info['surface_distance']:.1f} mm\n")
            f.write(f"  - Size: {info['volume']} voxels\n")
            f.write(f"  - Direction: [{info['principal_axis'][0]:.2f}, {info['principal_axis'][1]:.2f}, {info['principal_axis'][2]:.2f}]\n\n")
        
        f.write("Invalidated Bolt Heads:\n")
        for i, info in enumerate(invalidated_regions, 1):
            f.write(f"Bolt Head #{i}:\n")
            f.write(f"  - Position: ({info['physical_centroid'][0]:.1f}, {info['physical_centroid'][1]:.1f}, {info['physical_centroid'][2]:.1f}) mm\n")
            f.write(f"  - Surface Distance: {info['surface_distance']:.1f} mm\n")
            f.write(f"  - Size: {info['volume']} voxels\n")
            f.write(f"  - Direction: [{info['principal_axis'][0]:.2f}, {info['principal_axis'][1]:.2f}, {info['principal_axis'][2]:.2f}]\n\n")

def create_entry_points_volume(validated_regions, brain_mask, spacing, origin, volume_helper):
    entry_points_mask = np.zeros_like(brain_mask, dtype=np.uint8)
    
    for info in validated_regions:
        if 'brain_entry_point' in info and info['brain_entry_point'] is not None:
            # Convert physical coordinates to voxel coordinates
            entry_point_voxel = np.round(
                (np.array(info['brain_entry_point']) - np.array(origin)) / np.array(spacing)
            ).astype(int)

            try:
                x, y, z = entry_point_voxel
                # Small 3x3x3 neighborhood marking
                entry_points_mask[
                    max(0, x-1):min(entry_points_mask.shape[0], x+2),
                    max(0, y-1):min(entry_points_mask.shape[1], y+2),
                    max(0, z-1):min(entry_points_mask.shape[2], z+2)
                ] = 1
            except IndexError:
                print(f"Warning: Entry point {entry_point_voxel} out of brain mask bounds")

    volume_helper.create_volume(
        entry_points_mask, 
        "EntryPointsMask", 
        "P1_brain_entry_points.nrrd"
    )
    
    return entry_points_mask 
if __name__ == "__main__":
    main()

# exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Bolt_head\bolt_head_og.py').read())