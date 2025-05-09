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
import csv  # Added CSV module import
import time

def get_ras_coordinates_from_ijk(volume_node, ijk):
    ijk_to_ras = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(ijk_to_ras)
    
    homogeneous_ijk = [ijk[0], ijk[1], ijk[2], 1]
    ras = [
        sum(ijk_to_ras.GetElement(i, j) * homogeneous_ijk[j] for j in range(4))
        for i in range(4)
    ]
    return ras[:3]


CONFIG = {
    'threshold_value': 2416, #P1: 2240, P4:2340, P5: 2746, P7: 2806, P8> 2416
    'min_region_size': 100,         
    'max_region_size': 800,         
    'morph_kernel_size': 1,         
    'principal_axis_length': 15,    
    'output_dir': r"C:\Users\rocia\Downloads\TFG\Cohort\Bolt_heads\P8_08_05"  
}
def main():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    print("Loading volume data...")
    # start time
    start_time = time.time()
    # Load the volume and brain mask nodes
    volume_node = slicer.util.getNode('8_CTp.3D')
    brain_mask_node = slicer.util.getNode('patient8_mask_5')
    volume_array = slicer.util.arrayFromVolume(volume_node)
    brain_mask_array = slicer.util.arrayFromVolume(brain_mask_node)
    spacing = volume_node.GetSpacing()
    origin = volume_node.GetOrigin()
    ijkToRasMatrix = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASDirectionMatrix(ijkToRasMatrix)
    volume_helper = VolumeHelper(spacing, origin, ijkToRasMatrix, CONFIG['output_dir'])
    
    print("Performing initial segmentation...")
    binary_mask = volume_array > CONFIG['threshold_value']
    volume_helper.create_volume(binary_mask.astype(np.uint8), "Threshold_Result", "P8_threshold.nrrd")
    
    print("Removing structures inside brain mask...")
    outside_brain_mask = ~brain_mask_array.astype(bool)  
    bolt_heads_mask = binary_mask & outside_brain_mask   
    volume_helper.create_volume(bolt_heads_mask.astype(np.uint8), "Outside_Brain_Result", "P8_outside_brain.nrrd")
    
    print("Applying morphological operations...")
    kernel = morphology.ball(CONFIG['morph_kernel_size'])
    cleaned_mask = morphology.binary_closing(bolt_heads_mask, kernel)
    volume_helper.create_volume(cleaned_mask.astype(np.uint8), "Cleaned_Result", "P8_cleaned.nrrd")
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
    volume_helper.create_volume(filtered_mask, "Filtered_Bolt_Heads", "P8_filtered_bolt_heads.nrrd")

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

    try:
        plot_bolt_distances_and_orientations(
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

    create_entry_points_volume(
        validated_regions, 
        brain_mask_array, 
        spacing, 
        origin, 
        volume_helper
    )

    # finish time
    end_time = time.time()
    minutes = (end_time - start_time) / 60
    seconds = (end_time - start_time) % 60
    print(f"✅ Processing completed in {int(minutes)} minutes and {int(seconds)} seconds")
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


def plot_surface(ax, mask, spacing, origin, color='blue', alpha=0.7):
    try:
        verts, faces, _, _ = marching_cubes(mask, level=0.5, spacing=spacing)
        verts += origin  # Convert to physical coordinates
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                        triangles=faces, color=color, alpha=alpha, shade=True)
    except Exception as e:
        print(f"Surface generation error for {color} surface: {e}")


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
    
    print("\n✅ Processing complete!")
    print(f"✅ All results saved to: {CONFIG['output_dir']}")


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

    # Update to create a CSV report instead of a text report
    csv_path = os.path.join(CONFIG['output_dir'], "P8_bolt_heads_validation_report.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        # Define CSV writer
        fieldnames = ['type', 'bolt_id', 'position_x', 'position_y', 'position_z', 'surface_distance', 
                      'volume', 'direction_x', 'direction_y', 'direction_z']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write data for validated bolt heads
        for i, info in enumerate(validated_regions, 1):
            writer.writerow({
                'type': 'validated',
                'bolt_id': i,
                'position_x': round(info['physical_centroid'][0], 1),
                'position_y': round(info['physical_centroid'][1], 1),
                'position_z': round(info['physical_centroid'][2], 1),
                'surface_distance': round(info['surface_distance'], 1),
                'volume': info['volume'],
                'direction_x': round(info['principal_axis'][0], 2),
                'direction_y': round(info['principal_axis'][1], 2),
                'direction_z': round(info['principal_axis'][2], 2)
            })
        
        # Write data for invalidated bolt heads
        for i, info in enumerate(invalidated_regions, 1):
            writer.writerow({
                'type': 'invalidated',
                'bolt_id': i,
                'position_x': round(info['physical_centroid'][0], 1),
                'position_y': round(info['physical_centroid'][1], 1),
                'position_z': round(info['physical_centroid'][2], 1),
                'surface_distance': round(info['surface_distance'], 1),
                'volume': info['volume'],
                'direction_x': round(info['principal_axis'][0], 2),
                'direction_y': round(info['principal_axis'][1], 2),
                'direction_z': round(info['principal_axis'][2], 2)
            })
    
    print(f"✅ Saved bolt heads validation report to {csv_path}")

def create_entry_points_volume(validated_regions, brain_mask, spacing, origin, volume_helper):
    """
    Creates both a volume mask and markup fiducials for brain entry points
    using RAS coordinates calculated from IJK coordinates
    
    Args:
        validated_regions: List of validated bolt regions with entry point information
        brain_mask: 3D numpy array of the brain mask
        spacing: Voxel spacing of the volume
        origin: Origin coordinates of the volume
        volume_helper: VolumeHelper instance for creating volumes
        
    Returns:
        entry_points_mask: 3D numpy array marking entry points
    """
    # Create volume mask for entry points
    entry_points_mask = np.zeros_like(brain_mask, dtype=np.uint8)
    
    # Create a new markup fiducial node for the entry points
    markups_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "BoltEntryPoints")
    markups_node.CreateDefaultDisplayNodes()
    markups_node.GetDisplayNode().SetSelectedColor(0, 1, 0)  # Green color for points
    markups_node.GetDisplayNode().SetPointSize(5)  # Make points visible
    
    # Process each validated region and mark entry points in the mask
    for i, info in enumerate(validated_regions):
        if 'brain_entry_point' in info and info['brain_entry_point'] is not None:
            # Convert physical coordinates to voxel coordinates (IJK)
            entry_point_voxel = np.round(
                (np.array(info['brain_entry_point']) - np.array(origin)) / np.array(spacing)
            ).astype(int)
            
            try:
                x, y, z = entry_point_voxel
                # Small 3x3x3 neighborhood marking in the volume mask
                entry_points_mask[
                    max(0, x-1):min(entry_points_mask.shape[0], x+2),
                    max(0, y-1):min(entry_points_mask.shape[1], y+2),
                    max(0, z-1):min(entry_points_mask.shape[2], z+2)
                ] = i + 1  # Use region index as label
            except IndexError:
                print(f"Warning: Entry point {entry_point_voxel} out of brain mask bounds")
    
    # Create the entry points volume AFTER modifying the mask
    entry_mask_node = volume_helper.create_volume(
        entry_points_mask, 
        "EntryPointsMask",
        "P8_brain_entry_points.nrrd"
    )
    
    # Use regionprops to get the centroids of each labeled region
    from skimage.measure import label, regionprops
    labeled_image = label(entry_points_mask > 0)
    regions = regionprops(labeled_image)
    
    for i, region in enumerate(regions):
        # Get centroid in IJK coordinates (needs to be in [i,j,k] order for the conversion function)
        centroid_ijk = region.centroid
        
        # Correct IJK order - regionprops returns [z,y,x] but we need [i,j,k]
        # According to your get_ras_coordinates_from_ijk function, the order should be [k,j,i]
        ijk_for_conversion = [centroid_ijk[2], centroid_ijk[1], centroid_ijk[0]]
        
        # Convert IJK to RAS using get_ras_coordinates_from_ijk
        ras_coords = get_ras_coordinates_from_ijk(entry_mask_node, ijk_for_conversion)
        
        # Add entry point as markup fiducial
        # The AddControlPoint method expects RAS coordinates directly
        markups_node.AddControlPoint(
            ras_coords[0], ras_coords[1], ras_coords[2],
            f"Entry_{i+1}"
        )
    
    # Save the markup nodes
    save_path = os.path.join(CONFIG['output_dir'], "P8_entry_points_markups.fcsv")
    slicer.util.saveNode(markups_node, save_path)
    print(f"✅ Saved entry points markup file to {save_path}")
    
    return entry_points_mask
if __name__ == "__main__":
    main()

# exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Bolt_head\bolt_head.py').read())