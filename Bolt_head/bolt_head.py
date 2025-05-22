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
import pandas as pd


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
    'threshold_value': 2325, #P1: 2240, P4:2340, P5: 2746, P7: 2806, P8> 2416
    'min_region_size': 100,         
    'max_region_size': 800,         
    'morph_kernel_size': 1,         
    'principal_axis_length': 15,    
    'output_dir': r"C:\Users\rocia\Downloads\TFG\Cohort\Bolt_heads\P6_2325"  # tu directorio 
}

def main():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    print("Loading volume data...")
    # start time
    start_time = time.time()
    # Load the volume and brain mask nodes
    volume_node = slicer.util.getNode('6_CTp.3D') # CT del paciente 
    brain_mask_node = slicer.util.getNode('patient6_mask_5') # ROI del cerebro del paciente
    volume_array = slicer.util.arrayFromVolume(volume_node)
    brain_mask_array = slicer.util.arrayFromVolume(brain_mask_node)
    spacing = volume_node.GetSpacing()
    origin = volume_node.GetOrigin()
    ijkToRasMatrix = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASDirectionMatrix(ijkToRasMatrix)
    volume_helper = VolumeHelper(spacing, origin, ijkToRasMatrix, CONFIG['output_dir'])
    
    print("Performing initial segmentation...")
    binary_mask = volume_array > CONFIG['threshold_value']
    volume_helper.create_volume(binary_mask.astype(np.uint8), "Threshold_Result", "P6_threshold.nrrd")
    
    print("Removing structures inside brain mask...")
    outside_brain_mask = ~brain_mask_array.astype(bool)  
    bolt_heads_mask = binary_mask & outside_brain_mask   
    volume_helper.create_volume(bolt_heads_mask.astype(np.uint8), "Outside_Brain_Result", "P6_outside_brain.nrrd")
    
    print("Applying morphological operations...")
    kernel = morphology.ball(CONFIG['morph_kernel_size'])
    cleaned_mask = morphology.binary_closing(bolt_heads_mask, kernel)
    volume_helper.create_volume(cleaned_mask.astype(np.uint8), "Cleaned_Result", "P6_cleaned.nrrd")
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
    volume_helper.create_volume(filtered_mask, "Filtered_Bolt_Heads", "P6_filtered_bolt_heads.nrrd")

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

    entry_points_mask, ras_coordinates = create_entry_points_volume(
        validated_regions, 
        brain_mask_array, 
        spacing, 
        origin, 
        volume_helper
    )
    # finish time
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    print(f"Mask ensembling completed in {minutes}m {seconds:.2f}s")
    
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
    plt.savefig(os.path.join(CONFIG['output_dir'], "P6_bolt_heads_entry_points.png"), dpi=300)
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

 

def plot_surface(ax, mask, spacing, origin, color='blue', alpha=0.7):
    try:
        verts, faces, _, _ = marching_cubes(mask, level=0.5, spacing=spacing)
        verts += origin  # Convert to physical coordinates
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                        triangles=faces, color=color, alpha=alpha, shade=True)
    except Exception as e:
        print(f"Surface generation error for {color} surface: {e}")


# Plot bolts with brain context
def plot_bolt_brain_context(region_info, filtered_mask, brain_mask, spacing, origin, name = "P6_bolt_heads_brain_context.png"):
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
    plt.savefig(os.path.join(CONFIG['output_dir'], "P6_bolt_heads_brain_validation.png"), dpi=300)
    plt.close()

    print(f"✅ Saved bolt heads validation plot to P6_bolt_heads_brain_validation.png ")

def create_entry_points_volume(validated_regions, brain_mask, spacing, origin, volume_helper):
    # Create a mask to mark entry points
    entry_points_mask = np.zeros_like(brain_mask, dtype=np.uint8)
    
    # Create markups node for visualization
    markups_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "BoltEntryPoints")
    markups_node.CreateDefaultDisplayNodes()
    markups_node.GetDisplayNode().SetSelectedColor(0, 1, 0)
    markups_node.GetDisplayNode().SetPointSize(5)
    
    # Track which validated region corresponds to which mask value
    region_index_to_mask_value = {}
    
    # First pass: Create the mask with entry points
    for i, info in enumerate(validated_regions):
        if 'brain_entry_point' in info and info['brain_entry_point'] is not None:
            # Convert physical coordinates to voxel coordinates
            entry_point_voxel = np.round(
                (np.array(info['brain_entry_point']) - np.array(origin)) / np.array(spacing)
            ).astype(int)
            
            try:
                x, y, z = entry_point_voxel
                # Use a unique value for each region (i+1)
                mask_value = i + 1
                
                # Record which mask value corresponds to which validated region
                region_index_to_mask_value[i] = mask_value
                
                # Mark in the mask with unique label
                entry_points_mask[
                    max(0, x-1):min(entry_points_mask.shape[0], x+2),
                    max(0, y-1):min(entry_points_mask.shape[1], y+2),
                    max(0, z-1):min(entry_points_mask.shape[2], z+2)
                ] = mask_value
            except IndexError:
                print(f"Warning: Entry point {entry_point_voxel} out of brain mask bounds")
    
    # Create the volume for visualization
    entry_mask_node = volume_helper.create_volume(
        entry_points_mask, 
        "EntryPointsMask",
        "P7_brain_entry_points.nrrd"
    )
    
    # Use regionprops to get centroids in IJK space
    labeled_image = label(entry_points_mask)
    regions = regionprops(labeled_image)
    
    # Create a mapping from regionprops label to RAS coordinates
    label_to_ras = {}
    ras_coordinates_list = []
    
    # Process regionprops to get RAS coordinates
    for region in regions:
        # Get centroid in IJK coordinates
        centroid_ijk = region.centroid
        # Correct IJK order for the conversion function
        ijk_for_conversion = [centroid_ijk[2], centroid_ijk[1], centroid_ijk[0]]
        
        # Convert IJK to RAS
        ras_coords = get_ras_coordinates_from_ijk(entry_mask_node, ijk_for_conversion)
        
        # Add to markups node
        markups_node.AddControlPoint(
            ras_coords[0], ras_coords[1], ras_coords[2],
            f"Entry_{region.label}"
        )
        
        # Store RAS coordinates
        label_to_ras[region.label] = ras_coords
        ras_coordinates_list.append(ras_coords)
    
    # Save the markup nodes
    save_path = os.path.join(CONFIG['output_dir'], "P6_entry_points_markups.fcsv")
    slicer.util.saveNode(markups_node, save_path)
    print(f"✅ Saved entry points markup file to {save_path}")
    
    # Create mapping from mask values to regionprops labels
    # This is needed because regionprops may relabel regions
    mask_value_to_region_label = {}
    for region in regions:
        region_label = region.label
        region_mask = labeled_image == region_label
        unique_values = np.unique(entry_points_mask[region_mask])
        if len(unique_values) > 0 and unique_values[0] > 0:
            mask_value_to_region_label[unique_values[0]] = region_label
    
    # Create the report data using all the mappings
    report_data = []
    for i, info in enumerate(validated_regions):
        if 'brain_entry_point' in info and info['brain_entry_point'] is not None:
            # Get the mask value for this validated region
            mask_value = region_index_to_mask_value.get(i)
            if mask_value is None:
                continue
                
            # Get the regionprops label for this mask value
            region_label = mask_value_to_region_label.get(mask_value)
            if region_label is None:
                continue
                
            # Get the RAS coordinates for this region label
            ras_coords = label_to_ras.get(region_label)
            if ras_coords is None:
                continue
                
            # Create row for this entry point
            row = {
                # RAS coordinates
                'ras_x': round(ras_coords[0], 1),
                'ras_y': round(ras_coords[1], 1), 
                'ras_z': round(ras_coords[2], 1),
                
                # Original brain entry point coordinates
                'entry_point_x': round(info['brain_entry_point'][0], 1),
                'entry_point_y': round(info['brain_entry_point'][1], 1),
                'entry_point_z': round(info['brain_entry_point'][2], 1),
                
                # Additional metrics
                'surface_distance': round(info.get('surface_distance', 0), 1),
                'volume': info['volume'],
                'direction_x': round(info['principal_axis'][0], 2),
                'direction_y': round(info['principal_axis'][1], 2),
                'direction_z': round(info['principal_axis'][2], 2),
                'entry_distance': round(info.get('entry_distance', 0), 1),
            }
            report_data.append(row)
    
    # Create and save CSV report
    df = pd.DataFrame(report_data)
    csv_path = os.path.join(CONFIG['output_dir'], "P6_brain_entry_points_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved brain entry points report to {csv_path}")
    
    # Debug output
    print(f"Number of validated regions with entry points: {sum(1 for info in validated_regions if 'brain_entry_point' in info and info['brain_entry_point'] is not None)}")
    print(f"Number of regions found by regionprops: {len(regions)}")
    print(f"Number of report entries generated: {len(report_data)}")
    
    return entry_points_mask, ras_coordinates_list

if __name__ == "__main__":
    main()


# exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Bolt_head\bolt_head.py').read())