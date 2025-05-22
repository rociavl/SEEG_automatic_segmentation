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
import csv
import time
import pandas as pd

# Configuration Constants
CONFIG = {
    'threshold_value': 2450,
    'min_region_size': 100,
    'max_region_size': 800,
    'morph_kernel_size': 1,
    'principal_axis_length': 15,
    'output_dir': r"C:\Users\rocia\Downloads\TFG\Cohort\Bolt_heads\P1_2450_entry_correct"
}

class VolumeHelper:
    """Helper class for volume operations and file management."""
    def __init__(self, spacing, origin, direction_matrix, output_dir):
        self.spacing = spacing
        self.origin = origin
        self.direction_matrix = direction_matrix
        self.output_dir = output_dir
   
    def create_volume(self, array, name, save_filename=None):
        """Create and optionally save a volume from an array."""
        sitk_image = sitk.GetImageFromArray(array)
        sitk_image.SetSpacing(self.spacing)
        sitk_image.SetOrigin(self.origin)
        
        # Create a new volume node
        new_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
        
        # Set the spacing directly on the volume node
        new_node.SetSpacing(self.spacing)
        new_node.SetOrigin(self.origin)
        new_node.SetIJKToRASDirectionMatrix(self.direction_matrix)
        
        # Update the volume from the array
        slicer.util.updateVolumeFromArray(new_node, array)
        
        if save_filename:
            save_path = os.path.join(self.output_dir, save_filename)
            slicer.util.saveNode(new_node, save_path)
            print(f"✅ Saved {name} to {save_path}")
        return new_node

    # def create_volume(self, array, name, save_filename=None):
    #     """Create and optionally save a volume from an array."""
    #     sitk_image = sitk.GetImageFromArray(array)
    #     sitk_image.SetSpacing(self.spacing)
    #     sitk_image.SetOrigin(self.origin)
        
    #     new_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
    #     new_node.SetIJKToRASDirectionMatrix(self.direction_matrix)
    #     new_node.SetOrigin(self.origin)
    #     slicer.util.updateVolumeFromArray(new_node, array)
        
    #     if save_filename:
    #         save_path = os.path.join(self.output_dir, save_filename)
    #         slicer.util.saveNode(new_node, save_path)
    #         print(f"✅ Saved {name} to {save_path}")
    #     return new_node

class BoltHeadDetector:
    """Main class for bolt head detection and analysis."""
    def __init__(self, config):
        self.config = config
        os.makedirs(config['output_dir'], exist_ok=True)
        
    def run(self):
        """Main execution method."""
        print("Loading volume data...")
        start_time = time.time()
        
        # Load data and initialize helper
        volume_node, brain_mask_node = self._load_nodes()
        volume_array = slicer.util.arrayFromVolume(volume_node)
        brain_mask_array = slicer.util.arrayFromVolume(brain_mask_node)
        
        volume_helper = VolumeHelper(
            volume_node.GetSpacing(),
            volume_node.GetOrigin(),
            self._get_direction_matrix(volume_node),
            self.config['output_dir']
        )
        
        # Add threshold analysis before standard processing
        print("Performing threshold effects analysis...")
        threshold_results, voting_mask = self.analyze_threshold_effects_3d(
            volume_array, 
            brain_mask_array, 
            volume_helper
        )
        
        # Continue with the standard processing pipeline
        binary_mask = self._threshold_volume(volume_array, volume_helper)
        bolt_heads_mask = self._remove_brain_structures(binary_mask, brain_mask_array, volume_helper)
        cleaned_mask = self._apply_morphological_ops(bolt_heads_mask, volume_helper)
        
        if not np.any(cleaned_mask):
            print("No bolt head regions found at the given threshold outside the brain mask.")
            return
            
        filtered_mask, region_info = self._filter_bolt_heads(cleaned_mask, volume_node, brain_mask_array)
        validated_regions = self._validate_bolt_heads(region_info, brain_mask_array, volume_node)
        
        self._process_validated_regions(validated_regions, filtered_mask, brain_mask_array, volume_helper, volume_node)
        
        # Final reporting
        elapsed_time = time.time() - start_time
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        print(f"Processing completed in {minutes}m {seconds:.2f}s")
        print(f"\n✅ All results saved to: {self.config['output_dir']}")

    # Processing pipeline methods
    def _load_nodes(self):
        """Load required nodes from Slicer."""
        return (
            slicer.util.getNode('5_CTp.3D'),  # CT volume
            slicer.util.getNode('patient5_mask_5')  # Brain mask
        )
        
    def _get_direction_matrix(self, volume_node):
        """Get IJK to RAS direction matrix."""
        matrix = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASDirectionMatrix(matrix)
        return matrix
        
    def _threshold_volume(self, volume_array, volume_helper):
        """Apply threshold to create initial binary mask."""
        print("Performing initial segmentation...")
        binary_mask = volume_array > self.config['threshold_value']
        volume_helper.create_volume(binary_mask.astype(np.uint8), "Threshold_Result", "P6_threshold.nrrd")
        return binary_mask
        
    def _remove_brain_structures(self, binary_mask, brain_mask_array, volume_helper):
        """Remove structures inside the brain mask."""
        print("Removing structures inside brain mask...")
        outside_brain_mask = ~brain_mask_array.astype(bool)
        bolt_heads_mask = binary_mask & outside_brain_mask
        volume_helper.create_volume(bolt_heads_mask.astype(np.uint8), "Outside_Brain_Result", "P6_outside_brain.nrrd")
        return bolt_heads_mask
        
    def _apply_morphological_ops(self, mask, volume_helper):
        """Apply morphological operations to clean the mask."""
        print("Applying morphological operations...")
        kernel = morphology.ball(self.config['morph_kernel_size'])
        cleaned_mask = morphology.binary_closing(mask, kernel)
        volume_helper.create_volume(cleaned_mask.astype(np.uint8), "Cleaned_Result", "P6_cleaned.nrrd")
        return cleaned_mask
        
    def _filter_bolt_heads(self, cleaned_mask, volume_node, brain_mask_array):
        """Identify and filter bolt head components."""
        print("Identifying and filtering bolt head components...")
        labeled_image = label(cleaned_mask)
        regions = regionprops(labeled_image)
        
        filtered_mask = np.zeros_like(labeled_image, dtype=np.uint16)
        region_info = []
        spacing = volume_node.GetSpacing()
        origin = volume_node.GetOrigin()
        
        for region in regions:
            volume = region.area
            if self.config['min_region_size'] < volume < self.config['max_region_size']:
                filtered_mask[labeled_image == region.label] = region.label
                centroid_physical = tuple(origin[i] + region.centroid[i] * spacing[i] for i in range(3))
                coords = np.argwhere(labeled_image == region.label)
                
                principal_axis = self._calculate_principal_axis(coords, spacing)
                bolt_to_brain_center = self._estimate_brain_center(brain_mask_array, spacing, origin) - np.array(centroid_physical)
                
                if np.dot(principal_axis, bolt_to_brain_center) < 0:
                    principal_axis = -principal_axis
                    
                region_info.append({
                    'label': region.label,
                    'physical_centroid': centroid_physical,
                    'volume': volume,
                    'principal_axis': principal_axis
                })
        
        print(f"Found {len(region_info)} valid bolt head regions after filtering")
        volume_helper = VolumeHelper(spacing, origin, self._get_direction_matrix(volume_node), self.config['output_dir'])
        volume_helper.create_volume(filtered_mask, "Filtered_Bolt_Heads", "P6_filtered_bolt_heads.nrrd")
        
        return filtered_mask, region_info
        
    def _validate_bolt_heads(self, region_info, brain_mask_array, volume_node):
        """Validate bolt heads based on brain context."""
        spacing = volume_node.GetSpacing()
        origin = volume_node.GetOrigin()
        
        validated_regions, invalidated_regions = self._validate_bolt_head_in_brain_context(
            region_info, brain_mask_array, spacing, origin
        )
        
        print("Generating POST-VALIDATION visualizations...")
        self._plot_brain_context_with_validation(
            validated_regions, 
            invalidated_regions, 
            brain_mask_array, 
            spacing, 
            origin
        )
        
        return validated_regions
        
    def _process_validated_regions(self, validated_regions, filtered_mask, brain_mask_array, volume_helper, volume_node):
        """Process and analyze validated bolt heads."""
        print("Calculating brain entry points for validated bolt heads...")
        spacing = volume_node.GetSpacing()
        origin = volume_node.GetOrigin()
        
        for info in validated_regions:
            centroid = np.array(info['physical_centroid'])
            direction = np.array(info['principal_axis'])
            direction = direction / np.linalg.norm(direction)
            
            entry_point, distance = self._calculate_brain_intersection(
                centroid, direction, brain_mask_array, spacing, origin
            )
            
            info['brain_entry_point'] = entry_point
            info['entry_distance'] = distance
        
        # Visualization and output
        self._plot_entry_points(validated_regions, filtered_mask, brain_mask_array, spacing, origin)
        self._create_entry_points_volume(validated_regions, brain_mask_array, volume_helper, volume_node)

    # Helper methods
    def _estimate_brain_center(self, brain_mask, spacing, origin):
        """Estimate the physical center of the brain mask."""
        coords = np.argwhere(brain_mask > 0)
        if len(coords) == 0:
            return np.array([0, 0, 0])
        center_voxel = np.mean(coords, axis=0)
        return np.array([origin[i] + center_voxel[i] * spacing[i] for i in range(3)])
        
    def _calculate_principal_axis(self, coords, spacing):
        """Calculate principal axis using PCA."""
        if len(coords) > 2:
            pca = PCA(n_components=3)
            pca.fit(coords)
            principal_axis = pca.components_[0] * spacing
            return principal_axis / np.linalg.norm(principal_axis) * self.config['principal_axis_length']
        return np.array([0, 0, 1])  # Default if not enough points
        
    def _calculate_brain_intersection(self, centroid, direction, brain_mask, spacing, origin):
        """Calculate intersection point with brain surface."""
        try:
            voxel_centroid = np.array([
                (centroid[i] - origin[i]) / spacing[i] for i in range(3)
            ], dtype=np.float64)
            
            shape = brain_mask.shape
            direction = direction / np.linalg.norm(direction)
            strategies = [
                {'step_size': 0.5, 'max_multiplier': 3},
                {'step_size': 1.0, 'max_multiplier': 5},
                {'step_size': 0.25, 'max_multiplier': 10}
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
                    x, y, z = np.round(current_pos).astype(int)
                    
                    if (x < 0 or x >= shape[0] or y < 0 or y >= shape[1] or z < 0 or z >= shape[2]):
                        break
                        
                    if brain_mask[x, y, z] > 0:
                        intersection_voxel = (current_pos + last_pos) / 2
                        intersection_point = np.array([
                            origin[i] + intersection_voxel[i] * spacing[i] for i in range(3)
                        ])
                        
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
            
    def _compute_distance_to_surface(self, point, brain_mask, spacing, origin):
        """Compute distance from point to brain surface."""
        point = np.asarray(point)
        origin = np.asarray(origin)
        spacing = np.asarray(spacing)
        voxel_point = np.round((point - origin) / spacing).astype(int)
        
        if (np.any(voxel_point < 0) or np.any(voxel_point >= np.array(brain_mask.shape))):
            return np.inf
            
        surface_mask = self._compute_surface_mask(brain_mask)
        surface_voxels = np.argwhere(surface_mask)
        
        if len(surface_voxels) > 0:
            surface_points_physical = surface_voxels * spacing + origin
            distances = np.min(np.linalg.norm(surface_points_physical - point, axis=1))
            return distances
        return np.inf
        
    def _compute_surface_mask(self, mask, connectivity=1):
        """Compute surface mask using morphological operations."""
        dilated = binary_dilation(mask, iterations=1)
        eroded = binary_erosion(mask, iterations=1)
        return dilated ^ eroded
        
    def _validate_bolt_head_in_brain_context(self, region_info, brain_mask, spacing, origin, max_surface_distance=30.0):
        """Validate bolt heads based on distance to brain surface."""
        validated_regions = []
        invalidated_regions = []
        
        for info in region_info:
            centroid = np.array(info['physical_centroid'])
            surface_distance = self._compute_distance_to_surface(centroid, brain_mask, spacing, origin)
            info['surface_distance'] = surface_distance
            
            if surface_distance <= max_surface_distance:
                validated_regions.append(info)
            else:
                invalidated_regions.append(info)
                
        return validated_regions, invalidated_regions

    # Visualization methods
    def _plot_surface(self, ax, mask, spacing, origin, color='blue', alpha=0.7):
        """Plot a 3D surface from a binary mask."""
        try:
            verts, faces, _, _ = marching_cubes(mask, level=0.5, spacing=spacing)
            verts += origin  # Convert to physical coordinates
            ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                          triangles=faces, color=color, alpha=alpha, shade=True)
        except Exception as e:
            print(f"Surface generation error for {color} surface: {e}")
            
    def _plot_brain_context(self, region_info, filtered_mask, brain_mask, spacing, origin, name="bolt_heads_brain_context.png"):
        """Plot bolt heads with brain context."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        self._plot_surface(ax, brain_mask, spacing, origin, 'lightblue', 0.3)
        
        for info in region_info:
            self._plot_surface(ax, filtered_mask == info['label'], spacing, origin, 'orange', 0.8)
            centroid = np.array(info['physical_centroid'])
            vector = np.array(info['principal_axis'])
            ax.quiver(*centroid, *vector, color='red', linewidth=2, arrow_length_ratio=0.2)
            
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('Bolt Heads with Brain Context')
        ax.view_init(elev=30, azim=45)
        
        plt.savefig(os.path.join(self.config['output_dir'], name), dpi=300)
        plt.close()
        
    def _plot_brain_context_with_validation(self, validated_regions, invalidated_regions, brain_mask, spacing, origin):
        """Plot validation results with brain context."""
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot brain mask
        self._plot_surface(ax, brain_mask, spacing, origin, 'lightblue', 0.3)
        
        # Plot validated regions in green
        for info in validated_regions:
            # Note: This needs filtered_mask which isn't passed - might need adjustment
            # self._plot_surface(ax, filtered_mask == info['label'], spacing, origin, 'green', 0.8)
            centroid = np.array(info['physical_centroid'])
            vector = np.array(info['principal_axis'])
            ax.quiver(*centroid, *vector, color='blue', linewidth=2, arrow_length_ratio=0.2)
            ax.text(*centroid, f"{info['surface_distance']:.1f} mm", color='blue')
            
        # Plot invalidated regions in red
        for info in invalidated_regions:
            # self._plot_surface(ax, filtered_mask == info['label'], spacing, origin, 'red', 0.5)
            centroid = np.array(info['physical_centroid'])
            vector = np.array(info['principal_axis'])
            ax.quiver(*centroid, *vector, color='orange', linewidth=1, arrow_length_ratio=0.2)
            ax.text(*centroid, f"{info['surface_distance']:.1f} mm", color='red')
            
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('Bolt Heads Validation: Brain Surface Distance')
        ax.view_init(elev=30, azim=45)
        
        plt.savefig(os.path.join(self.config['output_dir'], "P6_bolt_heads_brain_validation.png"), dpi=300)
        plt.close()
        print(f"✅ Saved bolt heads validation plot to P6_bolt_heads_brain_validation.png")
        
    def _plot_entry_points(self, region_info, filtered_mask, brain_mask, spacing, origin):
        """Plot bolt heads with entry points."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        self._plot_surface(ax, brain_mask, spacing, origin, 'lightblue', 0.3)
        
        for info in region_info:
            self._plot_surface(ax, filtered_mask == info['label'], spacing, origin, 'yellow', 0.8)
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
        
        plt.savefig(os.path.join(self.config['output_dir'], "P6_bolt_heads_entry_points.png"), dpi=300)
        plt.close()
        
    def _create_entry_points_volume(self, validated_regions, brain_mask, volume_helper, volume_node):
        """Create volume with entry points and generate reports."""
        entry_points_mask = np.zeros_like(brain_mask, dtype=np.uint8)
        markups_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "BoltEntryPoints")
        markups_node.CreateDefaultDisplayNodes()
        markups_node.GetDisplayNode().SetSelectedColor(0, 1, 0)
        markups_node.GetDisplayNode().SetPointSize(5)
        
        region_index_to_mask_value = {}
        
        # Create entry points mask
        for i, info in enumerate(validated_regions):
            if 'brain_entry_point' in info and info['brain_entry_point'] is not None:
                entry_point_voxel = np.round(
                    (np.array(info['brain_entry_point']) - np.array(volume_node.GetOrigin())) / np.array(volume_node.GetSpacing())
                ).astype(int)
                
                try:
                    x, y, z = entry_point_voxel
                    mask_value = i + 1
                    region_index_to_mask_value[i] = mask_value
                    
                    entry_points_mask[
                        max(0, x-1):min(entry_points_mask.shape[0], x+2),
                        max(0, y-1):min(entry_points_mask.shape[1], y+2),
                        max(0, z-1):min(entry_points_mask.shape[2], z+2)
                    ] = mask_value
                except IndexError:
                    print(f"Warning: Entry point {entry_point_voxel} out of brain mask bounds")
        
        # Create volume and get region properties
        entry_mask_node = volume_helper.create_volume(
            entry_points_mask, 
            "EntryPointsMask",
            "P7_brain_entry_points.nrrd"
        )
        
        labeled_image = label(entry_points_mask)
        regions = regionprops(labeled_image)
        label_to_ras = {}
        ras_coordinates_list = []
        
        # Process regions and create markups
        for region in regions:
            centroid_ijk = region.centroid
            ijk_for_conversion = [centroid_ijk[2], centroid_ijk[1], centroid_ijk[0]]
            ras_coords = self._get_ras_coordinates_from_ijk(entry_mask_node, ijk_for_conversion)
            
            markups_node.AddControlPoint(
                ras_coords[0], ras_coords[1], ras_coords[2],
                f"Entry_{region.label}"
            )
            label_to_ras[region.label] = ras_coords
            ras_coordinates_list.append(ras_coords)
        
        # Save markups
        save_path = os.path.join(self.config['output_dir'], "P6_entry_points_markups.fcsv")
        slicer.util.saveNode(markups_node, save_path)
        print(f"✅ Saved entry points markup file to {save_path}")
        
        # Create report data
        mask_value_to_region_label = {}
        for region in regions:
            region_label = region.label
            region_mask = labeled_image == region_label
            unique_values = np.unique(entry_points_mask[region_mask])
            if len(unique_values) > 0 and unique_values[0] > 0:
                mask_value_to_region_label[unique_values[0]] = region_label
        
        report_data = []
        for i, info in enumerate(validated_regions):
            if 'brain_entry_point' in info and info['brain_entry_point'] is not None:
                mask_value = region_index_to_mask_value.get(i)
                if mask_value is None:
                    continue
                    
                region_label = mask_value_to_region_label.get(mask_value)
                if region_label is None:
                    continue
                    
                ras_coords = label_to_ras.get(region_label)
                if ras_coords is None:
                    continue
                    
                report_data.append({
                    'ras_x': round(ras_coords[0], 1),
                    'ras_y': round(ras_coords[1], 1), 
                    'ras_z': round(ras_coords[2], 1),
                    'entry_point_x': round(info['brain_entry_point'][0], 1),
                    'entry_point_y': round(info['brain_entry_point'][1], 1),
                    'entry_point_z': round(info['brain_entry_point'][2], 1),
                    'surface_distance': round(info.get('surface_distance', 0), 1),
                    'volume': info['volume'],
                    'direction_x': round(info['principal_axis'][0], 2),
                    'direction_y': round(info['principal_axis'][1], 2),
                    'direction_z': round(info['principal_axis'][2], 2),
                    'entry_distance': round(info.get('entry_distance', 0), 1),
                })
        
        # Save report
        df = pd.DataFrame(report_data)
        csv_path = os.path.join(self.config['output_dir'], "P6_brain_entry_points_report.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved brain entry points report to {csv_path}")
        
        print(f"Number of validated regions with entry points: {sum(1 for info in validated_regions if 'brain_entry_point' in info and info['brain_entry_point'] is not None)}")
        print(f"Number of regions found by regionprops: {len(regions)}")
        print(f"Number of report entries generated: {len(report_data)}")
        
        return entry_points_mask, ras_coordinates_list
        
    def _get_ras_coordinates_from_ijk(self, volume_node, ijk):
        """Convert IJK coordinates to RAS coordinates."""
        ijk_to_ras = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(ijk_to_ras)
        homogeneous_ijk = [ijk[0], ijk[1], ijk[2], 1]
        ras = [
            sum(ijk_to_ras.GetElement(i, j) * homogeneous_ijk[j] for j in range(4))
            for i in range(4)
        ]
        return ras[:3]


    def analyze_threshold_effects_3d(self, volume_array, brain_mask_array, volume_helper):
        """Optimized threshold effects analysis with combined visualizations."""
        threshold_values = [2275, 2300, 2325, 2350, 2375, 2400, 2425]
        results = []
        
        # Store original threshold
        original_threshold = self.config['threshold_value']
        
        # Create output directory
        threshold_dir = os.path.join(self.config['output_dir'], 'threshold_analysis')
        os.makedirs(threshold_dir, exist_ok=True)
        
        print("Performing optimized threshold analysis...")
        
        # Pre-compute brain surface for visualization
        brain_verts, brain_faces, _, _ = marching_cubes(
            brain_mask_array, 
            level=0.5, 
            spacing=volume_helper.spacing
        )
        brain_verts += volume_helper.origin  # Convert to physical coordinates
        
        # Initialize voting mask
        voting_mask = np.zeros_like(volume_array, dtype=np.uint8)
        all_masks = []
        
        # Prepare colormap for visualization
        colors = plt.cm.viridis(np.linspace(0, 1, len(threshold_values)))
        
        # Create a single figure for all threshold results
        fig, axes = plt.subplots(2, 4, figsize=(24, 12), 
                            subplot_kw={'projection': '3d'})
        axes = axes.ravel()
        
        # Process each threshold
        for i, (threshold, ax) in enumerate(zip(threshold_values, axes)):
            start_time = time.time()
            
            # Apply threshold and processing pipeline
            binary_mask = volume_array > threshold
            bolt_heads_mask = binary_mask & (~brain_mask_array.astype(bool))
            
            # Apply morphological operations
            kernel = morphology.ball(self.config['morph_kernel_size'])
            cleaned_mask = morphology.binary_closing(bolt_heads_mask, kernel)
            
            # Label and filter regions
            labeled_image = label(cleaned_mask)
            regions = regionprops(labeled_image)
            valid_regions = [r for r in regions if 
                            self.config['min_region_size'] < r.area < self.config['max_region_size']]
            
            # Create filtered mask
            filtered_mask = np.zeros_like(cleaned_mask, dtype=np.uint8)
            for region in valid_regions:
                filtered_mask[labeled_image == region.label] = 1
            
            # Update voting mask
            voting_mask += filtered_mask
            all_masks.append(filtered_mask)
            
            # Collect statistics
            stats = {
                'threshold': threshold,
                'total_voxels': np.sum(filtered_mask),
                'region_count': len(valid_regions),
                'avg_region_size': np.mean([r.area for r in valid_regions]) if valid_regions else 0,
                'min_region_size': np.min([r.area for r in valid_regions]) if valid_regions else 0,
                'max_region_size': np.max([r.area for r in valid_regions]) if valid_regions else 0,
            }
            results.append(stats)
            
            # Plot brain surface once on first subplot
            if i == 0:
                for a in axes:
                    a.plot_trisurf(
                        brain_verts[:, 0], brain_verts[:, 1], brain_verts[:, 2],
                        triangles=brain_faces, color='lightblue', alpha=0.2
                    )
            
            # Plot current threshold results
            if np.any(filtered_mask):
                verts, faces, _, _ = marching_cubes(
                    filtered_mask, 
                    level=0.5, 
                    spacing=volume_helper.spacing
                )
                verts += volume_helper.origin
                ax.plot_trisurf(
                    verts[:, 0], verts[:, 1], verts[:, 2],
                    triangles=faces, color=colors[i], alpha=0.8
                )
            
            ax.set_title(f'Threshold: {threshold}\nRegions: {len(valid_regions)}')
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.view_init(elev=30, azim=45)
            
            print(f"Processed threshold {threshold} in {time.time()-start_time:.2f}s")
        
        # Remove empty subplots
        for i in range(len(threshold_values), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(threshold_dir, "combined_threshold_results.png"), dpi=150)
        plt.close(fig)
        
        # Create voting confidence visualization
        self._create_voting_visualization(
            voting_mask, 
            brain_verts, 
            brain_faces, 
            threshold_values, 
            threshold_dir,
            volume_helper.spacing,
            volume_helper.origin
        )
        
        # Create statistical plots
        self._create_statistical_plots(results, threshold_dir)
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(threshold_dir, 'threshold_effects.csv'), index=False)
        
        # Reset original threshold
        self.config['threshold_value'] = original_threshold
        
        print(f"✅ Saved optimized threshold analysis to {threshold_dir}")
        return results, voting_mask

    def _create_voting_visualization(self, voting_mask, brain_verts, brain_faces, 
                                threshold_values, output_dir, spacing, origin):
        """Create visualization of voting confidence."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot brain surface
        ax.plot_trisurf(
            brain_verts[:, 0], brain_verts[:, 1], brain_verts[:, 2],
            triangles=brain_faces, color='lightblue', alpha=0.2
        )
        
        # Create colormap for voting confidence
        cmap = plt.cm.viridis
        norm = plt.Normalize(1, len(threshold_values))
        
        # Plot each confidence level separately
        for confidence in range(1, len(threshold_values)+1):
            mask = (voting_mask == confidence)
            if np.any(mask):
                verts, faces, _, _ = marching_cubes(mask, level=0.5, spacing=spacing)
                verts += origin
                ax.plot_trisurf(
                    verts[:, 0], verts[:, 1], verts[:, 2],
                    triangles=faces, 
                    color=cmap(norm(confidence)), 
                    alpha=0.7,
                    label=f'{confidence} thresholds'
                )
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Number of Thresholds Agreeing', rotation=270, labelpad=15)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('Threshold Voting Confidence')
        ax.view_init(elev=30, azim=45)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "threshold_voting_confidence.png"), dpi=150)
        plt.close(fig)

    def _create_statistical_plots(self, results, output_dir):
        """Create statistical plots of threshold effects."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Region count plot
        thresholds = [r['threshold'] for r in results]
        counts = [r['region_count'] for r in results]
        ax1.plot(thresholds, counts, 'o-', color='tab:blue')
        ax1.set_title('Number of Valid Regions by Threshold')
        ax1.set_ylabel('Region Count')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Region size plot
        avg_sizes = [r['avg_region_size'] for r in results]
        min_sizes = [r['min_region_size'] for r in results]
        max_sizes = [r['max_region_size'] for r in results]
        
        ax2.plot(thresholds, avg_sizes, 'o-', color='tab:green', label='Average')
        ax2.plot(thresholds, min_sizes, 's--', color='tab:red', label='Minimum')
        ax2.plot(thresholds, max_sizes, '^--', color='tab:purple', label='Maximum')
        
        ax2.set_title('Region Size Statistics by Threshold')
        ax2.set_xlabel('Threshold Value')
        ax2.set_ylabel('Region Size (voxels)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "threshold_statistics.png"), dpi=150)
        plt.close(fig)

def main():
    detector = BoltHeadDetector(CONFIG)
    detector.run()

if __name__ == "__main__":
    main()

# exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Bolt_head\bolt_head_2.py').read())