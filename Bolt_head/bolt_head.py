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
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.lib.enums import TA_CENTER
from io import BytesIO
import matplotlib
from scipy.stats import kurtosis, skew
matplotlib.use('Agg')  # Use non-interactive backend for report generation

# Configuration Constants
CONFIG = {
    'threshold_value': 2400,  
    'adaptive_threshold': True,  
    'min_region_size': 100,
    'max_region_size': 800,
    'morph_kernel_size': 1,
    'principal_axis_length': 15,
    'output_dir': r"C:\Users\rocia\Downloads\TFG\Cohort\Bolt_heads\P1_entry_adaptive_pdf_features_lines_complete_algorithm"
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
        
        # Perform threshold analysis and get consensus mask
        print("Performing threshold effects analysis...")
        threshold_results, consensus_mask = self.analyze_threshold_effects_3d(
            volume_array, 
            brain_mask_array, 
            volume_helper
        )
        
        # Use the consensus mask instead of fixed threshold processing
        bolt_heads_mask = consensus_mask & (~brain_mask_array.astype(bool))
        cleaned_mask = self._apply_morphological_ops(bolt_heads_mask, volume_helper)
        
        if not np.any(cleaned_mask):
            print("No bolt head regions found in consensus mask outside the brain mask.")
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
            slicer.util.getNode('P1_CTp.3D'),  # CT volume
            slicer.util.getNode('patient1_mask_5')  # Brain mask
        )
        
    def _get_direction_matrix(self, volume_node):
        """Get IJK to RAS direction matrix."""
        matrix = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASDirectionMatrix(matrix)
        return matrix
        
    def _apply_morphological_ops(self, mask, volume_helper):
        """Apply morphological operations to clean the mask."""
        print("Applying morphological operations...")
        kernel = morphology.ball(self.config['morph_kernel_size'])
        cleaned_mask = morphology.binary_closing(mask, kernel)
        volume_helper.create_volume(cleaned_mask.astype(np.uint8), "Cleaned_Result", "P5_cleaned.nrrd")
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
        volume_helper.create_volume(filtered_mask, "Filtered_Bolt_Heads", "P5_filtered_bolt_heads.nrrd")
        
        return filtered_mask, region_info
        
###################################################################################################################
#### Bolt Head adaptive thresholding and feature extraction methods                                   #############
####################################################################################################################


    def _extract_histogram_features(self, volume_node, name="current_patient", tail_percent=0.01):
        """Extract histogram features for adaptive thresholding."""
        from scipy.stats import kurtosis, skew
        
        # Get raw CT data and remove background (values <= -1000)
        array = slicer.util.arrayFromVolume(volume_node).flatten()
        filtered_array = array[array > -1000]
        total_voxels = len(filtered_array)
        
        # 2400 HU-specific features
        target_hu = 2400
        hu_features = {
            'voxels_at_2400±100': np.sum((filtered_array >= 2300) & (filtered_array <= 2500)),
            'voxels_above_2400': np.sum(filtered_array > target_hu),
            'ratio_above_2400': np.sum(filtered_array > target_hu) / total_voxels,
            'hu_2400_percentile': np.mean(filtered_array > target_hu) * 100,
            'density_gradient_2300_2500': np.mean(filtered_array[(filtered_array >= 2300) & 
                                                            (filtered_array <= 2500)]) - target_hu
        }
        
        # High percentiles
        percentiles = {
            "percentile_99.5": np.percentile(filtered_array, 99.5),
            "percentile_99.7": np.percentile(filtered_array, 99.7),
            "percentile_99.8": np.percentile(filtered_array, 99.8),
            "percentile_99.9": np.percentile(filtered_array, 99.9),
            "percentile_99.95": np.percentile(filtered_array, 99.95),
            "percentile_99.97": np.percentile(filtered_array, 99.97),
            "percentile_99.98": np.percentile(filtered_array, 99.98),
            "percentile_99.99": np.percentile(filtered_array, 99.99)
        }
        
        # Combine all features
        return {
            **percentiles,
            **hu_features,
            "mean_above_2400": np.mean(filtered_array[filtered_array > target_hu]) if np.any(filtered_array > target_hu) else 0,
            "std_above_2400": np.std(filtered_array[filtered_array > target_hu]) if np.sum(filtered_array > target_hu) > 1 else 0,
        }

    def _apply_adaptive_threshold_algorithm(self, features):
        """Apply complete adaptive threshold algorithm from LaTeX document."""
        
        # Extract key features
        P_99_95 = features['percentile_99.95']  # ← NOW USING P99.95!
        P_99_97 = features['percentile_99.97']
        P_99_98 = features['percentile_99.98']
        ratio = features['ratio_above_2400']
        gradient = features['density_gradient_2300_2500']
        
        threshold = 2400  # Start with baseline
        decision_reason = "Standard case - baseline threshold"
        
        # SPECIAL CASE 1: High ratio + steep gradient (P7-like)
        if ratio > 0.002 and gradient < -30:  # ← Changed from -35 to -30 
            threshold = 2815
            decision_reason = f"P7-like: High electrode density (ratio={ratio:.6f}) + steep gradient ({gradient:.2f})"
        
        # SPECIAL CASE 2: Very low ratio (P6-like)
        elif ratio < 0.0003:
            threshold = 2325
            decision_reason = f"P6-like: Very low electrode density (ratio={ratio:.6f})"
        
        # SPECIAL CASE 3: Scanner saturation (P4-like) - IMPROVED DETECTION
        elif P_99_97 == P_99_98 and P_99_97 > 3060:  # ← More flexible than hardcoded 3071
            # Use midpoint between 99.8% and 99.9% percentiles for saturation case
            P_99_8 = features.get('percentile_99.8', P_99_97)
            P_99_9 = features.get('percentile_99.9', P_99_97)
            threshold = int((P_99_8 + P_99_9) / 2)
            decision_reason = f"P4-like: Scanner saturation (99.97%={P_99_97}, 99.98%={P_99_98})"
        
        # GENERAL ALGORITHM - THE MISSING PIECE!
        else:
            # Determine base threshold using P99.95 percentile
            if ratio < 0.0005:
                # Low ratio case: threshold above P99.95
                base_threshold = min(2400, P_99_95 + 200)
                decision_reason = f"Low ratio case: P99.95 + 200 = {P_99_95:.1f} + 200"
            else:
                # Higher ratio case: threshold below P99.95
                distance = min(300, ratio * 100000)
                base_threshold = min(2400, P_99_95 - distance)
                decision_reason = f"Higher ratio case: P99.95 - {distance:.0f} = {P_99_95:.1f} - {distance:.0f}"
            
            threshold = base_threshold
            
            # CRITICAL: Never exceed 99.97 percentile (universal upper bound)
            upper_limit = P_99_97 - 100
            if threshold > upper_limit:
                threshold = int(upper_limit)
                decision_reason += f" → capped at P99.97-100 = {upper_limit:.0f}"
        
        # SAFETY BOUNDS - THE FINAL MISSING PIECE!
        threshold = max(2325, min(2815, threshold))
        
        return int(threshold), decision_reason
    
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
        self._create_entry_points_volume(validated_regions, brain_mask_array, filtered_mask, volume_helper, volume_node)

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
        
        plt.savefig(os.path.join(self.config['output_dir'], "P5_bolt_heads_brain_validation.png"), dpi=300)
        plt.close()
        print(f"✅ Saved bolt heads validation plot to P2_bolt_heads_brain_validation.png")
        

    def _generate_bolt_heads_report(self, report_data, plot_path):
        """Generate a comprehensive PDF report of bolt heads and entry points."""
        pdf_path = os.path.join(self.config['output_dir'], "P2_bolt_heads_report.pdf")
        
        # Create the PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Add title
        title_style = ParagraphStyle(
            name='CenteredTitle',
            parent=styles['Heading1'],
            alignment=TA_CENTER,
            fontSize=16,
            spaceAfter=12
        )
        elements.append(Paragraph("Bolt Heads Analysis Report", title_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add date
        date_style = ParagraphStyle(
            name='Date',
            parent=styles['Normal'],
            alignment=TA_CENTER,
            fontSize=10,
            spaceAfter=12
        )
        elements.append(Paragraph(f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}", date_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add visualization
        try:
            img = Image(plot_path, width=6*inch, height=5*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.15*inch))
            
            # Add caption
            caption_style = ParagraphStyle(
                name='Caption',
                parent=styles['Italic'],
                alignment=TA_CENTER,
                fontSize=10,
                spaceAfter=12
            )
            elements.append(Paragraph("Figure 1: 3D visualization of bolt heads (yellow), " +
                                    "their orientation vectors (red arrows), and brain entry points (green dots)", 
                                    caption_style))
            elements.append(Spacer(1, 0.3*inch))
        except Exception as e:
            print(f"Error adding image to PDF: {e}")
        
        # Add summary table
        summary_data = [
            ['Bolt ID', 'Location (mm)', 'Entry Point (mm)', 'Volume (mm³)', 
            'Distance to\nSurface (mm)', 'Distance to\nEntry (mm)', 'Dimensions (mm)', 
            'Elongation', 'Compactness']
        ]
        
        for data in report_data:
            bolt_id = f"B{data['bolt_id']}"
            location = f"({data['centroid'][0]:.1f}, {data['centroid'][1]:.1f}, {data['centroid'][2]:.1f})"
            entry = f"({data['entry_point'][0]:.1f}, {data['entry_point'][1]:.1f}, {data['entry_point'][2]:.1f})"
            volume = f"{data['volume_mm3']:.1f}"
            surface_dist = f"{data['surface_distance']:.1f}"
            entry_dist = f"{data['entry_distance']:.1f}"
            dimensions = f"({data['dimensions'][0]:.1f}, {data['dimensions'][1]:.1f}, {data['dimensions'][2]:.1f})"
            elongation = f"{data['elongation']:.2f}"
            compactness = f"{data['compactness']:.2f}"
            
            summary_data.append([
                bolt_id, location, entry, volume, surface_dist, entry_dist, 
                dimensions, elongation, compactness
            ])
        
        # Create table
        table = Table(summary_data, colWidths=[0.5*inch, 1.3*inch, 1.3*inch, 0.8*inch, 
                                            0.8*inch, 0.8*inch, 1.3*inch, 0.8*inch, 0.8*inch])
        
        # Apply table style
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ])
        table.setStyle(table_style)
        elements.append(table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Add explanation
        elements.append(Paragraph("Metrics Explanation:", styles['Heading4']))
        metrics_explanation = [
            "<b>Volume</b>: Total volume of the bolt head in cubic millimeters.",
            "<b>Distance to Surface</b>: Distance from bolt head centroid to the nearest brain surface point.",
            "<b>Distance to Entry</b>: Distance along the bolt's orientation vector to the brain entry point.",
            "<b>Dimensions</b>: Length along principal axes (major, middle, minor) in millimeters.",
            "<b>Elongation</b>: Ratio of major to minor axis length (higher values indicate more elongated shapes).",
            "<b>Compactness</b>: Measure of how compact the object is (values closer to 1 indicate more spherical shapes)."
        ]
        
        for explanation in metrics_explanation:
            elements.append(Paragraph(explanation, styles['Normal']))
        
        elements.append(Spacer(1, 0.2*inch))
        
        # Add detailed bolt information
        elements.append(Paragraph("Detailed Bolt Orientation Information:", styles['Heading4']))
        
        for data in report_data:
            bolt_id = f"B{data['bolt_id']}"
            elements.append(Paragraph(f"<b>{bolt_id}</b>", styles['Heading5']))
            
            vector = data['vector']
            normalized_vector = vector / np.linalg.norm(vector)
            
            vector_info = [
                f"Orientation Vector: ({vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f})",
                f"Normalized Direction: ({normalized_vector[0]:.3f}, {normalized_vector[1]:.3f}, {normalized_vector[2]:.3f})"
            ]
            
            for info in vector_info:
                elements.append(Paragraph(info, styles['Normal']))
            
            elements.append(Spacer(1, 0.1*inch))
        
        # Build the PDF
        try:
            doc.build(elements)
            print(f"✅ Saved bolt heads report to {pdf_path}")
        except Exception as e:
            print(f"Error building PDF: {e}")
            import traceback
            traceback.print_exc()     









        #####################
    def _create_entry_points_volume(self, validated_regions, brain_mask, filtered_mask, volume_helper, volume_node):
        """Create volume with entry points, bolt heads, and connecting trajectories, and generate reports."""
        # Create separate masks for different components
        bolt_head_mask = np.zeros_like(brain_mask, dtype=np.uint8)
        entry_points_mask = np.zeros_like(brain_mask, dtype=np.uint8)
        trajectory_mask = np.zeros_like(brain_mask, dtype=np.uint8)
        
        # Create markups node for entry points
        markups_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "BoltEntryPoints")
        markups_node.CreateDefaultDisplayNodes()
        markups_node.GetDisplayNode().SetSelectedColor(0, 1, 0)
        markups_node.GetDisplayNode().SetPointSize(5)


        region_index_to_mask_value = {}
        
        # First, add bolt heads to the mask (value = 1)
        for i, info in enumerate(validated_regions):
            bolt_head_mask[filtered_mask == info['label']] = 1
        
        # Then add entry points to the mask (value = 2) and trajectories (value = 3)
        for i, info in enumerate(validated_regions):
            if 'brain_entry_point' in info and info['brain_entry_point'] is not None:
                # Get centroid and entry point
                centroid = np.array(info['physical_centroid'])
                entry_point = np.array(info['brain_entry_point'])
                
                # Convert to voxel coordinates
                centroid_voxel = np.round(
                    (centroid - np.array(volume_node.GetOrigin())) / np.array(volume_node.GetSpacing())
                ).astype(int)
                
                entry_point_voxel = np.round(
                    (entry_point - np.array(volume_node.GetOrigin())) / np.array(volume_node.GetSpacing())
                ).astype(int)
                
                try:
                    x, y, z = entry_point_voxel
                    # Mark entry points with value 2
                    entry_points_mask[
                        max(0, x-1):min(entry_points_mask.shape[0], x+2),
                        max(0, y-1):min(entry_points_mask.shape[1], y+2),
                        max(0, z-1):min(entry_points_mask.shape[2], z+2)
                    ] = 2
                    
                    # Create trajectory line using Bresenham's algorithm to connect bolt head to entry point
                    points = self._bresenham_line_3d(
                        centroid_voxel[0], centroid_voxel[1], centroid_voxel[2],
                        entry_point_voxel[0], entry_point_voxel[1], entry_point_voxel[2]
                    )
                    
                    # Add trajectory line to the mask (value = 3)
                    for point in points:
                        x, y, z = point
                        if (0 <= x < trajectory_mask.shape[0] and 
                            0 <= y < trajectory_mask.shape[1] and 
                            0 <= z < trajectory_mask.shape[2]):
                            trajectory_mask[x, y, z] = 3
                    
                    
                except IndexError:
                    print(f"Warning: Entry point {entry_point_voxel} out of brain mask bounds")
        
        # Combine the masks (bolt heads = 1, entry points = 2, trajectories = 3)
        # We'll use a priority system where trajectories override bolt heads and entry points
        combined_mask = np.zeros_like(brain_mask, dtype=np.uint8)
        combined_mask[bolt_head_mask > 0] = 1
        combined_mask[entry_points_mask > 0] = 2
        combined_mask[trajectory_mask > 0] = 3
        
        # Create volume nodes for each mask
        bolt_head_node = volume_helper.create_volume(
            bolt_head_mask.astype(np.uint8), 
            "BoltHeadsMask",
            "P2_bolt_heads.nrrd"
        )
        
        entry_points_node = volume_helper.create_volume(
            entry_points_mask.astype(np.uint8), 
            "EntryPointsMask",
            "P2_brain_entry_points.nrrd"
        )
        
        trajectory_node_volume = volume_helper.create_volume(
            trajectory_mask.astype(np.uint8),
            "TrajectoryMask",
            "P2_bolt_trajectories.nrrd"
        )
        
        combined_node = volume_helper.create_volume(
            combined_mask.astype(np.uint8),
            "P2_CombinedBoltHeadEntryPointsTrajectoryMask",
            "P2_combined_visualization.nrrd"
        )

        
        # Continue with the existing code for entry points markups
        labeled_image = label(entry_points_mask)
        regions = regionprops(labeled_image)
        label_to_ras = {}
        ras_coordinates_list = []
        
        # Process regions and create markups
        for region in regions:
            centroid_ijk = region.centroid
            ijk_for_conversion = [centroid_ijk[2], centroid_ijk[1], centroid_ijk[0]]
            ras_coords = self._get_ras_coordinates_from_ijk(entry_points_node, ijk_for_conversion)
            
            markups_node.AddControlPoint(
                ras_coords[0], ras_coords[1], ras_coords[2],
                f"Entry_{region.label}"
            )
            label_to_ras[region.label] = ras_coords
            ras_coordinates_list.append(ras_coords)
        
        # Save markups
        save_path = os.path.join(self.config['output_dir'], "P2_entry_points_markups.fcsv")
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
        csv_path = os.path.join(self.config['output_dir'], "P2_brain_entry_points_report.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved brain entry points report to {csv_path}")
        
        print(f"Number of validated regions with entry points: {sum(1 for info in validated_regions if 'brain_entry_point' in info and info['brain_entry_point'] is not None)}")
        print(f"Number of regions found by regionprops: {len(regions)}")
        print(f"Number of report entries generated: {len(report_data)}")
        
        return entry_points_mask, ras_coordinates_list

    def _bresenham_line_3d(self, x0, y0, z0, x1, y1, z1):
        """
        Implementation of 3D Bresenham's line algorithm to create a line between two points in a 3D volume.
        Returns a list of points (voxel coordinates) along the line.
        """
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dz = abs(z1 - z0)
        
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        sz = 1 if z1 > z0 else -1
        
        if dx >= dy and dx >= dz:
            err_y = dx // 2
            err_z = dx // 2
            
            x, y, z = x0, y0, z0
            while x != x1:
                points.append((x, y, z))
                err_y -= dy
                if err_y < 0:
                    y += sy
                    err_y += dx
                
                err_z -= dz
                if err_z < 0:
                    z += sz
                    err_z += dx
                
                x += sx
            
        elif dy >= dx and dy >= dz:
            err_x = dy // 2
            err_z = dy // 2
            
            x, y, z = x0, y0, z0
            while y != y1:
                points.append((x, y, z))
                err_x -= dx
                if err_x < 0:
                    x += sx
                    err_x += dy
                
                err_z -= dz
                if err_z < 0:
                    z += sz
                    err_z += dy
                
                y += sy
        
        else:
            err_x = dz // 2
            err_y = dz // 2
            
            x, y, z = x0, y0, z0
            while z != z1:
                points.append((x, y, z))
                err_x -= dx
                if err_x < 0:
                    x += sx
                    err_x += dz
                
                err_y -= dy
                if err_y < 0:
                    y += sy
                    err_y += dz
                
                z += sz
        
        # Add the last point
        points.append((x1, y1, z1))
        
        # Ensure we don't have duplicate points
        return list(dict.fromkeys(map(tuple, points)))

    def _plot_entry_points(self, region_info, filtered_mask, brain_mask, spacing, origin):
        """
        Plot bolt heads with entry points and trajectories, and generate a comprehensive PDF report
        with quantitative metrics for each bolt head.
        """
        # PART 1: Create the 3D visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot brain surface
        self._plot_surface(ax, brain_mask, spacing, origin, 'lightblue', 0.3)
        
        # Data collection for report
        report_data = []
        bolt_idx = 1
        
        # Plot bolt heads, vectors and trajectories
        for info in region_info:
            if 'brain_entry_point' not in info or info['brain_entry_point'] is None:
                continue
                
            # Plot bolt head
            self._plot_surface(ax, filtered_mask == info['label'], spacing, origin, 'yellow', 0.8)
            
            # Extract data
            centroid = np.array(info['physical_centroid'])
            vector = np.array(info['principal_axis'])
            entry_point = np.array(info['brain_entry_point'])
            
            # Plot bolt vector
            ax.quiver(*centroid, *vector, color='red', linewidth=2, arrow_length_ratio=0.2)
            
            # Plot entry point
            ax.scatter(entry_point[0], entry_point[1], entry_point[2],
                    color='green', s=100, marker='o', label='Entry Points' if bolt_idx == 1 else "")
            
            # Plot trajectory line
            ax.plot([centroid[0], entry_point[0]], 
                    [centroid[1], entry_point[1]], 
                    [centroid[2], entry_point[2]], 
                    color='purple', linestyle='-', linewidth=1.5, 
                    label='Trajectories' if bolt_idx == 1 else "")
            
            # Add a label to each bolt head
            ax.text(centroid[0], centroid[1], centroid[2], f"B{bolt_idx}", color='black', 
                    fontsize=12, fontweight='bold')
            
            # Calculate additional metrics
            bolt_to_entry_distance = np.linalg.norm(entry_point - centroid)
            vector_norm = np.linalg.norm(vector)
            
            # Calculate 3D shape metrics
            bolt_mask = filtered_mask == info['label']
            volume_mm3 = np.sum(bolt_mask) * spacing[0] * spacing[1] * spacing[2]
            
            # Calculate approximate dimensions (along principal axes)
            try:
                bolt_points = np.argwhere(bolt_mask)
                if len(bolt_points) > 3:  # Need at least 3 points for PCA
                    # Center the points
                    centered_points = bolt_points - np.mean(bolt_points, axis=0)
                    
                    # Calculate covariance matrix and eigenvalues
                    cov_matrix = np.cov(centered_points, rowvar=False)
                    eigenvalues = np.linalg.eigvals(cov_matrix)
                    
                    # Sort eigenvalues in descending order
                    eigenvalues = np.sort(eigenvalues)[::-1]
                    
                    # Apply spacing to get dimensions in mm
                    dimensions = 2 * np.sqrt(eigenvalues) * np.mean(spacing)
                    
                    # Calculate elongation (ratio of major to minor axis)
                    elongation = dimensions[0] / dimensions[2] if dimensions[2] > 0 else float('inf')
                    
                    # Calculate compactness (ratio of volume to surface area)
                    # Approximating using eigenvalues
                    compactness = volume_mm3 / (4*np.pi*np.prod(dimensions)**(1/3))
                else:
                    dimensions = [0, 0, 0]
                    elongation = 0
                    compactness = 0
            except Exception as e:
                print(f"Error calculating bolt dimensions for bolt {bolt_idx}: {e}")
                dimensions = [0, 0, 0]
                elongation = 0
                compactness = 0
            
            # Collect data for report
            report_data.append({
                'bolt_id': bolt_idx,
                'centroid': centroid,
                'entry_point': entry_point,
                'vector': vector,
                'surface_distance': info.get('surface_distance', 0),
                'entry_distance': info.get('entry_distance', 0),
                'direct_distance': bolt_to_entry_distance,
                'volume': info['volume'],
                'volume_mm3': volume_mm3,
                'dimensions': dimensions,
                'elongation': elongation,
                'compactness': compactness,
            })
            
            bolt_idx += 1
        
        # Finalize the plot
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('Bolt Heads with Brain Entry Points and Trajectories')
        ax.view_init(elev=30, azim=45)
        
        # Save the plot
        plot_path = os.path.join(self.config['output_dir'], "P2_bolt_heads_entry_points.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"✅ Saved bolt heads entry points visualization to {plot_path}")
        
        # PART 2: Generate PDF report
        self._generate_bolt_heads_report(report_data, plot_path)
        
        return report_data
        
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
        """Enhanced threshold analysis with adaptive selection."""
        
        # Get the volume node for feature extraction
        volume_node, _ = self._load_nodes()
        
        print("🔍 Extracting CT features for adaptive thresholding...")
        
        # Extract features and apply adaptive threshold
        features = self._extract_histogram_features(volume_node)
        adaptive_threshold, decision_reason = self._apply_adaptive_threshold_algorithm(features)
        
        # Use adaptive threshold instead of fixed 2400
        threshold_values = [adaptive_threshold]
        
        print(f"\n🎯 ADAPTIVE THRESHOLD SELECTED: {adaptive_threshold} HU")
        print(f"📊 Decision: {decision_reason}")
        print(f"📊 Key metrics:")
        print(f"   • Ratio above 2400: {features['ratio_above_2400']:.6f}")
        print(f"   • 99.97 percentile: {features['percentile_99.97']:.1f} HU")
        print(f"   • 99.98 percentile: {features['percentile_99.98']:.1f} HU")
        print(f"   • Density gradient: {features['density_gradient_2300_2500']:.2f}")
        
        # Update config to use adaptive threshold
        original_threshold = self.config['threshold_value']
        self.config['threshold_value'] = adaptive_threshold
        
        # Create output directory
        threshold_dir = os.path.join(self.config['output_dir'], 'threshold_analysis')
        os.makedirs(threshold_dir, exist_ok=True)
        
        print("Performing optimized threshold analysis with adaptive threshold...")
        
        # Initialize voting mask and results
        voting_mask = np.zeros_like(volume_array, dtype=np.uint8)
        results = []
        
        # Process the adaptive threshold
        for threshold in threshold_values:
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
            
            # Create filtered mask and update voting
            filtered_mask = np.zeros_like(cleaned_mask, dtype=np.uint8)
            for region in valid_regions:
                filtered_mask[labeled_image == region.label] = 1
            
            # Apply weighting
            weight = 2
            voting_mask += filtered_mask * weight
            
            # Collect statistics
            stats = {
                'threshold': threshold,
                'adaptive_threshold': adaptive_threshold,
                'decision_reason': decision_reason,
                'total_voxels': np.sum(filtered_mask),
                'region_count': len(valid_regions),
                'weight': weight,
                'ratio_above_2400': features['ratio_above_2400'],
                'percentile_99_97': features['percentile_99.97'],
                'percentile_99_98': features['percentile_99.98'],
                'density_gradient': features['density_gradient_2300_2500']
            }
            results.append(stats)
            
            print(f"Processed adaptive threshold {threshold} in {time.time()-start_time:.2f}s")
        
        # Calculate consensus
        total_weight = sum(r['weight'] for r in results)
        consensus_mask = voting_mask >= (total_weight * 0.35)
        consensus_labeled = label(consensus_mask)
        consensus_regions = regionprops(consensus_labeled)
        
        # Add consensus metrics
        for r in results:
            r['consensus_voxels'] = np.sum(consensus_mask)
            r['consensus_regions'] = len(consensus_regions)
        
        # Visualization
        self._visualize_consensus_results(
            consensus_regions,
            consensus_labeled,
            brain_mask_array,
            volume_helper,
            threshold_dir
        )
        
        # Save enhanced results
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(threshold_dir, 'adaptive_threshold_results.csv'), index=False)
        
        # Save detailed features
        features_df = pd.DataFrame([features])
        features_df.to_csv(os.path.join(threshold_dir, 'extracted_features.csv'), index=False)
        
        print(f"✅ Saved adaptive threshold analysis to {threshold_dir}")
        print(f"📁 Files generated:")
        print(f"   • adaptive_threshold_results.csv (threshold decisions)")
        print(f"   • extracted_features.csv (detailed features)")
        
        return results, consensus_mask

    def _visualize_consensus_results(self, regions, labeled_image, brain_mask, volume_helper, output_dir):
        """Visualize only the consensus regions with brain context."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot brain surface
        brain_verts, brain_faces, _, _ = marching_cubes(
            brain_mask, 
            level=0.5, 
            spacing=volume_helper.spacing
        )
        brain_verts += volume_helper.origin
        ax.plot_trisurf(
            brain_verts[:, 0], brain_verts[:, 1], brain_verts[:, 2],
            triangles=brain_faces, color='lightblue', alpha=0.2
        )
        
        # Plot consensus regions
        for region in regions:
            region_mask = labeled_image == region.label
            if np.any(region_mask):
                verts, faces, _, _ = marching_cubes(
                    region_mask, 
                    level=0.5, 
                    spacing=volume_helper.spacing
                )
                verts += volume_helper.origin
                ax.plot_trisurf(
                    verts[:, 0], verts[:, 1], verts[:, 2],
                    triangles=faces, color='orange', alpha=0.8
                )
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'Consensus Bolt Head Regions ({len(regions)} found)')
        ax.view_init(elev=30, azim=45)
        
        plt.savefig(os.path.join(output_dir, "consensus_regions.png"), dpi=300)
        plt.close(fig)

def main():
    detector = BoltHeadDetector(CONFIG)
    detector.run()

if __name__ == "__main__":
    main()

# exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Bolt_head\bolt_head.py').read())