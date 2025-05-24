import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.measure import label, regionprops_table
import matplotlib.pyplot as plt
import os
import logging
import json
from pathlib import Path
import matplotlib
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ElectrodeDetector:

    def __init__(self):
        pass

    @staticmethod
    def binarize_mask(mask_array):
        return (mask_array > 0).astype(np.uint8)

    @staticmethod
    def read_nrrd_file(file_path):
        try:
            image = sitk.ReadImage(str(file_path))
            return sitk.GetArrayFromImage(image), image
        except Exception as e:
            logging.error(f"Failed to read NRRD file {file_path}: {e}")
            return None, None

    @staticmethod
    def get_ras_centroids_and_pixel_counts(image_array, image):
        labeled_array = label(image_array > 0)
        props = regionprops_table(labeled_array, properties=['label', 'centroid', 'area'])
        if not props or len(props['label']) == 0:
            return {}, {}
        df = pd.DataFrame(props)
        centroids = {}
        pixel_counts = {}
        for _, row in df.iterrows():
            try:
                # Convert from numpy (z,y,x) to SimpleITK (x,y,z) index coordinates
                idx_coord = (row['centroid-2'], row['centroid-1'], row['centroid-0'])
                physical_point = image.TransformContinuousIndexToPhysicalPoint(idx_coord)
                # Convert LPS (ITK) to RAS (Slicer)
                ras_coord = (-physical_point[0], -physical_point[1], physical_point[2])
                centroids[int(row['label'])] = ras_coord
                # Store pixel count (area) for this electrode
                pixel_counts[int(row['label'])] = int(row['area'])
            except Exception as e:
                logging.error(f"Error processing label {row['label']}: {e}")
        return centroids, pixel_counts

    def process_mask(self, mask_path, patient_id, save_intermediates=True):
        """Process a single mask file and extract electrode information"""
        mask_name = os.path.basename(mask_path)
        mask_file_stem = Path(mask_name).stem
        mask_array, mask_image = self.read_nrrd_file(mask_path)

        if mask_array is None:
            return {
                'results': [],
                'centroids': {},
                'pixel_counts': {}
            }

        # Get electrode information
        mask_centroids, pixel_counts = self.get_ras_centroids_and_pixel_counts(mask_array, mask_image)

        # Note: Only saving _results.csv files, not intermediate electrode data

        # Create results for each detected electrode
        results = []
        for m_label, m_cent in mask_centroids.items():
            results.append({
                'Patient ID': patient_id,
                'Mask': mask_name,
                'Electrode Label': m_label,
                'RAS Coordinates': m_cent,
                'X': m_cent[0],
                'Y': m_cent[1], 
                'Z': m_cent[2],
                'Pixel Count': pixel_counts.get(m_label, 0)
            })

        return {
            'results': results,
            'centroids': mask_centroids,
            'pixel_counts': pixel_counts
        }

    def analyze_patient(self, mask_paths, patient_id="P1", save_intermediates=True):
        """Analyze all masks for a single patient - electrode detection only"""
        all_results = []
        all_mask_data = {}

        # Process each mask sequentially
        for mask_path in mask_paths:
            result = self.process_mask(mask_path, patient_id, save_intermediates)
            mask_name = os.path.basename(mask_path)
            mask_file_stem = Path(mask_name).stem
            all_mask_data[mask_file_stem] = result
            all_results.extend(result['results'])

            if save_intermediates and result['results']:
                # Save only _results.csv files
                mask_results_df = pd.DataFrame(result['results'])
                mask_results_df.to_csv(f'patient_{patient_id}_{mask_file_stem}_results.csv', index=False)
                logging.info(f"Saved results for {mask_file_stem}")

        # Create combined results dataframe for visualization
        results_df = pd.DataFrame(all_results)

        return results_df, all_mask_data

    def plot_electrodes(self, results_df, patient_id):
        """Create visualizations showing detected electrodes"""
        if results_df.empty:
            logging.warning("No electrodes to visualize")
            return

        mask_names = results_df['Mask'].unique()
        colors = plt.cm.tab10.colors

        # Create output directory
        output_dir = Path(f'patient_{patient_id}_plots')
        output_dir.mkdir(exist_ok=True)

        # Get coordinate bounds for consistent plots
        all_coords = np.array([[row['X'], row['Y'], row['Z']] for _, row in results_df.iterrows()])
        coord_min = np.min(all_coords, axis=0)
        coord_max = np.max(all_coords, axis=0)
        coord_range = coord_max - coord_min
        padding = coord_range * 0.1
        plot_min = coord_min - padding
        plot_max = coord_max + padding

        # Individual plots for each mask
        for idx, mask_name in enumerate(mask_names):
            mask_file_stem = Path(mask_name).stem
            mask_data = results_df[results_df['Mask'] == mask_name]
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            coords = np.array([[row['X'], row['Y'], row['Z']] for _, row in mask_data.iterrows()])
            pixel_counts = mask_data['Pixel Count'].values
            color = colors[idx % len(colors)]
            
            # Size points by pixel count
            sizes = (pixel_counts / pixel_counts.max()) * 200 + 50
            
            scatter = ax.scatter(coords[:,0], coords[:,1], coords[:,2],
                               c=[color], s=sizes, alpha=0.7,
                               edgecolors='black', linewidths=0.5)
            
            # Add labels for electrodes
            for _, row in mask_data.iterrows():
                ax.text(row['X'], row['Y'], row['Z'], 
                       f"E{row['Electrode Label']}\n({row['Pixel Count']}px)",
                       fontsize=8)
            
            ax.set_xlim(plot_min[0], plot_max[0])
            ax.set_ylim(plot_min[1], plot_max[1])
            ax.set_zlim(plot_min[2], plot_max[2])
            
            ax.set_xlabel('X (RAS)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Y (RAS)', fontsize=12, fontweight='bold')
            ax.set_zlabel('Z (RAS)', fontsize=12, fontweight='bold')
            
            detected_count = len(mask_data)
            ax.set_title(f'Detected Electrodes - Patient {patient_id} - {mask_file_stem}\n'
                        f'Count: {detected_count}, Avg Pixels: {pixel_counts.mean():.1f}',
                        fontsize=14, fontweight='bold')
            
            output_path = output_dir / f'p{patient_id}_{mask_file_stem}_electrodes.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved visualization for {mask_file_stem}")

        # Combined plot with all masks
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        for idx, mask_name in enumerate(mask_names):
            mask_file_stem = Path(mask_name).stem
            mask_data = results_df[results_df['Mask'] == mask_name]
            coords = np.array([[row['X'], row['Y'], row['Z']] for _, row in mask_data.iterrows()])
            pixel_counts = mask_data['Pixel Count'].values
            color = colors[idx % len(colors)]
            
            sizes = (pixel_counts / results_df['Pixel Count'].max()) * 200 + 50
            
            ax.scatter(coords[:,0], coords[:,1], coords[:,2],
                      c=[color], s=sizes, alpha=0.7,
                      edgecolors='black', linewidths=0.5,
                      label=f'{mask_file_stem} ({len(coords)})')
        
        ax.set_xlim(plot_min[0], plot_max[0])
        ax.set_ylim(plot_min[1], plot_max[1])
        ax.set_zlim(plot_min[2], plot_max[2])
        
        ax.set_xlabel('X (RAS)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (RAS)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (RAS)', fontsize=12, fontweight='bold')
        
        ax.set_title(f'All Detected Electrodes - Patient {patient_id}',
                    fontsize=16, fontweight='bold')
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        combined_output_path = output_dir / f'p{patient_id}_all_electrodes.png'
        plt.savefig(combined_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved combined visualization")

        # Detection count comparison chart
        mask_counts = [len(results_df[results_df['Mask'] == mask]) for mask in mask_names]
        mask_labels = [Path(mask).stem for mask in mask_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(mask_labels, mask_counts, color=colors[:len(mask_names)], 
                      alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, count in zip(bars, mask_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.title(f'Electrode Detection Count - Patient {patient_id}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Mask', fontsize=12)
        plt.ylabel('Number of Electrodes Detected', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        count_output_path = output_dir / f'p{patient_id}_detection_counts.png'
        plt.savefig(count_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved detection count chart")

def main(mask_files, patient_id='P1', save_intermediates=True, output_dir=None):
    logging.info(f"Starting electrode detection for patient {patient_id}")
    logging.info(f"Found {len(mask_files)} mask files")

    # Create output directory structure if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    if not mask_files:
        logging.error("No mask files provided")
        return

    # Initialize detector and run analysis
    detector = ElectrodeDetector()
    start_time = time.time()
    results_df, mask_data = detector.analyze_patient(
        mask_paths=mask_files,
        patient_id=patient_id,
        save_intermediates=save_intermediates,
    )

    end_time = time.time()
    logging.info(f"Detection completed in {end_time - start_time:.2f} seconds")

    if not results_df.empty:
        detector.plot_electrodes(results_df, patient_id)
        logging.info("Generated visualizations")
    else:
        logging.warning("No electrodes detected")

    logging.info(f"All results saved to {output_dir}")
    logging.info("Electrode detection completed successfully")

if __name__ == "__main__":
    import glob
    
    mask_folder = r"C:\Users\rocia\Downloads\TFG\Cohort\Extension"

    # Automatically get all NRRD files in the folder
    all_nrrd_files = glob.glob(os.path.join(mask_folder, '*.nrrd'))
    mask_files = sorted([f for f in all_nrrd_files if not os.path.basename(f).startswith('Filtered_DESCARGAR_roi')])

    patient_id = 'P1_test'
    save_intermediates = True
    output_dir = r"C:\Users\rocia\Downloads\TFG\Cohort\Extension\Just_plot_extension"

    # Run the main analysis
    main(
        mask_files,
        patient_id=patient_id,
        save_intermediates=save_intermediates,
        output_dir=output_dir
    )

#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Centroids_feature_extraction\just_centroids_coordinates.py').read())