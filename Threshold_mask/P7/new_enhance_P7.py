import slicer
import numpy as np
import vtk
from vtk.util import numpy_support
import cv2
from skimage import exposure, filters, morphology
from skimage.exposure import rescale_intensity
import pywt
import pywt.data
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy import ndimage
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from skimage.morphology import remove_small_objects
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage import segmentation, measure, feature, draw
from skimage.filters import sobel
from scipy.ndimage import distance_transform_edt
from skimage.restoration import denoise_nl_means
from scipy.ndimage import watershed_ift, gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import active_contour
from skimage.draw import ellipse
from skimage import img_as_float
from skimage.filters import gaussian
from skimage.feature import canny
from skimage.transform import rescale
from skimage.filters import gaussian, laplace
from skimage.transform import rescale
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects
import pandas as pd
from skimage.measure import regionprops
from skimage.filters import frangi
from scipy.ndimage import median_filter
import vtk.util.numpy_support as ns
from skimage.morphology import disk
from skimage.filters import median
from skimage import morphology, measure, segmentation, filters, feature
from skimage import morphology as skmorph
from skimage.color import rgb2gray
from scipy import ndimage as ndi
from skimage.filters import gaussian
from skimage.restoration import denoise_wavelet
from skimage.exposure import adjust_gamma
from scipy.ndimage import distance_transform_edt, label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import logging
from skimage import restoration
from Threshold_mask.enhance_ctp import gamma_correction, sharpen_high_pass, log_transform_slices, wavelet_denoise, wavelet_nlm_denoise, morphological_operation, apply_clahe,morph_operations  

def shannon_entropy(image):
    """Calculate Shannon entropy of an image."""
    import numpy as np
    # Convert to probabilities by calculating histogram
    hist, _ = np.histogram(image, bins=256, density=True)
    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]
    # Calculate entropy
    return -np.sum(hist * np.log2(hist))

def extract_advanced_features(volume_array, hist=None, bin_centers=None):
    import numpy as np
    from scipy import stats
    
    features = {}
    features['min'] = np.min(volume_array)
    features['max'] = np.max(volume_array)
    features['mean'] = np.mean(volume_array)
    features['median'] = np.median(volume_array)
    features['std'] = np.std(volume_array)
    features['p25'] = np.percentile(volume_array, 25)
    features['p75'] = np.percentile(volume_array, 75)
    features['p95'] = np.percentile(volume_array, 95)
    features['p99'] = np.percentile(volume_array, 99)
    # Compute histogram if not provided
    if hist is None or bin_centers is None:
        hist, bin_edges = np.histogram(volume_array.flatten(), bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Handle zero-peak special case for small dot segmentation
    zero_idx = np.argmin(np.abs(bin_centers))  # Index closest to zero
    zero_peak_height = hist[zero_idx]
    features['zero_peak_height'] = zero_peak_height
    features['zero_peak_ratio'] = zero_peak_height / np.sum(hist) if np.sum(hist) > 0 else 0
    # Add very high percentiles that might better capture small bright dots
    features['p99.5'] = np.percentile(volume_array, 99.5)
    features['p99.9'] = np.percentile(volume_array, 99.9)
    features['p99.99'] = np.percentile(volume_array, 99.99)
    # Calculate skewness and kurtosis for the distribution
    features['skewness'] = stats.skew(volume_array.flatten())
    features['kurtosis'] = stats.kurtosis(volume_array.flatten())
    # Calculate non-zero statistics (ignoring background)
    non_zero_values = volume_array[volume_array > 0]
    if len(non_zero_values) > 0:
        features['non_zero_min'] = np.min(non_zero_values)
        features['non_zero_mean'] = np.mean(non_zero_values)
        features['non_zero_median'] = np.median(non_zero_values)
        features['non_zero_std'] = np.std(non_zero_values)
        features['non_zero_count'] = len(non_zero_values)
        features['non_zero_ratio'] = len(non_zero_values) / volume_array.size
        # Calculate skewness and kurtosis for non-zero values
        if len(non_zero_values) > 3:  # Need at least 3 points for skewness calculation
            features['non_zero_skewness'] = stats.skew(non_zero_values)
            features['non_zero_kurtosis'] = stats.kurtosis(non_zero_values)
        else:
            features['non_zero_skewness'] = 0
            features['non_zero_kurtosis'] = 0
    else:
        features['non_zero_min'] = 0
        features['non_zero_mean'] = 0
        features['non_zero_median'] = 0
        features['non_zero_std'] = 0
        features['non_zero_count'] = 0
        features['non_zero_ratio'] = 0
        features['non_zero_skewness'] = 0
        features['non_zero_kurtosis'] = 0
    # Calculate high-intensity island statistics (potential dots)
    high_threshold = features['p99']
    high_values = volume_array[volume_array >= high_threshold]
    if len(high_values) > 0:
        features['high_intensity_count'] = len(high_values)
        features['high_intensity_mean'] = np.mean(high_values)
        features['high_intensity_max'] = np.max(high_values)
        features['high_intensity_ratio'] = len(high_values) / volume_array.size
    else:
        features['high_intensity_count'] = 0
        features['high_intensity_mean'] = 0
        features['high_intensity_max'] = 0
        features['high_intensity_ratio'] = 0
    # Find peaks (ignoring the zero peak if it's dominant)
    peaks = []
    for i in range(1, len(hist)-1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            # Add this peak only if it's not the zero peak
            if abs(bin_centers[i]) > 0.01:  # Small tolerance to avoid numerical issues
                peaks.append((bin_centers[i], hist[i]))
    # Sort peaks by height (descending)
    peaks.sort(key=lambda x: x[1], reverse=True)
    # Extract info about top non-zero peaks
    if peaks:
        features['non_zero_peak1_value'] = peaks[0][0]
        features['non_zero_peak1_height'] = peaks[0][1]
        
        if len(peaks) > 1:
            features['non_zero_peak2_value'] = peaks[1][0]
            features['non_zero_peak2_height'] = peaks[1][1]
            features['non_zero_peak_distance'] = abs(features['non_zero_peak1_value'] - features['non_zero_peak2_value'])
        else:
            features['non_zero_peak2_value'] = features['non_zero_peak1_value']
            features['non_zero_peak2_height'] = 0
            features['non_zero_peak_distance'] = 0
    else:
        # No non-zero peaks found
        features['non_zero_peak1_value'] = features['mean']
        features['non_zero_peak1_height'] = 0
        features['non_zero_peak2_value'] = features['mean']
        features['non_zero_peak2_height'] = 0
        features['non_zero_peak_distance'] = 0
    # Add specialized dot detection features
    # Contrast ratios that might help identify dots
    features['contrast_ratio'] = features['max'] / features['mean'] if features['mean'] > 0 else 0
    features['p99_mean_ratio'] = features['p99'] / features['mean'] if features['mean'] > 0 else 0
    # Distance between percentiles
    features['p75_p25'] = features['p75'] - features['p25']  # Interquartile range
    features['p95_p5'] = np.percentile(volume_array, 95) - np.percentile(volume_array, 5)
    features['p99_p1'] = np.percentile(volume_array, 99) - np.percentile(volume_array, 1)
    # Entropy
    features['entropy'] = shannon_entropy(volume_array)
    return features

def collect_histogram_data(enhanced_volumes, threshold_tracker, outputDir=None):
    import numpy as np
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    from skimage import exposure
    
    if outputDir is None:
        outputDir = slicer.app.temporaryPath()
    
    # Create histograms directory
    hist_dir = os.path.join(outputDir, "histograms")
    if not os.path.exists(hist_dir):
        os.makedirs(hist_dir)
    
    histogram_data = {}
    hist_features = {}
    
    # Process each enhanced volume
    for method_name, volume_array in enhanced_volumes.items():
        # Skip binary threshold results (DESCARGAR_*)
        if method_name.startswith('DESCARGAR_'):
            continue
            
        # Create histogram
        hist, bin_edges = np.histogram(volume_array.flatten(), bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Save histogram data
        histogram_data[method_name] = {
            'hist': hist,
            'bin_centers': bin_centers
        }
        
        # Extract advanced features (instead of basic features)
        hist_features[method_name] = extract_advanced_features(volume_array, hist, bin_centers)
        
        # Add threshold if available
        if method_name in threshold_tracker:
            hist_features[method_name]['threshold'] = threshold_tracker[method_name]
        
        # # Plot histogram
        # plt.figure(figsize=(10, 6))
        # plt.plot(bin_centers, hist)
        # plt.title(f'Histogram for {method_name}')
        # plt.xlabel('Pixel Value')
        # plt.ylabel('Frequency')
        
        # # Add a vertical line for threshold if available
        # if method_name in threshold_tracker:
        #     threshold = threshold_tracker[method_name]
        #     plt.axvline(x=threshold, color='r', linestyle='--', 
        #                 label=f'Threshold = {threshold}')
        #     plt.legend()
        
        # # Save plot
        # plt.savefig(os.path.join(hist_dir, f'histogram_{method_name}.png'))
        # plt.close()
    
    # Save all histogram features to CSV
    features_df = pd.DataFrame.from_dict(hist_features, orient='index')
    features_df.to_csv(os.path.join(outputDir, 'histogram_features.csv'))
    
    # Create a comprehensive report with all histograms
    create_histogram_report(histogram_data, threshold_tracker, hist_features, outputDir)
    
    
    return histogram_data

def create_histogram_report(histogram_data, threshold_tracker,hist_features,  outputDir):
    """
    Create a comprehensive report with all histograms with improved visualization.
    
    Parameters:
    -----------
    histogram_data : dict
        Dictionary of histogram data
    threshold_tracker : dict
        Dictionary of thresholds
    outputDir : str
        Directory to save report
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from matplotlib import cm
    from matplotlib.colors import Normalize
    
    # Create plots directory
    plots_dir = os.path.join(outputDir, "combined_plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Group methods by approach
    approaches = {
        'Original CTP': ['OG_volume_array', 'OG_gaussian_volume_og', 'OG_gamma_volume_og', 'OG_sharpened'],
        'ROI with Gamma': ['PRUEBA_roi_plus_gamma_mask', 'PRUEBA_roi_plus_gamma_mask_clahe', 'PRUEBA_WAVELET_NL'],
        'ROI Only': ['roi_volume', 'wavelet_only_roi', 'gamma_only_roi', 'sharpened_wavelet_roi', 'sharpened_roi_only_roi', 'LOG_roi'],
        'ROI Plus Gamma After': ['2_gaussian_volume_roi', '2_gamma_correction', '2_tophat', '2_sharpened', '2_LOG', '2_wavelet_roi', '2_erode', '2_gaussian_2', '2_sharpening_2_trial'],
        'Wavelet ROI': ['NUEVO_NLMEANS'],
        'Original Idea': ['ORGINAL_IDEA_gaussian', 'ORGINAL_IDEA_gamma_correction', 'ORGINAL_IDEA_sharpened', 'ORIGINAL_IDEA_SHARPENED_OPENING', 'ORIGINAL_IDEA_wavelet', 'ORGINAL_IDEA_gaussian_2', 'ORIGINAL_IDEA_GAMMA_2', 'OG_tophat_1'],
        'First Try': ['FT_gaussian', 'FT_tophat_1', 'FT_RESTA_TOPHAT_GAUSSIAN', 'FT_gamma_correction', 'FT_sharpened', 'FT_gaussian_2', 'FT_gamma_2', 'FT_opening', 'FT_closing', 'FT_erode_2', 'FT_tophat', 'FT_gaussian_3']
    }
    
    # Define a better color palette
    colormap = cm.get_cmap('viridis', 10)
    
    # Plot histograms by approach
    for approach_name, methods in approaches.items():
        # Filter available methods
        available_methods = [m for m in methods if m in histogram_data]
        
        if not available_methods:
            continue
            
        # Create plot with subplots for this approach - adaptive size based on number of methods
        num_methods = len(available_methods)
        cols = min(3, num_methods)  # Maximum 3 columns
        rows = (num_methods + cols - 1) // cols  # Ceiling division
        
        # Adaptive figure size - width based on columns, height based on rows
        fig_width = 6 * cols
        fig_height = 4 * rows
        
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        fig.suptitle(f'Histograms for {approach_name} Approach', fontsize=16)
        
        # Flatten axes for easy indexing
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        axes_flat = axes.flatten()
        
        # Plot each method
        for i, method in enumerate(available_methods):
            if i < len(axes_flat):
                data = histogram_data[method]
                hist = data['hist']
                bin_centers = data['bin_centers']
                
                # Calculate mean and median for annotations
                if 'mean' in hist_features[method]:
                    mean_val = hist_features[method]['mean']
                    median_val = hist_features[method]['median']
                else:
                    # Approximate from histogram if not available
                    total = np.sum(hist)
                    cumsum = np.cumsum(hist)
                    median_idx = np.argmin(np.abs(cumsum - total/2))
                    median_val = bin_centers[median_idx]
                    mean_val = np.sum(bin_centers * hist) / total
                
                # Normalize histogram for better visualization
                normalized_hist = hist / np.max(hist)
                
                # Plot with better styling
                ax = axes_flat[i]
                color = colormap(i / len(available_methods))
                ax.plot(bin_centers, normalized_hist, color=color, linewidth=2)
                
                # Add semitransparent fill under curve
                ax.fill_between(bin_centers, normalized_hist, alpha=0.3, color=color)
                
                # Use log scale for y-axis to better see differences
                ax.set_yscale('log')
                ax.set_ylim(bottom=1e-8)  # Minimum value for log scale
                
                # Add grid for easier reading
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Set title and labels
                ax.set_title(method, fontsize=12, fontweight='bold')
                ax.set_xlabel('Pixel Value', fontsize=10)
                ax.set_ylabel('Normalized Frequency (log)', fontsize=10)
                
                # Add threshold line if available
                if method in threshold_tracker:
                    threshold = threshold_tracker[method]
                    ax.axvline(x=threshold, color='r', linestyle='-', 
                               linewidth=2, label=f'Threshold = {threshold:.2f}')
                
                # Add mean and median lines
                ax.axvline(x=mean_val, color='green', linestyle='--', 
                           linewidth=1.5, label=f'Mean = {mean_val:.2f}')
                ax.axvline(x=median_val, color='blue', linestyle=':', 
                           linewidth=1.5, label=f'Median = {median_val:.2f}')
                
                # Add legend
                ax.legend(loc='upper right', fontsize=8)
        
        # Hide unused subplots
        for j in range(num_methods, len(axes_flat)):
            axes_flat[j].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  
        plt.subplots_adjust(hspace=0.4, wspace=0.3)  
        plt.savefig(os.path.join(plots_dir, f'histograms_{approach_name.replace(" ", "_")}.png'), dpi=300)
        plt.close()
    
    # Enhanced plot for comparing all methods with thresholds
    threshold_methods = [m for m in threshold_tracker.keys() if m in histogram_data]
    
    # if threshold_methods:
    #     # Create two comparison plots: one regular and one with log scale
    #     for scale_type in ['linear', 'log']:
    #         # Determine appropriate figure size based on number of methods
    #         fig_width = min(20, 12 + len(threshold_methods) * 0.5)
            
    #         plt.figure(figsize=(fig_width, 10))
            
    #         # Use colormap for better differentiation between methods
    #         norm = Normalize(vmin=0, vmax=len(threshold_methods)-1)
            
    #         # Plot each method with distinct color
    #         for i, method in enumerate(threshold_methods):
    #             data = histogram_data[method]
    #             # Normalize histogram
    #             normalized_hist = data['hist'] / np.max(data['hist'])
    #             color = colormap(norm(i))
                
    #             plt.plot(data['bin_centers'], normalized_hist, 
    #                      label=method, color=color, linewidth=2)
                
    #             # Add threshold line with matching color
    #             threshold = threshold_tracker[method]
    #             plt.axvline(x=threshold, linestyle='--', color=color, alpha=0.7)
                
    #             # Annotate threshold value
    #             plt.text(threshold, 0.5 + i*0.05, f'{method}: {threshold:.2f}', 
    #                      rotation=90, fontsize=8, color=color)
            
    #         if scale_type == 'log':
    #             plt.yscale('log')
    #             plt.ylim(bottom=1e-8)
    #             plt.title('Normalized Histograms with Thresholds (Log Scale)')
    #         else:
    #             plt.title('Normalized Histograms with Thresholds')
                
    #         plt.xlabel('Pixel Value', fontsize=12)
    #         plt.ylabel('Normalized Frequency', fontsize=12)
    #         plt.grid(True, alpha=0.3, linestyle='--')
            
    #         # Create legend with two columns if many methods
    #         if len(threshold_methods) > 6:
    #             ncol = 2
    #         else:
    #             ncol = 1
                
    #         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=ncol, fontsize=10)
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(plots_dir, f'all_thresholds_comparison_{scale_type}.png'), dpi=300)
    #         plt.close()
            
    # Create a heatmap of threshold values for quick comparison
    if threshold_methods:
        plt.figure(figsize=(12, len(threshold_methods)/2 + 2))
        
        # Get threshold values and sort them
        thresholds = [threshold_tracker[m] for m in threshold_methods]
        sorted_indices = np.argsort(thresholds)
        sorted_methods = [threshold_methods[i] for i in sorted_indices]
        sorted_thresholds = [thresholds[i] for i in sorted_indices]
        
        # Create horizontal bar chart of thresholds
        bars = plt.barh(sorted_methods, sorted_thresholds, height=0.6)
        
        # Add threshold values as text
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                     f'{sorted_thresholds[i]:.2f}', 
                     va='center', fontsize=10)
        
        plt.xlabel('Threshold Value', fontsize=12)
        plt.title('Comparison of Threshold Values Across Methods', fontsize=14)
        plt.grid(True, axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'threshold_comparison_chart.png'), dpi=300)
        plt.close()

def process_original_ctp(enhanced_volumes, volume_array, threshold_tracker):
    """Process the original CTP volume with basic enhancement techniques"""
    print("Applying Original CTP Processing approach...")
    
    # Save original volume
    enhanced_volumes['OG_volume_array'] = volume_array
    print(f"OG_volume_array shape: {enhanced_volumes['OG_volume_array'].shape}")
    
    # Threshold original volume
    threshold = 2836
    enhanced_volumes['DESCARGAR_OG_volume_array'] = np.uint8(enhanced_volumes['OG_volume_array'] > threshold)
    threshold_tracker['OG_volume_array'] = threshold
    
    # Gaussian filter on original volume
    enhanced_volumes['OG_gaussian_volume_og'] = gaussian(enhanced_volumes['OG_volume_array'], sigma=0.3)
    threshold = 2836
    enhanced_volumes['DESCARGAR_OG_gaussian_volume_og_1886'] = np.uint8(enhanced_volumes['OG_gaussian_volume_og'] > threshold)
    threshold_tracker['OG_gaussian_volume_og'] = threshold
    
    # Gamma correction on gaussian filtered volume
    enhanced_volumes['OG_gamma_volume_og'] = gamma_correction(enhanced_volumes['OG_gaussian_volume_og'], gamma=3)
    # Threshold gamma corrected volume
    threshold = 215
    enhanced_volumes['DESCARGAR_OG_gamma_volume_og_85'] = np.uint8(enhanced_volumes['OG_gamma_volume_og'] > threshold)
    threshold_tracker['OG_gamma_volume_og'] = threshold
    
    # Sharpen gamma corrected volume
    enhanced_volumes['OG_sharpened'] = sharpen_high_pass(enhanced_volumes['OG_gamma_volume_og'], strenght=0.8)
    # Threshold sharpened volume
    threshold = 215
    enhanced_volumes['DESCARGAR_OG_sharpened_180'] = np.uint8(enhanced_volumes['OG_sharpened'] > threshold)
    threshold_tracker['OG_sharpened'] = threshold
    
    return enhanced_volumes, threshold_tracker


def process_roi_gamma_mask(enhanced_volumes, final_roi, volume_array, threshold_tracker):
    """Process the volume using ROI and gamma mask"""
    print("Applying ROI with Gamma Mask approach...")
    
    # Apply ROI mask to gamma corrected volume
    if 'OG_gamma_volume_og' not in enhanced_volumes:
        # First apply gaussian filter
        gaussian_volume = gaussian(volume_array, sigma=0.3)
        # Then apply gamma correction
        gamma_volume = gamma_correction(gaussian_volume, gamma=3)
        enhanced_volumes['OG_gamma_volume_og'] = gamma_volume
    
    # Combine ROI mask with gamma corrected volume
    enhanced_volumes['PRUEBA_roi_plus_gamma_mask'] = (final_roi > 0) * enhanced_volumes['OG_gamma_volume_og']
    # Threshold ROI plus gamma mask
    threshold = 163
    enhanced_volumes['DESCARGAR_PRUEBA_roi_plus_gamma_mask_153'] = np.uint8(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'] > threshold)
    threshold_tracker['PRUEBA_roi_plus_gamma_mask'] = threshold
    
    # Apply CLAHE to ROI plus gamma mask
    enhanced_volumes['PRUEBA_roi_plus_gamma_mask_clahe'] = apply_clahe(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'])
    # Threshold CLAHE result
    threshold = 170
    enhanced_volumes['DESCARGAR_PRUEBA_THRESHOLD_CLAHE'] = np.uint8(enhanced_volumes['PRUEBA_roi_plus_gamma_mask_clahe'] > threshold)
    threshold_tracker['PRUEBA_roi_plus_gamma_mask_clahe'] = threshold
    

    # Apply wavelet non-local means denoising
    enhanced_volumes['PRUEBA_WAVELET_NL'] = wavelet_nlm_denoise(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'], wavelet='db1')
    # Threshold wavelet NL denoised volume
    threshold = 175
    enhanced_volumes['DESCARGAR_PRUEBA_WAVELET_NL'] = np.uint8(enhanced_volumes['PRUEBA_WAVELET_NL'] > threshold)
    threshold_tracker['PRUEBA_WAVELET_NL'] = threshold
    
    return enhanced_volumes, threshold_tracker


def process_roi_only(enhanced_volumes, roi_volume, final_roi, threshold_tracker):
    """Process using only the ROI volume"""
    print("Applying ROI Only approach...")
    
    # Save ROI volume
    enhanced_volumes['roi_volume'] = roi_volume
    
    # Apply wavelet denoising
    enhanced_volumes['wavelet_only_roi'] = wavelet_denoise(enhanced_volumes['roi_volume'], wavelet='db1')
    threshold= 2720
    enhanced_volumes['DESCARGAR_WAVELET_ROI'] = np.uint8(enhanced_volumes['wavelet_only_roi'] > threshold)
    threshold_tracker['wavelet_only_roi'] = threshold

    
    # Apply gamma correction to wavelet denoised volume
    enhanced_volumes['gamma_only_roi'] = gamma_correction(enhanced_volumes['wavelet_only_roi'], gamma=0.8)
    threshold = 225
    enhanced_volumes['DESCARGAR_GAMMA_ONLY_ROI_225'] = np.uint8(enhanced_volumes['gamma_only_roi'] > threshold)
    threshold_tracker['gamma_only_roi'] = threshold
    
    # Threshold ROI volume
    threshold = 2736
    enhanced_volumes['DESCARGAR_Threshold_roi_volume'] = np.uint8(enhanced_volumes['roi_volume'] > threshold)
    threshold_tracker['roi_volume'] = threshold
    
    return enhanced_volumes, threshold_tracker


def process_roi_plus_gamma_after(enhanced_volumes, final_roi, threshold_tracker):
    """Process using ROI plus gamma correction after"""
    print("Applying ROI plus Gamma after approach...")
    
    # Apply gaussian filter to ROI plus gamma mask
    if 'PRUEBA_roi_plus_gamma_mask' not in enhanced_volumes:
        # This should have been created in process_roi_gamma_mask
        # If not available, return without processing
        print("Warning: PRUEBA_roi_plus_gamma_mask not found. Skipping this approach.")
        return enhanced_volumes, threshold_tracker
    
    enhanced_volumes['2_gaussian_volume_roi'] = gaussian(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'], sigma=0.3)
    # Threshold gaussian volume
    threshold = 0.806
    enhanced_volumes['DESCARGAR_2_gaussian_volume_roi_0.806'] = np.uint8(enhanced_volumes['2_gaussian_volume_roi'] > threshold)
    threshold_tracker['2_gaussian_volume_roi'] = threshold
    
    # Apply gamma correction
    enhanced_volumes['2_gamma_correction'] = gamma_correction(enhanced_volumes['2_gaussian_volume_roi'], gamma=0.8)
    # Threshold gamma corrected volume
    threshold = 184
    enhanced_volumes['DESCARGAR_2_gamma_correction'] = np.uint8(enhanced_volumes['2_gamma_correction'] > threshold)
    threshold_tracker['2_gamma_correction'] = threshold
    
    # Apply top-hat transformation
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    tophat_2 = cv2.morphologyEx(enhanced_volumes['2_gaussian_volume_roi'], cv2.MORPH_TOPHAT, kernel_2)
    enhanced_volumes['2_tophat'] = cv2.addWeighted(enhanced_volumes['2_gaussian_volume_roi'], 1, tophat_2, 2, 0)
    # Threshold top-hat result
    threshold = 0.533
    enhanced_volumes['DESCARGAR_2_tophat'] = np.uint8(enhanced_volumes['2_tophat'] > threshold)
    threshold_tracker['2_tophat'] = threshold
    
    # Sharpen gamma corrected volume
    enhanced_volumes['2_sharpened'] = sharpen_high_pass(enhanced_volumes['2_gamma_correction'], strenght=0.8)
    # Threshold sharpened volume
    threshold = 210
    enhanced_volumes['DESCARGAR_2_sharpened'] = np.uint8(enhanced_volumes['2_sharpened'] > threshold)
    threshold_tracker['2_sharpened'] = threshold
    
    # Apply LOG transform
    enhanced_volumes['2_LOG'] = log_transform_slices(enhanced_volumes['2_tophat'], c=3)
    # Threshold LOG transform
    threshold = 230
    enhanced_volumes['DESCARGAR_2_LOG'] = np.uint8(enhanced_volumes['2_LOG'] > threshold)
    threshold_tracker['2_LOG'] = threshold
    
    # Apply wavelet denoising
    enhanced_volumes['2_wavelet_roi'] = wavelet_denoise(enhanced_volumes['2_LOG'], wavelet='db4')
    # Threshold wavelet result
    threshold = 231
    enhanced_volumes['DESCARGAR_2_wavelet_roi'] = np.uint8(enhanced_volumes['2_wavelet_roi'] > threshold)
    threshold_tracker['2_wavelet_roi'] = threshold
    
    # Apply erosion
    enhanced_volumes['2_erode'] = morphological_operation(enhanced_volumes['2_sharpened'], operation='erode', kernel_size=1)
    # Threshold eroded volume
    threshold = 190
    enhanced_volumes['DESCARGAR_2'] = np.uint8(enhanced_volumes['2_erode'] > threshold)
    threshold_tracker['2_erode'] = threshold
    
    # Apply gaussian filter
    enhanced_volumes['2_gaussian_2'] = gaussian(enhanced_volumes['2_erode'], sigma=0.2)
    threshold = 0.915
    enhanced_volumes['DESCARGAR_2_gaussian_2'] = np.uint8(enhanced_volumes['2_gaussian_2'] > threshold)
    threshold_tracker['2_gaussian_2'] = threshold
    
    # Sharpen gaussian filtered volume
    enhanced_volumes['2_sharpening_2_trial'] = sharpen_high_pass(enhanced_volumes['2_gaussian_2'], strenght=0.8)
    # Threshold sharpened volume
    threshold = 0.915
    enhanced_volumes['DESCARGAR_2_sharpening_2_trial_0.915'] = np.uint8(enhanced_volumes['2_sharpening_2_trial'] > threshold)
    threshold_tracker['2_sharpening_2_trial'] = threshold
    
    return enhanced_volumes, threshold_tracker


def process_wavelet_roi(enhanced_volumes, roi_volume, threshold_tracker):
    """Process using wavelet denoising on ROI volume"""
    print("Applying Wavelet ROI approach...")
    
    # Apply non-local means denoising with wavelet
    enhanced_volumes['NUEVO_NLMEANS'] = wavelet_nlm_denoise(roi_volume)
    # Threshold NL means denoised volume
    threshold = 2558
    enhanced_volumes['DESCARGAR_NUEVO_NLMEANS'] = np.uint8(enhanced_volumes['NUEVO_NLMEANS'] > threshold)
    threshold_tracker['NUEVO_NLMEANS'] = threshold
    return enhanced_volumes, threshold_tracker


def process_original_idea(enhanced_volumes, threshold_tracker):
    """Process using the original idea approach"""
    print("Applying Original Idea approach...")
    
    if 'PRUEBA_roi_plus_gamma_mask' not in enhanced_volumes:
        # This should have been created in process_roi_gamma_mask
        # If not available, return without processing
        print("Warning: PRUEBA_roi_plus_gamma_mask not found. Skipping this approach.")
        return enhanced_volumes, threshold_tracker
    
    # Apply gaussian filter
    enhanced_volumes['ORGINAL_IDEA_gaussian'] = gaussian(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'], sigma=0.3)
    threshold = 0.7
    enhanced_volumes['DESCARGAR_ORGINAL_IDEA_gaussian_0.7'] = np.uint8(enhanced_volumes['ORGINAL_IDEA_gaussian'] > threshold)
    threshold_tracker['ORGINAL_IDEA_gaussian'] = threshold
    
    # Apply gamma correction
    enhanced_volumes['ORGINAL_IDEA_gamma_correction'] = gamma_correction(enhanced_volumes['ORGINAL_IDEA_gaussian'], gamma=2)
    # Threshold gamma corrected volume
    threshold = 130
    enhanced_volumes['DESCARGAR_ORGINAL_IDEA_gamma_correction'] = np.uint8(enhanced_volumes['ORGINAL_IDEA_gamma_correction'] > threshold)
    threshold_tracker['ORGINAL_IDEA_gamma_correction'] = threshold
    
    # Sharpen gamma corrected volume
    enhanced_volumes['ORGINAL_IDEA_sharpened'] = sharpen_high_pass(enhanced_volumes['ORGINAL_IDEA_gamma_correction'], strenght=0.8)
    # Threshold sharpened volume
    threshold = 130
    enhanced_volumes['DESCARGAR_ORGINAL_IDEA_sharpened_130'] = np.uint8(enhanced_volumes['ORGINAL_IDEA_sharpened'] > threshold)
    threshold_tracker['ORGINAL_IDEA_sharpened'] = threshold
    

    # Apply wavelet denoising
    enhanced_volumes['ORIGINAL_IDEA_wavelet'] = wavelet_denoise(enhanced_volumes['ORGINAL_IDEA_sharpened'])
    # Threshold wavelet result
    threshold = 120
    enhanced_volumes['DESCARGAR_ORIGINAL_IDEA_wavelet_120'] = np.uint8(enhanced_volumes['ORIGINAL_IDEA_wavelet'] > threshold)
    threshold_tracker['ORIGINAL_IDEA_wavelet'] = threshold
    
    # Apply gaussian filter
    enhanced_volumes['ORGINAL_IDEA_gaussian_2'] = gaussian(enhanced_volumes['ORGINAL_IDEA_sharpened'], sigma=0.4)
    # Threshold gaussian filtered volume
    threshold = 0.4
    enhanced_volumes['DESCARGAR_ORGINAL_IDEA_gaussian_2_0.4'] = np.uint8(enhanced_volumes['ORGINAL_IDEA_gaussian_2'] > threshold)
    threshold_tracker['ORGINAL_IDEA_gaussian_2'] = threshold
    
    # Apply gamma correction
    enhanced_volumes['ORIGINAL_IDEA_GAMMA_2'] = gamma_correction(enhanced_volumes['ORGINAL_IDEA_gaussian_2'], gamma=1.4)
    # Threshold gamma corrected volume
    threshold = 75
    enhanced_volumes['DESCARGAR_ORIGINAL_IDEA_GAMMA_2_75'] = np.uint8(enhanced_volumes['ORIGINAL_IDEA_GAMMA_2'] > threshold)
    threshold_tracker['ORIGINAL_IDEA_GAMMA_2'] = threshold
    
    # Apply top-hat transformation
    kernel_size_og = (1, 1)
    kernel_og = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size_og)
    tophat_og = cv2.morphologyEx(enhanced_volumes['ORGINAL_IDEA_gaussian_2'], cv2.MORPH_TOPHAT, kernel_og)
    enhanced_volumes['OG_tophat_1'] = cv2.addWeighted(enhanced_volumes['ORGINAL_IDEA_gaussian_2'], 1, tophat_og, 2, 0)
    # Threshold top-hat result
    threshold = 0.518
    enhanced_volumes['DESCARGAR_OG_tophat_1'] = np.uint8(enhanced_volumes['OG_tophat_1'] > threshold)
    threshold_tracker['OG_tophat_1'] = threshold
    
    return enhanced_volumes, threshold_tracker


def process_first_try(enhanced_volumes, roi_volume, threshold_tracker):
    """Process using the first try approach"""
    print("Applying First Try approach...")
    
    # Apply gaussian filter
    enhanced_volumes['FT_gaussian'] = gaussian(roi_volume, sigma=0.3)
    threshold = 2527
    enhanced_volumes['DESCARGAR_FT_gaussian'] = np.uint8(enhanced_volumes['FT_gaussian'] > threshold)
    threshold_tracker['FT_gaussian'] = threshold
    
    # Apply top-hat transformation
    kernel_size = (1, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    tophat_ft = cv2.morphologyEx(roi_volume, cv2.MORPH_TOPHAT, kernel)
    enhanced_volumes['FT_tophat_1'] = cv2.addWeighted(roi_volume, 1, tophat_ft, 2, 0)
    # Subtract gaussian from top-hat
    enhanced_volumes['FT_RESTA_TOPHAT_GAUSSIAN'] = enhanced_volumes['FT_tophat_1'] - gaussian(roi_volume, sigma=0.8)
    # Threshold subtracted volume
    threshold = 1060
    enhanced_volumes['DESCARGAR_FT_RESTA_TOPHAT_GAUSSIAN_1060'] = np.uint(enhanced_volumes['FT_RESTA_TOPHAT_GAUSSIAN'] > threshold)
    threshold_tracker['FT_RESTA_TOPHAT_GAUSSIAN'] = threshold
    
    # Apply gamma correction
    enhanced_volumes['FT_gamma_correction'] = gamma_correction(enhanced_volumes['FT_gaussian'], gamma=5)
    threshold = 120
    enhanced_volumes['DESCARGAR_FT_gamma_correction'] = np.uint8(enhanced_volumes['FT_gamma_correction'] > threshold)
    threshold_tracker['FT_gamma_correction'] = threshold

    
    # Sharpen gamma corrected volume
    enhanced_volumes['FT_sharpened'] = sharpen_high_pass(enhanced_volumes['FT_gamma_correction'], strenght=0.4)
    threshold= 118
    enhanced_volumes['DESCARGAR_FT_sharpened'] = np.uint8(enhanced_volumes['FT_sharpened'] > threshold)
    threshold_tracker['FT_sharpened'] = threshold

    
    # Apply gaussian filter
    enhanced_volumes['FT_gaussian_2'] = gaussian(enhanced_volumes['FT_sharpened'], sigma=0.4)
    # Threshold gaussian filtered volume
    threshold = 0.486
    enhanced_volumes['DESCARGAR_FT_gaussian_2_0.486'] = np.uint8(enhanced_volumes['FT_gaussian_2'] > threshold)
    threshold_tracker['FT_gaussian_2'] = threshold
    
    # Apply gamma correction
    enhanced_volumes['FT_gamma_2'] = gamma_correction(enhanced_volumes['FT_gaussian_2'], gamma=1.2)
    # Threshold gamma corrected volume
    threshold = 121
    enhanced_volumes['DESCARGAR_FT_GAMMA_2'] = np.uint8(enhanced_volumes['FT_gamma_2'] > threshold)
    threshold_tracker['FT_gamma_2'] = threshold

    # Apply opening operation
    enhanced_volumes['FT_opening'] = morph_operations(enhanced_volumes['FT_gamma_2'], iterations=2, kernel_shape='cross')
    threshold = 100
    enhanced_volumes['DESCARGAR_FT_OPENING'] = np.uint8(enhanced_volumes['FT_opening'] > threshold)
    threshold_tracker['FT_opening'] = threshold

    
    # Apply closing operation
    enhanced_volumes['FT_closing'] = morph_operations(enhanced_volumes['FT_opening'], operation='close')
    threshold = 100
    enhanced_volumes['DESCARGAR_FT_CLOSING'] = np.uint8(enhanced_volumes['FT_closing'] > threshold)
    threshold_tracker['FT_closing'] = threshold

    # Apply erosion
    enhanced_volumes['FT_erode_2'] = morphological_operation(enhanced_volumes['FT_closing'], operation='erode', kernel_size=1)
    # Threshold eroded volume
    threshold = 120
    enhanced_volumes['DESCARGAR_FT_ERODE_2'] = np.uint8(enhanced_volumes['FT_erode_2'] > threshold)
    threshold_tracker['FT_erode_2'] = threshold
    
    # Apply top-hat transformation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    tophat = cv2.morphologyEx(enhanced_volumes['FT_gaussian_2'], cv2.MORPH_TOPHAT, kernel)
    enhanced_volumes['FT_tophat'] = cv2.addWeighted(enhanced_volumes['FT_gaussian_2'], 1, tophat, 2, 0)
    # Threshold top-hat result
    threshold = 0.484
    enhanced_volumes['DESCARGAR_FT_TOPHAT'] = np.uint8(enhanced_volumes['FT_tophat'] > threshold)
    threshold_tracker['FT_tophat'] = threshold
    
    # Apply gaussian filter
    enhanced_volumes['FT_gaussian_3'] = gaussian(enhanced_volumes['FT_tophat'], sigma=0.1)
    # Threshold gaussian filtered volume
    threshold = 0.438
    enhanced_volumes['DESCARGAR_FT_gaussian_3_0.438'] = np.uint8(enhanced_volumes['FT_gaussian_3'] > threshold)
    threshold_tracker['FT_gaussian_3'] = threshold
    return enhanced_volumes, threshold_tracker


def enhance_ctp(inputVolume, inputROI=None, methods=None, outputDir=None, collect_histograms=True, train_model=True):
    """
    Enhance CT perfusion images using different image processing approaches.
    
    Parameters:
    -----------
    inputVolume : vtkMRMLScalarVolumeNode
        Input CT perfusion volume
    inputROI : vtkMRMLScalarVolumeNode, optional
        Region of interest mask
    methods : str or list, optional
        Methods to apply, can be 'all' or a list of method names
    outputDir : str, optional
        Directory to save output volumes
    collect_histograms : bool, optional
        Whether to collect histogram data
    train_model : bool, optional
        Whether to train a threshold prediction model
        
    Returns:
    --------
    dict
        Dictionary of enhanced volume nodes
    """
    # Initialize threshold tracker
    threshold_tracker = {}
    
    # Default to 'all' methods
    if methods is None:
        methods = 'all'
    
    # Convert input volume to numpy array
    volume_array = slicer.util.arrayFromVolume(inputVolume)
    if volume_array is None or volume_array.size == 0:
        print("Input volume data is empty or invalid.")
        return None

    # Process ROI if provided
    if inputROI is not None:
        roi_array = slicer.util.arrayFromVolume(inputROI)
        roi_array = np.uint8(roi_array > 0)  # Ensure binary mask (0 or 1)
        print(f"Shape of input volume: {volume_array.shape}")
        print(f"Shape of ROI mask: {roi_array.shape}")
        
        # Process ROI
        print("Filling inside the ROI...")
        filled_roi = ndimage.binary_fill_holes(roi_array)
        print("Applying morphological closing...")
        struct_elem = morphology.ball(10)
        closed_roi = morphology.binary_closing(filled_roi, struct_elem)
        
        if closed_roi.shape != volume_array.shape:
            print("🔄 Shapes don't match. Using spacing/origin-aware resampling...")
            final_roi = closed_roi
        else:
            final_roi = closed_roi
            print("No resizing needed: ROI already has the same shape as volume.")
    else:
        print("No ROI provided. Proceeding without ROI mask.")
        final_roi = np.ones_like(volume_array)
    
    # Apply the ROI mask to the volume
    print(f'Volume shape: {volume_array.shape}, ROI shape: {final_roi.shape}')
    print(f'Volume dtype: {volume_array.dtype}, ROI dtype: {final_roi.dtype}')
    roi_volume = np.multiply(volume_array, final_roi)
    final_roi = final_roi.astype(np.uint8)

    # Initialize results dictionary
    enhanced_volumes = {}
    
    if methods == 'all' or 'original' in methods:
        # ================ APPROACH 1: ORIGINAL CTP PROCESSING ================
        enhanced_volumes, threshold_tracker = process_original_ctp(
            enhanced_volumes, volume_array, threshold_tracker)
    
    if methods == 'all' or 'roi_gamma' in methods:
        # ================ APPROACH 2: ROI WITH GAMMA MASK ================
        enhanced_volumes, threshold_tracker = process_roi_gamma_mask(
            enhanced_volumes, final_roi, volume_array, threshold_tracker)
    
    if methods == 'all' or 'roi_only' in methods:
        # ================ APPROACH 3: ROI ONLY PROCESSING ================
        enhanced_volumes, threshold_tracker = process_roi_only(
            enhanced_volumes, roi_volume, final_roi, threshold_tracker)
        
    if methods == 'all' or 'roi_plus_gamma' in methods:
        # ================ APPROACH 4: ROI PLUS GAMMA AFTER ================
        enhanced_volumes, threshold_tracker = process_roi_plus_gamma_after(
            enhanced_volumes, final_roi, threshold_tracker)
        
    if methods == 'all' or 'wavelet_roi' in methods:
        # ================ APPROACH 5: WAVELET ON ROI ================
        enhanced_volumes, threshold_tracker = process_wavelet_roi(
            enhanced_volumes, roi_volume, threshold_tracker)
        
    if methods == 'all' or 'original_idea' in methods:
        # ================ APPROACH 6: ORIGINAL IDEA ================
        enhanced_volumes, threshold_tracker = process_original_idea(
            enhanced_volumes, threshold_tracker)
        
    if methods == 'all' or 'first_try' in methods:
        # ================ APPROACH 7: FIRST TRY ================
        enhanced_volumes, threshold_tracker = process_first_try(
            enhanced_volumes, roi_volume, threshold_tracker)
    
    # Save thresholds to a file
    if outputDir is None:
        outputDir = slicer.app.temporaryPath()  
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        
    # Save threshold values to a text file
    threshold_file = os.path.join(outputDir, f"thresholds_{inputVolume.GetName()}.txt")
    with open(threshold_file, 'w') as f:
        f.write(f"Thresholds for {inputVolume.GetName()}\n")
        f.write("=" * 50 + "\n\n")
        
        for method, threshold in threshold_tracker.items():
            f.write(f"{method}: {threshold}\n")
    
    print(f"Saved thresholds to: {threshold_file}")
    
    # Collect histogram data if requested
    if collect_histograms:
        histogram_data = collect_histogram_data(enhanced_volumes, threshold_tracker, outputDir)
        print(f"Saved histogram data to: {os.path.join(outputDir, 'histograms')}")
        
        # Train threshold prediction model if requested
        if train_model:
            model, scaler = train_threshold_model(outputDir)
            if model is not None:
                print(f"Saved threshold prediction model to: {os.path.join(outputDir, 'model')}")
    
    # Process each enhanced volume
    enhancedVolumeNodes = {}
    for method_name, enhanced_image in enhanced_volumes.items():
        enhancedVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        enhancedVolumeNode.SetName(f"Enhanced_{method_name}_{inputVolume.GetName()}")
        enhancedVolumeNode.SetOrigin(inputVolume.GetOrigin())
        enhancedVolumeNode.SetSpacing(inputVolume.GetSpacing())
        ijkToRasMatrix = vtk.vtkMatrix4x4()
        inputVolume.GetIJKToRASMatrix(ijkToRasMatrix)  
        enhancedVolumeNode.SetIJKToRASMatrix(ijkToRasMatrix) 
        slicer.util.updateVolumeFromArray(enhancedVolumeNode, enhanced_image)
        enhancedVolumeNodes[method_name] = enhancedVolumeNode
        
        output_file = os.path.join(outputDir, f"Filtered_{method_name}_{inputVolume.GetName()}.nrrd")
        slicer.util.saveNode(enhancedVolumeNode, output_file)
        print(f"Saved {method_name} enhancement as: {output_file}")
        
    return enhancedVolumeNodes


# Main execution
inputVolume = slicer.util.getNode('7_CTp.3D')  
inputROI = slicer.util.getNode('patient7_resampled_sy_mask')  # Brain Mask 
# Output directory
outputDir = r"C:\Users\rocia\Downloads\TFG\Cohort\Enhance_ctp_tests\P7\TH45_histograms_FILTERED"
# Run the enhancement function
enhancedVolumeNodes = enhance_ctp(inputVolume, inputROI, methods='all', outputDir=outputDir, collect_histograms=True, train_model=False)
# Print the enhanced volume nodes
for method, volume_node in enhancedVolumeNodes.items():
    if volume_node is not None:
        print(f"Enhanced volume for method '{method}': {volume_node.GetName()}")
    else:
        print(f"Enhanced volume for method '{method}': No volume node available.")

#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Threshold_mask\P7\new_enhance_P7.py').read())

