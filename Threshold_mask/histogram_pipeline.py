
def shannon_entropy(image):
    """Calculate Shannon entropy of an image."""
    import numpy as np
    # Convert to probabilities by calculating histogram
    hist, _ = np.histogram(image, bins=256, density=True)
    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]
    # Calculate entropy
    return -np.sum(hist * np.log2(hist))

def collect_histogram_data(enhanced_volumes, threshold_tracker, outputDir=None):
    """
    Collect histogram data for each enhanced volume and save as CSV.
    
    Parameters:
    -----------
    enhanced_volumes : dict
        Dictionary of enhanced volume arrays
    threshold_tracker : dict
        Dictionary of thresholds used for each method
    outputDir : str, optional
        Directory to save histogram data
        
    Returns:
    --------
    dict
        Dictionary of histogram data for each method
    """
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
        
        # Extract features from histogram
        hist_features[method_name] = {
            'min': np.min(volume_array),
            'max': np.max(volume_array),
            'mean': np.mean(volume_array),
            'median': np.median(volume_array),
            'std': np.std(volume_array),
            'p25': np.percentile(volume_array, 25),
            'p75': np.percentile(volume_array, 75),
            'p95': np.percentile(volume_array, 95),
            'p99': np.percentile(volume_array, 99),
            'entropy': shannon_entropy(volume_array),
        }
        
        # Add threshold if available
        if method_name in threshold_tracker:
            hist_features[method_name]['threshold'] = threshold_tracker[method_name]
        
        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, hist)
        plt.title(f'Histogram for {method_name}')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        
        # Add a vertical line for threshold if available
        if method_name in threshold_tracker:
            threshold = threshold_tracker[method_name]
            plt.axvline(x=threshold, color='r', linestyle='--', 
                        label=f'Threshold = {threshold}')
            plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(hist_dir, f'histogram_{method_name}.png'))
        plt.close()
    
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
                ax.set_ylim(bottom=0.001)  # Minimum value for log scale
                
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
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Add space between subplots
        plt.savefig(os.path.join(plots_dir, f'histograms_{approach_name.replace(" ", "_")}.png'), dpi=300)
        plt.close()
    
    # Enhanced plot for comparing all methods with thresholds
    threshold_methods = [m for m in threshold_tracker.keys() if m in histogram_data]
    
    if threshold_methods:
        # Create two comparison plots: one regular and one with log scale
        for scale_type in ['linear', 'log']:
            # Determine appropriate figure size based on number of methods
            fig_width = min(20, 12 + len(threshold_methods) * 0.5)
            
            plt.figure(figsize=(fig_width, 10))
            
            # Use colormap for better differentiation between methods
            norm = Normalize(vmin=0, vmax=len(threshold_methods)-1)
            
            # Plot each method with distinct color
            for i, method in enumerate(threshold_methods):
                data = histogram_data[method]
                # Normalize histogram
                normalized_hist = data['hist'] / np.max(data['hist'])
                color = colormap(norm(i))
                
                plt.plot(data['bin_centers'], normalized_hist, 
                         label=method, color=color, linewidth=2)
                
                # Add threshold line with matching color
                threshold = threshold_tracker[method]
                plt.axvline(x=threshold, linestyle='--', color=color, alpha=0.7)
                
                # Annotate threshold value
                plt.text(threshold, 0.5 + i*0.05, f'{method}: {threshold:.2f}', 
                         rotation=90, fontsize=8, color=color)
            
            if scale_type == 'log':
                plt.yscale('log')
                plt.ylim(bottom=0.001)
                plt.title('Normalized Histograms with Thresholds (Log Scale)')
            else:
                plt.title('Normalized Histograms with Thresholds')
                
            plt.xlabel('Pixel Value', fontsize=12)
            plt.ylabel('Normalized Frequency', fontsize=12)
            plt.grid(True, alpha=0.3, linestyle='--')
            
            # Create legend with two columns if many methods
            if len(threshold_methods) > 6:
                ncol = 2
            else:
                ncol = 1
                
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=ncol, fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'all_thresholds_comparison_{scale_type}.png'), dpi=300)
            plt.close()
            
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