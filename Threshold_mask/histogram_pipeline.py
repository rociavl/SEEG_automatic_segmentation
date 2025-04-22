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
            'entropy': exposure.shannon_entropy(volume_array),
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
    create_histogram_report(histogram_data, threshold_tracker, outputDir)
    
    return histogram_data

def create_histogram_report(histogram_data, threshold_tracker, outputDir):
    """
    Create a comprehensive report with all histograms.
    
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
    
    # Plot histograms by approach
    for approach_name, methods in approaches.items():
        # Filter available methods
        available_methods = [m for m in methods if m in histogram_data]
        
        if not available_methods:
            continue
            
        # Create plot with subplots for this approach
        num_methods = len(available_methods)
        nrows = (num_methods + 1) // 2  # Round up to nearest integer
        
        fig, axes = plt.subplots(nrows, 2, figsize=(16, 4 * nrows))
        fig.suptitle(f'Histograms for {approach_name} Approach', fontsize=16)
        
        # Flatten axes for easy indexing
        if nrows == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each method
        for i, method in enumerate(available_methods):
            if i < len(axes):
                data = histogram_data[method]
                axes[i].plot(data['bin_centers'], data['hist'])
                axes[i].set_title(method)
                axes[i].set_xlabel('Pixel Value')
                axes[i].set_ylabel('Frequency')
                
                # Add threshold line if available
                if method in threshold_tracker:
                    threshold = threshold_tracker[method]
                    axes[i].axvline(x=threshold, color='r', linestyle='--', 
                                  label=f'Threshold = {threshold}')
                    axes[i].legend()
        
        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        plt.savefig(os.path.join(plots_dir, f'histograms_{approach_name.replace(" ", "_")}.png'))
        plt.close()
    
    # Plot all thresholds in one figure
    threshold_methods = [m for m in threshold_tracker.keys() if m in histogram_data]
    
    if threshold_methods:
        # Create a large figure for all thresholds
        plt.figure(figsize=(15, 10))
        
        for method in threshold_methods:
            data = histogram_data[method]
            plt.plot(data['bin_centers'], data['hist'] / np.max(data['hist']), label=method)  # Normalize for comparison
            
            # Add threshold line
            threshold = threshold_tracker[method]
            plt.axvline(x=threshold, linestyle='--', color='gray', alpha=0.5)
        
        plt.title('Normalized Histograms with Thresholds')
        plt.xlabel('Pixel Value')
        plt.ylabel('Normalized Frequency')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'all_thresholds_comparison.png'))
        plt.close()

def train_threshold_model(outputDir):
    """
    Train a lightweight model to predict thresholds based on histogram features.
    
    Parameters:
    -----------
    outputDir : str
        Directory where histogram_features.csv is stored
        
    Returns:
    --------
    model
        Trained model for threshold prediction
    """
    import pandas as pd
    import numpy as np
    import os
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    # Load histogram features
    features_file = os.path.join(outputDir, 'histogram_features.csv')
    if not os.path.exists(features_file):
        print(f"Error: Features file not found at {features_file}")
        return None
        
    df = pd.read_csv(features_file, index_col=0)
    
    # Filter only rows with threshold values
    df = df.dropna(subset=['threshold'])
    
    if len(df) == 0:
        print("Error: No threshold data available for training")
        return None
    
    # Prepare features and target
    X = df.drop(columns=['threshold'])
    y = df['threshold']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"Model R² score on training data: {train_score:.4f}")
    print(f"Model R² score on test data: {test_score:.4f}")
    
    # Save model and scaler
    model_dir = os.path.join(outputDir, "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    joblib.dump(model, os.path.join(model_dir, "threshold_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    feature_importance.to_csv(os.path.join(model_dir, "feature_importance.csv"), index=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance for Threshold Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "feature_importance.png"))
    plt.close()
    
    return model, scaler