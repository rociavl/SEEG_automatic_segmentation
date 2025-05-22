import os
import numpy as np
import vtk
from vtk.util import numpy_support
import time
import joblib
import logging
import slicer
from skimage.filters import gaussian
from scipy import ndimage
from skimage import morphology

# Import necessary functions 
from Threshold_mask.enhance_ctp import (
    gamma_correction, 
    sharpen_high_pass, 
    log_transform_slices, 
    wavelet_denoise, 
    wavelet_nlm_denoise, 
    morphological_operation, 
    apply_clahe, 
    morph_operations
)

class CTPEnhancer:
    """
    Class for enhancing CT perfusion images and extracting features.
    """
    
    def __init__(self):
        """Initialize the CTP enhancer."""
        self.threshold_tracker = {}
        self.enhanced_volumes = {}
    
    def shannon_entropy(self, image):
        """Calculate Shannon entropy of an image."""
        # Convert to probabilities by calculating histogram
        hist, _ = np.histogram(image, bins=256, density=True)
        # Remove zeros to avoid log(0)
        hist = hist[hist > 0]
        # Calculate entropy
        return -np.sum(hist * np.log2(hist))
    
    def extract_advanced_features(self, volume_array, hist=None, bin_centers=None):
        """Extract advanced features from a volume array."""
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
        
        # Entropy
        features['entropy'] = self.shannon_entropy(volume_array)
        
        # Additional engineered features for model prediction
        features['range'] = features['max'] - features['min']
        features['iqr'] = features['p75'] - features['p25']
        features['iqr_to_std_ratio'] = features['iqr'] / (features['std'] + 1e-5)
        features['contrast_per_iqr'] = features['contrast_ratio'] / (features['iqr'] + 1e-5)
        features['range_to_iqr'] = features['range'] / (features['iqr'] + 1e-5)
        features['skewness_squared'] = features['skewness'] ** 2
        features['kurtosis_log'] = np.log1p(features['kurtosis'] - np.min(features['kurtosis']))
        
        return features
    
    def predict_threshold_for_volume(self, volume_array, model_path):
        """Predict threshold for a given volume array using the trained model, ensuring it's within min/max range."""
        # Extract features
        features = self.extract_advanced_features(volume_array)
        
        # Get min and max from features
        vol_min = features['min']
        vol_max = features['max']
        
        # Load the model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        model = model_data['model']
        feature_names = model_data.get('feature_names', [])
        
        # Convert features to DataFrame with correct feature order
        import pandas as pd
        feature_df = pd.DataFrame([features])
        
        # Ensure expected features
        for feat in feature_names:
            if feat not in feature_df.columns:
                feature_df[feat] = 0  # Add missing features with default value
        
        # Reorder columns to match training data
        feature_df = feature_df[feature_names]
        
        # Predict threshold
        threshold = model.predict(feature_df)[0]
        
        # Ensure threshold is within volume's min/max range
        if threshold < vol_min or threshold > vol_max:
            logging.warning(f"Predicted threshold {threshold} outside volume range [{vol_min}, {vol_max}]. Using 99.97th percentile instead.")
            threshold = np.percentile(volume_array, 99.97)
        
        return threshold
    
    def process_roi_gamma_mask(self, final_roi, volume_array, model_path=None):
        """Process the volume using ROI and gamma mask."""
        logging.info("Applying ROI with Gamma Mask approach...")
        
        # Apply gaussian filter
        gaussian_volume = gaussian(volume_array, sigma=0.3)
        # Then apply gamma correction
        gamma_volume = gamma_correction(gaussian_volume, gamma=3)
        self.enhanced_volumes['OG_gamma_volume_og'] = gamma_volume
        
        # Combine ROI mask with gamma corrected volume
        self.enhanced_volumes['PRUEBA_roi_plus_gamma_mask'] = (final_roi > 0) * self.enhanced_volumes['OG_gamma_volume_og']
        # Predict threshold for ROI plus gamma mask
        if model_path:
            threshold = self.predict_threshold_for_volume(self.enhanced_volumes['PRUEBA_roi_plus_gamma_mask'], model_path)
        else:
            threshold = 43  # Fallback to fixed threshold if no model
        self.enhanced_volumes['DESCARGAR_PRUEBA_roi_plus_gamma_mask_40'] = np.uint8(self.enhanced_volumes['PRUEBA_roi_plus_gamma_mask'] > threshold)
        self.threshold_tracker['PRUEBA_roi_plus_gamma_mask'] = threshold
        
        # Apply CLAHE to ROI plus gamma mask
        self.enhanced_volumes['PRUEBA_roi_plus_gamma_mask_clahe'] = apply_clahe(self.enhanced_volumes['PRUEBA_roi_plus_gamma_mask'])
        # Predict threshold for CLAHE result
        if model_path:
            threshold = self.predict_threshold_for_volume(self.enhanced_volumes['PRUEBA_roi_plus_gamma_mask_clahe'], model_path)
        else:
            threshold = 57  # Fallback to fixed threshold if no model
        self.enhanced_volumes['DESCARGAR_PRUEBA_THRESHOLD_CLAHE_57'] = np.uint8(self.enhanced_volumes['PRUEBA_roi_plus_gamma_mask_clahe'] > threshold)
        self.threshold_tracker['PRUEBA_roi_plus_gamma_mask_clahe'] = threshold
        
        # Apply wavelet non-local means denoising
        self.enhanced_volumes['PRUEBA_WAVELET_NL'] = wavelet_nlm_denoise(self.enhanced_volumes['PRUEBA_roi_plus_gamma_mask'], wavelet='db1')
        # Predict threshold for wavelet NL denoised volume
        if model_path:
            threshold = self.predict_threshold_for_volume(self.enhanced_volumes['PRUEBA_WAVELET_NL'], model_path)
        else:
            threshold = 40.4550  # Fallback to fixed threshold if no model
        self.enhanced_volumes['DESCARGAR_PRUEBA_WAVELET_NL_40.4550'] = np.uint8(self.enhanced_volumes['PRUEBA_WAVELET_NL'] > threshold)
        self.threshold_tracker['PRUEBA_WAVELET_NL'] = threshold
        
        return self.enhanced_volumes, self.threshold_tracker
    
    def process_roi_only(self, roi_volume, final_roi, model_path=None):
        """Process using only the ROI volume."""
        logging.info("Applying ROI Only approach...")
        
        # Save ROI volume - This is the non-binary volume we want to keep
        self.enhanced_volumes['roi_volume_features'] = roi_volume
        
        # Apply wavelet denoising
        self.enhanced_volumes['wavelet_only_roi'] = wavelet_denoise(roi_volume, wavelet='db1')
        # Predict threshold for wavelet denoised volume
        if model_path:
            threshold = self.predict_threshold_for_volume(self.enhanced_volumes['wavelet_only_roi'], model_path)
        else:
            threshold = 1000  # Fallback to fixed threshold if no model
        self.enhanced_volumes['DESCARGAR_WAVELET_ROI_1000'] = np.uint8(self.enhanced_volumes['wavelet_only_roi'] > threshold)
        self.threshold_tracker['wavelet_only_roi'] = threshold
        
        # Apply gamma correction to wavelet denoised volume
        self.enhanced_volumes['gamma_only_roi'] = gamma_correction(self.enhanced_volumes['wavelet_only_roi'], gamma=0.8)
        # Predict threshold for gamma corrected volume
        if model_path:
            threshold = self.predict_threshold_for_volume(self.enhanced_volumes['gamma_only_roi'], model_path)
        else:
            threshold = 160  # Fallback to fixed threshold if no model
        self.enhanced_volumes['DESCARGAR_GAMMA_ONLY_ROI_160'] = np.uint8(self.enhanced_volumes['gamma_only_roi'] > threshold)
        self.threshold_tracker['gamma_only_roi'] = threshold
        
        # Predict threshold for ROI volume
        if model_path:
            threshold = self.predict_threshold_for_volume(self.enhanced_volumes['roi_volume_features'], model_path)
        else:
            threshold = 980  # Fallback to fixed threshold if no model
        self.enhanced_volumes['DESCARGAR_Threshold_roi_volume_980'] = np.uint8(self.enhanced_volumes['roi_volume_features'] > threshold)
        self.threshold_tracker['roi_volume_features'] = threshold
        
        return self.enhanced_volumes, self.threshold_tracker
        
    def enhance_ctp(self, inputVolume, inputROI=None, outputDir=None, model_path=None):
        """
        Enhance CT perfusion images using different image processing approaches.
        Focused on creating DESCARGAR_ volumes and keeping roi_volume_features.
        
        Parameters:
        -----------
        inputVolume : vtkMRMLScalarVolumeNode
            Input CT perfusion volume
        inputROI : vtkMRMLScalarVolumeNode, optional
            Region of interest mask
        outputDir : str, optional
            Directory to save output volumes
        model_path : str, optional
            Path to trained model for threshold prediction
            
        Returns:
        --------
        dict
            Dictionary of enhanced volume nodes
        """
        # Initialize trackers
        self.threshold_tracker = {}
        self.enhanced_volumes = {}
        
        # Start timer
        start_time = time.time()
        
        # Convert input volume to numpy array
        volume_array = slicer.util.arrayFromVolume(inputVolume)
        if volume_array is None or volume_array.size == 0:
            logging.error("Input volume data is empty or invalid.")
            return None

        # Process ROI if provided
        if inputROI is not None:
            roi_array = slicer.util.arrayFromVolume(inputROI)
            roi_array = np.uint8(roi_array > 0)  # Ensure binary mask (0 or 1)
            logging.info(f"Shape of input volume: {volume_array.shape}")
            logging.info(f"Shape of ROI mask: {roi_array.shape}")
            
            # Process ROI
            logging.info("Filling inside the ROI...")
            filled_roi = ndimage.binary_fill_holes(roi_array)
            logging.info("Applying morphological closing...")
            struct_elem = morphology.ball(10)
            closed_roi = morphology.binary_closing(filled_roi, struct_elem)
            
            final_roi = closed_roi
        else:
            logging.info("No ROI provided. Proceeding without ROI mask.")
            final_roi = np.ones_like(volume_array)
        
        # Apply the ROI mask to the volume
        logging.info(f'Volume shape: {volume_array.shape}, ROI shape: {final_roi.shape}')
        roi_volume = np.multiply(volume_array, final_roi)
        final_roi = final_roi.astype(np.uint8)

        # Process only the two approaches we want: ROI with gamma mask and ROI only
        self.enhanced_volumes, self.threshold_tracker = self.process_roi_gamma_mask(
            final_roi, volume_array, model_path)
        
        self.enhanced_volumes, self.threshold_tracker = self.process_roi_only(
            roi_volume, final_roi, model_path)
        
        # Filter to keep only DESCARGAR_ volumes and roi_volume_features
        filtered_volumes = {}
        for key, value in self.enhanced_volumes.items():
            if key.startswith('DESCARGAR_') or key == 'roi_volume_features':
                filtered_volumes[key] = value
        
        self.enhanced_volumes = filtered_volumes
        
        # Create output directory if needed
        if outputDir is None:
            outputDir = slicer.app.temporaryPath()  
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        
        # Save threshold values to a text file
        threshold_file = os.path.join(outputDir, f"thresholds_{inputVolume.GetName()}.txt")
        with open(threshold_file, 'w') as f:
            f.write(f"Thresholds for {inputVolume.GetName()}\n")
            f.write("=" * 50 + "\n\n")
            
            for method, threshold in self.threshold_tracker.items():
                f.write(f"{method}: {threshold}\n")
        
        logging.info(f"Saved thresholds to: {threshold_file}")

        # Process each enhanced volume
        enhancedVolumeNodes = {}
        for method_name, enhanced_image in self.enhanced_volumes.items():
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
            logging.info(f"Saved {method_name} enhancement as: {output_file}")
        
        # Log execution time
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        logging.info(f"Enhancement process completed in {hours}h {minutes}m {seconds:.2f}s")
        
        return enhancedVolumeNodes