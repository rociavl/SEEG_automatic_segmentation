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
from Threshold_mask.enhance_ctp import gamma_correction, sharpen_high_pass, log_transform_slices, wavelet_denoise, wavelet_nlm_denoise, morphological_operation, apply_clahe, morph_operations  

def shannon_entropy(image):
    """Calculate Shannon entropy of an image."""
    import numpy as np
    # Convert to probabilities by calculating histogram
    hist, _ = np.histogram(image, bins=256, density=True)
    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]
    # Calculate entropy
    return -np.sum(hist * np.log2(hist))

def extract_advanced_features(volume_array):
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
    features['p99.5'] = np.percentile(volume_array, 99.5)
    features['p99.9'] = np.percentile(volume_array, 99.9)
    features['p99.98'] = np.percentile(volume_array, 99.98)  # Our key threshold
    features['skewness'] = stats.skew(volume_array.flatten())
    features['kurtosis'] = stats.kurtosis(volume_array.flatten())
    return features

def collect_histogram_data(enhanced_volumes, outputDir=None):
    import numpy as np
    import pandas as pd
    import os
    
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
            
        # Extract features (simplified)
        hist_features[method_name] = extract_advanced_features(volume_array)
    
    # Save all features to CSV
    features_df = pd.DataFrame.from_dict(hist_features, orient='index')
    features_df.to_csv(os.path.join(outputDir, 'histogram_features.csv'))
    
    return histogram_data

def process_original_ctp(enhanced_volumes, volume_array):
    """Process the original CTP volume with basic enhancement techniques"""
    print("Applying Original CTP Processing approach...")
    
    # Save original volume
    enhanced_volumes['OG_volume_array'] = volume_array
    
    # Threshold original volume using 99.98th percentile
    threshold = np.percentile(volume_array, 99.98)
    enhanced_volumes['DESCARGAR_OG_volume_array'] = np.uint8(volume_array > threshold)
    
    # Gaussian filter on original volume
    enhanced_volumes['OG_gaussian_volume_og'] = gaussian(volume_array, sigma=0.3)
    threshold = np.percentile(enhanced_volumes['OG_gaussian_volume_og'], 99.98)
    enhanced_volumes['DESCARGAR_OG_gaussian_volume_og'] = np.uint8(enhanced_volumes['OG_gaussian_volume_og'] > threshold)
    
    # Gamma correction on gaussian filtered volume
    enhanced_volumes['OG_gamma_volume_og'] = gamma_correction(enhanced_volumes['OG_gaussian_volume_og'], gamma=3)
    threshold = np.percentile(enhanced_volumes['OG_gamma_volume_og'], 99.98)
    enhanced_volumes['DESCARGAR_OG_gamma_volume_og'] = np.uint8(enhanced_volumes['OG_gamma_volume_og'] > threshold)
    
    # Sharpen gamma corrected volume
    enhanced_volumes['OG_sharpened'] = sharpen_high_pass(enhanced_volumes['OG_gamma_volume_og'], strenght=0.8)
    threshold = np.percentile(enhanced_volumes['OG_sharpened'], 99.98)
    enhanced_volumes['DESCARGAR_OG_sharpened'] = np.uint8(enhanced_volumes['OG_sharpened'] > threshold)
    
    return enhanced_volumes

def process_roi_gamma_mask(enhanced_volumes, final_roi, volume_array):
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
    threshold = np.percentile(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'], 99.98)
    enhanced_volumes['DESCARGAR_PRUEBA_roi_plus_gamma_mask'] = np.uint8(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'] > threshold)
    
    # Apply CLAHE to ROI plus gamma mask
    enhanced_volumes['PRUEBA_roi_plus_gamma_mask_clahe'] = apply_clahe(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'])
    threshold = np.percentile(enhanced_volumes['PRUEBA_roi_plus_gamma_mask_clahe'], 99.98)
    enhanced_volumes['DESCARGAR_PRUEBA_roi_plus_gamma_mask_clahe'] = np.uint8(enhanced_volumes['PRUEBA_roi_plus_gamma_mask_clahe'] > threshold)
    
    # Apply wavelet non-local means denoising
    enhanced_volumes['PRUEBA_WAVELET_NL'] = wavelet_nlm_denoise(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'], wavelet='db1')
    threshold = np.percentile(enhanced_volumes['PRUEBA_WAVELET_NL'], 99.98)
    enhanced_volumes['DESCARGAR_PRUEBA_WAVELET_NL'] = np.uint8(enhanced_volumes['PRUEBA_WAVELET_NL'] > threshold)
    
    return enhanced_volumes

def process_roi_only(enhanced_volumes, roi_volume, final_roi):
    """Process using only the ROI volume"""
    print("Applying ROI Only approach...")
    
    # Save ROI volume
    enhanced_volumes['roi_volume'] = roi_volume
    threshold = np.percentile(roi_volume, 99.98)
    enhanced_volumes['DESCARGAR_roi_volume'] = np.uint8(roi_volume > threshold)
    
    # Apply wavelet denoising
    enhanced_volumes['wavelet_only_roi'] = wavelet_denoise(roi_volume, wavelet='db1')
    threshold = np.percentile(enhanced_volumes['wavelet_only_roi'], 99.98)
    enhanced_volumes['DESCARGAR_wavelet_only_roi'] = np.uint8(enhanced_volumes['wavelet_only_roi'] > threshold)
    
    # Apply gamma correction to wavelet denoised volume
    enhanced_volumes['gamma_only_roi'] = gamma_correction(enhanced_volumes['wavelet_only_roi'], gamma=0.8)
    threshold = np.percentile(enhanced_volumes['gamma_only_roi'], 99.98)
    enhanced_volumes['DESCARGAR_gamma_only_roi'] = np.uint8(enhanced_volumes['gamma_only_roi'] > threshold)
    
    return enhanced_volumes

def process_roi_plus_gamma_after(enhanced_volumes, final_roi):
    """Process using ROI plus gamma correction after"""
    print("Applying ROI plus Gamma after approach...")
    
    if 'PRUEBA_roi_plus_gamma_mask' not in enhanced_volumes:
        print("Warning: PRUEBA_roi_plus_gamma_mask not found. Skipping this approach.")
        return enhanced_volumes
    
    # Apply gaussian filter to ROI plus gamma mask
    enhanced_volumes['2_gaussian_volume_roi'] = gaussian(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'], sigma=0.3)
    threshold = np.percentile(enhanced_volumes['2_gaussian_volume_roi'], 99.98)
    enhanced_volumes['DESCARGAR_2_gaussian_volume_roi'] = np.uint8(enhanced_volumes['2_gaussian_volume_roi'] > threshold)
    
    # Apply gamma correction
    enhanced_volumes['2_gamma_correction'] = gamma_correction(enhanced_volumes['2_gaussian_volume_roi'], gamma=0.8)
    threshold = np.percentile(enhanced_volumes['2_gamma_correction'], 99.98)
    enhanced_volumes['DESCARGAR_2_gamma_correction'] = np.uint8(enhanced_volumes['2_gamma_correction'] > threshold)
    
    # Apply top-hat transformation
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    tophat_2 = cv2.morphologyEx(enhanced_volumes['2_gaussian_volume_roi'], cv2.MORPH_TOPHAT, kernel_2)
    enhanced_volumes['2_tophat'] = cv2.addWeighted(enhanced_volumes['2_gaussian_volume_roi'], 1, tophat_2, 2, 0)
    threshold = np.percentile(enhanced_volumes['2_tophat'], 99.98)
    enhanced_volumes['DESCARGAR_2_tophat'] = np.uint8(enhanced_volumes['2_tophat'] > threshold)
    
    # Sharpen gamma corrected volume
    enhanced_volumes['2_sharpened'] = sharpen_high_pass(enhanced_volumes['2_gamma_correction'], strenght=0.8)
    threshold = np.percentile(enhanced_volumes['2_sharpened'], 99.98)
    enhanced_volumes['DESCARGAR_2_sharpened'] = np.uint8(enhanced_volumes['2_sharpened'] > threshold)
    
    # Apply LOG transform
    enhanced_volumes['2_LOG'] = log_transform_slices(enhanced_volumes['2_tophat'], c=3)
    threshold = np.percentile(enhanced_volumes['2_LOG'], 99.98)
    enhanced_volumes['DESCARGAR_2_LOG'] = np.uint8(enhanced_volumes['2_LOG'] > threshold)
    
    # Apply wavelet denoising
    enhanced_volumes['2_wavelet_roi'] = wavelet_denoise(enhanced_volumes['2_LOG'], wavelet='db4')
    threshold = np.percentile(enhanced_volumes['2_wavelet_roi'], 99.98)
    enhanced_volumes['DESCARGAR_2_wavelet_roi'] = np.uint8(enhanced_volumes['2_wavelet_roi'] > threshold)
    
    # Apply erosion
    enhanced_volumes['2_erode'] = morphological_operation(enhanced_volumes['2_sharpened'], operation='erode', kernel_size=1)
    threshold = np.percentile(enhanced_volumes['2_erode'], 99.98)
    enhanced_volumes['DESCARGAR_2_erode'] = np.uint8(enhanced_volumes['2_erode'] > threshold)
    
    # Apply gaussian filter
    enhanced_volumes['2_gaussian_2'] = gaussian(enhanced_volumes['2_erode'], sigma=0.2)
    threshold = np.percentile(enhanced_volumes['2_gaussian_2'], 99.98)
    enhanced_volumes['DESCARGAR_2_gaussian_2'] = np.uint8(enhanced_volumes['2_gaussian_2'] > threshold)
    
    # Sharpen gaussian filtered volume
    enhanced_volumes['2_sharpening_2_trial'] = sharpen_high_pass(enhanced_volumes['2_gaussian_2'], strenght=0.8)
    threshold = np.percentile(enhanced_volumes['2_sharpening_2_trial'], 99.98)
    enhanced_volumes['DESCARGAR_2_sharpening_2_trial'] = np.uint8(enhanced_volumes['2_sharpening_2_trial'] > threshold)
    
    return enhanced_volumes

def process_wavelet_roi(enhanced_volumes, roi_volume):
    """Process using wavelet denoising on ROI volume"""
    print("Applying Wavelet ROI approach...")
    
    # Apply non-local means denoising with wavelet
    enhanced_volumes['NUEVO_NLMEANS'] = wavelet_nlm_denoise(roi_volume)
    threshold = np.percentile(enhanced_volumes['NUEVO_NLMEANS'], 99.98)
    enhanced_volumes['DESCARGAR_NUEVO_NLMEANS'] = np.uint8(enhanced_volumes['NUEVO_NLMEANS'] > threshold)
    return enhanced_volumes

def process_original_idea(enhanced_volumes):
    """Process using the original idea approach"""
    print("Applying Original Idea approach...")
    
    if 'PRUEBA_roi_plus_gamma_mask' not in enhanced_volumes:
        print("Warning: PRUEBA_roi_plus_gamma_mask not found. Skipping this approach.")
        return enhanced_volumes
    
    # Apply gaussian filter
    enhanced_volumes['ORGINAL_IDEA_gaussian'] = gaussian(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'], sigma=0.3)
    threshold = np.percentile(enhanced_volumes['ORGINAL_IDEA_gaussian'], 99.98)
    enhanced_volumes['DESCARGAR_ORGINAL_IDEA_gaussian'] = np.uint8(enhanced_volumes['ORGINAL_IDEA_gaussian'] > threshold)
    
    # Apply gamma correction
    enhanced_volumes['ORGINAL_IDEA_gamma_correction'] = gamma_correction(enhanced_volumes['ORGINAL_IDEA_gaussian'], gamma=2)
    threshold = np.percentile(enhanced_volumes['ORGINAL_IDEA_gamma_correction'], 99.98)
    enhanced_volumes['DESCARGAR_ORGINAL_IDEA_gamma_correction'] = np.uint8(enhanced_volumes['ORGINAL_IDEA_gamma_correction'] > threshold)
    
    # Sharpen gamma corrected volume
    enhanced_volumes['ORGINAL_IDEA_sharpened'] = sharpen_high_pass(enhanced_volumes['ORGINAL_IDEA_gamma_correction'], strenght=0.8)
    threshold = np.percentile(enhanced_volumes['ORGINAL_IDEA_sharpened'], 99.98)
    enhanced_volumes['DESCARGAR_ORGINAL_IDEA_sharpened'] = np.uint8(enhanced_volumes['ORGINAL_IDEA_sharpened'] > threshold)
    
    # Apply wavelet denoising
    enhanced_volumes['ORIGINAL_IDEA_wavelet'] = wavelet_denoise(enhanced_volumes['ORGINAL_IDEA_sharpened'])
    threshold = np.percentile(enhanced_volumes['ORIGINAL_IDEA_wavelet'], 99.98)
    enhanced_volumes['DESCARGAR_ORIGINAL_IDEA_wavelet'] = np.uint8(enhanced_volumes['ORIGINAL_IDEA_wavelet'] > threshold)
    
    # Apply gaussian filter
    enhanced_volumes['ORGINAL_IDEA_gaussian_2'] = gaussian(enhanced_volumes['ORGINAL_IDEA_sharpened'], sigma=0.4)
    threshold = np.percentile(enhanced_volumes['ORGINAL_IDEA_gaussian_2'], 99.98)
    enhanced_volumes['DESCARGAR_ORGINAL_IDEA_gaussian_2'] = np.uint8(enhanced_volumes['ORGINAL_IDEA_gaussian_2'] > threshold)
    
    # Apply gamma correction
    enhanced_volumes['ORIGINAL_IDEA_GAMMA_2'] = gamma_correction(enhanced_volumes['ORGINAL_IDEA_gaussian_2'], gamma=1.4)
    threshold = np.percentile(enhanced_volumes['ORIGINAL_IDEA_GAMMA_2'], 99.98)
    enhanced_volumes['DESCARGAR_ORIGINAL_IDEA_GAMMA_2'] = np.uint8(enhanced_volumes['ORIGINAL_IDEA_GAMMA_2'] > threshold)
    
    # Apply top-hat transformation
    kernel_size_og = (1, 1)
    kernel_og = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size_og)
    tophat_og = cv2.morphologyEx(enhanced_volumes['ORGINAL_IDEA_gaussian_2'], cv2.MORPH_TOPHAT, kernel_og)
    enhanced_volumes['OG_tophat_1'] = cv2.addWeighted(enhanced_volumes['ORGINAL_IDEA_gaussian_2'], 1, tophat_og, 2, 0)
    threshold = np.percentile(enhanced_volumes['OG_tophat_1'], 99.98)
    enhanced_volumes['DESCARGAR_OG_tophat_1'] = np.uint8(enhanced_volumes['OG_tophat_1'] > threshold)
    
    return enhanced_volumes

def process_first_try(enhanced_volumes, roi_volume):
    """Process using the first try approach"""
    print("Applying First Try approach...")
    
    # Apply gaussian filter
    enhanced_volumes['FT_gaussian'] = gaussian(roi_volume, sigma=0.3)
    threshold = np.percentile(enhanced_volumes['FT_gaussian'], 99.98)
    enhanced_volumes['DESCARGAR_FT_gaussian'] = np.uint8(enhanced_volumes['FT_gaussian'] > threshold)
    
    # Apply top-hat transformation
    kernel_size = (1, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    tophat_ft = cv2.morphologyEx(roi_volume, cv2.MORPH_TOPHAT, kernel)
    enhanced_volumes['FT_tophat_1'] = cv2.addWeighted(roi_volume, 1, tophat_ft, 2, 0)
    
    # Subtract gaussian from top-hat
    enhanced_volumes['FT_RESTA_TOPHAT_GAUSSIAN'] = enhanced_volumes['FT_tophat_1'] - gaussian(roi_volume, sigma=0.8)
    threshold = np.percentile(enhanced_volumes['FT_RESTA_TOPHAT_GAUSSIAN'], 99.98)
    enhanced_volumes['DESCARGAR_FT_RESTA_TOPHAT_GAUSSIAN'] = np.uint8(enhanced_volumes['FT_RESTA_TOPHAT_GAUSSIAN'] > threshold)
    
    # Apply gamma correction
    enhanced_volumes['FT_gamma_correction'] = gamma_correction(enhanced_volumes['FT_gaussian'], gamma=5)
    threshold = np.percentile(enhanced_volumes['FT_gamma_correction'], 99.98)
    enhanced_volumes['DESCARGAR_FT_gamma_correction'] = np.uint8(enhanced_volumes['FT_gamma_correction'] > threshold)
    
    # Sharpen gamma corrected volume
    enhanced_volumes['FT_sharpened'] = sharpen_high_pass(enhanced_volumes['FT_gamma_correction'], strenght=0.4)
    threshold = np.percentile(enhanced_volumes['FT_sharpened'], 99.98)
    enhanced_volumes['DESCARGAR_FT_sharpened'] = np.uint8(enhanced_volumes['FT_sharpened'] > threshold)
    
    # Apply gaussian filter
    enhanced_volumes['FT_gaussian_2'] = gaussian(enhanced_volumes['FT_sharpened'], sigma=0.4)
    threshold = np.percentile(enhanced_volumes['FT_gaussian_2'], 99.98)
    enhanced_volumes['DESCARGAR_FT_gaussian_2'] = np.uint8(enhanced_volumes['FT_gaussian_2'] > threshold)
    
    # Apply gamma correction
    enhanced_volumes['FT_gamma_2'] = gamma_correction(enhanced_volumes['FT_gaussian_2'], gamma=1.2)
    threshold = np.percentile(enhanced_volumes['FT_gamma_2'], 99.98)
    enhanced_volumes['DESCARGAR_FT_gamma_2'] = np.uint8(enhanced_volumes['FT_gamma_2'] > threshold)
    
    # Apply opening operation
    enhanced_volumes['FT_opening'] = morph_operations(enhanced_volumes['FT_gamma_2'], iterations=2, kernel_shape='cross')
    threshold = np.percentile(enhanced_volumes['FT_opening'], 99.98)
    enhanced_volumes['DESCARGAR_FT_opening'] = np.uint8(enhanced_volumes['FT_opening'] > threshold)
    
    # Apply closing operation
    enhanced_volumes['FT_closing'] = morph_operations(enhanced_volumes['FT_opening'], operation='close')
    threshold = np.percentile(enhanced_volumes['FT_closing'], 99.98)
    enhanced_volumes['DESCARGAR_FT_closing'] = np.uint8(enhanced_volumes['FT_closing'] > threshold)
    
    # Apply erosion
    enhanced_volumes['FT_erode_2'] = morphological_operation(enhanced_volumes['FT_closing'], operation='erode', kernel_size=1)
    threshold = np.percentile(enhanced_volumes['FT_erode_2'], 99.98)
    enhanced_volumes['DESCARGAR_FT_erode_2'] = np.uint8(enhanced_volumes['FT_erode_2'] > threshold)
    
    # Apply top-hat transformation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    tophat = cv2.morphologyEx(enhanced_volumes['FT_gaussian_2'], cv2.MORPH_TOPHAT, kernel)
    enhanced_volumes['FT_tophat'] = cv2.addWeighted(enhanced_volumes['FT_gaussian_2'], 1, tophat, 2, 0)
    threshold = np.percentile(enhanced_volumes['FT_tophat'], 99.98)
    enhanced_volumes['DESCARGAR_FT_tophat'] = np.uint8(enhanced_volumes['FT_tophat'] > threshold)
    
    # Apply gaussian filter
    enhanced_volumes['FT_gaussian_3'] = gaussian(enhanced_volumes['FT_tophat'], sigma=0.1)
    threshold = np.percentile(enhanced_volumes['FT_gaussian_3'], 99.98)
    enhanced_volumes['DESCARGAR_FT_gaussian_3'] = np.uint8(enhanced_volumes['FT_gaussian_3'] > threshold)
    
    return enhanced_volumes

def enhance_ctp(inputVolume, inputROI=None, methods=None, outputDir=None, collect_histograms=True):
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
        
    Returns:
    --------
    dict
        Dictionary of enhanced volume nodes
    """
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
        
        # Process ROI
        filled_roi = ndimage.binary_fill_holes(roi_array)
        struct_elem = morphology.ball(10)
        closed_roi = morphology.binary_closing(filled_roi, struct_elem)
        
        if closed_roi.shape != volume_array.shape:
            final_roi = closed_roi
        else:
            final_roi = closed_roi
    else:
        final_roi = np.ones_like(volume_array)
    
    # Apply the ROI mask to the volume
    roi_volume = np.multiply(volume_array, final_roi)
    final_roi = final_roi.astype(np.uint8)

    # Initialize results dictionary
    enhanced_volumes = {}
    
    if methods == 'all' or 'original' in methods:
        enhanced_volumes = process_original_ctp(enhanced_volumes, volume_array)
    
    if methods == 'all' or 'roi_gamma' in methods:
        enhanced_volumes = process_roi_gamma_mask(enhanced_volumes, final_roi, volume_array)
    
    if methods == 'all' or 'roi_only' in methods:
        enhanced_volumes = process_roi_only(enhanced_volumes, roi_volume, final_roi)
        
    if methods == 'all' or 'roi_plus_gamma' in methods:
        enhanced_volumes = process_roi_plus_gamma_after(enhanced_volumes, final_roi)
        
    if methods == 'all' or 'wavelet_roi' in methods:
        enhanced_volumes = process_wavelet_roi(enhanced_volumes, roi_volume)
        
    if methods == 'all' or 'original_idea' in methods:
        enhanced_volumes = process_original_idea(enhanced_volumes)
        
    if methods == 'all' or 'first_try' in methods:
        enhanced_volumes = process_first_try(enhanced_volumes, roi_volume)
    
    # Save output if directory provided
    if outputDir is None:
        outputDir = slicer.app.temporaryPath()  
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    # Collect histogram data if requested
    if collect_histograms:
        histogram_data = collect_histogram_data(enhanced_volumes, outputDir)
    
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
outputDir = r"C:\Users\rocia\Downloads\TFG\Cohort\Enhance_ctp_tests\P7\TH45_histograms_only_percentile"
# Run the enhancement function
enhancedVolumeNodes = enhance_ctp(inputVolume, inputROI, methods='all', outputDir=outputDir, collect_histograms=True)
# Print the enhanced volume nodes
for method, volume_node in enhancedVolumeNodes.items():
    if volume_node is not None:
        print(f"Enhanced volume for method '{method}': {volume_node.GetName()}")
    else:
        print(f"Enhanced volume for method '{method}': No volume node available.")
    
#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Threshold_mask\threshold_percentile.py').read())