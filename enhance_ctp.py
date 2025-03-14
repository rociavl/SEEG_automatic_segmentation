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

logging.basicConfig(level=logging.DEBUG)

###################################
### Image processing filters ###
##################################


# Define enhancement methods

def threshold_metal_voxels_slice_by_slice(image_array, percentile=99.5):
    """Threshold metal voxels slice by slice."""
    print(f"DEBUG: Input array shape: {image_array.shape}, dtype: {image_array.dtype}") 
    if image_array.ndim != 3:
        raise ValueError("Input must be a 3D array.")

    mask = np.zeros_like(image_array, dtype=np.uint8)
    print(f"DEBUG: Initialized mask shape: {mask.shape}, dtype: {mask.dtype}") 
    for i in range(image_array.shape[2]):  
        print(f"DEBUG: Processing slice {i}")
        slice_data = image_array[:, :, i]
        print(f"DEBUG: Slice {i} shape: {slice_data.shape}, dtype: {slice_data.dtype}") 

        if not np.isfinite(slice_data).all():
            print(f"DEBUG: Slice {i} contains NaN or Inf values. Replacing with zeros.")
            slice_data = np.nan_to_num(slice_data)
            print(f"DEBUG: Slice {i} after NaN/Inf removal, dtype: {slice_data.dtype}") 

        threshold_value = np.nanpercentile(slice_data, percentile)
        print(f"DEBUG: Slice {i} threshold value: {threshold_value}") 

        binary_slice = (slice_data > threshold_value)
        print(f"DEBUG: Slice {i} binary slice shape: {binary_slice.shape}, dtype: {binary_slice.dtype}") 
        
        mask[:, :, i] = binary_slice.astype(np.uint8)
        print(f"DEBUG: Slice {i} mask slice shape: {mask[:,:,i].shape}, dtype: {mask[:,:,i].dtype}") 

    print(f"DEBUG: Final mask shape: {mask.shape}, dtype: {mask.dtype}") 
    print(f"DEBUG: unique values in mask: {np.unique(mask)}") 
    return mask

def apply_median_filter_2d(image_array, kernel_size_mm2=0.5, spacing=(1.0, 1.0)):

    kernel_radius_pixels = int(round(np.sqrt(kernel_size_mm2 / np.pi) / min(spacing[:2])))
    
    selem = disk(kernel_radius_pixels)

    filtered_slices = np.array([median(slice_, selem) for slice_ in image_array])

    print(f"✅ 2D Median filtering complete. Kernel radius: {kernel_radius_pixels} pixels")
    
    return filtered_slices

def apply_median_filter(image_array, kernel_size=3):
    return median_filter(image_array, size=kernel_size)

def resample_to_isotropic(image, spacing=(1.0, 1.0, 1.0)):
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled = resampler.Execute(image)
    return resampled


def threshold_metal_voxels(image_array, percentile=99.5):
    threshold_value = np.percentile(image_array, percentile)
    return image_array > threshold_value 


def thresholding_volume_histogram(volume, threshold=30):
        binary_volume = np.uint8(volume > threshold) 
        return binary_volume

def apply_clahe(roi_volume):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
        enhanced_slices = np.zeros_like(roi_volume)
        roi_volume_scaled = np.uint8(np.clip(roi_volume, 0, 255))  
        for i in range(roi_volume.shape[0]):
            enhanced_slices[i] = clahe.apply(roi_volume_scaled[i])
        return enhanced_slices


def gamma_correction(roi_volume, gamma=1.8):
        roi_volume_rescaled = rescale_intensity(roi_volume, in_range='image', out_range=(0, 1))
        gamma_corrected = exposure.adjust_gamma(roi_volume_rescaled, gamma=gamma)
        return np.uint8(np.clip(gamma_corrected * 255, 0, 255))

def adaptive_gamma_correction(volume, gamma=1.8):
    # Analyze the image histogram to adjust gamma dynamically
    volume_min = np.min(volume)
    volume_max = np.max(volume)
    gamma_adjusted = (volume_max - volume_min) / 255  
    return gamma_correction(volume, gamma=gamma_adjusted)

def local_otsu(roi_volume):
        block_size = 51
        local_thresh = filters.threshold_local(roi_volume, block_size, offset=10)
        binary_local = roi_volume > local_thresh
        binary_local_uint8 = np.uint8(binary_local)  
        return binary_local_uint8
    
    
def local_threshold(roi_volume):
        block_size = 51
        local_thresh = filters.threshold_local(roi_volume, block_size, offset=10)
        binary_local = roi_volume > local_thresh
        binary_local_uint8 = np.uint8(binary_local)  
        return binary_local_uint8
    
def denoise_2d_slices(volume, patch_size=5, patch_distance=6, h=0.1):
    """
    Denoise a 3D volume slice by slice using Non-Local Means Denoising.

    Parameters:
    - volume: 3D numpy array (CT/MRI volume)
    - patch_size: The size of the patches used for comparison (default is 5)
    - patch_distance: The maximum distance between patches to compare (default is 6)
    - h: Filtering parameter (higher values result in more smoothing)

    Returns:
    - Denoised 3D volume
    """
    # Initialize an empty array to hold the denoised volume
    denoised_volume = np.zeros_like(volume)

    # Apply Non-Local Means Denoising to each slice
    for i in range(volume.shape[0]): 
        slice_image = volume[i]  
        denoised_slice = denoise_nl_means(slice_image, patch_size=patch_size, patch_distance=patch_distance, h=h, multichannel=False)
        denoised_volume[i] = denoised_slice 

    return denoised_volume
   
def anisotropic_diffusion(roi_volume, n_iter=1, k=50, gamma=0.1):
        denoised_slices = np.zeros_like(roi_volume)
        for i in range(roi_volume.shape[0]):
            denoised_slice = filters.denoise_tv_bregman(roi_volume[i], n_iter, k, gamma)
            denoised_slices[i] = denoised_slice
        return denoised_slices

    
def bilateral_filter(roi_volume, d=3, sigma_color=75, sigma_space=40):
        # Convert roi_volume to uint8 (if not already) and ensure values are in the range 0-255
        roi_volume_uint8 = np.uint8(np.clip(roi_volume, 0, 255))

        filtered_slices = np.zeros_like(roi_volume_uint8)
        for i in range(roi_volume_uint8.shape[0]):
            filtered_slices[i] = cv2.bilateralFilter(roi_volume_uint8[i], d, sigma_color, sigma_space)
        return filtered_slices

def unsharp_masking(image, weight=1.5, blur_radius=1):
    blurred = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
    sharpened = cv2.addWeighted(image, 1 + weight, blurred, -weight, 0)
    return sharpened

def remove_large_objects_grayscale(image, size_threshold=5000):
    """
    Removes large objects from a grayscale image without binarization.
    
    Parameters:
        image (numpy.ndarray): The original grayscale image.
        size_threshold (int): The maximum object size to keep.
    
    Returns:
        numpy.ndarray: Image with large structures removed.
    """
    # Ensure input is a NumPy array
    image = np.asarray(image, dtype=np.float32)

    binary_mask = image > np.percentile(image, 60)

    labeled = label(binary_mask, connectivity=1)  
    for region in regionprops(labeled):
        if region.area > size_threshold:
            coords = region.coords  
            image[coords[:, 0], coords[:, 1], coords[:, 2]] = np.median(image)  
    return image
    
def wavelet_denoise(roi_volume, wavelet='db1', level=1):
        denoised_slices = np.zeros_like(roi_volume)
        for i in range(roi_volume.shape[0]):
            slice_image = roi_volume[i]
            coeffs = pywt.wavedec2(slice_image, wavelet, level=level)
            # Threshold the high-frequency coefficients
            coeffs_thresholded = list(coeffs)
            coeffs_thresholded[1:] = [tuple([pywt.threshold(c, value=0.2, mode='soft') for c in coeffs_level]) for coeffs_level in coeffs[1:]]
            denoised_slices[i] = pywt.waverec2(coeffs_thresholded, wavelet)
        return denoised_slices 

def sharpen_high_pass(roi_volume, kernel_size = 1, strenght=0.5):
        sharpened_slices = np.zeros_like(roi_volume)
        for i in range(roi_volume.shape[0]):
            slice_image = roi_volume[i]
            blurred = cv2.GaussianBlur(slice_image, (kernel_size, kernel_size), 0)
            high_pass = slice_image - blurred
            sharpened_slices[i] = np.clip(slice_image + strenght * high_pass, 0, 255)
        return sharpened_slices
    
def laplacian_sharpen(roi_volume, strength=1.5, iterations=3):
    """
    Apply Laplacian edge detection and sharpen the image while keeping the background black.
    
    Parameters:
    - roi_volume: 3D numpy array (CT/MRI volume)
    - strength: Intensity of sharpening
    - iterations: Number of iterations to apply sharpening
    
    Returns:
    - Sharpened 3D volume with preserved black background
    """
    sharpened_slices = np.zeros_like(roi_volume)

    for i in range(roi_volume.shape[0]):
        slice_image = roi_volume[i]

        # Convert to float32 for processing
        slice_image_float = slice_image.astype(np.float32)

        for _ in range(iterations):
            laplacian = cv2.Laplacian(slice_image_float, cv2.CV_32F)

            laplacian = np.clip(laplacian, -255, 255)  

            slice_image_float = np.clip(slice_image_float - strength * laplacian, 0, 255)

        slice_image_float[slice_image == 0] = 0

        sharpened_slices[i] = np.clip(slice_image_float, 0, 255).astype(np.uint8)

    return sharpened_slices


def adaptive_binarization(roi_volume, block_size=11, C=2, edge_strength=0.5):
        """
        Adaptive binarization with edge enhancement.
        """
        binarized_slices = np.zeros_like(roi_volume)

        for i in range(roi_volume.shape[0]):
            slice_image = roi_volume[i]
            
            # Apply adaptive threshold
            binarized = cv2.adaptiveThreshold(slice_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
            
            # Detect edges (Canny)
            edges = cv2.Canny(slice_image, 50, 150)

            # Blend edges with binarized image
            combined = cv2.addWeighted(binarized, 1 - edge_strength, edges, edge_strength, 0)

            binarized_slices[i] = combined

        return binarized_slices

def adaptive_threshold_high_pass_with_equalization(roi_volume, block_size=5, C=5, strength=0.5):
        """
        Apply histogram equalization, adaptive thresholding, and mild high-pass filtering 
        to sharpen the edges in a 3D volume.

        Parameters:
        - roi_volume: 3D numpy array (CT/MRI volume)
        - block_size: Size of the neighborhood for adaptive thresholding (must be odd)
        - C: Constant subtracted from the mean to fine-tune thresholding
        - strength: Mild sharpening strength to apply to the high-pass filter

        Returns:
        - Sharpened, equalized, and thresholded 3D volume
        """
        equalized_slices = np.zeros_like(roi_volume)

        for i in range(roi_volume.shape[0]):
            slice_image = roi_volume[i]
            
            # Apply histogram equalization to improve contrast
            equalized_image = cv2.equalizeHist(np.uint8(slice_image))  # Ensure the image is uint8
            
            # Apply adaptive threshold to the equalized slice
            thresholded_image = cv2.adaptiveThreshold(
                equalized_image, 
                255, 
                cv2.ADAPTIVE_THRESH_MEAN_C, 
                cv2.THRESH_BINARY, 
                block_size, 
                C
            )

            # Apply mild high-pass filter to sharpen the edges
            blurred = cv2.GaussianBlur(thresholded_image, (3, 3), 0)  # Using a slightly larger kernel for gentler blur
            high_pass = thresholded_image - blurred  # High-pass component
            sharpened_slice = np.clip(thresholded_image + strength * high_pass, 0, 255)  # Subtle sharpening
            
            equalized_slices[i] = sharpened_slice

        return equalized_slices
def morphological_operation(image, operation='dilate', kernel_size=3, iterations=1):

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if operation == 'dilate':
     processed_image = cv2.dilate(image, kernel, iterations=iterations)
    
    elif operation == 'erode':
        processed_image = cv2.erode(image, kernel, iterations=iterations)
   
    else:
        raise ValueError("Invalid operation. Use 'dilate' or 'erode'.")
    
    return processed_image

def morph_operations(roi_volume, kernel_size=1, operation="open", iterations=1, kernel_shape="square"):

    if kernel_shape == "square":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    elif kernel_shape == "cross":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    elif kernel_shape == "ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    else:
        raise ValueError("Invalid kernel_shape. Use 'square', 'cross', or 'ellipse'.")

    processed_slices = np.zeros_like(roi_volume)

    for i in range(roi_volume.shape[0]):
        if operation == "open":
            processed_slices[i] = cv2.morphologyEx(roi_volume[i], cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == "close":
            processed_slices[i] = cv2.morphologyEx(roi_volume[i], cv2.MORPH_CLOSE, kernel, iterations=iterations)

    return processed_slices

def apply_frangi_slice_by_slice(volume):
    """
    Applies the Frangi vesselness filter to a 3D volume slice-by-slice.
    
    Parameters:
        volume (numpy.ndarray): The 3D input image volume.

    Returns:
        numpy.ndarray: The enhanced 3D volume with Frangi applied per slice.
    """
    # Initialize an empty array with the same shape as input
    enhanced_volume = np.zeros_like(volume, dtype=np.float32)

    # Apply Frangi filter slice-by-slice along the Z-axis
    for i in range(volume.shape[0]):  
        enhanced_volume[i] = frangi(volume[i], scale_range=(1, 5), scale_step=2)

    return enhanced_volume   
    
def contrast_stretching(roi_volume, lower_percentile=2, upper_percentile=98):
        """
        Apply contrast stretching to enhance the image without over-amplifying noise.

        Parameters:
        - roi_volume: 3D numpy array (CT/MRI volume)
        - lower_percentile: Lower bound for intensity scaling (default 2%)
        - upper_percentile: Upper bound for intensity scaling (default 98%)

        Returns:
        - Contrast-enhanced 3D volume
        """
        stretched_slices = np.zeros_like(roi_volume, dtype=np.uint8)

        for i in range(roi_volume.shape[0]):
            slice_image = roi_volume[i]

            # Get intensity range based on percentiles
            p_low, p_high = np.percentile(slice_image, (lower_percentile, upper_percentile))

            # Apply contrast stretching
            stretched_slices[i] = exposure.rescale_intensity(slice_image, in_range=(p_low, p_high), out_range=(0, 255)).astype(np.uint8)

        return stretched_slices
        
def canny_edges(roi_volume, sigma=1.0, low_threshold=50, high_threshold=150):
        """
        Apply Canny edge detection to each slice in the volume with optional Gaussian blur.

        Parameters:
        - roi_volume: 3D numpy array (CT/MRI volume)
        - sigma: Standard deviation for Gaussian blur (affects edge detection)
        - low_threshold: Lower bound for edges
        - high_threshold: Upper bound for edges
        
        Returns:
        - Edge-detected 3D volume
        """
        edge_slices = np.zeros_like(roi_volume)

        for i in range(roi_volume.shape[0]):
            slice_image = roi_volume[i]

            # Apply Gaussian blur with sigma to smooth the image before edge detection
            blurred_image = cv2.GaussianBlur(slice_image, (5, 5), sigma)

            # Apply Canny edge detection after the blur
            edge_slices[i] = cv2.Canny(blurred_image, low_threshold, high_threshold)

        return edge_slices
def log_transform_slices(roi_volume, c=5, sigma=0.4):
    """
    Apply log transformation slice by slice, enhance small bright structures.
    
    Args:
        roi_volume (numpy array): 3D volume data (Z, Y, X)
        c (float): Scaling factor for log transformation
        sigma (float): Gaussian blur factor to preserve small structures
    
    Returns:
        numpy array: Processed volume with enhanced electrode brightness
    """
    roi_volume_log = np.zeros_like(roi_volume, dtype=np.float32) 

    for i in range(roi_volume.shape[0]):  
        slice_data = roi_volume[i].astype(np.float32)  
        slice_data_smoothed = gaussian_filter(slice_data, sigma=sigma)  
        slice_data_log = c * np.log1p(slice_data_smoothed)  

        # Enhance electrode edges
        slice_data_edges = laplace(slice_data_log)  
        slice_data_enhanced = slice_data_log + 0.5 * slice_data_edges  

        # Normalize slice to 0-255
        slice_data_enhanced = np.clip(slice_data_enhanced / np.max(slice_data_enhanced) * 255, 0, 255)

        roi_volume_log[i] = slice_data_enhanced  
    
    return roi_volume_log.astype(np.uint8)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Binarization
    return binary

def sobel_edge_detection(roi_volume):
    sobel_slices = np.zeros_like(roi_volume)

    for i in range(roi_volume.shape[0]):
        slice_image = roi_volume[i]

        # Apply Sobel edge detection in both x and y directions
        sobel_x = cv2.Sobel(slice_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(slice_image, cv2.CV_64F, 0, 1, ksize=3)

        # Compute the magnitude of the gradients (combined edges in both directions)
        sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

        # Convert back to 8-bit image
        sobel_edge_detected = np.uint8(np.clip(sobel_magnitude, 0, 255))

        # Store the processed slice
        sobel_slices[i] = sobel_edge_detected

    return sobel_slices


###############################
## masks and improvements ######
#################################

def remove_large_objects(segmented_image, size_threshold):
    """
    Removes large objects from a labeled image.

    Parameters:
        segmented_image (numpy.ndarray): 3D image with labeled connected components (after watershed).
        size_threshold (int): Minimum size (in number of voxels) for an object to be kept.

    Returns:
        numpy.ndarray: Modified image with large objects removed.
    """
    
    labeled_image = measure.label(segmented_image, connectivity=1)  
  
    mask = np.zeros_like(segmented_image, dtype=bool)
    
 
    for region in measure.regionprops(labeled_image):
        # If the region is small enough, keep it
        if region.area <= size_threshold:
            mask[labeled_image == region.label] = True

    filtered_image = segmented_image * mask
    
    return filtered_image

def denoise_2d_slices(volume, patch_size=2, patch_distance=2, h=0.1):
    """
    Denoise a 3D volume slice by slice using Non-Local Means Denoising.

    Parameters:
    - volume: 3D numpy array (CT/MRI volume)
    - patch_size: The size of the patches used for comparison (default is 5)
    - patch_distance: The maximum distance between patches to compare (default is 6)
    - h: Filtering parameter (higher values result in more smoothing)

    Returns:
    - Denoised 3D volume
    """
    # Initialize an empty array to hold the denoised volume
    denoised_volume = np.zeros_like(volume)

    for i in range(volume.shape[0]):  
        slice_image = volume[i]  
        denoised_slice = denoise_nl_means(slice_image, patch_size=patch_size, patch_distance=patch_distance, h=h)
        denoised_volume[i] = denoised_slice  

    return denoised_volume


def vtk_to_numpy(vtk_image_node):
    """Convert vtkMRMLScalarVolumeNode to numpy array."""
    np_array = slicer.util.arrayFromVolume(vtk_image_node)
    
    return np_array

def update_vtk_volume_from_numpy(np_array, vtk_image_node):
    """Update vtkMRMLScalarVolumeNode from a NumPy array."""
    slicer.util.updateVolumeFromArray(vtk_image_node, np_array)  
    slicer.app.processEvents()

def generate_contour_mask(brain_mask, dilation_iterations=1):

    if not isinstance(brain_mask, np.ndarray):
        brain_mask = slicer.util.arrayFromVolume(brain_mask)
    
    # Ensure brain_mask is binary (0 and 255)
    brain_mask = np.uint8(brain_mask > 0) * 255
    
    # Create structuring element
    kernel = np.ones((3, 3), np.uint8)
    # Dilate the brain mask
    dilated_mask = cv2.dilate(brain_mask, kernel, iterations=dilation_iterations)

    contour_mask = cv2.subtract(dilated_mask, brain_mask)
    
    return contour_mask


def get_watershed_markers(binary):
    binary = np.uint8(binary > 0) * 255  # Convert to binary mask

    # Compute distance transform
    distance = distance_transform_edt(binary)

    # ✅ Smooth distance transform to reduce noise
    smoothed_distance = gaussian_filter(distance, sigma=1)

    # Apply Otsu threshold on smoothed distance map
    otsu_threshold = filters.threshold_otsu(smoothed_distance)
    markers = measure.label(smoothed_distance > otsu_threshold)  # Label regions

    return markers

def apply_watershed_on_volume(volume_array):
    print(f"apply_watershed_on_volume - Input volume shape: {volume_array.shape}")
    
    # Initialize final watershed segmented volume
    watershed_segmented = np.zeros_like(volume_array, dtype=np.uint8)

    for i in range(volume_array.shape[0]):  # Iterate over slices
        binary_slice = np.uint8(volume_array[i] > 0) * 255  # Convert to binary
        marker_slice = get_watershed_markers(binary_slice)

        # Use distance transform instead of binary slice for better segmentation
        distance = distance_transform_edt(binary_slice)
        segmented_slice = segmentation.watershed(-distance, marker_slice, mask=binary_slice)

        # Remove tiny objects to reduce noise
        cleaned_segmented_slice = remove_small_objects(segmented_slice, min_size=10)

        watershed_segmented[i] = cleaned_segmented_slice

    print(f"Watershed segmentation - Final result shape: {watershed_segmented.shape}, Dtype: {watershed_segmented.dtype}")
    return watershed_segmented


# Function to apply DBSCAN clustering on 2D slices
def apply_dbscan_2d(volume_array, eps=5, min_samples=10):
    print(f"apply_dbscan_2d - Input volume shape: {volume_array.shape}")
    
    clustered_volume = np.zeros_like(volume_array, dtype=np.int32)  # Store cluster labels
    cluster_counts = {}

    for slice_idx in range(volume_array.shape[0]):
        print(f"Processing Slice {slice_idx} for DBSCAN...")
        
        slice_data = volume_array[slice_idx]

        # Extract nonzero points
        yx_coords = np.column_stack(np.where(slice_data > 0))
        print(f"Slice {slice_idx} - Non-zero points: {len(yx_coords)}")

        if len(yx_coords) == 0:
            print(f"Slice {slice_idx} - No non-zero points, skipping...")
            continue  # Skip empty slices

        # Apply DBSCAN
        print(f"Applying DBSCAN on Slice {slice_idx}...")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(yx_coords)

        # Convert labels back to 2D
        clustered_slice = np.zeros_like(slice_data, dtype=np.int32)
        for i, (y, x) in enumerate(yx_coords):
            clustered_slice[y, x] = labels[i] + 1  # Shift labels to be non-negative

        # Store results
        clustered_volume[slice_idx] = clustered_slice
        cluster_counts[slice_idx] = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise

    print(f"DBSCAN - Cluster counts per slice: {cluster_counts}")
    return clustered_volume, cluster_counts

def separate_merged_electrodes_mm(mask, spacing):
    """
    Uses distance transform + watershed to separate merged electrodes with mm² scaling.

    Parameters:
    - mask: 3D binary numpy array (1 = electrode, 0 = background)
    - spacing: Tuple (z_spacing, y_spacing, x_spacing) in mm

    Returns:
    - separated_mask: 3D numpy array with segmented electrodes.
    """
    separated_mask = np.zeros_like(mask, dtype=np.int32)
    
    # Compute area in mm² per voxel (y_spacing * x_spacing)
    voxel_area_mm2 = spacing[1] * spacing[2]
    
    for i in range(mask.shape[0]):  
        if np.sum(mask[i]) == 0:
            continue  

        distance = distance_transform_edt(mask[i]) * np.sqrt(voxel_area_mm2)

        footprint_size = max(1, int(2 / np.sqrt(voxel_area_mm2)))  
        local_maxi = peak_local_max(distance, footprint=np.ones((footprint_size, footprint_size)), labels=mask[i])

        markers, _ = label(local_maxi)

        separated_mask[i] = watershed(-distance, markers, mask=mask[i])
    
    return separated_mask

def apply_gmm(image, n_components=3):
   
    pixel_values = image[image > 0].reshape(-1, 1)  

    if pixel_values.shape[0] == 0:
        return np.zeros_like(image)  

    # Check for unique intensity values
    unique_values = np.unique(pixel_values)
    print(f"Unique intensity values count: {len(unique_values)}")

    if len(unique_values) < n_components:
        print("⚠️ Not enough unique intensity values for GMM clustering!")
        return np.zeros_like(image)  # Return blank mask

    try:
        # Dynamically adjust the number of clusters
        n_clusters = min(n_components, len(unique_values))
        gmm = GaussianMixture(n_components=n_clusters)
        gmm_labels = gmm.fit_predict(pixel_values) 

        gmm_image = np.zeros_like(image)

        # Get indices where image > 0
        indices = np.where(image > 0)

        gmm_image[indices] = gmm_labels + 1  

    except Exception as e:
        print(f"⚠️ GMM error: {e}")
        return np.zeros_like(image)  

    return gmm_image
def apply_snakes_tiny(volume):

    final_contours = np.zeros_like(volume, dtype=np.uint8)

  
    for i in range(volume.shape[0]): 
        slice_2d = volume[i] 
  
        edges = feature.canny(slice_2d)  
  
        s = np.zeros(slice_2d.shape)
        center = (slice_2d.shape[0] // 2, slice_2d.shape[1] // 2)
   
        rr, cc = draw.ellipse(center[0], center[1], 20, 20)
        s[rr, cc] = 1  # Create an initial contour
    
        snake = active_contour(edges, s, alpha=0.015, beta=10, gamma=0.001, max_num_iter=250)

        contour_mask = np.zeros_like(slice_2d)
        contour_mask[tuple(snake.T.astype(int))] = 1  
   
        final_contours[i] = contour_mask
    
    return final_contours


def get_auto_seeds(binary):

    # Compute distance transform
    distance = distance_transform_edt(binary)

    # Detect local maxima (potential seeds)
    local_max = peak_local_max(distance, min_distance=1, labels=binary)

    # Label detected seeds
    seeds, num_seeds = label(local_max)

    print(f"Generated {num_seeds} automatic seeds.")
    return seeds

def refine_labels(labels, min_size=5):

    refined = np.zeros_like(labels)
    for region in regionprops(labels):
        if region.area >= min_size:  
            refined[labels == region.label] = region.label
    return refined

def region_growing(binary, tolerance=3, min_size=5):
    seeds = get_auto_seeds(binary)

    segmented = np.zeros_like(binary, dtype=np.uint8)

    for label_id in range(1, seeds.max() + 1):
        mask = seeds == label_id
        region = binary.copy()

        region_grown = (distance_transform_edt(mask) < tolerance) & binary

        segmented[region_grown] = label_id  

    segmented = refine_labels(segmented, min_size)

    return segmented



def separate_by_erosion_and_closing(mask, kernel_size_mm2=0.5, spacing=(1.0, 1.0), erosion_iterations=1, closing_iterations=1):
    mask = np.uint8(mask > 0) 

    kernel_radius_pixels_x = int(round(np.sqrt(kernel_size_mm2 / np.pi) / spacing[0]))
    kernel_radius_pixels_y = int(round(np.sqrt(kernel_size_mm2 / np.pi) / spacing[1]))

    kernel_size_pixels = (max(3, kernel_radius_pixels_x * 2 + 1), 
                          max(3, kernel_radius_pixels_y * 2 + 1))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size_pixels)

    eroded_mask = cv2.erode(mask, kernel, iterations=erosion_iterations)

    closed_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)

    print(f"✅ Erosion and closing applied with kernel size {kernel_size_pixels} pixels")
    
    return closed_mask

def morphological_opening_slice_by_slice(mask, spacing, min_dist_mm=2.0):
    """
    Apply 2D morphological opening slice by slice for a 3D mask.
    """
    
    kernel_size = int(min_dist_mm / spacing[0])  

    kernel = disk(kernel_size)  

    opened_mask = np.zeros_like(mask)

    for i in range(mask.shape[0]):
        slice_image = mask[i]
        
        opened_slice = ndimage.binary_opening(slice_image, structure=kernel).astype(np.uint8)
        
        opened_mask[i] = opened_slice
    
    return opened_mask

def separate_merged_2d(mask, electrode_radius=0.4, voxel_size=(1, 1, 1), distance_threshold=1):

    mask = (mask > 0).astype(np.uint8)

    distance = distance_transform_edt(mask)

    thresholded_distance = distance > distance_threshold  

    labeled_mask, num_features = label(thresholded_distance)

    separated_mask = np.zeros_like(mask)
    for i in range(1, num_features + 1):
        separated_mask[labeled_mask == i] = 1  

    return separated_mask


###########################################    
# Function to enhance the CTP.3D images ###
###############################################

def enhance_ctp(inputVolume, inputROI=None, methods = 'all', outputDir=None):
    methods ='all'
    # Convert input volume to numpy array
    volume_array = slicer.util.arrayFromVolume(inputVolume)

    if volume_array is None or volume_array.size == 0:
        print("Input volume data is empty or invalid.")
        return None

    # If inputROI is provided
    if inputROI is not None:
        # Convert the ROI to a binary mask
        roi_array = slicer.util.arrayFromVolume(inputROI)
        roi_array = np.uint8(roi_array > 0)  # Ensure binary mask (0 or 1)
        
        # Print shapes for debugging
        print(f"Shape of input volume: {volume_array.shape}")
        print(f"Shape of ROI mask: {roi_array.shape}")

        # Perform any morphological operations if needed
        print("Filling inside the ROI...")
        filled_roi = ndimage.binary_fill_holes(roi_array)
        
        print("Applying morphological closing...")
        struct_elem = morphology.ball(10)
        closed_roi = morphology.binary_closing(filled_roi, struct_elem)
        
        # The final ROI after morphological operations
        final_roi = closed_roi
        
        # Ensure that final_roi and volume_array have the same shape
        # Ensure that final_roi and volume_array have the same shape
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
    
    # Apply mask with explicit element-wise multiplication
    
    roi_volume = np.multiply(volume_array, final_roi)
    final_roi = final_roi.astype(np.uint8)
    # Apply selected enhancement methods

    enhanced_volumes = {}
    if methods == 'all':
        
        ### Only CTP ###
        enhanced_volumes['OG_volume_array'] = volume_array
        print(f"OG_volume_array shape: {enhanced_volumes['OG_volume_array'].shape}")
        #enhanced_volumes['denoise_ctp'] = denoise_2d_slices(enhanced_volumes['gaussian_volume_og'], patch_size=2, patch_distance=2, h=0.8)
        enhanced_volumes['OG_gaussian_volume_og'] = gaussian(enhanced_volumes['OG_volume_array'], sigma=0.3)
        enhanced_volumes['OG_gamma_volume_og'] = gamma_correction(enhanced_volumes['OG_gaussian_volume_og'] , gamma=3)

        enhanced_volumes['og_THRESHOLD_gamma'] = np.uint8(enhanced_volumes['OG_gamma_volume_og'] > 129)
        ####4:102, 5:129
        enhanced_volumes['OG_sharpened_volume_og'] = sharpen_high_pass(enhanced_volumes['OG_gamma_volume_og'], strenght = 0.7)
        enhanced_volumes['og_THRESHOLD_sharpened'] = np.uint8(enhanced_volumes['OG_sharpened_volume_og'] > 132) ###4:149, 5:132

        enhanced_volumes['OG_LOG'] = log_transform_slices(enhanced_volumes['OG_sharpened_volume_og'], c= 3)
        enhanced_volumes['OG_FINAL_thresholded_ctp_per_og'] = np.uint8(enhanced_volumes['OG_LOG'] > 187) ### 5:187


        # enhanced_volumes['median_voxel_ctp'] = apply_median_filter_2d(enhanced_volumes['thresholded_ctp_per_og'], kernel_size_mm2=0.4)
        # enhanced_volumes['separate_erosion_ctp_og'] = separate_by_erosion(enhanced_volumes['median_voxel_ctp'], kernel_size_mm2=0.6, spacing=(1.0, 1.0), iterations=5)
        # enhanced_volumes['MASK_LABEL_ctp'] = np.uint8(enhanced_volumes['gamma_ctp_2'] > 0) * 255


        #### PRUEBA ROI####
        enhanced_volumes['PRUEBA_roi_plus_gamma_mask'] = enhanced_volumes['OG_gamma_volume_og']* final_roi
        enhanced_volumes['Prueba_final_roi'] = final_roi
        enhanced_volumes['PRUEBA_THRESHOLD'] = np.uint8(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'] > 133)  ### 4: 122, 5:133
        ######### This isn't working :C ########
        enhanced_volumes['PRUEBA_separate'] = separate_merged_2d(enhanced_volumes['PRUEBA_THRESHOLD'], electrode_radius=0.4, voxel_size=(1, 1, 1), distance_threshold=1)
        ########################################
        enhanced_volumes['gaussian_volume_roi'] = gaussian(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'], sigma=0.3)
        enhanced_volumes['PRUEBA_GAUSSIN_thre'] = np.uint8(enhanced_volumes['gaussian_volume_roi'] > 0.556) ### 4: 0.408, 5: 0.556
        enhanced_volumes['sharpened_roi'] = sharpen_high_pass(enhanced_volumes['gaussian_volume_roi'], strenght = 0.2)
        enhanced_volumes['gamma_volume_roi'] = gamma_correction(enhanced_volumes['sharpened_roi'], gamma=5)
        enhanced_volumes['PRUEBA_FINAL_thresholded_ctp_volume_roi'] = np.uint8(enhanced_volumes['gamma_volume_roi'] > 28) # 4: 16, 5: 28

        ####################
        enhanced_volumes['LOG_roi'] = log_transform_slices(enhanced_volumes['gamma_volume_roi'], c=3)
        enhanced_volumes['Morph_opening'] = morphological_opening_slice_by_slice(enhanced_volumes['gamma_volume_roi'], spacing=(1.0, 1.0), min_dist_mm=1)
        enhanced_volumes['FINAL_20_thresholded_ctp_volume_roi'] = threshold_metal_voxels_slice_by_slice(enhanced_volumes['Morph_opening'], percentile= 99.9)
        #######################

        ### Only ROI ###
        enhanced_volumes['roi_volume'] = roi_volume
        enhanced_volumes['Threshold_roi_volume_ONLY'] = np.uint8(enhanced_volumes['roi_volume'] > 2422) ## 4: 1526, 5: 2422
        enhanced_volumes['sharpened_roi'] = sharpen_high_pass(enhanced_volumes['roi_volume'], strenght = 0.8)
        enhanced_volumes['LOG_roi'] = log_transform_slices(enhanced_volumes['sharpened_roi'], c=1)
        #######
        enhanced_volumes['Threshold_roi_volume'] = threshold_metal_voxels_slice_by_slice(enhanced_volumes['LOG_roi'], percentile= 99.7)

       ### ROI gamma###
        enhanced_volumes['2_gaussian_volume_roi'] = gaussian(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'], sigma=0.3)
        enhanced_volumes['2_wavelet_denoised'] = wavelet_denoise(enhanced_volumes['2_gaussian_volume_roi'])
        enhanced_volumes['FINAL_2'] = np.uint8(enhanced_volumes['2_wavelet_denoised'] > 0.414) ### 4: 0.303, 5: 0.414


        enhanced_volumes['2_gamma_correction'] = gamma_correction(enhanced_volumes['2_gaussian_volume_roi'] , gamma = 3)
        enhanced_volumes['2_gamma_threshold'] = np.uint8(enhanced_volumes['2_gamma_correction'] > 37) 
        enhanced_volumes['2_sharpened'] = sharpen_high_pass(enhanced_volumes['2_gamma_correction'], strenght = 0.8)
        enhanced_volumes['2_THRESHOLD_sharpened'] = np.uint8(enhanced_volumes['2_sharpened'] > 45) 
        kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        # Apply white top-hat transformation
        tophat_2 = cv2.morphologyEx(enhanced_volumes['2_sharpened'], cv2.MORPH_TOPHAT, kernel_2)
        enhanced_volumes['2_tophat'] = cv2.addWeighted(enhanced_volumes['2_sharpened'], 1, tophat_2, 2, 0)
        enhanced_volumes['2_THRESHOLD_tophat'] = np.uint8(enhanced_volumes['2_tophat'] > 22) 
        enhanced_volumes['2_LOG'] = log_transform_slices(enhanced_volumes['2_tophat'], c=3)
        enhanced_volumes['2_FINAL_2_LOG_roi'] = np.uint8(enhanced_volumes['2_LOG'] > 157) 
        

        enhanced_volumes['2_erode'] = morphological_operation(enhanced_volumes['2_sharpened'], operation='erode', kernel_size=1)
        enhanced_volumes['2_threshold_erode'] = np.uint8(enhanced_volumes['2_erode'] > 33)
        enhanced_volumes['2_FINAL_20_erode'] = threshold_metal_voxels_slice_by_slice(enhanced_volumes['2_erode'], percentile= 99.9)
        enhanced_volumes['2_gaussian_2'] = gaussian(enhanced_volumes['2_erode'], sigma= 0.2)
        # enhanced_volumes['opening'] = morph_operations(enhanced_volumes['gamma_2'], iterations=2, kernel_shape= 'cross')
        # enhanced_volumes['gamma_3'] = gamma_correction(enhanced_volumes['tophat'], gamma= 2) 
        enhanced_volumes['2_denoised_roi_final'] = denoise_2d_slices(enhanced_volumes['2_gaussian_2'], patch_size=1, patch_distance=1, h=0.2)
        #enhanced_volumes['gamma_4'] = gamma_correction(enhanced_volumes['denoised_roi_final'], gamma= 3)
        enhanced_volumes['2_FINAL_20_MASK_LABEL'] = threshold_metal_voxels_slice_by_slice(enhanced_volumes['2_denoised_roi_final'], percentile= 99.9)

        
        
        
        ###ORGINAL_IDEA ####
        enhanced_volumes['ORGINAL_IDEA_gaussian'] = gaussian(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'], sigma= 0.3)
        enhanced_volumes['FINAL_ORGINAL_IDEA_GAUSSIAN_THRESHOLD'] = np.uint(enhanced_volumes['ORGINAL_IDEA_gaussian'] > 0.60) 
        enhanced_volumes['ORGINAL_IDEA_gamma_correction'] = gamma_correction(enhanced_volumes['ORGINAL_IDEA_gaussian'], gamma = 2)
        enhanced_volumes['ORGINAL_IDEA_sharpened'] = sharpen_high_pass(enhanced_volumes['ORGINAL_IDEA_gamma_correction'], strenght = 0.8)
        enhanced_volumes['ORIGINAL_IDEA_SHARPENED_LABEL'] = np.uint8(enhanced_volumes['ORGINAL_IDEA_sharpened'] > 81)
        enhanced_volumes['ORIGINAL_IDEA_SHARPENED_OPENING'] = morph_operations(enhanced_volumes['ORIGINAL_IDEA_SHARPENED_LABEL'], operation= 'open', kernel_shape= 'cross', kernel_size= 1)

        enhanced_volumes['ORIGINAL_IDEA_wavelet'] = wavelet_denoise(enhanced_volumes['ORGINAL_IDEA_sharpened'])
        enhanced_volumes['ORGINAL_IDEA_FINAL_MASK_LABEL'] = threshold_metal_voxels_slice_by_slice(enhanced_volumes['ORIGINAL_IDEA_wavelet'], percentile= 99.8)

        enhanced_volumes['ORGINAL_IDEA_gaussian_2'] = gaussian(enhanced_volumes['ORGINAL_IDEA_sharpened'], sigma= 0.4)
        enhanced_volumes['ORIGINAL_IDEA_GAMMA_2'] = gamma_correction(enhanced_volumes['ORGINAL_IDEA_gaussian_2'], gamma = 2)
        enhanced_volumes['ORIGINAL_IDEA_THRESHOLD_GAMMA_2'] = np.uint8(enhanced_volumes['ORIGINAL_IDEA_GAMMA_2'] > 9)
        enhanced_volumes['ORIGINAL_IDEA_OPENING'] = morph_operations(enhanced_volumes['ORIGINAL_IDEA_GAMMA_2'], iterations=2, kernel_shape= 'cross')
        enhanced_volumes['ORGINAL_IDEA__FINAL_MASK_LABEL'] = threshold_metal_voxels_slice_by_slice(enhanced_volumes['ORIGINAL_IDEA_OPENING'], percentile= 99.9)
        enhanced_volumes['ORIGINAL_OPENINING'] = morphological_opening_slice_by_slice(enhanced_volumes['ORGINAL_IDEA__FINAL_MASK_LABEL'], spacing=(1.0, 1.0), min_dist_mm=0.05)

        ### First try ###

        enhanced_volumes['FT_gaussian'] = gaussian(roi_volume, sigma= 0.3)
        enhanced_volumes['FINAL_FT_GAUSSIAN_THRESHOLD'] = np.uint(enhanced_volumes['ORGINAL_IDEA_gaussian'] > 1626) 

        enhanced_volumes['FT_gamma_correction'] = gamma_correction(enhanced_volumes['FT_gaussian'], gamma = 5)
        enhanced_volumes['FINAL_FT_gamma_THRESHOLD'] = np.uint(enhanced_volumes['FT_gamma_correction'] > 24) 

        
        enhanced_volumes['FT_sharpened'] = sharpen_high_pass(enhanced_volumes['FT_gamma_correction'], strenght = 0.4)
        enhanced_volumes['FT_gaussian_2'] = gaussian(enhanced_volumes['FT_sharpened'], sigma= 0.4)
        enhanced_volumes['FT_gaussinan_threshold'] = np.uint8(enhanced_volumes['FT_gaussian_2'] > 0.155) 


        enhanced_volumes['FT_gamma_2'] = gamma_correction(enhanced_volumes['FT_gaussian_2'], gamma= 2)
        enhanced_volumes['FT_opening'] = morph_operations(enhanced_volumes['FT_gamma_2'], iterations=2, kernel_shape= 'cross')
        enhanced_volumes['FT_closing'] = morph_operations(enhanced_volumes['FT_opening'], operation= 'close')
        #enhanced_volumes['wo_large_objects'] = remove_large_objects(enhanced_volumes['closing'], size_threshold= 9000)
        enhanced_volumes['FT_erode_2'] = morphological_operation(enhanced_volumes['FT_closing'], operation='erode', kernel_size=1)
        # Create an elliptical structuring element (adjust size if needed)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        # Apply white top-hat transformation
        tophat = cv2.morphologyEx(enhanced_volumes['FT_erode_2'], cv2.MORPH_TOPHAT, kernel)
        enhanced_volumes['FT_tophat'] = cv2.addWeighted(enhanced_volumes['FT_erode_2'], 1, tophat, 2, 0)
        enhanced_volumes['FT_THRESHOLD_TOPHAT'] = np.uint8(enhanced_volumes['FT_tophat'] > 2)

        #################
        enhanced_volumes['FT_gaussian_3'] = gaussian(enhanced_volumes['FT_tophat'], sigma= 0.1)
        #enhanced_volumes['FT_THRESHOLD_GAMMA_3'] = np.uint8(enhanced_volumes['FT_gaussian_3'] > 8) 
        ##################




    # Set output directory 
    if outputDir is None:
        outputDir = slicer.app.temporaryPath()  
    
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # Process each enhanced volume
    enhancedVolumeNodes = {}
    for method_name, enhanced_image in enhanced_volumes.items():
        # Create a new volume node to store the enhanced image data
        enhancedVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        enhancedVolumeNode.SetName(f"Enhanced_th20_{method_name}_{inputVolume.GetName()}")

        # Copy the original volume's transformation information to the enhanced volume
        enhancedVolumeNode.SetOrigin(inputVolume.GetOrigin())
        enhancedVolumeNode.SetSpacing(inputVolume.GetSpacing())

        # Get and set the IJK to RAS transformation matrix
        ijkToRasMatrix = vtk.vtkMatrix4x4()
        inputVolume.GetIJKToRASMatrix(ijkToRasMatrix)  
        enhancedVolumeNode.SetIJKToRASMatrix(ijkToRasMatrix) 

        # Update the volume node with the enhanced image data
        slicer.util.updateVolumeFromArray(enhancedVolumeNode, enhanced_image)
      
        # Store the volume node in the results for later access
        enhancedVolumeNodes[method_name] = enhancedVolumeNode
        
        # Save the volume as NRRD
        output_file = os.path.join(outputDir, f"Filtered_th_20_{method_name}_{inputVolume.GetName()}.nrrd")
        slicer.util.saveNode(enhancedVolumeNode, output_file)
        print(f"Saved {method_name} enhancement as: {output_file}")
    
    return enhancedVolumeNodes

#################################################
## Adding more filters in case it's necessary ###
################################################

def add_more_filter(inputVolume, selected_filters=None, outputDir=None):
    # Default selected_filters to an empty list if None is provided
    if selected_filters is None:
        selected_filters = []

    # Convert input volume to numpy array
    volume_array = slicer.util.arrayFromVolume(inputVolume)

    # Ensure the volume is valid
    if volume_array is None or volume_array.size == 0:
        slicer.util.errorDisplay("Input volume data is empty or invalid.")
        return None

    # Apply filters based on the selected options
    if 'morph_operations' in selected_filters:
        print("Applying morphological operations...")
        volume_array = morph_operations(volume_array)

    if 'canny_edge' in selected_filters:
        print("Applying Canny edge detection...")
        volume_array = canny_edges(volume_array)

    if 'high_pass_sharpening' in selected_filters:
        print("Applying high pass sharpening...")
        volume_array = sharpen_high_pass(volume_array)

    # Create the enhanced volume node
    enhancedVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    enhancedVolumeNode.SetName(f"Enhanced_{inputVolume.GetName()}")

    # Copy the original volume's transformation information to the enhanced volume
    enhancedVolumeNode.SetOrigin(inputVolume.GetOrigin())
    enhancedVolumeNode.SetSpacing(inputVolume.GetSpacing())

    # Get and set the IJK to RAS transformation matrix
    ijkToRasMatrix = vtk.vtkMatrix4x4()
    inputVolume.GetIJKToRASMatrix(ijkToRasMatrix)
    enhancedVolumeNode.SetIJKToRASMatrix(ijkToRasMatrix)

  
    slicer.util.updateVolumeFromArray(enhancedVolumeNode, volume_array)
  
    if outputDir:
        output_file = os.path.join(outputDir, f"Enhanced_more_filters_{inputVolume.GetName()}.nrrd")
        slicer.util.saveNode(enhancedVolumeNode, output_file)
        print(f"Saved enhanced volume with filters as: {output_file}")

    # Return the enhanced volume node
    return enhancedVolumeNode
      
####################

def label_and_get_centroids(mask, spacing):
    """
    Label a binary mask and extract centroids, volumes, and bounding box sizes for each region.

    Args:
        mask (ndarray): Binary mask (2D or 3D).
        spacing (tuple): Voxel spacing in mm (x, y, z).

    Returns:
        pd.DataFrame: DataFrame containing region properties.
    """
    # Label the mask
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)

    region_data = []

    for region in regions:
        label = region.label
        centroid = region.centroid
        # Ensure coordinates are 3D, even if only 2D
        z_coord = centroid[2] if len(centroid) > 2 else 0  # Default to 0 for 2D masks

        # Calculate the physical volume (in mm³) based on the voxel spacing
        region_volume_voxels = region.area
        region_volume_mm3 = region_volume_voxels * (spacing[0] * spacing[1] * spacing[2])

        # Get the bounding box dimensions in voxel units
        minr, minc, minz, maxr, maxc, maxz = region.bbox
        x_size_voxels = maxc - minc
        y_size_voxels = maxr - minr
        z_size_voxels = maxz - minz if len(centroid) > 2 else 1  # Default to 1 for 2D masks

        # Convert bounding box dimensions from voxels to mm
        x_size_mm = x_size_voxels * spacing[0]
        y_size_mm = y_size_voxels * spacing[1]
        z_size_mm = z_size_voxels * spacing[2]

        # Intensity statistics (optional)
        intensity_mean = np.mean(mask[labeled_mask == label])
        intensity_variance = np.var(mask[labeled_mask == label])

        # Store data in a dictionary format
        region_data.append({
            'Label': label,
            'Centroid_X': centroid[0],
            'Centroid_Y': centroid[1],
            'Centroid_Z': z_coord,
            'Volume_Voxels': region_volume_voxels,
            'Volume_mm3': region_volume_mm3,
            'X_Size_Voxels': x_size_voxels,
            'Y_Size_Voxels': y_size_voxels,
            'Z_Size_Voxels': z_size_voxels,
            'X_Size_mm': x_size_mm,
            'Y_Size_mm': y_size_mm,
            'Z_Size_mm': z_size_mm,
            'Intensity_Mean': intensity_mean,
            'Intensity_Variance': intensity_variance
        })
    
    # Create a DataFrame to store the data
    df = pd.DataFrame(region_data)
    
    return df

def save_centroids_to_csv(mask, file_path, inputVolume, enhancedVolumeNode):

    if mask is None or inputVolume is None:
        raise ValueError("Mask and input volume must be provided.")

    spacing = inputVolume.GetSpacing()

    df = label_and_get_centroids(mask, spacing)

    enhancedVolumeNode.SetOrigin(inputVolume.GetOrigin())
    enhancedVolumeNode.SetSpacing(inputVolume.GetSpacing())

    ijkToRasMatrix = vtk.vtkMatrix4x4()
    inputVolume.GetIJKToRASMatrix(ijkToRasMatrix)
    enhancedVolumeNode.SetIJKToRASMatrix(ijkToRasMatrix)

    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

inputVolume = slicer.util.getNode('CTp.3D')  
inputROI = slicer.util.getNode('patient8_mask_2')  # Brain Mask 
# # # # Define the file path to save the CSV
# # file_path = r'C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Enhance_ctp_tests\\P5\\P5_centroids_and_coordinates.csv'

#enhancedVolumeNode = slicer.util.getNode('Enhanced_gamma_4_CTp.3D')

# # volume_array = slicer.util.arrayFromVolume(enhancedVolumeNode)

# # mask_label = np.uint8(volume_array > 0) * 255

# # # # # Save the centroids and coordinates to CSV
# # save_centroids_to_csv(mask_label, file_path, inputVolume, enhancedVolumeNode)



# # # # # # Output directory
outputDir = r'C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Enhance_ctp_tests\\P8'  

# # # # # # # # # # Test the function 
enhancedVolumeNodes = enhance_ctp(inputVolume, inputROI, methods='all', outputDir=outputDir)

# # # # # # # # # # Access the enhanced volume nodes
for method, volume_node in enhancedVolumeNodes.items():
             if volume_node is not None:
                 print(f"Enhanced volume for method '{method}': {volume_node.GetName()}")
             else:
                 print(f"Enhanced volume for method '{method}': No volume node available.")


# vol_hist = slicer.util.getNode('ctp.3D')

# def show_histograms(volume_node, title='Histogram'):
#     # Convert volume node to NumPy array
#     vol_array = slicer.util.arrayFromVolume(volume_node)
    
#     if vol_array is None:
#         raise ValueError("Failed to get voxel data from the volume node.")

#     # Flatten the array and compute the histogram
#     histogram_og, bin_edges = np.histogram(vol_array.flatten(), bins=256, range=(0, 255))
    
#     # Create bin centers
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

#     # Stack data into a 2D array (Slicer expects a table format)
#     histogram_data = np.column_stack((bin_centers, histogram_og))

#     # Plot histogram
#     slicer.util.plot(histogram_data, title=title)

# show_histograms(vol_hist)


#exec(open('C:/Users/rocia/AppData/Local/slicer.org/Slicer 5.6.2/SEEG_module/SEEG_masking/enhance_ctp.py').read())
