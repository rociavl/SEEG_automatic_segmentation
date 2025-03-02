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
from skimage.feature import canny, peak_local_max
from skimage.transform import rescale
from skimage.feature import peak_local_max
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects
import pandas as pd
from skimage.measure import regionprops
from skimage.filters import frangi
from scipy.ndimage import median_filter
import vtk.util.numpy_support as ns
from skimage.morphology import disk
from skimage.filters import median

import logging
from skimage.feature import peak_local_max
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
    """
    Apply a 2D median filter with a circular kernel (in mm²) on each slice independently.

    Args:
    - image_array (numpy.ndarray): 3D image array (Slices, Height, Width).
    - kernel_size_mm2 (float): Area of the circular kernel in mm².
    - spacing (tuple): Voxel spacing (dx, dy) in mm.

    Returns:
    - numpy.ndarray: 3D filtered image (processed slice-by-slice).
    """
    kernel_radius_pixels = int(round(np.sqrt(kernel_size_mm2 / np.pi) / min(spacing[:2])))
    
    selem = disk(kernel_radius_pixels)

    filtered_slices = np.array([median(slice_, selem) for slice_ in image_array])

    print(f"✅ 2D Median filtering complete. Kernel radius: {kernel_radius_pixels} pixels")
    
    return filtered_slices

def apply_median_filter(image_array, kernel_size=3):
    return median_filter(image_array, size=kernel_size)

def resample_to_isotropic(image, spacing=(1.0, 1.0, 1.0)):
    """Resample image to isotropic spacing."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled = resampler.Execute(image)
    return resampled


def threshold_metal_voxels(image_array, percentile=99.5):
    threshold_value = np.percentile(image_array, percentile)
    return image_array > threshold_value 


def thresholding_volume_histogram(volume, threshold=30):
        """
        Apply thresholding to a 3D volume.
        
        Parameters:
        - volume: 3D numpy array (CT/MRI volume)
        - threshold: Intensity threshold for binarization
        
        Returns:
        - Binarized 3D volume
        """
        binary_volume = np.uint8(volume > threshold) * 255
        return binary_volume

def apply_clahe(roi_volume):
        '''
        clipLimit: limits the contrast enhancement 
        tileGridSize: control the contrast enhancement
        '''
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
        enhanced_slices = np.zeros_like(roi_volume)
        roi_volume_scaled = np.uint8(np.clip(roi_volume, 0, 255))  # Convert to 8-bit
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
        binary_local_uint8 = np.uint8(binary_local * 255)  # Convert to uint8 (0 or 255)
        return binary_local_uint8
    
    
def local_threshold(roi_volume):
        block_size = 51
        local_thresh = filters.threshold_local(roi_volume, block_size, offset=10)
        binary_local = roi_volume > local_thresh
        binary_local_uint8 = np.uint8(binary_local * 255)  # Convert to uint8 (0 or 255)
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
    for i in range(volume.shape[0]):  # Iterate through the slices (z-axis)
        slice_image = volume[i]  # Extract the i-th 2D slice
        denoised_slice = denoise_nl_means(slice_image, patch_size=patch_size, patch_distance=patch_distance, h=h, multichannel=False)
        denoised_volume[i] = denoised_slice  # Assign the denoised slice back to the volume

    return denoised_volume
   
def anisotropic_diffusion(roi_volume, n_iter=1, k=50, gamma=0.1):
        """
        Apply anisotropic diffusion (total variation denoising) to each slice in the 3D volume.

        Parameters:
        - roi_volume: 3D numpy array (CT/MRI volume)
        - n_iter: Number of iterations for the diffusion process
        - k: Edge-stopping parameter
        - gamma: Step size for the diffusion process

        Returns:
        - Denoised 3D volume
        """
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

    # 🔹 Step 1: Threshold the image to detect structures (modify percentile if needed)
    binary_mask = image > np.percentile(image, 60)

    # 🔹 Step 2: Label connected components (DO NOT use return_num=True)
    labeled = label(binary_mask, connectivity=1)  # Ensure 3D connectivity if needed

    # 🔹 Step 3: Remove large objects based on region area
    for region in regionprops(labeled):
        if region.area > size_threshold:
            coords = region.coords  # Get the coordinates of the object
            image[coords[:, 0], coords[:, 1], coords[:, 2]] = np.median(image)  # Replace with median intensity

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
            # Apply Laplacian filter to detect edges
            laplacian = cv2.Laplacian(slice_image_float, cv2.CV_32F)

            # Ensure the Laplacian response does not shift the overall intensity
            laplacian = np.clip(laplacian, -255, 255)  # Prevent overflow

            # Subtract laplacian from the original to enhance edges
            slice_image_float = np.clip(slice_image_float - strength * laplacian, 0, 255)

        # Ensure background remains black
        slice_image_float[slice_image == 0] = 0

        # Convert back to uint8 for the final result
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
    """
    Apply morphological dilation or erosion to an image.

    Parameters:
    - image: Input image (numpy array)
    - operation: 'dilate' or 'erode'
    - kernel_size: Size of the structuring element (default: 3)
    - iterations: Number of times to apply the operation (default: 1)

    Returns:
    - Processed image with dilation or erosion applied
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if operation == 'dilate':
     processed_image = cv2.dilate(image, kernel, iterations=iterations)
    
    elif operation == 'erode':
        processed_image = cv2.erode(image, kernel, iterations=iterations)
   
    else:
        raise ValueError("Invalid operation. Use 'dilate' or 'erode'.")
    
    return processed_image

def morph_operations(roi_volume, kernel_size=1, operation="open", iterations=1, kernel_shape="square"):
    """
    Apply morphological opening or closing to each slice with custom kernel shape.

    Parameters:
    - roi_volume: 3D numpy array (edge-detected volume)
    - kernel_size: Size of the structuring element
    - operation: "open" for noise removal, "close" for edge completion
    - iterations: Number of times the operation is applied
    - kernel_shape: "square", "cross" or "ellipse"

    Returns:
    - Processed 3D volume
    """
    # Define kernel shapes
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
    for i in range(volume.shape[0]):  # Assuming (Z, Y, X) ordering
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
def log_transform(roi_volume, c=5):
        # Apply a log transformation to enhance the brightness
        roi_volume_log = c * np.log(1 + roi_volume)
        
        # Normalize the values to [0, 255] range
        roi_volume_log = np.uint8(np.clip(roi_volume_log / np.max(roi_volume_log) * 255, 0, 255))
        
        return roi_volume_log

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Binarization
    return binary

def sobel_edge_detection(roi_volume):
    """
    Apply Sobel edge detection to each slice of a 3D volume.

    Parameters:
    - roi_volume: 3D numpy array (CT/MRI volume)

    Returns:
    - Edge-detected 3D volume using Sobel operator
    """
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

def enhanced_difference_of_gaussians(roi_volume, sigma1=1, sigma2=3, kernel_size=5, threshold=10):
    """
    Apply an enhanced Difference of Gaussians (DoG) filter to each slice of a 3D volume,
    with post-processing to highlight small bright objects like electrodes.

    Parameters:
    - roi_volume: 3D numpy array (e.g., CT/MRI volume)
    - sigma1: Standard deviation for the first Gaussian blur (smaller)
    - sigma2: Standard deviation for the second Gaussian blur (larger)
    - kernel_size: Size of the Gaussian kernel (must be odd)
    - threshold: Intensity threshold to remove weak edges

    Returns:
    - Processed 3D volume after DoG filtering
    """
    processed_slices = np.zeros_like(roi_volume, dtype=np.float32)

    for i in range(roi_volume.shape[0]):
        slice_image = roi_volume[i].astype(np.float32)

        # Apply Gaussian blurs
        blur1 = cv2.GaussianBlur(slice_image, (kernel_size, kernel_size), sigma1)
        blur2 = cv2.GaussianBlur(slice_image, (kernel_size, kernel_size), sigma2)

        # Compute Difference of Gaussians
        dog = blur1 - blur2

        # Apply absolute value to enhance contrast
        dog = np.abs(dog)

        # Normalize result
        dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)

        # Apply a binary threshold to remove weak responses
        _, binary_dog = cv2.threshold(dog, threshold, 255, cv2.THRESH_BINARY)

        # Store result
        processed_slices[i] = binary_dog

    return processed_slices.astype(np.uint8)

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
    # Label the connected components (returns labeled image and number of features)
    labeled_image = measure.label(segmented_image, connectivity=1)  # Only keep the labeled image
    
    # Create a mask to keep small objects
    mask = np.zeros_like(segmented_image, dtype=bool)
    
    # Iterate through each labeled component and check its size
    for region in measure.regionprops(labeled_image):
        # If the region is small enough, keep it
        if region.area <= size_threshold:
            mask[labeled_image == region.label] = True

    # Apply the mask to the original image to remove large objects
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

    # Apply Non-Local Means Denoising to each slice
    for i in range(volume.shape[0]):  # Iterate through the slices (z-axis)
        slice_image = volume[i]  # Extract the i-th 2D slice
        denoised_slice = denoise_nl_means(slice_image, patch_size=patch_size, patch_distance=patch_distance, h=h)
        denoised_volume[i] = denoised_slice  # Assign the denoised slice back to the volume

    return denoised_volume

def vtk_to_numpy(vtk_image_node):
    """Convert vtkMRMLScalarVolumeNode to numpy array."""
    
    # Convert the volume node directly into a NumPy array using slicer.util.arrayFromVolume
    np_array = slicer.util.arrayFromVolume(vtk_image_node)
    
    return np_array
# Update a vtkMRMLScalarVolumeNode with a NumPy array (assuming you have a NumPy array to update it)
def update_vtk_volume_from_numpy(np_array, vtk_image_node):
    """Update vtkMRMLScalarVolumeNode from a NumPy array."""
    
    # Directly update the volume using Slicer's updateVolumeFromArray method
    slicer.util.updateVolumeFromArray(vtk_image_node, np_array)  # Update volume with the numpy array

    # Ensure that the volume is properly updated in the scene
    slicer.app.processEvents()

def generate_contour_mask(brain_mask, dilation_iterations=1):
    """
    Generates a contour mask by dilating the brain mask and subtracting the original.
    
    Parameters:
      - brain_mask: Binary mask of the brain (NumPy array). If not, it will be converted.
      - dilation_iterations: Number of dilation iterations.
    
    Returns:
      - contour_mask: The computed contour mask.
    """

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



def apply_gmm(image, n_components=3):
   
    pixel_values = image[image > 0].reshape(-1, 1)  

    if pixel_values.shape[0] == 0:
        return np.zeros_like(image)  # If empty, return blank mask

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
        gmm_labels = gmm.fit_predict(pixel_values)  # Shape (num_points,)

        # Create an output image
        gmm_image = np.zeros_like(image)

        # Get indices where image > 0
        indices = np.where(image > 0)

        # Map the GMM labels back to the correct positions
        gmm_image[indices] = gmm_labels + 1  # Shift labels to be non-negative

    except Exception as e:
        print(f"⚠️ GMM error: {e}")
        return np.zeros_like(image)  

    return gmm_image
def apply_snakes_tiny(volume):
    """
    Apply active contour model (snakes) to each slice of a 3D volume.

    Parameters:
        volume (numpy.ndarray): 3D binary image (with electrodes or structures).
    
    Returns:
        numpy.ndarray: 3D image with applied contours on each slice.
    """
    # Initialize an empty volume for the final result
    final_contours = np.zeros_like(volume, dtype=np.uint8)

    # Iterate over each slice in the 3D volume
    for i in range(volume.shape[0]):  # Iterate through each slice
        slice_2d = volume[i]  # Extract 2D slice
        
        # Apply the canny edge detection (to detect edges)
        edges = feature.canny(slice_2d)  
        
        # Define an initial contour (e.g., an ellipse in the center of the image)
        s = np.zeros(slice_2d.shape)
        center = (slice_2d.shape[0] // 2, slice_2d.shape[1] // 2)
        
        # Use skimage.draw.ellipse to draw an ellipse in the center
        rr, cc = draw.ellipse(center[0], center[1], 20, 20)
        s[rr, cc] = 1  # Create an initial contour
        
        # Apply active contour (snakes)
        snake = active_contour(edges, s, alpha=0.015, beta=10, gamma=0.001, max_num_iter=250)

        # Convert the snake result back to a binary contour
        contour_mask = np.zeros_like(slice_2d)
        contour_mask[tuple(snake.T.astype(int))] = 1  # Set the contour points to 1

        # Store the result back into the final_contours volume
        final_contours[i] = contour_mask
    
    return final_contours


def get_auto_seeds(binary):
    """
    Automatically generate seeds based on the distance transform.
    Avoids erosion and ensures small electrodes are detected.
    """
    # Compute distance transform
    distance = distance_transform_edt(binary)

    # Detect local maxima (potential seeds)
    local_max = peak_local_max(distance, indices=False, min_distance=1, labels=binary)

    # Label detected seeds
    seeds, num_seeds = label(local_max)

    print(f"Generated {num_seeds} automatic seeds.")
    return seeds

def refine_labels(labels, min_size=5):
    """
    Removes falsely merged labels by filtering based on region properties.
    """
    refined = np.zeros_like(labels)
    for region in regionprops(labels):
        if region.area >= min_size:  # Keep only objects above threshold
            refined[labels == region.label] = region.label
    return refined

def region_growing(binary, tolerance=3, min_size=5):
    """
    Adaptive region growing for tiny structures like electrodes.

    Parameters:
        binary (numpy.ndarray): Binary input image.
        tolerance (int): Controls how much the region can grow.
        min_size (int): Minimum size of detected objects.

    Returns:
        numpy.ndarray: Segmented and labeled mask.
    """
    # Generate automatic seed points
    seeds = get_auto_seeds(binary)
    
    # Initialize output segmentation
    segmented = np.zeros_like(binary, dtype=np.uint8)
    
    # Perform growth from each seed
    for label_id in range(1, seeds.max() + 1):
        mask = seeds == label_id
        region = binary.copy()

        # Region growing: Expand mask based on tolerance
        region_grown = (distance_transform_edt(mask) < tolerance) & binary

        segmented[region_grown] = label_id  # Assign label

    # Remove merged artifacts
    segmented = refine_labels(segmented, min_size)

    return segmented

def separate_by_erosion_and_closing(mask, kernel_size_mm2=0.5, spacing=(1.0, 1.0), erosion_iterations=1, closing_iterations=1):
    """
    Apply gentle erosion and closing to shrink mask while keeping electrodes intact.

    Parameters:
    - mask (numpy.ndarray): Binary mask (electrodes = 255).
    - kernel_size_mm2 (float): Area of the structuring element in mm².
    - spacing (tuple): Voxel spacing (dx, dy) in mm.
    - erosion_iterations (int): Number of erosion iterations.
    - closing_iterations (int): Number of closing iterations.

    Returns:
    - numpy.ndarray: Mask after erosion and closing.
    """
    
    # Ensure binary mask (0 and 255)
    mask = np.uint8(mask > 0) * 255  

    # Compute kernel size in pixels
    kernel_radius_pixels_x = int(round(np.sqrt(kernel_size_mm2 / np.pi) / spacing[0]))
    kernel_radius_pixels_y = int(round(np.sqrt(kernel_size_mm2 / np.pi) / spacing[1]))

    # Ensure kernel size is at least 3x3 pixels
    kernel_size_pixels = (max(3, kernel_radius_pixels_x * 2 + 1), 
                          max(3, kernel_radius_pixels_y * 2 + 1))

    # Use a small circular kernel for gentle erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size_pixels)

    # Apply gentle erosion
    eroded_mask = cv2.erode(mask, kernel, iterations=erosion_iterations)

    # Apply closing to restore small details that may have been lost during erosion
    closed_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)

    print(f"✅ Erosion and closing applied with kernel size {kernel_size_pixels} pixels")
    
    return closed_mask

###########################################    
# Function to enhance the CTP.3D images ###
###############################################

def enhance_ctp(inputVolume, inputROI=None, methods = 'all', outputDir=None):
    methods ='all'
    # Convert input volume to numpy array
    volume_array = slicer.util.arrayFromVolume(inputVolume)

    # Ensure the volume is valid
    if volume_array is None or volume_array.size == 0:
        print("Input volume data is empty or invalid.")
        return None

    # Use inputVolume as inputROI if inputROI is not provided
    if inputROI is not None:

        # Convert ROI (binary mask) to numpy array
        roi_array = slicer.util.arrayFromVolume(inputROI)
        

        # Ensure ROI is binary (mask), converting to 0s and 1s
        roi_array = np.uint8(roi_array > 0)

        print(f"Shape of input volume: {volume_array.shape}")  
        print(f"Shape of ROI mask: {roi_array.shape}")
        
        # Find connected models 
        print('Finding connected components')
        labeled_array, num_features = ndimage.label(roi_array)

        # Selecting the largest component

        print("📏 Selecting the largest component...")
        sizes = ndimage.sum(roi_array, labeled_array, range(num_features + 1))
        largest_label = np.argmax(sizes[1:]) + 1  # Ignore background (label 0)
        largest_component = np.uint8(labeled_array == largest_label)

        
        # Fill Inside the ROI
        
        print("Filling inside the ROI...")
        filled_roi = ndimage.binary_fill_holes(largest_component)


        # Apply Morphological Closing for Smoothing
        print("Applying morphological closing...")
        struct_elem = morphology.ball(5)  
        closed_roi = morphology.binary_closing(filled_roi, struct_elem)

        print('Applying erosion to the brain mask...')
        erode_roi = morphology.binary_opening(closed_roi, morphology.ball(3))  

        final_roi = erode_roi


    # Check if the shapes match
        if final_roi.shape != volume_array.shape:
            print("ROI and input volume shapes do not match. Resampling ROI to match input volume (using SimpleITK)...")

            # Get the spacing, origin, and dimensions of the input volume
            reference_spacing = inputVolume.GetSpacing()
            reference_origin = inputVolume.GetOrigin()
            reference_size = volume_array.shape  # (x, y, z) dimensions

            print(f"Reference Spacing: {reference_spacing}")
            print(f"Reference Origin: {reference_origin}")
            print(f"Reference Dimensions: {reference_size}")

            # Get the ROI data as a NumPy array
            roi_array = slicer.util.arrayFromVolume(inputROI)

            # Transpose the ROI array *before* converting to SimpleITK
            roi_array = np.transpose(roi_array)

            # Convert ROI array to SimpleITK image
            roi_image_sitk = sitk.GetImageFromArray(roi_array)
            roi_image_sitk.SetSpacing(inputROI.GetSpacing())
            roi_image_sitk.SetOrigin(inputROI.GetOrigin())

            # Create SimpleITK image for the reference volume (for geometry)
            reference_image_sitk = sitk.Image(reference_size, sitk.sitkUInt8)  # Create an empty image
            reference_image_sitk.SetSpacing(reference_spacing)
            reference_image_sitk.SetOrigin(reference_origin)

            # Resample the ROI using SimpleITK
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(reference_image_sitk)  # Match geometry
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Preserve binary values
            resampler.SetOutputPixelType(sitk.sitkUInt8)
            resampled_roi_image_sitk = resampler.Execute(roi_image_sitk)

            # Convert the resampled SimpleITK image back to a NumPy array
            resampled_roi_array = sitk.GetArrayFromImage(resampled_roi_image_sitk)

            # Transpose the array *after* resampling
            resampled_roi_array = np.transpose(resampled_roi_array)

            resampled_roi_array = np.uint8(resampled_roi_array > 0) # Ensure binary

            print(f"Resampled ROI shape: {resampled_roi_array.shape}")

            final_roi = resampled_roi_array 

            # Ensure that the shapes match after resampling
            if final_roi.shape != volume_array.shape:
                print("Resampling failed. Shapes still do not match.")
                print(f"Volume shape: {volume_array.shape}, Resampled ROI shape: {final_roi.shape}")
                return None

            print("ROI successfully resampled: ;).")

        else:
            print("ROI and input volume shapes already match.")


        # Extract the region of interest from the input volume using the binary mask
        dimmed_mask = final_roi * 0.1  
        roi_volume = volume_array* dimmed_mask

    else:
         roi_volume = volume_array
    # Apply selected enhancement methods
    enhanced_volumes = {}
    if methods == 'all':
        
        ### Only CTP ###
        enhanced_volumes['volume_array'] = volume_array

        enhanced_volumes['thre_volume_og'] = (volume_array > 1200).astype(np.uint8)
        enhanced_volumes['gaussian_volume_og'] = gaussian(enhanced_volumes['thre_volume_og'], sigma=0.5)
        enhanced_volumes['gamma_volume_og_2'] = gamma_correction(enhanced_volumes['gaussian_volume_og'], gamma=2)
        enhanced_volumes['thresholded_ctp_per_og'] = threshold_metal_voxels_slice_by_slice(enhanced_volumes['gamma_volume_og_2'], percentile= 99.5)

        # enhanced_volumes['median_voxel_ctp'] = apply_median_filter_2d(enhanced_volumes['thresholded_ctp_per_og'], kernel_size_mm2=0.4)
        # enhanced_volumes['separate_erosion_ctp_og'] = separate_by_erosion(enhanced_volumes['median_voxel_ctp'], kernel_size_mm2=0.6, spacing=(1.0, 1.0), iterations=5)
        # enhanced_volumes['thresholded_ctp_per_og_2'] = threshold_metal_voxels_slice_by_slice(enhanced_volumes['separate_erosion_ctp_og'], percentile= 40)
        # enhanced_volumes['closing_ctp_og'] = morph_operations(enhanced_volumes['thresholded_ctp_per_og_2'], iterations=6, kernel_shape= 'cross', operation = 'close')

        # enhanced_volumes['sobel_ctp'] = sobel_edge_detection(enhanced_volumes['closing_ctp_og'])

        # enhanced_volumes['gamma_ctp'] = gamma_correction(enhanced_volumes['gaussian_only_ctp'], gamma= 10)
        # enhanced_volumes['gaussian_ctp_2'] = gaussian(enhanced_volumes['gamma_ctp'], sigma = 0.5)
        # enhanced_volumes['gamma_ctp_2'] = gamma_correction(enhanced_volumes['gaussian_ctp_2'], gamma= 2)
        # enhanced_volumes['MASK_LABEL_ctp'] = np.uint8(enhanced_volumes['gamma_ctp_2'] > 0) * 255

        ### Only ROI ###
        enhanced_volumes['roi_volume'] = roi_volume
        
        enhanced_volumes['thresholded_ctp_per_roi'] = threshold_metal_voxels_slice_by_slice(roi_volume, percentile= 99.9)



       ### ROI gamma###
        enhanced_volumes['gaussian_volume_roi'] = gaussian(roi_volume, sigma=0.3)
        enhanced_volumes['gamma_correction'] = gamma_correction(enhanced_volumes['gaussian_volume_roi'] , gamma = 3)
        enhanced_volumes['denoised_roi'] = denoise_2d_slices(enhanced_volumes['gaussian_volume_roi'], patch_size=6, patch_distance=6, h=0.4)
        enhanced_volumes['thresholded_denoised_roi'] = threshold_metal_voxels_slice_by_slice(enhanced_volumes['denoised_roi'], percentile= 99.9)
        enhanced_volumes['sharpened'] = sharpen_high_pass(enhanced_volumes['gamma_correction'], strenght = 1)
        enhanced_volumes['erode'] = morphological_operation(enhanced_volumes['sharpened'], operation='erode', kernel_size=1)
        enhanced_volumes['gaussian_2'] = gaussian(enhanced_volumes['erode'], sigma= 0.2)
        enhanced_volumes['gamma_2'] = gamma_correction(enhanced_volumes['gaussian_2'], gamma= 1.2)
        # enhanced_volumes['opening'] = morph_operations(enhanced_volumes['gamma_2'], iterations=2, kernel_shape= 'cross')
        # enhanced_volumes['closing'] = morph_operations(enhanced_volumes['opening'], operation= 'close')
        # #enhanced_volumes['wo_large_objects'] = remove_large_objects(enhanced_volumes['closing'], size_threshold= 9000)
        # enhanced_volumes['erode_2'] = morphological_operation(enhanced_volumes['closing'], operation='erode', kernel_size=1)
        # # Create an elliptical structuring element 
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        # # Apply white top-hat transformation
        # tophat = cv2.morphologyEx(enhanced_volumes['erode'], cv2.MORPH_TOPHAT, kernel)
        # enhanced_volumes['tophat'] = cv2.addWeighted(enhanced_volumes['erode_2'], 1, tophat, 2, 0)
        # enhanced_volumes['gamma_3'] = gamma_correction(enhanced_volumes['tophat'], gamma= 2)  
        enhanced_volumes['gaussian_3'] = gaussian(enhanced_volumes['gamma_2'], sigma = 0.3)
        enhanced_volumes['gamma_4'] = gamma_correction(enhanced_volumes['gaussian_3'], gamma= 3)
        enhanced_volumes['MASK_LABEL'] = threshold_metal_voxels_slice_by_slice(enhanced_volumes['gamma_4'] , percentile= 99.8)


        

        # enhanced_volumes['gamma_correction'] = gamma_correction(enhanced_volumes['closing_roi'], gamma = 5)
        # enhanced_volumes['sharpened'] = sharpen_high_pass(enhanced_volumes['gamma_correction'], strenght = 1)
        # enhanced_volumes['erode'] = morphological_operation(enhanced_volumes['sharpened'], operation='erode', kernel_size=1)
        # enhanced_volumes['gaussian_2'] = gaussian(enhanced_volumes['erode'], sigma= 0.2)
        # enhanced_volumes['gamma_2'] = gamma_correction(enhanced_volumes['gaussian_2'], gamma= 1.2)


        # enhanced_volumes['opening'] = morph_operations(enhanced_volumes['gamma_2'], iterations=2, kernel_shape= 'cross')
        # enhanced_volumes['closing'] = morph_operations(enhanced_volumes['opening'], operation= 'close')
        # #enhanced_volumes['wo_large_objects'] = remove_large_objects(enhanced_volumes['closing'], size_threshold= 9000)
        # enhanced_volumes['erode_2'] = morphological_operation(enhanced_volumes['closing'], operation='erode', kernel_size=1)
        # # Create an elliptical structuring element 
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        # # Apply white top-hat transformation
        # tophat = cv2.morphologyEx(enhanced_volumes['erode'], cv2.MORPH_TOPHAT, kernel)
        # enhanced_volumes['tophat'] = cv2.addWeighted(enhanced_volumes['erode_2'], 1, tophat, 2, 0)
        # enhanced_volumes['gamma_3'] = gamma_correction(enhanced_volumes['tophat'], gamma= 2)  
        # enhanced_volumes['gaussian_3'] = gaussian(enhanced_volumes['gamma_3'], sigma = 0.3)
        # enhanced_volumes['gamma_4'] = gamma_correction(enhanced_volumes['gaussian_3'], gamma= 3)
        # enhanced_volumes['MASK_LABEL'] = np.uint8(enhanced_volumes['gamma_4'] > 0) * 255


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
        enhancedVolumeNode.SetName(f"Enhanced_{method_name}_{inputVolume.GetName()}")

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
        output_file = os.path.join(outputDir, f"Filtered_{method_name}_{inputVolume.GetName()}.nrrd")
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
    """
    Save centroids and region properties to a CSV file.

    Args:
        mask (ndarray): Binary mask (2D or 3D).
        file_path (str): Path to save the CSV file.
        inputVolume: Input volume node (for spacing and transformation).
        enhancedVolumeNode: Enhanced volume node (for setting transformation).
    """
    # Validate inputs
    if mask is None or inputVolume is None:
        raise ValueError("Mask and input volume must be provided.")

    # Get the voxel spacing
    spacing = inputVolume.GetSpacing()

    # Get the labeled regions and their centroids, volumes, and bounding box sizes
    df = label_and_get_centroids(mask, spacing)

    # Copy the original volume's transformation information to the enhanced volume
    enhancedVolumeNode.SetOrigin(inputVolume.GetOrigin())
    enhancedVolumeNode.SetSpacing(inputVolume.GetSpacing())

    # Get and set the IJK to RAS transformation matrix
    ijkToRasMatrix = vtk.vtkMatrix4x4()
    inputVolume.GetIJKToRASMatrix(ijkToRasMatrix)
    enhancedVolumeNode.SetIJKToRASMatrix(ijkToRasMatrix)

    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")


# # Load the input volume and ROI
inputVolume = slicer.util.getNode('ctp.3D')  
inputROI = slicer.util.getNode('P1_brain_mask_good_2')  # Brain Mask 
# # # Define the file path to save the CSV
# file_path = r'C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Enhance_ctp_tests\\P5\\P5_centroids_and_coordinates.csv'

# enhancedVolumeNode = slicer.util.getNode('Enhanced_gamma_4_CTp.3D')

# volume_array = slicer.util.arrayFromVolume(enhancedVolumeNode)

# mask_label = np.uint8(volume_array > 0) * 255

# # # # Save the centroids and coordinates to CSV
# save_centroids_to_csv(mask_label, file_path, inputVolume, enhancedVolumeNode)



# # # # # Output directory
outputDir = r'C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Enhance_ctp_tests\\P1'  

# # # # # # # # # Test the function 
enhancedVolumeNodes = enhance_ctp(inputVolume, inputROI, methods='all', outputDir=outputDir)

# # # # # # # # # Access the enhanced volume nodes
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
