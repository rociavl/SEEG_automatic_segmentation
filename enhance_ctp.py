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

# Define enhancement methods
def apply_clahe(roi_volume):
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
    
def denoise_image(roi_volume, strength=10):
        """
        Denoise each slice in the 3D volume using Non-Local Means Denoising.
        
        Parameters:
        - roi_volume: 3D numpy array (CT/MRI volume)
        - strength: Denoising strength (higher means more denoising)
        
        Returns:
        - Denoised 3D volume
        """
        denoised_slices = np.zeros_like(roi_volume)
        for i in range(roi_volume.shape[0]):
            slice_image = roi_volume[i]
            denoised_slices[i] = cv2.fastNlMeansDenoising(slice_image, None, strength, 7, 21)
        return denoised_slices
   
def anisotropic_diffusion(roi_volume, n_iter =1 , k=50, gamma=0.1):
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
    
def laplacian_sharpen(roi_volume, strength=1.5):
        """
        Apply Laplacian edge detection and sharpen the image while keeping the background black.

        Parameters:
        - roi_volume: 3D numpy array (CT/MRI volume)
        - strength: Intensity of sharpening

        Returns:
        - Sharpened 3D volume with preserved black background
        """
        sharpened_slices = np.zeros_like(roi_volume)

        for i in range(roi_volume.shape[0]):
            slice_image = roi_volume[i]

            # Apply Laplacian filter
            laplacian = cv2.Laplacian(slice_image, cv2.CV_64F)

            # Ensure the Laplacian response does not shift the overall intensity
            laplacian = np.clip(laplacian, -255, 255)  # Prevent overflow

            # Subtract laplacian from the original to enhance edges
            sharpened_slice = np.clip(slice_image - strength * laplacian, 0, 255)

            # Ensure background remains black
            sharpened_slice[slice_image == 0] = 0

            sharpened_slices[i] = sharpened_slice.astype(np.uint8)

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

def morph_operations(roi_volume, kernel_size=1, operation="open"):
        """
        Apply morphological opening or closing to each slice.

        Parameters:
        - roi_volume: 3D numpy array (edge-detected volume)
        - kernel_size: Size of the structuring element
        - operation: "open" for noise removal, "close" for edge completion

        Returns:
        - Processed 3D volume
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        processed_slices = np.zeros_like(roi_volume)

        for i in range(roi_volume.shape[0]):
            if operation == "open":
                processed_slices[i] = cv2.morphologyEx(roi_volume[i], cv2.MORPH_OPEN, kernel)
            elif operation == "close":
                processed_slices[i] = cv2.morphologyEx(roi_volume[i], cv2.MORPH_CLOSE, kernel)

        return processed_slices
    
    
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

###
## 
###

def remove_large_objects(binary_image, min_size=500):
    # Use connected components to label objects and remove large ones
    num_labels, labels = cv2.connectedComponents(binary_image.astype(np.uint8))

    # Create an empty image to store the result
    cleaned_image = np.zeros_like(binary_image)

    for label in range(1, num_labels):  # Skip the background label (0)
        # Find the size of the object (number of pixels)
        object_size = np.sum(labels == label)

        if object_size < min_size:
            # If the object is smaller than the minimum size, keep it
            cleaned_image[labels == label] = 255

    return cleaned_image

def generate_contour_mask(brain_mask, dilation_iterations=1):
    # Dilate the brain mask slightly to get the contour region
    dilated_mask = cv2.dilate(brain_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=dilation_iterations)
    
    # Subtract the original brain mask from the dilated mask to get the contour region
    contour_mask = dilated_mask - brain_mask.astype(np.uint8)
    
    return contour_mask

def remove_brain_contour_from_image(binary_image, contour_mask):
    # Set pixels in the contour region to 0
    binary_image[contour_mask == 255] = 0
    return binary_image

def get_watershed_markers(binary, brain_mask=None, contour_mask=None):
    # If inputRoi is provided (brain_mask), remove the brain contour before applying watershed
    if brain_mask is not None and contour_mask is not None:
        cleaned_binary = remove_brain_contour_from_image(binary, contour_mask)
    else:
        cleaned_binary = binary

    # Convert to uint8 (0 and 255)
    cleaned_binary = np.uint8(cleaned_binary > 0) * 255

    # Use connected components to label the markers (ensure input is binary)
    num_labels, markers = cv2.connectedComponents(cleaned_binary.astype(np.uint8))

    # Label markers for watershed using connected components
    markers += 1  # Ensure background is not zero
    markers[cleaned_binary == 0] = 0  # Mark unknown region as zero (exclude contour areas)

    return markers



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

# Define the watershed function to be applied on a slice
def apply_watershed_on_volume(volume_array, brain_mask=None):
    markers = np.zeros_like(volume_array, dtype=np.int32)  # Initialize marker array
    watershed_segmented = np.zeros_like(volume_array, dtype=np.uint8)  # For final segmentation

    print(f"apply_watershed_on_volume - Input volume shape: {volume_array.shape}")

    for i in range(volume_array.shape[0]):  # Iterate over slices
        binary_slice = np.uint8(volume_array[i] > 0) * 255  # Convert to binary
        
        print(f"Slice {i} - Shape: {binary_slice.shape}, Dtype: {binary_slice.dtype}")

        # If brain_mask is provided, use it directly (it's already a NumPy array)
        if brain_mask is not None:
            brain_mask_slice = brain_mask[i]  # Get the corresponding slice
            binary_slice = cv2.bitwise_and(binary_slice, brain_mask_slice)

        marker_slice = get_watershed_markers(binary_slice)  # Get markers

        print(f"Slice {i} - Markers Shape: {marker_slice.shape}, Dtype: {marker_slice.dtype}")

        slice_bgr = cv2.cvtColor(binary_slice, cv2.COLOR_GRAY2BGR)

        # Apply watershed
        cv2.watershed(slice_bgr, marker_slice)

        segmented = np.uint8(marker_slice > 1) * 255  # Ignore background/unknown regions

        markers[i] = marker_slice
        watershed_segmented[i] = segmented
        
        print(f"Slice {i} - Segmented Shape: {segmented.shape}, Dtype: {segmented.dtype}")

    return watershed_segmented


def apply_dbscan_2d(volume_array, eps=5, min_samples=10):
    clustered_volume = np.zeros_like(volume_array, dtype=np.int32)  # Store cluster labels
    cluster_counts = {}

    for slice_idx in range(volume_array.shape[0]):
        slice_data = volume_array[slice_idx]

        # Extract nonzero points
        yx_coords = np.column_stack(np.where(slice_data > 0))

        if len(yx_coords) == 0:
            continue  # Skip empty slices

        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(yx_coords)

        # Convert labels back to 2D
        clustered_slice = np.zeros_like(slice_data, dtype=np.int32)
        for i, (y, x) in enumerate(yx_coords):
            clustered_slice[y, x] = labels[i] + 1  # Shift labels to be non-negative

        # Store results
        clustered_volume[slice_idx] = clustered_slice
        cluster_counts[slice_idx] = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise

    return clustered_volume, cluster_counts


def apply_gmm(image, n_components=3):
    # Extract nonzero pixel values instead of coordinates
    pixel_values = image[image > 0].reshape(-1, 1)  # Ensure it's (num_samples, 1)

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
        return np.zeros_like(image)  # Return blank mask on failure

    return gmm_image

def clean_segmentation(segmented_image, min_size=50):
    return np.uint8(remove_small_objects(segmented_image.astype(bool), min_size=min_size) * 255)

def separate_merged_electrodes(segmented_image, min_distance=5):
    # Perform distance transform to help separate close objects
    dist_transform = cv2.distanceTransform(segmented_image, cv2.DIST_L2, 5)

    # Threshold the distance transform to create a separation mask
    _, separation_mask = cv2.threshold(dist_transform, min_distance, 255, cv2.THRESH_BINARY)

    # Apply morphological opening to further separate electrodes if they are still too close
    kernel = np.ones((3, 3), np.uint8)
    separated = cv2.morphologyEx(separation_mask, cv2.MORPH_OPEN, kernel)
    
    return separated

###    
# Function to enhance the CTP.3D images 
###

def enhance_ctp(inputVolume, inputROI=None, methods = 'all', outputDir=None):
    methods ='all'
    # Convert input volume to numpy array
    volume_array = slicer.util.arrayFromVolume(inputVolume)

    # Ensure the volume is valid
    if volume_array is None or volume_array.size == 0:
        print("Input volume data is empty or invalid.")
        return None

    # Use inputVolume as inputROI if inputROI is not provided
    if inputROI is None:
        inputROI = inputVolume

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

    print(f"Number of components found: {num_features}")
    print(f"Largest component label: {largest_label}")

    
    # Fill Inside the ROI
    
    print("🖌️ Filling inside the ROI...")
    filled_roi = ndimage.binary_fill_holes(largest_component)


    # 4. Apply Morphological Closing for Smoothing
    print("⚙️ Applying morphological closing...")
    struct_elem = morphology.ball(5)  # Structuring element for closing
    closed_roi = morphology.binary_closing(filled_roi, struct_elem)


   # Check if the shapes match
    if closed_roi.shape != volume_array.shape:
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

        closed_roi = resampled_roi_array # Update roi_array

        # Ensure that the shapes match after resampling
        if closed_roi.shape != volume_array.shape:
            print("Resampling failed. Shapes still do not match.")
            print(f"Volume shape: {volume_array.shape}, Resampled ROI shape: {closed_roi.shape}")
            return None

        print("ROI successfully resampled (using SimpleITK).")

    else:
        print("ROI and input volume shapes already match.")


    # Extract the region of interest from the input volume using the binary mask
        
    roi_volume = volume_array * closed_roi

    # Apply selected enhancement methods
    enhanced_volumes = {}
    if methods == 'all':
        enhanced_volumes['gamma_correction'] = gamma_correction(roi_volume, gamma = 10)
        
        # Noise reduction
        enhanced_volumes['bilateral_filter'] = bilateral_filter(enhanced_volumes['gamma_correction'], sigma_color= 50, sigma_space=50)
        enhanced_volumes['sherpened'] = sharpen_high_pass(enhanced_volumes['bilateral_filter'], strenght = 0.5)
        enhanced_volumes['wavelet'] = wavelet_denoise(enhanced_volumes['sherpened'])
        
        enhanced_volumes['clahe'] = apply_clahe(enhanced_volumes['sherpened'])
        enhanced_volumes['laplacian'] = laplacian_sharpen(enhanced_volumes['clahe'])
        ## Morphological operations
        enhanced_volumes['closing'] = morph_operations(enhanced_volumes['laplacian'], operation= 'close')
        
        # Edge boosting?
        enhanced_volumes['high_pass'] = sharpen_high_pass(enhanced_volumes['closing'])
        
        # Enhance contrast 
        enhanced_volumes['gamma_2'] = gamma_correction(enhanced_volumes['high_pass'], gamma = 1.8)
        # Create an elliptical structuring element (adjust size if needed)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Apply white top-hat transformation
        tophat = cv2.morphologyEx(enhanced_volumes['gamma_2'], cv2.MORPH_TOPHAT, kernel)
        enhanced_volumes['tophat'] = cv2.addWeighted(enhanced_volumes['gamma_2'], 1, tophat, 2, 0)


    ## Watershed
    print('Applying watershed...')

    # Convert the 'tophat' enhanced volume to a binary mask
    binary = np.uint8(enhanced_volumes['tophat'] > 0) * 255

    print(f"Binary image shape: {binary.shape}, Dtype: {binary.dtype}")

    # Ensure inputROI is a valid vtkMRMLScalarVolumeNode before converting
    if inputROI is not None and isinstance(inputROI, slicer.vtkMRMLScalarVolumeNode):
        print("Using brain mask for watershed.")
        
        # Convert vtkMRMLScalarVolumeNode to NumPy array
        brain_mask_array = slicer.util.arrayFromVolume(inputROI)
        
        if brain_mask_array is None:
            raise ValueError("Failed to convert inputROI to NumPy array.")

        print(f"Brain mask shape: {brain_mask_array.shape}, Dtype: {brain_mask_array.dtype}")
        
        watershed_segmented = apply_watershed_on_volume(binary, brain_mask=brain_mask_array)
    else:
        print("No valid brain mask provided.")
        watershed_segmented = apply_watershed_on_volume(binary)

    print(f"Watershed result shape: {watershed_segmented.shape}, Dtype: {watershed_segmented.dtype}")

    # Ensure watershed_segmented is a NumPy array before updating the volume node
    if not isinstance(watershed_segmented, np.ndarray):
        raise TypeError("Expected a NumPy array for updateVolumeFromArray, but got a different type.")

    # Create or update the Slicer volume node with the segmented result
    segmented_volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")

    # Ensure updateVolumeFromArray gets a NumPy array, not a vtkMRMLScalarVolumeNode
    slicer.util.updateVolumeFromArray(segmented_volume_node, watershed_segmented)

    # Store the final segmentation in enhanced_volumes
    enhanced_volumes['watershed'] = segmented_volume_node



    # # **Apply DBSCAN Clustering on 2D Slices**
    print("🧩 Applying DBSCAN clustering to 2D slices...")
    clustered_volume, cluster_counts = apply_dbscan_2d(watershed_segmented, eps=7, min_samples=10)
    print(f"Cluster counts per slice: {cluster_counts}")

    enhanced_volumes['dbscan_clusters'] = clustered_volume
   
    #     # Apply GMM on DBSCAN-clustered regions
    print("🧩 Applying GMM for refined segmentation...")
    gmm_segmented = apply_gmm(clustered_volume, n_components=5)

    #     # Clean the final segmentation using small object removal
    cleaned_segmentation = clean_segmentation(gmm_segmented, min_size=500) 
    enhanced_volumes['final_segmentation'] = cleaned_segmentation
    
    # Set output directory (maybe the user doesn't have write access to the default output directory)
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
    
#  # Save and return the final DBSCAN + GMM segmented volume
    final_volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    final_volume_node.SetName(f"DBSCAN_GMM_Segmented_{inputVolume.GetName()}")
    slicer.util.updateVolumeFromArray(final_volume_node, cleaned_segmentation)
    slicer.util.saveNode(final_volume_node, os.path.join(outputDir, "final_segmented_volume.nrrd"))
    print(f"Saved the final segmented volume as: {os.path.join(outputDir, 'final_segmented_volume.nrrd')}")

    return enhancedVolumeNodes

###
## Adding more filters in case it's necessary
###

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
      

# # Load the input volume and ROI
# inputVolume = slicer.util.getNode('CTp.3D')  
# inputROI = slicer.util.getNode('P5_brain_mask')  # Brain Mask 

# # # # Output directory
# outputDir = r'C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Enhance_ctp_tests\\P5'  

# # # # Test the function 
# enhancedVolumeNodes = enhance_ctp(inputVolume, inputROI, methods='all', outputDir=outputDir)

# # # # Access the enhanced volume nodes
# for method, volume_node in enhancedVolumeNodes.items():
#       if volume_node is not None:
#           print(f"Enhanced volume for method '{method}': {volume_node.GetName()}")
#       else:
#           print(f"Enhanced volume for method '{method}': No volume node available.")


#exec(open('C:/Users/rocia/AppData/Local/slicer.org/Slicer 5.6.2/SEEG_module/SEEG_masking/enhance_ctp.py').read())
