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

logging.basicConfig(level=logging.DEBUG)

###################################
### Image processing filters ###
##################################


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
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
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
    denoised_volume = np.zeros_like(volume)
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

def sharpen_fourier(roi_volume, radius=30, strength=0.5):
    sharpened_slices = np.zeros_like(roi_volume, dtype=np.float32)
    for i in range(roi_volume.shape[0]):  
        slice_image = roi_volume[i]
        f = np.fft.fft2(slice_image)
        fshift = np.fft.fftshift(f)
        rows, cols = slice_image.shape
        crow, ccol = rows // 2 , cols // 2
        mask = np.ones((rows, cols), np.uint8)
        mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 0  
        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        sharpened_slice = np.clip(slice_image + strength * img_back, 0, 255)
        sharpened_slices[i] = sharpened_slice
    return sharpened_slices.astype(np.uint8)
    
def laplacian_sharpen(roi_volume, strength=1.5, iterations=3):
    sharpened_slices = np.zeros_like(roi_volume)
    for i in range(roi_volume.shape[0]):
        slice_image = roi_volume[i]
        slice_image_float = slice_image.astype(np.float32)
        for _ in range(iterations):
            laplacian = cv2.Laplacian(slice_image_float, cv2.CV_32F)
            laplacian = np.clip(laplacian, -255, 255)  
            slice_image_float = np.clip(slice_image_float - strength * laplacian, 0, 255)
        slice_image_float[slice_image == 0] = 0
        sharpened_slices[i] = np.clip(slice_image_float, 0, 255).astype(np.uint8)
    return sharpened_slices


def adaptive_binarization(roi_volume, block_size=11, C=2, edge_strength=0.5):
        binarized_slices = np.zeros_like(roi_volume)
        for i in range(roi_volume.shape[0]):
            slice_image = roi_volume[i]
            binarized = cv2.adaptiveThreshold(slice_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
            edges = cv2.Canny(slice_image, 50, 150)
            combined = cv2.addWeighted(binarized, 1 - edge_strength, edges, edge_strength, 0)
            binarized_slices[i] = combined
        return binarized_slices

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
    enhanced_volume = np.zeros_like(volume, dtype=np.float32)
    for i in range(volume.shape[0]):  
        enhanced_volume[i] = frangi(volume[i], scale_range=(1, 5), scale_step=2)
    return enhanced_volume   
    
def contrast_stretching(roi_volume, lower_percentile=2, upper_percentile=98):
        stretched_slices = np.zeros_like(roi_volume, dtype=np.uint8)
        for i in range(roi_volume.shape[0]):
            slice_image = roi_volume[i]
            p_low, p_high = np.percentile(slice_image, (lower_percentile, upper_percentile))
            stretched_slices[i] = exposure.rescale_intensity(slice_image, in_range=(p_low, p_high), out_range=(0, 255)).astype(np.uint8)

        return stretched_slices
        
def canny_edges(roi_volume, sigma=1.0, low_threshold=50, high_threshold=150):
        edge_slices = np.zeros_like(roi_volume)
        for i in range(roi_volume.shape[0]):
            slice_image = roi_volume[i]
            blurred_image = cv2.GaussianBlur(slice_image, (5, 5), sigma)
            edge_slices[i] = cv2.Canny(blurred_image, low_threshold, high_threshold)
        return edge_slices

def log_transform_slices(roi_volume, c=5, sigma=0.4):
    roi_volume_log = np.zeros_like(roi_volume, dtype=np.float32) 
    for i in range(roi_volume.shape[0]):  
        slice_data = roi_volume[i].astype(np.float32)  
        slice_data_smoothed = gaussian_filter(slice_data, sigma=sigma)  
        slice_data_log = c * np.log1p(slice_data_smoothed)  
        slice_data_edges = laplace(slice_data_log)  
        slice_data_enhanced = slice_data_log + 0.5 * slice_data_edges  
        slice_data_enhanced = np.clip(slice_data_enhanced / np.max(slice_data_enhanced) * 255, 0, 255)
        roi_volume_log[i] = slice_data_enhanced  
    return roi_volume_log.astype(np.uint8)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    return binary

def sobel_edge_detection(roi_volume):
    sobel_slices = np.zeros_like(roi_volume)
    for i in range(roi_volume.shape[0]):
        slice_image = roi_volume[i]
        sobel_x = cv2.Sobel(slice_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(slice_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
        sobel_edge_detected = np.uint8(np.clip(sobel_magnitude, 0, 255))
        sobel_slices[i] = sobel_edge_detected
    return sobel_slices

def wavelet_nlm_denoise(roi_volume,
                         wavelet='db1',
                         level=1,
                         patch_size=3,
                         patch_distance=5,
                         weight=0.05):

    denoised_volume = np.zeros_like(roi_volume)
    for z in range(roi_volume.shape[0]):
        slice_2d = roi_volume[z, :, :]

        wavelet_denoised_slice = wavelet_denoise(slice_2d, wavelet=wavelet, level=level)
        nlm_denoised_slice = restoration.denoise_nl_means(wavelet_denoised_slice,
                                                            patch_size=patch_size,
                                                            patch_distance=patch_distance,
                                                            h=weight * wavelet_denoised_slice.std())
        denoised_volume[z, :, :] = nlm_denoised_slice

    return denoised_volume

def wavelet_denoise(slice_image, wavelet='db1', level=1):
    coeffs = pywt.wavedec2(slice_image, wavelet, level=level)
    coeffs_thresholded = list(coeffs)
    coeffs_thresholded[1:] = [tuple([pywt.threshold(c, value=0.2, mode='soft') for c in coeffs_level]) for coeffs_level in coeffs[1:]]
    denoised_slice = pywt.waverec2(coeffs_thresholded, wavelet)
    return denoised_slice

###############################
## masks and improvements ######
#################################

def remove_large_objects(segmented_image, size_threshold):
    labeled_image = measure.label(segmented_image, connectivity=1)  
    mask = np.zeros_like(segmented_image, dtype=bool)
    for region in measure.regionprops(labeled_image):
        # If the region is small enough, keep it
        if region.area <= size_threshold:
            mask[labeled_image == region.label] = True
    filtered_image = segmented_image * mask
    return filtered_image

def denoise_2d_slices(volume, patch_size=2, patch_distance=2, h=0.1):
    denoised_volume = np.zeros_like(volume)
    for i in range(volume.shape[0]):  
        slice_image = volume[i]  
        denoised_slice = denoise_nl_means(slice_image, patch_size=patch_size, patch_distance=patch_distance, h=h)
        denoised_volume[i] = denoised_slice  
    return denoised_volume

def vtk_to_numpy(vtk_image_node):
    np_array = slicer.util.arrayFromVolume(vtk_image_node)
    return np_array

def update_vtk_volume_from_numpy(np_array, vtk_image_node):
    slicer.util.updateVolumeFromArray(vtk_image_node, np_array)  
    slicer.app.processEvents()

def generate_contour_mask(brain_mask, dilation_iterations=1):
    if not isinstance(brain_mask, np.ndarray):
        brain_mask = slicer.util.arrayFromVolume(brain_mask)
    brain_mask = np.uint8(brain_mask > 0) * 255
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(brain_mask, kernel, iterations=dilation_iterations)
    contour_mask = cv2.subtract(dilated_mask, brain_mask)
    return contour_mask

def get_watershed_markers(binary):
    binary = np.uint8(binary > 0) * 255  
    distance = distance_transform_edt(binary)
    smoothed_distance = gaussian_filter(distance, sigma=1)
    otsu_threshold = filters.threshold_otsu(smoothed_distance)
    markers = measure.label(smoothed_distance > otsu_threshold)  
    return markers

def apply_watershed_on_volume(volume_array):
    print(f"apply_watershed_on_volume - Input volume shape: {volume_array.shape}")
    watershed_segmented = np.zeros_like(volume_array, dtype=np.uint8)
    for i in range(volume_array.shape[0]):  
        binary_slice = np.uint8(volume_array[i] > 0) * 255  
        marker_slice = get_watershed_markers(binary_slice)
        distance = distance_transform_edt(binary_slice)
        segmented_slice = segmentation.watershed(-distance, marker_slice, mask=binary_slice)
        cleaned_segmented_slice = remove_small_objects(segmented_slice, min_size=10)
        watershed_segmented[i] = cleaned_segmented_slice
    print(f"Watershed segmentation - Final result shape: {watershed_segmented.shape}, Dtype: {watershed_segmented.dtype}")
    return watershed_segmented

def apply_dbscan_2d(volume_array, eps=5, min_samples=10):
    clustered_volume = np.zeros_like(volume_array, dtype=np.int32) 
    cluster_counts = {}
    for slice_idx in range(volume_array.shape[0]):
        print(f"Processing Slice {slice_idx} for DBSCAN...")
        slice_data = volume_array[slice_idx]
        yx_coords = np.column_stack(np.where(slice_data > 0))
        print(f"Slice {slice_idx} - Non-zero points: {len(yx_coords)}")
        if len(yx_coords) == 0:
            print(f"Slice {slice_idx} - No non-zero points, skipping...")
            continue  
        print(f"Applying DBSCAN on Slice {slice_idx}...")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(yx_coords)
        clustered_slice = np.zeros_like(slice_data, dtype=np.int32)
        for i, (y, x) in enumerate(yx_coords):
            clustered_slice[y, x] = labels[i] + 1  
        clustered_volume[slice_idx] = clustered_slice
        cluster_counts[slice_idx] = len(set(labels)) - (1 if -1 in labels else 0)  
    return clustered_volume, cluster_counts

def separate_merged_electrodes_mm(mask, spacing):
    separated_mask = np.zeros_like(mask, dtype=np.int32)
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
    unique_values = np.unique(pixel_values)
    print(f"Unique intensity values count: {len(unique_values)}")
    if len(unique_values) < n_components:
        print("⚠️ Not enough unique intensity values for GMM clustering!")
        return np.zeros_like(image) 
    try:
        n_clusters = min(n_components, len(unique_values))
        gmm = GaussianMixture(n_components=n_clusters)
        gmm_labels = gmm.fit_predict(pixel_values) 
        gmm_image = np.zeros_like(image)
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
    distance = distance_transform_edt(binary)
    local_max = peak_local_max(distance, min_distance=1, labels=binary)
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


def enhance_electrode_brightness_slice_by_slice(image, method='clahe', clip_limit=0.03, kernel_size=5):
    enhanced_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        slice_image = image[i, :, :]
        if method == 'clahe':
            clahe = exposure.CLAHE(clip_limit=clip_limit)
            enhanced_slice = clahe.apply(slice_image)
        elif method == 'tophat':
            selem = morphology.disk(kernel_size)
            enhanced_slice = morphology.white_tophat(slice_image, selem)
        elif method == 'log':
            enhanced_slice = filters.laplace(ndimage.gaussian_filter(slice_image, kernel_size))
        elif method == 'unsharp':
            gaussian_blurred = ndimage.gaussian_filter(slice_image, kernel_size)
            enhanced_slice = slice_image + 1.0 * (slice_image - gaussian_blurred)
        elif method == 'intensity_scaling':
            I_min = np.min(slice_image)
            I_max = np.max(slice_image)
            enhanced_slice = (slice_image - I_min)/(I_max - I_min)
        else:
            raise ValueError("Invalid enhancement method")
        enhanced_image[i, :, :] = enhanced_slice
    return enhanced_image



###########################################    
# Function to enhance the CTP.3D images ###
###############################################

def enhance_ctp(inputVolume, inputROI=None, methods = 'all', outputDir=None):
    methods ='all'
    volume_array = slicer.util.arrayFromVolume(inputVolume)
    if volume_array is None or volume_array.size == 0:
        print("Input volume data is empty or invalid.")
        return None

    if inputROI is not None:
        roi_array = slicer.util.arrayFromVolume(inputROI)
        roi_array = np.uint8(roi_array > 0)  # Ensure binary mask (0 or 1)
        print(f"Shape of input volume: {volume_array.shape}")
        print(f"Shape of ROI mask: {roi_array.shape}")
        print("Filling inside the ROI...")
        filled_roi = ndimage.binary_fill_holes(roi_array)
        print("Applying morphological closing...")
        struct_elem = morphology.ball(10)
        closed_roi = morphology.binary_closing(filled_roi, struct_elem)
        final_roi = closed_roi
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

    enhanced_volumes = {}
    if methods == 'all':
        
        ### Only CTP ###
        enhanced_volumes['OG_volume_array'] = volume_array
        print(f"OG_volume_array shape: {enhanced_volumes['OG_volume_array'].shape}")
        enhanced_volumes['DESCARGAR_OG_volume_array_2296'] = np.uint8(enhanced_volumes['OG_volume_array']>2296) #2: 2296
        #enhanced_volumes['denoise_ctp'] = denoise_2d_slices(enhanced_volumes['gaussian_volume_og'], patch_size=2, patch_distance=2, h=0.8)
        enhanced_volumes['OG_gaussian_volume_og'] = gaussian(enhanced_volumes['OG_volume_array'], sigma=0.3)
        #enhanced_volumes['DESCARGAR_OG_gaussian_volume_og'] = np.uint8(enhanced_volumes['OG_gaussian_volume_og']>1716) #8:1716
        enhanced_volumes['OG_gamma_volume_og'] = gamma_correction(enhanced_volumes['OG_gaussian_volume_og'] , gamma=3)
        enhanced_volumes['DESCARGAR_OG_gamma_volume_og_164'] = np.uint8(enhanced_volumes['OG_gamma_volume_og'] > 164) ####1: 164, 2: 164, 4:139, 5: 150, 6: 28, 7:76, 8: 49(queda contorno)
        enhanced_volumes['OG_sharpened'] = sharpen_high_pass(enhanced_volumes['OG_gamma_volume_og'], strenght=0.8)
        enhanced_volumes['DESCARGAR_OG_sharpened_167'] = np.uint8(enhanced_volumes['OG_sharpened']>167) ##1:92, 2: 167, 3:134, 4:124, 5> 149, 6:28, 7:54

        ### 8 ###
        # enhanced_volumes['OG_gamma_2_volume_og_8'] = gamma_correction(enhanced_volumes['OG_gauss_volume_og'], gamma=2)
        # enhanced_volumes['OG_sharpened_2_volume_og_8'] = sharpen_high_pass(enhanced_volumes['OG_gamma_2_volume_og_8'], strenght=0.4)
        # enhanced_volumes['OG_gamma_3_volume_og_8'] = gamma_correction(enhanced_volumes['OG_sharpened_2_volume_og_8'], gamma= 1.3)
        # enhanced_volumes['median_voxel_ctp'] = apply_median_filter_2d(enhanced_volumes['thresholded_ctp_per_og'], kernel_size_mm2=0.4)
        # enhanced_volumes['separate_erosion_ctp_og'] = separate_by_erosion(enhanced_volumes['median_voxel_ctp'], kernel_size_mm2=0.6, spacing=(1.0, 1.0), iterations=5)
        # enhanced_volumes['MASK_LABEL_ctp'] = np.uint8(enhanced_volumes['gamma_ctp_2'] > 0) * 255

        #### PRUEBA ROI####
        enhanced_volumes['PRUEBA_roi_plus_gamma_mask'] =  (final_roi > 0) * enhanced_volumes['OG_gamma_volume_og'] + (final_roi == 0) * 0 #enhanced_volumes['OG_gamma_volume_og']* final_roi
        enhanced_volumes['DESCARGAR_PRUEBA_roi_plus_gamma_mask_178'] = np.uint8(enhanced_volumes['PRUEBA_roi_plus_gamma_mask']>178) #1: 119,2:178,3:130, 4:108, 8:58  ### 4: 122, 5:114, 6: 21, 7:77
        enhanced_volumes['PRUEBA_roi_plus_gamma_mask_clahe'] = apply_clahe(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'])
        enhanced_volumes['DESCARGAR_PRUEBA_THRESHOLD_CLAHE_175'] = np.uint8(enhanced_volumes['PRUEBA_roi_plus_gamma_mask_clahe'] > 175) ## 1: 143, 3:149, 4:138,5:142, 6:25 2: 127, 8:57, 7:61
        enhanced_volumes['Prueba_final_roi'] = final_roi
        enhanced_volumes['PRUEBA_WAVELET_NL'] = wavelet_nlm_denoise(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'], wavelet='db1')
        enhanced_volumes['DESCARGAR_PRUEBA_WAVELET_NL'] = np.uint8(enhanced_volumes['PRUEBA_WAVELET_NL']> 182) ## 1: 106, 2:112,3:123, 4.115, 5:117,6:37,  8:42, 7:54
        ########################################
        enhanced_volumes['gaussian_volume_roi'] = gaussian(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'], sigma=0.3)
        enhanced_volumes['DESCARGAR_PRUEBA_GAUSSIAN_thre_0.000000081'] = np.uint8(enhanced_volumes['gaussian_volume_roi'] > 0.000000081) ### 1: 0.341,2:0.000000081, 3:0.430, 4: 0.352, 5: 0.525, 6: 0.0000000106, 8: 0.162, 7: 0.000000061
        enhanced_volumes['sharpened_roi'] = sharpen_high_pass(enhanced_volumes['gaussian_volume_roi'], strenght = 0.5)
        enhanced_volumes['gamma_volume_roi'] = gamma_correction(enhanced_volumes['sharpened_roi'], gamma=2)
        enhanced_volumes['DESCARGAR_gamma_volume_roi_167'] = np.uint8(enhanced_volumes['gamma_volume_roi'] > 167) # 1: 86, 2:167, 3:62, 4: 80, 5: 101, 6: 4, 8:17, 7:57
        ####################
        #######################


        ### Only ROI ###
        enhanced_volumes['roi_volume'] = roi_volume
        enhanced_volumes['wavelet_only_roi'] = wavelet_denoise(enhanced_volumes['roi_volume'], wavelet= 'db1')
        enhanced_volumes['gamma_only_roi'] = gamma_correction(enhanced_volumes['wavelet_only_roi'], gamma=0.8)
        enhanced_volumes['sharpened_wavelet_roi'] = sharpen_high_pass(enhanced_volumes['wavelet_only_roi'], strenght= 0.5) 
        enhanced_volumes['sharpened_roi_only_roi'] = sharpen_high_pass(enhanced_volumes['roi_volume'], strenght = 0.8)
        enhanced_volumes['LOG_roi'] = log_transform_slices(enhanced_volumes['sharpened_roi_only_roi'], c=3)
        #######
        enhanced_volumes['DESCARGAR_Threshold_roi_volume_2346'] = np.uint8(enhanced_volumes['roi_volume']>2346)## 1: 1646, 2: 23464: 1526, 5: 2422, 8: 1539
        enhanced_volumes['DESCARGAR_WAVELET_ROI_2626'] = np.uint8(enhanced_volumes['roi_volume']>2626) #1: 1876
        enhanced_volumes['DESCARGAR_GAMMA_ONLY_ROI'] = np.uint8(enhanced_volumes['gamma_only_roi']>217) ##1: 182,2:217,3:190, 4:166, 5:188, 6: 125, 8:163, 7:171
       ### Prueba roi_plus gamma después###
        enhanced_volumes['2_gaussian_volume_roi'] = gaussian(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'], sigma=0.3)
        enhanced_volumes['DESCARGAR_2_gaussian_volume_roi_0.00000008'] = np.uint8(enhanced_volumes['2_gaussian_volume_roi']> 0.000000080) #6
        ###enhanced_volumes['2_clahe'] = apply_clahe(enhanced_volumes['wavelet_only_roi'])
       
        #enhanced_volumes['2_wavelet_denoised'] = wavelet_denoise(enhanced_volumes['2_gaussian_volume_roi'])
        #enhanced_volumes['2_sharpened_wavelet'] = sharpen_high_pass(enhanced_volumes['2_wavelet_denoised'], strenght=0.7) 
        #enhanced_volumes['DESCARGAR_FINAL_2'] = np.uint8(enhanced_volumes['2_wavelet_denoised'] > 0.000000035) ### 1:0.307, 2: 0.370, 3:0.408, 4: 0.335, 5: 0.372, 8: 0229, 7:0.000000035
        #enhanced_volumes['DESCARGAR_SHARPENED_WAVELET'] = np.uint8(enhanced_volumes['2_sharpened_wavelet']>0.401) #1:0.298, 2:0.397, 3:0.415, 4:0.364, 5:0.401, 

        enhanced_volumes['2_gamma_correction'] = gamma_correction(enhanced_volumes['2_gaussian_volume_roi'] , gamma = 0.8)
        enhanced_volumes['DESCARGAR_2_gamma_correction_214'] = np.uint8(enhanced_volumes['2_gamma_correction'] >214) ## 8:9
        kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        tophat_2 = cv2.morphologyEx(enhanced_volumes['2_gaussian_volume_roi'], cv2.MORPH_TOPHAT, kernel_2)
        enhanced_volumes['2_tophat'] = cv2.addWeighted(enhanced_volumes['2_gaussian_volume_roi'], 1, tophat_2, 2, 0) ### 8:5 
        enhanced_volumes['DESCARGAR_2_tophat_0.000000083'] = np.uint8(enhanced_volumes['2_tophat'] > 0.000000083)  # 1: 0.415, 3:0.515, 4:0.427, 8:0.242,6: 0.0000000158,  7: 0.000000042
        enhanced_volumes['2_sharpened'] = sharpen_high_pass(enhanced_volumes['2_gamma_correction'], strenght = 0.8)
        enhanced_volumes['DESCARGAR_2_sharpened_199'] = np.uint8(enhanced_volumes['2_sharpened'] > 199) ##1: 134, 2:109,3:143, 4:121, 5: 121, 6: 51, 8:91, 7:134
        # Apply white top-hat transformation
        enhanced_volumes['2_LOG'] = log_transform_slices(enhanced_volumes['2_tophat'], c=3)
        enhanced_volumes['DESCARGAR_2_LOG_199'] = np.uint8(enhanced_volumes['2_LOG'] > 199) ### 1: 97, 2:100, 3:146, 4:106, 5 : 111, 8:91, 7: 78
        enhanced_volumes['2_wavelet_roi'] = wavelet_denoise(enhanced_volumes['2_LOG'], wavelet='db4')
        enhanced_volumes['DESCARGAR_2_wavelet_roi_154'] = np.uint8(enhanced_volumes['2_wavelet_roi']>154) ##1: 125, 2:80, 3:137, 4:130, 5: 160, 7:124
        enhanced_volumes['2_erode'] = morphological_operation(enhanced_volumes['2_sharpened'], operation='erode', kernel_size=1)
        enhanced_volumes['DESCARGAR_2__207'] = np.uint8(enhanced_volumes['2_erode'] > 207) ##1:150, 2:145, 4:106, 8:93
        enhanced_volumes['2_gaussian_2'] = gaussian(enhanced_volumes['2_erode'], sigma= 0.2)
        enhanced_volumes['2_sharpening_2_trial'] = sharpen_high_pass(enhanced_volumes['2_gaussian_2'], strenght = 0.8)
        enhanced_volumes['DESCARGAR_2_sharpening_2_trial_0.789'] = np.uint8(enhanced_volumes['2_sharpening_2_trial']>0.789) ##1: 0.47, 2:0.789, 3:0.571, 4:0.515, 5: 0.445, 6: 0.207, 7: 0.303


        ### Prueba otras cosas ###
        enhanced_volumes['NUEVO_NLMEANS'] = wavelet_nlm_denoise(enhanced_volumes['roi_volume'])
        enhanced_volumes['DESCARGAR_NUEVO_NLMEANS_2226'] = np.uint8(enhanced_volumes['NUEVO_NLMEANS']>2226) ###1: 1946,2: 1386, 3:1856, 4:1546,5: 2022, 6: 936,  8:1510; 7: 1368

        ###ORGINAL_IDEA ####
        enhanced_volumes['ORGINAL_IDEA_gaussian'] = gaussian(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'], sigma= 0.3)
       # enhanced_volumes['FINAL_ORGINAL_IDEA_GAUSSIAN_THRESHOLD'] = np.uint8(enhanced_volumes['ORGINAL_IDEA_gaussian'] > 0.055)  ### 8:0.055 (no completo)
        enhanced_volumes['ORGINAL_IDEA_gamma_correction'] = gamma_correction(enhanced_volumes['ORGINAL_IDEA_gaussian'], gamma = 2)
        enhanced_volumes['DESCARGAR_ORGINAL_IDEA_gamma_correction_164)'] = np.uint8(enhanced_volumes['ORGINAL_IDEA_gamma_correction'] > 164) ##1: 1, 2:2, 3:3, 4:1, 5:2, 6:0, 7:0, 8:0
        enhanced_volumes['ORGINAL_IDEA_sharpened'] = sharpen_high_pass(enhanced_volumes['ORGINAL_IDEA_gamma_correction'], strenght = 0.8)
        enhanced_volumes['DESCARGAR_ORGINAL_IDEA_sharpened_141'] = np.uint8(enhanced_volumes['ORGINAL_IDEA_sharpened'] > 141) ### 1: 70, 2:141, 3:84, 4:46, 5:82 8: 25, 7:69, 6: 3
        enhanced_volumes['ORIGINAL_IDEA_SHARPENED_OPENING'] = morph_operations(enhanced_volumes['DESCARGAR_ORGINAL_IDEA_sharpened_141'], operation= 'open', kernel_shape= 'cross', kernel_size= 1)
        enhanced_volumes['ORIGINAL_IDEA_wavelet'] = wavelet_denoise(enhanced_volumes['ORGINAL_IDEA_sharpened'])
        enhanced_volumes['DESCARGAR_ORIGINAL_IDEA_wavelet_127'] = np.uint8(enhanced_volumes['ORIGINAL_IDEA_wavelet']>127) ## 1: 58, 2:71,3:103, 4:48,5:72,  8:18, 7:93, 6:2
        enhanced_volumes['ORGINAL_IDEA_gaussian_2'] = gaussian(enhanced_volumes['ORGINAL_IDEA_sharpened'], sigma= 0.4)
        enhanced_volumes['DESCARGAR_ORGINAL_IDEA_gaussian_2_0.563'] = np.uint8(enhanced_volumes['ORGINAL_IDEA_gaussian_2']>0.563) #1: 0.161, 2:0.193,3:0.318 8:0.057
        enhanced_volumes['ORIGINAL_IDEA_GAMMA_2'] = gamma_correction(enhanced_volumes['ORGINAL_IDEA_gaussian_2'], gamma = 1.4)
        enhanced_volumes['DESCARGAR_ORIGINAL_IDEA_GAMMA_2_132'] = np.uint8(enhanced_volumes['ORIGINAL_IDEA_GAMMA_2']>132) ##1: 39, 2:56, 3:65,4:26,5:70,6: 3,  8:8, 7:39
        kernel_size_og = (1, 1) 
        kernel_og = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size_og)
        tophat_og = cv2.morphologyEx(enhanced_volumes['ORGINAL_IDEA_gaussian_2'], cv2.MORPH_TOPHAT, kernel_og)
        enhanced_volumes['OG_tophat_1'] = cv2.addWeighted(enhanced_volumes['ORGINAL_IDEA_gaussian_2'], 1, tophat_og, 2, 0)
        enhanced_volumes['DESCARGAR_OG_tophat_1_0.538'] = np.uint8(enhanced_volumes['OG_tophat_1']>0.538) #1: 0.217, 2:0.276 3:0.29, 4:0.128, 5: 0.372, 6: 0.027
        #enhanced_volumes['OG_GAUSSIAN_TOPHAT'] = gaussian(enhanced_volumes['OG_tophat_1'], sigma= 0.9)
        
        ### First try ###

        enhanced_volumes['FT_gaussian'] = gaussian(roi_volume, sigma= 0.3)
        kernel_size = (1, 1)  # Keep it small since 2x2 might be too large
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        tophat_ft = cv2.morphologyEx(roi_volume, cv2.MORPH_TOPHAT, kernel)
        enhanced_volumes['FT_tophat_1'] = cv2.addWeighted(roi_volume, 1, tophat_ft, 2, 0)
        enhanced_volumes['FT_RESTA_TOPHAT_GAUSSIAN'] = enhanced_volumes['FT_tophat_1'] - gaussian(roi_volume, sigma= 0.8)
        enhanced_volumes['DESCARGAR_FT_RESTA_TOPHAT_GAUSSIAN_1516'] = np.uint(enhanced_volumes['FT_RESTA_TOPHAT_GAUSSIAN'] >1516)  #1:594,2:929,3:1178,4:493,5:1103,  8:378, 7: 396,6: 129
        enhanced_volumes['FT_gamma_correction'] = gamma_correction(enhanced_volumes['FT_gaussian'], gamma = 5)
        #enhanced_volumes['FINAL_FT_gamma_THRESHOLD'] = np.uint8(enhanced_volumes['FT_gamma_correction'] > 24) 
        enhanced_volumes['FT_sharpened'] = sharpen_high_pass(enhanced_volumes['FT_gamma_correction'], strenght = 0.4)
        enhanced_volumes['DESCARGAR_FT_sharpened_146'] = np.uint8(enhanced_volumes['FT_sharpened']>146) #1:27, 2: 63
        enhanced_volumes['FT_gaussian_2'] = gaussian(enhanced_volumes['FT_sharpened'], sigma= 0.4)
        enhanced_volumes['DESCARGAR_FT_gaussian_2_0.482'] = np.uint8(enhanced_volumes['FT_gaussian_2'] > 0.482) #1:0.205, 2:0.257, 3:0.275, 8:0.155
        enhanced_volumes['FT_gamma_2'] = gamma_correction(enhanced_volumes['FT_gaussian_2'], gamma= 1.2)
        enhanced_volumes['DESCARGAR_FT_GAMMA_2_106'] = np.uint8(enhanced_volumes['FT_gamma_2']>106) #1: 32, 2:47, 3:72,4:16,5: 86,  8:15, 7:40,6 :4
        enhanced_volumes['FT_opening'] = morph_operations(enhanced_volumes['FT_gamma_2'], iterations=2, kernel_shape= 'cross')
        enhanced_volumes['FT_closing'] = morph_operations(enhanced_volumes['FT_opening'], operation= 'close')
        #enhanced_volumes['wo_large_objects'] = remove_large_objects(enhanced_volumes['closing'], size_threshold= 9000)
        enhanced_volumes['FT_erode_2'] = morphological_operation(enhanced_volumes['FT_closing'], operation='erode', kernel_size=1)
        enhanced_volumes['DESCARGAR_FT_ERODE_2_133'] = np.uint8(enhanced_volumes['FT_erode_2']>133) #1:35, 2:42,3:68,4:10,5: 85,  8:18, 7: 35, 6: 6
        # Create an elliptical structuring element (adjust size if needed)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        # Apply white top-hat transformation
        tophat = cv2.morphologyEx(enhanced_volumes['FT_gaussian_2'], cv2.MORPH_TOPHAT, kernel)
        enhanced_volumes['FT_tophat'] = cv2.addWeighted(enhanced_volumes['FT_gaussian_2'], 1, tophat, 2, 0)
        enhanced_volumes['DESCARGAR_FT_TOPHAT_0.490'] = np.uint8(enhanced_volumes['FT_tophat'] > 0.490) #1: 0.145, 3:0.313, 4:0.068, 5: 0.396 ,6: 0.034,  8:0.090; 7: 0.170
        #################
        enhanced_volumes['FT_gaussian_3'] = gaussian(enhanced_volumes['FT_tophat'], sigma= 0.1)
        enhanced_volumes['DESCARGAR_FT_gaussian_3_0.540'] = np.uint8(enhanced_volumes['FT_gaussian_3'] > 0.540) ##1: 0.123, 2:0.179, 3:0.305,4:0.064,5: 0.363,6:0.034,  8:0.101, 7: 0.313
        ##################
    if outputDir is None:
        outputDir = slicer.app.temporaryPath()  
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    # Process each enhanced volume
    enhancedVolumeNodes = {}
    for method_name, enhanced_image in enhanced_volumes.items():
        enhancedVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        enhancedVolumeNode.SetName(f"Enhanced_th45_{method_name}_{inputVolume.GetName()}")
        enhancedVolumeNode.SetOrigin(inputVolume.GetOrigin())
        enhancedVolumeNode.SetSpacing(inputVolume.GetSpacing())
        ijkToRasMatrix = vtk.vtkMatrix4x4()
        inputVolume.GetIJKToRASMatrix(ijkToRasMatrix)  
        enhancedVolumeNode.SetIJKToRASMatrix(ijkToRasMatrix) 
        slicer.util.updateVolumeFromArray(enhancedVolumeNode, enhanced_image)
        enhancedVolumeNodes[method_name] = enhancedVolumeNode
        output_file = os.path.join(outputDir, f"Filtered_th_45_{method_name}_{inputVolume.GetName()}.nrrd")
        slicer.util.saveNode(enhancedVolumeNode, output_file)
        print(f"Saved {method_name} enhancement as: {output_file}")
    return enhancedVolumeNodes


####################
# inputVolume = slicer.util.getNode('CTp.3D')  
# inputROI = slicer.util.getNode('patient2_mask_5')  # Brain Mask 
# # # # # # # # Output directory
# outputDir = r"C:\Users\rocia\Downloads\TFG\Cohort\Enhance_ctp_tests\P2\TH50"
# # # # # # # # # # # # Test the function 
# enhancedVolumeNodes = enhance_ctp(inputVolume, inputROI, methods='all', outputDir=outputDir)
# # # # # # # # # # # # Access the enhanced volume nodes
# for method, volume_node in enhancedVolumeNodes.items():
#               if volume_node is not None:
#                   print(f"Enhanced volume for method '{method}': {volume_node.GetName()}")
#               else:
#                   print(f"Enhanced volume for method '{method}': No volume node available.")
#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Threshold_mask\enhance_ctp.py').read())
