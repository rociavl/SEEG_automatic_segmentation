import slicer
import numpy as np
import vtk
import cv2
from skimage import exposure, filters, morphology
from skimage.exposure import rescale_intensity
import pywt
import pywt.data
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy import ndimage

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

    
def bilateral_filter(roi_volume, d=9, sigma_color=75, sigma_space=75):
        # Convert roi_volume to uint8 (if not already) and ensure values are in the range 0-255
        roi_volume_uint8 = np.uint8(np.clip(roi_volume, 0, 255))

        filtered_slices = np.zeros_like(roi_volume_uint8)
        for i in range(roi_volume_uint8.shape[0]):
            filtered_slices[i] = cv2.bilateralFilter(roi_volume_uint8[i], d, sigma_color, sigma_space)
        return filtered_slices
    
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
        enhanced_volumes['local_otsu'] = local_otsu(roi_volume)
        enhanced_volumes['local_threshold'] = local_threshold(roi_volume)
        enhanced_volumes['gamma_correction'] = gamma_correction(roi_volume, gamma=1.8)
        
        # Noise reduction
        enhanced_volumes['bilateral_filter'] = bilateral_filter(enhanced_volumes['gamma_correction'])
        enhanced_volumes['SHARPEN'] = sharpen_high_pass(enhanced_volumes['bilateral_filter'])
        enhanced_volumes['wavelet_gamma'] = wavelet_denoise(enhanced_volumes['SHARPEN']) 


        # Edge boosting?
        enhanced_volumes['Laplacian_sharpen'] = laplacian_sharpen(enhanced_volumes['wavelet_gamma'])
        enhanced_volumes['Sharpen_Laplacian_sharpen'] = sharpen_high_pass(enhanced_volumes['Laplacian_sharpen'])


        # # Morphological operations
        enhanced_volumes['morphological_filter_opening'] = morph_operations(enhanced_volumes['Sharpen_Laplacian_sharpen'])
        enhanced_volumes['morphological_filter_closing'] = morph_operations(enhanced_volumes['morphological_filter_opening'], operation='close')

        # # Enhance the contrast
        enhanced_volumes['gamma_2'] = gamma_correction(enhanced_volumes['morphological_filter_closing'], gamma=1.5)
        enhanced_volumes['Laplacian_sharpen_2'] = sharpen_high_pass(enhanced_volumes['gamma_2'])

      

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
# inputVolume = slicer.util.getNode('ctp.3D')  
# inputROI = slicer.util.getNode('P1_brain_mask')  # Brain Mask 

# # # Output directory
# outputDir = r'C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Enhance_ctp_tests\\P4'  

# # # Test the function 
# enhancedVolumeNodes = enhance_ctp(inputVolume, inputROI, methods='all', outputDir=outputDir)

# # # Access the enhanced volume nodes
# for method, volume_node in enhancedVolumeNodes.items():
#      if volume_node is not None:
#          print(f"Enhanced volume for method '{method}': {volume_node.GetName()}")
#      else:
#          print(f"Enhanced volume for method '{method}': No volume node available.")


#exec(open('C:/Users/rocia/AppData/Local/slicer.org/Slicer 5.6.2/SEEG_module/SEEG_masking/enhance_ctp.py').read())
