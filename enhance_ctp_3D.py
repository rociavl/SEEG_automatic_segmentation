import numpy as np
import slicer
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, median_filter
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means, estimate_sigma
import pywt
import os
import vtk

from skimage import measure
from scipy.ndimage import binary_erosion

def enhance_ctp_3D(inputVolume, methods=['gaussian'], outputDir=None):
    """
    Apply multiple 3D filtering techniques to enhance CT volumes in Slicer.

    Parameters:
    - inputVolume (vtkMRMLScalarVolumeNode): The input CT scan.
    - methods (list): List of filtering methods to apply in sequence.
    - outputDir (str, optional): Directory to save the enhanced volume.

    Returns:
    - enhancedVolumeNodes (dict): Dictionary of filtered volumes with method names as keys.
    """
    # Convert input volume to numpy array
    volume_array = slicer.util.arrayFromVolume(inputVolume)

    # Ensure volume is valid
    if volume_array is None or volume_array.size == 0:
        print("❌ Error: Input volume is empty or invalid.")
        return None

    print(f"📏 Input volume shape: {volume_array.shape}")

    # Start with the original volume
    processed_array = volume_array.copy()
    enhancedVolumeNodes = {}

    for method in methods:
        print(f"🔍 Applying {method}...")

        if method == 'gaussian':
            processed_array = gaussian_filter(processed_array, sigma=1.5)

        elif method == 'median':
            processed_array = median_filter(processed_array, size=3)

        elif method == 'anisotropic':
            processed_array = denoise_tv_chambolle(processed_array, weight=0.1, multichannel=False)

        elif method == 'nl_means':
            sigma_est = np.mean(estimate_sigma(processed_array))
            processed_array = denoise_nl_means(processed_array, h=1.15 * sigma_est, fast_mode=True)

        elif method == 'wavelet':
            coeffs = pywt.wavedecn(processed_array, wavelet='coif3', level=3)
            
            # Process all but the first element (which is an array, not a dict)
            coeffs_thresh = [coeffs[0]]  # Keep the approximation coefficients
            coeffs_thresh += [{key: pywt.threshold(value, 0.02 * np.max(value), mode='soft') 
                            for key, value in level.items()} for level in coeffs[1:]]
            
            processed_array = pywt.waverecn(coeffs_thresh, wavelet='coif3')
        
        # Apply morphological closing to merge dots
        if method == 'gaussian' or method == 'median':  # Example condition to apply morphological operations after smoothing
            processed_array = morphology.binary_closing(processed_array, morphology.ball(3))  # Increase size for larger effect

        # Connected component analysis (for removing small structures)
        labeled_array, num_features = measure.label(processed_array, connectivity=3)
        sizes = np.bincount(labeled_array.ravel())
        min_size = 500  # Filter small components (adjust this size as needed)
        processed_array = np.isin(labeled_array, np.where(sizes >= min_size)[0])

        # Create a new volume node to store the intermediate enhanced image
        enhancedVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        enhancedVolumeNode.SetName(f"Filtered_3D_{method}_{inputVolume.GetName()}")

        # Copy transformation information
        enhancedVolumeNode.SetOrigin(inputVolume.GetOrigin())
        enhancedVolumeNode.SetSpacing(inputVolume.GetSpacing())

        # Get and set the IJK to RAS transformation matrix
        ijkToRasMatrix = vtk.vtkMatrix4x4()
        inputVolume.GetIJKToRASMatrix(ijkToRasMatrix)
        enhancedVolumeNode.SetIJKToRASMatrix(ijkToRasMatrix)

        # Update the volume node with the enhanced image data
        slicer.util.updateVolumeFromArray(enhancedVolumeNode, processed_array)

        # Store the node in the results dictionary
        enhancedVolumeNodes[method] = enhancedVolumeNode

        # Set output directory
        if outputDir is None:
            outputDir = slicer.app.temporaryPath()
        
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        # Save the volume as NRRD
        output_file = os.path.join(outputDir, f"Filtered_3D_{method}_{inputVolume.GetName()}.nrrd")
        slicer.util.saveNode(enhancedVolumeNode, output_file)
        print(f"✅ Saved {method} enhancement as: {output_file}")

    return enhancedVolumeNodes



# Load the input volume and ROI
inputVolume = slicer.util.getNode('Filtered_gamma_3_ctp.3D')  


# # Output directory
outputDir = r'C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Enhance_ctp_tests\\P1\\Enhance_ctp_3D'  

# # Test the function 
enhancedVolumeNodes = enhance_ctp_3D(inputVolume,  methods=['gaussian', 'wavelet', 'nl_means'], outputDir=outputDir)

# # Access the enhanced volume nodes
for method, volume_node in enhancedVolumeNodes.items():
    if volume_node is not None:
         print(f"Enhanced volume for method '{method}': {volume_node.GetName()}")
    else:
         print(f"Enhanced volume for method '{method}': No volume node available.")

#exec(open('C:/Users/rocia/AppData/Local/slicer.org/Slicer 5.6.2/SEEG_module/SEEG_masking/enhance_ctp_3D.py').read())
