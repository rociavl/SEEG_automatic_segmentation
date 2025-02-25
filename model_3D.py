import numpy as np
import slicer
import vtk
from vtk.util import numpy_support
import scipy.ndimage as ndi 
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, median_filter, laplace, sobel
import logging
from skimage.filters import frangi
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from skimage.measure import regionprops

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def remove_large_objects(volume, size_threshold):
    """
    Removes objects that are larger than a given size threshold.
    
    Parameters:
        volume (numpy.ndarray): The 3D image (volume).
        size_threshold (int): The size threshold for removing large objects.
        
    Returns:
        numpy.ndarray: The volume with large objects removed.
    """
    try:
        # Label connected components
        labeled_volume, num_features = ndi.label(volume > 0)  # Only keep foreground
        
        # Get the properties of the labeled regions
        regions = regionprops(labeled_volume)
        
        # Create an empty volume to hold the filtered results
        filtered_volume = np.zeros_like(volume)
        
        # Loop through the regions and keep only those below the size threshold
        for region in regions:
            if region.area <= size_threshold:
                for coord in region.coords:
                    filtered_volume[tuple(coord)] = volume[tuple(coord)]  # Keep small objects
        
        print(f"✅ Removed {num_features - len([r for r in regions if r.area <= size_threshold])} large objects")
        return filtered_volume

    except Exception as e:
        logging.error(f"❌ Error removing large objects: {str(e)}")
        return volume 


# Vesselness Filter without SimpleITK Hessian (Custom Implementation)
def vesselness_filter(image, sigma=1.0, alpha=0.5, beta=0.5):
    """
    Applies custom vesselness filter for enhancing vessels.
    
    Parameters:
        image (numpy.ndarray): The 3D image (volume) to be filtered.
        sigma (float): The standard deviation of the Gaussian filter used in Hessian computation.
        alpha (float): The sensitivity parameter for vesselness calculation.
        beta (float): The sensitivity parameter for vesselness calculation.
        
    Returns:
        numpy.ndarray: The vesselness-enhanced image.
    """
    print(f"Applying vesselness filter with sigma={sigma}, alpha={alpha}, beta={beta}")
    try:
        # Compute Hessian manually
        hessian_image = np.array([gaussian_filter(image, sigma=sigma, order=(2, 0, 0)), 
                                  gaussian_filter(image, sigma=sigma, order=(0, 2, 0)), 
                                  gaussian_filter(image, sigma=sigma, order=(0, 0, 2))])
        
        # Compute eigenvalues of the Hessian matrix
        eigvals = np.linalg.eigvalsh(hessian_image)
        
        # Sort eigenvalues by absolute value
        eigvals = np.sort(np.abs(eigvals), axis=0)
        
        # Compute vesselness measure
        vesselness = np.zeros_like(image)
        lambda1, lambda2, lambda3 = eigvals[0], eigvals[1], eigvals[2]
        vesselness = (1 - np.exp(-lambda2**2 / (2 * alpha**2))) * np.exp(-lambda3**2 / (2 * beta**2))
        
        # Normalize vesselness
        vesselness = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min())
        
        return vesselness

    except Exception as e:
        logging.error(f"Error applying vesselness filter: {str(e)}")
        return None

def log_filter_3d(image, sigma=1.0):
    # Apply Gaussian filter first
    smoothed = gaussian_filter(image, sigma=sigma)
    # Apply Laplacian to detect edges
    log_filtered = laplace(smoothed)
    return log_filtered 


def sobel_filter(image):
    """Applies Sobel edge detection filter."""
    print("Applying Sobel edge detection filter")
    try:
        edges = np.sqrt(sobel(image, axis=0)**2 + sobel(image, axis=1)**2 + sobel(image, axis=2)**2)
        return edges
    except Exception as e:
        logging.error(f"Error applying Sobel filter: {str(e)}")
        return None

def gamma_correction(image, gamma=1.0):
    """Applies Gamma correction to the image."""
    print(f"Applying Gamma correction with gamma={gamma}")
    try:
        image = np.power(image, gamma)
        return image
    except Exception as e:
        logging.error(f"Error applying gamma correction: {str(e)}")
        return None


# Apply 3D filter method
def apply_3d_filter(volume, method, params=None):
    """Applies various 3D filters using CPU-based methods."""
    if params is None:
        params = {}

    try:
        print(f"Applying {method} filter with params: {params}")

        # Apply filters using SciPy or SimpleITK
        if method == 'gaussian':
            sigma = params.get('sigma', 1.5)
            result = gaussian_filter(volume, sigma=sigma)
        elif method == 'median':
            size = params.get('size', 1)
            result = median_filter(volume, size=size)
        elif method == 'laplacian':
            result = laplace(volume)
        elif method == 'frangi':
            result = frangi(volume, scale_range=params.get('scale_range', (1, 10)), scale_step=params.get('scale_step', 2))
        elif method == 'vesselness':
            result = vesselness_filter(volume, sigma=params.get('sigma', 1.0), alpha=params.get('alpha', 0.5), beta=params.get('beta', 0.5))
        elif method == 'sobel':
            result = sobel_filter(volume)
        elif method == 'gamma_correction':
            result = gamma_correction(volume, gamma=params.get('gamma', 1.0))
        else:
            logging.error(f"Unknown filter method: {method}")
            return None

        # Normalize output
        result = (result - result.min()) / (result.max() - result.min())
        print(f"{method} filter applied and normalized successfully")
        return result

    except Exception as e:
        logging.error(f"Error applying {method} filter: {str(e)}")
        return None


# Process volume with cascading filters 
def process_volume_3d(input_volume, methods=None, output_dir=None):
    """Processes volume with multiple 3D enhancement methods sequentially, cascading the filters."""
    if methods is None:
        methods = [
            ('gamma_correction', {'gamma': 5}),
            ('laplacian', {}),
            ('gaussian', {'sigma': 3}),
            ('vesselness', {'sigma': 1.0, 'alpha': 0.5, 'beta': 0.5}),
            ('frangi', {}),
            ('sobel', {}),
            ('gamma_correction', {'gamma': 1.5}),
        ]
    
    # Load and normalize input volume
    try:
        print("Loading input volume and normalizing")
        volume_array = slicer.util.arrayFromVolume(input_volume)
        print(f"Volume array shape: {volume_array.shape}")
        print(f"Initial volume range: {volume_array.min()} to {volume_array.max()}")
        
        if volume_array.min() == volume_array.max():
            print("❌ The input volume has no variation in intensity!")
            return None
        
        volume_array = volume_array.astype(np.float32)
        volume_array = (volume_array - volume_array.min()) / (volume_array.max() - volume_array.min())
        print(f"Normalized volume range: {volume_array.min()} to {volume_array.max()}")
        
    except Exception as e:
        logging.error(f"Error loading and normalizing input volume: {str(e)}")
        return None

    enhanced_nodes = {}

    # Apply filters in a cascade
    for method, params in methods:
        print(f"Processing method: {method}")
        
        # Apply filter to the current volume (cascading)
        volume_array = apply_3d_filter(volume_array, method, params)
        
        if volume_array is None:
            logging.warning(f"Skipping method {method} due to filter error")
            continue

        #volume_array = remove_large_objects(volume_array, size_threshold=100)
        print(f"🔍 After {method}, range: {volume_array.min()} to {volume_array.max()}")
        
        
        # Create new volume node for filtered data
        try:
            print(f"Creating new volume node for {method}")
            volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
            volume_node.SetName(f"Enhanced_3D_{method}_{input_volume.GetName()}")

            # Copy spatial transformations correctly
            ijkToRAS = vtk.vtkMatrix4x4()
            input_volume.GetIJKToRASMatrix(ijkToRAS)
            volume_node.SetIJKToRASMatrix(ijkToRAS)
            volume_node.SetOrigin(input_volume.GetOrigin())
            volume_node.SetSpacing(input_volume.GetSpacing())

            # Update with new data
            slicer.util.updateVolumeFromArray(volume_node, volume_array)

            enhanced_nodes[method] = volume_node
            print(f"Volume node created and updated for {method}")

            # Save output if directory provided
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"Enhanced_3D_{method}_{input_volume.GetName()}.nrrd"
                slicer.util.saveNode(volume_node, str(output_file))
                print(f"Saved enhanced volume to {output_file}")

        except Exception as e:
            logging.error(f"Error processing method {method}: {str(e)}")

    return enhanced_nodes

# #  Main function
# def main():
#     input_volume = slicer.util.getNode("Filtered_gamma_2_ctp.3D")
#     if not input_volume:
#         print("❌ No volume loaded in Slicer")
#         return

#     methods = [
#         ('gamma_correction', {'gamma': 1}),
#         ('log_filter_3d', {}),
#         ('sobel', {})
#     ]

#     enhanced_nodes = process_volume_3d(input_volume, methods=methods, output_dir=r'C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Enhance_ctp_tests\\P1\\Enhance_ctp_3D')

#     if enhanced_nodes:
#         for method, node in enhanced_nodes.items():
#             print(f"✅ Enhanced volume ({method}): {node.GetName()}")


# if __name__ == '__main__':
#     main()

# #exec(open('C:/Users/rocia/AppData/Local/slicer.org/Slicer 5.6.2/SEEG_module/SEEG_masking/model_3D.py').read())