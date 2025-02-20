import numpy as np
import slicer
import SimpleITK as sitk
import cupy as cp  # GPU acceleration
from scipy.ndimage import gaussian_filter
import cv2  # Fast OpenCV processing
from skimage.morphology import skeletonize_3d
import os
import vtk
from joblib import Parallel, delayed
import multiprocessing.shared_memory as shm
from skimage.measure import label, regionprops

# ----------- FAST GPU FILTERS ----------- Is too slow!!
def apply_gaussian_gpu(volume, sigma=1.5):
    """Uses GPU-based Gaussian filtering with CuPy."""
    volume_gpu = cp.asarray(volume)  # Move to GPU
    result_gpu = cp.array(gaussian_filter(volume_gpu, sigma=sigma))
    return cp.asnumpy(result_gpu)  # Move back to CPU

def apply_fast_bilateral_gpu(volume, d=5, sigmaColor=50, sigmaSpace=50):
    """Fast bilateral filter using OpenCV (optimized for large data)."""
    return np.array([cv2.bilateralFilter(slice.astype(np.float32), d, sigmaColor, sigmaSpace) for slice in volume])

def apply_skeleton_gpu(volume):
    """Skeletonization with optimized GPU processing."""
    volume_gpu = cp.asarray(volume)
    skeleton_gpu = cp.array(skeletonize_3d(volume_gpu))
    return cp.asnumpy(skeleton_gpu)

# ----------- OPTIMIZED ENHANCEMENT FUNCTION -----------

def enhance_ctp_3D(inputVolume, methods=['gaussian_gpu'], outputDir=None):
    """Apply ultra-fast 3D filtering with GPU and parallel processing."""
    
    # Load CT scan as numpy array
    volume_array = slicer.util.arrayFromVolume(inputVolume).astype(np.float16)  # Convert to float16 (faster!)

    if volume_array is None or volume_array.size == 0:
        print("❌ Error: Empty volume.")
        return None

    print(f"📏 Input shape: {volume_array.shape}")
    enhancedVolumeNodes = {}

    # Use multiprocessing shared memory to avoid slow memory copies
    shared_memory = shm.SharedMemory(create=True, size=volume_array.nbytes)
    shared_array = np.ndarray(volume_array.shape, dtype=volume_array.dtype, buffer=shared_memory.buf)
    np.copyto(shared_array, volume_array)  # Copy data into shared memory

    # Run filters in parallel
    def process_method(method):
        print(f"🔍 Applying {method}...")

        filtered_array = shared_array.copy()  # Work on a shared copy

        if method == 'gaussian_gpu':
            filtered_array = apply_gaussian_gpu(filtered_array)
        elif method == 'fast_bilateral_gpu':
            filtered_array = apply_fast_bilateral_gpu(filtered_array)
        elif method == 'skeleton_gpu':
            filtered_array = apply_skeleton_gpu(filtered_array)

        # Convert back to uint8 (if necessary)
        filtered_array = np.clip(filtered_array, 0, 255).astype(np.uint8)

        # Create & update Slicer volume node
        enhancedVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        enhancedVolumeNode.SetName(f"Filtered_3D_{method}_{inputVolume.GetName()}")
        slicer.util.updateVolumeFromArray(enhancedVolumeNode, filtered_array)
        enhancedVolumeNode.Copy(inputVolume)  # Copy metadata

        # Save if needed
        if outputDir:
            os.makedirs(outputDir, exist_ok=True)
            output_file = os.path.join(outputDir, f"Filtered_3D_{method}_{inputVolume.GetName()}.nrrd")
            slicer.util.saveNode(enhancedVolumeNode, output_file)
            print(f"✅ Saved {method}: {output_file}")

        return method, enhancedVolumeNode

    # Use parallel processing (all CPU cores + GPU)
    results = Parallel(n_jobs=-1, backend="loky")(delayed(process_method)(method) for method in methods)

    # Free shared memory
    shared_memory.close()
    shared_memory.unlink()

    # Collect results
    for method, node in results:
        enhancedVolumeNodes[method] = node

    return enhancedVolumeNodes


def generate_3D_model(volume_node, threshold_value=None, min_size=50):
    """
    Generates a 3D surface model from a volume using thresholding & filtering.

    Parameters:
    - volume_node: The input 3D volume node
    - threshold_value: Intensity threshold for surface extraction
    - min_size: Minimum size for objects to keep (remove noise)

    Returns:
    - modelNode: The generated 3D model node
    """
    print("🔍 Generating 3D model...")

    # Get volume array from Slicer
    volume_array = slicer.util.arrayFromVolume(volume_node)

    # Set a dynamic threshold if not provided
    if threshold_value is None:
        threshold_value = np.percentile(volume_array, 99)  # Set threshold to 99th percentile

    # Create segmentation node
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(volume_node, segmentationNode)

    # Apply threshold to filter out noise
    slicer.modules.segmentations.logic().ThresholdSegmentationNode(segmentationNode, threshold_value, 1e9)

    # Remove small components
    segmentationArray = slicer.util.arrayFromVolume(volume_node)
    labeled, num_features = label(segmentationArray, return_num=True)
    sizes = np.bincount(labeled.ravel())
    
    # Keep only objects larger than min_size
    filtered_array = np.isin(labeled, np.where(sizes >= min_size)[0])
    
    # Convert back to slicer volume
    slicer.util.updateVolumeFromArray(volume_node, filtered_array.astype(np.uint8))

    # Extract 3D model
    slicer.modules.segmentations.logic().ExportAllSegmentsToModels(segmentationNode)

    return segmentationNode


def detect_spheres(volume_node, min_radius=3, max_radius=10):
    """
    Detects sphere-like structures in a 3D volume using connected components.

    Parameters:
    - volume_node: The input 3D volume node
    - min_radius, max_radius: Size constraints for spheres

    Returns:
    - centers: List of detected sphere coordinates
    """
    print("🔍 Detecting spheres...")

    # Get numpy array from Slicer volume
    volume_array = slicer.util.arrayFromVolume(volume_node)

    # Connected component labeling
    labeled, num_features = label(volume_array, return_num=True)

    # Measure region properties
    properties = regionprops(labeled)

    # Find spheres based on size & shape
    centers = []
    for prop in properties:
        # Approximate volume filter based on spherical shape
        if min_radius**3 < prop.area < max_radius**3:  
            centers.append(prop.centroid)

    print(f"✅ Found {len(centers)} potential spheres")
    return centers


def create_markups(centers):
    """
    Converts a list of 3D coordinates into a Slicer Markups Node.

    Parameters:
    - centers: List of (x, y, z) coordinates

    Returns:
    - markupsNode: The created Markups node
    """
    print("🎯 Creating markups...")

    markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
    markupsNode.SetName("SphereCenters")

    for c in centers:
        markupsNode.AddFiducial(c[2], c[1], c[0])  # Convert from (z, y, x) to (x, y, z)

    return markupsNode


def merge_2D_3D(centers_2D, centers_3D, distance_threshold=5):
    """
    Merges 2D and 3D detected points by checking proximity.

    Parameters:
    - centers_2D: List of 2D-detected coordinates
    - centers_3D: List of 3D-detected coordinates
    - distance_threshold: Maximum distance to consider a match

    Returns:
    - merged_centers: Final list of merged 3D points
    """
    print("🔗 Merging 2D and 3D detections...")

    merged_centers = centers_3D.copy()  # Start with 3D points

   
    for p2d in centers_2D:
        if all(np.linalg.norm(np.array(p2d) - np.array(p3d)) > distance_threshold for p3d in centers_3D):
            merged_centers.append(p2d)

    print(f"✅ Merged {len(merged_centers)} total points")
    return merged_centers



# # Load the input volume and ROI
# inputVolume = slicer.util.getNode('Enhanced_Laplacian_sharpen_2_ctp.3D')  


# # # Output directory
# outputDir = r'C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Enhance_ctp_tests\\P1\\Enhance_ctp_3D'  

# # # Test the function 
# enhancedVolumeNodes = enhance_ctp_3D(inputVolume,  methods=['nl_means'], outputDir=outputDir)

# # # Access the enhanced volume nodes
# for method, volume_node in enhancedVolumeNodes.items():
#     if volume_node is not None:
#          print(f"Enhanced volume for method '{method}': {volume_node.GetName()}")
#     else:
#          print(f"Enhanced volume for method '{method}': No volume node available.")

#exec(open('C:/Users/rocia/AppData/Local/slicer.org/Slicer 5.6.2/SEEG_module/SEEG_masking/enhance_ctp_3D.py').read())
