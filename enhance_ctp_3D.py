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


# Load the input volume and ROI
inputVolume = slicer.util.getNode('Enhanced_Laplacian_sharpen_2_ctp.3D')  


# # Output directory
outputDir = r'C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Enhance_ctp_tests\\P1\\Enhance_ctp_3D'  

# # Test the function 
enhancedVolumeNodes = enhance_ctp_3D(inputVolume,  methods=['nl_means'], outputDir=outputDir)

# # Access the enhanced volume nodes
for method, volume_node in enhancedVolumeNodes.items():
    if volume_node is not None:
         print(f"Enhanced volume for method '{method}': {volume_node.GetName()}")
    else:
         print(f"Enhanced volume for method '{method}': No volume node available.")

#exec(open('C:/Users/rocia/AppData/Local/slicer.org/Slicer 5.6.2/SEEG_module/SEEG_masking/enhance_ctp_3D.py').read())
