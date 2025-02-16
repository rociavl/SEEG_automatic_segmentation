import slicer
import vtk
import numpy as np
import SimpleITK as sitk
from skimage import morphology, filters
from scipy import ndimage
import os

def create_brain_mask(inputVolume, outputDir=None):
    """
    Generate a binary brain mask from a regular scan volume. It doesn't work yet!!!!

    Args:
        inputVolume (vtkMRMLScalarVolumeNode): The input CT/MRI scan.
        outputDir (str): Directory to save the brain mask (optional).

    Returns:
        vtkMRMLScalarVolumeNode: The binary brain mask volume.
    """

    # Validate input
    if not inputVolume:
        slicer.util.errorDisplay("No input volume selected")
        return None
    print(" Input validation passed.")

    # Extract Array from Input Volume
    print(f"Loading volume: {inputVolume.GetName()}")
    inputArray = slicer.util.arrayFromVolume(inputVolume)
    if inputArray is None:
        slicer.util.errorDisplay("Failed to extract volume array.")
        return None
    print(f"📏 Volume shape: {inputArray.shape}, dtype: {inputArray.dtype}")

    # Apply Adaptive Thresholding
    print("Applying adaptive thresholding...")
    
    # Use Otsu's method to find a threshold
    otsu_threshold = filters.threshold_otsu(inputArray)
    
    # Apply threshold to get an initial binary mask
    binary_mask = inputArray > otsu_threshold

    # 4. Apply Morphological Operations
    print(" Refining with morphological operations...")

    # Morphological closing to fill gaps
    struct_elem = morphology.ball(5)
    closed_mask = morphology.binary_closing(binary_mask, struct_elem)

    # Morphological opening to remove small noise
    opened_mask = morphology.binary_opening(closed_mask, struct_elem)

    # Extract the Largest Connected Component (Brain)
    print("🧠 Extracting the largest connected component...")
    labeled_mask, num_features = ndimage.label(opened_mask)
    region_sizes = np.bincount(labeled_mask.ravel())

    # Ignore background (label 0)
    region_sizes[0] = 0
    largest_region = np.argmax(region_sizes)  # Find largest connected component
    brain_mask = (labeled_mask == largest_region)

    # Align Mask with Input Volume
    print(" Aligning mask with input volume...")
    brain_mask = brain_mask.astype(np.uint8)  # Convert to uint8 for display

    outputVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    outputVolumeNode.SetName(f"BrainMask_{inputVolume.GetName()}")

    # Update volume array
    slicer.util.updateVolumeFromArray(outputVolumeNode, brain_mask)

    # Copy spatial properties (origin, spacing, direction)
    outputVolumeNode.SetOrigin(inputVolume.GetOrigin())
    outputVolumeNode.SetSpacing(inputVolume.GetSpacing())

    # Fix: Correctly copy the transformation matrix using VTK
    matrix = vtk.vtkMatrix4x4()
    inputVolume.GetIJKToRASMatrix(matrix)
    outputVolumeNode.SetIJKToRASMatrix(matrix)

    # Retain transformations
    outputVolumeNode.SetAndObserveTransformNodeID(inputVolume.GetTransformNodeID())

    # Copy Display Properties
    inputDisplayNode = inputVolume.GetDisplayNode()
    if inputDisplayNode:
        outputDisplayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeDisplayNode")
        outputDisplayNode.Copy(inputDisplayNode)
        outputVolumeNode.SetAndObserveDisplayNodeID(outputDisplayNode.GetID())

    # Save Output Brain Mask
    if outputDir:
        outputPath = os.path.join(outputDir, f"brain_mask_{inputVolume.GetName()}.nrrd")
        slicer.util.saveNode(outputVolumeNode, outputPath)
        print(f"💾 Brain mask saved to: {outputPath}")

    print("Brain mask creation completed successfully.")
    return outputVolumeNode


# Example usage:
inputVolumeNode = slicer.util.getNode("3Dps.3D")  
outputDirectory = r'C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Brain_mask'   
brain_mask_node = create_brain_mask(inputVolumeNode, outputDir=outputDirectory)

#exec(open('C:/Users/rocia/AppData/Local/slicer.org/Slicer 5.6.2/SEEG_module/SEEG_masking/create_brain_mask.py').read())
