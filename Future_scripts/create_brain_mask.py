import slicer
import vtk
import numpy as np
import SimpleITK as sitk
from skimage import morphology, filters
from scipy import ndimage
import os
import slicer
import os
import torch
import monai
import numpy as np
import subprocess  # Add subprocess to run the training script

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

def create_brain_mask_monai(inputVolume=None, inputNrrd=None, outputDir=None, train_model=False, dataset=None):
    """
    Generate a binary brain mask using a MONAI deep learning model, with optional training. It's in progress!!
    
    Args:
        inputVolume (vtkMRMLScalarVolumeNode): The input MRI scan (optional).
        outputDir (str): Directory to save the brain mask (optional).
        train_model (bool): Whether to train the model (optional, default is False).
        dataset (str): Path to training dataset (if train_model is True).
    
    Returns:
        vtkMRMLScalarVolumeNode: The binary brain mask volume.
    """
    if not inputVolume and not inputNrrd:
        slicer.util.errorDisplay("No input volume selected")
        return None
    print("✅ Input validation passed.")

    # Load NRRD file if provided
    if inputNrrd:
        input_array = load_nrrd_image(inputNrrd)
    else:
        # Convert inputVolume to numpy array (MONAI works with numpy arrays)
        input_array = slicer.util.arrayFromVolume(inputVolume)

    # Ensure input_array has the right dimensions (C, H, W, D)
    if input_array.ndim == 3:  # If the array is 3D (H, W, D), add a channel dimension
        input_array = np.expand_dims(input_array, axis=0)  # (1, H, W, D)
        input_array = np.expand_dims(input_array, axis=0)  # (1, 1, H, W, D)
    elif input_array.ndim == 4:  # If it's already 4D (C, H, W, D), proceed as is
        input_array = np.expand_dims(input_array, axis=0)  # Add a batch dimension
    else:
        slicer.util.errorDisplay("Input volume shape is not supported.")
        return None
    
    # Normalize intensity (scale to 0-1 for MONAI)
    input_array = input_array.astype(np.float32)
    input_array = input_array / np.max(input_array)  # Scaling intensity

    # Convert the numpy array to a PyTorch tensor
    input_tensor = torch.tensor(input_array)

    # Define the target size to ensure dimensions are a multiple of 16 (or any multiple required by U-Net)
    spatial_size = input_tensor.shape[2:]  # (H, W, D)
    target_size = [((s + 15) // 16) * 16 for s in spatial_size]

    # Apply spatial padding (only for spatial dimensions, not batch/channel)
    pad_transform = SpatialPad(spatial_size=target_size)  
    padded_input_tensor = pad_transform(input_tensor[0])

    padded_input_tensor = padded_input_tensor.unsqueeze(0)  # Add batch dimension

    # Define a MONAI UNet model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=[16, 32, 64, 128],
        strides=[2, 2, 2],
    ).to(device)

    # If training is required, we call the training script
    if train_model:
        if not dataset:
            slicer.util.errorDisplay("No dataset provided for training")
            return None
        print("Training model with provided dataset...")
        
        # Call the training script (train_model_mask.py)
        subprocess.call(["python", "train_model_mask.py", dataset])
    
    # Run inference (or trained model if train_model is True)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        padded_input_tensor = padded_input_tensor.to(device)
        output_tensor = model(padded_input_tensor)

    # Convert output tensor back to numpy array (use only the first channel)
    output_array = output_tensor.cpu().numpy()[0, 0]  # Take first channel output

    # Threshold the output to get binary mask (0 or 1)
    output_array = (output_array > 0.5).astype(np.uint8)

    # Create a new Slicer volume node for the binary mask
    outputVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    outputVolumeNode.SetName(f"BrainMask_{inputVolume.GetName() if inputVolume else os.path.basename(inputNrrd)}")

    # Update volume array
    slicer.util.updateVolumeFromArray(outputVolumeNode, output_array)

    # Copy spatial properties (origin, spacing, direction)
    if inputVolume:
        outputVolumeNode.SetOrigin(inputVolume.GetOrigin())
        outputVolumeNode.SetSpacing(inputVolume.GetSpacing())

        # Copy the transformation matrix using VTK
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

    # Save Output Brain Mask (NRRD format)
    if outputDir:
        outputPath = os.path.join(outputDir, f"brain_mask_{inputVolume.GetName() if inputVolume else os.path.basename(inputNrrd)}.nrrd")
        slicer.util.saveNode(outputVolumeNode, outputPath)
        print(f"💾 Brain mask saved to: {outputPath}")

    print("✅ Brain mask created successfully using MONAI.")
    return outputVolumeNode

# Exampl
inputVolumeNode = slicer.util.getNode("3Dps.3D")  
outputDirectory = r'C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Brain_mask'   
brain_mask_node = create_brain_mask_monai(inputVolumeNode, outputDir=outputDirectory)

#exec(open('C:/Users/rocia/AppData/Local/slicer.org/Slicer 5.6.2/SEEG_module/SEEG_masking/create_brain_mask_monai.py').read())