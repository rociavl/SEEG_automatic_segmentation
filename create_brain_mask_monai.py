import slicer
import os
import vtk
import numpy as np
import torch
import monai
from monai.transforms import Compose, ScaleIntensity, ToTensor, Resize, SpatialPad
from monai.networks.nets import UNet

def create_brain_mask_monai(inputVolume, outputDir=None):
    """
    Generate a binary brain mask using a MONAI deep learning model. It doens't work yet!!
    
    Args:
        inputVolume (vtkMRMLScalarVolumeNode): The input MRI scan.
        outputDir (str): Directory to save the brain mask (optional).
    
    Returns:
        vtkMRMLScalarVolumeNode: The binary brain mask volume.
    """
    if not inputVolume:
        slicer.util.errorDisplay("No input volume selected")
        return None
    print("✅ Input validation passed.")

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

    # Check the tensor shape before applying any transformations
    print("Input tensor shape before padding:", input_tensor.shape)

    # Extract the spatial size (height, width, depth) for padding (ignore the batch and channel dimensions)
    spatial_size = input_tensor.shape[2:]  # (H, W, D)
    print(f"Extracted spatial_size: {spatial_size}")

    # Ensure the spatial size is a multiple of 16 (or another multiple based on U-Net requirements)
    target_size = [
    ((s + 15) // 16) * 16 for s in spatial_size]
    print(f"Computed target_size for padding: {target_size}")

    # Add debugging information to check the dimensions
    print("Tensor shape with batch and channel: ", input_tensor.shape)

    # Apply spatial padding (only for spatial dimensions, not batch/channel)
    pad_transform = SpatialPad(spatial_size=target_size)  # SpatialPad expects 3D size
    padded_input_tensor = pad_transform(input_tensor[0])

    # Check the tensor shape after padding
    print("Padded tensor shape:", padded_input_tensor.shape)

    padded_input_tensor = padded_input_tensor.unsqueeze(0)
    print("Padded tensor unsqueeze shape:", padded_input_tensor.shape)

    # Define a MONAI UNet model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=[16, 32, 64, 128],
        strides=[2, 2, 2],
    ).to(device)

    # Run inference
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
    outputVolumeNode.SetName(f"BrainMask_{inputVolume.GetName()}")

    # Update volume array
    slicer.util.updateVolumeFromArray(outputVolumeNode, output_array)

    # Copy spatial properties (origin, spacing, direction)
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
        outputPath = os.path.join(outputDir, f"brain_mask_{inputVolume.GetName()}.nrrd")
        slicer.util.saveNode(outputVolumeNode, outputPath)
        print(f"💾 Brain mask saved to: {outputPath}")

    print("✅ Brain mask created successfully using MONAI.")
    return outputVolumeNode




# Exampl
inputVolumeNode = slicer.util.getNode("3Dps.3D")  
outputDirectory = r'C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Brain_mask'   
brain_mask_node = create_brain_mask_monai(inputVolumeNode, outputDir=outputDirectory)

#exec(open('C:/Users/rocia/AppData/Local/slicer.org/Slicer 5.6.2/SEEG_module/SEEG_masking/create_brain_mask_monai.py').read())
