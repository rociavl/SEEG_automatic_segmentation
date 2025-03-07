import torch
import torch.nn.functional as F
import numpy as np
import slicer
import os
import vtk

from scipy.ndimage import gaussian_filter, median_filter

def enhance_ctp_pytorch(inputVolume, inputROI=None, methods='all', outputDir=None, device="cuda" if torch.cuda.is_available() else "cpu"):

    volume_array = slicer.util.arrayFromVolume(inputVolume)
    volume_tensor = torch.tensor(volume_array, dtype=torch.float32, device=device)

    if volume_tensor is None or volume_tensor.numel() == 0:
        print("Input volume data is empty or invalid.")
        return None

    if inputROI is not None:
        roi_array = slicer.util.arrayFromVolume(inputROI)
        roi_tensor = torch.tensor(roi_array > 0, dtype=torch.uint8, device=device)  
    else:
        roi_tensor = torch.ones_like(volume_tensor, dtype=torch.uint8, device=device) ç

    roi_volume = volume_tensor * roi_tensor.float()

    enhanced_volumes = {}

    if methods == 'all':
        enhanced_volumes['Gaussian'] = torch.tensor(gaussian_filter(roi_volume.cpu().numpy(), sigma=0.3)).to(device)

        enhanced_volumes['Median'] = torch.tensor(median_filter(enhanced_volumes['Gaussian'].cpu().numpy(), size=3)).to(device)

        gamma = 3
        enhanced_volumes['Gamma'] = torch.pow(enhanced_volumes['Median'], gamma)

        laplacian_kernel = torch.tensor([[[0, 1, 0], [1, -4, 1], [0, 1, 0]]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        enhanced_volumes['Edges'] = F.conv3d(enhanced_volumes['Gamma'].unsqueeze(0).unsqueeze(0), laplacian_kernel, padding=1).squeeze()

        threshold_value = 0.14  
        enhanced_volumes['Threshold'] = (enhanced_volumes['Edges'] > threshold_value).float() * 255

    if outputDir is None:
        outputDir = slicer.app.temporaryPath()
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    enhancedVolumeNodes = {}
    for method_name, enhanced_tensor in enhanced_volumes.items():
        enhanced_image = enhanced_tensor.cpu().numpy().astype(np.uint8)  

        enhancedVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        enhancedVolumeNode.SetName(f"Enhanced_pytorch_{method_name}_{inputVolume.GetName()}")
        enhancedVolumeNode.SetOrigin(inputVolume.GetOrigin())
        enhancedVolumeNode.SetSpacing(inputVolume.GetSpacing())

        # Copy transformation matrix
        ijkToRasMatrix = vtk.vtkMatrix4x4()
        inputVolume.GetIJKToRASMatrix(ijkToRasMatrix)
        enhancedVolumeNode.SetIJKToRASMatrix(ijkToRasMatrix)

        slicer.util.updateVolumeFromArray(enhancedVolumeNode, enhanced_image)

        output_file = os.path.join(outputDir, f"Filtered_pytorch_{method_name}_{inputVolume.GetName()}.nrrd")
        slicer.util.saveNode(enhancedVolumeNode, output_file)
        print(f"Saved {method_name} enhancement as: {output_file}")

        enhancedVolumeNodes[method_name] = enhancedVolumeNode

    return enhancedVolumeNodes
