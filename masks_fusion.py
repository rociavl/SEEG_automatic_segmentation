import slicer
import numpy as np
import SimpleITK as sitk
import vtk
from scipy import stats

def load_mask_to_array(mask_node_name):
    mask_node = slicer.util.getNode(mask_node_name)
    if mask_node is None:
        raise ValueError(f"Mask node {mask_node_name} not found.")
    
    mask_array = slicer.util.arrayFromVolume(mask_node)  
    origin = mask_node.GetOrigin()
    spacing = mask_node.GetSpacing()

    ijkToRasMatrix = vtk.vtkMatrix4x4()
    mask_node.GetIJKToRASMatrix(ijkToRasMatrix)
    
    return mask_array, origin, spacing, ijkToRasMatrix

def binarize_mask(mask_array):
    return np.where(mask_array > 0, 1, 0).astype(np.uint8)

def detect_outliers(mask_arrays):
    stacked_masks = np.stack(mask_arrays, axis=-1)
    z_scores = np.abs(stats.zscore(stacked_masks, axis=-1, nan_policy='omit'))
    return z_scores > 2  

def voting_weighted_fusion(mask_arrays, weights=None):
    if weights is None:
        weights = np.ones(len(mask_arrays))  
    
    weights = np.array(weights)
    weights = weights / np.sum(weights)  

    fused_mask = np.zeros_like(mask_arrays[0], dtype=np.float32)
    outliers = detect_outliers(mask_arrays)

    for i, mask_array in enumerate(mask_arrays):
        adjusted_weight = weights[i] * (1 - np.mean(outliers[..., i]))  
        fused_mask += mask_array * adjusted_weight  

    fused_mask = np.where(fused_mask >= 0.5, 1, 0).astype(np.uint8)  
    return fused_mask

def create_fused_volume_node(fused_mask, input_volume, output_node_name, output_dir=None):

    enhancedVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", output_node_name)

    enhancedVolumeNode.SetOrigin(input_volume.GetOrigin())
    enhancedVolumeNode.SetSpacing(input_volume.GetSpacing())
    ijkToRasMatrix = vtk.vtkMatrix4x4()
    input_volume.GetIJKToRASMatrix(ijkToRasMatrix)
    enhancedVolumeNode.SetIJKToRASMatrix(ijkToRasMatrix)

    slicer.util.updateVolumeFromArray(enhancedVolumeNode, fused_mask)

    if output_dir:
        file_path = f"{output_dir}/{output_node_name}.nrrd"
        slicer.util.saveNode(enhancedVolumeNode, file_path)
        print(f"Fused mask saved to: {file_path}")

    return enhancedVolumeNode

def process_masks(mask_node_names, weights=None, output_node_name="FusedMask", output_dir=None):

    mask_arrays, input_volume = [], None

    for i, mask_node_name in enumerate(mask_node_names):
        mask_array, origin, spacing, matrix = load_mask_to_array(mask_node_name)
        mask_arrays.append(binarize_mask(mask_array))

        # Use first mask's volume node as reference
        if i == 0:
            input_volume = slicer.util.getNode(mask_node_name)

    fused_mask = voting_weighted_fusion(mask_arrays, weights)

    create_fused_volume_node(fused_mask, input_volume, output_node_name, output_dir)

mask_node_names = [
    "Enhanced_th45_DESCARGAR_FT_gaussian_3_0.540_CTp.3D",
    "Enhanced_th45_DESCARGAR_FT_TOPHAT_0.490_CTp.3D",
    "Enhanced_th45_DESCARGAR_FT_ERODE_2_133_CTp.3D"
]  
weights = [0.4, 0.3, 0.3]  
output_dir = r'C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Enhance_ctp_tests\\P2' 
process_masks(mask_node_names, weights, output_dir=output_dir, output_node_name="patient2_mask_electrodes_1")

#exec(open('C:/Users/rocia/AppData/Local/slicer.org/Slicer 5.6.2/SEEG_module/SEEG_masking/masks_fusion.py').read())