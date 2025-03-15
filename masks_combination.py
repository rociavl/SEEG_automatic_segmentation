import slicer
import numpy as np

def combine_binary_masks_slicer(mask_node_names, method="majority", weights=None):

    mask_list = [slicer.util.arrayFromVolume(slicer.util.getNode(name)) for name in mask_node_names]
    
    stack = np.stack(mask_list, axis=0)  # Shape (num_masks, Z, Y, X)

    if method == "majority":
        combined = np.sum(stack, axis=0) > (len(mask_list) / 2)
    
    elif method == "weighted" and weights is not None:
        weights = np.array(weights) / np.sum(weights)  # Normalize weights
        weighted_sum = np.sum([w * mask for mask, w in zip(stack, weights)], axis=0)
        combined = weighted_sum > 0.5
    
    elif method == "any":
        combined = np.any(stack, axis=0)
    
    elif method == "all":
        combined = np.all(stack, axis=0)
    
    else:
        raise ValueError("Invalid method. Choose from: 'majority', 'weighted', 'any', or 'all'.")

    combined_mask_array = combined.astype(np.uint8)

    combined_mask_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "CombinedMask")

    reference_node = slicer.util.getNode(mask_node_names[0])
    combined_mask_node.CopyOrientation(reference_node)

    slicer.util.updateVolumeFromArray(combined_mask_node, combined_mask_array)

    return combined_mask_node

mask_nodes = ['Mask1', 'Mask2', 'Mask3']  
weights = [0.5, 0.3, 0.2]  

combined_mask_node = combine_binary_masks_slicer(mask_nodes, method="weighted", weights=weights)

print(f"Combined mask stored in node: {combined_mask_node.GetName()}")
