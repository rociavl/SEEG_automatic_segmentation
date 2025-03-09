import slicer
import numpy as np
import vtk
from scipy.ndimage import zoom, binary_dilation
from scipy.ndimage import generate_binary_structure

def get_bounds_from_volume(volume_node):
    """Calculate RAS bounds from a volume node using origin, spacing, and dimensions."""
    spacing = np.array(volume_node.GetSpacing())
    origin = np.array(volume_node.GetOrigin())
    dims = np.array(volume_node.GetImageData().GetDimensions())

    direction_matrix = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASDirectionMatrix(direction_matrix)

    corners_ijk = [
        [0, 0, 0], [dims[0]-1, 0, 0], [0, dims[1]-1, 0], [0, 0, dims[2]-1],
        [dims[0]-1, dims[1]-1, 0], [dims[0]-1, 0, dims[2]-1],
        [0, dims[1]-1, dims[2]-1], [dims[0]-1, dims[1]-1, dims[2]-1]
    ]

    bounds = [float('inf'), float('-inf'), float('inf'), float('-inf'), float('inf'), float('-inf')]
    for corner in corners_ijk:
        ijk_point = np.array(corner) * spacing
        ras_point = np.array([0, 0, 0, 1], dtype=float)
        for i in range(3):
            ras_point[i] = origin[i]
            for j in range(3):
                ras_point[i] += direction_matrix.GetElement(i, j) * ijk_point[j]
        bounds[0] = min(bounds[0], ras_point[0])  # X min
        bounds[1] = max(bounds[1], ras_point[0])  # X max
        bounds[2] = min(bounds[2], ras_point[1])  # Y min
        bounds[3] = max(bounds[3], ras_point[1])  # Y max
        bounds[4] = min(bounds[4], ras_point[2])  # Z min
        bounds[5] = max(bounds[5], ras_point[2])  # Z max
    return bounds

def get_mask_extent(mask_array):
    """Calculate the extent of non-zero voxels in the mask (number of non-zero voxels)."""
    return np.sum(mask_array > 0)

def get_mask_centroid(mask_array):
    """Calculate the centroid of non-zero voxels in the mask."""
    nz_indices = np.nonzero(mask_array)
    if len(nz_indices[0]) == 0:
        return np.array([0, 0, 0])
    return np.mean(nz_indices, axis=1)

def resample_roi(inputVolumeName, inputROIName, outputPath, output_dimensions=(180, 256, 256), 
                 max_dilation_iterations=5, tolerance=0.95, z_offset=0, verbose=True):
    try:
        # Fetch nodes
        inputVolume = slicer.util.getNode(inputVolumeName)
        inputROI = slicer.util.getNode(inputROIName)

        if inputVolume is None or inputROI is None:
            raise ValueError(f"❌ Error: Unable to fetch '{inputVolumeName}' or '{inputROIName}' from the scene.")

        if not inputVolume.GetImageData() or not inputROI.GetImageData():
            raise ValueError(f"❌ Error: Input volume or ROI has no image data.")

        volume_spacing = np.array(inputVolume.GetSpacing())
        volume_origin = np.array(inputVolume.GetOrigin())
        volume_dims = np.array(inputVolume.GetImageData().GetDimensions())

        roi_spacing = np.array(inputROI.GetSpacing())
        roi_origin = np.array(inputROI.GetOrigin())
        roi_dims = np.array(inputROI.GetImageData().GetDimensions())

        volume_direction_matrix = vtk.vtkMatrix4x4()
        inputVolume.GetIJKToRASDirectionMatrix(volume_direction_matrix)
        roi_direction_matrix = vtk.vtkMatrix4x4()
        inputROI.GetIJKToRASDirectionMatrix(roi_direction_matrix)

        volume_dir = np.array([[volume_direction_matrix.GetElement(i, j) for j in range(3)] for i in range(3)])
        roi_dir = np.array([[roi_direction_matrix.GetElement(i, j) for j in range(3)] for i in range(3)])

        if verbose:
            print(f"📌 Volume: Spacing={volume_spacing}, Origin={volume_origin}, Dims={volume_dims}")
            print(f"📌 ROI: Spacing={roi_spacing}, Origin={roi_origin}, Dims={roi_dims}")
            print(f"📌 Volume Direction Matrix (RAS):\n{volume_dir}")
            print(f"📌 ROI Direction Matrix (RAS):\n{roi_dir}")


        volume_array = slicer.util.arrayFromVolume(inputVolume)
        roi_array = slicer.util.arrayFromVolume(inputROI)

        if verbose:
            print(f"📌 Volume Array Shape: {volume_array.shape}")
            print(f"📌 ROI Array Shape: {roi_array.shape}")


        original_extent = get_mask_extent(roi_array)
        original_centroid = get_mask_centroid(roi_array)
        if verbose:
            print(f"📌 Original ROI Extent (non-zero voxels): {original_extent}")
            print(f"📌 Original ROI Centroid (IJK): {original_centroid}")

        zoom_factors = volume_spacing / roi_spacing
        if not np.allclose(zoom_factors, 1.0, atol=1e-6):
            resampled_roi_array = zoom(roi_array, zoom_factors, order=1)
        else:
            resampled_roi_array = roi_array.copy()

        resampled_roi_array = resampled_roi_array[::-1, ::-1, :]
        if verbose:
            print(f"📌 Applied manual flip [::-1, ::-1, :] to match visual alignment")

        axis_flips = np.sign(np.diag(volume_dir @ np.linalg.inv(roi_dir)))
        if verbose:
            print(f"📌 Axis Flips (Z, Y, X): {axis_flips}")

        volume_bounds = get_bounds_from_volume(inputVolume)
        roi_bounds = get_bounds_from_volume(inputROI)

        if verbose:
            print(f"📌 Volume Bounds (RAS): {volume_bounds}")
            print(f"📌 ROI Bounds (RAS): {roi_bounds}")

        offset_ras = volume_origin - roi_origin
        offset_voxel = offset_ras / volume_spacing
        if not np.allclose(offset_voxel, 0, atol=1e-6):
            for axis, shift in enumerate(offset_voxel):
                resampled_roi_array = np.roll(resampled_roi_array, int(shift), axis=axis)

        original_shape = roi_array.shape  
        current_shape = resampled_roi_array.shape

        if current_shape != original_shape:
            pad_or_crop = [(0, 0), (0, 0), (0, 0)]
            for i, (orig_dim, curr_dim) in enumerate(zip(original_shape, current_shape)):
                diff = orig_dim - curr_dim
                if diff > 0:  # Pad
                    pad_before = diff // 2
                    pad_after = diff - pad_before
                    pad_or_crop[i] = (pad_before, pad_after)
                elif diff < 0:  # Crop
                    start = -diff // 2
                    end = start + orig_dim
                    if i == 0:  # Z-axis
                        resampled_roi_array = resampled_roi_array[start:end, :, :]
                    elif i == 1:  # Y-axis
                        resampled_roi_array = resampled_roi_array[:, start:end, :]
                    else:  # X-axis
                        resampled_roi_array = resampled_roi_array[:, :, start:end]
            if any(p[0] > 0 or p[1] > 0 for p in pad_or_crop):
                resampled_roi_array = np.pad(resampled_roi_array, pad_or_crop, mode='constant', constant_values=0)

        ##### dilation#####
        structure = generate_binary_structure(10, 2)  
        current_extent = get_mask_extent(resampled_roi_array)
        iterations = 0
        while (current_extent < original_extent * tolerance) and (iterations < max_dilation_iterations):
            resampled_roi_array = binary_dilation(resampled_roi_array, structure=structure).astype(np.uint8)
            current_extent = get_mask_extent(resampled_roi_array)
            iterations += 1
            if verbose:
                print(f"📌 Dilation Iteration {iterations}, Current Extent: {current_extent}")

        if verbose:
            print(f"📌 Final ROI Extent (non-zero voxels): {current_extent}")
            print(f"📌 Total Dilation Iterations: {iterations}")

        resampled_roi_array = (resampled_roi_array > 0.5).astype(np.uint8)


        target_shape = volume_array.shape
        current_shape = resampled_roi_array.shape
        if current_shape != target_shape:
            z_diff = current_shape[0] - target_shape[0]
            z_start = (z_diff // 2) + z_offset
            z_end = z_start + target_shape[0]
            if z_diff > 0:  # Crop
                resampled_roi_array = resampled_roi_array[z_start:z_end, :, :]
            elif z_diff < 0:  # Pad
                pad_before = -z_diff // 2
                pad_after = -z_diff - pad_before
                resampled_roi_array = np.pad(resampled_roi_array, 
                                           ((pad_before, pad_after), (0, 0), (0, 0)), 
                                           mode='constant', constant_values=0)

        if verbose:
            print(f"📌 Aligned ROI Shape: {resampled_roi_array.shape}")

        aligned_roi_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "AlignedROI")
        slicer.util.updateVolumeFromArray(aligned_roi_node, resampled_roi_array)

        aligned_roi_node.SetSpacing(volume_spacing)
        aligned_roi_node.SetOrigin(volume_origin)
        aligned_roi_node.SetIJKToRASDirectionMatrix(volume_direction_matrix)

        if verbose:
            print(f"📌 Aligned ROI: Spacing={aligned_roi_node.GetSpacing()}, Origin={aligned_roi_node.GetOrigin()}")

        slicer.util.saveNode(aligned_roi_node, outputPath)
        print(f"✅ Saved resampled mask to {outputPath}")

    except Exception as e:
        slicer.util.errorDisplay(f"❌ Error: {str(e)}")
        raise

resample_roi('CTp.3D', 'patient6_mask', r'C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Enhance_ctp_tests\\patient6_resampled_mask.nrrd', 
             max_dilation_iterations=10, tolerance=0.95, z_offset=0, verbose=True)