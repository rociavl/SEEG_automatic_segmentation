import slicer 
import vtk

def apply_threshold_to_volume(input_volume_name, threshold_min, threshold_max, output_volume_name=None):

    input_volume = slicer.util.getNode(input_volume_name)
    if not input_volume:
        raise ValueError(f"Input volume '{input_volume_name}' not found")
    
    # Create an output volume name if not provided
    if output_volume_name is None:
        output_volume_name = f"{input_volume_name}_threshold_{threshold_min}_{threshold_max}"
    
    # Create a new volume for the output
    output_volume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', output_volume_name)
    
    # Set up the threshold filter - using vtk module directly
    threshold_filter = vtk.vtkImageThreshold()
    threshold_filter.SetInputData(input_volume.GetImageData())
    threshold_filter.ThresholdBetween(threshold_min, threshold_max)
    threshold_filter.SetInValue(1)  # Value to set for voxels within the threshold
    threshold_filter.SetOutValue(0)  # Value to set for voxels outside the threshold
    threshold_filter.SetOutputScalarTypeToUnsignedChar()  # Output as binary volume
    threshold_filter.Update()
    
    # Set the output image data and copy geometric information from input to output
    output_volume.SetAndObserveImageData(threshold_filter.GetOutput())
    output_volume.CopyOrientation(input_volume)
    
    # Copy the IJK to RAS matrix (ensures correct physical space representation)
    mat = vtk.vtkMatrix4x4()
    input_volume.GetIJKToRASMatrix(mat)
    output_volume.SetIJKToRASMatrix(mat)
    
    # Make sure display is updated
    output_volume.CreateDefaultDisplayNodes()
    
    # Return the new volume
    return output_volume

# Example usage:
inputVolume = slicer.util.getNode('Filtered_th_20_roi_volume_ctp.3D')
if inputVolume:
    # Apply threshold - change these values as needed for your specific case
    threshold_min = 2836 # Example minimum threshold value
    threshold_max = 3071 # Example maximum threshold value
    
    # Apply the thresholding
    thresholded_volume = apply_threshold_to_volume(
        input_volume_name='Filtered_th_20_roi_volume_ctp.3D',
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        output_volume_name='Thresholded_Volume'
    )
    
    print(f"Created thresholded volume: {thresholded_volume.GetName()}")
else:
    print("Input volume 'Filtered_th_20_roi_volume_ctp.3D' not found in the scene.")

#exec(open(r"C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Threshold_mask\threshold_roi.py").read())