import numpy as np
import slicer
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt

def register_ct_images(pre_ct_node, post_ct_node):

    pre_ct_array = slicer.util.arrayFromVolume(pre_ct_node)
    post_ct_array = slicer.util.arrayFromVolume(post_ct_node)
    pre_ct_image = sitk.GetImageFromArray(pre_ct_array)
    post_ct_image = sitk.GetImageFromArray(post_ct_array)

 
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattes()
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescentLineSearch()
    reg.SetNumberOfIterations(200)  
    reg.SetRelaxationFactor(0.5)

    initial_transform = sitk.AffineTransform(pre_ct_image.GetDimension())  
    reg.SetInitialTransform(initial_transform)
    final_transform = reg.Execute(pre_ct_image, post_ct_image)
    pre_ct_resampled_image = sitk.Resample(pre_ct_image, post_ct_image, final_transform, sitk.sitkLinear, 0.0)
    pre_ct_resampled_array = sitk.GetArrayFromImage(pre_ct_resampled_image)
    pre_ct_resampled_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', 'RegisteredPreCT')
    slicer.util.updateVolumeFromArray(pre_ct_resampled_node, pre_ct_resampled_array)

    return pre_ct_resampled_node, post_ct_node


def subtract_images(post_ct_node, pre_ct_resampled_node):
    """
    Subtract the pre-CT from the post-CT to highlight differences.
    """

    post_ct_array = slicer.util.arrayFromVolume(post_ct_node)
    pre_ct_resampled_array = slicer.util.arrayFromVolume(pre_ct_resampled_node)
    subtracted_array = np.abs(post_ct_array - pre_ct_resampled_array)  
    subtracted_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', 'SubtractedCT')
    slicer.util.updateVolumeFromArray(subtracted_node, subtracted_array)
    
    return subtracted_node


def apply_threshold(subtracted_node, threshold_value=1600):
    """
    Apply thresholding to the subtracted image to highlight high-intensity areas (likely electrode contacts).
    """

    subtracted_array = slicer.util.arrayFromVolume(subtracted_node)
    thresholded_array = np.where(subtracted_array >= threshold_value, 1, 0)
    thresholded_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', 'ThresholdedCT')
    slicer.util.updateVolumeFromArray(thresholded_node, thresholded_array)

    return thresholded_node


def save_and_visualize(thresholded_node, output_path="contacts_highlighted.nrrd"):
    """
    Save the thresholded result as an .nrrd file and visualize the first slice.
    """
    
    slicer.util.saveNode(thresholded_node, output_path)
    print(f"Thresholded result saved to: {output_path}")
    thresholded_array = slicer.util.arrayFromVolume(thresholded_node)
    plt.imshow(thresholded_array[0, :, :], cmap='gray')
    plt.title("Thresholded Electrode Contacts")
    plt.axis("off")
    plt.show()


def main():
    pre_ct_node = slicer.util.getNode('PreCT')  
    post_ct_node = slicer.util.getNode('PostCT')  
    pre_ct_resampled_node, post_ct_node = register_ct_images(pre_ct_node, post_ct_node)
    subtracted_node = subtract_images(post_ct_node, pre_ct_resampled_node)
    thresholded_node = apply_threshold(subtracted_node, threshold_value=1600)
    save_and_visualize(thresholded_node, output_path="contacts_highlighted.nrrd")

if __name__ == "__main__":
    main()
