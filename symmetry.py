import slicer
import numpy as np
import vtk
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, generate_binary_structure

def create_symmetric_mask(mask_name, axis='x', output_mask_name="Fully_Connected_Symmetric_Mask", output_path=None):
    try:
        input_mask = slicer.util.getNode(mask_name)
        if not input_mask:
            raise ValueError("Error: Could not find the input mask in the scene.")

        mask_array = slicer.util.arrayFromVolume(input_mask)

        spacing = np.array(input_mask.GetSpacing()) 
        origin = np.array(input_mask.GetOrigin())  
        dims = np.array(input_mask.GetImageData().GetDimensions())  
        direction_matrix = vtk.vtkMatrix4x4()
        input_mask.GetIJKToRASDirectionMatrix(direction_matrix)


        direction_matrix_np = np.array([[direction_matrix.GetElement(i, j) for j in range(3)] for i in range(3)])

        axis_map = {'x': 2, 'y': 1, 'z': 0}
        if axis not in axis_map:
            raise ValueError("Error: Axis must be 'x', 'y', or 'z'.")
        axis_index = axis_map[axis]

        flipped_mask = np.flip(mask_array, axis=axis_index)
        mirror_origin = origin.copy()
        mirror_origin[axis_index] = origin[axis_index] - (dims[axis_index] - 1) * spacing[axis_index] * direction_matrix_np[axis_index, axis_index]
        merged_mask = np.logical_or(mask_array, flipped_mask).astype(np.uint8)

        structure = generate_binary_structure(3, 3)  
        dilated_mask = binary_dilation(merged_mask, structure=structure, iterations=2)
        closed_mask = binary_closing(dilated_mask, structure=structure, iterations=3)
        filled_mask = binary_fill_holes(closed_mask).astype(np.uint8)

        output_mask = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", output_mask_name)
        output_mask.SetSpacing(spacing)
        output_mask.SetOrigin(mirror_origin) 
        output_mask.SetIJKToRASDirectionMatrix(direction_matrix)

        slicer.util.updateVolumeFromArray(output_mask, filled_mask)

        if output_path:
            slicer.util.saveNode(output_mask, output_path)
            print(f"Fully connected symmetric mask saved to {output_path}")

        print(f"Fully connected symmetric mask created: '{output_mask_name}'")

    except Exception as e:
        print(f"Error: {e}")

create_symmetric_mask("patient7_mask_2", axis='x', output_mask_name="patient7_symmetric_mask", output_path= r'C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Enhance_ctp_tests\\patient7_symmetry_mask.nrrd')

#exec(open('C:/Users/rocia/AppData/Local/slicer.org/Slicer 5.6.2/SEEG_module/SEEG_masking/symmetry.py').read())