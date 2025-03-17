import slicer
import vtk
import numpy as np

def resample_mask(input_mask_name, reference_volume_name, output_mask_name=None, output_path=None):

    try:

        input_mask = slicer.util.getNode(input_mask_name)
        reference_volume = slicer.util.getNode(reference_volume_name)
        if not input_mask or not reference_volume:
            raise ValueError("❌ Error: Could not find input or reference volume in the scene.")

        input_origin = np.array(input_mask.GetOrigin())
        reference_origin = np.array(reference_volume.GetOrigin())
        input_spacing = np.array(input_mask.GetSpacing())
        reference_spacing = np.array(reference_volume.GetSpacing())
        ijk_to_ras_matrix = vtk.vtkMatrix4x4()
        reference_volume.GetIJKToRASMatrix(ijk_to_ras_matrix)

        output_mask = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", output_mask_name)

        parameters = {
            "inputVolume": input_mask,
            "referenceVolume": reference_volume,
            "outputVolume": output_mask,
            "interpolationMode": "NearestNeighbor"  
        }
        slicer.cli.runSync(slicer.modules.resamplescalarvectordwivolume, None, parameters)
        output_mask.SetIJKToRASMatrix(ijk_to_ras_matrix)

        if output_path:
            slicer.util.saveNode(output_mask, output_path)
            print(f"Resampling complete. Saved to {output_path}")

        print(f"Resampled mask '{output_mask_name}' successfully created and aligned.")

    except Exception as e:
        print(f"Error: {e}")


resample_mask("patient6_mask_7", "CTp.3D",  'patient_mask_resampled',r'C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Enhance_ctp_tests\\patient6_resampled_sy_mask_7.nrrd')


#exec(open('C:/Users/rocia/AppData/Local/slicer.org/Slicer 5.6.2/SEEG_module/SEEG_masking/resampler_module_slicer.py').read())