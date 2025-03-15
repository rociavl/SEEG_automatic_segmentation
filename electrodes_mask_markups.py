import SimpleITK as sitk
import slicer
import numpy as np
import csv

def get_fiducials_from_slicer(node_name):
    node = slicer.util.getNode(node_name)  
    fiducial_data = []
    
    for i in range(node.GetNumberOfFiducials()):
        label = node.GetNthFiducialLabel(i)
        position = [0.0, 0.0, 0.0]
        node.GetNthFiducialPosition(i, position)
    
        fiducial_data.append((label, position[0], position[1], position[2]))
    
    return fiducial_data


def list_fiducials(fiducial_data):
    print("List of Fiducials:")
    for idx, (label, x, y, z) in enumerate(fiducial_data):
        print(f"{idx + 1}. Label: {label}, Coordinates (RAS): ({x}, {y}, {z})")


def create_electrode_mask_from_fiducials_and_save_csv(fiducial_data, volume_path, output_filename, csv_filename, radius_mm=0.4):

    print("Loading volume from Slicer...")
    image = sitk.ReadImage(volume_path)
    mask_image = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    mask_image.CopyInformation(image)

    successful_fiducials = 0
    fiducial_output_data = []  
    
    for idx, (label, x, y, z) in enumerate(fiducial_data):
        print(f"Processing Fiducial {idx+1}: {label} - Coordinates (RAS): ({x}, {y}, {z})")
        
        # Flip coordinates (RAS to LPS)
        flipped_x = -x  # Flip the X (Right/Left) axis
        flipped_y = -y  # Flip the Y (Anterior/Posterior) axis
        
        try:

            sphere = sitk.Image(image.GetSize(), sitk.sitkUInt8)
            sphere.CopyInformation(image)

            point_idx = image.TransformPhysicalPointToIndex((flipped_x, flipped_y, z))
            print(f"Fiducial {label} at RAS ({x}, {y}, {z}) converted to flipped index {point_idx}")
            sphere[point_idx] = 1

            distance_map = sitk.SignedMaurerDistanceMap(sphere, insideIsPositive=False, 
                                                       squaredDistance=False, useImageSpacing=True)
            sphere = sitk.BinaryThreshold(distance_map, -float('inf'), radius_mm, 1, 0)

            mask_image = sitk.Or(mask_image, sphere)
            successful_fiducials += 1

            fiducial_output_data.append([label, flipped_x, flipped_y, z])

        except Exception as e:
            print(f"Error processing fiducial {idx+1}: {e}")
    
    print(f"Saving fiducial data to CSV: {csv_filename}")
    with open(csv_filename, mode='w', newline='') as csvfile:
        fieldnames = ['Label', 'X (LPS)', 'Y (LPS)', 'Z (Superior)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for fiducial in fiducial_output_data:
            writer.writerow({'Label': fiducial[0], 'X (LPS)': fiducial[1], 'Y (LPS)': fiducial[2], 'Z (Superior)': fiducial[3]})
    
    print(f"Electrode mask creation completed. Successfully placed {successful_fiducials} out of {len(fiducial_data)} fiducials.")
    print(f"Saving electrode mask to {output_filename}...")
    sitk.WriteImage(mask_image, output_filename)

# Get fiducials from two different markups nodes
# fiducial_data_node_1 = get_fiducials_from_slicer("real-P")  
# fiducial_data_node_2 = get_fiducials_from_slicer("real-WM")  

# combined_fiducial_data = fiducial_data_node_1 + fiducial_data_node_2  

# list_fiducials(combined_fiducial_data)


# volume_path = r"C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Maks_tests\\P1_brain_mask_25.nrrd"
# output_filename = r"C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\P1_electrode_fiducials.nrrd"
# csv_filename = r"C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\P1_electrode_fiducials.csv"


# create_electrode_mask_from_fiducials_and_save_csv(combined_fiducial_data, volume_path, output_filename,csv_filename, radius_mm=0.4)


#exec(open('C:/Users/rocia/AppData/Local/slicer.org/Slicer 5.6.2/SEEG_module/SEEG_masking/electrodes_mask_markups.py').read())
