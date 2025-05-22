import csv
import SimpleITK as sitk
import os

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
            #print(f"Fiducial {label} at RAS ({x}, {y}, {z}) converted to flipped index {point_idx}")
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

def read_fiducial_csv(csv_path):
    fiducials = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header if present
        for row in reader:
            # Convert to float, handling potential whitespace
            label = row[0].strip()
            x = float(row[1].strip())
            y = float(row[2].strip())
            z = float(row[3].strip())
            fiducials.append((label, x, y, z))
    return fiducials

def main():
    # Paths - replace these with your actual paths
    fiducial_csv_path = r"C:\Users\rocia\Downloads\TFG\Cohort\Centroids\P2\P1_success_centroids.csv"
    volume_path = r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Model_brain_mask\Dataset\MASK\patient1_mask_5.nrrd"  
    output_mask_path = r"C:\Users\rocia\Downloads\TFG\Cohort\Centroids\P2\electrode_mask_success.nrrd"
    output_csv_path = r'C:\Users\rocia\Downloads\TFG\Cohort\Centroids\P2\processed_validated_LPS_fiducials.csv'

    # Read fiducials from CSV
    fiducials = read_fiducial_csv(fiducial_csv_path)

    # Create electrode mask
    create_electrode_mask_from_fiducials_and_save_csv(
        fiducial_data=fiducials, 
        volume_path=volume_path, 
        output_filename=output_mask_path, 
        csv_filename=output_csv_path
    )

if __name__ == "__main__":
    main()