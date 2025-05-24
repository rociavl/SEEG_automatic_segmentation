import slicer
import csv
import ast
import os
import time


def create_markups_from_csv(csv_file_path, markup_node_name=None):
    """
    Create Slicer markup fiducials from CSV file with electrode detection results
    
    Parameters:
    csv_file_path: Path to the CSV file with electrode data
    markup_node_name: Name for the markup node (if None, uses CSV filename)
    """

    start_time = time.time()
    
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        return None
    
    # Create markup node name from CSV filename if not provided
    if markup_node_name is None:
        markup_node_name = os.path.splitext(os.path.basename(csv_file_path))[0] + "_markups"
    
    # Create a new markups fiducial node
    markups_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
    markups_node.SetName(markup_node_name)
    
    electrodes_added = 0
    
    try:
        with open(csv_file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row_idx, row in enumerate(reader):
                try:
                    # Extract data from CSV row
                    patient_id = row.get('Patient ID', '')
                    mask_name = row.get('Mask', '')
                    electrode_label = row.get('Electrode Label', '')
                    
                    # Get coordinates - try X,Y,Z columns first, then RAS Coordinates
                    if 'X' in row and 'Y' in row and 'Z' in row:
                        x = float(row['X'])
                        y = float(row['Y']) 
                        z = float(row['Z'])
                    elif 'RAS Coordinates' in row:
                        # Parse the RAS Coordinates tuple string
                        ras_string = row['RAS Coordinates'].strip()
                        if ras_string.startswith('(') and ras_string.endswith(')'):
                            # Remove parentheses and parse as tuple
                            coords_str = ras_string[1:-1]
                            coords = [float(coord.strip()) for coord in coords_str.split(',')]
                            x, y, z = coords[0], coords[1], coords[2]
                        else:
                            # Try to evaluate as literal
                            coords = ast.literal_eval(ras_string)
                            x, y, z = coords[0], coords[1], coords[2]
                    else:
                        print(f"Row {row_idx + 1}: No valid coordinate data found")
                        continue
                    
                    pixel_count = row.get('Pixel Count', '')
                    
                    # Create fiducial label with electrode info
                    if electrode_label:
                        fiducial_label = f"E{electrode_label}"
                        if pixel_count:
                            fiducial_label += f"_px{pixel_count}"
                    else:
                        fiducial_label = f"Electrode_{row_idx + 1}"
                    
                    # Add fiducial point to markups node
                    point_index = markups_node.AddControlPoint([x, y, z])
                    markups_node.SetNthControlPointLabel(point_index, fiducial_label)
                    
                    # Set description with additional info
                    description = f"Patient: {patient_id}, Mask: {os.path.basename(mask_name)}"
                    if pixel_count:
                        description += f", Pixels: {pixel_count}"
                    markups_node.SetNthControlPointDescription(point_index, description)
                    
                    electrodes_added += 1
                    print(f"Added electrode {fiducial_label} at RAS ({x:.2f}, {y:.2f}, {z:.2f})")
                    
                except Exception as e:
                    print(f"Error processing row {row_idx + 1}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    # Set markup properties for better visualization
    display_node = markups_node.GetDisplayNode()
    if display_node:
        display_node.SetTextScale(3.0)  # Larger text
        display_node.SetGlyphScale(3.0)  # Larger markers
        display_node.SetColor(1, 0, 0)  # Red color
        display_node.SetPointLabelsVisibility(True)

    finish_time = time.time()
    execution_time = finish_time - start_time
    minutes = int(execution_time // 60)
    seconds = execution_time % 60
    print(f"- Total execution time: {minutes} min {seconds:.2f} sec")
    
    print(f"Successfully created markup node '{markup_node_name}' with {electrodes_added} electrodes")
    return markups_node



# Example usage:

# Method 1: Load from a single CSV file
csv_path = r"c:\Users\rocia\Downloads\TFG\Cohort\Extension\Just_plot_extension\patient_P1_test_Filtered_DESCARGAR_WAVELET_ROI_1000_ctp.3D_results.csv"
markup_node = create_markups_from_csv(csv_path, "P1_Mask_wavelet_Electrodes")



print("Markup creation completed!")

#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\csv_to_markups.py').read())