import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

# ==============================================
# Main Execution
# ==============================================

def process_csv_files_to_model_data(
    input_folder,
    output_folder,
    patient_id="P1",
    create_subfolders=True
):
    """
    Process all CSV files from input folder (including subfolders) and organize them into output folder.
    
    Parameters:
    - input_folder: Folder containing subfolders with CSV files
    - output_folder: Destination folder for the processed data
    - patient_id: Patient identifier for naming conventions
    - create_subfolders: Whether to create subfolders for each CSV file in the output folder
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all CSV files in the input folder and its subfolders
    csv_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    for i, csv_file in enumerate(csv_files):
        # Get the base filename without extension
        file_basename = os.path.basename(csv_file).split('.')[0]
        
        if create_subfolders:
            # Create a unique output directory for each file
            current_output_dir = os.path.join(output_folder, f"{patient_id}_mask_{i}")
            os.makedirs(current_output_dir, exist_ok=True)
            
            # Define destination file path
            destination_file = os.path.join(current_output_dir, os.path.basename(csv_file))
        else:
            # Just put all files in the main output folder
            destination_file = os.path.join(output_folder, f"{patient_id}_mask_{i}_{os.path.basename(csv_file)}")
        
        print(f"Processing file {i+1}/{len(csv_files)}: {csv_file}")
        print(f"Destination: {destination_file}")
        
        # Copy the CSV file to the destination
        shutil.copy2(csv_file, destination_file)
        

if __name__ == "__main__":
    # Define paths
    input_folder = r"C:\Users\rocia\Downloads\P7_DATASET\content\P7_DATASET"
    output_folder = r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Centroids_model\P7_DATA_READY"
    
    # Process all CSV files
    process_csv_files_to_model_data(
        input_folder=input_folder,
        output_folder=output_folder,
        patient_id="P7",
        create_subfolders=False  # Set to False if you prefer all CSVs in the main folder
    )

