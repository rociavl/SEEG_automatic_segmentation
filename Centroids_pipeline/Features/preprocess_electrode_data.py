import pandas as pd
import numpy as np

def preprocess_electrode_data(
    df: pd.DataFrame,
    is_gt: bool = False,
    is_success: bool = False,
    add_flags: bool = True,
    columns_to_drop: list = None,
    mask_id=None,
    reset_index: bool = True,
    add_confidence: bool = False  # <-- Toggle confidence computation
) -> pd.DataFrame:

    default_drop = [
        'Label', 'RAS Coordinates', 'GT RAS Coordinates', 'Mask Label',
        'GT_X', 'GT_Y', 'GT_Z', 'Local_Anisotropy', 'Error_X', 'Error_Y', 'Error_Z',
        'Closest GT Label', 'CT_centroid_x', 'CT_centroid_y', 'CT_centroid_z',
    ]
    cols_to_drop = columns_to_drop if columns_to_drop else default_drop

    processed_df = df.copy()
    existing_cols_to_drop = [col for col in cols_to_drop if col in processed_df.columns]
    processed_df.drop(columns=existing_cols_to_drop, inplace=True)

    # Optional: Add confidence column based on Distance (mm)
    if add_confidence and 'Distance (mm)' in processed_df.columns:
        processed_df['Confidence'] = 1 / (1 + (processed_df['Distance (mm)'] / 1.5) ** 5)

    # Add identifier flags
    if add_flags:
        processed_df['_gt'] = int(is_gt)
        processed_df['_success'] = int(is_success)
        processed_df['_all'] = int(not (is_gt or is_success))

    if mask_id is not None:
        processed_df['Mask'] = mask_id

    # Ensure required columns
    required_columns = ['RAS_X', 'RAS_Y', 'RAS_Z', 'Patient ID', 'Mask', 'PCA1', 'PCA2', 'PCA3']
    for col in required_columns:
        if col not in processed_df.columns:
            raise ValueError(f"Required column '{col}' missing in DataFrame")

    # Convert categorical columns
    if 'Hemisphere' in processed_df.columns:
        processed_df['Hemisphere'] = processed_df['Hemisphere'].astype('category')

    if reset_index:
        processed_df = processed_df.reset_index(drop=True)

    return processed_df
