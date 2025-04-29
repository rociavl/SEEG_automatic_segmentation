import pandas as pd
import ast

def process_centroids(input_csv, output_csv):

    df = pd.read_csv(input_csv)
    validated_centroids = pd.DataFrame()
    columns = df.columns.values.tolist()
    validated_centroids['Label'] = df['Mask Label']
    validated_centroids['RAS Coordinates'] = df['RAS Coordinates'].apply(ast.literal_eval)
    validated_centroids['x'] = validated_centroids['RAS Coordinates'].apply(lambda coord: coord[0])
    validated_centroids['y'] = validated_centroids['RAS Coordinates'].apply(lambda coord: coord[1])
    validated_centroids['z'] = validated_centroids['RAS Coordinates'].apply(lambda coord: coord[2])
    validated_centroids = validated_centroids.drop('RAS Coordinates', axis=1)
    validated_centroids.to_csv(output_csv, index=False)
    print(f'Columns{columns}')
    print(f"Validated centroids saved to {output_csv}")
    return validated_centroids

input_file = r"C:\Users\rocia\Downloads\content\P1_mask7_dataset\data_success\patient_P1_enhanced_analysis__success.csv"
output_file = r"C:\Users\rocia\Downloads\TFG\Cohort\Centroids\P1_28_04_electrodos_camino\P1_success_centroids.csv"
result = process_centroids(input_file, output_file)
print(result.head())


