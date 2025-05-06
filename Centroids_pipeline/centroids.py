import slicer
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table
import matplotlib.pyplot as plt
import os
import logging
import vtk
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def get_array_from_volume(volume_node):
    if volume_node is None:
        logging.error("Volume node is None")
        return None
    return slicer.util.arrayFromVolume(volume_node)

def binarize_array(array, threshold=0):
    return (array > threshold).astype(np.uint8) if array is not None else None

def calculate_centroids_numpy(binary_array):
    if binary_array is None:
        return pd.DataFrame(columns=['label', 'centroid-0', 'centroid-1', 'centroid-2'])
    labeled_array = label(binary_array)
    props = regionprops_table(labeled_array, properties=['label', 'centroid'])
    return pd.DataFrame(props)

def get_ras_coordinates_from_ijk(volume_node, ijk):
    ijk_to_ras = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(ijk_to_ras)
    homogeneous_ijk = [ijk[0], ijk[1], ijk[2], 1]
    ras = [sum(ijk_to_ras.GetElement(i, j) * homogeneous_ijk[j] for j in range(4)) for i in range(4)]
    return ras[:3]

def get_centroids_ras(volume_node, centroids_df):
    return {
        int(row['label']): tuple(get_ras_coordinates_from_ijk(volume_node, [row['centroid-2'], row['centroid-1'], row['centroid-0']]))
        for _, row in centroids_df.iterrows()
    }

def save_centroids_to_csv(centroids, patient_id, mask_name, output_dir=r"C:\\Users\\rocia\\Downloads\\TFG\\Cohort"):
    centroid_df = pd.DataFrame.from_dict(centroids, orient='index', columns=['R', 'A', 'S'])
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{patient_id}_{mask_name}_centroids.csv")
    centroid_df.to_csv(csv_path)
    print(f"Centroids saved at: {csv_path}")

def plot_centroids(mask_centroids_dict, gt_centroids, output_plot_path):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.get_cmap('tab10', len(mask_centroids_dict) + 1)  


    if gt_centroids:
        gt_x, gt_y, gt_z = zip(*gt_centroids.values())
        ax.scatter(gt_x, gt_y, gt_z, c='blue', marker='o', label="Ground Truth")

    for i, (mask_name, mask_centroids) in enumerate(mask_centroids_dict.items()):
        if mask_centroids:
            mask_x, mask_y, mask_z = zip(*mask_centroids.values())
            mask_color = colors(i)  
            ax.scatter(mask_x, mask_y, mask_z, c=[mask_color], marker='^', label=mask_name)

    # Labels 
    ax.set_xlabel("R (Right-Left)")
    ax.set_ylabel("A (Anterior-Posterior)")
    ax.set_zlabel("S (Superior-Inferior)")
    ax.set_title("3D Centroid Plot")
    ax.legend()

    plt.savefig(output_plot_path)
    print(f"Plot saved at: {output_plot_path}")
    plt.close(fig)

def create_markups_node(node_name):
    return slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", node_name)

def visualize_centroids_in_slicer(mask_centroids_dict, gt_centroids, patient_id):
    gt_markups = create_markups_node(f"GroundTruth_Centroids_{patient_id}")

    for label, centroid in gt_centroids.items():
        gt_markups.AddControlPoint(centroid)
        gt_markups.SetNthControlPointLabel(gt_markups.GetNumberOfControlPoints()-1, f"GT-{label}")
    
    gt_markups.GetDisplayNode().SetColor(0, 0, 1) 

    for mask_name, mask_centroids in mask_centroids_dict.items():
        mask_markups = create_markups_node(f"{mask_name}_MaskCentroids_{patient_id}")
        
        for label, centroid in mask_centroids.items():
            mask_markups.AddControlPoint(centroid)
            mask_markups.SetNthControlPointLabel(mask_markups.GetNumberOfControlPoints()-1, f"Mask-{label}")

        mask_markups.GetDisplayNode().SetColor(1, 0, 0)  

def analyze_centroids(mask_volumes, ground_truth_volume, patient_id="P1"):
    mask_centroids_dict = {}

    ground_truth_array = get_array_from_volume(ground_truth_volume)
    gt_binary = binarize_array(ground_truth_array)
    gt_centroids_df = calculate_centroids_numpy(gt_binary)
    gt_ras_centroids = get_centroids_ras(ground_truth_volume, gt_centroids_df)

    for mask_node in mask_volumes:
        mask_array = get_array_from_volume(mask_node)
        mask_binary = binarize_array(mask_array)
        mask_centroids_df = calculate_centroids_numpy(mask_binary)
        mask_ras_centroids = get_centroids_ras(mask_node, mask_centroids_df)
        mask_name = mask_node.GetName()
        mask_centroids_dict[mask_name] = mask_ras_centroids
        save_centroids_to_csv(mask_ras_centroids, patient_id, mask_name)

    save_centroids_to_csv(gt_ras_centroids, patient_id, 'ground_truth')
    return mask_centroids_dict, gt_ras_centroids

def calculate_euclidean_distances(mask_centroids, gt_centroids, patient_id, output_dir=r"C:\\Users\\rocia\\Downloads\\TFG\\Cohort"):
    if not mask_centroids or not gt_centroids:
        logging.warning("No centroids available for distance calculation.")
        return None

    gt_labels, gt_points = zip(*gt_centroids.items()) if gt_centroids else ([], [])
    gt_array = np.array(gt_points)

    distances = []
    
    for mask_name, mask_points in mask_centroids.items():
        mask_labels, mask_array = zip(*mask_points.items()) if mask_points else ([], [])
        mask_array = np.array(mask_array)

        pairwise_distances = cdist(mask_array, gt_array, metric='euclidean')
        for idx, mask_label in enumerate(mask_labels):
            min_distance = np.min(pairwise_distances[idx])
            closest_gt_label = gt_labels[np.argmin(pairwise_distances[idx])]

            distances.append({
                "Patient": patient_id,
                "Mask Name": mask_name,
                "Mask Label": mask_label,
                "GT Label": closest_gt_label,
                "Distance": min_distance
            })

    df_distances = pd.DataFrame(distances)
    csv_path = os.path.join(output_dir, f"{patient_id}_euclidean_distances.csv")
    df_distances.to_csv(csv_path, index=False)
    print(f"Distance results saved at: {csv_path}")

    return df_distances

def get_closest_centroids(mask_centroids, gt_centroids, patient_id, output_dir=r"C:\\Users\\rocia\\Downloads\\TFG\\Cohort"):

    df_distances = calculate_euclidean_distances(mask_centroids, gt_centroids, patient_id, output_dir)
    if df_distances is None or df_distances.empty:
        logging.warning("No distances computed.")
        return None

    closest_centroids = {}
    
    for _, row in df_distances.iterrows():
        mask_name = row["Mask Name"]
        mask_label = row["Mask Label"]
        gt_label = row["GT Label"]

        if mask_name not in closest_centroids:
            closest_centroids[mask_name] = {}

        closest_centroids[mask_name][mask_label] = {
            "Mask Centroid": mask_centroids[mask_name][mask_label],
            "GT Centroid": gt_centroids[gt_label],
            "Distance": row["Distance"]
        }

    closest_centroids_list = []
    for mask_name, matches in closest_centroids.items():
        for mask_label, data in matches.items():
            closest_centroids_list.append({
                "Patient": patient_id,
                "Mask Name": mask_name,
                "Mask Label": mask_label,
                "GT Label": gt_label,
                "Mask R": data["Mask Centroid"][0],
                "Mask A": data["Mask Centroid"][1],
                "Mask S": data["Mask Centroid"][2],
                "GT R": data["GT Centroid"][0],
                "GT A": data["GT Centroid"][1],
                "GT S": data["GT Centroid"][2],
                "Distance": data["Distance"]
            })

    df_closest = pd.DataFrame(closest_centroids_list)
    csv_path = os.path.join(output_dir, f"{patient_id}_closest_centroids.csv")
    df_closest.to_csv(csv_path, index=False)
    print(f"Closest centroids saved at: {csv_path}")

    return closest_centroids

def main():
    try:
        ground_truth_volume = slicer.util.getNode("P7_electrode_fiducials")
        mask_volumes = [
            slicer.util.getNode("electrode_mask_success"),
        ]
    except Exception as e:
        logging.error(f"Failed to retrieve volumes: {str(e)}")
        return
    
    mask_centroids_dict, gt_centroids = analyze_centroids(mask_volumes, ground_truth_volume, patient_id="P7")
    get_closest_centroids(mask_centroids_dict, gt_centroids, patient_id="P7")
    visualize_centroids_in_slicer(mask_centroids_dict, gt_centroids, patient_id="P7")
    plot_output_path = r"C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\centroid_plot_06_05.png"  
    plot_centroids(mask_centroids_dict, gt_centroids, plot_output_path)

main()

#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Centroids_pipeline\centroids.py').read())

