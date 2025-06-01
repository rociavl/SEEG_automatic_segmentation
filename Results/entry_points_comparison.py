import numpy as np
import os
from scipy.spatial import distance
import vtk
import slicer
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# ----------------------
# CONFIGURATION
# ----------------------
calculated_node_name = "P4_entry_points_markups"
ground_truth_node_name = "P4_EP_FIDUCIALS"
output_dir = r"C:\Users\rocia\Downloads\TFG\Cohort\Bolt_heads\Results_report\P4" 
threshold_mm = 3.5  # Detection threshold for true positives

# Visualization options
draw_error_lines = False
mark_fp_fn = True  
save_plots = True  

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# ----------------------
# FUNCTIONS
# ----------------------
def get_fiducial_points(node_name):
    """Extract fiducial points as numpy array."""
    try:
        node = slicer.util.getNode(node_name)
        points = []
        for i in range(node.GetNumberOfControlPoints()):
            coord = [0, 0, 0]
            node.GetNthControlPointPositionWorld(i, coord)
            points.append(coord)
        return np.array(points)
    except Exception as e:
        raise ValueError(f"Error loading {node_name}: {str(e)}")

def save_results_to_file(filename, text):
    """Save text results to file."""
    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write(text)

def save_plot(fig, filename):
    """Save matplotlib figure to output directory."""
    fig.savefig(os.path.join(output_dir, filename), 
                bbox_inches='tight', 
                dpi=300)
    plt.close(fig)

# ----------------------
# MAIN ANALYSIS
# ----------------------
# Load data
calculated_points = get_fiducial_points(calculated_node_name)
ground_truth_points = get_fiducial_points(ground_truth_node_name)

# Verify point counts
if len(calculated_points) == 0 or len(ground_truth_points) == 0:
    raise ValueError("Empty point sets detected")

# Initialize metrics
matched_gt_indices = set()
false_positives = []
true_positives = []
point_errors = []
paired_errors = []  # For matched points only

# Match points and calculate errors
for i, calc_point in enumerate(calculated_points):
    min_dist = float('inf')
    best_match_idx = -1
    
    for j, gt_point in enumerate(ground_truth_points):
        if j not in matched_gt_indices:
            dist = np.linalg.norm(calc_point - gt_point)
            if dist < threshold_mm and dist < min_dist:
                min_dist = dist
                best_match_idx = j
    
    if best_match_idx != -1:
        true_positives.append((i, best_match_idx))
        matched_gt_indices.add(best_match_idx)
        point_errors.append(min_dist)
        paired_errors.append((calc_point, ground_truth_points[best_match_idx], min_dist))
    else:
        false_positives.append(i)

# Find false negatives
false_negatives = [j for j in range(len(ground_truth_points)) if j not in matched_gt_indices]

# Calculate metrics
precision = len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0
recall = len(true_positives) / len(ground_truth_points) if len(ground_truth_points) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
mean_error = np.mean(point_errors) if point_errors else 0
std_error = np.std(point_errors) if point_errors else 0

# ----------------------
# VISUALIZATION (PLOTS)
# ----------------------
if save_plots:    
    # Plot 1: Error distribution
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax1.hist(point_errors, bins=15, color='#1f77b4', edgecolor='white', alpha=0.7)
    ax1.axvline(mean_error, color='#d62728', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_error:.2f} mm')
    ax1.set_xlabel('Error (mm)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Entry Point Error Distribution', fontsize=14, pad=20)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    save_plot(fig1, "error_distribution.png")

    # Plot 2: Detection metrics
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [precision, recall, f1_score]
    colors = ['#2ca02c', '#9467bd', '#e377c2']
    bars = ax2.bar(metrics, values, color=colors, width=0.6)
    ax2.set_ylim(0, 1.15)
    ax2.set_title('Detection Performance Metrics', fontsize=14, pad=20)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11)
    save_plot(fig2, "detection_metrics.png")

    # Plot 3: 3D Coordinate Comparison
    fig3 = plt.figure(figsize=(12, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    
    # Plot ground truth (green)
    ax3.scatter(ground_truth_points[:,0], ground_truth_points[:,1], ground_truth_points[:,2], 
               c='#2ca02c', s=60, label='Ground Truth', depthshade=False)
    
    # Plot calculated points (blue)
    ax3.scatter(calculated_points[:,0], calculated_points[:,1], calculated_points[:,2], 
               c='#1f77b4', s=60, label='Calculated', depthshade=False)
    
    # Draw error lines
    for pair in paired_errors:
        ax3.plot([pair[0][0], pair[1][0]], 
                [pair[0][1], pair[1][1]], 
                [pair[0][2], pair[1][2]], 
                'r-', linewidth=1, alpha=0.5)
    
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_zlabel('Z (mm)')
    ax3.set_title('3D Spatial Distribution of Points', fontsize=14, pad=20)
    ax3.legend()
    save_plot(fig3, "3d_coordinate_comparison.png")

    # Plot 4: FP/FN counts (if any)
    if false_positives or false_negatives:
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        counts = [len(false_positives), len(false_negatives)]
        labels = ['False Positives', 'False Negatives']
        colors = ['#ff7f0e', '#d62728']
        bars = ax4.bar(labels, counts, color=colors)
        ax4.set_title('Detection Errors', fontsize=14, pad=20)
        ax4.set_ylabel('Count', fontsize=12)
        ax4.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}',
                    ha='center', va='bottom', fontsize=11)
        save_plot(fig4, "fp_fn_counts.png")

# ----------------------
# SAVE RESULTS
# ----------------------
results_text = f"""
=== ENTRY POINT VALIDATION RESULTS ===
Patient: {calculated_node_name.split('_')[0]}
Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}

--- Point Matching ---
Total Calculated Points: {len(calculated_points)}
Total Ground Truth Points: {len(ground_truth_points)}
Matched Points: {len(true_positives)}

--- Detection Metrics ---
True Positives (TP): {len(true_positives)}
False Positives (FP): {len(false_positives)}
False Negatives (FN): {len(false_negatives)}
Precision: {precision:.3f} ({precision:.1%})
Recall (Sensitivity): {recall:.3f} ({recall:.1%})
F1-Score: {f1_score:.3f}

--- Error Analysis ---
Mean Error (±SD): {mean_error:.2f} ± {std_error:.2f} mm
Maximum Error: {max(point_errors) if point_errors else 0:.2f} mm
Minimum Error: {min(point_errors) if point_errors else 0:.2f} mm

--- Output Files ---
Saved plots:
- error_distribution.png
- detection_metrics.png
- 3d_coordinate_comparison.png
- {'fp_fn_counts.png' if (false_positives or false_negatives) else 'N/A'}
"""

# Save to file
save_results_to_file(f"{calculated_node_name}_results.txt", results_text)

# Print to console
print(results_text)
print(f"\nResults saved to: {output_dir}")

# ----------------------
# 3D SLICER VISUALIZATION (Optional)
# ----------------------
if draw_error_lines or mark_fp_fn:
    colors = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLColorTableNode')
    colors.SetTypeToCool1()

    # Create error lines model
    lines_model = slicer.modules.models.logic().AddModel(vtk.vtkPolyData())
    lines_model.GetDisplayNode().SetColor(1, 0, 0)
    lines_model.SetName("Error_Lines")

    if mark_fp_fn:
        # Mark false positives (red)
        fp_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "False_Positives")
        fp_node.GetDisplayNode().SetColor(1, 0, 0)
        for i in false_positives:
            fp_node.AddControlPoint(calculated_points[i])

        # Mark false negatives (blue)
        fn_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "False_Negatives")
        fn_node.GetDisplayNode().SetColor(0, 0, 1)
        for j in false_negatives:
            fn_node.AddControlPoint(ground_truth_points[j])

#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Results\entry_points_comparison.py').read())