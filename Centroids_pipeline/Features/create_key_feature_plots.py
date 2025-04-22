from matplotlib import gridspec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_absolute_error
from scipy.spatial.distance import cdist
import os

def create_key_feature_plots(results_df, patient_id):
    """Create two key plots for visualization"""

    # PLOT 1: 3D Error Vector Field
    # This shows directional error patterns
    mask_names = results_df['Mask'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(mask_names)))

    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    for idx, mask_name in enumerate(mask_names):
        mask_data = results_df[results_df['Mask'] == mask_name]
        mask_file_stem = Path(mask_name).stem
        color = colors[idx]

        # Only include points with valid ground truth
        valid_data = mask_data.dropna(subset=['GT RAS Coordinates'])

        for _, row in valid_data.iterrows():
            gt = np.array(row['GT RAS Coordinates'])
            pred = np.array(row['RAS Coordinates'])

            # Draw line connecting ground truth to prediction
            ax.plot([gt[0], pred[0]], [gt[1], pred[1]], [gt[2], pred[2]],
                    color=color, alpha=0.5, linewidth=1)

            # Ground truth point
            ax.scatter(gt[0], gt[1], gt[2], color='blue', s=20, alpha=0.7)

            # Prediction point - different markers for success/failure
            if row['Success']:
                ax.scatter(pred[0], pred[1], pred[2], color=color, s=30, marker='o', alpha=0.7)
            else:
                ax.scatter(pred[0], pred[1], pred[2], color=color, s=30, marker='x', alpha=0.7)

    # Add legend and labels
    mask_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
                               label=Path(name).stem, markersize=10)
                   for i, name in enumerate(mask_names)]

    gt_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                         label='Ground Truth', markersize=10)

    success_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                              label='Success', markersize=10)

    failure_patch = plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='gray',
                              label='Failure', markersize=10)

    ax.legend(handles=[gt_patch, success_patch, failure_patch] + mask_patches,
              bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_xlabel('X (RAS)')
    ax.set_ylabel('Y (RAS)')
    ax.set_zlabel('Z (RAS)')
    ax.set_title(f'Electrode Detection Error Vectors - Patient {patient_id}', pad=20)

    error_vector_path = f'patient_{patient_id}_error_vector_field.png'
    plt.savefig(error_vector_path, dpi=300, bbox_inches='tight')
    plt.close()

    # PLOT 2: Error heatmap by brain region (octants)
    # Create a figure with subplots for each mask
    fig, axes = plt.subplots(len(mask_names), 1, figsize=(15, 5*len(mask_names)))
    if len(mask_names) == 1:
        axes = [axes]

    for idx, mask_name in enumerate(mask_names):
        mask_data = results_df[results_df['Mask'] == mask_name]
        mask_file_stem = Path(mask_name).stem

        # Define octants in brain space
        mask_data['Octant'] = mask_data['RAS Coordinates'].apply(
            lambda p: f"{'P' if p[0] >= 0 else 'A'}"
                     f"{'S' if p[1] >= 0 else 'I'}"
                     f"{'R' if p[2] >= 0 else 'L'}"
        )

        # Get error statistics for each octant
        octant_stats = mask_data.groupby('Octant').agg({
            'Distance (mm)': ['mean', 'count'],
            'Success': ['mean', 'count']
        })

        # Fill in missing octants
        all_octants = ['PSR', 'PSL', 'PIR', 'PIL', 'ASR', 'ASL', 'AIR', 'AIL']
        for octant in all_octants:
            if octant not in octant_stats.index:
                octant_stats.loc[octant] = np.nan

        # Sort by anatomical position
        octant_stats = octant_stats.loc[all_octants]

        # Create the heatmap
        mean_distances = octant_stats[('Distance (mm)', 'mean')].values.reshape(2, 2, 2)
        electrode_counts = octant_stats[('Distance (mm)', 'count')].values.reshape(2, 2, 2)

        # Plot the data as 2D slices with annotations
        ax = axes[idx]

        # Create 2x2 grid for each of the 3 planes (axial, coronal, sagittal)
        gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gridspec.GridSpec(len(mask_names), 1)[idx])

        # Axial view (Superior vs Inferior)
        ax1 = fig.add_subplot(gs[0, 0])
        axial_data = np.nanmean(mean_distances, axis=2)
        axial_counts = np.nansum(electrode_counts, axis=2).astype(int)

        im1 = ax1.imshow(axial_data, cmap='hot_r', vmin=0, vmax=5)
        ax1.set_title(f'{mask_file_stem} - Axial View\n(Superior/Inferior)')
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['Posterior', 'Anterior'])
        ax1.set_yticklabels(['Superior', 'Inferior'])

        # Add electrode counts and mean distance
        for i in range(2):
            for j in range(2):
                if not np.isnan(axial_data[i, j]):
                    ax1.text(j, i, f"{axial_data[i, j]:.1f}mm\n({axial_counts[i, j]})",
                            ha="center", va="center", color="white" if axial_data[i, j] > 2.5 else "black")

        # Coronal view (Anterior vs Posterior)
        ax2 = fig.add_subplot(gs[0, 1])
        coronal_data = np.nanmean(mean_distances, axis=0)
        coronal_counts = np.nansum(electrode_counts, axis=0).astype(int)

        im2 = ax2.imshow(coronal_data, cmap='hot_r', vmin=0, vmax=5)
        ax2.set_title(f'{mask_file_stem} - Coronal View\n(Anterior/Posterior)')
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(['Right', 'Left'])
        ax2.set_yticklabels(['Superior', 'Inferior'])

        for i in range(2):
            for j in range(2):
                if not np.isnan(coronal_data[i, j]):
                    ax2.text(j, i, f"{coronal_data[i, j]:.1f}mm\n({coronal_counts[i, j]})",
                            ha="center", va="center", color="white" if coronal_data[i, j] > 2.5 else "black")

        # Sagittal view (Right vs Left)
        ax3 = fig.add_subplot(gs[0, 2])
        sagittal_data = np.nanmean(mean_distances, axis=1)
        sagittal_counts = np.nansum(electrode_counts, axis=1).astype(int)

        im3 = ax3.imshow(sagittal_data, cmap='hot_r', vmin=0, vmax=5)
        ax3.set_title(f'{mask_file_stem} - Sagittal View\n(Right/Left)')
        ax3.set_xticks([0, 1])
        ax3.set_yticks([0, 1])
        ax3.set_xticklabels(['Posterior', 'Anterior'])
        ax3.set_yticklabels(['Right', 'Left'])

        for i in range(2):
            for j in range(2):
                if not np.isnan(sagittal_data[i, j]):
                    ax3.text(j, i, f"{sagittal_data[i, j]:.1f}mm\n({sagittal_counts[i, j]})",
                            ha="center", va="center", color="white" if sagittal_data[i, j] > 2.5 else "black")

    plt.tight_layout()
    fig.colorbar(im1, ax=axes, shrink=0.6, label='Mean Distance Error (mm)')

    regional_error_path = f'patient_{patient_id}_regional_error_analysis.png'
    plt.savefig(regional_error_path, dpi=300, bbox_inches='tight')
    plt.close()

    return error_vector_path, regional_error_path

