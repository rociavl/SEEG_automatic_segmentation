import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
import slicer
import vtk

class CentroidVisualizer:
    def __init__(self, centroids_csv_path, volume_node_name):
        self.centroids_csv_path = centroids_csv_path
        self.volume_node_name = volume_node_name
        self._validate_inputs()
    
    def _validate_inputs(self):
        if not os.path.exists(self.centroids_csv_path):
            raise FileNotFoundError(f"Centroids CSV file not found: {self.centroids_csv_path}")
        
        volume_node = slicer.util.getNode(self.volume_node_name)
        if volume_node is None:
            raise ValueError(f"Volume node '{self.volume_node_name}' not found in Slicer scene")
    
    def load_centroids(self):
        centroids_df = pd.read_csv(self.centroids_csv_path)
        return centroids_df[['x', 'y', 'z']].values
    
    def convert_ras_to_ijk(self, ras_coords):

        volume_node = slicer.util.getNode(self.volume_node_name)
        ras_to_ijk_matrix = vtk.vtkMatrix4x4()
        volume_node.GetRASToIJKMatrix(ras_to_ijk_matrix)
        
        # Convert coordinates
        ijk_coords = []
        for ras_coord in ras_coords:
            ras_point = [float(ras_coord[0]), float(ras_coord[1]), float(ras_coord[2]), 1.0]
            ijk_point = [0, 0, 0, 1]
            ras_to_ijk_matrix.MultiplyPoint(ras_point, ijk_point)
            ijk_coords.append(ijk_point[:3])
        
        return np.array(ijk_coords)
    
    def get_volume_data(self):
        volume_node = slicer.util.getNode(self.volume_node_name)
        return slicer.util.array(volume_node.GetID())
    
    def create_markups_fiducials(self):

        centroids_df = pd.read_csv(self.centroids_csv_path)
        ras_coords = centroids_df[['x', 'y', 'z']].values
        markup_name = f"Centroids_{os.path.splitext(os.path.basename(self.centroids_csv_path))[0]}"
        markups_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', markup_name)
        markups_node.SetLocked(False)
        
        # Set color scheme
        color_node = slicer.mrmlScene.GetFirstNodeByName('Generic')
        if color_node:
            markups_node.SetDisplayNodeID(color_node.GetID())
        
        # Add centroids as fiducials in RAS coordinates
        for index, (x, y, z) in enumerate(ras_coords):
            markups_node.AddFiducial(x, y, z)
            markups_node.SetNthFiducialLabel(index, f"Centroid_{index+1}")
            markups_node.SetNthFiducialSelected(index, False)
        
        print(f"Created {len(ras_coords)} fiducial markups in node: {markup_name}")
        return markups_node
    
    def visualize(self, output_path=None, surface_alpha=0.3, create_2d_projections=True, mask_threshold=None):
        """
        Create 3D visualization of centroids and brain mask
        
        Parameters:
        -----------
        output_path : str, optional
            Custom output path for the plot
        surface_alpha : float, optional
            Transparency of the surface mesh
        create_2d_projections : bool, optional
            Whether to create 2D projection plots
        mask_threshold : float, optional
            Threshold for creating brain mask surface. 
            If None, uses mean intensity of the volume
        
        Returns:
        --------
        list
            Paths to the saved plots
        """
        # Load volume data
        volume_data = self.get_volume_data()
        
        # Load RAS centroids
        centroids_ras = self.load_centroids()
        
        # Convert RAS to IJK coordinates
        centroids_ijk = self.convert_ras_to_ijk(centroids_ras)
        
        # Create brain mask surface
        if mask_threshold is None:
            mask_threshold = volume_data.mean()
        
        # Use marching cubes to create surface mesh
        verts, faces, _, _ = marching_cubes(volume_data, level=mask_threshold)
        
        # Output paths list
        output_paths = []
        
        # 3D Plot
        plt.figure(figsize=(15, 12))
        ax = plt.subplot(111, projection='3d')
        
        # Plot brain mask surface
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                        cmap='gray', alpha=surface_alpha, edgecolor='none')
        
        # Plot centroids in IJK space
        scatter = ax.scatter(centroids_ijk[:, 0], 
                             centroids_ijk[:, 1], 
                             centroids_ijk[:, 2], 
                             c=centroids_ijk[:, 2],  # Color by Z coordinate 
                             cmap='viridis', 
                             s=200, 
                             alpha=0.8, 
                             edgecolors='black', 
                             linewidth=1)
        
        # Colorbar for centroids
        plt.colorbar(scatter, ax=ax, label='Z coordinate (IJK)')
        
        # Labeling
        ax.set_xlabel('X (IJK)')
        ax.set_ylabel('Y (IJK)')
        ax.set_zlabel('Z (IJK)')
        ax.set_title(f'Brain Mask and Centroids for {self.volume_node_name}')
        
        # Centroid annotations
        for i, (x, y, z) in enumerate(centroids_ijk):
            ax.text(x, y, z, f' {i+1}', fontsize=9)
        
        # View adjustment
        ax.view_init(elev=20, azim=45)
        
        # Output path for 3D plot
        if output_path is None:
            output_path = self.centroids_csv_path.replace('.csv', '_brain_mask_plot.png')
        
        # Save 3D plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        output_paths.append(output_path)
        
        print(f"3D Plot saved to {output_path}")
        
        # Optional 2D Projections
        if create_2d_projections:
            projection_views = [
                ('XY', 0, 1, 2),
                ('XZ', 0, 2, 1),
                ('YZ', 1, 2, 0)
            ]
            
            for view_name, x_idx, y_idx, z_idx in projection_views:
                plt.figure(figsize=(15, 12))

                brain_slice = np.take(volume_data, volume_data.shape[z_idx]//2, axis=z_idx)
                plt.imshow(brain_slice, cmap='gray', aspect='auto')
                scatter = plt.scatter(centroids_ijk[:, x_idx], 
                                      centroids_ijk[:, y_idx], 
                                      c=centroids_ijk[:, z_idx],  
                                      cmap='viridis', 
                                      s=200, 
                                      alpha=0.8, 
                                      edgecolors='black', 
                                      linewidth=1)
                
                plt.colorbar(scatter, label=f'Z coordinate (IJK)')
                
                plt.xlabel(f'{["X", "Y", "Z"][x_idx]} (IJK)')
                plt.ylabel(f'{["X", "Y", "Z"][y_idx]} (IJK)')
                plt.title(f'{view_name} Projection of Brain Mask and Centroids for {self.volume_node_name}')
                for i, (x, y) in enumerate(zip(centroids_ijk[:, x_idx], centroids_ijk[:, y_idx])):
                    plt.annotate(f' {i+1}', (x, y), fontsize=9)
                projection_output_path = self.centroids_csv_path.replace('.csv', f'_{view_name}_projection.png')
                plt.savefig(projection_output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                output_paths.append(projection_output_path)
                print(f"{view_name} Projection saved to {projection_output_path}")
        
        return output_paths

def main():
    centroids_csv_path = r"C:\Users\rocia\Downloads\TFG\Cohort\Centroids\P1\P1_colab\P1_validated_centroids.csv"
    volume_node_name = "patient1_mask_5"
    
    try:
        visualizer = CentroidVisualizer(centroids_csv_path, volume_node_name)
        visualizer.visualize(create_2d_projections=True)
        visualizer.create_markups_fiducials()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#exec(open('C:/Users/rocia/AppData/Local/slicer.org/Slicer 5.6.2/SEEG_module/SEEG_masking/Outermost_centroids_coordinates/outermost_centroids_slicer.py').read())