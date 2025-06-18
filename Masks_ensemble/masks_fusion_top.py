import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path

class EnhancedMaskSelector:
    def __init__(self, mask_folder_path, output_dir, excluded_files=None):
        """
        Initialize the enhanced mask selector with the path to the masks folder and output directory.
        
        Args:
            mask_folder_path: Path to the folder containing mask files
            output_dir: Path to the output directory for saving results
            excluded_files: List of filenames to exclude (optional)
        """
        self.mask_folder_path = Path(mask_folder_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store excluded files
        self.excluded_files = excluded_files or ["Filtered_DESCARGAR_roi_volume_features_ctp.3D.nrrd"]
        
        # Dictionary to store mask arrays
        self.masks = {}
        # Dictionary to store mask scores
        self.mask_scores = {}
        
        self.reference_origin = None
        self.reference_spacing = None
        self.reference_direction = None
        self.reference_size = None
        
        # Load all masks from the folder
        self.load_all_masks()
        
        # Initialize vote map
        if self.masks:
            self.global_vote_map = np.zeros_like(next(iter(self.masks.values())))
        
    def load_all_masks(self):
        """Load all NRRD mask files from the specified folder, excluding specified files"""
        print(f"Loading masks from {self.mask_folder_path}")
        
        mask_files = list(self.mask_folder_path.glob("*.nrrd"))
        
        # Filter out excluded files
        filtered_files = []
        for mask_file in mask_files:
            if mask_file.name not in self.excluded_files:
                filtered_files.append(mask_file)
            else:
                print(f"Excluding file: {mask_file.name}")
        
        mask_files = filtered_files
        
        if not mask_files:
            raise ValueError(f"No valid NRRD files found in {self.mask_folder_path}")
            
        print(f"Found {len(mask_files)} valid mask files")
        
        # Load each mask
        for i, mask_file in enumerate(mask_files):
            mask_sitk = sitk.ReadImage(str(mask_file))
            # Store reference information from the first mask
            if i == 0:
                self.reference_origin = mask_sitk.GetOrigin()
                self.reference_spacing = mask_sitk.GetSpacing()
                self.reference_direction = mask_sitk.GetDirection()
                self.reference_size = mask_sitk.GetSize()
            # Convert to numpy array and binarize
            mask_array = sitk.GetArrayFromImage(mask_sitk)
            mask_array = np.where(mask_array > 0, 1, 0).astype(np.uint8)
            # Store the mask array
            self.masks[mask_file.stem] = mask_array
        print(f"Successfully loaded {len(self.masks)} masks")
    
    def compute_global_agreement(self):
        """Compute the global agreement vote map across all masks"""
        if not self.masks:
            raise ValueError("No masks loaded")
        # Reset the global vote map
        self.global_vote_map = np.zeros_like(next(iter(self.masks.values())))
        # Sum all masks to create the vote map
        for mask_array in self.masks.values():
            self.global_vote_map += mask_array
        return self.global_vote_map
    
    def compute_overlap_score(self, mask_array, vote_map):
        """
        Compute the overlap score between a mask and the current vote map.
        This measures how much this mask contributes to the consensus.
        
        Args:
            mask_array: Binary mask array
            vote_map: Current vote map
        
        Returns:
            overlap_score: The weighted overlap score
        """
        # Calculate overlap: voxels where both mask and vote map are positive
        overlap = mask_array * (vote_map > 0)
        
        # Weight by the vote map values to favor voxels with higher consensus
        weighted_overlap = np.sum(overlap * vote_map)
        
        # Normalize by the sum of mask voxels to avoid favoring large masks
        mask_sum = np.sum(mask_array)
        if mask_sum == 0:
            return 0
            
        return weighted_overlap / mask_sum
    
    def select_best_masks(self, n_masks=4):
        """
        Select the best n_masks using the greedy voting strategy and store their scores.
        
        Args:
            n_masks: Number of masks to select
        
        Returns:
            selected_masks: List of selected mask names
        """
        if n_masks > len(self.masks):
            print(f"Warning: Requested {n_masks} masks but only {len(self.masks)} are available")
            n_masks = len(self.masks)
        
        # Compute initial global agreement
        self.compute_global_agreement()
        
        # Make a copy of all masks 
        remaining_masks = dict(self.masks)
        
        # Initialize list to store selected masks
        selected_masks = []
        
        # Clear previous scores
        self.mask_scores = {}
        
        for i in range(n_masks):
            best_mask_name = None
            best_score = -1
            
            # Evaluate each remaining mask
            for mask_name, mask_array in remaining_masks.items():
                # Compute how much this mask would contribute to the current selection
                score = self.compute_overlap_score(mask_array, self.global_vote_map)
                
                if score > best_score:
                    best_score = score
                    best_mask_name = mask_name
            
            if best_mask_name is None:
                print(f"Warning: Could not find a suitable mask at iteration {i}")
                break
                
            selected_masks.append(best_mask_name)
            # Store the score for this mask
            self.mask_scores[best_mask_name] = best_score
            
            # Remove the selected mask from consideration
            del remaining_masks[best_mask_name]
            
            print(f"Selected mask {i+1}/{n_masks}: {best_mask_name} (score: {best_score:.4f})")
        
        return selected_masks
    
    def create_weighted_fused_mask(self, mask_names, output_name):
        """
        Create a fused mask from the selected masks, using their scores as weights.
        
        Args:
            mask_names: List of mask names to fuse
            output_name: Name for the output fused mask
        
        Returns:
            fused_mask: The fused mask array
        """
        # Initialize the weighted sum and weight accumulator
        weighted_sum = np.zeros_like(next(iter(self.masks.values())), dtype=float)
        total_weight = 0
        
        # Add all selected masks with their weights
        for mask_name in mask_names:
            if mask_name in self.masks and mask_name in self.mask_scores:
                weight = self.mask_scores[mask_name]
                weighted_sum += self.masks[mask_name] * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            weighted_sum /= total_weight
        
        # Binarize with threshold of 0.5 (majority of weighted votes)
        fused_mask = np.where(weighted_sum >= 0.5, 1, 0).astype(np.uint8)
        
        # Save the fused mask
        self.save_mask(fused_mask, output_name)
        
        return fused_mask
        
    def save_mask(self, mask_array, output_name):
        """
        Save a mask array as a NRRD file.
        
        Args:
            mask_array: The mask array to save
            output_name: Name for the output file
        """
        # Create a SimpleITK image from the array
        mask_sitk = sitk.GetImageFromArray(mask_array)
        
        # Set the metadata from the reference mask
        mask_sitk.SetOrigin(self.reference_origin)
        mask_sitk.SetSpacing(self.reference_spacing)
        mask_sitk.SetDirection(self.reference_direction)
        
        # Save the mask
        output_path = self.output_dir / f"{output_name}.nrrd"
        sitk.WriteImage(mask_sitk, str(output_path))
        print(f"Saved mask to: {output_path}")
    
    def plot_voxel_count_comparison(self, selected_masks, plots_dir=None):
        """
        Plot comparison of total voxels between original and selected masks.
        
        Args:
            selected_masks: List of selected mask names
            plots_dir: Directory for saving plots (optional)
        """
        if plots_dir is None:
            plots_dir = self.output_dir / "plots"
        else:
            plots_dir = Path(plots_dir)
        plots_dir.mkdir(exist_ok=True)
        
        # Get all mask names
        all_mask_names = list(self.masks.keys())
        
        # Calculate total voxels for all original masks
        original_total_voxels = 0
        for mask_name in all_mask_names:
            if mask_name in self.masks:
                original_total_voxels += np.sum(self.masks[mask_name])
        
        # Calculate total voxels for selected masks
        selected_total_voxels = 0
        for mask_name in selected_masks:
            if mask_name in self.masks:
                selected_total_voxels += np.sum(self.masks[mask_name])
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        
        categories = ['Original Masks\n(All)', 'Selected Masks\n(Top 4)']
        voxel_counts = [original_total_voxels, selected_total_voxels]
        colors = ['lightblue', 'lightgreen']
        
        bars = plt.bar(categories, voxel_counts, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, count in zip(bars, voxel_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(voxel_counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.ylabel('Total Voxel Count')
        plt.title('Total Voxel Count Comparison: Original vs Selected Masks')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add percentage reduction text
        reduction_percent = ((original_total_voxels - selected_total_voxels) / original_total_voxels) * 100
        plt.text(0.5, max(voxel_counts) * 0.8, 
                f'Reduction: {reduction_percent:.1f}%\n({len(all_mask_names)} → {len(selected_masks)} masks)',
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(plots_dir / "voxel_count_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Voxel count comparison plot saved to: {plots_dir}")
        print(f"Original total voxels: {original_total_voxels:,}")
        print(f"Selected total voxels: {selected_total_voxels:,}")
        print(f"Reduction: {reduction_percent:.1f}% ({len(all_mask_names)} → {len(selected_masks)} masks)")
    
    def get_mask_statistics(self):
        """Get basic statistics about the loaded masks"""
        if not self.masks:
            return None
            
        stats = {
            'total_masks': len(self.masks),
            'mask_names': list(self.masks.keys()),
            'total_voxels': sum(np.sum(mask) for mask in self.masks.values()),
            'average_voxels_per_mask': sum(np.sum(mask) for mask in self.masks.values()) / len(self.masks),
            'excluded_files': self.excluded_files
        }
        return stats