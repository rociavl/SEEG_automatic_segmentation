import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path

class EnhancedMaskSelector:
    def __init__(self, mask_folder_path, output_dir):
        """
        Initialize the enhanced mask selector with the path to the masks folder and output directory.
        
        Args:
            mask_folder_path: Path to the folder containing mask files
            output_dir: Path to the output directory for saving results
        """
        self.mask_folder_path = Path(mask_folder_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
        self.global_vote_map = np.zeros_like(next(iter(self.masks.values())))
        
    def load_all_masks(self):
        """Load all NRRD mask files from the specified folder"""
        print(f"Loading masks from {self.mask_folder_path}")
        
        mask_files = list(self.mask_folder_path.glob("*.nrrd"))
        if not mask_files:
            raise ValueError(f"No NRRD files found in {self.mask_folder_path}")
            
        print(f"Found {len(mask_files)} mask files")
        
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
    
    def dice_score(self, mask1, mask2):
        """
        Compute Dice similarity coefficient between two binary masks.
        
        Args:
            mask1: First binary mask
            mask2: Second binary mask
            
        Returns:
            dice: Dice score between 0 and 1
        """
        intersection = np.sum(mask1 * mask2)
        sum_masks = np.sum(mask1) + np.sum(mask2)
        
        if sum_masks == 0:
            return 0.0
            
        return 2.0 * intersection / sum_masks
    
    def select_best_masks(self, n_masks=10):
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
    
    def create_progressive_fused_masks(self, mask_names, base_output_name):
        """
        Create a series of incrementally fused masks, adding one mask at a time
        based on their order in mask_names (which should be ordered by score).
        
        Args:
            mask_names: List of mask names ordered by score
            base_output_name: Base name for the output fused masks
        
        Returns:
            list of fused masks
        """
        fused_masks = []
        
        for i in range(1, len(mask_names) + 1):
            # Get subset of masks
            subset_masks = mask_names[:i]
            
            # Create fused mask from this subset
            output_name = f"{base_output_name}_{i}_masks"
            fused_mask = self.create_weighted_fused_mask(subset_masks, output_name)
            fused_masks.append(fused_mask)
            
            print(f"Created progressive fused mask {i} from {len(subset_masks)} masks")
            
        return fused_masks
        
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
    
    def create_comparison_plots(self, original_masks, selected_masks, fused_original, fused_selected, plots_dir=None):
        """
        Create comparison plots between original and selected masks.
        
        Args:
            original_masks: List of original mask names
            selected_masks: List of selected mask names
            fused_original: Original fused mask array
            fused_selected: Selected fused mask array
            plots_dir: Directory for saving plots (default: within output_dir)
        """
        # Create a directory for plots
        if plots_dir is None:
            plots_dir = self.output_dir / "plots"
        else:
            plots_dir = Path(plots_dir)
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Plot mask overlap histograms
        self._plot_mask_overlap(original_masks, selected_masks, plots_dir)
        
        # 2. Plot Dice scores
        self._plot_dice_scores(original_masks, selected_masks, fused_original, fused_selected, plots_dir)
        
        # 3. Plot mask size distribution
        self._plot_mask_size_distribution(original_masks, selected_masks, plots_dir)
        
        # 4. Plot the scores of selected masks
        self._plot_mask_scores(selected_masks, plots_dir)
        
        print(f"Saved comparison plots to: {plots_dir}")
    
    def _plot_mask_overlap(self, original_masks, selected_masks, plots_dir):
        """Plot histograms showing mask overlap distribution"""
        # Compute original overlap
        original_vote_map = np.zeros_like(next(iter(self.masks.values())))
        for mask_name in original_masks:
            if mask_name in self.masks:
                original_vote_map += self.masks[mask_name]
        
        # Compute selected overlap
        selected_vote_map = np.zeros_like(next(iter(self.masks.values())))
        for mask_name in selected_masks:
            if mask_name in self.masks:
                selected_vote_map += self.masks[mask_name]
        
        # Create histograms
        plt.figure(figsize=(12, 6))
        
        # Original masks overlap histogram
        plt.subplot(1, 2, 1)
        max_votes = len(original_masks)
        plt.hist(original_vote_map[original_vote_map > 0].flatten(), 
                 bins=range(1, max_votes + 2), 
                 alpha=0.7, 
                 color='blue')
        plt.title(f"Original Masks Overlap Distribution (n={len(original_masks)})")
        plt.xlabel("Number of Masks Agreeing")
        plt.ylabel("Voxel Count")
        plt.grid(True, alpha=0.3)
        
        # Selected masks overlap histogram
        plt.subplot(1, 2, 2)
        max_votes = len(selected_masks)
        plt.hist(selected_vote_map[selected_vote_map > 0].flatten(), 
                 bins=range(1, max_votes + 2), 
                 alpha=0.7, 
                 color='green')
        plt.title(f"Selected Masks Overlap Distribution (n={len(selected_masks)})")
        plt.xlabel("Number of Masks Agreeing")
        plt.ylabel("Voxel Count")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "mask_overlap_histogram.png", dpi=300)
        plt.close()
    
    def _plot_dice_scores(self, original_masks, selected_masks, fused_original, fused_selected, plots_dir):
        """Plot Dice scores comparing each mask to the fused mask"""
        plt.figure(figsize=(10, 6))
        
        # Compute Dice scores for original masks
        original_scores = []
        for mask_name in original_masks:
            if mask_name in self.masks:
                score = self.dice_score(self.masks[mask_name], fused_original)
                original_scores.append(score)
        
        # Compute Dice scores for selected masks
        selected_scores = []
        for mask_name in selected_masks:
            if mask_name in self.masks:
                score = self.dice_score(self.masks[mask_name], fused_selected)
                selected_scores.append(score)
        
        # Create bar plot
        x = np.arange(max(len(original_scores), len(selected_scores)))
        width = 0.35
        
        plt.bar(x[:len(original_scores)] - width/2, original_scores, width, label='Original Masks', alpha=0.7)
        plt.bar(x[:len(selected_scores)] + width/2, selected_scores, width, label='Selected Masks', alpha=0.7)
        
        plt.xlabel('Mask Index')
        plt.ylabel('Dice Score')
        plt.title('Dice Score Comparison with Fused Mask')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "dice_score_comparison.png", dpi=300)
        plt.close()
        
        print(f"Original masks average Dice: {np.mean(original_scores):.3f}")
        print(f"Selected masks average Dice: {np.mean(selected_scores):.3f}")
    
    def _plot_mask_size_distribution(self, original_masks, selected_masks, plots_dir):
        """Plot the distribution of mask sizes"""
        plt.figure(figsize=(10, 6))
        
        # Compute sizes for original masks
        original_sizes = []
        for mask_name in original_masks:
            if mask_name in self.masks:
                original_sizes.append(np.sum(self.masks[mask_name]))
        
        # Compute sizes for selected masks
        selected_sizes = []
        for mask_name in selected_masks:
            if mask_name in self.masks:
                selected_sizes.append(np.sum(self.masks[mask_name]))
        
        # Create box plot
        plt.boxplot([original_sizes, selected_sizes], 
                   tick_labels=[f'Original (n={len(original_sizes)})', f'Selected (n={len(selected_sizes)})'])
        
        plt.ylabel('Mask Size (voxels)')
        plt.title('Mask Size Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "mask_size_distribution.png", dpi=300)
        plt.close()
        
        print(f"Original masks average size: {np.mean(original_sizes):.1f} voxels")
        print(f"Selected masks average size: {np.mean(selected_sizes):.1f} voxels")
    
    def _plot_mask_scores(self, selected_masks, plots_dir):
        """Plot the scores of selected masks"""
        plt.figure(figsize=(12, 6))
        
        scores = [self.mask_scores[mask_name] for mask_name in selected_masks if mask_name in self.mask_scores]
        mask_indices = range(1, len(scores) + 1)
        
        # Create bar plot of scores
        plt.bar(mask_indices, scores, alpha=0.7, color='purple')
        
        plt.xlabel('Selected Mask Index')
        plt.ylabel('Overlap Score')
        plt.title('Overlap Scores of Selected Masks')
        plt.grid(True, alpha=0.3)
        
        # Add score values on top of bars
        for i, score in enumerate(scores):
            plt.text(i + 1, score + 0.01, f"{score:.3f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "mask_scores.png", dpi=300)
        plt.close()
    
    def plot_progressive_fused_masks_comparison(self, progressive_masks, plots_dir=None):
        """
        Plot comparison of the progressive fused masks.
        
        Args:
            progressive_masks: List of progressively fused mask arrays
            plots_dir: Directory for saving plots
        """
        if plots_dir is None:
            plots_dir = self.output_dir / "plots"
        else:
            plots_dir = Path(plots_dir)
        plots_dir.mkdir(exist_ok=True)
        
        # Plot sizes of progressive masks
        plt.figure(figsize=(10, 6))
        
        sizes = [np.sum(mask) for mask in progressive_masks]
        mask_indices = range(1, len(sizes) + 1)
        
        plt.plot(mask_indices, sizes, marker='o', linestyle='-', linewidth=2)
        plt.xlabel('Number of Masks in Fusion')
        plt.ylabel('Fused Mask Size (voxels)')
        plt.title('Fused Mask Size vs. Number of Masks Used')
        plt.grid(True, alpha=0.3)
        plt.xticks(mask_indices)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "progressive_mask_sizes.png", dpi=300)
        plt.close()
        
        # Plot Dice similarity between consecutive masks
        if len(progressive_masks) > 1:
            plt.figure(figsize=(10, 6))
            
            dice_scores = []
            for i in range(1, len(progressive_masks)):
                dice = self.dice_score(progressive_masks[i-1], progressive_masks[i])
                dice_scores.append(dice)
            
            plt.plot(range(2, len(progressive_masks) + 1), dice_scores, marker='o', linestyle='-', linewidth=2)
            plt.xlabel('Number of Masks in Fusion')
            plt.ylabel('Dice Similarity to Previous Fusion')
            plt.title('Dice Similarity Between Consecutive Fused Masks')
            plt.grid(True, alpha=0.3)
            plt.xticks(range(2, len(progressive_masks) + 1))
            
            plt.tight_layout()
            plt.savefig(plots_dir / "progressive_mask_dice.png", dpi=300)
            plt.close()

def main():
    """
    Main execution function to select and fuse the best masks
    """
    # Paths - update these to your actual paths
    mask_folder_path = r"C:\Users\rocia\Downloads\P6_ELECTRODES_MASK"
    output_dir = r"C:\Users\rocia\Downloads\TFG\Cohort\Enhance_ctp_tests\P6_weighted_fusion_fix"
    
    # Initialize the enhanced mask selector
    selector = EnhancedMaskSelector(mask_folder_path, output_dir)
    
    # Compute global agreement map across all masks
    selector.compute_global_agreement()
    
    # Select the best 10 masks based on their overlap score with the global vote map
    selected_masks = selector.select_best_masks(n_masks=10)
    
    # Save the individual top 10 masks
    for i, mask_name in enumerate(selected_masks, 1):
        selector.save_mask(selector.masks[mask_name], f"P8_top_mask_{i}")
    
    # Get all mask names for reference
    all_mask_names = list(selector.masks.keys())
    
    # Create a traditional fused mask from all masks (for comparison)
    # This uses a simple threshold of 45% agreement
    all_vote_map = np.zeros_like(next(iter(selector.masks.values())))
    for mask_name in all_mask_names:
        all_vote_map += selector.masks[mask_name]
    threshold = len(all_mask_names) * 0.45
    fused_all = np.where(all_vote_map >= threshold, 1, 0).astype(np.uint8)
    selector.save_mask(fused_all, "P8_all_masks_fused_traditional")
    
    # Create a single weighted fused mask from the top 10 masks
    fused_weighted = selector.create_weighted_fused_mask(selected_masks, "P8_top10_weighted_fused")
    
    # Create 10 progressive fused masks, incrementally adding one mask at a time
    # based on their score (from highest to lowest)
    progressive_masks = selector.create_progressive_fused_masks(selected_masks, "P8_progressive")
    
    # Create comparison plots
    selector.create_comparison_plots(all_mask_names, selected_masks, fused_all, fused_weighted)
    
    # Plot additional comparisons for the progressive masks
    selector.plot_progressive_fused_masks_comparison(progressive_masks)
    
    # Print mask scores for reference
    print("\nSelected mask scores:")
    for i, mask_name in enumerate(selected_masks, 1):
        score = selector.mask_scores.get(mask_name, 0)
        print(f"{i}. {mask_name}: {score:.4f}")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()