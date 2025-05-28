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
        Create comprehensive comparison plots between original and selected masks for SEEG electrode analysis.
        These plots help validate the mask selection strategy and assess the quality of the fusion process.
        
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
        """
        Plot histograms showing voxel-wise agreement across masks.
        
        This visualization helps understand the consensus patterns in brain segmentation,
        which is crucial for SEEG electrode localization accuracy. Higher agreement values
        indicate regions where multiple segmentation approaches consistently identify brain tissue.
        """
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
        
        # Create enhanced histograms with medical context
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Original masks overlap histogram
        max_votes_orig = len(original_masks)
        counts_orig, _, _ = ax1.hist(original_vote_map[original_vote_map > 0].flatten(), 
                                    bins=range(1, max_votes_orig + 2), 
                                    alpha=0.7, 
                                    color='steelblue',
                                    edgecolor='navy',
                                    linewidth=0.8)
        ax1.set_title(f"Original Masks: Inter-Rater Agreement Distribution\n"
                     f"Total Masks: {len(original_masks)} | Brain Segmentation Consensus", 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel("Number of Masks in Agreement\n(Higher values = stronger consensus)", fontsize=10)
        ax1.set_ylabel("Voxel Count", fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Add percentage annotations on bars
        total_voxels_orig = np.sum(counts_orig)
        for i, count in enumerate(counts_orig):
            if count > 0:
                percentage = (count / total_voxels_orig) * 100
                ax1.text(i + 1, count + max(counts_orig) * 0.01, 
                        f'{percentage:.1f}%', 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Selected masks overlap histogram
        max_votes_sel = len(selected_masks)
        counts_sel, _, _ = ax2.hist(selected_vote_map[selected_vote_map > 0].flatten(), 
                                   bins=range(1, max_votes_sel + 2), 
                                   alpha=0.7, 
                                   color='forestgreen',
                                   edgecolor='darkgreen',
                                   linewidth=0.8)
        ax2.set_title(f"Selected Masks: Optimized Agreement Distribution\n"
                     f"Selected: {len(selected_masks)} masks | Enhanced Consensus Quality", 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel("Number of Masks in Agreement\n(Optimized for electrode localization)", fontsize=10)
        ax2.set_ylabel("Voxel Count", fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Add percentage annotations on bars
        total_voxels_sel = np.sum(counts_sel)
        for i, count in enumerate(counts_sel):
            if count > 0:
                percentage = (count / total_voxels_sel) * 100
                ax2.text(i + 1, count + max(counts_sel) * 0.01, 
                        f'{percentage:.1f}%', 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Add interpretation text
        fig.text(0.5, 0.02, 
                "Interpretation: Higher agreement values indicate more reliable brain tissue identification.\n"
                "This is critical for accurate SEEG electrode placement and trajectory planning.",
                ha='center', fontsize=9, style='italic', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(plots_dir / "mask_overlap_histogram.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dice_scores(self, original_masks, selected_masks, fused_original, fused_selected, plots_dir):
        """
        Plot Dice similarity coefficients comparing individual masks to fused consensus masks.
        
        Dice scores measure spatial overlap quality, essential for validating brain segmentation
        accuracy in SEEG procedures. Higher scores indicate better agreement with the consensus.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
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
        
        # Create enhanced bar plots
        x_orig = np.arange(len(original_scores))
        x_sel = np.arange(len(selected_scores))
        
        # Original masks plot
        bars1 = ax1.bar(x_orig, original_scores, alpha=0.7, color='steelblue', 
                       edgecolor='navy', linewidth=0.8)
        ax1.axhline(y=np.mean(original_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(original_scores):.3f}', linewidth=2)
        ax1.set_xlabel('Mask Index', fontsize=10)
        ax1.set_ylabel('Dice Similarity Coefficient', fontsize=10)
        ax1.set_title(f'Original Masks vs. Consensus\n'
                     f'Mean Dice: {np.mean(original_scores):.3f} ± {np.std(original_scores):.3f}', 
                     fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim(0, 1.05)
        
        # Add value labels on bars
        for bar, score in zip(bars1, original_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)
        
        # Selected masks plot
        bars2 = ax2.bar(x_sel, selected_scores, alpha=0.7, color='forestgreen', 
                       edgecolor='darkgreen', linewidth=0.8)
        ax2.axhline(y=np.mean(selected_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(selected_scores):.3f}', linewidth=2)
        ax2.set_xlabel('Selected Mask Index', fontsize=10)
        ax2.set_ylabel('Dice Similarity Coefficient', fontsize=10)
        ax2.set_title(f'Selected Masks vs. Optimized Consensus\n'
                     f'Mean Dice: {np.mean(selected_scores):.3f} ± {np.std(selected_scores):.3f}', 
                     fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim(0, 1.05)
        
        # Add value labels on bars
        for bar, score in zip(bars2, selected_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)
        
        # Add clinical interpretation
        improvement = np.mean(selected_scores) - np.mean(original_scores)
        fig.text(0.5, 0.02, 
                f"Clinical Impact: Dice improvement of {improvement:+.3f} enhances brain boundary accuracy.\n"
                f"Values >0.8 indicate excellent segmentation quality for SEEG electrode localization.",
                ha='center', fontsize=9, style='italic',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen' if improvement > 0 else 'lightcoral', alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(plots_dir / "dice_score_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Segmentation Quality Assessment:")
        print(f"  Original masks average Dice: {np.mean(original_scores):.3f} ± {np.std(original_scores):.3f}")
        print(f"  Selected masks average Dice: {np.mean(selected_scores):.3f} ± {np.std(selected_scores):.3f}")
        print(f"  Quality improvement: {improvement:+.3f} ({improvement/np.mean(original_scores)*100:+.1f}%)")
    
    def _plot_mask_size_distribution(self, original_masks, selected_masks, plots_dir):
        """
        Analyze and visualize brain mask size distributions.
        
        Mask size analysis helps identify potential outliers and ensures consistent
        brain volume estimation, which affects electrode trajectory planning accuracy.
        """
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
        
        # Create enhanced visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Box plot comparison
        box_data = [original_sizes, selected_sizes]
        box_labels = [f'Original\n(n={len(original_sizes)})', f'Selected\n(n={len(selected_sizes)})']
        
        bp = ax1.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        
        # Color the boxes differently
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][1].set_facecolor('forestgreen')
        
        ax1.set_ylabel('Brain Mask Size (voxels)', fontsize=10)
        ax1.set_title('Brain Volume Distribution Analysis\nConsistency Check for SEEG Planning', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Add statistical annotations
        orig_stats = f"μ={np.mean(original_sizes):.0f}, σ={np.std(original_sizes):.0f}"
        sel_stats = f"μ={np.mean(selected_sizes):.0f}, σ={np.std(selected_sizes):.0f}"
        ax1.text(1, max(original_sizes) * 0.9, orig_stats, ha='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='steelblue', alpha=0.3))
        ax1.text(2, max(selected_sizes) * 0.9, sel_stats, ha='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='forestgreen', alpha=0.3))
        
        # Histogram overlay
        ax2.hist(original_sizes, bins=15, alpha=0.6, color='steelblue', 
                label=f'Original (n={len(original_sizes)})', density=True, edgecolor='navy')
        ax2.hist(selected_sizes, bins=15, alpha=0.6, color='forestgreen', 
                label=f'Selected (n={len(selected_sizes)})', density=True, edgecolor='darkgreen')
        
        ax2.axvline(np.mean(original_sizes), color='steelblue', linestyle='--', linewidth=2, alpha=0.8)
        ax2.axvline(np.mean(selected_sizes), color='forestgreen', linestyle='--', linewidth=2, alpha=0.8)
        
        ax2.set_xlabel('Brain Mask Size (voxels)', fontsize=10)
        ax2.set_ylabel('Density', fontsize=10)
        ax2.set_title('Size Distribution Comparison\nVolume Consistency Assessment', 
                     fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Add clinical context
        volume_ml_orig = np.mean(original_sizes) * np.prod(self.reference_spacing) / 1000  # Convert to mL
        volume_ml_sel = np.mean(selected_sizes) * np.prod(self.reference_spacing) / 1000
        
        fig.text(0.5, 0.02, 
                f"Clinical Context: Average brain volumes - Original: {volume_ml_orig:.0f}mL, Selected: {volume_ml_sel:.0f}mL\n"
                f"Consistent volumes ensure reliable electrode trajectory calculations and anatomical targeting.",
                ha='center', fontsize=9, style='italic',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(plots_dir / "mask_size_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Brain Volume Analysis:")
        print(f"  Original masks: {np.mean(original_sizes):.0f} ± {np.std(original_sizes):.0f} voxels "
              f"({volume_ml_orig:.0f} ± {np.std(original_sizes) * np.prod(self.reference_spacing) / 1000:.0f} mL)")
        print(f"  Selected masks: {np.mean(selected_sizes):.0f} ± {np.std(selected_sizes):.0f} voxels "
              f"({volume_ml_sel:.0f} ± {np.std(selected_sizes) * np.prod(self.reference_spacing) / 1000:.0f} mL)")
    
    def _plot_mask_scores(self, selected_masks, plots_dir):
        """
        Visualize the overlap scores that guided mask selection.
        
        These scores represent each mask's contribution to the global consensus,
        crucial for understanding the quality ranking in SEEG electrode segmentation.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        scores = [self.mask_scores[mask_name] for mask_name in selected_masks if mask_name in self.mask_scores]
        mask_indices = range(1, len(scores) + 1)
        
        # Create enhanced bar plot of scores
        bars = ax1.bar(mask_indices, scores, alpha=0.8, color='purple', 
                      edgecolor='darkviolet', linewidth=1.2)
        
        # Color gradient based on score
        norm = plt.Normalize(min(scores), max(scores))
        colors = plt.cm.viridis(norm(scores))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax1.set_xlabel('Mask Selection Order (Best to Worst)', fontsize=10)
        ax1.set_ylabel('Overlap Score with Global Consensus', fontsize=10)
        ax1.set_title('Mask Quality Ranking for SEEG Analysis\n'
                     'Higher Scores = Better Consensus Agreement', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Add score values on top of bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(scores) * 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Add rank labels
            ax1.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                    f'#{i+1}', ha='center', va='center', fontsize=10, 
                    color='white', fontweight='bold')
        
        # Score distribution and trends
        ax2.plot(mask_indices, scores, marker='o', linestyle='-', linewidth=3, 
                markersize=8, color='purple', markerfacecolor='white', 
                markeredgewidth=2, markeredgecolor='purple')
        ax2.fill_between(mask_indices, scores, alpha=0.3, color='purple')
        
        # Add trend line
        z = np.polyfit(mask_indices, scores, 1)
        p = np.poly1d(z)
        ax2.plot(mask_indices, p(mask_indices), "--", color='red', linewidth=2, 
                label=f'Trend: slope={z[0]:.4f}')
        
        ax2.set_xlabel('Mask Selection Order', fontsize=10)
        ax2.set_ylabel('Overlap Score', fontsize=10)
        ax2.set_title('Score Progression Analysis\n'
                     'Quality Degradation Pattern', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend()
        
        # Add percentile markers
        score_percentiles = [90, 75, 50, 25]
        percentile_values = np.percentile(scores, score_percentiles)
        for p, val in zip(score_percentiles, percentile_values):
            ax2.axhline(y=val, color='gray', linestyle=':', alpha=0.7)
            ax2.text(len(scores) * 0.95, val, f'{p}th %ile', fontsize=8, alpha=0.7)
        
        # Add selection quality assessment
        score_range = max(scores) - min(scores)
        consistency = 1 - (np.std(scores) / np.mean(scores))
        
        fig.text(0.5, 0.02, 
                f"Selection Quality: Range={score_range:.3f}, Consistency={consistency:.3f}\n"
                f"These scores guide optimal mask selection for precise SEEG electrode localization.",
                ha='center', fontsize=9, style='italic',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='plum', alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(plots_dir / "mask_scores.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Mask Selection Quality Metrics:")
        print(f"  Score range: {score_range:.3f}")
        print(f"  Score consistency: {consistency:.3f}")
        print(f"  Best mask score: {max(scores):.3f}")
        print(f"  Worst selected mask score: {min(scores):.3f}")
    
    def plot_progressive_fused_masks_comparison(self, progressive_masks, plots_dir=None):
        """
        Analyze the progressive fusion process to understand how mask quality evolves.
        
        This analysis is crucial for determining the optimal number of masks needed
        for reliable brain segmentation in SEEG electrode placement procedures.
        
        Args:
            progressive_masks: List of progressively fused mask arrays
            plots_dir: Directory for saving plots
        """
        if plots_dir is None:
            plots_dir = self.output_dir / "plots"
        else:
            plots_dir = Path(plots_dir)
        plots_dir.mkdir(exist_ok=True)
        
        # Enhanced plot for mask sizes progression
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        sizes = [np.sum(mask) for mask in progressive_masks]
        mask_indices = range(1, len(sizes) + 1)
        
        # 1. Mask size progression
        ax1.plot(mask_indices, sizes, marker='o', linestyle='-', linewidth=3, 
                markersize=8, color='darkblue', markerfacecolor='lightblue', 
                markeredgewidth=2, markeredgecolor='darkblue')
        ax1.fill_between(mask_indices, sizes, alpha=0.3, color='lightblue')
        
        # Add trend analysis
        if len(sizes) > 2:
            z = np.polyfit(mask_indices, sizes, 1)
            p = np.poly1d(z)
            ax1.plot(mask_indices, p(mask_indices), "--", color='red', linewidth=2, 
                    label=f'Trend: {z[0]:+.1f} voxels/mask')
            ax1.legend()
        
        ax1.set_xlabel('Number of Masks in Progressive Fusion', fontsize=10)
        ax1.set_ylabel('Brain Mask Size (voxels)', fontsize=10)
        ax1.set_title('Progressive Fusion: Brain Volume Evolution\n'
                     'Convergence to Optimal Brain Boundary', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xticks(mask_indices)
        
        # Add size annotations
        for i, size in enumerate(sizes):
            ax1.text(i + 1, size + max(sizes) * 0.02, f'{size:,}', 
                    ha='center', va='bottom', fontsize=8, rotation=45)
        
        # 2. Size change rate
        if len(sizes) > 1:
            size_changes = [sizes[i] - sizes[i-1] for i in range(1, len(sizes))]
            change_indices = range(2, len(sizes) + 1)
            
            colors = ['green' if change >= 0 else 'red' for change in size_changes]
            bars = ax2.bar(change_indices, size_changes, color=colors, alpha=0.7, 
                          edgecolor='black', linewidth=0.8)
            
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax2.set_xlabel('Fusion Step (Adding Nth Mask)', fontsize=10)
            ax2.set_ylabel('Volume Change (voxels)', fontsize=10)
            ax2.set_title('Incremental Volume Changes\n'
                         'Impact of Each Additional Mask', 
                         fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            
            # Add change annotations
            for bar, change in zip(bars, size_changes):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., 
                        height + (max(size_changes) - min(size_changes)) * 0.02 * (1 if height >= 0 else -1),
                        f'{change:+,}', ha='center', 
                        va='bottom' if height >= 0 else 'top', fontsize=8)
        
        # 3. Dice similarity between consecutive masks
        if len(progressive_masks) > 1:
            dice_scores = []
            for i in range(1, len(progressive_masks)):
                dice = self.dice_score(progressive_masks[i-1], progressive_masks[i])
                dice_scores.append(dice)
            
            ax3.plot(range(2, len(progressive_masks) + 1), dice_scores, 
                    marker='s', linestyle='-', linewidth=3, markersize=8, 
                    color='green', markerfacecolor='lightgreen', 
                    markeredgewidth=2, markeredgecolor='darkgreen')
            ax3.fill_between(range(2, len(progressive_masks) + 1), dice_scores, 
                           alpha=0.3, color='lightgreen')
            
            # Add stability threshold
            stability_threshold = 0.95
            ax3.axhline(y=stability_threshold, color='orange', linestyle='--', 
                       linewidth=2, label=f'Stability Threshold ({stability_threshold})')
            ax3.legend()
            
            ax3.set_xlabel('Number of Masks in Fusion', fontsize=10)
            ax3.set_ylabel('Dice Similarity to Previous Step', fontsize=10)
            ax3.set_title('Fusion Stability Analysis\n'
                         'Convergence to Stable Brain Boundary', 
                         fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, linestyle='--')
            ax3.set_xticks(range(2, len(progressive_masks) + 1))
            ax3.set_ylim(0.8, 1.02)
            
            # Add annotations for convergence
            for i, dice in enumerate(dice_scores):
                ax3.text(i + 2, dice + 0.005, f'{dice:.3f}', 
                        ha='center', va='bottom', fontsize=8)
            
            # Find optimal number of masks (where stability is achieved)
            stable_indices = [i for i, score in enumerate(dice_scores) if score >= stability_threshold]
            if stable_indices:
                optimal_n = stable_indices[0] + 2  # +2 because dice_scores starts from index 2
                ax3.axvline(x=optimal_n, color='red', linestyle=':', linewidth=2, 
                           label=f'Optimal N={optimal_n}')
                ax3.legend()
        
        # 4. Efficiency analysis - Quality vs Number of Masks
        # Calculate a quality metric combining size stability and Dice scores
        if len(progressive_masks) > 1 and 'dice_scores' in locals():
            # Normalize metrics
            size_stability = [1 - abs(change)/max(sizes) for change in size_changes] if len(size_changes) > 0 else []
            dice_stability = dice_scores
            
            if len(size_stability) == len(dice_stability):
                efficiency_scores = [(d + s)/2 for d, s in zip(dice_stability, size_stability)]
                
                ax4.plot(range(2, len(progressive_masks) + 1), efficiency_scores, 
                        marker='D', linestyle='-', linewidth=3, markersize=8, 
                        color='purple', markerfacecolor='plum', 
                        markeredgewidth=2, markeredgecolor='purple')
                ax4.fill_between(range(2, len(progressive_masks) + 1), efficiency_scores, 
                               alpha=0.3, color='plum')
                
                # Find the knee point (optimal efficiency)
                if len(efficiency_scores) >= 3:
                    # Simple knee detection - look for diminishing returns
                    improvements = [efficiency_scores[i] - efficiency_scores[i-1] 
                                  for i in range(1, len(efficiency_scores))]
                    if improvements:
                        knee_idx = next((i for i, imp in enumerate(improvements) 
                                       if imp < 0.01), len(improvements))
                        knee_point = knee_idx + 3  # +3 because we start from index 2 and add 1
                        
                        ax4.axvline(x=knee_point, color='red', linestyle='--', linewidth=2, 
                                   label=f'Efficiency Knee Point (N={knee_point})')
                        ax4.legend()
                
                ax4.set_xlabel('Number of Masks in Fusion', fontsize=10)
                ax4.set_ylabel('Combined Efficiency Score', fontsize=10)
                ax4.set_title('Fusion Efficiency Analysis\n'
                             'Optimal Balance of Quality vs. Complexity', 
                             fontsize=12, fontweight='bold')
                ax4.grid(True, alpha=0.3, linestyle='--')
                ax4.set_xticks(range(2, len(progressive_masks) + 1))
                ax4.set_ylim(0, 1.05)
        
        # Add comprehensive clinical interpretation
        volume_ml = [size * np.prod(self.reference_spacing) / 1000 for size in sizes]
        final_volume = volume_ml[-1] if volume_ml else 0
        volume_stability = np.std(volume_ml[-3:]) if len(volume_ml) >= 3 else np.std(volume_ml)
        
        fig.text(0.5, 0.02, 
                f"Clinical Summary: Final brain volume = {final_volume:.0f}mL, "
                f"Volume stability = ±{volume_stability:.1f}mL\n"
                f"Progressive fusion analysis guides optimal mask selection for SEEG electrode trajectory planning.\n"
                f"Stable convergence ensures reliable anatomical targeting and surgical safety.",
                ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcyan', alpha=0.7))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        plt.savefig(plots_dir / "progressive_fusion_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print detailed analysis
        print(f"\nProgressive Fusion Analysis:")
        print(f"  Final brain volume: {final_volume:.0f} mL")
        print(f"  Volume range: {min(volume_ml):.0f} - {max(volume_ml):.0f} mL")
        print(f"  Volume stability (last 3): ±{volume_stability:.1f} mL")
        if 'dice_scores' in locals():
            print(f"  Average Dice stability: {np.mean(dice_scores):.3f}")
            print(f"  Minimum Dice score: {min(dice_scores):.3f}")
        if 'optimal_n' in locals():
            print(f"  Recommended optimal masks: {optimal_n}")

def main():
    """
    Main execution function to select and fuse the best masks for SEEG electrode analysis
    """
    # Paths - update these to your actual paths
    mask_folder_path = r"C:\Users\rocia\Downloads\P6_ELECTRODES_MASK"
    output_dir = r"C:\Users\rocia\Downloads\TFG\Cohort\Enhance_ctp_tests\P6_weighted_fusion_fix_enhanced_plots_for_validation"
    
    # Initialize the enhanced mask selector
    selector = EnhancedMaskSelector(mask_folder_path, output_dir)
    
    # Compute global agreement map across all masks
    selector.compute_global_agreement()
    
    # Select the best 10 masks based on their overlap score with the global vote map
    selected_masks = selector.select_best_masks(n_masks=10)
    
    # Save the individual top 10 masks
    for i, mask_name in enumerate(selected_masks, 1):
        selector.save_mask(selector.masks[mask_name], f"P6_top_mask_{i}")
    
    # Get all mask names for reference
    all_mask_names = list(selector.masks.keys())
    
    # Create a traditional fused mask from all masks (for comparison)
    # This uses a simple threshold of 45% agreement
    all_vote_map = np.zeros_like(next(iter(selector.masks.values())))
    for mask_name in all_mask_names:
        all_vote_map += selector.masks[mask_name]
    threshold = len(all_mask_names) * 0.45
    fused_all = np.where(all_vote_map >= threshold, 1, 0).astype(np.uint8)
    selector.save_mask(fused_all, "P6_all_masks_fused_traditional")
    
    # Create a single weighted fused mask from the top 10 masks
    fused_weighted = selector.create_weighted_fused_mask(selected_masks, "P6_top10_weighted_fused")
    
    # Create 10 progressive fused masks, incrementally adding one mask at a time
    # based on their score (from highest to lowest)
    progressive_masks = selector.create_progressive_fused_masks(selected_masks, "P6_progressive")
    
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

#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Masks_ensemble\masks_fusion.py').read())