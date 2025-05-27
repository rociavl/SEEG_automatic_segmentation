import numpy as np
import os
import csv
import slicer
from scipy.stats import kurtosis, skew

# --- CONFIGURATION ---
CT = slicer.util.getNode('P3_CTp.3D')

fix_threshold_ranges_CT = [2400]

output_dir_CT = r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Image_bolt_head_model"
#output_dir_CT_AUGMENTED = r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Image_bolt_head_model"

tail_percent = 0.01  # Top 0.1%


# --- FUNCTIONS ---
def extract_histogram_features(volumeNode, name, fixed_threshold=None, tail_percent=0.01):
    """Enhanced histogram feature extraction with 2400 HU-specific features"""
    
    # Get raw CT data and remove background (values <= -1000)
    array = slicer.util.arrayFromVolume(volumeNode).flatten()
    filtered_array = array[array > -1000]
    total_voxels = len(filtered_array)
    
    # 2400 HU-specific features
    target_hu = 2400
    hu_features = {
        'voxels_at_2400±100': np.sum((filtered_array >= 2300) & (filtered_array <= 2500)),
        'voxels_above_2400': np.sum(filtered_array > target_hu),
        'ratio_above_2400': np.sum(filtered_array > target_hu) / total_voxels,
        'hu_2400_percentile': np.mean(filtered_array > target_hu) * 100,  # % of voxels > 2400
        'density_gradient_2300_2500': np.mean(filtered_array[(filtered_array >= 2300) & 
                                                          (filtered_array <= 2500)]) - target_hu
    }
    
    # Basic statistics
    stats = {
        "name": name,
        "min": np.min(filtered_array),
        "max": np.max(filtered_array),
        "mean": np.mean(filtered_array),
        "std": np.std(filtered_array),
        "median": np.median(filtered_array),
        "threshold": fixed_threshold,
        "total_voxels": total_voxels

    }
    
    # Tail statistics (top percent of values)
    n_tail = max(1, int(tail_percent * total_voxels))
    tail_values = np.sort(filtered_array)[-n_tail:]
    tail_stats = {
        "tail_min": np.min(tail_values),
        "tail_max": np.max(tail_values),
        "tail_mean": np.mean(tail_values),
        "tail_std": np.std(tail_values),
        "tail_median": np.median(tail_values)
    }
    
    # High percentiles (now including 2400 if it falls within these ranges)
    percentiles = {
        "percentile_99.5": np.percentile(filtered_array, 99.5),
        "percentile_99.7": np.percentile(filtered_array, 99.7),
        "percentile_99.8": np.percentile(filtered_array, 99.8),
        "percentile_99.9": np.percentile(filtered_array, 99.9),
        "percentile_99.95": np.percentile(filtered_array, 99.95),
        "percentile_99.97": np.percentile(filtered_array, 99.97),
        "percentile_99.98": np.percentile(filtered_array, 99.98),
        "percentile_99.99": np.percentile(filtered_array, 99.99)
    }
    
    # Histogram features
    hist, bin_edges = np.histogram(filtered_array, bins=512)
    hist_features = {
        "hist_peak_bin": bin_edges[np.argmax(hist)],
        "hist_peak_count": np.max(hist),
        "hist_skew": skew(filtered_array),
        "hist_kurtosis": kurtosis(filtered_array),
        "hist_2400_bin_count": hist[np.argmin(np.abs(bin_edges[:-1] - target_hu))] if len(hist) > 0 else 0
    }
    
    # Combine all features
    return {
        **stats,
        **tail_stats,
        **percentiles,
        **hist_features,
        **hu_features,
        
        # Additional distribution metrics
        "mean_above_2400": np.mean(filtered_array[filtered_array > target_hu]) if np.any(filtered_array > target_hu) else 0,
        "std_above_2400": np.std(filtered_array[filtered_array > target_hu]) if np.sum(filtered_array > target_hu) > 1 else 0,
        "thershold": fixed_threshold
    }


def save_features_to_csv(features_list, output_path):
    if not features_list:
        print("No features to save.")
        return
    keys = list(features_list[0].keys())
    with open(output_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in features_list:
            writer.writerow(row)


# --- ORIGINAL CT FEATURES (MULTIPLE RANGES) ---
features_original = []
for i, threshold in enumerate(fix_threshold_ranges_CT):
    features_original.append(
        extract_histogram_features(
            CT,
            name=f"P3_CT_original_threshold_{i+1}",
            fixed_threshold=threshold
        )
    )

# --- AUGMENTED CT FEATURES ---
# features_augmented = []
# for name, threshold in CT_AUGMENTED.items():
#     volumeNode = slicer.util.getNode(name)
#     features = extract_histogram_features(volumeNode, name, fixed_threshold=threshold)
#     features_augmented.append(features)

# --- SAVE RESULTS ---
save_features_to_csv(features_original, os.path.join(output_dir_CT, "features_P3_CT.csv"))
#save_features_to_csv(features_augmented, os.path.join(output_dir_CT_AUGMENTED, "features_P3_CT_augmented.csv"))

print("✅ Feature extraction complete.")


# exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Bolt_head\Threshold\features_extraction.py').read())




