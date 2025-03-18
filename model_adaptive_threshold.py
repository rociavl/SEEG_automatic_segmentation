import os
import numpy as np
import nrrd  # Handles NRRD files
from scipy.stats import entropy, skew, kurtosis
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data_dir = "data/nrrd_masks/"  

best_thresholds = {
    "patient_1_mask_1.nrrd": 0.45,
    "patient_1_mask_2.nrrd": 0.50,
    "patient_2_mask_1.nrrd": 0.42,
}

def load_nrrd(file_path):
    """Load an NRRD file and normalize intensities to [0,1]."""
    volume, _ = nrrd.read(file_path)  
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-6)  
    return volume

def extract_features(mask, bins=50):
    """Extract histogram and statistical features from a 3D mask."""
    mask_flat = mask.flatten()
    mean_intensity = np.mean(mask_flat)
    var_intensity = np.var(mask_flat)
    std_intensity = np.std(mask_flat)
    skewness = skew(mask_flat)
    kurt = kurtosis(mask_flat)

    p10, p25, p50, p75, p90 = np.percentile(mask_flat, [10, 25, 50, 75, 90])
    hist, _ = np.histogram(mask_flat, bins=bins, range=(0,1), density=True)
    ent = entropy(hist + 1e-6)  # Avoid log(0)

    return np.concatenate([hist, [mean_intensity, var_intensity, std_intensity, skewness, kurt, p10, p25, p50, p75, p90, ent]])


file_paths = sorted(os.listdir(data_dir))  
masks, y_labels = [], []

for file_name in file_paths:
    if file_name.endswith(".nrrd") and file_name in best_thresholds:
        file_path = os.path.join(data_dir, file_name)
        mask = load_nrrd(file_path)
        masks.append(mask)
        y_labels.append(best_thresholds[file_name])

X = np.array([extract_features(mask) for mask in masks])
y = np.array(y_labels)  

# Leave-One-Out Cross-Validation (LOO-CV)
loo = LeaveOneOut()
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)  
errors = []
print("\nLeave-One-Out Cross-Validation Results:")

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    error = mean_absolute_error(y_test, y_pred)
    errors.append(error)
    print(f"Test: {test_idx[0]}, True: {y_test[0]:.3f}, Predicted: {y_pred[0]:.3f}, Error: {error:.3f}")

print(f"\n Average MAE: {np.mean(errors):.3f}")


def predict_threshold(new_mask):
    """Predict the best threshold for a new pre-thresholded 3D mask."""
    features = extract_features(new_mask).reshape(1, -1)
    return model.predict(features)[0]

new_mask_path = "data/nrrd_masks/new_patient_mask.nrrd"
new_mask = load_nrrd(new_mask_path)
predicted_threshold = predict_threshold(new_mask)

print(f"\nPredicted Optimal Threshold for New Mask: {predicted_threshold:.3f}")
