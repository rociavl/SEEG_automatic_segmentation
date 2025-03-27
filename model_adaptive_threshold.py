import os
import numpy as np
import nrrd
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
import joblib

data_dir = "data/nrrd_masks/"
best_thresholds = {  
    "patient_1_mask_1.nrrd": 450,
    "patient_1_mask_2.nrrd": 500,
    "patient_2_mask_1.nrrd": 420,
}

def load_nrrd(file_path):
    """Load raw NRRD file without normalization."""
    volume, _ = nrrd.read(file_path)
    return volume

def extract_histogram_features(mask, bins=20):
    """Extract features from raw intensity histogram."""
    mask_flat = mask.flatten()
    vmin, vmax = np.min(mask_flat), np.max(mask_flat)
    hist, bin_edges = np.histogram(mask_flat, bins=bins, range=(vmin, vmax), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    mode_idx = np.argmax(hist)
    mode_intensity = bin_centers[mode_idx]
    hist_ent = entropy(hist + 1e-6)
    cum_hist = np.cumsum(hist) / np.sum(hist)
    p50_intensity = bin_centers[np.searchsorted(cum_hist, 0.5)]
    intensity_range = vmax - vmin
    
    return np.array([mode_intensity, hist_ent, p50_intensity, intensity_range])

# Load and prepare data (assuming ~20 masks)
file_paths = sorted(os.listdir(data_dir))
X, y = [], []
for file_name in file_paths:
    if file_name.endswith(".nrrd") and file_name in best_thresholds:
        file_path = os.path.join(data_dir, file_name)
        mask = load_nrrd(file_path)
        features = extract_histogram_features(mask)
        X.append(features)
        y.append(best_thresholds[file_name])

X = np.array(X)
y = np.array(y)
print(f"Loaded {len(y)} masks for training.")

# Model comparison: Linear Regression vs. Random Forest
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42)
}

loo = LeaveOneOut()

for model_name, model in models.items():
    errors = []
    print(f"\n{model_name} - Leave-One-Out Cross-Validation Results:")
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        error = mean_absolute_error(y_test, y_pred)
        errors.append(error)
        print(f"Test Mask: {test_idx[0]}, True: {y_test[0]:.1f}, Predicted: {y_pred[0]:.1f}, Error: {error:.1f}")
    
    avg_mae = np.mean(errors)
    print(f"\n{model_name} Average MAE: {avg_mae:.1f}")
    model.fit(X, y)
    model_file = f"raw_histogram_{model_name.lower().replace(' ', '_')}_model_loo.pkl"
    joblib.dump(model, model_file)


def predict_threshold(mask, model_path="raw_histogram_random_forest_model_loo.pkl", bins=20):
    """Predict threshold for a new mask."""
    model = joblib.load(model_path)
    features = extract_histogram_features(mask, bins).reshape(1, -1)
    return model.predict(features)[0]

# Test
new_mask_path = "data/nrrd_masks/new_patient_mask.nrrd"
new_mask = load_nrrd(new_mask_path)
predicted_threshold = predict_threshold(new_mask)  # Using Random Forest by default
print(f"\nPredicted Threshold (raw intensity, Random Forest): {predicted_threshold:.1f}")

#Plot
import matplotlib.pyplot as plt
hist, bin_edges = np.histogram(new_mask.flatten(), bins=20, density=True)
plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="black")
plt.axvline(predicted_threshold, color="red", linestyle="--", label=f"Threshold: {predicted_threshold:.1f}")
plt.xlabel("Raw Intensity")
plt.ylabel("Density")
plt.legend()
plt.show()