import os
import numpy as np
import joblib
import slicer
from slicer.util import *

def shannon_entropy(image):
    """Calculate Shannon entropy of an image."""
    # Convert to probabilities by calculating histogram
    hist, _ = np.histogram(image, bins=256, density=True)
    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]
    # Calculate entropy
    return -np.sum(hist * np.log2(hist))

def extractFeatures(volumeNode):
    """Extract features from volume using your feature extraction code."""
    from scipy import stats
    
    # Get volume data as numpy array
    volume_array = slicer.util.arrayFromVolume(volumeNode)
    if volume_array is None or volume_array.size == 0:
        raise ValueError("Input volume data is empty or invalid")
    
    # Initialize features dictionary
    features = {}
    
    # Basic statistics
    features['min'] = np.min(volume_array)
    features['max'] = np.max(volume_array)
    features['mean'] = np.mean(volume_array)
    features['median'] = np.median(volume_array)
    features['std'] = np.std(volume_array)
    features['p25'] = np.percentile(volume_array, 25)
    features['p75'] = np.percentile(volume_array, 75)
    features['p95'] = np.percentile(volume_array, 95)
    features['p99'] = np.percentile(volume_array, 99)

    
    # Compute histogram
    hist, bin_edges = np.histogram(volume_array.flatten(), bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Handle zero-peak special case for small dot segmentation
    zero_idx = np.argmin(np.abs(bin_centers))  # Index closest to zero
    zero_peak_height = hist[zero_idx]
    features['zero_peak_height'] = zero_peak_height
    features['zero_peak_ratio'] = zero_peak_height / np.sum(hist) if np.sum(hist) > 0 else 0
    
    # Add very high percentiles
    features['p99.5'] = np.percentile(volume_array, 99.5)
    features['p99.9'] = np.percentile(volume_array, 99.9)
    features['p99.99'] = np.percentile(volume_array, 99.99)

    # features['p99.8'] = np.percentile(volume_array, 99.5)
    # features['p99.95'] = np.percentile(volume_array, 99.9)
    # features['p99.99'] = np.percentile(volume_array, 99.99)
    # features['p99.999'] = np.percentile(volume_array, 99.999)
    
    # Calculate skewness and kurtosis
    features['skewness'] = stats.skew(volume_array.flatten())
    features['kurtosis'] = stats.kurtosis(volume_array.flatten())
    
    # Calculate non-zero statistics (ignoring background)
    non_zero_values = volume_array[volume_array > 0]
    if len(non_zero_values) > 0:
        features['non_zero_min'] = np.min(non_zero_values)
        features['non_zero_mean'] = np.mean(non_zero_values)
        features['non_zero_median'] = np.median(non_zero_values)
        features['non_zero_std'] = np.std(non_zero_values)
        features['non_zero_count'] = len(non_zero_values)
        features['non_zero_ratio'] = len(non_zero_values) / volume_array.size
        if len(non_zero_values) > 3:
            features['non_zero_skewness'] = stats.skew(non_zero_values)
            features['non_zero_kurtosis'] = stats.kurtosis(non_zero_values)
        else:
            features['non_zero_skewness'] = 0
            features['non_zero_kurtosis'] = 0
    else:
        features['non_zero_min'] = 0
        features['non_zero_mean'] = 0
        features['non_zero_median'] = 0
        features['non_zero_std'] = 0
        features['non_zero_count'] = 0
        features['non_zero_ratio'] = 0
        features['non_zero_skewness'] = 0
        features['non_zero_kurtosis'] = 0
    
    # # High-intensity island statistics
    # high_threshold = features['p99']
    # high_values = volume_array[volume_array >= high_threshold]
    # if len(high_values) > 0:
    #     features['high_intensity_count'] = len(high_values)
    #     features['high_intensity_mean'] = np.mean(high_values)
    #     features['high_intensity_max'] = np.max(high_values)
    #     features['high_intensity_ratio'] = len(high_values) / volume_array.size
    # else:
    #     features['high_intensity_count'] = 0
    #     features['high_intensity_mean'] = 0
    #     features['high_intensity_max'] = 0
    #     features['high_intensity_ratio'] = 0
    
    # Find peaks (ignoring the zero peak if it's dominant)
    peaks = []
    for i in range(1, len(hist)-1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            if abs(bin_centers[i]) > 0.01:  # Small tolerance to avoid numerical issues
                peaks.append((bin_centers[i], hist[i]))
    
    # Sort peaks by height (descending)
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    # Extract info about top non-zero peaks
    if peaks:
        features['non_zero_peak1_value'] = peaks[0][0]
        features['non_zero_peak1_height'] = peaks[0][1]
        
        if len(peaks) > 1:
            features['non_zero_peak2_value'] = peaks[1][0]
            features['non_zero_peak2_height'] = peaks[1][1]
            features['non_zero_peak_distance'] = abs(features['non_zero_peak1_value'] - features['non_zero_peak2_value'])
        else:
            features['non_zero_peak2_value'] = features['non_zero_peak1_value']
            features['non_zero_peak2_height'] = 0
            features['non_zero_peak_distance'] = 0
    else:
        features['non_zero_peak1_value'] = features['mean']
        features['non_zero_peak1_height'] = 0
        features['non_zero_peak2_value'] = features['mean']
        features['non_zero_peak2_height'] = 0
        features['non_zero_peak_distance'] = 0
    
    # Add specialized features
    features['contrast_ratio'] = features['max'] / features['mean'] if features['mean'] > 0 else 0
    features['p99_mean_ratio'] = features['p99'] / features['mean'] if features['mean'] > 0 else 0
    features['p75_p25'] = features['p75'] - features['p25']  # Interquartile range
    features['p95_p5'] = np.percentile(volume_array, 95) - np.percentile(volume_array, 5)
    features['p99_p1'] = np.percentile(volume_array, 99) - np.percentile(volume_array, 1)
    features['entropy'] = shannon_entropy(volume_array)
    
    # Add engineered features that were used in model training
    features['range'] = features['max'] - features['min']
    features['mean_centered_min'] = features['min'] - features['mean']
    features['mean_centered_max'] = features['max'] - features['mean']
    features['iqr'] = features['p75'] - features['p25']
    features['iqr_to_std_ratio'] = features['iqr'] / (features['std'] + 1e-5)
    features['contrast_per_iqr'] = features['contrast_ratio'] / (features['iqr'] + 1e-5)
    features['entropy_per_range'] = features['entropy'] / (features['range'] + 1e-5)
    features['peak_value_to_height_ratio'] = features['non_zero_peak2_value'] / (features['non_zero_peak2_height'] + 1e-5)
    features['range_to_iqr'] = features['range'] / (features['iqr'] + 1e-5)
    features['entropy_iqr_interaction'] = features['entropy'] * features['iqr']
    features['skewness_squared'] = features['skewness'] ** 2
    features['kurtosis_log'] = np.log1p(features['kurtosis'] - np.min(features['kurtosis']))
    
    return features

def predictThreshold(volumeNode, modelPath):
    """Predict threshold for given volume node."""
    import pandas as pd
    
    # Load the model
    if not os.path.exists(modelPath):
        raise FileNotFoundError(f"Model file not found: {modelPath}")
    
    model_data = joblib.load(modelPath)
    model = model_data['model']
    feature_names = model_data.get('feature_names', [])
    
    # Extract features
    features = extractFeatures(volumeNode)
    
    # Convert to DataFrame with correct feature order
    feature_df = pd.DataFrame([features])
    
    # Ensure we have all expected features
    for feat in feature_names:
        if feat not in feature_df.columns:
            feature_df[feat] = 0  # Add missing features with default value
    
    # Reorder columns to match training data
    feature_df = feature_df[feature_names]
    
    # Predict threshold
    threshold = model.predict(feature_df)[0]
    return threshold

def createThresholdPredictorUI():
    """Create the threshold predictor UI."""
    # Create main widget
    widget = qt.QWidget()
    layout = qt.QVBoxLayout(widget)
    
    # Volume selector
    volumeSelector = slicer.qMRMLNodeComboBox()
    volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    volumeSelector.selectNodeUponCreation = True
    volumeSelector.addEnabled = False
    volumeSelector.removeEnabled = False
    volumeSelector.noneEnabled = False
    volumeSelector.showHidden = False
    volumeSelector.showChildNodeTypes = False
    volumeSelector.setMRMLScene(slicer.mrmlScene)
    volumeSelector.setToolTip("Select volume to analyze")
    layout.addWidget(volumeSelector)
    
    # Model path selector
    modelPathLineEdit = qt.QLineEdit()
    modelPathLineEdit.setPlaceholderText("Path to trained model (best_model.joblib)")
    layout.addWidget(modelPathLineEdit)
    
    browseButton = qt.QPushButton("Browse...")
    layout.addWidget(browseButton)
    
    # Predict button
    predictButton = qt.QPushButton("Predict Threshold")
    layout.addWidget(predictButton)
    
    # Results display
    resultsLabel = qt.QLabel("Predicted threshold will appear here")
    resultsLabel.setAlignment(qt.Qt.AlignCenter)
    layout.addWidget(resultsLabel)
    
    # Connect signals
    def onBrowseModel():
        """Open file dialog to select model."""
        modelPath = qt.QFileDialog.getOpenFileName(
            widget, "Select Trained Model", "", "Joblib files (*.joblib)"
        )
        if modelPath:
            modelPathLineEdit.text = modelPath
    
    def onPredict():
        """Predict threshold for selected volume."""
        modelPath = modelPathLineEdit.text
        if not modelPath or not os.path.exists(modelPath):
            slicer.util.errorDisplay("Please select a valid model file first")
            return
            
        volumeNode = volumeSelector.currentNode()
        if not volumeNode:
            slicer.util.errorDisplay("Please select a volume first")
            return
            
        try:
            threshold = predictThreshold(volumeNode, modelPath)
            resultsLabel.text = f"Predicted Threshold: {threshold:.4f}"
            slicer.util.infoDisplay(f"Predicted threshold: {threshold:.4f}")
        except Exception as e:
            slicer.util.errorDisplay(f"Prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    browseButton.clicked.connect(onBrowseModel)
    predictButton.clicked.connect(onPredict)
    
    return widget

# Main execution
if __name__ == "__main__":
    # Create and show the UI
    thresholdPredictorUI = createThresholdPredictorUI()
    slicer.util.mainWindow().statusBar().showMessage("Threshold Predictor loaded")
    
    # Create a dock widget to show the UI
    dockWidget = qt.QDockWidget("Threshold Predictor")
    dockWidget.setWidget(thresholdPredictorUI)
    slicer.util.mainWindow().addDockWidget(qt.Qt.RightDockWidgetArea, dockWidget)
    dockWidget.show()
#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Threshold_mask\threshold_model.py').read())