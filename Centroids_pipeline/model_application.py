"""
CLEAN MODEL PREDICTION FOR SLICER
lightgbm required
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Essential imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor
import os

# Minimal class definitions for joblib loading
class PatientEnsemblePipeline:
    def __init__(self, numerical_features=None, categorical_features=None, test_size=0.2, random_state=42):
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.model_features = self.numerical_features + self.categorical_features
        self.patient_models = {}
        self.model_weights = {}
        self.test_size = test_size
        self.random_state = random_state

    def predict_leave_one_out(self, test_df, exclude_patient_id=None):
        """Make predictions using leave-one-out weighted ensemble"""
        if not self.patient_models:
            raise ValueError("No models trained.")

        # Get predictions from all relevant models
        predictions = {}
        for patient_id, model in self.patient_models.items():
            if patient_id == exclude_patient_id:
                continue  # Skip the left-out model

            # Get features for this model
            X_test = test_df[model.numerical_features + model.categorical_features]
            preds = model.model.predict(X_test)
            predictions[patient_id] = preds

        if not predictions:
            raise ValueError("No models available for prediction")

        # Calculate weighted average predictions
        weighted_preds = np.zeros(len(test_df))
        total_weight = 0

        for patient_id, preds in predictions.items():
            weight = self.model_weights.get(patient_id, 1.0)
            weighted_preds += weight * preds
            total_weight += weight

        if total_weight > 0:
            weighted_preds /= total_weight

        return weighted_preds


class CentroidConfidencePipeline:
    """Minimal class definition for joblib loading"""
    def __init__(self, numerical_features=None, categorical_features=None, test_size=0.2, random_state=42):
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = None
        self.model = None
        # Initialize other attributes that might be referenced
        self.gt_tree = None
        self.preprocessing_time = 0
        self.model_training_time = 0
        self.total_training_time = 0
        self.test_metrics = None
        self.feature_names = None
        self.best_params = None


def load_model_and_predict(csv_path, model_path, exclude_patient="P8"):
    """
    Simple function to load model and make predictions.

    Parameters:
    -----------
    csv_path : str
        Path to CSV with extracted features
    model_path : str
        Path to trained .joblib model
    exclude_patient : str
        Patient to exclude from ensemble

    Returns:
    --------
    pd.DataFrame with predictions
    """

    print(f"Loading data from: {csv_path}")

    # Load feature data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} electrodes")

    # Fix categorical columns (CRITICAL for model compatibility)
    categorical_columns = ['Hemisphere', 'has_neighbors']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
            print(f"Fixed {col} as categorical")

    # Load trained model
    print(f"Loading model from: {model_path}")
    ensemble = joblib.load(model_path)
    print(f"Model loaded successfully")

    # Make predictions
    print(f"Making predictions (excluding {exclude_patient})...")
    confidence_scores = ensemble.predict_leave_one_out(df, exclude_patient_id=exclude_patient)

    # Add predictions to dataframe
    df['Ensemble_Confidence'] = confidence_scores
    df['Binary_Prediction'] = (confidence_scores >= 0.5).astype(int)

    # Sort by confidence (highest first)
    df = df.sort_values('Ensemble_Confidence', ascending=False).reset_index(drop=True)
    df['Confidence_Rank'] = range(1, len(df) + 1)

    # Print summary
    print(f"Predictions complete!")
    print(f"  Top confidence score: {confidence_scores.max():.4f}")
    print(f"  Mean confidence score: {confidence_scores.mean():.4f}")
    print(f"  High confidence predictions: {np.sum(confidence_scores >= 0.5)}/{len(confidence_scores)}")

    return df


def predict_electrode_confidence_no_optuna(
    feature_csv_path,
    model_path,
    patient_id="P8",
    output_path=None
):
    """
    Main function for electrode confidence prediction in Slicer.

    Parameters:
    -----------
    feature_csv_path : str
        Path to CSV file with extracted electrode features
    model_path : str
        Path to trained ensemble model (.joblib)
    patient_id : str
        Patient ID for leave-one-out prediction
    output_path : str, optional
        Path to save results CSV

    Returns:
    --------
    pd.DataFrame
        DataFrame with electrode predictions
    """

    try:
        # Load model and predict
        results_df = load_model_and_predict(
            csv_path=feature_csv_path,
            model_path=model_path,
            exclude_patient=patient_id
        )

        # Save results if path provided
        if output_path:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")

        # Show top predictions
        print(f"\nTop 5 predictions:")
        top_5 = results_df[['Confidence_Rank', 'RAS_X', 'RAS_Y', 'RAS_Z', 'Ensemble_Confidence']].head(5)
        for _, row in top_5.iterrows():
            print(f"  #{int(row['Confidence_Rank'])}: "
                  f"({row['RAS_X']:.1f}, {row['RAS_Y']:.1f}, {row['RAS_Z']:.1f}) "
                  f"confidence: {row['Ensemble_Confidence']:.4f}")

        return results_df

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

# Your exact paths
csv_path = r"C:\Users\rocia\Downloads\TFG\Cohort\Extension\P1_Feature_Extraction\P1_mask_0\target_features_P1_mask_0.csv"
model_path = r"C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Models\patient_leave_one_out_ensemble.joblib"

# Run prediction with your paths
results_df = predict_electrode_confidence_no_optuna(
    feature_csv_path=csv_path,
    model_path=model_path,
    patient_id="P1",
    output_path=r"C:\Users\rocia\Downloads\TFG\Cohort\Extension\P1_trial_model\P1_predictions.csv"
)

print(f"✅ Prediction complete!")
print(f"📊 Results shape: {results_df.shape}")
print(f"🎯 Top confidence: {results_df['Ensemble_Confidence'].max():.4f}")

#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Centroids_pipeline\model_application.py').read())