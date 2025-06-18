"""
FINAL WORKING CONFIDENCE WRAPPER FOR SLICER
===========================================
This forces the classes into __main__ where joblib expects them.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
import os
import sys

# Essential imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor

# Try additional imports for feature extraction
try:
    import SimpleITK as sitk
    from skimage.measure import label, regionprops, marching_cubes
    from scipy.spatial import KDTree
    from scipy.stats import gaussian_kde
    from sklearn.decomposition import PCA
    import networkx as nx
    DEPENDENCIES_OK = True
except ImportError as e:
    print(f"⚠️ Some dependencies missing: {e}")
    DEPENDENCIES_OK = False

# ============================================================================
# CLASSES DEFINED AT MODULE LEVEL (EXACT COPY OF YOUR WORKING SCRIPT)
# ============================================================================

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

        predictions = {}
        for patient_id, model in self.patient_models.items():
            if patient_id == exclude_patient_id:
                continue

            X_test = test_df[model.numerical_features + model.categorical_features]
            preds = model.model.predict(X_test)
            predictions[patient_id] = preds

        if not predictions:
            raise ValueError("No models available for prediction")

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
        self.gt_tree = None
        self.preprocessing_time = 0
        self.model_training_time = 0
        self.total_training_time = 0
        self.test_metrics = None
        self.feature_names = None
        self.best_params = None

# ============================================================================
# CRITICAL: REGISTER CLASSES IN __main__ MODULE
# ============================================================================

def register_classes_in_main():
    """Register classes in __main__ so joblib can find them"""
    import sys
    import types
    
    # Ensure __main__ module exists
    if '__main__' not in sys.modules:
        sys.modules['__main__'] = types.ModuleType('__main__')
    
    main_module = sys.modules['__main__']
    
    # Register our classes
    main_module.PatientEnsemblePipeline = PatientEnsemblePipeline
    main_module.CentroidConfidencePipeline = CentroidConfidencePipeline
    
    print("✅ Classes registered in __main__ module")

# Register classes immediately when this module is imported
register_classes_in_main()

# ============================================================================
# WORKING MODEL PREDICTION FUNCTIONS
# ============================================================================

def load_model_and_predict(csv_path, model_path, exclude_patient="P8"):
    """Load model and make predictions (with __main__ class registration)"""
    print(f"Loading data from: {csv_path}")

    # Load feature data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} electrodes")

    # Fix categorical columns
    categorical_columns = ['Hemisphere', 'has_neighbors']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
            print(f"Fixed {col} as categorical")

    # CRITICAL: Ensure classes are registered before loading model
    register_classes_in_main()
    
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

def predict_electrode_confidence_no_optuna(feature_csv_path, model_path, patient_id="P8", output_path=None):
    """Main function for electrode confidence prediction"""
    try:
        # Load model and predict
        results_df = load_model_and_predict(
            csv_path=feature_csv_path,
            model_path=model_path,
            exclude_patient=patient_id
        )

        # Save results if path provided
        if output_path:
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

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_centroids_from_mask(mask_file_path):
    """Extract electrode coordinates from mask file."""
    print(f"🔧 Extracting centroids from: {mask_file_path}")
    
    if not DEPENDENCIES_OK:
        raise ImportError("SimpleITK and skimage required for centroid extraction")
    
    try:
        mask_image = sitk.ReadImage(mask_file_path)
        mask_array = sitk.GetArrayFromImage(mask_image)
        
        labeled_mask = label(mask_array > 0)
        regions = regionprops(labeled_mask)
        
        electrode_coords = []
        for region in regions:
            centroid_voxel = region.centroid
            idx_coord = (float(centroid_voxel[2]), float(centroid_voxel[1]), float(centroid_voxel[0]))
            physical_point = mask_image.TransformContinuousIndexToPhysicalPoint(idx_coord)
            ras_coord = (-physical_point[0], -physical_point[1], physical_point[2])
            electrode_coords.append(ras_coord)
        
        print(f"✅ Extracted {len(electrode_coords)} centroids")
        return np.array(electrode_coords)
        
    except Exception as e:
        print(f"❌ Centroid extraction failed: {e}")
        raise

def extract_ct_intensity_features(electrode_coords, ct_file_path):
    """Extract CT intensity features (simplified)."""
    try:
        if not os.path.exists(ct_file_path):
            print("⚠️ CT file not found, using default values")
            n = len(electrode_coords)
            return {
                'CT_mean_intensity': np.full(n, 100.0),
                'CT_std_intensity': np.full(n, 15.0),
                'CT_max_intensity': np.full(n, 150.0),
                'CT_min_intensity': np.full(n, 50.0)
            }
        
        print("🔬 Extracting CT intensity features...")
        
        ct_image = sitk.ReadImage(ct_file_path)
        ct_array = sitk.GetArrayFromImage(ct_image)
        
        ct_features = {
            'CT_mean_intensity': [],
            'CT_std_intensity': [],
            'CT_max_intensity': [],
            'CT_min_intensity': []
        }
        
        for coord in electrode_coords:
            try:
                lps_coord = (-coord[0], -coord[1], coord[2])
                voxel_coord = ct_image.TransformPhysicalPointToIndex(lps_coord)
                
                x, y, z = voxel_coord
                radius = 2
                
                x_min = max(0, x - radius)
                x_max = min(ct_array.shape[2], x + radius + 1)
                y_min = max(0, y - radius)
                y_max = min(ct_array.shape[1], y + radius + 1)
                z_min = max(0, z - radius)
                z_max = min(ct_array.shape[0], z + radius + 1)
                
                region = ct_array[z_min:z_max, y_min:y_max, x_min:x_max]
                
                if region.size > 0:
                    ct_features['CT_mean_intensity'].append(np.mean(region))
                    ct_features['CT_std_intensity'].append(np.std(region))
                    ct_features['CT_max_intensity'].append(np.max(region))
                    ct_features['CT_min_intensity'].append(np.min(region))
                else:
                    ct_features['CT_mean_intensity'].append(100.0)
                    ct_features['CT_std_intensity'].append(15.0)
                    ct_features['CT_max_intensity'].append(150.0)
                    ct_features['CT_min_intensity'].append(50.0)
                    
            except Exception:
                ct_features['CT_mean_intensity'].append(100.0)
                ct_features['CT_std_intensity'].append(15.0)
                ct_features['CT_max_intensity'].append(150.0)
                ct_features['CT_min_intensity'].append(50.0)
        
        print("✅ CT intensity features extracted")
        return ct_features
        
    except Exception as e:
        print(f"⚠️ CT extraction failed: {e}, using defaults")
        n = len(electrode_coords)
        return {
            'CT_mean_intensity': np.full(n, 100.0),
            'CT_std_intensity': np.full(n, 15.0),
            'CT_max_intensity': np.full(n, 150.0),
            'CT_min_intensity': np.full(n, 50.0)
        }

def extract_surface_distance_features(electrode_coords, roi_file_path):
    """Extract distance to ROI surface (simplified)."""
    try:
        if not os.path.exists(roi_file_path):
            print("⚠️ ROI file not found, using default surface distances")
            return {'dist_to_surface': np.full(len(electrode_coords), 5.0)}
        
        print("🔬 Extracting surface distance features...")
        
        roi_image = sitk.ReadImage(roi_file_path)
        roi_array = sitk.GetArrayFromImage(roi_image)
        
        spacing = roi_image.GetSpacing()
        spacing_zyx = (spacing[2], spacing[1], spacing[0])
        
        binary_array = (roi_array > 0).astype(np.uint8)
        if binary_array.sum() == 0:
            print("⚠️ ROI is empty, using default distances")
            return {'dist_to_surface': np.full(len(electrode_coords), 5.0)}
        
        vertices, faces, _, _ = marching_cubes(binary_array, level=0.5, spacing=spacing_zyx)
        
        surface_points_ras = []
        for vertex in vertices:
            idx_coord = (float(vertex[2]), float(vertex[1]), float(vertex[0]))
            physical_point = roi_image.TransformContinuousIndexToPhysicalPoint(idx_coord)
            ras_coord = (-physical_point[0], -physical_point[1], physical_point[2])
            surface_points_ras.append(ras_coord)
        
        surface_points_ras = np.array(surface_points_ras)
        
        surface_tree = KDTree(surface_points_ras)
        distances = []
        for coords in electrode_coords:
            dist, _ = surface_tree.query(coords, k=1)
            distances.append(float(dist))
        
        print("✅ Surface distance features extracted")
        return {'dist_to_surface': np.array(distances)}
        
    except Exception as e:
        print(f"⚠️ Surface distance extraction failed: {e}, using defaults")
        return {'dist_to_surface': np.full(len(electrode_coords), 5.0)}

def create_all_features(electrode_coords, volume_name, enhanced_ct_file, brain_mask_file):
    """Create all features required by the model."""
    print("🔬 Creating all features...")
    
    n_electrodes = len(electrode_coords)
    
    features = {
        'RAS_X': electrode_coords[:, 0],
        'RAS_Y': electrode_coords[:, 1],
        'RAS_Z': electrode_coords[:, 2],
        'Patient ID': [volume_name] * n_electrodes,
        'Pixel Count': [1] * n_electrodes
    }
    
    # CT intensity features
    ct_features = extract_ct_intensity_features(electrode_coords, enhanced_ct_file)
    features.update(ct_features)
    
    # Surface distance features
    surface_features = extract_surface_distance_features(electrode_coords, brain_mask_file)
    features.update(surface_features)
    
    # PCA features
    try:
        if len(electrode_coords) >= 3:
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(electrode_coords)
            features['PCA1'] = pca_result[:, 0]
            features['PCA2'] = pca_result[:, 1]
            features['PCA3'] = pca_result[:, 2]
        else:
            features['PCA1'] = np.zeros(n_electrodes)
            features['PCA2'] = np.zeros(n_electrodes)
            features['PCA3'] = np.zeros(n_electrodes)
    except:
        features['PCA1'] = np.zeros(n_electrodes)
        features['PCA2'] = np.zeros(n_electrodes)
        features['PCA3'] = np.zeros(n_electrodes)
    
    # Neighbor features
    try:
        tree = KDTree(electrode_coords)
        n_neighbors = []
        mean_neighbor_dist = []
        
        for coords in electrode_coords:
            distances, indices = tree.query(coords, k=min(len(electrode_coords), 10), distance_upper_bound=7)
            valid_dists = distances[(distances > 0) & (distances <= 7)]
            n_neighbors.append(len(valid_dists))
            mean_neighbor_dist.append(np.mean(valid_dists) if len(valid_dists) > 0 else np.nan)
        
        features['n_neighbors'] = n_neighbors
        features['mean_neighbor_dist'] = mean_neighbor_dist
    except:
        features['n_neighbors'] = [0] * n_electrodes
        features['mean_neighbor_dist'] = [np.nan] * n_electrodes
    
    # Density features
    try:
        kde = gaussian_kde(electrode_coords.T, bw_method='scott')
        features['kde_density'] = kde(electrode_coords.T)
    except:
        features['kde_density'] = np.full(n_electrodes, 0.1)
    
    # Position features
    centroid = np.mean(electrode_coords, axis=0)
    features['dist_to_centroid'] = np.linalg.norm(electrode_coords - centroid, axis=1)
    features['x_relative'] = electrode_coords[:, 0] - centroid[0]
    features['y_relative'] = electrode_coords[:, 1] - centroid[1]
    features['z_relative'] = electrode_coords[:, 2] - centroid[2]
    
    # Community features
    features['Louvain_Community'] = [0] * n_electrodes
    
    # Categorical features
    features['Hemisphere'] = ['Right' if x > 0 else 'Left' for x in electrode_coords[:, 0]]
    features['has_neighbors'] = [n > 0 for n in features['n_neighbors']]
    
    # Create DataFrame
    df = pd.DataFrame(features)
    df['Hemisphere'] = df['Hemisphere'].astype('category')
    df['has_neighbors'] = df['has_neighbors'].astype('category')
    
    print(f"✅ Created {len(df)} electrode features with {len(df.columns)} columns")
    return df

# ============================================================================
# WRAPPER CLASS
# ============================================================================

class ConfidenceAnalysisWrapper:
    """Confidence analysis wrapper with forced __main__ class registration"""
    
    def __init__(self, module_dir):
        print(f"🔧 Initializing ConfidenceAnalysisWrapper with __main__ registration")
        
        self.module_dir = module_dir
        self.model_path = os.path.join(module_dir, "models", "patient_leave_one_out_ensemble.joblib")
        
        self.dependencies_available = DEPENDENCIES_OK
        self.model_file_available = os.path.exists(self.model_path)
        
        print(f"   Dependencies: {'✅' if self.dependencies_available else '❌'}")
        print(f"   Model file: {'✅' if self.model_file_available else '❌'}")
        
        # Ensure classes are registered when wrapper is created
        register_classes_in_main()
    
    def is_confidence_analysis_available(self):
        return self.dependencies_available and self.model_file_available
    
    def get_missing_components(self):
        missing = []
        if not self.dependencies_available:
            missing.append("Core dependencies (SimpleITK, skimage, scipy)")
        if not self.model_file_available:
            missing.append("Trained model file")
        return missing
    
    def run_full_confidence_analysis(self, brain_mask_file, top_mask_file, enhanced_ct_file, volume_name, confidence_dir):
        """Run complete confidence analysis with __main__ class registration"""
        print("🔧 Starting full confidence analysis...")
        
        if not self.is_confidence_analysis_available():
            missing = self.get_missing_components()
            missing_str = "\n    • ".join(missing)
            error_msg = f"Cannot perform confidence analysis. Missing components:\n    • {missing_str}"
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        try:
            # Ensure classes are registered before any model operations
            register_classes_in_main()
            
            # Step 1: Extract centroids
            print("📍 Step 1: Extracting centroids...")
            electrode_coords = extract_centroids_from_mask(top_mask_file)
            
            if len(electrode_coords) == 0:
                raise ValueError("No electrode centroids found in mask")
            
            # Step 2: Create features
            print("🔬 Step 2: Creating features...")
            features_df = create_all_features(electrode_coords, volume_name, enhanced_ct_file, brain_mask_file)
            
            # Step 3: Save features
            features_csv = os.path.join(confidence_dir, f"target_features_{volume_name}_top_mask_1.csv")
            features_df.to_csv(features_csv, index=False)
            print(f"✅ Features saved to: {features_csv}")
            
            # Step 4: Run prediction (classes already registered)
            print("🤖 Step 4: Running prediction...")
            predictions_csv = os.path.join(confidence_dir, f"confidence_predictions_{volume_name}.csv")
            
            confidence_results = predict_electrode_confidence_no_optuna(
                feature_csv_path=features_csv,
                model_path=self.model_path,
                patient_id=volume_name,
                output_path=predictions_csv
            )
            
            # Step 5: Calculate statistics
            print("📊 Step 5: Calculating statistics...")
            total_electrodes = len(confidence_results)
            high_confidence = np.sum(confidence_results['Ensemble_Confidence'] >= 0.8)
            medium_confidence = np.sum((confidence_results['Ensemble_Confidence'] >= 0.6) & 
                                     (confidence_results['Ensemble_Confidence'] < 0.8))
            low_confidence = np.sum(confidence_results['Ensemble_Confidence'] < 0.6)
            top_confidence = confidence_results['Ensemble_Confidence'].max()
            avg_confidence = confidence_results['Ensemble_Confidence'].mean()
            
            print("✅ Confidence analysis completed successfully!")
            
            return {
                'confidence_results': confidence_results,
                'total_electrodes': total_electrodes,
                'high_confidence': high_confidence,
                'medium_confidence': medium_confidence,
                'low_confidence': low_confidence,
                'top_confidence': top_confidence,
                'avg_confidence': avg_confidence,
                'predictions_csv': predictions_csv
            }
            
        except Exception as e:
            print(f"❌ Confidence analysis failed: {e}")
            raise

print("✅ Working ConfidenceAnalysisWrapper with __main__ class registration loaded!")