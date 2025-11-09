# SEEG Automatic Segmentation: Algorithms and Implementation

![image](https://github.com/user-attachments/assets/be33d580-feb4-4caa-9a48-30ebd59ee7e1)
![image](https://github.com/user-attachments/assets/4e0f3fa7-2de5-4efc-b5d4-10d8878caf77)
<p align="center">
  <a href="https://www.youtube.com/watch?v=mSGtHaBInJM">
    <img src="https://img.youtube.com/vi/mSGtHaBInJM/0.jpg" alt="Watch the demo video" width="70%">
  </a>
  <br>
  <b>🎥 Click the image to watch the SEEG trajectory demo</b>
</p>

## Abstract

This repository contains the core algorithms, software architecture, and implementation details for automated SEEG (Stereoelectroencephalography) electrode localization and trajectory reconstruction. The system employs a novel multi-stage ensemble approach combining deep learning-based brain extraction, adaptive image enhancement, global voting consensus, and machine learning-based confidence estimation to achieve sub-millimeter localization accuracy (mean error: 0.33 mm).

**For end-users and clinical deployment**, please refer to [SlicerSEEG](https://github.com/rociavl/SlicerSEEG) - the production-ready 3D Slicer extension.

**For complete research documentation**, see the [full bachelor's thesis](docs/SEEG_medical_software_detection_Rocio_Avalos.pdf) included in this repository.

## Repository Purpose

This repository serves as the **research and development foundation** for the SEEG electrode localization system. It contains:

- Algorithm implementations and experimental code
- Software architecture and design patterns
- Pipeline development and optimization
- Research validation scripts and analysis
- Technical documentation for developers and researchers

The companion repository [SlicerSEEG](https://github.com/rociavl/SlicerSEEG) provides the packaged extension for clinical users, while this repository focuses on the underlying computational methods and technical implementation.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Algorithm Details](#algorithm-details)
3. [Machine Learning Models](#machine-learning-models)
4. [Repository Structure](#repository-structure)
5. [Experimental Results](#experimental-results)
6. [Technical Implementation](#technical-implementation)
7. [Development](#development)
8. [Research Context](#research-context)
9. [Citation](#citation)

---

## System Architecture

### Six-Stage Processing Pipeline

The system implements a modular pipeline architecture with distinct processing stages:

```
INPUT: MRI + CT volumes
    ↓
[1] Brain Extraction → [2] Multi-Modal Enhancement → [3] Ensemble Voting
    ↓
[4] Centroid Detection → [5] Confidence Scoring → [6] Trajectory Reconstruction
    ↓
OUTPUT: Electrode positions + trajectories + confidence scores
```

#### Stage 1: Brain Extraction

**Algorithm**: Otsu thresholding with morphological refinement

**Implementation**: `Brain_mask_methods/brain_mask_extractor_otsu.py`

**Processing steps**:
1. **Gaussian smoothing** (σ=2.0) - Noise reduction
2. **Otsu's thresholding** - Automatic threshold selection
3. **Morphological closing** (6 iterations, connectivity-1 structure) - Structure connection
4. **Binary hole filling** - Internal cavity completion

**Output**: Binary brain mask (uint8, 0/1 values)

**Key parameters**:
- `threshold_value`: 20 (fallback threshold)
- `sigma`: 2.0 (Gaussian smoothing)
- `closing_iterations`: 6

---

#### Stage 2: Multi-Modal Image Enhancement

**Implementation**: `Threshold_mask/ctp_enhancer.py`, `Threshold_mask/enhance_ctp.py`

**Purpose**: Maximize electrode contrast across varying CT acquisition parameters

The system applies **7 parallel enhancement pipelines**, generating ~38 enhanced volume variants:

##### Enhancement Approach 1: Original CTP Processing
- Gaussian filtering (σ=0.3)
- Gamma correction (γ=3)
- High-pass sharpening (strength=0.8)

##### Enhancement Approach 2: ROI with Gamma Mask
- ROI masking × gamma correction
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Clip limit: 2.0
  - Tile grid: (3, 3)
- Wavelet denoising (Daubechies-1) + Non-local means

##### Enhancement Approach 3: ROI Only Processing
- Wavelet-only denoising (db1)
- Conservative gamma correction (γ=0.8)

##### Enhancement Approach 4: ROI Plus Gamma After
Complex 10-step pipeline:
- Gaussian (σ=0.3) → Gamma (γ=0.8) → Top-hat morphology → Sharpening
- LOG transform (c=3) → Wavelet (db4) → Morphological erosion

##### Enhancement Approach 5: Wavelet on ROI
- Combined wavelet + non-local means denoising

##### Enhancement Approach 6: Original Idea
Multi-stage enhancement:
- Gaussian (σ=0.3) → Gamma (γ=2) → Sharpen → Wavelet
- Gaussian (σ=0.4) → Gamma (γ=1.4)
- Top-hat transform

##### Enhancement Approach 7: First Try
14-step complex chain:
- Gaussian → Top-hat → Subtraction → Gamma (γ=5)
- Sharpening → Opening → Closing → Erosion

**Enhancement techniques employed**:
- **Gaussian filtering**: σ ∈ {0.1, 0.2, 0.3, 0.4, 0.8}
- **Gamma correction**: γ ∈ {0.8, 1.2, 1.4, 2, 3, 5}
- **CLAHE**: clipLimit=2.0, tileGridSize=(3,3)
- **Wavelet denoising**: db1, db4 wavelets, soft thresholding
- **Non-local means**: patch_size=2-5, patch_distance=2-6
- **High-pass sharpening**: strength=0.4-0.8
- **LOG transform**: c ∈ {3, 5}
- **Morphological operations**: top-hat, opening, closing, erosion

**Rationale**: Different enhancement methods optimize for different CT protocols and image quality levels.

---

#### Stage 3: Adaptive Thresholding & Ensemble Voting

**Adaptive Thresholding** (`Threshold_mask/ctp_enhancer.py`)

**Algorithm**: Machine learning-based threshold prediction

**Feature extraction** (33 statistical features):
- Basic statistics: min, max, mean, median, std
- Percentiles: p25, p75, p95, p99, p99.5, p99.9, p99.99
- Distribution metrics: skewness, kurtosis
- Non-zero statistics: mean, median, std, count, ratio
- Histogram analysis: peak detection, peak distance, entropy
- Engineered features: IQR, contrast ratio, CV

**Model**: Random Forest ensemble (`random_forest_modelP1.joblib`)
- Input: 33 statistical features
- Output: Optimal threshold value
- Fallback: 99.97th percentile if prediction out of range

**Ensemble Voting** (`Masks_ensemble/masks_fusion_top.py`)

**Algorithm**: Global voting with greedy mask selection

**Process**:
1. **Mask generation**: Apply predicted threshold to all 38 enhanced volumes
2. **Global vote map**: `vote_map[voxel] = Σ(mask_i[voxel])` (range: 0-38)
3. **Best mask selection**: Greedy algorithm selecting top 4 masks
   - Score: `weighted_overlap / mask_sum`
   - Higher agreement with consensus → higher score
4. **Consensus formation**: Threshold at 35% agreement
   - `consensus_mask = (vote_map >= 0.35 × num_masks)`

**Output masks**:
- Top mask 1 & 2: Highest quality individual masks
- Consensus mask: 35% voting threshold
- Weighted fusion: Score-weighted combination

---

#### Stage 4: Centroid Detection & Feature Extraction

**Centroid Extraction**

**Algorithm**: Connected component analysis
- Implementation: `skimage.measure.label` + `regionprops`
- Centroid calculation: Center of mass per component
- Coordinate conversion: Voxel IJK → Physical LPS → RAS

**Feature Extraction** (`Centroids_pipeline/centroids_feature_extraction.py`)

**Total features**: 22 (20 numerical + 2 categorical)

##### 1. Spatial Coordinates (3 features)
- `RAS_X`, `RAS_Y`, `RAS_Z`: 3D position in RAS coordinate system

##### 2. CT Intensity Features (4 features)
- ROI extraction: radius=2 voxels around centroid
- `CT_mean_intensity`, `CT_std_intensity`
- `CT_max_intensity`, `CT_min_intensity`

##### 3. PCA Features (3 features)
- `PCA1`, `PCA2`, `PCA3`: First 3 principal components
- Implementation: `sklearn.decomposition.PCA(n_components=3)`

##### 4. Geometric Features (4 features)
- `dist_to_surface`: Distance to brain ROI surface
  - Marching cubes surface extraction
  - KDTree nearest neighbor search
- `dist_to_centroid`: Distance to global centroid
- `x_relative`, `y_relative`, `z_relative`: Relative position

##### 5. Neighborhood Features (3 features)
- `mean_neighbor_dist`: Average distance to neighbors
- `n_neighbors`: Count within 7mm threshold
- `kde_density`: Gaussian KDE (Scott's bandwidth)

##### 6. Community Detection (1 feature)
- `Louvain_Community`: Graph-based clustering
  - Algorithm: Louvain modularity optimization
  - Edge weight: `1/distance`

##### 7. Size & Categorical (3 features)
- `Pixel_Count`: Voxels in connected component
- `Hemisphere`: "Left" or "Right" (based on X coordinate)
- `has_neighbors`: Boolean flag

---

#### Stage 5: Confidence Scoring

**Implementation**: `confidence.py`, `Centroids_pipeline/model_application.py`

**Model Architecture**: Patient Leave-One-Out Ensemble

**File**: `patient_leave_one_out_ensemble.joblib` (9.1 MB)

**Structure**:
- **PatientEnsemblePipeline**: 8 patient-specific models
- **Individual models**: LightGBM regressors
- **Preprocessor**: ColumnTransformer
  - Numerical features: StandardScaler
  - Categorical features: OneHotEncoder

**Prediction algorithm** (`predict_leave_one_out`):
```python
def predict(test_data, exclude_patient):
    predictions = []
    weights = []

    for patient_model in ensemble:
        if patient_model.id != exclude_patient:
            pred = patient_model.predict(test_data)
            weight = patient_model.validation_score
            predictions.append(pred * weight)
            weights.append(weight)

    return sum(predictions) / sum(weights)
```

**Leave-one-out strategy**: When analyzing patient P8, uses models trained on P1-P7.

**Output**:
- `Ensemble_Confidence`: Continuous score (0-1)
- `Binary_Prediction`: Binary classification (≥0.5 = electrode)
- `Confidence_Rank`: Ranking by confidence

**Confidence thresholds**:
- High (≥0.8): Direct clinical use
- Medium (0.6-0.8): Review recommended
- Low (<0.6): Manual validation required

---

#### Stage 6: Trajectory Reconstruction

**Implementation**: `Electrode_path/test_slicer.py`

**Algorithm**: Hybrid clustering with geometric validation

**Function**: `integrated_trajectory_analysis`

##### Substage 6.1: DBSCAN Clustering
- **Algorithm**: Density-Based Spatial Clustering
- **Parameters**:
  - `eps`: 8mm (maximum neighbor distance)
  - `min_samples`: 3 (minimum neighbors for core point)
- **Output**: Initial cluster labels

##### Substage 6.2: Graph Construction
- **Nodes**: Electrode contact positions
- **Edges**: Contacts within `max_neighbor_distance` (7mm)
- **Edge weights**: `1 / (distance + ε)`
- **Implementation**: NetworkX Graph

##### Substage 6.3: Louvain Community Detection
- **Algorithm**: Modularity optimization
- **Resolution**: 1.0
- **Output**: Refined community assignments
- **Metric**: Modularity score

##### Substage 6.4: PCA-Based Trajectory Fitting
For each cluster:
- **Principal Component Analysis**: 3 components
- **Primary axis**: `pca.components_[0]`
- **Linearity metric**: `explained_variance_ratio_[0]`
- **Trajectory metrics**:
  - Direction vector
  - Center point (mean of coordinates)
  - Total length (sum of inter-contact distances)
  - Spacing regularity: `std(distances) / mean(distances)`

##### Substage 6.5: Spline Interpolation
- **Algorithm**: B-spline interpolation
- **Implementation**: `scipy.interpolate.splprep`
- **Points**: 50 interpolated points per trajectory
- **Smoothing**: s=0 (exact fit)

##### Substage 6.6: Spacing Validation
**Expected spacing**: 3.0-5.0 mm (DIXI Medical electrodes)

**Validation metrics**:
- `min_spacing`, `max_spacing`, `mean_spacing`, `std_spacing`
- `cv_spacing`: Coefficient of variation
- `valid_percentage`: % spacings within expected range
- `is_valid`: True if ≥75% spacings are valid
- `too_close_indices`: Contacts <3mm apart
- `too_far_indices`: Contacts >5mm apart

**Adaptive optimization** (`adaptive_clustering_parameters`):
- Iteratively adjusts `eps` and `min_samples`
- Expected contact counts: [5, 8, 10, 12, 15, 18]
- Scoring: `0.7 × valid_percentage + 0.3 × clustered_percentage`
- Maximum iterations: 10

**Quality metrics**:
1. **Linearity**: PCA variance ratio
2. **Spacing regularity**: CV of inter-contact distances
3. **Modularity**: Community detection score
4. **Cluster purity**: DBSCAN-Louvain agreement
5. **Spacing validity**: % correct spacing

---

## Machine Learning Models

### Model 1: Threshold Predictor

**File**: `Models/random_forest_modelP1.joblib` (1.1 MB)

**Type**: Random Forest Regressor

**Purpose**: Predict optimal threshold for electrode segmentation

**Input**: 33 statistical features from enhanced CT volumes

**Output**: Optimal threshold value (HU units)

**Training**: Optimized on development cohort to minimize false positives while maintaining sensitivity

---

### Model 2: Confidence Ensemble

**File**: `Models/patient_leave_one_out_ensemble.joblib` (9.1 MB)

**Type**: LightGBM Ensemble (8 patient-specific models)

**Purpose**: Electrode contact authentication and confidence scoring

**Input**: 22 features (spatial, intensity, geometric, neighborhood)

**Output**: Confidence score (0-1, continuous)

**Architecture**:
```
PatientEnsemblePipeline
├── Patient 1 Model: CentroidConfidencePipeline
│   ├── Preprocessor: ColumnTransformer
│   │   ├── StandardScaler (numerical features)
│   │   └── OneHotEncoder (categorical features)
│   └── Regressor: LGBMRegressor
├── Patient 2 Model: ...
├── ...
└── Patient 8 Model: ...
```

**Training strategy**: Leave-One-Patient-Out Cross-Validation (LOPOCV)
- 8 patients in development cohort
- Each model excludes one patient from training
- Prediction uses weighted ensemble of 7 models

**Performance**: 98.8% accuracy at 2mm clinical threshold

---

## Repository Structure

```
SEEG_automatic_segmentation/
│
├── SEEG_masking.py                          # Main pipeline orchestration
├── confidence.py                             # Confidence threshold UI
├── resampler_module_slicer.py               # Mask resampling utilities
├── overlay.py                                # Visualization overlays
│
├── Brain_mask_methods/                       # Brain extraction algorithms
│   └── brain_mask_extractor_otsu.py         # Otsu + morphological processing
│
├── Threshold_mask/                           # Image enhancement & thresholding
│   ├── ctp_enhancer.py                      # 7 enhancement pipelines
│   ├── enhance_ctp.py                       # Enhancement utilities
│   └── ...                                  # Additional enhancement methods
│
├── Masks_ensemble/                           # Ensemble voting algorithms
│   └── masks_fusion_top.py                  # Global voting & mask selection
│
├── Centroids_pipeline/                       # Contact detection & features
│   ├── centroids_feature_extraction.py      # 22-feature extraction
│   ├── model_application.py                 # LightGBM ensemble application
│   └── Features/
│       └── ct_features.py                   # CT intensity analysis
│
├── Electrode_path/                           # Trajectory reconstruction
│   └── test_slicer.py                       # DBSCAN + Louvain + PCA
│
├── Outermost_centroids_coordinates/          # Surface distance calculations
│   └── outermost_centroids_vol_slicer.py    # Marching cubes + KDTree
│
├── Models/                                   # Pre-trained ML models
│   ├── random_forest_modelP1.joblib         # Threshold predictor (1.1 MB)
│   └── patient_leave_one_out_ensemble.joblib # Confidence ensemble (9.1 MB)
│
├── Resources/                                # UI resources
│   └── Icons/                               # Extension icons
│
└── docs/                                    # Technical documentation
    ├── SEEG_medical_software_detection_Rocio_Avalos.pdf  # Full bachelor's thesis
    ├── monai_model_slide.html               # Brain extraction methodology
    ├── state_of_art_comparison.html         # Benchmarking analysis
    └── trajectory_consensus_slide.html      # Trajectory methods
```

---

## Experimental Results

### Validation Methodology

**Dataset**: 8-patient development cohort from Hospital del Mar (Barcelona)

**Validation strategy**:
- Leave-One-Patient-Out Cross-Validation (LOPOCV)
- External validation on independent test cases
- Real-world clinical deployment

**Ground truth**: Manual annotations by experienced neurophysiologists

**Evaluation metric**: Euclidean distance with 2mm clinical tolerance threshold

**Electrode type**: DIXI Medical SEEG electrodes (standard clinical configuration)

---

### Quantitative Performance

| Metric | Value | Clinical Threshold |
|--------|-------|-------------------|
| Localization accuracy (≤2mm) | 98.8% | ≥95% |
| Mean Euclidean distance | 0.33 mm | <1mm |
| Sensitivity (contact detection) | 100% | >98% |
| False positive rate (w/ confidence filter) | <5% | <10% |
| Processing time per case | ~30 min | <1 hour |
| Manual correction time | ~30 min | <2 hours |
| Total time (automated + manual) | ~60 min | <3 hours |

**Baseline comparison**: Manual localization requires 4+ hours per case

**Efficiency gain**: >95% reduction in specialist workload

---

### Computational Performance

**Hardware requirements**:
- CPU: Standard clinical workstation (no GPU required for inference)
- RAM: ~4GB peak usage
- Storage: ~500MB for models and dependencies

**Scalability**:
- Time complexity: Linear with number of contacts
- Typical case: 12-16 electrodes, 150-200 contacts
- Throughput: 1 patient case per 30 minutes

**Software environment**:
- 3D Slicer: 5.0+
- Python: 3.9+
- OS: Windows, Linux, macOS

---

### Ablation Studies

**Impact of ensemble size**:
- Single best mask: 94.2% accuracy
- Top 4 masks: 96.8% accuracy
- Full ensemble (38 masks): 98.8% accuracy

**Confidence threshold analysis**:
- Threshold ≥0.8: 99.4% precision, 89% recall
- Threshold ≥0.6: 98.8% precision, 97% recall
- Threshold ≥0.4: 96.2% precision, 99% recall

**Enhancement method comparison**:
- Single enhancement: 92-95% accuracy (varies by approach)
- All 7 approaches: 98.8% accuracy
- Best single approach: Approach 4 (ROI Plus Gamma After) - 95.1%

---

## Technical Implementation

### Libraries & Dependencies

**Core scientific computing**:
- NumPy: Array operations and numerical computing
- SciPy: Optimization, signal processing, spatial algorithms
- scikit-learn: Machine learning utilities, preprocessing
- scikit-image: Image processing, morphology, filters

**Image processing**:
- SimpleITK: Medical image I/O and transformations
- OpenCV (cv2): CLAHE, morphological operations
- PyWavelets: Wavelet denoising

**Machine learning**:
- LightGBM: Gradient boosting classifier/regressor
- joblib: Model serialization

**Graph analysis**:
- NetworkX: Graph construction, Louvain clustering

**Visualization**:
- Matplotlib: Static plotting
- VTK: 3D visualization
- Qt: User interface

**3D Slicer integration**:
- slicer: Slicer Python API
- vtk: Visualization Toolkit
- qt: Qt bindings

---

### Coordinate Systems

The system handles multiple coordinate reference frames:

**RAS (Right-Anterior-Superior)**:
- Slicer's standard coordinate system
- X: Right (+) / Left (-)
- Y: Anterior (+) / Posterior (-)
- Z: Superior (+) / Inferior (-)

**LPS (Left-Posterior-Superior)**:
- DICOM standard
- X: Left (+) / Right (-)
- Y: Posterior (+) / Anterior (-)
- Z: Superior (+) / Inferior (-)

**IJK (Image Indices)**:
- Voxel array indices
- I, J, K: Column, row, slice

**Transformations**:
- IJK ↔ RAS: Via IJK-to-RAS matrix (4×4 affine)
- LPS ↔ RAS: Via SimpleITK direction matrices
- All conversions preserve geometric accuracy

---

### Data Flow & File Formats

**Input formats**:
- CT/MRI: NRRD, NIfTI, DICOM
- Brain mask: NRRD, NIfTI

**Intermediate outputs** (saved to `~/Documents/SEEG_Results/[folder_name]/`):
- Brain masks: `brain_mask_*.nrrd`
- Enhanced volumes (38 files): `DESCARGAR_*_volume_*.nrrd`
- Binary masks (38 files): `DESCARGAR_*_mask_*.nrrd`
- Vote maps: `global_vote_map.nrrd`
- Consensus masks: `consensus_mask_*.nrrd`

**Analysis outputs**:
- Centroid features: `*_features.csv` (22 columns)
- Confidence scores: `*_ensemble_predictions.csv`
- Trajectory analysis: `trajectory_report_*.html`

**Visualization formats**:
- 3D models: VTK polydata
- Plots: PNG, HTML (Plotly interactive)

---

### Performance Optimizations

**Computational efficiency**:
- Slice-by-slice processing for memory-intensive 3D operations
- KDTree spatial indexing for nearest neighbor queries (O(log n))
- Parallel enhancement pipelines (independent processing)
- Cached vote maps to avoid recomputation

**Memory management**:
- Lazy loading of large volumes
- Incremental processing for ensemble voting
- Garbage collection after major processing stages

**Accuracy-performance trade-offs**:
- Adaptive parameter optimization (iterative refinement)
- Early stopping for confidence prediction
- Sparse matrix representations for vote maps

---

## Development

### Installation for Developers

```bash
# Clone repository
git clone https://github.com/rociavl/SEEG_automatic_segmentation.git
cd SEEG_automatic_segmentation

# Install dependencies
pip install -r requirements.txt

# Install in 3D Slicer (Developer mode)
# 1. Open 3D Slicer
# 2. Extension Manager → Developer Tools
# 3. Add module path: path/to/SEEG_masking
# 4. Restart Slicer
```

---

### Key Development Files

**Main pipeline**: `SEEG_masking.py`
- Class: `SEEGMaskingWidget`
- Entry point: `onApplyButton()` (line 408)
- Processing logic: Lines 408-494

**Brain extraction**: `Brain_mask_methods/brain_mask_extractor_otsu.py`
- Class: `BrainMaskExtractor`
- Key method: `extract_mask()`

**Enhancement**: `Threshold_mask/ctp_enhancer.py`
- Class: `CTPEnhancer`
- Key methods: `enhance_all_approaches()`, `predict_threshold_for_volume()`

**Ensemble voting**: `Masks_ensemble/masks_fusion_top.py`
- Class: `EnhancedMaskSelector`
- Key method: `process_masks_in_folder()`

**Feature extraction**: `Centroids_pipeline/centroids_feature_extraction.py`
- Function: `extract_all_target_features()`

**Confidence scoring**: `Centroids_pipeline/model_application.py`
- Function: `predict_leave_one_out()`

**Trajectory analysis**: `Electrode_path/test_slicer.py`
- Function: `integrated_trajectory_analysis()`

---

### Extending the System

**Adding new enhancement methods**:
1. Add method to `CTPEnhancer` class
2. Register in `enhance_all_approaches()`
3. Save output with naming convention: `DESCARGAR_{method}_volume_*.nrrd`

**Modifying ML models**:
1. Train new model with same 22-feature interface
2. Save as joblib: `joblib.dump(model, 'new_model.joblib')`
3. Update model path in `model_application.py`

**Custom trajectory algorithms**:
1. Implement in `Electrode_path/`
2. Maintain input/output interface (DataFrame with RAS coordinates)
3. Integrate in `integrated_trajectory_analysis()`

---

### Testing & Validation

**Unit tests**: (to be implemented)
- Brain extraction: Test Otsu threshold vs. manual masks
- Enhancement: Verify contrast improvement metrics
- Ensemble: Validate vote map statistics
- Confidence: Test prediction ranges [0, 1]
- Trajectory: Verify spacing validation logic

**Integration tests**:
- End-to-end pipeline on test patient
- Coordinate transformation accuracy
- Model loading and prediction

**Validation datasets**:
- Development cohort: 8 patients (Hospital del Mar)
- External validation: Independent test cases
- Ground truth: Manual annotations by neurophysiologists

---

## Research Context

### Academic Foundation

This work originates from a comprehensive bachelor's thesis:

**Title**: "Medical Software Module in 3D Slicer for Automatic Segmentation and Trajectory Reconstruction of SEEG Electrodes Using AI and Data Science"

**Author**: Rocío Ávalos Morillas
**Institution**: Universitat Politècnica de Catalunya (UPC)
**Collaboration**: Hospital del Mar (Barcelona), Center for Brain and Cognition (UPF)
**Year**: 2025

📄 **[Read the full thesis](docs/SEEG_medical_software_detection_Rocio_Avalos.pdf)** - Complete documentation of methodology, algorithms, validation, and clinical deployment.

---

### Clinical Context

**SEEG (Stereoelectroencephalography)**:
- Minimally invasive technique for epilepsy surgery planning
- 12-16 depth electrodes with 8-18 contacts each (150-200 contacts total)
- Implanted stereotactically to record epileptic activity
- Localization critical for epileptogenic zone mapping

**Clinical challenge**:
- Manual localization: 4+ hours per patient
- Requires expert neurophysiologist/neurosurgeon
- Prone to human error and inter-rater variability
- Bottleneck in epilepsy surgery workflow

**Clinical deployment**:
- Hospital del Mar Epilepsy Unit (Barcelona)
- 8-patient validation cohort
- Real-world clinical usage since 2024

---

### Related Work & State-of-the-Art

**Electrode localization methods**:
1. **Manual localization**: Gold standard, 4+ hours
2. **Semi-automated (GARDEL, iELVis)**: 2-3 hours, requires manual initialization
3. **Fully automated (ours)**: 30 minutes, no manual input

**Key innovations vs. existing methods**:
- **Multi-mask ensemble**: Robust to varying image quality (vs. single threshold)
- **Leave-one-out confidence**: Patient-specific validation (vs. pooled training)
- **Hybrid trajectory clustering**: DBSCAN + Louvain (vs. DBSCAN alone)
- **Conservative confidence scoring**: Clinical decision support (vs. binary classification)

**Benchmark comparison** (see `docs/state_of_art_comparison.html`):
- GARDEL: 96.2% accuracy, 2-3 hours
- iELVis: 94.8% accuracy, 2-3 hours
- Ours: 98.8% accuracy, 0.5 hours

---

### Future Research Directions

**Algorithm improvements**:
- Deep learning-based electrode segmentation (U-Net, nnU-Net)
- Transfer learning for multi-institutional generalization
- Attention mechanisms for enhanced feature extraction
- Uncertainty quantification (Bayesian neural networks)

**Clinical extensions**:
- Multi-electrode manufacturer support (Medtronic, Abbott, Ad-Tech)
- Subdural grid electrode localization
- Integration with electrical source imaging
- Automated epileptogenic zone prediction

**Software engineering**:
- GPU acceleration for deep learning inference
- Cloud-based processing pipeline
- PACS/DICOM integration
- Real-time intraoperative guidance

**Validation studies**:
- Multi-center clinical trial
- Larger patient cohorts (100+ patients)
- Prospective randomized controlled trial
- Long-term clinical outcome analysis

---

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{avalos2025seeg,
  title={Medical Software Module in 3D Slicer for Automatic Segmentation and Trajectory Reconstruction of SEEG Electrodes Using AI and Data Science},
  author={Ávalos Morillas, Rocío},
  year={2025},
  school={Universitat Politècnica de Catalunya},
  type={Bachelor's Thesis},
  note={Biomedical Engineering},
  url={https://github.com/rociavl/SEEG_automatic_segmentation},
  pdf={https://github.com/rociavl/SEEG_automatic_segmentation/blob/main/docs/SEEG_medical_software_detection_Rocio_Avalos.pdf}
}
```

**Full thesis**: Available in [`docs/SEEG_medical_software_detection_Rocio_Avalos.pdf`](docs/SEEG_medical_software_detection_Rocio_Avalos.pdf)

**Related repository** (clinical extension):
```bibtex
@software{avalos2025slicerseeg,
  title={SlicerSEEG: 3D Slicer Extension for SEEG Electrode Localization},
  author={Ávalos Morillas, Rocío},
  year={2025},
  url={https://github.com/rociavl/SlicerSEEG}
}
```

---

## Contributing

We welcome contributions from the medical imaging and epilepsy research communities!

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/algorithm-improvement`)
3. Implement changes with tests
4. Document new algorithms/methods
5. Submit pull request with detailed description

### Areas for Contribution

**Algorithms**:
- Enhanced segmentation methods
- Alternative ensemble strategies
- Novel trajectory reconstruction algorithms
- Improved confidence calibration

**Validation**:
- Testing on additional patient cohorts
- Cross-institutional validation
- Comparison with alternative methods

**Documentation**:
- Algorithm tutorials and notebooks
- API documentation
- Clinical workflow guides
- Code examples

**Software Engineering**:
- Performance optimization
- Unit test coverage
- Continuous integration
- Code refactoring

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Hospital del Mar Epilepsy Unit**: Clinical collaboration, patient data, validation
- **Center for Brain and Cognition (UPF)**: Research environment and scientific guidance
- **3D Slicer Community**: Platform, development tools, and technical support
- **Open Source Libraries**: NumPy, scikit-learn, PyTorch, MONAI, LightGBM, NetworkX

---

## Contact

**Rocío Ávalos Morillas**
*Biomedical Engineer*
*Universitat Politècnica de Catalunya (UPC)*

- 📧 Email: rocio.avalos029@gmail.com
- 🔗 LinkedIn: [Rocío Ávalos Morillas](https://www.linkedin.com/in/roc%C3%ADo-%C3%A1valos-morillas-04a5372b1/)
- 🐙 GitHub: [@rociavl](https://github.com/rociavl)

---

## Medical Device Notice

⚠️ **Research and Educational Use Only**

This software is intended for research and educational purposes. Clinical use requires:
- Appropriate validation in your institution
- Regulatory approval (FDA, CE Mark, etc.) in your jurisdiction
- Integration with validated clinical workflows
- Oversight by qualified medical professionals

The software is provided "as is" without warranty. Clinical decisions must be made by qualified healthcare professionals with appropriate training and oversight.

---

**Repository**: [SEEG_automatic_segmentation](https://github.com/rociavl/SEEG_automatic_segmentation)
**Production Extension**: [SlicerSEEG](https://github.com/rociavl/SlicerSEEG)
**Full Thesis**: [SEEG Medical Software Detection](docs/SEEG_medical_software_detection_Rocio_Avalos.pdf)
**Documentation**: See `docs/` directory for thesis, technical slides, and analysis
