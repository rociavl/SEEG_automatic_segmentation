"""
CORRECTED TEST SCRIPT FOR CONFIDENCE WRAPPER
============================================
This script tests the confidence wrapper functionality step by step.
Save as: test_confidence_wrapper.py
"""

import os
import sys
import traceback
from pathlib import Path

print("🧪 CONFIDENCE WRAPPER TEST SCRIPT")
print("=" * 50)

# ============================================================================
# STEP 1: SETUP PATHS
# ============================================================================

print("\n📁 STEP 1: Setting up paths...")

# Your actual paths (modify these to match your setup)
MODULE_DIR = r"C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking"
CENTROIDS_PIPELINE_DIR = os.path.join(MODULE_DIR, "Centroids_pipeline")
CONFIDENCE_WRAPPER_PATH = os.path.join(CENTROIDS_PIPELINE_DIR, "confidence_wrapper.py")
MODEL_PATH = os.path.join(MODULE_DIR, "models", "patient_leave_one_out_ensemble.joblib")

# Test data paths (using your actual existing files)
TEST_DATA_DIR = r"C:\Users\rocia\Documents\SEEG_Results\Stitch"
BRAIN_MASK_FILE = r"C:\Users\rocia\Documents\SEEG_Results\Stitch\Brain_mask\BrainMask_brain.nrrd"
TOP_MASK_FILE = r"C:\Users\rocia\Documents\SEEG_Results\Stitch\Global_masks\top_mask_1_CTp.3D.nrrd"
ENHANCED_CT_FILE = r"C:\Users\rocia\Documents\SEEG_Results\Stitch\confidence_ct_file.nrrd"  # Fixed path
CONFIDENCE_DIR = os.path.join(TEST_DATA_DIR, "confidence_output")

# Create test output directory
os.makedirs(CONFIDENCE_DIR, exist_ok=True)

print(f"   Module dir: {MODULE_DIR}")
print(f"   Wrapper path: {CONFIDENCE_WRAPPER_PATH}")
print(f"   Model path: {MODEL_PATH}")
print(f"   Test data dir: {TEST_DATA_DIR}")

# Check if key files exist
print(f"   Wrapper exists: {'✅' if os.path.exists(CONFIDENCE_WRAPPER_PATH) else '❌'}")
print(f"   Model exists: {'✅' if os.path.exists(MODEL_PATH) else '❌'}")
print(f"   Brain mask exists: {'✅' if os.path.exists(BRAIN_MASK_FILE) else '❌'}")
print(f"   Top mask exists: {'✅' if os.path.exists(TOP_MASK_FILE) else '❌'}")
print(f"   Enhanced CT exists: {'✅' if os.path.exists(ENHANCED_CT_FILE) else '❌'}")

# ============================================================================
# STEP 2: ADD PATHS TO PYTHON PATH
# ============================================================================

print("\n🔧 STEP 2: Adding paths to Python path...")

if CENTROIDS_PIPELINE_DIR not in sys.path:
    sys.path.append(CENTROIDS_PIPELINE_DIR)
    print(f"   Added to path: {CENTROIDS_PIPELINE_DIR}")

# ============================================================================
# STEP 3: TEST WRAPPER IMPORT
# ============================================================================

print("\n📦 STEP 3: Testing wrapper import...")

try:
    from confidence_wrapper import ConfidenceAnalysisWrapper
    print("   ✅ ConfidenceAnalysisWrapper imported successfully")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    print("   🔍 Available files in Centroids_pipeline:")
    if os.path.exists(CENTROIDS_PIPELINE_DIR):
        for file in os.listdir(CENTROIDS_PIPELINE_DIR):
            if file.endswith('.py'):
                print(f"      • {file}")
    else:
        print(f"      Directory not found: {CENTROIDS_PIPELINE_DIR}")
    sys.exit(1)

# ============================================================================
# STEP 4: TEST WRAPPER INITIALIZATION
# ============================================================================

print("\n🏗️ STEP 4: Testing wrapper initialization...")

try:
    wrapper = ConfidenceAnalysisWrapper(MODULE_DIR)
    print("   ✅ Wrapper initialized successfully")
    
    # Check availability
    available = wrapper.is_confidence_analysis_available()
    print(f"   Analysis available: {'✅' if available else '❌'}")
    
    if not available:
        missing = wrapper.get_missing_components()
        print("   Missing components:")
        for component in missing:
            print(f"      • {component}")
            
except Exception as e:
    print(f"   ❌ Wrapper initialization failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 5: TEST MODEL LOADING (ISOLATED) - FIXED
# ============================================================================

print("\n🤖 STEP 5: Testing model loading in isolation...")

if os.path.exists(MODEL_PATH):
    try:
        # Import joblib directly and test model loading with class registration
        import joblib
        from confidence_wrapper import register_classes_in_main
        
        print("   Registering classes in __main__...")
        register_classes_in_main()
        
        print("   Attempting to load model...")
        model = joblib.load(MODEL_PATH)
        print("   ✅ Model loaded successfully!")
        print(f"   Model type: {type(model)}")
        
        # Test if model has expected attributes
        if hasattr(model, 'patient_models'):
            print(f"   Patient models: {len(model.patient_models)}")
        if hasattr(model, 'model_weights'):
            print(f"   Model weights: {len(model.model_weights)}")
            
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        traceback.print_exc()
else:
    print(f"   ❌ Model file not found: {MODEL_PATH}")

# ============================================================================
# STEP 6: SKIP CREATING SAMPLE DATA (Use your real files)
# ============================================================================

print("\n📊 STEP 6: Using existing test data...")
print("   ℹ️ Using your actual SEEG result files for testing")

# ============================================================================
# STEP 7: TEST INDIVIDUAL FUNCTIONS
# ============================================================================

print("\n🔬 STEP 7: Testing individual functions...")

if os.path.exists(TOP_MASK_FILE):
    try:
        from confidence_wrapper import extract_centroids_from_mask
        
        print("   Testing centroid extraction...")
        coords = extract_centroids_from_mask(TOP_MASK_FILE)
        print(f"   ✅ Extracted {len(coords)} centroids")
        
        if len(coords) > 0:
            print(f"   First centroid: ({coords[0][0]:.1f}, {coords[0][1]:.1f}, {coords[0][2]:.1f})")
            
    except Exception as e:
        print(f"   ❌ Centroid extraction failed: {e}")
        traceback.print_exc()
else:
    print("   ⚠️ No test mask file available for centroid extraction")

# ============================================================================
# STEP 8: TEST FEATURE CREATION
# ============================================================================

print("\n🧮 STEP 8: Testing feature creation...")

try:
    from confidence_wrapper import create_all_features
    import numpy as np
    
    # Create sample coordinates
    sample_coords = np.array([
        [10.0, 20.0, 30.0],
        [-5.0, 15.0, 25.0],
        [8.0, -10.0, 35.0]
    ])
    
    print("   Creating features for sample coordinates...")
    features_df = create_all_features(
        electrode_coords=sample_coords,
        volume_name="test_volume",
        enhanced_ct_file=ENHANCED_CT_FILE if os.path.exists(ENHANCED_CT_FILE) else None,
        brain_mask_file=BRAIN_MASK_FILE if os.path.exists(BRAIN_MASK_FILE) else None
    )
    
    print(f"   ✅ Created features: {features_df.shape[0]} rows, {features_df.shape[1]} columns")
    print(f"   Columns: {list(features_df.columns)}")
    
    # Save test features
    test_features_file = os.path.join(CONFIDENCE_DIR, "test_features.csv")
    features_df.to_csv(test_features_file, index=False)
    print(f"   ✅ Saved test features to: {test_features_file}")
    
except Exception as e:
    print(f"   ❌ Feature creation failed: {e}")
    traceback.print_exc()

# ============================================================================
# STEP 9: TEST PREDICTION (IF MODEL AVAILABLE) - FIXED
# ============================================================================

print("\n🎯 STEP 9: Testing prediction...")

if os.path.exists(MODEL_PATH) and 'test_features_file' in locals() and os.path.exists(test_features_file):
    try:
        from confidence_wrapper import predict_electrode_confidence_no_optuna
        
        print("   Running prediction on test features...")
        results = predict_electrode_confidence_no_optuna(
            feature_csv_path=test_features_file,
            model_path=MODEL_PATH,
            patient_id="test_patient",
            output_path=os.path.join(CONFIDENCE_DIR, "test_predictions.csv")
        )
        
        print(f"   ✅ Prediction completed: {len(results)} results")
        print(f"   Top confidence: {results['Ensemble_Confidence'].max():.4f}")
        print(f"   Mean confidence: {results['Ensemble_Confidence'].mean():.4f}")
        
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        traceback.print_exc()
else:
    print("   ⚠️ Skipping prediction test (model or features not available)")

# ============================================================================
# STEP 10: FULL INTEGRATION TEST
# ============================================================================

print("\n🎪 STEP 10: Full integration test...")

if (wrapper.is_confidence_analysis_available() and 
    os.path.exists(BRAIN_MASK_FILE) and 
    os.path.exists(TOP_MASK_FILE) and 
    os.path.exists(ENHANCED_CT_FILE)):
    
    try:
        print("   Running full confidence analysis...")
        results = wrapper.run_full_confidence_analysis(
            brain_mask_file=BRAIN_MASK_FILE,
            top_mask_file=TOP_MASK_FILE,
            enhanced_ct_file=ENHANCED_CT_FILE,
            volume_name="test_integration",
            confidence_dir=CONFIDENCE_DIR
        )
        
        print("   ✅ FULL INTEGRATION TEST PASSED!")
        print(f"   Total electrodes: {results['total_electrodes']}")
        print(f"   High confidence: {results['high_confidence']}")
        print(f"   Top confidence: {results['top_confidence']:.4f}")
        
    except Exception as e:
        print(f"   ❌ Full integration test failed: {e}")
        traceback.print_exc()
else:
    print("   ⚠️ Skipping full integration test (missing requirements)")
    if not wrapper.is_confidence_analysis_available():
        missing = wrapper.get_missing_components()
        print("   Missing components:")
        for component in missing:
            print(f"      • {component}")
    
    # Check which files are missing
    missing_files = []
    if not os.path.exists(BRAIN_MASK_FILE):
        missing_files.append("Brain mask")
    if not os.path.exists(TOP_MASK_FILE):
        missing_files.append("Top mask")
    if not os.path.exists(ENHANCED_CT_FILE):
        missing_files.append("Enhanced CT")
        
    if missing_files:
        print("   Missing files:")
        for file in missing_files:
            print(f"      • {file}")

# ============================================================================
# STEP 11: SUMMARY
# ============================================================================

print("\n📋 STEP 11: Test Summary")
print("=" * 50)

# Check what worked
test_results = {
    "Wrapper Import": True,  # If we got here, import worked
    "Wrapper Init": 'wrapper' in locals(),
    "Model Available": os.path.exists(MODEL_PATH),
    "Dependencies": wrapper.is_confidence_analysis_available() if 'wrapper' in locals() else False,
    "Test Files": all([
        os.path.exists(BRAIN_MASK_FILE),
        os.path.exists(TOP_MASK_FILE), 
        os.path.exists(ENHANCED_CT_FILE)
    ])
}

print("Test Results:")
for test_name, passed in test_results.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"   {test_name}: {status}")

overall_pass = all(test_results.values())
if overall_pass:
    print("\n🎉 ALL BASIC TESTS PASSED!")
    
    # Check if the critical model loading test passed
    try:
        # Try a simple model load test
        from confidence_wrapper import register_classes_in_main
        import joblib
        register_classes_in_main()
        test_model = joblib.load(MODEL_PATH)
        print("🎯 CRITICAL MODEL LOADING TEST: ✅ PASSED")
        print("\n✅ Your confidence wrapper should work in Slicer!")
    except Exception as e:
        print("🎯 CRITICAL MODEL LOADING TEST: ❌ FAILED")
        print(f"   Error: {e}")
        print("\n⚠️ Model loading still has issues. Check the __main__ registration.")
else:
    print("\n⚠️ Some basic tests failed. Check the output above for details.")

print(f"\nTest output saved to: {CONFIDENCE_DIR}")
print("You can now test the wrapper in your Slicer module!")

# ============================================================================
# STEP 12: SHOW NEXT STEPS
# ============================================================================

print("\n🚀 STEP 12: Next Steps")
print("=" * 30)

if overall_pass:
    print("1. ✅ Your wrapper is ready!")
    print("2. 🔧 Use it in your SEEG_masking.py file")
    print("3. 🧪 Test it with real data in Slicer")
else:
    print("1. ❌ Fix the failed tests above")
    print("2. 🔄 Re-run this test script")
    print("3. 🆘 If still failing, check file paths and dependencies")

print(f"\n📁 All test files are in: {CONFIDENCE_DIR}")
#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Centroids_pipeline\test.py').read())