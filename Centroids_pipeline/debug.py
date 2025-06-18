"""
MULTIPLE IMPORT METHODS TEST
===========================
Try different ways to import the ConfidenceAnalysisWrapper
"""

import os
import sys

def test_all_import_methods():
    """Test multiple ways to import the ConfidenceAnalysisWrapper."""
    
    print("🔧 TESTING MULTIPLE IMPORT METHODS")
    print("=" * 50)
    
    module_dir = r"C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking"
    wrapper_path = os.path.join(module_dir, "Centroids_pipeline", "confidence_wrapper.py")
    
    # Check file exists first
    if not os.path.exists(wrapper_path):
        print(f"❌ Wrapper file not found: {wrapper_path}")
        return None
    else:
        print(f"✅ Wrapper file found: {wrapper_path}")
    
    # METHOD 1: Your suggested Slicer module import
    print("\n1️⃣ Method 1: Slicer module import")
    print("   from SEEG_masking.Centroids_pipeline.confidence_wrapper import ConfidenceAnalysisWrapper")
    try:
        from SEEG_masking.Centroids_pipeline.confidence_wrapper import ConfidenceAnalysisWrapper
        wrapper1 = ConfidenceAnalysisWrapper(module_dir)
        print("✅ Method 1 SUCCESS!")
        return ("Method 1", ConfidenceAnalysisWrapper)
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")

    # METHOD 4: Import spec method
    print("\n4️⃣ Method 4: Import spec method")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("confidence_wrapper", wrapper_path)
        confidence_wrapper_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(confidence_wrapper_module)
        
        if hasattr(confidence_wrapper_module, 'ConfidenceAnalysisWrapper'):
            ConfidenceAnalysisWrapper = confidence_wrapper_module.ConfidenceAnalysisWrapper
            wrapper4 = ConfidenceAnalysisWrapper(module_dir)
            print("✅ Method 4 SUCCESS!")
            return ("Method 4", ConfidenceAnalysisWrapper)
        else:
            print("❌ Method 4: Module loaded but ConfidenceAnalysisWrapper not found")
    except Exception as e:
        print(f"❌ Method 4 failed: {e}")
    
    # METHOD 2: Add to path then import
    print("\n2️⃣ Method 2: Add to path then import")
    try:
        centroids_dir = os.path.join(module_dir, "Centroids_pipeline")
        if centroids_dir not in sys.path:
            sys.path.insert(0, centroids_dir)
        
        import confidence_wrapper
        if hasattr(confidence_wrapper, 'ConfidenceAnalysisWrapper'):
            ConfidenceAnalysisWrapper = confidence_wrapper.ConfidenceAnalysisWrapper
            wrapper2 = ConfidenceAnalysisWrapper(module_dir)
            print("✅ Method 2 SUCCESS!")
            return ("Method 2", ConfidenceAnalysisWrapper)
        else:
            print("❌ Method 2: Module imported but ConfidenceAnalysisWrapper not found")
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    # METHOD 3: Add module dir to path
    print("\n3️⃣ Method 3: Add module directory to path")
    try:
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        
        from Centroids_pipeline.confidence_wrapper import ConfidenceAnalysisWrapper
        wrapper3 = ConfidenceAnalysisWrapper(module_dir)
        print("✅ Method 3 SUCCESS!")
        return ("Method 3", ConfidenceAnalysisWrapper)
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
    
    
    # METHOD 5: Execute file directly
    print("\n5️⃣ Method 5: Execute file directly")
    try:
        # Execute the file in current namespace
        globals_dict = {}
        with open(wrapper_path, 'r') as f:
            exec(f.read(), globals_dict)
        
        if 'ConfidenceAnalysisWrapper' in globals_dict:
            ConfidenceAnalysisWrapper = globals_dict['ConfidenceAnalysisWrapper']
            wrapper5 = ConfidenceAnalysisWrapper(module_dir)
            print("✅ Method 5 SUCCESS!")
            return ("Method 5", ConfidenceAnalysisWrapper)
        else:
            print("❌ Method 5: File executed but ConfidenceAnalysisWrapper not found")
    except Exception as e:
        print(f"❌ Method 5 failed: {e}")
    
    print("\n❌ All import methods failed!")
    return None

def check_wrapper_file_content():
    """Check what's actually in the wrapper file."""
    print("🔍 CHECKING WRAPPER FILE CONTENT")
    print("=" * 40)
    
    wrapper_path = r"C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Centroids_pipeline\confidence_wrapper.py"
    
    if not os.path.exists(wrapper_path):
        print(f"❌ File not found: {wrapper_path}")
        return False
    
    try:
        with open(wrapper_path, 'r') as f:
            content = f.read()
        
        print(f"✅ File size: {len(content)} characters")
        
        # Check for key elements
        checks = [
            ("class ConfidenceAnalysisWrapper:", "Main class definition"),
            ("def __init__(self, module_dir):", "Constructor"),
            ("def run_full_confidence_analysis", "Main method"),
            ("class PatientEnsemblePipeline:", "Model class 1"),
            ("def predict_electrode_confidence_no_optuna", "Prediction function"),
            ("import numpy as np", "NumPy import"),
            ("import pandas as pd", "Pandas import"),
        ]
        
        print("\n📋 Content check:")
        for check, description in checks:
            if check in content:
                print(f"✅ {description}")
            else:
                print(f"❌ Missing: {description}")
        
        # Check for syntax errors
        try:
            compile(content, wrapper_path, 'exec')
            print("✅ Syntax is valid")
        except SyntaxError as e:
            print(f"❌ Syntax error: {e}")
            print(f"   Line {e.lineno}: {e.text}")
            return False
        
        # Show first few lines
        lines = content.split('\n')
        print(f"\n📄 First 10 lines:")
        for i, line in enumerate(lines[:10], 1):
            print(f"   {i:2d}: {line}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def test_stitch_with_working_import():
    """Test Stitch analysis using whichever import method works."""
    
    print("🏥 TESTING STITCH WITH WORKING IMPORT")
    print("=" * 50)
    
    # First find a working import method
    import_result = test_all_import_methods()
    
    if import_result is None:
        print("❌ No working import method found")
        return False
    
    method_name, ConfidenceAnalysisWrapper = import_result
    print(f"\n✅ Using {method_name} for Stitch test")
    
    # Setup paths
    module_dir = r"C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking"
    stitch_dir = r"C:\Users\rocia\Documents\SEEG_Results\STITCH"
    
    brain_mask = os.path.join(stitch_dir, "Brain_mask", "BrainMask_norm.nrrd")
    top_mask = os.path.join(stitch_dir, "Global_masks", "top_mask_1_patient1_CT.nrrd")
    enhanced_dir = os.path.join(stitch_dir, "Enhanced_masks")
    output_dir = os.path.join(stitch_dir, "Confidence_Working_Import_Test")
    
    # Check files exist
    print("\n📋 Checking Stitch files...")
    files_to_check = [brain_mask, top_mask, enhanced_dir]
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ {os.path.basename(file_path)}")
        else:
            print(f"❌ {os.path.basename(file_path)}")
            return False
    
    # Find enhanced CT file
    enhanced_ct = None
    for filename in os.listdir(enhanced_dir):
        if filename.endswith('.nrrd') and 'Filtered_DESCARGAR' not in filename:
            enhanced_ct = os.path.join(enhanced_dir, filename)
            break
    
    if not enhanced_ct:
        print("❌ No suitable enhanced CT file found")
        return False
    
    print(f"✅ Enhanced CT: {os.path.basename(enhanced_ct)}")
    
    # Initialize wrapper
    try:
        wrapper = ConfidenceAnalysisWrapper(module_dir)
        print("✅ Wrapper initialized")
        
        if not wrapper.is_confidence_analysis_available():
            missing = wrapper.get_missing_components()
            print(f"❌ Missing components: {', '.join(missing)}")
            return False
        
        print("✅ All components available")
        
    except Exception as e:
        print(f"❌ Wrapper initialization failed: {e}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Test centroid extraction
    print("\n📍 Testing centroid extraction...")
    try:
        coords = wrapper.extract_centroids_from_mask(top_mask)
        print(f"✅ Extracted {len(coords)} centroids")
        
        if len(coords) == 0:
            print("❌ No centroids found")
            return False
            
    except Exception as e:
        print(f"❌ Centroid extraction failed: {e}")
        return False
    
    # Run full analysis
    print("\n🤖 Running full confidence analysis...")
    try:
        results = wrapper.run_full_confidence_analysis(
            brain_mask_file=brain_mask,
            top_mask_file=top_mask,
            enhanced_ct_file=enhanced_ct,
            volume_name="stitch_test",
            confidence_dir=output_dir
        )
        
        print(f"\n🎉 SUCCESS! Analysis completed!")
        print(f"📊 Results:")
        print(f"   • Total electrodes: {results['total_electrodes']}")
        print(f"   • High confidence: {results['high_confidence']}")
        print(f"   • Top score: {results['top_confidence']:.4f}")
        print(f"   • Results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_debug():
    """Quick debug function."""
    print("🔧 Quick debug...")
    
    # Check if file exists and show basic info
    wrapper_path = r"C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Centroids_pipeline\confidence_wrapper.py"
    
    if os.path.exists(wrapper_path):
        size = os.path.getsize(wrapper_path)
        print(f"✅ File exists: {size} bytes")
        
        # Try to read first few lines
        try:
            with open(wrapper_path, 'r') as f:
                first_lines = [f.readline().strip() for _ in range(5)]
            print("First 5 lines:")
            for i, line in enumerate(first_lines, 1):
                print(f"  {i}: {line}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    else:
        print(f"❌ File not found: {wrapper_path}")
    
    return True

# Available functions
print("Available debug functions:")
print("  - quick_debug()")
print("  - check_wrapper_file_content()")
print("  - test_all_import_methods()")
print("  - test_stitch_with_working_import()")
print("\nStart with: quick_debug()")
#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Centroids_pipeline\debug.py').read())