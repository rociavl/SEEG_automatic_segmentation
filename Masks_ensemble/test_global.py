# Quick test script for global masking functionality
# Run this in Slicer's Python console or as a separate script

import os
import sys
import logging

def test_global_masking():
    """Quick test for the global masking functionality"""
    
    # === SETUP PATHS ===
    # Update these paths to match your setup:
    enhanced_masks_dir = r"C:\Users\rocia\Documents\SEEG_Results\Stitch\Enhanced_masks"  # Your enhanced masks folder
    test_output_dir = r"C:\Users\rocia\Documents\SEEG_Results\Stitch\Global_masks_test"  # Test output folder
    
    print(f"Testing global masking...")
    print(f"Input dir: {enhanced_masks_dir}")
    print(f"Output dir: {test_output_dir}")
    
    # === CHECK INPUT DIRECTORY ===
    if not os.path.exists(enhanced_masks_dir):
        print(f"❌ ERROR: Enhanced masks directory not found: {enhanced_masks_dir}")
        print("Please update the enhanced_masks_dir path in this script.")
        return False
    
    # Check for .nrrd files
    nrrd_files = [f for f in os.listdir(enhanced_masks_dir) if f.endswith('.nrrd')]
    print(f"Found {len(nrrd_files)} .nrrd files in input directory:")
    for i, file in enumerate(nrrd_files, 1):
        print(f"  {i}. {file}")
    
    if len(nrrd_files) == 0:
        print("❌ ERROR: No .nrrd files found in enhanced masks directory")
        return False
    
    # === CREATE OUTPUT DIRECTORY ===
    try:
        os.makedirs(test_output_dir, exist_ok=True)
        print(f"✅ Created output directory: {test_output_dir}")
    except Exception as e:
        print(f"❌ ERROR: Failed to create output directory: {e}")
        return False
    
    # === IMPORT AND TEST ===
    try:
        # Add the module directory to path if needed
        # module_dir = r"C:\path\to\your\SEEG_module"  # Update if needed
        # if module_dir not in sys.path:
        #     sys.path.append(module_dir)
        
        from masks_fusion_top import EnhancedMaskSelector
        print("✅ Successfully imported EnhancedMaskSelector")
        
    except ImportError as e:
        print(f"❌ ERROR: Failed to import EnhancedMaskSelector: {e}")
        print("Make sure masks_fusion_top.py is in your Python path")
        return False
    
    # === RUN GLOBAL MASKING ===
    try:
        print("\n🔄 Starting global masking process...")
        
        # Initialize selector
        selector = EnhancedMaskSelector(enhanced_masks_dir, test_output_dir)
        print(f"✅ Initialized selector with {len(selector.masks)} masks")
        
        # Remove excluded file if it exists
        excluded_file = "Filtered_DESCARGAR_roi_volume_features_ctp.3D"  # Without .nrrd extension
        if excluded_file in selector.masks:
            del selector.masks[excluded_file]
            print(f"✅ Removed excluded file: {excluded_file}")
        
        if len(selector.masks) < 4:
            print(f"⚠️  WARNING: Only {len(selector.masks)} masks available (need 4 for full test)")
            n_masks = min(len(selector.masks), 2)  # Use what we have
        else:
            n_masks = 4
        
        # Select best masks
        print(f"\n🔍 Selecting best {n_masks} masks...")
        selected_masks = selector.select_best_masks(n_masks=n_masks)
        print(f"✅ Selected masks: {selected_masks}")
        
        # Save individual top masks
        n_individual = min(2, len(selected_masks))
        print(f"\n💾 Saving top {n_individual} individual masks...")
        for i, mask_name in enumerate(selected_masks[:n_individual], 1):
            output_name = f"test_top_mask_{i}"
            selector.save_mask(selector.masks[mask_name], output_name)
            print(f"✅ Saved: {output_name}.nrrd")
        
        # Create progressive masks
        if len(selected_masks) >= 1:
            print(f"\n🔗 Creating progressive fused masks...")
            
            # Progressive 1: Just best mask
            progressive_1 = selector.create_weighted_fused_mask(
                selected_masks[:1], 
                "test_progressive_1mask"
            )
            print("✅ Created progressive_1mask.nrrd")
            
            # Progressive 2: Best 2 masks (if available)
            if len(selected_masks) >= 2:
                progressive_2 = selector.create_weighted_fused_mask(
                    selected_masks[:2], 
                    "test_progressive_2masks"
                )
                print("✅ Created progressive_2masks.nrrd")
        
        # Test voxel count calculation
        print(f"\n📊 Calculating voxel statistics...")
        all_mask_names = list(selector.masks.keys())
        original_voxels = sum(selector.masks[name].sum() for name in all_mask_names)
        selected_voxels = sum(selector.masks[name].sum() for name in selected_masks)
        reduction = ((original_voxels - selected_voxels) / original_voxels) * 100
        
        print(f"✅ Original total voxels: {original_voxels:,}")
        print(f"✅ Selected total voxels: {selected_voxels:,}")
        print(f"✅ Reduction: {reduction:.1f}%")
        
        # Test plotting if method exists
        try:
            if hasattr(selector, 'plot_voxel_count_comparison'):
                plots_dir = os.path.join(test_output_dir, "plots")
                selector.plot_voxel_count_comparison(selected_masks, plots_dir)
                print("✅ Created voxel count comparison plot")
            else:
                print("ℹ️  Voxel count plot method not available (using basic stats)")
        except Exception as e:
            print(f"⚠️  Plot creation failed: {e}")
        
        print(f"\n✅ GLOBAL MASKING TEST COMPLETED SUCCESSFULLY!")
        print(f"📁 Check results in: {test_output_dir}")
        
        # List created files
        created_files = [f for f in os.listdir(test_output_dir) if f.endswith('.nrrd')]
        print(f"\n📄 Created {len(created_files)} mask files:")
        for file in created_files:
            print(f"  • {file}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR during global masking: {e}")
        import traceback
        traceback.print_exc()
        return False

# === RUN THE TEST ===
if __name__ == "__main__":
    success = test_global_masking()
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed - check errors above")
else:
    # If running in Slicer console, just call the function
    print("Run: test_global_masking()")

#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\Masks_ensemble\test_global.py').read())