#!/usr/bin/env python3
"""
Check CT-PET Alignment Verification
Quick visual and quantitative check to verify CT and PET are properly aligned
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_analyze_images(ct_path, pet_path):
    """
    Load CT and PET images and analyze their properties
    """
    print("📊 Loading and analyzing images...")
    
    # Load images
    ct_img = sitk.ReadImage(ct_path)
    pet_img = sitk.ReadImage(pet_path)
    
    # Get arrays
    ct_array = sitk.GetArrayFromImage(ct_img)
    pet_array = sitk.GetArrayFromImage(pet_img)
    
    # Image properties
    ct_props = {
        'size': ct_img.GetSize(),
        'spacing': ct_img.GetSpacing(),
        'origin': ct_img.GetOrigin(),
        'direction': ct_img.GetDirection()
    }
    
    pet_props = {
        'size': pet_img.GetSize(),
        'spacing': pet_img.GetSpacing(), 
        'origin': pet_img.GetOrigin(),
        'direction': pet_img.GetDirection()
    }
    
    return ct_img, pet_img, ct_array, pet_array, ct_props, pet_props

def check_geometric_alignment(ct_props, pet_props):
    """
    Check geometric properties for alignment
    """
    print("\n🔍 Checking geometric alignment...")
    
    alignment_score = 0
    total_checks = 0
    
    # Check image sizes
    print(f"CT size: {ct_props['size']}")
    print(f"PET size: {pet_props['size']}")
    if ct_props['size'] == pet_props['size']:
        print("✅ Image sizes match")
        alignment_score += 1
    else:
        print("⚠️ Image sizes differ")
    total_checks += 1
    
    # Check spacing
    ct_spacing = np.array(ct_props['spacing'])
    pet_spacing = np.array(pet_props['spacing'])
    spacing_diff = np.abs(ct_spacing - pet_spacing)
    print(f"CT spacing: {ct_spacing}")
    print(f"PET spacing: {pet_spacing}")
    print(f"Spacing difference: {spacing_diff}")
    
    if np.all(spacing_diff < 0.1):  # Within 0.1mm
        print("✅ Spacing closely matches")
        alignment_score += 1
    else:
        print("⚠️ Spacing differs significantly")
    total_checks += 1
    
    # Check origins
    ct_origin = np.array(ct_props['origin'])
    pet_origin = np.array(pet_props['origin'])
    origin_diff = np.abs(ct_origin - pet_origin)
    print(f"CT origin: {ct_origin}")
    print(f"PET origin: {pet_origin}")
    print(f"Origin difference: {origin_diff}")
    
    if np.all(origin_diff < 5.0):  # Within 5mm
        print("✅ Origins closely match")
        alignment_score += 1
    else:
        print("⚠️ Origins differ significantly")
    total_checks += 1
    
    # Check directions (orientation)
    ct_dir = np.array(ct_props['direction']).reshape(3,3)
    pet_dir = np.array(pet_props['direction']).reshape(3,3)
    dir_diff = np.abs(ct_dir - pet_dir)
    print(f"Direction matrix difference max: {np.max(dir_diff)}")
    
    if np.max(dir_diff) < 0.01:
        print("✅ Image orientations match")
        alignment_score += 1
    else:
        print("⚠️ Image orientations differ")
    total_checks += 1
    
    return alignment_score, total_checks

def create_alignment_visualization(ct_array, pet_array, output_path):
    """
    Create visual overlay to check alignment
    """
    print("\n🎨 Creating alignment visualization...")
    
    # Get middle slices (handle different dimensions)
    ct_mid_z = ct_array.shape[0] // 2
    pet_mid_z = pet_array.shape[0] // 2
    mid_y = min(ct_array.shape[1], pet_array.shape[1]) // 2
    mid_x = min(ct_array.shape[2], pet_array.shape[2]) // 2
    
    print(f"CT middle slice: {ct_mid_z}/{ct_array.shape[0]}")
    print(f"PET middle slice: {pet_mid_z}/{pet_array.shape[0]}")
    
    # Extract slices (use respective middle slices)
    ct_axial = ct_array[ct_mid_z, :, :]
    pet_axial = pet_array[pet_mid_z, :, :]
    
    ct_sagittal = ct_array[:, mid_y, :]
    pet_sagittal = pet_array[:, mid_y, :]
    
    ct_coronal = ct_array[:, :, mid_x]
    pet_coronal = pet_array[:, :, mid_x]
    
    # Normalize for display
    def normalize_for_display(img):
        img_norm = img - np.min(img)
        if np.max(img_norm) > 0:
            img_norm = img_norm / np.max(img_norm)
        return img_norm
    
    ct_axial = normalize_for_display(ct_axial)
    pet_axial = normalize_for_display(pet_axial)
    ct_sagittal = normalize_for_display(ct_sagittal)
    pet_sagittal = normalize_for_display(pet_sagittal)
    ct_coronal = normalize_for_display(ct_coronal)
    pet_coronal = normalize_for_display(pet_coronal)
    
    # Create overlays
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row: Individual images
    axes[0, 0].imshow(ct_axial, cmap='gray')
    axes[0, 0].set_title('CT - Axial')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ct_sagittal, cmap='gray')
    axes[0, 1].set_title('CT - Sagittal')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(ct_coronal, cmap='gray')
    axes[0, 2].set_title('CT - Coronal')
    axes[0, 2].axis('off')
    
    # Bottom row: PET overlays on CT
    axes[1, 0].imshow(ct_axial, cmap='gray', alpha=0.7)
    axes[1, 0].imshow(pet_axial, cmap='hot', alpha=0.5, vmin=0.1)
    axes[1, 0].set_title('CT + PET Overlay - Axial')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(ct_sagittal, cmap='gray', alpha=0.7)
    axes[1, 1].imshow(pet_sagittal, cmap='hot', alpha=0.5, vmin=0.1)
    axes[1, 1].set_title('CT + PET Overlay - Sagittal')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(ct_coronal, cmap='gray', alpha=0.7)
    axes[1, 2].imshow(pet_coronal, cmap='hot', alpha=0.5, vmin=0.1)
    axes[1, 2].set_title('CT + PET Overlay - Coronal')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved: {output_path}")

def analyze_pet_hotspots(pet_array):
    """
    Analyze PET hotspots to understand uptake pattern
    """
    print("\n🎯 Analyzing PET hotspots...")
    
    # Remove background
    valid_pet = pet_array[pet_array > 0]
    
    if len(valid_pet) == 0:
        print("⚠️ No valid PET data found")
        return
    
    # Statistics
    mean_uptake = np.mean(valid_pet)
    max_uptake = np.max(valid_pet)
    p95_uptake = np.percentile(valid_pet, 95)
    p99_uptake = np.percentile(valid_pet, 99)
    
    print(f"PET uptake statistics:")
    print(f"  Mean: {mean_uptake:.2f}")
    print(f"  Max: {max_uptake:.2f}")
    print(f"  95th percentile: {p95_uptake:.2f}")
    print(f"  99th percentile: {p99_uptake:.2f}")
    
    # Count hotspots at different thresholds
    hotspots_95 = np.sum(pet_array > p95_uptake)
    hotspots_99 = np.sum(pet_array > p99_uptake)
    
    print(f"Hotspot voxels:")
    print(f"  Above 95th percentile: {hotspots_95} voxels")
    print(f"  Above 99th percentile: {hotspots_99} voxels")

def main():
    """
    Main function to check CT-PET alignment
    """
    # File paths - Second case
    ct_path = "/Users/anish/output/AC38B1178240A5/#PP_TB_ABDOMEN_3.0_HD_FOV_L+H_H_SN120KV_12.nii.gz"
    pet_path = "/Users/anish/output/AC38B1178240A5/PET_DYNAMIC_PASS_4.nii.gz"
    
    print("🏥 CT-PET Alignment Verification")
    print("=" * 50)
    print(f"CT file: {ct_path}")
    print(f"PET file: {pet_path}")
    
    # Check if files exist
    if not Path(ct_path).exists():
        print(f"❌ CT file not found: {ct_path}")
        return
    
    if not Path(pet_path).exists():
        print(f"❌ PET file not found: {pet_path}")
        return
    
    try:
        # Load and analyze
        ct_img, pet_img, ct_array, pet_array, ct_props, pet_props = load_and_analyze_images(ct_path, pet_path)
        
        # Check geometric alignment
        alignment_score, total_checks = check_geometric_alignment(ct_props, pet_props)
        
        # Analyze PET data
        analyze_pet_hotspots(pet_array)
        
        # Create visualization
        output_viz = "/Users/anish/from_scratch_G/ct_pet_alignment_check.png"
        create_alignment_visualization(ct_array, pet_array, output_viz)
        
        # Overall assessment
        print(f"\n📊 ALIGNMENT ASSESSMENT:")
        print(f"Geometric alignment score: {alignment_score}/{total_checks}")
        
        if alignment_score == total_checks:
            print("✅ CT and PET appear to be WELL ALIGNED")
            print("✅ Safe to proceed with PSMA landmark-based registration")
        elif alignment_score >= total_checks * 0.75:
            print("⚠️ CT and PET have MINOR alignment differences")
            print("⚠️ May still work for landmark registration, but check visual overlay")
        else:
            print("❌ CT and PET appear to be POORLY ALIGNED")
            print("❌ Recommend proper CT-PET registration before MRI registration")
        
        print(f"\n📋 Next steps:")
        print(f"1. Review overlay visualization: {output_viz}")
        print(f"2. Check if anatomical structures align between CT and PET")
        print(f"3. Look for PSMA hotspots in expected anatomical regions")
        
    except Exception as e:
        print(f"❌ Error during alignment check: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()