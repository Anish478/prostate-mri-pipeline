#!/usr/bin/env python3
"""
Create overlay images for coregistered cases to visualize registration quality.
Generates overlays of:
1. T2W (reference) with registered ADC
2. T2W (reference) with registered HBV
3. Registered ADC with registered HBV
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def normalize_image(image_array, percentile_clip=99.5):
    """Normalize image to 0-1 range with percentile clipping."""
    lower = np.percentile(image_array, 100 - percentile_clip)
    upper = np.percentile(image_array, percentile_clip)
    
    normalized = np.clip(image_array, lower, upper)
    normalized = (normalized - lower) / (upper - lower)
    return normalized

def create_overlay_image(fixed_array, moving_array, alpha=0.6, colormap_moving='hot'):
    """Create overlay of two normalized images."""
    # Normalize both images
    fixed_norm = normalize_image(fixed_array)
    moving_norm = normalize_image(moving_array)
    
    # Create RGB overlay
    # Fixed image in grayscale
    overlay = np.stack([fixed_norm, fixed_norm, fixed_norm], axis=-1)
    
    # Apply colormap to moving image
    if colormap_moving == 'hot':
        # Hot colormap: black -> red -> yellow -> white
        moving_colored = np.zeros((*moving_norm.shape, 3))
        moving_colored[..., 0] = moving_norm  # Red channel
        moving_colored[..., 1] = np.clip(2 * moving_norm - 1, 0, 1)  # Green channel
        moving_colored[..., 2] = np.clip(3 * moving_norm - 2, 0, 1)  # Blue channel
    elif colormap_moving == 'jet':
        # Simple jet-like colormap
        moving_colored = plt.cm.jet(moving_norm)[..., :3]
    else:
        # Default to grayscale
        moving_colored = np.stack([moving_norm, moving_norm, moving_norm], axis=-1)
    
    # Blend images
    overlay = (1 - alpha) * overlay + alpha * moving_colored
    overlay = np.clip(overlay, 0, 1)
    
    return overlay

def get_middle_slices(image_arrays, num_slices=5):
    """Get middle slices from 3D volumes, using the minimum size."""
    # Find the minimum z-size across all arrays
    min_z_size = min(arr.shape[0] for arr in image_arrays)  # SimpleITK arrays are (z, y, x)
    
    middle = min_z_size // 2
    start = max(0, middle - num_slices // 2)
    end = min(min_z_size, start + num_slices)
    
    # Ensure we have at least one slice
    if end <= start:
        start = max(0, min_z_size // 2)
        end = min(min_z_size, start + 1)
    
    return range(start, end)

def create_case_overlays(case_dir, output_dir):
    """Create overlay images for a single case."""
    case_id = os.path.basename(case_dir).replace('_coregistered', '')
    print(f"  📸 Creating overlays for {case_id}")
    
    # Find the required files
    t2w_file = None
    adc_registered = None
    hbv_registered = None
    
    # Look for T2W reference file
    t2w_candidates = glob.glob(os.path.join(case_dir, "*t2w_reference*.nii.gz"))
    if t2w_candidates:
        t2w_file = t2w_candidates[0]
    
    # Look for registered ADC
    adc_dir = os.path.join(case_dir, "adc_registration")
    if os.path.exists(adc_dir):
        for ext in ["result.0.nii.gz", "result.nii", "result.nii.gz"]:
            potential = os.path.join(adc_dir, ext)
            if os.path.exists(potential):
                adc_registered = potential
                break
    
    # Look for registered HBV
    hbv_dir = os.path.join(case_dir, "hbv_registration")
    if os.path.exists(hbv_dir):
        for ext in ["result.nii", "result.nii.gz"]:
            potential = os.path.join(hbv_dir, ext)
            if os.path.exists(potential):
                hbv_registered = potential
                break
    
    if not all([t2w_file, adc_registered, hbv_registered]):
        print(f"    ❌ Missing files for {case_id}")
        print(f"       T2W: {t2w_file is not None}")
        print(f"       ADC: {adc_registered is not None}")
        print(f"       HBV: {hbv_registered is not None}")
        return False
    
    try:
        # Read images
        t2w_img = sitk.ReadImage(t2w_file)
        adc_img = sitk.ReadImage(adc_registered)
        hbv_img = sitk.ReadImage(hbv_registered)
        
        # Convert to numpy arrays
        t2w_array = sitk.GetArrayFromImage(t2w_img)  # (z, y, x)
        adc_array = sitk.GetArrayFromImage(adc_img)
        hbv_array = sitk.GetArrayFromImage(hbv_img)
        
        # Create output directory for this case
        case_output_dir = os.path.join(output_dir, f"{case_id}_overlays")
        os.makedirs(case_output_dir, exist_ok=True)
        
        # Get middle slices based on minimum volume size
        middle_slices = get_middle_slices([t2w_array, adc_array, hbv_array], num_slices=5)
        
        # Create overlays for each middle slice
        for slice_idx in middle_slices:
            # Extract 2D slices
            t2w_slice = t2w_array[slice_idx, :, :]
            adc_slice = adc_array[slice_idx, :, :]
            hbv_slice = hbv_array[slice_idx, :, :]
            
            # Create overlay images
            t2w_adc_overlay = create_overlay_image(t2w_slice, adc_slice, alpha=0.5, colormap_moving='hot')
            t2w_hbv_overlay = create_overlay_image(t2w_slice, hbv_slice, alpha=0.5, colormap_moving='jet')
            adc_hbv_overlay = create_overlay_image(adc_slice, hbv_slice, alpha=0.5, colormap_moving='jet')
            
            # Save overlay images
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'{case_id} - Slice {slice_idx}', fontsize=14)
            
            # Top row: Individual images
            axes[0, 0].imshow(t2w_slice, cmap='gray')
            axes[0, 0].set_title('T2W (Reference)')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(adc_slice, cmap='gray')
            axes[0, 1].set_title('ADC (Registered)')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(hbv_slice, cmap='gray')
            axes[0, 2].set_title('HBV (Registered)')
            axes[0, 2].axis('off')
            
            # Bottom row: Overlays
            axes[1, 0].imshow(t2w_adc_overlay)
            axes[1, 0].set_title('T2W + ADC Overlay')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(t2w_hbv_overlay)
            axes[1, 1].set_title('T2W + HBV Overlay')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(adc_hbv_overlay)
            axes[1, 2].set_title('ADC + HBV Overlay')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(case_output_dir, f'{case_id}_slice_{slice_idx:03d}_overlays.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"    ✅ Created overlays for {len(middle_slices)} slices")
        return True
        
    except Exception as e:
        print(f"    ❌ Error creating overlays for {case_id}: {str(e)}")
        return False

def main():
    # Configuration
    coregistered_dir = "/Users/anish/from_scratch_G/coregistered_results"
    output_dir = "/Users/anish/from_scratch_G/overlay_results"
    
    print("🎨 Creating Overlay Images for Coregistered Cases")
    print("=" * 60)
    print(f"📂 Coregistered results: {coregistered_dir}")
    print(f"📂 Overlay output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all coregistered case directories
    case_dirs = glob.glob(os.path.join(coregistered_dir, "*_coregistered"))
    
    if not case_dirs:
        print("❌ No coregistered case directories found!")
        return
    
    print(f"✅ Found {len(case_dirs)} coregistered cases")
    
    # Process each case
    successful = 0
    failed = 0
    
    for i, case_dir in enumerate(sorted(case_dirs), 1):
        print(f"\n[{i}/{len(case_dirs)}] Processing {os.path.basename(case_dir)}")
        
        success = create_case_overlays(case_dir, output_dir)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n🎉 Overlay creation complete!")
    print(f"✅ Successful: {successful}/{len(case_dirs)}")
    print(f"❌ Failed: {failed}/{len(case_dirs)}")
    print(f"📂 Overlays saved in: {output_dir}")
    
    # Create summary HTML file
    create_summary_html(output_dir, successful)

def create_summary_html(output_dir, num_cases):
    """Create an HTML summary file with links to all overlay images."""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Coregistration Overlay Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .case {{ margin: 20px 0; border: 1px solid #ddd; padding: 15px; }}
        .case h3 {{ margin-top: 0; color: #666; }}
        .overlay-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 10px; }}
        .overlay-item {{ text-align: center; }}
        .overlay-item img {{ max-width: 100%; height: auto; border: 1px solid #ccc; }}
        .overlay-item p {{ margin: 5px 0; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <h1>Coregistration Overlay Results</h1>
    <p>Generated overlay images for {num_cases} successfully coregistered cases.</p>
    <p>Each case shows T2W (reference), registered ADC, and registered HBV images along with their overlays.</p>
    
    <div id="cases">
"""
    
    # Find all case overlay directories
    case_overlay_dirs = glob.glob(os.path.join(output_dir, "*_overlays"))
    
    for case_dir in sorted(case_overlay_dirs):
        case_id = os.path.basename(case_dir).replace('_overlays', '')
        html_content += f"""
        <div class="case">
            <h3>{case_id}</h3>
            <div class="overlay-grid">
"""
        
        # Find all overlay images for this case
        overlay_images = glob.glob(os.path.join(case_dir, "*.png"))
        
        for img_path in sorted(overlay_images):
            img_name = os.path.basename(img_path)
            relative_path = os.path.relpath(img_path, output_dir)
            
            html_content += f"""
                <div class="overlay-item">
                    <img src="{relative_path}" alt="{img_name}">
                    <p>{img_name}</p>
                </div>
"""
        
        html_content += """
            </div>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    # Write HTML file
    html_file = os.path.join(output_dir, "overlay_summary.html")
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    print(f"📄 Summary HTML created: {html_file}")

if __name__ == "__main__":
    main()