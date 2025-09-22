#!/usr/bin/env python3
"""
Batch coregistration script using improved_rigid.txt parameters.
Processes all cases with complete T2W+ADC+HBV sequences.
"""

import os
import subprocess
import shutil
from pathlib import Path

def run_coregistration_improved():
    """
    Run coregistration on all complete cases using improved_rigid.txt parameters.
    Results saved in coregistered_results_improved/ folder.
    """
    
    # Set elastix library path
    os.environ['DYLD_LIBRARY_PATH'] = '/Users/anish/elastix-5/lib'
    
    # Define paths
    nifti_dir = '/Users/anish/from_scratch_G/nif'
    output_dir = '/Users/anish/from_scratch_G/coregistered_results_pranav_nif'
    parameter_file = '/Users/anish/from_scratch_G/preprocessing/co_registration/parameters/rigid.txt'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(" Starting batch coregistration with improved_rigid.txt parameters")
    print("=" * 70)
    
    # Get all case directories
    case_dirs = [d for d in os.listdir(nifti_dir) if os.path.isdir(os.path.join(nifti_dir, d))]
    
    complete_cases = []
    incomplete_cases = []
    
    # First pass: identify complete cases
    for case_id in sorted(case_dirs):
        case_path = os.path.join(nifti_dir, case_id)
        mri_path = os.path.join(case_path, f"{case_id}_mri")
        
        if not os.path.exists(mri_path):
            incomplete_cases.append((case_id, "No MRI folder"))
            continue
        
        # ------------------------------------------------------------------
        # The MRI extractor  writes canonical names:
        #     t2w.nii.gz   adc.nii.gz   hbv.nii.gz   (plus _2, _3 … if duplicates)
        # We just need the first instance of each.
        # ------------------------------------------------------------------
        t2w_path = Path(mri_path, "t2w.nii.gz")
        adc_path = Path(mri_path, "adc.nii.gz")
        hbv_path = Path(mri_path, "hbv.nii.gz")

        if not t2w_path.exists():
            incomplete_cases.append((case_id, "No T2W"))
            continue
        if not adc_path.exists():
            incomplete_cases.append((case_id, "No ADC"))
            continue
        if not hbv_path.exists():
            incomplete_cases.append((case_id, "No HBV"))
            continue

        complete_cases.append(
            (case_id, str(t2w_path), str(adc_path), str(hbv_path))
        )
    
    print(f" Found {len(complete_cases)} complete cases, {len(incomplete_cases)} incomplete")
    
    if incomplete_cases:
        print(f"\n Incomplete cases:")
        for case_id, reason in incomplete_cases:
            print(f"  {case_id}: {reason}")
    
    print(f"\n Processing {len(complete_cases)} complete cases:")
    
    successful = 0
    failed = 0
    
    # Process each complete case
    for i, (case_id, t2w_file, adc_file, hbv_file) in enumerate(complete_cases, 1):
        print(f"\n[{i}/{len(complete_cases)}] Processing {case_id}")
        
        # Create case output directory
        case_output_dir = os.path.join(output_dir, f"{case_id}_coregistered")
        os.makedirs(case_output_dir, exist_ok=True)
        
        try:
            # Copy T2W as reference
            t2w_ref_path = os.path.join(case_output_dir, f"t2w_reference_{case_id}.nii.gz")
            shutil.copy2(t2w_file, t2w_ref_path)
            print(f"   T2W reference: {os.path.basename(t2w_file)}")
            
            # 1. Register ADC to T2W
            adc_reg_dir = os.path.join(case_output_dir, "adc_registration")
            os.makedirs(adc_reg_dir, exist_ok=True)
            
            print(f"   Registering ADC...")
            adc_cmd = [
                '/Users/anish/elastix-5/bin/elastix',
                '-f', t2w_file,
                '-m', adc_file,
                '-out', adc_reg_dir,
                '-p', parameter_file
            ]
            
            result = subprocess.run(adc_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"   ADC registration failed")
                failed += 1
                continue
                
            # 2. Transform HBV using ADC transformation
            hbv_reg_dir = os.path.join(case_output_dir, "hbv_registration")
            os.makedirs(hbv_reg_dir, exist_ok=True)
            
            print(f"   Transforming HBV...")
            transform_params = os.path.join(adc_reg_dir, "TransformParameters.0.txt")
            
            hbv_cmd = [
                '/Users/anish/elastix-5/bin/transformix',
                '-in', hbv_file,
                '-out', hbv_reg_dir,
                '-tp', transform_params
            ]
            
            result = subprocess.run(hbv_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"   HBV transformation failed")
                failed += 1
                continue
            
            print(f"   Completed successfully")
            successful += 1
            
        except Exception as e:
            print(f"   Error: {e}")
            failed += 1
            continue
    
    print(f"\n Batch coregistration completed!")
    print(f" Successful: {successful}")
    print(f" Failed: {failed}")
    print(f" Results saved in: {output_dir}")

if __name__ == "__main__":
    run_coregistration_improved()