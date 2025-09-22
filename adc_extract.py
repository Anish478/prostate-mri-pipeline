import numpy as np
import nibabel as nib
import os

# --- Parameters ---
b0_path    = '/Users/anish/from_scratch_G/IUSM_MRI_nifti/STUDY_P45270/SB0.nii.gz'
b1_path    = '/Users/anish/from_scratch_G/IUSM_MRI_nifti/STUDY_P45270/SB1500.nii.gz'
b_value    = 1500.0  # s/mm² for SB1000
output_path = '/Users/anish/from_scratch_G/IUSM_MRI_nifti/STUDY_P45270/ADC_map.nii.gz'

# --- Load data ---
img0 = nib.load(b0_path)
img1 = nib.load(b1_path)

S0 = img0.get_fdata().astype(np.float32)
S1 = img1.get_fdata().astype(np.float32)

# --- Compute ADC ---
# Avoid division by zero / log of zero
mask = S0 > 0
ADC = np.zeros_like(S0, dtype=np.float32)
ADC[mask] = -np.log(S1[mask] / S0[mask]) / b_value

# Optional: clip to physiologic range (0 – 0.003 mm^2/s)
ADC = np.clip(ADC, 0, 0.003)

# --- Save ADC map ---
adc_img = nib.Nifti1Image(ADC, img0.affine, img0.header)
nib.save(adc_img, output_path)

print(f"Saved ADC map to {output_path}")
