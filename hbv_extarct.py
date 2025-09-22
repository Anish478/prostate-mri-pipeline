import SimpleITK as sitk
import numpy as np

# Load the 4D image
img4d = sitk.ReadImage('/Users/anish/from_scratch_G/IUSM_MRI_nifti/STUDY_P33931/Ax_DWI_3_B_VALUES.nii.gz')
data4d = sitk.GetArrayFromImage(img4d)  # shape: [Z,Y,X,Time]

# Compute the mean across the time dimension
mean_data = np.mean(data4d, axis=3)

# Create and save the 3D image (swapping axes back to [X,Y,Z])
mean_img = sitk.GetImageFromArray(mean_data)
mean_img.SetDirection(img4d.GetDirection())
mean_img.SetOrigin(img4d.GetOrigin())
mean_img.SetSpacing(img4d.GetSpacing()[:3])
sitk.WriteImage(mean_img, 'HBV_eDWI3_1_mean.nii.gz')

print("Saved 3D mean high‑b image to HBV_eDWI3_1_mean.nii.gz")



