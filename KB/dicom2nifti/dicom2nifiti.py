import os

import SimpleITK as sitk

import pydicom
 
parent_dir = '/N/project/prostate_cancer_ai/In_house_MRI_dataset/2021-00119_MRI'

output_root = '/N/project/prostate_cancer_ai/Sumedh/output_nifti1'

os.makedirs(output_root, exist_ok=True)
 
for case_folder in os.listdir(parent_dir):

    case_path = os.path.join(parent_dir, case_folder)

    if not os.path.isdir(case_path):

        continue
 
    print(f" Processing case: {case_folder}")

    output_dir = os.path.join(output_root, case_folder)

    os.makedirs(output_dir, exist_ok=True)
 
    reader = sitk.ImageSeriesReader()

    series_ids = reader.GetGDCMSeriesIDs(case_path)

    if not series_ids:

        print(f"âš ï¸ No DICOM series found in {case_folder}")

        continue
 
    for idx, series_id in enumerate(series_ids):

        file_names = reader.GetGDCMSeriesFileNames(case_path, series_id)

        reader.SetFileNames(file_names)

        try:

            image = reader.Execute()

        except Exception as e:

            print(f"âŒ Failed to read series {series_id} in case {case_folder}: {e}")

            continue
 
        # Get SeriesDescription from metadata

        try:

            ds = pydicom.dcmread(file_names[0], stop_before_pixels=True)

            description = ds.get('SeriesDescription', f'series{idx+1}')

            description_clean = description.replace(' ', '_').replace('/', '_')

        except Exception:

            description_clean = f'series{idx+1}'
 
        output_path = os.path.join(output_dir, f'{description_clean}.nii.gz')

        sitk.WriteImage(image, output_path)

        print(f"âœ… Saved: {output_path}")
 
print(" ALL DONE! Converted all readable series.")
 
 