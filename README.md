# Multi-Modal Medical Image Processing Pipeline

A comprehensive image processing pipeline for medical imaging research, built on elastix registration framework and supporting multiple imaging modalities including MRI, CT-PET, and PSMA.

## Overview

This pipeline provides:
1. **DICOM to NIFTI Conversion** - Multi-modal support with intelligent sequence detection
2. **Co-registration** - Elastix-based registration (implemented separately)
3. **Intensity Harmonization** - Nyul histogram normalization
4. **Segmentation** - CNN and Foundation Model based approaches
5. **Feature Extraction** - Handcrafted and deep learning features
6. **Classification** - ML and deep learning models
7. **Biostatistics** - Survival analysis and Cox regression

## Installation

### Requirements
```bash
pip install SimpleITK pydicom nibabel numpy scipy
```

### Dependencies
- Python 3.7+
- SimpleITK
- pydicom
- nibabel
- numpy
- scipy

## Module 1: DICOM to NIFTI Conversion

### Overview
The `MultiModalDicom2Nifti` class converts DICOM files to NIFTI format with intelligent sequence detection and standardized naming conventions.

### Supported Modalities

#### MRI Sequences
- **T2W** (T2-weighted): `t2_tse_tra`, `ax_t2`, `t2.*tra`, `t2.*axial`
- **ADC** (Apparent Diffusion Coefficient): `adc`, `apparent.*diffusion`
- **HBV** (Hepatic Blood Volume): `hbv`, `blood.*volume`, `perfusion.*`
- **DWI** (Diffusion Weighted): `dwi`, `diffusion.*weighted`, `b\d+`

#### CT-PET Sequences
- **CT**: `ct`, `ctac`, `attenuation.*correction`
- **PET**: `pet`, `suv`, `emission`, `positron.*emission`

#### PSMA Sequences
- **PSMA**: `psma`, `ga68`, `lu177`, `gallium.*`
- **Post-Contrast**: `post.*contrast`, `gd.*`, `gadolinium`

### Directory Structure Support

The converter supports flexible directory structures:

#### Structure 1: Flat Organization
```
Dataset/
├── Patient_001/
│   ├── file001.dcm          # T2W DICOM files
│   ├── file002.dcm
│   ├── file003.dcm          # ADC DICOM files
│   └── ...
├── Patient_002/
│   └── *.dcm
└── Patient_003/
    └── *.dcm
```

#### Structure 2: Nested by Sequence
```
Dataset/
├── Patient_001/
│   ├── T2_Sequence/
│   │   └── *.dcm           # T2W DICOM files
│   ├── DWI_ADC/
│   │   └── *.dcm           # ADC DICOM files
│   └── Perfusion_HBV/
│       └── *.dcm           # HBV DICOM files
├── Patient_002/
│   ├── T2_Axial/
│   ├── Diffusion/
│   └── Blood_Volume/
└── ...
```

#### Structure 3: Nested by Study/Series
```
Dataset/
├── Patient_001/
│   ├── Study_20231201/
│   │   ├── Series_001_T2/
│   │   │   └── *.dcm
│   │   ├── Series_002_DWI/
│   │   │   └── *.dcm
│   │   └── Series_003_ADC/
│   │       └── *.dcm
│   └── Study_20231202/
│       └── ...
└── ...
```

#### Structure 4: Mixed Modalities
```
Dataset/
├── MRI_Patient_001/
│   ├── T2W/
│   ├── ADC/
│   └── HBV/
├── CT_PET_Patient_002/
│   ├── CT_Series/
│   └── PET_Series/
├── PSMA_Patient_003/
│   ├── PSMA_Pre/
│   └── PSMA_Post/
└── ...
```

### Usage

#### Basic Usage
```python
from multimodal_dicom2nifti import MultiModalDicom2Nifti

# Initialize converter
converter = MultiModalDicom2Nifti()

# Convert entire dataset
results = converter.convert_dataset(
    parent_dir='/path/to/dicom/dataset',
    output_root='/path/to/nifti/output'
)
```

#### Single Case Conversion
```python
# Convert single case
converted_files = converter.convert_case(
    case_path='/path/to/patient/folder',
    output_dir='/path/to/output/case',
    case_id='Patient_001'
)
```

#### Sequence Filtering for Pipeline
```python
# Filter for specific MRI sequences
required_sequences = ['T2W', 'ADC', 'HBV']

for case_id, converted_files in results.items():
    sequence_files = converter.filter_sequences(
        converted_files, 
        required_sequences
    )
    
    print(f"Case {case_id}:")
    for seq_type, file_path in sequence_files.items():
        print(f"  {seq_type}: {file_path}")
```

### Output Format

#### Naming Convention
Files are saved with standardized names:
- Format: `{CaseID}_{SequenceType}[_SeriesNumber].nii.gz`
- Examples:
  - `Patient_001_T2W.nii.gz`
  - `Patient_001_ADC.nii.gz`
  - `Patient_001_HBV.nii.gz`
  - `Patient_001_DWI_b1000.nii.gz`

#### Multiple Instances
When multiple series of the same type are found:
- `Patient_001_T2W.nii.gz` (first instance)
- `Patient_001_T2W_2.nii.gz` (second instance)
- `Patient_001_T2W_3.nii.gz` (third instance)

#### Output Structure
```
Output/
├── Patient_001/
│   ├── Patient_001_T2W.nii.gz
│   ├── Patient_001_ADC.nii.gz
│   ├── Patient_001_HBV.nii.gz
│   └── Patient_001_DWI_b1000.nii.gz
├── Patient_002/
│   ├── Patient_002_T2W.nii.gz
│   ├── Patient_002_ADC.nii.gz
│   └── Patient_002_CT.nii.gz
└── ...
```

### Sequence Detection Logic

The converter uses pattern matching on DICOM metadata:
- **SeriesDescription**: Primary field for sequence identification
- **ProtocolName**: Secondary field for additional context
- **Modality**: Used for high-level modality classification

#### Custom Sequence Patterns
You can extend sequence detection by modifying the `sequence_patterns` dictionary:

```python
converter = MultiModalDicom2Nifti()

# Add custom patterns
converter.sequence_patterns['FLAIR'] = [
    r'flair', r'fluid.*attenuated', r'dark.*fluid'
]

converter.sequence_patterns['SWI'] = [
    r'swi', r'susceptibility.*weighted', r'venous.*bold'
]
```

### Error Handling

The converter includes robust error handling:
- **Missing DICOM files**: Warns and continues with other cases
- **Corrupted series**: Skips problematic series and continues
- **Unknown sequences**: Labels as 'UNKNOWN' for manual review
- **Duplicate series**: Automatically numbered for disambiguation

### Console Output Example

```
🔄 Processing case: Patient_001
📁 Found DICOM files in 3 directories
  📂 Processing subdirectory: T2_Sequence
    ✅ Converted: T2W -> Patient_001_T2W.nii.gz
       📋 MR | t2_tse_tra_p2_iso
  📂 Processing subdirectory: DWI_ADC
    ✅ Converted: ADC -> Patient_001_ADC.nii.gz
       📋 MR | ep2d_diff_tra_DYNDIST_ADC
  📂 Processing subdirectory: Perfusion
    ✅ Converted: HBV -> Patient_001_HBV.nii.gz
       📋 MR | tfl_artperfusion_tra_blood_volume

🎉 Conversion complete! Processed 1 cases.
```

### Integration with Downstream Pipeline

#### Elastix Registration
After conversion, files are ready for elastix-based co-registration:
```python
# Get standardized sequence files
sequence_files = converter.filter_sequences(converted_files, ['T2W', 'ADC'])

# Use with elastix (next pipeline step)
fixed_image = sequence_files['T2W']    # Reference image
moving_image = sequence_files['ADC']   # Image to register
```

#### Quality Control
The converter maintains metadata for quality assessment:
```python
metadata = converter.get_dicom_metadata(dicom_file)
print(f"Slice Thickness: {metadata.get('SliceThickness', 'N/A')} mm")
print(f"Field Strength: {metadata.get('MagneticFieldStrength', 'N/A')} T")
```

### Troubleshooting

#### Common Issues

1. **No DICOM series found**
   - Check if files have DICOM headers
   - Verify directory permissions
   - Ensure files aren't compressed/corrupted

2. **Unknown sequence types**
   - Review SeriesDescription in console output
   - Add custom patterns for your institution's naming
   - Check if sequences are non-standard

3. **Missing required sequences**
   - Use `filter_sequences()` to identify missing data
   - Review sequence detection patterns
   - Check if sequences have different names

4. **Memory issues with large datasets**
   - Process cases individually rather than entire dataset
   - Monitor available RAM during conversion
   - Consider processing subsets

#### DICOM File Validation
```python
# Check if directory contains valid DICOM files
dicom_dirs = converter.find_dicom_directories('/path/to/data')
print(f"Found DICOM files in {len(dicom_dirs)} directories")

# Validate individual files
is_valid = converter.is_dicom_file('/path/to/file')
```

## Next Pipeline Steps

After DICOM to NIFTI conversion, the pipeline continues with:

1. **Co-registration** (elastix-based)
2. **Intensity harmonization** (Nyul method)
3. **Segmentation** (CNN/Foundation models)
4. **Feature extraction** (Radiomics/Deep features)
5. **Classification** (ML/DL models)
6. **Statistical analysis** (Survival/Cox regression)

## Contributing

When adding new sequence types or modalities:
1. Update `sequence_patterns` dictionary
2. Add corresponding examples to this README
3. Test with representative datasets
4. Update documentation

## References

Based on existing DICOM conversion workflows and extended for multi-modal medical imaging research. Compatible with elastix registration framework and ITK-based processing pipelines.