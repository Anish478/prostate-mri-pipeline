# Co-registration Module

This stage rigidly aligns all study volumes to a common reference (by default each case's `t2w.nii.gz`). Implementation is 100% elastix/transformix; Python is only used as a thin executor and for QA metrics/visualization.

```
preprocessing/co_registration/
├── parameters/
│   └── rigid.txt                # similarity-transform parameters (from elastix manual)
├── registration_wrapper.py      # run_elastix(), apply_transform()
├── hbv_registration_fix.py     # specialized HBV registration with preprocessing
├── evaluation.py                # dice / jaccard / mse helpers
├── visualize.py                 # quick matplotlib overlay
└── README.md                    # this file
```

## Usage

### Standard Registration (ADC)
```python
from preprocessing.co_registration.registration_wrapper import run_elastix, apply_transform

# ADC registration works well with standard rigid parameters
transform_params = run_elastix(
    fixed_image="t2w.nii.gz",
    moving_image="adc.nii.gz", 
    output_dir="output/adc_to_t2w",
    param_files=["parameters/rigid.txt"]
)

result = apply_transform(
    moving_image="adc.nii.gz",
    transform_param=transform_params,
    output_dir="output/adc_to_t2w"
)
```

### HBV Registration (Special Handling)
```python
from preprocessing.co_registration.hbv_registration_fix import register_hbv_with_fix

# HBV requires special preprocessing due to different acquisition characteristics
result = register_hbv_with_fix(
    fixed_image="t2w.nii.gz",
    moving_image="hbv.nii.gz",
    output_dir="output/hbv_to_t2w"
)
```

## Pipeline Integration

For complete case processing:
```python
from pathlib import Path
from preprocessing.co_registration.registration_wrapper import run_elastix, apply_transform
from preprocessing.co_registration.hbv_registration_fix import register_hbv_with_fix

def register_case(case_dir: Path, output_dir: Path):
    """Register ADC and HBV to T2W for a complete case."""
    
    # Input files
    t2w = case_dir / "t2w.nii.gz"
    adc = case_dir / "adc.nii.gz" 
    hbv = case_dir / "hbv.nii.gz"
    
    # ADC registration
    adc_output = output_dir / "adc_to_t2w"
    adc_transform = run_elastix(t2w, adc, adc_output, ["parameters/rigid.txt"])
    adc_result = apply_transform(adc, adc_transform, adc_output)
    
    # HBV registration (specialized)
    hbv_output = output_dir / "hbv_to_t2w"
    hbv_result = register_hbv_with_fix(t2w, hbv, hbv_output)
    
    return {
        "t2w": t2w,
        "adc_registered": adc_result, 
        "hbv_registered": hbv_result
    }
```

## Quality Assessment

Load these files in 3D Slicer for visual verification:
1. `t2w.nii.gz` (reference)
2. `adc_to_t2w/result.nii.gz` (ADC aligned to T2W grid)
3. `hbv_to_t2w/hbv_enhanced.nii.gz` (HBV aligned, preserving native resolution)

## Key Notes

- **ADC**: Uses standard rigid registration with mutual information
- **HBV**: Requires specialized handling due to different geometric origins and contrast characteristics
- **Result grids**: ADC is resampled to T2W grid; HBV preserves its native high-resolution grid
- **Transform reuse**: ADC and HBV transforms are computed independently due to different acquisition geometries

## FAQ

**Q: Why can't I reuse the ADC transform for HBV?**  
A: ADC and HBV have different geometric origins (~81mm apart in Z-direction) and different contrast characteristics, requiring separate registration.

**Q: Why does HBV keep its original resolution?**  
A: HBV has 56 thin slices (1.47mm) vs T2W's 28 thick slices (3mm). Downsampling would lose significant detail.

**Q: What if registration fails?**  
A: Check that elastix libraries are properly configured with `export DYLD_FALLBACK_LIBRARY_PATH=/path/to/elastix/lib`
