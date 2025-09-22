import SimpleITK as sitk
import os
import sys
from pathlib import Path

def convert_mha_to_nifti(mha_file_path, output_dir=None):
    """
    Convert a single .mha file to .nii.gz format
    
    Args:
        mha_file_path (str/Path): Path to the .mha file
        output_dir (str/Path): Output directory (default: same as input)
    
    Returns:
        str: Path to the converted .nii.gz file
    """
    mha_path = Path(mha_file_path)
    
    if not mha_path.exists():
        raise FileNotFoundError(f"MHA file not found: {mha_path}")
    
    # Set output directory
    if output_dir is None:
        output_dir = mha_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename
    nifti_filename = mha_path.stem + ".nii.gz"
    nifti_path = output_dir / nifti_filename
    
    try:
        # Read MHA image
        print(f"Reading: {mha_path}")
        image = sitk.ReadImage(str(mha_path))
        
        # Write as NIfTI
        print(f"Writing: {nifti_path}")
        sitk.WriteImage(image, str(nifti_path))
        
        # Verify file was created
        if nifti_path.exists():
            print(f" Successfully converted: {mha_path.name} -> {nifti_filename}")
            return str(nifti_path)
        else:
            raise RuntimeError("Output file was not created")
            
    except Exception as e:
        print(f" Failed to convert {mha_path.name}: {e}")
        return None

def batch_convert_mha_to_nifti(input_dir, output_dir=None):
    """
    Convert all .mha files in a directory to .nii.gz format
    
    Args:
        input_dir (str/Path): Directory containing .mha files
        output_dir (str/Path): Output directory (default: same as input)
    
    Returns:
        list: List of successfully converted files
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    
    # Find all .mha files
    mha_files = list(input_path.glob("*.mha"))
    
    if not mha_files:
        print(f"No .mha files found in: {input_path}")
        return []
    
    print(f"Found {len(mha_files)} .mha files to convert")
    
    # Convert each file
    converted_files = []
    for mha_file in mha_files:
        result = convert_mha_to_nifti(mha_file, output_dir)
        if result:
            converted_files.append(result)
    
    print(f"\n Conversion complete!")
    print(f"Successfully converted: {len(converted_files)}/{len(mha_files)} files")
    
    return converted_files

def convert_nested_folders(root_dir, output_root_dir):
    """
    Convert .mha files in nested folder structure to separate output directory
    Each case folder contains: t2w, adc, hbv, mask .mha files
    """
    root_path = Path(root_dir)
    output_root_path = Path(output_root_dir)
    
    if not root_path.exists():
        print(f" Directory not found: {root_path}")
        return
    
    # Create output root directory
    output_root_path.mkdir(parents=True, exist_ok=True)
    print(f" Output directory: {output_root_path}")
    
    # Find all case folders
    case_folders = [d for d in root_path.iterdir() if d.is_dir()]
    
    if not case_folders:
        print(f"No case folders found in: {root_path}")
        return
    
    print(f"Found {len(case_folders)} case folders to process")
    
    total_converted = 0
    total_files = 0
    
    # Process each case folder
    for case_folder in sorted(case_folders):
        print(f"\n Processing case: {case_folder.name}")
        
        # Create corresponding output directory for this case
        case_output_dir = output_root_path / case_folder.name
        case_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find .mha files in this case folder
        mha_files = list(case_folder.glob("*.mha"))
        
        if not mha_files:
            print(f"    No .mha files found in {case_folder.name}")
            continue
        
        print(f"  Found {len(mha_files)} .mha files")
        total_files += len(mha_files)
        
        # Convert each .mha file (output to separate directory)
        case_converted = 0
        for mha_file in mha_files:
            result = convert_mha_to_nifti(mha_file, case_output_dir)
            if result:
                case_converted += 1
                total_converted += 1
        
        print(f"  Converted {case_converted}/{len(mha_files)} files in {case_folder.name}")
    
    print(f"\n Batch conversion complete!")
    print(f"Successfully converted: {total_converted}/{total_files} files across {len(case_folders)} cases")
    print(f" All files saved to: {output_root_path}")

def main():
    """
    Main function - converts .mha files in nested folder structure to separate output
    """
    # Default input and output directories
    default_input_dir = "/Users/anish/from_scratch_G/images"
    default_output_dir = "/Users/anish/from_scratch_G/images_nifti"
    
    print(" MHA to NIfTI Nested Folder Converter")
    print(f"Input:  {default_input_dir}")
    print(f"Output: {default_output_dir}")
    
    # Perform nested folder conversion
    try:
        convert_nested_folders(default_input_dir, default_output_dir)
        
    except Exception as e:
        print(f" Conversion failed: {e}")

if __name__ == "__main__":
    # Check if SimpleITK is installed
    try:
        import SimpleITK as sitk
    except ImportError:
        print(" SimpleITK not found. Please install with: pip install SimpleITK")
        sys.exit(1)
    
    # Run conversion
    main()