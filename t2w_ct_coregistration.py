#!/usr/bin/env python3
"""
T2W MRI to CT Coregistration with PSMA Landmark Guidance
Registers T2W MRI to CT using pre-aligned PET/PSMA data for landmark detection.

Directory Structure:
- CT/PET: ct_pet_directory/case_id/ct/ and ct_pet_directory/case_id/pet/
- MRI: mri_directory/folder/subfolder/subfolder/t2w/

Prerequisites:
- CT and PET/PSMA are already co-registered
- DICOM data converted to NIfTI format
"""

import os
import subprocess
import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from scipy.ndimage import label, center_of_mass, gaussian_filter

class PSMALandmarkDetector:
    """Detect PSMA hotspots as landmarks for registration guidance"""
    
    def __init__(self, threshold_percentile=95, min_hotspot_size=5, max_landmarks=8):
        self.threshold_percentile = threshold_percentile
        self.min_hotspot_size = min_hotspot_size
        self.max_landmarks = max_landmarks
    
    def detect_psma_landmarks_in_ct_space(self, psma_aligned_path):
        """
        Extract PSMA hotspots as landmarks - already in CT coordinate space
        """
        print(f"    🎯 Detecting PSMA landmarks...")
        
        try:
            # Load PSMA (already aligned to CT)
            psma_img = sitk.ReadImage(psma_aligned_path)
            psma_array = sitk.GetArrayFromImage(psma_img)
            
            # Remove background noise
            valid_mask = psma_array > 0
            if not np.any(valid_mask):
                print(f"    ⚠️ No valid PSMA data found")
                return []
            
            # Detect PSMA hotspots using percentile threshold
            threshold = np.percentile(psma_array[valid_mask], self.threshold_percentile)
            hotspot_mask = psma_array > threshold
            
            # Find connected components
            labeled_hotspots, num_hotspots = label(hotspot_mask)
            
            if num_hotspots == 0:
                print(f"    ⚠️ No PSMA hotspots found at {self.threshold_percentile}th percentile")
                return []
            
            # Extract landmark coordinates (already in CT space!)
            ct_landmarks = []
            hotspot_sizes = []
            
            for i in range(1, num_hotspots + 1):
                hotspot_mask_i = labeled_hotspots == i
                hotspot_size = np.sum(hotspot_mask_i)
                
                if hotspot_size >= self.min_hotspot_size:
                    # Get centroid in voxel coordinates
                    centroid_voxel = center_of_mass(hotspot_mask_i)
                    
                    try:
                        # Convert to physical coordinates (CT space)
                        centroid_physical = psma_img.TransformIndexToPhysicalPoint(
                            [int(c) for c in reversed(centroid_voxel)]
                        )
                        ct_landmarks.append(centroid_physical)
                        hotspot_sizes.append(hotspot_size)
                        
                    except Exception as e:
                        print(f"    ⚠️ Failed to convert landmark {i}: {e}")
                        continue
            
            # Sort by hotspot size (largest first) and limit number
            if ct_landmarks:
                sorted_indices = np.argsort(hotspot_sizes)[::-1]
                ct_landmarks = [ct_landmarks[i] for i in sorted_indices[:self.max_landmarks]]
            
            print(f"    ✅ Found {len(ct_landmarks)} PSMA landmarks in CT space")
            return ct_landmarks
            
        except Exception as e:
            print(f"    ❌ Error detecting PSMA landmarks: {e}")
            return []

class T2WCTCoregistration:
    """Main class for T2W MRI to CT coregistration using PSMA landmarks"""
    
    def __init__(self, elastix_path, parameter_file, output_dir):
        self.elastix_path = Path(elastix_path)
        self.parameter_file = Path(parameter_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.psma_detector = PSMALandmarkDetector()
        
        # Validate paths
        if not self.elastix_path.exists():
            raise FileNotFoundError(f"Elastix not found: {self.elastix_path}")
        if not self.parameter_file.exists():
            raise FileNotFoundError(f"Parameter file not found: {self.parameter_file}")
    
    def register_t2w_to_ct_with_psma(self, ct_fixed_path, t2w_moving_path, 
                                    psma_aligned_path, case_id):
        """
        Register T2W MRI to CT using PSMA landmarks for guidance
        """
        print(f"  🎯 Registering T2W to CT using PSMA landmarks for {case_id}")
        
        # Step 1: Extract PSMA landmarks directly in CT coordinate space
        ct_landmarks = self.psma_detector.detect_psma_landmarks_in_ct_space(psma_aligned_path)
        
        if len(ct_landmarks) < 3:
            print(f"    ❌ Insufficient PSMA landmarks ({len(ct_landmarks)}) for registration")
            return False, None
        
        # Step 2: Find corresponding landmarks in T2W MRI space
        t2w_landmarks = self.find_corresponding_t2w_landmarks(
            t2w_moving_path, ct_landmarks
        )
        
        if len(t2w_landmarks) != len(ct_landmarks):
            print(f"    ❌ Landmark correspondence mismatch: CT={len(ct_landmarks)}, T2W={len(t2w_landmarks)}")
            return False, None
        
        # Step 3: Perform landmark-guided registration
        success, reg_output_dir = self.perform_t2w_ct_registration(
            fixed_image=ct_fixed_path,
            moving_image=t2w_moving_path,
            fixed_landmarks=ct_landmarks,
            moving_landmarks=t2w_landmarks,
            case_id=case_id
        )
        
        return success, reg_output_dir
    
    def find_corresponding_t2w_landmarks(self, t2w_path, ct_landmarks):
        """
        Find corresponding anatomical landmarks in T2W MRI
        """
        print(f"    🔍 Finding corresponding T2W landmarks...")
        
        try:
            t2w_img = sitk.ReadImage(t2w_path)
            t2w_array = sitk.GetArrayFromImage(t2w_img)
            
            t2w_landmarks = []
            
            for i, ct_landmark in enumerate(ct_landmarks):
                try:
                    # Convert CT physical coordinate to T2W voxel space
                    t2w_voxel = t2w_img.TransformPhysicalPointToIndex(ct_landmark)
                    
                    # Refine landmark location by finding nearest high-contrast feature
                    refined_t2w_landmark = self.refine_t2w_landmark_location(
                        t2w_array, t2w_img, t2w_voxel, search_radius=15
                    )
                    
                    t2w_landmarks.append(refined_t2w_landmark)
                    
                except Exception as e:
                    print(f"    ⚠️ Failed to find T2W correspondence for landmark {i+1}: {e}")
                    # Use approximate coordinate transformation as fallback
                    try:
                        fallback_landmark = self.approximate_t2w_coordinate(ct_landmark, t2w_img)
                        t2w_landmarks.append(fallback_landmark)
                    except:
                        continue
            
            print(f"    ✅ Found {len(t2w_landmarks)} corresponding T2W landmarks")
            return t2w_landmarks
            
        except Exception as e:
            print(f"    ❌ Error finding T2W landmarks: {e}")
            return []
    
    def refine_t2w_landmark_location(self, t2w_array, t2w_img, initial_voxel, search_radius=15):
        """
        Refine landmark location by finding nearest high-contrast feature in T2W
        """
        z, y, x = initial_voxel
        
        # Define search region
        z_start = max(0, z - search_radius)
        z_end = min(t2w_array.shape[0], z + search_radius)
        y_start = max(0, y - search_radius)
        y_end = min(t2w_array.shape[1], y + search_radius)
        x_start = max(0, x - search_radius)
        x_end = min(t2w_array.shape[2], x + search_radius)
        
        search_region = t2w_array[z_start:z_end, y_start:y_end, x_start:x_end]
        
        if search_region.size > 0:
            # Smooth to reduce noise and find local maxima
            smoothed = gaussian_filter(search_region, sigma=1.5)
            
            # Find maximum intensity location (likely tissue/lesion)
            max_location = np.unravel_index(np.argmax(smoothed), smoothed.shape)
            
            # Convert back to full image coordinates
            refined_voxel = (
                z_start + max_location[0],
                y_start + max_location[1],
                x_start + max_location[2]
            )
        else:
            refined_voxel = initial_voxel
        
        # Convert to physical coordinates
        refined_physical = t2w_img.TransformIndexToPhysicalPoint(
            [int(c) for c in reversed(refined_voxel)]
        )
        
        return refined_physical
    
    def approximate_t2w_coordinate(self, ct_landmark, t2w_img):
        """
        Simple spatial approximation for landmark coordinate
        """
        # Use center of image as reference
        t2w_center = t2w_img.TransformIndexToPhysicalPoint(
            [int(s//2) for s in reversed(t2w_img.GetSize())]
        )
        
        # Simple offset-based approximation
        approx_landmark = [
            ct_landmark[0] + (t2w_center[0] - ct_landmark[0]) * 0.1,
            ct_landmark[1] + (t2w_center[1] - ct_landmark[1]) * 0.1,
            ct_landmark[2] + (t2w_center[2] - ct_landmark[2]) * 0.1
        ]
        
        return approx_landmark
    
    def perform_t2w_ct_registration(self, fixed_image, moving_image, 
                                   fixed_landmarks, moving_landmarks, case_id):
        """
        Perform T2W to CT registration using Elastix with landmark guidance
        """
        print(f"    🔧 Running Elastix registration...")
        
        try:
            # Create landmark files
            fixed_pts_file = self.create_landmark_file(
                fixed_landmarks, f"{case_id}_ct_landmarks.txt"
            )
            moving_pts_file = self.create_landmark_file(
                moving_landmarks, f"{case_id}_t2w_landmarks.txt"
            )
            
            # Create case-specific parameter file with landmark paths
            case_param_file = self.create_case_parameter_file(
                fixed_pts_file, moving_pts_file, case_id
            )
            
            # Create output directory
            reg_output_dir = self.output_dir / f"{case_id}_t2w_to_ct_registration"
            reg_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run Elastix registration
            registration_cmd = [
                str(self.elastix_path / 'elastix'),
                '-f', str(fixed_image),        # CT (fixed)
                '-m', str(moving_image),       # T2W (moving)
                '-out', str(reg_output_dir),
                '-p', str(case_param_file)
            ]
            
            print(f"    📝 Running: elastix -f CT -m T2W -out {reg_output_dir.name}")
            
            result = subprocess.run(registration_cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                print(f"    ✅ Registration successful")
                return True, reg_output_dir
            else:
                print(f"    ❌ Registration failed:")
                print(f"    Error: {result.stderr[:200]}...")
                return False, None
                
        except subprocess.TimeoutExpired:
            print(f"    ❌ Registration timed out after 30 minutes")
            return False, None
        except Exception as e:
            print(f"    ❌ Registration error: {e}")
            return False, None
    
    def create_landmark_file(self, landmarks, filename):
        """
        Create Elastix-compatible landmark file
        """
        landmark_path = self.output_dir / filename
        
        with open(landmark_path, 'w') as f:
            f.write("index\n")
            f.write(f"{len(landmarks)}\n")
            for i, (x, y, z) in enumerate(landmarks):
                f.write(f"{i} {x:.6f} {y:.6f} {z:.6f}\n")
        
        print(f"    💾 Saved {len(landmarks)} landmarks to {filename}")
        return str(landmark_path)
    
    def create_case_parameter_file(self, fixed_landmarks_file, moving_landmarks_file, case_id):
        """
        Create case-specific parameter file with landmark paths
        """
        # Read base parameter file
        with open(self.parameter_file, 'r') as f:
            param_content = f.read()
        
        # Add landmark file paths
        landmark_line = f'(CorrespondingPointsEuclideanDistancePointSetFileName "{fixed_landmarks_file}" "{moving_landmarks_file}")\n'
        param_content += '\n' + landmark_line
        
        # Write case-specific parameter file
        case_param_file = self.output_dir / f"{case_id}_t2w_ct_params.txt"
        with open(case_param_file, 'w') as f:
            f.write(param_content)
        
        return case_param_file

def find_first_nifti(directory):
    """Helper function to find first NIfTI file in directory"""
    if not directory or not os.path.exists(directory):
        return None
    
    nifti_files = list(Path(directory).glob("*.nii.gz"))
    if not nifti_files:
        nifti_files = list(Path(directory).glob("*.nii"))
    
    return str(nifti_files[0]) if nifti_files else None

def find_mri_cases(mri_directory):
    """
    Find all T2W MRI cases in nested folder structure
    Returns dict: {case_id: t2w_file_path}
    """
    mri_cases = {}
    mri_root = Path(mri_directory)
    
    if not mri_root.exists():
        print(f"❌ MRI directory not found: {mri_directory}")
        return mri_cases
    
    # Search for t2w folders in nested structure
    t2w_folders = list(mri_root.rglob("t2w"))
    
    print(f"🔍 Found {len(t2w_folders)} T2W folders in MRI directory")
    
    for t2w_folder in t2w_folders:
        # Find NIfTI file in t2w folder
        t2w_file = find_first_nifti(t2w_folder)
        if t2w_file:
            # Extract case ID from path structure
            # Assuming structure: .../folder/subfolder/subfolder/t2w/
            parent_path = t2w_folder.parent
            case_id = parent_path.name  # Use immediate parent as case ID
            
            mri_cases[case_id] = t2w_file
            print(f"  📁 Found T2W for case {case_id}: {t2w_file}")
    
    return mri_cases

def find_ct_pet_cases(ct_pet_directory):
    """
    Find all CT/PET case pairs
    Returns dict: {case_id: {'ct': ct_file, 'pet': pet_file}}
    """
    ct_pet_cases = {}
    ct_pet_root = Path(ct_pet_directory)
    
    if not ct_pet_root.exists():
        print(f"❌ CT/PET directory not found: {ct_pet_directory}")
        return ct_pet_cases
    
    # Find all case directories
    case_dirs = [d for d in ct_pet_root.iterdir() if d.is_dir()]
    
    print(f"🔍 Found {len(case_dirs)} case directories in CT/PET directory")
    
    for case_dir in case_dirs:
        case_id = case_dir.name
        ct_dir = case_dir / "ct"
        pet_dir = case_dir / "pet"
        
        # Find CT and PET files
        ct_file = find_first_nifti(ct_dir) if ct_dir.exists() else None
        pet_file = find_first_nifti(pet_dir) if pet_dir.exists() else None
        
        if ct_file and pet_file:
            ct_pet_cases[case_id] = {
                'ct': ct_file,
                'pet': pet_file
            }
            print(f"  📁 Found CT/PET for case {case_id}")
        else:
            print(f"  ⚠️ Incomplete data for case {case_id} (CT: {ct_file is not None}, PET: {pet_file is not None})")
    
    return ct_pet_cases

def run_t2w_ct_coregistration(ct_pet_directory, mri_directory, output_directory, 
                             elastix_path, parameter_file):
    """
    Main function for T2W to CT coregistration pipeline
    """
    # Set environment
    os.environ['DYLD_LIBRARY_PATH'] = '/Users/anish/elastix-5/lib'
    
    print("🏥 T2W MRI to CT Coregistration with PSMA Landmark Guidance")
    print("=" * 65)
    print("ℹ️  Prerequisites: CT and PET/PSMA are already co-registered")
    print(f"📂 CT/PET directory: {ct_pet_directory}")
    print(f"📂 MRI directory: {mri_directory}")
    print(f"📂 Output directory: {output_directory}")
    print(f"⚙️  Parameter file: {parameter_file}")
    
    # Initialize registration
    try:
        t2w_ct_reg = T2WCTCoregistration(elastix_path, parameter_file, output_directory)
    except Exception as e:
        print(f"❌ Failed to initialize registration: {e}")
        return
    
    # Find all cases
    print(f"\n🔍 Scanning directories...")
    ct_pet_cases = find_ct_pet_cases(ct_pet_directory)
    mri_cases = find_mri_cases(mri_directory)
    
    if not ct_pet_cases:
        print(f"❌ No CT/PET cases found")
        return
    
    if not mri_cases:
        print(f"❌ No MRI cases found")
        return
    
    # Find matching cases between CT/PET and MRI
    matching_cases = set(ct_pet_cases.keys()) & set(mri_cases.keys())
    
    if not matching_cases:
        print(f"❌ No matching cases found between CT/PET and MRI directories")
        print(f"CT/PET cases: {list(ct_pet_cases.keys())[:5]}...")
        print(f"MRI cases: {list(mri_cases.keys())[:5]}...")
        return
    
    print(f"\n✅ Found {len(matching_cases)} matching cases")
    
    successful = 0
    failed = 0
    
    for i, case_id in enumerate(sorted(matching_cases), 1):
        print(f"\n[{i}/{len(matching_cases)}] Processing case: {case_id}")
        
        try:
            # Get file paths
            ct_file = ct_pet_cases[case_id]['ct']
            pet_file = ct_pet_cases[case_id]['pet']
            t2w_file = mri_cases[case_id]
            
            print(f"  📁 Files:")
            print(f"     CT (fixed): {Path(ct_file).name}")
            print(f"     PET (landmarks): {Path(pet_file).name}")
            print(f"     T2W (moving): {Path(t2w_file).name}")
            
            # Register T2W MRI to CT using PSMA landmarks
            t2w_success, t2w_output = t2w_ct_reg.register_t2w_to_ct_with_psma(
                ct_fixed_path=ct_file,
                t2w_moving_path=t2w_file,
                psma_aligned_path=pet_file,
                case_id=case_id
            )
            
            if t2w_success:
                print(f"  ✅ T2W registration successful for {case_id}")
                successful += 1
            else:
                print(f"  ❌ T2W registration failed for {case_id}")
                failed += 1
                
        except Exception as e:
            print(f"  ❌ Error processing {case_id}: {e}")
            failed += 1
    
    # Final summary
    print(f"\n🎉 T2W to CT Coregistration Complete!")
    print(f"✅ Successful: {successful}/{len(matching_cases)} cases")
    print(f"❌ Failed: {failed}/{len(matching_cases)} cases")
    print(f"📂 Results saved in: {output_directory}")
    
    if successful > 0:
        print(f"\n📋 Output structure:")
        print(f"   case_id_t2w_to_ct_registration/")
        print(f"   ├── result.0.nii.gz              # T2W registered to CT space")
        print(f"   ├── TransformParameters.0.txt    # Transformation parameters")
        print(f"   └── elastix.log                  # Registration log")
        
        print(f"\n📋 Next steps:")
        print(f"1. Quality check: Review T2W-CT alignment")
        print(f"2. Apply transforms: Use TransformParameters.0.txt for ADC/HBV")
        print(f"3. Intensity harmonization: Proceed with registered data")

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description='T2W MRI to CT Coregistration with PSMA Landmark Guidance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python t2w_ct_coregistration.py \\
    --ct-pet-dir /path/to/ct_pet_data \\
    --mri-dir /path/to/mri_data \\
    --output-dir /path/to/output \\
    --elastix-path /Users/anish/elastix-5/bin \\
    --parameter-file /Users/anish/from_scratch_G/preprocessing/co_registration/parameters/ct_mri_landmark_rigid.txt
        """
    )
    
    parser.add_argument('--ct-pet-dir', required=True,
                       help='Directory containing CT/PET data (structure: case_id/ct/ and case_id/pet/)')
    parser.add_argument('--mri-dir', required=True,
                       help='Directory containing MRI data (structure: .../folder/subfolder/subfolder/t2w/)')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for registration results')
    parser.add_argument('--elastix-path', default='/Users/anish/elastix-5/bin',
                       help='Path to Elastix binary directory')
    parser.add_argument('--parameter-file', 
                       default='/Users/anish/from_scratch_G/preprocessing/co_registration/parameters/ct_mri_landmark_rigid.txt',
                       help='Elastix parameter file for CT-MRI registration')
    
    args = parser.parse_args()
    
    # Run registration
    run_t2w_ct_coregistration(
        ct_pet_directory=args.ct_pet_dir,
        mri_directory=args.mri_dir,
        output_directory=args.output_dir,
        elastix_path=args.elastix_path,
        parameter_file=args.parameter_file
    )

if __name__ == "__main__":
    main()