import os
import SimpleITK as sitk
import pydicom
import re
from typing import Dict, List, Tuple, Optional


class MultiModalDicom2Nifti:
    """
    Multi-modal DICOM to NIFTI converter based on existing dicom2nifti code.
    Supports MRI (T2W, ADC, HBV), CT-PET, PSMA with standardized naming.
    """
    
    def __init__(self):
        # Define sequence patterns for different modalities
        self.sequence_patterns = {
            # MRI sequences
            'T2W': [
                # Exact filename matches (case-insensitive)
                r'^[aA][xX]_[tT]2_[pP][rR][oO][pP][eE][lL][lL][eE][rR]\.nii\.gz$',
                r'^[aA][xX]_[dD][yY][nN][aA][cC][aA][dD]_[tT]2\.nii\.gz$',
                r'^[aA][xX]_[tT]2\.nii\.gz$',
                r'^[aA][xX]_[tT]2_[tT][sS][eE]\.nii\.gz$',
                r'^[aA][xX]_[tT]2_[cC][uU][bB][eE]--[pP][eE][lL][vV][iI][sS]\.nii\.gz$',
                r'^[aA][xX]_[tT]2_[pP][rR][oO][pP][eE][lL][lL][eE][rR]\.nii\.gz$',
                r'^[aA][xX]_[dD][yY][nN][aA][cC][aA][dD]_[tT]2\.nii\.gz$',
                r'^[aA][xX]_[tT]2\.nii\.gz$',
                r'^[aA][xX]_[tT]2_[tT][sS][eE]\.nii\.gz$',
                r'^[aA][xX]_[tT]2_[cC][uU][bB][eE]--[pP][eE][lL][vV][iI][sS]\.nii\.gz$',
                # New T2W patterns
                r'^AX_T2_\(3_SKIP_0\)\.nii\.gz$',
                r'^AX_T2_3D_SPACE_\+sag_&_cor_mpr\.nii\.gz$',
                r'^t2_tra\.nii\.gz$',
                r'^T2_Ax_FSE_PROP\.nii\.gz$',
                # Generic patterns for new sequences
                r'ax_t2_\(3_skip_0\)',
                r'ax_t2_3d_space_\+sag_&_cor_mpr',
                r't2_tra',
                r't2_ax_fse_prop'
                
            ],
            'ADC': [
                # Original exact filename matches (case-insensitive) - ONLY 1400 b-value
                r'^ax_dynacad_dwi_adc\.nii\.gz$',
                r'^ax_dynacad_dwladc_dfc_mix\.nii\.gz$',
                r'^ax_dynacad_dwi_1400_b_value_adc\.nii\.gz$',
                r'^ax_dynacad_dwi_adc_dfc\.nii\.gz$',
                r'^ax_dynacad_dwi_adc_dfc_mix\.nii\.gz$',
                r'^apparent_diffusion_coefficient_\(mm2_s\)\.nii\.gz$',
                r'^dadc_map\.nii\.gz$',
                r'^ax_dwi_b100_800_1400_adc_dfc_mix\.nii\.gz$',
                r'^ax_dwi_b100,_800,_1400_adc_dfc_mix\.nii\.gz$',
                r'^adc_\(10\^-6_mm²_s\):dec_21_2021_09-15-46_est\.nii\.gz$',
                r'^adc_\(10\^-6_mm²_s\):feb_19_2022_11-42-24_cst\.nii\.gz$',
                r'^adc_\(10\^-6_mm²_s\):Dec_21_2021_09-15-46_EST\.nii\.gz$',
                r'^adc_\(10\^-6_mm²_s\):Feb_19_2022_11-42-24_CST\.nii\.gz$',
                r'^adc_\(10\^-6_mm²_s\):feb_19_2022_11-40-05_cst\.nii\.gz$',
                r'^diff_b50_100_1000_1200_adc_dfc\.nii\.gz$',
                # New ADC patterns
                r'^dDWI_2000_ADC\.nii\.gz$',
                r'^ADC_map\.nii\.gz$',
                r'^DIFF_b50_100_1000_1200_ADC_DFC\.nii\.gz$',
                r'^AX_DIFF_WHOLE_PELVIS_ADC_DFC_MIX\.nii\.gz$',
                r'^ep2d_diff_b50_800_tra_ADC\.nii\.gz$',
                r'^ep2d_diff_b50_500_1000_tra_p2_ADC_DFC_MIX\.nii\.gz$',
                # Specific generic patterns (more restrictive) - ONLY 1400 b-value
                r'ax_dynacad_dwi_adc', 
                r'ax_dynacad_dwi_1400_b_value_adc',
                r'apparent_diffusion_coefficient_\(mm2_s\)',
                r'dadc_map', 
                r'ax_dwi_b100_800_1400_adc_dfc_mix',
                r'ax_dwi_b100,_800,_1400_adc_dfc_mix',
                r'adc_\(10\^-6_mm²_s\):',
                # New generic patterns
                r'ddwi_2000_adc',
                r'adc_map',
                r'diff_b50_100_1000_1200_adc_dfc',
                r'ax_diff_whole_pelvis_adc_dfc_mix',
                r'ep2d_diff_b50_800_tra_adc',
                r'ep2d_diff_b50_500_1000_tra_p2_adc_dfc_mix'
            ],
            'HBV': [
                # Original patterns
                r'^ax_dynacad_dwi_1400_b_value_tracew\.nii\.gz$',
                r'^ax_dynacad_dwi_1400_b_value_tracew_dfc_mix\.nii\.gz$',
                r'^ax_edwi_3_and_1\.nii\.gz$',
                r'^ax_dwi_\(b1400_nsa_20\)\.nii\.gz$',
                r'^ax_dwi_b100_800_1400_tracew_dfc_mix\.nii\.gz$',
                r'^ax_dwi_b100,_800,_1400_tracew_dfc_mix\.nii\.gz$',
                r'^ax_dwi_3_b_values\.nii\.gz$',
                r'^ax_dwi_focus_pros_b1400\.nii\.gz$',
                # New HBV patterns
                r'^sDWI_B2000\.nii\.gz$',
                r'^SB1500\.nii\.gz$',
                r'^DIFF_b50_100_1000_1200_TRACEW_DFC\.nii\.gz$',
                r'^AX_DIFF_WHOLE_PELVIS_TRACEW_DFC_MIX\.nii\.gz$',
                r'^ep2d_diff_b50_800_tra_TRACEW\.nii\.gz$',
                r'^ep2d_diff_b50_500_1000_tra_p2_TRACEW_DFC_MIX\.nii\.gz$',
                r'^DWI_Ax_b-2000\.nii\.gz$',
                # Missing patterns for incomplete cases
                r'^AX_DWI_Focus\.nii\.gz$',
                r'^Ax_DWI_FOCUS_PROS_B1400\.nii\.gz$',
                # Generic patterns for new sequences
                r'sdwi_b2000',
                r'sb1500',
                r'diff_b50_100_1000_1200_tracew_dfc',
                r'ax_diff_whole_pelvis_tracew_dfc_mix',
                r'ep2d_diff_b50_800_tra_tracew',
                r'ep2d_diff_b50_500_1000_tra_p2_tracew_dfc_mix',
                r'dwi_ax_b-2000',
                # Generic patterns for missing cases
                r'ax_dwi_focus',
                r'ax_dwi_focus_pros_b1400'
            ],
            'DWI': [
                r'dwi', r'.*diffusion.*weighted', r'.*dwi.*',
                r'b\d+', r'.*b-value.*', r'.*tracew.*', r'.*dwi.*tracew.*'
            ],
            # Additional MRI sequences found in test
            'T1W': [
                r't1.*vibe', r'axial.*t1.*vibe', r'.*t1.*weighted',
                r't1w', r'.*t1_.*'
            ],
            'DYNAMIC': [
                r'dynacad.*vibe.*\+c', r'.*dynamic.*', r'.*\+c.*',
                r'.*contrast.*enhanced', r'.*post.*contrast'
            ],
            # CT-PET sequences
            'CT': [
                r'ct', r'.*ct.*', r'ctac', r'attenuation.*correction',
                r'computed.*tomography',
                # PSMA-specific CT patterns
                r'.*tb_abdomen.*', r'.*abdomen.*', r'.*monoe.*', 
                r'.*iodine.*', r'.*pp.*tb.*', r'.*hd_fov.*'
            ],
            'PET': [
                r'pet', r'.*pet.*', r'suv', r'.*suv.*', r'emission',
                r'positron.*emission',
                # PSMA-specific PET patterns
                r'static.*min', r'.*static.*', r'.*\d+-\d+.*min.*'
            ],
            # PSMA sequences
            'PSMA': [
                r'psma', r'.*psma.*', r'ga68', r'.*ga-68.*',
                r'lu177', r'.*lu-177.*', r'.*gallium.*',
                # Additional PSMA PET patterns
                r'static.*min', r'.*static.*', r'.*\d+-\d+.*min.*'
            ],
            'POST_CONTRAST': [
                r'post.*contrast', r'.*post.*', r'contrast.*enhanced',
                r'.*gd.*', r'gadolinium'
            ]
        }
        
        # Modality detection patterns from DICOM headers
        self.modality_patterns = {
            'MR': ['MR', 'MRI'],
            'CT': ['CT'],
            'PT': ['PT', 'PET'],
            'NM': ['NM', 'NUCLEAR']
        }
    
    def detect_sequence_type(self, series_description: str, protocol_name: str = '', 
                           modality: str = '') -> str:
        """
        Detect sequence type based on series description and protocol name.
        
        Args:
            series_description: SeriesDescription from DICOM header
            protocol_name: ProtocolName from DICOM header  
            modality: Modality from DICOM header
            
        Returns:
            Detected sequence type or 'UNKNOWN'
        """
        # Combine description and protocol for matching
        combined_text = f"{series_description} {protocol_name}".lower()
        
        # For PSMA datasets, use modality-aware detection
        if modality == 'CT':
            # Check CT-specific patterns first
            for pattern in self.sequence_patterns['CT']:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    return 'CT'
        elif modality == 'PT':
            # Check PET-specific patterns first
            for pattern in self.sequence_patterns['PET']:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    return 'PET'
            # Also check PSMA patterns for PT modality
            for pattern in self.sequence_patterns['PSMA']:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    return 'PSMA'
        
        # Fall back to general pattern matching
        for seq_type, patterns in self.sequence_patterns.items():
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    return seq_type
        
        return 'UNKNOWN'
    
    def get_dicom_metadata(self, file_path: str) -> Dict[str, str]:
        """
        Extract relevant metadata from DICOM file.
        
        Args:
            file_path: Path to DICOM file
            
        Returns:
            Dictionary with metadata
        """
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            
            metadata = {
                'SeriesDescription': ds.get('SeriesDescription', ''),
                'ProtocolName': ds.get('ProtocolName', ''),
                'Modality': ds.get('Modality', ''),
                'SeriesNumber': ds.get('SeriesNumber', ''),
                'StudyDescription': ds.get('StudyDescription', ''),
                'ManufacturerModelName': ds.get('ManufacturerModelName', ''),
                'SliceThickness': ds.get('SliceThickness', ''),
                'RepetitionTime': ds.get('RepetitionTime', ''),
                'EchoTime': ds.get('EchoTime', ''),
                'MagneticFieldStrength': ds.get('MagneticFieldStrength', '')
            }
            
            return metadata
            
        except Exception as e:
            print(f"❌ Error reading DICOM metadata from {file_path}: {e}")
            return {}
    
    def generate_output_filename(self, case_id: str, sequence_type: str, 
                                series_number: str = '', metadata: Dict = None,
                                use_original_naming: bool = True) -> str:
        """
        Generate output filename - either original DICOM naming or standardized naming.
        
        Args:
            case_id: Patient/case identifier
            sequence_type: Detected sequence type
            series_number: Series number from DICOM
            metadata: Additional metadata for filename generation
            use_original_naming: If True, use original SeriesDescription; if False, use standardized naming
            
        Returns:
            Generated filename
        """
        if use_original_naming and metadata and metadata.get('SeriesDescription'):
            # Use original SeriesDescription similar to IUSM_MRI_nifti format
            series_desc = metadata.get('SeriesDescription', '').strip()
            # Clean up the description for filename (remove invalid characters)
            clean_desc = re.sub(r'[<>:"/\\|?*]', '_', series_desc)
            clean_desc = re.sub(r'\s+', '_', clean_desc)  # Replace spaces with underscores
            clean_desc = clean_desc.upper()  # Match IUSM format (uppercase)
            return f"{clean_desc}.nii.gz"
        else:
            # Use standardized naming (original behavior)
            base_name = f"{case_id}_{sequence_type}"
            
            # Add series number if available and needed for disambiguation
            if series_number:
                base_name += f"_S{series_number}"
            
            # Add specific suffixes for certain sequences
            if sequence_type == 'DWI' and metadata:
                # Try to extract b-value from description
                desc = metadata.get('SeriesDescription', '').lower()
                b_value_match = re.search(r'b(\d+)', desc)
                if b_value_match:
                    base_name += f"_b{b_value_match.group(1)}"
            
            return f"{base_name}.nii.gz"
    
    def find_dicom_directories(self, root_path: str) -> List[str]:
        """
        Recursively find all directories containing DICOM files.
        
        Args:
            root_path: Root directory to search
            
        Returns:
            List of directories containing DICOM files
        """
        dicom_dirs = []
        
        for root, dirs, files in os.walk(root_path):
            # Check if directory contains DICOM files
            has_dicom = False
            for file in files:
                if file.lower().endswith(('.dcm', '.dicom')) or self.is_dicom_file(os.path.join(root, file)):
                    has_dicom = True
                    break
            
            # Also check with SimpleITK if no obvious DICOM extensions
            if not has_dicom:
                try:
                    reader = sitk.ImageSeriesReader()
                    series_ids = reader.GetGDCMSeriesIDs(root)
                    if series_ids:
                        has_dicom = True
                except:
                    pass
            
            if has_dicom:
                dicom_dirs.append(root)
        
        return dicom_dirs
    
    def is_dicom_file(self, file_path: str) -> bool:
        """
        Check if a file is a DICOM file by reading its header.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file is DICOM, False otherwise
        """
        try:
            pydicom.dcmread(file_path, stop_before_pixels=True)
            return True
        except:
            return False

    def convert_case(self, case_path: str, output_dir: str, case_id: str = None, 
                     use_original_naming: bool = True) -> List[str]:
        """
        Convert all DICOM series in a case folder (including subdirectories) to NIFTI.
        
        Args:
            case_path: Path to case folder containing DICOM files (may have subdirectories)
            output_dir: Output directory for NIFTI files
            case_id: Case identifier (if None, uses folder name)
            use_original_naming: If True, use original SeriesDescription naming (IUSM format)
            
        Returns:
            List of successfully converted files
        """
        if case_id is None:
            case_id = os.path.basename(case_path)
        
        print(f"🔄 Processing case: {case_id}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all directories with DICOM files (including subdirectories)
        dicom_dirs = self.find_dicom_directories(case_path)
        
        if not dicom_dirs:
            # Fallback: case directory already contains NIfTI files (IUSM dataset)
            nii_files = []
            for root, _dirs, files in os.walk(case_path):
                for fn in files:
                    if fn.lower().endswith(('.nii', '.nii.gz')):
                        nii_files.append(os.path.join(root, fn))

            if nii_files:
                print(f"ℹ️ Detected pre-converted NIfTI files: {len(nii_files)}")
                # Simply return the list – they will be copied/filtered later
                return nii_files

            print(f"⚠️ No DICOM or NIfTI files found in {case_id}")
            return []
        
        print(f"📁 Found DICOM files in {len(dicom_dirs)} directories")
        
        converted_files = []
        sequence_counts = {}  # Track multiple instances of same sequence type
        
        # Process each directory containing DICOM files
        for dicom_dir in dicom_dirs:
            print(f"  📂 Processing subdirectory: {os.path.relpath(dicom_dir, case_path)}")
            
            # Initialize SimpleITK reader for this directory
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
            
            if not series_ids:
                print(f"    ⚠️ No DICOM series found in {os.path.basename(dicom_dir)}")
                continue
            
            # Process each series in this directory
            for series_id in series_ids:
                try:
                    # Get file names for this series in current directory
                    file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_id)
                    if not file_names:
                        continue
                    
                    # Extract metadata from first file
                    metadata = self.get_dicom_metadata(file_names[0])
                    if not metadata:
                        continue
                    
                    # Detect sequence type
                    sequence_type = self.detect_sequence_type(
                        metadata.get('SeriesDescription', ''),
                        metadata.get('ProtocolName', ''),
                        metadata.get('Modality', '')
                    )
                    
                    # Handle multiple instances of same sequence type
                    if sequence_type in sequence_counts:
                        sequence_counts[sequence_type] += 1
                        sequence_suffix = f"_{sequence_counts[sequence_type]}"
                    else:
                        sequence_counts[sequence_type] = 1
                        sequence_suffix = ""
                    
                    # Generate output filename
                    if use_original_naming:
                        # For original naming, handle duplicates by adding series number
                        filename = self.generate_output_filename(
                            case_id, 
                            sequence_type,
                            metadata.get('SeriesNumber', ''),
                            metadata,
                            use_original_naming=True
                        )
                        # If duplicate, add series number or counter
                        if sequence_counts[sequence_type] > 1:
                            base_name = filename.replace('.nii.gz', '')
                            filename = f"{base_name}_{sequence_counts[sequence_type]}.nii.gz"
                    else:
                        # Standardized naming with sequence suffix handling
                        if sequence_counts[sequence_type] > 1:
                            filename = self.generate_output_filename(
                                case_id, 
                                f"{sequence_type}{sequence_suffix}",
                                metadata.get('SeriesNumber', ''),
                                metadata,
                                use_original_naming=False
                            )
                        else:
                            filename = self.generate_output_filename(
                                case_id, 
                                sequence_type,
                                metadata.get('SeriesNumber', ''),
                                metadata,
                                use_original_naming=False
                            )
                    
                    output_path = os.path.join(output_dir, filename)
                    
                    # Read and convert series
                    reader.SetFileNames(file_names)
                    image = reader.Execute()
                    
                    # Write NIFTI file
                    sitk.WriteImage(image, output_path)
                    
                    print(f"    ✅ Converted: {sequence_type} -> {filename}")
                    converted_files.append(output_path)
                    
                    # Print metadata summary
                    modality = metadata.get('Modality', 'Unknown')
                    series_desc = metadata.get('SeriesDescription', 'No description')
                    print(f"       📋 {modality} | {series_desc}")
                    
                except Exception as e:
                    print(f"    ❌ Failed to convert series {series_id} in {os.path.basename(dicom_dir)}: {e}")
                    continue
        
        return converted_files
    
    def convert_dataset(self, parent_dir: str, output_root: str, 
                       use_original_naming: bool = True) -> Dict[str, List[str]]:
        """
        Convert entire dataset of DICOM cases to NIFTI.
        Based on the original dicom2nifiti.py structure.
        
        Args:
            parent_dir: Root directory containing case folders
            output_root: Root output directory
            use_original_naming: If True, use original SeriesDescription naming (IUSM format)
            
        Returns:
            Dictionary mapping case IDs to converted file lists
        """
        # Create output root directory
        os.makedirs(output_root, exist_ok=True)
        
        results = {}
        
        for case_folder in os.listdir(parent_dir):
            case_path = os.path.join(parent_dir, case_folder)
            
            if not os.path.isdir(case_path):
                continue
            
            output_dir = os.path.join(output_root, case_folder)
            converted_files = self.convert_case(case_path, output_dir, case_folder, use_original_naming)
            results[case_folder] = converted_files
        
        print(f"\n🎉 Conversion complete! Processed {len(results)} cases.")
        return results
    
    def filter_sequences(self, converted_files: List[str], 
                        required_sequences: List[str]) -> Dict[str, str]:
        """
        Filter and validate required sequences for downstream processing.
        
        Args:
            converted_files: List of converted NIFTI file paths
            required_sequences: List of required sequence types
            
        Returns:
            Dictionary mapping sequence types to file paths
        """
        sequence_files = {}
        
        for file_path in converted_files:
            filename = os.path.basename(file_path)
            
            # Extract sequence type from filename
            for seq_type in required_sequences:
                if f"_{seq_type}" in filename or filename.startswith(f"{seq_type}_"):
                    sequence_files[seq_type] = file_path
                    break
        
        # Check for missing sequences
        missing = set(required_sequences) - set(sequence_files.keys())
        if missing:
            print(f"⚠️ Missing required sequences: {missing}")
        
        return sequence_files
    
    def extract_pipeline_sequences(self, converted_files: List[str], 
                                  output_dir: str, case_id: str) -> Dict[str, str]:
        """
        Extract and rename the three key sequences needed for the pipeline:
        - T2W: AX_DYNACAD_T2 variants
        - HBV: AX_DYNACAD_DWI_1400_B_VALUE_TRACEW variants 
        - ADC: AX_DYNACAD_DWI_ADC variants
        
        Args:
            converted_files: List of converted NIFTI file paths
            output_dir: Output directory for pipeline-ready files
            case_id: Case identifier
            
        Returns:
            Dictionary mapping standardized names to file paths
        """
        import shutil
        
        # Create pipeline output directory
        pipeline_dir = os.path.join(output_dir, f"{case_id}_pipeline")
        os.makedirs(pipeline_dir, exist_ok=True)
        
        # Define sequence mapping patterns
        sequence_patterns = {
            't2w': ['AX_DYNACAD_T2'],
            'hbv': ['AX_DYNACAD_DWI_1400_B_VALUE_TRACEW'],
            'adc': ['AX_DYNACAD_DWI_ADC']
        }
        
        pipeline_files = {}
        
        for target_name, patterns in sequence_patterns.items():
            found_file = None
            
            # Find matching file
            for file_path in converted_files:
                filename = os.path.basename(file_path)
                
                for pattern in patterns:
                    if pattern in filename:
                        found_file = file_path
                        break
                
                if found_file:
                    break
            
            if found_file:
                # Copy and rename file
                target_filename = f"{target_name}.nii.gz"
                target_path = os.path.join(pipeline_dir, target_filename)
                
                shutil.copy2(found_file, target_path)
                pipeline_files[target_name] = target_path
                
                print(f"    📋 {target_name.upper()}: {os.path.basename(found_file)} -> {target_filename}")
            else:
                print(f"    ❌ {target_name.upper()}: Not found")
        
        return pipeline_files

    # ------------------------------------------------------------------
    # Generic extractor that supports multiple imaging blocks (MRI / PETCT / PSMA)
    # ------------------------------------------------------------------
    def extract_pipeline_sequences_multimodal(self,
                                              converted_files: List[str],
                                              output_dir: str,
                                              case_id: str,
                                              modality_block: str = 'MRI') -> Dict[str, str]:
        """
        Create a <case>_<block>/ sub-folder and copy the required sequences
        for the chosen imaging block.

        modality_block options and the sequences they expect:

        MRI   ➜  t2w  adc  hbv
        PETCT ➜  ct   pet
        PSMA  ➜  ct   psma_pet
        """

        import shutil

        modality_block = modality_block.upper()

        target_map = {
            'MRI': {
                # only explicit axial T2 variants requested by user
                't2w':      [
                    'ax_dynacad_t2.nii.gz',
                    'ax_t2_propeller.nii.gz', 
                    'ax_t2.nii.gz',
                    'ax_t2_tse.nii.gz',
                    'ax_t2_cube--pelvis.nii.gz',
                    'AX_T2_(3_SKIP_0).nii.gz',
                    'AX_T2_3D_SPACE_+sag_&_cor_mpr.nii.gz',
                    't2_tra.nii.gz',
                    'T2_Ax_FSE_PROP.nii.gz'
                ],
                # explicit ADC variants – only specific ones (ONLY 1400 b-value)
                'adc': [
                    'ax_dynacad_dwi_adc',
                    'ax_dynacad_dwi_adc_dfc_mix',
                    'ax_dynacad_dwi_1400_b_value_adc',
                    'dadc_map',
                    'ax_dwi_b100_800_1400_adc_dfc_mix',
                    'ax_dwi_b100,_800,_1400_adc_dfc_mix',
                    'apparent_diffusion_coefficient_(mm2_s)',
                    'adc_(10^-6_mm²_s):',
                    'dDWI_2000_ADC.nii.gz',
                    'ADC_map.nii.gz',
                    'DIFF_b50_100_1000_1200_ADC_DFC.nii.gz',
                    'AX_DIFF_WHOLE_PELVIS_ADC_DFC_MIX.nii.gz',
                    'ep2d_diff_b50_800_tra_ADC.nii.gz',
                    'ep2d_diff_b50_500_1000_tra_p2_ADC_DFC_MIX.nii.gz'
                ],
                # explicit HBV/TRACEW variants – only specific ones (ONLY 1400 b-value)
                'hbv': [
                    'ax_dynacad_dwi_1400_b_value_tracew',
                    'ax_dwi_(b1400_nsa_20)',
                    'ax_dwi_b100_800_1400_tracew_dfc_mix',
                    'ax_dwi_b100,_800,_1400_tracew_dfc_mix',
                    'ax_dwi_3_b_values',
                    'ax_edwi_3_and_1',
                    'ax_dwi_focus',
                    'sDWI_B2000.nii.gz',
                    'SB1500.nii.gz',
                    'DIFF_b50_100_1000_1200_TRACEW_DFC.nii.gz',
                    'AX_DIFF_WHOLE_PELVIS_TRACEW_DFC_MIX.nii.gz',
                    'ep2d_diff_b50_800_tra_TRACEW.nii.gz',
                    'ep2d_diff_b50_500_1000_tra_p2_TRACEW_DFC_MIX.nii.gz',
                    'DWI_Ax_b-2000.nii.gz'
                ]
            },
            'PETCT': {
                'ct':       ['CT'],
                'pet':      ['PET']
            },
            'PSMA': {
                'ct':       ['CT'],
                'psma_pet': ['PSMA', 'PET']
            }
        }

        if modality_block not in target_map:
            print(f"❌ Unknown modality block: {modality_block}. Supported: {list(target_map.keys())}")
            return {}

        # create <case>_<block>/ output dir
        block_dir = os.path.join(output_dir, f"{case_id}_{modality_block.lower()}")
        os.makedirs(block_dir, exist_ok=True)

        mapping   = target_map[modality_block]
        collected = {}

        # ------------------------------------------------------------
        # MRI block
        #     – detect T2-w / ADC / HBV matches
        #     – copy them into <case>_mri/ and rename:
        #         t2w.nii.gz   /  adc.nii.gz   /  hbv.nii.gz
        #       duplicates →   t2w_2.nii.gz / adc_3.nii.gz …
        # ------------------------------------------------------------
        for fpath in converted_files:
            fname_lower = os.path.basename(fpath).lower()

            seq_tag = None
            if modality_block == 'MRI':
                if any(k.lower() in fname_lower for k in mapping['t2w']):
                    seq_tag = 't2w'
                elif any(k.lower() in fname_lower for k in mapping['hbv']):
                    seq_tag = 'hbv'
                elif any(k.lower() in fname_lower for k in mapping['adc']):
                    seq_tag = 'adc'
            elif modality_block == 'PSMA':
                # For PSMA with original naming, detect based on DICOM series descriptions
                if any(pattern in fname_lower for pattern in ['tb_abdomen', 'abdomen', 'monoe', 'iodine', 'pp', 'hd_fov']):
                    seq_tag = 'ct'
                elif any(pattern in fname_lower for pattern in ['static', 'min']):
                    seq_tag = 'psma_pet'
            else:
                # PETCT blocks – keep previous logic
                for std_name, key_list in mapping.items():
                    if any(k.lower() in fname_lower for k in key_list):
                        seq_tag = std_name
                        break

            if seq_tag is None:
                continue

            # Decide canonical file-name (handle duplicates)
            counter   = len(collected.get(seq_tag, [])) + 1
            base_name = f"{seq_tag}.nii.gz" if counter == 1 else f"{seq_tag}_{counter}.nii.gz"
            dest_path = os.path.join(block_dir, base_name)

            shutil.copy2(fpath, dest_path)

            collected.setdefault(seq_tag, []).append(dest_path)
            print(f"    📋 {seq_tag.upper()}: copied {os.path.basename(fpath)}  ➜  {base_name}")

        # ensure at least one file per required key for MRI
        for key in mapping.keys():
            if key not in collected:
                print(f"    ❌ {key.upper()}: Not found in {case_id}")

        return collected


def main():
    """
    Main function for production use.
    Usage: python multimodal_dicom2nifti.py
    """
    import sys
    
    # Initialize converter
    converter = MultiModalDicom2Nifti()
    
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python multimodal_dicom2nifti.py <input_path> <output_path> [modality_block] [case_id]")
        print("\nExamples:")
        print("  # Convert single case:")
        print("  python multimodal_dicom2nifti.py /path/to/STUDY_P10101 /path/to/output MRI")
        print("  python multimodal_dicom2nifti.py /path/to/STUDY_P10101 /path/to/output PETCT")
        print("\n  # Convert multiple cases:")
        print("  python multimodal_dicom2nifti.py /path/to/parent_directory /path/to/output_root MRI")
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    # optional modality block arg (defaults to MRI)
    modality_block = sys.argv[3].upper() if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else 'MRI'
    # optional case id (only for single-case convert)
    case_id = sys.argv[4] if len(sys.argv) > 4 else None
    
    print("🚀 Multi-Modal DICOM to NIFTI Converter")
    print("=" * 50)
    print(f"📂 Input: {input_path}")
    print(f"📂 Output: {output_path}")
    
    if not os.path.exists(input_path):
        print(f"❌ Error: Input path {input_path} does not exist!")
        return
    
    try:
        # Determine if single-case or multi-case based on directory structure
        is_single_case = False
        
        if os.path.isfile(input_path):
            is_single_case = True
        elif os.path.isdir(input_path):
            # Check if input directory contains DICOM files directly or NIfTI files
            direct_dicom_files = False
            direct_nifti_files = False
            subdirectories = []
            
            for item in os.listdir(input_path):
                item_path = os.path.join(input_path, item)
                if os.path.isdir(item_path):
                    subdirectories.append(item)
                elif item.lower().endswith(('.dcm', '.dicom')):
                    direct_dicom_files = True
                elif item.lower().endswith(('.nii', '.nii.gz')):
                    direct_nifti_files = True
            
            # Single case if: direct DICOM/NIfTI files OR only one subdirectory
            if direct_dicom_files or direct_nifti_files or len(subdirectories) <= 1:
                is_single_case = True
            # Multi-case if: multiple subdirectories with no direct DICOM/NIfTI files
            else:
                is_single_case = False

        if is_single_case:
            # Single case conversion
            if case_id is None:
                case_id = os.path.basename(input_path)
            
            print(f"🔄 Converting single case: {case_id}")
            # Use original naming for MRI and PSMA blocks to preserve DICOM series names
            use_original_naming = (modality_block in ['MRI', 'PSMA'])

            converted_files = converter.convert_case(
                case_path=input_path,
                output_dir=output_path,
                case_id=case_id,
                use_original_naming=use_original_naming
            )
            
            if converted_files:
                print(f"\n✅ Converted {len(converted_files)} series")
                
                # Extract pipeline sequences
                print(f"\n🎯 Extracting pipeline sequences for block {modality_block} ...")
                pipeline_files = converter.extract_pipeline_sequences_multimodal(
                    converted_files, output_path, case_id, modality_block
                )
                
                if pipeline_files:
                    print(f"\n✅ Pipeline-ready files created in: {case_id}_{modality_block.lower()}/")
                    for seq_name, paths in pipeline_files.items():
                        if isinstance(paths, list):
                            for p in paths:
                                size_mb = os.path.getsize(p)/(1024*1024)
                                print(f"  {seq_name.upper()}: {os.path.basename(p)} ({size_mb:.1f} MB)")
                        else:
                            size_mb = os.path.getsize(paths)/(1024*1024)
                            print(f"  {seq_name.upper()}: {os.path.basename(paths)} ({size_mb:.1f} MB)")
            else:
                print("⚠️ No files were converted")
        
        else:
            # Multiple cases conversion
            subdirs = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
            print(f"🔄 Converting multiple cases found in {len(subdirs)} subdirectories:")
            for subdir in subdirs:
                print(f"  📁 {subdir}")
            print()
            
            use_original_naming = (modality_block in ['MRI', 'PSMA'])
            results = converter.convert_dataset(input_path, output_path, use_original_naming=use_original_naming)
            
            print(f"\n✅ Processed {len(results)} cases")
            
            # Extract pipeline sequences for each case
            for case_id, converted_files in results.items():
                if converted_files:
                    print(f"\n🎯 Extracting pipeline sequences for {case_id} ({modality_block}) ...")
                    case_output_dir = os.path.join(output_path, case_id)
                    pipeline_files = converter.extract_pipeline_sequences_multimodal(
                        converted_files, case_output_dir, case_id, modality_block
                    )
                    
                    if pipeline_files:
                        print(f"  ✅ Pipeline files found: {len(pipeline_files)}/{len(pipeline_files)}")
    
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()