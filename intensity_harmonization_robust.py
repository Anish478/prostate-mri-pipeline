#!/usr/bin/env python3
"""
Robust Intensity Harmonization Pipeline for Prostate MRI
Handles edge cases and provides better error handling for NYUL normalization.
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys
from scipy.interpolate import interp1d

class RobustIntensityHarmonizer:
    """
    Robust intensity harmonization using improved NYUL method.
    """
    
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.harmonized_dir = self.output_dir / "harmonized"
        self.models_dir = self.output_dir / "models" 
        self.histograms_dir = self.output_dir / "histograms"
        self.quality_dir = self.output_dir / "quality_assessment"
        
        for dir_path in [self.harmonized_dir, self.models_dir, self.histograms_dir, self.quality_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.sequences = ['t2w', 'adc', 'hbv']
        
    def robust_get_landmarks(self, img, percs):
        """Get landmarks with robust error handling."""
        if len(img) == 0:
            print("  ⚠️ Empty image after masking, skipping...")
            return None
        
        # Remove extreme outliers
        q1, q99 = np.percentile(img, [1, 99])
        img_filtered = img[(img >= q1) & (img <= q99)]
        
        if len(img_filtered) < 100:  # Need minimum points
            print("  ⚠️ Insufficient valid pixels after filtering, skipping...")
            return None
            
        landmarks = np.percentile(img_filtered, percs)
        return landmarks
        
    def train_robust_nyul(self, img_paths, sequence_name):
        """Train NYUL model with robust error handling."""
        print(f"  📚 Training {sequence_name.upper()} model...")
        
        percs = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99])
        all_landmarks = []
        
        valid_images = 0
        for img_path in tqdm(img_paths, desc=f"Processing {sequence_name}"):
            try:
                # Load image
                img_data = nib.load(img_path).get_fdata()
                
                # Create robust mask (tissue regions)
                if sequence_name == 'adc':
                    # ADC specific masking - values should be positive and reasonable
                    mask = (img_data > 0) & (img_data < 5000)  # ADC typically 0-3000 x10^-6
                elif sequence_name == 'hbv':
                    # HBV specific masking
                    mask = img_data > np.percentile(img_data[img_data > 0], 5)
                else:  # T2W
                    # Standard tissue masking
                    mask = img_data > np.percentile(img_data[img_data > 0], 10)
                
                masked_data = img_data[mask]
                
                if len(masked_data) < 1000:  # Need sufficient data
                    print(f"    ⚠️ Insufficient tissue pixels in {Path(img_path).name}")
                    continue
                
                landmarks = self.robust_get_landmarks(masked_data, percs)
                if landmarks is not None:
                    all_landmarks.append(landmarks)
                    valid_images += 1
                    
            except Exception as e:
                print(f"    ❌ Error processing {Path(img_path).name}: {e}")
                continue
        
        if len(all_landmarks) < 5:  # Need minimum training images
            print(f"  ❌ Insufficient valid images for {sequence_name} training ({len(all_landmarks)}/5 minimum)")
            return None
            
        # Calculate standard scale as mean of all landmarks
        standard_scale = np.mean(all_landmarks, axis=0)
        
        print(f"  ✅ {sequence_name.upper()} model trained on {valid_images} images")
        return standard_scale, percs
    
    def apply_robust_nyul(self, img_data, standard_scale, percs, sequence_name):
        """Apply NYUL normalization with robust handling."""
        try:
            # Create mask based on sequence type
            if sequence_name == 'adc':
                mask = (img_data > 0) & (img_data < 5000)
            elif sequence_name == 'hbv':
                mask = img_data > np.percentile(img_data[img_data > 0], 5)
            else:  # T2W
                mask = img_data > np.percentile(img_data[img_data > 0], 10)
            
            masked_data = img_data[mask]
            
            if len(masked_data) < 100:
                print(f"    ⚠️ Insufficient pixels for normalization")
                return img_data  # Return original
            
            # Get image landmarks
            img_landmarks = self.robust_get_landmarks(masked_data, percs)
            if img_landmarks is None:
                return img_data
            
            # Create interpolation function
            f = interp1d(img_landmarks, standard_scale, kind='linear', 
                        fill_value='extrapolate', bounds_error=False)
            
            # Apply transformation
            normalized = f(img_data)
            
            # Ensure reasonable output range
            if sequence_name == 'adc':
                normalized = np.clip(normalized, 0, 5000)
            elif sequence_name in ['t2w', 'hbv']:
                normalized = np.clip(normalized, 0, np.percentile(normalized, 99.5))
            
            return normalized.astype(img_data.dtype)
            
        except Exception as e:
            print(f"    ❌ Normalization failed: {e}")
            return img_data  # Return original on failure
    
    def collect_images(self):
        """Collect all coregistered images."""
        image_collections = {seq: [] for seq in self.sequences}
        
        print("🔍 Collecting coregistered images...")
        
        # Check if using new standardized structure
        nif_dir = Path("/Users/anish/from_scratch_G/nif")
        if nif_dir.exists():
            print("  📁 Using standardized NIfTI structure...")
            for case_dir in nif_dir.glob("*"):
                if case_dir.is_dir():
                    mri_dir = case_dir / f"{case_dir.name}_mri"
                    if mri_dir.exists():
                        # Check for standardized files
                        t2w_file = mri_dir / "t2w.nii.gz"
                        adc_file = mri_dir / "adc.nii.gz"
                        hbv_file = mri_dir / "hbv.nii.gz"
                        
                        if t2w_file.exists():
                            image_collections['t2w'].append(str(t2w_file))
                        if adc_file.exists():
                            image_collections['adc'].append(str(adc_file))
                        if hbv_file.exists():
                            image_collections['hbv'].append(str(hbv_file))
        else:
            # Fall back to coregistered results
            print("  📁 Using coregistered results structure...")
            for case_dir in self.data_dir.glob("*_coregistered"):
                case_id = case_dir.name.replace("_coregistered", "")
                
                # T2W reference
                t2w_ref = case_dir / f"t2w_reference_{case_id}.nii.gz"
                if t2w_ref.exists():
                    image_collections['t2w'].append(str(t2w_ref))
                
                # ADC and HBV results
                adc_result = case_dir / "adc_registration" / "result.0.nii.gz"
                if adc_result.exists():
                    image_collections['adc'].append(str(adc_result))
                    
                hbv_result = case_dir / "hbv_registration" / "result.nii.gz"
                if hbv_result.exists():
                    image_collections['hbv'].append(str(hbv_result))
        
        print(f"📊 Collected images:")
        for seq, imgs in image_collections.items():
            print(f"  {seq.upper()}: {len(imgs)} images")
            
        return image_collections
    
    def generate_comparison_histograms(self, image_collections, harmonized_paths=None):
        """Generate histograms for before/after comparison."""
        print("📈 Generating histograms...")
        
        n_sample = 5  # Sample size for speed
        
        fig, axes = plt.subplots(2 if harmonized_paths else 1, 3, figsize=(15, 10 if harmonized_paths else 5))
        if not harmonized_paths:
            axes = [axes]  # Make it 2D for consistency
        
        colors = ['blue', 'orange', 'green']
        
        for i, seq in enumerate(self.sequences):
            if not image_collections[seq]:
                continue
            
            # Before harmonization
            original_intensities = []
            for img_path in image_collections[seq][:n_sample]:
                try:
                    img_data = nib.load(img_path).get_fdata()
                    # Get tissue intensities
                    if seq == 'adc':
                        tissue = img_data[(img_data > 0) & (img_data < 5000)]
                    else:
                        tissue = img_data[img_data > np.percentile(img_data[img_data > 0], 10)]
                    
                    if len(tissue) > 0:
                        original_intensities.extend(tissue.flatten()[:2000])  # Sample
                except:
                    continue
            
            if original_intensities:
                axes[0][i].hist(original_intensities, bins=50, alpha=0.7, 
                               color=colors[i], density=True)
                axes[0][i].set_title(f'{seq.upper()} - Original')
                axes[0][i].set_xlabel('Intensity')
                axes[0][i].set_ylabel('Density')
            
            # After harmonization (if available)
            if harmonized_paths and seq in harmonized_paths:
                harmonized_intensities = []
                for img_path in harmonized_paths[seq][:n_sample]:
                    try:
                        img_data = nib.load(img_path).get_fdata()
                        # Get tissue intensities
                        if seq == 'adc':
                            tissue = img_data[(img_data > 0) & (img_data < 5000)]
                        else:
                            tissue = img_data[img_data > np.percentile(img_data[img_data > 0], 10)]
                        
                        if len(tissue) > 0:
                            harmonized_intensities.extend(tissue.flatten()[:2000])  # Sample
                    except:
                        continue
                
                if harmonized_intensities:
                    axes[1][i].hist(harmonized_intensities, bins=50, alpha=0.7, 
                                   color=colors[i], density=True)
                    axes[1][i].set_title(f'{seq.upper()} - Harmonized')
                    axes[1][i].set_xlabel('Intensity')
                    axes[1][i].set_ylabel('Density')
        
        plt.tight_layout()
        filename = "before_after_harmonization.png" if harmonized_paths else "original_histograms.png"
        plt.savefig(self.histograms_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Histograms saved: {filename}")
    
    def run_harmonization_pipeline(self):
        """Run the complete robust harmonization pipeline."""
        print("🚀 Starting Robust Intensity Harmonization Pipeline")
        print("=" * 60)
        
        # Step 1: Collect images
        image_collections = self.collect_images()
        if not any(image_collections.values()):
            print("❌ No images found! Check your data directory.")
            return None, None
        
        # Step 2: Generate original histograms
        self.generate_comparison_histograms(image_collections)
        
        # Step 3: Train models
        models = {}
        for seq in self.sequences:
            if image_collections[seq]:
                result = self.train_robust_nyul(image_collections[seq], seq)
                if result is not None:
                    standard_scale, percs = result
                    models[seq] = (standard_scale, percs)
                    
                    # Save model
                    model_path = self.models_dir / f"{seq}_robust_nyul_model.npz"
                    np.savez(model_path, standard_scale=standard_scale, percs=percs)
                    print(f"  💾 {seq.upper()} model saved: {model_path}")
        
        if not models:
            print("❌ No models could be trained!")
            return None, None
        
        # Step 4: Apply harmonization
        print("🎨 Applying harmonization...")
        harmonized_paths = {seq: [] for seq in self.sequences}
        
        for seq, (standard_scale, percs) in models.items():
            print(f"  🔄 Harmonizing {seq.upper()} images...")
            
            for img_path in tqdm(image_collections[seq], desc=f"Harmonizing {seq}"):
                try:
                    # Load image
                    img_nib = nib.load(img_path)
                    img_data = img_nib.get_fdata()
                    
                    # Apply robust normalization
                    normalized_data = self.apply_robust_nyul(img_data, standard_scale, percs, seq)
                    
                    # Create output path
                    case_name = Path(img_path).parent.parent.name
                    if "mri" in case_name:
                        case_name = case_name.replace("_mri", "")
                    
                    output_path = self.harmonized_dir / case_name / f"{seq}_harmonized.nii.gz"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save harmonized image
                    harmonized_nib = nib.Nifti1Image(normalized_data, img_nib.affine, img_nib.header)
                    nib.save(harmonized_nib, output_path)
                    
                    harmonized_paths[seq].append(str(output_path))
                    
                except Exception as e:
                    print(f"    ❌ Failed to process {Path(img_path).name}: {e}")
                    continue
        
        # Step 5: Generate after histograms
        self.generate_comparison_histograms(image_collections, harmonized_paths)
        
        # Step 6: Quality report
        self.generate_quality_report(image_collections, harmonized_paths)
        
        print("\\n🎉 Robust intensity harmonization completed!")
        print(f"📁 Results saved in: {self.output_dir}")
        
        return harmonized_paths, models
    
    def generate_quality_report(self, original_collections, harmonized_paths):
        """Generate quality assessment report."""
        print("📋 Generating quality report...")
        
        report_path = self.quality_dir / "harmonization_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Robust NYUL Harmonization Report\\n")
            f.write("=" * 50 + "\\n\\n")
            
            for seq in self.sequences:
                if seq not in harmonized_paths:
                    continue
                    
                f.write(f"{seq.upper()} Sequence:\\n")
                f.write(f"  Original images: {len(original_collections[seq])}\\n")
                f.write(f"  Harmonized images: {len(harmonized_paths[seq])}\\n")
                f.write(f"  Success rate: {len(harmonized_paths[seq])/len(original_collections[seq])*100:.1f}%\\n\\n")
        
        print(f"  ✅ Quality report saved: {report_path}")

def main():
    """Main function to run robust intensity harmonization."""
    
    # Define paths
    coregistered_dir = "/Users/anish/from_scratch_G/coregistered_results_nif"
    output_dir = "/Users/anish/from_scratch_G/robust_intensity_harmonization"
    
    # Create harmonizer
    harmonizer = RobustIntensityHarmonizer(coregistered_dir, output_dir)
    
    # Run pipeline
    harmonized_paths, models = harmonizer.run_harmonization_pipeline()
    
    if harmonized_paths:
        print("\\n📋 Harmonization Summary:")
        for seq, paths in harmonized_paths.items():
            print(f"  {seq.upper()}: {len(paths)} harmonized images")
    else:
        print("❌ Harmonization failed!")

if __name__ == "__main__":
    main()