#!/bin/bash
#SBATCH --job-name=dicom2nifti
#SBATCH --account=r01271
#SBATCH --partition=general
#SBATCH --qos=allocated
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
#SBATCH --output=dicom_conversion_%j.out
#SBATCH --error=dicom_conversion_%j.err

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load Python module
module load python/3.11.4

# Load Python module (adjust if needed)
# module load python/3.8  # uncomment if needed

# Run the conversion
cd /N/slate/agada
python multimodal_dicom2nifti.py /N/project/prostate_cancer_ai/In_house_MRI_dataset/2021-00119 /N/slate/agada/output MRI

echo "Job completed at: $(date)"
