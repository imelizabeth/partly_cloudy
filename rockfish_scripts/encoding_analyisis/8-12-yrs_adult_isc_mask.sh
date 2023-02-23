#!/bin/bash
#SBATCH --job-name=8-12-yrs_adult_isc_mask
#SBATCH --time=12:00:00
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

#SBATCH --mail-type=end
#SBATCH --mail-user=ashirah1@jhu.edu

#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

#---------------------------------------------------------------------
# SLURM job script
#---------------------------------------------------------------------

ml anaconda
ml # confirm modules used
conda activate partly_cloudy_env
python -u 8-12-yrs_adult_isc_mask.py
conda deactivate
echo “Finished with job $SLURM_JOBID”[ashirah1@login03 social_encoding_mask]$
