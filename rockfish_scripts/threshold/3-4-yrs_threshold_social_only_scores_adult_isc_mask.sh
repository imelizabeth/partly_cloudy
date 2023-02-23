#!/bin/bash
#SBATCH --job-name=3-4-yrs_social_only_scores_adult_isc_fdr_results
#SBATCH --time=1:00:00
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB

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
python -u 3-4-yrs_threshold_social_only_scores_adult_isc_mask.py
conda deactivate
echo “Finished with job $SLURM_JOBID”