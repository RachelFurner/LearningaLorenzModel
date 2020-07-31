#!/bin/bash
#
#SBATCH -J LorenzIntegrator
#SBATCH -o /data/hpcdata/users/racfur/DynamicPrediction/code_git/LorenzModel/LorenzIntegrator.%j.out
#SBATCH -D /data/hpcdata/users/racfur/DynamicPrediction/code_git/LorenzModel
#SBATCH --chdir=/data/hpcdata/users/racfur/DynamicPrediction/code_git/LorenzModel
#SBATCH --mem=50gb
#SBATCH --time=13-24:00:00         
#
# Send mail when the job beings, ends, fails or is requeued
#SBATCH --mail-type=begin,end,fail,requeue
#SBATCH --mail-user=racfur@bas.ac.uk
#
# Pick the partition (equivalent to SGE queue) either short, medium or long.
#SBATCH --partition=long  
# Select the billing account to use, it must match the partition ie. short, medium or long
#SBATCH --account=long  
#
# Load the module
module add hpc/jupyterhub/20190513
source /etc/profile.d/modules.sh
. /hpcpackages/jupyterhub/20190513/etc/profile.d/conda.sh
conda activate /data/hpcdata/users/racfur/conda-envs/RF_hpc_clean
module add hpc/intel/2017
module add hpc/netcdf/intel/4.4.1.1

#
# run program
MPLBACKEND='Agg' python /data/hpcdata/users/racfur/DynamicPrediction/code_git/LorenzModel/CreateLorenzDataset.py
#
