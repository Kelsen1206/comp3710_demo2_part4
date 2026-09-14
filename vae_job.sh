#!/bin/bash
#SBATCH --job-name=vae-oasis
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=vae_%j.out
#SBATCH --error=vae_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

python part4_vae_rangpur.py
