#!/bin/bash
#
#========== Slurm options ==========
#SBATCH --job-name=pert_GRPO
#SBATCH --nodes=3                 # 3 nodes
#SBATCH --ntasks-per-node=1       # one Slurm task per node
#SBATCH --gpus-per-task=8         # 8 GPUs on every node
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16        # tune if you want more CPU threads per GPU
#SBATCH --output=logs/train/GRPO/%x-%j.log
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=0-00:00:00
#SBATCH -p cu_0001
#===================================

# ---------- container / paths ----------



