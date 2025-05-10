#!/bin/bash
#SBATCH --job-name=qwen3-0.6B-gsm8k
#SBATCH --output=../logs/job_%j.out
#SBATCH --error=../logs/job_%j.err
#SBATCH --partition=low-priority
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# >>> Conda / module setup (edit as needed) >>>
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/xfs/home/kbzhu/conda_envs/mosaic
# <<< ---------------------------------------------------------------

export WANDB_DISABLED=true

# Create the log directory (in case it doesn't exist yet)
mkdir -p ../logs/job_${SLURM_JOB_ID}

echo "Starting qwen3-0.6B GSM8K fine-tune"
python -u full_ft_qwen3-0.6B_gsm8k.py
echo "Finished"
