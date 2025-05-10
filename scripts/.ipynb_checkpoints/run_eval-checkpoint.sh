#!/bin/bash
#SBATCH --job-name=tinyllama-eval       
#SBATCH --output=../logs/job_%j/job_%j.out    # there is something wrong with this
#SBATCH --error=../logs/job_%j/job_%j.err     # there is something wrong with this
#SBATCH --partition=low-priority
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# >>> Conda / module setup (edit as needed) >>>
module load cuda/12.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mosaic
# <<< ---------------------------------------------------------------

export WANDB_DISABLED=true

# Create the log directory (in case it doesn't exist yet)
mkdir -p ../logs/job_${SLURM_JOB_ID}

echo "Starting TinyLlama evaluation"
python -u eval_tinyllama_gsm8k_svamp.py
echo "Finished evaluation"
