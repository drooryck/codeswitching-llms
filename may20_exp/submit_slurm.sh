#!/bin/bash
#SBATCH --job-name=tense_sweep
#SBATCH --array=0-19
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=seas_gpu

# Define your 20 proportions (0.05, 0.10, …, 1.00)
PROPS=( \
  0.00 0.05 0.10 0.15 0.20 0.25 \
  0.30 0.35 0.40 0.45 0.50 \
  0.55 0.60 0.65 0.70 0.75 \
  0.80 0.85 0.90 0.95 1.00 \
)

# Pick this job’s prop based on the array index
PROP=${PROPS[$SLURM_ARRAY_TASK_ID]}
# Use array index as run_id
RUN_ID=$SLURM_ARRAY_TASK_ID

echo "Starting job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID: prop=$PROP run_id=$RUN_ID"

module load cuda/11.7  # adjust to your environment
source ~/venv/bin/activate

python train_tense_sweep.py \
  --prop ${PROP} \
  --run_id ${RUN_ID}
