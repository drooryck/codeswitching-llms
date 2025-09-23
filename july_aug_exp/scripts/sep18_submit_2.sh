#!/bin/bash
#SBATCH --job-name=ablation_inf
#SBATCH --output=/n/home06/drooryck/codeswitching-llms/july_aug_exp/results/sep15.9/ablation_logs/ablation_%A_%a.out
#SBATCH --error=/n/home06/drooryck/codeswitching-llms/july_aug_exp/results/sep15.9/ablation_logs/ablation_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_requeue
#SBATCH --array=0-40  # Adjust this based on number of runs

# Load modules and activate environment
module load python/3.10.9-fasrc01
source /n/home06/drooryck/envs/codeswitching-py310/bin/activate

# Add project root to PYTHONPATH
export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:$PYTHONPATH

# Get the list of run directories and select the one for this array task
BASE_DIR="/n/home06/drooryck/codeswitching-llms/july_aug_exp/results/sep15.9"
RUN_DIRS=($(ls -d ${BASE_DIR}/p*_run*))
RUN_DIR="${RUN_DIRS[$SLURM_ARRAY_TASK_ID]}"

echo "Processing run directory: $RUN_DIR"

# Run the script with the selected run directory
python /n/home06/drooryck/codeswitching-llms/july_aug_exp/scripts/run_ablation_inference.py "$RUN_DIR"