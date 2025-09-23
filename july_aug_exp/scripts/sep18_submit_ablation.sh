#!/bin/bash
#SBATCH --job-name=ablation_inf
#SBATCH --output=/n/home06/drooryck/codeswitching-llms/july_aug_exp/results/sep18.0/ablation_logs/ablation_%j.out
#SBATCH --error=/n/home06/drooryck/codeswitching-llms/july_aug_exp/results/sep18.0/ablation_logs/ablation_%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_requeue

# Load modules and activate environment
module load python/3.10.9-fasrc01
source /n/home06/drooryck/envs/codeswitching-py310/bin/activate

# Add project root to PYTHONPATH
export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:$PYTHONPATH

# Run the script
python /n/home06/drooryck/codeswitching-llms/july_aug_exp/scripts/sep18_run_abl_inf_parallel.py
