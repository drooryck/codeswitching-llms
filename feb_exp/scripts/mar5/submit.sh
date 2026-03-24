#!/bin/bash
#SBATCH --job-name=mar5_sweeps
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/mar5/logs/slurm_%j.out
#SBATCH --error=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/mar5/logs/slurm_%j.err

module load python/3.10.9-fasrc01
source /n/home06/drooryck/envs/codeswitching-py310/bin/activate
export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:${PYTHONPATH:-}

mkdir -p /n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/mar5/logs

python /n/home06/drooryck/codeswitching-llms/feb_exp/scripts/mar5/run_sweep.py
