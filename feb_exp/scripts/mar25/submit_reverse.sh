#!/bin/bash
#SBATCH --job-name=mar25_rev
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/mar25/logs/slurm_%j.out
#SBATCH --error=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/mar25/logs/slurm_%j.err

module load python/3.10.9-fasrc01
source /n/home06/drooryck/envs/codeswitching-py310/bin/activate
export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:${PYTHONPATH:-}

python /n/home06/drooryck/codeswitching-llms/feb_exp/scripts/mar25/run_sweep.py --reverse
