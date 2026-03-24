#!/bin/bash
#SBATCH --job-name=plot_tok_type
#SBATCH --partition=seas_compute
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --output=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/mar5/logs/plot_token_type_%j.out
#SBATCH --error=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/mar5/logs/plot_token_type_%j.err

module load python/3.10.9-fasrc01
source /n/home06/drooryck/envs/codeswitching-py310/bin/activate
export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:${PYTHONPATH:-}

mkdir -p /n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/mar5/logs

python /n/home06/drooryck/codeswitching-llms/feb_exp/scripts/mar5/plot_token_type_by_position.py
