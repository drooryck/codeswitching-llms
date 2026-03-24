#!/bin/bash
#SBATCH --job-name=alignment_scores
#SBATCH --partition=seas_compute
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/mar4/logs/alignment_%j.out
#SBATCH --error=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/mar4/logs/alignment_%j.err

module load python/3.10.9-fasrc01
source /n/home06/drooryck/envs/codeswitching-py310/bin/activate
export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:${PYTHONPATH:-}

RUNS_ROOT=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/mar4

echo "[$(date)] Computing alignment scores for version1_plurality_mixing..."
python /n/home06/drooryck/codeswitching-llms/feb_exp/scripts/compute_language_alignment.py \
    "${RUNS_ROOT}/version1_plurality_mixing/runs"

echo "[$(date)] Computing alignment scores for version1_no_plurality_mixing..."
python /n/home06/drooryck/codeswitching-llms/feb_exp/scripts/compute_language_alignment.py \
    "${RUNS_ROOT}/version1_no_plurality_mixing/runs"

echo "[$(date)] Done."
