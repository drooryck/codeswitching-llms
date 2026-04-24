#!/bin/bash
#SBATCH --job-name=hyp_test
#SBATCH --output=/n/home06/drooryck/codeswitching-llms/april_exp/trans_conj/scripts/apr14/hypothesis_test_output.txt
#SBATCH --error=/n/home06/drooryck/codeswitching-llms/april_exp/trans_conj/scripts/apr14/hypothesis_test_output.txt
#SBATCH --partition=seas_compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

module load python/3.10.9-fasrc01
export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:${PYTHONPATH:-}

python /n/home06/drooryck/codeswitching-llms/april_exp/trans_conj/scripts/apr14/hypothesis_test_alignment.py
