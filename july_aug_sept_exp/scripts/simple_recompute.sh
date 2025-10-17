#!/bin/bash
#SBATCH --job-name=recompute_parallel
#SBATCH --output=/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/scripts/temp_plots/recompute_%j.out
#SBATCH --error=/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/scripts/temp_plots/recompute_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Load modules and activate environment (copied from your working script)
module load python/3.10.9-fasrc01
source /n/home06/drooryck/envs/codeswitching-py310/bin/activate

# Add project root to PYTHONPATH (copied from your working script)
export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:$PYTHONPATH

cd /n/home06/drooryck/codeswitching-llms/july_aug_sept_exp

echo "Starting parallel recomputation job at $(date)"
echo "Python path: $(which python)"
echo "Working directory: $(pwd)"
echo "Available CPUs: $(nproc)"
echo "SLURM CPUs allocated: $SLURM_CPUS_PER_TASK"

# Force unbuffered output and run the parallel Python script
python -u scripts/recompute_metrics.py 2>&1

echo "Job finished at $(date)"