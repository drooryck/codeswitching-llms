#!/bin/bash
#SBATCH --job-name=lang_20k
#SBATCH --partition=seas_gpu
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/n/home06/drooryck/circuits_languages_2/jul_1/results/sweep_20k/logs/slurm_%A_%a.out
#SBATCH --error=/n/home06/drooryck/circuits_languages_2/jul_1/results/sweep_20k/logs/slurm_%A_%a.err
#SBATCH --account=dam_lab
#SBATCH --array=0-0

# Load modules and activate environment
module load python/3.10.9-fasrc01
source /n/home06/drooryck/circuits_languages_2/venv39/bin/activate

# Job mapping
case $SLURM_ARRAY_TASK_ID in
  0) PROP=0.2; RUN_ID=48 ;;
  *) echo "Invalid array index"; exit 1 ;;
esac

# Run experiment
python -m src.run_single --config /n/home06/drooryck/circuits_languages_2/jul_1/results/sweep_20k/model_config.json --output-dir /n/home06/drooryck/circuits_languages_2/jul_1/results/sweep_20k --prop $PROP --run-id $RUN_ID --eval-prop 1.0
