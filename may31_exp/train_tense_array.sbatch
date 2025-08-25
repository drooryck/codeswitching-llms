#!/bin/bash
#SBATCH --job-name=tense
#SBATCH --partition=seas_gpu            # ← edit to your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-65               # 22 props × 3 seeds = 66 jobs
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# --- activate our environment with all packages installed
source ../venv39/bin/activate

# --- ensure data symlink exists
ln -sf ../may20_exp/data .

# --- decode the array index into (prop, run_id) ----------------------------
PROPS=(0 0.025 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 \
       0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.975 1.0)
NUM_RUNS=3

idx=$SLURM_ARRAY_TASK_ID
prop_idx=$(( idx / NUM_RUNS ))
run_id=$(( idx % NUM_RUNS ))
prop=${PROPS[$prop_idx]}

echo "SLURM_ARRAY_TASK_ID=$idx → prop=$prop  run_id=$run_id"

python train_tense.py \
       --prop      "$prop" \
       --run_id    "$run_id" \
       --eval_prop 0.1 \
       --out_root  interactive_results_v2
