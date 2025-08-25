#!/usr/bin/env bash
# ------------------------------------------------------------------
# submit_tense_experiment.sh
#  • Submits SLURM array for all proportions & seeds
#  • Chains a dependent job that aggregates metrics and plots
# ------------------------------------------------------------------

# ----- editable parameters -------------------------------------------------
PARTITION=seas_gpu
ACCOUNT=dam_lab
SEEDS=(0 1 2 3 4)                # 5 independent runs per p
PROPS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
PYENV=~/circuits_languages_2/venv39     # virtual‑env with torch/transformers
CODEDIR=~/circuits_languages_2/may6_exp # contains train_tense_model_v5.py
LOGDIR=$CODEDIR/logs
mkdir -p "$LOGDIR"

# ---------- 1) submit training array --------------------------------------
ARRAY_COUNT=$(( ${#PROPS[@]} * ${#SEEDS[@]} - 1 ))

ARRAY_JOB_ID=$(sbatch <<EOF | awk '{print $4}'
#!/bin/bash
#SBATCH --job-name=tense-train
#SBATCH --partition=$PARTITION
#SBATCH --account=$ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --array=0-$ARRAY_COUNT
#SBATCH --output=$LOGDIR/train_%A_%a.out
#SBATCH --error=$LOGDIR/train_%A_%a.err

module load python/3.10.9-fasrc01
source $PYENV/bin/activate
cd $CODEDIR

# map array index -> (prop, seed)
SEEDS=(${SEEDS[@]})
PROPS=(${PROPS[@]})
p_idx=\$(( SLURM_ARRAY_TASK_ID / \${#SEEDS[@]} ))
s_idx=\$(( SLURM_ARRAY_TASK_ID % \${#SEEDS[@]} ))

PROP=\${PROPS[\$p_idx]}
SEED=\${SEEDS[\$s_idx]}

echo "[$(date)]  start  p=\$PROP  run_id=\$SEED"
# Removed the --seed parameter as it's not supported
python train_tense_model_v5.py --prop "\$PROP" --run_id "\$SEED"
echo "[$(date)]  done   p=\$PROP  run_id=\$SEED"
EOF
)

echo "Submitted training array  JobID=${ARRAY_JOB_ID}"

# ---------- 2) submit dependent aggregation job ---------------------------
sbatch --dependency=afterok:${ARRAY_JOB_ID} --partition=seas_compute <<EOF
#!/bin/bash
#SBATCH --job-name=tense-plot
#SBATCH --account=$ACCOUNT
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=$LOGDIR/plot_%j.out
#SBATCH --error=$LOGDIR/plot_%j.err

module load python/3.10.9-fasrc01
source $PYENV/bin/activate
cd $CODEDIR

python aggregate_accuracy.py
EOF