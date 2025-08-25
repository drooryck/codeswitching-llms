#!/bin/bash
#SBATCH --job-name=tense-small
#SBATCH --partition=seas_gpu
#SBATCH --account=dam_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4          # data‑loading & tokenisation
#SBATCH --mem=16G
#SBATCH --time=4:00:00             # plenty for 20 k steps tiny model
#SBATCH --array=0-10               # 11 proportions
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# -------- setup Python env -------------------------------------------------
module load python/3.10.9-fasrc01   # or whatever module gives you python
source ~/circuits_languages_2/venv39/bin/activate

cd ~/circuits_languages_2/may5_exp   # <== path where train_tense_model_v5.py lives

# -------- map array index -> proportion ------------------------------------
props=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
PROP=${props[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)]  Starting run  p=${PROP}"
python train_tense_model_v5.py --prop "$PROP"
echo "[$(date)]  Finished run  p=${PROP}"
