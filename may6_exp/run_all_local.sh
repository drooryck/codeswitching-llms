#!/usr/bin/env bash
# ------------------------------------------------------------------
# run_all_local.sh  – run all p × seed combinations on THIS node
# ------------------------------------------------------------------
set -e

# ---------- editable -------------------------------------------------------
PYENV=~/circuits_languages_2/venv39           # activate this venv
CODEDIR=~/circuits_languages_2/may6_exp       # folder with train_tense_model_v5.py
LOGDIR=$CODEDIR/logs
mkdir -p "$LOGDIR"

PROPS=(0.0 0.025 0.05 0.075 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.85 0.9 0.925 0.95 0.975 1.0)   # proportions
SEEDS=(1)                                     # independent runs

# ---------- activate environment ------------------------------------------
cd "$CODEDIR"

# ---------- run training loops --------------------------------------------
for p in "${PROPS[@]}"; do
  for s in "${SEEDS[@]}"; do
    echo "[$(date)]  START  p=$p  seed=$s"
    # Fixed: use tee to write to both the file and standard output
    python train_tense_model_v5.py \
        --prop "$p" \
        --run_id "$s" \
        2>&1 | tee "$LOGDIR/run_p${p}_seed${s}.out"
    echo "[$(date)]  DONE   p=$p  seed=$s"
  done
done

# ---------- aggregate & plot ----------------------------------------------
echo "[$(date)]  Aggregating results..."
python aggregate_accuracy.py    # creates accuracy_summary.csv + accuracy_seaborn.png
echo "All runs complete. Results saved in accuracy_seaborn.png and accuracy_summary.csv"