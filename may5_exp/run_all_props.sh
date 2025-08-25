#!/usr/bin/env bash
# ------------------------------------------------------------------
# run_all_props.sh  – sequentially train 11 proportions (0 … 1.0)
#
# Assumes:
#   • virtual‑env already activated (e.g., `source venv39/bin/activate`)
#   • train_tense_model_v5.py is in the current directory
#   • logs/ directory exists (or we create it)
# ------------------------------------------------------------------

mkdir -p logs

declare -a props=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for p in "${props[@]}"; do
    echo "[$(date)]  Starting training run p=${p}"
    python train_tense_model_v5.py --prop "${p}" \
        >  logs/run_p${p}.out  2>&1
    echo "[$(date)]  Finished training run p=${p}"
done
