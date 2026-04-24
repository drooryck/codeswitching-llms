#!/bin/bash
#SBATCH --job-name=cfg_apr14
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-8
#SBATCH --output=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/april_exp/cfg/results/sweep_apr14/logs/slurm_%A_%a.out
#SBATCH --error=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/april_exp/cfg/results/sweep_apr14/logs/slurm_%A_%a.err

# ── Sweep grid ─────────────────────────────────────────────────────
# Each array task handles one prop value (9 trans_fracs × 3 seeds = 27 runs)
PROPS=(0 0.01 0.1 0.25 0.5 0.75 0.9 0.99 1.0)
TRANS_FRACS=(0 0.01 0.1 0.25 0.5 0.75 0.9 0.99 1.0)
SEEDS=(1 2 3)
TRANSLATION_LEVEL=tense_separate
EVAL_PROP=0.1

PROP=${PROPS[$SLURM_ARRAY_TASK_ID]}
echo "[$(date)] Array task ${SLURM_ARRAY_TASK_ID}: prop=${PROP}"

# ── Paths ──────────────────────────────────────────────────────────
OUTPUT_ROOT=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/april_exp/cfg/results/sweep_apr14
CONFIG_JSON=/n/home06/drooryck/codeswitching-llms/april_exp/cfg/configs/default_model.json
DATA_DIR=/n/home06/drooryck/codeswitching-llms/april_exp/cfg/data
LEXICON=/n/home06/drooryck/codeswitching-llms/april_exp/cfg/data/lexicon.json
PY_ENV=/n/home06/drooryck/envs/codeswitching-py310/bin/activate

# ── Setup ──────────────────────────────────────────────────────────
module load python/3.10.9-fasrc01
source "${PY_ENV}"
export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:${PYTHONPATH:-}

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/runs"

# Save a copy of this script and config on first task
if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
    cp -f "${CONFIG_JSON}" "${OUTPUT_ROOT}/model_config.json"
    cp -f "$0" "${OUTPUT_ROOT}/submitted_script.sh"
fi

# ── Generate pairs once (first task to arrive creates them) ───────
LOCK="${DATA_DIR}/.gen_lock"
if [[ ! -f "${DATA_DIR}/train_pairs.csv" ]]; then
    (
        flock -n 200 || { echo "Waiting for data generation..."; flock 200; }
        if [[ ! -f "${DATA_DIR}/train_pairs.csv" ]]; then
            echo "[$(date)] Generating sentence pairs..."
            python -c "
import sys; sys.path.insert(0, '/n/home06/drooryck/codeswitching-llms')
import json
from april_exp.cfg.src.dataset_manager import DatasetManager
from april_exp.cfg.src.model_config import ModelConfig
config = ModelConfig(**json.load(open('${CONFIG_JSON}')))
dm = DatasetManager('${DATA_DIR}', config, lexicon_path='${LEXICON}')
dm.generate_and_save_pairs(n_trees_per_struct=40000, test_size=0.2, seed=0)
"
            echo "[$(date)] Pair generation complete."
        fi
    ) 200>"${LOCK}"
fi

# ── Run loop for this prop ────────────────────────────────────────
n_total=${#TRANS_FRACS[@]}
run_idx=0
for TF in "${TRANS_FRACS[@]}"; do
    for SEED in "${SEEDS[@]}"; do

        if [[ "${TF}" == "0" ]]; then
            LEVEL="none"
        else
            LEVEL="${TRANSLATION_LEVEL}"
        fi

        RUN_NAME=$(printf "mask_%s_prop%s_tf%s_run%02d" \
                   "${LEVEL}" "${PROP}" "${TF}" "${SEED}")
        RUN_DIR="${OUTPUT_ROOT}/runs/${RUN_NAME}"

        if [[ -f "${RUN_DIR}/test_predictions.csv" ]]; then
            echo "[$(date)] SKIP (already complete): ${RUN_NAME}"
            ((run_idx++))
            continue
        fi

        ((run_idx++))
        echo "[$(date)] Run ${run_idx}/27: ${RUN_NAME}"

        mkdir -p "${RUN_DIR}"

        python -m april_exp.cfg.src.run_single \
            --config "${CONFIG_JSON}" \
            --output-dir "${RUN_DIR}" \
            --data-dir "${DATA_DIR}" \
            --lexicon-path "${LEXICON}" \
            --prop "${PROP}" \
            --trans-frac "${TF}" \
            --translation-level "${LEVEL}" \
            --run-id "${SEED}" \
            --eval-prop "${EVAL_PROP}"

        echo "[$(date)] Finished: ${RUN_NAME}"
    done
done

echo "[$(date)] Array task ${SLURM_ARRAY_TASK_ID} complete: ${run_idx} runs."
