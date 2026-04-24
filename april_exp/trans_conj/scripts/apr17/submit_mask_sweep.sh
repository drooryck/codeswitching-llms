#!/bin/bash
#SBATCH --job-name=tc_mask_apr17
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --output=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/april_exp/trans_conj/results/mask_sweep_apr17/logs/slurm_%A_%a.out
#SBATCH --error=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/april_exp/trans_conj/results/mask_sweep_apr17/logs/slurm_%A_%a.err

# ── Sweep grid ─────────────────────────────────────────────────────
#
# Fixed-budget design: total training examples = P = 262936 for all runs.
# SequentialSampler bug fixed — Trainer uses default RandomSampler.
# <pad> stripping added to inference.
#
# Primary question: does masking input tokens help conjugation accuracy?
# Secondary: interaction with prop (FR fraction) and trans_frac.
#
# Each array task handles one prop value.

PROPS=(0.01 0.1 0.5 0.9 0.99)
TRANS_FRACS=(0.0 0.1 0.5)
SEEDS=(1 2 3)
MASK_MODES=(mask nomask)
EVAL_PROP=0.1

PROP=${PROPS[$SLURM_ARRAY_TASK_ID]}
echo "[$(date)] Array task ${SLURM_ARRAY_TASK_ID}: prop=${PROP}"

# ── Paths ──────────────────────────────────────────────────────────
OUTPUT_ROOT=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/april_exp/trans_conj/results/mask_sweep_apr17
CONFIG_JSON=/n/home06/drooryck/codeswitching-llms/april_exp/trans_conj/configs/default_model.json
DATA_DIR=/n/home06/drooryck/codeswitching-llms/april_exp/trans_conj/data
LEXICON=/n/home06/drooryck/codeswitching-llms/april_exp/trans_conj/data/lexicon_sep22.json
PY_ENV=/n/home06/drooryck/envs/codeswitching-py310/bin/activate

# ── Setup ──────────────────────────────────────────────────────────
module load python/3.10.9-fasrc01
source "${PY_ENV}"
export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:${PYTHONPATH:-}

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/runs"

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
    cp -f "${CONFIG_JSON}" "${OUTPUT_ROOT}/model_config.json"
    cp -f "$0" "${OUTPUT_ROOT}/submitted_script.sh"
fi

# ── Run loop ───────────────────────────────────────────────────────
# Per array task: 2 masks × 3 trans_fracs × 3 seeds = 18 runs
# At ~25 min/run (1 epoch, 262K examples), ~7.5 hours total.

run_idx=0
total=18

for MODE in "${MASK_MODES[@]}"; do
    if [[ "${MODE}" == "nomask" ]]; then
        MASK_FLAG="--no-mask"
    else
        MASK_FLAG=""
    fi

    for TF in "${TRANS_FRACS[@]}"; do
        if [[ "${TF}" == "0.0" ]]; then
            LEVEL="none"
        else
            LEVEL="tense_separate"
        fi

        for SEED in "${SEEDS[@]}"; do
            RUN_NAME=$(printf "%s_%s_prop%s_tf%s_run%02d" \
                       "${MODE}" "${LEVEL}" "${PROP}" "${TF}" "${SEED}")
            RUN_DIR="${OUTPUT_ROOT}/runs/${RUN_NAME}"

            if [[ -f "${RUN_DIR}/test_predictions.csv" ]]; then
                echo "[$(date)] SKIP (already complete): ${RUN_NAME}"
                ((run_idx++))
                continue
            fi

            ((run_idx++))
            echo "[$(date)] Run ${run_idx}/${total}: ${RUN_NAME}"

            mkdir -p "${RUN_DIR}"

            python -m april_exp.trans_conj.src.run_single \
                --config "${CONFIG_JSON}" \
                --output-dir "${RUN_DIR}" \
                --data-dir "${DATA_DIR}" \
                --lexicon-path "${LEXICON}" \
                --prop "${PROP}" \
                --trans-frac "${TF}" \
                --translation-level "${LEVEL}" \
                --run-id "${SEED}" \
                --eval-prop "${EVAL_PROP}" \
                ${MASK_FLAG}

            echo "[$(date)] Finished: ${RUN_NAME}"
        done
    done
done

echo "[$(date)] Array task ${SLURM_ARRAY_TASK_ID} complete: ${run_idx}/${total} runs."
