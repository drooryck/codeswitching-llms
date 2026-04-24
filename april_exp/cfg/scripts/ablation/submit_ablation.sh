#!/bin/bash
#SBATCH --job-name=cfg_ablation
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-26
#SBATCH --output=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/april_exp/cfg/results/ablation/logs/slurm_%A_%a.out
#SBATCH --error=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/april_exp/cfg/results/ablation/logs/slurm_%A_%a.err

# 27 tasks: 9 props × 3 seeds (tf=0 only, from apr14 sweep)
PROPS=(0 0.01 0.1 0.25 0.5 0.75 0.9 0.99 1.0)
SEEDS=(1 2 3)

PROP_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
PROP=${PROPS[$PROP_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: prop=${PROP} seed=${SEED}"

# ── Paths ──────────────────────────────────────────────────────────
SWEEP_ROOT=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/april_exp/cfg/results/sweep_apr14/runs
RUN_NAME=$(printf "mask_none_prop%s_tf0_run%02d" "${PROP}" "${SEED}")
MODEL_DIR="${SWEEP_ROOT}/${RUN_NAME}"

DATA_DIR=/n/home06/drooryck/codeswitching-llms/april_exp/cfg/data
LEXICON=/n/home06/drooryck/codeswitching-llms/april_exp/cfg/data/lexicon.json
OUTPUT_DIR=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/april_exp/cfg/results/ablation/results

PY_ENV=/n/home06/drooryck/envs/codeswitching-py310/bin/activate

# ── Setup ──────────────────────────────────────────────────────────
module load python/3.10.9-fasrc01
source "${PY_ENV}"
export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:${PYTHONPATH:-}

mkdir -p "${OUTPUT_DIR}"
mkdir -p /n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/april_exp/cfg/results/ablation/logs

# ── Check model exists ─────────────────────────────────────────────
if [[ ! -d "${MODEL_DIR}/final" ]]; then
    echo "ERROR: Model not found at ${MODEL_DIR}/final"
    exit 1
fi

OUT_FILE="${OUTPUT_DIR}/ablation_prop${PROP}_seed${SEED}.csv"
if [[ -f "${OUT_FILE}" ]]; then
    echo "SKIP: Output already exists at ${OUT_FILE}"
    exit 0
fi

# ── Run ablation ───────────────────────────────────────────────────
echo "[$(date)] Running ablation for ${RUN_NAME}..."

python -m april_exp.cfg.scripts.ablation.run_ablation \
    --model-dir "${MODEL_DIR}" \
    --data-dir "${DATA_DIR}" \
    --lexicon-path "${LEXICON}" \
    --output-dir "${OUTPUT_DIR}" \
    --prop "${PROP}" \
    --seed "${SEED}"

echo "[$(date)] Done: prop=${PROP} seed=${SEED}"
