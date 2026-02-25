#!/bin/bash
#SBATCH --job-name=feb23_balanced_data_v2_plurality_mixing
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/feb23-v2-plurality-mixing/logs/slurm_%A_%a.out
#SBATCH --error=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/feb23-v2-plurality-mixing/logs/slurm_%A_%a.err

# Proportion sweep 10% through 100%, with plurality mixing (main data)
PROPS=(0 0.5 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0)
RUNS=(1)
# PROPS=(0.5)
# RUNS=(1)
EVAL_PROP=0.1

OUTPUT_ROOT=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/feb23-v2-plurality-mixing
CONFIG_JSON=/n/home06/drooryck/codeswitching-llms/feb_exp/configs/default_model.json
DATA_DIR=/n/home06/drooryck/codeswitching-llms/feb_exp/data/balanced_data_feb23/version2_plurality_mixing
LEXICON=/n/home06/drooryck/codeswitching-llms/feb_exp/data/lexicon_sep22.json
PY_ENV=/n/home06/drooryck/envs/codeswitching-py310/bin/activate

module load python/3.10.9-fasrc01
source "${PY_ENV}"
export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:${PYTHONPATH:-}

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/plots" "${OUTPUT_ROOT}/runs"
cp -f "${CONFIG_JSON}" "${OUTPUT_ROOT}/model_config.json"
cp -f "$0" "${OUTPUT_ROOT}/submitted_script.sh"

run_idx=0
for PROP in "${PROPS[@]}"; do
  for RUN_ID in "${RUNS[@]}"; do
    ((run_idx++))
    echo "[$(date)] Starting run ${run_idx}: prop=${PROP}, run_id=${RUN_ID}"

    RUN_DIR=$(printf "%s/runs/p%05.2f_run%02d" "${OUTPUT_ROOT}" "$(echo "${PROP} * 100" | bc)" "${RUN_ID}")
    mkdir -p "${RUN_DIR}"
    cp -f "${CONFIG_JSON}" "${RUN_DIR}/model_config.json"

    python - <<PYCODE
import json
from pathlib import Path
from feb_exp.src.dataset_manager import DatasetManager
from feb_exp.src.metrics import Metrics
from feb_exp.src.model_config import ModelConfig
from feb_exp.src.experiment import Experiment

config = ModelConfig(**json.load(open("${RUN_DIR}/model_config.json")))
dm = DatasetManager("${DATA_DIR}", config, lexicon_path="${LEXICON}")
metrics = Metrics("${LEXICON}")
exp = Experiment(config, dm, metrics, Path("${RUN_DIR}"))
exp.run_single(prop=${PROP}, run_id=${RUN_ID}, eval_prop=${EVAL_PROP})
PYCODE

    echo "[$(date)] Finished run ${run_idx}"
  done
done
