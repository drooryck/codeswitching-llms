#!/bin/bash
#SBATCH --job-name=nov11_sweep
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# remember to do:  mkdir -p /n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/results/nov11.0/logs .
# slurm directives cant expand variables and i dont want to have another wrapper script.

#SBATCH --output=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/results/nov11.1/logs/slurm_%A_%a.out
#SBATCH --error=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/results/nov11.1/logs/slurm_%A_%a.err


# PROPS=(0.000 0.001 0.005 0.010 0.015 0.020 0.025 0.030 0.040 0.050 0.075 0.100 0.150 0.200 0.250 0.300 0.400 0.450 0.500 0.550 0.600 0.650 0.700 0.750 0.800 0.850 0.900 0.925 0.950 0.960 0.970 0.980 0.985 0.990 0.995 0.999 1.000)
# RUNS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
PROPS=(0.5)
RUNS=(1)
EVAL_PROP=0.05

OUTPUT_ROOT=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/results/nov11.1
CONFIG_JSON=/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/configs/default_model.json
DATA_DIR=/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/data
LEXICON=/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/data/lexicon_sep22.json
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
from july_aug_sept_exp.src.dataset_manager import DatasetManager
from july_aug_sept_exp.src.metrics import Metrics
from july_aug_sept_exp.src.model_config import ModelConfig
from july_aug_sept_exp.src.experiment import Experiment

config = ModelConfig(**json.load(open("${RUN_DIR}/model_config.json")))
dm = DatasetManager("${DATA_DIR}", config, lexicon_path="${LEXICON}")
metrics = Metrics("${LEXICON}")
exp = Experiment(config, dm, metrics, Path("${RUN_DIR}"))
exp.run_single_super_debug(prop=${PROP}, run_id=${RUN_ID}, eval_prop=${EVAL_PROP})
PYCODE

    echo "[$(date)] Finished run ${run_idx}"
  done
done