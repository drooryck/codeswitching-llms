#!/bin/bash
#SBATCH --job-name=tc_april
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/april_exp/trans_conj/results/sweep/logs/slurm_%A_%a.out
#SBATCH --error=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/april_exp/trans_conj/results/sweep/logs/slurm_%A_%a.err

# prop now means overall French token fraction (not just conjugation FR frac)
PROPS=(0.0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 1.0)
TRANS_FRACS=(0.0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 1.0)
TRANSLATION_LEVELS=(tense_separate full_sequence)
RUN_ID=1
EVAL_PROP=0.1

OUTPUT_ROOT=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/april_exp/trans_conj/results/sweep
CONFIG_JSON=/n/home06/drooryck/codeswitching-llms/april_exp/trans_conj/configs/default_model.json
DATA_DIR=/n/home06/drooryck/codeswitching-llms/april_exp/trans_conj/data
LEXICON=/n/home06/drooryck/codeswitching-llms/april_exp/trans_conj/data/lexicon_sep22.json
PY_ENV=/n/home06/drooryck/envs/codeswitching-py310/bin/activate

module load python/3.10.9-fasrc01
source "${PY_ENV}"
export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:${PYTHONPATH:-}

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/runs"
cp -f "${CONFIG_JSON}" "${OUTPUT_ROOT}/model_config.json"
cp -f "$0" "${OUTPUT_ROOT}/submitted_script.sh"

run_idx=0
skipped=0
for LEVEL in "${TRANSLATION_LEVELS[@]}"; do
  for PROP in "${PROPS[@]}"; do
    for TF in "${TRANS_FRACS[@]}"; do

      # tf=0 → no translation, level is irrelevant; run only under tense_separate
      if [[ "${TF}" == "0.0" && "${LEVEL}" == "full_sequence" ]]; then
        continue
      fi

      # tf=1 → all translation, prop is irrelevant; run only for prop=0.5
      if [[ "${TF}" == "1.0" && "${PROP}" != "0.5" ]]; then
        continue
      fi

      RUN_DIR=$(printf "%s/runs/%s_prop%.2f_tf%.2f_run%02d" \
                "${OUTPUT_ROOT}" "${LEVEL}" "${PROP}" "${TF}" "${RUN_ID}")

      if [[ -d "${RUN_DIR}" ]]; then
        ((skipped++))
        continue
      fi

      RUN_DIR=$(printf "%s/runs/%s_prop%.3f_tf%.3f_run%02d" \
                "${OUTPUT_ROOT}" "${LEVEL}" "${PROP}" "${TF}" "${RUN_ID}")

      if [[ -d "${RUN_DIR}" ]]; then
        ((skipped++))
        continue
      fi

      ((run_idx++))
      echo "[$(date)] Run ${run_idx}: level=${LEVEL} prop=${PROP} tf=${TF}"

      mkdir -p "${RUN_DIR}"
      cp -f "${CONFIG_JSON}" "${RUN_DIR}/model_config.json"

      python - <<PYCODE
import json
from pathlib import Path
from april_exp.trans_conj.src.dataset_manager import DatasetManager
from april_exp.trans_conj.src.metrics import Metrics
from april_exp.trans_conj.src.model_config import ModelConfig
from april_exp.trans_conj.src.experiment import Experiment
from april_exp.trans_conj.src.translation import TranslationLevel

config = ModelConfig(**json.load(open("${RUN_DIR}/model_config.json")))
dm = DatasetManager("${DATA_DIR}", config, lexicon_path="${LEXICON}")
metrics = Metrics("${LEXICON}")
exp = Experiment(config, dm, metrics, Path("${RUN_DIR}"))
exp.run_single(
    prop=${PROP}, run_id=${RUN_ID},
    trans_frac=${TF},
    translation_level=TranslationLevel("${LEVEL}"),
    eval_prop=${EVAL_PROP},
)
PYCODE

      echo "[$(date)] Finished run ${run_idx}"
    done
  done
done

echo "[$(date)] All done: ${run_idx} new runs, ${skipped} skipped (already existed)."
