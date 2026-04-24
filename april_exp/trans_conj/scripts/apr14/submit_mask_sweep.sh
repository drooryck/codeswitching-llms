#!/bin/bash
#SBATCH --job-name=mask_sweep_apr14
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/april_exp/trans_conj/results/mask_sweep_apr14/logs/slurm_%j.out
#SBATCH --error=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/april_exp/trans_conj/results/mask_sweep_apr14/logs/slurm_%j.err

PROPS=(0.01 0.1 0.5 0.9 0.99)
TRANS_FRACS=(0.01 0.1 0.5 0.9 0.99)
TRANSLATION_LEVELS=(tense_separate full_sequence)
MASK_MODES=(mask nomask)
RUN_ID=1
EVAL_PROP=0.1

OUTPUT_ROOT=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/april_exp/trans_conj/results/mask_sweep_apr14
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

# Count total runs
total=0
for MODE in "${MASK_MODES[@]}"; do
  for LEVEL in "${TRANSLATION_LEVELS[@]}"; do
    for PROP in "${PROPS[@]}"; do
      for TF in "${TRANS_FRACS[@]}"; do
        ((total++))
      done
    done
  done
done
echo "[$(date)] Total runs planned: ${total}"

run_idx=0
for MODE in "${MASK_MODES[@]}"; do
  if [[ "${MODE}" == "nomask" ]]; then
    MASK_FLAG="--no-mask"
  else
    MASK_FLAG=""
  fi

  for LEVEL in "${TRANSLATION_LEVELS[@]}"; do
    for PROP in "${PROPS[@]}"; do
      for TF in "${TRANS_FRACS[@]}"; do

        RUN_NAME=$(printf "%s_%s_prop%s_tf%s_run%02d" \
                   "${MODE}" "${LEVEL}" "${PROP}" "${TF}" "${RUN_ID}")
        RUN_DIR="${OUTPUT_ROOT}/runs/${RUN_NAME}"

        if [[ -f "${RUN_DIR}/test_predictions.csv" ]]; then
          echo "[$(date)] SKIP (already complete): ${RUN_NAME}"
          ((run_idx++))
          continue
        fi

        ((run_idx++))
        echo "[$(date)] Run ${run_idx}/${total}: ${RUN_NAME}"

        mkdir -p "${RUN_DIR}"
        cp -f "${CONFIG_JSON}" "${RUN_DIR}/model_config.json"

        python -m april_exp.trans_conj.src.run_single \
            --config "${RUN_DIR}/model_config.json" \
            --output-dir "${RUN_DIR}" \
            --data-dir "${DATA_DIR}" \
            --lexicon-path "${LEXICON}" \
            --prop "${PROP}" \
            --trans-frac "${TF}" \
            --translation-level "${LEVEL}" \
            --run-id "${RUN_ID}" \
            --eval-prop "${EVAL_PROP}" \
            ${MASK_FLAG}

        echo "[$(date)] Finished: ${RUN_NAME}"
      done
    done
  done
done

echo "[$(date)] All done: ${run_idx}/${total} runs."
