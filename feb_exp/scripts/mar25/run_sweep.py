#!/usr/bin/env python3
"""Mar25 sweep: 21 proportions × 20 seeds = 420 experiments.

Runs all experiments in a single Python process to avoid repeated
torch/transformers import overhead (~30s per run). Shared objects
(config, data_manager, metrics, tokenizer) are loaded once.

Data: feb_exp/data/full_data_mar25 (full plurality mixing, ~591K train)
Output: netscratch .../feb_exp/results/mar25/

Usage:
  export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:$PYTHONPATH
  python feb_exp/scripts/mar25/run_sweep.py [--reverse]
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch
from feb_exp.src.dataset_manager import DatasetManager
from feb_exp.src.metrics import Metrics
from feb_exp.src.model_config import ModelConfig
from feb_exp.src.experiment import Experiment

PROPS = [
    0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0,
]
RUNS = list(range(1, 21))
EVAL_PROP = 0.1

OUTPUT_ROOT = Path(
    "/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/mar25"
)
CONFIG_PATH = Path(__file__).parent / "model_config.json"
DATA_DIR = REPO_ROOT / "feb_exp" / "data" / "full_data_mar25"
LEXICON_PATH = REPO_ROOT / "feb_exp" / "data" / "lexicon_sep22.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reverse", action="store_true",
                        help="Sweep proportions from 1.0 down to 0.0")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "runs").mkdir(exist_ok=True)
    (OUTPUT_ROOT / "logs").mkdir(exist_ok=True)

    config = ModelConfig(**json.load(open(CONFIG_PATH)))
    data_manager = DatasetManager(str(DATA_DIR), config, lexicon_path=str(LEXICON_PATH))
    metrics = Metrics(str(LEXICON_PATH))

    tokenizer = data_manager.build_tokenizer()
    print(f"Shared objects ready  (vocab_size={len(tokenizer)})")

    props = list(reversed(PROPS)) if args.reverse else PROPS
    print(f"Sweep order: {'reversed (1.0→0.0)' if args.reverse else 'forward (0.0→1.0)'}")

    total = len(props) * len(RUNS)
    idx = 0
    skipped = 0
    t0 = time.time()

    for prop in props:
        for run_id in RUNS:
            idx += 1
            run_dir = OUTPUT_ROOT / "runs" / f"p{prop * 100:05.2f}_run{run_id:02d}"

            if (run_dir / "ablation_predictions.csv").exists():
                skipped += 1
                print(f"[{idx}/{total}] SKIP  prop={prop:.4f} run={run_id}")
                continue

            print(f"[{idx}/{total}] START prop={prop:.4f} run={run_id}")
            run_t0 = time.time()

            root = logging.getLogger()
            for h in root.handlers[:]:
                root.removeHandler(h)
                h.close()

            exp = Experiment(
                config=config,
                data_manager=data_manager,
                metrics=metrics,
                output_dir=run_dir,
            )
            exp.run_single(prop=prop, run_id=run_id, eval_prop=EVAL_PROP)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            elapsed = time.time() - run_t0
            print(f"[{idx}/{total}] DONE  prop={prop:.4f} run={run_id}  ({elapsed:.1f}s)")

    total_elapsed = time.time() - t0
    print(
        f"\nAll {total} runs complete ({skipped} skipped) "
        f"in {total_elapsed / 60:.1f} minutes"
    )


if __name__ == "__main__":
    main()
