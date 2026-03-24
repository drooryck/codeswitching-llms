#!/usr/bin/env python3
"""Mar5 sweep: run all 185 experiments in a single Python process.

Avoids re-importing torch/transformers/wandb for each run (~30s startup
saved per run). Shared objects (config, data_manager, metrics, tokenizer)
are loaded once and reused across all (prop, run_id) combinations.

Usage:
  export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:$PYTHONPATH
  python feb_exp/scripts/mar5/run_sweep.py
"""
import json
import logging
import subprocess
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

# feb20 proportions (37 values) × 5 seeds = 185 runs
PROPS = [
    0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05,
    0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5, 0.55,
    0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.96,
    0.97, 0.98, 0.985, 0.99, 0.995, 0.999, 1.0,
]
RUNS = [1, 2, 3]
EVAL_PROP = 0.1

OUTPUT_ROOT = Path(
    "/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/mar5"
)
CONFIG_PATH = Path(__file__).parent / "model_config.json"
DATA_DIR = REPO_ROOT / "feb_exp" / "data" / "balanced_data_feb23" / "version1_plurality_mixing"
LEXICON_PATH = REPO_ROOT / "feb_exp" / "data" / "lexicon_sep22.json"


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "runs").mkdir(exist_ok=True)
    (OUTPUT_ROOT / "logs").mkdir(exist_ok=True)

    # --- Load shared objects once ---
    config = ModelConfig(**json.load(open(CONFIG_PATH)))
    data_manager = DatasetManager(str(DATA_DIR), config, lexicon_path=str(LEXICON_PATH))
    metrics = Metrics(str(LEXICON_PATH))

    tokenizer = data_manager.build_tokenizer()
    print(f"Shared objects ready  (vocab_size={len(tokenizer)})")

    total = len(PROPS) * len(RUNS)
    idx = 0
    skipped = 0
    t0 = time.time()

    for prop in PROPS:
        for run_id in RUNS:
            idx += 1
            run_dir = OUTPUT_ROOT / "runs" / f"p{prop * 100:05.2f}_run{run_id:02d}"

            if (run_dir / "ablation_predictions.csv").exists():
                skipped += 1
                print(f"[{idx}/{total}] SKIP  prop={prop:.4f} run={run_id}")
                continue

            print(f"[{idx}/{total}] START prop={prop:.4f} run={run_id}")
            run_t0 = time.time()

            # Clear root-logger handlers so Experiment._setup_logging
            # can attach a fresh FileHandler for this run's log file.
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

    # --- Plotting ---
    print("Generating plots...")
    subprocess.run(
        [sys.executable, str(Path(__file__).parent / "plot.py")],
        check=True,
    )
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
