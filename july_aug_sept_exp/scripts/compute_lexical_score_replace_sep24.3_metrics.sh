#!/bin/bash
#SBATCH --job-name=lexical_refresh_sep24
#SBATCH --partition=seas_compute
#SBATCH --account=dam_lab
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/scripts/logs/lexical_refresh_%j.out
#SBATCH --error=/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/scripts/logs/lexical_refresh_%j.err

# Refresh all existing metrics.json files for sep24.3 using the updated lexical metric.

module load python/3.10.9-fasrc01
source /n/home06/drooryck/envs/codeswitching-py310/bin/activate

export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:${PYTHONPATH}

python <<'PYTHON_SCRIPT'
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed

from july_aug_sept_exp.src.metrics import Metrics

RESULTS_ROOT = Path("/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/results/sep24.3")
RESUME_FROM = {"p99.9_run1", "p99.9_run2", "p99.9_run3", "p99.9_run4", "p99.9_run5",
               "p100.0_run1", "p100.0_run2", "p100.0_run3", "p100.0_run4", "p100.0_run5"}
LEXICON_PATH = Path("/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/data/lexicon_sep22.json")
ABLA_TIONS = ["none", "subject", "verb", "object"]
N_JOBS = 8

def process_run(run_dir: Path) -> dict:
    """Recompute metrics for every ablation in a run directory."""
    print(f"[START] {run_dir.name}", flush=True)

    predictions_path = run_dir / "ablation_predictions.csv"
    if not predictions_path.exists():
        print(f"[SKIP ] {run_dir.name}: missing ablation_predictions.csv", flush=True)
        return {"run": run_dir.name, "status": "missing_predictions"}

    # Load predictions once per run.
    df = pd.read_csv(predictions_path)

    # Instantiate Metrics per process to avoid shared-state surprises.
    metrics_obj = Metrics(LEXICON_PATH)

    for ablation in ABLA_TIONS:
        subset = df[df["ablation"] == ablation]
        if subset.empty:
            print(f"  [WARN] {run_dir.name}: no rows for ablation={ablation}", flush=True)
            continue

        predictions = [
            {
                "language": row.language,
                "prediction": row.prediction,
                "gold": row.gold,
                "input": row.input,
            }
            for row in subset.itertuples(index=False)
        ]

        try:
            metrics = metrics_obj.compute_all_metrics(predictions, ablation_type=ablation)
        except Exception as exc:  # surface lexical/OOV issues immediately
            print(f"  [ERROR] {run_dir.name}: ablation={ablation} | {exc}", flush=True)
            raise

        output_path = run_dir / f"ablation_{ablation}_metrics.json"
        metrics_obj.save_metrics(metrics, output_path)
        print(f"  [DONE] {run_dir.name}: ablation={ablation} -> {output_path.name}", flush=True)

    print(f"[END ] {run_dir.name}", flush=True)
    return {"run": run_dir.name, "status": "ok"}


def main():
    run_dirs = sorted(
        d for d in RESULTS_ROOT.iterdir()
        if d.is_dir() and d.name.startswith("p") and d.name in RESUME_FROM
    )
    if not run_dirs:
        print(f"No run directories found in {RESULTS_ROOT}")
        return

    print(f"Found {len(run_dirs)} run directories. Starting recomputation...")

    summaries = Parallel(n_jobs=N_JOBS)(
        delayed(process_run)(run_dir) for run_dir in run_dirs
    )

    num_ok = sum(1 for s in summaries if s.get("status") == "ok")
    num_missing = sum(1 for s in summaries if s.get("status") == "missing_predictions")

    print("\n===== Summary =====")
    print(f"Successful recomputations: {num_ok}")
    print(f"Runs missing predictions : {num_missing}")
    print("====================")


if __name__ == "__main__":
    main()
PYTHON_SCRIPT

