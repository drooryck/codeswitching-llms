#!/usr/bin/env python3
"""Generate all metric plots (including alignment) for mar5 runs.

For each ablation type creates:
  - token_share, structure_followed  (existing)
  - syntax_score, morphology_score, alignment_score  (new)

Usage (from repo root, with venv activated):
  export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:$PYTHONPATH
  python feb_exp/scripts/mar5/plot.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from feb_exp.src.plotting import BilingualPlotter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

NETSCRATCH_RESULTS = Path(
    "/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results"
)
REPO_ROOT = Path(__file__).resolve().parents[3]
ABLATION_TYPES = ("none", "subject", "verb", "object")


def main():
    runs_dir = (NETSCRATCH_RESULTS / "mar5" / "runs").resolve()
    output_dir = (REPO_ROOT / "feb_exp" / "results" / "mar5" / "mar5-plurality-mixing").resolve()

    if not runs_dir.exists():
        logger.error("Runs dir not found: %s", runs_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== %s -> %s ===", runs_dir, output_dir)

    for ablation_type in ABLATION_TYPES:
        try:
            plotter = BilingualPlotter.create_plotter_from_run_metrics_dir(
                runs_dir, plots_subdir="plots", ablation_type=ablation_type
            )
        except ValueError as e:
            logger.warning("  %s", e)
            continue

        plotter.output_dir = output_dir
        logger.info(
            "  Ablation %s: loaded %d runs", ablation_type, len(plotter.results_df)
        )

        plotter.plot_token_share_by_proportion(suffix=ablation_type)
        plotter.plot_structure_followed_by_proportion(suffix=ablation_type)
        plotter.plot_syntax_score_by_proportion(suffix=ablation_type)
        plotter.plot_morphology_score_by_proportion(suffix=ablation_type)
        plotter.plot_alignment_score_by_proportion(suffix=ablation_type)

    logger.info("All plots saved to %s", output_dir)


if __name__ == "__main__":
    main()
