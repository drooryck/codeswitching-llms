#!/usr/bin/env python3
"""Generate all metric plots for mar5 runs, organized by ablation type in separate folders.

For each ablation type (none, subject, verb, object) creates a subfolder with:
  - token_share.png
  - structure_followed.png
  - syntax_score.png
  - morphology_score.png
  - alignment_score.png

All plots show the average (mean) as a bold line + lighter lines for individual runs (3 per proportion).

Usage (from repo root, with venv activated):
  export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:$PYTHONPATH
  python feb_exp/scripts/mar5/plot_all.py
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
    output_root = (REPO_ROOT / "feb_exp" / "results" / "mar5" / "mar5-plurality-mixing").resolve()

    if not runs_dir.exists():
        logger.error("Runs dir not found: %s", runs_dir)
        return

    output_root.mkdir(parents=True, exist_ok=True)
    logger.info("Runs: %s", runs_dir)
    logger.info("Output root: %s", output_root)

    for ablation_type in ABLATION_TYPES:
        abl_dir = output_root / ablation_type
        abl_dir.mkdir(parents=True, exist_ok=True)

        try:
            plotter = BilingualPlotter.create_plotter_from_run_metrics_dir(
                runs_dir, plots_subdir="plots", ablation_type=ablation_type
            )
        except ValueError as e:
            logger.warning("  Ablation %s: %s", ablation_type, e)
            continue

        plotter.output_dir = abl_dir
        n_runs_per_prop = plotter.results_df.groupby("prop").size()
        logger.info(
            "  Ablation %s: loaded %d rows (%d proportions, %d-%d runs each)",
            ablation_type,
            len(plotter.results_df),
            len(n_runs_per_prop),
            n_runs_per_prop.min(),
            n_runs_per_prop.max(),
        )

        plotter.plot_token_share_by_proportion(suffix=ablation_type)
        plotter.plot_structure_followed_by_proportion(suffix=ablation_type)
        plotter.plot_syntax_score_by_proportion(suffix=ablation_type)
        plotter.plot_morphology_score_by_proportion(suffix=ablation_type)
        plotter.plot_alignment_score_by_proportion(suffix=ablation_type)

        logger.info("  -> Saved 5 plots to %s", abl_dir)

    logger.info("All plots saved under %s", output_root)


if __name__ == "__main__":
    main()
