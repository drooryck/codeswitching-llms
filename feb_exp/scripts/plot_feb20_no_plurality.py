#!/usr/bin/env python3
"""Generate token-share and structure-followed plots from feb20-no-plurality runs.

Creates 8 plots: 2 per ablation type (none, subject, verb, object).
- Token share by proportion
- Structure followed by proportion (follows_fr, follows_nl, follows_either on same plot)

Usage (from repo root, with your venv activated):
  export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:$PYTHONPATH
  python -m feb_exp.scripts.plot_feb20_no_plurality -o feb_exp/results/feb22
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feb_exp.src.plotting import BilingualPlotter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RUNS_DIR = Path(
    "/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/feb20-no-plurality/runs"
)
PLURALITY_RUNS_DIR = Path(
    "/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/feb20/runs"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
PLURALITY_OUTPUT_DIR = REPO_ROOT / "feb_exp" / "results" / "feb20-plurality"
ABLATION_TYPES = ("none", "subject", "verb", "object")


def main():
    parser = argparse.ArgumentParser(description="Plot token share and structure followed per ablation type.")
    parser.add_argument(
        "runs_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_RUNS_DIR,
        help="Directory containing p*_run* subdirs with ablation_*_metrics.json",
    )
    parser.add_argument(
        "plots_subdir",
        type=str,
        nargs="?",
        default="../plots",
        help="Subdirectory for plots (relative to runs_dir) when -o not set",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Absolute directory to save all 8 plots (overrides plots_subdir if set)",
    )
    parser.add_argument(
        "--plurality",
        action="store_true",
        help="Use feb20/runs (plurality mixed data) and save to feb_exp/results/feb20-plurality",
    )
    args = parser.parse_args()

    if args.plurality:
        runs_dir = PLURALITY_RUNS_DIR.resolve()
        output_dir = PLURALITY_OUTPUT_DIR.resolve()
        logger.info("Using plurality data: %s -> %s", runs_dir, output_dir)
    else:
        runs_dir = args.runs_dir.resolve()
        output_dir = Path(args.output_dir).resolve() if args.output_dir else (runs_dir / args.plots_subdir).resolve()

    if not runs_dir.exists():
        logger.error("Runs directory does not exist: %s", runs_dir)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ablation_type in ABLATION_TYPES:
        plotter = BilingualPlotter.create_plotter_from_run_metrics_dir(
            runs_dir, plots_subdir=args.plots_subdir, ablation_type=ablation_type
        )
        plotter.output_dir = output_dir
        logger.info("Ablation %s: loaded %d runs", ablation_type, len(plotter.results_df))
        plotter.plot_token_share_by_proportion(suffix=ablation_type)
        plotter.plot_structure_followed_by_proportion(suffix=ablation_type)

    logger.info("All 8 plots saved to %s", output_dir)


if __name__ == "__main__":
    main()
