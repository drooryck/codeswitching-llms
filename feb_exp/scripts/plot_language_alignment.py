#!/usr/bin/env python3
"""Plot language alignment scores (syntax, morphology, combined) per ablation type.

Reads alignment_{ablation}.json files from each p*_run* directory and produces
side-by-side (FR / NL) plots matching the existing token_share plot format.

For each ablation type, produces three plots:
  - alignment_score_{ablation}.png  (combined score, 0=NL-aligned, 1=FR-aligned)
  - syntax_score_{ablation}.png     (word order component only)
  - morphology_score_{ablation}.png (token share component only)

Usage:
  python feb_exp/scripts/plot_language_alignment.py --mar4
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from feb_exp.src.plotting import BilingualPlotter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

NETSCRATCH_RESULTS = Path(
    "/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
MAR4_OUTPUT_ROOT = REPO_ROOT / "feb_exp" / "results" / "mar4"

MAR4_PRESETS = [
    ("mar4-v1-no-plurality", "mar4/version1_no_plurality_mixing"),
    ("mar4-v1-plurality-mixing", "mar4/version1_plurality_mixing"),
]
ABLATION_TYPES = ("none", "subject", "verb", "object")


def load_alignment_df(runs_dir: Path, ablation_type: str) -> pd.DataFrame:
    """Load alignment JSONs from all run dirs into a DataFrame."""
    results = []
    for run_dir in sorted(runs_dir.glob("p*_run*")):
        json_path = run_dir / f"alignment_{ablation_type}.json"
        if not json_path.exists():
            logger.warning("Missing %s in %s", json_path.name, run_dir.name)
            continue

        with open(json_path) as f:
            metrics = json.load(f)

        dir_name = run_dir.name
        parts = dir_name.split("_run")
        prop = float(parts[0][1:]) / 100.0
        run_id = int(parts[1])

        results.append({
            "prop": prop,
            "run_id": run_id,
            "ablation": ablation_type,
            **metrics,
        })

    if not results:
        raise ValueError(f"No alignment files found in {runs_dir} for ablation={ablation_type}")

    return pd.DataFrame(results).sort_values("prop")


def plot_alignment_metric(
    df: pd.DataFrame,
    fr_col: str,
    nl_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
):
    """Create a dual-panel plot matching the token_share style."""
    plotter = BilingualPlotter(df, output_path.parent)
    plotter._plot_dual_panel_with_runs(
        fr_col=fr_col,
        nl_col=nl_col,
        title=title,
        ylabel=ylabel,
        output_path=output_path,
        ylim=(0, 1),
    )


def main():
    parser = argparse.ArgumentParser(description="Plot language alignment scores.")
    parser.add_argument("--mar4", action="store_true", help="Plot for mar4 sweeps")
    parser.add_argument(
        "--runs-dir", type=Path, default=None,
        help="Custom runs dir (overrides --mar4)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=None,
        help="Output directory for plots",
    )
    args = parser.parse_args()

    if args.runs_dir:
        presets = [("custom", str(args.runs_dir))]
        output_root = args.output_dir or Path(".")
    elif args.mar4:
        presets = MAR4_PRESETS
        output_root = MAR4_OUTPUT_ROOT
    else:
        parser.error("Specify --mar4 or --runs-dir")

    output_root.mkdir(parents=True, exist_ok=True)

    for out_name, netscratch_name in presets:
        if args.runs_dir:
            runs_dir = args.runs_dir
        else:
            runs_dir = (NETSCRATCH_RESULTS / netscratch_name / "runs").resolve()

        output_dir = (output_root / out_name).resolve()
        if not runs_dir.exists():
            logger.warning("Skipping %s: runs dir not found: %s", out_name, runs_dir)
            continue
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=== %s -> %s ===", runs_dir, output_dir)

        for ablation_type in ABLATION_TYPES:
            try:
                df = load_alignment_df(runs_dir, ablation_type)
            except ValueError as e:
                logger.warning("  %s", e)
                continue

            n_runs = df.groupby("prop").size().iloc[0]
            logger.info("  Ablation %s: loaded %d runs (%d per proportion)",
                        ablation_type, len(df), n_runs)

            suffix = ablation_type

            plot_alignment_metric(
                df,
                fr_col="fr_alignment_score",
                nl_col="nl_alignment_score",
                title=f"Language alignment (syntax + morphology) - {suffix} ablation",
                ylabel="Alignment score (0=Dutch, 1=French)",
                output_path=output_dir / f"alignment_score_{suffix}.png",
            )

            plot_alignment_metric(
                df,
                fr_col="fr_syntax_score",
                nl_col="nl_syntax_score",
                title=f"Syntax score (word order) - {suffix} ablation",
                ylabel="Syntax score (0=NL order, 1=FR order)",
                output_path=output_dir / f"syntax_score_{suffix}.png",
            )

            plot_alignment_metric(
                df,
                fr_col="fr_morphology_score",
                nl_col="nl_morphology_score",
                title=f"Morphology score (token share) - {suffix} ablation",
                ylabel="Morphology score (0=Dutch tokens, 1=French tokens)",
                output_path=output_dir / f"morphology_score_{suffix}.png",
            )

        logger.info("  All plots saved to %s", output_dir)


if __name__ == "__main__":
    main()
