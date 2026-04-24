#!/usr/bin/env python3
"""Generate metric plots for mar25 runs.

Per ablation type (none, subject, verb, object) creates:
  - syntax_score   (word order, 0=NL 1=FR)
  - lexical_score  (French token fraction, 0=NL 1=FR)  [stored as morphology_score]
  - alignment_score (avg of syntax + lexical)
  - structure_followed (follows FR / NL / either)

All plots: dual panel (French | Dutch), mean line + blue shaded ±1 SD.

Usage (from repo root, with venv activated):
  export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:$PYTHONPATH
  python feb_exp/scripts/mar25/plot.py
"""
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

NETSCRATCH_RESULTS = Path(
    "/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results"
)
REPO_ROOT = Path(__file__).resolve().parents[3]
ABLATION_TYPES = ("none", "subject", "verb", "object")


def load_results(runs_dir: Path, ablation_type: str) -> pd.DataFrame:
    """Load metrics from all run directories for a given ablation type."""
    results = []
    metrics_filename = f"ablation_{ablation_type}_metrics.json"

    for run_dir in sorted(runs_dir.glob("p*_run*")):
        metrics_file = run_dir / metrics_filename
        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        name = run_dir.name
        parts = name.split("_run")
        prop = float(parts[0][1:]) / 100.0
        run_id = int(parts[1])

        results.append({"prop": prop, "run_id": run_id, **metrics})

    if not results:
        raise ValueError(f"No results found in {runs_dir} for ablation={ablation_type}")

    return pd.DataFrame(results).sort_values("prop")


def plot_dual_panel_mean_sem(
    df: pd.DataFrame,
    fr_col: str,
    nl_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    ylim: tuple[float, float] = (0, 1),
    color: str = "#1f77b4",
) -> None:
    """Dual-panel plot with mean line and shaded ±1 SD band."""
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    def plot_panel(ax, col, subtitle):
        grouped = df.groupby("prop")[col]
        means = grouped.mean()
        sds = grouped.std()
        props = means.index.values

        ax.plot(props, means.values, color=color, linewidth=2.5, zorder=3)
        ax.fill_between(
            props,
            (means - sds).values,
            (means + sds).values,
            color=color,
            alpha=0.25,
            zorder=2,
        )
        ax.set_title(subtitle, fontsize=13, pad=10, fontweight="semibold")
        ax.set_xlabel("Proportion of French in Training Data")
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        ax.grid(True, linestyle="--", alpha=0.3)

    plot_panel(ax1, fr_col, "French test sentences")
    plot_panel(ax2, nl_col, "Dutch test sentences")

    n_runs = df.groupby("prop").size().iloc[0] if len(df) else 0
    fig.suptitle(
        f"{title}\n(mean ± SD, {n_runs} seeds per proportion)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_structure_followed(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    """Dual-panel structure plot: 3 curves (FR / NL / either) with mean + SD."""
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    series_spec = [
        ("follows_fr", "Follows FR structure", "#1f77b4"),
        ("follows_nl", "Follows NL structure", "#ff7f0e"),
        ("follows_either", "Follows either", "#2ca02c"),
    ]

    def plot_panel(ax, lang_prefix, subtitle):
        for suffix, label, color in series_spec:
            col = f"{lang_prefix}_{suffix}"
            grouped = df.groupby("prop")[col]
            means = grouped.mean()
            sds = grouped.std()
            props = means.index.values

            ax.plot(props, means.values, color=color, linewidth=2.5, label=label, zorder=3)
            ax.fill_between(
                props,
                (means - sds).values,
                (means + sds).values,
                color=color,
                alpha=0.18,
                zorder=2,
            )

        ax.set_title(subtitle, fontsize=13, pad=10, fontweight="semibold")
        ax.set_xlabel("Proportion of French in Training Data")
        ax.set_ylabel("Proportion of test sentences")
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=10)

    plot_panel(ax1, "fr", "French test sentences")
    plot_panel(ax2, "nl", "Dutch test sentences")

    n_runs = df.groupby("prop").size().iloc[0] if len(df) else 0
    fig.suptitle(
        f"{title}\n(mean ± SD, {n_runs} seeds per proportion)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def main():
    runs_dir = (NETSCRATCH_RESULTS / "mar25" / "runs").resolve()
    output_base = (REPO_ROOT / "feb_exp" / "results" / "mar25" / "sd_plots").resolve()

    if not runs_dir.exists():
        logger.error("Runs dir not found: %s", runs_dir)
        return

    for ablation_type in ABLATION_TYPES:
        out_dir = output_base / ablation_type
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            df = load_results(runs_dir, ablation_type)
        except ValueError as e:
            logger.warning("  %s", e)
            continue

        logger.info("Ablation %s: loaded %d runs", ablation_type, len(df))
        abl_label = f" — {ablation_type} ablation" if ablation_type != "none" else ""

        plot_dual_panel_mean_sem(
            df,
            fr_col="fr_syntax_score",
            nl_col="nl_syntax_score",
            title=f"Syntax Score (word order){abl_label}",
            ylabel="Syntax score (0=NL order, 1=FR order)",
            output_path=out_dir / f"syntax_score_{ablation_type}.png",
        )

        plot_dual_panel_mean_sem(
            df,
            fr_col="fr_morphology_score",
            nl_col="nl_morphology_score",
            title=f"Lexical Score (French token fraction){abl_label}",
            ylabel="Lexical score (0=all Dutch, 1=all French)",
            output_path=out_dir / f"lexical_score_{ablation_type}.png",
        )

        plot_dual_panel_mean_sem(
            df,
            fr_col="fr_alignment_score",
            nl_col="nl_alignment_score",
            title=f"Alignment Score (syntax + lexical avg){abl_label}",
            ylabel="Alignment score (0=Dutch, 1=French)",
            output_path=out_dir / f"alignment_score_{ablation_type}.png",
        )

        plot_structure_followed(
            df,
            title=f"Structure Followed (FR / NL / either){abl_label}",
            output_path=out_dir / f"structure_followed_{ablation_type}.png",
        )

    logger.info("All SD plots saved to %s", output_base)


if __name__ == "__main__":
    main()
