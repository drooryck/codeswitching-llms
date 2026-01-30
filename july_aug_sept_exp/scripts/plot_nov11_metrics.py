#!/usr/bin/env python3
"""
Aggregate Nov-11 lexical/syntax metrics across runs and create summary plots.

The script looks for either:
  1. a summary.csv in the results root (preferred), or
  2. the individual ablation_{type}_metrics.json files inside each run directory.

It filters to the ablation tag passed via --ablation and writes a two-panel plot
to OUTPUT_DIR/plots/nov11/, alongside a CSV snapshot of the aggregated data.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

# Ensure plotting works on headless login nodes.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_PATTERN = re.compile(r"p(?P<prop>\d+\.\d+)_run(?P<run>\d+)")


def load_metrics(run_dir: Path, ablation: str) -> Dict[str, float] | None:
    metrics_path = run_dir / f"ablation_{ablation}_metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open() as f:
        return json.load(f)


def collect_records(root: Path, ablation: str) -> pd.DataFrame:
    records: List[Dict[str, float]] = []

    summary_path = root / "summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        if not {"prop", "run_id", "ablation", "input_lang"}.issubset(df.columns):
            raise ValueError("summary.csv is missing required columns {prop, run_id, ablation, input_lang}")

        df = df[df["ablation"] == ablation]
        if df.empty:
            raise ValueError(f"No rows found in summary.csv for ablation='{ablation}'")

        for row in df.itertuples(index=False):
            prefix = f"{row.input_lang}_"
            records.append(
                {
                    "prop": row.prop,
                    "run_id": row.run_id,
                    "input_lang": row.input_lang,
                    "ablation": row.ablation,
                    "syntax_fr": getattr(row, f"{prefix}follows_fr"),
                    "syntax_nl": getattr(row, f"{prefix}follows_nl"),
                    "syntax_either": getattr(row, f"{prefix}follows_either"),
                    "lexical": getattr(row, f"{prefix}lexical_score"),
                    "pct_expected_len": getattr(row, f"{prefix}pct_expected_len", math.nan),
                }
            )
    else:
        for run_dir in sorted((root / "runs").glob("p*_run*")):
            match = RUN_PATTERN.match(run_dir.name)
            if not match:
                continue

            metrics = load_metrics(run_dir, ablation)
            if metrics is None:
                print(f"No metrics found in {run_dir}")
                continue

            prop = float(match.group("prop"))
            run_id = int(match.group("run"))

            for lang in ("fr", "nl"):
                prefix = f"{lang}_"
                try:
                    record = {
                        "prop": prop / 100.0,
                        "run_id": run_id,
                        "input_lang": lang,
                        "ablation": ablation,
                        "syntax_fr": metrics[f"{prefix}follows_fr"],
                        "syntax_nl": metrics[f"{prefix}follows_nl"],
                        "syntax_either": metrics[f"{prefix}follows_either"],
                        "lexical": metrics[f"{prefix}lexical_score"],
                        "pct_expected_len": metrics.get(f"{prefix}pct_expected_len", math.nan),
                    }
                except KeyError as exc:
                    missing = exc.args[0]
                    raise KeyError(
                        f"Missing key '{missing}' in {metrics_path}. "
                        "Ensure metrics.py supports lexical metrics before plotting."
                    ) from exc
                records.append(record)

    if not records:
        raise ValueError(f"No metrics found for ablation='{ablation}' in {root}")

    return pd.DataFrame.from_records(records)


def mean_sem(df: pd.DataFrame, metric: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    props = np.sort(df["prop"].unique())
    means = []
    sems = []
    for prop in props:
        values = df.loc[df["prop"] == prop, metric].dropna()
        means.append(values.mean() if len(values) else np.nan)
        sems.append(values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0)
    return props, np.array(means), np.array(sems)


def plot_metrics(df: pd.DataFrame, output_dir: Path) -> Path:
    colors = {
        "syntax_fr": "#1f77b4",
        "syntax_nl": "#ff7f0e",
        "syntax_either": "#2ca02c",
        "lexical": "#d62728",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, lang in zip(axes, ("fr", "nl")):
        subset = df[df["input_lang"] == lang]
        title = "French inputs" if lang == "fr" else "Dutch inputs"
        ax.set_title(title)
        ax.set_xlabel("Training French proportion")
        if lang == "fr":
            ax.set_ylabel("Mean score across runs")

        for metric, label in [
            ("syntax_fr", "French syntax"),
            ("syntax_nl", "Dutch syntax"),
            ("syntax_either", "Either syntax"),
            ("lexical", "Lexical FR share"),
        ]:
            props, means, sems = mean_sem(subset, metric)
            ax.errorbar(
                props,
                means,
                yerr=sems,
                marker="o",
                linewidth=2,
                capsize=4,
                color=colors[metric],
                label=label,
            )

        props, means, sems = mean_sem(subset, "pct_expected_len")
        ax.errorbar(
            props,
            means,
            yerr=sems,
            linestyle="--",
            linewidth=1,
            color="#7f7f7f",
            alpha=0.6,
            label="Correct length" if lang == "fr" else None,
        )

        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "nov11_metrics.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot aggregate syntax/lexical metrics.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Experiment output directory (contains runs/ subfolder).",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        help="Ablation tag (e.g. none, subject, verb, object).",
    )
    parser.add_argument(
        "--plots-subdir",
        type=str,
        default="plots/nov11",
        help="Where to store derived plots/CSVs within OUTPUT_DIR.",
    )
    args = parser.parse_args()

    records = collect_records(args.output_dir, args.ablation)

    plots_dir = args.output_dir / args.plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = plots_dir / f"nov11_metrics_{args.ablation}.csv"
    records.to_csv(raw_csv, index=False)

    plot_path = plot_metrics(records, plots_dir)

    print(f"Saved metrics snapshot to {raw_csv}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()

