"""
Aggregate and plot ablation results across props, seeds, and structures.

Processes one file at a time to avoid OOM. Aggregates per-pair deltas
into (prop, seed, source_lang, structure, ablation) level means, then
plots.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms"
                   "/april_exp/cfg/results/ablation/results")
OUTPUT_DIR = Path(__file__).resolve().parent


def aggregate_one_file(path: Path) -> pd.DataFrame:
    """Load one file, compute deltas, return aggregated rows."""
    df = pd.read_csv(path)

    # Separate baseline and ablated
    baseline = df[df["ablation"] == "none"].set_index(
        ["source_lang", "pair_idx"]
    )[["lexical_score", "syntax_score"]].rename(
        columns={"lexical_score": "base_lex", "syntax_score": "base_syn"})

    ablated = df[df["ablation"] != "none"].copy()
    ablated = ablated.join(baseline, on=["source_lang", "pair_idx"])
    ablated["delta_lex"] = ablated["lexical_score"] - ablated["base_lex"]

    # Aggregate: mean delta per (prop, seed, source_lang, structure, ablation)
    agg = ablated.groupby(
        ["prop", "seed", "source_lang", "structure", "ablation"]
    ).agg(
        mean_delta_lex=("delta_lex", "mean"),
        mean_lex=("lexical_score", "mean"),
        n=("delta_lex", "count"),
    ).reset_index()

    # Also get baseline lexical per group
    base_agg = df[df["ablation"] == "none"].groupby(
        ["prop", "seed", "source_lang", "structure"]
    )["lexical_score"].mean().reset_index().rename(
        columns={"lexical_score": "base_lex_mean"})

    # Add "none" rows with delta=0
    none_rows = base_agg.copy()
    none_rows["ablation"] = "none"
    none_rows["mean_delta_lex"] = 0.0
    none_rows["mean_lex"] = none_rows["base_lex_mean"]
    none_rows["n"] = len(df[df["ablation"] == "none"])

    agg = agg.merge(base_agg, on=["prop", "seed", "source_lang", "structure"], how="left")
    result = pd.concat([agg, none_rows], ignore_index=True)

    del df, ablated, baseline
    return result


def load_all_aggregated() -> pd.DataFrame:
    """Load all result files, aggregate each, combine."""
    dfs = []
    for f in sorted(RESULTS_DIR.glob("ablation_prop*.csv")):
        print(f"  Processing {f.name}...")
        dfs.append(aggregate_one_file(f))
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total aggregated rows: {len(df)}")
    return df


def plot_main(agg: pd.DataFrame):
    """Line plot: Δ lexical by ablation type across prop, panels for EN/NL."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, src_lang, title in zip(axes, ["en", "nl"],
                                   ["EN test sentences", "NL test sentences"]):
        sub = agg[(agg["source_lang"] == src_lang) & (agg["ablation"] != "none")]
        # Average across structures for each (prop, seed, ablation)
        seed_level = sub.groupby(["prop", "seed", "ablation"])["mean_delta_lex"].mean().reset_index()
        # Then average across seeds
        summary = seed_level.groupby(["prop", "ablation"]).agg(
            mean=("mean_delta_lex", "mean"),
            std=("mean_delta_lex", "std"),
        ).reset_index()

        colors = {"subject": "#e41a1c", "verb": "#377eb8", "object": "#4daf4a"}
        markers = {"subject": "o", "verb": "s", "object": "^"}

        for abl_type in ["subject", "verb", "object"]:
            s = summary[summary["ablation"] == abl_type].sort_values("prop")
            ax.plot(s["prop"], s["mean"], color=colors[abl_type],
                    marker=markers[abl_type], label=abl_type, linewidth=2, markersize=6)
            ax.fill_between(s["prop"],
                            s["mean"] - s["std"],
                            s["mean"] + s["std"],
                            color=colors[abl_type], alpha=0.15)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Training proportion (EN)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)

    axes[0].set_ylabel("Δ Lexical Score\n(ablated − baseline)", fontsize=12)
    fig.suptitle("Effect of Cross-Lingual Ablation on Output Language",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTPUT_DIR / "ablation_delta_lexical.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_by_structure(agg: pd.DataFrame):
    """Faceted plot: one row per structure, Δ lexical by ablation type."""
    structures = sorted(agg["structure"].unique())
    n_struct = len(structures)

    fig, axes = plt.subplots(n_struct, 2, figsize=(14, 3 * n_struct), sharey=True)
    if n_struct == 1:
        axes = axes.reshape(1, -1)

    colors = {"subject": "#e41a1c", "verb": "#377eb8", "object": "#4daf4a"}
    markers = {"subject": "o", "verb": "s", "object": "^"}

    for row_idx, struct in enumerate(structures):
        for col_idx, (src_lang, lang_label) in enumerate(
            [("en", "EN test"), ("nl", "NL test")]
        ):
            ax = axes[row_idx, col_idx]
            sub = agg[(agg["source_lang"] == src_lang) &
                      (agg["structure"] == struct) &
                      (agg["ablation"] != "none")]

            summary = sub.groupby(["prop", "ablation"]).agg(
                mean=("mean_delta_lex", "mean"),
                std=("mean_delta_lex", "std"),
            ).reset_index()

            for abl_type in ["subject", "verb", "object"]:
                s = summary[summary["ablation"] == abl_type].sort_values("prop")
                ax.plot(s["prop"], s["mean"], color=colors[abl_type],
                        marker=markers[abl_type], label=abl_type,
                        linewidth=1.5, markersize=4)
                ax.fill_between(s["prop"],
                                s["mean"] - s["std"],
                                s["mean"] + s["std"],
                                color=colors[abl_type], alpha=0.12)

            ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
            if row_idx == 0:
                ax.set_title(lang_label, fontsize=12)
            if row_idx == n_struct - 1:
                ax.set_xlabel("Training prop (EN)", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(struct.replace("_", " "), fontsize=10, fontweight="bold")
            if row_idx == 0 and col_idx == 1:
                ax.legend(fontsize=8)

    fig.suptitle("Ablation Effect by Structure Type",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = OUTPUT_DIR / "ablation_by_structure.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_absolute_lexical(agg: pd.DataFrame):
    """Show absolute lexical score under each ablation condition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, src_lang, title in zip(axes, ["en", "nl"],
                                   ["EN test sentences", "NL test sentences"]):
        sub = agg[agg["source_lang"] == src_lang]
        # Compute absolute lexical: for "none" use base_lex_mean, for ablated use mean_lex
        sub = sub.copy()
        sub["lex_val"] = np.where(sub["ablation"] == "none",
                                  sub["base_lex_mean"], sub["mean_lex"])

        seed_level = sub.groupby(["prop", "seed", "ablation"])["lex_val"].mean().reset_index()
        summary = seed_level.groupby(["prop", "ablation"]).agg(
            mean=("lex_val", "mean"),
            std=("lex_val", "std"),
        ).reset_index()

        colors = {"none": "#333333", "subject": "#e41a1c",
                  "verb": "#377eb8", "object": "#4daf4a"}
        markers = {"none": "D", "subject": "o", "verb": "s", "object": "^"}
        labels = {"none": "original", "subject": "ablate subject",
                  "verb": "ablate verb", "object": "ablate object"}

        for abl_type in ["none", "subject", "verb", "object"]:
            s = summary[summary["ablation"] == abl_type].sort_values("prop")
            ax.plot(s["prop"], s["mean"], color=colors[abl_type],
                    marker=markers[abl_type], label=labels[abl_type],
                    linewidth=2, markersize=6)
            ax.fill_between(s["prop"],
                            s["mean"] - s["std"],
                            s["mean"] + s["std"],
                            color=colors[abl_type], alpha=0.12)

        ax.set_xlabel("Training proportion (EN)", fontsize=12)
        ax.set_ylabel("Lexical Score (EN=1, NL=0)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Output Language Under Cross-Lingual Ablation",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTPUT_DIR / "ablation_absolute_lexical.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def save_summary_table(agg: pd.DataFrame):
    """Save aggregate summary as CSV."""
    # Overall (across structures)
    abl = agg[agg["ablation"] != "none"]
    seed_level = abl.groupby(
        ["prop", "seed", "source_lang", "ablation"]
    )["mean_delta_lex"].mean().reset_index()
    summary = seed_level.groupby(["prop", "source_lang", "ablation"]).agg(
        mean_delta_lex=("mean_delta_lex", "mean"),
        std_delta_lex=("mean_delta_lex", "std"),
    ).reset_index()
    out = OUTPUT_DIR / "ablation_summary.csv"
    summary.to_csv(out, index=False, float_format="%.4f")
    print(f"Saved: {out}")

    # By structure
    summary2 = abl.groupby(
        ["prop", "source_lang", "structure", "ablation"]
    ).agg(
        mean_delta_lex=("mean_delta_lex", "mean"),
        std_delta_lex=("mean_delta_lex", "std"),
    ).reset_index()
    out2 = OUTPUT_DIR / "ablation_by_structure_summary.csv"
    summary2.to_csv(out2, index=False, float_format="%.4f")
    print(f"Saved: {out2}")


def main():
    print("Loading and aggregating results...")
    agg = load_all_aggregated()
    print(f"Props: {sorted(agg.prop.unique())}")
    print(f"Seeds: {sorted(agg.seed.unique())}")
    print(f"Structures: {sorted(agg.structure.unique())}")

    print("\nGenerating plots...")
    plot_main(agg)
    plot_by_structure(agg)
    plot_absolute_lexical(agg)
    save_summary_table(agg)

    # Print quick overview
    print("\n" + "=" * 70)
    print("SUMMARY: Mean Δ lexical by ablation type (averaged over all props)")
    print("=" * 70)
    abl = agg[agg["ablation"] != "none"]
    for src_lang in ["en", "nl"]:
        print(f"\n── {src_lang.upper()} test sentences ──")
        sub = abl[abl["source_lang"] == src_lang]
        overview = sub.groupby("ablation")["mean_delta_lex"].agg(
            ["mean", "count"]
        ).reindex(["subject", "verb", "object"])
        overview["abs_mean"] = overview["mean"].abs()
        print(overview.to_string(float_format="%.4f"))

    # Detailed table: by prop, for NL test sentences
    print("\n" + "=" * 70)
    print("NL test: Mean Δ lexical by prop and ablation type")
    print("=" * 70)
    nl = abl[abl["source_lang"] == "nl"]
    seed_level = nl.groupby(["prop", "seed", "ablation"])["mean_delta_lex"].mean().reset_index()
    table = seed_level.groupby(["prop", "ablation"])["mean_delta_lex"].mean().unstack("ablation")
    table = table[["subject", "verb", "object"]]
    print(table.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
