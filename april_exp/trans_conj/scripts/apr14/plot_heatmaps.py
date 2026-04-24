"""Generate 10 heatmap plots (5 metrics × 2 languages) for the mask sweep."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RUNS_ROOT = Path("/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/"
                 "april_exp/trans_conj/results/mask_sweep_apr14/runs")
PLOT_DIR = Path("/n/home06/drooryck/codeswitching-llms/"
                "april_exp/trans_conj/scripts/apr14/heatmap_plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODES = ["mask", "nomask"]
LEVELS = ["tense_separate", "full_sequence"]
PROPS = [0.01, 0.1, 0.5, 0.9, 0.99]
TFS = [0.01, 0.1, 0.5, 0.9, 0.99]


def load_all() -> pd.DataFrame:
    records = []
    for mode in MODES:
        for level in LEVELS:
            for prop in PROPS:
                for tf in TFS:
                    name = f"{mode}_{level}_prop{prop}_tf{tf}_run01"
                    f = RUNS_ROOT / name / "eval_sample_metrics.csv"
                    if not f.exists():
                        continue
                    df = pd.read_csv(f)
                    df["mode"] = mode
                    df["level"] = level
                    df["prop"] = prop
                    df["tf"] = tf
                    records.append(df)
    all_df = pd.concat(records, ignore_index=True)
    all_df = all_df.rename(columns={
        "nl_morphology": "nl_lexical", "fr_morphology": "fr_lexical"})
    all_df["nl_alignment"] = (all_df["nl_lexical"] + all_df["nl_syntax"]) / 2
    all_df["fr_alignment"] = (all_df["fr_lexical"] + all_df["fr_syntax"]) / 2
    print(f"Loaded {len(records)} runs, {len(all_df)} total eval rows")
    return all_df


def plot_metric_heatmap(final: pd.DataFrame, col: str, title: str, fname: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    panels = [
        ("mask", "tense_separate", axes[0, 0]),
        ("mask", "full_sequence", axes[0, 1]),
        ("nomask", "tense_separate", axes[1, 0]),
        ("nomask", "full_sequence", axes[1, 1]),
    ]

    vmin, vmax = 0, 1

    for mode, level, ax in panels:
        subset = final[(final["mode"] == mode) & (final.level == level)]
        if subset.empty:
            ax.set_title(f"{mode} — {level}\n(no data)")
            continue
        pivot = subset.pivot_table(
            index="prop", columns="tf", values=col, aggfunc="mean")
        pivot = pivot.reindex(index=PROPS, columns=TFS)
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlBu_r",
                    vmin=vmin, vmax=vmax, ax=ax,
                    xticklabels=[str(t) for t in TFS],
                    yticklabels=[str(p) for p in PROPS])
        ax.set_title(f"{mode} — {level}")
        ax.set_xlabel("trans_frac")
        ax.set_ylabel("prop (FR conj frac)")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def main():
    all_df = load_all()

    final = all_df.loc[all_df.groupby(
        ["mode", "level", "prop", "tf"])["step"].idxmax()]

    metrics = [
        ("nl_lexical",     "NL Lexical Score (FR token fraction in NL outputs)",   "nl_lexical.png"),
        ("fr_lexical",     "FR Lexical Score (FR token fraction in FR outputs)",   "fr_lexical.png"),
        ("nl_syntax",      "NL Syntax Score (word-order adherence, NL outputs)",   "nl_syntax.png"),
        ("fr_syntax",      "FR Syntax Score (word-order adherence, FR outputs)",   "fr_syntax.png"),
        ("nl_conformity",  "NL Conformity (structural validity, NL outputs)",      "nl_conformity.png"),
        ("fr_conformity",  "FR Conformity (structural validity, FR outputs)",      "fr_conformity.png"),
        ("nl_exact_match", "NL Exact Match",                                       "nl_exact_match.png"),
        ("fr_exact_match", "FR Exact Match",                                       "fr_exact_match.png"),
        ("nl_alignment",   "NL Alignment (mean of lexical + syntax, NL outputs)",  "nl_alignment.png"),
        ("fr_alignment",   "FR Alignment (mean of lexical + syntax, FR outputs)",  "fr_alignment.png"),
    ]

    for col, title, fname in metrics:
        plot_metric_heatmap(final, col, title, fname)

    print(f"\nAll 10 heatmaps saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
