"""Multi-metric timeseries plots for NL outputs: lexical, conformity, syntax, exact match.

One figure per tf value. Each figure has rows=props, cols=levels (tense_separate, full_sequence).
Solid = mask, dotted = nomask. Colors distinguish metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RUNS_ROOT = Path("/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/"
                 "april_exp/trans_conj/results/mask_sweep_apr14/runs")
PLOT_DIR = Path("/n/home06/drooryck/codeswitching-llms/"
                "april_exp/trans_conj/scripts/apr14/timeseries_plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODES = ["mask", "nomask"]
LEVELS = ["tense_separate", "full_sequence"]
PROPS = [0.01, 0.1, 0.5, 0.9, 0.99]
TFS = [0.01, 0.1, 0.5, 0.9, 0.99]

LANG_METRICS = {
    "nl": [
        ("nl_lexical",     "lexical",     "#e41a1c"),
        ("nl_conformity",  "conformity",  "#377eb8"),
        ("nl_syntax",      "syntax",      "#4daf4a"),
        ("nl_exact_match", "exact match", "#984ea3"),
    ],
    "fr": [
        ("fr_lexical",     "lexical",     "#e41a1c"),
        ("fr_conformity",  "conformity",  "#377eb8"),
        ("fr_syntax",      "syntax",      "#4daf4a"),
        ("fr_exact_match", "exact match", "#984ea3"),
    ],
}


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
    print(f"Loaded {len(records)} runs, {len(all_df)} total eval rows")
    return all_df


def plot_timeseries_for_tf(all_df: pd.DataFrame, tf_val: float, lang: str = "nl"):
    metrics = LANG_METRICS[lang]
    lang_upper = lang.upper()

    fig, axes = plt.subplots(len(PROPS), 2, figsize=(16, 3.2 * len(PROPS)),
                             sharex=False, sharey=True)

    for row_i, prop in enumerate(PROPS):
        for col_i, level in enumerate(LEVELS):
            ax = axes[row_i, col_i]

            for col, label, color in metrics:
                for mode in MODES:
                    sub = all_df[(all_df["mode"] == mode) &
                                 (all_df.level == level) &
                                 (all_df.prop == prop) &
                                 (all_df.tf == tf_val)].sort_values("step")
                    if sub.empty:
                        continue
                    ls = "-" if mode == "mask" else "--"
                    lw = 1.5
                    alpha = 1.0 if mode == "mask" else 0.85
                    ax.plot(sub.step, sub[col], color=color, ls=ls, lw=lw,
                            alpha=alpha)

            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel("score")
            if row_i == 0:
                ax.set_title(level, fontsize=12, fontweight="bold")
            ax.text(0.02, 0.95, f"prop={prop}", transform=ax.transAxes,
                    fontsize=10, va="top", bbox=dict(boxstyle="round,pad=0.2",
                    fc="wheat", alpha=0.8))
            if row_i == len(PROPS) - 1:
                ax.set_xlabel("Step")

    from matplotlib.lines import Line2D
    handles = []
    for _, label, color in metrics:
        handles.append(Line2D([0], [0], color=color, ls="-", lw=1.5, label=f"{label} (mask)"))
        handles.append(Line2D([0], [0], color=color, ls="--", lw=1.5, label=f"{label} (nomask)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 1.0), frameon=True)

    fig.suptitle(f"{lang_upper} Metrics Over Training — tf={tf_val}",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fname = f"{lang}_metrics_tf{tf_val}.png"
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def main():
    all_df = load_all()
    for lang in ("nl", "fr"):
        for tf in TFS:
            plot_timeseries_for_tf(all_df, tf, lang=lang)
    print(f"\nAll timeseries saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
