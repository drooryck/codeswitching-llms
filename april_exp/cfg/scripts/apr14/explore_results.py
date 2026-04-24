"""
Evaluation & plotting for the CFG apr14 sweep.

Run from the repo root:
    python april_exp/cfg/scripts/apr14/explore_results.py

Outputs PDFs and PNGs to the results directory.
"""
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Configuration ──────────────────────────────────────────────────
RESULTS_ROOT = Path(
    "/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/"
    "april_exp/cfg/results/sweep_apr14/runs"
)
FIGS_DIR = RESULTS_ROOT.parent / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_COLS = [
    "en_lexical", "nl_lexical",
    "en_syntax", "nl_syntax",
    "en_conformity", "nl_conformity",
    "en_pos_validity", "nl_pos_validity",
    "en_pos_coverage", "nl_pos_coverage",
    "en_exact_match", "nl_exact_match",
]

NICE_NAMES = {
    "lexical": "Lexical Score",
    "syntax": "Syntax Score",
    "conformity": "Conformity Score",
    "pos_validity": "POS Validity",
    "pos_coverage": "POS Coverage",
    "exact_match": "Exact Match",
}


def savefig(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIGS_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"  Saved {name}")


# =====================================================================
# 1. LOAD DATA
# =====================================================================

def load_all_runs():
    """Load eval_sample_metrics.csv from every completed run."""
    all_metrics = []
    run_summary = []

    for run_dir in sorted(RESULTS_ROOT.iterdir()):
        m = re.match(
            r"mask_(\w+)_prop([\d.]+)_tf([\d.]+)_run(\d+)", run_dir.name
        )
        if not m:
            continue
        level = m.group(1)
        prop = float(m.group(2))
        tf = float(m.group(3))
        run_id = int(m.group(4))

        metrics_file = run_dir / "eval_sample_metrics.csv"
        if not metrics_file.exists():
            continue

        df = pd.read_csv(metrics_file)
        if df.empty:
            continue

        df["level"] = level
        df["prop"] = prop
        df["tf"] = tf
        df["run_id"] = run_id
        all_metrics.append(df)

        final = df.iloc[-1]
        row = {
            "level": level, "prop": prop, "tf": tf, "run_id": run_id,
            "final_step": int(final["step"]),
        }
        for lang in ("en", "nl"):
            for metric in ("lexical", "syntax", "conformity",
                           "pos_validity", "pos_coverage", "exact_match"):
                col = f"{lang}_{metric}"
                if col in df.columns:
                    row[f"{lang}_{metric}_final"] = final[col]
                    if metric in ("syntax", "lexical"):
                        row[f"{lang}_{metric}_peak"] = df[col].max()
        run_summary.append(row)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    summary_df = pd.DataFrame(run_summary)

    print(f"Loaded {len(summary_df)} runs, {len(metrics_df)} metric rows")
    print(f"Props:  {sorted(summary_df['prop'].unique())}")
    print(f"TFs:    {sorted(summary_df['tf'].unique())}")
    print(f"Seeds:  {sorted(summary_df['run_id'].unique())}")
    return metrics_df, summary_df


def load_test_predictions():
    """Load test_predictions.csv from every completed run (final eval)."""
    all_preds = []
    for run_dir in sorted(RESULTS_ROOT.iterdir()):
        m = re.match(
            r"mask_(\w+)_prop([\d.]+)_tf([\d.]+)_run(\d+)", run_dir.name
        )
        if not m:
            continue
        pred_file = run_dir / "test_predictions.csv"
        if not pred_file.exists():
            continue
        df = pd.read_csv(pred_file)
        df["level"] = m.group(1)
        df["prop"] = float(m.group(2))
        df["tf"] = float(m.group(3))
        df["run_id"] = int(m.group(4))
        all_preds.append(df)
    if not all_preds:
        return pd.DataFrame()
    return pd.concat(all_preds, ignore_index=True)


def load_wandb_logs():
    """Load trainer_state.json training loss from each run."""
    import json
    all_logs = []
    for run_dir in sorted(RESULTS_ROOT.iterdir()):
        m = re.match(
            r"mask_(\w+)_prop([\d.]+)_tf([\d.]+)_run(\d+)", run_dir.name
        )
        if not m:
            continue
        state_file = run_dir / "checkpoint-last" / "trainer_state.json"
        if not state_file.exists():
            state_file = run_dir / "trainer_state.json"
        if not state_file.exists():
            for ckpt_dir in sorted(run_dir.glob("checkpoint-*")):
                candidate = ckpt_dir / "trainer_state.json"
                if candidate.exists():
                    state_file = candidate
                    break
        if not state_file.exists():
            continue
        with open(state_file) as f:
            state = json.load(f)
        for entry in state.get("log_history", []):
            row = {
                "level": m.group(1), "prop": float(m.group(2)),
                "tf": float(m.group(3)), "run_id": int(m.group(4)),
            }
            row.update(entry)
            all_logs.append(row)
    if not all_logs:
        return pd.DataFrame()
    return pd.DataFrame(all_logs)


# =====================================================================
# 2. METRICS OVER TIME BY PROPORTION (avg ± SD over seeds)
# =====================================================================

def plot_metrics_over_time(metrics_df: pd.DataFrame,
                           metric: str = "syntax",
                           fixed_tf: Optional[float] = None):
    """
    For each prop, plot the metric over training steps.
    Average over seeds, shaded SD.
    """
    df = metrics_df.copy()
    if fixed_tf is not None:
        df = df[df["tf"] == fixed_tf]
    else:
        df = df[df["tf"] == 0.0]

    props = sorted(df["prop"].unique())
    cmap = plt.cm.coolwarm
    colors = [cmap(i / max(len(props) - 1, 1)) for i in range(len(props))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, lang, title in zip(axes, ["en", "nl"], ["English", "Dutch"]):
        col = f"{lang}_{metric}"
        if col not in df.columns:
            continue
        for color, prop in zip(colors, props):
            sub = df[df["prop"] == prop].groupby("step")[col]
            mean = sub.mean()
            std = sub.std().fillna(0)
            ax.plot(mean.index, mean.values, color=color,
                    label=f"p={prop:.2f}", linewidth=1.2)
            ax.fill_between(mean.index, mean.values - std.values,
                            mean.values + std.values, color=color, alpha=0.15)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(NICE_NAMES.get(metric, metric))
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                   fontsize=7, framealpha=0.9)
    tf_label = f"tf={fixed_tf}" if fixed_tf is not None else "tf=0"
    fig.suptitle(f"{NICE_NAMES.get(metric, metric)} over time ({tf_label})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, f"metrics_over_time_{metric}_tf{fixed_tf or 0}")
    plt.close(fig)


# =====================================================================
# 3. HEATMAPS: final metric value as prop × trans_frac
# =====================================================================

def plot_heatmaps(summary_df: pd.DataFrame):
    """Heatmaps of final metric values: prop × tf, averaged over seeds."""
    metric_pairs = [
        ("nl_syntax_final", "NL Syntax (final)"),
        ("en_syntax_final", "EN Syntax (final)"),
        ("nl_exact_match_final", "NL Exact Match (final)"),
        ("en_exact_match_final", "EN Exact Match (final)"),
        ("nl_conformity_final", "NL Conformity (final)"),
        ("en_conformity_final", "EN Conformity (final)"),
    ]
    available = [(col, name) for col, name in metric_pairs
                 if col in summary_df.columns]
    if not available:
        print("No final metrics available for heatmaps.")
        return

    n = len(available)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, (col, name) in zip(axes, available):
        pivot = summary_df.groupby(["prop", "tf"])[col].mean().unstack("tf")
        vmin, vmax = (0, 1) if "exact_match" in col else (None, None)
        sns.heatmap(pivot, ax=ax, vmin=vmin, vmax=vmax,
                    cmap="YlOrRd" if "syntax" in col or "lexical" in col else "RdYlGn",
                    annot=True, fmt=".2f", annot_kws={"size": 7})
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("trans_frac")
        ax.set_ylabel("prop (EN fraction)")

    for ax in axes[len(available):]:
        ax.set_visible(False)

    fig.suptitle("Final Metrics: prop × trans_frac (avg over seeds)", fontsize=14, y=1.01)
    fig.tight_layout()
    savefig(fig, "heatmaps_final_metrics")
    plt.close(fig)


# =====================================================================
# 4. LOSS CURVES
# =====================================================================

def plot_loss_curves(logs_df: pd.DataFrame,
                     fixed_tf: Optional[float] = None):
    """Training and eval loss over time, by prop."""
    if logs_df.empty:
        print("No training logs found — skipping loss curves.")
        return

    df = logs_df.copy()
    if fixed_tf is not None:
        df = df[df["tf"] == fixed_tf]
    else:
        df = df[df["tf"] == 0.0]

    props = sorted(df["prop"].unique())
    cmap = plt.cm.coolwarm
    colors = [cmap(i / max(len(props) - 1, 1)) for i in range(len(props))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    ax = axes[0]
    train_df = df.dropna(subset=["loss"])
    for color, prop in zip(colors, props):
        sub = train_df[train_df["prop"] == prop].groupby("step")["loss"]
        mean = sub.mean()
        std = sub.std().fillna(0)
        ax.plot(mean.index, mean.values, color=color,
                label=f"p={prop:.2f}", linewidth=1)
        ax.fill_between(mean.index, mean.values - std.values,
                        mean.values + std.values, color=color, alpha=0.1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    # Eval loss
    ax = axes[1]
    eval_df = df.dropna(subset=["eval_loss"])
    for color, prop in zip(colors, props):
        sub = eval_df[eval_df["prop"] == prop].groupby("step")["eval_loss"]
        mean = sub.mean()
        std = sub.std().fillna(0)
        if len(mean) == 0:
            continue
        ax.plot(mean.index, mean.values, color=color,
                label=f"p={prop:.2f}", linewidth=1)
        ax.fill_between(mean.index, mean.values - std.values,
                        mean.values + std.values, color=color, alpha=0.1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Eval Loss")
    ax.set_title("Eval Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=7, framealpha=0.9)

    tf_label = f"tf={fixed_tf}" if fixed_tf is not None else "tf=0"
    fig.suptitle(f"Loss curves ({tf_label})", fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, f"loss_curves_tf{fixed_tf or 0}")
    plt.close(fig)


# =====================================================================
# 5. RESULTS BY STRUCTURE TYPE (from test predictions)
# =====================================================================

def plot_by_structure(preds_df: pd.DataFrame, metrics_obj=None):
    """Compute and plot exact match & metrics by structure type."""
    if preds_df.empty or "structure" not in preds_df.columns:
        print("No test predictions with structure info — skipping.")
        return

    df = preds_df.copy()
    df["em"] = (df["prediction"] == df["gold"]).astype(float)

    # If we have the metrics object, compute scores
    if metrics_obj is not None:
        df["lexical"] = df["prediction"].apply(metrics_obj.lexical_score)
        df["syntax"] = df["prediction"].apply(
            lambda p: metrics_obj.syntax_score(
                p, df.loc[df["prediction"] == p, "structure"].iloc[0]
            )["score"]
        )
    else:
        for col in ("lexical", "syntax"):
            if col not in df.columns:
                df[col] = np.nan

    structures = sorted(df["structure"].unique())
    props = sorted(df["prop"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row_idx, lang in enumerate(["en", "nl"]):
        lang_df = df[df["language"] == lang]

        # Exact match by structure
        ax = axes[row_idx, 0]
        pivot = lang_df.groupby(["prop", "structure"])["em"].mean().unstack("structure")
        pivot.plot(ax=ax, marker="o", markersize=3, linewidth=1)
        ax.set_title(f"{'English' if lang == 'en' else 'Dutch'} — Exact Match by Structure")
        ax.set_xlabel("prop")
        ax.set_ylabel("Exact Match")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        # Token-level exact match distribution
        ax = axes[row_idx, 1]
        em_by_struct = lang_df.groupby(["structure", "prop"])["em"].mean().reset_index()
        for struct in structures:
            sub = em_by_struct[em_by_struct["structure"] == struct]
            ax.plot(sub["prop"], sub["em"], marker="o", markersize=3,
                    label=struct, linewidth=1)
        ax.set_title(f"{'English' if lang == 'en' else 'Dutch'} — EM vs prop by structure")
        ax.set_xlabel("prop")
        ax.set_ylabel("Exact Match")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Prediction length distribution
        ax = axes[row_idx, 2]
        lang_df_copy = lang_df.copy()
        lang_df_copy["pred_len"] = lang_df_copy["prediction"].apply(
            lambda x: len(str(x).split()))
        lang_df_copy["gold_len"] = lang_df_copy["gold"].apply(
            lambda x: len(str(x).split()))
        lang_df_copy["len_diff"] = lang_df_copy["pred_len"] - lang_df_copy["gold_len"]
        for struct in structures:
            sub = lang_df_copy[lang_df_copy["structure"] == struct]
            mean_diff = sub.groupby("prop")["len_diff"].mean()
            ax.plot(mean_diff.index, mean_diff.values, marker="o",
                    markersize=3, label=struct, linewidth=1)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.set_title(f"{'English' if lang == 'en' else 'Dutch'} — Pred length error by struct")
        ax.set_xlabel("prop")
        ax.set_ylabel("Mean (pred_len - gold_len)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Results by Structure Type (tf=0, avg over seeds)", fontsize=14, y=1.01)
    fig.tight_layout()
    savefig(fig, "results_by_structure")
    plt.close(fig)


# =====================================================================
# 6. ADDITIONAL INTERESTING ANALYSES
# =====================================================================

def plot_codeswitching_dynamics(metrics_df: pd.DataFrame):
    """
    Plot how NL lexical/syntax scores evolve with prop,
    identifying codeswitching patterns.
    """
    df = metrics_df[metrics_df["tf"] == 0.0].copy()
    if df.empty:
        return

    props = sorted(df["prop"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: NL lexical over time by prop
    ax = axes[0, 0]
    cmap = plt.cm.coolwarm
    colors = [cmap(i / max(len(props) - 1, 1)) for i in range(len(props))]
    for color, prop in zip(colors, props):
        sub = df[df["prop"] == prop].groupby("step")["nl_lexical"]
        mean = sub.mean()
        ax.plot(mean.index, mean.values, color=color,
                label=f"p={prop:.2f}", linewidth=1)
    ax.set_title("NL Lexical Score over time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Lexical Score (1=EN, 0=NL)")
    ax.grid(True, alpha=0.3)

    # Panel 2: EN lexical over time by prop
    ax = axes[0, 1]
    for color, prop in zip(colors, props):
        sub = df[df["prop"] == prop].groupby("step")["en_lexical"]
        mean = sub.mean()
        ax.plot(mean.index, mean.values, color=color,
                label=f"p={prop:.2f}", linewidth=1)
    ax.set_title("EN Lexical Score over time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Lexical Score (1=EN, 0=NL)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: Final NL syntax vs prop (each seed as a dot)
    ax = axes[1, 0]
    final = df.groupby(["prop", "run_id"]).last().reset_index()
    for metric, marker, label in [
        ("nl_syntax", "o", "NL Syntax"),
        ("nl_conformity", "s", "NL Conformity"),
        ("nl_exact_match", "^", "NL Exact Match"),
    ]:
        if metric not in final.columns:
            continue
        mean = final.groupby("prop")[metric].mean()
        std = final.groupby("prop")[metric].std().fillna(0)
        ax.errorbar(mean.index, mean.values, yerr=std.values,
                    marker=marker, label=label, capsize=3, linewidth=1)
    ax.set_title("Final NL metrics vs prop (tf=0)")
    ax.set_xlabel("prop (EN fraction)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Same for EN
    ax = axes[1, 1]
    for metric, marker, label in [
        ("en_syntax", "o", "EN Syntax"),
        ("en_conformity", "s", "EN Conformity"),
        ("en_exact_match", "^", "EN Exact Match"),
    ]:
        if metric not in final.columns:
            continue
        mean = final.groupby("prop")[metric].mean()
        std = final.groupby("prop")[metric].std().fillna(0)
        ax.errorbar(mean.index, mean.values, yerr=std.values,
                    marker=marker, label=label, capsize=3, linewidth=1)
    ax.set_title("Final EN metrics vs prop (tf=0)")
    ax.set_xlabel("prop (EN fraction)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Codeswitching Dynamics", fontsize=14, y=1.01)
    fig.tight_layout()
    savefig(fig, "codeswitching_dynamics")
    plt.close(fig)


def plot_translation_effect(metrics_df: pd.DataFrame):
    """
    At fixed props (0.25, 0.5, 0.75), show how trans_frac
    affects final syntax and exact match.
    """
    target_props = [0.25, 0.5, 0.75]
    df = metrics_df.copy()

    fig, axes = plt.subplots(2, len(target_props),
                             figsize=(6 * len(target_props), 8),
                             sharey="row")

    for col_idx, prop in enumerate(target_props):
        sub = df[df["prop"] == prop]
        if sub.empty:
            continue
        final = sub.groupby(["tf", "run_id"]).last().reset_index()
        tfs = sorted(final["tf"].unique())

        for row_idx, (lang, title) in enumerate(
            [("nl", "Dutch"), ("en", "English")]
        ):
            ax = axes[row_idx, col_idx]
            for metric, marker, label in [
                (f"{lang}_syntax", "o", "Syntax"),
                (f"{lang}_conformity", "s", "Conformity"),
                (f"{lang}_exact_match", "^", "Exact Match"),
            ]:
                if metric not in final.columns:
                    continue
                mean = final.groupby("tf")[metric].mean()
                std = final.groupby("tf")[metric].std().fillna(0)
                ax.errorbar(mean.index, mean.values, yerr=std.values,
                            marker=marker, label=label, capsize=3, linewidth=1)
            ax.set_title(f"{title} (prop={prop})")
            ax.set_xlabel("trans_frac")
            ax.grid(True, alpha=0.3)
            if col_idx == 0:
                ax.set_ylabel("Score")
            if col_idx == len(target_props) - 1:
                ax.legend(fontsize=7)

    fig.suptitle("Effect of Translation Fraction on Final Metrics", fontsize=14, y=1.01)
    fig.tight_layout()
    savefig(fig, "translation_effect")
    plt.close(fig)


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 60)
    print("CFG apr14 sweep — Evaluation")
    print("=" * 60)

    metrics_df, summary_df = load_all_runs()
    if metrics_df.empty:
        print("No completed runs found. Exiting.")
        return

    print(f"\n{'─' * 40}")
    print("1. Metrics over time by proportion")
    print(f"{'─' * 40}")
    for metric in ("lexical", "syntax", "conformity", "exact_match",
                    "pos_validity", "pos_coverage"):
        plot_metrics_over_time(metrics_df, metric=metric, fixed_tf=0.0)
        if 0.5 in metrics_df["tf"].unique():
            plot_metrics_over_time(metrics_df, metric=metric, fixed_tf=0.5)

    print(f"\n{'─' * 40}")
    print("2. Heatmaps (final metrics)")
    print(f"{'─' * 40}")
    plot_heatmaps(summary_df)

    print(f"\n{'─' * 40}")
    print("3. Loss curves")
    print(f"{'─' * 40}")
    logs_df = load_wandb_logs()
    plot_loss_curves(logs_df, fixed_tf=0.0)
    if 0.5 in metrics_df["tf"].unique():
        plot_loss_curves(logs_df, fixed_tf=0.5)

    print(f"\n{'─' * 40}")
    print("4. Results by structure type")
    print(f"{'─' * 40}")
    preds_df = load_test_predictions()
    plot_by_structure(preds_df)

    print(f"\n{'─' * 40}")
    print("5. Codeswitching dynamics")
    print(f"{'─' * 40}")
    plot_codeswitching_dynamics(metrics_df)

    print(f"\n{'─' * 40}")
    print("6. Translation effect")
    print(f"{'─' * 40}")
    plot_translation_effect(metrics_df)

    print(f"\n{'=' * 60}")
    print(f"All figures saved to: {FIGS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
