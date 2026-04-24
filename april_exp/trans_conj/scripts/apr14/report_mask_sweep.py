"""
Comprehensive analysis of mask vs no-mask sweep (apr14).

100 runs: 5 props × 5 tfs × 2 levels × 2 mask modes.
Generates plots and prints summary statistics.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

RUNS_ROOT = Path("/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/"
                 "april_exp/trans_conj/results/mask_sweep_apr14/runs")
PLOT_DIR = Path("/n/home06/drooryck/codeswitching-llms/"
                "april_exp/trans_conj/scripts/apr14/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

PROPS = [0.01, 0.1, 0.5, 0.9, 0.99]
TFS = [0.01, 0.1, 0.5, 0.9, 0.99]
LEVELS = ["tense_separate", "full_sequence"]
MODES = ["mask", "nomask"]


def load_all_runs():
    records = []
    missing = []
    for mode in MODES:
        for level in LEVELS:
            for prop in PROPS:
                for tf in TFS:
                    name = f"{mode}_{level}_prop{prop}_tf{tf}_run01"
                    f = RUNS_ROOT / name / "eval_sample_metrics.csv"
                    if not f.exists():
                        missing.append(name)
                        continue
                    df = pd.read_csv(f)
                    df["mode"] = mode
                    df["level"] = level
                    df["prop"] = prop
                    df["tf"] = tf
                    df["run_name"] = name
                    records.append(df)
    if missing:
        print(f"WARNING: {len(missing)} runs missing")
        for m in missing[:5]:
            print(f"  {m}")
    all_df = pd.concat(records, ignore_index=True)
    all_df = all_df.rename(columns={
        "nl_morphology": "nl_lexical", "fr_morphology": "fr_lexical"})
    print(f"Loaded {len(records)} runs, {len(all_df)} total eval rows")
    return all_df


def get_final(all_df):
    return all_df.loc[all_df.groupby("run_name").step.idxmax()].copy()


def plot_heatmaps(final, metric, title, fname, cmap="RdYlBu_r", vmin=0, vmax=1):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ci, level in enumerate(LEVELS):
        for ri, mode in enumerate(MODES):
            ax = axes[ri, ci]
            sub = final[(final.level == level) & (final["mode"] == mode)]
            piv = sub.pivot_table(index="prop", columns="tf",
                                  values=metric, aggfunc="mean")
            piv = piv.reindex(index=PROPS, columns=TFS)
            im = ax.imshow(piv.values, cmap=cmap, vmin=vmin, vmax=vmax,
                           aspect="auto", origin="lower")
            ax.set_xticks(range(len(TFS)))
            ax.set_xticklabels([str(v) for v in TFS])
            ax.set_yticks(range(len(PROPS)))
            ax.set_yticklabels([str(v) for v in PROPS])
            ax.set_xlabel("trans_frac")
            ax.set_ylabel("prop (FR conj frac)")
            ax.set_title(f"{mode} — {level}", fontsize=11)
            for i in range(len(PROPS)):
                for j in range(len(TFS)):
                    val = piv.values[i, j]
                    if not np.isnan(val):
                        color = "black" if 0.3 < val < 0.7 else "white"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=8, color=color)
            plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_diff_heatmaps(final, metric, title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ci, level in enumerate(LEVELS):
        ax = axes[ci]
        mask_piv = final[(final.level == level) & (final["mode"] == "mask")].pivot_table(
            index="prop", columns="tf", values=metric, aggfunc="mean")
        nomask_piv = final[(final.level == level) & (final["mode"] == "nomask")].pivot_table(
            index="prop", columns="tf", values=metric, aggfunc="mean")
        mask_piv = mask_piv.reindex(index=PROPS, columns=TFS)
        nomask_piv = nomask_piv.reindex(index=PROPS, columns=TFS)
        diff = nomask_piv - mask_piv
        vals = diff.values[~np.isnan(diff.values)]
        if len(vals) == 0:
            continue
        vmax = max(abs(vals.min()), abs(vals.max()), 0.005)
        im = ax.imshow(diff.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto", origin="lower")
        ax.set_xticks(range(len(TFS)))
        ax.set_xticklabels([str(v) for v in TFS])
        ax.set_yticks(range(len(PROPS)))
        ax.set_yticklabels([str(v) for v in PROPS])
        ax.set_xlabel("trans_frac")
        ax.set_ylabel("prop (FR conj frac)")
        ax.set_title(f"{level}", fontsize=11)
        for i in range(len(PROPS)):
            for j in range(len(TFS)):
                val = diff.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                            fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_timeseries_grid(all_df, metric, ylabel, title, fname, tf_fixed=0.5):
    fig, axes = plt.subplots(len(PROPS), 2, figsize=(16, 4 * len(PROPS)),
                             sharex=True)
    for row, prop in enumerate(PROPS):
        for col, level in enumerate(LEVELS):
            ax = axes[row, col]
            for mode, color, ls in [("mask", "tab:blue", "-"),
                                    ("nomask", "tab:red", "--")]:
                sub = all_df[(all_df.level == level) & (all_df.prop == prop) &
                             (all_df.tf == tf_fixed) & (all_df["mode"] == mode)]
                if len(sub) > 0:
                    ax.plot(sub.step, sub[metric], color=color, ls=ls,
                            linewidth=1.5, label=mode, alpha=0.8)
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(f"{level}", fontsize=12)
                ax.legend(fontsize=9)
            ax.text(0.02, 0.95, f"prop={prop}", transform=ax.transAxes,
                    fontsize=10, va="top", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="wheat", alpha=0.7))
            if row == len(PROPS) - 1:
                ax.set_xlabel("Step")
    fig.suptitle(f"{title} (tf={tf_fixed})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_scatter_mask_vs_nomask(final, metric, label, fname):
    mask_f = final[final["mode"] == "mask"][["level", "prop", "tf", metric]].copy()
    mask_f = mask_f.set_index(["level", "prop", "tf"]).rename(columns={metric: "mask"})
    nomask_f = final[final["mode"] == "nomask"][["level", "prop", "tf", metric]].copy()
    nomask_f = nomask_f.set_index(["level", "prop", "tf"]).rename(columns={metric: "nomask"})
    joined = mask_f.join(nomask_f, how="inner")

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    for level, marker in [("tense_separate", "o"), ("full_sequence", "s")]:
        sub = joined.loc[level]
        ax.scatter(sub["mask"], sub["nomask"], marker=marker, s=60, alpha=0.7,
                   label=level)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel(f"{label} (masked)", fontsize=12)
    ax.set_ylabel(f"{label} (no-mask)", fontsize=12)
    ax.set_title(f"Masked vs No-Mask: {label}", fontsize=13)
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_convergence_speed(all_df, fname):
    """How quickly does each run reach 95% exact match?"""
    thresh = 0.95
    records = []
    for name, grp in all_df.groupby("run_name"):
        row = grp.iloc[0]
        for lang in ("fr", "nl"):
            col = f"{lang}_exact_match"
            above = grp[grp[col] >= thresh]
            step = int(above.step.min()) if len(above) > 0 else np.nan
            records.append({
                "mode": row["mode"], "level": row["level"],
                "prop": row["prop"], "tf": row["tf"],
                "lang": lang, "converge_step": step,
            })
    conv = pd.DataFrame(records)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ci, level in enumerate(LEVELS):
        for ri, lang in enumerate(["fr", "nl"]):
            ax = axes[ri, ci]
            for mode, color, offset in [("mask", "tab:blue", -0.15),
                                        ("nomask", "tab:red", 0.15)]:
                sub = conv[(conv.level == level) & (conv.lang == lang) &
                           (conv["mode"] == mode) & (conv.tf == 0.5)]
                sub = sub.sort_values("prop")
                ax.bar([i + offset for i in range(len(PROPS))],
                       sub.converge_step.values, width=0.28,
                       color=color, alpha=0.7, label=mode)
            ax.set_xticks(range(len(PROPS)))
            ax.set_xticklabels([str(p) for p in PROPS])
            ax.set_xlabel("prop")
            ax.set_ylabel("Step to 95% exact match")
            ax.set_title(f"{level} — {lang.upper()} outputs")
            ax.grid(True, alpha=0.3, axis="y")
            if ri == 0 and ci == 0:
                ax.legend()
    fig.suptitle("Convergence Speed: Step to 95% Exact Match (tf=0.5)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_peak_codeswitching(all_df, fname):
    """Peak NL lexical score (max FR fraction in NL outputs) across training."""
    records = []
    for name, grp in all_df.groupby("run_name"):
        row = grp.iloc[0]
        peak = grp.nl_lexical.max()
        peak_step = int(grp.loc[grp.nl_lexical.idxmax(), "step"])
        records.append({
            "mode": row["mode"], "level": row["level"],
            "prop": row["prop"], "tf": row["tf"],
            "peak_nl_lex": peak, "peak_step": peak_step,
        })
    peaks = pd.DataFrame(records)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ci, level in enumerate(LEVELS):
        for ri, mode in enumerate(MODES):
            ax = axes[ri, ci]
            sub = peaks[(peaks.level == level) & (peaks["mode"] == mode)]
            piv = sub.pivot_table(index="prop", columns="tf",
                                  values="peak_nl_lex", aggfunc="mean")
            piv = piv.reindex(index=PROPS, columns=TFS)
            im = ax.imshow(piv.values, cmap="hot_r", vmin=0, vmax=0.5,
                           aspect="auto", origin="lower")
            ax.set_xticks(range(len(TFS)))
            ax.set_xticklabels([str(v) for v in TFS])
            ax.set_yticks(range(len(PROPS)))
            ax.set_yticklabels([str(v) for v in PROPS])
            ax.set_xlabel("trans_frac")
            ax.set_ylabel("prop (FR conj frac)")
            ax.set_title(f"{mode} — {level}", fontsize=11)
            for i in range(len(PROPS)):
                for j in range(len(TFS)):
                    val = piv.values[i, j]
                    if not np.isnan(val):
                        color = "white" if val > 0.25 else "black"
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                                fontsize=7, color=color)
            plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Peak NL Morphology (max FR frac in NL outputs) During Training",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def print_summary_table(final):
    print("\n" + "=" * 100)
    print("FINAL-STEP SUMMARY: Key metrics by (mode, level, prop) at tf=0.5")
    print("=" * 100)
    sub = final[final.tf == 0.5].sort_values(["level", "mode", "prop"])
    cols = ["mode", "level", "prop",
            "nl_lexical", "fr_lexical",
            "nl_exact_match", "fr_exact_match",
            "nl_conformity", "fr_conformity"]
    avail = [c for c in cols if c in sub.columns]
    print(sub[avail].to_string(index=False, float_format="%.4f"))

    print("\n" + "=" * 100)
    print("DIFFERENCE (nomask − mask) at tf=0.5")
    print("=" * 100)
    for level in LEVELS:
        print(f"\n  {level}:")
        for prop in PROPS:
            m = final[(final.level == level) & (final["mode"] == "mask") &
                      (final.prop == prop) & (final.tf == 0.5)]
            n = final[(final.level == level) & (final["mode"] == "nomask") &
                      (final.prop == prop) & (final.tf == 0.5)]
            if len(m) == 0 or len(n) == 0:
                continue
            m, n = m.iloc[0], n.iloc[0]
            d_nl_lex = n.nl_lexical - m.nl_lexical
            d_fr_lex = n.fr_lexical - m.fr_lexical
            d_nl_em = n.nl_exact_match - m.nl_exact_match
            d_fr_em = n.fr_exact_match - m.fr_exact_match
            d_nl_conf = n.nl_conformity - m.nl_conformity if "nl_conformity" in final.columns else 0
            d_fr_conf = n.fr_conformity - m.fr_conformity if "fr_conformity" in final.columns else 0
            print(f"    prop={prop:4.2f}  Δnl_lex={d_nl_lex:+.4f}  Δfr_lex={d_fr_lex:+.4f}  "
                  f"Δnl_em={d_nl_em:+.4f}  Δfr_em={d_fr_em:+.4f}  "
                  f"Δnl_conf={d_nl_conf:+.4f}  Δfr_conf={d_fr_conf:+.4f}")


def print_extreme_props(final):
    print("\n" + "=" * 100)
    print("EXTREME PROPORTIONS: Detailed view at prop=0.01 and prop=0.99")
    print("=" * 100)
    for prop in [0.01, 0.99]:
        sub = final[final.prop == prop].sort_values(["level", "tf", "mode"])
        print(f"\n--- prop = {prop} ---")
        cols = ["mode", "level", "tf",
                "nl_lexical", "fr_lexical",
                "nl_exact_match", "fr_exact_match",
                "nl_conformity", "fr_conformity"]
        avail = [c for c in cols if c in sub.columns]
        print(sub[avail].to_string(index=False, float_format="%.4f"))


def main():
    all_df = load_all_runs()
    final = get_final(all_df)

    print_summary_table(final)
    print_extreme_props(final)

    print("\nGenerating plots...")

    # 1. Heatmaps for final metrics
    plot_heatmaps(final, "nl_lexical",
                  "Final NL Lexical Score (FR frac in NL outputs)",
                  "heatmap_nl_lexical.png")
    plot_heatmaps(final, "fr_lexical",
                  "Final FR Lexical Score (FR frac in FR outputs)",
                  "heatmap_fr_lexical.png")
    plot_heatmaps(final, "nl_exact_match",
                  "Final NL Exact Match",
                  "heatmap_nl_exact_match.png")
    plot_heatmaps(final, "fr_exact_match",
                  "Final FR Exact Match",
                  "heatmap_fr_exact_match.png")
    plot_heatmaps(final, "nl_conformity",
                  "Final NL Conformity (structural validity)",
                  "heatmap_nl_conformity.png")

    # 2. Difference heatmaps
    plot_diff_heatmaps(final, "nl_lexical",
                       "Δ NL Lexical Score (nomask − mask)\nPositive = more FR in NL outputs without masking",
                       "diff_nl_lexical.png")
    plot_diff_heatmaps(final, "nl_exact_match",
                       "Δ NL Exact Match (nomask − mask)\nPositive = better without masking",
                       "diff_nl_exact_match.png")
    plot_diff_heatmaps(final, "fr_exact_match",
                       "Δ FR Exact Match (nomask − mask)",
                       "diff_fr_exact_match.png")
    plot_diff_heatmaps(final, "nl_conformity",
                       "Δ NL Conformity (nomask − mask)",
                       "diff_nl_conformity.png")

    # 3. Time series
    plot_timeseries_grid(all_df, "nl_lexical", "NL lexical",
                         "NL Lexical Score Over Training: Mask vs No-Mask",
                         "timeseries_nl_lex_tf05.png", tf_fixed=0.5)
    plot_timeseries_grid(all_df, "fr_lexical", "FR lexical",
                         "FR Lexical Score Over Training: Mask vs No-Mask",
                         "timeseries_fr_lex_tf05.png", tf_fixed=0.5)
    plot_timeseries_grid(all_df, "nl_exact_match", "NL EM",
                         "NL Exact Match Over Training: Mask vs No-Mask",
                         "timeseries_nl_em_tf05.png", tf_fixed=0.5)
    plot_timeseries_grid(all_df, "nl_conformity", "NL conf",
                         "NL Conformity Over Training: Mask vs No-Mask",
                         "timeseries_nl_conf_tf05.png", tf_fixed=0.5)

    # Also do tf=0.01 (almost no translation)
    plot_timeseries_grid(all_df, "nl_lexical", "NL lexical",
                         "NL Lexical Score Over Training: Mask vs No-Mask",
                         "timeseries_nl_lex_tf001.png", tf_fixed=0.01)

    # 4. Scatter
    plot_scatter_mask_vs_nomask(final, "nl_lexical", "NL Lexical Score",
                                "scatter_nl_lexical.png")
    plot_scatter_mask_vs_nomask(final, "nl_exact_match", "NL Exact Match",
                                "scatter_nl_exact_match.png")
    plot_scatter_mask_vs_nomask(final, "fr_exact_match", "FR Exact Match",
                                "scatter_fr_exact_match.png")

    # 5. Convergence speed
    plot_convergence_speed(all_df, "convergence_speed.png")

    # 6. Peak codeswitching
    plot_peak_codeswitching(all_df, "peak_nl_lexical.png")

    print(f"\nAll plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
