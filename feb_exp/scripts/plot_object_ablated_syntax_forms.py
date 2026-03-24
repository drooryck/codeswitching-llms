#!/usr/bin/env python3
"""Plot proportion of each syntax form (pos 4-6) by training proportion per ablation type.

X-axis: proportion of French in training (0, 0.1, ..., 1.0)
Y-axis: proportion of predictions (0–1) with that syntax form
Grouped bars: at each x, one bar per syntax form (part det noun, det noun part, etc.)

Output: 2-panel figure (French test sentences | Dutch test sentences).
Supports all ablation types (none, subject, verb, object) and mar4 presets (plurality-mixing, no-plurality).
"""
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from feb_exp.src.metrics import Metrics
from feb_exp.scripts.compute_language_alignment import get_token_role

REPO_ROOT = Path(__file__).resolve().parents[2]
NETSCRATCH_RESULTS = Path("/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results")
OUTPUT_ROOT = REPO_ROOT / "feb_exp" / "results"
LEXICON = REPO_ROOT / "feb_exp" / "data" / "lexicon_sep22.json"

ABLATION_TYPES = ("none", "subject", "verb", "object")
PRESETS = {
    "mar4-plurality": ("mar4/mar4-v1-plurality-mixing", "mar4/version1_plurality_mixing"),
    "mar4-no-plurality": ("mar4/mar4-v1-no-plurality", "mar4/version1_no_plurality_mixing"),
    "mar5": ("mar5/mar5-plurality-mixing", "mar5"),
}


def get_form_key(pred, m):
    toks = m.tokenize(pred.lower())
    if len(toks) < 6:
        return "other"
    roles = [get_token_role(t, m) for t in toks]
    if roles[0] != "det" or roles[1] != "noun" or roles[2] != "aux":
        return "other"
    return " ".join(roles[3:6])


def load_form_proportions(runs_dir: Path, lexicon_path: Path, ablation_type: str, run_id: int = 1):
    m = Metrics(lexicon_path)
    counts = {}
    props = []
    for d in sorted(runs_dir.glob(f"p*_run{run_id:02d}")):
        if not (d / "ablation_predictions.csv").exists():
            continue
        prop = float(d.name.split("_run")[0][1:]) / 100.0
        props.append(prop)
        df = pd.read_csv(d / "ablation_predictions.csv")
        sub = df[df["ablation"] == ablation_type]
        for _, row in sub.iterrows():
            lang = row["language"]
            form = get_form_key(str(row["prediction"]).strip(), m)
            key = (prop, lang)
            if key not in counts:
                counts[key] = {}
            counts[key][form] = counts[key].get(form, 0) + 1

    props = sorted(set(props))
    rows = []
    for (prop, lang), form_counts in counts.items():
        total = sum(form_counts.values())
        for form, cnt in form_counts.items():
            rows.append({"prop": prop, "lang": lang, "form": form, "proportion": cnt / total})
    return pd.DataFrame(rows), props


def plot_panel(ax, df: pd.DataFrame, props: list, lang: str, title: str, form_order: list, colors: dict):
    df_lang = df[df["lang"] == lang]
    n_props = len(props)
    n_forms = len(form_order)
    bar_width = 0.8 / max(n_forms, 1)
    x = np.arange(n_props)

    for i, form in enumerate(form_order):
        heights = []
        for prop in props:
            row = df_lang[(df_lang["prop"] == prop) & (df_lang["form"] == form)]
            h = row["proportion"].sum() if len(row) else 0.0
            heights.append(h)
        offset = (i - n_forms / 2 + 0.5) * bar_width
        ax.bar(x + offset, heights, bar_width, label=form, color=colors.get(form, "gray"))

    ax.set_xticks(x)
    ax.set_xticklabels([str(round(p, 2)) for p in props], rotation=45, ha="right")
    ax.set_xlabel("Proportion of French in training")
    ax.set_ylabel("Proportion of predictions")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(axis="y", alpha=0.3)


def build_form_order_and_colors(df: pd.DataFrame):
    forms_fr = sorted(df[df["lang"] == "fr"]["form"].unique())
    forms_nl = sorted(df[df["lang"] == "nl"]["form"].unique())
    all_forms = sorted(set(forms_fr) | set(forms_nl))
    order = ["part det noun", "det noun part"]
    for f in all_forms:
        if f not in order:
            order.append(f)
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    colors = {f: palette[i % len(palette)] for i, f in enumerate(order)}
    return order, colors


def plot_one(runs_dir: Path, output_path: Path, lexicon_path: Path, ablation_type: str,
             preset_label: str, run_id: int = 1):
    df, props = load_form_proportions(runs_dir, lexicon_path, ablation_type, run_id)
    if df.empty:
        print("  No data for ablation=%s, skipping" % ablation_type)
        return False
    order, colors = build_form_order_and_colors(df)
    abl_label = ablation_type + "-ablated" if ablation_type != "none" else "no ablation"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plot_panel(ax1, df, props, "fr", "French test sentences (%s)" % abl_label, order, colors)
    plot_panel(ax2, df, props, "nl", "Dutch test sentences (%s)" % abl_label, order, colors)
    fig.suptitle("Syntax form (positions 4–6) by proportion of French in training\n%s ablation — %s" % (ablation_type, preset_label), fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved", output_path)
    return True


def main():
    preset_names = list(PRESETS.keys()) + ["all"]
    parser = argparse.ArgumentParser(description="Plot syntax form proportions by training proportion")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Runs directory (optional if using --preset)")
    parser.add_argument("--output-dir", "-o", type=Path, default=None, help="Output directory for PNGs (optional if using --preset)")
    parser.add_argument("--preset", choices=preset_names, default=None,
                        help="Preset name or 'all' for every preset")
    parser.add_argument("--ablation", default="all",
                        help="Ablation type: none, subject, verb, object, or all (default: all)")
    parser.add_argument("--lexicon", type=Path, default=LEXICON, help="Lexicon path")
    parser.add_argument("--run-id", type=int, default=1, help="Run number (e.g. 1 for run01)")
    args = parser.parse_args()

    if args.preset is not None:
        selected = PRESETS if args.preset == "all" else {args.preset: PRESETS[args.preset]}
        for name, (out_subpath, netscratch_subpath) in selected.items():
            runs_dir = (NETSCRATCH_RESULTS / netscratch_subpath / "runs").resolve()
            output_dir = (OUTPUT_ROOT / out_subpath).resolve()
            if not runs_dir.exists():
                print("Skipping %s: runs dir not found: %s" % (name, runs_dir))
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            ablations = ABLATION_TYPES if args.ablation == "all" else [args.ablation]
            for ablation_type in ablations:
                abl_output = output_dir / ablation_type
                abl_output.mkdir(parents=True, exist_ok=True)
                out_path = abl_output / ("syntax_form_by_proportion_%s.png" % ablation_type)
                plot_one(runs_dir, out_path, args.lexicon, ablation_type, name, args.run_id)
        return

    runs_dir = args.runs_dir or (NETSCRATCH_RESULTS / "mar5/runs").resolve()
    output_dir = args.output_dir or (OUTPUT_ROOT / "mar5" / "mar5-plurality-mixing").resolve()
    ablations = ABLATION_TYPES if args.ablation == "all" else [args.ablation]
    preset_label = output_dir.name
    for ablation_type in ablations:
        abl_output = output_dir / ablation_type
        abl_output.mkdir(parents=True, exist_ok=True)
        out_path = abl_output / ("syntax_form_by_proportion_%s.png" % ablation_type)
        plot_one(runs_dir, out_path, args.lexicon, ablation_type, preset_label, args.run_id)


if __name__ == "__main__":
    main()
