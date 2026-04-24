#!/usr/bin/env python3
"""Analyze per-POS language alignment across prop and structure type."""

import re
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_ROOT = Path(
    "/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/"
    "april_exp/cfg/results/sweep_apr14/runs"
)
LEX_PATH = Path("/n/home06/drooryck/codeswitching-llms/april_exp/cfg/data/lexicon.json")
OUT_DIR = Path("/n/home06/drooryck/codeswitching-llms/april_exp/cfg/scripts/apr14")

# ── Build per-POS word sets ──────────────────────────────────────────
lex = json.load(open(LEX_PATH))
pos_en = defaultdict(set)
pos_nl = defaultdict(set)

for form in lex["DET"]["en"].values():
    pos_en["det"].add(form.lower())
for key in ("nl_de", "nl_het"):
    for form in lex["DET"][key].values():
        pos_nl["det"].add(form.lower())
for noun_data in lex["NOUNS"].values():
    for form in noun_data["en"].values():
        if isinstance(form, str):
            pos_en["noun"].add(form.lower())
    for form in noun_data["nl"].values():
        if isinstance(form, str):
            pos_nl["noun"].add(form.lower())
for verb_data in lex["VERBS"].values():
    for form in verb_data["en"]["present"].values():
        pos_en["verb"].add(form.lower())
    for form in verb_data["nl"]["present"].values():
        pos_nl["verb"].add(form.lower())
    pos_en["part"].add(verb_data["en"]["participle"].lower())
    pos_nl["part"].add(verb_data["nl"]["participle"].lower())
for form in lex["AUX"]["en"].values():
    pos_en["aux"].add(form.lower())
for form in lex["AUX"]["nl"].values():
    pos_nl["aux"].add(form.lower())
for prep_data in lex["PREP"].values():
    pos_en["prep"].add(prep_data["en"].lower())
    pos_nl["prep"].add(prep_data["nl"].lower())
pos_en["rel"].add(lex["REL"]["en"].lower())
pos_nl["rel"].add(lex["REL"]["nl_de"].lower())
pos_nl["rel"].add(lex["REL"]["nl_het"].lower())

all_en = set()
all_nl = set()
for s in pos_en.values():
    all_en |= s
for s in pos_nl.values():
    all_nl |= s

TOKEN_RE = re.compile(r"\w+|[^\s\w]")

ROLE_TEMPLATES = {
    "plain": {
        "en": ["det", "noun", "aux", "part", "det", "noun"],
        "nl": ["det", "noun", "aux", "det", "noun", "part"],
    },
    "subj_pp": {
        "en": ["det", "noun", "prep", "det", "noun", "aux", "part", "det", "noun"],
        "nl": ["det", "noun", "prep", "det", "noun", "aux", "det", "noun", "part"],
    },
    "obj_pp": {
        "en": ["det", "noun", "aux", "part", "det", "noun", "prep", "det", "noun"],
        "nl": ["det", "noun", "aux", "det", "noun", "prep", "det", "noun", "part"],
    },
    "subj_rc": {
        "en": ["det", "noun", "rel", "verb", "det", "noun", "aux", "part", "det", "noun"],
        "nl": ["det", "noun", "rel", "det", "noun", "verb", "aux", "det", "noun", "part"],
    },
    "obj_rc": {
        "en": ["det", "noun", "aux", "part", "det", "noun", "rel", "verb", "det", "noun"],
        "nl": ["det", "noun", "aux", "det", "noun", "rel", "det", "noun", "verb", "part"],
    },
    "subj_pp+obj_pp": {
        "en": ["det", "noun", "prep", "det", "noun", "aux", "part", "det", "noun", "prep", "det", "noun"],
        "nl": ["det", "noun", "prep", "det", "noun", "aux", "det", "noun", "prep", "det", "noun", "part"],
    },
}


def main():
    print("Starting POS-language analysis...", flush=True)

    # Aggregate incrementally to avoid OOM
    # key = (prop, seed, test_lang, structure, actual_pos) -> [sum_lang, count]
    agg_struct = defaultdict(lambda: [0.0, 0])
    # key = (prop, seed, test_lang, actual_pos) -> [sum_lang, count]
    agg_all = defaultdict(lambda: [0.0, 0])

    for run_dir in sorted(RESULTS_ROOT.iterdir()):
        m = re.match(r"mask_(\w+)_prop([\d.]+)_tf([\d.]+)_run(\d+)", run_dir.name)
        if not m:
            continue
        r_prop = float(m.group(2))
        r_tf = float(m.group(3))
        r_seed = int(m.group(4))
        if r_tf != 0.0:
            continue

        pf = run_dir / "test_predictions.csv"
        if not pf.exists():
            continue

        df = pd.read_csv(pf)
        n_valid = 0

        for _, row in df.iterrows():
            pred = str(row["prediction"])
            struct = row["structure"]
            if struct not in ROLE_TEMPLATES:
                continue

            tokens = TOKEN_RE.findall(pred.lower())
            expected_len = len(ROLE_TEMPLATES[struct]["en"])
            if len(tokens) != expected_len:
                continue

            n_valid += 1

            for tok in tokens:
                is_en = tok in all_en
                is_nl = tok in all_nl
                if not is_en and not is_nl:
                    continue

                if is_en and not is_nl:
                    lang_score = 1.0
                elif is_nl and not is_en:
                    lang_score = 0.0
                else:
                    lang_score = 0.5

                actual_pos = "unknown"
                for pos in ["rel", "aux", "part", "prep", "det", "verb", "noun"]:
                    if tok in pos_en[pos] or tok in pos_nl[pos]:
                        actual_pos = pos
                        break

                test_lang = row["language"]
                agg_struct[(r_prop, r_seed, test_lang, struct, actual_pos)][0] += lang_score
                agg_struct[(r_prop, r_seed, test_lang, struct, actual_pos)][1] += 1
                agg_all[(r_prop, r_seed, test_lang, actual_pos)][0] += lang_score
                agg_all[(r_prop, r_seed, test_lang, actual_pos)][1] += 1

        print(f"  prop={r_prop}, seed={r_seed}: {n_valid} valid preds", flush=True)

    # Convert to DataFrames
    rows_s = [
        (k[0], k[1], k[2], k[3], k[4], v[0] / v[1], v[1])
        for k, v in agg_struct.items()
    ]
    df_struct = pd.DataFrame(
        rows_s,
        columns=["prop", "seed", "test_lang", "structure", "pos", "en_frac", "n"],
    )

    rows_a = [
        (k[0], k[1], k[2], k[3], v[0] / v[1], v[1])
        for k, v in agg_all.items()
    ]
    df_all = pd.DataFrame(
        rows_a, columns=["prop", "seed", "test_lang", "pos", "en_frac", "n"]
    )

    df_struct.to_csv(OUT_DIR / "pos_lang_by_struct.csv", index=False)
    df_all.to_csv(OUT_DIR / "pos_lang_overall.csv", index=False)
    print(f"Saved CSVs to {OUT_DIR}", flush=True)

    # Print summary separated by test language
    for tl in ["en", "nl"]:
        sub = df_all[df_all["test_lang"] == tl]
        print(f"\n=== EN fraction by POS and prop — {tl.upper()} test sentences ===", flush=True)
        avg = sub.groupby(["prop", "pos"])["en_frac"].mean().unstack()
        print(avg.to_string(float_format="%.3f"), flush=True)

    # ── Plotting ──────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    POS_ORDER = ["det", "noun", "aux", "part", "verb", "prep", "rel"]
    POS_COLORS = {
        "det": "#e41a1c", "noun": "#377eb8", "aux": "#4daf4a",
        "part": "#984ea3", "verb": "#ff7f00", "prep": "#a65628", "rel": "#999999",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, tl in zip(axes, ["en", "nl"]):
        sub = df_all[df_all["test_lang"] == tl]
        avg = sub.groupby(["prop", "pos"])["en_frac"].mean().unstack()
        for pos in POS_ORDER:
            if pos in avg.columns:
                ax.plot(avg.index, avg[pos], "o-", label=pos.upper(),
                        color=POS_COLORS[pos], linewidth=2, markersize=5)
        ax.set_xlabel("English proportion in training (prop)", fontsize=12)
        ax.set_title(f"{'English' if tl=='en' else 'Dutch'} test sentences", fontsize=13)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", ls="--", alpha=0.4)
        ax.set_xticks(sorted(df_all["prop"].unique()))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Fraction English tokens (per POS)", fontsize=12)
    axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    fig.suptitle("Per-POS Language Alignment vs Training Proportion",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pos_lang_alignment_v1.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {OUT_DIR / 'pos_lang_alignment_v1.png'}", flush=True)

    # ── Per-structure faceted plot ────────────────────────────────
    structs = sorted(ROLE_TEMPLATES.keys())
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10), sharey=True, sharex=True)
    for idx, struct in enumerate(structs):
        ax = axes2.flat[idx]
        sub = df_struct[(df_struct["test_lang"] == "nl") & (df_struct["structure"] == struct)]
        if sub.empty:
            continue
        avg = sub.groupby(["prop", "pos"])["en_frac"].mean().unstack()
        for pos in POS_ORDER:
            if pos in avg.columns:
                ax.plot(avg.index, avg[pos], "o-", label=pos.upper(),
                        color=POS_COLORS[pos], linewidth=1.5, markersize=4)
        ax.set_title(struct, fontsize=11, fontweight="bold")
        ax.axhline(0.5, color="gray", ls="--", alpha=0.4)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.2)
        if idx >= 3:
            ax.set_xlabel("prop")
        if idx % 3 == 0:
            ax.set_ylabel("EN frac")
    axes2.flat[-1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    fig2.suptitle("Per-POS Language of NL Test Predictions, by Structure Type",
                  fontsize=14, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "pos_lang_by_structure_v1.png", dpi=150, bbox_inches="tight")
    print(f"Plot saved: {OUT_DIR / 'pos_lang_by_structure_v1.png'}", flush=True)


if __name__ == "__main__":
    main()
