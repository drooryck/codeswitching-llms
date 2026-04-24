#!/usr/bin/env python3
"""
Analyze which POS positions drive language alignment (word order).

Adapted from apr14/pos_syntax_analysis.py to analyze sweep_apr16 (fixed-budget design).

For each structure, EN and NL templates differ at specific positions.
At each divergent position, classify whether the predicted token's POS
follows the EN or NL template.
"""

import re
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_ROOT = Path(
    "/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/"
    "april_exp/cfg/results/sweep_apr16/runs"
)
LEX_PATH = Path("/n/home06/drooryck/codeswitching-llms/april_exp/cfg/data/lexicon.json")
OUT_DIR = Path("/n/home06/drooryck/codeswitching-llms/april_exp/cfg/scripts/apr16")

# ── Lexicon: build POS membership sets ───────────────────────────
lex = json.load(open(LEX_PATH))
pos_sets = defaultdict(set)

for form in lex["DET"]["en"].values():
    pos_sets["det"].add(form.lower())
for key in ("nl_de", "nl_het"):
    for form in lex["DET"][key].values():
        pos_sets["det"].add(form.lower())
for noun_data in lex["NOUNS"].values():
    for lang in ("en", "nl"):
        for form in noun_data[lang].values():
            if isinstance(form, str):
                pos_sets["noun"].add(form.lower())
for verb_data in lex["VERBS"].values():
    for lang in ("en", "nl"):
        for form in verb_data[lang]["present"].values():
            pos_sets["verb"].add(form.lower())
        pos_sets["part"].add(verb_data[lang]["participle"].lower())
for lang in ("en", "nl"):
    for form in lex["AUX"][lang].values():
        pos_sets["aux"].add(form.lower())
for prep_data in lex["PREP"].values():
    for lang in ("en", "nl"):
        pos_sets["prep"].add(prep_data[lang].lower())
pos_sets["rel"].add(lex["REL"]["en"].lower())
pos_sets["rel"].add(lex["REL"]["nl_de"].lower())
pos_sets["rel"].add(lex["REL"]["nl_het"].lower())

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


def classify_role(tok):
    t = tok.lower()
    for pos in ["rel", "aux", "part", "prep", "det", "verb", "noun"]:
        if t in pos_sets[pos]:
            return pos
    return "unknown"


def main():
    print("Starting divergent-position syntax analysis (sweep_apr16)...", flush=True)

    div_info = {}
    for struct, templates in ROLE_TEMPLATES.items():
        en_t = templates["en"]
        nl_t = templates["nl"]
        divs = []
        for i in range(len(en_t)):
            if en_t[i] != nl_t[i]:
                divs.append((i, en_t[i], nl_t[i]))
        div_info[struct] = divs

    print("\nDivergent positions per structure:")
    for struct, divs in sorted(div_info.items()):
        desc = ", ".join(f"pos{i}: EN={e}/NL={n}" for i, e, n in divs)
        print(f"  {struct}: {desc}", flush=True)

    agg = defaultdict(lambda: [0, 0, 0, 0])
    agg_role = defaultdict(lambda: [0, 0, 0, 0])

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
            test_lang = row["language"]
            if struct not in div_info:
                continue

            tokens = TOKEN_RE.findall(pred.lower())
            expected_len = len(ROLE_TEMPLATES[struct]["en"])
            if len(tokens) != expected_len:
                continue

            n_valid += 1

            for pos_idx, en_role, nl_role in div_info[struct]:
                tok_role = classify_role(tokens[pos_idx])
                en_match = int(tok_role == en_role)
                nl_match = int(tok_role == nl_role)
                neither = int(not en_match and not nl_match)

                k1 = (r_prop, r_seed, test_lang, struct, pos_idx, en_role, nl_role)
                agg[k1][0] += en_match
                agg[k1][1] += nl_match
                agg[k1][2] += neither
                agg[k1][3] += 1

                k2 = (r_prop, r_seed, test_lang, en_role, nl_role)
                agg_role[k2][0] += en_match
                agg_role[k2][1] += nl_match
                agg_role[k2][2] += neither
                agg_role[k2][3] += 1

        print(f"  prop={r_prop}, seed={r_seed}: {n_valid} valid preds", flush=True)

    rows_d = []
    for k, v in agg.items():
        total = v[3]
        rows_d.append({
            "prop": k[0], "seed": k[1], "test_lang": k[2],
            "structure": k[3], "pos_idx": k[4],
            "en_role": k[5], "nl_role": k[6],
            "en_frac": v[0] / total, "nl_frac": v[1] / total,
            "neither_frac": v[2] / total, "n": total,
        })
    df_detail = pd.DataFrame(rows_d)

    rows_r = []
    for k, v in agg_role.items():
        total = v[3]
        rows_r.append({
            "prop": k[0], "seed": k[1], "test_lang": k[2],
            "en_role": k[3], "nl_role": k[4],
            "en_frac": v[0] / total, "nl_frac": v[1] / total,
            "neither_frac": v[2] / total, "n": total,
        })
    df_role = pd.DataFrame(rows_r)

    df_detail.to_csv(OUT_DIR / "syntax_divergent_detail.csv", index=False)
    df_role.to_csv(OUT_DIR / "syntax_divergent_role.csv", index=False)
    print(f"Saved CSVs to {OUT_DIR}", flush=True)

    for tl in ["en", "nl"]:
        sub = df_role[df_role["test_lang"] == tl]
        print(f"\n=== {tl.upper()} test: EN-order fraction at divergent positions ===")
        sub = sub.copy()
        sub["role_pair"] = sub["en_role"] + "→" + sub["nl_role"]
        pivot = sub.groupby(["prop", "role_pair"])["en_frac"].mean().unstack()
        print(pivot.to_string(float_format="%.3f"), flush=True)

    # ── Plotting ──────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PAIR_COLORS = {
        "part→det": "#e41a1c",
        "det→noun": "#377eb8",
        "noun→part": "#4daf4a",
        "verb→det": "#984ea3",
        "det→verb": "#ff7f00",
        "part→noun": "#a65628",
        "noun→verb": "#f781bf",
        "verb→noun": "#999999",
        "part→prep": "#66c2a5",
        "prep→det": "#fc8d62",
        "det→det": "#8da0cb",
        "noun→prep": "#e78ac3",
        "prep→noun": "#a6d854",
        "noun→det": "#ffd92f",
        "det→part": "#e5c494",
        "verb→part": "#b3b3b3",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, tl in zip(axes, ["nl", "en"]):
        sub = df_role[df_role["test_lang"] == tl].copy()
        sub["role_pair"] = sub["en_role"] + "→" + sub["nl_role"]

        avg = sub.groupby(["prop", "role_pair"])["en_frac"].mean().unstack()
        for rp in sorted(avg.columns):
            color = PAIR_COLORS.get(rp, "#333333")
            ax.plot(avg.index, avg[rp], "o-", label=rp, color=color,
                    linewidth=2, markersize=5)

        lang_label = "Dutch" if tl == "nl" else "English"
        ax.set_title(f"{lang_label} test sentences", fontsize=13)
        ax.set_xlabel("English proportion in training (prop)", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", ls="--", alpha=0.4)
        ax.set_xticks(sorted(sub["prop"].unique()))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Fraction following EN word order\n(at divergent positions)", fontsize=11)
    axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9,
                   title="EN role → NL role")
    fig.suptitle("Word-Order Alignment at Divergent Positions (apr16, fixed budget)\n"
                 "(1.0 = EN order, 0.0 = NL order)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "syntax_alignment_by_role_v1.png", dpi=150,
                bbox_inches="tight")
    print(f"\nPlot saved: {OUT_DIR / 'syntax_alignment_by_role_v1.png'}", flush=True)

    structs = sorted(ROLE_TEMPLATES.keys())
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10), sharey=True, sharex=True)

    for idx, struct in enumerate(structs):
        ax = axes2.flat[idx]
        sub = df_detail[
            (df_detail["test_lang"] == "nl") & (df_detail["structure"] == struct)
        ].copy()
        if sub.empty:
            continue
        sub["role_pair"] = sub["en_role"] + "→" + sub["nl_role"]
        avg = sub.groupby(["prop", "role_pair"])["en_frac"].mean().unstack()
        for rp in sorted(avg.columns):
            color = PAIR_COLORS.get(rp, "#333333")
            ax.plot(avg.index, avg[rp], "o-", label=rp, color=color,
                    linewidth=1.5, markersize=4)

        ax.set_title(struct, fontsize=11, fontweight="bold")
        ax.axhline(0.5, color="gray", ls="--", alpha=0.4)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.2)
        if idx >= 3:
            ax.set_xlabel("prop")
        if idx % 3 == 0:
            ax.set_ylabel("EN order frac")
        ax.legend(fontsize=7, loc="best")

    fig2.suptitle("Word-Order Alignment by Structure (NL test, apr16 fixed budget)\n"
                  "At each divergent position: fraction following EN template",
                  fontsize=14, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "syntax_alignment_by_structure_v1.png", dpi=150,
                 bbox_inches="tight")
    print(f"Plot saved: {OUT_DIR / 'syntax_alignment_by_structure_v1.png'}", flush=True)


if __name__ == "__main__":
    main()
