#!/usr/bin/env python3
"""
Clean v2 plot: POS-level word-order alignment.

Uses only seed=1 (the only seed with stable NL training) to get
a clean signal. Groups divergent positions into interpretable
categories.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/n/home06/drooryck/codeswitching-llms/april_exp/cfg/scripts/apr14")

df = pd.read_csv(OUT / "syntax_divergent_detail.csv")
print(f"Loaded {len(df)} rows", flush=True)
print(f"Seeds: {sorted(df['seed'].unique())}", flush=True)

# ── Focus on seed 1 (stable training) ─────────────────────────────
df1 = df[df["seed"] == 1].copy()

# Create a readable label for each divergent position
# The key linguistic distinction: participle shift vs RC verb shift
def categorize(row):
    en, nl = row["en_role"], row["nl_role"]
    pair = f"{en}→{nl}"
    if en == "part" or nl == "part":
        return "participle position"
    if en == "verb" or nl == "verb":
        return "RC verb position"
    if en == "rel" or nl == "rel":
        return "relativizer position"
    return "NP-internal shift"

df1["category"] = df1.apply(categorize, axis=1)
df1["role_pair"] = df1["en_role"] + "→" + df1["nl_role"]

# ═══════════════════════════════════════════════════════════════════
# PLOT 1: Main overview — NL test, grouped by category
# ═══════════════════════════════════════════════════════════════════
nl = df1[df1["test_lang"] == "nl"]

cat_colors = {
    "participle position": "#e41a1c",
    "RC verb position": "#377eb8",
    "NP-internal shift": "#999999",
    "relativizer position": "#4daf4a",
}
cat_order = ["participle position", "RC verb position",
             "NP-internal shift", "relativizer position"]

fig, ax = plt.subplots(figsize=(8, 5))
cat_avg = nl.groupby(["prop", "category"])["en_frac"].mean().unstack()
for cat in cat_order:
    if cat not in cat_avg.columns:
        continue
    ax.plot(cat_avg.index, cat_avg[cat], "o-", label=cat,
            color=cat_colors[cat], linewidth=2.5, markersize=6)

ax.set_xlabel("English proportion in training (prop)", fontsize=12)
ax.set_ylabel("Fraction following EN word order", fontsize=12)
ax.set_title("Which POS positions follow English word order?\n"
             "(Dutch test sentences, seed=1, tf=0)", fontsize=13, fontweight="bold")
ax.set_ylim(-0.05, 1.05)
ax.axhline(0, color="gray", ls="-", alpha=0.3)
ax.axhline(1, color="gray", ls="-", alpha=0.3)
ax.legend(fontsize=10, loc="upper left")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "pos_alignment_overview_v2.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT / 'pos_alignment_overview_v2.png'}", flush=True)

# ═══════════════════════════════════════════════════════════════════
# PLOT 2: Per-structure, showing all individual role pairs
# ═══════════════════════════════════════════════════════════════════
structs = sorted(nl["structure"].unique())
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 9), sharey=True, sharex=True)

for idx, struct in enumerate(structs):
    ax = axes2.flat[idx]
    sub = nl[nl["structure"] == struct]
    if sub.empty:
        continue

    avg = sub.groupby(["prop", "category"])["en_frac"].mean().unstack()
    for cat in cat_order:
        if cat not in avg.columns:
            continue
        ax.plot(avg.index, avg[cat], "o-", label=cat,
                color=cat_colors[cat], linewidth=2, markersize=4)

    ax.set_title(struct.replace("_", " "), fontsize=11, fontweight="bold")
    ax.axhline(0, color="gray", ls="-", alpha=0.2)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.15)
    if idx >= 3:
        ax.set_xlabel("prop")
    if idx % 3 == 0:
        ax.set_ylabel("EN order fraction")

handles, labels = axes2.flat[0].get_legend_handles_labels()
fig2.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
            bbox_to_anchor=(0.5, -0.02))
fig2.suptitle("Word-Order Alignment by Structure Type\n"
              "(Dutch test sentences, seed=1, tf=0)",
              fontsize=14, fontweight="bold")
fig2.tight_layout(rect=[0, 0.05, 1, 0.95])
fig2.savefig(OUT / "pos_alignment_by_struct_v2.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT / 'pos_alignment_by_struct_v2.png'}", flush=True)

# ═══════════════════════════════════════════════════════════════════
# PLOT 3: Combined panel — THE key plot
# Left: NL test by category
# Right: heatmap of all role pairs × prop
# ═══════════════════════════════════════════════════════════════════
fig3, (ax_line, ax_heat) = plt.subplots(1, 2, figsize=(15, 5.5),
                                         gridspec_kw={"width_ratios": [1, 1.3]})

# Left: line plot by category
for cat in cat_order:
    if cat not in cat_avg.columns:
        continue
    ax_line.plot(cat_avg.index, cat_avg[cat], "o-", label=cat,
                 color=cat_colors[cat], linewidth=2.5, markersize=6)
ax_line.set_xlabel("prop (EN fraction in training)", fontsize=11)
ax_line.set_ylabel("Fraction following EN word order", fontsize=11)
ax_line.set_title("By POS category", fontsize=12, fontweight="bold")
ax_line.set_ylim(-0.05, 1.05)
ax_line.legend(fontsize=9, loc="upper left")
ax_line.grid(alpha=0.2)

# Right: heatmap of individual role pairs
rp_avg = nl.groupby(["prop", "role_pair"])["en_frac"].mean().unstack()
rp_order = sorted(rp_avg.columns, key=lambda x: rp_avg[x].mean())
rp_avg = rp_avg[rp_order]

im = ax_heat.imshow(rp_avg.values.T, aspect="auto", cmap="RdBu_r",
                     vmin=0, vmax=1)
ax_heat.set_xticks(range(len(rp_avg.index)))
ax_heat.set_xticklabels([f"{p:.2f}" for p in rp_avg.index], rotation=45, fontsize=9)
ax_heat.set_yticks(range(len(rp_avg.columns)))
ax_heat.set_yticklabels(rp_avg.columns, fontsize=9)
ax_heat.set_xlabel("prop", fontsize=11)
ax_heat.set_title("By individual role pair", fontsize=12, fontweight="bold")
cb = fig3.colorbar(im, ax=ax_heat, shrink=0.8, label="EN order fraction")

fig3.suptitle("POS-Level Language Alignment in Dutch Predictions\n"
              "(seed=1, tf=0 — 0 = Dutch order, 1 = English order)",
              fontsize=13, fontweight="bold")
fig3.tight_layout()
fig3.savefig(OUT / "pos_alignment_combined_v2.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT / 'pos_alignment_combined_v2.png'}", flush=True)

# ═══════════════════════════════════════════════════════════════════
# PLOT 4: THE definitive plot — faceted by structure, heatmaps
# ═══════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(2, 3, figsize=(18, 10))

for idx, struct in enumerate(structs):
    ax = axes4.flat[idx]
    sub = nl[nl["structure"] == struct]
    if sub.empty:
        ax.set_visible(False)
        continue

    hm = sub.groupby(["prop", "role_pair"])["en_frac"].mean().unstack()
    rp_sorted = sorted(hm.columns, key=lambda x: hm[x].mean())
    hm = hm[rp_sorted]

    im = ax.imshow(hm.values.T, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(hm.index)))
    ax.set_xticklabels([f"{p:.1g}" for p in hm.index], fontsize=8, rotation=45)
    ax.set_yticks(range(len(hm.columns)))
    ax.set_yticklabels(hm.columns, fontsize=9)
    ax.set_title(struct.replace("_", " "), fontsize=11, fontweight="bold")
    if idx >= 3:
        ax.set_xlabel("prop")

fig4.colorbar(im, ax=axes4.ravel().tolist(), shrink=0.6,
              label="EN order fraction (0=NL, 1=EN)")
fig4.suptitle("POS-Level Word Order by Structure\n"
              "(Dutch test, seed=1, tf=0 — blue = NL order, red = EN order)",
              fontsize=14, fontweight="bold")
fig4.tight_layout(rect=[0, 0, 0.92, 0.93])
fig4.savefig(OUT / "pos_alignment_heatmaps_v2.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT / 'pos_alignment_heatmaps_v2.png'}", flush=True)

# Print key summary numbers
print("\n=== KEY FINDINGS (NL test, seed=1) ===", flush=True)
for struct in structs:
    sub = nl[nl["structure"] == struct]
    avg_struct = sub.groupby(["prop", "category"])["en_frac"].mean().unstack()
    print(f"\n{struct}:", flush=True)
    for cat in cat_order:
        if cat in avg_struct.columns:
            vals = avg_struct[cat]
            print(f"  {cat:25s}: min={vals.min():.3f} max={vals.max():.3f} "
                  f"at p=0.5: {vals.get(0.5, float('nan')):.3f}", flush=True)
