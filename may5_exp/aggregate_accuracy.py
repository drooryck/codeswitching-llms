#!/usr/bin/env python
# -------------------------------------------------------------------------
# aggregate_accuracy.py – gather mean ± CI per p & language, plot with Seaborn
# -------------------------------------------------------------------------
import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = Path("weights")           # parent folder

records = []
for mfile in ROOT.rglob("metrics.json"):
    # parent dir looks like  small_p50_run3
    parts = mfile.parent.name.split("_")
    if len(parts) != 3 or not parts[0].startswith("small"):
        continue
    prop = int(parts[1][1:]) / 100.0          # 'p50' -> 0.5
    run  = int(parts[2][3:])                  # 'run3' -> 3

    js = json.load(open(mfile))
    records.append({
        "prop": prop,
        "run":  run,
        "fr_acc": js["fr_exact"],
        "nl_acc": js["nl_exact"]
    })

df = pd.DataFrame(records)
if df.empty:
    raise RuntimeError("No metrics.json files found under weights/")

# ── reshape to long form for Seaborn ────────────────────────────────────────
df_long = (
    df.melt(id_vars=["prop", "run"],
            value_vars=["fr_acc", "nl_acc"],
            var_name="lang", value_name="acc")
)

# ── save summary CSV (mean & count only) ───────────────────────────────────
summary = (df_long
           .groupby(["prop", "lang"])
           .agg(mean=("acc","mean"), n_runs=("acc","size"))
           .reset_index())
summary.to_csv("accuracy_summary.csv", index=False)
print("✓ wrote accuracy_summary.csv")

# ── plot with bootstrap CI (Seaborn default) ───────────────────────────────
plt.figure(figsize=(6,4))
sns.lineplot(
    data=df_long,
    x="prop", y="acc", hue="lang",
    estimator="mean",     # mean across runs
    errorbar=("ci", 95),  # 95 % bootstrap CI
    marker="o"
)
plt.ylim(0,1.02)
plt.xlabel("Training proportion p (French fraction)")
plt.ylabel("Exact‑match accuracy")
plt.title("Accuracy with 95 % bootstrap CI (Seaborn)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("accuracy_seaborn.png", dpi=150)
print("✓ saved accuracy_seaborn.png")
