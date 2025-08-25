#!/usr/bin/env python
# ------------------------------------------------------------------
# eval_and_plot.py   – aggregate test accuracies & plot vs. proportion
# ------------------------------------------------------------------
import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path, PurePath
import re, json

ROOT = Path("weights")            # parent directory of small_p00 … small_p100
rows = []

for folder in ROOT.glob("small_p*"):
    m = re.search(r"small_p(\d\d)", folder.name)
    if not m: 
        continue
    prop = int(m.group(1)) / 100.0                 # 0.0 … 1.0

    csv_path = folder / "test_generations.csv"
    if not csv_path.exists():
        print(f"!! {csv_path} missing, skipping")
        continue

    df = pd.read_csv(csv_path)
    df["correct"] = (df["prediction"].str.strip() == df["target"].str.strip())

    fr_acc = df[df.lang == "fr"]["correct"].mean()
    nl_acc = df[df.lang == "nl"]["correct"].mean()

    rows.append({"prop": prop, "fr_acc": fr_acc, "nl_acc": nl_acc})

# ---------------- summary CSV ------------------------------------------------
summary = pd.DataFrame(rows).sort_values("prop")
summary.to_csv("accuracy_by_prop.csv", index=False)
print("✓ wrote accuracy_by_prop.csv")
print(summary)

# ---------------- plot -------------------------------------------------------
plt.figure(figsize=(6,4))
plt.plot(summary["prop"], summary["fr_acc"], marker="o", label="French test acc")
plt.plot(summary["prop"], summary["nl_acc"], marker="o", label="Dutch test acc")
plt.xlabel("Training proportion p  (fraction of French sentences)")
plt.ylabel("Exact‑match accuracy on 5 k‑item test set")
plt.title("Present → Past tense conversion accuracy")
plt.grid(True, linestyle="--", alpha=0.3)
plt.ylim(0, 1.02)
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=150)
print("✓ saved accuracy_curve.png")
