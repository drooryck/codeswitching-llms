#!/usr/bin/env python3
"""Plot token type distribution at each output position for mar5 runs.

For each ablation type (none, subject, verb, object), creates a plot showing:
  - X-axis: output position (0, 1, 2, 3, 4, 5)
  - Y-axis: proportion of predictions (0–1) with that token type at that position
  - Stacked bars: det, noun, aux, part, unknown

Output: 4 PNG files (one per ablation type), each with 2 panels (French | Dutch test sentences).

Usage (from repo root, with venv activated):
  export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:$PYTHONPATH
  python feb_exp/scripts/mar5/plot_token_type_by_position.py
"""
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from feb_exp.src.metrics import Metrics
from feb_exp.scripts.compute_language_alignment import get_token_role

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

NETSCRATCH_RESULTS = Path(
    "/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results"
)
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "feb_exp" / "results" / "mar5" / "mar5-plurality-mixing"
LEXICON_PATH = REPO_ROOT / "feb_exp" / "data" / "lexicon_sep22.json"
ABLATION_TYPES = ("none", "subject", "verb", "object")

TOKEN_TYPES = ("det", "noun", "aux", "part", "unknown")
TOKEN_COLORS = {
    "det": "#1f77b4",
    "noun": "#ff7f0e",
    "aux": "#2ca02c",
    "part": "#d62728",
    "unknown": "#7f7f7f",
}


def load_all_predictions(
    runs_dir: Path,
    run_ids: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    """Load all ablation predictions from run dirs (one CSV read per run)."""
    all_rows = []
    for run_dir in sorted(runs_dir.glob("p*_run*")):
        csv_path = run_dir / "ablation_predictions.csv"
        if not csv_path.exists():
            continue
        run_id = int(run_dir.name.split("_run")[1])
        if run_id not in run_ids:
            continue
        df = pd.read_csv(csv_path)
        all_rows.append(df)
    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def compute_position_counts(
    df: pd.DataFrame,
    lexicon_path: Path,
    ablation_type: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Count token types at each position for given ablation type."""
    m = Metrics(str(lexicon_path))
    sub = df[df["ablation"] == ablation_type]
    counts: dict[tuple[str, int, str], int] = {}
    total_by_lang: dict[str, int] = {"fr": 0, "nl": 0}

    for _, row in sub.iterrows():
        lang = row["language"]
        pred = str(row["prediction"]).strip()
        tokens = m.tokenize(pred.lower())
        total_by_lang[lang] = total_by_lang.get(lang, 0) + 1
        for pos, tok in enumerate(tokens):
            if pos >= 6:
                break
            role = get_token_role(tok, m)
            key = (lang, pos, role)
            counts[key] = counts.get(key, 0) + 1

    pos_totals: dict[tuple[str, int], int] = {}
    for (lang, pos, _), cnt in counts.items():
        key = (lang, pos)
        pos_totals[key] = pos_totals.get(key, 0) + cnt

    rows = []
    for (lang, pos, role), cnt in counts.items():
        total_at_pos = pos_totals.get((lang, pos), 1)
        rows.append({
            "lang": lang,
            "position": pos,
            "token_type": role,
            "count": cnt,
            "proportion": cnt / total_at_pos,
        })
    return pd.DataFrame(rows), total_by_lang


def plot_one(
    df: pd.DataFrame,
    total_by_lang: dict[str, int],
    output_path: Path,
    ablation_type: str,
) -> None:
    """Create 2-panel figure: token type distribution by position (FR | NL)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.rcParams.update({
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    positions = list(range(6))
    abl_label = f"{ablation_type}-ablated" if ablation_type != "none" else "no ablation"

    for ax, lang, subtitle in [
        (ax1, "fr", f"French test sentences ({abl_label})"),
        (ax2, "nl", f"Dutch test sentences ({abl_label})"),
    ]:
        df_lang = df[df["lang"] == lang]
        n_preds = total_by_lang.get(lang, 0)

        # Build matrix: position x token_type -> proportion
        bottom = np.zeros(6)
        for tt in TOKEN_TYPES:
            heights = []
            for pos in positions:
                row = df_lang[(df_lang["position"] == pos) & (df_lang["token_type"] == tt)]
                h = row["proportion"].sum() if len(row) else 0.0
                heights.append(h)
            heights = np.array(heights)
            ax.bar(
                positions,
                heights,
                bottom=bottom,
                label=tt,
                color=TOKEN_COLORS.get(tt, "gray"),
                width=0.7,
            )
            bottom += heights

        ax.set_xticks(positions)
        ax.set_xticklabels([f"pos {i}" for i in positions])
        ax.set_xlabel("Output position")
        ax.set_ylabel("Proportion of predictions")
        ax.set_title(f"{subtitle}\n(n={n_preds:,} predictions)")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Token type distribution by output position — {ablation_type} ablation",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def main():
    runs_dir = (NETSCRATCH_RESULTS / "mar5" / "runs").resolve()
    if not runs_dir.exists():
        logger.error("Runs dir not found: %s", runs_dir)
        return

    output_root = OUTPUT_ROOT.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logger.info("Runs: %s", runs_dir)
    logger.info("Output: %s", output_root)

    logger.info("Loading predictions (once)...")
    all_df = load_all_predictions(runs_dir)
    if all_df.empty:
        logger.error("No predictions found")
        return
    logger.info("Loaded %d predictions", len(all_df))

    for ablation_type in ABLATION_TYPES:
        try:
            df, total_by_lang = compute_position_counts(
                all_df, LEXICON_PATH, ablation_type
            )
        except Exception as e:
            logger.warning("  Ablation %s: %s", ablation_type, e)
            continue

        if df.empty:
            logger.warning("  Ablation %s: no data, skipping", ablation_type)
            continue

        abl_dir = output_root / ablation_type
        abl_dir.mkdir(parents=True, exist_ok=True)
        out_path = abl_dir / "token_type_by_position.png"
        plot_one(df, total_by_lang, out_path, ablation_type)

    logger.info("All plots saved under %s", output_root)


if __name__ == "__main__":
    main()
