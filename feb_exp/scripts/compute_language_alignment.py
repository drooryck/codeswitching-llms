#!/usr/bin/env python3
"""Compute language alignment scores from ablation_predictions.csv for each run.

For each prediction, computes:
  - syntax_score: How French-like the word order is (0=NL, 1=FR), based on positions 4-6.
      FR template: pos4=participle, pos5=det, pos6=noun
      NL template: pos4=det, pos5=noun, pos6=participle
      Each position matching FR adds 1/3.
  - morphology_score: Proportion of tokens that are French (fr_share from token_lang_frac).
  - alignment_score: Average of syntax_score and morphology_score.

Saves per-run, per-ablation JSON files:  alignment_{ablation_type}.json

Usage:
  source /n/home06/drooryck/envs/codeswitching-py310/bin/activate
  python feb_exp/scripts/compute_language_alignment.py \\
      /n/netscratch/.../results/mar4/version1_plurality_mixing/runs
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from feb_exp.src.metrics import Metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LEXICON_PATH = Path(__file__).resolve().parents[1] / "data" / "lexicon_sep22.json"

FR_TEMPLATE_456 = ("part", "det", "noun")
NL_TEMPLATE_456 = ("det", "noun", "part")


def get_token_role(token: str, m: Metrics) -> str:
    """Classify a token into its grammatical role."""
    t = token.lower()
    if t in m.det_fr or t in m.det_nl:
        return "det"
    if t in m.nouns_fr or t in m.nouns_nl:
        return "noun"
    if t in m.aux_fr or t in m.aux_nl:
        return "aux"
    if t in m.part_fr or t in m.part_nl:
        return "part"
    return "unknown"


def compute_syntax_score(prediction: str, m: Metrics) -> float | None:
    """Compute syntax score (0=NL order, 1=FR order) from positions 4-6.

    Returns None if the prediction doesn't have exactly 6 tokens or
    positions 1-3 don't follow the shared prefix (det noun aux).
    """
    tokens = m.tokenize(prediction.lower())
    if len(tokens) != 6:
        return None

    roles = [get_token_role(t, m) for t in tokens]

    if roles[0] != "det" or roles[1] != "noun" or roles[2] != "aux":
        return None

    score = 0.0
    for i, fr_role in enumerate(FR_TEMPLATE_456):
        if roles[3 + i] == fr_role:
            score += 1.0 / 3.0
    return score


def compute_morphology_score(prediction: str, m: Metrics) -> float:
    """Compute morphology score = fraction of tokens that are French."""
    tokens = m.tokenize(prediction.lower())
    if not tokens:
        return 0.0
    fr_frac, _ = m.token_lang_frac(tokens)
    return fr_frac


def process_run(run_dir: Path, m: Metrics) -> None:
    """Process a single run directory."""
    csv_path = run_dir / "ablation_predictions.csv"
    if not csv_path.exists():
        logger.warning("No ablation_predictions.csv in %s, skipping", run_dir)
        return

    df = pd.read_csv(csv_path)

    for ablation_type in df["ablation"].unique():
        sub = df[df["ablation"] == ablation_type].copy()

        records = []
        for _, row in sub.iterrows():
            pred = str(row["prediction"])
            lang = row["language"]

            syntax = compute_syntax_score(pred, m)
            morph = compute_morphology_score(pred, m)
            alignment = ((syntax + morph) / 2.0) if syntax is not None else None

            records.append({
                "lang": lang,
                "syntax_score": syntax,
                "morphology_score": morph,
                "alignment_score": alignment,
            })

        rec_df = pd.DataFrame(records)

        result = {}
        for lang in rec_df["lang"].unique():
            lang_df = rec_df[rec_df["lang"] == lang]

            valid_syntax = lang_df["syntax_score"].dropna()
            valid_align = lang_df["alignment_score"].dropna()

            prefix = lang
            result[f"{prefix}_syntax_score"] = float(valid_syntax.mean()) if len(valid_syntax) else None
            result[f"{prefix}_morphology_score"] = float(lang_df["morphology_score"].mean())
            result[f"{prefix}_alignment_score"] = float(valid_align.mean()) if len(valid_align) else None
            result[f"{prefix}_syntax_valid_pct"] = float(len(valid_syntax) / len(lang_df)) if len(lang_df) else 0.0

        out_path = run_dir / f"alignment_{ablation_type}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("  Saved %s (%d predictions)", out_path.name, len(sub))


def main():
    parser = argparse.ArgumentParser(description="Compute language alignment scores per run.")
    parser.add_argument("runs_dir", type=Path, help="Directory containing p*_run* subdirs")
    parser.add_argument("--lexicon", type=Path, default=LEXICON_PATH, help="Path to lexicon JSON")
    args = parser.parse_args()

    m = Metrics(args.lexicon)

    run_dirs = sorted(args.runs_dir.glob("p*_run*"))
    if not run_dirs:
        logger.error("No p*_run* directories found in %s", args.runs_dir)
        sys.exit(1)

    logger.info("Found %d run directories in %s", len(run_dirs), args.runs_dir)

    for run_dir in run_dirs:
        logger.info("Processing %s ...", run_dir.name)
        process_run(run_dir, m)

    logger.info("Done.")


if __name__ == "__main__":
    main()
