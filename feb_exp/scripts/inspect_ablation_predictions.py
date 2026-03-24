"""
Inspect ablation_predictions.csv: head/tail and structure conformity.

Structure (metrics.py) is function-based: 6 slots as det, noun, aux, part, det, noun.
- FR structure: positions 0-5 = det, noun, aux, part, det, noun
- NL structure: positions 0-5 = det, noun, aux, det, noun, part

Token language (FR vs NL) does not define structure; only slot identity does.

Usage:
  python inspect_ablation_predictions.py <path-to-ablation_predictions.csv> [--lexicon path]
  python inspect_ablation_predictions.py results/feb20/runs/p50.00_run01/ablation_predictions.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running from repo root or from feb_exp/scripts/
_script_dir = Path(__file__).resolve().parent
_feb_exp = _script_dir.parent
_repo_root = _feb_exp.parent
sys.path.insert(0, str(_repo_root))
from feb_exp.src.metrics import Metrics


def main():
    parser = argparse.ArgumentParser(description="Inspect ablation_predictions.csv")
    parser.add_argument("csv_path", type=Path, help="Path to ablation_predictions.csv")
    parser.add_argument("--lexicon", type=Path, default=None, help="Lexicon path (default: feb_exp/data/lexicon_sep22.json)")
    parser.add_argument("--n", type=int, default=10, help="Number of head/tail rows to show")
    parser.add_argument("--by-structure", action="store_true", help="Report follows_fr / follows_nl / follows_either by language and ablation")
    args = parser.parse_args()

    csv_path = args.csv_path
    if not csv_path.is_absolute():
        csv_path = Path(__file__).resolve().parent.parent / csv_path
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    print()

    print("--- Head ---")
    print(df.head(args.n).to_string())
    print()
    print("--- Tail ---")
    print(df.tail(args.n).to_string())
    print()

    if args.by_structure:
        if args.lexicon is None:
            args.lexicon = Path(__file__).resolve().parent.parent / "data" / "lexicon_sep22.json"
        if not args.lexicon.exists():
            print(f"Lexicon not found: {args.lexicon}, skipping structure report")
        else:
            metrics = Metrics(args.lexicon)
            records = []
            for _, row in df.iterrows():
                st = metrics.check_structure_conformity(row["prediction"])
                records.append({
                    "language": row["language"],
                    "ablation": row["ablation"],
                    "follows_fr": st["follows_fr_structure"],
                    "follows_nl": st["follows_nl_structure"],
                    "follows_either": st["follows_either_structure"],
                })
            struct_df = pd.DataFrame(records)
            print("--- Structure conformity (by language and ablation) ---")
            for (lang, abl), g in struct_df.groupby(["language", "ablation"]):
                fr = g["follows_fr"].mean()
                nl = g["follows_nl"].mean()
                either = g["follows_either"].mean()
                print(f"  {lang} / {abl}: follows_fr={fr:.3f}  follows_nl={nl:.3f}  follows_either={either:.3f}  n={len(g)}")
            print()
            # Show a few predictions that fail follows_either
            fail = struct_df[~struct_df["follows_either"]]
            if not fail.empty:
                print("--- Sample rows where prediction follows NEITHER structure ---")
                sample = fail.head(5).index
                for idx in sample:
                    r = df.loc[idx]
                    print(f"  lang={r['language']} ablation={r['ablation']}")
                    print(f"    input:      {r['input']}")
                    print(f"    gold:      {r['gold']}")
                    print(f"    prediction: {r['prediction']}")
                    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
