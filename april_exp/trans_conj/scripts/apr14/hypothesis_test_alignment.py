"""Paired hypothesis test: does masking produce different codeswitching (lexical score)
than no-masking, sentence-by-sentence on test predictions?

Memory-efficient: accumulates running sums per slice instead of storing all diffs.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict

RUNS_ROOT = Path("/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/"
                 "april_exp/trans_conj/results/mask_sweep_apr14/runs")
LEXICON_PATH = Path("/n/home06/drooryck/codeswitching-llms/"
                    "april_exp/trans_conj/data/lexicon_sep22.json")
OUT_DIR = Path("/n/home06/drooryck/codeswitching-llms/"
               "april_exp/trans_conj/scripts/apr14")

LEVELS = ["tense_separate", "full_sequence"]
PROPS = [0.01, 0.1, 0.5, 0.9, 0.99]
TFS = [0.01, 0.1, 0.5, 0.9, 0.99]


def build_vocab_sets(lexicon_path):
    with open(lexicon_path) as f:
        lex = json.load(f)
    fr_words, nl_words = set(), set()

    def _collect(obj, target_set):
        if isinstance(obj, str):
            target_set.add(obj.lower())
        elif isinstance(obj, dict):
            for v in obj.values():
                _collect(v, target_set)

    for cat_data in lex.values():
        if "fr" in cat_data:
            _collect(cat_data["fr"], fr_words)
        if "nl" in cat_data:
            _collect(cat_data["nl"], nl_words)

    shared = fr_words & nl_words
    return fr_words - shared, nl_words - shared


def lexical_scores(preds: np.ndarray, fr_only: set, nl_only: set) -> np.ndarray:
    """FR-fraction for each prediction string."""
    out = np.zeros(len(preds), dtype=np.float64)
    for i, pred in enumerate(preds):
        if not isinstance(pred, str) or not pred.strip():
            continue
        tokens = pred.lower().split()
        n_fr = sum(1 for t in tokens if t in fr_only)
        n_nl = sum(1 for t in tokens if t in nl_only)
        total = n_fr + n_nl
        if total > 0:
            out[i] = n_fr / total
    return out


class RunningStats:
    """Welford online algorithm for mean/variance without storing all values."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update_batch(self, values: np.ndarray):
        for x in values:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2

    @property
    def std(self):
        return np.sqrt(self.M2 / self.n) if self.n > 1 else 0.0

    @property
    def se(self):
        return self.std / np.sqrt(self.n) if self.n > 0 else 0.0


def main():
    print("Building vocabulary sets...")
    fr_only, nl_only = build_vocab_sets(LEXICON_PATH)
    print(f"  FR-only: {len(fr_only)} words, NL-only: {len(nl_only)} words")

    # Accumulators for sliced aggregation
    global_stats = RunningStats()
    lang_stats = {l: RunningStats() for l in ["nl", "fr"]}
    level_stats = {l: RunningStats() for l in LEVELS}
    ts_tf_stats = {  # tense_separate by tf bucket
        "low": RunningStats(),   # tf in 0.01, 0.1
        "mid": RunningStats(),   # tf = 0.5
        "high": RunningStats(),  # tf in 0.9, 0.99
    }
    config_results = []
    total_configs = 0

    for level in LEVELS:
        for prop in PROPS:
            for tf in TFS:
                mask_path = RUNS_ROOT / f"mask_{level}_prop{prop}_tf{tf}_run01" / "test_predictions.csv"
                nomask_path = RUNS_ROOT / f"nomask_{level}_prop{prop}_tf{tf}_run01" / "test_predictions.csv"
                if not mask_path.exists() or not nomask_path.exists():
                    continue
                total_configs += 1

                mask_df = pd.read_csv(mask_path)
                nomask_df = pd.read_csv(nomask_path)

                m_lex = lexical_scores(mask_df["prediction"].values, fr_only, nl_only)
                n_lex = lexical_scores(nomask_df["prediction"].values, fr_only, nl_only)

                for lang_code in ["nl", "fr"]:
                    lang_mask = (mask_df.language == lang_code).values
                    m_sub = m_lex[lang_mask]
                    n_sub = n_lex[lang_mask]

                    if len(m_sub) == 0 or len(m_sub) != len(n_sub):
                        continue

                    diff = m_sub - n_sub

                    # Feed running stats
                    global_stats.update_batch(diff)
                    lang_stats[lang_code].update_batch(diff)
                    level_stats[level].update_batch(diff)
                    if level == "tense_separate":
                        if tf <= 0.1:
                            ts_tf_stats["low"].update_batch(diff)
                        elif tf <= 0.5:
                            ts_tf_stats["mid"].update_batch(diff)
                        else:
                            ts_tf_stats["high"].update_batch(diff)

                    # Per-config paired t-test
                    t_stat, p_val = stats.ttest_rel(m_sub, n_sub)
                    config_results.append({
                        "level": level, "prop": prop, "tf": tf, "language": lang_code,
                        "n_sentences": len(m_sub),
                        "mask_lex_mean": float(m_sub.mean()),
                        "nomask_lex_mean": float(n_sub.mean()),
                        "diff_mean": float(diff.mean()),
                        "diff_std": float(diff.std()),
                        "t_stat": float(t_stat),
                        "p_value": float(p_val),
                    })

                # Free memory
                del mask_df, nomask_df, m_lex, n_lex
                if total_configs % 5 == 0:
                    print(f"  Processed {total_configs}/50 configs...")

    cfg = pd.DataFrame(config_results)

    print(f"\nProcessed {total_configs} configs, {global_stats.n:,} total sentence pairs")

    # ═══════════════════════════════════════════════════════════════
    def report(label, rs):
        if rs.n == 0:
            print(f"  (no data)")
            return
        t_val = rs.mean / rs.se if rs.se > 0 else 0
        p_val = 2 * (1 - stats.t.cdf(abs(t_val), df=rs.n - 1)) if rs.n > 1 else 1.0
        ci_lo = rs.mean - 1.96 * rs.se
        ci_hi = rs.mean + 1.96 * rs.se
        print(f"  N sentence pairs:        {rs.n:,}")
        print(f"  Mean diff (mask-nomask): {rs.mean:+.6f}")
        print(f"  Std of diffs:            {rs.std:.6f}")
        print(f"  SE:                      {rs.se:.6f}")
        print(f"  t-statistic:             {t_val:.4f}")
        print(f"  p-value (two-sided):     {p_val:.2e}")
        print(f"  95% CI:                  [{ci_lo:+.6f}, {ci_hi:+.6f}]")
        print(f"  Significant at α=0.05?   {'YES' if p_val < 0.05 else 'NO'}")
        if rs.mean < 0:
            print(f"  Direction: NOMASK has higher FR lexical (more CS in NL outputs)")
        elif rs.mean > 0:
            print(f"  Direction: MASK has higher FR lexical (more CS in NL outputs)")
        else:
            print(f"  Direction: no difference")

    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("GLOBAL PAIRED TEST: mask vs nomask lexical score (all sentences pooled)")
    print("  H0: mean(mask_lexical - nomask_lexical) = 0")
    print("  Positive diff → mask has MORE FR tokens in output")
    print("=" * 80)
    report("global", global_stats)

    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("BY OUTPUT LANGUAGE")
    print("  NL outputs: higher lexical = more FR words in Dutch = more codeswitching")
    print("  FR outputs: lower lexical = more NL words in French = more codeswitching")
    print("=" * 80)
    for lang_code in ["nl", "fr"]:
        print(f"\n  ── {lang_code.upper()} outputs ──")
        report(lang_code, lang_stats[lang_code])

    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("BY TRANSLATION LEVEL")
    print("=" * 80)
    for level in LEVELS:
        print(f"\n  ── {level} ──")
        report(level, level_stats[level])

    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("TENSE_SEPARATE: BY TF BUCKET")
    print("=" * 80)
    for bucket, label in [("low", "low tf (0.01, 0.1)"),
                           ("mid", "mid tf (0.5)"),
                           ("high", "high tf (0.9, 0.99)")]:
        print(f"\n  ── {label} ──")
        report(bucket, ts_tf_stats[bucket])

    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PER-CONFIG RESULTS (top 20 by |effect|)")
    print("=" * 80)
    cfg_sorted = cfg.reindex(cfg["diff_mean"].abs().sort_values(ascending=False).index)
    sig = cfg_sorted[cfg_sorted.p_value < 0.05]
    pos_sig = sig[sig.diff_mean > 0]
    neg_sig = sig[sig.diff_mean < 0]
    print(f"\n  {len(sig)} / {len(cfg_sorted)} configs significant at α=0.05")
    print(f"  {len(pos_sig)} where mask > nomask (mask has more FR tokens)")
    print(f"  {len(neg_sig)} where nomask > mask (nomask has more FR tokens)")
    print()

    print(f"  {'level':<18s} {'prop':>5s} {'tf':>5s} {'lang':>4s}  "
          f"{'mask_lex':>8s} {'no_lex':>8s} {'diff':>8s} {'std':>8s} {'p':>10s} {'sig':>3s}")
    print("  " + "-" * 88)
    for _, r in cfg_sorted.head(20).iterrows():
        star = " * " if r.p_value < 0.05 else "   "
        print(f"  {r.level:<18s} {r.prop:5.2f} {r.tf:5.2f} {r.language:>4s}  "
              f"{r.mask_lex_mean:8.4f} {r.nomask_lex_mean:8.4f} "
              f"{r.diff_mean:+8.4f} {r.diff_std:8.4f} {r.p_value:10.2e}{star}")

    # Save CSV
    cfg.to_csv(OUT_DIR / "hypothesis_test_results.csv", index=False)
    print(f"\nFull per-config results saved to {OUT_DIR / 'hypothesis_test_results.csv'}")


if __name__ == "__main__":
    main()
