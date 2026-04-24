"""
Ablation experiment for CFG bilingual models.

For each test sentence, swap the subject-head (det+noun), main verb,
or object-head (det+noun) with the equivalent tokens from the other
language.  Feed the ablated present-tense sentence to the model and
measure how the output metrics (lexical_score, syntax_score) shift.

Usage:
    python -m april_exp.cfg.scripts.ablation.run_ablation \
        --model-dir <path_to_final_model> \
        --prop <prop_value> --seed <seed_value>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

# ── Make sure project root is importable ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from april_exp.cfg.src.metrics import Metrics
from april_exp.cfg.src.translation import TASK_TOKENS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Token positions in present-tense sentences ────────────────────
# Present tense is always SVO in both EN and NL.
# subj_head = (start, end) slice for the subject det+noun
# verb = index of the main verb
# obj_head = (start, end) slice for the object det+noun

PRESENT_POSITIONS: Dict[str, Dict] = {
    "plain":          {"subj_head": (0, 2), "verb": 2, "obj_head": (3, 5)},
    "subj_pp":        {"subj_head": (0, 2), "verb": 5, "obj_head": (6, 8)},
    "obj_pp":         {"subj_head": (0, 2), "verb": 2, "obj_head": (3, 5)},
    "subj_rc":        {"subj_head": (0, 2), "verb": 6, "obj_head": (7, 9)},
    "obj_rc":         {"subj_head": (0, 2), "verb": 2, "obj_head": (3, 5)},
    "subj_pp+obj_pp": {"subj_head": (0, 2), "verb": 5, "obj_head": (6, 8)},
}


def create_ablated_inputs(
    pairs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create ablated present-tense inputs from test pair data.

    For each pair and each source language (en/nl), produce:
      - original: unmodified present-tense input
      - subject:  subject det+noun swapped with other language
      - verb:     main verb swapped with other language
      - object:   object det+noun swapped with other language

    Returns a DataFrame with columns:
      source_lang, ablation, input_text, gold_perfect, structure, pair_idx
    """
    rows: List[Dict] = []

    for idx, pair in pairs_df.iterrows():
        struct = pair["structure"]
        pos = PRESENT_POSITIONS[struct]
        s0, s1 = pos["subj_head"]
        v = pos["verb"]
        o0, o1 = pos["obj_head"]

        en_tokens = pair["en_present"].split()
        nl_tokens = pair["nl_present"].split()

        for src_lang, src_tokens, tgt_tokens, gold in [
            ("en", en_tokens, nl_tokens, pair["en_perfect"]),
            ("nl", nl_tokens, en_tokens, pair["nl_perfect"]),
        ]:
            base = {
                "source_lang": src_lang,
                "gold_perfect": gold,
                "structure": struct,
                "pair_idx": idx,
            }

            # Original (no ablation)
            rows.append({**base, "ablation": "none",
                         "input_text": " ".join(src_tokens)})

            # Subject ablation
            abl = src_tokens.copy()
            abl[s0:s1] = tgt_tokens[s0:s1]
            rows.append({**base, "ablation": "subject",
                         "input_text": " ".join(abl)})

            # Verb ablation
            abl = src_tokens.copy()
            abl[v] = tgt_tokens[v]
            rows.append({**base, "ablation": "verb",
                         "input_text": " ".join(abl)})

            # Object ablation
            abl = src_tokens.copy()
            abl[o0:o1] = tgt_tokens[o0:o1]
            rows.append({**base, "ablation": "object",
                         "input_text": " ".join(abl)})

    return pd.DataFrame(rows)


def run_inference(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    inputs: List[str],
    device: torch.device,
    batch_size: int = 512,
) -> List[str]:
    """Run inference on a list of present-tense input strings."""
    model.eval()
    task_tok = TASK_TOKENS["conjugate"]
    predictions: List[str] = []

    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        prompts = [f"<sos> {inp} {task_tok}" for inp in batch]

        with torch.no_grad():
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            outputs = model.generate(
                **encoded, max_new_tokens=30, do_sample=False,
                num_beams=1, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        for text in tokenizer.batch_decode(outputs, skip_special_tokens=False):
            pred = (text.split(task_tok)[1]
                    .replace("<eos>", "").replace("<pad>", "").strip()
                    if task_tok in text else "")
            predictions.append(pred)

    return predictions


def compute_metrics_batch(
    metrics: Metrics,
    predictions: List[str],
    structures: List[str],
) -> pd.DataFrame:
    """Compute lexical and syntax scores for each prediction."""
    rows = []
    for pred, struct in zip(predictions, structures):
        lex = metrics.lexical_score(pred)
        syn = metrics.syntax_score(pred, struct)
        conf = metrics.conformity_score(pred, struct)
        rows.append({
            "lexical_score": lex,
            "syntax_score": syn["score"],
            "syntax_valid": syn["valid_positions"],
            "conformity_score": conf["score"],
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to model directory containing 'final/' subdirectory")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to data directory with test_pairs.csv")
    parser.add_argument("--lexicon-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Where to save ablation results")
    parser.add_argument("--prop", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="If >0, subsample test pairs (stratified by structure)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load test pairs ───────────────────────────────────────────
    pairs_df = pd.read_csv(Path(args.data_dir) / "test_pairs.csv")
    log.info("Loaded %d test pairs", len(pairs_df))

    if args.max_pairs > 0 and args.max_pairs < len(pairs_df):
        pairs_df = (
            pairs_df.groupby("structure", group_keys=False)
            .apply(lambda g: g.sample(
                min(len(g), args.max_pairs // g.name.__class__(pairs_df["structure"].nunique())),
                random_state=42))
            .reset_index(drop=True)
        )
        # Simpler approach: just sample uniformly
        pairs_df = pairs_df.sample(n=min(args.max_pairs, len(pairs_df)),
                                   random_state=42).reset_index(drop=True)
        log.info("Subsampled to %d pairs", len(pairs_df))

    # ── Create ablated inputs ─────────────────────────────────────
    abl_df = create_ablated_inputs(pairs_df)
    log.info("Created %d ablated inputs (%d pairs × 2 langs × 4 conditions)",
             len(abl_df), len(pairs_df))

    # ── Load model and tokenizer ──────────────────────────────────
    model_path = Path(args.model_dir) / "final"
    log.info("Loading model from %s", model_path)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(str(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log.info("Model loaded on %s", device)

    # ── Run inference ─────────────────────────────────────────────
    log.info("Running inference on %d inputs...", len(abl_df))
    preds = run_inference(model, tokenizer, abl_df["input_text"].tolist(), device)
    abl_df["prediction"] = preds
    log.info("Inference complete.")

    # ── Compute metrics ───────────────────────────────────────────
    metrics = Metrics(args.lexicon_path)
    log.info("Computing metrics...")
    metrics_df = compute_metrics_batch(
        metrics, abl_df["prediction"].tolist(), abl_df["structure"].tolist())
    result_df = pd.concat([abl_df.reset_index(drop=True), metrics_df], axis=1)
    result_df["prop"] = args.prop
    result_df["seed"] = args.seed

    # ── Save full results ─────────────────────────────────────────
    out_file = output_dir / f"ablation_prop{args.prop}_seed{args.seed}.csv"
    result_df.to_csv(out_file, index=False)
    log.info("Saved %d rows to %s", len(result_df), out_file)

    # ── Print summary ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"ABLATION SUMMARY  prop={args.prop}  seed={args.seed}")
    print("=" * 70)

    for src_lang in ("en", "nl"):
        print(f"\n── Source language: {src_lang.upper()} ──")
        sub = result_df[result_df["source_lang"] == src_lang]
        summary = sub.groupby("ablation").agg(
            lexical_mean=("lexical_score", "mean"),
            lexical_std=("lexical_score", "std"),
            syntax_mean=("syntax_score", lambda x: x.dropna().mean()),
            conformity_mean=("conformity_score", lambda x: x.dropna().mean()),
            n=("lexical_score", "count"),
        ).reindex(["none", "subject", "verb", "object"])

        print(summary.to_string(float_format="%.4f"))

        # Compute deltas relative to original
        baseline_lex = summary.loc["none", "lexical_mean"]
        for abl in ["subject", "verb", "object"]:
            delta = summary.loc[abl, "lexical_mean"] - baseline_lex
            print(f"  Δ lexical ({abl:>7s}): {delta:+.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
