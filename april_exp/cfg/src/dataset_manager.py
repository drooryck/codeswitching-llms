"""
Dataset management for the bilingual CFG experiment (april_exp).

Token format: <sos> input <task_token> target <eos>
  - The task token (<conjugate>/<translate>) sits at the input/target boundary.
  - Loss is masked over everything up to and including the task token (when mask_input=True).

Data is generated from the CFG grammar: unique trees are sampled per structure
type, linearized in both EN and NL for present and perfect tenses, then stored
as CSVs.  Train/test is split by (subj_key, verb_key) to prevent leakage.
"""
from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import WordPiece
from transformers import PreTrainedTokenizerFast

from .grammar import Lexicon, SentenceTree, STRUCTURE_TYPES, linearize, sample_tree
from .model_config import ModelConfig
from .translation import (
    EXTRA_SPECIAL_TOKENS,
    TASK_TOKENS,
    TranslationLevel,
    examples_per_pair,
    format_translation_examples,
)

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"\w+|[^\s\w]")


def _count_tokens_by_lang(
    rows: List[Dict[str, str]],
    en_words: set,
    nl_words: set,
) -> Tuple[int, int, int]:
    """Count EN, NL, and total tokens across all rows (input + target)."""
    n_en = n_nl = n_total = 0
    for row in rows:
        for field in ("input", "target"):
            for tok in TOKEN_RE.findall(row[field].lower()):
                n_total += 1
                if tok in en_words:
                    n_en += 1
                elif tok in nl_words:
                    n_nl += 1
    return n_en, n_nl, n_total


def _tree_key(tree: SentenceTree) -> tuple:
    """Hashable key for tree dedup."""
    parts: list = [
        tree.subj.noun_key, tree.subj.plural,
        tree.verb_key,
        tree.obj.noun_key, tree.obj.plural,
    ]
    for np_node in (tree.subj, tree.obj):
        if np_node.pp:
            parts.extend([np_node.pp.prep_key,
                          np_node.pp.np.noun_key, np_node.pp.np.plural])
        if np_node.rc:
            parts.extend([np_node.rc.verb_key,
                          np_node.rc.obj.noun_key, np_node.rc.obj.plural])
    return tuple(parts)


class DatasetManager:

    def __init__(self, data_dir: str, config: ModelConfig, lexicon_path: str):
        self.data_dir = Path(data_dir)
        self.config = config
        self.lexicon_path = Path(lexicon_path)
        self.tokenizer: Optional[PreTrainedTokenizerFast] = None
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pairs generation & persistence
    # ------------------------------------------------------------------

    def generate_and_save_pairs(
        self,
        n_trees_per_struct: int = 40_000,
        test_size: float = 0.2,
        seed: int = 0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Sample unique trees per structure type, linearize, and save as CSV.

        Split is by (subj_key, verb_key) so those combos never leak.
        """
        lex = Lexicon(self.lexicon_path)
        rng = random.Random(seed)

        rows: List[Dict[str, Any]] = []
        for struct in STRUCTURE_TYPES:
            seen: set = set()
            count = 0
            max_attempts = n_trees_per_struct * 3
            attempts = 0
            while count < n_trees_per_struct and attempts < max_attempts:
                tree = sample_tree(rng, lex, struct)
                key = _tree_key(tree)
                attempts += 1
                if key in seen:
                    continue
                seen.add(key)
                count += 1

                en_pre = " ".join(linearize(tree, "en", "present", lex))
                en_perf = " ".join(linearize(tree, "en", "perfect", lex))
                nl_pre = " ".join(linearize(tree, "nl", "present", lex))
                nl_perf = " ".join(linearize(tree, "nl", "perfect", lex))
                rows.append({
                    "structure": struct,
                    "subj_key": tree.subj.noun_key,
                    "verb_key": tree.verb_key,
                    "en_present": en_pre,
                    "en_perfect": en_perf,
                    "nl_present": nl_pre,
                    "nl_perfect": nl_perf,
                })

            logger.info("  %-20s  sampled %d unique trees (%d attempts)",
                        struct, count, attempts)

        df = pd.DataFrame(rows)

        sv_pairs = list(set(zip(df["subj_key"], df["verb_key"])))
        n_test = int(test_size * len(sv_pairs))
        test_sv = set(rng.sample(sv_pairs, n_test))

        mask = np.array(
            [sv in test_sv for sv in zip(df["subj_key"], df["verb_key"])],
            dtype=bool)
        train_pairs = df[~mask].reset_index(drop=True)
        test_pairs = df[mask].reset_index(drop=True)

        train_pairs.to_csv(self.data_dir / "train_pairs.csv", index=False)
        test_pairs.to_csv(self.data_dir / "test_pairs.csv", index=False)
        logger.info("Saved %d train + %d test pairs (%d trees/struct)",
                     len(train_pairs), len(test_pairs), n_trees_per_struct)
        return train_pairs, test_pairs

    # ------------------------------------------------------------------
    # Build training DataFrame from proportions
    # ------------------------------------------------------------------

    def build_training_data(
        self,
        prop: float,
        trans_frac: float,
        translation_level: TranslationLevel,
        seed: int,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Construct training DataFrame from saved pairs.

        ``prop`` is the fraction of English examples within conjugation.
        Translation is always balanced (50/50 en2nl / nl2en).
        ``trans_frac`` is the desired fraction of total training examples
        that are translation.

        Returns (DataFrame[input, target, task, lang, structure], stats dict).
        """
        pairs = pd.read_csv(self.data_dir / "train_pairs.csv")
        rng = np.random.RandomState(seed)
        pairs = pairs.sample(frac=1, random_state=seed).reset_index(drop=True)
        P = len(pairs)

        # One-per-pair design: total examples = P regardless of trans_frac.
        # Each translation pair contributes exactly 1 randomly sampled example
        # (out of 4 for tense_separate, 2 for full_sequence).
        use_trans = (translation_level != TranslationLevel.NONE and trans_frac > 0)
        n_trans = int(round(trans_frac * P)) if use_trans else 0
        n_trans = min(n_trans, P)

        trans_pairs_df = pairs.iloc[:n_trans]
        conj_pairs_df = pairs.iloc[n_trans:]

        # ── Translation examples (1 per pair, randomly sampled) ──
        trans_rows: List[Dict[str, str]] = []
        for r in trans_pairs_df.itertuples():
            pair_dict = {
                "en_present": r.en_present, "en_perfect": r.en_perfect,
                "nl_present": r.nl_present, "nl_perfect": r.nl_perfect,
            }
            candidates: List[Tuple[str, str, str]] = []
            for direction in ("en2nl", "nl2en"):
                for inp, tgt in format_translation_examples(
                    pair_dict, translation_level, direction
                ):
                    candidates.append((inp, tgt, direction))
            idx = rng.randint(len(candidates))
            inp, tgt, direction = candidates[idx]
            trans_rows.append({
                "input": inp, "target": tgt,
                "task": "translate", "lang": direction[:2],
                "structure": r.structure,
            })

        # ── Conjugation examples ──
        n_conj = P - n_trans
        en_rows = [
            {"input": r.en_present, "target": r.en_perfect,
             "task": "conjugate", "lang": "en", "structure": r.structure}
            for r in conj_pairs_df.itertuples()
        ]
        nl_rows = [
            {"input": r.nl_present, "target": r.nl_perfect,
             "task": "conjugate", "lang": "nl", "structure": r.structure}
            for r in conj_pairs_df.itertuples()
        ]
        rng.shuffle(en_rows)
        rng.shuffle(nl_rows)

        want_en = int(n_conj * prop)
        want_nl = n_conj - want_en

        conj_selected = en_rows[:want_en] + nl_rows[:want_nl]
        all_rows = conj_selected + trans_rows

        df = pd.DataFrame(all_rows)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # ── Exact token counts against the lexicon ──
        en_words, nl_words = self._extract_word_sets()

        conj_en, conj_nl, conj_total = _count_tokens_by_lang(
            conj_selected, en_words, nl_words)
        trans_en, trans_nl, trans_total = _count_tokens_by_lang(
            trans_rows, en_words, nl_words)
        all_en = conj_en + trans_en
        all_nl = conj_nl + trans_nl
        all_total = conj_total + trans_total

        stats = {
            "n_conj_examples": len(conj_selected),
            "n_trans_examples": len(trans_rows),
            "n_total_examples": len(all_rows),
            "conj_en_examples": want_en,
            "conj_nl_examples": want_nl,
            "conj_en_tokens": conj_en,
            "conj_nl_tokens": conj_nl,
            "conj_total_tokens": conj_total,
            "conj_en_token_frac": conj_en / conj_total if conj_total else 0.0,
            "trans_en_tokens": trans_en,
            "trans_nl_tokens": trans_nl,
            "trans_total_tokens": trans_total,
            "overall_en_tokens": all_en,
            "overall_nl_tokens": all_nl,
            "overall_total_tokens": all_total,
            "overall_en_token_frac": all_en / all_total if all_total else 0.0,
        }
        return df, stats

    # ------------------------------------------------------------------
    # Eval data (conjugation only, both languages)
    # ------------------------------------------------------------------

    def build_eval_data(self) -> pd.DataFrame:
        pairs = pd.read_csv(self.data_dir / "test_pairs.csv")
        rows: List[Dict[str, str]] = []
        for r in pairs.itertuples():
            rows.append({"input": r.en_present, "target": r.en_perfect,
                         "task": "conjugate", "lang": "en",
                         "structure": r.structure})
            rows.append({"input": r.nl_present, "target": r.nl_perfect,
                         "task": "conjugate", "lang": "nl",
                         "structure": r.structure})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Word sets for token counting
    # ------------------------------------------------------------------

    def _extract_word_sets(self) -> Tuple[set, set]:
        with open(self.lexicon_path, encoding="utf-8") as f:
            lex = json.load(f)

        en_words: Set[str] = set()
        nl_words: Set[str] = set()

        for form in lex["DET"]["en"].values():
            en_words.add(form.lower())
        for key in ("nl_de", "nl_het"):
            for form in lex["DET"][key].values():
                nl_words.add(form.lower())

        for noun_data in lex["NOUNS"].values():
            for form in noun_data["en"].values():
                if isinstance(form, str):
                    en_words.add(form.lower())
            for form in noun_data["nl"].values():
                if isinstance(form, str):
                    nl_words.add(form.lower())

        for verb_data in lex["VERBS"].values():
            for form in verb_data["en"]["present"].values():
                en_words.add(form.lower())
            en_words.add(verb_data["en"]["participle"].lower())
            for form in verb_data["nl"]["present"].values():
                nl_words.add(form.lower())
            nl_words.add(verb_data["nl"]["participle"].lower())

        for form in lex["AUX"]["en"].values():
            en_words.add(form.lower())
        for form in lex["AUX"]["nl"].values():
            nl_words.add(form.lower())

        for prep_data in lex["PREP"].values():
            en_words.add(prep_data["en"].lower())
            nl_words.add(prep_data["nl"].lower())

        en_words.add(lex["REL"]["en"].lower())
        nl_words.add(lex["REL"]["nl_de"].lower())
        nl_words.add(lex["REL"]["nl_het"].lower())

        return en_words, nl_words

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def build_tokenizer(self) -> PreTrainedTokenizerFast:
        if self.tokenizer is not None:
            return self.tokenizer

        vocab: Set[str] = set()
        for csv_name in ("train_pairs.csv", "test_pairs.csv"):
            path = self.data_dir / csv_name
            if not path.exists():
                continue
            df = pd.read_csv(path)
            for col in ("en_present", "en_perfect", "nl_present", "nl_perfect"):
                for text in df[col]:
                    vocab.update(TOKEN_RE.findall(str(text)))

        base_special = ["<pad>", "<sos>", "<eos>", "<unk>", "<sep>"]
        all_special = base_special + EXTRA_SPECIAL_TOKENS
        vocab_list = all_special + sorted(vocab)

        tokenizer = Tokenizer(
            WordLevel(vocab={w: i for i, w in enumerate(vocab_list)},
                      unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.decoder = WordPiece()

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            additional_special_tokens=EXTRA_SPECIAL_TOKENS,
            **self.config.tokenizer_config,
        )
        return self.tokenizer

    # ------------------------------------------------------------------
    # PyTorch dataset & collator
    # ------------------------------------------------------------------

    def create_pytorch_dataset(
        self, df: pd.DataFrame,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        mask_input: bool = True,
    ) -> torch.utils.data.Dataset:
        if tokenizer is None:
            tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("No tokenizer. Call build_tokenizer() first.")

        task_token_ids = {
            task: tokenizer.convert_tokens_to_ids(tok)
            for task, tok in TASK_TOKENS.items()
        }

        def encode(input_str: str, target_str: str, task: str) -> Dict[str, list]:
            """Format: <sos> input_tokens <task_token> target_tokens <eos>"""
            input_ids = tokenizer.encode(input_str, add_special_tokens=False)
            target_ids = tokenizer.encode(target_str, add_special_tokens=False)
            task_tok_id = task_token_ids[task]
            ids = (
                [tokenizer.bos_token_id] + input_ids
                + [task_tok_id] + target_ids
                + [tokenizer.eos_token_id]
            )
            task_pos = 1 + len(input_ids)
            if mask_input:
                labels = [-100] * (task_pos + 1) + ids[task_pos + 1:]
            else:
                labels = list(ids)
            return {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": labels,
            }

        class _Dataset(torch.utils.data.Dataset):
            def __init__(self, df):
                self.data = [
                    encode(r.input, r.target, r.task)
                    for r in df.itertuples()
                ]
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]

        return _Dataset(df)

    def create_collator(self, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizer

        class _Collator:
            def __init__(self, tok, mult=8):
                self.tok = tok
                self.mult = mult
            def __call__(self, feats):
                ml = max(len(f["input_ids"]) for f in feats)
                if self.mult:
                    ml = ((ml + self.mult - 1) // self.mult) * self.mult
                pad = self.tok.pad_token_id
                return {
                    "input_ids": torch.tensor(
                        [f["input_ids"] + [pad] * (ml - len(f["input_ids"]))
                         for f in feats], dtype=torch.long),
                    "attention_mask": torch.tensor(
                        [f["attention_mask"] + [0] * (ml - len(f["attention_mask"]))
                         for f in feats], dtype=torch.long),
                    "labels": torch.tensor(
                        [f["labels"] + [-100] * (ml - len(f["labels"]))
                         for f in feats], dtype=torch.long),
                }

        return _Collator(tokenizer)
