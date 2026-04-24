"""
Dataset management for translation + conjugation experiments (april_exp).

Two key changes from march_exp:

1. Token format: the task token (<conjugate>/<translate>) now sits at the
   input/target boundary, replacing <sep>.
   Old: <sos> <task> input <sep> target <eos>
   New: <sos> input <task> target <eos>

2. Proportion semantics: ``prop`` now means the desired fraction of French
   *tokens* across ALL training data (conjugation + translation).  The
   conjugation FR/NL balance is derived so that the overall proportion
   matches the requested value, keeping translation balanced (50/50).
"""
import json
import logging
import random as rnd
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

from .model_config import ModelConfig
from .translation import (
    EXTRA_SPECIAL_TOKENS,
    TASK_TOKENS,
    TranslationLevel,
    examples_per_pair,
    format_translation_examples,
)

logger = logging.getLogger(__name__)


def _count_tokens_by_lang(
    rows: List[Dict[str, str]],
    fr_words: set,
    nl_words: set,
    tokenize_fn,
) -> Tuple[int, int, int]:
    """Count FR, NL, and total tokens across all rows (input + target).

    Returns (n_fr, n_nl, n_total).
    """
    n_fr = n_nl = n_total = 0
    for row in rows:
        for field in ("input", "target"):
            for tok in tokenize_fn(row[field]):
                n_total += 1
                t = tok.lower()
                if t in fr_words:
                    n_fr += 1
                elif t in nl_words:
                    n_nl += 1
    return n_fr, n_nl, n_total


class DatasetManager:

    def __init__(self, data_dir: str, config: ModelConfig, lexicon_path: str):
        self.data_dir = Path(data_dir)
        self.config = config
        self.tokenizer: Optional[PreTrainedTokenizerFast] = None
        self.lexicon: Optional[dict] = None
        self.lexicon_path = Path(lexicon_path)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Lexicon
    # ------------------------------------------------------------------

    def load_lexicon(self) -> dict:
        if self.lexicon is None:
            with open(self.lexicon_path, "r") as f:
                self.lexicon = json.load(f)
        return self.lexicon

    # ------------------------------------------------------------------
    # Sentence generation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _verb_form(verb_info: dict, lang: str, plural: bool) -> str:
        if lang == "nl":
            return verb_info["present"]["zijpl" if plural else "hij"]
        return verb_info["present"]["ils" if plural else "il"]

    def _make_sentences(
        self, lang: str, subj_key: str, obj_key: str, verb_key: str,
        subj_plural: bool, obj_plural: bool,
    ) -> Tuple[str, str]:
        """Return (present, present_perfect) for one language."""
        lex = self.load_lexicon()
        DET, NOUNS, VERBS, AUX = lex["DET"], lex["NOUNS"], lex["VERBS"], lex["AUX"]

        sd = DET[lang]["pl" if subj_plural else "sgl"]
        od = DET[lang]["pl" if obj_plural else "sgl"]
        s = NOUNS[lang][subj_key]["pl" if subj_plural else "sgl"]
        o = NOUNS[lang][obj_key]["pl" if obj_plural else "sgl"]

        v_info = VERBS[lang][verb_key]
        pres_v = self._verb_form(v_info, lang, subj_plural)
        part = v_info["participle"]
        aux_k = ("zijpl" if subj_plural else "hij") if lang == "nl" else (
            "ils" if subj_plural else "il")
        aux = AUX[lang][aux_k]

        present = f"{sd} {s} {pres_v} {od} {o}"
        if lang == "fr":
            perfect = f"{sd} {s} {aux} {part} {od} {o}"
        else:
            perfect = f"{sd} {s} {aux} {od} {o} {part}"
        return present, perfect

    # ------------------------------------------------------------------
    # Pairs generation & persistence
    # ------------------------------------------------------------------

    def generate_and_save_pairs(
        self, test_size: float = 0.2, seed: int = 0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate all sentence pairs from the lexicon and split train/test.

        Each row is a unique (subj, obj, verb, subj_plural, obj_plural) tuple
        with columns: subj_idx, obj_idx, verb_idx, subj_plural, obj_plural,
        fr_present, fr_perfect, nl_present, nl_perfect, subj_fr, obj_fr,
        verb_fr, subj_nl, obj_nl, verb_nl.

        Split is by (subj_idx, verb_idx) so subject–verb combos never leak.
        Saves train_pairs.csv and test_pairs.csv to self.data_dir.
        """
        lex = self.load_lexicon()
        fr_nouns = list(lex["NOUNS"]["fr"].keys())
        nl_nouns = list(lex["NOUNS"]["nl"].keys())
        fr_verbs = list(lex["VERBS"]["fr"].keys())
        nl_verbs = list(lex["VERBS"]["nl"].keys())

        rows: List[Dict[str, Any]] = []
        pl_options = [(False, False), (True, True), (False, True), (True, False)]

        for si in range(len(fr_nouns)):
            for oi in range(len(fr_nouns)):
                if si == oi:
                    continue
                for vi in range(len(fr_verbs)):
                    for sp, op in pl_options:
                        fr_pre, fr_prf = self._make_sentences(
                            "fr", fr_nouns[si], fr_nouns[oi], fr_verbs[vi], sp, op)
                        nl_pre, nl_prf = self._make_sentences(
                            "nl", nl_nouns[si], nl_nouns[oi], nl_verbs[vi], sp, op)
                        rows.append({
                            "subj_idx": si, "obj_idx": oi, "verb_idx": vi,
                            "subj_plural": sp, "obj_plural": op,
                            "fr_present": fr_pre, "fr_perfect": fr_prf,
                            "nl_present": nl_pre, "nl_perfect": nl_prf,
                            "subj_fr": fr_nouns[si], "obj_fr": fr_nouns[oi],
                            "verb_fr": fr_verbs[vi],
                            "subj_nl": nl_nouns[si], "obj_nl": nl_nouns[oi],
                            "verb_nl": nl_verbs[vi],
                        })

        df = pd.DataFrame(rows)

        sv_pairs = list(set(zip(df["subj_idx"], df["verb_idx"])))
        rng = rnd.Random(seed)
        n_test = int(test_size * len(sv_pairs))
        test_sv = set(rng.sample(sv_pairs, n_test))

        sv_col = list(zip(df["subj_idx"], df["verb_idx"]))
        mask = np.array([sv in test_sv for sv in sv_col], dtype=bool)
        train_pairs = df[~mask].reset_index(drop=True)
        test_pairs = df[mask].reset_index(drop=True)

        train_pairs.to_csv(self.data_dir / "train_pairs.csv", index=False)
        test_pairs.to_csv(self.data_dir / "test_pairs.csv", index=False)
        return train_pairs, test_pairs

    # ------------------------------------------------------------------
    # Build the actual training DataFrame from proportions
    # ------------------------------------------------------------------

    def build_training_data(
        self,
        prop: float,
        trans_frac: float,
        translation_level: TranslationLevel,
        seed: int,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Construct the training DataFrame from saved pairs.

        ``prop`` is the fraction of French examples within the conjugation
        task.  Translation is always balanced (50/50 fr2nl / nl2fr).

        After building the dataset, exact token counts are computed by
        matching every token against the lexicon to report both the
        conjugation FR token fraction and the overall FR token fraction.

        ``trans_frac`` is the desired fraction of *total training examples*
        that are translation.

        Returns (DataFrame with columns [input, target, task, lang],
                 stats dict with exact token counts).
        """
        pairs = pd.read_csv(self.data_dir / "train_pairs.csv")
        rng = np.random.RandomState(seed)
        P = len(pairs)

        pairs = pairs.sample(frac=1, random_state=seed).reset_index(drop=True)

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
                "fr_present": r.fr_present, "fr_perfect": r.fr_perfect,
                "nl_present": r.nl_present, "nl_perfect": r.nl_perfect,
            }
            candidates: List[Tuple[str, str, str]] = []
            for direction in ("fr2nl", "nl2fr"):
                for inp, tgt in format_translation_examples(
                    pair_dict, translation_level, direction
                ):
                    candidates.append((inp, tgt, direction))
            idx = rng.randint(len(candidates))
            inp, tgt, direction = candidates[idx]
            trans_rows.append({
                "input": inp, "target": tgt,
                "task": "translate", "lang": direction[:2],
            })

        # ── Conjugation examples ──
        n_conj = P - n_trans
        fr_rows = [
            {"input": r.fr_present, "target": r.fr_perfect,
             "task": "conjugate", "lang": "fr"}
            for r in conj_pairs_df.itertuples()
        ]
        nl_rows = [
            {"input": r.nl_present, "target": r.nl_perfect,
             "task": "conjugate", "lang": "nl"}
            for r in conj_pairs_df.itertuples()
        ]
        rng.shuffle(fr_rows)
        rng.shuffle(nl_rows)

        want_fr = int(n_conj * prop)
        want_nl = n_conj - want_fr

        conj_selected = fr_rows[:want_fr] + nl_rows[:want_nl]
        all_rows = conj_selected + trans_rows

        df = pd.DataFrame(all_rows)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # ── Exact token counts against the lexicon ──
        lex = self.load_lexicon()
        fr_words: Set[str] = set()
        nl_words: Set[str] = set()
        def _extract(obj, words):
            if isinstance(obj, str):
                words.add(obj.lower())
            elif isinstance(obj, dict):
                for v in obj.values():
                    _extract(v, words)
            elif isinstance(obj, list):
                for v in obj:
                    _extract(v, words)
        for section in lex.values():
            if "fr" in section:
                _extract(section["fr"], fr_words)
            if "nl" in section:
                _extract(section["nl"], nl_words)

        tok_re = re.compile(r"\w+|[^\s\w]")
        def _tokenize(s):
            return tok_re.findall(s.lower())

        conj_fr, conj_nl, conj_total = _count_tokens_by_lang(
            conj_selected, fr_words, nl_words, _tokenize)
        trans_fr, trans_nl, trans_total = _count_tokens_by_lang(
            trans_rows, fr_words, nl_words, _tokenize)
        all_fr = conj_fr + trans_fr
        all_nl = conj_nl + trans_nl
        all_total = conj_total + trans_total

        stats = {
            "n_conj_examples": len(conj_selected),
            "n_trans_examples": len(trans_rows),
            "n_total_examples": len(all_rows),
            "conj_fr_examples": want_fr,
            "conj_nl_examples": want_nl,
            "conj_fr_tokens": conj_fr,
            "conj_nl_tokens": conj_nl,
            "conj_total_tokens": conj_total,
            "conj_fr_token_frac": conj_fr / conj_total if conj_total else 0.0,
            "trans_fr_tokens": trans_fr,
            "trans_nl_tokens": trans_nl,
            "trans_total_tokens": trans_total,
            "trans_fr_token_frac": trans_fr / trans_total if trans_total else 0.0,
            "overall_fr_tokens": all_fr,
            "overall_nl_tokens": all_nl,
            "overall_total_tokens": all_total,
            "overall_fr_token_frac": all_fr / all_total if all_total else 0.0,
        }

        return df, stats

    # ------------------------------------------------------------------
    # Build a test/val DataFrame (conjugation only, both languages)
    # ------------------------------------------------------------------

    def build_eval_data(self) -> pd.DataFrame:
        """Load test pairs and return a conjugation-only DataFrame."""
        pairs = pd.read_csv(self.data_dir / "test_pairs.csv")
        rows: List[Dict[str, str]] = []
        for r in pairs.itertuples():
            rows.append({"input": r.fr_present, "target": r.fr_perfect,
                         "task": "conjugate", "lang": "fr"})
            rows.append({"input": r.nl_present, "target": r.nl_perfect,
                         "task": "conjugate", "lang": "nl"})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def build_tokenizer(self) -> PreTrainedTokenizerFast:
        if self.tokenizer is not None:
            return self.tokenizer

        vocab: Set[str] = set()
        tok_re = re.compile(r"\w+|[^\s\w]")
        for csv_name in ("train_pairs.csv", "test_pairs.csv"):
            path = self.data_dir / csv_name
            if not path.exists():
                continue
            df = pd.read_csv(path)
            for col in ("fr_present", "fr_perfect", "nl_present", "nl_perfect"):
                for text in df[col]:
                    vocab.update(tok_re.findall(str(text)))

        # <sep> kept in vocabulary for backward-compat with model_config's
        # tokenizer_config, but is never used in training or inference.
        base_special = ["<pad>", "<sos>", "<eos>", "<unk>", "<sep>"]
        all_special = base_special + EXTRA_SPECIAL_TOKENS
        vocab_list = all_special + sorted(vocab)

        tokenizer = Tokenizer(
            WordLevel(vocab={w: i for i, w in enumerate(vocab_list)},
                      unk_token="<unk>")
        )
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
        self, df: pd.DataFrame, tokenizer: Optional[PreTrainedTokenizerFast] = None,
        mask_input: bool = True,
    ) -> torch.utils.data.Dataset:
        if tokenizer is None:
            tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("No tokenizer. Call build_tokenizer() first.")

        # Pre-resolve task-token ids so encode() is a fast lookup
        task_token_ids = {
            task: tokenizer.convert_tokens_to_ids(tok)
            for task, tok in TASK_TOKENS.items()
        }

        def encode(input_str: str, target_str: str, task: str) -> Dict[str, list]:
            """Encode a single example with the new boundary-task-token format.

            Format: <sos> input_tokens <task_token> target_tokens <eos>
            Loss mask: everything up to and including <task_token> is -100.
            """
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
                        [f["input_ids"] + [pad] * (ml - len(f["input_ids"])) for f in feats],
                        dtype=torch.long),
                    "attention_mask": torch.tensor(
                        [f["attention_mask"] + [0] * (ml - len(f["attention_mask"])) for f in feats],
                        dtype=torch.long),
                    "labels": torch.tensor(
                        [f["labels"] + [-100] * (ml - len(f["labels"])) for f in feats],
                        dtype=torch.long),
                }

        return _Collator(tokenizer)
