"""
Translation pair generation for the bilingual CFG experiment.

Supports multiple granularity levels:
  NONE            — no translation examples
  TENSE_SEPARATE  — present→present AND perfect→perfect as separate examples
  FULL_SEQUENCE   — present <pair> perfect → present <pair> perfect (concatenated)

Token format (april_exp):
  <sos> input_sentence <task_token> target_sentence <eos>
"""
from enum import Enum
from typing import Dict, List, Tuple


class TranslationLevel(Enum):
    NONE = "none"
    TENSE_SEPARATE = "tense_separate"
    FULL_SEQUENCE = "full_sequence"


TASK_TOKENS = {
    "conjugate": "<conjugate>",
    "translate": "<translate>",
}
PAIR_TOKEN = "<pair>"

EXTRA_SPECIAL_TOKENS = ["<conjugate>", "<translate>", "<pair>"]


def examples_per_pair(level: TranslationLevel) -> int:
    """How many training examples one pair produces (both directions combined)."""
    if level == TranslationLevel.TENSE_SEPARATE:
        return 4   # 2 tenses × 2 directions
    elif level == TranslationLevel.FULL_SEQUENCE:
        return 2   # 1 concatenated × 2 directions
    return 0


def format_translation_examples(
    pair: Dict[str, str],
    level: TranslationLevel,
    direction: str,
) -> List[Tuple[str, str]]:
    """Build (input_str, target_str) tuples for one translation direction.

    The returned strings do NOT include <sos>/<eos> or task tokens.

    Args:
        pair: Dict with keys en_present, en_perfect, nl_present, nl_perfect.
        level: Translation granularity.
        direction: "en2nl" or "nl2en".
    """
    if direction == "en2nl":
        src_pre, src_perf = pair["en_present"], pair["en_perfect"]
        tgt_pre, tgt_perf = pair["nl_present"], pair["nl_perfect"]
    elif direction == "nl2en":
        src_pre, src_perf = pair["nl_present"], pair["nl_perfect"]
        tgt_pre, tgt_perf = pair["en_present"], pair["en_perfect"]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    if level == TranslationLevel.TENSE_SEPARATE:
        return [(src_pre, tgt_pre), (src_perf, tgt_perf)]
    elif level == TranslationLevel.FULL_SEQUENCE:
        return [(
            f"{src_pre} {PAIR_TOKEN} {src_perf}",
            f"{tgt_pre} {PAIR_TOKEN} {tgt_perf}",
        )]
    else:
        raise ValueError(f"Cannot format translation for level={level}")
