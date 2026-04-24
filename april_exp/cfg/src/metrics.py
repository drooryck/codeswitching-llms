"""
Metrics for the CFG bilingual experiment.

Lexical score: fraction of tokens that are English (1.0) vs Dutch (0.0).
Syntax score: at positions where EN/NL role templates diverge, check whether
              the predicted token's role matches EN or NL.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


TOKEN_RE = re.compile(r"\w+|[^\s\w]")

# Perfect-tense role templates per structure type.
# These are deterministic: structure type fully determines the role sequence.
ROLE_TEMPLATES = {
    "plain": {
        "en": ["det","noun","aux","part","det","noun"],
        "nl": ["det","noun","aux","det","noun","part"],
    },
    "subj_pp": {
        "en": ["det","noun","prep","det","noun","aux","part","det","noun"],
        "nl": ["det","noun","prep","det","noun","aux","det","noun","part"],
    },
    "obj_pp": {
        "en": ["det","noun","aux","part","det","noun","prep","det","noun"],
        "nl": ["det","noun","aux","det","noun","prep","det","noun","part"],
    },
    "subj_rc": {
        "en": ["det","noun","rel","verb","det","noun","aux","part","det","noun"],
        "nl": ["det","noun","rel","det","noun","verb","aux","det","noun","part"],
    },
    "obj_rc": {
        "en": ["det","noun","aux","part","det","noun","rel","verb","det","noun"],
        "nl": ["det","noun","aux","det","noun","rel","det","noun","verb","part"],
    },
    "subj_pp+obj_pp": {
        "en": ["det","noun","prep","det","noun","aux","part","det","noun","prep","det","noun"],
        "nl": ["det","noun","prep","det","noun","aux","det","noun","prep","det","noun","part"],
    },
}


class Metrics:

    def __init__(self, lexicon_path: str | Path):
        with open(lexicon_path, encoding="utf-8") as f:
            lex = json.load(f)

        # Build role-classification sets (language-agnostic: we only need role, not language)
        self.dets: set[str] = set()
        for key in ("en", "nl_de", "nl_het"):
            for form in lex["DET"][key].values():
                self.dets.add(form.lower())

        self.nouns: set[str] = set()
        for noun_data in lex["NOUNS"].values():
            for lang in ("en", "nl"):
                for form in noun_data[lang].values():
                    if isinstance(form, str):
                        self.nouns.add(form.lower())

        self.verbs_present: set[str] = set()
        self.participles: set[str] = set()
        for verb_data in lex["VERBS"].values():
            for lang in ("en", "nl"):
                for form in verb_data[lang]["present"].values():
                    self.verbs_present.add(form.lower())
                self.participles.add(verb_data[lang]["participle"].lower())

        self.auxes: set[str] = set()
        for lang in ("en", "nl"):
            for form in lex["AUX"][lang].values():
                self.auxes.add(form.lower())

        self.preps: set[str] = set()
        for prep_data in lex["PREP"].values():
            for form in prep_data.values():
                self.preps.add(form.lower())

        self.rels: set[str] = set()
        self.rels.add(lex["REL"]["en"].lower())
        self.rels.add(lex["REL"]["nl_de"].lower())
        self.rels.add(lex["REL"]["nl_het"].lower())

        # Build language-membership sets for lexical score
        self.en_words: set[str] = set()
        self.nl_words: set[str] = set()
        for key in ("en",):
            for form in lex["DET"][key].values():
                self.en_words.add(form.lower())
        for key in ("nl_de", "nl_het"):
            for form in lex["DET"][key].values():
                self.nl_words.add(form.lower())

        for noun_data in lex["NOUNS"].values():
            for form in noun_data["en"].values():
                if isinstance(form, str):
                    self.en_words.add(form.lower())
            for form in noun_data["nl"].values():
                if isinstance(form, str):
                    self.nl_words.add(form.lower())

        for verb_data in lex["VERBS"].values():
            for form in verb_data["en"]["present"].values():
                self.en_words.add(form.lower())
            self.en_words.add(verb_data["en"]["participle"].lower())
            for form in verb_data["nl"]["present"].values():
                self.nl_words.add(form.lower())
            self.nl_words.add(verb_data["nl"]["participle"].lower())

        for lang_set, lang in ((self.en_words, "en"), (self.nl_words, "nl")):
            for form in lex["AUX"][lang].values():
                lang_set.add(form.lower())

        for prep_data in lex["PREP"].values():
            self.en_words.add(prep_data["en"].lower())
            self.nl_words.add(prep_data["nl"].lower())

        self.en_words.add(lex["REL"]["en"].lower())
        self.nl_words.add(lex["REL"]["nl_de"].lower())
        self.nl_words.add(lex["REL"]["nl_het"].lower())

        # Pre-compute divergent positions per structure type
        self.divergent_positions: dict[str, list[int]] = {}
        self.en_expected: dict[str, list[str]] = {}
        self.nl_expected: dict[str, list[str]] = {}
        for struct, templates in ROLE_TEMPLATES.items():
            en_t = templates["en"]
            nl_t = templates["nl"]
            divs = [i for i in range(len(en_t)) if en_t[i] != nl_t[i]]
            self.divergent_positions[struct] = divs
            self.en_expected[struct] = [en_t[i] for i in divs]
            self.nl_expected[struct] = [nl_t[i] for i in divs]

    def tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in TOKEN_RE.findall(text)]

    def classify_role(self, token: str) -> str:
        t = token.lower()
        if t in self.rels:
            return "rel"
        if t in self.auxes:
            return "aux"
        if t in self.participles:
            return "part"
        if t in self.preps:
            return "prep"
        if t in self.dets:
            return "det"
        if t in self.verbs_present:
            return "verb"
        if t in self.nouns:
            return "noun"
        return "unknown"

    def lexical_score(self, text: str) -> float:
        """Fraction of tokens that are English. EN-only=1.0, NL-only=0.0, shared=0.5."""
        tokens = self.tokenize(text)
        if not tokens:
            return 0.5
        total = 0.0
        count = 0
        for t in tokens:
            is_en = t in self.en_words
            is_nl = t in self.nl_words
            if is_en and not is_nl:
                total += 1.0
            elif is_nl and not is_en:
                total += 0.0
            elif is_en and is_nl:
                total += 0.5
            else:
                continue
            count += 1
        return total / count if count > 0 else 0.5

    def syntax_score(self, text: str, structure_tag: str) -> dict:
        """Score word order: 1.0 = EN syntax, 0.0 = NL syntax.

        Only evaluates divergent positions where the token actually matches
        one of the two expected roles (EN or NL). Positions matching neither
        are excluded, giving a cleaner signal of syntactic preference.
        Pair with conformity_score to measure overall well-formedness.

        Returns dict with:
            score: float or None (None if no valid divergent positions)
            valid_positions: int (how many divergent positions matched EN or NL)
            total_divergent: int (total divergent positions for this structure)
        """
        if structure_tag not in ROLE_TEMPLATES:
            return {"score": None, "valid_positions": 0, "total_divergent": 0}

        tokens = self.tokenize(text)
        expected_len = len(ROLE_TEMPLATES[structure_tag]["en"])
        if len(tokens) != expected_len:
            return {"score": None, "valid_positions": 0,
                    "total_divergent": len(self.divergent_positions.get(structure_tag, []))}

        divs = self.divergent_positions[structure_tag]
        en_exp = self.en_expected[structure_tag]
        nl_exp = self.nl_expected[structure_tag]

        total = 0.0
        valid = 0
        for idx, en_role, nl_role in zip(divs, en_exp, nl_exp):
            pred_role = self.classify_role(tokens[idx])
            if pred_role == en_role:
                total += 1.0
                valid += 1
            elif pred_role == nl_role:
                valid += 1

        score = total / valid if valid > 0 else None
        return {"score": score, "valid_positions": valid, "total_divergent": len(divs)}

    def conformity_score(self, text: str, structure_tag: str) -> dict:
        """Fraction of output tokens conforming to EITHER language's grammar.

        For each position i in the output:
          - If i < template_length: 1 if classify_role(token[i]) matches
            en_template[i] OR nl_template[i], else 0.
          - If i >= template_length: 0 (extra tokens are always wrong).

        Denominator is max(len(output), template_length) so both too-short
        and too-long outputs are penalized.

        Returns dict with:
            score: float in [0, 1]
            matches: int (number of conforming positions)
            denominator: int
        """
        if structure_tag not in ROLE_TEMPLATES:
            return {"score": None, "matches": 0, "denominator": 0}

        tokens = self.tokenize(text)
        en_t = ROLE_TEMPLATES[structure_tag]["en"]
        nl_t = ROLE_TEMPLATES[structure_tag]["nl"]
        template_len = len(en_t)

        matches = 0
        for i, tok in enumerate(tokens):
            if i >= template_len:
                break
            role = self.classify_role(tok)
            if role == en_t[i] or role == nl_t[i]:
                matches += 1

        denom = max(len(tokens), template_len)
        return {"score": matches / denom if denom > 0 else 0.0,
                "matches": matches, "denominator": denom}

    def per_position_pos_validity(self, text: str, structure_tag: str) -> Optional[float]:
        """Fraction of token positions whose POS matches either template.

        Returns None when the prediction length doesn't match the template
        length for this structure type.
        """
        if structure_tag not in ROLE_TEMPLATES:
            return None
        en_t = ROLE_TEMPLATES[structure_tag]["en"]
        nl_t = ROLE_TEMPLATES[structure_tag]["nl"]
        tokens = self.tokenize(text)
        if len(tokens) != len(en_t):
            return None
        hits = sum(
            self.classify_role(t) in {en_t[i], nl_t[i]}
            for i, t in enumerate(tokens)
        )
        return hits / len(en_t)

    def pos_coverage_rate(self, text: str, structure_tag: str) -> float:
        """Position-aware POS coverage over the output.

        For each of the first template_len positions, checks whether the
        token's POS matches what either the EN or NL template expects.
        Tokens beyond the template length always score 0.  The denominator
        is max(len(tokens), template_len) so both short and long outputs
        are penalised.

        Returns a value in [0, 1].  Never returns None.
        """
        if structure_tag not in ROLE_TEMPLATES:
            return 0.0
        en_t = ROLE_TEMPLATES[structure_tag]["en"]
        nl_t = ROLE_TEMPLATES[structure_tag]["nl"]
        template_len = len(en_t)
        tokens = self.tokenize(text)
        if not tokens:
            return 0.0
        denom = max(len(tokens), template_len)
        hits = 0
        for i, t in enumerate(tokens):
            if i >= template_len:
                break
            if self.classify_role(t) in {en_t[i], nl_t[i]}:
                hits += 1
        return hits / denom

    def alignment_score(self, text: str, structure_tag: str) -> Optional[float]:
        """Average of lexical and syntax scores."""
        lex = self.lexical_score(text)
        syn = self.syntax_score(text, structure_tag)["score"]
        if syn is None:
            return None
        return (lex + syn) / 2.0

    def exact_match(self, pred: str, gold: str) -> bool:
        return pred.strip() == gold.strip()
