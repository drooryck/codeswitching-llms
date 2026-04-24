"""
Bilingual CFG for EN/NL sentence generation.

Grammar (PPs + relative clauses):
    S  -> NP_subj VP
    VP -> V NP_obj
    NP -> DET N | DET N PP | DET N RC
    PP -> P NP
    RC -> REL VP_sub          (subject-extracted relative clause)

Abstract syntax trees are language-neutral. Linearization is language-specific:
    EN: always SVO, even in subordinate clauses
    NL main clause: V OBJ (present), AUX OBJ PART (perfect)
    NL subordinate: OBJ V (present), OBJ PART AUX (perfect)
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set


# ---------------------------------------------------------------------------
# Lexicon wrapper
# ---------------------------------------------------------------------------

class Lexicon:
    def __init__(self, path: str | Path):
        with open(path, encoding="utf-8") as f:
            self._lex = json.load(f)
        self.noun_keys: list[str] = list(self._lex["NOUNS"].keys())
        self.verb_keys: list[str] = list(self._lex["VERBS"].keys())
        self.prep_keys: list[str] = list(self._lex["PREP"].keys())

    def det(self, lang: str, number: str, gender: str = "de") -> str:
        """Return determiner. *gender* only matters for NL singular."""
        if lang == "en":
            return self._lex["DET"]["en"][number]
        key = f"nl_{gender}" if number == "sgl" else "nl_de"
        return self._lex["DET"][key][number]

    def noun(self, lang: str, key: str, number: str) -> str:
        return self._lex["NOUNS"][key][lang][number]

    def noun_gender(self, key: str) -> str:
        return self._lex["NOUNS"][key]["nl"]["gender"]

    def verb_present(self, lang: str, key: str, plural: bool) -> str:
        form = "pl" if plural else "sg"
        return self._lex["VERBS"][key][lang]["present"][form]

    def participle(self, lang: str, key: str) -> str:
        return self._lex["VERBS"][key][lang]["participle"]

    def aux(self, lang: str, plural: bool) -> str:
        form = "pl" if plural else "sg"
        return self._lex["AUX"][lang][form]

    def prep(self, lang: str, key: str) -> str:
        return self._lex["PREP"][key][lang]

    def rel(self, lang: str, gender: str = "de", plural: bool = False) -> str:
        if lang == "en":
            return self._lex["REL"]["en"]
        if plural:
            return self._lex["REL"]["nl_de"]   # all NL plurals → "die"
        return self._lex["REL"][f"nl_{gender}"]


# ---------------------------------------------------------------------------
# Abstract syntax tree nodes
# ---------------------------------------------------------------------------

@dataclass
class PPNode:
    prep_key: str = ""
    np: NPNode | None = None


@dataclass
class RCNode:
    """Subject-extracted RC: 'that/die eats the wolf' / 'die de wolf eet'."""
    verb_key: str = ""
    obj: NPNode | None = None


@dataclass
class NPNode:
    noun_key: str = ""
    plural: bool = False
    pp: PPNode | None = None
    rc: RCNode | None = None


@dataclass
class SentenceTree:
    subj: NPNode = field(default_factory=NPNode)
    verb_key: str = ""
    obj: NPNode = field(default_factory=NPNode)
    structure_tag: str = ""


# ---------------------------------------------------------------------------
# Structure tag computation
# ---------------------------------------------------------------------------

def _np_tag(np: NPNode, role: str) -> str:
    parts = []
    if np.pp is not None:
        parts.append(f"{role}_pp")
    if np.rc is not None:
        parts.append(f"{role}_rc")
    return "+".join(parts) if parts else ""


def compute_structure_tag(tree: SentenceTree) -> str:
    parts = []
    stag = _np_tag(tree.subj, "subj")
    if stag:
        parts.append(stag)
    otag = _np_tag(tree.obj, "obj")
    if otag:
        parts.append(otag)
    return "+".join(parts) if parts else "plain"


# ---------------------------------------------------------------------------
# Linearization
# ---------------------------------------------------------------------------

def _linearize_np(np: NPNode, lang: str, lex: Lexicon) -> list[str]:
    gender = lex.noun_gender(np.noun_key)
    number = "pl" if np.plural else "sgl"
    tokens = [
        lex.det(lang, number, gender),
        lex.noun(lang, np.noun_key, number),
    ]
    if np.pp is not None:
        tokens += _linearize_pp(np.pp, lang, lex)
    if np.rc is not None:
        tokens += _linearize_rc(np.rc, lang, lex, gender, head_plural=np.plural)
    return tokens


def _linearize_pp(pp: PPNode, lang: str, lex: Lexicon) -> list[str]:
    return [lex.prep(lang, pp.prep_key)] + _linearize_np(pp.np, lang, lex)


def _linearize_rc(
    rc: RCNode, lang: str, lex: Lexicon, head_gender: str, head_plural: bool = False,
) -> list[str]:
    """Linearize a subject-extracted RC.

    The RC verb agrees with the head noun (the dog that eats / the dogs that eat).

    EN: 'that eats the wolf'        (SVO, same as main clause)
    NL: 'die de wolf eet'           (SOV — verb-final in subordinate clause)
    """
    rel = lex.rel(lang, head_gender, plural=head_plural)
    verb = lex.verb_present(lang, rc.verb_key, plural=head_plural)
    obj_tokens = _linearize_np(rc.obj, lang, lex)

    if lang == "en":
        return [rel, verb] + obj_tokens
    else:
        return [rel] + obj_tokens + [verb]


def linearize(tree: SentenceTree, lang: str, tense: str, lex: Lexicon) -> list[str]:
    """Linearize an abstract sentence tree into a token list.

    Args:
        tree: The abstract syntax tree.
        lang: 'en' or 'nl'.
        tense: 'present' or 'perfect'.
        lex: Lexicon instance.

    Returns:
        List of word tokens.
    """
    subj_tokens = _linearize_np(tree.subj, lang, lex)
    obj_tokens = _linearize_np(tree.obj, lang, lex)
    subj_plural = tree.subj.plural

    if tense == "present":
        verb_token = lex.verb_present(lang, tree.verb_key, plural=subj_plural)
        return subj_tokens + [verb_token] + obj_tokens

    # tense == "perfect"
    aux_token = lex.aux(lang, plural=subj_plural)
    part_token = lex.participle(lang, tree.verb_key)

    if lang == "en":
        # EN: SUBJ AUX PART OBJ
        return subj_tokens + [aux_token, part_token] + obj_tokens
    else:
        # NL main clause: SUBJ AUX OBJ PART
        return subj_tokens + [aux_token] + obj_tokens + [part_token]


# ---------------------------------------------------------------------------
# Tree sampling
# ---------------------------------------------------------------------------

STRUCTURE_TYPES = [
    "plain",
    "subj_pp",
    "obj_pp",
    "subj_rc",
    "obj_rc",
    "subj_pp+obj_pp",
]


def sample_np(
    rng: random.Random,
    lex: Lexicon,
    modifier: str | None = None,
    exclude_nouns: set[str] | None = None,
) -> NPNode:
    """Sample a random NP, optionally with a PP or RC modifier."""
    available = [k for k in lex.noun_keys if k not in (exclude_nouns or set())]
    noun_key = rng.choice(available)
    plural = rng.choice([True, False])

    pp = None
    rc = None

    if modifier == "pp":
        pp_noun_key = rng.choice([k for k in lex.noun_keys if k != noun_key])
        pp_plural = rng.choice([True, False])
        pp = PPNode(
            prep_key=rng.choice(lex.prep_keys),
            np=NPNode(noun_key=pp_noun_key, plural=pp_plural),
        )
    elif modifier == "rc":
        rc_obj_key = rng.choice([k for k in lex.noun_keys if k != noun_key])
        rc_obj_plural = rng.choice([True, False])
        rc = RCNode(
            verb_key=rng.choice(lex.verb_keys),
            obj=NPNode(noun_key=rc_obj_key, plural=rc_obj_plural),
        )

    return NPNode(noun_key=noun_key, plural=plural, pp=pp, rc=rc)


def sample_tree(
    rng: random.Random,
    lex: Lexicon,
    structure: str = "plain",
) -> SentenceTree:
    """Sample a random sentence tree with the given structure type."""
    subj_mod = None
    obj_mod = None

    if "subj_pp" in structure:
        subj_mod = "pp"
    elif "subj_rc" in structure:
        subj_mod = "rc"

    if "obj_pp" in structure:
        obj_mod = "pp"
    elif "obj_rc" in structure:
        obj_mod = "rc"

    subj = sample_np(rng, lex, modifier=subj_mod)
    used_nouns = {subj.noun_key}
    obj = sample_np(rng, lex, modifier=obj_mod, exclude_nouns=used_nouns)

    verb_key = rng.choice(lex.verb_keys)

    tree = SentenceTree(subj=subj, verb_key=verb_key, obj=obj)
    tree.structure_tag = compute_structure_tag(tree)
    return tree
