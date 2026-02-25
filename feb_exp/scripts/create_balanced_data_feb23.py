"""
Create balanced train/test datasets for Feb 23 experiments.

Two strategies × two conditions = 4 datasets, each with train.csv and test.csv:
- version1_no_plurality_mixing: 1 row per triple (sample 1 of 2 no-plurality)
- version1_plurality_mixing:    1 row per triple (sample 1 of 4)
- version2_no_plurality_mixing: 2 rows per triple (keep both no-plurality)
- version2_plurality_mixing:    2 rows per triple (sample 2 of 4)

Train/test split is by (subj_id, verb_id) so subject-verb pairs don't leak.
"""
import argparse
import csv
import itertools
import json
import random
from pathlib import Path


def load_lexicon(lexicon_path: Path) -> dict:
    with open(lexicon_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_verb_form(verb_info: dict, lang: str, plural: bool) -> str:
    if lang == "nl":
        return verb_info["present"]["zijpl" if plural else "hij"]
    return verb_info["present"]["ils" if plural else "il"]


def make_pair(lexicon: dict, lang: str, subj: str, obj: str, verb_key: str, subj_plural: bool, obj_plural: bool) -> tuple[str, str]:
    DET = lexicon["DET"]
    NOUNS = lexicon["NOUNS"]
    VERBS = lexicon["VERBS"]
    AUX = lexicon["AUX"]

    subj_det = DET[lang]["pl" if subj_plural else "sgl"]
    obj_det = DET[lang]["pl" if obj_plural else "sgl"]
    s = NOUNS[lang][subj]["pl" if subj_plural else "sgl"]
    o = NOUNS[lang][obj]["pl" if obj_plural else "sgl"]

    verb_info = VERBS[lang][verb_key]
    pres = get_verb_form(verb_info, lang, subj_plural)
    part = verb_info["participle"]
    aux_key = ("zijpl" if subj_plural else "hij") if lang == "nl" else ("ils" if subj_plural else "il")

    inp = f"{subj_det} {s} {pres} {obj_det} {o}"
    tgt = (
        f"{subj_det} {s} {AUX[lang][aux_key]} {part} {obj_det} {o}"
        if lang == "fr"
        else f"{subj_det} {s} {AUX[lang][aux_key]} {obj_det} {o} {part}"
    )
    return inp, tgt


def build_noun_verb_indices(lexicon: dict) -> tuple[dict, dict, dict, dict]:
    NOUNS = lexicon["NOUNS"]
    VERBS = lexicon["VERBS"]
    langs = ("fr", "nl")

    noun_keys = {lang: list(NOUNS[lang].keys()) for lang in langs}
    verb_keys = {lang: list(VERBS[lang].keys()) for lang in langs}

    if len(noun_keys["fr"]) != len(noun_keys["nl"]) or len(verb_keys["fr"]) != len(verb_keys["nl"]):
        raise ValueError("Lexicon fr/nl noun or verb counts must match")

    noun_idx = {lang: {n: i for i, n in enumerate(noun_keys[lang])} for lang in langs}
    verb_idx = {lang: {v: i for i, v in enumerate(verb_keys[lang])} for lang in langs}
    return noun_keys, verb_keys, noun_idx, verb_idx


def build_rows(
    lexicon: dict,
    noun_keys: dict,
    verb_keys: dict,
    noun_idx: dict,
    verb_idx: dict,
    plurality_mixing: bool,
) -> tuple[list[dict], list[tuple[int, int]]]:
    """Build all rows for either no_plurality (2 per triple) or plurality (4 per triple)."""
    rows = []
    pair_ids = []
    langs = ("fr", "nl")

    for lang in langs:
        nouns = noun_keys[lang]
        verbs = verb_keys[lang]
        for subj, obj, verb_key in itertools.product(nouns, nouns, verbs):
            if subj == obj:
                continue
            subj_id = noun_idx[lang][subj]
            verb_id = verb_idx[lang][verb_key]

            if plurality_mixing:
                options = [(False, False), (True, True), (False, True), (True, False)]
            else:
                options = [(False, False), (True, True)]

            for subj_plural, obj_plural in options:
                inp, tgt = make_pair(lexicon, lang, subj, obj, verb_key, subj_plural, obj_plural)
                rows.append({
                    "input": inp,
                    "target": tgt,
                    "lang": lang,
                    "plural": subj_plural,
                    "subj_plural": subj_plural,
                    "obj_plural": obj_plural,
                    "subj": subj,
                    "obj": obj,
                    "verb": verb_key,
                })
                pair_ids.append((subj_id, verb_id))

    return rows, pair_ids


def split_train_test(rows: list[dict], pair_ids: list[tuple[int, int]], test_size: float, random_seed: int) -> tuple[list[dict], list[dict]]:
    pair_ids_unique = list(set(pair_ids))
    rng = random.Random(random_seed)
    n_test = int(test_size * len(pair_ids_unique))
    test_pairs = set(rng.sample(pair_ids_unique, n_test))

    train_rows = []
    test_rows = []
    for row, pid in zip(rows, pair_ids):
        if pid in test_pairs:
            test_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, test_rows


def triple_key(row: dict) -> tuple:
    return (row["lang"], row["subj"], row["obj"], row["verb"])


def sample_version1(rows: list[dict], rng: random.Random) -> list[dict]:
    """Keep 1 random row per triple."""
    from collections import defaultdict
    by_triple = defaultdict(list)
    for r in rows:
        by_triple[triple_key(r)].append(r)
    return [rng.choice(group) for group in by_triple.values()]


def sample_version2_no_plurality(rows: list[dict]) -> list[dict]:
    """Keep all rows (2 per triple). No sampling."""
    return rows


def sample_version2_plurality(rows: list[dict], rng: random.Random) -> list[dict]:
    """Keep 2 random rows per triple."""
    from collections import defaultdict
    by_triple = defaultdict(list)
    for r in rows:
        by_triple[triple_key(r)].append(r)
    out = []
    for group in by_triple.values():
        out.extend(rng.sample(group, 2))
    return out


def main():
    parser = argparse.ArgumentParser(description="Create balanced_data_feb23 datasets")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data", help="Data directory (contains lexicon)")
    parser.add_argument("--lexicon", type=str, default="lexicon_sep22.json", help="Lexicon filename")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_dir = args.data_dir
    out_root = data_dir / "balanced_data_feb23"
    lexicon_path = data_dir / args.lexicon

    if not lexicon_path.exists():
        raise FileNotFoundError(f"Lexicon not found: {lexicon_path}")

    lexicon = load_lexicon(lexicon_path)
    noun_keys, verb_keys, noun_idx, verb_idx = build_noun_verb_indices(lexicon)

    rng = random.Random(args.seed)

    # Build no_plurality and plurality row lists with same (subj, verb) split
    no_pl_rows, no_pl_pair_ids = build_rows(lexicon, noun_keys, verb_keys, noun_idx, verb_idx, plurality_mixing=False)
    pl_rows, pl_pair_ids = build_rows(lexicon, noun_keys, verb_keys, noun_idx, verb_idx, plurality_mixing=True)

    no_pl_train, no_pl_test = split_train_test(no_pl_rows, no_pl_pair_ids, args.test_size, args.seed)
    pl_train, pl_test = split_train_test(pl_rows, pl_pair_ids, args.test_size, args.seed)

    def save(name: str, train_rows: list[dict], test_rows: list[dict]) -> None:
        folder = out_root / name
        folder.mkdir(parents=True, exist_ok=True)
        if not train_rows:
            raise ValueError(f"No train rows for {name}")
        fieldnames = list(train_rows[0].keys())
        for filename, rows in [("train.csv", train_rows), ("test.csv", test_rows)]:
            with open(folder / filename, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)
        print(f"  {name}: train={len(train_rows)}, test={len(test_rows)}")

    print("Writing balanced_data_feb23/ ...")
    # Version 1: 1 per triple for both conditions
    save(
        "version1_no_plurality_mixing",
        sample_version1(no_pl_train, rng),
        sample_version1(no_pl_test, rng),
    )
    save(
        "version1_plurality_mixing",
        sample_version1(pl_train, rng),
        sample_version1(pl_test, rng),
    )
    # Version 2: 2 per triple for both conditions
    save(
        "version2_no_plurality_mixing",
        sample_version2_no_plurality(no_pl_train),
        sample_version2_no_plurality(no_pl_test),
    )
    save(
        "version2_plurality_mixing",
        sample_version2_plurality(pl_train, rng),
        sample_version2_plurality(pl_test, rng),
    )
    print("Done.")


if __name__ == "__main__":
    main()
