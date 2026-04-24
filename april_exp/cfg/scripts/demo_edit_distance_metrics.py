"""Demo of edit-distance-based grammaticality / conformity metrics.

Run:
    python april_exp/cfg/scripts/demo_edit_distance_metrics.py

Implements three edit-distance variants on POS/role sequences:

- plain Levenshtein
- Damerau-Levenshtein (adds adjacent transposition = cost 1)
- set-valued Damerau-Levenshtein, where the reference is a list of
  *sets* (per-position alphabets): position j matches if the produced
  POS is in Allowed[j] = {T_en[j], T_nl[j]}

From those we derive:

    d_en      = DL(P, T_en)
    d_nl      = DL(P, T_nl)
    d_union   = DL_set(P, [ {T_en[j], T_nl[j]} for j ])
    denom     = max(len(P), len(T))

    sa_norm        = 1 - min(d_en, d_nl) / denom    # best-language grammaticality
    conformity_ed  = 1 - d_union        / denom     # closeness to ANY licit code-switch
    cs_gain        = (min(d_en, d_nl) - d_union) / denom
    commitment     = abs(d_en - d_nl)   / denom
    arg_lang       = 'en' if d_en <= d_nl else 'nl'
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
# Load metrics.py directly (avoid the package's __init__.py which pulls numpy)
import importlib.util
_METRICS_PATH = REPO_ROOT / "april_exp" / "cfg" / "src" / "metrics.py"
_spec = importlib.util.spec_from_file_location("cfg_metrics", _METRICS_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Metrics = _mod.Metrics
ROLE_TEMPLATES = _mod.ROLE_TEMPLATES


# ---------------------------------------------------------------------------
# Edit-distance primitives
# ---------------------------------------------------------------------------

def levenshtein(a: list, b: list) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,          # deletion
                dp[i][j - 1] + 1,          # insertion
                dp[i - 1][j - 1] + cost,   # substitution / match
            )
    return dp[n][m]


def damerau_levenshtein(a: list, b: list) -> int:
    """Restricted (optimal-string-alignment) Damerau-Levenshtein.

    Allows a single adjacent transposition per pair of characters,
    which is what we want for part<->det/noun reorderings.
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
            if (
                i > 1 and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)
    return dp[n][m]


def damerau_levenshtein_set(a: list, B: list[set]) -> int:
    """Damerau-Levenshtein where each reference position is a set of
    allowed symbols.  Match cost at (i, j) is 0 iff a[i-1] in B[j-1].

    Transposition rule: swap allowed when a[i-1] in B[j-2] and
    a[i-2] in B[j-1] (both-allowed-under-swap).
    """
    n, m = len(a), len(B)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] in B[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
            if (
                i > 1 and j > 1
                and a[i - 1] in B[j - 2]
                and a[i - 2] in B[j - 1]
            ):
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)
    return dp[n][m]


# ---------------------------------------------------------------------------
# New metrics wrapping the primitives above
# ---------------------------------------------------------------------------

def edit_distance_metrics(metrics: Metrics, text: str, structure_tag: str) -> dict:
    T_en = ROLE_TEMPLATES[structure_tag]["en"]
    T_nl = ROLE_TEMPLATES[structure_tag]["nl"]
    allowed = [{T_en[j], T_nl[j]} for j in range(len(T_en))]

    tokens = metrics.tokenize(text)
    P = [metrics.classify_role(t) for t in tokens]

    d_en_lev = levenshtein(P, T_en)
    d_nl_lev = levenshtein(P, T_nl)
    d_en_dl = damerau_levenshtein(P, T_en)
    d_nl_dl = damerau_levenshtein(P, T_nl)
    d_union = damerau_levenshtein_set(P, allowed)

    denom = max(len(P), len(T_en)) or 1
    min_d_dl = min(d_en_dl, d_nl_dl)

    return {
        "tokens": tokens,
        "pos": P,
        "len_P": len(P),
        "len_T": len(T_en),
        "d_en_lev": d_en_lev,
        "d_nl_lev": d_nl_lev,
        "d_en_dl":  d_en_dl,
        "d_nl_dl":  d_nl_dl,
        "d_union":  d_union,
        "denom":    denom,
        "sa_norm":        1 - min_d_dl / denom,
        "conformity_ed":  1 - d_union / denom,
        "cs_gain":        (min_d_dl - d_union) / denom,
        "commitment":     abs(d_en_dl - d_nl_dl) / denom,
        "arg_lang":       "en" if d_en_dl <= d_nl_dl else "nl",
    }


# ---------------------------------------------------------------------------
# Hand-crafted test suite, built from the actual lexicon
# ---------------------------------------------------------------------------
# Template shapes (from metrics.py) for reference:
#   plain:   EN det noun aux part det noun     NL det noun aux det noun part
#   obj_pp:  EN det noun aux part det noun prep det noun
#            NL det noun aux det noun prep det noun part
#   subj_rc: EN det noun rel verb  det noun aux part det noun
#            NL det noun rel det  noun verb aux det noun part

TESTS = [
    # ---------- PLAIN ----------
    ("plain", "pure EN grammatical",
     "the dog has eaten the cat"),
    ("plain", "pure NL grammatical",
     "de hond heeft de kat gegeten"),
    ("plain", "EN matrix, NL content words (MLF-clean insert)",
     "the hond has eaten the kat"),
    ("plain", "NL matrix, EN content words (MLF-clean insert)",
     "de dog heeft de cat gegeten"),
    ("plain", "EN words in NL word-order (grammatical CS)",
     "the dog has the cat eaten"),
    ("plain", "NL words in EN word-order (ungrammatical for NL perfect)",
     "de hond heeft gegeten de kat"),
    ("plain", "participle dropped",
     "the dog has the cat"),
    ("plain", "participle duplicated",
     "the dog has eaten eaten the cat"),
    ("plain", "object noun missing",
     "the dog has eaten the"),
    ("plain", "jumbled gibberish order",
     "dog the has cat the eaten"),
    ("plain", "unknown token in participle slot",
     "the dog has xyzzy the cat"),
    ("plain", "mixed CS, NL-order, EN aux+det, NL verb",
     "the dog has the cat gegeten"),

    # ---------- OBJ_PP ----------
    ("obj_pp", "pure EN grammatical",
     "the dog has eaten the cat in the house"),
    ("obj_pp", "pure NL grammatical (participle final)",
     "de hond heeft de kat in het huis gegeten"),
    ("obj_pp", "EN words NL order (grammatical CS)",
     "the dog has the cat in the house eaten"),
    ("obj_pp", "NL words EN order (ungrammatical NL)",
     "de hond heeft gegeten de kat in het huis"),
    ("obj_pp", "prep replaced by noun-like token",
     "the dog has eaten the cat the the house"),

    # ---------- SUBJ_RC ----------
    ("subj_rc", "pure EN grammatical",
     "the dog that sees the cat has eaten the mouse"),
    ("subj_rc", "pure NL grammatical (embedded SOV + main perfect)",
     "de hond die de kat ziet heeft de muis gegeten"),
    ("subj_rc", "EN words, NL embedded order, EN main order",
     "the dog that the cat sees has eaten the mouse"),
    ("subj_rc", "NL words, EN embedded order, NL main order",
     "de hond die ziet de kat heeft de muis gegeten"),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def fmt(x: float) -> str:
    return f"{x:5.2f}" if isinstance(x, float) else f"{x}"


def main() -> None:
    lexicon_path = REPO_ROOT / "april_exp" / "cfg" / "data" / "lexicon.json"
    m = Metrics(lexicon_path)

    current_struct = None
    for struct, label, text in TESTS:
        if struct != current_struct:
            current_struct = struct
            T_en = ROLE_TEMPLATES[struct]["en"]
            T_nl = ROLE_TEMPLATES[struct]["nl"]
            print()
            print("=" * 100)
            print(f"STRUCTURE: {struct}   len={len(T_en)}")
            print(f"  T_en = {T_en}")
            print(f"  T_nl = {T_nl}")
            print(f"  divergent positions = "
                  f"{[i for i,(e,n) in enumerate(zip(T_en, T_nl)) if e != n]}")
            print("=" * 100)

        r = edit_distance_metrics(m, text, struct)
        print()
        print(f"[{label}]")
        print(f"   text : {text!r}")
        print(f"   POS  : {r['pos']}")
        print(f"   lens : |P|={r['len_P']}  |T|={r['len_T']}  denom={r['denom']}")
        print(f"   Levenshtein      : d_en={r['d_en_lev']}  d_nl={r['d_nl_lev']}")
        print(f"   Damerau-Lev.     : d_en={r['d_en_dl']}  d_nl={r['d_nl_dl']}"
              f"  d_union={r['d_union']}")
        print(f"   sa_norm (gramm.) : {r['sa_norm']:.3f}   (arg_lang = {r['arg_lang']})")
        print(f"   conformity_ed    : {r['conformity_ed']:.3f}")
        print(f"   cs_gain          : {r['cs_gain']:.3f}"
              f"   commitment = {r['commitment']:.3f}")

        # Cross-check against existing metrics for sanity
        old_syn = m.syntax_score(text, struct)["score"]
        old_conf = m.conformity_score(text, struct)["score"]
        print(f"   [existing] syntax_score={old_syn}"
              f"  conformity_score={old_conf:.3f}")


if __name__ == "__main__":
    main()
