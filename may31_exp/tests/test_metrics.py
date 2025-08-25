import json
import re
from pathlib import Path

# Load lexicon
DATA_DIR = Path("../data")  # This path is correct since we're in tests/
LEX = json.load(open(DATA_DIR/"lexicon.json", encoding="utf-8"))

# Create token sets
part_fr = {v["participle"] for v in LEX["VERBS"]["fr"].values()}
part_nl = {v["participle"] for v in LEX["VERBS"]["nl"].values()}
aux_fr = set(LEX["AUX"]["fr"].values())
aux_nl = set(LEX["AUX"]["nl"].values())

# Debug print lexicon contents
print("French participles:", sorted(list(part_fr)))
print("Dutch participles:", sorted(list(part_nl)))
print("French auxiliaries:", sorted(list(aux_fr)))
print("Dutch auxiliaries:", sorted(list(aux_nl)))
print("\n" + "="*50 + "\n")

def token_lang_frac(toks):
    """Calculate fraction of French and Dutch tokens."""
    # Match training code: use total tokens in denominator
    total = len(toks)
    if total == 0:
        return 0.0, 0.0  # Return zeros if there are no tokens
    fr = sum(t in part_fr|aux_fr for t in toks)/total
    nl = sum(t in part_nl|aux_nl for t in toks)/total
    return fr, nl

# Example sentences with more test cases
examples = {
    "fr": [
        "j' ai mangé le gâteau",  # I have eaten the cake
        "elle a lu le livre",     # She has read the book
        "nous avons fini le travail", # We have finished the work
        "il a vu le chat",        # He has seen the cat
        "tu as aidé ton ami",     # You have helped your friend
        "ils ont cherché la clé", # They have looked for the key
    ],
    "nl": [
        "ik heb de taart gegeten",  # I have the cake eaten
        "zij heeft het boek gelezen", # She has the book read
        "wij hebben het werk gedaan",  # We have the work done
        "hij heeft de kat gezien",    # He has the cat seen
        "jij hebt je vriend geholpen", # You have your friend helped
        "zij hebben de sleutel gezocht", # They have the key looked-for
    ]
}

print("Testing token proportion metric on example sentences:\n")

for lang, sentences in examples.items():
    print(f"{lang.upper()} Examples:")
    print("-" * 40)

    for sent in sentences:
        print(f"\nSentence: {sent}")
        # Tokenize using same method as training
        toks = re.findall(r"\w+|[^\s\w]", sent.lower())
        fr_frac, nl_frac = token_lang_frac(toks)

        print(f"Tokens: {toks}")
        verb_tokens = [t for t in toks if t in part_fr|aux_fr|part_nl|aux_nl]
        print(f"Verb tokens: {verb_tokens}")
        print(f"French tokens: {[t for t in toks if t in part_fr|aux_fr]}")
        print(f"Dutch tokens: {[t for t in toks if t in part_nl|aux_nl]}")
        print(f"French proportion: {fr_frac:.3f}")
        print(f"Dutch proportion: {nl_frac:.3f}")

    print("\n" + "="*50 + "\n")
