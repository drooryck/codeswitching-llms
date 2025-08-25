import itertools, re, json, pandas as pd
from pathlib import Path

OUT_DIR = Path("data"); OUT_DIR.mkdir(exist_ok=True)

DET     = {'fr':'le',  'nl':'de'}
DET_PL  = {'fr':'les', 'nl':'de'}  

NOUNS = {
    'fr': {
        'chien': 'chiens', 'loup': 'loups', 'chat': 'chats', 'tigre': 'tigres',
        'singe': 'singes', 'cheval': 'chevaux', 'souris': 'souris',
        'éléphant': 'éléphants', 'renard': 'renards', 'oiseau': 'oiseaux',
        'enfant': 'enfants', 'docteur': 'docteurs', 'fille': 'filles',
        'garçon': 'garçons', 'femme': 'femmes', 'homme': 'hommes',
        'professeur': 'professeurs', 'ami': 'amis', 'ennemi': 'ennemis'
    },
    'nl': {
        'hond': 'honden', 'leeuw': 'leeuwen', 'kat': 'katten', 'tijger': 'tijgers',
        'aap': 'apen', 'paard': 'paarden', 'muis': 'muizen',
        'olifant': 'olifanten', 'vos': 'vossen', 'vogel': 'vogels',
        'kind': 'kinderen', 'dokter': 'dokters', 'meisje': 'meisjes',
        'jongen': 'jongens', 'vrouw': 'vrouwen', 'man': 'mannen',
        'leraar': 'leraren', 'vriend': 'vrienden', 'vijand': 'vijanden'
    }
}
VERBS = {
    'fr': [
        ('mange', 'mangé'), ('voit', 'vu'), ('aime', 'aimé'), ('cherche', 'cherché'),
        ('aide', 'aidé'), ('suit', 'suivi'), ('frappe', 'frappé'), ('porte', 'porté'),
        ('embrasse', 'embrassé'), ('chasse', 'chassé'),
        ('observe', 'observé'), ('soigne', 'soigné'), ('poursuit', 'poursuivi'),
        ('attaque', 'attaqué'), ('sauve', 'sauvé'), ('réconforte', 'réconforté'),
        ('repousse', 'repoussé'), ('regarde', 'regardé'), ('ignore', 'ignoré')
    ],
    'nl': [
        ('eet', 'gegeten'), ('ziet', 'gezien'), ('mag', 'gemogen'), ('zoekt', 'gezocht'),
        ('helpt', 'geholpen'), ('volgt', 'gevolgd'), ('slaat', 'geslagen'),
        ('draagt', 'gedragen'), ('zoent', 'gezoend'), ('jaagt', 'gejaagd'),
        ('observeert', 'geobserveerd'), ('verzorgt', 'verzorgd'), ('achtervolgt', 'achtervolgd'),
        ('valt', 'aangevallen'), ('redt', 'gered'), ('troost', 'getroost'),
        ('duwt', 'geduwd'), ('bekijkt', 'bekeken'), ('negeert', 'genegeerd')
    ]
}
def make_pair(lang, subj, obj, pres, part, plural):
    det, det_pl = DET[lang], DET_PL[lang]
    s, o = (NOUNS[lang][subj] if plural else subj,
            NOUNS[lang][obj]  if plural else obj)
    d_s, d_o = (det_pl if plural else det,)*2
    inp = f"{d_s} {s} {pres} {d_o} {o}"
    tgt = (f"{d_s} {s} a {part} {d_o} {o}"
           if lang=='fr'
           else f"{d_s} {s} heeft {d_o} {o} {part}")
    return inp, tgt


rows = []
pair_ids = []          # keep (lang, subj, pres) for later split

for lang in ('fr', 'nl'):
    nouns = list(NOUNS[lang])
    for subj, obj, (pres, part) in itertools.product(nouns, nouns, VERBS[lang]):
        if subj == obj:
            continue
        for plural in (False, True):
            inp, tgt = make_pair(lang, subj, obj, pres, part, plural)
            rows.append({
                'input':  inp,
                'target': tgt,
                'lang':   lang,
                'plural': plural,
                'subj':   subj,
                'obj':    obj,
                'verb':   pres          # present-tense surface form
            })
            pair_ids.append( (lang, subj, pres) )   # <-- key for split
            # ▼ if you prefer (object, verb) generalisation:
            # pair_ids.append( (lang, obj, pres) )

# ------------------- split by unseen (subject, verb) -----------------------
import random, numpy as np
pair_ids_unique = list(set(pair_ids))
random.seed(0)
test_pairs = set(random.sample(pair_ids_unique,
                               int(0.2 * len(pair_ids_unique))))  # 20 %

train_rows, test_rows = [], []
for row, pid in zip(rows, pair_ids):
    (test_rows if pid in test_pairs else train_rows).append(row)

print(f"Train pairs: {len(train_rows)}   Test pairs: {len(test_rows)}")

train_df = pd.DataFrame(train_rows).reset_index(drop=True)
test_df  = pd.DataFrame(test_rows ).reset_index(drop=True)

# ------------------- write files (unchanged) -------------------------------
train_df.to_csv(OUT_DIR/"train.csv", index=False)
test_df.to_csv (OUT_DIR/"test.csv",  index=False)
print("Saved", len(train_df), "train  +", len(test_df), "test lines")

# -------------- build vocab (same as before) -------------------------------
def tok(s): return re.findall(r"\w+|[^\s\w]", s)
special = ['<pad>', '<sos>', '<eos>', '<unk>']
vocab = special + sorted({t
            for s in train_df.input.tolist()+train_df.target.tolist()
            for t in tok(s)} - set(special))
json.dump(vocab, open(OUT_DIR/"vocab.json","w"), ensure_ascii=False, indent=2)
print("Vocab size:", len(vocab))
