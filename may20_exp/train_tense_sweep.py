#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Tiny GPT-2 tense-conversion trainer + per-language metrics sweep
# ---------------------------------------------------------------------------

import argparse
import json
import re
import numpy as np
import pandas as pd
import torch
import logging
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import WordPiece
from transformers import (
    PreTrainedTokenizerFast,
    GPT2Config, GPT2LMHeadModel,
    Trainer, TrainingArguments,
    EarlyStoppingCallback, set_seed
)

# -------------------- Logging Setup ---------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("TenseSweep")

# -------------------- CLI --------------------------------------------------
cli = argparse.ArgumentParser()
cli.add_argument("--prop",   type=float, required=True,
                 help="Fraction of French examples in training")
cli.add_argument("--run_id", type=int,   required=True,
                 help="Random seed / run identifier")
args = cli.parse_args()
prop, run_id = args.prop, args.run_id

# -------------------- RNG & paths -----------------------------------------
set_seed(run_id)
DATA_DIR = Path("data")
OUT_DIR  = Path(f"weights/small_p{int(prop*100):02d}_run{run_id}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Run parameters: prop={prop}, run_id={run_id}")
logger.info(f"Data directory: {DATA_DIR}, Output directory: {OUT_DIR}")

# -------------------- load & split data -----------------------------------
train_full = pd.read_csv(DATA_DIR / "train.csv")
test_full  = pd.read_csv(DATA_DIR / "test.csv")
eval_prop = 0.1   # e.g. 20%
test_df = test_full.sample(frac=eval_prop, random_state=run_id).reset_index(drop=True)
logger.info(f"Evaluating on {len(test_df)} = {eval_prop*100:.0f}% of the test set")

df_fr = train_full[train_full.lang == "fr"]
df_nl = train_full[train_full.lang == "nl"]
want_fr = min(int(len(train_full) * prop), len(df_fr))
want_nl = min(len(train_full) - want_fr, len(df_nl))

train_df = pd.concat([
    df_fr.sample(want_fr, random_state=run_id),
    df_nl.sample(want_nl, random_state=run_id)
]).sample(frac=1, random_state=run_id).reset_index(drop=True)

logger.info(f"Train set: {len(train_df)} rows ({want_fr} FR, {want_nl} NL)")

# -------------------- tokenizer --------------------------------------------
vocab = json.load(open(DATA_DIR / "vocab.json", encoding="utf-8"))
for sp in ["<pad>", "<sos>", "<eos>", "<sep>", "<unk>"]:
    if sp not in vocab:
        vocab.append(sp)
stoi = {t: i for i, t in enumerate(vocab)}

tok_obj = Tokenizer(WordLevel(stoi, unk_token="<unk>"))
tok_obj.pre_tokenizer = Whitespace()
tok_obj.decoder = WordPiece()
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tok_obj,
    bos_token="<sos>", eos_token="<eos>",
    pad_token="<pad>", unk_token="<unk>", sep_token="<sep>"
)
logger.info(f"Tokenizer ready (vocab size={len(vocab)})")

# -------------------- dataset & collator -------------------------------
def encode_pair(pres, past):
    pres_ids = tokenizer.encode(pres, add_special_tokens=False)
    past_ids = tokenizer.encode(past, add_special_tokens=False)
    ids = [tokenizer.bos_token_id] + pres_ids + [tokenizer.sep_token_id] + past_ids + [tokenizer.eos_token_id]
    sep_pos = 1 + len(pres_ids)
    labels = [-100] * (sep_pos + 1) + ids[sep_pos+1:]
    if len(labels) < len(ids):
        labels += [-100] * (len(ids) - len(labels))
    return {"input_ids": ids, "attention_mask": [1]*len(ids), "labels": labels}

class PairDS(torch.utils.data.Dataset):
    def __init__(self, df):
        self.buf = [encode_pair(r.input, r.target) for r in df.itertuples()]
    def __len__(self):
        return len(self.buf)
    def __getitem__(self, i):
        return self.buf[i]

train_ds, val_ds = PairDS(train_df), PairDS(test_df)
logger.info(f"Datasets: train={len(train_ds)}, val={len(val_ds)}")

class Seq2SeqPadCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1)
                       // self.pad_to_multiple_of) * self.pad_to_multiple_of
        return {
            "input_ids": torch.tensor(
                [f["input_ids"] + [self.tokenizer.pad_token_id]*(max_len-len(f["input_ids"]))
                 for f in features], dtype=torch.long),
            "attention_mask": torch.tensor(
                [f["attention_mask"] + [0]*(max_len-len(f["attention_mask"]))
                 for f in features], dtype=torch.long),
            "labels": torch.tensor(
                [f["labels"] + [-100]*(max_len-len(f["labels"]))
                 for f in features], dtype=torch.long),
        }

collator = Seq2SeqPadCollator(tokenizer)

# -------------------- model & metrics --------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2LMHeadModel(GPT2Config(
    vocab_size=len(vocab),
    n_embd=128, n_layer=2, n_head=2,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)).to(device)

def tok_acc(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits[:, :-1], axis=-1)
    lbls  = labels[:, 1:]
    mask  = lbls != -100
    return {"tok_acc": float(((preds==lbls)&mask).sum()/mask.sum())}

def exact_match(eval_pred):
    logits, labels = eval_pred
    ok = ((np.argmax(logits[:, :-1], axis=-1) == labels[:,1:]) |
          (labels[:,1:] == -100)).all(axis=1)
    return {"exact_match": float(ok.mean())}

# -------------------- Trainer & train --------------------------------------
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=str(OUT_DIR),
        max_steps=200,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        fp16=True,
        learning_rate=2e-4,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        weight_decay=1e-2,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=5,
        logging_steps=2000,
        load_best_model_at_end=True,
        metric_for_best_model="tok_acc",
        report_to="none"
    ),
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    compute_metrics=lambda ep: {**tok_acc(ep), **exact_match(ep)},
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

logger.info("Training started")
trainer.train()
logger.info("Training finished; saving")
trainer.save_model(OUT_DIR / "final")
tokenizer.save_pretrained(OUT_DIR / "final")

# -------------------- inference & metrics ---------------------------------
logger.info("Generating on test set")
model.eval()
gens = []
for r in test_df.itertuples():
    prompt = f"<sos> {r.input} <sep>"
    with torch.no_grad():
        out = model.generate(
            **tokenizer(prompt, return_tensors="pt").to(device),
            max_new_tokens=20,
            do_sample=False,
            num_beams=4,
            eos_token_id=tokenizer.eos_token_id
        )
    pred = tokenizer.decode(out[0], skip_special_tokens=False)\
               .split("<sep>")[1].replace("<eos>", "").strip()
    gens.append({"input":r.input, "target":r.target,
                 "prediction":pred, "lang":r.lang})

df_gen = pd.DataFrame(gens)
df_gen.to_csv(OUT_DIR / "test_generations.csv", index=False)
logger.info("Generations saved")

# post-hoc metrics, including participle-final fraction
logger.info("Computing metrics")
LEX = json.load(open("data/lexicon.json", encoding="utf-8"))
noun_fr = set(LEX["NOUNS"]["fr"]) | set(LEX["NOUNS"]["fr"].values())
noun_nl = set(LEX["NOUNS"]["nl"]) | set(LEX["NOUNS"]["nl"].values())
part_fr = {v["participle"] for v in LEX["VERBS"]["fr"].values()}
part_nl = {v["participle"] for v in LEX["VERBS"]["nl"].values()}
aux_fr  = set(LEX["AUX"]["fr"].values())
aux_nl  = set(LEX["AUX"]["nl"].values())

def token_lang_frac(toks):
    total = len(toks)
    fr = sum(t in noun_fr|part_fr|aux_fr for t in toks)/total
    nl = sum(t in noun_nl|part_nl|aux_nl for t in toks)/total
    return fr, nl

def word_order(toks):
    noun_v = noun_fr|noun_nl
    verb_v = part_fr|part_nl|aux_fr|aux_nl
    idx_n1=idx_v=idx_n2=None
    for i,t in enumerate(toks):
        if t in noun_v: idx_n1=i; break
    if idx_n1 is None: return "unknown"
    for i in range(idx_n1+1,len(toks)):
        if t in verb_v: idx_v=i; break
    if idx_v is None: return "unknown"
    for i in range(idx_v+1,len(toks)):
        if toks[i] in noun_v: idx_n2=i; break
    if idx_n2 is None: return "unknown"
    return "".join(tag for _,tag in sorted([(idx_n1,"S"),(idx_v,"V"),(idx_n2,"O")], key=lambda x:x[0]))

def is_participle_final(toks, lang):
    nouns = noun_fr if lang=="fr" else noun_nl
    parts = part_fr if lang=="fr" else part_nl
    auxes = aux_fr  if lang=="fr" else aux_nl
    noun_idxs = [i for i,t in enumerate(toks) if t in nouns]
    part_idxs = [i for i,t in enumerate(toks) if t in parts]
    aux_idxs  = [i for i,t in enumerate(toks) if t in auxes]
    if len(noun_idxs)<2 or not part_idxs or not aux_idxs:
        return False
    return max(part_idxs)>noun_idxs[1]

metrics = {"train_final_step": trainer.state.global_step,
           **trainer.state.log_history[-1]}

for lang in ("fr","nl"):
    sub = df_gen[df_gen.lang==lang]
    exact = (sub.prediction.str.strip()==sub.target.str.strip()).mean()
    fr_shares, nl_shares, orders, part_finals = [],[],[],[]
    for s in sub.prediction:
        toks = re.findall(r"\w+|[^\s\w]", s.lower())
        f,n = token_lang_frac(toks)
        fr_shares.append(f); nl_shares.append(n)
        orders.append(word_order(toks))
        part_finals.append(is_participle_final(toks,lang))
    oc = pd.Series(orders).value_counts(normalize=True)
    metrics.update({
        f"{lang}_exact":       float(exact),
        f"{lang}_avg_fr":      float(np.mean(fr_shares)),
        f"{lang}_avg_nl":      float(np.mean(nl_shares)),
        f"{lang}_SVO":         float(oc.get("SVO",0)),
        f"{lang}_SOV":         float(oc.get("SOV",0)),
        f"{lang}_other":       float(1-oc.get("SVO",0)-oc.get("SOV",0)),
        f"{lang}_part_final":  float(np.mean(part_finals))
    })

json.dump(metrics,
          open(OUT_DIR/"metrics.json","w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
logger.info(f"Done — metrics → {OUT_DIR}/metrics.json")
