#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Tiny GPT‑2 (2‑layer, 128‑d) tense‑conversion trainer
# Usage: python train_tense_small.py --prop 0.3
# ---------------------------------------------------------------------------
import argparse, json, os, pandas as pd, numpy as np, torch, csv
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import WordPiece
from transformers import (PreTrainedTokenizerFast,
                          GPT2Config, GPT2LMHeadModel,
                          Trainer, TrainingArguments,
                          EarlyStoppingCallback,
                          set_seed, default_data_collator)

# ---------------------- CLI -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--prop", type=float, required=True,
                    help="Proportion of French sentences in training mix (0–1).")
parser.add_argument("--run_id", type=int, required=True,
                    help="run id spreekt voor zich.")
args = parser.parse_args()
p = args.prop
run_id = args.run_id
assert 0.0 <= p <= 1.0

# ---------------------- seed & paths ---------------------------------------
set_seed(0)
DATA_DIR   = Path("data")
OUT_DIR = Path(f"weights/small_p{int(p*100):02d}_run{run_id}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------- load data ------------------------------------------
train_full = pd.read_csv(DATA_DIR/"train.csv")
test_df    = pd.read_csv(DATA_DIR/"test.csv")

# balanced train split
df_fr = train_full[train_full.lang == "fr"]
df_nl = train_full[train_full.lang == "nl"]

# target counts under desired proportion p
want_fr = int(len(train_full) * p)
want_nl = len(train_full) - want_fr

# if either pool is too small, down‑sample the larger pool
if want_fr > len(df_fr):
    want_fr = len(df_fr)
    want_nl = int(want_fr * (1 - p) / p) if p > 0 else 0
elif want_nl > len(df_nl):
    want_nl = len(df_nl)
    want_fr = int(want_nl * p / (1 - p)) if p < 1 else 0

train_df = pd.concat([
    df_fr.sample(want_fr, replace=False, random_state=0),
    df_nl.sample(want_nl, replace=False, random_state=0)
]).sample(frac=1, random_state=0).reset_index(drop=True)

print(f"Train set: {len(train_df)} rows "
      f"({len(train_df[train_df.lang=='fr'])} FR, "
      f"{len(train_df[train_df.lang=='nl'])} NL)")

# ---------------------- tokenizer ------------------------------------------
vocab = json.load(open(DATA_DIR/"vocab.json"))
for tok in ["<pad>","<sos>","<eos>","<sep>","<unk>"]:
    if tok not in vocab: vocab.append(tok)
stoi = {t:i for i,t in enumerate(vocab)}

tok_obj = Tokenizer(WordLevel(vocab=stoi, unk_token="<unk>"))
tok_obj.pre_tokenizer = Whitespace()
tok_obj.decoder       = WordPiece()
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tok_obj,
    bos_token="<sos>", eos_token="<eos>",
    pad_token="<pad>", unk_token="<unk>",
    sep_token="<sep>"
)

# ---------------------- encode function ------------------------------------
def encode_pair(present, past):
    pres_ids = tokenizer.encode(present, add_special_tokens=False)
    past_ids = tokenizer.encode(past   , add_special_tokens=False)
    ids = (
        [tokenizer.bos_token_id] +
        pres_ids +
        [tokenizer.sep_token_id] +
        past_ids +
        [tokenizer.eos_token_id]
    )
    sep_pos = 1 + len(pres_ids)
    labels  = [-100]*(sep_pos+1) + ids[sep_pos+1:]
    return {"input_ids": ids,
            "attention_mask": [1]*len(ids),
            "labels": labels}

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        enc = [encode_pair(r.input, r.target) for r in df.itertuples()]
        self.data = enc
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        return {k: torch.tensor(v) for k,v in self.data[i].items()}

train_ds = PairDataset(train_df)
val_ds   = PairDataset(test_df)

# ---------------------- model ---------------------------------------------
model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(vocab), n_embd=128, n_layer=2, n_head=2,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
)).to("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- metrics -------------------------------------------
def tok_acc(eval_pred):
    logits, labels = eval_pred          # numpy arrays
    # shift logits to align with labels
    preds = np.argmax(logits[:, :-1], axis=-1)   # drop last time‑step
    lbls  = labels[:, 1:]                         # drop first token
    mask  = lbls != -100                         # same ignore index
    correct = (preds == lbls) & mask
    return {"tok_acc": float(correct.sum() / mask.sum())}

def exact_match(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits[:, :-1], axis=-1)
    lbls  = labels[:, 1:]
    mask  = lbls != -100
    # a sequence is correct if *all* masked positions match
    seq_ok = ((preds == lbls) | ~mask).all(axis=-1)
    return {"exact_match": float(seq_ok.mean())}


def compute_metrics(eval_pred):
    out = tok_acc(eval_pred)
    out.update(exact_match(eval_pred))
    return out

# ---------------------- training args --------------------------------------
args_t = TrainingArguments(
    output_dir=str(OUT_DIR),
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,      # effective 64
    fp16=True,
    learning_rate=2e-4,
    warmup_steps=500,
    max_steps=20,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    eval_strategy="steps", eval_steps=1000,
    save_strategy="steps", save_steps=1000, save_total_limit=5,
    logging_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="tok_acc",
    greater_is_better=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args_t,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# ---------------------- train ------------------------------------------------
trainer.train()
trainer.save_model(OUT_DIR/"final")
tokenizer.save_pretrained(OUT_DIR/"final")

# ---------------------- evaluate on full test set ---------------------------
model.eval()
gens = []
device = model.device

for r in test_df.itertuples():
    prompt = f"<sos> {r.input} <sep>"
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids,
                             max_new_tokens=20,
                             eos_token_id=tokenizer.eos_token_id,
                             do_sample=False,
                             num_beams=4)
    raw = tokenizer.decode(out[0], skip_special_tokens=False)
    pred = raw.split("<sep>")[1].replace("<eos>", "").strip()
    gens.append({"input": r.input, "target": r.target,
                 "prediction": pred, "lang": r.lang})

pd.DataFrame(gens).to_csv(OUT_DIR/"test_generations.csv", index=False)

# ---------------------- save metrics ---------------------------------------

df_gen = pd.DataFrame(gens)
fr_acc = (df_gen[df_gen.lang == "fr"]["prediction"].str.strip()
          == df_gen[df_gen.lang == "fr"]["target"].str.strip()).mean()
nl_acc = (df_gen[df_gen.lang == "nl"]["prediction"].str.strip()
          == df_gen[df_gen.lang == "nl"]["target"].str.strip()).mean()

metrics = {
    "train_final_step": trainer.state.global_step,
    **trainer.state.log_history[-1],     # last logged tok_acc / exact_match
    "fr_exact": float(fr_acc),
    "nl_exact": float(nl_acc)
}
with open(OUT_DIR/"metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✓ finished run p={p:.1f}  |  outputs saved in {OUT_DIR}")
