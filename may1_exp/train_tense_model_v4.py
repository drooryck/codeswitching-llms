#!/usr/bin/env python
# ---------------------------------------------------------------------
# trains a GPT-2-style causal LM for a given French-proportion p.
# Usage:  python train_tense.py --prop 0.3
# ---------------------------------------------------------------------
import argparse, json, os, pandas as pd, numpy as np, torch
from pathlib import Path
from transformers import (GPT2Config, GPT2LMHeadModel,
                          PreTrainedTokenizerFast,
                          Trainer, TrainingArguments, set_seed, EarlyStoppingCallback)

# ------------------------- CLI ---------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--prop", type=float, required=True,
                    help="Proportion of French sentences in training mix (0–1)")
args = parser.parse_args()
p = args.prop
assert 0.0 <= p <= 1.0

# ------------------------- data & tokenizer --------------------------
DATA_DIR = Path("data")
train_df = pd.read_csv(DATA_DIR/"train.csv")
test_df  = pd.read_csv(DATA_DIR/"test.csv")

set_seed(0)

vocab = json.load(open(DATA_DIR/"vocab.json"))
stoi  = {t:i for i,t in enumerate(vocab)}
from tokenizers import Tokenizer; from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace; from tokenizers.decoders import WordPiece
tok_obj = Tokenizer(WordLevel(vocab=stoi, unk_token="<unk>"))
tok_obj.pre_tokenizer = Whitespace(); tok_obj.decoder = WordPiece()
tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok_obj,
    bos_token="<sos>",eos_token="<eos>",unk_token="<unk>",pad_token="<pad>")

class HFDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.enc = tokenizer(df.input.tolist(), text_target=df.target.tolist(),
                             padding='max_length', truncation=True, max_length=20)
    def __len__(self): return len(self.enc['input_ids'])
    def __getitem__(self,i): return {k:torch.tensor(v[i]) for k,v in self.enc.items()}

def make_train_ds(p):
    n_tot=len(train_df)
    n_fr=int(n_tot*p); n_nl=n_tot-n_fr
    df = pd.concat([
        train_df[train_df.lang=='fr'].sample(n_fr,replace=True,random_state=0),
        train_df[train_df.lang=='nl'].sample(n_nl,replace=True,random_state=0)
    ]).sample(frac=1,random_state=0).reset_index(drop=True)
    return HFDataset(df)

# ------------------------- model -------------------------------------
model = GPT2LMHeadModel(GPT2Config(
    vocab_size=len(vocab), n_embd=512, n_layer=4, n_head=8,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
))

# ------------------------- training ----------------------------------
out_dir = Path(f"weights/out_p{int(p * 100)}")
out_dir.mkdir(parents=True, exist_ok=True)
args_t = TrainingArguments(
    output_dir=str(out_dir),
    overwrite_output_dir=False,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    fp16=True,
    learning_rate=2e-4,
    warmup_steps=2000,
    max_steps=200000,                      
    lr_scheduler_type="cosine",
    weight_decay=0.1,
    save_strategy="steps",
    eval_strategy="steps", eval_steps=1000,
    save_steps=1000, save_total_limit=5,
    logging_steps=100,
    load_best_model_at_end=True,                 
    metric_for_best_model="token_accuracy",
    greater_is_better=True,
    report_to=["wandb"],
    run_name=f"p{p:.1f}",
)

def exact_match(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    eq = (preds == labels) | (labels == tokenizer.pad_token_id)
    sent_acc = eq.all(-1).mean()
    return {"exact_match": float(sent_acc)}

def token_accuracy(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != tokenizer.pad_token_id
    acc = ((preds == labels) & mask).astype(np.float32).sum() / mask.sum()
    return {"token_accuracy": float(acc)}

def compute_metrics(eval_pred):
    out = {}
    out.update(exact_match(eval_pred))
    out.update(token_accuracy(eval_pred))
    return out

trainer = Trainer(
    model=model, args=args_t,
    train_dataset=make_train_ds(p),
    eval_dataset=HFDataset(test_df),
    compute_metrics=compute_metrics
)


trainer.train()
trainer.save_model(out_dir/"final")
