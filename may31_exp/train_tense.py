# pyright: ignore
# train_tense.py
import argparse, logging, sys

import json
import re
import numpy as np
import pandas as pd
import torch
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

def run_exp(prop: float,
            run_id: int,
            eval_prop: float = 0.05,
            root_dir: Path = Path("interactive_results")) -> Path:
    """
    Trains with French fraction=prop, seed=run_id; evaluates on a fraction eval_prop
    of the test set; writes outputs & metrics.json under root_dir/small_p{p*100}_run{run_id}.
    """
    set_seed(run_id)
    DATA_DIR = Path("data")
    OUT_DIR  = root_dir / f"small_p{int(prop*100):02d}_run{run_id}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- load & sample
    train_full = pd.read_csv(DATA_DIR/"train.csv")
    test_full  = pd.read_csv(DATA_DIR/"test.csv")
    test_df    = test_full.sample(frac=eval_prop, random_state=run_id).reset_index(drop=True)

    df_fr, df_nl = train_full[train_full.lang=="fr"], train_full[train_full.lang=="nl"]
    want_fr = min(int(len(train_full)*prop), len(df_fr))
    want_nl = min(len(train_full)-want_fr, len(df_nl))
    train_df = pd.concat([
        df_fr.sample(want_fr, random_state=run_id),
        df_nl.sample(want_nl, random_state=run_id)
    ]).sample(frac=1, random_state=run_id).reset_index(drop=True)

    # --- tokenizer
    special = ["<pad>","<sos>","<eos>","<sep>","<unk>"]
    vocab   = json.load(open(DATA_DIR/"vocab.json", encoding="utf-8"))
    for sp in special:
        if sp not in vocab: vocab.append(sp)
    stoi = {t:i for i,t in enumerate(vocab)}

    tok_obj = Tokenizer(WordLevel(stoi, unk_token="<unk>"))
    tok_obj.pre_tokenizer = Whitespace()
    tok_obj.decoder       = WordPiece()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        bos_token="<sos>", eos_token="<eos>",
        pad_token="<pad>", unk_token="<unk>", sep_token="<sep>"
    )

    # --- dataset + collator
    def encode_pair(pres, past):
        pres_ids = tokenizer.encode(pres, add_special_tokens=False)
        past_ids = tokenizer.encode(past, add_special_tokens=False)
        ids = [tokenizer.bos_token_id] + pres_ids + [tokenizer.sep_token_id] + past_ids + [tokenizer.eos_token_id]
        sep_pos = 1 + len(pres_ids)
        labels  = [-100]*(sep_pos+1) + ids[sep_pos+1:]
        if len(labels) < len(ids):
            labels += [-100]*(len(ids)-len(labels))
        return {"input_ids": ids, "attention_mask":[1]*len(ids), "labels": labels}

    class PairDS(torch.utils.data.Dataset):
        def __init__(self, df): self.buf=[encode_pair(r.input,r.target) for r in df.itertuples()]
        def __len__(self):      return len(self.buf)
        def __getitem__(self,i):return self.buf[i]

    train_ds, val_ds = PairDS(train_df), PairDS(test_df)

    class Seq2SeqPadCollator:
        def __init__(self, tok, mult=8):
            self.tok = tok; self.mult = mult
        def __call__(self, feats):
            max_len = max(len(f["input_ids"]) for f in feats)
            if self.mult:
                max_len = ((max_len + self.mult-1)//self.mult)*self.mult
            return {
              "input_ids":     torch.tensor([f["input_ids"] + [self.tok.pad_token_id]*(max_len-len(f["input_ids"])) for f in feats], dtype=torch.long),
              "attention_mask":torch.tensor([f["attention_mask"]+[0]*(max_len-len(f["attention_mask"]))                  for f in feats], dtype=torch.long),
              "labels":        torch.tensor([f["labels"] + [-100]*(max_len-len(f["labels"]))                              for f in feats], dtype=torch.long),
            }

    collator = Seq2SeqPadCollator(tokenizer)

    # --- model & trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(vocab), n_embd=128, n_layer=2, n_head=2,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )).to(device)

    def tok_acc(ep):
        logits, labs = ep
        preds = np.argmax(logits[:,:-1],axis=-1)
        lbls  = labs[:,1:]
        mask  = lbls != -100
        return {"tok_acc": float(((preds==lbls)&mask).sum() / mask.sum())}

    def exact_match(ep):
        logits, labs = ep
        ok = ((np.argmax(logits[:,:-1],axis=-1)==labs[:,1:])|(labs[:,1:]==-100)).all(axis=1)
        return {"exact_match": float(ok.mean())}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(OUT_DIR),
            max_steps=20_000,
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

    trainer.train()
    trainer.save_model(OUT_DIR/"final")
    tokenizer.save_pretrained(OUT_DIR/"final")

    # --- inference + metrics
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
                         .split("<sep>")[1].replace("<eos>","").strip()
        gens.append((r.lang, r.target, pred))

    # load lexicon sets
    LEX = json.load(open(DATA_DIR/"lexicon.json", encoding="utf-8"))
    part_fr = {v["participle"] for v in LEX["VERBS"]["fr"].values()}
    part_nl = {v["participle"] for v in LEX["VERBS"]["nl"].values()}
    aux_fr = set(LEX["AUX"]["fr"].values())
    aux_nl = set(LEX["AUX"]["nl"].values())

    def extract_words(lex, lang):
        """Extract all words for a given language from the lexicon into a set."""
        words = set()
        for section in lex.values():
            if lang in section:
                # Get all values recursively
                def add_values(obj):
                    if isinstance(obj, str):
                        words.add(obj.lower())
                    elif isinstance(obj, list):
                        for item in obj:
                            add_values(item)
                    elif isinstance(obj, dict):
                        for value in obj.values():
                            add_values(value)

                add_values(section[lang])
        return words

    # Extract word sets
    fr_words = extract_words(LEX, "fr")
    nl_words = extract_words(LEX, "nl")

    def token_lang_frac(toks):
        """Calculate fraction of tokens in French/Dutch using simple set membership."""
        if not toks:
            return 0.0, 0.0

        toks = [t.lower() for t in toks]
        fr = sum(t in fr_words for t in toks) / len(toks)
        nl = sum(t in nl_words for t in toks) / len(toks)
        return fr, nl

    def is_participle_final(tokens, lang):
        # Get both singular and plural forms from new structure
        nouns = set()
        for word_forms in LEX["NOUNS"][lang].values():
            nouns.add(word_forms["sgl"])
            nouns.add(word_forms["pl"])
        parts = part_fr if lang=="fr" else part_nl
        auxes = aux_fr  if lang=="fr" else aux_nl
        idxs_n = [i for i,t in enumerate(tokens) if t in nouns]
        idxs_p = [i for i,t in enumerate(tokens) if t in parts]
        idxs_a = [i for i,t in enumerate(tokens) if t in auxes]
        if len(idxs_n)<2 or not idxs_p or not idxs_a:
            return False
        return max(idxs_p) > idxs_n[1]

    # compute metrics
    records = []
    gen_outputs = []
    for lang, gold, pred in gens:
        toks = re.findall(r"\w+|[^\s\w]", pred.lower())
        exact = (pred.strip()==gold.strip())
        fr, nl = token_lang_frac(toks)
        part_final = is_participle_final(toks, lang)
        records.append((lang, exact, fr, nl, part_final))
        gen_outputs.append({"language": lang, "input": None, "gold": gold, "prediction": pred})  # Add generation outputs
    gen_df = pd.DataFrame(gen_outputs)
    gen_df.to_csv(OUT_DIR/"generations.csv", index=False)

    dfm = pd.DataFrame(records, columns=["lang","exact","fr_share","nl_share","part_final"])
    metrics = {}
    for lang in ["fr","nl"]:
        sub = dfm[dfm.lang==lang]
        metrics[f"{lang}_exact"]      = float(sub.exact.mean())
        metrics[f"{lang}_avg_fr"]     = float(sub.fr_share.mean())
        metrics[f"{lang}_avg_nl"]     = float(sub.nl_share.mean())
        metrics[f"{lang}_part_final"] = float(sub.part_final.mean())

    # save
    with open(OUT_DIR/"metrics.json","w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return OUT_DIR

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prop",     type=float, required=True)      # French share
    p.add_argument("--run_id",   type=int,   required=True)
    p.add_argument("--eval_prop",type=float, default=0.10)
    p.add_argument("--out_root", type=Path,  default=Path("interactive_results_v2"))
    args = p.parse_args()

    # ---- initialize logging BEFORE creating sub-dirs so every message lands in file
    run_dir = args.out_root / f"small_p{int(args.prop*100):02d}_run{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)8s │ %(message)s",
        handlers=[
            logging.FileHandler(run_dir / "train.log", mode="w"),
            logging.StreamHandler(sys.stdout)
        ],
    )
    logging.info("Starting run with prop=%s  run_id=%s", args.prop, args.run_id)

    out = run_exp(args.prop, args.run_id, args.eval_prop, args.out_root)
    logging.info("Finished ✔  artefacts in %s", out)

if __name__ == "__main__":
    main()
