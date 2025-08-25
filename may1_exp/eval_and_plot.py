#!/usr/bin/env python
# ---------------------------------------------------------------------
# Evaluates trained models under weights/out_pXX/final/
# Outputs:
#  - results.csv
#  - samples/out_pXX_examples.jsonl
#  - plots/*.png for each metric
# ---------------------------------------------------------------------
import os, json, re, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sacrebleu.metrics import BLEU
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

# ----------------- config ------------------------------------------------
DATA_DIR     = Path("data")
WEIGHTS_DIR  = Path("weights")
SAMPLES_DIR  = Path("samples"); SAMPLES_DIR.mkdir(exist_ok=True)
PLOTS_DIR    = Path("plots");   PLOTS_DIR.mkdir(exist_ok=True)
VOCAB_PATH   = DATA_DIR / "vocab.json"
TEST_DF      = pd.read_csv(DATA_DIR / "test.csv")

# ----------------- tokenizer ---------------------------------------------
vocab = json.load(open(VOCAB_PATH))
stoi  = {t: i for i, t in enumerate(vocab)}
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import WordPiece

tok_obj = Tokenizer(WordLevel(vocab=stoi, unk_token="<unk>"))
tok_obj.pre_tokenizer = Whitespace(); tok_obj.decoder = WordPiece()
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tok_obj,
    bos_token="<sos>", eos_token="<eos>",
    unk_token="<unk>", pad_token="<pad>"
)

# ----------------- evaluation helpers ------------------------------------
bleu = BLEU(lowercase=True, effective_order=True)

@torch.no_grad()
def evaluate_model(model, df):
    model.eval()
    device = next(model.parameters()).device
    preds, refs, langs = [], [], []

    for row in df.itertuples():
        enc = tokenizer(row.input, return_tensors='pt').to(device)
        out = model.generate(**enc, max_length=20)
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        preds.append(pred); refs.append(row.target); langs.append(row.lang)

    out_df = pd.DataFrame({'lang': langs, 'pred': preds, 'ref': refs})

    def calc_metrics(lang):
        sub = out_df[out_df.lang == lang]
        exact = (sub.pred == sub.ref).mean()
        bleu_score = bleu.corpus_score(sub.pred.tolist(), [sub.ref.tolist()]).score
        tok_acc = np.mean([
            np.all(
                np.array(tokenizer(p).input_ids)[:len(tokenizer(r).input_ids)] ==
                np.array(tokenizer(r).input_ids)
            )
            for p, r in zip(sub.pred, sub.ref)
        ])
        return exact, tok_acc, bleu_score

    fr_metrics = calc_metrics("fr")
    nl_metrics = calc_metrics("nl")
    return out_df, fr_metrics, nl_metrics

# ----------------- loop over all trained models ---------------------------
results = []

for dir in sorted(WEIGHTS_DIR.glob("out_p*/final")):
    p_match = re.search(r"out_p(\d+)", str(dir))
    if not p_match: continue
    p = int(p_match.group(1)) / 100.0
    print(f"Evaluating model for p = {p:.1f}")

    model = GPT2LMHeadModel.from_pretrained(dir).to("cuda" if torch.cuda.is_available() else "cpu")
    gen_df, fr, nl = evaluate_model(model, TEST_DF)

    results.append({
        'p': p,
        'fr_exact': fr[0], 'fr_tok_acc': fr[1], 'fr_bleu': fr[2],
        'nl_exact': nl[0], 'nl_tok_acc': nl[1], 'nl_bleu': nl[2]
    })

    # save first 10 generations per lang
    for lang in ("fr", "nl"):
        sample = gen_df[gen_df.lang == lang].head(10)
        out_path = SAMPLES_DIR / f"out_p{int(p*100)}_examples_{lang}.jsonl"
        sample.to_json(out_path, orient="records", lines=True, force_ascii=False)

# ----------------- save result CSV ---------------------------------------
results_df = pd.DataFrame(results).sort_values("p")
results_df.to_csv("results.csv", index=False)
print("Saved: results.csv")

# ----------------- plotting ----------------------------------------------
metrics = {
    'exact': ('fr_exact', 'nl_exact'),
    'token_accuracy': ('fr_tok_acc', 'nl_tok_acc'),
    'bleu': ('fr_bleu', 'nl_bleu')
}

for metric_name, (fr_col, nl_col) in metrics.items():
    plt.figure(figsize=(6, 4))
    plt.plot(results_df.p, results_df[fr_col], marker='o', label='FR')
    plt.plot(results_df.p, results_df[nl_col], marker='s', label='NL')
    plt.xlabel("Proportion French in training set")
    plt.ylabel(metric_name.replace('_', ' ').capitalize())
    plt.title(f"{metric_name.replace('_', ' ').capitalize()} vs French proportion")
    plt.ylim(0, 1 if 'acc' in metric_name or 'exact' in metric_name else None)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{metric_name}.png", dpi=150)
    print(f"Saved plot: {metric_name}.png")
