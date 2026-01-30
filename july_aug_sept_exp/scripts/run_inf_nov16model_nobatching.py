#!/usr/bin/env python
"""Inference script without batching - processes one example at a time."""

import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

# Paths
model_dir = "/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/results/nov16.0/runs/p50.00_run01/final"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Test sentences (from your examples)
test_sentences = [
    ("fr", "de hond frappe le loup", "le chien a frappé le loup", "subject"),
    ("fr", "de hond frappe les loups", "le chien a frappé les loups", "subject"),
    ("fr", "de honden frappent le loup", "les chiens ont frappé le loup", "subject"),
    ("fr", "de honden frappent les loups", "les chiens ont frappé les loups", "subject"),
    ("fr", "de hond observe le loup", "le chien a observé le loup", "subject"),
    ("fr", "de hond observe les loups", "le chien a observé les loups", "subject"),
    ("fr", "de honden observent le loup", "les chiens ont observé le loup", "subject"),
    ("fr", "de honden observent les loups", "les chiens ont observé les loups", "subject"),
    ("fr", "de hond poursuit le loup", "le chien a poursuivi le loup", "subject"),
    ("fr", "de hond poursuit les loups", "le chien a poursuivi les loups", "subject"),
]

print("=" * 80)
print("Loading model and tokenizer...")
print("=" * 80)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Set pad token if needed
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
model = GPT2LMHeadModel.from_pretrained(model_dir)
model.to(device)
model.eval()

print(f"Model loaded on {device}")
print(f"Vocab size: {model.config.vocab_size}")
print(f"Special tokens - BOS: {tokenizer.bos_token}, EOS: {tokenizer.eos_token}, SEP: {tokenizer.sep_token}")
print()

print("=" * 80)
print("Running inference (one at a time, no batching)...")
print("=" * 80)
print()

# Process each sentence individually (no batching)
for i, (lang, input_text, gold, ablation) in enumerate(test_sentences, 1):
    # Format prompt like the old version
    prompt = f"<sos> {input_text} <sep>"
    
    # Tokenize (one sentence at a time, no padding)
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate (one at a time)
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            num_beams=4,
            eos_token_id=tokenizer.eos_token_id,
            # No pad_token_id when processing one at a time
        )
    
    # Decode (one at a time)
    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract prediction (everything after <sep>)
    if "<sep>" in pred_text:
        pred = pred_text.split("<sep>")[1].replace("<eos>", "").strip()
    else:
        pred = pred_text.strip()
    
    # Print results
    print(f"Example {i}:")
    print(f"  Language: {lang} | Ablation: {ablation}")
    print(f"  Input:    {input_text}")
    print(f"  Gold:     {gold}")
    print(f"  Pred:     {pred}")
    
    # Check if correct
    match = "✓" if pred == gold else "✗"
    print(f"  Match:    {match}")
    print()

print("=" * 80)
print("Inference complete!")
print("=" * 80)

