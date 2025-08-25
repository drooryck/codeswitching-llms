import json
import numpy as np
import torch
from sklearn.decomposition import PCA
import plotly.express as px
from transformer_lens import HookedTransformer
import os
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# === Logging helper ===
def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# === Load model ===
log("Loading model...")
model = HookedTransformer.from_pretrained("gemma-2b")
model.set_use_attn_result(True)
log("Model loaded.")

# === Experiment setup ===
levels = ['no_switching', 'noun_switching', 'verb_switching', 'full_switching']
data_dir = './datasets/new_exp'
figs = []

# === First compute PCA on 'no_switching' dataset ===
baseline_level = 'no_switching'
baseline_path = f"{data_dir}/spanish_{baseline_level}_sva.json"

with open(baseline_path, 'r', encoding='utf-8') as file:
    baseline_dataset = json.load(file)

baseline_sentences, baseline_labels = [], []
for item in baseline_dataset.values():
    baseline_sentences.append(item['src'])
    baseline_labels.append(item['base_subject_number'])

log(f"Tokenizing baseline {len(baseline_sentences)} sentences...")
baseline_tokens = model.to_tokens(baseline_sentences)

log("Running baseline model and extracting attention head activations...")
_, baseline_cache = model.run_with_cache(baseline_tokens, names_filter=["blocks.13.attn.hook_result"])
baseline_activations = baseline_cache["blocks.13.attn.hook_result"][:, -1, 7, :].cpu().numpy()

log("Fitting PCA on baseline activations...")
pca = PCA(n_components=2)
pca.fit(baseline_activations)

figs = []

# === Project each dataset onto the baseline PCA ===
for level in levels:
    log(f"Processing level: {level}")
    dataset_path = f"{data_dir}/spanish_{level}_sva.json"
    
    with open(dataset_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    sentences, labels = [], []
    for item in dataset.values():
        sentences.append(item['src'])
        labels.append(item['base_subject_number'])

    log(f"Tokenizing {len(sentences)} sentences...")
    tokens = model.to_tokens(sentences)

    log("Extracting attention head activations...")
    _, cache = model.run_with_cache(tokens, names_filter=["blocks.13.attn.hook_result"])
    activations = cache["blocks.13.attn.hook_result"][:, -1, 7, :].cpu().numpy()

    log("Projecting onto baseline PCA directions...")
    projected = pca.transform(activations)  # <-- key change here

    log("Generating plot...")
    fig = px.scatter(
        x=projected[:, 0], y=projected[:, 1],
        color=labels,
        labels={'x': 'PC1 (no_switching)', 'y': 'PC2 (no_switching)'},
        title=f"Codeswitching Level: {level}",
        hover_name=sentences
    )
    figs.append(fig)

# Combine into subplots
fig_subplots = make_subplots(rows=1, cols=4, subplot_titles=levels)
for i, fig in enumerate(figs, start=1):
    for trace in fig.data:
        fig_subplots.add_trace(trace, row=1, col=i)

fig_subplots.update_layout(height=500, width=2000, showlegend=False)

# Save
output_html = "codeswitching_levels_2.html"
output_png = "codeswitching_levels_2.png"

fig_subplots.write_html(output_html)
fig_subplots.write_image(output_png)

log("Finished PCA plotting with shared directions.")
