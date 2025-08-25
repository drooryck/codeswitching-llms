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

# === Run PCA & visualize each switching level ===
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

    log("Running model and extracting attention head activations...")
    logits, cache = model.run_with_cache(tokens, names_filter=["blocks.13.attn.hook_result"])
    activations = cache["blocks.13.attn.hook_result"][:, -1, 7, :].cpu().numpy()


    log("Running PCA...")
    pca = PCA(n_components=2)
    projected = pca.fit_transform(activations)

    log("Generating plot...")
    fig = px.scatter(
        x=projected[:, 0], y=projected[:, 1],
        color=labels,
        labels={'x': 'PC1', 'y': 'PC2'},
        title=f"Codeswitching Level: {level}",
        hover_name=sentences
    )
    figs.append(fig)

# === Combine all plots into a single subplot ===
log("Combining plots into subplot layout...")
fig_subplots = make_subplots(rows=1, cols=4, subplot_titles=levels)

for i, fig in enumerate(figs, start=1):
    for trace in fig.data:
        fig_subplots.add_trace(trace, row=1, col=i)

fig_subplots.update_layout(height=500, width=2000, showlegend=False)

# === Save plots ===
output_html = "codeswitching_levels.html"
output_png = "codeswitching_levels.png"

log(f"Saving to {output_html} and {output_png}...")
fig_subplots.write_html(output_html)
fig_subplots.write_image(output_png)
log("Done.")

# === Optional display (commented out for cluster use)
# fig_subplots.show()
