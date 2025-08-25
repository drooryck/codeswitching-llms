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
data_dir = './corrupted_cs_datasets'  # Path to corrupted datasets
figs = []

# === Run PCA & visualize each switching level with corrupted data ===
for level in levels:
    log(f"Processing corrupted level: {level}")
    dataset_path = f"{data_dir}/corrupted_spanish_{level}_sva.json"
    
    with open(dataset_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    sentences, labels = [], []
    for item in dataset.values():
        sentences.append(item['src'])
        labels.append(item['base_subject_number'])

    log(f"Tokenizing {len(sentences)} sentences...")
    tokens = model.to_tokens(sentences)

    log("Running model and extracting attention head activations...")
    # Using the same layer (13) and head (7) as in the original file
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
        title=f"Corrupted Codeswitching: {level}",
        hover_name=sentences
    )
    figs.append(fig)

# === Combine all plots into a single subplot ===
log("Combining plots into subplot layout...")
fig_subplots = make_subplots(
    rows=1, cols=4, 
    subplot_titles=[f"Corrupted {level}" for level in levels]
)

for i, fig in enumerate(figs, start=1):
    for trace in fig.data:
        fig_subplots.add_trace(trace, row=1, col=i)

# Update layout for better visualization
fig_subplots.update_layout(
    height=600, 
    width=2000, 
    title_text="PCA of Corrupted Spanish Codeswitched Datasets",
    showlegend=True,
    legend=dict(
        title="Subject Number",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Update x and y axes labels
for i in range(1, 5):
    fig_subplots.update_xaxes(title_text="PC1", row=1, col=i)
    fig_subplots.update_yaxes(title_text="PC2", row=1, col=i)

# === Save plots ===
output_dir = "corrupted_analysis"
os.makedirs(output_dir, exist_ok=True)
output_html = f"{output_dir}/corrupted_codeswitching_levels.html"
output_png = f"{output_dir}/corrupted_codeswitching_levels.png"

log(f"Saving to {output_html} and {output_png}...")
fig_subplots.write_html(output_html)
fig_subplots.write_image(output_png)
log("Done.")

# === Optional display (uncomment to show in notebook) ===
# fig_subplots.show()