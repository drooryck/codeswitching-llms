import torch
from transformer_lens import HookedTransformer
from utils_sva import run_pca, paper_plot
import json
import numpy as np
from neel_plotly import scatter
import time
import os

# === Logger helper ===
def log_time(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# === Load Dataset ===
def load_dataset(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)['examples']

# === Extract Activations ===
def extract_activations(model, dataset, layer, head):
    activations = []
    labels = []
    
    for example in dataset.values():
        masc_tokens = model.to_tokens(example["masc_sentence"])
        fem_tokens = model.to_tokens(example["fem_sentence"])
        
        _, masc_cache = model.run_with_cache(masc_tokens)
        _, fem_cache = model.run_with_cache(fem_tokens)
        
        masc_act = masc_cache[f'blocks.{layer}.attn.hook_result'][0, -1, head].cpu().numpy()
        fem_act = fem_cache[f'blocks.{layer}.attn.hook_result'][0, -1, head].cpu().numpy()
        
        activations.append(masc_act)
        activations.append(fem_act)
        labels.append('Masculine')
        labels.append('Feminine')
    
    return np.array(activations), labels

# === Perform PCA ===
def perform_pca(activations):
    pca_matrix, pca, scaler = run_pca(activations, n_components=2)
    return pca_matrix

# === Main Analysis Workflow ===
def main():
    log_time("Loading model...")
    model = HookedTransformer.from_pretrained("gemma-2b")
    model.set_use_attn_result(True)
    log_time("Model loaded.")

    log_time("Loading dataset...")
    dataset_path = "./datasets/gender_agreement/gender_agreement_dataset.json"
    dataset = load_dataset(dataset_path)
    log_time(f"Dataset loaded with {len(dataset)} examples.")

    layer, head = 13, 7  # Example attention head
    log_time(f"Extracting activations from layer {layer}, head {head}...")
    activations, labels = extract_activations(model, dataset, layer, head)
    log_time("Activations extracted.")

    log_time("Performing PCA...")
    pca_matrix = perform_pca(activations)
    log_time("PCA performed.")

    hover_texts = []
    for example in dataset.values():
        # Add once for masculine
        hover_texts.append(f"MASCULINE: {example['masc_sentence']}")
        # Add once for feminine
        hover_texts.append(f"FEMININE: {example['fem_sentence']}")

    log_time("Creating PCA plot...")
    fig = scatter(
        pca_matrix[:, 0],
        pca_matrix[:, 1],
        color=labels,
        hover=hover_texts,
        title="PCA of Gender Agreement Activations",
        return_fig=True
    )
    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(legend_title_text='Gender')
    fig = paper_plot(fig)

    os.makedirs('./images/', exist_ok=True)
    fig.write_image("./images/gender_agreement_pca.png")
    fig.write_html("./images/gender_agreement_pca.html")
    log_time("PCA plot saved.")

if __name__ == "__main__":
    main()
