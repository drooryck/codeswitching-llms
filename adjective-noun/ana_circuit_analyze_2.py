import torch
from transformer_lens import HookedTransformer, utils, patching
from utils_sva import get_logit_diff
import json
import numpy as np
import plotly.express as px
import os
import time
from tqdm import tqdm

# === Logging helper ===
def log_time(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# === Load Dataset ===
def load_gender_dataset(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)['examples']

# === Compute logit difference ===
def compute_logit_diff(model, tokens_masc, tokens_fem, label_masc, label_fem):
    logits_masc = model(tokens_masc)[:, -1]
    logits_fem = model(tokens_fem)[:, -1]
    diff = (logits_fem[:, label_fem] - logits_fem[:, label_masc]) - (logits_masc[:, label_fem] - logits_masc[:, label_masc])
    return diff.mean().item()

# === Activation patching function ===
def run_activation_patching(model, dataset, layer_range=None, head_range=None, device='cuda'):
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    if layer_range is None:
        layer_range = range(n_layers)
    if head_range is None:
        head_range = range(n_heads)

    results = torch.zeros((n_layers, n_heads), device=device)

    for layer in tqdm(layer_range, desc="Layers"):
        for head in head_range:
            contributions = []
            for example in dataset.values():
                tokens_masc = model.to_tokens(example['masc_sentence']).to(device)
                tokens_fem = model.to_tokens(example['fem_sentence']).to(device)

                fem_tokens = model.to_tokens(example['fem_label'], prepend_bos=False)[0].to(device)
                masc_tokens = model.to_tokens(example['masc_label'], prepend_bos=False)[0].to(device)
                label_fem = fem_tokens[-1].item()
                label_masc = masc_tokens[-1].item()

                baseline_diff = compute_logit_diff(model, tokens_masc, tokens_fem, label_masc, label_fem)

                def patch_attn(activations, hook):
                    activations[:, -1, head, :] = fem_cache[hook.name][:, -1, head, :]
                    return activations

                hook_name = f'blocks.{layer}.attn.hook_result'
                _, fem_cache = model.run_with_cache(tokens_fem, names_filter=[hook_name])

                patched_logits = model.run_with_hooks(tokens_masc, fwd_hooks=[(hook_name, patch_attn)])
                patched_diff = compute_logit_diff(model, tokens_masc, tokens_fem, label_masc, label_fem)

                contribution = patched_diff - baseline_diff
                contributions.append(contribution)

                torch.cuda.empty_cache()

            results[layer, head] = np.mean(contributions)

    return results.cpu().numpy()


# === Visualization ===
def visualize_heatmap(contrib_matrix, output_path):
    fig = px.imshow(
        contrib_matrix,
        labels={'x': 'Head', 'y': 'Layer'},
        title='Activation Patching: Contribution to Gender Agreement',
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    fig.update_layout(xaxis=dict(tickmode='linear'), yaxis=dict(tickmode='linear'))
    fig.write_image(output_path)
    fig.write_html(output_path.replace(".png", ".html"))

# === Main workflow ===
def main():
    log_time("Loading model...")
    model = HookedTransformer.from_pretrained("gemma-2b")
    model.set_use_attn_result(True)
    log_time("Model loaded.")

    log_time("Loading dataset...")
    dataset_path = "./datasets/gender_agreement/gender_agreement_dataset.json"
    dataset = load_gender_dataset(dataset_path)
    log_time(f"Loaded dataset with {len(dataset)} examples.")

    log_time("Running activation patching...")
    contrib_matrix = run_activation_patching(model, dataset)
    log_time("Activation patching complete.")

    # Save and visualize results
    os.makedirs('./results', exist_ok=True)
    result_path = './results/gender_agreement_patching.npy'
    np.save(result_path, contrib_matrix)
    log_time(f"Saved results to {result_path}")

    plot_path = './images/gender_agreement_patching.png'
    visualize_heatmap(contrib_matrix, plot_path)
    log_time(f"Saved heatmap to {plot_path}")

if __name__ == "__main__":
    main()
