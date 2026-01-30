#!/usr/bin/env python
"""Analyze language distribution in training dataset."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load dataset
train_file = "/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/results/nov20.0/runs/p50.00_run01/logs/train_dataset.csv"
df = pd.read_csv(train_file)

print("=" * 80)
print("Language Distribution Analysis")
print("=" * 80)
print(f"Total rows: {len(df):,}")
print(f"FR count: {(df['lang'] == 'fr').sum():,}")
print(f"NL count: {(df['lang'] == 'nl').sum():,}")
print(f"FR percentage: {(df['lang'] == 'fr').sum() / len(df) * 100:.2f}%")
print(f"NL percentage: {(df['lang'] == 'nl').sum() / len(df) * 100:.2f}%")

# Analyze language patches (consecutive sequences)
lang_col = df['lang'].values
patches = []
current_lang = lang_col[0]
current_length = 1

for i in range(1, len(lang_col)):
    if lang_col[i] == current_lang:
        current_length += 1
    else:
        patches.append((current_lang, current_length))
        current_lang = lang_col[i]
        current_length = 1
patches.append((current_lang, current_length))

fr_patches = [p[1] for p in patches if p[0] == 'fr']
nl_patches = [p[1] for p in patches if p[0] == 'nl']

print(f"\nPatch Statistics:")
print(f"Total patches: {len(patches):,}")
print(f"FR patches: {len(fr_patches):,}")
print(f"NL patches: {len(nl_patches):,}")
print(f"\nFR Patch Lengths:")
print(f"  Max: {max(fr_patches) if fr_patches else 0}")
print(f"  Mean: {np.mean(fr_patches):.2f}")
print(f"  Median: {np.median(fr_patches):.2f}")
print(f"  Q95: {np.percentile(fr_patches, 95):.2f}")
print(f"  Q99: {np.percentile(fr_patches, 99):.2f}")
print(f"\nNL Patch Lengths:")
print(f"  Max: {max(nl_patches) if nl_patches else 0}")
print(f"  Mean: {np.mean(nl_patches):.2f}")
print(f"  Median: {np.median(nl_patches):.2f}")
print(f"  Q95: {np.percentile(nl_patches, 95):.2f}")
print(f"  Q99: {np.percentile(nl_patches, 99):.2f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Language sequence over time (first 5000 and last 5000)
ax1 = axes[0, 0]
sample_size = 5000
lang_binary = (df['lang'] == 'fr').astype(int).values
ax1.plot(range(sample_size), lang_binary[:sample_size], alpha=0.6, linewidth=0.5, label='First 5000')
ax1.plot(range(sample_size), lang_binary[-sample_size:], alpha=0.6, linewidth=0.5, label='Last 5000', color='orange')
ax1.set_xlabel('Position in Dataset')
ax1.set_ylabel('Language (1=FR, 0=NL)')
ax1.set_title(f'Language Distribution (First & Last {sample_size} examples)')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Histogram of patch lengths
ax2 = axes[0, 1]
ax2.hist(fr_patches, bins=50, alpha=0.7, label='FR patches', color='blue', edgecolor='black')
ax2.hist(nl_patches, bins=50, alpha=0.7, label='NL patches', color='orange', edgecolor='black')
ax2.set_xlabel('Patch Length (consecutive examples)')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Language Patch Lengths')
ax2.legend()
ax2.set_yscale('log')
ax2.grid(alpha=0.3)

# 3. Patch lengths over time (first 1000 patches)
ax3 = axes[1, 0]
patch_langs = [p[0] for p in patches]
patch_lengths = [p[1] for p in patches]
fr_indices = [i for i, lang in enumerate(patch_langs[:1000]) if lang == 'fr']
nl_indices = [i for i, lang in enumerate(patch_langs[:1000]) if lang == 'nl']
ax3.scatter(fr_indices, [patch_lengths[i] for i in fr_indices], 
           alpha=0.6, label='FR', s=10, color='blue')
ax3.scatter(nl_indices, [patch_lengths[i] for i in nl_indices], 
           alpha=0.6, label='NL', s=10, color='orange')
ax3.set_xlabel('Patch Index (position in sequence)')
ax3.set_ylabel('Patch Length')
ax3.set_title('Patch Lengths Over Dataset (First 1000 Patches)')
ax3.legend()
ax3.grid(alpha=0.3)
ax3.set_yscale('log')

# 4. Running average of language proportion (sliding window)
ax4 = axes[1, 1]
window_size = 1000
running_fr_frac = []
for i in range(window_size, len(lang_binary), window_size):
    window_fr_frac = lang_binary[i-window_size:i].mean()
    running_fr_frac.append(window_fr_frac)

ax4.plot(range(window_size, len(lang_binary), window_size), running_fr_frac, 
         linewidth=2, alpha=0.7, color='purple')
ax4.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Expected 50%')
ax4.set_xlabel('Position in Dataset')
ax4.set_ylabel('Fraction of FR in Window')
ax4.set_title(f'Running Average of FR Fraction (window={window_size})')
ax4.legend()
ax4.grid(alpha=0.3)
ax4.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/scripts/train_lang_distribution.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: train_lang_distribution.png")
plt.close()

# Find where the longest NL patch occurs
max_nl_patch_idx = nl_patches.index(max(nl_patches))
nl_patch_positions = [i for i, (lang, _) in enumerate(patches) if lang == 'nl']
max_nl_patch_pos = nl_patch_positions[max_nl_patch_idx]
cumulative_pos = sum(p[1] for p in patches[:max_nl_patch_pos])
print(f"\nLongest NL patch:")
print(f"  Length: {max(nl_patches)} consecutive examples")
print(f"  Starts at position: {cumulative_pos:,} in dataset")
print(f"  That's {cumulative_pos/len(df)*100:.2f}% through the dataset")

