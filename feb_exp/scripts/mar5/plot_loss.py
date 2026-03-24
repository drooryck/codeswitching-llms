#!/usr/bin/env python3
"""Parse experiment logs and SLURM output to create loss plots for mar5 runs.

Creates two figures:
  1. loss_over_time.png  — grid of subplots (one per proportion): train loss, eval FR, eval NL vs step
  2. loss_3d.png         — 3D surface plots: training loss, eval FR, eval NL (axes: step, FR%, log10(loss))

Data sources:
  - Train loss: SLURM stdout  {'loss': ..., 'epoch': ...}
  - Eval loss (FR/NL): experiment.log  "eval loss: overall=... fr=... nl=..."
"""
import ast
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = Path("/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/mar5/runs")
SLURM_LOG = Path("/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/mar5/logs/slurm_63961245.out")
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "feb_exp" / "results" / "mar5"


def parse_slurm_train_losses(slurm_path: Path):
    """Parse SLURM output to get per-step training losses for each (prop, run_id).

    Returns dict: (prop, run_id) -> list of (step_index, loss)
    step_index is 1-based (1=first logging step, 2=second, etc.)
    """
    results = {}
    current_key = None
    step_counter = 0

    with open(slurm_path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r'\[\d+/\d+\] START\s+prop=([\d.]+)\s+run=(\d+)', line)
            if m:
                prop = float(m.group(1))
                run_id = int(m.group(2))
                current_key = (prop, run_id)
                results[current_key] = []
                step_counter = 0
                continue

            if current_key and line.startswith("{'loss':"):
                try:
                    d = ast.literal_eval(line)
                    if 'loss' in d and 'train_runtime' not in d:
                        step_counter += 1
                        results[current_key].append((step_counter, d['loss']))
                except (ValueError, SyntaxError):
                    pass

            if line.startswith("[") and "DONE" in line:
                current_key = None

    return results


def parse_eval_losses(runs_dir: Path):
    """Parse experiment.log files to get per-step eval losses.

    Returns dict: (prop, run_id) -> list of (step_index, overall, fr, nl)
    """
    results = {}
    eval_re = re.compile(r'eval loss: overall=([\d.]+)\s+fr=([\d.]+)\s+nl=([\d.]+)')

    for run_dir in sorted(runs_dir.glob("p*_run*")):
        log_path = run_dir / "experiment.log"
        if not log_path.exists():
            continue
        name = run_dir.name
        parts = name.split("_run")
        prop = float(parts[0][1:]) / 100.0
        run_id = int(parts[1])

        step_counter = 0
        losses = []
        with open(log_path) as f:
            for line in f:
                m = eval_re.search(line)
                if m:
                    step_counter += 1
                    losses.append((
                        step_counter,
                        float(m.group(1)),
                        float(m.group(2)),
                        float(m.group(3)),
                    ))
        results[(prop, run_id)] = losses

    return results


def plot_loss_over_time(train_losses, eval_losses, output_path: Path, eval_interval: int = 200):
    """Grid of subplots: one per proportion, showing train/eval-FR/eval-NL loss over steps."""
    props = sorted({k[0] for k in eval_losses})
    target_props = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    selected = []
    for tp in target_props:
        closest = min(props, key=lambda p: abs(p - tp))
        if closest not in selected and abs(closest - tp) < 0.03:
            selected.append(closest)
    if not selected:
        selected = props[:12]

    ncols = 4
    nrows = (len(selected) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows), squeeze=False)

    def step_to_actual(step_idx):
        return step_idx * eval_interval

    for idx, prop in enumerate(selected):
        ax = axes[idx // ncols][idx % ncols]
        run_ids = sorted({k[1] for k in eval_losses if k[0] == prop})

        for run_id in run_ids:
            alpha = 0.3 if len(run_ids) > 1 else 1.0

            tl = train_losses.get((prop, run_id), [])
            if tl:
                steps_t = [step_to_actual(s) for s, _ in tl]
                vals_t = [v for _, v in tl]
                ax.plot(steps_t, vals_t, color='gray', alpha=alpha, linewidth=1)

            el = eval_losses.get((prop, run_id), [])
            if el:
                steps_e = [step_to_actual(s) for s, _, _, _ in el]
                fr_vals = [fr for _, _, fr, _ in el]
                nl_vals = [nl for _, _, _, nl in el]
                ax.plot(steps_e, fr_vals, color='blue', alpha=alpha, linewidth=1)
                ax.plot(steps_e, nl_vals, color='red', alpha=alpha, linewidth=1)

        all_run_ids = sorted({k[1] for k in eval_losses if k[0] == prop})
        if len(all_run_ids) > 1:
            max_steps_t = max(len(train_losses.get((prop, r), [])) for r in all_run_ids)
            if max_steps_t > 0:
                avg_t = []
                for si in range(max_steps_t):
                    vals = [train_losses[(prop, r)][si][1] for r in all_run_ids
                            if (prop, r) in train_losses and len(train_losses[(prop, r)]) > si]
                    if vals:
                        avg_t.append(np.mean(vals))
                actual_steps = [step_to_actual(i + 1) for i in range(len(avg_t))]
                ax.plot(actual_steps, avg_t, color='gray', linewidth=2.5, label='train loss')

            max_steps_e = max(len(eval_losses.get((prop, r), [])) for r in all_run_ids)
            if max_steps_e > 0:
                avg_fr, avg_nl = [], []
                for si in range(max_steps_e):
                    fr_vals = [eval_losses[(prop, r)][si][2] for r in all_run_ids
                               if (prop, r) in eval_losses and len(eval_losses[(prop, r)]) > si]
                    nl_vals = [eval_losses[(prop, r)][si][3] for r in all_run_ids
                               if (prop, r) in eval_losses and len(eval_losses[(prop, r)]) > si]
                    if fr_vals:
                        avg_fr.append(np.mean(fr_vals))
                    if nl_vals:
                        avg_nl.append(np.mean(nl_vals))
                actual_steps_e = [step_to_actual(i + 1) for i in range(len(avg_fr))]
                ax.plot(actual_steps_e, avg_fr, color='blue', linewidth=2.5, label='eval FR')
                ax.plot(actual_steps_e, avg_nl, color='red', linewidth=2.5, label='eval NL')

        ax.set_yscale('log')
        ax.set_title(f"FR proportion: {prop*100:.0f}%", fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss (log scale)")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')

    # Hide unused axes
    for idx in range(len(selected), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Loss over training — mar5 plurality-mixing", fontsize=16, fontweight='bold')
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def plot_loss_3d(train_losses, eval_losses, output_path: Path):
    """Three 3D surface plots: training loss, eval FR, eval NL."""
    props = sorted({k[0] for k in eval_losses})
    run_ids_per_prop = {p: sorted({k[1] for k in eval_losses if k[0] == p}) for p in props}

    # Determine step grid (use first run's step count)
    sample_key = next(iter(eval_losses))
    n_steps = len(eval_losses[sample_key])
    eval_interval = 200  # eval_steps from model config

    # Build grids: X=Step (left axis), Y=FR% (right axis)
    step_indices = np.arange(1, n_steps + 1) * eval_interval
    fr_pcts = np.array([p * 100 for p in props])
    X, Y = np.meshgrid(step_indices, fr_pcts)

    # Average losses across runs (Z indexed as [prop, step])
    Z_train = np.full_like(X, np.nan, dtype=float)
    Z_eval_fr = np.full_like(X, np.nan, dtype=float)
    Z_eval_nl = np.full_like(X, np.nan, dtype=float)

    for i, prop in enumerate(props):
        rids = run_ids_per_prop[prop]
        for j in range(n_steps):
            # Train loss
            t_vals = [train_losses[(prop, r)][j][1] for r in rids
                      if (prop, r) in train_losses and len(train_losses[(prop, r)]) > j]
            if t_vals:
                Z_train[i, j] = np.mean(t_vals)

            # Eval losses
            e_vals_fr = [eval_losses[(prop, r)][j][2] for r in rids
                         if (prop, r) in eval_losses and len(eval_losses[(prop, r)]) > j]
            e_vals_nl = [eval_losses[(prop, r)][j][3] for r in rids
                         if (prop, r) in eval_losses and len(eval_losses[(prop, r)]) > j]
            if e_vals_fr:
                Z_eval_fr[i, j] = np.mean(e_vals_fr)
            if e_vals_nl:
                Z_eval_nl[i, j] = np.mean(e_vals_nl)

    Z_train_log = np.log10(np.clip(Z_train, 1e-6, None))
    Z_eval_fr_log = np.log10(np.clip(Z_eval_fr, 1e-6, None))
    Z_eval_nl_log = np.log10(np.clip(Z_eval_nl, 1e-6, None))

    fig = plt.figure(figsize=(20, 6))
    panels = [
        (Z_train_log, "Training loss", "Greys_r"),
        (Z_eval_fr_log, "Eval loss — FR", "Blues_r"),
        (Z_eval_nl_log, "Eval loss — NL", "Reds_r"),
    ]

    for pidx, (Z, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(1, 3, pidx + 1, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85, edgecolor='gray', linewidth=0.3)
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel("FR %", fontsize=9)
        ax.set_zlabel("log₁₀(loss)", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.view_init(elev=25, azim=315)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="log₁₀(loss)")

    fig.suptitle("mar5 — plurality-mixing", fontsize=14, fontweight='bold')
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def main():
    print("Parsing SLURM log for train losses...")
    train_losses = parse_slurm_train_losses(SLURM_LOG)
    print(f"  Found train losses for {len(train_losses)} runs")

    print("Parsing experiment logs for eval losses...")
    eval_losses = parse_eval_losses(RUNS_DIR)
    print(f"  Found eval losses for {len(eval_losses)} runs")

    # Verify alignment
    props_t = sorted({k[0] for k in train_losses})
    props_e = sorted({k[0] for k in eval_losses})
    print(f"  Train: {len(props_t)} proportions, Eval: {len(props_e)} proportions")

    plot_loss_over_time(train_losses, eval_losses, OUTPUT_DIR / "loss_over_time_mar5.png")
    plot_loss_3d(train_losses, eval_losses, OUTPUT_DIR / "loss_3d_mar5.png")


if __name__ == "__main__":
    main()
