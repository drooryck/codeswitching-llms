#!/usr/bin/env python3
"""Parse experiment logs and SLURM output to create loss plots for mar25 runs.

Creates:
  loss_over_time_mar25.png — grid of subplots (one per proportion):
      train loss, eval FR, eval NL vs step

Data sources:
  - Train loss: SLURM stdout  {'loss': ..., 'epoch': ...}
  - Eval loss (FR/NL): experiment.log  "eval loss: overall=... fr=... nl=..."

Usage:
  python feb_exp/scripts/mar25/plot_loss.py [--slurm-log PATH]
"""
import argparse
import ast
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NETSCRATCH = Path("/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/mar25")
RUNS_DIR = NETSCRATCH / "runs"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "feb_exp" / "results" / "mar25"


def find_slurm_log() -> Path:
    """Find the most recent SLURM .out log in the logs directory."""
    logs_dir = NETSCRATCH / "logs"
    candidates = sorted(logs_dir.glob("slurm_*.out"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No SLURM .out files in {logs_dir}")
    return candidates[-1]


def parse_slurm_train_losses(slurm_path: Path):
    """Parse SLURM output for per-step training losses per (prop, run_id).

    Returns dict: (prop, run_id) -> list of (step_index, loss)
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
    """Parse experiment.log files for per-step eval losses.

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
    """Grid of subplots: one per proportion, averaged across seeds."""
    props = sorted({k[0] for k in eval_losses})
    if not props:
        print("No eval losses found, skipping loss plot.")
        return

    ncols = 4
    nrows = (len(props) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows), squeeze=False)

    def step_to_actual(step_idx):
        return step_idx * eval_interval

    for idx, prop in enumerate(props):
        ax = axes[idx // ncols][idx % ncols]
        run_ids = sorted({k[1] for k in eval_losses if k[0] == prop})

        for run_id in run_ids:
            alpha = 0.15 if len(run_ids) > 1 else 1.0

            tl = train_losses.get((prop, run_id), [])
            if tl:
                steps_t = [step_to_actual(s) for s, _ in tl]
                vals_t = [v for _, v in tl]
                ax.plot(steps_t, vals_t, color='gray', alpha=alpha, linewidth=0.8)

            el = eval_losses.get((prop, run_id), [])
            if el:
                steps_e = [step_to_actual(s) for s, _, _, _ in el]
                fr_vals = [fr for _, _, fr, _ in el]
                nl_vals = [nl for _, _, _, nl in el]
                ax.plot(steps_e, fr_vals, color='blue', alpha=alpha, linewidth=0.8)
                ax.plot(steps_e, nl_vals, color='red', alpha=alpha, linewidth=0.8)

        if len(run_ids) > 1:
            max_steps_t = max((len(train_losses.get((prop, r), [])) for r in run_ids), default=0)
            if max_steps_t > 0:
                avg_t = []
                for si in range(max_steps_t):
                    vals = [train_losses[(prop, r)][si][1] for r in run_ids
                            if (prop, r) in train_losses and len(train_losses[(prop, r)]) > si]
                    if vals:
                        avg_t.append(np.mean(vals))
                actual_steps = [step_to_actual(i + 1) for i in range(len(avg_t))]
                ax.plot(actual_steps, avg_t, color='gray', linewidth=2.5, label='train loss (mean)')

            max_steps_e = max((len(eval_losses.get((prop, r), [])) for r in run_ids), default=0)
            if max_steps_e > 0:
                avg_fr, avg_nl = [], []
                for si in range(max_steps_e):
                    fr_vals = [eval_losses[(prop, r)][si][2] for r in run_ids
                               if (prop, r) in eval_losses and len(eval_losses[(prop, r)]) > si]
                    nl_vals = [eval_losses[(prop, r)][si][3] for r in run_ids
                               if (prop, r) in eval_losses and len(eval_losses[(prop, r)]) > si]
                    if fr_vals:
                        avg_fr.append(np.mean(fr_vals))
                    if nl_vals:
                        avg_nl.append(np.mean(nl_vals))
                actual_steps_e = [step_to_actual(i + 1) for i in range(len(avg_fr))]
                ax.plot(actual_steps_e, avg_fr, color='blue', linewidth=2.5, label='eval FR (mean)')
                ax.plot(actual_steps_e, avg_nl, color='red', linewidth=2.5, label='eval NL (mean)')

        ax.set_yscale('log')
        ax.set_title(f"FR: {prop*100:.0f}%", fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss (log)")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')

    for idx in range(len(props), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        "Loss over training — mar25 (21 props × 20 seeds)",
        fontsize=16,
        fontweight='bold',
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slurm-log", type=Path, default=None,
                        help="Path to SLURM .out log (auto-detected if omitted)")
    args = parser.parse_args()

    slurm_path = args.slurm_log or find_slurm_log()
    print(f"Using SLURM log: {slurm_path}")

    print("Parsing SLURM log for train losses...")
    train_losses = parse_slurm_train_losses(slurm_path)
    print(f"  Found train losses for {len(train_losses)} runs")

    print("Parsing experiment logs for eval losses...")
    eval_losses = parse_eval_losses(RUNS_DIR)
    print(f"  Found eval losses for {len(eval_losses)} runs")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss_over_time(train_losses, eval_losses, OUTPUT_DIR / "loss_over_time_mar25.png")


if __name__ == "__main__":
    main()
