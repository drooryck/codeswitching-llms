#!/bin/bash
#SBATCH --job-name=orientation
#SBATCH --partition=seas_compute
#SBATCH --account=dam_lab
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/scripts/logs/orientation_%j.out
#SBATCH --error=/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/scripts/logs/orientation_%j.err

# Load environment
module load python/3.10.9-fasrc01
source /n/home06/drooryck/envs/codeswitching-py310/bin/activate

# Set PYTHONPATH
export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:$PYTHONPATH

# Run Python code
python << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '/n/home06/drooryck/codeswitching-llms')

from pathlib import Path
import pandas as pd
from july_aug_sept_exp.src.metrics import Metrics
from july_aug_sept_exp.src.plotting import BilingualPlotter

print("Starting orientation score computation...")

# Setup
results_dir = Path("/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/results/sep24.3")
base_plots_dir = Path("/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/scripts/temp_plots_sep24")

metrics_obj = Metrics("/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/data/lexicon_sep22.json")

run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('p')])

import time
from joblib import Parallel, delayed

def process_run(run_dir, ablation_type):
    """Process a single run directory for given ablation type."""
    start_time = time.time()
    print(f"[START] {ablation_type:8s} | {run_dir.name}", flush=True)
    
    parts = run_dir.name.split("_run")
    prop = float(parts[0][1:]) / 100.0
    run_id = int(parts[1])
    
    pred_file = run_dir / "ablation_predictions.csv"
    if not pred_file.exists():
        print(f"[SKIP] {ablation_type:8s} | {run_dir.name} - no predictions file", flush=True)
        return []
    
    df = pd.read_csv(pred_file)
    df_abl = df[df['ablation'] == ablation_type]
    
    results = []
    for lang in ['fr', 'nl']:
        lang_preds = df_abl[df_abl['language'] == lang]['prediction']
        scores = [metrics_obj.language_orientation_score(pred) for pred in lang_preds]
        avg_score = sum(scores) / len(scores) if scores else 0.5
        
        results.append({
            'prop': prop,
            'run_id': run_id,
            f'{lang}_orientation': avg_score
        })
    
    elapsed = time.time() - start_time
    print(f"[DONE] {ablation_type:8s} | {run_dir.name} in {elapsed:.2f}s", flush=True)
    return results

# Process each ablation type
for ablation_type in ['none', 'subject', 'verb', 'object']:
    print(f"\n{'='*70}")
    print(f"Processing ablation={ablation_type}")
    print(f"{'='*70}")
    
    start_abl = time.time()
    
    # Run in parallel
    all_results = Parallel(n_jobs=8, verbose=0)(
        delayed(process_run)(run_dir, ablation_type) for run_dir in run_dirs
    )
    
    # Flatten results
    orientation_results = [item for sublist in all_results for item in sublist]
    
    elapsed_abl = time.time() - start_abl
    print(f"\nAblation {ablation_type} processing time: {elapsed_abl:.2f}s ({elapsed_abl/60:.2f} minutes)")
    
    # Create DataFrame
    orientation_df = pd.DataFrame(orientation_results)
    orientation_df = orientation_df.groupby(['prop', 'run_id']).first().reset_index()
    
    # Save and plot
    plots_dir = base_plots_dir / f"ablation_{ablation_type}"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = plots_dir / 'orientation_scores.csv'
    orientation_df.to_csv(csv_path, index=False)
    print(f"Saved data to {csv_path}")
    
    plotter = BilingualPlotter(orientation_df, plots_dir)
    plotter.plot_language_orientation()
    print(f"Plot saved to {plots_dir / 'orientation.png'}")

print(f"\n{'='*70}")
print(f"All done! Plots saved to {base_plots_dir}/ablation_*/orientation.png")
print(f"{'='*70}")
PYTHON_SCRIPT

