#!/bin/bash
#SBATCH --job-name=orientation_oct29
#SBATCH --partition=seas_compute
#SBATCH --account=dam_lab
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=32
#SBATCH --output=/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/scripts/logs/orientation_oct29_%j.out
#SBATCH --error=/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/scripts/logs/orientation_oct29_%j.err

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

print("Starting orientation score computation WITH COMPONENT BREAKDOWNS...")

# Setup
results_dir = Path("/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/results/oct21_overnight")
base_plots_dir = Path("/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/scripts/temp_plots_oct29")
base_plots_dir.mkdir(parents=True, exist_ok=True)

metrics_obj = Metrics("/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/data/lexicon_sep22.json")

run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('p')])
print(f"Found {len(run_dirs)} run directories")

import time
from joblib import Parallel, delayed

def process_run(run_dir, ablation_type):
    """Process a single run directory for given ablation type - WITH COMPONENT BREAKDOWNS."""
    start_time = time.time()
    
    parts = run_dir.name.split("_run")
    if len(parts) != 2:
        return []
    
    try:
        prop = float(parts[0][1:]) / 100.0
        run_id = int(parts[1])
    except ValueError:
        return []
    
    pred_file = run_dir / "ablation_predictions.csv"
    if not pred_file.exists():
        return []
    
    df = pd.read_csv(pred_file)
    df_abl = df[df['ablation'] == ablation_type]
    
    results = []
    for lang in ['fr', 'nl']:
        lang_preds = df_abl[df_abl['language'] == lang]
        
        # Process each prediction individually with breakdown
        for idx, row in lang_preds.iterrows():
            pred = row['prediction']
            result = metrics_obj.language_orientation_score(pred, return_breakdown=True)
            
            results.append({
                'prop': prop,
                'run_id': run_id,
                'language': lang,
                'input': row['input'],
                'prediction': pred,
                'gold': row['gold'],
                'orientation_score': result['score'],
                'structure': result['components']['structure'],
                'aux': result['components']['aux'],
                'participle': result['components']['participle'],
                'determiner': result['components']['determiner'],
                'noun': result['components']['noun']
            })
    
    elapsed = time.time() - start_time
    if len(results) > 0:
        print(f"[DONE] {ablation_type:8s} | {run_dir.name} | {len(results)} predictions in {elapsed:.2f}s", flush=True)
    
    return results

# Process each ablation type
for ablation_type in ['none', 'subject', 'verb', 'object']:
    print(f"\n{'='*70}")
    print(f"Processing ablation={ablation_type}")
    print(f"{'='*70}")
    
    start_abl = time.time()
    
    # Run in parallel with 32 CPUs
    all_results = Parallel(n_jobs=32, verbose=5)(
        delayed(process_run)(run_dir, ablation_type) for run_dir in run_dirs
    )
    
    # Flatten results - now we have per-prediction details
    detailed_results = [item for sublist in all_results for item in sublist]
    
    elapsed_abl = time.time() - start_abl
    print(f"\nAblation {ablation_type} processing time: {elapsed_abl:.2f}s ({elapsed_abl/60:.2f} minutes)")
    print(f"Total predictions processed: {len(detailed_results)}")
    
    # Save DETAILED results (every prediction with all components)
    df_detailed = pd.DataFrame(detailed_results)
    plots_dir = base_plots_dir / f"ablation_{ablation_type}"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    detailed_path = plots_dir / 'orientation_detailed.csv'
    df_detailed.to_csv(detailed_path, index=False)
    print(f"✓ Saved detailed data to {detailed_path}")
    
    # Create SUMMARY by run (aggregated statistics)
    summary_results = []
    for (prop, run_id, lang), group in df_detailed.groupby(['prop', 'run_id', 'language']):
        summary_results.append({
            'prop': prop,
            'run_id': run_id,
            'language': lang,
            f'{lang}_orientation': group['orientation_score'].mean(),
            f'{lang}_structure': group['structure'].mean(),
            f'{lang}_aux': group['aux'].mean(),
            f'{lang}_participle': group['participle'].mean(),
            f'{lang}_determiner': group['determiner'].mean(),
            f'{lang}_noun': group['noun'].mean(),
            'n_predictions': len(group)
        })
    
    df_summary = pd.DataFrame(summary_results)
    summary_path = plots_dir / 'orientation_summary.csv'
    df_summary.to_csv(summary_path, index=False)
    print(f"✓ Saved summary to {summary_path}")

print(f"\n{'='*70}")
print(f"✅ All done! Results saved to {base_plots_dir}")
print(f"{'='*70}")
print(f"\nOutput structure:")
print(f"  temp_plots_oct29/")
print(f"    ablation_none/")
print(f"      orientation_detailed.csv  (all predictions with components)")
print(f"      orientation_summary.csv   (aggregated by run)")
print(f"    ablation_subject/")
print(f"    ablation_verb/")
print(f"    ablation_object/")
PYTHON_SCRIPT


