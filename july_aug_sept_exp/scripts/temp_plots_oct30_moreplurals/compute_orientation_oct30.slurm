#!/bin/bash
#SBATCH --job-name=orientation_oct30
#SBATCH --partition=seas_compute
#SBATCH --account=dam_lab
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --output=/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/scripts/logs/orientation_oct30_%j.out
#SBATCH --error=/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/scripts/logs/orientation_oct30_%j.err

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

print("Starting orientation score computation for oct30_moreplurals...")

# Setup
results_dir = Path("/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/results/oct30_moreplurals")
base_plots_dir = Path("/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/scripts/temp_plots_oct30_moreplurals")

metrics_obj = Metrics("/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/data/lexicon_sep22.json")

run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('p')])
print(f"Found {len(run_dirs)} run directories")

import time
from joblib import Parallel, delayed

def process_run(run_dir, ablation_type):
    """Process a single run directory with component breakdowns."""
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
    
    # Build ONE result dict with both languages
    result = {'prop': prop, 'run_id': run_id}
    
    for lang in ['fr', 'nl']:
        lang_preds = df_abl[df_abl['language'] == lang]['prediction']
        
        # Accumulate ALL component scores and quality metrics
        scores, structures, auxs, participles, determiners, nouns = [], [], [], [], [], []
        structure_bads, aux_missings, part_missings, det_missings, noun_missings = [], [], [], [], []
        wrong_token_counts, wrong_noun_counts, wrong_det_counts = [], [], []
        wrong_aux_counts, wrong_part_counts = [], []
        
        for pred in lang_preds:
            breakdown = metrics_obj.language_orientation_score(pred, return_breakdown=True)
            scores.append(breakdown['score'])
            structures.append(breakdown['components']['structure'])
            auxs.append(breakdown['components']['aux'])
            participles.append(breakdown['components']['participle'])
            determiners.append(breakdown['components']['determiner'])
            nouns.append(breakdown['components']['noun'])
            
            # Quality metrics
            structure_bads.append(breakdown['quality']['structure_bad'])
            aux_missings.append(breakdown['quality']['aux_missing'])
            part_missings.append(breakdown['quality']['participle_missing'])
            det_missings.append(breakdown['quality']['determiners_missing'])
            noun_missings.append(breakdown['quality']['nouns_missing'])
            wrong_token_counts.append(breakdown['quality']['wrong_token_count'])
            wrong_noun_counts.append(breakdown['quality']['wrong_noun_count'])
            wrong_det_counts.append(breakdown['quality']['wrong_det_count'])
            wrong_aux_counts.append(breakdown['quality']['wrong_aux_count'])
            wrong_part_counts.append(breakdown['quality']['wrong_part_count'])
        
        # Add ALL averaged metrics to result dict
        if scores:
            result[f'{lang}_orientation'] = sum(scores) / len(scores)
            result[f'{lang}_structure'] = sum(structures) / len(structures)
            result[f'{lang}_aux'] = sum(auxs) / len(auxs)
            result[f'{lang}_participle'] = sum(participles) / len(participles)
            result[f'{lang}_determiner'] = sum(determiners) / len(determiners)
            result[f'{lang}_noun'] = sum(nouns) / len(nouns)
            
            # Quality metrics
            result[f'{lang}_structure_bad'] = sum(structure_bads) / len(structure_bads)
            result[f'{lang}_aux_missing'] = sum(aux_missings) / len(aux_missings)
            result[f'{lang}_participle_missing'] = sum(part_missings) / len(part_missings)
            result[f'{lang}_determiners_missing'] = sum(det_missings) / len(det_missings)
            result[f'{lang}_nouns_missing'] = sum(noun_missings) / len(noun_missings)
            result[f'{lang}_wrong_token_count'] = sum(wrong_token_counts) / len(wrong_token_counts)
            result[f'{lang}_wrong_noun_count'] = sum(wrong_noun_counts) / len(wrong_noun_counts)
            result[f'{lang}_wrong_det_count'] = sum(wrong_det_counts) / len(wrong_det_counts)
            result[f'{lang}_wrong_aux_count'] = sum(wrong_aux_counts) / len(wrong_aux_counts)
            result[f'{lang}_wrong_part_count'] = sum(wrong_part_counts) / len(wrong_part_counts)
        else:
            result[f'{lang}_orientation'] = 0.5
            result[f'{lang}_structure'] = 0.0
            result[f'{lang}_aux'] = 0.0
            result[f'{lang}_participle'] = 0.0
            result[f'{lang}_determiner'] = 0.0
            result[f'{lang}_noun'] = 0.0
            result[f'{lang}_structure_bad'] = 0.0
            result[f'{lang}_aux_missing'] = 0.0
            result[f'{lang}_participle_missing'] = 0.0
            result[f'{lang}_determiners_missing'] = 0.0
            result[f'{lang}_nouns_missing'] = 0.0
            result[f'{lang}_wrong_token_count'] = 0.0
            result[f'{lang}_wrong_noun_count'] = 0.0
            result[f'{lang}_wrong_det_count'] = 0.0
            result[f'{lang}_wrong_aux_count'] = 0.0
            result[f'{lang}_wrong_part_count'] = 0.0
    
    elapsed = time.time() - start_time
    print(f"[DONE] {ablation_type:8s} | {run_dir.name} in {elapsed:.2f}s", flush=True)
    return [result]

# Process each ablation type
for ablation_type in ['none', 'subject', 'verb', 'object']:
    print(f"\n{'='*70}")
    print(f"Processing ablation={ablation_type}")
    print(f"{'='*70}")
    
    start_abl = time.time()
    
    # Run in parallel
    all_results = Parallel(n_jobs=32, verbose=0)(
        delayed(process_run)(run_dir, ablation_type) for run_dir in run_dirs
    )
    
    # Flatten results (already one row per run)
    orientation_results = [item for sublist in all_results for item in sublist]
    
    elapsed_abl = time.time() - start_abl
    print(f"\nAblation {ablation_type} processing time: {elapsed_abl:.2f}s ({elapsed_abl/60:.2f} minutes)")
    print(f"Total runs processed: {len(orientation_results)}")
    
    # Create DataFrame (already aggregated, no need to groupby)
    orientation_df = pd.DataFrame(orientation_results)
    
    # Save with component breakdowns
    plots_dir = base_plots_dir / f"ablation_{ablation_type}"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = plots_dir / 'orientation_scores_with_components.csv'
    orientation_df.to_csv(csv_path, index=False)
    print(f"Saved data to {csv_path} ({len(orientation_df)} rows, {len(orientation_df.columns)} columns)")
    
    plotter = BilingualPlotter(orientation_df, plots_dir)
    plotter.plot_language_orientation()
    print(f"Plot saved to {plots_dir / 'orientation.png'}")

print(f"\n{'='*70}")
print(f"All done! Results saved to {base_plots_dir}/ablation_*/orientation_scores_with_components.csv")
print(f"Each CSV has 34 columns: prop, run_id, and 16 metrics per language")
print(f"  Components (6): orientation, structure, aux, participle, determiner, noun")
print(f"  Quality (10): structure_bad, aux_missing, participle_missing, determiners_missing, nouns_missing,")
print(f"                wrong_token_count, wrong_noun_count, wrong_det_count, wrong_aux_count, wrong_part_count")
print(f"Plots saved to {base_plots_dir}/ablation_*/orientation.png")
print(f"{'='*70}")
PYTHON_SCRIPT



