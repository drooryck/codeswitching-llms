#!/usr/bin/env python3
"""Recompute metrics from existing predictions with parallel processing."""
from pathlib import Path
import pandas as pd
import sys
import multiprocessing as mp
import logging
import time
from datetime import datetime

# Add the project to path
sys.path.insert(0, '/n/home06/drooryck/codeswitching-llms')

from july_aug_sept_exp.src.metrics import Metrics

def recompute_metrics_for_run(args):
    """Recompute metrics for a single run from existing predictions."""
    run_dir, lexicon_path, process_id = args
    
    # Setup logging for this process
    logger = logging.getLogger(f"Process-{process_id}")
    logger.setLevel(logging.INFO)
    
    # Create formatter with timestamp and process ID
    formatter = logging.Formatter(
        f'%(asctime)s [PID-{process_id}] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Add handler if not already added
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.info(f"Starting {run_dir.name}")
    
    predictions_file = run_dir / "ablation_predictions.csv"
    if not predictions_file.exists():
        logger.warning(f"No predictions file found in {run_dir}")
        return False
    
    try:
        # Load predictions
        logger.info(f"Loading predictions for {run_dir.name}")
        pred_df = pd.read_csv(predictions_file)
        
        # Initialize metrics with fixed code
        logger.info(f"Initializing metrics for {run_dir.name}")
        metrics = Metrics(lexicon_path)
        
        # Recompute metrics for each ablation type
        for ablation_type in ["none", "subject", "verb", "object"]:
            logger.info(f"  Computing {ablation_type} metrics for {run_dir.name}")
            type_preds = pred_df[pred_df["ablation"] == ablation_type].to_dict("records")
            
            if type_preds:  # Only if we have predictions for this ablation type
                new_metrics = metrics.compute_all_metrics(type_preds, ablation_type)
                metrics.save_metrics(new_metrics, run_dir / f"ablation_{ablation_type}_metrics.json")
                logger.info(f"    ✓ Saved {ablation_type} metrics for {run_dir.name}")
            else:
                logger.info(f"    No {ablation_type} predictions found for {run_dir.name}")
        
        logger.info(f"✓ Completed {run_dir.name}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error processing {run_dir.name}: {str(e)}")
        return False

if __name__ == "__main__":
    # Setup main logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [MAIN] %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stdout
    )
    main_logger = logging.getLogger("main")
    
    main_logger.info("Starting parallel metrics recomputation...")
    
    # Paths
    lexicon_path = Path("/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/data/lexicon_sep22.json")
    results_dir = Path("/n/home06/drooryck/codeswitching-llms/july_aug_sept_exp/results/sep24.3")
    
    main_logger.info(f"Lexicon path: {lexicon_path}")
    main_logger.info(f"Results dir: {results_dir}")
    
    if not lexicon_path.exists():
        main_logger.error(f"ERROR: Lexicon file not found: {lexicon_path}")
        sys.exit(1)
    
    if not results_dir.exists():
        main_logger.error(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)
    
    run_dirs = sorted(results_dir.glob("p*_run*"))
    main_logger.info(f"Found {len(run_dirs)} runs to process")
    
    if len(run_dirs) == 0:
        main_logger.warning("No run directories found!")
        sys.exit(0)
    
    # Determine number of processes (use all available CPUs or limit to 8)
    num_processes = min(mp.cpu_count(), 8, len(run_dirs))
    main_logger.info(f"Using {num_processes} parallel processes")
    
    # Prepare arguments for each process
    process_args = [
        (run_dir, lexicon_path, i) 
        for i, run_dir in enumerate(run_dirs)
    ]
    
    start_time = time.time()
    
    # Run in parallel
    try:
        with mp.Pool(processes=num_processes) as pool:
            main_logger.info("Starting parallel processing...")
            results = pool.map(recompute_metrics_for_run, process_args)
            
        # Count results
        successful = sum(1 for r in results if r)
        failed = len(results) - successful
        
        end_time = time.time()
        duration = end_time - start_time
        
        main_logger.info(f"✓ Parallel recomputation completed!")
        main_logger.info(f"  Successful: {successful}/{len(run_dirs)} runs")
        main_logger.info(f"  Failed: {failed}/{len(run_dirs)} runs")
        main_logger.info(f"  Total time: {duration:.1f} seconds")
        main_logger.info(f"  Average per run: {duration/len(run_dirs):.1f} seconds")
        
    except KeyboardInterrupt:
        main_logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        main_logger.error(f"Error in parallel processing: {str(e)}")
        sys.exit(1)