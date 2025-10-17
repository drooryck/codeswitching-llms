#!/usr/bin/env python
"""
Script to submit 20k step language experiments to SLURM.
"""
from pathlib import Path

from src.metrics import Metrics
from src.dataset_manager import DatasetManager
from src.model_config import ModelConfig, SlurmConfig
from src.experiment import Experiment


def submit_sweep():
    """Submit parameter sweep jobs to SLURM."""
    print("Preparing SLURM jobs for 20k step sweep...")

    # Setup paths - use relative paths from jul_1 directory
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "data"
    output_dir = root_dir / "results/sweep_20k"

    # Load default configs
    config = ModelConfig.load(root_dir / "configs/default_model.json")
    slurm_config = SlurmConfig.load(root_dir / "configs/slurm_default.json")

    # Override/set additional parameters not in default config
    config = config.copy(
        # Hardware specific
        fp16=True,

        # Logging/saving
        save_total_limit=5,  # Keep last 5 checkpoints
        logging_steps=1000,

        # Metrics
        metric_for_best_model="exact_match",  # Focus on exact match accuracy
        load_best_model_at_end=True
    )

    # Override only experiment-specific SLURM settings
    slurm_config.job_name = "lang_20k"  # Specific to this 20k steps experiment
    slurm_config.output_pattern = str(output_dir / "logs/slurm_%A_%a.out")
    slurm_config.error_pattern = str(output_dir / "logs/slurm_%A_%a.err")

    # Setup components
    data_manager = DatasetManager(str(data_dir), config=config)
    metrics = Metrics(data_dir / "lexicon_new.json")

    # Create experiment
    experiment = Experiment(config, data_manager, metrics, output_dir)

    # Save config for SLURM jobs
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "model_config.json")
    slurm_config.save(output_dir / "slurm_config.json")

    # Submit jobs
    props = [0.2]  # French proportions
    runs = [48]      # Two runs per proportion

    print(f"Submitting SLURM jobs with:")
    print(f"  Proportions: {props}")
    print(f"  Run IDs: {runs}")
    print(f"  Steps: {config.max_steps}")
    print(f"  Eval frequency: {config.eval_steps}")
    print(f"  Early stopping patience: {config.early_stopping_patience}")
    print(f"  Time limit: {slurm_config.time}")
    print(f"  Memory: {slurm_config.mem}")

    job_ids = experiment.submit_to_slurm(
        props=props,
        runs=runs,
        slurm_config=slurm_config,
        eval_prop=1.0  # Use full test set for final evaluation
    )

    print(f"\nSubmitted {len(props) * len(runs)} jobs")
    print(f"Job array ID: {job_ids[0]}")
    print(f"\nResults will be saved to: {output_dir}")
    print(f"Monitor progress in: {output_dir}/logs/")


if __name__ == "__main__":
    submit_sweep()
