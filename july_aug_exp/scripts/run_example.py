#!/usr/bin/env python
"""
Example script showing how to use the language experiment framework.
"""
import sys
from pathlib import Path
import json
import pandas as pd

# Add src to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.metrics import Metrics
from src.dataset_manager import DatasetManager
from src.model_config import ModelConfig, SlurmConfig
from src.experiment import Experiment
from src.plotting import BilingualPlotter


def run_local_example():
    """Run a simple local experiment."""
    print("Running local example experiment...")

    # Setup paths
    data_dir = root_dir.parent   # Use existing data
    output_dir = root_dir / "results/example"

    # Load configuration
    config = ModelConfig(
        n_layer=2,
        max_steps=1000,  # Short run for testing
        eval_steps=500
    )

    # Setup components
    data_manager = DatasetManager(data_dir)
    data_manager.load_lexicon()
    data_manager.load_vocab()

    metrics = Metrics(data_dir / "lexicon.json")

    # Create experiment
    experiment = Experiment(
        config=config,
        data_manager=data_manager,
        metrics=metrics,
        output_dir=output_dir
    )

    # Run single experiment
    result_dir = experiment.run_single(prop=0.5, run_id=42, eval_prop=0.01)
    print(f"Experiment completed. Results in: {result_dir}")


def run_sweep_example(props=None, runs=None, eval_prop=0.01):
    """Run a parameter sweep.

    Args:
        props (list[float], optional): List of proportions to sweep over. Defaults to [0.2, 0.5, 0.8].
        runs (list[int], optional): List of run IDs to use. Defaults to [1, 2].
        eval_prop (float, optional): Proportion of test set to use for evaluation. Defaults to 0.01.
    """
    print("Running parameter sweep...")

    # Setup
    data_dir = root_dir / "data"  # Use the same data directory as SLURM example
    output_dir = root_dir / "results/sweep"  # Changed from ../jul_1/results/sweep

    config = ModelConfig(
        n_layer=2,
        n_head=2,
        n_embd=128,
        max_steps=500,  # Short run for testing
        eval_steps=200,
        batch_size=16
    )

    data_manager = DatasetManager(
        data_dir,
        tokenizer_config={
            "bos_token": "<sos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "sep_token": "<sep>",
            "unk_token": "<unk>",
            "padding_side": "right"
        }
    )
    data_manager.load_lexicon()
    data_manager.load_vocab()

    metrics = Metrics(data_dir / "lexicon_new.json")  # Use new lexicon

    experiment = Experiment(config, data_manager, metrics, output_dir)

    # Use provided parameters or defaults
    props = props or [0.2, 0.5, 0.8]
    runs = runs or [1, 2]

    # Run sweep
    result_dirs = experiment.run_sweep(props, runs, eval_prop=eval_prop)

    # Collect and analyze results
    results_df = experiment.collect_results(result_dirs)
    print("\nAvailable columns in results:")
    print(results_df.columns)
    print("\nFirst few rows of results:")
    print(results_df.head())

    experiment.save_summary(results_df)
    experiment.create_plots(results_df)

    print(f"Sweep completed. {len(result_dirs)} experiments run.")
    print(f"Summary saved to: {output_dir}/summary.csv")


def prepare_and_run_slurm_example(prop=0.5, run_id=1):
    """Prepare and submit SLURM job."""
    print("Preparing SLURM example...")

    # Setup
    data_dir = root_dir / "data"  # Use our new data
    output_dir = root_dir / "results/slurm_test"
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Simple config for testing
    config = ModelConfig(
        n_layer=2,
        n_head=2,
        n_embd=128,
        max_steps=1000,  # Short run for testing
        eval_steps=200,
        batch_size=16
    )

    slurm_config = SlurmConfig(
        partition="seas_gpu",
        account="dam_lab",
        time="00:30:00",  # 30 minutes should be enough for test
        mem="16G",
        cpus_per_task=4,
        gpus=1,
        output_pattern=str(logs_dir / "slurm_%A_%a.out"),
        error_pattern=str(logs_dir / "slurm_%A_%a.err")
    )

    data_manager = DatasetManager(
        data_dir,
        tokenizer_config={
            "bos_token": "<sos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "sep_token": "<sep>",
            "unk_token": "<unk>",
            "padding_side": "right"
        }
    )
    metrics = Metrics(data_dir / "lexicon_new.json")  # Use new lexicon

    experiment = Experiment(config, data_manager, metrics, output_dir)

    # Save configurations
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "model_config.json")
    slurm_config.save(output_dir / "slurm_config.json")

    print(f"SLURM configuration prepared in: {output_dir}")
    print("Files created:")
    print(f"  - {output_dir}/model_config.json")
    print(f"  - {output_dir}/slurm_config.json")

    # Submit job to SLURM
    print("\nSubmitting job to SLURM...")
    job_ids = experiment.submit_to_slurm(
        props=[prop],  # Single proportion
        runs=[run_id],  # Single run
        slurm_config=slurm_config,
        eval_prop=0.1  # Use 10% of test set for evaluation
    )

    print(f"\nJob submitted successfully!")
    print(f"Job IDs: {job_ids}")
    print(f"\nTo monitor your job:")
    print(f"  squeue -u $USER")
    print(f"  tail -f {output_dir}/logs/slurm_*.out")


def plot_sweep_results():
    """Plot results from existing sweep directory using BilingualPlotter."""
    print("Plotting sweep results...")

    sweep_dir = root_dir / "results/sweep"
    plotter = BilingualPlotter.create_plotter_from_run_metrics_dir(sweep_dir)

    # Print summary statistics
    plotter.print_metrics_summary()

    # Create all plots
    plotter.create_all_plots()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run parameter sweep experiment")
    parser.add_argument("--props", type=float, nargs="+", default=[0.2, 0.5, 0.8],
                       help="List of proportions to sweep over (default: [0.2, 0.5, 0.8])")
    parser.add_argument("--runs", type=int, nargs="+", default=[1, 2],
                       help="List of run IDs to use (default: [1, 2])")
    parser.add_argument("--eval-prop", type=float, default=0.01,
                       help="Proportion of test set to use for evaluation (default: 0.01)")
    parser.add_argument("--plot-only", action="store_true",
                       help="Only plot existing results without running new experiments")

    args = parser.parse_args()

    if args.plot_only:
        plot_sweep_results()
    else:
        # Update run_sweep_example to use command line arguments
        print(f"Running sweep with:")
        print(f"  Proportions: {args.props}")
        print(f"  Run IDs: {args.runs}")
        print(f"  Eval proportion: {args.eval_prop}")

        run_sweep_example(props=args.props, runs=args.runs, eval_prop=args.eval_prop)
