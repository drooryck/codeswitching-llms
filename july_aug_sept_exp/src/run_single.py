"""
Command-line interface for running single experiments.
"""
import argparse
from pathlib import Path
import json

from .dataset_manager import DatasetManager
from .metrics import Metrics
from .model_config import ModelConfig
from .experiment import Experiment


def main():
    """Main entry point for running single experiments."""
    parser = argparse.ArgumentParser(description="Run single language experiment")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to model configuration file")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--prop", type=float, required=True,
                       help="Proportion of French data (0.0 to 1.0)")
    parser.add_argument("--run-id", type=int, required=True,
                       help="Random seed for reproducibility")
    parser.add_argument("--eval-prop", type=float, default=0.05,
                       help="Proportion of test data to use for evaluation")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory containing data files")
    parser.add_argument("--lexicon-path", type=str, required=True,
                    help="Path to lexicon JSON file")

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    config = ModelConfig(**config_dict)

    # Setup components
    data_manager = DatasetManager(args.data_dir, config, lexicon_path=args.lexicon_path)
    metrics = Metrics(args.lexicon_path)

    # Create and run experiment
    experiment = Experiment(
        config=config,
        data_manager=data_manager,
        metrics=metrics,
        output_dir=Path(args.output_dir)
    )

    experiment.run_single(args.prop, args.run_id, args.eval_prop)


if __name__ == "__main__":
    main()
