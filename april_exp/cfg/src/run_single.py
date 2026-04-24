"""
CLI entry point for running a single CFG experiment
(april_exp — boundary task-token format, EN/NL).
"""
import argparse
import json
from pathlib import Path

from .dataset_manager import DatasetManager
from .metrics import Metrics
from .model_config import ModelConfig
from .experiment import Experiment
from .translation import TranslationLevel


def main():
    parser = argparse.ArgumentParser(
        description="Run a single CFG experiment (april_exp)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Dir for train_pairs.csv / test_pairs.csv")
    parser.add_argument("--lexicon-path", type=str, required=True)
    parser.add_argument("--prop", type=float, required=True,
                        help="EN fraction in conjugation task (0-1)")
    parser.add_argument("--trans-frac", type=float, default=0.0)
    parser.add_argument("--translation-level", type=str, default="none",
                        choices=[e.value for e in TranslationLevel])
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--eval-prop", type=float, default=0.1)
    parser.add_argument("--prediction-save-frac", type=float, default=0.1)
    parser.add_argument("--no-mask", action="store_true",
                        help="Disable loss masking over input tokens")
    args = parser.parse_args()

    config = ModelConfig(**json.load(open(args.config)))
    dm = DatasetManager(args.data_dir, config, lexicon_path=args.lexicon_path)
    metrics = Metrics(args.lexicon_path)

    exp = Experiment(config=config, data_manager=dm, metrics=metrics,
                     output_dir=Path(args.output_dir))
    exp.run_single(
        prop=args.prop, run_id=args.run_id,
        trans_frac=args.trans_frac,
        translation_level=TranslationLevel(args.translation_level),
        eval_prop=args.eval_prop,
        prediction_save_frac=args.prediction_save_frac,
        mask_input=not args.no_mask)


if __name__ == "__main__":
    main()
