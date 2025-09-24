"""
Main experiment orchestration for language experiments.
"""
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import torch
import numpy as np
from transformers import (
    GPT2LMHeadModel,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from sklearn.model_selection import train_test_split
import wandb

from .metrics import Metrics
from .dataset_manager import DatasetManager
from .model_config import ModelConfig, SlurmConfig
from .plotting import BilingualPlotter


class Experiment:
    """Main experiment orchestrator for language model training and evaluation."""

    def __init__(self,
                 config: ModelConfig,
                 data_manager: DatasetManager,
                 metrics: Metrics,
                 output_dir: Path):
        """Initialize experiment.

        Args:
            config: Model configuration
            data_manager: Dataset manager
            metrics: Metrics calculator
            output_dir: Output directory for results
        """
        self.config = config
        self.data_manager = data_manager
        self.metrics = metrics
        self.output_dir = Path(output_dir)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.output_dir / "experiment.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)8s | %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode="w"),
                logging.StreamHandler()
            ],
        )

        self.logger = logging.getLogger(__name__)

    def run_single(self,
                prop: float,
                run_id: int,
                eval_prop: float = 0.05) -> Path:
        """Run single experiment with given parameters.

        Changes vs. previous version:
        - Create a validation split **from the training set** (size = eval_prop).
        - Remove early stopping entirely.
        - Log actual post-sampling counts for FR/NL.
        - Keep the full test set untouched for final metrics (no downsampling).
        """
        set_seed(run_id)

        # Create run-specific output directory
        run_dir = self.output_dir / f"p{prop*100:04.1f}_run{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb
        wandb.init(
            project="codeswitching-llms",
            name=f"prop{prop*100:04.1f}_run{run_id}",
            config={
                "prop": prop,
                "run_id": run_id,
                "eval_prop": eval_prop,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
        )

        self.logger.info(f"Starting experiment: prop={prop}, run_id={run_id}")

        # Load data
        self.logger.info("Loading data...")
        train_full = pd.read_csv(self.data_manager.data_dir / "train.csv")
        test_df = pd.read_csv(self.data_manager.data_dir / "test.csv")

        # Apply proportion to training data (intentional constant-size budget across props)
        df_fr = train_full[train_full.lang == "fr"]
        df_nl = train_full[train_full.lang == "nl"]

        total_budget = min(len(df_fr), len(df_nl))
        want_fr = int(total_budget * prop)
        want_nl = total_budget - want_fr

        fr_take = df_fr.sample(want_fr, random_state=run_id)
        nl_take = df_nl.sample(want_nl, random_state=run_id)
        train_df = (
            pd.concat([fr_take, nl_take])
            .sample(frac=1, random_state=run_id)
            .reset_index(drop=True)
        )

        self.logger.info(f"Train set (post-sampling): {len(train_df)} rows (FR={len(fr_take)}, NL={len(nl_take)})")
        self.logger.info(f"Full held-out test set: {len(test_df)} rows")

        # === Validation split from TRAIN ===
        # Use eval_prop as the validation fraction of the training data.
        train_df, val_df = train_test_split(
            train_df, test_size=eval_prop, random_state=run_id, shuffle=True
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        wandb.log({
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "fr_train_samples": len(fr_take),
            "nl_train_samples": len(nl_take)
        })
        self.logger.info(f"Final train size: {len(train_df)} | Val size: {len(val_df)}")

        # Keep an untouched copy of the full test set for FINAL metrics after training
        full_test_df_for_final = test_df.copy()

        # Create tokenizer and datasets
        self.logger.info("Creating tokenizer and datasets...")
        tokenizer = self.data_manager.build_tokenizer()
        train_dataset, _ = self.data_manager.create_pytorch_datasets(
            train_df, train_df, tokenizer
        )
        _, val_dataset = self.data_manager.create_pytorch_datasets(
            val_df, val_df, tokenizer
        )
        collator = self.data_manager.create_collator(tokenizer)

        # Create model
        self.logger.info("Creating model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {device}")
        model_config = self.config.to_gpt2_config(len(tokenizer))
        model = GPT2LMHeadModel(model_config).to(device)

        # ===== HF Trainer metrics (unchanged) =====
        def tok_acc(eval_pred):
            logits, labels = eval_pred
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            preds = np.argmax(logits, axis=-1)
            p = preds[:, :-1]
            y = labels[:, 1:]
            mask = y != -100
            return {"tok_acc": float(((p == y) & mask).sum() / mask.sum())}

        def exact_match(eval_pred):
            logits, labels = eval_pred
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            preds = np.argmax(logits, axis=-1)
            p = preds[:, :-1]
            y = labels[:, 1:]
            ok = ((p == y) | (y == -100)).all(axis=1)
            return {"exact_match": float(ok.mean())}

        def compute_metrics(eval_pred):
            out = {}
            out.update(tok_acc(eval_pred))
            out.update(exact_match(eval_pred))
            return out

        # Create trainer (NO early stopping; eval on validation split)
        self.logger.info("Creating trainer...")
        training_args = self.config.to_training_args(run_dir)
        training_args.report_to = ["wandb"]
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=[]  # explicitly no early stopping
        )

        # Train model
        self.logger.info("Starting training...")
        trainer.train()

        # Save model and tokenizer
        final_dir = run_dir / "final"
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)

        # === Final evaluation on FULL held-out test (with ablations) ===
        self.logger.info("Preparing full held-out test data with ablations for final metrics...")
        final_test = full_test_df_for_final.copy()
        final_test["ablation"] = "none"
        ablated_df = self.data_manager.create_ablated_dataset(final_test)
        full_eval_df = ablated_df
        self.logger.info(f"Created ablated final eval set with {len(full_eval_df)} total examples")

        # Run inference on full eval set
        self.logger.info("Running inference on final eval set...")
        predictions = self._run_inference(model, tokenizer, full_eval_df, device)

        # Save predictions
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(run_dir / "ablation_predictions.csv", index=False)

        # Compute and save metrics for each ablation type (on full held-out test)
        for ablation_type in ["none", "subject", "verb", "object"]:
            self.logger.info(f"Computing final metrics for ablation={ablation_type}...")
            type_preds = pred_df[pred_df["ablation"] == ablation_type].to_dict("records")
            metrics = self.metrics.compute_all_metrics(type_preds, ablation_type)
            wandb.log({f"{ablation_type}_{k}": v for k, v in metrics.items()})
            self.metrics.save_metrics(
                metrics, run_dir / f"ablation_{ablation_type}_metrics.json"
            )

        # Save configuration
        self.config.save(run_dir / "config.json")
        wandb.finish()

        self.logger.info(f"Experiment completed. Results saved to {run_dir}")
        return run_dir



    def _run_inference(self,
                    model: GPT2LMHeadModel,
                    tokenizer,
                    test_df: pd.DataFrame,
                    device: str) -> List[Dict[str, Any]]:
        """Run inference on test data.

        Args:
            model: Trained model
            tokenizer: Tokenizer
            test_df: Test dataframe (may or may not have ablation column)
            device: Device to run on

        Returns:
            List of prediction dictionaries
        """
        model.eval()
        predictions = []

        # Set pad token ID to avoid warnings
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        for row in test_df.itertuples():
            prompt = f"<sos> {row.input} <sep>"

            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    num_beams=4,
                    eos_token_id=tokenizer.eos_token_id
                )

            pred = tokenizer.decode(outputs[0], skip_special_tokens=False)
            pred = pred.split("<sep>")[1].replace("<eos>", "").strip()

            pred_dict = {
                'language': row.lang,
                'input': row.input,
                'gold': row.target,
                'prediction': pred
            }
            
            # need to handle ablation calls and normal calls.
            if hasattr(row, 'ablation'):
                pred_dict['ablation'] = row.ablation

            predictions.append(pred_dict)

        return predictions

    def run_sweep(self,
                 props: List[float],
                 runs: List[int],
                 eval_prop: float = 0.05) -> List[Path]:
        """Run parameter sweep across multiple proportions and runs.

        Args:
            props: List of French proportions to test
            runs: List of run IDs (random seeds)
            eval_prop: Proportion of test data to use

        Returns:
            List of output directories
        """
        results = []

        for prop in props:
            for run_id in runs:
                try:
                    result_dir = self.run_single(prop, run_id, eval_prop)
                    results.append(result_dir)
                except Exception as e:
                    self.logger.error(f"Failed prop={prop}, run_id={run_id}: {e}")

        return results

    def submit_to_slurm(self, props: List[float], runs: List[int],
                   slurm_config: SlurmConfig, eval_prop: float = 0.1) -> List[str]:
        """Submit jobs to SLURM."""
        logging.info("Submitting %d jobs to SLURM...", len(props) * len(runs))

        # make the logs directory
        logs_dir = self.output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Create submit script
        script_path = self.output_dir / "submit_jobs.sh"
        with open(script_path, "w") as f:
            # Write SLURM directives
            f.write("#!/bin/bash\n")
            f.write(f"#SBATCH --job-name={slurm_config.job_name}\n")
            f.write(f"#SBATCH --partition={slurm_config.partition}\n")
            f.write(f"#SBATCH --time={slurm_config.time}\n")
            f.write(f"#SBATCH --mem={slurm_config.mem}\n")
            f.write(f"#SBATCH --cpus-per-task={slurm_config.cpus_per_task}\n")
            if slurm_config.gpus:
                f.write(f"#SBATCH --gres=gpu:{slurm_config.gpus}\n")
            f.write(f"#SBATCH --output={slurm_config.output_pattern}\n")
            f.write(f"#SBATCH --error={slurm_config.error_pattern}\n")
            if slurm_config.account:
                f.write(f"#SBATCH --account={slurm_config.account}\n")

            # Calculate array size
            n_tasks = len(props) * len(runs)
            f.write(f"#SBATCH --array=0-{n_tasks-1}\n\n")

            # Load modules and activate environment
            f.write("# Load modules and activate environment\n")
            f.write("module load python/3.10.9-fasrc01\n")
            f.write("source /n/home06/drooryck/envs/codeswitching-py310/bin/activate\n\n")

            # Add project root to PYTHONPATH
            f.write("# Add project root to PYTHONPATH\n")
            f.write("export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:$PYTHONPATH\n\n")

            # Create job mapping
            f.write("# Job mapping\n")
            f.write("case $SLURM_ARRAY_TASK_ID in\n")
            task_id = 0
            for prop in props:
                for run_id in runs:
                    f.write(f"  {task_id}) PROP={prop}; RUN_ID={run_id} ;;\n")
                    task_id += 1
            f.write("  *) echo \"Invalid array index\"; exit 1 ;;\n")
            f.write("esac\n\n")

            # Write run command
            f.write("# Run experiment\n")
            cmd = [
                "python", "-m", "july_aug_exp.src.run_single",
                "--config", str(self.output_dir / "model_config.json"),
                "--output-dir", str(self.output_dir),
                "--prop", "$PROP",
                "--run-id", "$RUN_ID",
                "--eval-prop", str(eval_prop),
                "--data-dir", str(Path("/n/home06/drooryck/codeswitching-llms/july_aug_exp/data")),
                "--lexicon-path", str(self.data_manager.lexicon_path)
            ]
            f.write(" ".join(cmd) + "\n")

        # Make script executable
        script_path.chmod(0o755)

        # Submit job
        result = subprocess.run(["sbatch", str(script_path)],
                              capture_output=True, text=True, check=True)

        # Extract job ID from output
        job_id = result.stdout.strip().split()[-1]
        logging.info("Submitted job array %s", job_id)

        return [job_id]

    # TODO: should this go in data mangement 
    def collect_results(self, run_dirs: List[Path]) -> pd.DataFrame:
        """Collect metrics from multiple experiment runs.

        Args:
            run_dirs: List of experiment output directories

        Returns:
            DataFrame with aggregated results
        """
        results = []

        for run_dir in run_dirs:
            metrics_file = run_dir / "metrics.json"
            if not metrics_file.exists():
                self.logger.warning(f"No metrics found in {run_dir}")
                continue

            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # Extract run parameters from directory name
            dir_name = run_dir.name
            if dir_name.startswith("p") and "_run" in dir_name:
                parts = dir_name.split("_run")
                prop = int(parts[0][1:]) / 100.0
                run_id = int(parts[1])
            else:
                prop = None
                run_id = None

            result = {
                'prop': prop,
                'run_id': run_id,
                'run_dir': str(run_dir),
                **metrics
            }
            results.append(result)

        return pd.DataFrame(results)

    def save_summary(self, results_df: pd.DataFrame,
                    output_path: Optional[Path] = None) -> None:
        """Save experiment summary.

        Args:
            results_df: Results dataframe
            output_path: Path to save summary (defaults to output_dir/summary.csv)
        """
        if output_path is None:
            output_path = self.output_dir / "summary.csv"

        results_df.to_csv(output_path, index=False)
        self.logger.info(f"Summary saved to {output_path}")

    def create_plots(self, results_df: pd.DataFrame) -> None:
        """Create visualization plots from results.

        Args:
            results_df: Results dataframe with metrics computed on model outputs
        """
        plotter = BilingualPlotter(results_df, self.output_dir / "plots")
        plotter.create_all_plots()
