"""
Main experiment orchestration for language experiments.
"""
import json
import logging
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import torch
import numpy as np
import time

# Suppress common warnings
warnings.filterwarnings("ignore", message="Setting `pad_token_id`")
from transformers import (
    GPT2LMHeadModel,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    set_seed
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SequentialSampler
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

        self.output_dir.mkdir(parents=True, exist_ok=True)

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


    class RichMetricsCallback(TrainerCallback):
        """Compute and log rich validation metrics including ablation metrics after each evaluation."""

        def __init__(self, experiment: 'Experiment', val_df: pd.DataFrame, ablated_val_df: pd.DataFrame, tokenizer):
            self.experiment = experiment
            self.val_df = val_df
            self.ablated_val_df = ablated_val_df
            self.tokenizer = tokenizer

        def on_evaluate(self, args, state, control, model=None, **kwargs):  # noqa: D401
            if model is None or self.val_df.empty:
                return control

            logger = self.experiment.logger
            logger.info(f"Running rich metrics callback on validation set (step={state.global_step})...")

            device = model.device if hasattr(model, 'device') else (
                'cuda' if torch.cuda.is_available() else 'cpu'
            )

            # Run inference on ablated validation set
            predictions = self.experiment._run_inference(
                model=model,
                tokenizer=self.tokenizer,
                test_df=self.ablated_val_df,
                device=device,
            )

            pred_df = pd.DataFrame(predictions)
            
            # Collect all metrics to log
            all_wandb_metrics = {}
            structure_metrics_log = []
            
            # Compute metrics for each ablation type
            for ablation_type in ["none", "subject", "verb", "object"]:
                type_preds = pred_df[pred_df["ablation"] == ablation_type].to_dict("records")
                if not type_preds:
                    continue
                    
                metrics_dict = self.experiment.metrics.compute_all_metrics(
                    type_preds,
                    ablation_type=ablation_type
                )
                
                # Extract structure conformity metrics
                fr_follows_either = metrics_dict.get('fr_follows_either', 0.0)
                nl_follows_either = metrics_dict.get('nl_follows_either', 0.0)
                overall_follows = metrics_dict.get('overall_follows_structure', 0.0)
                
                # Log to wandb with prefix
                for k, v in metrics_dict.items():
                    all_wandb_metrics[f"eval_abl_{ablation_type}_{k}"] = v
                
                # Collect for formatted log message
                structure_metrics_log.append(
                    f"  {ablation_type:8s}: FR={fr_follows_either:.3f}, NL={nl_follows_either:.3f}, Overall={overall_follows:.3f}"
                )
            
            # Log all metrics to wandb
            wandb.log(all_wandb_metrics, step=state.global_step)
            
            # Log formatted structure conformity metrics to logger
            logger.info(f"[step={state.global_step}] Ablation Metrics - Structure Conformity:")
            logger.info("\n".join(structure_metrics_log))
            
            logger.info(
                "Logged %d ablation metrics to wandb (step=%s)",
                len(all_wandb_metrics),
                state.global_step,
            )

            model.train()

            return control

    def run_single(self,
                prop: float,
                run_id: int,
                eval_prop: float = 0.05) -> Path:
        """Run single experiment with given parameters.
        High level:

        """
        set_seed(run_id)

        # Ensure run-specific output directory exists (expect it to be supplied by caller)
        run_dir = self.output_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        # Determine wandb group name from output path (default to last directory component)
        group_name = run_dir.name
        if len(run_dir.parents) >= 2 and run_dir.parent.name == "runs":
            group_name = run_dir.parents[1].name

        # Initialize wandb
        wandb.init(
            project="codeswitching-llms",
            name=f"prop{prop*100:04.1f}_run{run_id}",
            group=group_name,
            config={
                "prop": prop,
                "run_id": run_id,
                "eval_prop": eval_prop,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
        )

        self.logger.info(f"Starting experiment: prop={prop}, run_id={run_id}")

        self.logger.info("Loading data...")
        train_df_full = pd.read_csv(self.data_manager.data_dir / "train.csv")
        test_df_full = pd.read_csv(self.data_manager.data_dir / "test.csv")
        
        # Log initial dataset sizes and language counts
        self.logger.info(f"Initial train set: {len(train_df_full)} total | "
                        f"FR: {(train_df_full['lang'] == 'fr').sum()} | "
                        f"NL: {(train_df_full['lang'] == 'nl').sum()}")
        self.logger.info(f"Test set: {len(test_df_full)} total | "
                        f"FR: {(test_df_full['lang'] == 'fr').sum()} | "
                        f"NL: {(test_df_full['lang'] == 'nl').sum()}")

        rng = np.random.RandomState(run_id)
        train_df_full["global_key"] = rng.random(len(train_df_full))
        train_df_full["orig_idx"] = np.arange(len(train_df_full))

        # for any run, the eval rows will be the same. not stratified by language though.
        eval_mask = train_df_full["global_key"] < eval_prop
        val_df = train_df_full[eval_mask].reset_index(drop=True)
        train_df_full = train_df_full[~eval_mask].reset_index(drop=True)
        
        # Log validation split
        self.logger.info(f"Validation set: {len(val_df)} total | "
                        f"FR: {(val_df['lang'] == 'fr').sum()} | "
                        f"NL: {(val_df['lang'] == 'nl').sum()}")
        self.logger.info(f"Remaining train set after val split: {len(train_df_full)} total | "
                        f"FR: {(train_df_full['lang'] == 'fr').sum()} | "
                        f"NL: {(train_df_full['lang'] == 'nl').sum()}")

        train_df_full["lang_rank"] = (
            train_df_full.groupby("lang")["global_key"]
                        .rank(method="first")
                        .astype(int)
        )

        total_budget = min(
            (train_df_full.lang == "fr").sum(),
            (train_df_full.lang == "nl").sum()
        )

        want_fr = int(total_budget * prop)
        want_nl = total_budget - want_fr

        fr_take = train_df_full[
            (train_df_full.lang == "fr") & (train_df_full.lang_rank <= want_fr)
        ]
        nl_take = train_df_full[
            (train_df_full.lang == "nl") & (train_df_full.lang_rank <= want_nl)
        ]

        train_df = (
            pd.concat([fr_take, nl_take], ignore_index=True)
            .sort_values(["global_key","orig_idx"], kind="mergesort")
            .reset_index(drop=True)
        )
        
        # Log final training set composition
        self.logger.info(f"Final training set (after proportion selection):")
        self.logger.info(f"  Total: {len(train_df)} | FR: {len(fr_take)} ({len(fr_take)/len(train_df)*100:.1f}%) | "
                        f"NL: {len(nl_take)} ({len(nl_take)/len(train_df)*100:.1f}%)")
        self.logger.info(f"  Target proportion: FR={prop*100:.1f}%, NL={100-prop*100:.1f}%")

        wandb.log({
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df_full),
            "fr_train_samples": len(fr_take),
            "nl_train_samples": len(nl_take),
            "fr_train_pct": len(fr_take) / len(train_df) * 100,
            "nl_train_pct": len(nl_take) / len(train_df) * 100,
            "fr_val_samples": (val_df['lang'] == 'fr').sum(),
            "nl_val_samples": (val_df['lang'] == 'nl').sum(),
            "fr_test_samples": (test_df_full['lang'] == 'fr').sum(),
            "nl_test_samples": (test_df_full['lang'] == 'nl').sum()
        })

        self.logger.info("Preparing held-out test data with ablations...")
        test_df_full["ablation"] = "none"
        full_eval_df = self.data_manager.create_ablated_dataset(test_df_full)
        self.logger.info(
            "Created ablated final eval set with %d total examples",
            len(full_eval_df)
        )

        self.logger.info("Creating tokenizer and datasets...")
        tokenizer = self.data_manager.build_tokenizer()
        train_dataset, val_dataset = self.data_manager.create_pytorch_datasets(
            train_df, val_df, tokenizer
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
        rich_metrics_callback = self.RichMetricsCallback(
            experiment=self,
            val_df=val_df.copy(),
            tokenizer=tokenizer,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            compute_metrics=compute_metrics,
            #callbacks=[rich_metrics_callback],
        )

        # monkey patch the trainer to use the SequentialSampler
        trainer.get_train_dataloader = lambda: DataLoader(
            train_dataset,
            batch_size=training_args.train_batch_size,
            sampler=SequentialSampler(train_dataset),
            collate_fn=collator,
            drop_last=training_args.dataloader_drop_last,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )

        # Train model
        self.logger.info("Starting training...")
        trainer.train()

        # Save model and tokenizer
        final_dir = run_dir / "final"
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)

        self.logger.info("Running inference on final eval set...")
        predictions = self._run_inference(model, tokenizer, full_eval_df, device)

        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(run_dir / "ablation_predictions.csv", index=False)

        for ablation_type in ["none", "subject", "verb", "object"]:
            self.logger.info(f"Computing final metrics for ablation={ablation_type}...")
            type_preds = pred_df[pred_df["ablation"] == ablation_type].to_dict("records")
            metrics = self.metrics.compute_all_metrics(type_preds, ablation_type)
            wandb.log({f"{ablation_type}_{k}": v for k, v in metrics.items()})
            self.metrics.save_metrics(
                metrics, run_dir / f"ablation_{ablation_type}_metrics.json"
            )

        self.config.save(run_dir / "config.json")
        wandb.finish()

        self.logger.info(f"Experiment completed. Results saved to {run_dir}")
        return run_dir

    def run_single_super_debug(self,
                prop: float,
                run_id: int,
                eval_prop: float = 0.05) -> Path:
        """Verbose version of run_single with timing + dataset dumps for debugging."""
        set_seed(run_id)

        run_dir = self.output_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        debug_dir = run_dir / "logs"
        debug_dir.mkdir(parents=True, exist_ok=True)

        def log_split(name: str, df: pd.DataFrame) -> None:
            fr = int((df["lang"] == "fr").sum()) if "lang" in df else 0
            nl = int((df["lang"] == "nl").sum()) if "lang" in df else 0
            self.logger.info(f"[debug] {name}: total={len(df)} | FR={fr} | NL={nl}")

        group_name = run_dir.name
        if len(run_dir.parents) >= 2 and run_dir.parent.name == "runs":
            group_name = run_dir.parents[1].name

        wandb.init(
            project="codeswitching-llms",
            name=f"prop{prop*100:04.1f}_run{run_id}",
            group=group_name,
            config={
                "prop": prop,
                "run_id": run_id,
                "eval_prop": eval_prop,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
        )

        timings: Dict[str, float] = {}
        total_start = time.perf_counter()
        try:
            self.logger.info(f"[debug] Starting run_single_super_debug prop={prop}, run_id={run_id}")

            t = time.perf_counter()
            train_df_full = pd.read_csv(self.data_manager.data_dir / "train.csv")
            test_df_full = pd.read_csv(self.data_manager.data_dir / "test.csv")
            timings["load_data"] = time.perf_counter() - t
            self.logger.info(f"[timing] load_data: {timings['load_data']:.2f}s")
            log_split("initial_train", train_df_full)
            log_split("initial_test", test_df_full)

            rng = np.random.RandomState(run_id)
            
            # Balance initial dataset FIRST - keep min(FR_count, NL_count) from each language
            t = time.perf_counter()
            fr_count = (train_df_full['lang'] == 'fr').sum()
            nl_count = (train_df_full['lang'] == 'nl').sum()
            min_count = min(fr_count, nl_count)
            fr_df = train_df_full[train_df_full['lang'] == 'fr'].sample(n=min_count, random_state=run_id)
            nl_df = train_df_full[train_df_full['lang'] == 'nl'].sample(n=min_count, random_state=run_id)
            train_df_full = pd.concat([fr_df, nl_df], ignore_index=True).reset_index(drop=True)
            timings["balance_initial"] = time.perf_counter() - t
            self.logger.info(f"[timing] balance_initial: {timings['balance_initial']:.2f}s")
            log_split("balanced_train", train_df_full)

            # Stratified validation split (on balanced data)
            t = time.perf_counter()
            train_df_full, val_df = train_test_split(
                train_df_full,
                test_size=eval_prop,
                random_state=run_id,
                shuffle=True,
                stratify=train_df_full['lang']
            )
            train_df_full = train_df_full.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
            timings["create_val_split"] = time.perf_counter() - t
            self.logger.info(f"[timing] create_val_split: {timings['create_val_split']:.2f}s")
            log_split("validation_split", val_df)
            log_split("post_val_train_pool", train_df_full)

            # Apply global_key to training data only (after validation split)
            train_df_full["global_key"] = rng.random(len(train_df_full))
            train_df_full["orig_idx"] = np.arange(len(train_df_full))
            
            train_df_full["lang_rank"] = (
                train_df_full.groupby("lang")["global_key"]
                            .rank(method="first")
                            .astype(int)
            )

            total_budget = min(
                (train_df_full.lang == "fr").sum(),
                (train_df_full.lang == "nl").sum()
            )

            want_fr = int(total_budget * prop)
            want_nl = total_budget - want_fr

            fr_take = train_df_full[
                (train_df_full.lang == "fr") & (train_df_full.lang_rank <= want_fr)
            ]
            nl_take = train_df_full[
                (train_df_full.lang == "nl") & (train_df_full.lang_rank <= want_nl)
            ]

            train_df = (
                pd.concat([fr_take, nl_take], ignore_index=True)
                .sample(frac=1, random_state=run_id)
                .reset_index(drop=True)
            )
            log_split("final_train", train_df)

            wandb.log({
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df_full),
                "fr_train_samples": len(fr_take),
                "nl_train_samples": len(nl_take),
                "fr_train_pct": len(fr_take) / len(train_df) * 100 if len(train_df) else 0,
                "nl_train_pct": len(nl_take) / len(train_df) * 100 if len(train_df) else 0,
                "fr_val_samples": (val_df['lang'] == 'fr').sum(),
                "nl_val_samples": (val_df['lang'] == 'nl').sum(),
                "fr_test_samples": (test_df_full['lang'] == 'fr').sum(),
                "nl_test_samples": (test_df_full['lang'] == 'nl').sum()
            })

            # Save the exact datasets Trainer / evaluation will see
            train_csv = debug_dir / "train_dataset.csv"
            val_csv = debug_dir / "val_dataset.csv"
            test_csv = debug_dir / "test_dataset.csv"
            train_df.to_csv(train_csv, index=False)
            val_df.to_csv(val_csv, index=False)
            test_df_full.to_csv(test_csv, index=False)
            self.logger.info(f"[debug] Saved dataset snapshots to {debug_dir}")

            t = time.perf_counter()
            test_df_full["ablation"] = "none"
            full_eval_df = self.data_manager.create_ablated_dataset(test_df_full)
            timings["prepare_ablations"] = time.perf_counter() - t
            self.logger.info(f"[timing] prepare_ablations: {timings['prepare_ablations']:.2f}s")
            log_split("ablated_eval_full", full_eval_df)
            full_eval_df.to_csv(debug_dir / "full_eval_dataset.csv", index=False)

            t = time.perf_counter()
            tokenizer = self.data_manager.build_tokenizer()
            
            # Create ablated validation set for training-time evaluation
            val_df_ablated = val_df.copy()
            val_df_ablated["ablation"] = "none"
            val_df_ablated_full = self.data_manager.create_ablated_dataset(val_df_ablated)
            self.logger.info(f"Created ablated validation set with {len(val_df_ablated_full)} examples for training-time evaluation")
            
            train_dataset, val_dataset = self.data_manager.create_pytorch_datasets(
                train_df, val_df, tokenizer
            )
            collator = self.data_manager.create_collator(tokenizer)
            timings["build_tokenizer_and_datasets"] = time.perf_counter() - t
            self.logger.info(f"[timing] build_tokenizer_and_datasets: {timings['build_tokenizer_and_datasets']:.2f}s")

            self.logger.info("Creating model (super debug)...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {device}")
            model_config = self.config.to_gpt2_config(len(tokenizer))
            model = GPT2LMHeadModel(model_config).to(device)

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

            self.logger.info("Creating trainer (super debug)...")
            t = time.perf_counter()
            training_args = self.config.to_training_args(run_dir)
            training_args.report_to = ["wandb"]
            
            # Create callback for ablation metrics during training
            rich_metrics_callback = self.RichMetricsCallback(
                experiment=self,
                val_df=val_df,
                ablated_val_df=val_df_ablated_full,
                tokenizer=tokenizer
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=collator,
                compute_metrics=compute_metrics,
                callbacks=[rich_metrics_callback],
            )
            timings["create_trainer"] = time.perf_counter() - t
            self.logger.info(f"[timing] create_trainer: {timings['create_trainer']:.2f}s")

            # Use default RandomSampler (shuffling) instead of SequentialSampler

            self.logger.info("Starting training (super debug)...")
            t = time.perf_counter()
            trainer.train()
            timings["training"] = time.perf_counter() - t
            self.logger.info(f"[timing] training: {timings['training']:.2f}s")

            final_dir = run_dir / "final"
            trainer.save_model(final_dir)
            tokenizer.save_pretrained(final_dir)

            self.logger.info("Running inference on final eval set (super debug)...")
            t = time.perf_counter()
            predictions = self._run_inference(model, tokenizer, full_eval_df, device)
            timings["inference"] = time.perf_counter() - t
            self.logger.info(f"[timing] inference: {timings['inference']:.2f}s")

            pred_df = pd.DataFrame(predictions)
            pred_df.to_csv(run_dir / "ablation_predictions.csv", index=False)

            t = time.perf_counter()
            for ablation_type in ["none", "subject", "verb", "object"]:
                self.logger.info(f"Computing final metrics for ablation={ablation_type}...")
                type_preds = pred_df[pred_df["ablation"] == ablation_type].to_dict("records")
                metrics = self.metrics.compute_all_metrics(type_preds, ablation_type)
                wandb.log({f"{ablation_type}_{k}": v for k, v in metrics.items()})
                self.metrics.save_metrics(
                    metrics, run_dir / f"ablation_{ablation_type}_metrics.json"
                )
            timings["compute_metrics"] = time.perf_counter() - t
            self.logger.info(f"[timing] compute_metrics: {timings['compute_metrics']:.2f}s")

            self.config.save(run_dir / "config.json")
            total_time = time.perf_counter() - total_start
            timings["total"] = total_time
            self.logger.info(f"[timing] total_runtime: {total_time/60:.2f} minutes")
            with open(debug_dir / "timings.json", "w") as f:
                json.dump({k: round(v, 3) for k, v in timings.items()}, f, indent=2)

            wandb.finish()
            self.logger.info(f"Super debug experiment completed. Results saved to {run_dir}")
            return run_dir
        finally:
            if wandb.run is not None:
                wandb.finish()

    def run_single_nov21(self,
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

        # Use output_dir directly (script already provides run-specific path)
        run_dir = self.output_dir
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
        
        # Create ablated validation set for training-time evaluation
        val_df_ablated = val_df.copy()
        val_df_ablated["ablation"] = "none"
        val_df_ablated_full = self.data_manager.create_ablated_dataset(val_df_ablated)
        self.logger.info(f"Created ablated validation set with {len(val_df_ablated_full)} examples for training-time evaluation")
        
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
        
        # Create callback for ablation metrics during training
        rich_metrics_callback = self.RichMetricsCallback(
            experiment=self,
            val_df=val_df,
            ablated_val_df=val_df_ablated_full,
            tokenizer=tokenizer
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=[rich_metrics_callback]
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
        batch_size = 1024
        
        # Ensure pad token is set for batched generation
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Process in batches
        for i in range(0, len(test_df), batch_size):
            batch_df = test_df.iloc[i:i+batch_size]
            prompts = [f"<sos> {row.input} <sep>" for row in batch_df.itertuples()]
            
            with torch.no_grad():
                inputs = tokenizer(prompts, return_tensors="pt").to(device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    num_beams=4,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Decode batch
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            
            for j, (pred_text, row) in enumerate(zip(decoded, batch_df.itertuples())):
                pred = pred_text.split("<sep>")[1].replace("<eos>", "").strip()
                
                pred_dict = {
                    'language': row.lang,
                    'input': row.input,
                    'gold': row.target,
                    'prediction': pred
                }
                
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
                "python", "-m", "july_aug_sept_exp.src.run_single",
                "--config", str(self.output_dir / "model_config.json"),
                "--output-dir", str(self.output_dir),
                "--prop", "$PROP",
                "--run-id", "$RUN_ID",
                "--eval-prop", str(eval_prop),
                "--data-dir", str(self.data_manager.data_dir),
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
