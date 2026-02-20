"""
Main experiment orchestration for language experiments.
"""
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import torch
import numpy as np

warnings.filterwarnings("ignore", message="Setting `pad_token_id`")
from transformers import (
    GPT2LMHeadModel,
    Trainer,
    TrainerCallback,
    set_seed
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SequentialSampler
import wandb

from .metrics import Metrics
from .dataset_manager import DatasetManager
from .model_config import ModelConfig


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


    WANDB_RICH_KEYS = frozenset({
        "nl_follows_fr", "nl_follows_nl", "nl_follows_either",
        "fr_follows_fr", "fr_follows_nl", "fr_follows_either",
        "nl_avg_nl", "nl_avg_fr", "fr_avg_nl", "fr_avg_fr",
    })

    class RichMetricsCallback(TrainerCallback):
        """Compute per-language eval loss and selected structure metrics after each evaluation."""

        def __init__(self, experiment: 'Experiment', val_df: pd.DataFrame, tokenizer, collator):
            self.experiment = experiment
            self.val_df = val_df
            self.tokenizer = tokenizer
            self.collator = collator

        def _compute_loss_for_df(self, model, df, device):
            """Forward-pass loss over a dataframe subset."""
            if df.empty:
                return float("nan")
            dataset = self.experiment.data_manager.create_pytorch_datasets(df, df, self.tokenizer)[0]
            loader = DataLoader(dataset, batch_size=64, collate_fn=self.collator)
            total_loss, n = 0.0, 0
            with torch.no_grad():
                for batch in loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    total_loss += model(**batch).loss.item() * batch["input_ids"].size(0)
                    n += batch["input_ids"].size(0)
            return total_loss / n if n > 0 else float("nan")

        def on_evaluate(self, args, state, control, model=None, **kwargs):
            if model is None or self.val_df.empty:
                return control

            logger = self.experiment.logger
            step = state.global_step
            logger.info(f"Rich metrics callback (step={step})...")

            device = model.device if hasattr(model, "device") else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            model.eval()

            wandb_metrics = {}

            # --- Per-language eval loss ---
            fr_df = self.val_df[self.val_df["lang"] == "fr"]
            nl_df = self.val_df[self.val_df["lang"] == "nl"]

            loss_fr = self._compute_loss_for_df(model, fr_df, device)
            loss_nl = self._compute_loss_for_df(model, nl_df, device)
            loss_all = self._compute_loss_for_df(model, self.val_df, device)

            wandb_metrics["eval/loss_fr"] = loss_fr
            wandb_metrics["eval/loss_nl"] = loss_nl
            wandb_metrics["eval/loss"] = loss_all

            logger.info(f"  eval loss: overall={loss_all:.4f}  fr={loss_fr:.4f}  nl={loss_nl:.4f}")

            # --- Structure / language metrics (ablation=none only) ---
            self.val_df["ablation"] = "none"
            predictions = self.experiment._run_inference(
                model=model, tokenizer=self.tokenizer,
                test_df=self.val_df, device=device,
            )
            all_metrics = self.experiment.metrics.compute_all_metrics(predictions, ablation_type="none")

            for k, v in all_metrics.items():
                if k in Experiment.WANDB_RICH_KEYS:
                    wandb_metrics[f"eval/{k}"] = v

            wandb.log(wandb_metrics, step=step)
            logger.info("  Logged %d metrics to wandb", len(wandb_metrics))

            model.train()
            return control

    def run_single(self,
                prop: float,
                run_id: int,
                eval_prop: float = 0.05) -> Path:
        """Run single experiment with given parameters.
        Uses data_prep_debug logic: balanced dataset, stratified validation, shuffled training.
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

        # --- 1) Make validation come from the test set (NOT the train set) ---
        # Split test into (val_df, test_df_eval) stratified by lang
        test_df_eval, val_df = train_test_split(
            test_df_full,
            test_size=eval_prop,
            random_state=run_id,
            shuffle=True,
            stratify=test_df_full["lang"],
        )
        test_df_eval = test_df_eval.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        self.logger.info(
            f"Validation set (from TEST): {len(val_df)} total | "
            f"FR: {(val_df['lang'] == 'fr').sum()} | NL: {(val_df['lang'] == 'nl').sum()}"
        )
        self.logger.info(
            f"Final eval test set (post-val split): {len(test_df_eval)} total | "
            f"FR: {(test_df_eval['lang'] == 'fr').sum()} | NL: {(test_df_eval['lang'] == 'nl').sum()}"
        )

        # --- 2) Train set balancing by taking FIRST min_count from each language ---
        train_fr = train_df_full[train_df_full["lang"] == "fr"].reset_index(drop=True)
        train_nl = train_df_full[train_df_full["lang"] == "nl"].reset_index(drop=True)

        fr_count = len(train_fr)
        nl_count = len(train_nl)
        min_count = min(fr_count, nl_count)

        train_fr_bal = train_fr.head(min_count)
        train_nl_bal = train_nl.head(min_count)

        self.logger.info(
            f"Balanced train pool (head): {2*min_count} total | FR: {len(train_fr_bal)} | NL: {len(train_nl_bal)}"
        )

        # --- 3) Compute budget + take FIRST want_fr / want_nl from each ---
        total_budget = min_count
        want_fr = int(total_budget * prop)
        want_nl = total_budget - want_fr

        fr_take = train_fr_bal.head(want_fr)
        nl_take = train_nl_bal.head(want_nl)

        # --- 4) Shuffle final training set ---
        train_df = (
            pd.concat([fr_take, nl_take], ignore_index=True)
            .sample(frac=1, random_state=run_id)
            .reset_index(drop=True)
        )

        self.logger.info("Final training set (after proportion selection):")
        self.logger.info(
            f"  Total: {len(train_df)} | FR: {len(fr_take)} ({len(fr_take)/len(train_df)*100:.1f}%) | "
            f"NL: {len(nl_take)} ({len(nl_take)/len(train_df)*100:.1f}%)"
        )
        self.logger.info(f"  Target proportion: FR={prop*100:.1f}%, NL={100-prop*100:.1f}%")

        # ===== END NEW DATA PREP LOGIC =====

        wandb.run.summary.update({
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df_eval),
            "fr_train_samples": int(len(fr_take)),
            "nl_train_samples": int(len(nl_take)),
            "fr_train_pct": len(fr_take) / len(train_df) * 100,
            "nl_train_pct": len(nl_take) / len(train_df) * 100,
        })

        self.logger.info("Preparing held-out test data with ablations...")
        test_df_eval["ablation"] = "none"
        full_eval_df = self.data_manager.create_ablated_dataset(test_df_eval)
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
            collator=collator,
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
            self.metrics.save_metrics(
                metrics, run_dir / f"ablation_{ablation_type}_metrics.json"
            )

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

