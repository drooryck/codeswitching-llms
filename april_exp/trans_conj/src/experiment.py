"""
Main experiment orchestration for translation + conjugation experiments
(april_exp — boundary task-token format, global French proportion).
"""
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (
    GPT2LMHeadModel,
    Trainer,
    TrainerCallback,
    set_seed,
)

from .dataset_manager import DatasetManager
from .metrics import Metrics
from .model_config import ModelConfig
from .translation import TranslationLevel, TASK_TOKENS

warnings.filterwarnings("ignore", message="Setting `pad_token_id`")


class Experiment:

    def __init__(self, config: ModelConfig, data_manager: DatasetManager,
                 metrics: Metrics, output_dir: Path):
        self.config = config
        self.data_manager = data_manager
        self.metrics = metrics
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self) -> None:
        log_file = self.output_dir / "experiment.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)8s | %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode="w"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Callback: save 10 % of val predictions at every eval step
    # ------------------------------------------------------------------

    class EvalCallback(TrainerCallback):
        """At each eval step: run inference on a val sample, compute
        lexical/syntax scores, log to wandb, and accumulate predictions."""

        def __init__(self, experiment: "Experiment",
                     val_sample_df: pd.DataFrame, tokenizer):
            self.experiment = experiment
            self.val_sample_df = val_sample_df
            self.tokenizer = tokenizer
            self.step_preds: Dict[int, List[str]] = {}
            self.step_metrics: List[Dict[str, Any]] = []

        def on_evaluate(self, args, state, control, model=None, **kwargs):
            if model is None or self.val_sample_df.empty:
                return control
            step = state.global_step
            device = getattr(model, "device", "cpu")
            model.eval()

            preds = self.experiment._run_inference(
                model, self.tokenizer, self.val_sample_df, device)
            self.step_preds[step] = [p["prediction"] for p in preds]

            metrics_obj = self.experiment.metrics
            row = {"step": step}
            for lang in ("fr", "nl"):
                lang_preds = [p["prediction"] for p in preds if p["language"] == lang]
                lex = [metrics_obj.compute_lexical_score(p) for p in lang_preds]
                syn_results = [metrics_obj.compute_syntax_score(p) for p in lang_preds]
                syn_valid = [r["score"] for r in syn_results if r is not None and r["score"] is not None]
                conf = [metrics_obj.compute_conformity_score(p)["score"] for p in lang_preds]
                pos_val_results = [metrics_obj.per_position_pos_validity(p) for p in lang_preds]
                pos_val_valid = [v for v in pos_val_results if v is not None]
                pos_cov = [metrics_obj.pos_coverage_rate(p) for p in lang_preds]

                row[f"{lang}_lexical"] = float(np.mean(lex)) if lex else float("nan")
                row[f"{lang}_syntax"] = float(np.mean(syn_valid)) if syn_valid else float("nan")
                row[f"{lang}_conformity"] = float(np.mean(conf)) if conf else float("nan")
                row[f"{lang}_pos_validity"] = float(np.mean(pos_val_valid)) if pos_val_valid else float("nan")
                row[f"{lang}_pos_coverage"] = float(np.mean(pos_cov)) if pos_cov else float("nan")
                row[f"{lang}_exact_match"] = float(np.mean([
                    p["prediction"] == p["gold"] for p in preds if p["language"] == lang
                ]))

            self.step_metrics.append(row)
            wandb.log({f"eval_sample/{k}": v for k, v in row.items()}, step=step)
            self.experiment.logger.info(
                "Step %d — FR lex=%.3f syn=%.3f conf=%.3f pv=%.3f pc=%.3f em=%.3f"
                " | NL lex=%.3f syn=%.3f conf=%.3f pv=%.3f pc=%.3f em=%.3f",
                step,
                row["fr_lexical"], row["fr_syntax"], row["fr_conformity"],
                row["fr_pos_validity"], row["fr_pos_coverage"], row["fr_exact_match"],
                row["nl_lexical"], row["nl_syntax"], row["nl_conformity"],
                row["nl_pos_validity"], row["nl_pos_coverage"], row["nl_exact_match"])

            model.train()
            return control

        def save(self):
            """Write predictions (wide) and metrics (long) CSVs."""
            out_dir = self.experiment.output_dir

            df = self.val_sample_df[["lang", "input", "target"]].rename(
                columns={"lang": "language", "target": "gold"}).copy()
            for step in sorted(self.step_preds):
                df[f"pred_{step}"] = self.step_preds[step]
            df.to_csv(out_dir / "eval_sample_predictions.csv", index=False)

            pd.DataFrame(self.step_metrics).to_csv(
                out_dir / "eval_sample_metrics.csv", index=False)

            self.experiment.logger.info(
                "Saved eval sample: %d sentences × %d steps (predictions + metrics)",
                len(df), len(self.step_preds))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_single(
        self,
        prop: float,
        run_id: int,
        trans_frac: float = 0.0,
        translation_level: TranslationLevel = TranslationLevel.NONE,
        eval_prop: float = 0.1,
        prediction_save_frac: float = 0.1,
        mask_input: bool = True,
    ) -> Path:
        set_seed(run_id)
        run_dir = self.output_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        group_name = run_dir.name
        if len(run_dir.parents) >= 2 and run_dir.parent.name == "runs":
            group_name = run_dir.parents[1].name

        mask_tag = "mask" if mask_input else "nomask"
        wandb.init(
            project="codeswitching-llms",
            name=f"{mask_tag}_prop{prop*100:04.1f}_tf{trans_frac:.2f}_run{run_id}",
            group=group_name,
            config={"prop": prop, "trans_frac": trans_frac,
                    "translation_level": translation_level.value,
                    "run_id": run_id, "eval_prop": eval_prop,
                    "mask_input": mask_input},
        )

        self.logger.info(
            "Starting: prop=%.3f  trans_frac=%.3f  level=%s  run=%d  mask=%s",
            prop, trans_frac, translation_level.value, run_id, mask_input)

        # ── 0. Generate pairs from lexicon if not already saved ──────
        pairs_path = self.data_manager.data_dir / "train_pairs.csv"
        if not pairs_path.exists():
            self.logger.info("Generating sentence pairs from lexicon...")
            self.data_manager.generate_and_save_pairs(test_size=0.2, seed=0)

        # ── 1. Build training data from pairs ────────────────────────
        train_df, token_stats = self.data_manager.build_training_data(
            prop=prop, trans_frac=trans_frac,
            translation_level=translation_level, seed=run_id)

        ct = train_df.groupby(["task", "lang"]).size().unstack(fill_value=0)
        n_conj = token_stats["n_conj_examples"]
        n_trans = token_stats["n_trans_examples"]
        actual_trans_frac = n_trans / len(train_df) if len(train_df) > 0 else 0

        self.logger.info("Training set: %d examples", len(train_df))
        self.logger.info("  task \\ lang    %8s %8s %8s", "fr", "nl", "total")
        for task in ["conjugate", "translate"]:
            if task in ct.index:
                fr = int(ct.loc[task].get("fr", 0))
                nl = int(ct.loc[task].get("nl", 0))
                self.logger.info("  %-14s %8d %8d %8d", task, fr, nl, fr + nl)
        self.logger.info("  %-14s %8d %8d %8d", "total",
                         int((train_df["lang"] == "fr").sum()),
                         int((train_df["lang"] == "nl").sum()), len(train_df))
        self.logger.info("  prop=%.3f (conjugation FR example fraction)", prop)
        self.logger.info("  trans_frac=%.3f (requested) -> %.3f actual (%d/%d examples)",
                         trans_frac, actual_trans_frac, n_trans, len(train_df))
        self.logger.info("  Exact token counts (input + target, lexicon-matched):")
        self.logger.info("    Conjugation: FR=%d NL=%d total=%d (FR frac=%.4f)",
                         token_stats["conj_fr_tokens"], token_stats["conj_nl_tokens"],
                         token_stats["conj_total_tokens"], token_stats["conj_fr_token_frac"])
        self.logger.info("    Translation: FR=%d NL=%d total=%d (FR frac=%.4f)",
                         token_stats["trans_fr_tokens"], token_stats["trans_nl_tokens"],
                         token_stats["trans_total_tokens"], token_stats["trans_fr_token_frac"])
        self.logger.info("    Overall:     FR=%d NL=%d total=%d (FR frac=%.4f)",
                         token_stats["overall_fr_tokens"], token_stats["overall_nl_tokens"],
                         token_stats["overall_total_tokens"], token_stats["overall_fr_token_frac"])

        # ── 2. Build eval data (conjugation only) ────────────────────
        eval_df = self.data_manager.build_eval_data()

        test_df, val_df = train_test_split(
            eval_df, test_size=eval_prop, random_state=run_id,
            shuffle=True, stratify=eval_df["lang"])
        test_df = test_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        self.logger.info("Eval set: %d (val %d, test %d) — conjugation only, both langs",
                         len(eval_df), len(val_df), len(test_df))

        wandb.run.summary.update({
            "train_total": len(train_df), "train_conj": n_conj,
            "train_trans": n_trans, "actual_trans_frac": actual_trans_frac,
            "conj_fr_token_frac": token_stats["conj_fr_token_frac"],
            "overall_fr_token_frac": token_stats["overall_fr_token_frac"],
            "val_size": len(val_df), "test_size": len(test_df),
            **{f"tokens/{k}": v for k, v in token_stats.items()}})

        # ── 3. Tokenizer & datasets ──────────────────────────────────
        tokenizer = self.data_manager.build_tokenizer()
        train_dataset = self.data_manager.create_pytorch_dataset(
            train_df, tokenizer, mask_input=mask_input)
        val_dataset = self.data_manager.create_pytorch_dataset(val_df, tokenizer)
        collator = self.data_manager.create_collator(tokenizer)

        # ── 4. Model ─────────────────────────────────────────────────
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info("Device: %s", device)
        model = GPT2LMHeadModel(
            self.config.to_gpt2_config(len(tokenizer))).to(device)

        # ── 5. Trainer setup ─────────────────────────────────────────
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            preds = np.argmax(logits, axis=-1)
            p, y = preds[:, :-1], labels[:, 1:]
            mask = y != -100
            tok_acc = float(((p == y) & mask).sum() / mask.sum())
            exact = float(((p == y) | (y == -100)).all(axis=1).mean())
            return {"tok_acc": tok_acc, "exact_match": exact}

        val_sample = val_df.sample(
            frac=prediction_save_frac, random_state=42
        ).reset_index(drop=True)
        pred_callback = self.EvalCallback(
            experiment=self, val_sample_df=val_sample, tokenizer=tokenizer)

        training_args = self.config.to_training_args(run_dir)
        training_args.report_to = ["wandb"]

        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=train_dataset, eval_dataset=val_dataset,
            data_collator=collator, compute_metrics=compute_metrics,
            callbacks=[pred_callback])

        # Trainer default uses RandomSampler — no override needed.

        # ── 6. Train ─────────────────────────────────────────────────
        self.logger.info("Starting training...")
        trainer.train()

        pred_callback.save()

        final_dir = run_dir / "final"
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)

        # ── 7. Final predictions on held-out test set ────────────────
        self.logger.info("Final inference on test set...")
        predictions = self._run_inference(model, tokenizer, test_df, device)
        pd.DataFrame(predictions).to_csv(
            run_dir / "test_predictions.csv", index=False)

        self.config.save(run_dir / "config.json")
        wandb.finish()
        self.logger.info("Done. Results in %s", run_dir)
        return run_dir

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(self, model, tokenizer, df, device) -> List[Dict]:
        """Generate predictions using the new boundary-task-token format.

        Prompt: ``<sos> input <task_token>``
        The model generates everything after the task token.
        """
        model.eval()
        predictions: List[Dict[str, Any]] = []
        batch_size = 1024

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            prompts = [
                f"<sos> {r.input} {TASK_TOKENS[r.task]}"
                for r in batch.itertuples()
            ]
            with torch.no_grad():
                inputs = tokenizer(prompts, return_tensors="pt",
                                   padding=True).to(device)
                outputs = model.generate(
                    **inputs, max_new_tokens=20, do_sample=False,
                    num_beams=1, eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id)

            for text, row in zip(
                tokenizer.batch_decode(outputs, skip_special_tokens=False),
                batch.itertuples(),
            ):
                task_tok = TASK_TOKENS[row.task]
                pred = (text.split(task_tok)[1]
                        .replace("<eos>", "").replace("<pad>", "").strip()
                        if task_tok in text else "")
                predictions.append({
                    "language": row.lang, "input": row.input,
                    "gold": row.target, "prediction": pred})
        return predictions
