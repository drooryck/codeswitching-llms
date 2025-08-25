"""
Model configuration for language experiments.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import GPT2Config, TrainingArguments


@dataclass
class ModelConfig:
    """Configuration for model architecture and training parameters."""

    # Architecture parameters
    vocab_size: Optional[int] = None  # Auto-computed from tokenizer
    n_embd: int = 128
    n_layer: int = 2
    n_head: int = 2
    n_positions: int = 1024

    # Tokenizer parameters
    tokenizer_config: Dict[str, str] = field(default_factory=lambda: {
        "bos_token": "<sos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "sep_token": "<sep>",
        "unk_token": "<unk>",
        "padding_side": "right"
    })

    # Training parameters
    max_steps: int = 20_000
    batch_size: int = 16
    grad_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 500
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 1e-2

    # Evaluation parameters
    eval_strategy: str = "steps"
    eval_steps: int = 1000
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 5
    logging_steps: int = 2000
    early_stopping_patience: int = 5

    # Hardware parameters
    fp16: bool = True
    dataloader_num_workers: int = 0

    # Metrics
    metric_for_best_model: str = "tok_acc"
    greater_is_better: bool = True
    load_best_model_at_end: bool = True

    # Reporting
    report_to: str = "none"
    run_name: Optional[str] = None

    # Additional parameters
    additional_config: Dict[str, Any] = field(default_factory=dict)

    def to_gpt2_config(self, vocab_size: int) -> GPT2Config:
        """Convert to HuggingFace GPT2Config.

        Args:
            vocab_size: Vocabulary size from tokenizer

        Returns:
            GPT2Config object
        """
        return GPT2Config(
            vocab_size=vocab_size,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_positions=self.n_positions,
            **self.additional_config
        )

    def to_training_args(self, output_dir: Path) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments.

        Args:
            output_dir: Directory for training outputs

        Returns:
            TrainingArguments object
        """
        return TrainingArguments(
            output_dir=str(output_dir),
            max_steps=self.max_steps,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accumulation_steps,
            fp16=self.fp16,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            lr_scheduler_type=self.lr_scheduler_type,
            weight_decay=self.weight_decay,
            eval_strategy=self.eval_strategy,
            eval_steps=self.eval_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            logging_steps=self.logging_steps,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            report_to=self.report_to,
            run_name=self.run_name,
            dataloader_num_workers=self.dataloader_num_workers,
        )

    def save(self, path: Path) -> None:
        """Save configuration to JSON file.

        Args:
            path: Path to save configuration
        """
        import json

        config_dict = {
            'vocab_size': self.vocab_size,
            'n_embd': self.n_embd,
            'n_layer': self.n_layer,
            'n_head': self.n_head,
            'n_positions': self.n_positions,
            'max_steps': self.max_steps,
            'batch_size': self.batch_size,
            'grad_accumulation_steps': self.grad_accumulation_steps,
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'lr_scheduler_type': self.lr_scheduler_type,
            'weight_decay': self.weight_decay,
            'eval_strategy': self.eval_strategy,
            'eval_steps': self.eval_steps,
            'save_strategy': self.save_strategy,
            'save_steps': self.save_steps,
            'save_total_limit': self.save_total_limit,
            'logging_steps': self.logging_steps,
            'early_stopping_patience': self.early_stopping_patience,
            'fp16': self.fp16,
            'dataloader_num_workers': self.dataloader_num_workers,
            'metric_for_best_model': self.metric_for_best_model,
            'greater_is_better': self.greater_is_better,
            'load_best_model_at_end': self.load_best_model_at_end,
            'report_to': self.report_to,
            'run_name': self.run_name,
            'additional_config': self.additional_config,
        }

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'ModelConfig':
        """Load configuration from JSON file.

        Args:
            path: Path to configuration file

        Returns:
            ModelConfig instance
        """
        import json

        with open(path, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def copy(self, **kwargs) -> 'ModelConfig':
        """Create a copy with optional parameter overrides.

        Args:
            **kwargs: Parameters to override

        Returns:
            New ModelConfig instance
        """
        import copy

        new_config = copy.deepcopy(self)
        for key, value in kwargs.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

        return new_config


@dataclass
class SlurmConfig:
    """Configuration for SLURM job submission."""

    partition: str = "gpu"
    account: Optional[str] = None
    time: str = "02:00:00"
    mem: str = "16G"
    cpus_per_task: int = 4
    gpus: int = 1
    job_name: str = "lang_exp"
    output_pattern: str = "logs/slurm_%A_%a.out"
    error_pattern: str = "logs/slurm_%A_%a.err"

    # Additional SLURM parameters
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def to_sbatch_script(self, command: str, array_spec: Optional[str] = None) -> str:
        """Generate SLURM batch script.

        Args:
            command: Command to run
            array_spec: Array specification (e.g., "0-9" for array job)

        Returns:
            SLURM batch script as string
        """
        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --time={self.time}",
            f"#SBATCH --mem={self.mem}",
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
            f"#SBATCH --gres=gpu:{self.gpus}",
            f"#SBATCH --output={self.output_pattern}",
            f"#SBATCH --error={self.error_pattern}",
        ]

        if self.account:
            script_lines.append(f"#SBATCH --account={self.account}")

        if array_spec:
            script_lines.append(f"#SBATCH --array={array_spec}")

        # Add additional parameters
        for key, value in self.additional_params.items():
            script_lines.append(f"#SBATCH --{key}={value}")

        script_lines.extend([
            "",
            "# Load modules and activate environment",
            "module load python/3.10.9-fasrc01",
            "source /n/home06/drooryck/circuits_languages_2/venv39/bin/activate",
            "",
            "# Run command",
            command
        ])

        return "\n".join(script_lines)

    def save(self, path: Path) -> None:
        """Save SLURM configuration to JSON file.

        Args:
            path: Path to save configuration
        """
        import json

        config_dict = {
            'partition': self.partition,
            'account': self.account,
            'time': self.time,
            'mem': self.mem,
            'cpus_per_task': self.cpus_per_task,
            'gpus': self.gpus,
            'job_name': self.job_name,
            'output_pattern': self.output_pattern,
            'error_pattern': self.error_pattern,
            'additional_params': self.additional_params,
        }

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'SlurmConfig':
        """Load SLURM configuration from JSON file.

        Args:
            path: Path to configuration file

        Returns:
            SlurmConfig instance
        """
        import json

        with open(path, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)
