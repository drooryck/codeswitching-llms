# French-Dutch Language Model Experiments

This codebase implements experiments for training language models on French-Dutch translation tasks, with a focus on verb tense transformations.

## Core Components

### Experiment Class
The main experiment orchestrator that handles:
- Single experiment runs with configurable French/Dutch proportions
- Parameter sweeps across multiple proportions and random seeds
- SLURM job submission for distributed experiments
- Result collection and visualization

### Dataset Management
The `DatasetManager` class implements sophisticated data preparation:
- Loads and processes lexicons with verb conjugations and noun forms
- Splits data ensuring subject-verb pairs don't overlap between train/test
- Builds custom tokenizers for the bilingual task
- Creates PyTorch datasets with proper padding and masking

### Metrics
Evaluation metrics include:
- Token-level accuracy
- Exact match accuracy
- Language-specific performance tracking

### Model Configuration
Configurable model parameters including:
- Architecture (GPT-2 base)
- Training hyperparameters
- Early stopping criteria
- SLURM job specifications

## Usage

### Single Experiment
```python
from src.experiment import Experiment
from src.dataset_manager import DatasetManager
from src.metrics import Metrics
from src.model_config import ModelConfig

# Initialize components
config = ModelConfig(...)
data_manager = DatasetManager("data/", tokenizer_config={...})
metrics = Metrics()

# Create and run experiment
experiment = Experiment(config, data_manager, metrics, "output/")
experiment.run_single(prop=0.5, run_id=42)
```

### Parameter Sweep
```python
# Run multiple experiments
props = [0.0, 0.25, 0.5, 0.75, 1.0]
runs = list(range(5))
results = experiment.run_sweep(props, runs)

# Collect and visualize results
df = experiment.collect_results(results)
experiment.create_plots(df)
```

### SLURM Submission
```python
from src.slurm_config import SlurmConfig

# Configure SLURM parameters
slurm_config = SlurmConfig(
    partition="gpu",
    time="24:00:00",
    mem="32G",
    gpus=1
)

# Submit jobs
experiment.submit_to_slurm(props, runs, slurm_config)
```

## Data Format

### Lexicon Structure
```json
{
  "NOUNS": {
    "fr": {
      "professeur": {"sgl": "professeur", "pl": "professeurs"}
    },
    "nl": {
      "leraar": {"sgl": "leraar", "pl": "leraren"}
    }
  },
  "VERBS": {
    "fr": [["voit", "vu"], ["mange", "mangé"]],
    "nl": [["ziet", "gezien"], ["eet", "gegeten"]]
  }
}
```

### Dataset Format
The training and test datasets are CSV files with columns:
- `input`: Source sentence (present tense)
- `target`: Target sentence (past tense)
- `lang`: Language code ("fr" or "nl")
- `plural`: Boolean indicating plural subject
- `subj`: Subject noun
- `obj`: Object noun
- `verb`: Present tense verb form

## Key Features

1. **Sophisticated Data Splitting**: Training and test sets are split to ensure subject-verb pairs don't overlap, forcing the model to generalize to unseen combinations.

2. **Bilingual Evaluation**: Separate tracking of performance on French and Dutch sentences to analyze cross-lingual learning.

3. **Flexible Training**: Support for varying proportions of French vs Dutch data to study transfer learning effects.

4. **HuggingFace Integration**: Uses HuggingFace's Trainer for efficient training with built-in early stopping and evaluation.

5. **Distributed Experiments**: SLURM support for running large parameter sweeps on compute clusters.

6. **Rich Visualizations**: Comprehensive plotting capabilities for analyzing results:
   - Exact match accuracy by language proportion
   - Token-level accuracy trends
   - BLEU score comparisons
   - Error bars and confidence intervals
   - Consistent styling and high-resolution outputs

## Directory Structure
```
jul_1/
├── src/
│   ├── __init__.py
│   ├── experiment.py      # Main experiment orchestration
│   ├── dataset_manager.py # Data preparation and handling
│   ├── metrics.py         # Evaluation metrics
│   ├── model_config.py    # Model configuration
│   ├── plotting.py        # Visualization utilities
│   └── slurm_config.py    # SLURM job configuration
├── data/
│   ├── lexicon.json      # Vocabulary and conjugations
│   ├── train.csv         # Training data
│   └── test.csv          # Test data
└── README.md
```

## Configuration

### Model Configuration

```json
{
  "n_embd": 128,
  "n_layer": 2,
  "n_head": 2,
  "max_steps": 20000,
  "batch_size": 16,
  "learning_rate": 0.0002,
  "early_stopping_patience": 5
}
```

### SLURM Configuration

```json
{
  "partition": "seas_gpu",
  "account": "dam_lab",
  "time": "04:00:00",
  "mem": "16G",
  "cpus_per_task": 4,
  "gpus": 1
}
```

## Command Line Interface

Run individual experiments:

```bash
python src/run_experiment.py --prop 0.5 --run-id 42 --data-dir data --output-dir results
```

## Examples

See `scripts/run_example.py` for complete examples:

```bash
# Run local experiment
python scripts/run_example.py local

# Run parameter sweep
python scripts/run_example.py sweep

# Prepare SLURM submission
python scripts/run_example.py slurm
```

## Metrics

The framework computes several language-specific metrics:

- **Token Language Fraction**: Proportion of tokens appearing in French/Dutch lexicon
- **Exact Match**: Whether prediction exactly matches target
- **Participle Final**: Whether participle appears after second noun (tense-specific)

## Benefits

1. **Modularity**: Each component has clear responsibilities
2. **Reusability**: Mix and match components for different experiments
3. **Reproducibility**: Comprehensive logging and configuration saving
4. **Scalability**: Built-in SLURM support for cluster computing
5. **Extensibility**: Easy to add new metrics, model types, or data formats

## Migration from Previous Code

This framework consolidates and improves upon the experimental code from `may31_exp/`, `may6_exp/`, etc.:

- Replaces monolithic `train_tense.py` with modular classes
- Standardizes configuration and output formats
- Adds robust error handling and logging
- Provides consistent CLI and programmatic interfaces

## Future Extensions

The framework is designed to easily support:

- Different model architectures (beyond GPT-2)
- Additional languages and metrics
- Different training objectives
- Integration with experiment tracking tools (W&B, MLflow)
- Automated hyperparameter optimization
