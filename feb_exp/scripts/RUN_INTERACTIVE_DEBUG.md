# Running `run_single` interactively (no Slurm) with debug on non-plurality data

## 1. Get an interactive GPU node

On FASRC (Harvard) with Slurm, request an interactive session with a GPU:

```bash
srun --pty -p seas_gpu --gres=gpu:1 --mem=32G --cpus-per-task=4 --time=2:00:00 bash
```

Or without GPU (CPU-only, slower but fine for sanity-checking batches/tokenization):

```bash
srun --pty -p seas_gpu --mem=32G --cpus-per-task=4 --time=2:00:00 bash
```

Once you get a shell on the node, continue below.

## 2. Load environment and set paths

```bash
cd /n/home06/drooryck/codeswitching-llms
module load python/3.10.9-fasrc01
source /n/home06/drooryck/envs/codeswitching-py310/bin/activate
export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:${PYTHONPATH:-}
```

## 3. Non-plurality data directories

Use one of these as `--data-dir` for **non-plurality** (no plurality mixing) data:

- **Legacy:**  
  `DATA_DIR=/n/home06/drooryck/codeswitching-llms/feb_exp/data/data_no_plurality_mixing`

- **Feb23 balanced (v1 or v2 no plurality):**  
  - v1: `feb_exp/data/balanced_data_feb23/version1_no_plurality_mixing`  
  - v2: `feb_exp/data/balanced_data_feb23/version2_no_plurality_mixing`

## 4. Run `run_single` with `--debug` (no job submission)

Single run with **debug output** (train/eval batches, pre/post tokenization, tokenizer vocab, per-example loss):

```bash
CONFIG=feb_exp/configs/default_model.json
OUTPUT_DIR=/n/home06/drooryck/codeswitching-llms/feb_exp/results/debug_run
DATA_DIR=/n/home06/drooryck/codeswitching-llms/feb_exp/data/balanced_data_feb23/version1_no_plurality_mixing
LEXICON=/n/home06/drooryck/codeswitching-llms/feb_exp/data/lexicon_sep22.json

mkdir -p "$OUTPUT_DIR"

# Single line (safe to copy-paste; backslash continuation can break when pasting):
python -m feb_exp.src.run_single --config "$CONFIG" --output-dir "$OUTPUT_DIR" --data-dir "$DATA_DIR" --lexicon-path "$LEXICON" --prop 0.5 --run-id 1 --eval-prop 0.05 --debug
```

Training will run in the foreground so you can watch logs and sanity-check.

## 5. What `--debug` prints

- **Tokenizer vocabulary:** `id -> token` for every token the model sees (and vocab size).
- **Pre-tokenization:** First 5 train and 5 val rows as `(lang, input, target)`.
- **Post-tokenization:** First 3 train and 3 val examples as tokenized sequences (decoded back to text, including labels where applicable).
- **First train batch:** Full first training batch decoded (as seen by the Trainer).
- **First eval batch:** Full first eval batch decoded.
- **Loss for some examples:** Per-example loss on the first train batch, plus batch mean (before any training).

Use this to verify data mix, tokenization, and that loss is reasonable before relying on full runs.
