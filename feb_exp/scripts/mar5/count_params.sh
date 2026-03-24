#!/bin/bash
#SBATCH --job-name=count_params
#SBATCH --partition=seas_compute
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/mar5/logs/count_params_%j.out
#SBATCH --error=/n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/mar5/logs/count_params_%j.err

module load python/3.10.9-fasrc01
source /n/home06/drooryck/envs/codeswitching-py310/bin/activate
export PYTHONPATH=/n/home06/drooryck/codeswitching-llms:${PYTHONPATH:-}

python -c "
import json
from pathlib import Path
from feb_exp.src.dataset_manager import DatasetManager
from feb_exp.src.model_config import ModelConfig
from transformers import GPT2LMHeadModel

config = ModelConfig(**json.load(open('feb_exp/scripts/mar5/model_config.json')))
data_dir = 'feb_exp/data/balanced_data_feb23/version1_plurality_mixing'
lexicon = 'feb_exp/data/lexicon_sep22.json'
dm = DatasetManager(data_dir, config, lexicon_path=lexicon)
tokenizer = dm.build_tokenizer()

vocab_size = len(tokenizer)
model_config = config.to_gpt2_config(vocab_size)
model = GPT2LMHeadModel(model_config)

total_params = sum(p.numel() for p in model.parameters())
print(f'vocab_size: {vocab_size}')
print(f'model_config: n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}')
print(f'total_params: {total_params:,}')
"
