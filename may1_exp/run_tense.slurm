#!/bin/bash
#SBATCH -J tense-${P}
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=seas_gpu
#SBATCH --account=dam_lab
#SBATCH --mem=32G
#SBATCH -t 10:00:00
#SBATCH -o logs/%x-%j.out

# Activate your venv
source ~/circuits_languages_2/venv39/bin/activate

# Run Python script with French proportion passed via env var
echo "Running with p = $P"
python train_tense_model.py --prop $P
