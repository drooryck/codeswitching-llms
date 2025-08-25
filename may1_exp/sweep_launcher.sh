#!/bin/bash
# sweep_launcher.sh

for i in $(seq 0 10); do
    P=$(echo "$i / 10" | bc -l)
    sbatch --export=ALL,P=$P run_tense.slurm
done
