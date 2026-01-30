# Essential SLURM Commands

```bash
# Check your job queue
squeue -u $USER

# Watch queue in real-time (refresh every 2s)
watch -n 2 'squeue -u $USER'

# Cancel a specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# Monitor job output in real-time
tail -f slurm_<job_id>_*.out

# Monitor job errors in real-time
tail -f slurm_<job_id>_*.err

# Quick interactive GPU session (2 hours)
srun --pty -p seas_gpu --gres=gpu:1 --mem=16G -t 0-02:00 /bin/bash

# Check GPU/node availability
sinfo -p seas_gpu

