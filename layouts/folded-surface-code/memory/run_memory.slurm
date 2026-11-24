#!/bin/bash
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J qec-memory
#SBATCH -o results/raw/slurm-%x-%j.out
# Usage: sbatch --export=ALL,NOISE=SD6,PS="0.001 0.002",DISTANCES="3 5 7 9 11",MAX_SHOTS=1e7 run_memory.slurm
# Re-submitting the same command resumes from results/raw/memory-$NOISE.csv.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
PY=${QEC_PYTHON:-python}   # an environment with `pip install -e ".[memory]"`
timeout 1650 "$PY" run_memory.py --noise "$NOISE" --distances $DISTANCES --ps $PS \
    --max-shots "${MAX_SHOTS:-1e7}" --max-errors "${MAX_ERRORS:-1000}" --workers "${WORKERS:-250}" \
    --out "results/raw/memory-${NOISE}.csv" --quiet || echo "stopped by time limit; resubmit to resume"
