#!/bin/bash
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J qec-gates
#SBATCH -o results/raw/slurm-%x-%j.out
# Memory with a fold-transversal logical gate between every two rounds (folded code), plus the
# plain Z memory with sector detectors as the baseline. Resubmit to resume.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
PY=${QEC_PYTHON:-python}   # an environment with `pip install -e ".[memory]"`
COMMON="--codes folded --noise $NOISE --distances 3 5 7 9 --ps $PS --max-shots 1e8 --max-errors 2000 --workers 250 --quiet"
timeout 540 "$PY" run_memory.py $COMMON --gate none --bases Z --detectors sector --out "results/raw/gates-${NOISE}-none.csv" || true
timeout 540 "$PY" run_memory.py $COMMON --gate H --bases X Z --out "results/raw/gates-${NOISE}-H.csv" || true
timeout 540 "$PY" run_memory.py $COMMON --gate S --bases Z --out "results/raw/gates-${NOISE}-S.csv" || true
