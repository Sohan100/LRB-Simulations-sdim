#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --job-name=qscal
#SBATCH --output=logs/slurm-%j.out
# Runs every line of the commands file given as $1 (each a 10_sample_lrb.py
# argument list) one after another on one node; 10_sample_lrb.py skips p files
# that already exist, so a resubmission resumes.
# Submit from this folder (studies/qudit-scaling) after `mkdir -p logs`.
cd "$SLURM_SUBMIT_DIR"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1
PY=../../.venv-sdim140/bin/python
while read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  echo "=== $(date -Is) $line"
  $PY 10_sample_lrb.py $line --workers ${WORKERS:-250}
done < "$1"
echo "=== $(date -Is) DONE"
