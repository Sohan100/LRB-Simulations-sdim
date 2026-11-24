#!/bin/bash
# Run the full correction-decoder study on one Perlmutter CPU node.
#
# Usage (from the worktree root), STAGE = baseline | variants | all:
#   salloc -N 1 -C cpu -q debug -t 00:30:00 \
#       srun -n 1 -c 256 bash scripts/correction/run_all_on_node.sh STAGE
#
# Every task writes its own output file and is skipped when that file
# exists, so re-running after a time-out resumes where it stopped.
set -euo pipefail

ROOT="${LRB_WORKTREE_ROOT:-$(pwd)}"
RUN="${LRB_BASELINE_RUN:-/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000}"
CACHE="${ROOT}/.cache_correction"
WORKERS="${LRB_WORKERS:-128}"
STAGE="${1:-all}"

module load python/3.11
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${ROOT}"

if [[ "${STAGE}" == "baseline" || "${STAGE}" == "all" ]]; then
echo "[$(date +%T)] baseline prepare"
python scripts/correction/run_decoder_study.py prepare \
    --run-folder "${RUN}" --cache "${CACHE}/structures" --workers "${WORKERS}"

echo "[$(date +%T)] baseline run"
python scripts/correction/run_decoder_study.py run \
    --run-folder "${RUN}" --cache "${CACHE}/structures" \
    --out "${CACHE}/results_baseline" --workers "${WORKERS}" \
    --sampled-probs 1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2 --shots 100000

fi

if [[ "${STAGE}" == "variants" || "${STAGE}" == "all" ]]; then
echo "[$(date +%T)] variant prepare"
python scripts/correction/run_variant_study.py prepare \
    --cache "${CACHE}/variant_structures" --workers "${WORKERS}"

echo "[$(date +%T)] variant run"
python scripts/correction/run_variant_study.py run \
    --cache "${CACHE}/variant_structures" --out "${CACHE}/results_variants" \
    --workers "${WORKERS}"

fi

if [[ "${STAGE}" == "dense" ]]; then
echo "[$(date +%T)] dense exact prepare"
python scripts/correction/run_variant_study.py prepare --depths dense \
    --variants base,chk,par_term --cache "${CACHE}/dense_structures" --workers "${WORKERS}"
echo "[$(date +%T)] dense exact run"
python scripts/correction/run_variant_study.py run --depths dense \
    --variants base,chk,par_term --cache "${CACHE}/dense_structures" \
    --out "${CACHE}/results_dense" --sampled-probs "" \
    --exact-probs 2e-4,5e-4,1e-3,2e-3,5e-3 --workers "${WORKERS}"
fi

echo "[$(date +%T)] done ${STAGE}"
