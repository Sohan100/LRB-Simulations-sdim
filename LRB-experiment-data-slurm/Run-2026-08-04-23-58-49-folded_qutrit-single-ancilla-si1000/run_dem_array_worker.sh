#!/bin/bash
#SBATCH --job-name=lrb_dem_single_si1000_array
#SBATCH -C cpu
#SBATCH -q preempt
#SBATCH -t 47:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --requeue

set -euo pipefail
umask 027

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Sbatch executes a copy of this script from the Slurm spool directory, so
# BASH_SOURCE cannot locate the run folder there; LRB_RUN_DIR must name it.
RUN_DIR="${LRB_RUN_DIR:-${SCRIPT_DIR}}"
if [[ ! -f "${RUN_DIR}/code_name.txt" ]]; then
    echo "Run directory '${RUN_DIR}' has no code_name.txt. Submit with" \
        "LRB_RUN_DIR set to the run folder that holds this worker."
    exit 1
fi
PROJECT_ROOT="$(cd "${RUN_DIR}/../.." && pwd)"
RUN_NAME="${RUN_NAME_OVERRIDE:-$(basename "${RUN_DIR}")}"
PROBABILITY_INDEX="${SLURM_ARRAY_TASK_ID:?Submit this worker with --array.}"

if [[ ! "${PROBABILITY_INDEX}" =~ ^([0-9]|1[0-4])$ ]]; then
    echo "Probability index must be between 0 and 14; got '${PROBABILITY_INDEX}'."
    exit 1
fi

if [[ "$(tr -d '[:space:]' < "${RUN_DIR}/code_name.txt")" != "folded_qutrit" ]]; then
    echo "Expected folded_qutrit metadata in ${RUN_DIR}/code_name.txt."
    exit 1
fi

if [[ "$(tr -d '[:space:]' < "${RUN_DIR}/noise_model.txt")" != "si1000" ]]; then
    echo "Expected si1000 metadata in ${RUN_DIR}/noise_model.txt."
    exit 1
fi

module load python/3.11

export OMP_NUM_THREADS=128
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export LRB_BATCH_SIZE="${LRB_BATCH_SIZE:-100000}"
export PYTHONUNBUFFERED=1

echo "worker_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "run_name=${RUN_NAME}"
echo "probability_index=${PROBABILITY_INDEX}"
echo "slurm_job_id=${SLURM_JOB_ID:-local}"
echo "slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-local}"
echo "slurm_restart_count=${SLURM_RESTART_COUNT:-0}"
echo "batch_size=${LRB_BATCH_SIZE}"

cd "${PROJECT_ROOT}"
exec python3 -u scripts/run_lrb_experiment.py \
    "${RUN_NAME}" \
    "${PROBABILITY_INDEX}" \
    --simulation-backend dem
