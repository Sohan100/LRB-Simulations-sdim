#!/bin/bash
#SBATCH --job-name=lrb_rb_cmp_split
#SBATCH --output=lrb_rb_cmp_split_%j.out
#SBATCH --error=lrb_rb_cmp_split_%j.err
#SBATCH -C cpu
#SBATCH -q preempt
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/src" && -d "${SCRIPT_DIR}/scripts" ]]; then
    PROJECT_ROOT="${SCRIPT_DIR}"
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

# Optional manual override. Leave empty to use the split-unif marker.
RUN_NAME_OVERRIDE="${RUN_NAME_OVERRIDE:-}"
AXIS_SCALE="${AXIS_SCALE:-5.0}"
RB_SHOTS="${RB_SHOTS:-10000}"
RB_INDICES="${RB_INDICES:-all}"
RB_SEED="${RB_SEED:-314159}"
SIM_BACKEND="${LRB_SIMULATION_BACKEND:-dem}"

module load python/3.11

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

ROOT_DIR="${PROJECT_ROOT}/LRB-experiment-data-slurm"
WORKING_FOLDER_FILE="${ROOT_DIR}/working-folder-folded_qutrit_split_unif.txt"

if [[ -n "${RUN_NAME_OVERRIDE}" ]]; then
    RUN_NAME="${RUN_NAME_OVERRIDE}"
elif [[ -f "${WORKING_FOLDER_FILE}" ]]; then
    RUN_NAME="$(tr -d '[:space:]' < "${WORKING_FOLDER_FILE}")"
else
    echo "Could not determine RUN_NAME from ${WORKING_FOLDER_FILE}"
    exit 1
fi

echo "Run Name: ${RUN_NAME}"
echo "Axis scale: ${AXIS_SCALE}"
echo "RB shots: ${RB_SHOTS}"
echo "RB indices: ${RB_INDICES}"
echo "Simulation backend: ${SIM_BACKEND}"

srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    python3 "${PROJECT_ROOT}/scripts/run_physical_rb_comparison.py" \
    "${RUN_NAME}" \
    --axis-scale "${AXIS_SCALE}" \
    --shots "${RB_SHOTS}" \
    --indices "${RB_INDICES}" \
    --seed "${RB_SEED}" \
    --simulation-backend "${SIM_BACKEND}"
