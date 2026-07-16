#!/bin/bash
#SBATCH --job-name=lrb_rb_cmp_single
#SBATCH --output=lrb_rb_cmp_single_%j.out
#SBATCH --error=lrb_rb_cmp_single_%j.err
#SBATCH -C cpu
#SBATCH -q preempt
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-}"
if [[ -n "${SUBMIT_DIR}" && -d "${SUBMIT_DIR}/src" && -d "${SUBMIT_DIR}/scripts" ]]; then
    PROJECT_ROOT="${SUBMIT_DIR}"
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

# Run this after the one-ancilla DEM LRB sweep is complete. The physical RB
# comparison uses p_RB = 5 * raw LRB p, which is exactly the displayed
# lambda_LRB coordinate in the DEM notebook. For example, raw LRB p=0.005 is
# paired with lambda_LRB=p_RB=0.025.
RUN_NAME_OVERRIDE="${RUN_NAME_OVERRIDE:-}"
AXIS_SCALE="5.0"
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
WORKING_FOLDER_FILE="${ROOT_DIR}/working-folder-folded_qutrit.txt"

if [[ -n "${RUN_NAME_OVERRIDE}" ]]; then
    RUN_NAME="${RUN_NAME_OVERRIDE}"
elif [[ -f "${WORKING_FOLDER_FILE}" ]]; then
    RUN_NAME="$(tr -d '[:space:]' < "${WORKING_FOLDER_FILE}")"
else
    echo "Could not determine RUN_NAME from ${WORKING_FOLDER_FILE}"
    exit 1
fi

WORKDIR="${ROOT_DIR}/${RUN_NAME}"
if [[ "$(tr -d '[:space:]' < "${WORKDIR}/code_name.txt")" != "folded_qutrit" ]]; then
    echo "Expected the one-ancilla folded_qutrit profile in ${WORKDIR}."
    exit 1
fi
if [[ "$(tr -d '[:space:]' < "${WORKDIR}/noise_model.txt")" != "si1000" ]]; then
    echo "Expected the SI1000 noise model in ${WORKDIR}."
    exit 1
fi

IFS=',' read -r -a RAW_PROBABILITIES < "${WORKDIR}/probs.txt"
PROBABILITY_COUNT=0
for probability in "${RAW_PROBABILITIES[@]}"; do
    if [[ -n "$(echo "${probability}" | tr -d '[:space:]')" ]]; then
        PROBABILITY_COUNT=$((PROBABILITY_COUNT + 1))
    fi
done
if [[ "${PROBABILITY_COUNT}" -eq 0 ]]; then
    echo "No LRB probabilities found in ${WORKDIR}/probs.txt."
    exit 1
fi
for index in $(seq 0 $((PROBABILITY_COUNT - 1))); do
    DONE_FILE="${WORKDIR}/progress/${index}/done.txt"
    if [[ ! -f "${DONE_FILE}" || "$(tr -d '[:space:]' < "${DONE_FILE}")" != "1" ]]; then
        echo "LRB probability index ${index} is not complete (${DONE_FILE})."
        echo "Finish the one-ancilla DEM LRB sweep before running matched RB."
        exit 1
    fi
done

echo "Run name: ${RUN_NAME}"
echo "Pairing rule: p_RB = lambda_LRB = ${AXIS_SCALE} * raw LRB p"
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
