#!/bin/bash
#SBATCH --job-name=lrb_qgrm_si1000
#SBATCH --output=lrb_qgrm_si1000_%j.out
#SBATCH --error=lrb_qgrm_si1000_%j.err
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 47:30:00
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Generate circuits before submitting this file, for example:
# python3 scripts/generate_circuits_qgrm.py --noise-model si1000 \
#     --custom-name si1000-all --probabilities <comma-separated p list>
#
# This launcher runs one probability index per srun. Each probability job
# processes every constant-check and uniform-interval policy listed in the
# generated run folder. If const=0 is present, run_lrb_experiment.py uses the
# generated experiments/LRB_const0 circuits for that special direct-X check.

# --- BEGIN USER CONFIGURABLE SECTION ---
RUN_NAME_OVERRIDE="${RUN_NAME_OVERRIDE:-}"
NUM_SHOTS="${NUM_SHOTS:-1000000}"
ROOT_DIR="${LRB_RUNS_ROOT:-${PROJECT_ROOT}/LRB-experiment-data-slurm}"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts"
PYTHON_BIN="${LRB_PYTHON:-python3}"
# --- END USER CONFIGURABLE SECTION ---

EXPECTED_CODE_NAME="qgrm_3_1_2"
WORKING_FOLDER_FILE="${ROOT_DIR}/working-folder-${EXPECTED_CODE_NAME}.txt"
LEGACY_WORKING_FOLDER_FILE="${ROOT_DIR}/working-folder.txt"

if command -v module >/dev/null 2>&1; then
    module load python/3.11
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-128}"
export OMP_PLACES="${OMP_PLACES:-threads}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-256}"

if [[ -n "${RUN_NAME_OVERRIDE}" ]]; then
    RUN_NAME="${RUN_NAME_OVERRIDE}"
elif [[ -f "${WORKING_FOLDER_FILE}" ]]; then
    RUN_NAME="$(tr -d '[:space:]' < "${WORKING_FOLDER_FILE}")"
elif [[ -f "${LEGACY_WORKING_FOLDER_FILE}" ]]; then
    RUN_NAME="$(tr -d '[:space:]' < "${LEGACY_WORKING_FOLDER_FILE}")"
else
    LATEST_RUN_PATH="$(ls -1dt "${ROOT_DIR}"/Run-* 2>/dev/null | head -n 1)"
    RUN_NAME="$(basename "${LATEST_RUN_PATH}")"
fi

if [[ -z "${RUN_NAME}" ]]; then
    echo "Could not determine RUN_NAME from ${WORKING_FOLDER_FILE} or Run-*"
    exit 1
fi

WORKDIR="${ROOT_DIR}/${RUN_NAME}"
LOG_DIR="${WORKDIR}/logs_job_${SLURM_JOB_ID:-local}"
CODE_NAME_FILE="${WORKDIR}/code_name.txt"
SHOTS_FILE="${WORKDIR}/shots.txt"
PROBS_FILE="${WORKDIR}/probs.txt"
CHECK_CONST_FILE="${WORKDIR}/check_const.txt"
NOISE_MODEL_FILE="${WORKDIR}/noise_model.txt"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${CODE_NAME_FILE}" ]]; then
    echo "Missing ${CODE_NAME_FILE}"
    exit 1
fi

ACTUAL_CODE_NAME="$(tr -d '[:space:]' < "${CODE_NAME_FILE}")"
if [[ "${ACTUAL_CODE_NAME}" != "${EXPECTED_CODE_NAME}" ]]; then
    echo "Expected code '${EXPECTED_CODE_NAME}' but found" \
        "'${ACTUAL_CODE_NAME}' in ${CODE_NAME_FILE}"
    exit 1
fi

if [[ ! -f "${PROBS_FILE}" ]]; then
    echo "Missing ${PROBS_FILE}"
    exit 1
fi

IFS=',' read -r -a PROBABILITIES < "${PROBS_FILE}"
NUM_PROBS="${#PROBABILITIES[@]}"
if [[ "${NUM_PROBS}" -eq 0 ]]; then
    echo "No probabilities found in ${PROBS_FILE}"
    exit 1
fi

NOISE_MODEL="unknown"
if [[ -f "${NOISE_MODEL_FILE}" ]]; then
    NOISE_MODEL="$(tr -d '[:space:]' < "${NOISE_MODEL_FILE}")"
fi

if [[ "${NOISE_MODEL}" != "si1000" ]]; then
    echo "This launcher is for SI1000 runs, but ${NOISE_MODEL_FILE} says '${NOISE_MODEL}'."
    exit 1
fi

for p in "${PROBABILITIES[@]}"; do
    if ! awk -v p="${p}" 'BEGIN { exit (p <= 0.2 ? 0 : 1) }'; then
        echo "SI1000 run contains p=${p} > 0.2, which makes 5p > 1."
        exit 1
    fi
done

if [[ -f "${CHECK_CONST_FILE}" ]]; then
    IFS=',' read -r -a CONST_CHECKS < "${CHECK_CONST_FILE}"
    for check in "${CONST_CHECKS[@]}"; do
        check="$(echo "${check}" | tr -d '[:space:]')"
        if [[ "${check}" == "0" ]]; then
            CONST0_SENTINEL="${WORKDIR}/experiments/LRB_const0/0/0/0.chp"
            if [[ ! -f "${CONST0_SENTINEL}" ]]; then
                echo "check_const.txt requests const=0 but ${CONST0_SENTINEL} is missing."
                echo "Regenerate circuits with the current generator before submitting."
                exit 1
            fi
        fi
    done
fi

echo "${NUM_SHOTS}" > "${SHOTS_FILE}"

echo "Project root: ${PROJECT_ROOT}"
echo "Run name: ${RUN_NAME}"
echo "Code name: ${ACTUAL_CODE_NAME}"
echo "Noise model: ${NOISE_MODEL}"
echo "Shots: ${NUM_SHOTS}"
echo "Number of probabilities: ${NUM_PROBS}"
echo "Allocated tasks: ${SLURM_NTASKS:-unknown}"
echo "Logs: ${LOG_DIR}"

PIDS=()
for idx in $(seq 0 $((NUM_PROBS - 1))); do
    prob_val_for_log="${PROBABILITIES[idx]}"
    safe_prob_label="$(echo "${prob_val_for_log}" | tr '+.-' 'p__')"
    echo "Launching idx ${idx} (p=${prob_val_for_log})"
    srun --exclusive --nodes=1 --ntasks=1 \
        --cpus-per-task="${SRUN_CPUS_PER_TASK}" \
        "${PYTHON_BIN}" "${SCRIPTS_DIR}/run_lrb_experiment.py" \
        "${RUN_NAME}" "${idx}" \
        > "${LOG_DIR}/run_p_idx${idx}_p${safe_prob_label}.log" 2>&1 &
    PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

if [[ "${status}" -ne 0 ]]; then
    echo "One or more probability jobs failed. Check ${LOG_DIR}."
    exit "${status}"
fi

echo "Job ${SLURM_JOB_ID:-local} completed successfully."
