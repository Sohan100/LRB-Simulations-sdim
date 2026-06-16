#!/bin/bash
#SBATCH --job-name=lrb_folded_si1000
#SBATCH --output=lrb_folded_si1000_%j.out
#SBATCH --error=lrb_folded_si1000_%j.err
#SBATCH -C cpu
#SBATCH -q preempt
#SBATCH -t 47:30:00
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=256
#SBATCH --signal=B:USR1@600

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-}"
if [[ -n "${SUBMIT_DIR}" && -d "${SUBMIT_DIR}/src" && -d "${SUBMIT_DIR}/scripts" ]]; then
    PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -d "${SCRIPT_DIR}/src" && -d "${SCRIPT_DIR}/scripts" ]]; then
    PROJECT_ROOT="${SCRIPT_DIR}"
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

# Generate circuits before submitting this file, for example:
# python3 scripts/generate_circuits_folded.py --noise-model si1000 \
#     --custom-name si1000-all --probabilities <comma-separated p list>
#
# This launcher runs one probability index per srun. Each probability job
# processes every constant-check and uniform-interval policy listed in the
# generated run folder. If const=0 is present, run_lrb_experiment.py uses the
# generated experiments/LRB_const0 circuits for that special direct-X check.

# --- BEGIN USER CONFIGURABLE SECTION ---
RUN_NAME_OVERRIDE="${RUN_NAME_OVERRIDE:-}"
NUM_SHOTS=1000000
SCRIPTS_DIR="${PROJECT_ROOT}/scripts"
PYTHON_BIN="${LRB_PYTHON:-python3}"
SIMULATION_BACKEND="${LRB_SIMULATION_BACKEND:-sdim}"
BATCH_SIZE="${LRB_BATCH_SIZE:-100000}"
AUTO_RESUBMIT="${LRB_AUTO_RESUBMIT:-1}"
RESUBMIT_SCRIPT="${LRB_RESUBMIT_SCRIPT:-${PROJECT_ROOT}/slurm/run_lrb_si1000_all_checks_folded.sh}"
DRY_RUN="${DRY_RUN:-0}"
# --- END USER CONFIGURABLE SECTION ---

EXPECTED_CODE_NAME="folded_qutrit"
ROOT_DIR="${PROJECT_ROOT}/LRB-experiment-data-slurm"

module load python/3.11

export OMP_NUM_THREADS=128
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export LRB_BATCH_SIZE="${BATCH_SIZE}"

WORKING_FOLDER_FILE="${ROOT_DIR}/working-folder-${EXPECTED_CODE_NAME}.txt"
LEGACY_WORKING_FOLDER_FILE="${ROOT_DIR}/working-folder.txt"

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
LOG_DIR="${WORKDIR}/logs_job_${SLURM_JOB_ID}"
CODE_NAME_FILE="${WORKDIR}/code_name.txt"
SHOTS_FILE="${WORKDIR}/shots.txt"
PROBS_FILE="${WORKDIR}/probs.txt"
CHECK_CONST_FILE="${WORKDIR}/check_const.txt"
NOISE_MODEL_FILE="${WORKDIR}/noise_model.txt"
LRB_SENTINEL="${WORKDIR}/experiments/LRB/0/0/0.chp"
LRB_CONST0_SENTINEL="${WORKDIR}/experiments/LRB_const0/0/0/0.chp"
RB_SENTINEL="${WORKDIR}/experiments/RB/0/0/0.chp"

mkdir -p "${LOG_DIR}"

AUTO_RESUBMIT_DONE=0

progress_complete() {
    local done_file
    shopt -s nullglob
    local done_files=("${WORKDIR}"/progress/*/done.txt)
    shopt -u nullglob
    if [[ "${#done_files[@]}" -eq 0 ]]; then
        return 1
    fi
    for done_file in "${done_files[@]}"; do
        if [[ "$(tr -d '[:space:]' < "${done_file}")" != "1" ]]; then
            return 1
        fi
    done
    return 0
}

submit_resume_job() {
    local reason="$1"
    if [[ "${AUTO_RESUBMIT}" != "1" || "${AUTO_RESUBMIT_DONE}" == "1" ]]; then
        return
    fi
    if progress_complete; then
        echo "Run is complete; not auto-resubmitting."
        return
    fi
    AUTO_RESUBMIT_DONE=1
    local sbatch_args=(--export="ALL,RUN_NAME_OVERRIDE=${RUN_NAME}")
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        sbatch_args+=(--dependency="afterany:${SLURM_JOB_ID}")
    fi
    echo "Auto-resubmitting ${RUN_NAME} after ${reason} with ${RESUBMIT_SCRIPT}"
    sbatch "${sbatch_args[@]}" "${RESUBMIT_SCRIPT}"
}

handle_pretimeout_signal() {
    echo "Received pre-timeout signal; scheduling an automatic resume."
    submit_resume_job "pre-timeout signal"
}

trap handle_pretimeout_signal USR1

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

IFS=',' read -r -a RAW_PROBABILITIES < "${PROBS_FILE}"
PROBABILITIES=()
for p in "${RAW_PROBABILITIES[@]}"; do
    p="$(echo "${p}" | tr -d '[:space:]')"
    if [[ -n "${p}" ]]; then
        PROBABILITIES+=("${p}")
    fi
done
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
    IFS=',' read -r -a RAW_CONST_CHECKS < "${CHECK_CONST_FILE}"
    CONST_CHECKS=()
    for check in "${RAW_CONST_CHECKS[@]}"; do
        check="$(echo "${check}" | tr -d '[:space:]')"
        if [[ -n "${check}" ]]; then
            CONST_CHECKS+=("${check}")
        fi
    done
    for check in "${CONST_CHECKS[@]}"; do
        if [[ "${check}" == "0" ]]; then
            if [[ ! -f "${LRB_CONST0_SENTINEL}" ]]; then
                echo "check_const.txt requests const=0 but ${LRB_CONST0_SENTINEL} is missing."
                echo "Regenerate circuits with the current generator before submitting."
                exit 1
            fi
        fi
    done
fi

if [[ "${SIMULATION_BACKEND}" == "dem" ]]; then
    if [[ ! -f "${LRB_SENTINEL}" ]]; then
        echo "Missing ${LRB_SENTINEL}"
        echo "Regenerate circuits before submitting this launcher."
        exit 1
    fi
    if [[ ! -f "${RB_SENTINEL}" ]]; then
        echo "Missing ${RB_SENTINEL}"
        echo "Regenerate circuits before submitting this launcher."
        exit 1
    fi
    if ! grep -q 'DETECTOR .*label="lrb_stab_r0_w5"' "${LRB_SENTINEL}"; then
        echo "${LRB_SENTINEL} does not contain folded LRB detector labels."
        echo "Regenerate circuits with the detector-enabled generator before submitting."
        exit 1
    fi
    if ! grep -q 'LOGICAL_OBSERVABLE .*label="lrb_logical"' "${LRB_SENTINEL}"; then
        echo "${LRB_SENTINEL} does not contain the LRB logical observable label."
        echo "Regenerate circuits with the detector-enabled generator before submitting."
        exit 1
    fi
    if [[ -f "${CHECK_CONST_FILE}" ]] \
            && printf '%s\n' "${CONST_CHECKS[@]}" | grep -qx '0'; then
        if ! grep -q 'DETECTOR .*label="lrb_const0_xstab_0"' "${LRB_CONST0_SENTINEL}"; then
            echo "${LRB_CONST0_SENTINEL} does not contain const0 detector labels."
            echo "Regenerate circuits with the detector-enabled generator before submitting."
            exit 1
        fi
        if ! grep -q 'LOGICAL_OBSERVABLE .*label="lrb_const0_logical"' "${LRB_CONST0_SENTINEL}"; then
            echo "${LRB_CONST0_SENTINEL} does not contain the const0 logical observable label."
            echo "Regenerate circuits with the detector-enabled generator before submitting."
            exit 1
        fi
    fi
    if ! grep -q 'LOGICAL_OBSERVABLE .*label="rb_logical"' "${RB_SENTINEL}"; then
        echo "${RB_SENTINEL} does not contain the RB logical observable label."
        echo "Regenerate circuits with the detector-enabled generator before submitting."
        exit 1
    fi
fi

echo "${NUM_SHOTS}" > "${SHOTS_FILE}"

echo "Project root: ${PROJECT_ROOT}"
echo "Run name: ${RUN_NAME}"
echo "Code name: ${ACTUAL_CODE_NAME}"
echo "Noise model: ${NOISE_MODEL}"
echo "Shots: ${NUM_SHOTS}"
echo "Simulation backend: ${SIMULATION_BACKEND}"
echo "Checkpoint batch size: ${BATCH_SIZE}"
echo "Auto-resubmit enabled: ${AUTO_RESUBMIT}"
echo "Number of probabilities: ${NUM_PROBS}"
echo "Allocated tasks: ${SLURM_NTASKS:-unknown}"
echo "Logs: ${LOG_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1, validation completed without launching srun workers."
    exit 0
fi

PIDS=()
for idx in $(seq 0 $((NUM_PROBS - 1))); do
    prob_val_for_log="${PROBABILITIES[idx]}"
    safe_prob_label="$(echo "${prob_val_for_log}" | tr '+.-' 'p__')"
    echo "Launching idx ${idx} (p=${prob_val_for_log})"
    srun --exclusive --nodes=1 --ntasks=1 \
        --cpus-per-task=${SLURM_CPUS_PER_TASK} \
        "${PYTHON_BIN}" "${SCRIPTS_DIR}/run_lrb_experiment.py" \
        "${RUN_NAME}" "${idx}" \
        --simulation-backend "${SIMULATION_BACKEND}" \
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
    submit_resume_job "worker failure"
    exit "${status}"
fi

if ! progress_complete; then
    submit_resume_job "incomplete run"
fi

echo "Job ${SLURM_JOB_ID:-local} completed successfully."
