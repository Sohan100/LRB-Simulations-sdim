#!/bin/bash
#SBATCH --job-name=lrb_qgrm_parallel
#SBATCH --output=lrb_qgrm_parallel_%j.out
#SBATCH --error=lrb_qgrm_parallel_%j.err
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 47:30:00
#SBATCH --nodes=18
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=256

# --- BEGIN USER CONFIGURABLE SECTION ---
# Optional manual override. Leave empty to auto-resolve latest run.
RUN_NAME_OVERRIDE=""
NUM_SHOTS=1000000
SCRIPTS_DIR="."
# --- END USER CONFIGURABLE SECTION ---

EXPECTED_CODE_NAME="qgrm_3_1_2"

PROBABILITIES=(
3.35981829e-05 0.000615848211 0.0112883789 0.0206130785 0.0233572147 \
0.0311537409 0.0362021775 0.0420687089 0.0483293024 0.0547144504 \
0.0635808794 0.0738841056 0.0858569606 0.0925524149 0.1 \
0.143844989 0.206913808 0.335981829
)
NUM_PROBS=${#PROBABILITIES[@]}

module load python/3.11

export OMP_NUM_THREADS=128
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

ROOT_DIR="${PWD}/LRB-experiment-data-slurm"
WORKING_FOLDER_FILE="${ROOT_DIR}/working-folder-qgrm_3_1_2.txt"
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

echo "${NUM_SHOTS}" > "${SHOTS_FILE}"

echo "Run Name: ${RUN_NAME}"
echo "Code Name: ${ACTUAL_CODE_NAME}"
echo "Shots: ${NUM_SHOTS}"
echo "Number of probabilities: ${NUM_PROBS}"
echo "Logs: ${LOG_DIR}"

for idx in $(seq 0 $((${NUM_PROBS} - 1))); do
    prob_val_for_log="${PROBABILITIES[idx]}"
    echo "Launching idx ${idx} (p ~ ${prob_val_for_log})"
    srun --exclusive --nodes=1 --ntasks=1 \
        --cpus-per-task=${SLURM_CPUS_PER_TASK} \
        python3 "${SCRIPTS_DIR}/run_lrb_experiment.py" "${RUN_NAME}" "${idx}" \
        > "${LOG_DIR}/run_p_idx${idx}_val${prob_val_for_log}.log" 2>&1 &
done

wait
echo "Job ${SLURM_JOB_ID} completed successfully."
