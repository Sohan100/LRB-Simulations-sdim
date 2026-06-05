#!/bin/bash
#SBATCH --job-name=lrb_split_unif_si1000
#SBATCH --output=lrb_split_unif_si1000_%j.out
#SBATCH --error=lrb_split_unif_si1000_%j.err
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 47:30:00
#SBATCH --nodes=15
#SBATCH --ntasks=15
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-}"
if [[ -n "${SUBMIT_DIR}" && -d "${SUBMIT_DIR}/src" && -d "${SUBMIT_DIR}/scripts" ]]; then
    PROJECT_ROOT="${SUBMIT_DIR}"
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

# Generate circuits before submitting this file, for example:
#
# python3 scripts/generate_circuits_folded.py \
#     --noise-model si1000 \
#     --ancilla-mode split-unif \
#     --custom-name split-unif-si1000
#
# The split-unif ancilla mode uses the 11-wire split-boundary folded profile,
# writes code_name.txt as folded_qutrit_split_unif, and appends SDIM
# DETECTOR/LOGICAL_OBSERVABLE operations to newly generated .chp files. This
# launcher expects the same all-checks output families as the conventional
# folded SI1000 launcher. Nonzero constant checks use the split-ancilla LRB
# stabilizer-check circuits. Only const=0 uses the separate direct terminal
# data-qudit X readout in experiments/LRB_const0. Uniform checks also use the
# split-ancilla LRB stabilizer-check circuits. By default this launcher also
# enforces the old folded all-check SI1000 metadata: const checks 0..22, uniform
# checks 1..22, and the recommended lower SI1000 probability grid from 0
# through 0.05.

# --- BEGIN USER CONFIGURABLE SECTION ---
RUN_NAME_OVERRIDE="${RUN_NAME_OVERRIDE:-}"
NUM_SHOTS="${NUM_SHOTS:-1000000}"
ROOT_DIR="${LRB_RUNS_ROOT:-${PROJECT_ROOT}/LRB-experiment-data-slurm}"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts"
PYTHON_BIN="${LRB_PYTHON:-python3}"
REQUIRE_DETECTORS="${REQUIRE_DETECTORS:-1}"
REQUIRE_SPLIT_SI1000_PARAMETERS="${REQUIRE_SPLIT_SI1000_PARAMETERS:-${REQUIRE_OLD_SI1000_PARAMETERS:-1}}"
DRY_RUN="${DRY_RUN:-0}"
# --- END USER CONFIGURABLE SECTION ---

EXPECTED_CODE_NAME="folded_qutrit_split_unif"
# These are the all-check folded SI1000 settings this split-ancilla launcher is
# meant to mirror. The old folded metadata uses every constant-check policy from
# 0 through 22 and every uniform-interval policy from 1 through 22. The
# old folded metadata also uses 30 Clifford sequences, depths
# 0,2,4,6,10,14,18,20,22, and 1,000,000 shots. The probability list below is
# intentionally lower than the old depolarizing grid because SI1000 assigns
# measurement events a 5p rate and reset/idles a 2p rate.
EXPECTED_CONST_CHECKS="${EXPECTED_CONST_CHECKS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22}"
EXPECTED_UNIF_CHECKS="${EXPECTED_UNIF_CHECKS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22}"
EXPECTED_PROBABILITIES="${EXPECTED_PROBABILITIES:-0.0,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.0075,0.01,0.0125,0.015,0.02,0.03,0.04,0.05}"
EXPECTED_DEPTHS="${EXPECTED_DEPTHS:-0,2,4,6,10,14,18,20,22}"
EXPECTED_NUM_CLIFFS="${EXPECTED_NUM_CLIFFS:-30}"
EXPECTED_SHOTS="${EXPECTED_SHOTS:-1000000}"
WORKING_FOLDER_FILE="${ROOT_DIR}/working-folder-${EXPECTED_CODE_NAME}.txt"
LEGACY_WORKING_FOLDER_FILE="${ROOT_DIR}/working-folder.txt"

join_csv() {
    local IFS=,
    echo "$*"
}

require_expected_csv() {
    local label="$1"
    local metadata_file="$2"
    local actual_csv="$3"
    local expected_csv="$4"

    if [[ "${actual_csv}" != "${expected_csv}" ]]; then
        echo "${metadata_file} does not match the expected split SI1000 ${label}."
        echo "Expected: ${expected_csv}"
        echo "Actual:   ${actual_csv}"
        echo "Regenerate the split-unif run with the expected SI1000 ${label}, or set"
        echo "REQUIRE_SPLIT_SI1000_PARAMETERS=0 if this is an intentional custom sweep."
        exit 1
    fi
}

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
DEPTHS_FILE="${WORKDIR}/depths.txt"
NUM_CLIFFS_FILE="${WORKDIR}/num_cliffs.txt"
PROBS_FILE="${WORKDIR}/probs.txt"
CHECK_CONST_FILE="${WORKDIR}/check_const.txt"
CHECK_UNIF_FILE="${WORKDIR}/check_unif.txt"
NOISE_MODEL_FILE="${WORKDIR}/noise_model.txt"
LRB_SENTINEL="${WORKDIR}/experiments/LRB/0/0/0.chp"
LRB_CONST0_SENTINEL="${WORKDIR}/experiments/LRB_const0/0/0/0.chp"
RB_SENTINEL="${WORKDIR}/experiments/RB/0/0/0.chp"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${CODE_NAME_FILE}" ]]; then
    echo "Missing ${CODE_NAME_FILE}"
    exit 1
fi

ACTUAL_CODE_NAME="$(tr -d '[:space:]' < "${CODE_NAME_FILE}")"
if [[ "${ACTUAL_CODE_NAME}" != "${EXPECTED_CODE_NAME}" ]]; then
    echo "Expected code '${EXPECTED_CODE_NAME}' but found" \
        "'${ACTUAL_CODE_NAME}' in ${CODE_NAME_FILE}"
    echo "Regenerate with --ancilla-mode split-unif before submitting this launcher."
    exit 1
fi

if [[ ! -f "${PROBS_FILE}" ]]; then
    echo "Missing ${PROBS_FILE}"
    exit 1
fi

if [[ ! -f "${DEPTHS_FILE}" ]]; then
    echo "Missing ${DEPTHS_FILE}"
    exit 1
fi

IFS=',' read -r -a RAW_DEPTHS < "${DEPTHS_FILE}"
DEPTHS=()
for depth in "${RAW_DEPTHS[@]}"; do
    depth="$(echo "${depth}" | tr -d '[:space:]')"
    if [[ -n "${depth}" ]]; then
        DEPTHS+=("${depth}")
    fi
done
DEPTHS_CSV="$(join_csv "${DEPTHS[@]}")"
if [[ "${#DEPTHS[@]}" -eq 0 ]]; then
    echo "No benchmark depths found in ${DEPTHS_FILE}"
    exit 1
fi
if [[ "${REQUIRE_SPLIT_SI1000_PARAMETERS}" == "1" ]]; then
    require_expected_csv "depth grid" \
        "${DEPTHS_FILE}" \
        "${DEPTHS_CSV}" \
        "${EXPECTED_DEPTHS}"
fi

if [[ ! -f "${NUM_CLIFFS_FILE}" ]]; then
    echo "Missing ${NUM_CLIFFS_FILE}"
    exit 1
fi

NUM_CLIFFS="$(tr -d '[:space:]' < "${NUM_CLIFFS_FILE}")"
if [[ -z "${NUM_CLIFFS}" ]]; then
    echo "No Clifford count found in ${NUM_CLIFFS_FILE}"
    exit 1
fi
if [[ "${REQUIRE_SPLIT_SI1000_PARAMETERS}" == "1" ]]; then
    require_expected_csv "Clifford-count value" \
        "${NUM_CLIFFS_FILE}" \
        "${NUM_CLIFFS}" \
        "${EXPECTED_NUM_CLIFFS}"
fi

if [[ ! -f "${SHOTS_FILE}" ]]; then
    echo "Missing ${SHOTS_FILE}"
    exit 1
fi

GENERATED_SHOTS="$(tr -d '[:space:]' < "${SHOTS_FILE}")"
if [[ -z "${GENERATED_SHOTS}" ]]; then
    echo "No shot count found in ${SHOTS_FILE}"
    exit 1
fi
if [[ "${REQUIRE_SPLIT_SI1000_PARAMETERS}" == "1" ]]; then
    require_expected_csv "generated shot-count value" \
        "${SHOTS_FILE}" \
        "${GENERATED_SHOTS}" \
        "${EXPECTED_SHOTS}"
    require_expected_csv "launcher shot-count value" \
        "NUM_SHOTS" \
        "${NUM_SHOTS}" \
        "${EXPECTED_SHOTS}"
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
PROBABILITIES_CSV="$(join_csv "${PROBABILITIES[@]}")"

NOISE_MODEL="unknown"
if [[ -f "${NOISE_MODEL_FILE}" ]]; then
    NOISE_MODEL="$(tr -d '[:space:]' < "${NOISE_MODEL_FILE}")"
fi

if [[ "${NOISE_MODEL}" != "si1000" ]]; then
    echo "This launcher is for SI1000 runs, but ${NOISE_MODEL_FILE} says '${NOISE_MODEL}'."
    exit 1
fi

if [[ "${REQUIRE_SPLIT_SI1000_PARAMETERS}" == "1" ]]; then
    require_expected_csv "probability grid" \
        "${PROBS_FILE}" \
        "${PROBABILITIES_CSV}" \
        "${EXPECTED_PROBABILITIES}"
fi

for p in "${PROBABILITIES[@]}"; do
    if ! awk -v p="${p}" 'BEGIN { exit (p <= 0.2 ? 0 : 1) }'; then
        echo "SI1000 run contains p=${p} > 0.2, which makes 5p > 1."
        exit 1
    fi
done

CONST_CHECKS=()
HAS_CONST0=0
HAS_MAIN_CONST=0
if [[ -f "${CHECK_CONST_FILE}" ]]; then
    IFS=',' read -r -a RAW_CONST_CHECKS < "${CHECK_CONST_FILE}"
    for check in "${RAW_CONST_CHECKS[@]}"; do
        check="$(echo "${check}" | tr -d '[:space:]')"
        if [[ -n "${check}" ]]; then
            CONST_CHECKS+=("${check}")
            if [[ "${check}" == "0" ]]; then
                HAS_CONST0=1
            else
                HAS_MAIN_CONST=1
            fi
        fi
    done
fi
if [[ "${#CONST_CHECKS[@]}" -eq 0 ]]; then
    echo "No constant-check policies found in ${CHECK_CONST_FILE}."
    echo "Regenerate with --ancilla-mode split-unif and non-empty --stab-checks-const."
    exit 1
fi
CONST_CHECKS_CSV="$(join_csv "${CONST_CHECKS[@]}")"
if [[ "${REQUIRE_SPLIT_SI1000_PARAMETERS}" == "1" ]]; then
    require_expected_csv "constant-check grid" \
        "${CHECK_CONST_FILE}" \
        "${CONST_CHECKS_CSV}" \
        "${EXPECTED_CONST_CHECKS}"
fi
if [[ "${HAS_CONST0}" -ne 1 ]]; then
    echo "${CHECK_CONST_FILE} does not contain const=0."
    echo "Include 0 in --stab-checks-const so the direct-X const0 circuits are generated."
    exit 1
fi
if [[ "${HAS_MAIN_CONST}" -ne 1 ]]; then
    echo "${CHECK_CONST_FILE} contains const=0 but no nonzero constant checks."
    echo "Include at least one nonzero value in --stab-checks-const so ordinary const checks use the split-ancilla LRB circuits."
    exit 1
fi

UNIF_CHECKS=()
if [[ -f "${CHECK_UNIF_FILE}" ]]; then
    IFS=',' read -r -a RAW_UNIF_CHECKS < "${CHECK_UNIF_FILE}"
    for check in "${RAW_UNIF_CHECKS[@]}"; do
        check="$(echo "${check}" | tr -d '[:space:]')"
        if [[ -n "${check}" ]]; then
            UNIF_CHECKS+=("${check}")
        fi
    done
fi
if [[ "${#UNIF_CHECKS[@]}" -eq 0 ]]; then
    echo "No uniform-interval checks found in ${CHECK_UNIF_FILE}."
    exit 1
fi
UNIF_CHECKS_CSV="$(join_csv "${UNIF_CHECKS[@]}")"
if [[ "${REQUIRE_SPLIT_SI1000_PARAMETERS}" == "1" ]]; then
    require_expected_csv "uniform-check grid" \
        "${CHECK_UNIF_FILE}" \
        "${UNIF_CHECKS_CSV}" \
        "${EXPECTED_UNIF_CHECKS}"
fi

if [[ ! -f "${LRB_SENTINEL}" ]]; then
    echo "Missing ${LRB_SENTINEL}"
    echo "Regenerate circuits before submitting this launcher."
    exit 1
fi

if [[ ! -f "${LRB_CONST0_SENTINEL}" ]]; then
    echo "Missing ${LRB_CONST0_SENTINEL}"
    echo "Regenerate with const=0 included in --stab-checks-const."
    exit 1
fi

if [[ ! -f "${RB_SENTINEL}" ]]; then
    echo "Missing ${RB_SENTINEL}"
    echo "Regenerate circuits before submitting this launcher."
    exit 1
fi

if [[ "${REQUIRE_DETECTORS}" == "1" ]]; then
    if ! grep -q 'DETECTOR .*label="lrb_stab_r0_w5"' "${LRB_SENTINEL}"; then
        echo "${LRB_SENTINEL} does not contain split LRB detector labels."
        echo "Regenerate circuits with the detector-enabled generator before submitting."
        exit 1
    fi
    if ! grep -q 'LOGICAL_OBSERVABLE .*label="lrb_logical"' "${LRB_SENTINEL}"; then
        echo "${LRB_SENTINEL} does not contain the LRB logical observable label."
        echo "Regenerate circuits with the detector-enabled generator before submitting."
        exit 1
    fi
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
echo "Generated shots metadata: ${GENERATED_SHOTS}"
echo "Clifford sequences: ${NUM_CLIFFS}"
echo "Depths: ${DEPTHS[*]}"
echo "Constant checks: ${CONST_CHECKS[*]}"
echo "Uniform checks: ${UNIF_CHECKS[*]}"
echo "Detector annotations required: ${REQUIRE_DETECTORS}"
echo "Strict split SI1000 parameter grid required: ${REQUIRE_SPLIT_SI1000_PARAMETERS}"
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
