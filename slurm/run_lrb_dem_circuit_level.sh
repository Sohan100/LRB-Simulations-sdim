#!/bin/bash
#SBATCH --job-name=lrb_dem_cl
#SBATCH --output=lrb_dem_cl_%j.out
#SBATCH --error=lrb_dem_cl_%j.err
#SBATCH -C cpu
#SBATCH -q preempt
#SBATCH -t 08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --signal=B:USR1@600

# One-node DEM launcher for SI1000/SD6 folded-code runs (sdim >= 1.4.0).
#
# Generate the run first, for example:
#
# .venv-sdim140/bin/python scripts/generate_circuits_folded.py \
#     --noise-model sd6 --ancilla-mode split-unif --custom-name split-sd6 \
#     --n-cliff 30 --depths 0,2,4,6,10,14,18,20,22 --n-shots 1000000 \
#     --probabilities 0.0,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.0075,0.01,0.0125,0.015,0.02,0.03,0.04,0.05 \
#     --stab-checks-const 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22 \
#     --stab-checks-unif 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22
#
# then submit from the repository root, naming the run explicitly:
#
# sbatch --export=ALL,RUN_NAME_OVERRIDE=<Run-folder> slurm/run_lrb_dem_circuit_level.sh
#
# Sampling with sdim.dem and the vectorized postselection takes minutes per
# probability index, so every index of the run runs concurrently on this one
# node, each with LRB_THREADS_PER_INDEX sampler threads. Indices resume from
# their checkpoints; an incomplete run resubmits itself once per job.

set -euo pipefail

# sbatch runs a spool copy of this script, so locate the project from the
# submit directory.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ ! -f "${PROJECT_ROOT}/scripts/run_lrb_experiment.py" ]]; then
    echo "Submit from the repository root; ${PROJECT_ROOT} is not it."
    exit 1
fi
RUN_NAME="${RUN_NAME_OVERRIDE:?Set RUN_NAME_OVERRIDE to the run folder name.}"
WORKDIR="${PROJECT_ROOT}/LRB-experiment-data-slurm/${RUN_NAME}"
PYTHON_BIN="${LRB_PYTHON:-${PROJECT_ROOT}/.venv-sdim140/bin/python}"
THREADS_PER_INDEX="${LRB_THREADS_PER_INDEX:-8}"
AUTO_RESUBMIT="${LRB_AUTO_RESUBMIT:-1}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${WORKDIR}/logs_job_${SLURM_JOB_ID:-local}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export LRB_BATCH_SIZE="${LRB_BATCH_SIZE:-100000}"
export NUMBA_NUM_THREADS="${THREADS_PER_INDEX}"
export OMP_NUM_THREADS="${THREADS_PER_INDEX}"

fail() {
    echo "$1"
    exit 1
}

[[ -d "${WORKDIR}" ]] || fail "Missing run folder ${WORKDIR}"
[[ -x "${PYTHON_BIN}" ]] || fail "Missing Python ${PYTHON_BIN}"

CODE_NAME="$(tr -d '[:space:]' < "${WORKDIR}/code_name.txt")"
NOISE_MODEL="$(tr -d '[:space:]' < "${WORKDIR}/noise_model.txt")"
case "${CODE_NAME}" in
    folded_qutrit|folded_qutrit_split_unif|folded_qubit|folded_qubit_split_unif|folded_qutrit_conv_sites|folded_qubit_conv_sites) ;;
    *) fail "Unsupported code '${CODE_NAME}' in ${WORKDIR}/code_name.txt" ;;
esac
case "${NOISE_MODEL}" in
    si1000|sd6) ;;
    *) fail "Unsupported noise model '${NOISE_MODEL}'; expected si1000 or sd6." ;;
esac

SDIM_VERSION="$("${PYTHON_BIN}" -c 'import importlib.metadata as m; print(m.version("sdim"))')"
"${PYTHON_BIN}" -c 'import sys; from sdim.dem import DetectorErrorModel' \
    || fail "sdim ${SDIM_VERSION} at ${PYTHON_BIN} has no sdim.dem; need >= 1.4.0."

for sentinel in \
        "experiments/LRB/0/0/0.chp:DETECTOR .*label=\"lrb_stab_r0_w5\"" \
        "experiments/LRB/0/0/0.chp:LOGICAL_OBSERVABLE .*label=\"lrb_logical\"" \
        "experiments/LRB_const0/0/0/0.chp:LOGICAL_OBSERVABLE .*label=\"lrb_const0_logical\"" \
        "experiments/RB/0/0/0.chp:LOGICAL_OBSERVABLE .*label=\"rb_logical\""; do
    file="${WORKDIR}/${sentinel%%:*}"
    pattern="${sentinel#*:}"
    grep -q "${pattern}" "${file}" \
        || fail "${file} lacks ${pattern}; regenerate the circuits."
done
if grep -q "prob_scale" "${WORKDIR}/experiments/LRB/0/0/0.chp"; then
    fail "Circuits carry prob_scale, which sdim 1.4 cannot read; regenerate them."
fi

IFS=',' read -r -a PROBABILITIES < "${WORKDIR}/probs.txt"
NUM_PROBS="${#PROBABILITIES[@]}"
[[ "${NUM_PROBS}" -gt 0 ]] || fail "No probabilities in ${WORKDIR}/probs.txt"

progress_complete() {
    local idx
    for idx in $(seq 0 $((NUM_PROBS - 1))); do
        local done_file="${WORKDIR}/progress/${idx}/done.txt"
        [[ -f "${done_file}" ]] || return 1
        [[ "$(tr -d '[:space:]' < "${done_file}")" == "1" ]] || return 1
    done
    return 0
}

RESUBMITTED=0
submit_resume_job() {
    if [[ "${AUTO_RESUBMIT}" != "1" || "${RESUBMITTED}" == "1" ]] \
            || progress_complete; then
        # Explicit status: a bare return inside a trap handler returns the
        # interrupted wait's status (>128), which set -e turns into an exit.
        return 0
    fi
    RESUBMITTED=1
    echo "Resubmitting ${RUN_NAME} after $1."
    (cd "${PROJECT_ROOT}" && sbatch \
        --export="ALL,RUN_NAME_OVERRIDE=${RUN_NAME}" \
        --dependency="afterany:${SLURM_JOB_ID}" \
        --job-name="${SLURM_JOB_NAME:-lrb_dem_cl}" \
        "${PROJECT_ROOT}/slurm/run_lrb_dem_circuit_level.sh")
}
trap 'submit_resume_job "pre-timeout signal"' USR1

echo "Project root: ${PROJECT_ROOT}"
echo "Run: ${RUN_NAME} (${CODE_NAME}, ${NOISE_MODEL})"
echo "Python: ${PYTHON_BIN} (sdim ${SDIM_VERSION})"
echo "Probability indices: ${NUM_PROBS}, threads each: ${THREADS_PER_INDEX}"
echo "Checkpoint batch size: ${LRB_BATCH_SIZE}"
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: validation passed, nothing launched."
    exit 0
fi

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"
PIDS=()
for idx in $(seq 0 $((NUM_PROBS - 1))); do
    p="$(echo "${PROBABILITIES[idx]}" | tr -d '[:space:]')"
    "${PYTHON_BIN}" -u scripts/run_lrb_experiment.py "${RUN_NAME}" "${idx}" \
        --simulation-backend dem \
        > "${LOG_DIR}/run_p_idx${idx}_p${p}.log" 2>&1 &
    PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
    while true; do
        rc=0
        wait "${pid}" || rc=$?
        # A trapped USR1 interrupts wait with a status above 128 while the
        # worker keeps running.
        if [[ "${rc}" -gt 128 ]] && kill -0 "${pid}" 2>/dev/null; then
            continue
        fi
        break
    done
    [[ "${rc}" -eq 0 ]] || status=1
done

if progress_complete; then
    echo "All ${NUM_PROBS} indices complete."
    exit 0
fi
echo "Run incomplete (worker status ${status}); see ${LOG_DIR}."
submit_resume_job "incomplete run"
exit 1
