#!/bin/bash
#SBATCH --job-name=lrb_dem_single_verify
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail
umask 027

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Sbatch executes a copy of this script from the Slurm spool directory, so
# BASH_SOURCE cannot locate the run folder there; LRB_RUN_DIR must name it.
RUN_DIR="${LRB_RUN_DIR:-${SCRIPT_DIR}}"
if [[ ! -f "${RUN_DIR}/code_name.txt" ]]; then
    echo "Run directory '${RUN_DIR}' has no code_name.txt. Submit with" \
        "LRB_RUN_DIR set to the run folder that holds this verifier."
    exit 1
fi
PROJECT_ROOT="$(cd "${RUN_DIR}/../.." && pwd)"
RUN_NAME="$(basename "${RUN_DIR}")"
WORKER_SCRIPT="${RUN_DIR}/run_dem_array_worker.sh"
VERIFY_SCRIPT="${RUN_DIR}/verify_or_resume_dem_array.sh"
VALIDATOR="${PROJECT_ROOT}/scripts/validate_lrb_run.py"
VALIDATION_JSON="${RUN_DIR}/last_validation.json"
CONTROLLER_STATE="${RUN_DIR}/controller_state.json"
ATTEMPT="${LRB_ARRAY_ATTEMPT:-1}"
MAX_ATTEMPTS="${LRB_ARRAY_MAX_ATTEMPTS:-6}"

exec 9>"${RUN_DIR}/.verifier.lock"
if ! flock -n 9; then
    echo "Another verifier owns ${RUN_DIR}/.verifier.lock; exiting."
    exit 0
fi

if [[ ! "${ATTEMPT}" =~ ^[1-9][0-9]*$ ]] \
        || [[ ! "${MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "LRB_ARRAY_ATTEMPT and LRB_ARRAY_MAX_ATTEMPTS must be positive integers."
    exit 1
fi

module load python/3.11
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

python3 -u "${VALIDATOR}" \
    "${RUN_NAME}" \
    --batch-size 100000 \
    --expected-code folded_qutrit \
    --expected-noise si1000 \
    --strict-single-ancilla-si1000 \
    --output "${VALIDATION_JSON}"

complete="$(
    python3 - "${VALIDATION_JSON}" <<'PY'
import json
import sys
with open(sys.argv[1]) as report_file:
    print(int(json.load(report_file)["complete"]))
PY
)"

if [[ "${complete}" == "1" ]]; then
    python3 -u "${VALIDATOR}" \
        "${RUN_NAME}" \
        --batch-size 100000 \
        --expected-code folded_qutrit \
        --expected-noise si1000 \
        --strict-single-ancilla-si1000 \
        --require-complete \
        --output "${RUN_DIR}/completion_verified.json"
    python3 - "${CONTROLLER_STATE}" "${ATTEMPT}" <<'PY'
import json
import os
import sys
import time
from lrb.lrb_simulation import atomic_write_text

state = {
    "schema_version": 1,
    "status": "COMPLETE_VERIFIED",
    "attempt": int(sys.argv[2]),
    "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "verifier_job_id": os.environ.get("SLURM_JOB_ID", "local"),
}
atomic_write_text(sys.argv[1], json.dumps(state, sort_keys=True, indent=2) + "\n")
PY
    echo "DEM single-ancilla SI1000 run is complete and deeply verified."
    exit 0
fi

incomplete_csv="$(
    python3 - "${VALIDATION_JSON}" <<'PY'
import json
import sys
with open(sys.argv[1]) as report_file:
    report = json.load(report_file)
if report["invalid_indices"]:
    raise SystemExit(
        "Invalid checkpoints require inspection: "
        + ",".join(map(str, report["invalid_indices"]))
    )
print(",".join(map(str, report["resubmittable_indices"])))
PY
)"
if [[ -z "${incomplete_csv}" ]]; then
    echo "Run is incomplete but no index is safely resubmittable."
    exit 1
fi
if [[ "${ATTEMPT}" -ge "${MAX_ATTEMPTS}" ]]; then
    echo "Maximum array attempts reached; not resubmitting ${incomplete_csv}."
    exit 1
fi

active_array_job_id="$(
    python3 - "${CONTROLLER_STATE}" <<'PY'
import json
import os
import sys
path = sys.argv[1]
if not os.path.isfile(path):
    print("")
else:
    try:
        print(json.load(open(path)).get("active_array_job_id", ""))
    except (OSError, ValueError):
        print("")
PY
)"
if [[ "${active_array_job_id}" =~ ^[0-9]+$ ]] \
        && [[ -n "$(squeue -h -j "${active_array_job_id}" 2>/dev/null)" ]]; then
    echo "Array ${active_array_job_id} is still active; no duplicate submitted."
    exit 0
fi

next_attempt="$((ATTEMPT + 1))"
token="a${next_attempt}_v${SLURM_JOB_ID:-local}_$(date -u +%Y%m%dT%H%M%SZ)"
python3 - "${CONTROLLER_STATE}" "${ATTEMPT}" "${token}" "${incomplete_csv}" <<'PY'
import json
import os
import sys
import time
from lrb.lrb_simulation import atomic_write_text

state = {
    "schema_version": 1,
    "status": "SUBMITTING_ARRAY",
    "attempt": int(sys.argv[2]),
    "submission_token": sys.argv[3],
    "indices": [int(value) for value in sys.argv[4].split(",")],
    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "verifier_job_id": os.environ.get("SLURM_JOB_ID", "local"),
}
atomic_write_text(sys.argv[1], json.dumps(state, sort_keys=True, indent=2) + "\n")
PY

array_submission="$(
    sbatch --parsable \
        --array="${incomplete_csv}" \
        --job-name="lrb_dem_single_${token}" \
        --comment="${token}" \
        --export="ALL,RUN_NAME_OVERRIDE=${RUN_NAME},LRB_BATCH_SIZE=100000,LRB_RUN_DIR=${RUN_DIR}" \
        --output="${RUN_DIR}/lrb_dem_array_%A_%a.out" \
        --error="${RUN_DIR}/lrb_dem_array_%A_%a.err" \
        "${WORKER_SCRIPT}"
)"
array_job_id="${array_submission%%;*}"
if [[ ! "${array_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Invalid array job ID returned by sbatch: ${array_submission}"
    exit 1
fi

python3 - "${CONTROLLER_STATE}" "${next_attempt}" "${token}" \
    "${incomplete_csv}" "${array_job_id}" <<'PY'
import json
import sys
import time
from lrb.lrb_simulation import atomic_write_text

state = {
    "schema_version": 1,
    "status": "ARRAY_SUBMITTED_NEEDS_VERIFIER",
    "attempt": int(sys.argv[2]),
    "submission_token": sys.argv[3],
    "indices": [int(value) for value in sys.argv[4].split(",")],
    "active_array_job_id": sys.argv[5],
    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
atomic_write_text(sys.argv[1], json.dumps(state, sort_keys=True, indent=2) + "\n")
PY

verification_submission="$(
    sbatch --parsable \
        --dependency="afterany:${array_job_id}" \
        --export="ALL,LRB_ARRAY_ATTEMPT=${next_attempt},LRB_ARRAY_MAX_ATTEMPTS=${MAX_ATTEMPTS},LRB_RUN_DIR=${RUN_DIR}" \
        --output="${RUN_DIR}/lrb_dem_verify_%j.out" \
        --error="${RUN_DIR}/lrb_dem_verify_%j.err" \
        "${VERIFY_SCRIPT}"
)"
verification_job_id="${verification_submission%%;*}"
if [[ ! "${verification_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Invalid verifier job ID returned by sbatch: ${verification_submission}"
    exit 1
fi

python3 - "${CONTROLLER_STATE}" "${next_attempt}" "${token}" \
    "${incomplete_csv}" "${array_job_id}" "${verification_job_id}" <<'PY'
import json
import sys
import time
from lrb.lrb_simulation import atomic_write_text

state = {
    "schema_version": 1,
    "status": "MONITORING_ARRAY",
    "attempt": int(sys.argv[2]),
    "submission_token": sys.argv[3],
    "indices": [int(value) for value in sys.argv[4].split(",")],
    "active_array_job_id": sys.argv[5],
    "active_verifier_job_id": sys.argv[6],
    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
atomic_write_text(sys.argv[1], json.dumps(state, sort_keys=True, indent=2) + "\n")
PY

echo "Resubmitted indices ${incomplete_csv} as array ${array_job_id}."
echo "Scheduled verifier ${verification_job_id} after the array."
