#!/bin/bash
# PROD5B LRB-D campaign (folded_qutrit_prod5b: the paper's one-ancilla round with its
# gates reordered into 5 gate moments, user's go-ahead 2026-11-19). Four runs, each
# paired with a current one-ancilla run (same Clifford sequences, grids and noise;
# scripts/make_round_variant.py): SI1000 and SD6 production and their high-p
# extensions. Runs in 30-minute debug jobs until every run is complete, then
# validates each run. Jobs are submitted from the one-ancilla-prod5b worktree (branch
# one-ancilla-prod5b) so they use its code; its LRB-experiment-data-slurm is a
# symlink to this data folder. Runs detached:
#   setsid nohup bash LRB-experiment-data-slurm/local-runner/prod5b-campaign-driver.sh \
#       >> LRB-experiment-data-slurm/local-runner/prod5b-campaign-driver.log 2>&1 &
# Stop it with: touch LRB-experiment-data-slurm/local-runner/PROD5B_STOP
set -u
WORKTREE=/pscratch/sd/s/sohan100/worktrees/LRB-Simulations-sdim-2bc490701489f8e7/one-ancilla-prod5b
DATA=/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm
STOP=$DATA/local-runner/PROD5B_STOP
RUNS_FILE=$DATA/local-runner/prod5b-runs.txt   # one "RUN NAME THREADS STRICT" per line
cd "$WORKTREE"
MAX_QUEUED=3
MAX_SUBMITS=30
declare -A SUBMITS

complete() {
    local f n=0
    for f in "$DATA/$1"/progress/*/done.txt; do
        [[ -f "$f" ]] || return 1
        [[ "$(tr -d '[:space:]' < "$f")" == "1" ]] || return 1
        n=$((n + 1))
    done
    (( n > 0 ))
}

echo "$(date) PROD5B driver started, pid $$"
declare -A VALIDATED
while true; do
    if [[ -e "$STOP" ]]; then
        echo "$(date) STOP file found; exiting (queued jobs left alone)"
        exit 0
    fi
    if ! queued=$(squeue -h -u "$USER" -o %j 2>/dev/null); then
        sleep 60
        continue
    fi
    n_queued=$(grep -c '^lrbP' <<< "$queued")
    all_done=1
    while read -r run name threads strict; do
        [[ -z "$run" || "$run" == \#* ]] && continue
        if complete "$run"; then
            if [[ -z "${VALIDATED[$run]:-}" ]]; then
                extra=()
                [[ "$strict" == strict ]] && extra=(--strict-production-grid)
                PYTHONNOUSERSITE=1 .venv-sdim140/bin/python scripts/validate_lrb_run.py \
                    "$run" --batch-size 100000 --require-complete "${extra[@]}" \
                    > "$DATA/$run/validation.txt" 2>&1
                echo "$(date) $name complete; validate status $?"
                VALIDATED[$run]=1
            fi
            continue
        fi
        all_done=0
        grep -qx "$name" <<< "$queued" && continue
        (( n_queued >= MAX_QUEUED )) && continue
        if (( ${SUBMITS[$name]:-0} >= MAX_SUBMITS )); then
            echo "$(date) $name hit $MAX_SUBMITS submissions; skipping it"
            continue
        fi
        dir=$DATA/$run
        if sbatch -q debug -t 00:30:00 --job-name="$name" \
                --output="$dir/slurm-%j.out" --error="$dir/slurm-%j.err" \
                --export=ALL,RUN_NAME_OVERRIDE=$run,LRB_THREADS_PER_INDEX=$threads,LRB_AUTO_RESUBMIT=0 \
                slurm/run_lrb_dem_circuit_level.sh; then
            SUBMITS[$name]=$(( ${SUBMITS[$name]:-0} + 1 ))
            n_queued=$((n_queued + 1))
            echo "$(date) submitted $name (#${SUBMITS[$name]})"
        fi
    done < "$RUNS_FILE"
    (( all_done )) && break
    sleep 120
done
echo "$(date) all PROD5B runs complete"
