#!/bin/bash
# Run the qubit [[5,1,2]] sdim140 campaign (4 production + 4 high-p extension
# runs, same grids as the qutrit campaign) in 30-minute debug jobs until every
# run is complete, then validate each run. Jobs are submitted from the qubit
# worktree so they use its code; the worktree's LRB-experiment-data-slurm is a
# symlink to this data folder. Runs detached so it survives Claude restarts:
#   setsid nohup bash LRB-experiment-data-slurm/local-runner/qubit-campaign-driver.sh \
#       >> LRB-experiment-data-slurm/local-runner/qubit-campaign-driver.log 2>&1 &
# Stop it with: touch LRB-experiment-data-slurm/local-runner/QUBIT_STOP
set -u
WORKTREE=/pscratch/sd/s/sohan100/worktrees/LRB-Simulations-sdim-2bc490701489f8e7/qubit-lrbd-sdim140
DATA=/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm
STOP=$DATA/local-runner/QUBIT_STOP
cd "$WORKTREE"
# Production (done 10:56) first, then the near-threshold refinements (the
# qubit crossings sit at p = 0.003-0.021, below the high-p extension ranges),
# then the high-p extensions, kept for parity with the qutrit campaign.
RUNS=(Run-2026-10-31-10-12-55-folded_qubit-single-ancilla-si1000-sdim140
      Run-2026-10-31-10-12-58-folded_qubit_split_unif-split-ancilla-si1000-sdim140
      Run-2026-10-31-10-13-03-folded_qubit-single-ancilla-sd6-sdim140
      Run-2026-10-31-10-13-07-folded_qubit_split_unif-split-ancilla-sd6-sdim140
      Run-2026-10-31-11-09-27-folded_qubit-single-ancilla-si1000-sdim140-threshold-refine
      Run-2026-10-31-11-09-29-folded_qubit_split_unif-split-ancilla-si1000-sdim140-threshold-refine
      Run-2026-10-31-11-09-31-folded_qubit-single-ancilla-sd6-sdim140-threshold-refine
      Run-2026-10-31-11-09-33-folded_qubit_split_unif-split-ancilla-sd6-sdim140-threshold-refine
      Run-2026-10-31-10-12-57-folded_qubit-single-ancilla-si1000-sdim140-highp-ext
      Run-2026-10-31-10-13-01-folded_qubit_split_unif-split-ancilla-si1000-sdim140-highp-ext
      Run-2026-10-31-10-13-05-folded_qubit-single-ancilla-sd6-sdim140-highp-ext
      Run-2026-10-31-10-13-09-folded_qubit_split_unif-split-ancilla-sd6-sdim140-highp-ext)
NAMES=(lrbq-1-si lrbq-2-si lrbq-1-sd lrbq-2-sd
       lrbqr-1-si lrbqr-2-si lrbqr-1-sd lrbqr-2-sd
       lrbqx-1-si lrbqx-2-si lrbqx-1-sd lrbqx-2-sd)
# About 250 sampler threads per node: 15, 23 (SI1000 refine), 21 (SD6
# refine), 12 (SI1000 ext) and 8 (SD6 ext) indices.
THREADS=(16 16 16 16 11 11 12 12 20 20 30 30)
MAX_QUEUED=3
MAX_SUBMITS=10
SUBMITS=(0 0 0 0 0 0 0 0 0 0 0 0)

complete() {
    local f
    for f in "$DATA/$1"/progress/*/done.txt; do
        [[ -f "$f" && "$(tr -d '[:space:]' < "$f")" == "1" ]] || return 1
    done
}

echo "$(date) qubit driver started, pid $$"
while true; do
    if [[ -e "$STOP" ]]; then
        echo "$(date) STOP file found; exiting (queued jobs left alone)"
        exit 0
    fi
    # Skip this round if squeue does not answer.
    if ! queued=$(squeue -h -u "$USER" -o %j 2>/dev/null); then
        sleep 60
        continue
    fi
    n_queued=$(grep -c '^lrbq' <<< "$queued")
    all_done=1
    for i in "${!RUNS[@]}"; do
        run=${RUNS[$i]}
        name=${NAMES[$i]}
        complete "$run" && continue
        all_done=0
        grep -qx "$name" <<< "$queued" && continue
        (( n_queued >= MAX_QUEUED )) && continue
        if (( SUBMITS[i] >= MAX_SUBMITS )); then
            echo "$(date) $name hit $MAX_SUBMITS submissions; skipping it"
            continue
        fi
        dir=$DATA/$run
        if sbatch -q debug -t 00:30:00 --job-name="$name" \
                --output="$dir/slurm-%j.out" --error="$dir/slurm-%j.err" \
                --export=ALL,RUN_NAME_OVERRIDE=$run,LRB_THREADS_PER_INDEX=${THREADS[$i]},LRB_AUTO_RESUBMIT=0 \
                slurm/run_lrb_dem_circuit_level.sh; then
            SUBMITS[i]=$((SUBMITS[i] + 1))
            n_queued=$((n_queued + 1))
            echo "$(date) submitted $name (#${SUBMITS[i]})"
        fi
    done
    (( all_done )) && break
    sleep 120
done

echo "$(date) all qubit runs complete; validating"
for i in "${!RUNS[@]}"; do
    extra=()
    [[ ${NAMES[$i]} == lrbq-* ]] && extra=(--strict-production-grid)
    PYTHONNOUSERSITE=1 .venv-sdim140/bin/python scripts/validate_lrb_run.py \
        "${RUNS[$i]}" --batch-size 100000 --require-complete "${extra[@]}" \
        > "$DATA/${RUNS[$i]}/validation.txt" 2>&1
    echo "$(date) validate ${NAMES[$i]}: status $?"
done
echo "$(date) qubit driver finished"
