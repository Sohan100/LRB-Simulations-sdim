#!/bin/bash
# Run the conventional-site two-ancilla campaign (folded_qutrit_conv_sites and
# folded_qubit_conv_sites: G1/G4 by a data relay on the four conventional
# ancilla sites, every site read once per round) in 30-minute debug jobs until
# every run is complete, then validate each run. Same grids as the existing
# two-ancilla campaigns: qutrit production + high-p extensions, qubit
# production + near-threshold refinements. Jobs are submitted from the
# conventional-sites worktree so they use its code; its LRB-experiment-data-slurm
# is a symlink to this data folder. Runs detached:
#   setsid nohup bash LRB-experiment-data-slurm/local-runner/conv-sites-campaign-driver.sh \
#       >> LRB-experiment-data-slurm/local-runner/conv-sites-campaign-driver.log 2>&1 &
# Stop it with: touch LRB-experiment-data-slurm/local-runner/CONV_STOP
set -u
WORKTREE=/pscratch/sd/s/sohan100/worktrees/LRB-Simulations-sdim-2bc490701489f8e7/conventional-sites
DATA=/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm
STOP=$DATA/local-runner/CONV_STOP
cd "$WORKTREE"
RUNS=(Run-2026-11-01-09-22-32-folded_qutrit_conv_sites-conv-sites-si1000-sdim140
      Run-2026-11-01-09-24-10-folded_qutrit_conv_sites-conv-sites-sd6-sdim140
      Run-2026-11-01-09-26-44-folded_qubit_conv_sites-conv-sites-si1000-sdim140
      Run-2026-11-01-09-28-00-folded_qubit_conv_sites-conv-sites-sd6-sdim140
      Run-2026-11-01-09-29-15-folded_qubit_conv_sites-conv-sites-si1000-sdim140-threshold-refine
      Run-2026-11-01-09-31-12-folded_qubit_conv_sites-conv-sites-sd6-sdim140-threshold-refine
      Run-2026-11-01-09-25-50-folded_qutrit_conv_sites-conv-sites-si1000-sdim140-highp-ext
      Run-2026-11-01-09-26-20-folded_qutrit_conv_sites-conv-sites-sd6-sdim140-highp-ext)
NAMES=(lrbc-3-si lrbc-3-sd lrbc-2-si lrbc-2-sd
       lrbcr-2-si lrbcr-2-sd lrbcx-3-si lrbcx-3-sd)
# About 250 sampler threads per node: 15 (production), 23 (SI1000 refine),
# 21 (SD6 refine), 12 (SI1000 ext) and 8 (SD6 ext) indices.
THREADS=(16 16 16 16 11 12 20 30)
MAX_QUEUED=4
MAX_SUBMITS=14
SUBMITS=(0 0 0 0 0 0 0 0)

complete() {
    local f
    for f in "$DATA/$1"/progress/*/done.txt; do
        [[ -f "$f" && "$(tr -d '[:space:]' < "$f")" == "1" ]] || return 1
    done
}

echo "$(date) conv-sites driver started, pid $$"
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
    n_queued=$(grep -c '^lrbc' <<< "$queued")
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

echo "$(date) all conv-sites runs complete; validating"
for i in "${!RUNS[@]}"; do
    extra=()
    [[ ${NAMES[$i]} == lrbc-* ]] && extra=(--strict-production-grid)
    PYTHONNOUSERSITE=1 .venv-sdim140/bin/python scripts/validate_lrb_run.py \
        "${RUNS[$i]}" --batch-size 100000 --require-complete "${extra[@]}" \
        > "$DATA/${RUNS[$i]}/validation.txt" 2>&1
    echo "$(date) validate ${NAMES[$i]}: status $?"
done
echo "$(date) conv-sites driver finished"
