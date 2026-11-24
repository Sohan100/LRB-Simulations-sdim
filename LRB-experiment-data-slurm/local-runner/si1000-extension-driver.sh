#!/bin/bash
# Keep the SI1000 high-p extension running in 30-minute debug jobs until both
# runs are complete, then merge them into the thresholds and rewrite
# comparisons/pseudo-thresholds-current.{md,csv}. Runs detached so it
# survives session restarts:
#   setsid nohup bash LRB-experiment-data-slurm/local-runner/si1000-extension-driver.sh \
#       >> LRB-experiment-data-slurm/local-runner/si1000-extension-driver.log 2>&1 &
set -u
cd /pscratch/sd/s/sohan100/LRB-Simulations-sdim
RUNS=(Run-2026-10-30-10-01-06-folded_qutrit-single-ancilla-si1000-sdim140-highp-ext
      Run-2026-10-30-10-01-28-folded_qutrit_split_unif-split-ancilla-si1000-sdim140-highp-ext)
NAMES=(lrbx-single-si1000 lrbx-split-si1000)
MAX_SUBMITS=8
SUBMITS=(0 0)

complete() {
    local f
    for f in LRB-experiment-data-slurm/$1/progress/*/done.txt; do
        [[ "$(tr -d '[:space:]' < "$f")" == "1" ]] || return 1
    done
}

echo "$(date) driver started, pid $$"
while true; do
    all_done=1
    for i in 0 1; do
        run=${RUNS[$i]}
        name=${NAMES[$i]}
        complete "$run" && continue
        all_done=0
        # Submit only when squeue answered and shows no job of this name.
        if jobs=$(squeue -h -u "$USER" -n "$name" -o %i 2>/dev/null) \
                && [[ -z "$jobs" ]]; then
            if (( SUBMITS[i] >= MAX_SUBMITS )); then
                echo "$(date) $name hit $MAX_SUBMITS submissions; giving up"
                exit 1
            fi
            dir=$PWD/LRB-experiment-data-slurm/$run
            sbatch -q debug -t 00:30:00 --job-name="$name" \
                --output="$dir/slurm-%j.out" --error="$dir/slurm-%j.err" \
                --export=ALL,RUN_NAME_OVERRIDE=$run,LRB_THREADS_PER_INDEX=10,LRB_AUTO_RESUBMIT=0 \
                slurm/run_lrb_dem_circuit_level.sh
            SUBMITS[i]=$((SUBMITS[i] + 1))
            echo "$(date) submitted $name (#${SUBMITS[i]})"
        fi
    done
    (( all_done )) && break
    sleep 120
done

echo "$(date) both extensions complete; merging"
PYTHONNOUSERSITE=1 .venv-sdim140/bin/python \
    scripts/merge_highp_threshold_extension.py si1000
echo "$(date) merge finished with status $?"
