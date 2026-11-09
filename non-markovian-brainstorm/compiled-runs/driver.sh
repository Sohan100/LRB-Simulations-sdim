#!/bin/bash
# Keep the eight compiled-recovery runs going in 30-minute debug jobs until all
# are complete, at most two jobs queued at once (debug allows 2 running per
# user), production grids first. Each job resumes from its checkpoints. When
# everything is done, validate each run and write pooled r_L/d_L fits into the
# new production run folders. Runs detached:
#   setsid nohup bash non-markovian-brainstorm/compiled-runs/driver.sh \
#       >> non-markovian-brainstorm/compiled-runs/driver.log 2>&1 &
# Stop it by creating non-markovian-brainstorm/compiled-runs/STOP.
set -u
cd /pscratch/sd/s/sohan100/LRB-Simulations-sdim
D=non-markovian-brainstorm/compiled-runs
DATA=LRB-experiment-data-slurm
# name|run folder|threads per index
JOBS=(
  "lrbc-1-si|Run-2026-10-31-00-39-01-folded_qutrit-single-ancilla-si1000-sdim140-compiled-recovery|8"
  "lrbc-1-sd6|Run-2026-10-31-00-40-37-folded_qutrit-single-ancilla-sd6-sdim140-compiled-recovery|8"
  "lrbc-2-si|Run-2026-10-31-00-41-58-folded_qutrit_split_unif-split-ancilla-si1000-sdim140-compiled-recovery|8"
  "lrbc-2-sd6|Run-2026-10-31-00-43-31-folded_qutrit_split_unif-split-ancilla-sd6-sdim140-compiled-recovery|8"
  "lrbc-1-si-hp|Run-2026-10-31-00-45-00-folded_qutrit-single-ancilla-si1000-sdim140-compiled-recovery-highp|10"
  "lrbc-2-si-hp|Run-2026-10-31-00-45-27-folded_qutrit_split_unif-split-ancilla-si1000-sdim140-compiled-recovery-highp|10"
  "lrbc-1-sd6-hp|Run-2026-10-31-00-45-57-folded_qutrit-single-ancilla-sd6-sdim140-compiled-recovery-highp|12"
  "lrbc-2-sd6-hp|Run-2026-10-31-00-46-19-folded_qutrit_split_unif-split-ancilla-sd6-sdim140-compiled-recovery-highp|12"
)
MAX_ACTIVE=2
MAX_SUBMITS=8
declare -A SUBMITS

complete() {
    local f
    for f in "$DATA/$1"/progress/*/done.txt; do
        [[ "$(tr -d '[:space:]' < "$f")" == "1" ]] || return 1
    done
}

echo "$(date '+%F %T') driver started, pid $$"
echo $$ > "$D/driver.pid"
while true; do
    [[ -e "$D/STOP" ]] && { echo "$(date '+%F %T') STOP file found; exiting"; exit 0; }
    queue_ok=1
    names=$(squeue -h -u "$USER" -o %j 2>/dev/null) || queue_ok=0
    active=$(grep -c '^lrbc-' <<< "$names")
    all_done=1
    for entry in "${JOBS[@]}"; do
        IFS='|' read -r name run threads <<< "$entry"
        complete "$run" && continue
        all_done=0
        (( queue_ok )) || continue
        grep -qx "$name" <<< "$names" && continue
        (( active < MAX_ACTIVE )) || continue
        n=${SUBMITS[$name]:-0}
        if (( n >= MAX_SUBMITS )); then
            echo "$(date '+%F %T') $name hit $MAX_SUBMITS submissions; skipping it"
            continue
        fi
        dir=$PWD/$DATA/$run
        if sbatch -q debug -t 00:30:00 --job-name="$name" \
                --output="$dir/slurm-%j.out" --error="$dir/slurm-%j.err" \
                --export=ALL,RUN_NAME_OVERRIDE=$run,LRB_THREADS_PER_INDEX=$threads,LRB_AUTO_RESUBMIT=0 \
                slurm/run_lrb_dem_circuit_level.sh; then
            SUBMITS[$name]=$((n + 1))
            active=$((active + 1))
            echo "$(date '+%F %T') submitted $name (#$((n + 1)))"
        fi
    done
    (( all_done )) && break
    sleep 120
done

echo "$(date '+%F %T') all eight runs complete; validating"
for entry in "${JOBS[@]}"; do
    IFS='|' read -r name run threads <<< "$entry"
    PYTHONNOUSERSITE=1 .venv-sdim140/bin/python scripts/validate_lrb_run.py "$run" \
        --batch-size 100000 --require-complete > "$D/validate-$name.json" 2>&1
    echo "$(date '+%F %T') validate $name: exit $?"
done

echo "$(date '+%F %T') pooled fits (production + high-p merged)"
run_of() { local name run threads; IFS='|' read -r name run threads <<< "${JOBS[$1]}"; echo "$run"; }
fit() {  # prod-index highp-index split
    local prod highp
    prod=$(run_of "$1"); highp=$(run_of "$2")
    PYTHONNOUSERSITE=1 .venv-sdim140/bin/python scripts/fit_lrb_pooled.py "$DATA/$prod" \
        --highp-run "$DATA/$highp" --split "$3" --out-dir "$DATA/$prod/results/plots/pooled_fits" \
        > "$D/pooled-$(basename "$prod").txt" 2>&1
    echo "$(date '+%F %T') pooled fit $prod: exit $?"
}
fit 0 4 0.015
fit 1 6 0.025
fit 2 5 0.015
fit 3 7 0.025
echo "$(date '+%F %T') driver finished"
