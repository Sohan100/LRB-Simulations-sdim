#!/bin/bash
# Generate the compiled-recovery run folders (opt-in --inverse-mode compiled).
# generate_circuits_folded.py rewrites LRB-experiment-data-slurm/working-folder*.txt,
# so every generation is followed by restoring the saved pointers.
set -euo pipefail
cd "$(dirname "$0")/../.."
D=non-markovian-brainstorm/compiled-runs
PY="env PYTHONNOUSERSITE=1 nice -n 10 .venv-sdim140/bin/python"
PROD_P=0.0,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.0075,0.01,0.0125,0.015,0.02,0.03,0.04,0.05
SI_HIGHP=0.015,0.0175,0.02,0.0225,0.025,0.0275,0.03,0.0325,0.035,0.04,0.045,0.05
SD6_HIGHP=0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07
gen() {  # noise ancilla-mode label depths shots probs const unif
    $PY scripts/generate_circuits_folded.py --inverse-mode compiled --noise-model "$1" --ancilla-mode "$2" \
        --custom-name "$3" --n-cliff 30 --depths "$4" --n-shots "$5" --probabilities "$6" \
        --stab-checks-const "$7" --stab-checks-unif "$8" | tail -1
    cp "$D"/wf-backup/working-folder*.txt LRB-experiment-data-slurm/
    md5sum -c --quiet "$D"/wf-backup/pointers.md5
}
FULL_C=$(seq -s, 0 22); FULL_U=$(seq -s, 1 22)
gen sd6 single single-ancilla-sd6-sdim140-compiled-recovery 0,1,2,3,4,6,10,14,18,22 1000000 $PROD_P $FULL_C $FULL_U
gen si1000 split-unif split-ancilla-si1000-sdim140-compiled-recovery 0,1,2,3,4,6,10,14,18,22 1000000 $PROD_P $FULL_C $FULL_U
gen sd6 split-unif split-ancilla-sd6-sdim140-compiled-recovery 0,1,2,3,4,6,10,14,18,22 1000000 $PROD_P $FULL_C $FULL_U
gen si1000 single single-ancilla-si1000-sdim140-compiled-recovery-highp 0,1,2,3,4 20000000 $SI_HIGHP 0,1,2,3,4 1,2,3,4
gen si1000 split-unif split-ancilla-si1000-sdim140-compiled-recovery-highp 0,1,2,3,4 20000000 $SI_HIGHP 0,1,2,3,4 1,2,3,4
gen sd6 single single-ancilla-sd6-sdim140-compiled-recovery-highp 0,1,2,3,4 20000000 $SD6_HIGHP 0,1,2,3,4 1,2,3,4
gen sd6 split-unif split-ancilla-sd6-sdim140-compiled-recovery-highp 0,1,2,3,4 20000000 $SD6_HIGHP 0,1,2,3,4 1,2,3,4
