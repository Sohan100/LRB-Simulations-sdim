#!/bin/bash
# Two-fault expansion and model envelope for the five-moment round.
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=1 PYTHONNOUSERSITE=1 QSCAL_ROUND=five
PY=${LRB_PYTHON:-../../.venv-sdim140/bin/python}
$PY 20_pair_expansion.py --q 2,3,5,7,11,13 --noise si1000,sd6 --compile production --depths 1,2,3,4 --n-seq 10 --workers 6 --tag five
$PY 23_model_pstar.py --q 2,3,5,7,11,13,17,19,23,29,31 --n 96 --workers 6 --compiles production
echo FIVE_ANALYSIS_DONE
