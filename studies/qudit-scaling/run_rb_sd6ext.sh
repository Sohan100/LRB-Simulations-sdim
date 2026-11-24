#!/bin/bash
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=1 PYTHONNOUSERSITE=1
PY=../../.venv-sdim140/bin/python
$PY 12_sample_rb.py --q 5 --noise sd6 --p 0.03,0.04,0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.1 --depths 0,1,2 --seq-seed 5602 --tag final-ext --workers 6
$PY 12_sample_rb.py --q 7 --noise sd6 --p 0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.08 --depths 0,1,2 --seq-seed 7602 --tag final-ext --workers 6
echo RB_DONE
