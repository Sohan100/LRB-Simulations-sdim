#!/bin/bash
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=1 PYTHONNOUSERSITE=1
PY=../../.venv-sdim140/bin/python
PP=0.0001,0.0002,0.0005,0.001,0.002,0.005,0.0075,0.01,0.0125,0.015,0.02,0.025
D=0,2,4,6,10,14,18,20,22
for q in 5 7; do for n in si1000 sd6; do
  i=$([ $n = si1000 ] && echo 1 || echo 2)
  $PY 12_sample_rb.py --q $q --noise $n --p $PP --depths $D --seq-seed $((q*1000+500+i)) --tag final-prod --workers 10
done; done
$PY 12_sample_rb.py --q 5 --noise si1000 --p 0.03,0.035,0.04,0.045,0.05,0.0525,0.055,0.0575,0.06,0.0625,0.065,0.07,0.08 --depths 0,1,2 --seq-seed 5601 --tag final-ext --workers 10
$PY 12_sample_rb.py --q 7 --noise si1000 --p 0.03,0.04,0.05,0.055,0.06,0.065,0.07,0.0725,0.075,0.0775,0.08,0.085,0.09 --depths 0,1,2 --seq-seed 7601 --tag final-ext --workers 10
echo RB_DONE
