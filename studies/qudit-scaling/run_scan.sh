#!/bin/bash
# Coarse crossing scan: q = 5, 7 (and 2, 3 for a matched comparison), SI1000 and SD6,
# depths 0..6, 30 sequences, 2e5 shots, production compile.
cd "$(dirname "$0")"
PY="env PYTHONNOUSERSITE=1 ../../.venv-sdim140/bin/python"
P=0.001,0.002,0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.06,0.08,0.1,0.12,0.15
for q in 5 7 3 2; do
  for noise in si1000 sd6; do
    $PY 10_sample_lrb.py --q $q --noise $noise --p $P --shots 200000 --seq-seed $((100*q+1)) --tag scan --workers 12
  done
done
echo SCAN_DONE
