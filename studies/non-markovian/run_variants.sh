#!/bin/bash
# Login-node protocol-variant runs used in REPORT.md (about 30 min total, 8 workers, nice 10).
cd "$(dirname "$0")"
PY="env PYTHONNOUSERSITE=1 nice -n 10 ${LRB_PYTHON:-../../.venv-sdim140/bin/python}"
$PY variant_sim.py --noise sd6 --p 0.002 --shots 1000000 --workers 8
$PY variant_sim.py --noise si1000 --p 0.005 --shots 1000000 --workers 8
$PY variant_sim.py --noise sd6 --p 0.025 --shots 2000000 --workers 8 --depths 0,1,2,3,4,5,6
