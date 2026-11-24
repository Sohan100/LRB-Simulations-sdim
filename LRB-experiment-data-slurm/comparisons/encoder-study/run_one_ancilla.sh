#!/bin/bash
# One-ancilla layout: encoder search for every helper subset of size 0-2, d=3 and d=2.
cd "$(dirname "$0")"
PY=/pscratch/sd/s/sohan100/LRB-Simulations-sdim/.venv-sdim140/bin/python
mkdir -p out
for d in 2 3; do
  for h in "" 5 6 7 8 5,6 5,7 5,8 6,7 6,8 7,8; do
    tag=$(echo "h$h" | tr ',' '-')
    PYTHONNOUSERSITE=1 $PY encoder_search.py one-ancilla $d --helpers=$h --all-optimal > out/one-ancilla-d$d-$tag.out 2>&1 &
  done
done
wait
grep -h RESULT out/one-ancilla-*.out | sort
