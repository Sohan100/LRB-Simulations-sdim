#!/bin/bash
# RB is the same physical experiment for both rounds: sample the extra p values
# into the seven-moment folders (same RB seeds, so the same RB sequences), then
# link every RB file into the five-moment folders (paired RB, as production does).
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=1 PYTHONNOUSERSITE=1
PY=../../.venv-sdim140/bin/python
$PY 12_sample_rb.py --q 5 --noise si1000 --p 0.0675,0.075 --depths 0,1,2 --seq-seed 5601 --tag final-ext --workers 6
$PY 12_sample_rb.py --q 7 --noise si1000 --p 0.06,0.0825,0.095 --depths 0,1,2 --seq-seed 7601 --tag final-ext --workers 6
for tag in final-prod final-ext; do for q in 5 7; do for n in si1000 sd6; do
  src=cache/$tag/q$q-$n-production; dst=cache/$tag/q$q-$n-production-five
  mkdir -p $dst; for f in $src/rb_p*.npz $src/rb_words.json; do ln -sf "$(realpath $f)" $dst/; done
done; done; done
echo RB_FIVE_DONE
