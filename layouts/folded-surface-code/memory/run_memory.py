"""Sample memory experiments with sinter and append the statistics to a CSV file.

The CSV doubles as a resume file: running the same command again continues where it stopped,
so the sweep can be split over several short SLURM jobs.

Example:
    python run_memory.py --noise SD6 SI1000 --codes folded rotated --distances 3 5 7 9 \
        --ps 0.001 0.002 0.003 --out stats.csv --workers 128
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main() -> None:
    import sinter

    import memory_circuits as mc

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noise", nargs="+", default=["SD6", "SI1000"])
    ap.add_argument("--codes", nargs="+", default=["folded", "rotated"])
    ap.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7, 9])
    ap.add_argument("--ps", nargs="+", type=float, required=True)
    ap.add_argument("--bases", nargs="+", default=["X", "Z"])
    ap.add_argument("--max-shots", type=float, default=1e8)
    ap.add_argument("--max-errors", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--gate", default="none", choices=["none", "H", "S"],
                    help="fold-transversal logical gate between every two rounds (folded code only)")
    ap.add_argument("--detectors", default="all", choices=["all", "sector"])
    ap.add_argument("--decoder", default="pymatching")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    gate = None if a.gate == "none" else a.gate
    tasks = list(mc.tasks(a.codes, a.distances, a.noise, a.ps, a.bases, logical_gate=gate,
                          detectors=a.detectors))
    print("%d tasks, %d workers, decoder %s, max %.0e shots / %d errors per task"
          % (len(tasks), a.workers, a.decoder, a.max_shots, a.max_errors), flush=True)
    sinter.collect(
        num_workers=a.workers,
        tasks=tasks,
        decoders=[a.decoder],
        max_shots=int(a.max_shots),
        max_errors=a.max_errors,
        save_resume_filepath=a.out,
        print_progress=not a.quiet,
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
