# Shared compute allocation (maintained by the lead session)

The lead keeps ONE single-node interactive allocation alive for agents' heavier
runs and renews it about hourly. Find the current one with:

    squeue -u sohan100 -n q512shared -h -o "%i %M %L %N"

(job id, elapsed, time left, node). If one is RUNNING with more than 10 minutes
left, run heavy jobs on it with:

    srun --jobid=<JOBID> --overlap -N 1 -n 1 -c <CORES> bash -c "module load python/3.11; cd <WORKTREE>; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; python ..."

Rules: at most 24 cores per agent at a time, each run at most 20 minutes, use
Bash run_in_background for long runs, and write outputs under
studies/correction-decoders/qubit/results/<your-WP>/. The lead's main grid
uses about 120 cores until it finishes.

If no allocation has more than 10 minutes left, run on the login node with at
most 4 processes, nice -n 19, OMP_NUM_THREADS=1, each run at most 15 minutes,
and scale statistics down; or write the exact command you need to
studies/correction-decoders/qubit/results/requests/<WP>-<n>.sh and describe it
in your notes as PENDING so the lead can run it later. Never call sbatch or
salloc yourself.

## IMPORTANT: BLAS threads

A site hook in the user's Python environment imports numpy at interpreter
startup, and numpy's OpenBLAS then starts 64 threads per process. Setting
thread variables inside a Python script is too late. Always launch Python with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` exported in the
shell (forked multiprocessing workers inherit it), or re-exec the interpreter
with them set as `studies/correction-decoders/qubit/scripts/run_grid.py` does. Do NOT call
`threadpoolctl.threadpool_limits` inside forked workers: the startup hook also
loads numba, and threadpoolctl aborts the worker. Without single-threaded BLAS
the small matrix products in the MLD decoder run about 7x slower and overload
the node.

## CPU binding (reported by work package B)

Overlapping `srun --overlap -c N` steps can all be bound to the same few CPUs
(0-11 and 128-139), which starves every worker. Add `--cpu-bind=none` to every
`srun --jobid=<id> --overlap ...` command, or request `-c 256` and still run
at most 24 worker processes.
