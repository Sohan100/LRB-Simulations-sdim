"""Run DEM probability indices on the login node until SLURM takes over.

While a run's SLURM job (``slurm/run_lrb_dem_circuit_level.sh``) is still
queued, this driver works through the run's unfinished probability indices
with a few niced local workers. Each worker is an ordinary
``scripts/run_lrb_experiment.py`` process, so it checkpoints after every
``LRB_BATCH_SIZE`` shots and holds the index's ``.worker.lock``.

As soon as a SLURM job for a run is RUNNING, the driver kills that run's
local workers. Their batch workers wait on the lock (see
``LRB_WORKER_LOCK_WAIT_SECONDS``), take it when the local worker dies, and
resume from the last committed checkpoint, so at most one uncommitted batch
per index is lost. Runs are handled in the order given, highest probability
(slowest) index first.

Usage (from the repository root, detached so it survives the session):

    setsid nohup .venv-sdim140/bin/python -u \\
        scripts/run_lrb_dem_local_until_slurm.py <Run-folder> [...] \\
        > LRB-experiment-data-slurm/local-runner/driver.log 2>&1 &

Stop it with ``touch LRB-experiment-data-slurm/local-runner/STOP`` (workers
are killed and resume later from their checkpoints). Touch
``<run>/.local_handoff`` to make the driver yield one run by hand.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "LRB-experiment-data-slurm"
STATE_DIR = RUNS_ROOT / "local-runner"
STOP_FILE = STATE_DIR / "STOP"
# Must match slurm/run_lrb_dem_circuit_level.sh: the checkpoint fingerprint
# includes the batch size, so a mismatch would block the SLURM resume.
BATCH_SIZE = "100000"
SLURM_ACTIVE_STATES = "RUNNING,CONFIGURING"
MAX_FAILURES_PER_INDEX = 2


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


@dataclass
class Task:
    run: str
    index: int
    probability: str
    process: subprocess.Popen | None = None
    log_file: object = None
    killed_by_driver: bool = False


@dataclass
class RunState:
    name: str
    path: Path
    probabilities: list[str]
    failures: dict[int, int] = field(default_factory=dict)

    def done(self, index: int) -> bool:
        marker = self.path / "progress" / str(index) / "done.txt"
        try:
            return marker.read_text().strip() == "1"
        except FileNotFoundError:
            return False

    def num_done(self) -> int:
        return sum(self.done(i) for i in range(len(self.probabilities)))

    def complete(self) -> bool:
        return self.num_done() == len(self.probabilities)

    def shots_remaining(self, index: int) -> str:
        counter = self.path / "progress" / str(index) / "shots_processed.txt"
        try:
            return counter.read_text().strip()
        except FileNotFoundError:
            return "?"


def index_locked(run: RunState, index: int) -> bool:
    """Whether another process currently holds the index's worker lock."""
    lock_path = run.path / "progress" / str(index) / ".worker.lock"
    if not lock_path.exists():
        return False
    with open(lock_path, "a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return False


class SlurmWatcher:
    """Map the user's active SLURM jobs to the run folders they process."""

    def __init__(self) -> None:
        self.user = os.environ.get("USER", "")
        self._job_runs: dict[str, str | None] = {}

    def _run_of_job(self, job_id: str) -> str | None:
        if job_id not in self._job_runs:
            result = subprocess.run(
                ["scontrol", "show", "job", "-d", job_id],
                capture_output=True, text=True, timeout=60, check=False)
            run = None
            for token in result.stdout.replace(",", " ").split():
                if token.startswith("RUN_NAME_OVERRIDE="):
                    run = token.split("=", 1)[1]
                    break
            if result.returncode != 0:
                return None
            self._job_runs[job_id] = run
        return self._job_runs[job_id]

    def active_runs(self) -> dict[str, str] | None:
        """Return {run: job_id} for running jobs, or None if SLURM is down."""
        try:
            result = subprocess.run(
                ["squeue", "-h", "-u", self.user, "-t", SLURM_ACTIVE_STATES,
                 "-o", "%i"],
                capture_output=True, text=True, timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        active = {}
        for job_id in result.stdout.split():
            run = self._run_of_job(job_id)
            if run:
                active[run] = job_id
        return active


class LocalDriver:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.runs: list[RunState] = []
        for name in args.runs:
            path = RUNS_ROOT / name
            if not (path / "probs.txt").is_file():
                raise SystemExit(f"Missing run folder or probs.txt: {path}")
            probabilities = [
                p.strip() for p in (path / "probs.txt").read_text().split(",")
            ]
            self.runs.append(RunState(name, path, probabilities))
        self.watcher = SlurmWatcher()
        self.tasks: list[Task] = []
        self.stopping = False

    def _handoff_runs(self, active: dict[str, str]) -> set[str]:
        owned = set(active)
        for run in self.runs:
            if (run.path / ".local_handoff").exists():
                owned.add(run.name)
        return owned

    def _kill(self, task: Task, reason: str) -> None:
        if task.process is None or task.process.poll() is not None:
            return
        task.killed_by_driver = True
        log(f"Stopping local worker {task.run} idx {task.index} ({reason}); "
            f"shots remaining at last checkpoint: "
            f"{self._run(task.run).shots_remaining(task.index)}")
        os.killpg(task.process.pid, signal.SIGTERM)
        try:
            task.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(task.process.pid, signal.SIGKILL)
            task.process.wait()

    def _run(self, name: str) -> RunState:
        return next(run for run in self.runs if run.name == name)

    def _start(self, run: RunState, index: int) -> None:
        probability = run.probabilities[index]
        log_dir = run.path / "logs_local"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"run_p_idx{index}_p{probability}.log"
        log_file = open(log_path, "a")
        log_file.write(f"\n===== local start {datetime.now().isoformat()} "
                       f"on {os.uname().nodename} =====\n")
        log_file.flush()
        threads = str(self.args.threads)
        env = dict(
            os.environ,
            PYTHONNOUSERSITE="1",
            PYTHONUNBUFFERED="1",
            LRB_BATCH_SIZE=BATCH_SIZE,
            LRB_WORKER_LOCK_WAIT_SECONDS="0",
            NUMBA_NUM_THREADS=threads,
            OMP_NUM_THREADS=threads,
        )
        env.pop("SLURM_JOB_ID", None)
        nice = self.args.nice
        process = subprocess.Popen(
            [sys.executable, "-u", "scripts/run_lrb_experiment.py",
             run.name, str(index), "--simulation-backend", "dem"],
            cwd=PROJECT_ROOT, env=env, stdout=log_file,
            stderr=subprocess.STDOUT, start_new_session=True,
            preexec_fn=lambda: os.nice(nice))
        self.tasks.append(Task(run.name, index, probability, process,
                               log_file))
        log(f"Started local worker {run.name} idx {index} (p={probability}) "
            f"pid {process.pid}; shots remaining at last checkpoint: "
            f"{run.shots_remaining(index)}")

    def _reap(self) -> None:
        for task in list(self.tasks):
            rc = task.process.poll()
            if rc is None:
                continue
            task.log_file.close()
            self.tasks.remove(task)
            run = self._run(task.run)
            if task.killed_by_driver:
                continue
            if rc == 0 and run.done(task.index):
                log(f"Finished {task.run} idx {task.index} "
                    f"({run.num_done()}/{len(run.probabilities)} done)")
                continue
            failures = run.failures.get(task.index, 0) + 1
            run.failures[task.index] = failures
            log(f"Worker {task.run} idx {task.index} exited with {rc} "
                f"(failure {failures}/{MAX_FAILURES_PER_INDEX}); see "
                f"{run.path / 'logs_local'}")

    def _next_index(self, owned: set[str]) -> tuple[RunState, int] | None:
        busy = {(task.run, task.index) for task in self.tasks}
        for run in self.runs:
            if run.name in owned:
                continue
            order = sorted(range(len(run.probabilities)),
                           key=lambda i: float(run.probabilities[i]),
                           reverse=True)
            for index in order:
                if ((run.name, index) in busy or run.done(index)
                        or run.failures.get(index, 0)
                        >= MAX_FAILURES_PER_INDEX
                        or index_locked(run, index)):
                    continue
                return run, index
        return None

    def _write_status(self, active: dict[str, str] | None,
                      owned: set[str]) -> None:
        lines = [f"updated {datetime.now().isoformat()} on "
                 f"{os.uname().nodename}, driver pid {os.getpid()}"]
        if active is None:
            lines.append("SLURM query failed this poll")
        for run in self.runs:
            if run.name in (active or {}):
                owner = f"SLURM job {active[run.name]}"
            elif run.name in owned:
                owner = "handed off by .local_handoff"
            else:
                owner = "local"
            local = sorted(t.index for t in self.tasks if t.run == run.name)
            lines.append(f"{run.name}: {run.num_done()}/"
                         f"{len(run.probabilities)} done, owner {owner}, "
                         f"local indices {local}")
        (STATE_DIR / "status.txt").write_text("\n".join(lines) + "\n")

    def _request_stop(self, signum, _frame) -> None:
        log(f"Received signal {signum}; stopping workers.")
        self.stopping = True

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self._request_stop)
        signal.signal(signal.SIGINT, self._request_stop)
        STATE_DIR.mkdir(exist_ok=True)
        (STATE_DIR / "driver.pid").write_text(f"{os.getpid()}\n")
        log(f"Driver on {os.uname().nodename}: {self.args.workers} workers x "
            f"{self.args.threads} threads, nice {self.args.nice}, poll "
            f"{self.args.poll_seconds} s, runs: "
            + ", ".join(run.name for run in self.runs))

        while True:
            self._reap()
            if self.stopping or STOP_FILE.exists():
                for task in list(self.tasks):
                    self._kill(task, "driver stop")
                self._reap()
                log("Driver stopped; workers resume from checkpoints later.")
                return 0

            active = self.watcher.active_runs()
            owned = self._handoff_runs(active or {})
            for task in list(self.tasks):
                if task.run in owned:
                    reason = (f"SLURM job {active[task.run]} is running"
                              if task.run in (active or {})
                              else ".local_handoff present")
                    self._kill(task, reason)
            self._reap()

            # If SLURM cannot be queried, keep current workers but start
            # nothing new; a batch job that starts meanwhile waits on the
            # lock until the next successful poll hands the index over.
            if active is not None:
                while len(self.tasks) < self.args.workers:
                    choice = self._next_index(owned)
                    if choice is None:
                        break
                    self._start(*choice)
            self._write_status(active, owned)

            if not self.tasks:
                if all(run.complete() for run in self.runs):
                    log("All runs complete.")
                    return 0
                # Keep polling while SLURM or another worker holds unfinished
                # work, so a preempted-and-requeued run is picked up again.
                if all(run.done(i) or run.failures.get(i, 0)
                       >= MAX_FAILURES_PER_INDEX
                       for run in self.runs
                       for i in range(len(run.probabilities))):
                    log("Only repeatedly failing indices are left; see "
                        "their logs_local/ files.")
                    return 1
            time.sleep(self.args.poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="Run folder names, in order.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return LocalDriver(parser.parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())
