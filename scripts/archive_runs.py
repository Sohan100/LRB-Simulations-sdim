"""Pack the raw data of LRB run folders into xz archives under archive/runs/, and list them.

A run's raw data are its circuits (``experiments/``) and its committed checkpoints (``progress/``
and ``progress_rb_comparison/``): the per-sequence outcome tallies behind every fit and every
sequence bootstrap. They are plain text and small integer arrays, so xz shrinks them by a factor of
a few hundred (a run's 400 to 500 MB of circuits become about 1 MB). ``results/`` is already in git
and is not repacked.

An archive ``archive/runs/<run>.tar.xz`` holds ``<run>/`` with

* the run's top-level ``*.txt`` files: the grid, ``run_instructions.txt``, ``validation.txt``,
  ``inverse_mode.txt``;
* ``experiments/``: every ``.chp`` circuit file;
* ``progress*/<index>/``: the committed checkpoint (for crash-safe runs the ``.checkpoint_current``
  link, its generation and the links into it), ``completion_manifest.json`` and the ``done.txt``
  and elapsed-time markers;
* any other small files of the run folder (for example a run-local ``archive/``).

It leaves out ``results/`` (in git), superseded checkpoint generations, ``timing_metrics.csv``,
locks, and job logs (``logs_*/``, ``slurm-*.out``, ``*.err``). Restoring a run is plain tar:

    tar -xJf archive/runs/<run>.tar.xz -C LRB-experiment-data-slurm/

Usage:

    python scripts/archive_runs.py pack LRB-experiment-data-slurm/Run-... [Run-... ...]
    python scripts/archive_runs.py list
    python scripts/archive_runs.py index      # rebuild index.json from the archives
"""
from __future__ import annotations

import argparse
import fcntl
import fnmatch
import glob
import json
import os
import subprocess
import sys
import tarfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "archive", "runs")
INDEX = "index.json"

SKIP_DIRS = ("results", "logs_*", "__pycache__")
SKIP_FILES = ("timing_metrics.csv", "*.lock", ".worker.lock", "slurm-*.out", "slurm-*.err",
              "*.out", "*.err", "*.pid", "STOP*")


def _skipped(name: str, patterns) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _current_generation(index_dir: str) -> str | None:
    link = os.path.join(index_dir, ".checkpoint_current")
    if not os.path.islink(link):
        return None
    return os.path.normpath(os.readlink(link))


def run_members(run: str) -> list[str]:
    """Paths (relative to the run's parent folder) that go into the archive, in order."""
    parent, name = os.path.split(os.path.abspath(run))
    members: list[str] = []

    def add(path: str) -> None:
        members.append(os.path.relpath(path, parent))

    for dirpath, dirnames, filenames in os.walk(os.path.join(parent, name)):
        rel = os.path.relpath(dirpath, os.path.join(parent, name))
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth == 0:
            dirnames[:] = [d for d in dirnames if not _skipped(d, SKIP_DIRS)]
        # crash-safe checkpoints: keep only the generation .checkpoint_current points to
        if os.path.basename(dirpath) == ".checkpoint_generations":
            current = _current_generation(os.path.dirname(dirpath))
            keep = os.path.basename(current) if current else None
            dirnames[:] = [d for d in dirnames if d == keep]
        dirnames.sort()
        for filename in sorted(filenames):
            if _skipped(filename, SKIP_FILES):
                continue
            add(os.path.join(dirpath, filename))
        # links to directories (.checkpoint_current, unif_check_data, ...) are listed in
        # dirnames by os.walk but not followed; store them as links
        for d in list(dirnames):
            if os.path.islink(os.path.join(dirpath, d)):
                add(os.path.join(dirpath, d))
                dirnames.remove(d)
    return members


def raw_size(run: str, members: list[str]) -> int:
    parent = os.path.dirname(os.path.abspath(run))
    return sum(os.lstat(os.path.join(parent, m)).st_size for m in members)


def pack(run: str, out_dir: str, threads: int) -> dict:
    run = os.path.abspath(run)
    parent, name = os.path.split(run)
    if not (os.path.isdir(os.path.join(run, "experiments")) or os.path.isdir(os.path.join(run, "progress"))):
        raise SystemExit(f"{name}: no experiments/ or progress/ to archive")
    members = run_members(run)
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, f"{name}.tar.xz")
    tmp = target + ".tmp"
    tar = subprocess.Popen(
        ["tar", "-C", parent, "--no-recursion", "--owner=0", "--group=0", "--numeric-owner",
         "--null", "-T", "-", "-cf", "-"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    with open(tmp, "wb") as fh:
        xz = subprocess.Popen(["xz", "-9e", f"-T{threads}", "-c"], stdin=tar.stdout, stdout=fh)
        tar.stdout.close()
        tar.stdin.write(b"\0".join(m.encode() for m in members) + b"\0")
        tar.stdin.close()
        if tar.wait() or xz.wait():
            os.unlink(tmp)
            raise SystemExit(f"{name}: tar or xz failed")
    # check: the archive lists exactly the members, with their sizes
    with tarfile.open(tmp, "r:xz") as archive:
        listed = {info.name: info for info in archive.getmembers()}
    missing = [m for m in members if m not in listed]
    wrong = [m for m in members if m in listed and listed[m].isfile()
             and listed[m].size != os.lstat(os.path.join(parent, m)).st_size]
    if missing or wrong or len(listed) != len(members):
        os.unlink(tmp)
        raise SystemExit(f"{name}: archive check failed ({len(missing)} missing, {len(wrong)} wrong size)")
    os.replace(tmp, target)
    entry = {
        "run": name,
        "archive": os.path.relpath(target, REPO),
        "members": len(members),
        "circuit_files": sum(1 for m in members if f"{os.sep}experiments{os.sep}" in m and m.endswith(".chp")),
        "raw_bytes": raw_size(run, members),
        "archive_bytes": os.path.getsize(target),
    }
    for key in ("code_name", "noise_model"):
        path = os.path.join(run, f"{key}.txt")
        entry[key] = open(path).read().strip() if os.path.exists(path) else None
    update_index(out_dir, entry)
    return entry


def _write_index(out_dir: str, index: dict) -> None:
    path = os.path.join(out_dir, INDEX)
    with open(path + ".tmp", "w") as fh:
        json.dump(dict(sorted(index.items())), fh, indent=1)
        fh.write("\n")
    os.replace(path + ".tmp", path)


def update_index(out_dir: str, entry: dict) -> None:
    path = os.path.join(out_dir, INDEX)
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index = json.load(open(path)) if os.path.exists(path) else {}
        index[entry["run"]] = entry
        _write_index(out_dir, index)


def describe_archive(path: str) -> dict:
    """Index entry of an existing archive, read from the archive itself."""
    name = os.path.basename(path)[: -len(".tar.xz")]
    entry = {"run": name, "archive": os.path.relpath(path, REPO), "members": 0, "circuit_files": 0,
             "raw_bytes": 0, "archive_bytes": os.path.getsize(path), "code_name": None,
             "noise_model": None}
    with tarfile.open(path, "r:xz") as archive:
        for info in archive:
            entry["members"] += 1
            entry["raw_bytes"] += info.size
            if "/experiments/" in info.name and info.name.endswith(".chp"):
                entry["circuit_files"] += 1
            for key in ("code_name", "noise_model"):
                if info.name == f"{name}/{key}.txt":
                    entry[key] = archive.extractfile(info).read().decode().strip()
    return entry


def rebuild_index(out_dir: str) -> None:
    index = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "*.tar.xz"))):
        entry = describe_archive(path)
        index[entry["run"]] = entry
    _write_index(out_dir, index)


def list_archives(out_dir: str) -> None:
    path = os.path.join(out_dir, INDEX)
    index = json.load(open(path)) if os.path.exists(path) else {}
    total_raw = total_xz = 0
    for entry in index.values():
        total_raw += entry["raw_bytes"]
        total_xz += entry["archive_bytes"]
        print(f"{entry['archive_bytes'] / 1e6:7.2f} MB  {entry['raw_bytes'] / 1e6:8.1f} MB raw  "
              f"{entry['circuit_files']:6d} circuits  {entry['run']}")
    print(f"{total_xz / 1e6:7.2f} MB  {total_raw / 1e6:8.1f} MB raw  in {len(index)} archives")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_pack = sub.add_parser("pack", help="archive one or more run folders")
    p_pack.add_argument("runs", nargs="+")
    p_pack.add_argument("--out", default=DEFAULT_OUT)
    p_pack.add_argument("--threads", type=int, default=4)
    p_list = sub.add_parser("list", help="list the archives in the index")
    p_list.add_argument("--out", default=DEFAULT_OUT)
    p_index = sub.add_parser("index", help="rebuild index.json from the archives themselves")
    p_index.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.command == "pack":
        for run in args.runs:
            entry = pack(run, args.out, args.threads)
            print(f"{entry['run']}: {entry['members']} files, {entry['raw_bytes'] / 1e6:.1f} MB "
                  f"-> {entry['archive_bytes'] / 1e6:.2f} MB", flush=True)
    elif args.command == "index":
        rebuild_index(args.out)
        list_archives(args.out)
    else:
        list_archives(args.out)


if __name__ == "__main__":
    sys.exit(main())
