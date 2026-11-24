"""Pin the const-0 (terminal-check) protocol from the production files and
validate the exact engine against the production tallies.

For each of the four one-ancilla production runs (qutrit/qubit x SI1000/SD6):

1. Recover every sequence's Clifford words from the run's own LRB files
   (``scripts/verify_default_circuits_unchanged.decode_lrb_words``).
2. Rebuild each const-0 circuit from those words with ``tclib.build_circuit``
   (ideal encoder, 2m noisy Cliffords = forward + mirror inverse, no check
   round, ideal X readout) and compare it byte for byte with the file in
   ``experiments/LRB_const0``. A match at every depth proves what the
   simulated const-0 circuit contains.
3. Scan the const-0 files for anything that would contradict "no mid-circuit
   check": measurements or resets before the terminal readout, gates on the
   ancilla wires, noise inside the encoder or the terminal readout.
4. Compare the exact outcome distribution (product of DEM characters) with the
   sampled production tallies in ``progress/<p>/const_check_data/0*.npy``.

Reads only; writes ``results/production_words.json`` and
``results/protocol_check.json`` in this folder.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

import tclib as T

sys.path.insert(0, os.path.join(T.REPO, "scripts"))
SCRATCH = os.environ.get("TC_SCRATCH", tempfile.gettempdir())
tempfile.tempdir = SCRATCH
import verify_default_circuits_unchanged as V  # noqa: E402

from sdim.circuit_io import read_circuit  # noqa: E402
from sdim.dem import DetectorErrorModel  # noqa: E402

P_INDICES_TEXT = (5, 10)        # p = 0.002 and 0.015
P_INDICES_TALLY = (4, 6, 8)     # p = 0.001, 0.005, 0.01


def run_path(key):
    return os.path.join(T.MAIN_DATA, T.RUNS[key])


def read_list(path, cast):
    with open(path) as fh:
        return [cast(x) for x in fh.read().strip().split(",") if x.strip()]


def scan_const0_text(text: str, d: int) -> dict:
    """Structural facts of one const-0 .chp file."""
    lines = text.splitlines()[V.HEADER_LINES:]
    first_noise = next((i for i, ln in enumerate(lines) if ln.startswith(("N1", "N2"))), len(lines))
    last_noise = max((i for i, ln in enumerate(lines) if ln.startswith(("N1", "N2"))), default=-1)
    encoder = lines[:first_noise]
    tail = lines[last_noise + 1:]
    gates_on_ancilla = 0
    early_collapse = 0
    for i, ln in enumerate(lines):
        tok = ln.split()
        name = tok[0]
        if name in ("DETECTOR", "LOGICAL_OBSERVABLE"):
            continue
        wires = [int(t) for t in tok[1:] if t.isdigit()]
        if name not in ("N1", "N2") and any(w >= 5 for w in wires):
            gates_on_ancilla += 1
        if name in ("M", "M_X", "RESET") and i <= last_noise:
            early_collapse += 1
    return {
        "encoder_has_7_gates_then_noise_starts": len([ln for ln in encoder if not ln.startswith("M")]) >= 7,
        "encoder_prefix": encoder[:7],
        "tail": tail,
        "gates_on_ancilla_wires": gates_on_ancilla,
        "collapse_before_last_noise": early_collapse,
        "n_measurements": sum(ln.startswith("M ") for ln in lines),
        "n_two_qudit_noise": sum(ln.startswith("N2") for ln in lines),
    }


def main():
    summary = {}
    all_words = {}
    for key in T.RUNS:
        code, noise = key
        run = run_path(key)
        gen, cd, _ = T.setup(code, noise)
        d = cd.dimension
        depths = read_list(os.path.join(run, "depths.txt"), int)
        probs = read_list(os.path.join(run, "probs.txt"), float)
        n_seq = int(open(os.path.join(run, "num_cliffs.txt")).read().strip())
        shots = int(open(os.path.join(run, "shots.txt")).read().strip())
        vgen = V.generator_for_run(run)
        words = [V.decode_lrb_words(run, vgen, s) for s in range(n_seq)]
        all_words["/".join(key)] = words

        # 2. byte-for-byte rebuild
        rend = V.Renderer(gen)
        n_cmp = n_ok = 0
        for s in range(n_seq):
            for di, m in enumerate(depths):
                circ = T.build_circuit(code, noise, T.mirror_segments(words[s][:m]))
                for pi in P_INDICES_TEXT:
                    mine = rend.text(circ, probs[pi])
                    theirs = V._chp(run, "LRB_const0", s, pi, di)
                    n_cmp += 1
                    n_ok += int(mine == theirs)

        # 3. structural scan (all sequences, all depths, one p)
        scans = [scan_const0_text(V._chp(run, "LRB_const0", s, 5, di), d)
                 for s in range(n_seq) for di in range(len(depths))]
        enc = {tuple(sc["encoder_prefix"]) for sc in scans}
        tails = {tuple(sc["tail"]) for sc in scans}

        # 4. exact vs sampled
        z_acc, z_err = [], []
        for pi in P_INDICES_TALLY:
            base = os.path.join(run, "progress", str(pi), "const_check_data")
            tal = np.load(os.path.join(base, "0.npy")).astype(float)       # depth x seq x d
            rej = np.load(os.path.join(base, "0_rejected.npy")).astype(float)
            for s in range(n_seq):
                for di, m in enumerate(depths):
                    if m == 0:
                        assert rej[di, s] == 0 and tal[di, s, 1:].sum() == 0
                        continue
                    circ = read_circuit(os.path.join(run, "experiments", "LRB_const0", str(s), str(pi), f"{di}.chp"))
                    P = T.dist_from_chars(T.chars_from_dem(DetectorErrorModel.from_circuit(circ)), d)
                    A, J, _ = T.accept_and_polarization(P, d)
                    _, _, acc = T.index_sets(d)
                    n_acc = tal[di, s].sum()
                    n_tot = n_acc + rej[di, s]
                    assert abs(n_tot - shots) <= 1, (n_tot, shots)
                    z_acc.append((n_acc - n_tot * A) / np.sqrt(n_tot * A * (1 - A)))
                    e = 1.0 - P[acc[0]] / A          # P(logical != 0 | accepted)
                    n_e = tal[di, s, 1:].sum()
                    if n_acc * e > 5:
                        z_err.append((n_e - n_acc * e) / np.sqrt(n_acc * e * (1 - e)))
        z_acc, z_err = np.asarray(z_acc), np.asarray(z_err)
        summary["/".join(key)] = {
            "run": T.RUNS[key],
            "depths": depths,
            "n_sequences": n_seq,
            "shots_per_circuit": shots,
            "const0_files_compared": n_cmp,
            "const0_files_identical_to_rebuild": n_ok,
            "distinct_encoder_prefixes": [list(e) for e in enc],
            "distinct_tails_after_last_noise": [list(t) for t in tails],
            "max_gates_on_ancilla_wires": max(sc["gates_on_ancilla_wires"] for sc in scans),
            "max_collapse_before_last_noise": max(sc["collapse_before_last_noise"] for sc in scans),
            "measurements_per_circuit": sorted({sc["n_measurements"] for sc in scans}),
            "z_acceptance_mean_std_n": [float(z_acc.mean()), float(z_acc.std()), int(z_acc.size)],
            "z_accepted_error_mean_std_n": [float(z_err.mean()), float(z_err.std()), int(z_err.size)],
            "z_acceptance_max_abs": float(np.abs(z_acc).max()),
            "z_accepted_error_max_abs": float(np.abs(z_err).max()),
        }
        print(key, json.dumps({k: v for k, v in summary["/".join(key)].items()
                               if k not in ("distinct_encoder_prefixes", "distinct_tails_after_last_noise")}), flush=True)
        print("   encoder:", [list(e) for e in enc])
        print("   tail   :", [list(t) for t in tails], flush=True)
    os.makedirs(os.path.join(T.HERE, "results"), exist_ok=True)
    with open(os.path.join(T.HERE, "results", "production_words.json"), "w") as fh:
        json.dump(all_words, fh)
    with open(os.path.join(T.HERE, "results", "protocol_check.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
