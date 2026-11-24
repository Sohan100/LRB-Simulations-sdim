"""Small login-node test of protocol fixes for the unchecked mirror inverse.

Builds single-ancilla [[5,1,2]]_3 LRB-D circuits in memory with the
production generator's own pieces (src/lrb is imported read-only, nothing
is written outside this folder) and samples sdim 1.4.0's exact DEM:

* ``mirror``           production protocol: C1..Cm each followed by a check
                       round, then Cm^-1..C1^-1 with no checks, then one
                       final check (2m noisy Cliffords, m+1 check rounds).
* ``checked_inverse``  same, plus a check round after every inverse Clifford
                       except the last (2m noisy Cliffords, 2m check rounds).
* ``compiled``         C1..Cm each followed by a check round, then ONE
                       recovery Clifford equal to (Cm...C1)^-1 drawn from the
                       216-element catalog, then the final check (m+1 noisy
                       Cliffords, m+1 check rounds). Standard RB recovery.

Postselection uses every round (unif 1). For each circuit the script stores
accepted counts per terminal outcome, so the same estimators as
``nmlib.curves`` apply. Usage:

    python variant_sim.py --noise sd6 --p 0.002 --shots 200000 --workers 6
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # studies/non-markovian -> repo root
sys.path.insert(0, os.path.join(REPO, "src"))

DEPTHS = (0, 2, 4, 6, 10, 14, 18, 20, 22)
VARIANTS = ("mirror", "checked_inverse", "compiled")
# m=0-consistent closings (tested after the user asked to keep depth 0):
#   compiled_I0    compiled, and depth 0 also gets its "recovery", the catalog
#                  identity word I (one noisy identity moment)
#   compiled_pair  every depth, 0 included, closes with a uniformly random
#                  catalog Clifford G and then R = (G Cm...C1)^-1, no check
#                  between them, then the final check
#   compiled_lead  compiled, plus one postselected check round right after the
#                  ideal preparation at every depth, so the depth-0 final check
#                  inherits a previous round's leftover errors like every other
#   mirror_lead    the production mirror protocol with the same leading round

_W = np.exp(2j * np.pi / 3)
_TOKEN = {
    "I": np.eye(3, dtype=complex),
    "X": np.roll(np.eye(3), 1, axis=0).astype(complex),  # X|j> = |j+1>
    "Z": np.diag([1, _W, _W ** 2]),
    "H": np.array([[_W ** (j * k) for k in range(3)] for j in range(3)]) / np.sqrt(3),
    "P": np.diag([1, 1, _W]),
}  # checked against sdim's Program on 800 random words: check_gate_conventions.py


def word_unitary(word: str) -> np.ndarray:
    """Catalog words act right to left, so U = M(w0) M(w1) ... M(wn)."""
    u = np.eye(3, dtype=complex)
    for t in word:
        u = u @ _TOKEN[t]
    return u


def _phase_key(u: np.ndarray) -> tuple:
    flat = u.reshape(-1)
    i = int(np.argmax(np.abs(flat) > 1e-9))
    v = flat / (flat[i] / abs(flat[i]))
    return tuple(np.round(v.real, 6)) + tuple(np.round(v.imag, 6))


def recovery_word(words: list[str], catalog: list[str], lookup: dict) -> str:
    total = np.eye(3, dtype=complex)
    for w in words:  # C1 first
        total = word_unitary(w) @ total
    return lookup[_phase_key(total.conj().T)]


def _setup(noise: str):
    from lrb.circuit_generator import LRBCircuitGenerator
    from lrb.code_simulation_profiles import CodeSimulationProfileRegistry

    profile = CodeSimulationProfileRegistry.resolve_code_profile("folded_qutrit")
    gen = LRBCircuitGenerator(code_definition=profile.code_definition, noise_model=noise)
    cd = profile.code_definition
    catalog = list(cd.clifford_strings)
    lookup = {_phase_key(word_unitary(w)): w for w in catalog}
    assert len(lookup) == 216, len(lookup)
    return gen, cd, catalog, lookup


def build_circuit(gen, cd, catalog, lookup, variant, words, depth, p, pad_word=None):
    from sdim.circuit import Circuit
    from lrb.lrb_simulation import NoiseModelUtils

    fwd = [cd.clifford_to_gate_sequence(w) for w in words[:depth]]
    inv = [cd.clifford_inverse_map[w] for w in reversed(words[:depth])]
    seg = lambda gates: gen._noisy_segment(gen._logical_clifford_circuit(gates), True)
    chk = lambda: gen._noisy_segment(gen._check_round_circuit(), True)
    c = Circuit(dimension=cd.dimension, num_qudits=cd.encoded_num_qudits)
    c = c + cd.logical_plus_initial_state()
    rounds = 0
    if variant in ("compiled_lead", "mirror_lead"):
        c = c + chk()
        rounds += 1
    for g in fwd:
        c = c + seg(g)
        c = c + chk()
        rounds += 1
    if variant in ("mirror", "mirror_lead"):
        for g in inv:
            c = c + seg(g)
    elif variant == "checked_inverse":
        for j, g in enumerate(inv):
            c = c + seg(g)
            if j < len(inv) - 1:
                c = c + chk()
                rounds += 1
    elif variant in ("compiled", "compiled_lead"):
        if depth > 0:
            c = c + seg(cd.clifford_to_gate_sequence(recovery_word(words[:depth], catalog, lookup)))
    elif variant == "compiled_I0":
        c = c + seg(cd.clifford_to_gate_sequence(recovery_word(words[:depth], catalog, lookup)))
    elif variant == "compiled_pair":
        g = pad_word
        c = c + seg(cd.clifford_to_gate_sequence(g))
        c = c + seg(cd.clifford_to_gate_sequence(recovery_word(words[:depth] + [g], catalog, lookup)))
    else:
        raise ValueError(variant)
    c = c + chk()
    rounds += 1
    c = c + cd.terminal_measurement()
    gen.append_lrb_detector_events(c, rounds - 1)
    gen.update_noise_param(c, p)
    NoiseModelUtils.ensure_noise_params(c)
    return c, rounds


def run_one(args):
    noise, p, variant, seq_idx, words, depth, shots, seed, pad_word = args
    from sdim.dem import DetectorErrorModel
    from lrb.dem_simulation import reference_shot

    gen, cd, catalog, lookup = _setup(noise)
    circ, rounds = build_circuit(gen, cd, catalog, lookup, variant, words, depth, p, pad_word)
    _, ref = reference_shot(circ)
    dem = DetectorErrorModel.from_circuit(circ)
    det_ref = np.asarray(ref["detectors"], dtype=np.int64)
    log_ref = np.asarray(ref["logicals"], dtype=np.int64)
    assert np.all(det_ref % 3 == 0), "noiseless syndromes must be trivial"
    assert np.all(log_ref % 3 == 0), "noiseless logical outcome must be 0"
    sampler = dem.compile_sampler(seed=seed)
    counts = np.zeros(3, dtype=np.int64)
    rejected = 0
    left = shots
    while left > 0:
        n = min(left, 100_000)
        det, log = sampler.sample(n)
        det = (np.asarray(det, dtype=np.int64) + det_ref[None, :]) % 3
        acc = np.all(det == 0, axis=1)
        out = (np.asarray(log, dtype=np.int64)[:, 0] + log_ref[0]) % 3
        counts += np.bincount(out[acc], minlength=3)
        rejected += int((~acc).sum())
        left -= n
    return variant, seq_idx, depth, counts, rejected, rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", default="sd6")
    ap.add_argument("--p", type=float, default=0.002)
    ap.add_argument("--shots", type=int, default=200_000)
    ap.add_argument("--n-seq", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--depths", default=",".join(map(str, DEPTHS)))
    a = ap.parse_args()
    depths = [int(x) for x in a.depths.split(",")]
    variants = a.variants.split(",")
    _, _, catalog, _ = _setup(a.noise)
    rng = random.Random(a.seed)
    seqs = [[rng.choice(catalog) for _ in range(max(depths))] for _ in range(a.n_seq)]
    pads = [rng.choice(catalog) for _ in range(a.n_seq)]  # one G per sequence, shared by its depths
    jobs = []
    for v in variants:
        for s, words in enumerate(seqs):
            for d in depths:
                jobs.append((a.noise, a.p, v, s, words, d, a.shots,
                             a.seed + 7919 * s + 104729 * d + 13 * variants.index(v), pads[s]))
    t0 = time.time()
    tallies = {v: np.zeros((len(depths), a.n_seq, 3), np.int64) for v in variants}
    rej = {v: np.zeros((len(depths), a.n_seq), np.int64) for v in variants}
    rounds = {v: np.zeros(len(depths), np.int64) for v in variants}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (v, s, d, cnt, r, nr) in enumerate(ex.map(run_one, jobs, chunksize=1)):
            k = depths.index(d)
            tallies[v][k, s] = cnt
            rej[v][k, s] = r
            rounds[v][k] = nr
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)} circuits, {time.time() - t0:.0f}s", flush=True)
    tag = "" if variants == list(VARIANTS) else "_" + "-".join(variants)
    out = os.path.join(HERE, "results", f"variant_sim_{a.noise}_p{a.p:g}_shots{a.shots}{tag}.npz")
    np.savez_compressed(
        out, depths=np.asarray(depths), p=a.p, noise=a.noise, shots=a.shots, seed=a.seed,
        **{f"{v}_tallies": tallies[v] for v in variants},
        **{f"{v}_rejected": rej[v] for v in variants},
        **{f"{v}_rounds": rounds[v] for v in variants},
    )
    print("wrote", out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
