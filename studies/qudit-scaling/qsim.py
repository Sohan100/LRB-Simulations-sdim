"""LRB-D and physical-RB sampling for the folded [[5,1,2]]_q code at any prime q.

Circuits come from the production generator (``src/lrb``), so the protocol is
the production one exactly: ideal |+>_L preparation, C1..Cm each followed by a
noisy check round, the mirror inverses Cm^-1..C1^-1 with no checks, one final
noisy check round and an ideal logical-X readout; SI1000/SD6 moment-scheduled
noise.  Every check round is executed for every interval k; k only chooses
which rounds are postselected (``postselect_uniform_interval``: the final
round plus the rounds after C_k, C_2k, ...).  ``const1`` postselects only the
final round.

Sampling uses sdim 1.4.0's exact DEM, as production does.  For each p and
circuit the output is the accepted tally per logical outcome and the rejected
count, per k, in the production checkpoint layout (depth, sequence, outcome).

Physical RB: one qudit, the same catalogue words with the production RB
construction (one noisy segment per Clifford, mirror inverse, ideal
preparation and readout), sampled through the DEM as production does.
"""
from __future__ import annotations

import random
import time

import numpy as np

import qslib  # noqa: F401  (puts the repository's src/ on sys.path)
from lrb.dem_simulation import reference_shot  # noqa: E402
from lrb.lrb_simulation import NoiseModelUtils  # noqa: E402

BATCH = 100_000


def unif_slots(depth: int, k: int) -> list[int]:
    """Production ``postselect_uniform_interval`` check slots (round indices)."""
    slots = [depth]
    if depth >= k and k > 0:
        slots.extend(j * k - 1 for j in range(1, depth // k + 1) if j * k - 1 < depth)
    return sorted(set(slots))


def draw_sequences(q: int, compile_mode: str, n_seq: int, max_depth: int, seed: int):
    cat = qslib.catalogue(q, compile_mode)
    rng = random.Random(seed)
    return [[rng.choice(cat) for _ in range(max_depth)] for _ in range(n_seq)]


def lrb_circuits(q, noise, compile_mode, words, depths, round_name=None):
    """Production generator; ``words`` must have exactly max(depths) entries
    (the generator's mirror suffix assumes the sample length is the maximum
    depth)."""
    assert len(words) == max(depths), (len(words), max(depths))
    gen = qslib.generator(q, noise, compile_mode, round_name)
    return gen, gen.generate_lrb_clifford_sequence(list(depths), clifford_strings=list(words))


def rb_circuits(q, noise, compile_mode, words, depths):
    """Production ``generate_rb_clifford_sequence`` with given words."""
    from sdim.circuit import Circuit

    gen = qslib.generator(q, noise, compile_mode)
    cd = gen.code_definition
    fwd, inv = gen._clifford_and_inverse_gate_lists(list(words))
    max_depth = len(words)
    out = []
    for depth in sorted(depths):
        c = Circuit(dimension=q, num_qudits=1)
        if depth == 0:
            gates, closing = [], []
        else:
            gates = fwd[:depth]
            closing = gen._closing_gate_lists(list(words), inv, max_depth, depth)
        for cliff in gates + closing:
            seg = Circuit(dimension=q, num_qudits=1)
            for g in cliff:
                cd.apply_physical_gate(seg, g)
            c = c + gen._noisy_segment(seg, True)
        c.add_gate("M", 0)
        gen.append_rb_detector_events(c)
        out.append(c)
    return gen, out


def _prepare(gen, circuit, p):
    gen.update_noise_param(circuit, p)
    NoiseModelUtils.ensure_noise_params(circuit)


def sample_lrb_circuit(gen, circuit, depth, p, shots, seed, ks, q, ref=None):
    """Return {k: counts(q)}, {k: rejected}, plus const1, for one circuit at p."""
    from sdim.dem import DetectorErrorModel

    _prepare(gen, circuit, p)
    if ref is None:
        _, ref = reference_shot(circuit)
    det_ref = np.asarray(ref["detectors"], dtype=np.int64)
    log_ref = np.asarray(ref["logicals"], dtype=np.int64)
    assert np.all(det_ref % q == 0) and np.all(log_ref % q == 0)
    n_rounds = depth + 1
    assert len(det_ref) == 4 * n_rounds, (len(det_ref), n_rounds)
    counts = {k: np.zeros(q, np.int64) for k in ks}
    counts["c1"] = np.zeros(q, np.int64)
    rejected = {k: 0 for k in counts}
    if p == 0.0:
        for k in counts:
            counts[k][0] = shots
        return counts, rejected, ref
    dem = DetectorErrorModel.from_circuit(circuit)
    sampler = dem.compile_sampler(seed=seed)
    slots = {k: unif_slots(depth, k) for k in ks}
    left = shots
    while left > 0:
        n = min(left, BATCH)
        det, log = sampler.sample(n)
        det = (np.asarray(det, dtype=np.int64) + det_ref[None, :]) % q
        round_ok = np.all(det.reshape(n, n_rounds, 4) == 0, axis=2)  # (n, rounds)
        out = (np.asarray(log, dtype=np.int64)[:, 0] + log_ref[0]) % q
        for k in ks:
            acc = np.all(round_ok[:, slots[k]], axis=1)
            counts[k] += np.bincount(out[acc], minlength=q)
            rejected[k] += int((~acc).sum())
        acc = round_ok[:, depth]
        counts["c1"] += np.bincount(out[acc], minlength=q)
        rejected["c1"] += int((~acc).sum())
        left -= n
    return counts, rejected, ref


def sample_rb_circuit(gen, circuit, p, shots, seed, q, ref=None):
    from sdim.dem import DetectorErrorModel

    _prepare(gen, circuit, p)
    if ref is None:
        _, ref = reference_shot(circuit)
    log_ref = np.asarray(ref["logicals"], dtype=np.int64)
    if p == 0.0:
        c = np.zeros(q, np.int64)
        c[int(log_ref[0]) % q] = shots
        return c, ref
    dem = DetectorErrorModel.from_circuit(circuit)
    sampler = dem.compile_sampler(seed=seed)
    counts = np.zeros(q, np.int64)
    left = shots
    while left > 0:
        n = min(left, BATCH)
        _, log = sampler.sample(n)
        out = (np.asarray(log, dtype=np.int64)[:, 0] + log_ref[0]) % q
        counts += np.bincount(out, minlength=q)
        left -= n
    return counts, ref


def polarization(tallies: np.ndarray) -> np.ndarray:
    """Pipeline statistic |sum_i omega^i n_i| / N over the last axis."""
    t = np.asarray(tallies, dtype=float)
    d = t.shape[-1]
    w = np.exp(2j * np.pi * np.arange(d) / d)
    n = t.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(t @ w) / n


def timed(fn, *a, **kw):
    t0 = time.time()
    out = fn(*a, **kw)
    return out, time.time() - t0
