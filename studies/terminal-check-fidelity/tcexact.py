"""Exact outcome-distribution tables for families of terminal-check circuits.

A table holds, for every (sequence, depth) circuit, the integer counts
``n[type, k]`` of ``tclib.CircuitChars``; the exact distribution over the
d^3 outcomes (s1, s2, l) then follows for any p in microseconds. Tables are
cached as ``results/cache/<name>.npz``.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import tclib as T

CACHE = os.path.join(T.HERE, "results", "cache")


def _one(args):
    code, noise, words, depth, closing, noisy_prep, noisy_meas = args
    if closing == "mirror":
        segs = T.mirror_segments(words[:depth])
    elif closing == "compiled":
        segs = T.compiled_segments(code, words[:depth])
    else:
        raise ValueError(closing)
    cc = T.CircuitChars(code, noise, T.build_circuit(code, noise, segs, noisy_prep, noisy_meas))
    return cc.types, cc.n


def _one_segments(args):
    code, noise, segs, noisy_prep, noisy_meas, crosstalk = args
    cc = T.CircuitChars(code, noise, T.build_circuit(code, noise, segs, noisy_prep, noisy_meas, crosstalk))
    return cc.types, cc.n


def _collect(results, shape, d):
    types = sorted({t for ty, _ in results for t in ty})
    n = np.zeros((len(results), len(types), d ** 3), dtype=np.int32)
    for i, (ty, cnt) in enumerate(results):
        for t, row in zip(ty, cnt):
            n[i, types.index(t)] = row
    return types, n.reshape(shape + (len(types), d ** 3))


def _run(fn, jobs, workers):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, jobs, chunksize=8))


class Table:
    def __init__(self, d, types, n, meta=None):
        self.d, self.types, self.n, self.meta = d, [tuple(t) for t in types], n, meta or {}

    def chars(self, p, scale_override=None):
        """chi[..., k] for scalar p (leading axes = circuit axes)."""
        logchi = np.zeros(self.n.shape[:-2] + (self.d ** 3,))
        for i, (kind, scale) in enumerate(self.types):
            s = scale if scale_override is None else scale_override.get((kind, scale), scale)
            logchi += np.log1p(-T._pi(kind, s * p, self.d)) * self.n[..., i, :]
        return np.exp(logchi)

    def dist(self, p, scale_override=None):
        return np.clip(T.dist_from_chars(self.chars(p, scale_override), self.d), 0.0, None)

    def save(self, path):
        np.savez_compressed(path, d=self.d, types=json.dumps(self.types), n=self.n, meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(int(z["d"]), json.loads(str(z["types"])), z["n"], json.loads(str(z["meta"])))


def sequence_table(code, noise, words_list, depths, closing="mirror", noisy_prep=False,
                   noisy_meas=False, workers=1, name=None) -> Table:
    """Table with axes (sequence, depth) for nested-prefix sequences."""
    key = hashlib.sha1(json.dumps([code, noise, words_list, list(depths), closing, noisy_prep, noisy_meas]).encode()).hexdigest()[:12]
    path = os.path.join(CACHE, f"{name or 'seq'}_{code}_{noise}_{closing}_{key}.npz")
    if os.path.exists(path):
        return Table.load(path)
    jobs = [(code, noise, words, m, closing, noisy_prep, noisy_meas) for words in words_list for m in depths]
    res = _run(_one, jobs, workers)
    d = T.setup(code, noise)[1].dimension
    types, n = _collect(res, (len(words_list), len(depths)), d)
    tab = Table(d, types, n, {"code": code, "noise": noise, "closing": closing, "depths": list(depths),
                              "noisy_prep": noisy_prep, "noisy_meas": noisy_meas})
    os.makedirs(CACHE, exist_ok=True)
    tab.save(path)
    return tab


def segments_table(code, noise, segs_list, noisy_prep=False, noisy_meas=False, workers=1, name=None,
                   crosstalk=()) -> Table:
    """Table with one axis over arbitrary segment lists (ground-truth constructions)."""
    crosstalk = tuple(tuple(c) for c in crosstalk)
    spec = [code, noise, segs_list, noisy_prep, noisy_meas] + ([list(crosstalk)] if crosstalk else [])
    key = hashlib.sha1(json.dumps(spec).encode()).hexdigest()[:12]
    path = os.path.join(CACHE, f"{name or 'segs'}_{code}_{noise}_{key}.npz")
    if os.path.exists(path):
        return Table.load(path)
    jobs = [(code, noise, segs, noisy_prep, noisy_meas, crosstalk) for segs in segs_list]
    res = _run(_one_segments, jobs, workers)
    d = T.setup(code, noise)[1].dimension
    types, n = _collect(res, (len(segs_list),), d)
    tab = Table(d, types, n, {"code": code, "noise": noise})
    os.makedirs(CACHE, exist_ok=True)
    tab.save(path)
    return tab


def random_words(code, n_seq, length, seed):
    rng = random.Random(seed)
    cat = T.catalog(code)
    return [[rng.choice(cat) for _ in range(length)] for _ in range(n_seq)]


def production_words(code, noise):
    with open(os.path.join(T.HERE, "results", "production_words.json")) as fh:
        return json.load(fh)[f"{code}/{noise}"]


def pooled(P, d):
    """Pooled (sequence-averaged) acceptance, joint and accepted polarization.

    P has axes (sequence, depth, outcome); sequences get equal shots.
    """
    return T.accept_and_polarization(P.mean(0), d)
