"""Adversarial verification of work package K (Steane [[7,1,3]] LRB).

Independent of the implementer's analysis code: own Steane stabilizers (from
the [7,4,3] Hamming parity-check matrix), own memory circuits, own DEM parser,
own brute-force joint, own exact decoder (whole check rounds, flags as plain
detectors, as time slots), and an exact second-order (pair-enumeration)
expansion of the maximum-likelihood failure probability. The implementer's
circuit builder (``lrb_qubit.steane_circuits.build_lrb_circuit``) and decoder
(``SlotMLD``) are used only as objects under test.

Subcommands (run from the repository root, ``module load python/3.11``):

``algebra``       K1 (transversal gates) and K3 (weight-2 equivalences).
``distance``      K4/K5: own one- and two-round memory circuits, plus the
                  implementer's circuits, with Stim's searches and an exact
                  DEM pair bound.
``first-order``   E1/R1/A2 per step (own code) and the frame decomposition of
                  the naive first-order floor (K7, K8, K13, K14).
``second-order``  Exact p^2 coefficient of the MLD failure probability for
                  the flagged circuits, all 10 sequences and grid depths, and
                  exact const0 curves (K9-K12, K18). Process pool.
``decoder``       K6 and task 5: SlotMLD against an own brute force and an own
                  wide-slot exact decoder; flag sensitivity. Process pool.
``sample``        Sampled MLD failure rates at two small p (depth 4) against
                  the exact second-order prediction; fresh |LLR| >= 4 sample
                  at p = 1e-3 (K16). Process pool.
``thresholds``    Spot-check of the pseudo-thresholds (K10-K12).

Outputs go to ``studies/correction-decoders/qubit/results/K-verify/``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if any(os.environ.get(_v) != "1" for _v in _THREAD_VARS):
    for _v in _THREAD_VARS:
        os.environ[_v] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import stim  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import code as C  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import fits as FT  # noqa: E402
from lrb_qubit.circuits import SI1000, emit_operations  # noqa: E402

OUT = ROOT / "studies/correction-decoders/qubit/results/K-verify"
K_DIR = ROOT / "studies/correction-decoders/qubit/results/K"
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
SEQ_SEED = 20261028
NUM_SEQ = 10


def sequences(n=NUM_SEQ):
    return E.clifford_sequences(n, max(DEPTHS), SEQ_SEED)


def under_test():
    """The implementer's modules (objects under test only)."""
    from lrb_qubit import steane_circuits as SV
    from lrb_qubit import steane_code as SC
    return SC, SV


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float))


# ----------------------------------------------------------------------
# Own code construction
# ----------------------------------------------------------------------
def hamming_binary():
    """Rows = bits of the column index q + 1 (qubit q <-> column q + 1)."""
    return np.array([[((q + 1) >> b) & 1 for q in range(7)]
                     for b in (2, 1, 0)], dtype=np.uint8)


def hamming_cyclic():
    """An inequivalently labelled [7,4,3] parity-check matrix."""
    return np.array([[1, 1, 1, 0, 1, 0, 0], [0, 1, 1, 1, 0, 1, 0],
                     [1, 1, 0, 1, 0, 0, 1]], dtype=np.uint8)


def pauli(kind, support, n=7):
    chars = ["_"] * n
    for q in support:
        chars[q] = kind
    return stim.PauliString("".join(chars))


def stabilizer_group(H):
    gens = ([pauli("Z", np.flatnonzero(r)) for r in H]
            + [pauli("X", np.flatnonzero(r)) for r in H])
    elems = {}
    for mask in range(64):
        P = stim.PauliString(7)
        for i in range(6):
            if (mask >> i) & 1:
                P *= gens[i]
        key = tuple(np.concatenate(P.to_numpy()).astype(int))
        if key in elems:
            raise AssertionError("dependent generators")
        elems[key] = P
    return gens, elems


def classify(P, elems, XL, ZL):
    """('I'|'X'|'Y'|'Z', sign) with P = sign * logical * stabilizer, or None."""
    YL = XL * ZL * 1j
    for name, L in (("I", stim.PauliString(7)), ("X", XL), ("Y", YL),
                    ("Z", ZL)):
        R = P * L
        key = tuple(np.concatenate(R.to_numpy()).astype(int))
        if key in elems:
            S = elems[key]
            if R == S:
                return name, 1
            if R == -S:
                return name, -1
            return name, "complex"
    return None


def cmd_algebra(args):
    out = {}
    for label, H in (("binary", hamming_binary()), ("cyclic", hamming_cyclic())):
        gens, elems = stabilizer_group(H)
        assert all(a.commutes(b) for a in gens for b in gens)
        XL7, ZL7 = stim.PauliString("X" * 7), stim.PauliString("Z" * 7)
        w3 = next(c for c in itertools.combinations(range(7), 3)
                  if not (H[:, list(c)].sum(axis=1) % 2).any())
        entry = {"weight3_logical_support": list(w3), "gates": {}}
        for XL, ZL, tag in ((XL7, ZL7, "X^7/Z^7"),
                            (pauli("X", w3), pauli("Z", w3), "weight-3")):
            for gname in ("H", "S", "S_DAG"):
                T1 = stim.Tableau.from_named_gate(gname)
                T = stim.Tableau(0)
                for _ in range(7):
                    T = T + T1
                entry["gates"][f"{tag} {gname}^7"] = {
                    "stabilizers_preserved_with_plus_sign": all(
                        classify(T(g), elems, XL, ZL) == ("I", 1) for g in gens),
                    "X_L ->": classify(T(XL), elems, XL, ZL),
                    "Z_L ->": classify(T(ZL), elems, XL, ZL)}
        w1 = [pauli(k, [q]) for q in range(7) for k in "XYZ"]
        counts = {"same_pauli": [0, 0], "mixed_pauli": [0, 0]}
        counterexamples = []
        und = {1: 0, 2: 0, 3: 0}
        for e in w1:
            und[1] += classify(e, elems, XL7, ZL7) is not None
        for c, d in itertools.combinations(range(7), 2):
            for a, b in itertools.product("XYZ", repeat=2):
                E2 = stim.PauliString(7)
                E2[c] = a
                E2[d] = b
                und[2] += classify(E2, elems, XL7, ZL7) is not None
                ok = False
                for e1 in w1:
                    cl = classify(E2 * e1, elems, XL7, ZL7)
                    if cl is not None and cl[0] != "I":
                        ok = True
                        break
                key = "same_pauli" if a == b else "mixed_pauli"
                counts[key][0 if ok else 1] += 1
                if not ok and len(counterexamples) < 6:
                    counterexamples.append(str(E2))
        for qs in itertools.combinations(range(7), 3):
            for ps in itertools.product("XYZ", repeat=3):
                E3 = stim.PauliString(7)
                for q, ch in zip(qs, ps):
                    E3[q] = ch
                cl = classify(E3, elems, XL7, ZL7)
                und[3] += cl is not None and cl[0] != "I"
        entry["weight2_is_weight1_times_nontrivial_logical"] = {
            k: {"yes": v[0], "no": v[1]} for k, v in counts.items()}
        entry["weight2_counterexamples"] = counterexamples
        entry["undetectable_logicals_by_weight"] = und
        out[label] = entry
        print(label, json.dumps(entry, indent=1, default=str))
    dump("algebra.json", out)


# ----------------------------------------------------------------------
# Own memory circuits (K4, K5)
# ----------------------------------------------------------------------
def own_memory_circuit(H, extraction, schedule, basis, p, rounds=1,
                       orders=None, logical_support=None, noise=SI1000()):
    """Ideal |0>_L or |+>_L, `rounds` noisy rounds of all six weight-4 checks
    (naive one-ancilla or Chao-Reichardt flagged), ancilla reset between
    rounds, ideal readout in `basis`. SI1000 via lrb_qubit.circuits."""
    supports = [tuple(np.flatnonzero(r)) for r in H]
    stabs = [("Z", s) for s in supports] + [("X", s) for s in supports]
    orders = [s for _, s in stabs] if orders is None else orders
    anc = [7 + i for i in range(6)]
    flags = ([13 + i for i in range(6)] if extraction == "flag"
             else [None] * 6)
    nq = 19 if extraction == "flag" else 13
    logical_support = (tuple(range(7)) if logical_support is None
                       else logical_support)
    gens = [pauli(k, s) for k, s in stabs]
    enc = stim.Tableau.from_stabilizers(
        gens + [pauli(basis, logical_support)]).to_circuit("elimination")
    enc_ops = []
    for inst in enc:
        t = [x.value for x in inst.targets_copy()]
        ar = 2 if stim.gate_data(inst.name).is_two_qubit_gate else 1
        enc_ops += [(inst.name, tuple(t[k:k + ar])) for k in range(0, len(t), ar)]
    circ = stim.Circuit()
    emit_operations(circ, enc_ops, p, None, nq)
    syn_rec, flag_rec = [], []
    for r in range(rounds):
        ops, plan, meas = [], [], []
        for i, ((kind, _), order) in enumerate(zip(stabs, orders)):
            a, f, d = anc[i], flags[i], list(order)
            if f is None:
                ops += ([("CX", (q, a)) for q in d] if kind == "Z" else
                        [("H", (a,))] + [("CX", (a, q)) for q in d] + [("H", (a,))])
            elif kind == "Z":
                ops += [("H", (f,)), ("CX", (d[0], a)), ("CX", (f, a)),
                        ("CX", (d[1], a)), ("CX", (d[2], a)), ("CX", (f, a)),
                        ("CX", (d[3], a)), ("H", (f,))]
            else:
                ops += [("H", (a,)), ("CX", (a, d[0])), ("CX", (a, f)),
                        ("CX", (a, d[1])), ("CX", (a, d[2])), ("CX", (a, f)),
                        ("CX", (a, d[3])), ("H", (a,))]
            block = [("M", (a,))] + ([("M", (f,))] if f is not None else [])
            plan += [(i, "a")] + ([(i, "f")] if f is not None else [])
            if schedule == "sequential":
                ops += block
            else:
                meas += block
        ops += meas
        if r < rounds - 1:
            ops += [("R", (q,)) for q in anc + [x for x in flags if x is not None]]
        recs = emit_operations(circ, ops, p, noise, nq, noisy_measurements=True)
        sr, fr = [None] * 6, [None] * 6
        for (i, k), rec in zip(plan, recs):
            (sr if k == "a" else fr)[i] = rec
        syn_rec.append(sr)
        flag_rec.append(fr)
    ro = (([("H", (q,)) for q in range(7)] if basis == "X" else [])
          + [("M", (q,)) for q in range(7)])
    data = emit_operations(circ, ro, p, None, nq)
    nm = circ.num_measurements

    def rec(ids):
        return [stim.target_rec(i - nm) for i in ids]

    for r in range(rounds):
        for i in range(6):
            circ.append("DETECTOR", rec([syn_rec[r][i]] + (
                [syn_rec[r - 1][i]] if r else [])), [i, r])
        if extraction == "flag":
            for i in range(6):
                circ.append("DETECTOR", rec([flag_rec[r][i]]), [100 + i, r])
    for i, (kind, s) in enumerate(stabs):
        if kind == basis:
            circ.append("DETECTOR", rec([data[q] for q in s]
                                        + [syn_rec[-1][i]]), [i, rounds])
    circ.append("OBSERVABLE_INCLUDE", rec([data[q] for q in logical_support]), 0)
    return circ


def drop_detectors(circuit, keep):
    out = stim.Circuit()
    k = 0
    for inst in circuit.flattened():
        if inst.name == "DETECTOR":
            if keep[k]:
                out.append(inst)
            k += 1
        else:
            out.append(inst)
    return out


# ----------------------------------------------------------------------
# Own DEM tools
# ----------------------------------------------------------------------
def mechanisms(dem):
    """(probability, sorted detector tuple, L flip) per DEM error."""
    out = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets, L = set(), 0
        for t in inst.targets_copy():
            if t.is_separator():
                continue
            if t.is_relative_detector_id():
                dets ^= {t.val}
            elif t.is_logical_observable_id() and t.val == 0:
                L ^= 1
        if p > 0:
            out.append((p, tuple(sorted(dets)), L))
    return out


def pair_bound(dem):
    """Exact for d <= 3: 1 if a single mechanism is an undetectable logical,
    2 if two mechanisms share a symptom and differ in L, else 3 (>= 3)."""
    classes = {}
    for _, dets, L in mechanisms(dem):
        classes.setdefault(dets, set()).add(L)
    if 1 in classes.get((), set()):
        return 1, 0
    amb = sum(1 for k, v in classes.items() if k and len(v) == 2)
    return (2 if amb else 3), amb


def stim_search(circ, size=6, degree=8):
    found = circ.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=size,
        dont_explore_edges_with_degree_above=degree,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=True)
    try:
        g = len(circ.shortest_graphlike_error(
            ignore_ungraphlike_errors=True, canonicalize_circuit_errors=True))
    except ValueError:
        g = None
    witness = []
    for e in found:
        loc = e.circuit_error_locations[0] if e.circuit_error_locations else None
        if loc is None:
            witness.append(str(e.dem_error_terms))
            continue
        paulis = " ".join(f"{'XYZ'[[t.gate_target.is_x_target, t.gate_target.is_y_target, t.gate_target.is_z_target].index(True)]}{t.gate_target.value}"
                          for t in loc.flipped_pauli_product)
        tgt = [t.gate_target.value
               for t in loc.instruction_targets.targets_in_range]
        witness.append(f"{paulis} after {loc.instruction_targets.gate}{tgt}")
    return len(found), g, witness


def first_order(dem):
    classes = {}
    for p, dets, L in mechanisms(dem):
        e = classes.setdefault(dets, [0.0, 0.0])
        e[L] = e[L] + p - 2 * e[L] * p
    U1 = classes.get((), [0.0, 0.0])[1]
    R1 = E1 = A2 = 0.0
    amb = 0
    for d, (p0, p1) in classes.items():
        if not d:
            continue
        R1 += p0 + p1
        E1 += min(p0, p1)
        A2 += p0 * p1
        amb += int(p0 > 0 and p1 > 0)
    return {"U1": U1, "R1": R1, "E1": E1, "A2": A2, "ambiguous": amb}


def second_order(dem, seed=12345):
    """Exact second-order MLD failure probability for a DEM with E1 = 0.

    Every syndrome s != 0 contributes, at order p^2: if s is the symptom of a
    single mechanism (logical value L*), the pair weight of the other coset;
    otherwise min(W0(s), W1(s)) over pairs. s = 0 contributes the pair weight
    with L = 1 (undetectable logical pairs, 0 for d = 3). Pairs are grouped by
    a 128-bit linear hash of their symptom (hash(s1 ^ s2) = h1 ^ h2).
    """
    mechs = mechanisms(dem)
    n = dem.num_detectors
    N = len(mechs)
    rng = np.random.default_rng(seed)
    R = rng.integers(0, 2 ** 64, size=(n, 2), dtype=np.uint64)
    h = np.zeros((N, 2), dtype=np.uint64)
    prob = np.empty(N)
    L = np.empty(N, dtype=np.int8)
    for i, (p, dets, l) in enumerate(mechs):
        for d in dets:
            h[i] ^= R[d]
        prob[i], L[i] = p, l
    single = {}
    for i in range(N):
        e = single.setdefault((int(h[i, 0]), int(h[i, 1])), [0.0, 0.0])
        e[L[i]] += prob[i]
    if (0, 0) in single and single[(0, 0)][1] > 0:
        raise ValueError("first-order undetectable logical")
    if any(v[0] > 0 and v[1] > 0 for k, v in single.items() if k != (0, 0)):
        raise ValueError("first-order ambiguity (E1 > 0)")
    keys = sorted(single)
    sk = np.array(keys, dtype=np.uint64).reshape(-1, 2)
    s_lstar = np.array([0 if single[k][0] > 0 else 1 for k in keys], np.int8)
    hp = np.concatenate([h[i + 1:] ^ h[i] for i in range(N - 1)])
    wp = np.concatenate([prob[i] * prob[i + 1:] for i in range(N - 1)])
    lp = np.concatenate([L[i + 1:] ^ L[i] for i in range(N - 1)])
    rec = np.ascontiguousarray(hp).view([("a", np.uint64), ("b", np.uint64)]).ravel()
    uniq, inv = np.unique(rec, return_inverse=True)
    W0 = np.bincount(inv, weights=wp * (lp == 0), minlength=len(uniq))
    W1 = np.bincount(inv, weights=wp * (lp == 1), minlength=len(uniq))
    ua, ub = uniq["a"], uniq["b"]
    idx = np.clip(np.searchsorted(sk[:, 0], ua), 0, len(sk) - 1)
    is_single = (sk[idx, 0] == ua) & (sk[idx, 1] == ub)
    zero = (ua == 0) & (ub == 0)
    contrib = np.where(zero, W1, np.where(
        is_single, np.where(s_lstar[idx] == 0, W1, W0), np.minimum(W0, W1)))
    return {"E2": float(contrib.sum()),
            "undetectable_pairs": float(W1[zero].sum()),
            "from_single_classes": float(contrib[is_single & ~zero].sum()),
            "from_pair_only_classes": float(contrib[~is_single & ~zero].sum()),
            "num_mechanisms": N, "num_pairs": int(len(wp))}


def xor_convolve(arr, bits, p, nbits):
    axes = tuple(nbits - 1 - b for b in bits)
    return (1.0 - p) * arr + p * np.flip(arr, axis=axes)


def brute_force_joint(dem):
    """Full joint P(detectors, L) over 2**(n+1) outcomes (bit n = L)."""
    n = dem.num_detectors
    nb = n + 1
    J = np.zeros((2,) * nb)
    J[(0,) * nb] = 1.0
    for p, dets, L in mechanisms(dem):
        bits = list(dets) + ([n] if L else [])
        if bits:
            J = xor_convolve(J, bits, p, nb)
    return J.reshape(-1)


def brute_llr(joint, n, events):
    idx = np.asarray(events, np.int64) @ (1 << np.arange(n)).astype(np.int64)
    with np.errstate(divide="ignore"):
        return np.log(joint[idx]) - np.log(joint[idx + (1 << n)])


class WideSlotMLD:
    """Own exact MLD. Slots are whole check rounds with the flags as ordinary
    detectors (12 per flagged round) plus the final readout slot; mechanisms
    may touch at most two consecutive slots (checked, raises otherwise)."""

    def __init__(self, dem, slot_of_detector):
        slots = np.asarray(slot_of_detector)
        self.T = int(slots.max()) + 1
        self.sd = [np.flatnonzero(slots == t) for t in range(self.T)]
        self.size = [len(d) for d in self.sd]
        self.pos = np.zeros(len(slots), dtype=np.int64)
        for d in self.sd:
            self.pos[d] = np.arange(len(d))
        self.slots = slots
        self.groups = {t: [] for t in range(self.T)}
        for p, dets, L in mechanisms(dem):
            if not dets:
                if L:
                    self.groups[0].append((p, (), 1))
                continue
            ts = slots[list(dets)]
            t = int(ts.max())
            if int(ts.min()) < t - 1:
                raise ValueError(f"mechanism spans slots {sorted(set(ts.tolist()))}")
            self.groups[t].append((p, dets, L))

    def _kernel(self, t):
        n_prev = self.size[t - 1] if t > 0 else 0
        n_cur = self.size[t]
        nb = n_prev + n_cur + 1
        K = np.zeros((2,) * nb)
        K[(0,) * nb] = 1.0
        for p, dets, L in self.groups[t]:
            bits = [(n_prev + self.pos[d]) if self.slots[d] == t else self.pos[d]
                    for d in dets] + ([n_prev + n_cur] if L else [])
            K = xor_convolve(K, bits, p, nb)
        return K.reshape(-1)

    def llr(self, events, batch=1000):
        events = np.asarray(events, dtype=bool)
        shots = events.shape[0]
        vals = [events[:, d].astype(np.int64) @ (1 << np.arange(len(d))).astype(np.int64)
                for d in self.sd]
        K0 = self._kernel(0)
        psi = np.broadcast_to(K0, (shots, K0.size)).copy()
        for t in range(1, self.T):
            n_prev, n_cur = self.size[t - 1], self.size[t]
            K = self._kernel(t).reshape(2, 2 ** n_cur, 2 ** n_prev)
            M = np.empty((2 * 2 ** n_prev, 2 * 2 ** n_cur))
            for li in range(2):
                for lo in range(2):
                    M[li * 2 ** n_prev:(li + 1) * 2 ** n_prev,
                      lo * 2 ** n_cur:(lo + 1) * 2 ** n_cur] = K[li ^ lo].T
            del K
            size = 2 ** n_prev
            base = np.arange(2 * size)
            new = np.empty((shots, 2 * 2 ** n_cur))
            for a in range(0, shots, batch):
                b = min(shots, a + batch)
                cols = (((base[None, :] & (size - 1)) ^ vals[t - 1][a:b, None])
                        | ((base[None, :] >> n_prev) << n_prev))
                new[a:b] = np.take_along_axis(psi[a:b], cols, axis=1) @ M
            del M
            psi = new / new.max(axis=1, keepdims=True)
        s = vals[-1]
        p0 = psi[np.arange(shots), s]
        p1 = psi[np.arange(shots), s + 2 ** self.size[-1]]
        with np.errstate(divide="ignore"):
            return np.log(p0) - np.log(p1)


def wide_slots(lc):
    slot = np.zeros(lc.num_detectors, dtype=np.int64)
    for k in range(lc.num_rounds):
        slot[lc.syndrome_columns[k]] = k
        if lc.flag_columns.size:
            slot[lc.flag_columns[k]] = k
    slot[lc.final_columns] = lc.num_rounds
    return slot


# ----------------------------------------------------------------------
# distance (K4, K5)
# ----------------------------------------------------------------------
def cmd_distance(args):
    SC, SV = under_test()
    rng = np.random.default_rng(7)
    own = []
    for hname, H in (("binary", hamming_binary()), ("cyclic", hamming_cyclic())):
        supports = [tuple(np.flatnonzero(r)) for r in H]
        order_sets = {"support order": None}
        for k in range(2):
            order_sets[f"random order {k}"] = [tuple(int(x) for x in rng.permutation(s))
                                               for s in supports + supports]
        for oname, orders in order_sets.items():
            for ext in ("naive", "flag"):
                for sched in ("sequential", "parallel"):
                    for basis in ("X", "Z"):
                        for rounds in (1, 2):
                            c = own_memory_circuit(H, ext, sched, basis, 1e-3,
                                                   rounds=rounds, orders=orders)
                            lb, amb = pair_bound(c.detector_error_model())
                            ds, g, wit = stim_search(c)
                            row = {"H": hname, "cnot_order": oname,
                                   "orders": orders, "extraction": ext,
                                   "schedule": sched, "basis": basis,
                                   "rounds": rounds, "pair_bound": lb,
                                   "ambiguous_classes": amb, "stim_search": ds,
                                   "shortest_graphlike": g, "witness": wit}
                            if ext == "flag":
                                nd = c.num_detectors
                                keep = np.ones(nd, bool)
                                for r in range(rounds):
                                    keep[12 * r + 6:12 * r + 12] = False
                                cf = drop_detectors(c, keep)
                                row["flags_dropped_pair_bound"] = pair_bound(
                                    cf.detector_error_model())[0]
                                row["flags_dropped_stim_search"] = stim_search(cf)[0]
                            own.append(row)
    summary = {}
    for r in own:
        key = f"{r['extraction']}"
        summary.setdefault(key, set()).add(
            (r["pair_bound"], r["stim_search"], r["shortest_graphlike"]))
    print("own memory circuits (pair bound, stim search, graphlike):",
          {k: sorted(v) for k, v in summary.items()})
    print("flagged with flag detectors dropped:",
          sorted({(r["flags_dropped_pair_bound"], r["flags_dropped_stim_search"])
                  for r in own if r["extraction"] == "flag"}))
    print("naive witness example:", own[0]["witness"])

    seqs = sequences()
    theirs = []
    for ext in ("naive", "flag"):
        code = SC.SteaneCode(ext)
        for sched in ("sequential", "parallel"):
            for si, seq in enumerate(seqs):
                lc = SV.build_lrb_circuit(code, seq, 2, 1e-3, schedule=sched)
                ds, g, wit = stim_search(lc.circuit)
                lb, amb = pair_bound(lc.circuit.detector_error_model())
                row = {"extraction": ext, "schedule": sched, "seq": si,
                       "depth": 2, "stim_search": ds, "shortest_graphlike": g,
                       "pair_bound": lb, "ambiguous_classes": amb,
                       "witness": wit if si == 0 else None}
                if ext == "flag":
                    keep = np.ones(lc.num_detectors, bool)
                    keep[lc.flag_columns.ravel()] = False
                    cf = drop_detectors(lc.circuit, keep)
                    row["flags_dropped_pair_bound"] = pair_bound(
                        cf.detector_error_model())[0]
                    row["flags_dropped_stim_search"] = stim_search(cf)[0]
                theirs.append(row)
            deep = set()
            for si, seq in enumerate(seqs):
                for m in DEPTHS:
                    lc = SV.build_lrb_circuit(code, seq, m, 1e-3, schedule=sched)
                    deep.add(pair_bound(lc.circuit.detector_error_model())[0])
            theirs.append({"extraction": ext, "schedule": sched,
                           "pair_bound_all_seqs_all_grid_depths": sorted(deep)})
            sub = [r for r in theirs if r.get("depth") == 2
                   and r["extraction"] == ext and r["schedule"] == sched]
            print(ext, sched, "depth 2, 10 seqs (stim, graphlike, pair):",
                  sorted({(r["stim_search"], r["shortest_graphlike"],
                           r["pair_bound"]) for r in sub}),
                  "| pair bound over all grid depths:", sorted(deep),
                  "| seq0 witness:", sub[0]["witness"], flush=True)
    dump("distance.json", {"own_memory_circuits": own,
                           "implementer_circuits": theirs})


# ----------------------------------------------------------------------
# first-order (K7, K8, K13, K14)
# ----------------------------------------------------------------------
def _fo_task(job):
    ext, sched, si, seq, m, p = job
    SC, SV = under_test()
    lc = SV.build_lrb_circuit(SC.SteaneCode(ext), seq, m, p, schedule=sched)
    fo = first_order(lc.circuit.detector_error_model())
    return ext, sched, si, m, {k: (v / p if k in ("U1", "R1", "E1") else
                                   v / p ** 2 if k == "A2" else v)
                               for k, v in fo.items()}


def cmd_first_order(args):
    p = 1e-6
    seqs = sequences()
    variants = [(e, s) for e in ("naive", "flag")
                for s in ("sequential", "parallel")]
    jobs = [(e, s, si, seqs[si], m, p) for e, s in variants
            for si in range(NUM_SEQ) for m in DEPTHS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_fo_task, jobs, chunksize=4))
    table = {}
    for e, s, si, m, fo in rows:
        table.setdefault(f"{e}:{s}", {}).setdefault(si, {})[m] = fo
    out = {"p": p, "units": "U1, R1, E1 per p; A2 per p^2", "variants": {}}
    for key, per_seq in table.items():
        entry = {}
        for q in ("R1", "E1", "A2", "U1"):
            slopes = [np.polyfit(DEPTHS, [per_seq[si][m][q] for m in DEPTHS], 1)[0]
                      for si in range(NUM_SEQ)]
            entry[q] = {"slope_mean": float(np.mean(slopes)),
                        "slope_sem": float(np.std(slopes, ddof=1) / np.sqrt(NUM_SEQ)),
                        "depth0": float(np.mean([per_seq[si][0][q] for si in range(NUM_SEQ)]))}
        entry["max_ambiguous_classes"] = int(max(per_seq[si][m]["ambiguous"]
                                                 for si in range(NUM_SEQ) for m in DEPTHS))
        out["variants"][key] = entry
        print(key, {q: (round(v["slope_mean"], 4), round(v["slope_sem"], 4),
                        round(v["depth0"], 4)) for q, v in entry.items()
                    if isinstance(v, dict)}, flush=True)

    # Frame decomposition of the naive first-order floor. With the sequence
    # [U, I, I, ...] every forward round sees the Heisenberg observable
    # U X_L U^dagger (X, Z or Y frame); the slope of E1 is that frame's
    # per-round floor. A uniformly random Clifford frame is X, Y, Z with
    # probability 1/3 each, so the ensemble floor is their mean.
    SC, SV = under_test()
    frames = {"X": "I", "Z": "H", "Y": "P"}
    for desc in frames.values():
        assert desc in C.SINGLE_QUBIT_CLIFFORDS
    frame_out = {}
    for ext in ("naive",):
        for sched in ("sequential", "parallel"):
            vals = {}
            for fr, desc in frames.items():
                seq = [desc] + ["I"] * 22
                e1 = []
                for m in (4, 8, 12):
                    lc = SV.build_lrb_circuit(SC.SteaneCode(ext), seq, m, p,
                                              schedule=sched)
                    e1.append(first_order(lc.circuit.detector_error_model())["E1"] / p)
                vals[fr] = float((e1[2] - e1[0]) / 8)
                assert abs((e1[1] - e1[0]) / 4 - vals[fr]) < 1e-6
            vals["mean_over_frames_identity_neighbours"] = float(
                np.mean([vals[f] for f in "XYZ"]))
            # predicted slope for the 10 sequences from their frames
            frame_out[f"{ext}:{sched}"] = vals
            print(ext, sched, "per-round first-order floor by frame:", vals, flush=True)
    # frames of the 10 sequences (fraction of forward rounds in each frame)
    frac = {"X": 0, "Y": 0, "Z": 0}
    for seq in seqs:
        U = stim.Tableau(1)
        for desc in seq[:22]:
            T = stim.Tableau(1)
            for tok in C.clifford_gate_sequence(desc):
                g = {"P": "S", "I": "I", "H": "H", "X": "X", "Z": "Z"}[tok[0]]
                T = T.then(stim.Tableau.from_named_gate(g))
            U = U.then(T)
            obs = U(stim.PauliString("X"))
            frac["XYZ"[[obs[0] == 1, obs[0] == 2, obs[0] == 3].index(True)]] += 1
    tot = sum(frac.values())
    frame_out["frame_fractions_10_sequences"] = {k: v / tot for k, v in frac.items()}
    nv = frame_out["naive:parallel"]
    frame_out["predicted_E1_slope_10_sequences"] = float(
        sum(frame_out["frame_fractions_10_sequences"][f] * nv[f] for f in "XYZ"))
    print("frame fractions over the 10 sequences:", frame_out["frame_fractions_10_sequences"],
          "-> predicted E1 slope", frame_out["predicted_E1_slope_10_sequences"])
    # Per-round increments for sequence 0 (neighbouring Cliffords matter).
    inc = []
    for m in range(1, 23):
        e1 = [first_order(SV.build_lrb_circuit(SC.SteaneCode("naive"), seqs[0], d, p,
                                               schedule="parallel").circuit
                          .detector_error_model())["E1"] / p for d in (m - 1, m)]
        inc.append(round(15 * (e1[1] - e1[0]), 6))
    frame_out["seq0_per_round_increments_times_15"] = inc
    print("seq0 per-round increments x15:", inc, flush=True)
    out["naive_frame_decomposition"] = frame_out
    # Ensemble per-step floor from fresh random sequences: E[E1(22) - E1(4)]/18.
    rng_seqs = E.clifford_sequences(args.ensemble, 22, 777000)
    jobs = [("naive", "parallel", si, s, m, p) for si, s in enumerate(rng_seqs)
            for m in (4, 22)]
    jobs += [("naive", "sequential", si, s, m, p)
             for si, s in enumerate(rng_seqs[:40]) for m in (4, 22)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_fo_task, jobs, chunksize=8))
    val = {}
    for e, s, si, m, fo in rows:
        val.setdefault((s, si), {})[m] = fo["E1"]
    ens = {}
    for sched in ("parallel", "sequential"):
        d = np.array([(v[22] - v[4]) / 18 for (s, _), v in val.items() if s == sched])
        ens[sched] = {"num_sequences": int(len(d)), "mean": float(d.mean()),
                      "sem": float(d.std(ddof=1) / np.sqrt(len(d))),
                      "min": float(d.min()), "max": float(d.max())}
    same = [abs(val[("parallel", i)][m] - val[("sequential", i)][m])
            for i in range(40) for m in (4, 22)]
    ens["max_abs_diff_parallel_vs_sequential"] = float(max(same))
    out["naive_ensemble_E1_per_step"] = ens
    print("naive ensemble E1 per step:", ens, flush=True)
    dump("first_order.json", out)


# ----------------------------------------------------------------------
# second-order (K9-K12, K18)
# ----------------------------------------------------------------------
def _so_task(job):
    sched, si, seq, m, p = job
    SC, SV = under_test()
    t0 = time.time()
    lc = SV.build_lrb_circuit(SC.SteaneCode("flag"), seq, m, p, schedule=sched)
    so = second_order(lc.circuit.detector_error_model())
    so["seconds"] = time.time() - t0
    return sched, si, m, so


def _const0_task(job):
    si, seq, m, ps = job
    SC, SV = under_test()
    out = {}
    for p in ps:
        lc = SV.build_lrb_const0_circuit(SC.SteaneCode("flag"), seq, m, p)
        dem = lc.circuit.detector_error_model()
        J = brute_force_joint(dem)
        n = dem.num_detectors
        out[f"{p:g}"] = float(np.minimum(J[: 1 << n], J[1 << n:]).sum())
    return si, m, out


def offset0_fit_exact(depths, ler_by_seq, shots=None):
    """Repo LRB fit of F = 1 - 2 LER (sequence mean; weights from the
    sequence std, floored by binomial shot noise at `shots` per circuit)."""
    F = 1.0 - 2.0 * np.asarray(ler_by_seq)
    mean = F.mean(axis=0)
    std = F.std(axis=0)
    if shots:
        ler = np.asarray(ler_by_seq).mean(axis=0)
        std = np.sqrt(std ** 2 + 4 * ler * (1 - ler) / shots)
    fit = FT.fit_fixed_offset(depths, mean, std)
    return (1 - fit["f"]) / 2


def cmd_second_order(args):
    p = 1e-6
    seqs = sequences()
    jobs = [(s, si, seqs[si], m, p) for s in ("parallel", "sequential")
            for si in range(NUM_SEQ) for m in DEPTHS]
    jobs.sort(key=lambda j: -j[3])
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for sched, si, m, so in pool.map(_so_task, jobs):
            res.setdefault(sched, {}).setdefault(si, {})[m] = so
    print(f"second-order sweep {time.time() - t0:.0f}s", flush=True)
    out = {"p_dem": p, "units": "E2 in units of p^2", "variants": {}}
    for sched, per_seq in res.items():
        E2 = np.array([[per_seq[si][m]["E2"] / p ** 2 for m in DEPTHS]
                       for si in range(NUM_SEQ)])
        und = max(per_seq[si][m]["undetectable_pairs"] for si in range(NUM_SEQ)
                  for m in DEPTHS)
        slopes = np.array([np.polyfit(DEPTHS, E2[si], 1)[0] for si in range(NUM_SEQ)])
        slopes_ge2 = np.array([np.polyfit(DEPTHS[1:], E2[si, 1:], 1)[0]
                               for si in range(NUM_SEQ)])
        quad = np.polyfit(DEPTHS, E2.mean(axis=0), 2)
        entry = {"E2_mean_by_depth": dict(zip(DEPTHS, E2.mean(axis=0).tolist())),
                 "E2_by_seq": E2.tolist(),
                 "linear_slope_mean": float(slopes.mean()),
                 "linear_slope_sem": float(slopes.std(ddof=1) / np.sqrt(NUM_SEQ)),
                 "linear_slope_depths_ge_2_mean": float(slopes_ge2.mean()),
                 "quadratic_fit_coeffs_m2_m1_m0": quad.tolist(),
                 "max_undetectable_pair_weight": und}
        # Offset-0 repo fit applied to the exact second-order curves.
        fits = {}
        for pp in (1e-5, 1e-4, 5e-4, 1e-3):
            for shots in (None, 100000):
                eps = offset0_fit_exact(DEPTHS, E2 * pp ** 2, shots)
                fits[f"p={pp:g}, shots={shots}"] = eps / pp ** 2
        entry["offset0_fit_eps_over_p2"] = fits
        # pseudo-threshold from the exact coefficient (pure p^2 law)
        entry["pstar_pure_p2_vs_pRB_p"] = float(0.338889 / slopes.mean())
        entry["pstar_pure_p2_vs_pRB_5p"] = float(5 * 0.338889 / slopes.mean())
        out["variants"][f"flag:{sched}"] = entry
        print(sched, "E2(m)/p^2 mean:", np.round(E2.mean(axis=0), 1).tolist())
        print(sched, "slope per step", round(slopes.mean(), 1), "+-",
              round(slopes.std(ddof=1) / np.sqrt(NUM_SEQ), 1),
              "| depths>=2:", round(slopes_ge2.mean(), 1),
              "| quad coeffs", np.round(quad, 3).tolist(),
              "| offset-0 fits", {k: round(v, 1) for k, v in fits.items()},
              "| undetectable pair weight", und, flush=True)

    # const0: exact curves (3 detectors, brute force at the actual p)
    ps = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    jobs0 = [(si, seqs[si], m, ps) for si in range(NUM_SEQ) for m in DEPTHS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows0 = list(pool.map(_const0_task, jobs0))
    ler = {f"{pp:g}": np.zeros((NUM_SEQ, len(DEPTHS))) for pp in ps}
    for si, m, vals in rows0:
        for k, v in vals.items():
            ler[k][si, DEPTHS.index(m)] = v
    c0 = {}
    for k, arr in ler.items():
        pp = float(k)
        c0[k] = {"LER_mean_by_depth": dict(zip(DEPTHS, arr.mean(axis=0).tolist())),
                 "eps_over_p_offset0_fit_exact": offset0_fit_exact(DEPTHS, arr) / pp,
                 "eps_over_p_offset0_fit_shot_floor_1e5": offset0_fit_exact(DEPTHS, arr, 100000) / pp,
                 "LER22_over_LER10": float(arr.mean(axis=0)[-1] / arr.mean(axis=0)[4])}
        print("const0", k, {kk: (round(v, 4) if isinstance(v, float) else None)
                            for kk, v in c0[k].items() if kk != "LER_mean_by_depth"},
              "LER(m)/p^2:", np.round(arr.mean(axis=0) / pp ** 2, 2).tolist(), flush=True)
    out["const0_exact"] = c0
    dump("second_order.json", out)


# ----------------------------------------------------------------------
# decoder (K6, task 5)
# ----------------------------------------------------------------------
def _dec_task(job):
    kind, ext, sched, si, m, p, shots, seed = job
    SC, SV = under_test()
    seq = sequences()[si]
    lc = SV.build_lrb_circuit(SC.SteaneCode(ext), seq, m, p, schedule=sched)
    dem = lc.circuit.detector_error_model()
    dets, obs = lc.circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)
    # add every single-mechanism symptom as an extra "shot"
    mechs = mechanisms(dem)
    single = np.zeros((len(mechs), dem.num_detectors), dtype=bool)
    for i, (_, d, _) in enumerate(mechs):
        single[i, list(d)] = True
    allev = np.concatenate([dets, single])
    t0 = time.time()
    theirs = SV.mld_for(lc, dem).log_likelihood_ratio(allev)
    t_theirs = time.time() - t0
    t0 = time.time()
    if kind == "brute":
        mine = brute_llr(brute_force_joint(dem), dem.num_detectors, allev)
    else:
        mine = WideSlotMLD(dem, wide_slots(lc)).llr(allev)
    t_mine = time.time() - t0
    fin = np.isfinite(mine) & np.isfinite(theirs)
    flag_cols = lc.flag_columns.ravel()
    raised = dets[:, flag_cols].any(axis=1) if flag_cols.size else np.zeros(len(dets), bool)
    return {"reference": kind, "extraction": ext, "schedule": sched, "seq": si,
            "depth": m, "p": p, "detectors": dem.num_detectors,
            "shots": shots, "single_mechanism_symptoms": len(mechs),
            "shots_with_raised_flag": int(raised.sum()),
            "max_abs_llr_diff": float(np.max(np.abs(mine[fin] - theirs[fin]))),
            "same_infinite_pattern": bool(np.array_equal(np.isfinite(mine), np.isfinite(theirs))),
            "decisions_differ": int(np.sum((mine < 0) != (theirs < 0))),
            "seconds_slotmld": t_theirs, "seconds_reference": t_mine}


def cmd_decoder(args):
    SC, SV = under_test()
    jobs = []
    for sched in ("sequential", "parallel"):
        jobs.append(("brute", "flag", sched, 1, 0, 2e-2, 20000, 11))
        jobs.append(("brute", "naive", sched, 2, 2, 8e-3, 20000, 12))
        for m, si in ((1, 3), (2, 4), (4, 5)):
            jobs.append(("wide", "flag", sched, si, m, 5e-3,
                         3000 if m == 4 else 4000, 13 + m))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(_dec_task, jobs):
            results.append(r)
            print({k: (round(v, 3) if isinstance(v, float) and k.startswith("seconds") else v)
                   for k, v in r.items()}, flush=True)

    # Flag sensitivity: for every flag-raising single mechanism of a depth-4
    # flagged circuit, clear its flag bits and decode again.
    sens = {}
    seq = sequences()[0]
    for sched in ("sequential", "parallel"):
        lc = SV.build_lrb_circuit(SC.SteaneCode("flag"), seq, 4, 1e-4, schedule=sched)
        dem = lc.circuit.detector_error_model()
        dec = SV.mld_for(lc, dem)
        flag_cols = set(lc.flag_columns.ravel().tolist())
        mechs = [m for m in mechanisms(dem) if set(m[1]) & flag_cols]
        ev = np.zeros((len(mechs), dem.num_detectors), dtype=bool)
        for i, (_, d, _) in enumerate(mechs):
            ev[i, list(d)] = True
        Ls = np.array([m[2] for m in mechs])
        with_flags = dec.log_likelihood_ratio(ev)
        cleared = ev.copy()
        cleared[:, sorted(flag_cols)] = False
        without = dec.log_likelihood_ratio(cleared)
        # A decoder that marginalises the flags (flag detectors deleted).
        keep = np.ones(dem.num_detectors, bool)
        keep[sorted(flag_cols)] = False
        nf_circ = drop_detectors(lc.circuit, keep)
        nf_dem = nf_circ.detector_error_model()
        fo_nf = first_order(nf_dem)
        from lrb_qubit import mld as M
        slots = wide_slots(lc)[keep]
        nf_dec = M.TransferMatrixMLD(nf_dem, slots)
        nf_llr = nf_dec.log_likelihood_ratio(ev[:, keep])
        sens[sched] = {
            "flag_raising_single_mechanisms": len(mechs),
            "decoded_correctly_with_flags": int(np.sum((with_flags < 0) == Ls)),
            "decision_changes_when_flags_cleared": int(np.sum((with_flags < 0) != (without < 0))),
            "flag_marginalised_decoder_wrong_on_these": int(np.sum((nf_llr < 0) != Ls)),
            "flag_marginalised_DEM_E1_over_p": fo_nf["E1"] / 1e-4,
            "flag_marginalised_DEM_ambiguous_classes": fo_nf["ambiguous"],
            "cached_flag_kernels_after_decoding": len(dec._cache)}
        print(sched, sens[sched], flush=True)
    dump("decoder.json", {"exactness": results, "flag_sensitivity": sens})


# ----------------------------------------------------------------------
# sample (task 4 sampled check; K16)
# ----------------------------------------------------------------------
def _pack_rows(dets):
    n = dets.shape[1]
    nb = (n + 63) // 64
    padded = np.zeros((dets.shape[0], nb * 64), dtype=bool)
    padded[:, :n] = dets
    packed = np.packbits(padded, axis=1, bitorder="little").view(np.uint64)
    return np.ascontiguousarray(packed).view([(f"w{i}", np.uint64) for i in range(nb)]).ravel()


def _ler_task(job):
    sched, si, m, p, shots, seed = job
    SC, SV = under_test()
    lc = SV.build_lrb_circuit(SC.SteaneCode("flag"), sequences()[si], m, p,
                              schedule=sched)
    dec = SV.mld_for(lc)
    sampler = lc.circuit.compile_detector_sampler(seed=seed)
    wrong = 0
    done = 0
    while done < shots:
        n = min(500000, shots - done)
        dets, obs = sampler.sample(n, separate_observables=True)
        keys = _pack_rows(dets)
        uniq, first, inv = np.unique(keys, return_index=True, return_inverse=True)
        pred = (dec.log_likelihood_ratio(dets[first]) < 0)
        wrong += int(np.sum(pred[inv] != obs[:, 0]))
        done += n
    return sched, si, m, p, shots, wrong


def _llr4_task(job):
    sched, si, m, p, shots, seed = job
    SC, SV = under_test()
    lc = SV.build_lrb_circuit(SC.SteaneCode("flag"), sequences()[si], m, p,
                              schedule=sched)
    dec = SV.mld_for(lc)
    dets, obs = lc.circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)
    llr = dec.log_likelihood_ratio(dets)
    wrong = (llr < 0) != obs[:, 0]
    acc = np.abs(llr) >= 4.0
    return sched, si, m, p, shots, int(acc.sum()), int((acc & wrong).sum()), int(wrong.sum())


def cmd_sample(args):
    out = {}
    if "ler" in args.parts:
        jobs = []
        for sched, ps in (("parallel", (1e-4, 2e-4, 5e-4)),
                          ("sequential", (5e-5, 1e-4, 2e-4))):
            for p in ps:
                for k in range(args.chunks):
                    jobs.append((sched, 0, 4, p, args.ler_shots, 9000 + 37 * k
                                 + int(p * 1e7) + (1 if sched == "parallel" else 0)))
        agg = {}
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for sched, si, m, p, n, w in pool.map(_ler_task, jobs):
                a = agg.setdefault((sched, p), [0, 0])
                a[0] += n
                a[1] += w
        print(f"LER sampling {time.time() - t0:.0f}s", flush=True)
        SC, SV = under_test()
        ler = {}
        for (sched, p), (n, w) in sorted(agg.items()):
            lc = SV.build_lrb_circuit(SC.SteaneCode("flag"), sequences()[0], 4,
                                      1e-6, schedule=sched)
            e2 = second_order(lc.circuit.detector_error_model())["E2"] / 1e-12
            rate = w / n
            ler[f"{sched} p={p:g}"] = {"shots": n, "errors": w,
                                       "LER_over_p2": rate / p ** 2,
                                       "LER_over_p2_1sigma": np.sqrt(w) / n / p ** 2,
                                       "exact_second_order_E2_over_p2": e2}
            print(sched, p, ler[f"{sched} p={p:g}"], flush=True)
        # Higher-order term: fit LER/p^2 = E2 + c3 p (weighted) per schedule.
        for sched in ("parallel", "sequential"):
            pts = [(float(k.split("p=")[1]), v) for k, v in ler.items()
                   if k.startswith(sched)]
            x = np.array([q for q, _ in pts])
            y = np.array([v["LER_over_p2"] - v["exact_second_order_E2_over_p2"] for _, v in pts])
            w = 1 / np.array([v["LER_over_p2_1sigma"] for _, v in pts]) ** 2
            c3 = float(np.sum(w * x * y) / np.sum(w * x * x))
            chi2 = float(np.sum(w * (y - c3 * x) ** 2))
            e2 = pts[0][1]["exact_second_order_E2_over_p2"]
            ler[f"{sched} higher-order fit"] = {
                "c3_over_E2 (relative change per unit p)": c3 / e2,
                "relative_shift_at_p=5e-4": c3 * 5e-4 / e2,
                "relative_shift_at_p=1e-3": c3 * 1e-3 / e2,
                "chi2_of_E2_plus_c3p_with_E2_fixed": chi2, "points": len(pts)}
            print(sched, ler[f"{sched} higher-order fit"], flush=True)
        out["ler_depth4_seq0"] = ler
    if "llr4" in args.parts:
        p = 1e-3
        jobs = [("parallel", si, m, p, args.llr_shots, 500 + 31 * si + m)
                for si in range(NUM_SEQ) for m in DEPTHS]
        acc = np.zeros((NUM_SEQ, len(DEPTHS)), dtype=np.int64)
        err = np.zeros_like(acc)
        tot = np.zeros_like(acc)
        werr = np.zeros_like(acc)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for sched, si, m, pp, n, a, e, w in pool.map(_llr4_task, jobs):
                d = DEPTHS.index(m)
                acc[si, d], err[si, d], tot[si, d], werr[si, d] = a, e, n, w
        print(f"|LLR|>=4 sampling {time.time() - t0:.0f}s", flush=True)
        fit = FT.fit_from_counts(DEPTHS, acc, err, "lrb")
        fit_all = FT.fit_from_counts(DEPTHS, tot, werr, "lrb")
        rng = np.random.default_rng(1)
        boots = []
        for _ in range(200):
            pick = rng.integers(0, NUM_SEQ, NUM_SEQ)
            f = FT.fit_from_counts(DEPTHS, acc[pick], err[pick], "lrb")
            boots.append((1 - f["f"]) / 2 / p)
        a_mean = (acc / tot).mean(axis=0)
        a_step = float(np.exp(np.polyfit(DEPTHS, np.log(a_mean), 1)[0]))
        out["llr4_flag_parallel_p1e-3"] = {
            "shots_per_circuit": args.llr_shots,
            "eps_over_p": (1 - fit["f"]) / 2 / p, "eps_over_p_boot_std": float(np.std(boots)),
            "acceptance_per_step": a_step,
            "mld_no_rejection_eps_over_p": (1 - fit_all["f"]) / 2 / p}
        print(out["llr4_flag_parallel_p1e-3"], flush=True)
    name = "sample.json" if set(args.parts) == {"ler", "llr4"} else f"sample_{'_'.join(args.parts)}.json"
    dump(name, out)


# ----------------------------------------------------------------------
# thresholds (K10-K12 spot check)
# ----------------------------------------------------------------------
def cmd_thresholds(args):
    import csv
    rows = list(csv.DictReader(open(K_DIR / "fits.csv")))
    get = {(r["method"], float(r["p"])): r for r in rows}
    out = {}
    ps = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    for var in ("flag:parallel", "flag:sequential", "naive:parallel", "naive:sequential"):
        for rb in ("phys_rb_p", "phys_rb_5p"):
            ratio = [float(get[(f"{var}:mld", p)]["eps"]) / float(get[(rb, p)]["eps"])
                     for p in ps]
            g = np.log(ratio)
            star = None
            for i in range(len(ps) - 1):
                if g[i] < 0 <= g[i + 1]:
                    x0, x1 = np.log(ps[i]), np.log(ps[i + 1])
                    star = float(np.exp(x0 + (x1 - x0) * (-g[i]) / (g[i + 1] - g[i])))
                    break
            out[f"{var} vs {rb}"] = {"ratio_at_grid": dict(zip(map(str, ps), ratio)),
                                     "loglog_interpolated_pstar": star,
                                     "status": "crossing" if star else (
                                         "below grid" if g[0] >= 0 else "none on grid")}
            print(var, rb, out[f"{var} vs {rb}"], flush=True)
    summ = json.loads((K_DIR / "summary.json").read_text())
    out["implementer_summary_pseudo_thresholds"] = {
        k: {kk: v[kk] for kk in ("p_star", "status", "boot_lo", "boot_hi")}
        for k, v in summ["pseudo_thresholds"].items()}
    out["implementer_model_crossings"] = summ.get("model_crossings")
    so_path = OUT / "second_order.json"
    if so_path.exists():
        so = json.loads(so_path.read_text())
        for var, entry in so["variants"].items():
            c2 = entry["linear_slope_mean"]
            out[f"{var} exact c2 crossing"] = {
                "c2_exact_slope": c2,
                "pstar_vs_pRB_p_pure_p2": 0.338889 / c2,
                "pstar_vs_pRB_5p_pure_p2": 5 * 0.338889 / c2}
            print(var, out[f"{var} exact c2 crossing"])
    dump("thresholds.json", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("algebra")
    sub.add_parser("distance")
    for name in ("first-order", "second-order", "decoder"):
        sp = sub.add_parser(name)
        sp.add_argument("--workers", type=int, default=4)
        if name == "first-order":
            sp.add_argument("--ensemble", type=int, default=400)
    sp = sub.add_parser("sample")
    sp.add_argument("--workers", type=int, default=4)
    sp.add_argument("--parts", default="ler,llr4", type=lambda s: s.split(","))
    sp.add_argument("--ler-shots", type=int, default=5_000_000)
    sp.add_argument("--chunks", type=int, default=8)
    sp.add_argument("--llr-shots", type=int, default=100_000)
    sub.add_parser("thresholds")
    args = ap.parse_args()
    {"algebra": cmd_algebra, "distance": cmd_distance,
     "first-order": cmd_first_order, "second-order": cmd_second_order,
     "decoder": cmd_decoder, "sample": cmd_sample,
     "thresholds": cmd_thresholds}[args.cmd](args)


if __name__ == "__main__":
    main()
