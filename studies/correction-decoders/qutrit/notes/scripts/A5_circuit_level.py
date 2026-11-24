#!/usr/bin/env python3
"""A5 circuit-level picture.

Part A (simulation of the actual baseline circuits, exact at first order in p):
  For stored [[5,1,2]]_3 LRB circuits of the baseline SI1000 run (read-only), the
  repository's CompactDetectorErrorModel gives the linear Z_3 response of every
  detector and of the logical observable to every unit Weyl component of every
  noise location.  Every single fault (location x non-zero Weyl vector) is
  enumerated with its probability (prob/8 for N1 'd', prob/80 for N2, prob/2 for
  the 'f' shift channel) and grouped by its full detector signature (all
  rounds).  At first order in p:
    detection (unif 1):  rejection = mass(signature != 0),
                         undetected logical = mass(signature = 0, shift != 0)
    detection terminal-only (const 1): same with the final round's detectors only
    optimal correction (p -> 0 limit of MLD on the whole detector history):
                         failure = undetected + sum_sig [mass(sig) - max_k mass(sig, k)]
    hybrid: reject signatures whose single-fault explanations give >1 shift.
  Each quantity (in units of p) is fitted linearly in the depth m; the slope is
  the per-depth-step coefficient.  The fidelity metric |sum_k w^k P(k)| equals
  1 - 1.5 P(shift != 0) at first order.  The failing mass of correction is
  attributed to fault sources.

Part B (analysis, no new simulation): SI1000 noise budget per data qutrit per
  check round for each candidate code, the d = 2 prediction from the
  code-capacity tie rule, and the d = 3 estimate eps ~ A (kappa p)^2 built from
  the exact phenomenological coefficients of A5_code_capacity_compare.py.

Part C (simulation of a d = 3 memory under SI1000, exact at second order in p):
  see section C below.

Run: module load python/3.11; python A5_circuit_level.py [--cliffs 6] [--quick]
Reads only the baseline run folder; writes A5_out/A5_circuit_level.json and
A5_out/A5_circuit_level_log.txt.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "A5_out")
WORKTREE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
       "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
P_IDX, P_VAL = 4, 1e-3
DATA = set(range(5))
LOG: list[str] = []
RES: dict = {}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------

def nonzero_vectors(width):
    return [v for v in itertools.product(range(3), repeat=width) if any(v)]


V2 = np.array(nonzero_vectors(2), dtype=np.int64)
V4 = np.array(nonzero_vectors(4), dtype=np.int64)


def category(loc, op_before):
    ch = loc.channel
    scale = float(loc.params.get("prob_scale", "nan"))
    q, t = loc.qudit, loc.target
    if loc.gate == "N2":
        if q in DATA and t in DATA:
            return "logical two-qutrit gate (SWAP/CZ)"
        return "check CNOT"
    if ch == "f":
        return "measurement flip (5p)" if abs(scale - 5) < 1e-9 else "reset error (2p)"
    if abs(scale - 0.1) < 1e-9:
        return "single-qutrit logical gate (0.1p)" if q in DATA else "ancilla H/H_INV (0.1p)"
    if abs(scale - 2) < 1e-9:
        return "data idle in measurement/reset layers (2p)" if q in DATA else "ancilla idle while others are measured (2p)"
    if abs(scale - 1) < 1e-9:
        return "measured-ancilla depolarizing (p)"
    return f"other ({loc.gate} {ch} {scale})"


def first_order(model, circuit, p):
    """Enumerate single faults; return dict of first-order masses (units of p)."""
    nd = model.num_detectors
    final = [i for i, lab in enumerate(model.detector_labels)]
    rounds = sorted({int(lab.split("_r")[1].split("_")[0]) for lab in model.detector_labels})
    last = max(rounds)
    final = [i for i, lab in enumerate(model.detector_labels) if f"_r{last}_" in lab]
    sigs, shifts, masses, cats = [], [], [], []
    for loc in model.locations:
        if loc.gate == "N1":
            if loc.channel == "d":
                vecs, pv = V2, loc.probability / 8.0
            elif loc.channel == "f":
                vecs, pv = np.array([[1, 0], [2, 0]]), loc.probability / 2.0
            elif loc.channel == "p":
                vecs, pv = np.array([[0, 1], [0, 2]]), loc.probability / 2.0
            else:
                raise ValueError(loc.channel)
        else:
            vecs, pv = V4, loc.probability / 80.0
            pd = loc.params.get("prob_dist")
            if pd is not None:
                arr = np.array(pd, dtype=float)
                assert np.allclose(arr[1:], arr[1]), "non-uniform N2"
        D = np.zeros((len(loc.component_terms), nd), dtype=np.int64)
        Lg = np.zeros(len(loc.component_terms), dtype=np.int64)
        for c, terms in enumerate(loc.component_terms):
            for (i, v) in terms.detector:
                D[c, i] = v
            for (i, v) in terms.logical:
                Lg[c] = v
        S = (vecs @ D) % 3
        Lsh = (vecs @ Lg) % 3
        cat = category(loc, None)
        for k in range(len(vecs)):
            sigs.append(S[k])
            shifts.append(int(Lsh[k]))
            masses.append(pv / p)
            cats.append(cat)
    sigs = np.array(sigs, dtype=np.int8)
    shifts = np.array(shifts)
    masses = np.array(masses)
    uniq, inv = np.unique(sigs, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    zero = np.nonzero(~uniq.any(axis=1))[0]
    zero = int(zero[0]) if len(zero) else -1
    M = np.zeros((len(uniq), 3))
    np.add.at(M, (inv, shifts), masses)
    out = {}
    nzmask = np.ones(len(uniq), dtype=bool)
    if zero >= 0:
        nzmask[zero] = False
    out["detect_rej"] = float(M[nzmask].sum())
    out["undetected"] = float(M[zero, 1:].sum()) if zero >= 0 else 0.0
    kstar = np.argmax(M, axis=1)
    if zero >= 0:
        kstar[zero] = 0
    fail_mask = shifts != kstar[inv]
    out["mld_fail"] = float(masses[fail_mask].sum())
    amb = (M > 0).sum(axis=1) > 1
    amb[zero] = False if zero >= 0 else amb[zero]
    out["hybrid_rej"] = float(M[amb].sum())
    # terminal-only detection
    term_nz = np.any(sigs[:, final] != 0, axis=1)
    out["terminal_rej"] = float(masses[term_nz].sum())
    out["total_mass"] = float(masses.sum())
    # failure attribution and total mass by category
    cats = np.array(cats)
    out["mld_fail_by_cat"] = {c: float(masses[fail_mask & (cats == c)].sum()) for c in sorted(set(cats))}
    out["mass_by_cat"] = {c: float(masses[cats == c].sum()) for c in sorted(set(cats))}
    out["detected_mass_by_cat"] = {c: float(masses[(cats == c) & (inv != zero)].sum()) for c in sorted(set(cats))}
    return out


def part_A(args):
    log("=== Part A: baseline [[5,1,2]]_3 LRB circuits, exact first order in p (repository DEM) ===")
    sys.path.insert(0, os.path.join(WORKTREE, "src"))
    from sdim.circuit_io import read_circuit
    from lrb.dem_simulation import build_compact_detector_error_model
    cliffs = list(range(args.cliffs))
    didx = [0, 1, 2, 3, 4] if not args.quick else [0, 2, 4]
    per = {}
    t0 = time.time()
    for c in cliffs:
        for di in didx:
            path = os.path.join(RUN, "experiments", "LRB", str(c), str(P_IDX), f"{di}.chp")
            circ = read_circuit(path)
            model = build_compact_detector_error_model(circ)
            per[(c, DEPTHS[di])] = first_order(model, circ, P_VAL)
    log(f"   {len(per)} circuits compiled and enumerated in {time.time() - t0:.1f}s "
        f"(sequences {cliffs}, depths {[DEPTHS[d] for d in didx]}, p = {P_VAL})")
    keys = ["detect_rej", "undetected", "mld_fail", "hybrid_rej", "terminal_rej", "total_mass"]
    fits = {}
    ms = np.array([DEPTHS[d] for d in didx], dtype=float)
    for k in keys:
        slopes, inters = [], []
        for c in cliffs:
            y = np.array([per[(c, DEPTHS[d])][k] for d in didx])
            b, a = np.polyfit(ms, y, 1)
            slopes.append(b)
            inters.append(a)
        fits[k] = {"slope_mean": float(np.mean(slopes)), "slope_sd": float(np.std(slopes)),
                   "intercept_mean": float(np.mean(inters))}
    cats = sorted(per[(cliffs[0], DEPTHS[didx[-1]])]["mass_by_cat"])
    catfit = {}
    for cat in cats:
        sl_f, sl_m, sl_d = [], [], []
        for c in cliffs:
            y_f = np.array([per[(c, DEPTHS[d])]["mld_fail_by_cat"].get(cat, 0.0) for d in didx])
            y_m = np.array([per[(c, DEPTHS[d])]["mass_by_cat"].get(cat, 0.0) for d in didx])
            y_d = np.array([per[(c, DEPTHS[d])]["detected_mass_by_cat"].get(cat, 0.0) for d in didx])
            sl_f.append(np.polyfit(ms, y_f, 1)[0])
            sl_m.append(np.polyfit(ms, y_m, 1)[0])
            sl_d.append(np.polyfit(ms, y_d, 1)[0])
        catfit[cat] = {"mld_fail_per_step": float(np.mean(sl_f)), "mass_per_step": float(np.mean(sl_m)),
                       "detected_mass_per_step": float(np.mean(sl_d))}
    RES["A"] = {"fits_units_of_p": fits, "by_category_per_step": catfit,
                "per_circuit": {f"{k[0]}_{k[1]}": v for k, v in per.items()}}
    F = fits
    log("   per depth step (slope) and intercept, units of p, mean over sequences:")
    for k in keys:
        log(f"     {k:14s}: slope {F[k]['slope_mean']:8.3f} (sd {F[k]['slope_sd']:.3f}), "
            f"intercept {F[k]['intercept_mean']:8.3f}")
    one_minus_f = 1.5 * F["mld_fail"]["slope_mean"]
    eps_corr = 8.0 / 9.0 * one_minus_f
    rb_f = 0.9943650488263496          # fit table, p_raw = 1e-3, unif 1 row (rb_f): physical RB per step
    one_minus_f_rb = (1 - rb_f) / P_VAL
    RES["A"]["summary"] = {"correction_1_minus_f_per_step": one_minus_f, "correction_eps_per_step": eps_corr,
                           "physical_RB_1_minus_f_per_step": one_minus_f_rb,
                           "physical_RB_eps_per_step": 8.0 / 9.0 * one_minus_f_rb,
                           "ratio": one_minus_f / one_minus_f_rb}
    log(f"   optimal correction: 1 - f = 1.5 x {F['mld_fail']['slope_mean']:.3f} p = {one_minus_f:.2f} p per depth step "
        f"(twirled eps = {eps_corr:.2f} p); physical RB at p_RB = 5p [fit table]: 1 - f = {one_minus_f_rb:.3f} p "
        f"(eps = {8 / 9 * one_minus_f_rb:.2f} p); ratio {one_minus_f / one_minus_f_rb:.2f}")
    log("   failure and mass per depth step by source (units of p):")
    for cat, v in sorted(catfit.items(), key=lambda kv: -kv[1]["mld_fail_per_step"]):
        log(f"     {cat:48s}: correction failure {v['mld_fail_per_step']:6.3f}   fault mass {v['mass_per_step']:7.3f}"
            f"   detected mass {v['detected_mass_per_step']:7.3f}")
    return RES["A"]


# ---------------------------------------------------------------------------
# Part B: SI1000 budget per data qutrit per check round
# ---------------------------------------------------------------------------

def part_B():
    log("\n=== Part B: SI1000 noise budget per data qutrit per check round (analysis) ===")
    # catalog averages of logical tokens per Clifford (216-element catalog, from the repository)
    sys.path.insert(0, os.path.join(WORKTREE, "src"))
    try:
        from lrb.code_definitions import single_qutrit_cliffords_exhaustive_strings as strings  # type: ignore
    except Exception:
        strings = None
    tok = {"H": 0.0, "P": 0.0, "X": 0.0, "Z": 0.0}
    if strings:
        for s_ in strings:
            toks = s_ if isinstance(s_, (list, tuple)) else list(s_)
            for t in toks:
                t0 = str(t)[0]
                if t0 in tok:
                    tok[t0] += 1
        for k in tok:
            tok[k] /= len(strings)
    RES["B_catalog_tokens_per_clifford"] = tok
    log(f"   logical tokens per Clifford (catalog average): {tok}  [{len(strings) if strings else 'catalog not found'} strings]")
    nH, nP, nX, nZ = tok["H"], tok["P"], tok["X"], tok["Z"]
    # d = 2 per-qutrit marginal depolarizing per depth step (forward Clifford + check round + inverse Clifford)
    cnots = {0: 2, 1: 4, 2: 2, 3: 2, 4: 2}
    idle = 4 * 2 + 2
    gate1 = {0: 0.1 * (nH + nP + nX + nZ), 1: 0.1 * (nH + nP), 2: 0.1 * (nH + nZ), 3: 0.1 * (nH + nX), 4: 0.1 * (nH + nP)}
    gate2 = {2: 0.9 * (nH + nP), 3: 0.9 * (nH + nP)}
    q = {j: idle + 0.9 * cnots[j] + 2 * (gate1[j] + gate2.get(j, 0.0)) for j in range(5)}
    eps_tie = 0.25 * (min(q[0], q[3]) + min(q[2], q[4]) + min(q[0], q[2]) + min(q[3], q[4]))
    RES["B_d2"] = {"q_per_qutrit_per_step": q, "eps_tie_rule": eps_tie}
    log(f"   [[5,1,2]]_3 per data qutrit per depth step (units of p): " + ", ".join(f"q{j} = {q[j]:.2f}" for j in q))
    log(f"   code-capacity tie rule eps = (1/4)[min(q0,q3)+min(q2,q4)+min(q0,q2)+min(q3,q4)] = {eps_tie:.2f} p per step")
    return RES["B_d2"]


# ---------------------------------------------------------------------------
# Part C: SI1000 circuit-level memory / LRB-like rounds for the candidate codes,
# exact leading order (first and second order in p) of space-time MLD
# ---------------------------------------------------------------------------
#
# Circuit per round (all choices stated, following the brief's SI1000 rules):
#   optional logical gates (fold-transversal H_L / P_L tokens, X_L, Z_L);
#   Z checks then X checks, one ancilla per check; Z check: CNOT(^-1) data -> anc;
#   X check: H anc, CNOT(^-1) anc -> data, H_INV anc; coupling order = support order
#   (or a given hook-safe order);
#   readout 'seq': every check is measured right after its CNOTs, one ancilla per
#   measurement layer (the repository style); 'par': one measurement layer for all
#   ancillas after all CNOTs;
#   noise: N1 0.1p after 1-qutrit gates, N2 p after 2-qutrit gates; measurement
#   layer: N1 2p on every other wire, N1 p on each measured ancilla, X-shift 5p
#   before M; reset layer: N1 2p on data, X-shift 2p on each reset ancilla;
#   no idle noise during gate layers.
# T noisy rounds, then a perfect syndrome readout of the data.  Signature = raw
# ancilla outcomes of all rounds + final syndrome; label = logical class of the
# final data error.  All single faults are propagated exactly (Pauli frames);
# fault pairs are formed over distinct single-fault effects with the
# same-location exclusion applied exactly.

sys.path.insert(0, HERE)


def load_codes():
    import A5_code_capacity_compare as M5
    return M5, {"512": M5.code_512(), "513": M5.code_513(), "713": M5.code_713(),
                "913": M5.code_913(), "1313": M5.code_1313()}


def round_ops(M5, code, readout, logical_tokens=(), orders=None):
    n, r = code.n, code.r
    W = n + r
    ops = []

    def g1(name, q):
        ops.append(("G1", name, q))
        ops.append(("N1", q, 0.1, "d"))

    def g2(name, a, b):
        ops.append(("G2", name, a, b))
        ops.append(("N2", a, b, 1.0))

    for tok in logical_tokens:
        if tok == "H":
            circ = M5.fold_H(code) if "pairs" in code.meta else M5.repo_H(code)
        elif tok == "P":
            circ = M5.fold_P(code) if "pairs" in code.meta else M5.repo_P(code)
        elif tok == "X":
            circ = [("X", q) for q in range(n) if code.XL[q] % 3]
        elif tok == "Z":
            circ = [("Z", q) for q in range(n) if code.ZL[n + q] % 3]
        else:
            raise ValueError(tok)
        for g in circ:
            if len(g) == 2:
                g1(g[0], g[1])
            else:
                g2(g[0], g[1], g[2])
    phen = readout == "phen"
    if phen:                      # validation mode: data depolarizing q and outcome shifts q_m = q only
        ops = [op for op in ops if op[0] not in ("N1", "N2")]
        for q in range(n):
            ops.append(("N1", q, 1.0, "d"))

        def g1(name, q):          # noqa: F811
            ops.append(("G1", name, q))

        def g2(name, a, b):       # noqa: F811
            ops.append(("G2", name, a, b))
    supp = code.supports()
    zchecks = [k for k in range(r) if not np.any(code.G[k, :n])]
    xchecks = [k for k in range(r) if not np.any(code.G[k, n:])]
    mixed = [k for k in range(r) if k not in zchecks and k not in xchecks]
    order_of = orders or {k: supp[k] for k in range(r)}

    def meas_layer(ancs):
        anc_set = set(ancs)
        if phen:
            for a in ancs:
                ops.append(("N1", a, 1.0, "f"))
                ops.append(("M", a))
            return
        for w in range(W):
            if w not in anc_set:
                ops.append(("N1", w, 2.0, "d"))
        for a in ancs:
            ops.append(("N1", a, 1.0, "d"))
            ops.append(("N1", a, 5.0, "f"))
            ops.append(("M", a))

    for k in zchecks + xchecks + mixed:
        a = n + k
        g = code.G[k]
        if k in zchecks:
            for q in order_of[k]:
                g2("CNOT" if g[n + q] == 1 else "CNOT_INV", q, a)
        elif k in xchecks:
            g1("H", a)
            for q in order_of[k]:
                g2("CNOT" if g[q] == 1 else "CNOT_INV", a, q)
            g1("H_INV", a)
        else:
            # mixed check X^x Z^z on each qutrit: controlled-X^x then controlled-Z^z from an
            # ancilla in the Fourier basis (CNOT^x anc->q, CZ^z anc-q)
            g1("H", a)
            for q in order_of[k]:
                if g[q]:
                    g2("CNOT" if g[q] == 1 else "CNOT_INV", a, q)
                if g[n + q]:
                    g2("CZ" if g[n + q] == 1 else "CZ_INV", a, q)
            g1("H_INV", a)
        if readout == "seq":
            meas_layer([a])
    if readout in ("par", "phen"):
        meas_layer([n + k for k in zchecks + xchecks + mixed])
    for k in range(r):
        ops.append(("R", n + k))
    if phen:
        return ops
    for q in range(n):
        ops.append(("N1", q, 2.0, "d"))
    for k in range(r):
        ops.append(("N1", n + k, 2.0, "f"))
    return ops


def apply_sym(F, W, name, a, b=None):
    if b is None:
        if name in ("X", "Z", "I"):
            return
        x = F[:, a].copy()
        z = F[:, W + a].copy()
        if name == "H":
            F[:, a], F[:, W + a] = (-z) % 3, x
        elif name == "H_INV":
            F[:, a], F[:, W + a] = z, (-x) % 3
        elif name == "P":
            F[:, W + a] = (z + x) % 3
        elif name == "P_INV":
            F[:, W + a] = (z - x) % 3
        else:
            raise ValueError(name)
        return
    if name == "CNOT":
        F[:, b] = (F[:, b] + F[:, a]) % 3
        F[:, W + a] = (F[:, W + a] - F[:, W + b]) % 3
    elif name == "CNOT_INV":
        F[:, b] = (F[:, b] - F[:, a]) % 3
        F[:, W + a] = (F[:, W + a] + F[:, W + b]) % 3
    elif name == "CZ":
        F[:, W + b] = (F[:, W + b] + F[:, a]) % 3
        F[:, W + a] = (F[:, W + a] + F[:, b]) % 3
    elif name == "CZ_INV":
        F[:, W + b] = (F[:, W + b] - F[:, a]) % 3
        F[:, W + a] = (F[:, W + a] - F[:, b]) % 3
    elif name == "SWAP":
        F[:, [a, b]] = F[:, [b, a]]
        F[:, [W + a, W + b]] = F[:, [W + b, W + a]]
    else:
        raise ValueError(name)


PAULI8 = [(x, z) for x in range(3) for z in range(3) if (x, z) != (0, 0)]
PAULI80 = [v for v in itertools.product(range(3), repeat=4) if any(v)]


def single_fault_effects(code, ops_rounds):
    """Propagate every single fault of the circuit (list of per-round op lists).
    Returns sig (F, (T+1) r) int8, cls (F,), mass (F,) in units of p, loc (F,)."""
    n, r = code.n, code.r
    W = n + r
    T = len(ops_rounds)
    # enumerate faults
    fl_loc, fl_vec, fl_mass, fl_where = [], [], [], []
    li = 0
    for t, ops in enumerate(ops_rounds):
        for oi, op in enumerate(ops):
            if op[0] == "N1":
                _, q, sc, ch = op
                vecs = PAULI8 if ch == "d" else [(1, 0), (2, 0)]
                for (x, z) in vecs:
                    v = np.zeros(2 * W, dtype=np.int8)
                    v[q], v[W + q] = x, z
                    fl_loc.append(li)
                    fl_vec.append(v)
                    fl_mass.append(sc / len(vecs))
                    fl_where.append((t, oi))
                li += 1
            elif op[0] == "N2":
                _, a, b, sc = op
                for (xa, za, xb, zb) in PAULI80:
                    v = np.zeros(2 * W, dtype=np.int8)
                    v[a], v[W + a], v[b], v[W + b] = xa, za, xb, zb
                    fl_loc.append(li)
                    fl_vec.append(v)
                    fl_mass.append(sc / 80.0)
                    fl_where.append((t, oi))
                li += 1
    Fn = len(fl_loc)
    vecs = np.array(fl_vec, dtype=np.int8)
    where = {}
    for idx, w in enumerate(fl_where):
        where.setdefault(w, []).append(idx)
    Fr = np.zeros((Fn, 2 * W), dtype=np.int8)
    outcomes = np.zeros((Fn, (T + 1) * r), dtype=np.int8)
    for t, ops in enumerate(ops_rounds):
        for oi, op in enumerate(ops):
            kind = op[0]
            if kind in ("N1", "N2"):
                rows = where[(t, oi)]
                Fr[rows] = (Fr[rows] + vecs[rows]) % 3
            elif kind == "G1":
                apply_sym(Fr, W, op[1], op[2])
            elif kind == "G2":
                apply_sym(Fr, W, op[1], op[2], op[3])
            elif kind == "M":
                a = op[1]
                outcomes[:, t * r + (a - n)] = Fr[:, a]
            elif kind == "R":
                a = op[1]
                Fr[:, a] = 0
                Fr[:, W + a] = 0
    data = np.hstack([Fr[:, :n], Fr[:, W:W + n]]).astype(np.int64)
    SL = code.syn_cls(data)
    outcomes[:, T * r:] = SL[:, :r]
    cls = SL[:, r] * 3 + SL[:, r + 1]
    return outcomes, cls, np.array(fl_mass), np.array(fl_loc)


def leading_order_mld(sig, cls, mass, loc, want_second=True):
    """First- and second-order failure mass (units of p and p^2) of the
    asymptotically optimal decoder (MLD on the full signature)."""
    L = sig.shape[1]
    # distinct effects (signature, class)
    eff_rows = np.hstack([sig.astype(np.int16), cls[:, None].astype(np.int16)])
    ue, einv = np.unique(eff_rows, axis=0, return_inverse=True)
    einv = einv.reshape(-1)
    E = len(ue)
    esig = ue[:, :L].astype(np.int8)
    ecls = ue[:, L].astype(np.int64)
    Me = np.bincount(einv, weights=mass, minlength=E)
    # per (location, effect) masses
    le = loc.astype(np.int64) * E + einv
    ule, leinv = np.unique(le, return_inverse=True)
    m_le = np.bincount(leinv, weights=mass)
    l_of = ule // E
    e_of = ule % E
    # signature keys (exact): chunks of 20 trits -> int64
    nch = (L + 19) // 20
    PW = np.zeros((L, nch))
    for j in range(L):
        PW[j, j // 20] = 3.0 ** (j % 20)

    def keyrows(S):
        return (S.astype(np.float64) @ PW).astype(np.int64)

    ekey = keyrows(esig)
    zero_key = np.zeros(nch, dtype=np.int64)
    # singles grouped by signature
    keys_all = [ekey]
    cls_all = [ecls]
    m1_all = [Me]
    m2_all = [np.zeros(E)]
    if want_second:
        # all unordered effect pairs (including e with itself), chunked
        blocks = []
        for e1 in range(E):
            e2 = np.arange(e1, E)
            pm = Me[e1] * Me[e2]
            pm[0] = (Me[e1] ** 2) / 2.0           # same effect from two different faults
            S2 = (esig[e1][None, :] + esig[e2]) % 3
            l1a, l2a = divmod(int(ecls[e1]), 3)
            l1b, l2b = np.divmod(ecls[e2], 3)
            c2 = ((l1a + l1b) % 3) * 3 + (l2a + l2b) % 3
            blocks.append((keyrows(S2), c2, pm))
        # same-location exclusion
        order = np.argsort(l_of, kind="stable")
        lo_sorted = l_of[order]
        bounds = np.flatnonzero(np.diff(lo_sorted)) + 1
        groups = np.split(order, bounds)
        corr_keys, corr_cls, corr_m = [], [], []
        for gidx in groups:
            ee = e_of[gidx]
            mm = m_le[gidx]
            k = len(ee)
            if k == 0:
                continue
            ii, jj = np.triu_indices(k, k=0)
            pm = mm[ii] * mm[jj]
            pm[ii == jj] = pm[ii == jj] / 2.0
            S2 = (esig[ee[ii]] + esig[ee[jj]]) % 3
            l1a, l2a = np.divmod(ecls[ee[ii]], 3)
            l1b, l2b = np.divmod(ecls[ee[jj]], 3)
            corr_keys.append(keyrows(S2))
            corr_cls.append(((l1a + l1b) % 3) * 3 + (l2a + l2b) % 3)
            corr_m.append(-pm)
        for kk, cc, mm in blocks:
            keys_all.append(kk)
            cls_all.append(cc)
            m1_all.append(np.zeros(len(mm)))
            m2_all.append(mm)
        keys_all.append(np.vstack(corr_keys))
        cls_all.append(np.concatenate(corr_cls))
        m1_all.append(np.zeros(sum(len(x) for x in corr_m)))
        m2_all.append(np.concatenate(corr_m))
    K = np.vstack(keys_all)
    C = np.concatenate(cls_all)
    M1v = np.concatenate(m1_all)
    M2v = np.concatenate(m2_all)
    uk, kinv = np.unique(K, axis=0, return_inverse=True)
    kinv = kinv.reshape(-1)
    Sg = len(uk)
    M1 = np.zeros((Sg, 9))
    M2 = np.zeros((Sg, 9))
    np.add.at(M1, (kinv, C), M1v)
    np.add.at(M2, (kinv, C), M2v)
    M2[np.abs(M2) < 1e-12] = 0.0
    assert M2.min() > -1e-9, M2.min()
    zrow = np.nonzero(np.all(uk == zero_key, axis=1))[0]
    has1 = M1.sum(axis=1) > 0
    key1 = np.where(has1[:, None], M1 * 1e6 + M2, M2)
    cstar = np.argmax(key1, axis=1)
    if len(zrow):
        cstar[zrow[0]] = 0
    rows = np.arange(Sg)
    F1 = float((M1.sum(1) - M1[rows, cstar]).sum())
    F2 = float((M2.sum(1) - M2[rows, cstar]).sum())
    det_rej1 = float(M1.sum() - (M1[zrow[0]].sum() if len(zrow) else 0.0))
    undet1 = float(M1[zrow[0], 1:].sum()) if len(zrow) else 0.0
    # strict hybrid: reject a signature whose minimum-weight explanations span >1 class
    n1 = (M1 > 1e-12).sum(axis=1)
    n2 = (M2 > 1e-12).sum(axis=1)
    rej = np.where(has1, n1 > 1, n2 > 1)
    if len(zrow):
        rej[zrow[0]] = False
    ch = np.where(has1, np.argmax(M1, axis=1), np.argmax(M2, axis=1))
    if len(zrow):
        ch[zrow[0]] = 0
    acc = ~rej
    hyb_rej1 = float(M1[rej].sum())
    hyb_rej2 = float(M2[rej & ~has1].sum())
    hyb_fail2 = float((M2[acc].sum(1) - M2[acc][np.arange(acc.sum()), ch[acc]]).sum())
    return {"F1": F1, "F2": F2, "detect_rej1": det_rej1, "undetected1": undet1,
            "hybrid_rej1": hyb_rej1, "hybrid_rej2": hyb_rej2, "hybrid_fail2": hyb_fail2,
            "effects": int(E), "signatures": int(Sg), "faults": int(len(mass))}


def parse_chp(path):
    """Minimal reader of the repository's .chp text: returns (ops, detectors, observable)
    with ops in the engine format plus ('M', q) for every measurement in order."""
    import re
    ops, dets, obs = [], [], []
    nq = 0
    for line in open(path):
        tok = line.split()
        if not tok or tok[0].startswith("#") or tok[0] in ("Randomly-generated", "d"):
            continue
        g = tok[0]
        if g in ("H", "H_INV", "P", "P_INV", "X", "Z", "X_INV", "Z_INV", "I"):
            for q in tok[1:]:
                ops.append(("G1", "X" if g.startswith("X") or g.startswith("Z") or g == "I" else g, int(q)))
                nq = max(nq, int(q) + 1)
        elif g in ("CNOT", "CNOT_INV", "CZ", "CZ_INV", "SWAP"):
            ops.append(("G2", g, int(tok[1]), int(tok[2])))
            nq = max(nq, int(tok[1]) + 1, int(tok[2]) + 1)
        elif g == "N1":
            ch = re.search(r'noise_channel="(\w)"', line).group(1)
            sc = float(re.search(r'prob_scale="([0-9.eE+-]+)"', line).group(1))
            ops.append(("N1", int(tok[1]), sc, ch))
        elif g == "N2":
            sc = float(re.search(r'prob_scale="([0-9.eE+-]+)"', line).group(1))
            ops.append(("N2", int(tok[1]), int(tok[2]), sc))
        elif g == "M":
            for q in tok[1:]:
                ops.append(("M", int(q)))
        elif g == "RESET":
            ops.append(("R", int(tok[1])))
        elif g == "DETECTOR":
            dets.append([int(x) for x in re.findall(r"rec\[(\d+)\]", line)])
        elif g == "LOGICAL_OBSERVABLE":
            obs = [int(x) for x in re.findall(r"rec\[(\d+)\]", line)]
        else:
            raise ValueError(line)
    return ops, dets, obs, nq


def generic_fault_records(ops, W):
    """Propagate every single fault through a flat op list on W wires; return the
    shift of every measurement record (F, n_meas), masses and location ids."""
    fl_vec, fl_mass, fl_at, fl_loc = [], [], [], []
    li = 0
    for oi, op in enumerate(ops):
        if op[0] == "N1":
            _, q, sc, ch = op
            vecs = PAULI8 if ch == "d" else ([(1, 0), (2, 0)] if ch == "f" else [(0, 1), (0, 2)])
            for (x, z) in vecs:
                v = np.zeros(2 * W, dtype=np.int8)
                v[q], v[W + q] = x, z
                fl_vec.append(v); fl_mass.append(sc / len(vecs)); fl_at.append(oi); fl_loc.append(li)
            li += 1
        elif op[0] == "N2":
            _, a, b, sc = op
            for (xa, za, xb, zb) in PAULI80:
                v = np.zeros(2 * W, dtype=np.int8)
                v[a], v[W + a], v[b], v[W + b] = xa, za, xb, zb
                fl_vec.append(v); fl_mass.append(sc / 80.0); fl_at.append(oi); fl_loc.append(li)
            li += 1
    vecs = np.array(fl_vec, dtype=np.int8)
    at = {}
    for i, oi in enumerate(fl_at):
        at.setdefault(oi, []).append(i)
    Fr = np.zeros((len(vecs), 2 * W), dtype=np.int8)
    recs = []
    for oi, op in enumerate(ops):
        if op[0] in ("N1", "N2"):
            rows = at[oi]
            Fr[rows] = (Fr[rows] + vecs[rows]) % 3
        elif op[0] == "G1":
            apply_sym(Fr, W, op[1], op[2])
        elif op[0] == "G2":
            apply_sym(Fr, W, op[1], op[2], op[3])
        elif op[0] == "M":
            recs.append(Fr[:, op[1]].copy())
        elif op[0] == "R":
            Fr[:, op[1]] = 0
            Fr[:, W + op[1]] = 0
    return np.stack(recs, axis=1), np.array(fl_mass), np.array(fl_loc)


def validate_engine(args):
    """Engine vs repository DEM on stored baseline circuits (first order)."""
    rows = []
    for c, di in ((0, 2), (1, 3)):
        path = os.path.join(RUN, "experiments", "LRB", str(c), str(P_IDX), f"{di}.chp")
        ops, dets, obs, W = parse_chp(path)
        recs, mass, loc = generic_fault_records(ops, W)
        D = np.stack([recs[:, d].sum(axis=1) % 3 for d in dets], axis=1).astype(np.int8)
        ob = recs[:, obs].sum(axis=1) % 3
        uq, inv = np.unique(D, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        Mm = np.zeros((len(uq), 3))
        np.add.at(Mm, (inv, ob), mass)
        z = np.nonzero(~uq.any(axis=1))[0]
        kstar = Mm.argmax(axis=1)
        if len(z):
            kstar[z[0]] = 0
        mine = {"detect_rej": float(mass[inv != (z[0] if len(z) else -1)].sum()),
                "mld_fail": float(mass[ob != kstar[inv]].sum())}
        ref = RES.get("A", {}).get("per_circuit", {}).get(f"{c}_{DEPTHS[di]}")
        rows.append({"circuit": f"cliff {c} depth {DEPTHS[di]}", "engine": mine,
                     "dem": {k: ref[k] for k in mine} if ref else None})
        log(f"   engine vs repository DEM, cliff {c} depth {DEPTHS[di]}: detection rejection "
            f"{mine['detect_rej']:.4f} p vs {ref['detect_rej'] if ref else float('nan'):.4f} p; correction failure "
            f"{mine['mld_fail']:.4f} p vs {ref['mld_fail'] if ref else float('nan'):.4f} p")
    RES["C_validation"] = rows


def part_C(args):
    log("\n=== Part C: SI1000 circuit-level rounds, exact leading order of space-time MLD ===")
    validate_engine(args)
    M5, codes = load_codes()
    # hook-safe orders for the rotated code (first individually safe order of each check)
    c9 = codes["913"]
    singles9 = np.vstack([np.zeros((1, 18), dtype=np.int64), M5.weight_w_errors(9, 1)])
    orders913 = {}
    for k, sp_ in enumerate(c9.supports()):
        for o in itertools.permutations(sp_):
            if M5.collision_free(c9, np.vstack([singles9, M5.hook_set(c9, k, o)])):
                orders913[k] = o
                break
    cases = [
        ("1313", "phen", (), None),
        ("913", "phen", (), orders913),
        ("512", "seq", (), None),
        ("512", "seq", ("H", "P", "H", "X", "Z"), None),
        ("1313", "seq", (), None),
        ("1313", "par", (), None),
        ("1313", "par", ("H", "P", "H", "X", "Z"), None),
        ("1313", "seq", ("H", "P", "H", "X", "Z"), None),
        ("913", "seq", (), orders913),
        ("913", "par", (), orders913),
        ("513", "par", (), None),
        ("713", "par", (), None),
    ]
    if args.quick:
        cases = [("512", "seq", (), None), ("1313", "par", (), None)]
    out = {}
    for key, ro, toks, orders in cases:
        code = codes[key]
        t0 = time.time()
        res = {}
        for T in (2, 3):
            ops = [round_ops(M5, code, ro, toks, orders) for _ in range(T)]
            sig, cls, mass, loc = single_fault_effects(code, ops)
            res[T] = leading_order_mld(sig, cls, mass, loc, want_second=(code.n > 5 or True))
        per = {k: res[3][k] - res[2][k] for k in ("F1", "F2", "detect_rej1", "undetected1")}
        name = f"{code.name} readout={ro} gates/round={''.join(toks) or 'none'}"
        out[name] = {"per_round": per, "T2": res[2], "T3": res[3]}
        for k in ("hybrid_rej1", "hybrid_rej2", "hybrid_fail2"):
            per[k] = res[3][k] - res[2][k]
        unit = "q" if ro == "phen" else "p"
        if per["F1"] > 1e-9:
            msg = (f"first-order failure {per['F1']:.3f} {unit} per round (total logical error at first order)")
        else:
            msg = (f"eps = {per['F2']:.2f} {unit}^2 per round (no first-order failure); strict hybrid: accepted "
                   f"error {per['hybrid_fail2']:.2f} {unit}^2, rejection {per['hybrid_rej2']:.1f} {unit}^2 per round")
        log(f"{name}: {msg}; detection rejection {per['detect_rej1']:.1f} {unit} per round; "
            f"{res[3]['faults']} single faults, {res[3]['effects']} distinct effects, "
            f"{res[3]['signatures']} signatures (T=3) [{time.time() - t0:.1f}s]")
    RES["C"] = out
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cliffs", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--parts", default="A,B,C")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    parts = set(args.parts.split(","))
    if "A" in parts:
        part_A(args)
    if "B" in parts:
        part_B()
    if "C" in parts and "part_C" in globals():
        part_C(args)  # noqa: F821
    log(f"\ntotal runtime {time.time() - t0:.1f}s")
    suffix = "_quick" if args.quick else ""

    def js(x):
        if isinstance(x, dict):
            return {str(k): js(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [js(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        return x
    with open(os.path.join(OUT, f"A5_circuit_level{suffix}.json"), "w") as fh:
        json.dump(js(RES), fh, indent=1)
    with open(os.path.join(OUT, f"A5_circuit_level_log{suffix}.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
