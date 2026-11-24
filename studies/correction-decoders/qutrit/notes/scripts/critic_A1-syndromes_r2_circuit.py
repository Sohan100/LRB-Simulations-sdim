#!/usr/bin/env python3
"""Critic check (round 2) of A1-syndromes circuit-level claims C13-C20.

Independent of A1_syndrome_classification.py:
  * circuits are read with sdim.circuit_io.read_circuit (baseline files) or
    built with the repository's own LRBCircuitGenerator (memory windows and
    one check round), never with A1's generator;
  * the as-run view (V1) is validated against the repository's
    CompactDetectorErrorModel (src/lrb/dem_simulation.py);
  * my own Pauli-frame propagator (sdim/program.py rules) supplies the
    perfect-final-syndrome / full-class view (V2) and the one-round table.
Leading-order coefficients are coefficients of p (prob_scale weights).

Run: module load python/3.11; OMP_NUM_THREADS=2 python critic_A1-syndromes_r2_circuit.py
Runtime: a few minutes on one or two cores (login node).  Read-only on the
baseline run folder.
"""
import dataclasses
import itertools
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

REPO = ("/pscratch/sd/s/sohan100/worktrees/LRB-Simulations-sdim-2bc490701489f8e7/"
        "correction-decoders-qutrit-2c455b")
RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
       "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
sys.path.insert(0, REPO + "/src")
from sdim.circuit_io import read_circuit                        # noqa: E402
from lrb.dem_simulation import build_compact_detector_error_model  # noqa: E402
from lrb.code_simulation_profiles import CodeSimulationProfileRegistry  # noqa: E402
from lrb.circuit_generator import LRBCircuitGenerator, SI1000_NOISE_MODEL  # noqa: E402

T0 = time.time()
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------ code algebra
def symp(a, b):
    return int((np.dot(a[5:], b[:5]) - np.dot(a[:5], b[5:])) % 3)


E10 = np.eye(10, dtype=int)
S_ = [np.array(v) for v in ([0, 0, 0, 0, 0, 1, -1, 0, -1, 0], [0, 0, 0, 0, 0, 0, 1, 1, 0, -1],
                            [1, 1, -1, 0, 0, 0, 0, 0, 0, 0], [0, -1, 0, 1, -1, 0, 0, 0, 0, 0])]
XL = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
ZL = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0])
FUNC = np.array([[symp(s, E10[i]) for i in range(10)] for s in S_]
                + [[symp(ZL, E10[i]) for i in range(10)]]
                + [[(-symp(XL, E10[i])) % 3 for i in range(10)]])   # rows: s1..s4, a, b
KS = np.array(list(itertools.product(range(9), repeat=5)))
VV = np.concatenate([KS % 3, KS // 3], axis=1)
SY = (VV @ FUNC[:4].T) % 3
CL = (VV @ FUNC[4:].T) % 3
SIDX = SY @ np.array([1, 3, 9, 27])
CIDX = CL[:, 0] + 3 * CL[:, 1]
WT = np.count_nonzero(KS, axis=1)
MINW = np.full((81, 9), 99)
np.minimum.at(MINW, (SIDX, CIDX), WT)
SCLS = {}
for s in range(81):
    wmin = min(MINW[s])
    cos = [c for c in range(9) if MINW[s, c] == wmin]
    SCLS[s] = "T" if wmin == 0 else ("U1" if (wmin == 1 and len(cos) == 1) else
                                     ("A1" if wmin == 1 else "W2"))
W1IDX = {(int(SIDX[i]), int(CIDX[i])): i for i in np.nonzero(WT == 1)[0]}


def plab(k):
    s = []
    for q in range(5):
        x, z = k[q] % 3, k[q] // 3
        if x or z:
            s.append((f"X{q}" + ("^-1" if x == 2 else "") if x else "") +
                     (f"Z{q}" + ("^-1" if z == 2 else "") if z else ""))
    return " ".join(s) or "I"


def residual_class(v):
    """v: 10-vector data Pauli. Returns (kind, detail)."""
    s = int(((FUNC[:4] @ v) % 3) @ np.array([1, 3, 9, 27]))
    a, b = (FUNC[4:] @ v) % 3
    c = int(a + 3 * b)
    if s == 0 and c == 0:
        return "stab", None
    w = MINW[s, c]
    if s == 0:
        return f"logical_w{w}", None
    if w == 1:
        i = W1IDX[(s, c)]
        q = int(np.nonzero(KS[i])[0][0])
        return f"w1_{SCLS[s]}", (q, plab(KS[i]))
    return f"w{w}_{SCLS[s]}", None


# ------------------------------------------------------------ propagator
def parse_terms(expr):
    out = []
    for t in expr.replace(" ", "").split("+"):
        if not t:
            continue
        sign = -1 if t.startswith("-") else 1
        m = re.search(r"rec\[(-?\d+)\]", t)
        coef = re.match(r"-?(\d+)\*", t)
        c = sign * (int(coef.group(1)) if coef else 1)
        out.append((int(m.group(1)), c))
    return out


def propagate(circ, snapshot_before_terminal=True):
    ops = circ.operations
    nq = circ.num_qudits
    locs = []
    c = 0
    for i, o in enumerate(ops):
        if o.gate_name in ("N1", "N2"):
            sc = float((o.params or {}).get("prob_scale", 0.0))
            if sc <= 0:
                continue
            nc = 2 if o.gate_name == "N1" else 4
            ch = (o.params or {}).get("noise_channel", "d") if o.gate_name == "N1" else "d"
            locs.append(dict(i=i, g=o.gate_name, q=o.qudit_index, t=o.target_index,
                             ch=ch, sc=sc, c0=c, nc=nc))
            c += nc
    loc_at = {l["i"]: l for l in locs}
    F = np.zeros((2 * nq, c), dtype=np.int64)
    meas, meas_q = [], []
    dets, obs = [], []
    # terminal start: right after the last ancilla measurement
    last_anc = max((i for i, o in enumerate(ops) if o.gate_name == "M" and o.qudit_index >= 5),
                   default=None)
    snap = None
    for i, o in enumerate(ops):
        g, a, b = o.gate_name, o.qudit_index, o.target_index
        if snapshot_before_terminal and last_anc is not None and i == last_anc + 1:
            snap = F % 3
        if g == "N1":
            if i in loc_at:
                l = loc_at[i]
                F[a, l["c0"]] += 1
                F[nq + a, l["c0"] + 1] += 1
        elif g == "N2":
            if i in loc_at:
                l = loc_at[i]
                F[a, l["c0"]] += 1
                F[nq + a, l["c0"] + 1] += 1
                F[b, l["c0"] + 2] += 1
                F[nq + b, l["c0"] + 3] += 1
        elif g == "H":
            F[a], F[nq + a] = -F[nq + a], F[a].copy()
        elif g == "H_INV":
            F[a], F[nq + a] = F[nq + a].copy(), -F[a]
        elif g == "P":
            F[nq + a] += F[a]
        elif g == "P_INV":
            F[nq + a] -= F[a]
        elif g == "CNOT":
            F[b] += F[a]
            F[nq + a] -= F[nq + b]
        elif g == "CNOT_INV":
            F[b] -= F[a]
            F[nq + a] += F[nq + b]
        elif g == "CZ":
            F[nq + b] += F[a]
            F[nq + a] += F[b]
        elif g == "CZ_INV":
            F[nq + b] -= F[a]
            F[nq + a] -= F[b]
        elif g == "SWAP":
            F[[a, b]] = F[[b, a]]
            F[[nq + a, nq + b]] = F[[nq + b, nq + a]]
        elif g == "M":
            meas.append(F[a] % 3)
            meas_q.append(a)
        elif g == "RESET":
            F[a] = 0
            F[nq + a] = 0
        elif g in ("DETECTOR", "LOGICAL_OBSERVABLE"):
            row = np.zeros(c, dtype=np.int64)
            for idx, cf in parse_terms(o.params["expr"]):
                row += cf * meas[idx]
            (dets if g == "DETECTOR" else obs).append(row % 3)
        elif g in ("I", "X", "X_INV", "Z", "Z_INV"):
            pass
        else:
            raise ValueError(g)
        if i % 12 == 0:
            F %= 3
    if snap is None:
        snap = F % 3
    data = np.vstack([snap[0:5], snap[nq:nq + 5]])
    return dict(locs=locs, ncols=c, det=np.array(dets).reshape(len(dets), c),
                obs=np.array(obs).reshape(len(obs), c), data_final=data,
                meas=np.array(meas).reshape(len(meas), c), meas_q=meas_q, F_end=F % 3, nq=nq)


VEC = {"d": np.array([(x, z) for x in range(3) for z in range(3) if (x, z) != (0, 0)]),
       "f": np.array([(1, 0), (2, 0)]), "p": np.array([(0, 1), (0, 2)]),
       "N2": np.array([v for v in itertools.product(range(3), repeat=4) if any(v)])}


def expand(prop, rows):
    """All single faults: response matrix over `rows` (list of (n, ncols)
    arrays), weights (coeff of p), location id."""
    U = np.vstack(rows)
    R, Wt, L = [], [], []
    for li, l in enumerate(prop["locs"]):
        vs = VEC["N2"] if l["g"] == "N2" else VEC[l["ch"]]
        R.append((vs @ U[:, l["c0"]:l["c0"] + l["nc"]].T) % 3)
        Wt.append(np.full(len(vs), l["sc"] / len(vs)))
        L.append(np.full(len(vs), li))
    return np.vstack(R), np.concatenate(Wt), np.concatenate(L)


def leading(resp, w, nsig, cls):
    """cls: integer class per fault.  Returns det_rej, undet, mld, hyb_rej,
    group index, ambiguous-group flag."""
    sig = np.ascontiguousarray(resp[:, :nsig].astype(np.int8))
    keyv = sig.view(np.dtype((np.void, nsig))).ravel()
    _, first, g = np.unique(keyv, return_index=True, return_inverse=True)
    g = g.reshape(-1)
    zero = ~np.any(sig[first] != 0, axis=1)
    ncl = int(cls.max()) + 1 if len(cls) else 1
    Wm = np.zeros((len(first), max(ncl, 1)))
    np.add.at(Wm, (g, cls), w)
    tot = Wm.sum(1)
    amb = ((Wm > 0).sum(1) > 1) & ~zero
    undet = Wm[zero][:, 1:].sum()
    out = dict(det_rej=tot[~zero].sum(), undet=undet,
               mld=(tot - Wm.max(1))[~zero].sum() + undet, hyb_rej=tot[amb].sum())
    return out, g, amb, zero


def roles(circ, prop):
    ops = circ.operations
    out = []
    for l in prop["locs"]:
        i = l["i"]
        prev = ops[i - 1]
        if l["g"] == "N2":
            r = "check_CNOT" if max(l["q"], l["t"]) >= 5 else "gate_2q"
        elif l["ch"] == "f":
            r = "reset_flip" if prev.gate_name == "RESET" else "meas_flip"
        elif l["sc"] == 0.1:
            r = "gate_1q_data" if l["q"] < 5 else "anc_H"
        elif l["sc"] == 1.0:
            r = "meas_depol"
        elif l["sc"] == 2.0:
            j = i
            while ops[j].gate_name == "N1":
                j += 1
            r = ("idle_reset_" if ops[j].gate_name == "RESET" else "idle_meas_") + \
                ("data" if l["q"] < 5 else "anc")
        else:
            r = "other"
        out.append(r)
    return out


# ------------------------------------------------------------ DEM validation
def dem_validate(circ):
    model = build_compact_detector_error_model(circ)
    prop = propagate(circ)
    by_i = {l["i"]: l for l in prop["locs"]}
    nbad = ncmp = 0
    for loc in model.locations:
        l = by_i[loc.operation_index]
        for ci, terms in enumerate(loc.component_terms):
            d = np.zeros(prop["det"].shape[0], int)
            for k, v in terms.detector:
                d[k] = v
            o = np.zeros(prop["obs"].shape[0], int)
            for k, v in terms.logical:
                o[k] = v
            ncmp += 1
            col = l["c0"] + ci
            if not (np.array_equal(d, prop["det"][:, col]) and np.array_equal(o, prop["obs"][:, col])):
                nbad += 1
    return ncmp, nbad, len(model.locations), len(prop["locs"])


def v1_v2(circ, want_roles=False):
    prop = propagate(circ)
    nd = prop["det"].shape[0]
    extra = (FUNC @ prop["data_final"]) % 3                 # perfect s1..s4, a, b
    resp, w, L = expand(prop, [prop["det"], prop["obs"], extra])
    assert np.array_equal(resp[:, nd], resp[:, nd + 6]), "observable != b(final frame)"
    v1, g1, amb1, z1 = leading(resp, w, nd, resp[:, nd].astype(int))
    sig2 = np.hstack([resp[:, :nd], resp[:, nd + 1:nd + 5]])
    cls2 = (resp[:, nd + 5] + 3 * resp[:, nd + 6]).astype(int)
    v2, g2, amb2, z2 = leading(np.hstack([sig2, cls2[:, None]]), w, nd + 4, cls2)
    out = dict(V1=v1, V2=v2, total=w.sum())
    if want_roles:
        rl = np.array(roles(circ, prop))[L]
        bd = defaultdict(float)
        for r_, ww in zip(rl[amb2[g2]], w[amb2[g2]]):
            bd[r_] += ww
        out["V2_amb_by_role"] = dict(bd)
        bd1 = defaultdict(float)
        m = z1[g1] & (resp[:, nd] != 0)
        for r_, ww in zip(rl[m], w[m]):
            bd1[r_] += ww
        out["V1_undet_by_role"] = dict(bd1)
    return out, prop, resp, w, L


# ============================================================ 1. DEM validation
log("=" * 70)
log("1. Own propagator vs repository CompactDetectorErrorModel")
val = []
for sub, cl, di in (("LRB", 0, 0), ("LRB", 0, 1), ("LRB", 7, 2), ("LRB", 23, 4),
                    ("LRB_const0", 0, 1), ("LRB_const0", 12, 3)):
    c = read_circuit(f"{RUN}/experiments/{sub}/{cl}/4/{di}.chp")
    r = dem_validate(c)
    val.append(r)
    log(f"   {sub}/{cl}/4/{di}.chp: components {r[0]}, mismatches {r[1]}, DEM locs {r[2]}, own locs {r[3]}")

# ============================================================ 2. memory windows
log("=" * 70)
log("2. Memory windows built with the repository generator (no logical gates)")
cd = CodeSimulationProfileRegistry._build_code_definition("folded_qutrit")
cd_mem = dataclasses.replace(cd, clifford_to_gate_sequence=lambda s: [],
                             clifford_inverse_map={"E": []})
gen = LRBCircuitGenerator(code_definition=cd_mem, noise_model=SI1000_NOISE_MODEL)
mem = {}
for R in range(0, 5):
    circ = gen.generate_lrb_clifford_sequence(depths=[R], clifford_strings=["E"] * R)[0]
    gen.update_noise_param(circ, 1e-3)
    if R in (0, 2):
        r = dem_validate(circ)
        log(f"   R={R}: DEM validation components {r[0]}, mismatches {r[1]}")
    o, prop, resp, w, L = v1_v2(circ, want_roles=True)
    mem[R] = o
    log(f"   R={R}: total {o['total']:.3f} | V1 det {o['V1']['det_rej']:.3f} undet {o['V1']['undet']:.3f}"
        f" MLD {o['V1']['mld']:.3f} hyb {o['V1']['hyb_rej']:.3f} | V2 det {o['V2']['det_rej']:.3f}"
        f" MLD {o['V2']['mld']:.3f} hyb {o['V2']['hyb_rej']:.3f}")
for key in ("det_rej", "mld", "hyb_rej"):
    for V in ("V1", "V2"):
        inc = [round(mem[R + 1][V][key] - mem[R][V][key], 4) for R in range(4)]
        log(f"   increment {V} {key}: {inc}")
inc_roles = defaultdict(float)
for k in set(mem[3]["V2_amb_by_role"]) | set(mem[2]["V2_amb_by_role"]):
    inc_roles[k] = mem[3]["V2_amb_by_role"].get(k, 0) - mem[2]["V2_amb_by_role"].get(k, 0)
log("   V2 ambiguous weight by role, increment R=2->3:", {k: round(v, 4) for k, v in inc_roles.items()})
log("   V2 ambiguous weight by role, R=0:", {k: round(v, 4) for k, v in mem[0]["V2_amb_by_role"].items()})
log("   V1 undetected by role, R=0:", {k: round(v, 4) for k, v in mem[0]["V1_undet_by_role"].items()})

# ============================================================ 3. one round table
log("=" * 70)
log("3. One check round + reset, built from the repository blocks with SI1000 noise")
from sdim.circuit import Circuit  # noqa: E402
one = Circuit(dimension=3, num_qudits=9)
for fac, _ in cd.stabilizer_check_blocks:
    one = one + gen.apply_si1000_noise_to_circuit(fac(), noisy_measurements=True)
one = one + gen.apply_si1000_noise_to_circuit(cd.reset_measurement_wires(), noisy_measurements=False)
gen.update_noise_param(one, 1e-3)
prop = propagate(one, snapshot_before_terminal=False)
F = prop["F_end"]
data = np.vstack([F[0:5], F[9:14]])
nxt = F[5:9]                                     # ancilla x after reset -> next-round flips
resp, w, L = expand(prop, [prop["meas"], data, nxt])
flips, resid, nx = resp[:, :4], resp[:, 4:14], resp[:, 14:18]
cat = defaultdict(float)
md = np.zeros(4)
rf = np.zeros(4)
w1 = defaultdict(float)
w1pauli = defaultdict(float)
for k in range(len(w)):
    kind, det = residual_class(resid[k])
    fnow, fnext = flips[k].any(), nx[k].any()
    if kind == "stab":
        cat["meas_only" if (fnow or fnext) else "no_effect"] += w[k]
        md += (flips[k] != 0) * w[k]
    else:
        cat["data+flip" if fnow else "data_only"] += w[k]
        if kind.startswith("w1"):
            w1[(det[0], kind)] += w[k]
            w1pauli[det[1]] += w[k]
        else:
            w1[kind] += w[k]
    rf += (nx[k] != 0) * w[k]
log("   total fault weight", round(w.sum(), 4), " categories", {k: round(v, 4) for k, v in cat.items()})
log("   measurement-only flips per check (S_Z1,S_Z2,S_X1,S_X2):", np.round(md, 4).tolist(),
    " next-round flips:", np.round(rf, 4).tolist())
log("   data residual classes:", {str(k): round(v, 4) for k, v in sorted(w1.items(), key=str)})
pairs = [("X0", "X3^-1"), ("X0^-1", "X3"), ("X2", "X4^-1"), ("X2^-1", "X4"),
         ("Z0", "Z2^-1"), ("Z0^-1", "Z2"), ("Z3", "Z4^-1"), ("Z3^-1", "Z4")]
log("   tied-pair per-Pauli rates:", [(a, round(w1pauli[a], 4), b, round(w1pauli[b], 4)) for a, b in pairs])
log("   sum min over pairs:", round(sum(min(w1pauli[a], w1pauli[b]) for a, b in pairs), 4),
    " sum over pairs:", round(sum(w1pauli[a] + w1pauli[b] for a, b in pairs), 4))

# ============================================================ 4. LRB circuits
log("=" * 70)
log("4. Baseline LRB and const0 circuits, all 30 sequences, own propagation (V1, V2)")
tab = defaultdict(lambda: defaultdict(list))
for cl in range(30):
    for di, m in enumerate(DEPTHS):
        circ = read_circuit(f"{RUN}/experiments/LRB/{cl}/1/{di}.chp")
        o, *_ = v1_v2(circ)
        for V in ("V1", "V2"):
            for k, v in o[V].items():
                tab[(V, k)][m].append(v)
        c0 = read_circuit(f"{RUN}/experiments/LRB_const0/{cl}/1/{di}.chp")
        p0 = propagate(c0, snapshot_before_terminal=False)
        if p0["ncols"] == 0:
            for k in ("det_rej", "undet", "mld", "hyb_rej"):
                tab[("C0", k)][m].append(0.0)
        else:
            r0, w0, _ = expand(p0, [p0["det"], p0["obs"]])
            v0, *_ = leading(r0, w0, p0["det"].shape[0], r0[:, -1].astype(int))
            for k, v in v0.items():
                tab[("C0", k)][m].append(v)
log(f"   ({time.time() - T0:.0f} s so far)")


def csv(pidx, pol, key):
    rows = {}
    for line in open(f"{RUN}/results/LRB/{pidx}/{pol}/{key}.csv"):
        parts = line.strip().split(",")
        rows[parts[0]] = [float(t) for t in parts[1:]]
    return rows


bu = csv(1, "unif_check_data", 1)
b0 = csv(1, "const_check_data", 0)
p = 1e-4
ms = np.array(DEPTHS)
sel = ms >= 2


def fit(y):
    A = np.vstack([np.ones(sel.sum()), ms[sel]]).T
    return np.linalg.lstsq(A, np.asarray(y)[sel], rcond=None)[0]


for lab_, idx in (("seq 0-9", slice(0, 10)), ("seq 0-29", slice(0, 30))):
    det = [np.mean(tab[("V1", "det_rej")][m][idx]) for m in DEPTHS]
    mld = [np.mean(tab[("V1", "mld")][m][idx]) for m in DEPTHS]
    hyb = [np.mean(tab[("V1", "hyb_rej")][m][idx]) for m in DEPTHS]
    und = [np.mean(tab[("V1", "undet")][m][idx]) for m in DEPTHS]
    mld2 = [np.mean(tab[("V2", "mld")][m][idx]) for m in DEPTHS]
    hyb2 = [np.mean(tab[("V2", "hyb_rej")][m][idx]) for m in DEPTHS]
    c0d = [np.mean(tab[("C0", "det_rej")][m][idx]) for m in DEPTHS]
    c0m = [np.mean(tab[("C0", "mld")][m][idx]) for m in DEPTHS]
    c0h = [np.mean(tab[("C0", "hyb_rej")][m][idx]) for m in DEPTHS]
    log(f"   {lab_}: per-depth table (m, det, MLD, hyb, undet | const0 det, MLD, hyb)")
    for i, m in enumerate(DEPTHS):
        log(f"      m={m:2d}: {det[i]:8.2f} {mld[i]:7.2f} {hyb[i]:7.2f} {und[i]:.3f} |"
            f" {c0d[i]:7.2f} {c0m[i]:6.2f} {c0h[i]:6.2f}")
    for nm, y in (("det", det), ("MLD", mld), ("hyb", hyb), ("V2 MLD", mld2), ("V2 hyb", hyb2),
                  ("const0 det", c0d), ("const0 MLD", c0m), ("const0 hyb", c0h)):
        a, b = fit(y)
        log(f"      fit {nm:10s}: c0 = {a:8.3f}  c1 = {b:8.3f}")
    log(f"      ratio V1/V2 MLD slope = {fit(mld)[1] / fit(mld2)[1]:.4f}")
bl = [-np.log(1 - bu["Rejected Runs"][i]) / p for i in range(9)]
bl0 = [-np.log(1 - b0["Rejected Runs"][i]) / p for i in range(9)]
log("   baseline (30 seq, p=1e-4) -ln(acc)/p unif1:", np.round(bl, 2).tolist())
log("   baseline fit unif1:", np.round(fit(bl), 3).tolist())
log("   baseline (30 seq, p=1e-4) -ln(acc)/p const0:", np.round(bl0, 2).tolist())
log("   baseline fit const0:", np.round(fit(bl0), 3).tolist())
log("   baseline (1-F)/(1.5p) unif1:", np.round([(1 - f) / 1.5 / p for f in bu["Fidelity averages"]], 3).tolist())
log("   baseline Fidelity std (over Cliffords) unif1 at p=1e-4:", bu.get("Fidelity std", None))

# ============================================================ 5. second order
log("=" * 70)
log("5. Second-order (two-fault) coefficients, V1, own pair enumeration")


def order2(circ):
    """Two-fault coefficients of P(fail|acc) for detection and the hybrid
    (accept a signature iff all single-fault explanations share one
    observable class; zero signature -> class 0).  Exact base-3 keys."""
    prop = propagate(circ)
    nd = prop["det"].shape[0]
    assert nd <= 39
    resp, w, L = expand(prop, [prop["det"], prop["obs"]])
    sc = np.array([l["sc"] for l in prop["locs"]])
    Ttot = sc.sum()
    sig = resp[:, :nd].astype(np.int64)
    p3 = 3 ** np.arange(nd, dtype=np.int64)
    key = sig @ p3
    ob = resp[:, nd].astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(-1)
    Wg = np.zeros((len(uk), 3))
    np.add.at(Wg, (inv, ob), w)
    ncls = (Wg > 0).sum(1)
    acc = ncls == 1
    best = Wg.argmax(1)
    iz = np.searchsorted(uk, 0)
    if iz < len(uk) and uk[iz] == 0:
        acc[iz] = True
        best[iz] = 0
    undet_mask = (key == 0) & (ob != 0)
    U1 = w[undet_mask].sum()
    D1 = w[key != 0].sum()
    H1 = w[~acc[inv]].sum()
    dep = float(sum(w[f] * (Ttot - sc[L[f]]) for f in np.nonzero(undet_mask)[0]))
    A_det = 0.0
    A_hyb = 0.0
    n = len(w)
    for i in range(n - 1):
        j = np.arange(i + 1, n)
        j = j[L[j] != L[i]]
        k2 = ((sig[i] + sig[j]) % 3) @ p3
        c2 = (ob[i] + ob[j]) % 3
        w2 = w[i] * w[j]
        zr = k2 == 0
        A_det += w2[zr & (c2 != 0)].sum()
        pos = np.minimum(np.searchsorted(uk, k2), len(uk) - 1)
        hit = (uk[pos] == k2) & acc[pos]
        A_hyb += w2[hit & (c2 != best[pos])].sum()     # includes zero signature (best 0)
    det2 = A_det - dep + U1 * D1
    hyb2 = A_hyb - dep + U1 * H1
    return dict(U1=U1, D1=D1, H1=H1, det2=det2, hyb2=hyb2, nfaults=n)


c = read_circuit(f"{RUN}/experiments/LRB/0/4/0.chp")
t1 = time.time()
r0 = order2(c)
log(f"   m=0: undetected {r0['U1']:.4f}, det rej {r0['D1']:.3f}, hyb rej {r0['H1']:.3f},"
    f" det P(f|a) p^2 coeff {r0['det2']:.3f}, hybrid p^2 coeff {r0['hyb2']:.3f}"
    f" ({time.time() - t1:.0f} s)")
# baseline fidelities at p_idx 3,4,5 for m = 0 and m = 2 (unif 1)
c2s = []
for cl in range(4):
    t1 = time.time()
    rr = order2(read_circuit(f"{RUN}/experiments/LRB/{cl}/4/1.chp"))
    c2s.append(rr["det2"])
    log(f"   m=2 seq {cl}: det rej {rr['D1']:.3f}, hyb rej {rr['H1']:.3f}, det p^2 {rr['det2']:.3f},"
        f" hyb p^2 {rr['hyb2']:.3f} ({time.time() - t1:.0f} s)")
probs = [float(t) for t in open(f"{RUN}/probs.txt").read().replace("\n", ",").split(",") if t.strip()]
for pidx in (2, 3, 4, 5, 6):
    pv = probs[pidx]
    b = csv(pidx, "unif_check_data", 1)
    f0 = b["Fidelity averages"][0]
    se0 = b["Fidelity Standard Deviations"][0] / np.sqrt(30)
    pred0 = 1 - 1.5 * (r0["U1"] * pv + r0["det2"] * pv ** 2)
    pred0_lin = 1 - 1.5 * (r0["U1"] * pv)
    f2 = b["Fidelity averages"][1]
    se2 = b["Fidelity Standard Deviations"][1] / np.sqrt(30)
    pred2 = [1 - 1.5 * (4.95 * pv + c2 * pv ** 2) for c2 in (min(c2s), max(c2s))]
    log(f"   p={pv:g}: m=0 baseline F {f0:.6f} (SE {se0:.1e}), predicted {pred0:.6f} (diff {f0 - pred0:+.2e};"
        f" first-order only {pred0_lin:.6f}, diff {f0 - pred0_lin:+.2e}) | m=2 baseline {f2:.6f}"
        f" (SE {se2:.1e}), predicted [{pred2[0]:.6f}, {pred2[1]:.6f}] for c2 in [{min(c2s):.0f},{max(c2s):.0f}]")
log(f"done in {time.time() - T0:.0f} s")
