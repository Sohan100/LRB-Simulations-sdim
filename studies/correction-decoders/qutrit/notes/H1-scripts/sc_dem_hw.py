"""H1 (rewrite) - schedules and duration-aware hardware noise on the repo DEM.

Three check-round schedules, all built with the repo generator
(LRBCircuitGenerator, SI1000 insertion) and compiled with the repo compact
DEM (lrb.dem_simulation):

  seq9    the production schedule (folded_qutrit profile): four ancilla
          readout layers per round, any-to-any couplings assumed;
  par9    same couplings, all four ancillas read in one layer;
  relay11 square-lattice layout of sc_graphs.py (native fold D2-D3, relay
          qutrits R1 = wire 9, R2 = wire 10, six relay hops, 24 two-qutrit
          gates per round); X-type checks first using freshly reset relays,
          then Z-type checks using the X ancillas (still holding their
          X-syndrome in a basis state) as collector relays; all six
          ancilla/relay qutrits read in ONE layer, then reset.

Noise models:
  SI1000(p)      the repo model (no idle during gate layers);
  HW-now         best published qutrit components (fetched sources, see
                 H1-sc-hardware.md): CZ 2.7% Weyl error in 580 ns, single-
                 qutrit H/X 6e-3 in 100 ns, diagonal gates virtual, logical
                 SWAP virtual, three-state readout error 3% in 500 ns, reset
                 0.2% in 300 ns, idle from the Goss2022-Q4 echo Lindblad model
                 for every idle gap of the ASAP schedule (gate layers too);
  HW-now-R       same with Ramsey T2 (pessimistic dephasing);
  HW-target      what-if: CZ 1% in 200 ns, 1q 1e-3 in 50 ns, readout 1% in
                 300 ns, reset 0.1% in 250 ns, coherence 2x Goss-Q4 echo;
  HW-qubit       what-if: HW-target timing with CZ 0.5%, 1q 5e-4, readout
                 0.5% (IQM's median qubit CZ error is 5e-3).
  A CNOT is compiled as CZ plus on average 1.33 single-qutrit basis changes.
  The const-0 circuits (no ancilla checks, ideal terminal X readout of all
  data) are run under the same models.  An empirical lookup decoder
  (train/test split, m = 2) compares raw, corrected, hybrid and LRB-D.

Outputs per (schedule, model, depth): unif-1 (all rounds) and const-1
(terminal round) acceptance and accepted fidelity, raw fidelity (exact
product formula), round duration; first-order fault classes (silent-logical,
detected-unambiguous, detected-ambiguous probability mass); per-sequence
spread of the raw fidelity; noiseless determinism of the relay schedule.

Run: module load python/3.11; OMP_NUM_THREADS=1 python sc_dem_hw.py
Runtime: about 5 minutes on one core.
"""
from __future__ import annotations

import dataclasses
import itertools
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

from sdim.circuit import Circuit  # noqa: E402
from sdim.program import Program  # noqa: E402

from lrb import code_definitions as defns  # noqa: E402
from lrb.circuit_generator import LRBCircuitGenerator  # noqa: E402
from lrb.code_simulation_profiles import CodeSimulationProfileRegistry  # noqa: E402
from lrb.dem_simulation import (build_compact_detector_error_model,  # noqa: E402
                                sample_detector_error_model,
                                two_qudit_depol_prob_dist)
from sc_idle import COH, e_idle  # noqa: E402

OMEGA = np.exp(2j * np.pi / 3)
SEED = 20261028


# ------------------------------------------------------------ schedules
def par9_block() -> Circuit:
    c = Circuit(dimension=3, num_qudits=9)
    for g, a, b in (("CNOT", 0, 5), ("CNOT_INV", 1, 5), ("CNOT_INV", 3, 5),
                    ("CNOT", 1, 6), ("CNOT", 2, 6), ("CNOT_INV", 4, 6)):
        c.add_gate(g, a, b)
    for anc, gs in ((7, (("CNOT", 7, 0), ("CNOT", 7, 1), ("CNOT_INV", 7, 2))),
                    (8, (("CNOT", 8, 3), ("CNOT_INV", 8, 1), ("CNOT_INV", 8, 4)))):
        c.add_gate("H", anc)
        for g, a, b in gs:
            c.add_gate(g, a, b)
        c.add_gate("H_INV", anc)
    for a in (5, 6, 7, 8):
        c.add_gate("M", a)
    return c


RELAY_X = [  # X-type checks, relays A5 (for A7-D0) and A6 (for A8-D1) fresh
    ("H", 7), ("H", 8),
    ("CNOT", 7, 1), ("CNOT_INV", 7, 2),
    ("CNOT", 7, 5), ("CNOT", 5, 0), ("CNOT_INV", 7, 5),          # A7 -> D0
    ("CNOT", 8, 3), ("CNOT_INV", 8, 4),
    ("CNOT", 8, 6), ("CNOT_INV", 6, 1), ("CNOT_INV", 8, 6),      # A8 -> D1
    ("H_INV", 7), ("H_INV", 8),
]
RELAY_Z = [  # Z-type checks, A7/A8 (holding s7/s8) and chain R2,R1 as relays
    ("CNOT", 0, 5),
    ("CNOT_INV", 1, 7), ("CNOT", 7, 5), ("CNOT", 1, 7),          # D1 -> A5 via A7
    ("CNOT_INV", 3, 10), ("CNOT", 10, 9), ("CNOT", 9, 5),        # D3 -> A5 via R2,R1
    ("CNOT_INV", 10, 9), ("CNOT", 3, 10),
    ("CNOT", 1, 6), ("CNOT", 2, 6),
    ("CNOT_INV", 4, 8), ("CNOT", 8, 6), ("CNOT", 4, 8),          # D4 -> A6 via A8
]
LAYOUT_COUPLERS = {frozenset(e) for e in [(5, 0), (6, 1), (6, 2), (7, 1), (7, 2),
                                          (8, 3), (8, 4), (2, 3), (5, 7), (5, 9),
                                          (9, 10), (10, 3), (6, 8)]}


def relay11_block() -> Circuit:
    c = Circuit(dimension=3, num_qudits=11)
    for op in RELAY_X + RELAY_Z:
        if len(op) == 2:
            c.add_gate(op[0], op[1])
        else:
            assert frozenset(op[1:]) in LAYOUT_COUPLERS, op
            c.add_gate(*op)
    for a in (5, 6, 7, 8, 9, 10):
        c.add_gate("M", a)
    return c


def code_definitions():
    seq = CodeSimulationProfileRegistry.resolve_code_profile("folded_qutrit").code_definition
    par = dataclasses.replace(seq, stabilizer_check_blocks=((par9_block, {5, 6, 7, 8}),))
    spl = CodeSimulationProfileRegistry.resolve_code_profile(
        "folded_qutrit_split_unif").code_definition
    rel = dataclasses.replace(spl, stabilizer_check_blocks=((relay11_block, {5, 6, 7, 8, 9, 10}),))
    return {"seq9": seq, "par9": par, "relay11": rel}


# ------------------------------------------------------------ noise models
HW = {
    "HW-now": dict(e_cz=0.027, e1=0.006, n_bc=1.33, e_meas=0.03, e_reset=0.002,
                   t1=0.10, t_cz=0.58, t_meas=0.50, t_reset=0.30,
                   coh=COH["Goss2022 Q4 echo"]),
    "HW-target": dict(e_cz=0.010, e1=0.001, n_bc=1.33, e_meas=0.01, e_reset=0.001,
                      t1=0.05, t_cz=0.20, t_meas=0.30, t_reset=0.25,
                      coh=COH["target (2x Goss Q4 echo)"]),
}
HW["HW-now-R"] = dict(HW["HW-now"], coh=COH["Goss2022 Q4 Ramsey"])
HW["HW-qubit"] = dict(HW["HW-target"], e_cz=0.005, e1=0.0005, e_meas=0.005, e_reset=0.001)
TIMING_OF = {"HW-now": "HW-now", "HW-now-R": "HW-now", "HW-target": "HW-target",
             "HW-qubit": "HW-target"}
PHYS_1Q = {"H", "H_INV", "X", "X_INV"}
VIRT_1Q = {"P", "P_INV", "Z", "Z_INV", "I"}
TWO_Q = {"CNOT", "CNOT_INV", "CZ", "CZ_INV", "SWAP"}


def e_cnot(m):
    return 1 - (1 - m["e_cz"]) * (1 - m["e1"]) ** m["n_bc"]


def duration(name, m):
    if name in ("CNOT", "CNOT_INV"):
        return m["t_cz"] + m["n_bc"] * m["t1"]
    if name in ("CZ", "CZ_INV"):
        return m["t_cz"]
    if name in PHYS_1Q:
        return m["t1"]
    return 0.0


def classify_noise(ops, i, nxt_name):
    op = ops[i]
    prm = op.params or {}
    sc = float(prm.get("prob_scale", 1.0))
    ch = prm.get("noise_channel", "d")
    j = i - 1
    while j >= 0 and ops[j].gate_name in ("N1", "N2"):
        j -= 1
    prev = ops[j].gate_name if j >= 0 else None
    if op.gate_name == "N2":
        return "swap" if prev == "SWAP" else ("cz" if prev in ("CZ", "CZ_INV") else "cnot")
    if ch == "f":
        return "meas_flip" if sc == 5.0 else "reset_flip"
    if abs(sc - 0.1) < 1e-12:
        return "1q_phys" if prev in PHYS_1Q else "1q_virt"
    if sc == 2.0:
        return "idle_meas" if nxt_name == "M" else "idle_reset"
    if sc == 1.0:
        return "meas_depol"
    raise ValueError((i, prm))


def next_gate_names(ops):
    out = [None] * len(ops)
    nxt = None
    for i in range(len(ops) - 1, -1, -1):
        out[i] = nxt
        if ops[i].gate_name not in ("N1", "N2"):
            nxt = ops[i].gate_name
    return out


def schedule_and_label(circ, n_enc, timing=None):
    """Return (new circuit, labels by op index, round durations).

    timing=None: SI1000 structure only (labels, no inserted idle).
    timing=model dict: ASAP schedule of gates, measurement and reset layers
    block all qudits; an idle N1 is inserted before an operation for every
    gap of an active qudit (data always active; ancillas active from their
    first gate after a reset until measured).
    """
    ops = circ.operations
    nxt = next_gate_names(ops)
    nq = circ.num_qudits
    ann = [i for i, o in enumerate(ops) if o.qudit_index is None]
    body_end = ann[0] if ann else len(ops)
    term_start = body_end                        # ideal terminal X readout of data
    while term_start > 0 and ops[term_start - 1].gate_name in ("H_INV", "M") \
            and int(ops[term_start - 1].qudit_index) < 5:
        term_start -= 1
    assert body_end - term_start in (4, 10), body_end - term_start
    tmp = Circuit(dimension=3, num_qudits=nq)
    new_ops, labels = [], {}
    avail = [0.0] * nq
    active = [q < 5 for q in range(nq)]
    layer_starts = []

    def add_idle(q, gap):
        tmp.add_gate("N1", q, prob=1e-3, noise_channel="d")
        labels[len(new_ops)] = ("idle_gate", q, gap)
        new_ops.append(tmp.operations.pop())

    i = 0
    while i < len(ops):
        op = ops[i]
        name = op.gate_name
        if i < n_enc or i >= term_start:
            new_ops.append(op)
            i += 1
            continue
        if name in ("N1", "N2"):
            cls = classify_noise(ops, i, nxt[i])
            q = int(op.qudit_index)
            act = active[q] if timing is not None else True
            labels[len(new_ops)] = (cls, q, act)
            new_ops.append(op)
            i += 1
            continue
        if name in ("M", "RESET"):
            # gather the whole layer (noise ops interleaved)
            j = i
            members = []
            while j < term_start and ops[j].gate_name in ("N1", "N2", name):
                if ops[j].gate_name == name:
                    members.append(int(ops[j].qudit_index))
                j += 1
            # the SI1000 idle ops of this layer precede it; they were already
            # copied, so idle gaps must be inserted before them: handled by
            # synchronizing here and relabelling below.
            if timing is not None:
                T = max(avail)
                for q in range(nq):
                    if active[q] and T - avail[q] > 1e-9:
                        add_idle(q, T - avail[q])
                tl = timing["t_meas"] if name == "M" else timing["t_reset"]
                layer_starts.append((name, T))
                for q in range(nq):
                    avail[q] = T + tl
            for k in range(i, j):
                o = ops[k]
                if o.gate_name in ("N1", "N2"):
                    cls = classify_noise(ops, k, nxt[k])
                    q = int(o.qudit_index)
                    act = active[q] if timing is not None else True
                    labels[len(new_ops)] = (cls, q, act)
                new_ops.append(o)
            if timing is not None:
                for q in members:
                    active[q] = False      # measured / reset: no further idle cost
            i = j
            continue
        # gate
        qs = [int(op.qudit_index)] + ([int(op.target_index)] if op.target_index is not None else [])
        if timing is not None:
            if name == "SWAP":           # virtual: relabel the two wires
                a, b = qs
                avail[a], avail[b] = avail[b], avail[a]
                active[a], active[b] = active[b], active[a]
            elif name in VIRT_1Q:
                pass
            else:
                start = max(avail[q] for q in qs)
                for q in qs:
                    if active[q] and start - avail[q] > 1e-9:
                        add_idle(q, start - avail[q])
                    avail[q] = start + duration(name, timing)
                    active[q] = True
        new_ops.append(op)
        i += 1
    # move SI1000 idle labels that sit before a layer but were emitted before
    # the inserted gap idles: harmless (all are independent N1 locations)
    out = Circuit(dimension=3, num_qudits=nq)
    out.operations = new_ops
    return out, labels, layer_starts


def prob_of(label, model, p=None):
    cls, q, extra = label
    if model == "SI1000":
        return {"cnot": p, "cz": p, "swap": p, "1q_phys": 0.1 * p, "1q_virt": 0.1 * p,
                "idle_meas": 2 * p, "idle_reset": 2 * p, "meas_depol": p,
                "meas_flip": 5 * p, "reset_flip": 2 * p, "idle_gate": 0.0}[cls]
    m = HW[model]
    if cls == "cnot":
        return e_cnot(m)
    if cls == "cz":
        return m["e_cz"]
    if cls == "swap":
        return 0.0
    if cls == "1q_phys":
        return m["e1"]
    if cls == "1q_virt":
        return 0.0
    if cls == "meas_flip":
        return m["e_meas"]
    if cls == "meas_depol":
        return 0.0
    if cls == "reset_flip":
        return m["e_reset"]
    if cls == "idle_meas":
        return e_idle(m["t_meas"], m["coh"]) if extra else 0.0
    if cls == "idle_reset":
        return e_idle(m["t_reset"], m["coh"]) if extra else 0.0
    if cls == "idle_gate":
        return e_idle(extra, m["coh"])
    raise KeyError(cls)


def reweight(dem, labels, model, p=None):
    locs = []
    for loc in dem.locations:
        r = prob_of(labels[loc.operation_index], model, p)
        if r <= 0:
            continue
        if loc.gate == "N2":
            prm = dict(loc.params)
            prm["prob_dist"] = two_qudit_depol_prob_dist(3, r)
            locs.append(dataclasses.replace(loc, probability=r, params=prm))
        else:
            locs.append(dataclasses.replace(loc, probability=r))
    return dataclasses.replace(dem, locations=tuple(locs))


# ------------------------------------------------------------ statistics
def round_of(label, m):
    # lrb_stab_r{round}_w{wire}; const0 terminal detectors count as round m
    if label.startswith("lrb_stab_r"):
        return int(label.split("_r")[1].split("_w")[0])
    return m


def stats(dem, payload, m):
    det = np.stack([d["data"] for d in payload["detectors"]], axis=1)
    log = payload["logicals"][0]["data"]
    rnd = np.array([round_of(l, m) for l in dem.detector_labels])
    acc_all = ~np.any(det != 0, axis=1)
    acc_term = ~np.any(det[:, rnd == m] != 0, axis=1)
    def fid(mask):
        return float(np.abs(np.mean(OMEGA ** log[mask]))) if mask.sum() else float("nan")
    return dict(acc_u1=acc_all.mean(), fid_u1=fid(acc_all), acc_c1=acc_term.mean(),
                fid_c1=fid(acc_term), raw_s=fid(np.ones(len(log), bool)))


def exact_raw(dem):
    f = 1.0
    for loc in dem.locations:
        co = [sum(c for _, c in t.logical) % 3 for t in loc.component_terms]
        if loc.gate == "N1" and loc.channel == "f":
            sens, n = co[0] != 0, 2
        elif loc.gate == "N1":
            sens, n = any(co[:2]), 8
        else:
            sens, n = any(co), 80
        if sens:
            f *= 1 - loc.probability * (n + 1) / n
    return f


VEC = {
    "d": np.array([v for v in itertools.product(range(3), repeat=2) if any(v)]),
    "f": np.array([[1, 0], [2, 0]]),
    "N2": np.array([v for v in itertools.product(range(3), repeat=4) if any(v)]),
}


def first_order(dem):
    nd = dem.num_detectors
    classes = {}
    for loc in dem.locations:
        V = VEC["N2"] if loc.gate == "N2" else VEC[loc.channel]
        C = np.zeros((loc.vector_width, nd), dtype=np.int64)
        L = np.zeros(loc.vector_width, dtype=np.int64)
        for i, t in enumerate(loc.component_terms):
            for d, c in t.detector:
                C[i, d] = (C[i, d] + c) % 3
            L[i] = sum(c for _, c in t.logical) % 3
        S = (V @ C) % 3
        Lg = (V @ L) % 3
        w = loc.probability / len(V)
        for s, lg in zip(S, Lg):
            key = s.tobytes()
            cl = classes.setdefault(key, np.zeros(3))
            cl[lg] += w
    zero = np.zeros(nd, dtype=np.int64).tobytes()
    silent = classes.get(zero, np.zeros(3))
    det_mass = amb_mass = mld_fail = 0.0
    n_amb = 0
    for key, cl in classes.items():
        if key == zero:
            continue
        tot = cl.sum()
        det_mass += tot
        if (cl > 1e-15).sum() > 1:
            amb_mass += tot
            n_amb += 1
        mld_fail += tot - cl.max()
    return dict(silent_logical=silent[1] + silent[2], detected=det_mass,
                ambiguous=amb_mass, mld_fail=mld_fail, n_sig=len(classes) - 1,
                n_amb_sig=n_amb)


# ------------------------------------------------------------ checks
def noiseless_check(cd, n_seq=6, depth=3, shots=3, rng=None):
    gen = LRBCircuitGenerator(code_definition=cd, noise_model="si1000")
    bad = 0
    tot = 0
    for _ in range(n_seq):
        cs = [rng.choice(cd.clifford_strings) for _ in range(depth)]
        circ = gen.generate_lrb_clifford_sequence([depth], with_noise=False, clifford_strings=cs)[0]
        c2 = Circuit(dimension=3, num_qudits=circ.num_qudits)
        c2.operations = [o for o in circ.operations if o.qudit_index is not None]
        for _s in range(shots):
            res = Program(c2).simulate(shots=1, force_tableau=True)
            anc = [r for r in res if r.qudit_index >= 5]
            term = [r for r in res if r.qudit_index in (0, 3)][-2:]
            tot += 1
            if any((not r.deterministic) or r.measurement_value % 3 for r in anc) or \
                    (term[0].measurement_value + term[1].measurement_value) % 3:
                bad += 1
    return bad, tot


def lookup_decoder(dem, m, n_train, n_test, threshold=0.0):
    """Empirical maximum-likelihood lookup decoder (train/test split).

    In LRB the ideal logical outcome of every shot is 0, so each shot's
    logical shift is observed; a syndrome -> most-frequent-shift table can be
    learned from data.  Returns fidelities of: raw, corrected (all shots),
    hybrid (reject unseen syndromes and those whose empirical posterior max is
    below `threshold`), and LRB-D unif 1 (all rounds).
    """
    tr = sample_detector_error_model(dem, extra_shots=n_train)
    te = sample_detector_error_model(dem, extra_shots=n_test)
    def arr(pl):
        det = np.stack([d["data"] for d in pl["detectors"]], axis=1).astype(np.int64)
        key = det @ (3 ** np.arange(det.shape[1], dtype=np.int64))
        return key, pl["logicals"][0]["data"].astype(np.int64)
    ktr, ltr = arr(tr)
    kte, lte = arr(te)
    table = {}
    for k, l in zip(ktr.tolist(), ltr.tolist()):
        c = table.setdefault(k, [0, 0, 0])
        c[l] += 1
    pred = np.zeros(len(kte), dtype=np.int64)
    conf = np.zeros(len(kte))
    seen = np.zeros(len(kte), bool)
    for i, k in enumerate(kte.tolist()):
        c = table.get(k)
        if c is None:
            continue
        seen[i] = True
        j = int(np.argmax(c))
        pred[i] = j
        conf[i] = c[j] / sum(c)
    resid = (lte - pred) % 3
    f = lambda mask: float(np.abs(np.mean(OMEGA ** resid[mask]))) if mask.sum() else float("nan")
    keep = seen & (conf >= threshold)
    acc_all = kte == 0
    return dict(raw=float(np.abs(np.mean(OMEGA ** lte))), corrected=f(np.ones(len(kte), bool)),
                hybrid=f(keep), hybrid_acc=keep.mean(), seen=seen.mean(),
                lrbd=float(np.abs(np.mean(OMEGA ** lte[acc_all]))) if acc_all.sum() else float("nan"),
                lrbd_acc=acc_all.mean())


MODELS = [("SI1000 p=1e-3", "SI1000", 1e-3), ("SI1000 p=5e-3", "SI1000", 5e-3),
          ("HW-now", "HW-now", None), ("HW-now-R", "HW-now-R", None),
          ("HW-target", "HW-target", None), ("HW-qubit", "HW-qubit", None)]


def main():
    t0 = time.time()
    rng = random.Random(SEED)
    np.random.seed(SEED)
    cds = code_definitions()
    cat = cds["seq9"].clifford_strings

    print("== noiseless determinism (all ancilla/relay outcomes 0, logical 0) ==")
    for name in ("par9", "relay11"):
        bad, tot = noiseless_check(cds[name], rng=rng)
        print(f"  {name}: bad shots {bad}/{tot}")
    print(f"  [{time.time() - t0:.0f}s]")

    print("\n== model parameters ==")
    for name, m in HW.items():
        print(f"  {name}: CNOT error {e_cnot(m):.4f} ({m['t_cz'] + m['n_bc'] * m['t1']:.3f} us),"
              f" CZ {m['e_cz']} ({m['t_cz']} us), 1q {m['e1']} ({m['t1']} us), readout flip"
              f" {m['e_meas']} ({m['t_meas']} us), reset {m['e_reset']} ({m['t_reset']} us),"
              f" idle per readout layer {e_idle(m['t_meas'], m['coh']):.4f}, per us"
              f" {e_idle(1.0, m['coh']):.4f}")

    depths = [2, 6, 10]
    K = 6
    seqs = [[rng.choice(cat) for _ in range(max(depths))] for _ in range(K)]
    shots = 15000
    results, fo, durations, demcache = {}, {}, {}, {}
    for sname in ("seq9", "par9", "relay11", "const0"):
        cd = cds["seq9"] if sname == "const0" else cds[sname]
        gen = LRBCircuitGenerator(code_definition=cd, noise_model="si1000")
        n_enc = len(cd.logical_plus_initial_state().operations)
        for k, cs in enumerate(seqs):
            if sname == "const0":
                circs = gen.generate_lrb_const0_clifford_sequence(depths, with_noise=True,
                                                                  clifford_strings=cs)
            else:
                circs = gen.generate_lrb_clifford_sequence(depths, with_noise=True,
                                                           clifford_strings=cs)
            for m, circ in zip(sorted(depths), circs):
                gen.update_noise_param(circ, 0.001)
                built = {}
                c_si, lab_si, _ = schedule_and_label(circ, n_enc, None)
                built["SI1000"] = (build_compact_detector_error_model(c_si), lab_si)
                for tname in ("HW-now", "HW-target"):
                    c_hw, lab_hw, layers = schedule_and_label(circ, n_enc, HW[tname])
                    built[tname] = (build_compact_detector_error_model(c_hw), lab_hw)
                    if k == 0 and sname != "const0":
                        rs = [T for nm, T in layers if nm == "RESET"]
                        if len(rs) >= 2:
                            durations[(sname, tname)] = rs[1] - rs[0]
                for vname, model, p in MODELS:
                    key = "SI1000" if model == "SI1000" else TIMING_OF[model]
                    dem, lab = built[key]
                    rdem = reweight(dem, lab, model, p)
                    st = stats(rdem, sample_detector_error_model(rdem, extra_shots=shots), m)
                    st["raw_x"] = exact_raw(rdem)
                    results.setdefault((sname, vname, m), []).append(st)
                    if k == 0 and m == 2:
                        fo[(sname, vname, m)] = first_order(rdem)
                    if k < 3 and m == 2 and vname in ("SI1000 p=5e-3", "HW-target", "HW-qubit") \
                            and sname != "const0":
                        demcache[(sname, vname, k)] = rdem
        print(f"  {sname} done [{time.time() - t0:.0f}s]")

    print("\n== LRB step duration (us; forward Clifford + check round + reset) ==")
    for (s, t), d in sorted(durations.items()):
        print(f"  {s:8s} {t:10s} {d:6.2f}")

    print(f"\n== acceptance / accepted fidelity (mean over K={K} sequences, {shots} shots;"
          " const0: terminal X checks only) ==")
    print("  schedule  model            m   unif1 acc / fid      const1 acc / fid     raw(exact)")
    for (s, v, m), rs in sorted(results.items(), key=lambda x: ([mm[0] for mm in MODELS].index(x[0][1]), x[0][2], x[0][0])):
        a = lambda kk: np.nanmean([r[kk] for r in rs])
        print(f"  {s:8s}  {v:15s} {m:3d}   {a('acc_u1'):.4f} / {a('fid_u1'):.4f}"
              f"     {a('acc_c1'):.4f} / {a('fid_c1'):.4f}     {a('raw_x'):.4f}")

    print("\n== first-order fault classes at m=2 (sequence 0): expected faults per shot"
          " and fractions ==")
    print("  schedule  model            faults/shot  silent-logical  ambiguous  MLD-1st-order-fail  (fractions of all faults)")
    for (s, v, m), f in sorted(fo.items(), key=lambda x: ([mm[0] for mm in MODELS].index(x[0][1]), x[0][0])):
        tot = f["detected"] + f["silent_logical"]
        tot_all = tot  # silent-trivial mass excluded from both
        print(f"  {s:8s}  {v:15s}  {tot:9.3f}     {f['silent_logical'] / tot_all:.3f}"
              f"          {f['ambiguous'] / tot_all:.3f}      {f['mld_fail'] / tot_all:.4f}")

    print("\n== sequence-to-sequence spread (exact raw fidelity, unif1 acceptance) ==")
    for (s, v, m), rs in sorted(results.items()):
        if v in ("HW-target", "HW-qubit", "SI1000 p=5e-3"):
            x = np.array([r["raw_x"] for r in rs])
            acc = np.array([r["acc_u1"] for r in rs])
            fu = np.array([r["fid_u1"] for r in rs])
            print(f"  {s:8s} {v:14s} m={m:2d}: raw mean {x.mean():.4f} sd {x.std(ddof=1):.4f};"
                  f" unif1 acc mean {acc.mean():.4f} sd {acc.std(ddof=1):.4f};"
                  f" unif1 fid sd {np.nanstd(fu, ddof=1):.4f}")

    print("\n== empirical lookup decoder at m=2 (train 2e5 / test 1e5 shots, 3 sequences) ==")
    print("  schedule  model            raw     corrected   hybrid(conf>=0.6) acc   LRB-D unif1 acc   test syndromes seen")
    for (s, v) in sorted({(a, b) for a, b, _ in demcache}):
        rs = [lookup_decoder(demcache[(s, v, k)], 2, 200000, 100000, 0.6) for k in range(3)]
        a = lambda kk: np.nanmean([r[kk] for r in rs])
        print(f"  {s:8s}  {v:15s} {a('raw'):.4f}   {a('corrected'):.4f}     {a('hybrid'):.4f} {a('hybrid_acc'):.3f}"
              f"     {a('lrbd'):.4f} {a('lrbd_acc'):.4f}     {a('seen'):.3f}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
