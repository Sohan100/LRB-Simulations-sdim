"""Critic DEM checks of H1 claims C13, C14, C24, C25.

Uses the repo generator and compact DEM (as H1 did) but asks different
questions:
 A. C13 decomposition: how much of the sequential -> parallel acceptance
    gain at SI1000 p = 0.005 comes from data idle versus from idle on
    ancillas (unmeasured ancillas waiting through other ancillas'
    measurement layers)?
 B. C14 sensitivity of the "hardware-calibrated" HW-A numbers to terms H1
    leaves out or treats inconsistently with its own protocol:
      - SWAP(2,3) of logical H costs one N2 at p2 in HW-A, but H1's protocol
        makes it virtual (cost 0); a physical qutrit SWAP would cost ~3 CZ;
      - gate-layer idle (H1's own C12): data 0/2/3/4 idle ~1.23 us per
        parallel round with 580 ns CZ (1.7% with Goss-Q4 echo coherence),
        Z ancillas 5/6 wait ~1.23 us before readout, and data 0/1/4 idle
        during every CZ(2,3) of logical P (0.58 us, 0.79%);
      - relays on a square lattice: 24 instead of 12 two-qutrit gates per
        round (crude: check-round N2 probability 1-(1-p2)^2).
 C. C24/C25: Clifford-to-Clifford spread of the RAW logical fidelity
    (exact per sequence from the DEM: product over logical-sensitive
    locations) and of unif-1 accepted fidelity, compared with the
    shot-noise-only standard errors used in h1_shot_budget.py.

Run: module load python/3.11; python critic_H1-sc-hardware_dem.py
Runtime: about 3-4 minutes on one core.
"""
from __future__ import annotations

import dataclasses
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))

from sdim.circuit import Circuit  # noqa: E402

from lrb.circuit_generator import LRBCircuitGenerator  # noqa: E402
from lrb.code_simulation_profiles import CodeSimulationProfileRegistry  # noqa
from lrb.correction.compact_dem import (build_compact_detector_error_model,  # noqa
                                sample_detector_error_model,
                                two_qudit_depol_prob_dist)

SEED = 4242
OMEGA = np.exp(2j * np.pi / 3)


def parallel_block() -> Circuit:
    c = Circuit(dimension=3, num_qudits=9)
    for g, a, b in (("CNOT", 0, 5), ("CNOT_INV", 1, 5), ("CNOT_INV", 3, 5),
                    ("CNOT", 1, 6), ("CNOT", 2, 6), ("CNOT_INV", 4, 6)):
        c.add_gate(g, a, b)
    for anc, gs in ((7, (("CNOT", 7, 0), ("CNOT", 7, 1), ("CNOT_INV", 7, 2))),
                    (8, (("CNOT", 8, 3), ("CNOT_INV", 8, 1),
                         ("CNOT_INV", 8, 4)))):
        c.add_gate("H", anc)
        for g, a, b in gs:
            c.add_gate(g, a, b)
        c.add_gate("H_INV", anc)
    for a in (5, 6, 7, 8):
        c.add_gate("M", a)
    return c


def code_defs():
    seq = CodeSimulationProfileRegistry.resolve_code_profile(
        "folded_qutrit").code_definition
    par = dataclasses.replace(seq, stabilizer_check_blocks=((parallel_block,
                                                             {5, 6, 7, 8}),))
    return seq, par


def label_locations(circ):
    """Per noise op index: (class, qudit, context) with context telling
    whether the op sits in a check round, after SWAP, after CZ(2,3), or
    after a P/P_INV gate."""
    ops = circ.operations
    out = {}
    in_check = False
    for i, op in enumerate(ops):
        name = op.gate_name
        if name in ("N1", "N2"):
            prm = op.params or {}
            sc = float(prm.get("prob_scale", 1.0))
            ch = prm.get("noise_channel", "d")
            prev = ops[i - 1]
            j = i + 1
            while j < len(ops) and ops[j].gate_name in ("N1", "N2"):
                j += 1
            nxt = ops[j].gate_name if j < len(ops) else None
            q = int(op.qudit_index)
            if name == "N2":
                cls = ("2q_swap" if prev.gate_name == "SWAP" else
                       "2q_fold" if prev.gate_name in ("CZ", "CZ_INV") else
                       "2q_check")
            elif ch == "f":
                cls = "meas_flip" if sc == 5.0 else "reset_flip"
            elif abs(sc - 0.1) < 1e-12:
                cls = ("1q_P" if prev.gate_name in ("P", "P_INV")
                       and int(prev.qudit_index) in (0, 1, 4) else "1q")
            elif sc == 2.0:
                cls = "idle_reset" if nxt == "RESET" else "idle_meas"
            elif sc == 1.0:
                cls = "meas_depol"
            else:
                raise ValueError((i, prm))
            out[i] = (cls, q)
    return out


def rates_si1000(p):
    return {"2q": p, "1q": 0.1 * p, "idle_meas": 2 * p, "idle_reset": 2 * p,
            "meas_depol": p, "meas_flip": 5 * p, "reset_flip": 2 * p}


HW_A = {"2q": 0.027, "1q": 0.003, "idle_meas": 0.0068,
        "idle_reset": 0.0034, "meas_depol": 0.0, "meas_flip": 0.025,
        "reset_flip": 0.002}
HW_C = {"2q": 0.005, "1q": 0.001, "idle_meas": 0.0068,
        "idle_reset": 0.0034, "meas_depol": 0.0, "meas_flip": 0.01,
        "reset_flip": 0.001}


def rate_of(base, cls, q, opts):
    """Location probability for a model `base` with option flags."""
    if cls.startswith("2q"):
        r = base["2q"]
        if cls == "2q_swap" and "virtual_swap" in opts:
            return 0.0
        if cls == "2q_check" and "relay_double" in opts:
            return 1 - (1 - r) ** 2
        return r
    if cls == "1q_P":
        r = base["1q"]
        return r + (0.0079 if "gate_idle" in opts else 0.0)
    if cls == "idle_meas":
        r = base["idle_meas"]
        if "no_anc_idle" in opts and q >= 5:
            return 0.0
        if "gate_idle" in opts and q in (0, 2, 3, 4):
            return 1 - (1 - r) * (1 - 0.017)
        return r
    if cls == "meas_depol":
        r = base["meas_depol"]
        if "gate_idle" in opts and q in (5, 6):
            return 1 - (1 - r) * (1 - 0.017)
        return r
    return base[cls]


def reweight(model, labels, base, opts=()):
    locs = []
    for loc in model.locations:
        cls, q = labels[loc.operation_index]
        r = rate_of(base, cls, q, opts)
        if r <= 0:
            continue
        if loc.gate == "N2":
            prm = dict(loc.params)
            prm["prob_dist"] = two_qudit_depol_prob_dist(3, r)
            locs.append(dataclasses.replace(loc, probability=r, params=prm))
        else:
            locs.append(dataclasses.replace(loc, probability=r))
    return dataclasses.replace(model, locations=tuple(locs))


def exact_raw_fidelity(model):
    """|E[w^L]| = prod over locations of (1 - p (n+1)/n) when the logical
    response is a non-zero linear functional on the location's support."""
    f = 1.0
    for loc in model.locations:
        coeffs = [sum(c for _, c in t.logical) % 3 for t in loc.component_terms]
        if loc.gate == "N1" and loc.channel == "f":
            sens, n = coeffs[0] != 0, 2
        elif loc.gate == "N1":
            sens, n = any(coeffs), 8
        else:
            sens, n = any(coeffs), 80
        if sens:
            f *= 1 - loc.probability * (n + 1) / n
    return abs(f)


def stats(payload, depth):
    det = np.stack([d["data"] for d in payload["detectors"]], axis=1)
    log = payload["logicals"][0]["data"]
    acc = ~np.any(det != 0, axis=1)
    n = int(acc.sum())
    fid = float(np.abs(np.mean(OMEGA ** log[acc]))) if n else float("nan")
    raw = float(np.abs(np.mean(OMEGA ** log)))
    return n / len(log), fid, raw


def var_re(F):
    q = (2 * F + 1) / 3
    return q + (1 - q) / 4 - F ** 2


def main():
    t0 = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    seq_cd, par_cd = code_defs()
    cat = seq_cd.clifford_strings

    # ---------------- A: decomposition at SI1000 p=0.005, m=2 ----------
    print("== A. C13 decomposition (SI1000 p=0.005, m=2, 6 seqs x 2e4) ==")
    gen_s = LRBCircuitGenerator(code_definition=seq_cd, noise_model="si1000")
    gen_p = LRBCircuitGenerator(code_definition=par_cd, noise_model="si1000")
    resA = {"seq": [], "seq_no_anc_idle": [], "par": []}
    for k in range(6):
        cs = [random.choice(cat) for _ in range(2)]
        for tag, gen in (("seq", gen_s), ("par", gen_p)):
            circ = gen.generate_lrb_clifford_sequence([2], with_noise=True,
                                                      clifford_strings=cs)[0]
            gen.update_noise_param(circ, 0.001)
            lab = label_locations(circ)
            mdl = build_compact_detector_error_model(circ)
            optsets = [((), tag)]
            if tag == "seq":
                optsets.append((("no_anc_idle",), "seq_no_anc_idle"))
            for opts, key in optsets:
                pl = sample_detector_error_model(
                    reweight(mdl, lab, rates_si1000(0.005), opts),
                    extra_shots=20000)
                resA[key].append(stats(pl, 2))
    for key, rs in resA.items():
        print(f"  {key}: unif-1 acceptance {np.mean([r[0] for r in rs]):.3f},"
              f" accepted fidelity {np.nanmean([r[1] for r in rs]):.4f}")
    print(f"  [{time.time() - t0:.0f}s]")

    # ---------------- B + C: parallel variant, K sequences -------------
    K = 16
    depths = [2, 6, 10]
    print(f"\n== B/C. parallel variant, K={K} sequences, depths {depths} ==")
    models = {}
    seqs = [[random.choice(cat) for _ in range(max(depths))] for _ in range(K)]
    for k, cs in enumerate(seqs):
        circs = gen_p.generate_lrb_clifford_sequence(depths, with_noise=True,
                                                     clifford_strings=cs)
        for m, circ in zip(sorted(depths), circs):
            gen_p.update_noise_param(circ, 0.001)
            models[(k, m)] = (build_compact_detector_error_model(circ),
                              label_locations(circ))
    print(f"  compiled {len(models)} circuits [{time.time() - t0:.0f}s]")

    variants = {
        "HW-A as in H1": (HW_A, ()),
        "HW-A + virtual SWAP": (HW_A, ("virtual_swap",)),
        "HW-A + virtual SWAP + gate-layer idle": (HW_A, ("virtual_swap",
                                                         "gate_idle")),
        "HW-A + virtual SWAP + gate idle + relay x2 (crude)":
            (HW_A, ("virtual_swap", "gate_idle", "relay_double")),
    }
    print("  B. unif-1 acceptance / accepted fidelity / raw fidelity "
          "(mean over first 8 sequences, 2e4 shots):")
    for vname, (base, opts) in variants.items():
        line = f"   {vname}:"
        for m in (2, 6):
            rs = []
            for k in range(8):
                mdl, lab = models[(k, m)]
                pl = sample_detector_error_model(reweight(mdl, lab, base, opts),
                                                 extra_shots=20000)
                rs.append(stats(pl, m))
            line += (f" m={m}: {np.mean([r[0] for r in rs]):.3f}/"
                     f"{np.nanmean([r[1] for r in rs]):.3f}/"
                     f"{np.mean([r[2] for r in rs]):.3f};")
        print(line)
    print(f"  [{time.time() - t0:.0f}s]")

    print("\n  C. sequence-to-sequence spread at HW-C (parallel), K="
          f"{K}:")
    N = 30000
    for m in depths:
        raw_exact = []
        acc, fid = [], []
        for k in range(K):
            mdl, lab = models[(k, m)]
            rm = reweight(mdl, lab, HW_C)
            raw_exact.append(exact_raw_fidelity(rm))
            a, f, _ = stats(sample_detector_error_model(rm, extra_shots=N), m)
            acc.append(a)
            fid.append(f)
        raw_exact = np.array(raw_exact)
        Fm = raw_exact.mean()
        shot_se_per_circ = np.sqrt(var_re(Fm) / N)
        fid = np.array(fid)
        acc = np.array(acc)
        shot_fid = np.sqrt(var_re(np.nanmean(fid)) / (N * acc.mean()))
        print(f"   m={m:2d}: RAW exact mean {Fm:.4f}, std over sequences "
              f"{raw_exact.std(ddof=1):.2e} (shot noise per circuit at N=3e4:"
              f" {shot_se_per_circ:.2e}); SE of K=30 mean: "
              f"sequence-limited {raw_exact.std(ddof=1) / np.sqrt(30):.2e} vs "
              f"H1 shot-only {np.sqrt(var_re(Fm) / (30 * N)):.2e}")
        print(f"          unif-1: acceptance {acc.mean():.3f} (std "
              f"{acc.std(ddof=1):.3f}, binomial {np.sqrt(acc.mean() * (1 - acc.mean()) / N):.3f}),"
              f" accepted fidelity std {np.nanstd(fid, ddof=1):.2e} vs shot "
              f"noise {shot_fid:.2e}")
    print(f"\ntotal runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
