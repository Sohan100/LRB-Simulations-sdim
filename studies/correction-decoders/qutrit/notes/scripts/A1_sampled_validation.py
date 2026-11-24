#!/usr/bin/env python3
"""A1 revision: sampled-shot validation of the leading-order decoders.

The critique of A1 (A1-syndromes-critique.md, missing item 1) noted that the
MLD and hybrid numbers of A1 are perturbative only.  This script samples
shots from the exact Pauli-frame noise model of the baseline LRB circuits
and decodes them with lookup-table decoders built from the single-fault
table (the leading-order decoders of A1), at p = 5e-4 ... 5e-3.

Noise model: every noise location of a baseline circuit fires independently
with probability p * prob_scale and then applies one of its non-identity
Pauli components uniformly (N1 'd': 8, N1 'f': 2, N2: 80), which is the
repository CompactDetectorErrorModel.  The unit responses come from
A1_syndrome_classification.propagate_units, which reproduces that DEM
exactly (A1 section G, critique C20).  Firing shots are drawn by geometric
skipping, which is an exact Bernoulli process.

Decoders (V1: the detector record of the circuit as run; class = shift of
the terminal observable x0 + x3):
  * detection: accept iff all detectors are zero (the LRB-D baseline);
  * hybrid(tau): accept the zero signature and every signature that appears
    in the single-fault table with leading-order posterior >= tau; decode to
    the majority class.  tau = 1 is the strict hybrid of A1;
  * table MLD: accept everything; decode table signatures to the majority
    class and every other signature (two or more faults) to class 0.  This
    equals MLD only at first order.
  * space-time hybrid(tau) and space-time MLD (function greedy): the raw
    ancilla outcomes are converted to detection events with the syndrome
    maps of the logical Cliffords, so that every single fault occupies at
    most two consecutive event rounds (asserted), and the events are peeled
    round by round with the single-fault groups.  For one fault this is the
    leading-order decoder of A1; unlike the table decoders it also decodes
    shots with several faults that are separated in space-time.
Fidelity of the accepted shots: |sum_k w^k P(k)| with k = observable shift
minus decoded class (the LRB metric of the brief).

Run (login node, one or two cores, about four minutes):
    module load python/3.11
    OMP_NUM_THREADS=2 python A1_sampled_validation.py
Writes A1_sampled_validation.json and prints a log.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "a1", os.path.join(HERE, "A1_syndrome_classification.py"))
a1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a1)
D = 3
W = np.exp(2j * np.pi / 3)
TAUS = (1.0, 0.95, 0.9)


def log(*args):
    print(*args, flush=True)


def _token_syndrome_maps():
    """4x4 syndrome map M of every logical token (sigma(U E U^dag) = M
    sigma(E)), from the symplectic frame rules of A1 section F."""
    Ek = []
    for k in range(4):
        tgt = np.zeros(4, dtype=np.int64)
        tgt[k] = 1
        i = np.flatnonzero(np.all(a1.SYN == tgt, axis=1))[0]
        Ek.append(a1.VECS[i])
    out = {}
    for tok in ("H", "H_INV", "P", "P_INV", "X", "X_INV", "Z", "Z_INV", "I"):
        G = a1.symplectic_matrix(a1.logical_gate_ops(tok))
        out[tok] = np.array([a1.syn_of((G @ e) % D) for e in Ek]).T % D
    return out


MTOK = _token_syndrome_maps()
P3 = np.array([1, 3, 9, 27])


def round_maps(ops):
    """A_r maps the round r-1 syndrome of a persisting data error to its round
    r syndrome: the forward Clifford between rounds r-1 and r (r < m) or the
    whole inverse block (r = m)."""
    fwd, inv, _ = a1.tokens_from_ops(ops)
    m = len(fwd)

    def seq(toks):
        A = np.eye(4, dtype=np.int64)
        for t in toks:
            A = (MTOK[t] @ A) % D
        return A
    return [None] + [seq(fwd[r]) for r in range(1, m)] + ([seq(inv)] if m else [])


def to_events(raw, As):
    """Detection events e_0 = o_0, e_r = o_r - A_r o_{r-1} (raw: (n, m+1, 4))."""
    ev = raw.astype(np.int16).copy()
    for r in range(1, raw.shape[1]):
        ev[:, r] = (raw[:, r] - raw[:, r - 1].astype(np.int64) @ As[r].T) % D
    return ev


def build(ops):
    """Single-fault table and per-location response blocks (V1)."""
    t0 = a1.terminal_start(ops)
    prop = a1.propagate_units(ops, snapshot_at=t0)
    resp, wts, locid, _ = a1.single_faults(prop)
    nd = prop["det"].shape[0]
    sig = (resp[:, :nd] % D).astype(np.int8)
    cls = (resp[:, nd] % D).astype(np.int64)
    # ---- space-time tables for the greedy decoder -------------------------
    m1 = nd // 4
    As = round_maps(ops)
    ev = to_events(sig.reshape(-1, m1, 4), As)               # (n, m1, 4)
    nzr = np.any(ev != 0, axis=2)
    first_r = np.where(nzr.any(axis=1), nzr.argmax(axis=1), -1)
    last_r = np.where(nzr.any(axis=1), m1 - 1 - nzr[:, ::-1].argmax(axis=1), -1)
    assert np.all((last_r - first_r) <= 1), "a fault spans more than two event rounds"
    evc = (ev @ P3).astype(np.int64)                          # (n, m1) codes 0..80
    gkey = np.where(first_r >= 0, first_r, m1) * 6561 + \
        evc[np.arange(len(evc)), np.maximum(first_r, 0)] * 81 + \
        np.where((first_r >= 0) & (first_r + 1 < m1),
                 evc[np.arange(len(evc)), np.minimum(first_r + 1, m1 - 1)], 0)
    ug, ginv = np.unique(gkey, return_inverse=True)
    ginv = ginv.reshape(-1)
    Wg = np.zeros((len(ug), 3))
    np.add.at(Wg, (ginv, cls), wts)
    g_tot = Wg.sum(1)
    g_best = Wg.argmax(1)
    g_post = np.where(g_tot > 0, Wg.max(1) / np.where(g_tot > 0, g_tot, 1), 1.0)
    g_r = ug // 6561
    g_x = (ug // 81) % 81
    g_y = ug % 81
    T2 = -np.ones((m1, 81, 81), dtype=np.int64)
    T1 = -np.ones((m1, 81), dtype=np.int64)
    for gi in range(len(ug)):
        if g_r[gi] >= m1:
            continue                    # zero event pattern (undetected / no effect)
        if g_y[gi]:
            T2[g_r[gi], g_x[gi], g_y[gi]] = gi
        else:
            T1[g_r[gi], g_x[gi]] = gi
    g_ydig = np.array([[(y // 3 ** k) % 3 for k in range(4)] for y in g_y],
                      dtype=np.int16)
    st = dict(As=As, m1=m1, T1=T1, T2=T2, post=g_post, best=g_best, ydig=g_ydig)
    # leading-order table
    sv = np.ascontiguousarray(sig).view(np.dtype((np.void, nd))).ravel()
    uq, first, g = np.unique(sv, return_index=True, return_inverse=True)
    g = g.reshape(-1)
    Wt = np.zeros((len(uq), 3))
    np.add.at(Wt, (g, cls), wts)
    tot = Wt.sum(1)
    best = Wt.argmax(1)
    post = np.where(tot > 0, Wt.max(1) / np.where(tot > 0, tot, 1), 1.0)
    zero = ~np.any(sig[first] != 0, axis=1)
    best[zero] = 0
    # sampling blocks: one per location (rows are contiguous per location)
    blocks = []
    bounds = np.flatnonzero(np.diff(locid)) + 1
    for rows in np.split(np.arange(len(locid)), bounds):
        l = int(locid[rows[0]])
        scale = prop["locs"][l][4]
        R = np.hstack([sig[rows], cls[rows, None].astype(np.int8)]).astype(np.int16)
        blocks.append((scale, R))
    return dict(nd=nd, table_keys=sig[first], post=post, best=best, zero=zero,
                blocks=blocks, total_weight=float(wts.sum()), st=st)


def greedy(model, S, tau, reject=True):
    """Space-time greedy decoder.  Raw outcomes are turned into detection
    events with the Clifford syndrome maps; every single fault then occupies
    at most two consecutive event rounds (asserted in build).  Rounds are
    processed in time order: the events left in round r are explained by the
    single-fault group with pattern (round r, round r+1) equal to the events
    left there if it exists, otherwise by the group with that round-r
    pattern and nothing in round r+1.  The group's majority class is added
    to the correction and its pattern removed.  Hybrid mode (reject=True)
    rejects a shot when some step finds no group or a group whose
    leading-order posterior is below tau; MLD mode (reject=False) accepts
    every shot and drops unexplained events.  For one fault this is exactly
    the leading-order decoder of A1; it also handles faults that are
    separated in space-time."""
    st = model["st"]
    nd = model["nd"]
    m1 = st["m1"]
    N = len(S)
    ev = to_events(S[:, :nd].reshape(N, m1, 4), st["As"])
    obs = S[:, nd].astype(np.int64)
    corr = np.zeros(N, dtype=np.int64)
    ok = np.ones(N, dtype=bool)
    for r in range(m1):
        x = (ev[:, r] @ P3).astype(np.int64)
        need = x != 0
        if not need.any():
            continue
        y = (ev[:, r + 1] @ P3).astype(np.int64) if r + 1 < m1 else np.zeros(N, np.int64)
        g = np.where(st["T2"][r][x, y] >= 0, st["T2"][r][x, y], st["T1"][r][x])
        g = np.where(need, g, -1)
        unexpl = need & (g < 0)
        good = need & (g >= 0)
        if reject:
            ok &= ~unexpl
            ok &= ~(good & (st["post"][np.maximum(g, 0)] < tau - 1e-12))
        corr[good] += st["best"][g[good]]
        ev[need, r] = 0
        if r + 1 < m1:
            ev[good, r + 1] = (ev[good, r + 1] - st["ydig"][g[good]]) % D
    k = (obs - corr) % D
    n = int(ok.sum())
    if n == 0:
        return dict(n_acc=0, P_acc=0.0, P_fail_given_acc=float("nan"),
                    fidelity=float("nan"))
    P = np.bincount(k[ok], minlength=3) / n
    return dict(n_acc=n, P_acc=n / N, P_fail_given_acc=float(1 - P[0]),
                fidelity=float(abs(P[0] + W * P[1] + W * W * P[2])))


def sample(model, p, N, rng):
    nd = model["nd"]
    S = np.zeros((N, nd + 1), dtype=np.int16)
    for i, (scale, R) in enumerate(model["blocks"]):
        pi = p * scale
        if pi <= 0:
            continue
        exp_n = N * pi
        pos = np.cumsum(rng.geometric(pi, size=int(exp_n + 8 * np.sqrt(exp_n) + 20))) - 1
        while pos[-1] < N:            # rare: extend the Bernoulli process
            more = np.cumsum(rng.geometric(pi, size=int(exp_n + 20))) + pos[-1]
            pos = np.concatenate([pos, more])
        pos = pos[pos < N]
        if len(pos):
            S[pos] += R[rng.integers(len(R), size=len(pos))]
        if i % 256 == 255:
            S %= D
    S %= D
    return S


def decode(model, S):
    nd = model["nd"]
    sig = np.ascontiguousarray(S[:, :nd].astype(np.int8))
    obs = S[:, nd].astype(np.int64)
    T = np.ascontiguousarray(model["table_keys"])
    allv = np.concatenate([T, sig]).view(np.dtype((np.void, nd))).ravel()
    _, inv = np.unique(allv, return_inverse=True)
    inv = inv.reshape(-1)
    nT = len(T)
    gid = np.full(inv.max() + 1, -1, dtype=np.int64)
    gid[inv[:nT]] = np.arange(nT)
    g = gid[inv[nT:]]                              # table group of each shot
    shot_zero = ~np.any(sig != 0, axis=1)
    in_t = g >= 0
    out = {}

    def stats(acc, dec):
        k = (obs[acc] - dec[acc]) % D
        n = int(acc.sum())
        if n == 0:
            return dict(n_acc=0, P_acc=0.0, P_fail_given_acc=float("nan"),
                        fidelity=float("nan"))
        P = np.bincount(k, minlength=3) / n
        return dict(n_acc=n, P_acc=n / len(obs), P_fail_given_acc=float(1 - P[0]),
                    fidelity=float(abs(P[0] + W * P[1] + W * W * P[2])))

    zero_dec = np.zeros(len(obs), dtype=np.int64)
    out["detection"] = stats(shot_zero, zero_dec)
    dec = np.where(in_t, model["best"][np.maximum(g, 0)], 0)
    dec[shot_zero] = 0
    for tau in TAUS:
        acc = shot_zero | (in_t & (model["post"][np.maximum(g, 0)] >= tau - 1e-12))
        out[f"hybrid_{tau}"] = stats(acc, dec)
    out["table_MLD"] = stats(np.ones(len(obs), dtype=bool), dec)
    for tau in (1.0, 0.95):
        out[f"st_hybrid_{tau}"] = greedy(model, S, tau, reject=True)
    out["st_MLD"] = greedy(model, S, 0.0, reject=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/pscratch/sd/s/sohan100/LRB-Simulations-sdim/"
                    "LRB-experiment-data-slurm/Run-2026-08-04-23-58-49-"
                    "folded_qutrit-single-ancilla-si1000")
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--seed", type=int, default=20261028)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    run = args.run
    depths = [int(t) for t in open(os.path.join(run, "depths.txt")).read()
              .strip().split(",")]
    probs = [float(t) for t in open(os.path.join(run, "probs.txt")).read()
             .strip().split(",")]
    a1r = json.load(open(os.path.join(HERE, "A1_results.json")))
    o2 = a1r["G4b_order2_all_sequences"]["per_sequence"]
    REPORT = dict(seed=args.seed)
    t_start = time.time()

    def circ(c, di):
        return a1.parse_chp(os.path.join(run, "experiments", "LRB", str(c), "4",
                                         f"{di}.chp"))

    # ---- (a) m = 0 and m = 2: sampled values against the series ----------
    log("=" * 78)
    log("S1. m = 0 and m = 2: sampled shots versus the perturbative series")
    rows = []
    for (c, di, N) in [(0, 0, 1_000_000)] + [(c, 1, 1_000_000) for c in range(4)]:
        model = build(circ(c, di))
        m = depths[di]
        lo = a1.analyse_circuit(circ(c, di), want_breakdown=False)["V1_as_run"]
        c2 = o2[str(m)][c]["V1"]
        for p in ((5e-4, 1e-3, 2e-3, 5e-3) if m == 0 else (1e-3, 2e-3, 5e-3)):
            res = decode(model, sample(model, p, N, rng))
            pred = dict(
                det_acc=float(np.exp(-c2["det_rej1"] * p)),
                det_fail=lo["undetected_fail"] * p + c2["det_fail_given_acc_2"] * p * p,
                hyb_acc=float(np.exp(-c2["hyb_rej1"] * p)),
                hyb_fail=lo["hyb_fail"] * p + c2["hyb_fail_given_acc_2"] * p * p,
                mld_fail_order1=lo["mld_fail"] * p)
            row = dict(seq=c, m=m, p=p, N=N, sampled=res, predicted=pred)
            rows.append(row)
            d, h, t = res["detection"], res["hybrid_1.0"], res["table_MLD"]
            sh, sm = res["st_hybrid_1.0"], res["st_MLD"]
            se = lambda P, n: np.sqrt(max(P, 1e-12) * (1 - P) / max(n, 1))
            log(f"  seq {c} m={m:2d} p={p:.0e}: detection acc {d['P_acc']:.4f} (exp(-r1 p)"
                f" {pred['det_acc']:.4f}), P(f|a) {d['P_fail_given_acc']:.2e} +- "
                f"{se(d['P_fail_given_acc'], d['n_acc']):.1e} (series {pred['det_fail']:.2e})"
                f" | table hybrid acc {h['P_acc']:.4f} ({pred['hyb_acc']:.4f}), P(f|a) "
                f"{h['P_fail_given_acc']:.2e} +- {se(h['P_fail_given_acc'], h['n_acc']):.1e}"
                f" (series {pred['hyb_fail']:.2e}) | space-time hybrid acc {sh['P_acc']:.4f},"
                f" P(f|a) {sh['P_fail_given_acc']:.2e} | MLD fail: table {t['P_fail_given_acc']:.2e},"
                f" space-time {sm['P_fail_given_acc']:.2e} (order 1 {pred['mld_fail_order1']:.2e})")
    # baseline at m = 0 (all sequences share this circuit)
    for pi, p in enumerate(probs):
        if p in (5e-4, 1e-3, 2e-3, 5e-3):
            b = a1.baseline_csv(run, pi, "unif_check_data", 1)
            s_ = next(r for r in rows if r["m"] == 0 and r["p"] == p)["sampled"]["detection"]
            log(f"  baseline m=0 p={p:.0e}: rejection {b['Rejected Runs'][0]:.5f} vs sampled"
                f" {1 - s_['P_acc']:.5f}; fidelity {b['Fidelity averages'][0]:.6f} vs sampled"
                f" {s_['fidelity']:.6f}")
    REPORT["S1_series"] = rows

    # ---- (a') m = 22 at small p: the space-time decoders must reproduce the
    # leading-order rates when faults are sparse
    log("=" * 78)
    log("S0. m = 22, sequence 0, p = 2e-4: sampled versus leading order")
    ops22 = circ(0, depths.index(22))
    model = build(ops22)
    lo = a1.analyse_circuit(ops22, want_breakdown=False)["V1_as_run"]
    p = 2e-4
    res = decode(model, sample(model, p, 400_000, rng))
    s0 = dict(p=p, N=400_000, sampled=res, exp_det=float(np.exp(-lo["det_rej"] * p)),
              exp_hyb=float(np.exp(-lo["hyb_rej"] * p)), mld_order1=lo["mld_fail"] * p,
              undetected_order1=lo["undetected_fail"] * p)
    log(f"  detection acc {res['detection']['P_acc']:.4f} (exp(-r p) {s0['exp_det']:.4f});"
        f" space-time hybrid acc {res['st_hybrid_1.0']['P_acc']:.4f} (exp(-r p)"
        f" {s0['exp_hyb']:.4f}); table hybrid acc {res['hybrid_1.0']['P_acc']:.4f};"
        f" space-time MLD fail {res['st_MLD']['P_fail_given_acc']:.4f} (order 1"
        f" {s0['mld_order1']:.4f}); table MLD fail {res['table_MLD']['P_fail_given_acc']:.4f}")
    REPORT["S0_small_p_m22"] = s0

    # ---- (b) decay with depth at finite p ------------------------------------
    log("=" * 78)
    log("S2. Decay with depth at finite p (sequences 0-5, depths 2, 6, 10, 14, 22)")
    seqs = list(range(6))
    dis = [depths.index(m) for m in (2, 6, 10, 14, 22)]
    ps = (1e-3, 2e-3, 4e-3)
    Ns = {1e-3: 100_000, 2e-3: 100_000, 4e-3: 200_000}
    dec_names = ["detection", "hybrid_1.0", "st_hybrid_1.0", "st_hybrid_0.95", "st_MLD"]
    acc_F = {p: {dn: np.zeros((len(seqs), len(dis))) for dn in dec_names} for p in ps}
    acc_A = {p: {dn: np.zeros((len(seqs), len(dis))) for dn in dec_names} for p in ps}
    for si, c in enumerate(seqs):
        for j, di in enumerate(dis):
            model = build(circ(c, di))
            for p in ps:
                res = decode(model, sample(model, p, Ns[p], rng))
                for dn in dec_names:
                    acc_F[p][dn][si, j] = res[dn]["fidelity"]
                    acc_A[p][dn][si, j] = res[dn]["P_acc"]
        log(f"  sequence {c} done ({time.time() - t_start:.0f} s)")
    ms = np.array([depths[di] for di in dis], dtype=float)
    fit_rows = []
    rb = {}
    fit_csv = os.path.join(run, "results", "plots", "unif_lrb_vs_rb_table_all_mixed_fits.csv")
    for line in open(fit_csv).read().strip().splitlines()[1:]:
        t = line.split(",")
        if t[0] == "1":
            rb[float(t[3])] = 1 - float(t[16])          # p_raw -> RB 1 - f
    hyb_slope = a1r["G4b_order2_all_sequences"]["summary"][
        "V1_hyb_fail_given_acc_2"]["slope_m2to6_per_seq"]["mean"]
    tau_fits = a1r["G3_lrb_circuits"]["threshold_family"]["V1_as_run"]["fits"]
    for p in ps:
        for dn in dec_names:
            F = acc_F[p][dn]
            Fm = np.nanmean(F, axis=0)
            ok = np.isfinite(Fm) & (Fm > 0)
            coef = np.polyfit(ms[ok], np.log(Fm[ok]), 1)
            f = float(np.exp(coef[0]))
            # sequence-bootstrap of the fitted 1 - f
            bs = []
            brng = np.random.default_rng(1)
            for _ in range(400):
                pick = brng.integers(len(seqs), size=len(seqs))
                Fb = np.nanmean(F[pick], axis=0)
                okb = np.isfinite(Fb) & (Fb > 0)
                bs.append(1 - np.exp(np.polyfit(ms[okb], np.log(Fb[okb]), 1)[0]))
            pred = None
            if dn in ("hybrid_1.0", "st_hybrid_1.0"):
                pred = 1.5 * hyb_slope * p * p        # A1 second-order slope
            elif dn == "st_hybrid_0.95":
                pred = 1.5 * tau_fits["fail_0.95"]["slope_per_depth"] * p   # first order only
            elif dn == "st_MLD":
                pred = 1.5 * tau_fits["fail_0.0"]["slope_per_depth"] * p    # first order only
            row = dict(p=p, decoder=dn, one_minus_f=1 - f, one_minus_f_bootstrap_std=float(np.std(bs)),
                       F_by_depth=Fm.tolist(), acc_by_depth=np.mean(acc_A[p][dn], axis=0).tolist(),
                       prediction=pred, rb_one_minus_f=rb.get(p))
            fit_rows.append(row)
            log(f"  p={p:.0e} {dn:12s}: 1-f = {1 - f:.2e} +- {np.std(bs):.1e}"
                + (f" (perturbative {pred:.2e})" if pred is not None else "")
                + (f" | physical RB 1-f {rb[p]:.2e}" if p in rb else
                   f" | physical RB 1-f ~ {5.64 * p:.2e} (5.64p)")
                + f" | mean acceptance by depth {np.round(row['acc_by_depth'], 3).tolist()}"
                + f" | mean F by depth {np.round(Fm, 4).tolist()}")
    REPORT["S2_decay"] = fit_rows
    REPORT["S2_settings"] = dict(sequences=seqs, depths=ms.tolist(), shots=
                                 {str(k): v for k, v in Ns.items()},
                                 hybrid_p2_slope_used=hyb_slope)
    out = os.path.join(args.out, "A1_sampled_validation.json")
    with open(out, "w") as fh:
        json.dump(REPORT, fh, indent=1)
    log(f"wrote {out}; total time {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
