"""Independent adversarial verification of work package F (non-Markovianity).

Written by the F verifier; it does not import the implementer's scripts
(``nonmarkov_runs.py``, ``nonmarkov_analysis.py``). It uses the lead's model
(``src/lrb_qubit``) and reads the implementer's raw result files only where a
spot check of their numbers is the point.

Subcommands (outputs in ``studies/correction-decoders/qubit/results/F-verify``):

``abs``
    Claim 1 / estimator check: per-sequence signed polarization
    ``F_s = 1 - 2 e/a`` versus the repo's ``|F_s|`` on every sampled and exact
    curve of WP-F and on the lead's 200-sequence exact null curves. Lists the
    curves where the absolute value changes the mean, and refits them both
    ways (rate and relative drift).
``twirl``
    Claim 3: "each logical fault multiplies <X_L> by -1/3 in expectation".
    (a) the physical logical gates permute the three logical representatives
    {X0X3, Z0Z2, Y0Z2X3}; (b) Bell-pair probe circuits give the exact logical
    Pauli channel of every twirl block (forward C_j + its check round + the
    inverse C_{j+1}^-1); (c) these channels, composed along each sequence's
    actual frames, reproduce the lead's exact null curves sequence by
    sequence; (d) replacing each block by its twirl average (each
    non-identity block class weighted -1/3) predicts the sequence-averaged
    curve, while the alternatives (-1 or 0 per fault) do not.
``validate``
    The lead's ``RawPolicyEvaluator`` on ``inverse_check_every=1`` circuits
    (not covered by the lead's tests): brute force at depths 1-2 and Stim
    sampling at larger depths.
``exact``
    Exact curves with ``RawPolicyEvaluator`` / ``exact_condition`` for fresh
    sequences (default 20 sequences, seed 5150000, p = 1e-3, depths 2..30):
    standard circuit (unif=1,2,4, const=1, null) and the
    ``inverse_check_every=1`` circuit (forward + terminal rounds used, i.e.
    inverse checks executed but unused; all rounds used; null).
``spacing``
    Claim 12 from the ``exact`` output: per-block rates of unif=1,2,4 on
    depths 4..24 and their ratios.
``invunused``
    Claim 16/17 from the ``exact`` output: curvature of LRB-D with the inverse
    checks executed but not used.
``mlddrift``
    Claim 7 spot check: MLD drift at p = 5e-3 from the implementer's
    ``std_p5e-3.npz`` with this script's own estimators.
``misc``
    Spot checks of claims 5, 8, 9, 10 from the implementer's raw files.

Usage (from the repository root):

    module load python/3.11
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    export PYTHONPATH=$PWD/src
    python studies/correction-decoders/qubit/scripts/verify_nonmarkov.py abs
    python studies/correction-decoders/qubit/scripts/verify_nonmarkov.py twirl --p 5e-4,5e-3
    python studies/correction-decoders/qubit/scripts/verify_nonmarkov.py validate
    python studies/correction-decoders/qubit/scripts/verify_nonmarkov.py exact --workers 4
    python studies/correction-decoders/qubit/scripts/verify_nonmarkov.py spacing
    python studies/correction-decoders/qubit/scripts/verify_nonmarkov.py invunused
    python studies/correction-decoders/qubit/scripts/verify_nonmarkov.py mlddrift
    python studies/correction-decoders/qubit/scripts/verify_nonmarkov.py misc
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if any(os.environ.get(_var) != "1" for _var in _THREAD_VARS):
    for _var in _THREAD_VARS:
        os.environ[_var] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import stim  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import baseline as B  # noqa: E402
from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import code as C  # noqa: E402
from lrb_qubit import exact as X  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

DOC = ROOT / "studies" / "correction-decoders" / "qubit"
RES_F = DOC / "results" / "F"
OUT = DOC / "results" / "F-verify"
NBOOT = 4000
RNG_SEED = 424242


def save(name: str, obj) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(obj, indent=1, default=float))
    print(f"wrote {path.relative_to(ROOT)}", flush=True)
    return path


def pct(samples, axis=0):
    lo, hi = np.nanpercentile(samples, [2.5, 97.5], axis=axis)
    return lo, hi


# ---------------------------------------------------------------------------
# Simple estimators (deliberately different from the implementer's weighted
# fits): unweighted OLS of log(mean F) and its sequence bootstrap.
# ---------------------------------------------------------------------------

def ols_slope(x, y):
    """Slope of an unweighted least-squares line; y may be (n,) or (B, n)."""
    x = np.asarray(x, float)
    xc = x - x.mean()
    y = np.asarray(y, float)
    if y.ndim == 1:
        return float(np.sum(xc * (y - y.mean())) / np.sum(xc ** 2))
    return (y - y.mean(axis=1, keepdims=True)) @ xc / np.sum(xc ** 2)


def ols_quad(x, y):
    """Quadratic coefficient of an unweighted quadratic fit (y (n,) or (B, n))."""
    x = np.asarray(x, float)
    Xm = np.stack([np.ones_like(x), x, x ** 2], axis=1)
    G = np.linalg.pinv(Xm)
    y = np.asarray(y, float)
    return (G @ y.T).T[..., 2] if y.ndim == 2 else float((G @ y)[2])


def drift(depths, logF, lo, hi):
    """rho = (r_late - r_early) / r_all on [lo, hi] split in thirds (OLS)."""
    d = np.asarray(depths, float)
    third = (hi - lo) / 3.0
    sel = (d >= lo) & (d <= hi)
    early = sel & (d <= lo + third)
    late = sel & (d >= hi - third)
    r_all = -ols_slope(d[sel], logF[..., sel])
    r_e = -ols_slope(d[early], logF[..., early])
    r_l = -ols_slope(d[late], logF[..., late])
    return r_all, r_e, r_l, (r_l - r_e) / r_all


def curve_stats(depths, Fs, lo, hi, rng, nboot=NBOOT):
    """Rate, quadratic coefficient and drift of log(mean_s F_s) with a
    sequence bootstrap. ``Fs`` is (n_seq, n_depth)."""
    d = np.asarray(depths, float)
    Fbar = np.nanmean(Fs, axis=0)
    idx = rng.integers(0, Fs.shape[0], (nboot, Fs.shape[0]))
    Fb = np.nanmean(Fs[idx], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        logF = np.log(Fbar)
        logFb = np.log(np.where(Fb > 0, Fb, np.nan))
    sel = (d >= lo) & (d <= hi)
    r_all, r_e, r_l, rho = drift(d, logF, lo, hi)
    rb, reb, rlb, rhob = drift(d, logFb, lo, hi)
    c2 = ols_quad(d[sel], logF[sel])
    c2b = ols_quad(d[sel], logFb[:, sel])
    out = {"window": [lo, hi], "r": r_all, "r_ci": list(pct(rb)),
           "r_early": r_e, "r_late": r_l, "rho": rho, "rho_ci": list(pct(rhob)),
           "c2": c2, "c2_ci": list(pct(c2b))}
    return out, {"r": rb, "rho": rhob, "c2": c2b}


# ---------------------------------------------------------------------------
# abs: signed versus absolute polarization
# ---------------------------------------------------------------------------

def _fs_from_counts(acc, err):
    acc = acc.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(acc > 0, 1.0 - 2.0 * err / acc, np.nan)


def cmd_abs(args) -> int:
    rng = np.random.default_rng(RNG_SEED)
    report = {"sampled": [], "exact": [], "null200": [], "refits": []}
    for variant in ("std", "inv1"):
        for tag in ("p5e-4", "p2e-3", "p5e-3"):
            path = RES_F / f"{variant}_{tag}.npz"
            if not path.exists():
                continue
            pr = E.PointResult.from_npz(str(path))
            depths = np.asarray(pr.depths)
            for name, t in sorted(pr.tables.items()):
                keep = t.shots.sum(axis=1) > 0
                acc, err = t.accepted[keep], t.errors[keep]
                Fs = _fs_from_counts(acc, err)
                Fbar = np.nanmean(Fs, axis=0)
                Fabs = np.nanmean(np.abs(Fs), axis=0)
                se = np.nanstd(Fs, axis=0, ddof=1) / np.sqrt(Fs.shape[0])
                usable = (acc.min(axis=0) >= 100) & (Fbar > 4 * se)
                # contiguous from depth 0
                stop = np.argmin(usable) if not usable.all() else len(usable)
                usable[stop:] = False
                neg = (Fs < 0).mean(axis=0)
                rel = np.where(usable, Fabs / Fbar - 1.0, np.nan)
                row = {"variant": variant, "p": tag, "method": name,
                       "min_Fbar": float(np.nanmin(Fbar)),
                       "m_hi_usable": int(depths[usable].max())
                       if usable.any() else None,
                       "max_rel_bias_usable": float(np.nanmax(rel))
                       if np.isfinite(rel).any() else None,
                       "max_rel_bias_all": float(np.nanmax(Fabs / Fbar - 1.0)),
                       "max_neg_fraction_usable": float(neg[usable].max())
                       if usable.any() else None}
                report["sampled"].append(row)
                if (row["max_rel_bias_usable"] or 0) > 0.01 and \
                        usable.sum() >= 8:
                    hi = float(depths[usable].max())
                    s1, _ = curve_stats(depths[usable], Fs[:, usable], 2.0, hi,
                                        rng, 1000)
                    s2, _ = curve_stats(depths[usable],
                                        np.abs(Fs[:, usable]), 2.0, hi, rng,
                                        1000)
                    report["refits"].append({
                        "variant": variant, "p": tag, "method": name,
                        "window": [2.0, hi],
                        "signed": {k: s1[k] for k in ("r", "rho", "rho_ci")},
                        "abs": {k: s2[k] for k in ("r", "rho", "rho_ci")}})
            xpath = RES_F / f"exact_{variant}_{tag}.npz"
            if xpath.exists():
                data = np.load(xpath)
                for m in data["methods"]:
                    m = str(m)
                    a = data[f"{m}__accept"]
                    f = data[f"{m}__acc_flip"]
                    Fs = 1.0 - 2.0 * f / a
                    report["exact"].append({
                        "variant": variant, "p": tag, "method": m,
                        "min_Fs": float(np.nanmin(Fs)),
                        "n_negative_Fs": int((Fs < 0).sum()),
                        "min_Fbar": float(np.nanmin(Fs.mean(axis=0)))})
    for p in ("0.0005", "0.001", "0.002", "0.005"):
        path = DOC / "results" / "exact_null" / f"exact_null_p{p}.npz"
        data = np.load(path)
        for key in ("flip_checks_data5", "flip_nochecks"):
            Fs = 1.0 - 2.0 * data[key]
            report["null200"].append({"p": p, "key": key,
                                      "min_Fs": float(Fs.min()),
                                      "n_negative": int((Fs < 0).sum()),
                                      "min_Fbar": float(Fs.mean(axis=0).min())})
    save("abs_check.json", report)
    print("\nSampled curves where |F_s| changes the mean by > 1% at usable "
          "depths:")
    for r in report["sampled"]:
        if (r["max_rel_bias_usable"] or 0) > 0.01:
            print(f"  {r['variant']} {r['p']} {r['method']:22s} min Fbar="
                  f"{r['min_Fbar']:+.2e} usable to {r['m_hi_usable']}: "
                  f"max (|F|-F)/F = {r['max_rel_bias_usable']:.3f}, "
                  f"neg frac {r['max_neg_fraction_usable']:.2f}")
    print("\nRefits (OLS of log mean, window [2, m_hi]):")
    for r in report["refits"]:
        print(f"  {r['variant']} {r['p']} {r['method']:22s} signed r="
              f"{r['signed']['r']:.4g} rho={r['signed']['rho']:+.3f} | abs r="
              f"{r['abs']['r']:.4g} rho={r['abs']['rho']:+.3f}")
    print("\nExact curves with negative per-sequence F_s:",
          [(r["variant"], r["p"], r["method"], r["n_negative_Fs"])
           for r in report["exact"] if r["n_negative_Fs"]])
    print("Exact null 200: negatives:",
          [(r["p"], r["key"], r["n_negative"]) for r in report["null200"]])
    return 0


# ---------------------------------------------------------------------------
# twirl: claim 3 on the real circuits
# ---------------------------------------------------------------------------

REP_X = stim.PauliString("X__X_")
REP_Z = stim.PauliString("Z_Z__")
REP_Y = stim.PauliString("Y_ZX_")   # proportional to REP_X * REP_Z
REF = 9                              # reference qubit of the Bell probe


def _same_up_to_sign(a: stim.PauliString, b: stim.PauliString) -> bool:
    ax, az = a.to_numpy()
    bx, bz = b.to_numpy()
    return bool((ax == bx).all() and (az == bz).all())


def axis_of(pauli: stim.PauliString) -> int:
    """0 = X rep, 1 = Z rep, 2 = Y rep; raises if not a representative."""
    for k, rep in enumerate((REP_X, REP_Z, REP_Y)):
        if _same_up_to_sign(pauli, rep):
            return k
    raise ValueError(f"{pauli} is not one of the three representatives")


def rep_closure() -> dict:
    """Check that every physical logical gate (and every one of the 24
    Cliffords) permutes the three representatives, and that the axis of
    C X0X3 C^dag is uniform over the 24 Cliffords."""
    tokens = ["H", "P", "X", "Z", "I", "H_INV", "P_INV", "X_INV", "Z_INV"]
    gate_maps = {}
    for tok in tokens:
        tab = C.data_tableau(C.logical_gate_operations(tok))
        gate_maps[tok] = [axis_of(tab(rep)) for rep in (REP_X, REP_Z, REP_Y)]
    counts = [0, 0, 0]
    minus_third = []
    for desc in C.SINGLE_QUBIT_CLIFFORDS:
        tab = C.data_tableau(C.logical_clifford_operations(desc))
        counts[axis_of(tab(REP_X))] += 1
        # E[(-1)^{anticommute}] for a fixed logical fault (X_L, Y_L, Z_L)
    for fault in (REP_X, REP_Y, REP_Z):
        signs = []
        for desc in C.SINGLE_QUBIT_CLIFFORDS:
            tab = C.data_tableau(C.logical_clifford_operations(desc))
            readout = tab(REP_X)            # readout pulled back to the frame
            signs.append(1.0 if fault.commutes(readout) else -1.0)
        minus_third.append(float(np.mean(signs)))
    return {"gate_axis_maps": gate_maps, "axis_counts_over_24": counts,
            "mean_sign_for_fixed_logical_fault_XYZ": minus_third}


def bell_prep() -> stim.Circuit:
    """Ideal circuit preparing the code state Bell-paired with qubit 9."""
    stabs = []
    pad = "____"  # ancillas 5-8
    for s in C.STABILIZER_PAULIS:
        stabs.append(stim.PauliString(s + pad + "_"))
    for a in range(4):
        stabs.append(stim.PauliString("_____" + "".join(
            "Z" if b == a else "_" for b in range(4)) + "_"))
    stabs.append(stim.PauliString("X__X_" + pad + "X"))
    stabs.append(stim.PauliString("Z_Z__" + pad + "Z"))
    tab = stim.Tableau.from_stabilizers(stabs)
    return tab.to_circuit("elimination")


def _mpp(circuit: stim.Circuit, pauli5: stim.PauliString, ref: str) -> None:
    targets = []
    for q in range(5):
        ch = "_XYZ"[pauli5[q]]
        if ch == "_":
            continue
        targets.append({"X": stim.target_x, "Y": stim.target_y,
                        "Z": stim.target_z}[ch](q))
        targets.append(stim.target_combiner())
    targets.append({"X": stim.target_x, "Z": stim.target_z}[ref](REF))
    circuit.append("MPP", targets)


def probe_channel(noisy_ops: list, ideal_frame_ops: list, p: float
                  ) -> np.ndarray:
    """Exact logical eigenvalues of a probe block.

    ``noisy_ops`` is a list of (ops, noisy, measurements_noisy) segments
    executed in order after the ideal Bell preparation. ``ideal_frame_ops``
    are the data operations whose net ideal action the final observables
    must be conjugated by. Returns lambda along the three representatives
    (X0X3, Z0Z2, Y0Z2X3), each conjugated by the net ideal frame.
    """
    circ = bell_prep()
    noise = Q.SI1000()
    for ops, noisy, meas_noisy in noisy_ops:
        Q.emit_operations(circ, ops, p if noisy else 0.0,
                          noise if noisy else None, C.NUM_QUBITS,
                          noisy_measurements=meas_noisy)
    V = C.data_tableau(ideal_frame_ops)
    _mpp(circ, V(REP_X), "X")
    _mpp(circ, V(REP_Z), "Z")
    circ.append("OBSERVABLE_INCLUDE", [stim.target_rec(-2)], 0)
    circ.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 1)
    dem = circ.detector_error_model()
    lam = np.ones(3)
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        prob = inst.args_copy()[0]
        o = [0, 0]
        for t in inst.targets_copy():
            if t.is_logical_observable_id():
                o[t.val] ^= 1
            elif t.is_relative_detector_id():
                raise AssertionError("probe has no detectors")
        flips = (o[0], o[1], o[0] ^ o[1])
        for k in range(3):
            if flips[k]:
                lam[k] *= 1.0 - 2.0 * prob
    return lam


def _fwd(desc):
    return C.logical_clifford_operations(desc)


def _inv(desc):
    return C.logical_clifford_operations(desc, inverse=True)


def block_tables(p: float) -> dict:
    """Probe channels for every block type and Clifford (pair)."""
    cl = C.SINGLE_QUBIT_CLIFFORDS
    n = len(cl)
    check = C.check_round_operations()
    terminal = (C.z_stabilizer_check_operations()
                + C.x_stabilizer_check_operations())
    full = np.zeros((n, n, 3))      # block j: noisy C, check, ideal C', noisy C'^-1
    last = np.zeros((n, 3))         # block m: noisy C, check
    c0 = np.zeros(n)                # block 0 at m >= 1: ideal C1, noisy C1^-1, terminal
    for a, ca in enumerate(cl):
        last[a] = probe_channel([(_fwd(ca), True, False), (check, True, True)],
                                _fwd(ca), p)
        c0[a] = probe_channel([(_fwd(ca), False, False),
                               (_inv(ca), True, False),
                               (terminal, True, True)], [], p)[0]
        for b, cb in enumerate(cl):
            full[a, b] = probe_channel(
                [(_fwd(ca), True, False), (check, True, True),
                 (_fwd(cb), False, False), (_inv(cb), True, False)],
                _fwd(ca), p)
    c0_depth0 = probe_channel([(terminal, True, True)], [], p)[0]
    return {"full": full, "last": last, "c0": c0, "c0_depth0": c0_depth0}


def frames(sequence: list[str], depth: int) -> list[int]:
    """Axis (0 X, 1 Z, 2 Y) of the readout X0X3 pulled back to the start of
    each forward Clifford C_j, j = 1..depth."""
    axes = []
    R = REP_X.copy()
    for desc in sequence[:depth]:
        axes.append(axis_of(R))
        R = C.data_tableau(_fwd(desc))(R)
    return axes


def cmd_twirl(args) -> int:
    closure = rep_closure()
    print("representative closure:", closure, flush=True)
    seqs = E.clifford_sequences(200, 40, 20261028)
    index = {d: i for i, d in enumerate(C.SINGLE_QUBIT_CLIFFORDS)}
    report = {"closure": closure, "points": {}}
    for p in (float(x) for x in args.p.split(",")):
        t0 = time.perf_counter()
        tabs = block_tables(p)
        print(f"p={p:g}: probes in {time.perf_counter() - t0:.1f}s",
              flush=True)
        exact = np.load(DOC / "results" / "exact_null" /
                        f"exact_null_p{p:g}.npz")
        F_exact = 1.0 - 2.0 * exact["flip_checks_data5"]   # (200, 41)
        depths = np.asarray(exact["depths"])
        full, last, c0 = tabs["full"], tabs["last"], tabs["c0"]
        f_full = full.mean(axis=2)          # twirl average per (C, C')
        f_last = last.mean(axis=1)
        # (1) per-sequence reconstruction with the actual frames
        recon = np.zeros_like(F_exact)
        twirl = np.zeros_like(F_exact)
        bad_x = np.zeros_like(F_exact)      # alternative: no twirl (axis X)
        for s, seq in enumerate(seqs):
            ids = [index[d] for d in seq]
            for di, m in enumerate(depths):
                if m == 0:
                    recon[s, di] = twirl[s, di] = bad_x[s, di] = \
                        tabs["c0_depth0"]
                    continue
                ax = frames(seq, m)
                val = c0[ids[0]]
                tw = c0[ids[0]]
                bx = c0[ids[0]]
                for j in range(1, m):          # blocks 1..m-1
                    lam = full[ids[j - 1], ids[j]]
                    val *= lam[ax[j - 1]]
                    tw *= f_full[ids[j - 1], ids[j]]
                    bx *= lam[0]
                lam = last[ids[m - 1]]
                val *= lam[ax[m - 1]]
                tw *= f_last[ids[m - 1]]
                bx *= lam[0]
                recon[s, di] = val
                twirl[s, di] = tw
                bad_x[s, di] = bx
        max_abs = float(np.max(np.abs(recon - F_exact)))
        Fbar = F_exact.mean(axis=0)
        Tbar = twirl.mean(axis=0)
        diff = F_exact - twirl
        se = diff.std(axis=0, ddof=1) / math.sqrt(diff.shape[0])
        z = np.where(se > 0, diff.mean(axis=0) / np.where(se > 0, se, 1), 0)
        # K picture: per-block probability of a non-identity logical class
        pI = (1.0 + full.sum(axis=2)) / 4.0
        eps_bar = float(1.0 - pI.mean())
        f_bar = float(f_full.mean())
        pI_last = (1.0 + last.sum(axis=1)) / 4.0
        # i.i.d. block model with the -1/3 rule and its alternatives
        c0_bar = float(c0.mean())
        iid = {}
        for label, factor in (("minus_third", -1.0 / 3.0), ("minus_one", -1.0),
                              ("zero", 0.0)):
            fb = 1.0 - (1.0 - factor) * eps_bar
            fl = 1.0 - (1.0 - factor) * float(1.0 - pI_last.mean())
            pred = np.array([tabs["c0_depth0"] if m == 0 else
                             c0_bar * fb ** (m - 1) * fl for m in depths])
            iid[label] = {"f_block": fb,
                          "max_rel_dev_vs_exact": float(np.max(np.abs(
                              pred / Fbar - 1.0))),
                          "pred_m40": float(pred[-1])}
        sel = (depths >= 2)
        r_exact = -ols_slope(depths[sel], np.log(Fbar[sel]))
        r_twirl = -ols_slope(depths[sel], np.log(Tbar[sel]))
        pt = {"probe_seconds": time.perf_counter() - t0,
              "max_abs_recon_minus_exact_per_sequence": max_abs,
              "F0_exact": float(Fbar[0]), "c0_depth0": tabs["c0_depth0"],
              "eps_bar_per_block": eps_bar, "f_bar_twirled_block": f_bar,
              "one_minus_4eps_over_3": 1.0 - 4.0 * eps_bar / 3.0,
              "r_exact_ols_2_40": r_exact, "r_twirl_pred_ols_2_40": r_twirl,
              "Fbar_exact": Fbar.tolist(), "Fbar_twirl_pred": Tbar.tolist(),
              "Fbar_no_twirl_axisX": bad_x.mean(axis=0).tolist(),
              "z_exact_minus_twirl": z.tolist(),
              "max_abs_z": float(np.max(np.abs(z))),
              "max_rel_dev_twirl_vs_exact": float(np.max(np.abs(Tbar / Fbar
                                                                 - 1.0))),
              "iid_models": iid}
        report["points"][f"{p:g}"] = pt
        print(f"p={p:g}: max |recon - exact| per sequence = {max_abs:.2e}; "
              f"F(0) exact {Fbar[0]:.6f} vs c0 {tabs['c0_depth0']:.6f}; "
              f"eps_bar = {eps_bar:.5f} ({eps_bar / p:.2f} p), "
              f"1-4eps/3 = {1 - 4 * eps_bar / 3:.6f}, twirl f = {f_bar:.6f}; "
              f"r exact {r_exact:.5f} vs twirl {r_twirl:.5f}; "
              f"max|z| exact-twirl {pt['max_abs_z']:.2f}; "
              f"iid rel dev: " + ", ".join(
                  f"{k} {v['max_rel_dev_vs_exact']:.3g}"
                  for k, v in iid.items()), flush=True)
    save("twirl_check.json", report)
    return 0


# ---------------------------------------------------------------------------
# validate: RawPolicyEvaluator on inverse_check_every circuits
# ---------------------------------------------------------------------------

def fwd_terminal_positions(lc: Q.LRBCircuit) -> list[int]:
    """Forward rounds 0..m-1 and the terminal round m (inverse rounds,
    labelled >= INVERSE_ROUND_OFFSET, are skipped)."""
    m = lc.depth
    pos = [k for k, lab in enumerate(lc.rounds)
           if lab < Q.INVERSE_ROUND_OFFSET]
    assert [lc.rounds[k] for k in pos] == list(range(m)) + [m]
    return pos


def cmd_validate(args) -> int:
    seq = E.clifford_sequences(3, 12, 9090)[1]
    report = {"brute": [], "sampled": []}
    for depth in (1, 2):
        lc = Q.build_lrb_circuit(seq, depth, 0.01, terminal="data5",
                                 inverse_check_every=1)
        dem = lc.circuit.detector_error_model()
        dec = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lc))
        ev = X.RawPolicyEvaluator(lc, dec)
        joint = M.brute_force_joint(dem)
        n = dem.num_detectors
        pats = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(bool)
        raw = lc.raw_syndromes(pats)
        for name, pos in (("fwd_terminal", fwd_terminal_positions(lc)),
                          ("all", list(range(lc.num_rounds)))):
            for final in (False, True):
                ok = ~raw[:, pos, :].any(axis=(1, 2))
                if final:
                    ok &= ~pats[:, 4 * lc.num_rounds:].any(axis=1)
                acc = joint[:2 ** n][ok].sum() + joint[2 ** n:][ok].sum()
                flip = joint[2 ** n:][ok].sum()
                a, f = ev.evaluate(pos, final)
                report["brute"].append({
                    "depth": depth, "rounds": lc.rounds, "policy": name,
                    "final": final, "acc_rel_err": abs(a / acc - 1),
                    "flip_rel_err": abs(f / flip - 1) if flip > 0 else
                    abs(f)})
                print(report["brute"][-1], flush=True)
    rng_seed = 31
    for depth, p, shots in ((4, 5e-3, 1_000_000), (8, 2e-3, 1_000_000)):
        lc = Q.build_lrb_circuit(seq, depth, p, terminal="data5",
                                 inverse_check_every=1)
        dem = lc.circuit.detector_error_model()
        ev = X.RawPolicyEvaluator(lc, M.TransferMatrixMLD(
            dem, M.lrb_detector_slots(lc)))
        sampler = lc.circuit.compile_detector_sampler(seed=rng_seed)
        dets, obs = sampler.sample(shots, separate_observables=True)
        raw = lc.raw_syndromes(dets)
        final_clean = ~dets[:, 4 * lc.num_rounds:].any(axis=1)
        for name, pos in (("fwd_terminal", fwd_terminal_positions(lc)),
                          ("all", list(range(lc.num_rounds)))):
            for final in (False, True):
                ok = ~raw[:, pos, :].any(axis=(1, 2))
                if final:
                    ok &= final_clean
                a, f = ev.evaluate(pos, final)
                n_acc, n_flip = int(ok.sum()), int((ok & obs[:, 0]).sum())
                z_acc = (n_acc - shots * a) / math.sqrt(shots * a * (1 - a))
                z_flip = (n_flip - shots * f) / math.sqrt(
                    max(shots * f * (1 - f), 1e-12))
                report["sampled"].append({
                    "depth": depth, "p": p, "shots": shots, "policy": name,
                    "final": final, "exact_acc": a, "sampled_acc":
                    n_acc / shots, "z_acc": z_acc, "exact_acc_flip": f,
                    "sampled_acc_flip": n_flip / shots, "z_flip": z_flip})
                print(report["sampled"][-1], flush=True)
    save("validate_inv1_evaluator.json", report)
    return 0


# ---------------------------------------------------------------------------
# exact: curves for fresh sequences
# ---------------------------------------------------------------------------

def my_unif_rounds(k: int, depth: int) -> list[int]:
    """The repo's unif=k rule, re-implemented: terminal round plus forward
    rounds j*k - 1 for j = 1..floor(m/k), index < m."""
    rounds = {depth}
    for j in range(1, depth // k + 1):
        if j * k - 1 < depth:
            rounds.add(j * k - 1)
    return sorted(rounds)


def _exact_task(task):
    kind, p, si, seq, depth = task
    out = {}
    if kind == "std":
        lc = Q.build_lrb_circuit(seq, depth, p, terminal="data5")
        dec = M.TransferMatrixMLD(lc.circuit.detector_error_model(),
                                  M.lrb_detector_slots(lc))
        ev = X.RawPolicyEvaluator(lc, dec)
        for k in (1, 2, 4):
            labels = my_unif_rounds(k, depth)
            assert labels == B.used_rounds("unif", k, depth)
            pos = [lc.rounds.index(r) for r in labels]
            out[f"unif{k}"] = ev.evaluate(pos, False)
            out[f"unif{k}_final"] = ev.evaluate(pos, True)
        pos = [lc.rounds.index(depth)]           # const=1: terminal only
        out["const1_final"] = ev.evaluate(pos, True)
        out["null"] = X.exact_condition(dec, [])
    else:
        lc = Q.build_lrb_circuit(seq, depth, p, terminal="data5",
                                 inverse_check_every=1)
        dec = M.TransferMatrixMLD(lc.circuit.detector_error_model(),
                                  M.lrb_detector_slots(lc))
        ev = X.RawPolicyEvaluator(lc, dec)
        pos = fwd_terminal_positions(lc)
        out["fwdterm"] = ev.evaluate(pos, False)
        out["fwdterm_final"] = ev.evaluate(pos, True)
        allr = list(range(lc.num_rounds))
        out["all_final"] = ev.evaluate(allr, True)
        out["null"] = X.exact_condition(dec, [])
    return kind, si, depth, out


def cmd_exact(args) -> int:
    p = float(args.p)
    depths = list(range(args.dmin, args.dmax + 1))
    seqs = E.clifford_sequences(args.num_seq, args.dmax, args.seed)
    tasks = [(kind, p, si, seq, d) for kind in ("std", "inv1")
             for si, seq in enumerate(seqs) for d in depths]
    tasks.sort(key=lambda t: -t[4])
    store: dict = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for kind, si, d, out in pool.map(_exact_task, tasks, chunksize=4):
            for key, (a, f) in out.items():
                name = f"{kind}:{key}"
                if name not in store:
                    store[name] = (np.full((len(seqs), len(depths)), np.nan),
                                   np.full((len(seqs), len(depths)), np.nan))
                store[name][0][si, depths.index(d)] = a
                store[name][1][si, depths.index(d)] = f
    arrays = {}
    for name, (a, f) in store.items():
        arrays[name.replace(":", "__") + "__accept"] = a
        arrays[name.replace(":", "__") + "__acc_flip"] = f
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"exact_fresh_p{p:g}.npz"
    np.savez_compressed(path, p=p, depths=np.array(depths), seed=args.seed,
                        num_seq=args.num_seq, **arrays)
    print(f"wrote {path.relative_to(ROOT)} in {time.perf_counter() - t0:.0f}s",
          flush=True)
    return 0


def _load_fresh(p: float):
    data = np.load(OUT / f"exact_fresh_p{p:g}.npz")
    depths = np.asarray(data["depths"])
    curves = {}
    for key in data.files:
        if key.endswith("__accept"):
            base = key[:-len("__accept")]
            a = data[key]
            f = data[base + "__acc_flip"]
            curves[base.replace("__", ":")] = (a, 1.0 - 2.0 * f / a)
    return depths, curves


# ---------------------------------------------------------------------------
# spacing (claim 12)
# ---------------------------------------------------------------------------

def cmd_spacing(args) -> int:
    p = float(args.p)
    depths, curves = _load_fresh(p)
    rng = np.random.default_rng(RNG_SEED + 1)
    lo, hi = 4, 24
    n_seq = curves["std:unif1"][1].shape[0]
    idx = rng.integers(0, n_seq, (NBOOT, n_seq))
    report = {"p": p, "n_seq": int(n_seq), "window": [lo, hi], "sets": {}}
    for suffix in ("_final", ""):
        rates, boots = {}, {}
        for k in (1, 2, 4):
            Fs = curves[f"std:unif{k}{suffix}"][1]
            A = curves[f"std:unif{k}{suffix}"][0]
            Fbar = Fs.mean(axis=0)
            Fb = Fs[idx].mean(axis=1)
            for scheme in ("lattice_k", "common_lattice_4", "all_depths",
                           "endpoints"):
                if scheme == "lattice_k":
                    sel = (depths >= lo) & (depths <= hi) & (depths % k == 0)
                elif scheme == "common_lattice_4":
                    sel = (depths >= lo) & (depths <= hi) & (depths % 4 == 0)
                elif scheme == "all_depths":
                    sel = (depths >= lo) & (depths <= hi)
                else:
                    sel = (depths == lo) | (depths == hi)
                r = -ols_slope(depths[sel], np.log(Fbar[sel]))
                rb = -ols_slope(depths[sel], np.log(Fb[:, sel]))
                rates[(k, scheme)] = r
                boots[(k, scheme)] = rb
            # acceptance per block on the lattice
            sel = (depths >= lo) & (depths <= hi) & (depths % k == 0)
            rates[(k, "accept_per_block")] = math.exp(
                ols_slope(depths[sel], np.log(A.mean(axis=0)[sel])))
        out = {}
        for scheme in ("lattice_k", "common_lattice_4", "all_depths",
                       "endpoints"):
            row = {}
            for k in (1, 2, 4):
                row[f"r{k}"] = rates[(k, scheme)]
                row[f"r{k}_ci"] = list(pct(boots[(k, scheme)]))
            for k in (2, 4):
                ratio = boots[(k, scheme)] / boots[(1, scheme)]
                row[f"r{k}/r1"] = rates[(k, scheme)] / rates[(1, scheme)]
                row[f"r{k}/r1_ci"] = list(pct(ratio))
            out[scheme] = row
        out["accept_per_block"] = {f"unif{k}": rates[(k, "accept_per_block")]
                                   for k in (1, 2, 4)}
        report["sets"]["final" if suffix else "data2_semantics"] = out
        print(f"\nunif=k{suffix or ' (no final check)'} at p={p:g}, "
              f"{n_seq} fresh sequences, depths {lo}..{hi}:")
        for scheme, row in out.items():
            if scheme == "accept_per_block":
                print(f"  acceptance per block: {row}")
                continue
            print(f"  {scheme:17s} r1={row['r1']:.3e} r2={row['r2']:.3e} "
                  f"r4={row['r4']:.3e}  r2/r1={row['r2/r1']:.3f} "
                  f"[{row['r2/r1_ci'][0]:.3f}, {row['r2/r1_ci'][1]:.3f}]  "
                  f"r4/r1={row['r4/r1']:.3f} [{row['r4/r1_ci'][0]:.3f}, "
                  f"{row['r4/r1_ci'][1]:.3f}]")
    save(f"spacing_p{p:g}.json", report)
    return 0


# ---------------------------------------------------------------------------
# invunused (claim 16/17)
# ---------------------------------------------------------------------------

def cmd_invunused(args) -> int:
    p = float(args.p)
    depths, curves = _load_fresh(p)
    rng = np.random.default_rng(RNG_SEED + 2)
    lo, hi = 2, 30
    report = {"p": p, "window": [lo, hi], "curves": {}}
    names = ["inv1:fwdterm_final", "inv1:fwdterm", "inv1:all_final",
             "inv1:null", "std:unif1_final", "std:unif1", "std:const1_final",
             "std:null"]
    for name in names:
        A, Fs = curves[name]
        st, _ = curve_stats(depths, Fs, lo, hi, rng)
        Fbar = Fs.mean(axis=0)
        logF = np.log(Fbar)
        # local rates over windows of 4 blocks
        local = []
        for m1 in range(lo, hi - 3, 4):
            m2 = m1 + 4
            i, j = list(depths).index(m1), list(depths).index(m2)
            local.append([m1, m2, float(-(logF[j] - logF[i]) / 4)])
        # Markov surrogate from short depths [2, 8]
        s = (depths >= 2) & (depths <= 8)
        slope = ols_slope(depths[s], logF[s])
        icpt = float(np.mean(logF[s]) - slope * np.mean(depths[s]))
        delta30 = float(logF[list(depths).index(30)] - (icpt + slope * 30))
        st.update({"F_m2": float(Fbar[list(depths).index(2)]),
                   "F_m30": float(Fbar[list(depths).index(30)]),
                   "A_m30": float(A.mean(axis=0)[list(depths).index(30)]),
                   "local_rates_w4": local,
                   "delta_logF30_vs_short_markov": delta30,
                   "late_over_early_local": local[-1][2] / local[0][2]})
        report["curves"][name] = st
        print(f"{name:20s} r={st['r']:.3e} c2={st['c2']:+.2e} "
              f"[{st['c2_ci'][0]:+.1e}, {st['c2_ci'][1]:+.1e}] rho="
              f"{st['rho']:+.3f} [{st['rho_ci'][0]:+.3f}, "
              f"{st['rho_ci'][1]:+.3f}] local r(2-6)={local[0][2]:.2e} "
              f"r(26-30)={local[-1][2]:.2e} ratio "
              f"{st['late_over_early_local']:.2f} Delta30={delta30:+.4f} "
              f"F30={st['F_m30']:.5f} A30={st['A_m30']:.3g}", flush=True)
    save(f"invunused_p{p:g}.json", report)
    return 0


# ---------------------------------------------------------------------------
# mlddrift (claim 7 spot check)
# ---------------------------------------------------------------------------

def cmd_mlddrift(args) -> int:
    rng = np.random.default_rng(RNG_SEED + 3)
    report = {}
    for tag in ("p5e-3", "p2e-3", "p5e-4"):
        pr = E.PointResult.from_npz(str(RES_F / f"std_{tag}.npz"))
        depths = np.asarray(pr.depths, float)
        res = {}
        for name in ("mld5", "raw"):
            t = pr.tables[name]
            Fs = _fs_from_counts(t.accepted, t.errors)
            n_seq = Fs.shape[0]
            # (a) OLS thirds on [2, 30]
            st, bt = curve_stats(depths, Fs, 2.0, 30.0, rng)
            # (b) two-point local rates over the first and last 9 blocks
            Fbar = Fs.mean(axis=0)
            idx = rng.integers(0, n_seq, (NBOOT, n_seq))
            Fb = Fs[idx].mean(axis=1)

            def two_point(F):
                lF = np.log(np.where(F > 0, F, np.nan))
                e = -(lF[..., 11] - lF[..., 2]) / 9.0
                l_ = -(lF[..., 30] - lF[..., 21]) / 9.0
                a = -(lF[..., 30] - lF[..., 2]) / 28.0
                return (l_ - e) / a
            tp = float(two_point(Fbar))
            tpb = two_point(Fb)
            # (c) shot noise only: parametric binomial resampling of the
            # same 30 sequences (removes sequence-sampling noise)
            shots = t.shots.astype(float)
            pflip = t.errors / shots
            sims = []
            for _ in range(1000):
                e = rng.binomial(t.shots, pflip)
                Fsim = (1.0 - 2.0 * e / shots).mean(axis=0)
                sims.append(drift(depths, np.log(Fsim), 2.0, 30.0)[3])
            sims = np.asarray(sims)
            res[name] = {"ols_thirds": st, "two_point_rho": tp,
                         "two_point_rho_ci": list(pct(tpb)),
                         "shot_noise_only_rho_sd": float(np.std(sims)),
                         "F_m30": float(Fbar[30])}
            print(f"{tag} {name:5s}: OLS-thirds rho={st['rho']:+.4f} "
                  f"[{st['rho_ci'][0]:+.4f}, {st['rho_ci'][1]:+.4f}] r="
                  f"{st['r']:.4f} c2={st['c2']:+.2e} [{st['c2_ci'][0]:+.1e}, "
                  f"{st['c2_ci'][1]:+.1e}]; two-point rho={tp:+.4f} "
                  f"[{pct(tpb)[0]:+.4f}, {pct(tpb)[1]:+.4f}]; shot-noise-only "
                  f"sd(rho)={np.std(sims):.4f}; F(30)={Fbar[30]:.4g}",
                  flush=True)
        # paired MLD - null on the same 30 sequences, using the LEAD's exact
        # null file (first 30 of 200 sequences; same per-index RNG streams)
        s30 = E.clifford_sequences(30, 30, 20261028)
        s200 = E.clifford_sequences(30, 40, 20261028)
        assert all(a == b[:30] for a, b in zip(s30, s200))
        pval = {"p5e-3": 5e-3, "p2e-3": 2e-3, "p5e-4": 5e-4}[tag]
        lead = np.load(DOC / "results" / "exact_null" /
                       f"exact_null_p{pval:g}.npz")
        Fn = 1.0 - 2.0 * lead["flip_checks_data5"][:30, :31]
        t = pr.tables["mld5"]
        Fm = _fs_from_counts(t.accepted, t.errors)
        idx = rng.integers(0, Fm.shape[0], (NBOOT, Fm.shape[0]))
        rho_m = drift(depths, np.log(Fm[idx].mean(axis=1)), 2.0, 30.0)[3]
        rho_n = drift(depths, np.log(Fn[idx].mean(axis=1)), 2.0, 30.0)[3]
        d0 = drift(depths, np.log(Fm.mean(axis=0)), 2.0, 30.0)[3] - drift(
            depths, np.log(Fn.mean(axis=0)), 2.0, 30.0)[3]
        res["paired_mld_minus_exact_null"] = {"diff": float(d0),
                                              "ci": list(pct(rho_m - rho_n))}
        # exact null on 200 sequences (lead), same window
        p = {"p5e-3": 5e-3, "p2e-3": 2e-3, "p5e-4": 5e-4}[tag]
        n200 = np.load(DOC / "results" / "exact_null" /
                       f"exact_null_p{p:g}.npz")
        F200 = 1.0 - 2.0 * n200["flip_checks_data5"][:, :31]
        st200, _ = curve_stats(np.arange(31.0), F200, 2.0, 30.0, rng)
        res["null200_ols_thirds"] = st200
        print(f"{tag}: paired rho(MLD) - rho(exact null, same 30 seq) = "
              f"{d0:+.4f} [{pct(rho_m - rho_n)[0]:+.4f}, "
              f"{pct(rho_m - rho_n)[1]:+.4f}]; exact null 200 seq rho = "
              f"{st200['rho']:+.4f} [{st200['rho_ci'][0]:+.4f}, "
              f"{st200['rho_ci'][1]:+.4f}]", flush=True)
        report[tag] = res
    save("mld_drift.json", report)
    return 0


# ---------------------------------------------------------------------------
# misc spot checks (claims 5, 8, 9, 10)
# ---------------------------------------------------------------------------

def cmd_misc(args) -> int:
    rng = np.random.default_rng(RNG_SEED + 4)
    report = {"tanh_floor": {}, "exact_rho": {}, "first_order": {}}
    # claim 5: |LLR| >= t floor from the gap histograms (signed F per seq)
    for tag in ("p5e-4", "p2e-3", "p5e-3"):
        pr = E.PointResult.from_npz(str(RES_F / f"std_{tag}.npz"))
        hist = pr.gap_hist["mld5"]                # (seq, depth, bins, 2)
        rows = {}
        for t in (0.5, 1.0, 2.0, 4.0):
            i = int(np.searchsorted(E.GAP_EDGES, t))
            acc = hist[:, :, i:, :].sum(axis=(2, 3)).astype(float)
            err = hist[:, :, i:, 1].sum(axis=2)
            ok = (acc >= 100).all(axis=0)
            Fs = np.where(acc > 0, 1.0 - 2.0 * err / np.maximum(acc, 1), np.nan)
            Fbar = Fs.mean(axis=0)
            # per-cell lower bound check with pooled shots (all sequences)
            pooled = 1.0 - 2.0 * err.sum(axis=0) / acc.sum(axis=0)
            rows[f"{t:g}"] = {"floor": math.tanh(t / 2),
                              "min_Fbar_usable": float(np.nanmin(Fbar[ok]))
                              if ok.any() else None,
                              "min_pooled_usable": float(np.nanmin(pooled[ok]))
                              if ok.any() else None,
                              "n_usable_depths": int(ok.sum())}
        report["tanh_floor"][tag] = rows
        print(tag, "tanh floor:", {k: (round(v['floor'], 3),
                                        v['min_Fbar_usable'])
                                    for k, v in rows.items()}, flush=True)
    # claims 9, 10: drift of the exact LRB-D curves with this script's OLS
    for tag in ("p5e-4", "p2e-3", "p5e-3"):
        x = np.load(RES_F / f"exact_std_{tag}.npz")
        depths = np.asarray(x["depths"], float)
        rows = {}
        for m in ("lrbd_unif_1_final", "lrbd_const_1_final",
                  "lrbd_const_4_final", "null"):
            Fs = 1.0 - 2.0 * x[f"{m}__acc_flip"] / x[f"{m}__accept"]
            st, _ = curve_stats(depths, Fs, 2.0, 30.0, rng, 2000)
            rows[m] = {k: st[k] for k in ("r", "rho", "rho_ci", "c2",
                                           "c2_ci")}
        report["exact_rho"][tag] = rows
        print(tag, {m: (f"{v['r']:.3g}", f"{v['rho']:+.3f}",
                        [round(z, 3) for z in v['rho_ci']])
                    for m, v in rows.items()}, flush=True)
    # const=0 is sampled only in WP-F: OLS drift from the sampled counts
    for tag in ("p5e-4", "p2e-3", "p5e-3"):
        pr = E.PointResult.from_npz(str(RES_F / f"std_{tag}.npz"))
        t = pr.tables["const0"]
        Fs = _fs_from_counts(t.accepted, t.errors)
        st, _ = curve_stats(np.asarray(pr.depths, float), Fs, 2.0, 30.0, rng,
                            2000)
        report["exact_rho"].setdefault(tag, {})["const0_sampled"] = {
            k: st[k] for k in ("r", "rho", "rho_ci")}
        print(tag, "const0 sampled:", f"r={st['r']:.3g} rho={st['rho']:+.3f}",
              [round(z, 3) for z in st["rho_ci"]], flush=True)
    # claim 8: first-order MLD weight, 150 sequences
    lo = np.load(RES_F / "loworder_std_p5e-4_n150.npz")
    d = np.asarray(lo["depths"], float)
    for key in ("mld5__S1", "mld5__S1_late", "raw__S1"):
        y = lo[key] / float(lo["p"])
        sel = d >= 2
        Y = y[:, sel]
        idx = rng.integers(0, Y.shape[0], (NBOOT, Y.shape[0]))
        Yb = Y[idx].mean(axis=1)
        q = ols_quad(d[sel], Y.mean(axis=0))
        qb = ols_quad(d[sel], Yb)
        s = ols_slope(d[sel], Y.mean(axis=0))
        report["first_order"][key] = {"slope": s, "quad": q,
                                      "quad_ci": list(pct(qb))}
        print(f"{key}: slope {s:.3f} p/block, quad {q:+.4f} "
              f"[{pct(qb)[0]:+.4f}, {pct(qb)[1]:+.4f}]", flush=True)
    save("misc_checks.json", report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cmd", choices=["abs", "twirl", "validate", "exact",
                                        "spacing", "invunused", "mlddrift",
                                        "misc"])
    parser.add_argument("--p", default=None)
    parser.add_argument("--num-seq", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5150000)
    parser.add_argument("--dmin", type=int, default=2)
    parser.add_argument("--dmax", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.p is None:
        args.p = "5e-4,5e-3" if args.cmd == "twirl" else "1e-3"
    return {"abs": cmd_abs, "twirl": cmd_twirl, "validate": cmd_validate,
            "exact": cmd_exact, "spacing": cmd_spacing,
            "invunused": cmd_invunused, "mlddrift": cmd_mlddrift,
            "misc": cmd_misc}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
