"""H2 (other platforms): LRB-D and maximum-likelihood correction of the baseline
circuits under platform-specific noise models.

Run:  module load python/3.11 && python3 h2_platform_noise_study.py [--seqs 6] [--shots 4000] [--workers 2]
Runtime: about 3-5 min wall on two cores.

Read-only inputs
  <worktree>/.cache_correction/structures/c<cc>_d<i>.pkl  chain structures cached by the
      decoder study (compiled from the baseline circuits at p = 1e-3, so every noise
      location of the SI1000 circuit is present with its exact detector/logical response)
  <run>/experiments/LRB/<c>/4/<i>.chp  the same circuits, read only to label each noise
      location by the operation it belongs to
  <run>/results/LRB/4/{unif,const}_check_data/1.csv  baseline LRB-D at p = 1e-3 (validation)
  <worktree>/.cache_correction/results_baseline/c<cc>_p0p001.npz  the decoder study's MLD
      results at p = 1e-3 (cross-check of the sampled MLD path)
Writes h2_platform_noise_study.csv next to this script (one row per model, sequence, depth).

Method
  A noise model gives every location of the compact DEM an activation probability and a Weyl
  distribution (uniform over the 8 one-qutrit or 80 two-qutrit non-identity Weyl operators, or
  Z-only dephasing for idles). Locations keep their exact responses. The exact chain machinery of
  src/lrb/correction (forward recursion over 81 x 3 states) then gives, per circuit:
    exact LRB-D acceptance and accepted-shot fidelity for unif 1 and const 1;
    exact raw fidelity (no correction, no rejection);
    sampled MLD correction (all rounds; terminal round only), scored with the Rao-Blackwellized
    estimator used by study.evaluate_chain (expected residual from the exact posterior).
  Signed fidelity lambda = Re E[w^R] = (3 S - 1)/2 with S = P(residual = 0); the legacy metric
  |E[w^R]| is printed for comparison with the baseline CSVs.

Noise models (the inputs are literature values or my assumptions; see the comments in MODELS)
  si1000_1e-3   the brief's SI1000 at p = 1e-3 (validation and reference)
  si1000_2e-3, si1000_5e-3   the same at p = 2e-3 and 5e-3 (to read off an equivalent SI1000 p)
  ion_now       40Ca+ qutrits as published: LS-gate CNOT/CZ, Clifford-RB single-qutrit gates,
                parallel camera readout, measured mid-circuit spectator error, SWAP(2,3) and
                logical X, Z done in software
  ion_slowread  ion_now with the data dephasing expected from a 4.5 ms qutrit readout at T2 = 100 ms
  ion_next      the same with the improvements the ion papers project (conjecture)
  qubit_pairs   qutrits emulated by qubit pairs on an all-to-all QCCD processor, with the gate
                counts of Iqbal et al. 2025 and 2e-3 per two-qubit gate
  transmon_opt  transmon qutrits with the best published gates and readout, no idle noise at all
                and native CZ(2,3) (deliberately optimistic; see H1 for a full transmon budget)
  ion_worstidle ion_slowread plus exponential dephasing during ~5 ms of sequential gates per round
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(WORKTREE, "src"))
RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
       "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
CACHE = os.path.join(WORKTREE, ".cache_correction")
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
USE_DI = [0, 1, 2, 3, 4]                       # m = 0, 2, 4, 6, 10
W3 = np.exp(2j * np.pi * np.arange(3) / 3)

# --------------------------------------------------------------------------- models
# class -> (activation probability, Weyl distribution: "u" uniform, "z" Z-only dephasing)
def si1000(p):
    """The brief's SI1000 model (sequential single-ancilla readout layers)."""
    return {
        "2q_check": (p, "u"), "2q_SWAP": (p, "u"), "2q_CZ": (p, "u"),
        "1q_check": (0.1 * p, "u"), "1q_H": (0.1 * p, "u"), "1q_P": (0.1 * p, "u"),
        "1q_X": (0.1 * p, "u"), "1q_Z": (0.1 * p, "u"), "1q_I": (0.1 * p, "u"),
        "idle_meas_data_last": (2 * p, "u"), "idle_meas_data_other": (2 * p, "u"),
        "idle_meas_anc": (2 * p, "u"), "idle_reset_data": (2 * p, "u"),
        "anc_depol": (p, "u"), "meas_flip": (5 * p, "u"), "reset_flip": (2 * p, "u"),
    }


SI1000 = si1000(1e-3)


def platform(two_check, two_cz, one_h, one_p, one_check, meas, spect, reset, swap=0.0, one_x=0.0, one_z=0.0):
    """One parallel readout layer per round: data idle only in the last readout layer of a round."""
    return {
        "2q_check": (two_check, "u"), "2q_SWAP": (swap, "u"), "2q_CZ": (two_cz, "u"),
        "1q_check": (one_check, "u"), "1q_H": (one_h, "u"), "1q_P": (one_p, "u"),
        "1q_X": (one_x, "u"), "1q_Z": (one_z, "u"), "1q_I": (0.0, "u"),
        "idle_meas_data_last": (spect, "z"), "idle_meas_data_other": (0.0, "z"),
        "idle_meas_anc": (0.0, "u"), "idle_reset_data": (0.0, "z"),
        "anc_depol": (0.0, "u"), "meas_flip": (meas, "u"), "reset_flip": (reset, "u"),
    }


def depol1(avg_infidelity):      # one-qutrit depolarizing: F_avg = 1 - 3q/4
    return 4.0 * avg_infidelity / 3.0


def depol2(avg_infidelity):      # two-qutrit depolarizing: F_avg = 1 - 9q/10 (also the pure-state fidelity)
    return 10.0 * avg_infidelity / 9.0


MODELS = {
    "si1000_1e-3": SI1000,
    "si1000_2e-3": si1000(2e-3),
    "si1000_5e-3": si1000(5e-3),
    # Hrmo et al. 2023: qutrit LS gate 98.7(2)% (SPAM-corrected per-gate decay); Ringbauer et al.
    # 2022: qutrit Clifford RB 2(2)e-3 per Clifford, worst-case qutrit readout error 3e-3;
    # Yu et al. 2025: data-qubit error 1.2-1.8% per in-situ mid-circuit measurement (shelving),
    # taken as Z dephasing of each data qutrit once per round; reset error 5e-3 (assumption,
    # between the 0.9(3)% hands-off reset of Yu et al. and the readout error).
    "ion_now": platform(two_check=depol2(0.013), two_cz=depol2(0.013), one_h=depol1(2e-3), one_p=depol1(2e-3),
                        one_check=depol1(2e-3), meas=3e-3, spect=0.015, reset=5e-3),
    # Sensitivity: the same, but the data qutrits dephase during a 4.5 ms qutrit readout (Meth et al.
    # 2025) with T2 = 100 ms and exponential decay: q = (2/3)(1 - exp(-0.045)) = 0.029 (conjecture).
    "ion_slowread": platform(two_check=depol2(0.013), two_cz=depol2(0.013), one_h=depol1(2e-3), one_p=depol1(2e-3),
                             one_check=depol1(2e-3), meas=3e-3, spect=0.029, reset=5e-3),
    # Conjecture: technical noise removed from the Hrmo d = 3 budget (about 5e-3 left), local
    # pulses 4x better, readout 3x better (Ringbauer: "an order of magnitude" possible),
    # spectator error 1e-3 (Yu et al. projection "< 0.1%"), reset 1e-3.
    "ion_next": platform(two_check=depol2(0.005), two_cz=depol2(0.005), one_h=depol1(5e-4), one_p=depol1(5e-4),
                         one_check=depol1(5e-4), meas=1e-3, spect=1e-3, reset=1e-3),
    # Iqbal et al. 2025 on Quantinuum H2 (>99.8% two-qubit gates): qutrit Fourier = 3 ZZPhase,
    # controlled-Z = 4 controlled-RZ; check CNOT = F^dag CZ F = 10 two-qubit gates -> 0.02;
    # logical CZ(2,3) = 4 -> 8e-3; Fourier-type 1q gates 6e-3, phase-type 2e-3 (1 ZZPhase);
    # readout 2 qubits -> 2e-3; spectator 2e-3 and reset 1e-3 per round are assumptions.
    "qubit_pairs": platform(two_check=0.02, two_cz=8e-3, one_h=6e-3, one_p=2e-3, one_check=6e-3,
                            meas=2e-3, spect=2e-3, reset=1e-3),
    # Goss et al. 2022: CZ^dag process fidelity 97.3(1)% (p = 1 - F_pro = 0.027); Morvan et al.
    # 2021: single-qutrit infidelity 2.38e-3; QubiC 2.0: 97% qutrit classification -> 0.03;
    # no idle noise anywhere, native CZ(2,3), reset 0.01 (assumptions, optimistic).
    "transmon_opt": platform(two_check=0.027, two_cz=0.027, one_h=depol1(2.38e-3), one_p=depol1(2.38e-3),
                             one_check=depol1(2.38e-3), meas=0.03, spect=0.0, reset=0.01),
    # Worst-case sensitivity (appended last so the other models keep their random streams): ion_slowread
    # plus exponential dephasing of each data qutrit during the ~5 ms of sequential gates per round,
    # (2/3)(1 - exp(-0.05)) = 0.033, lumped into the once-per-round spectator location (conjecture).
    "ion_worstidle": platform(two_check=depol2(0.013), two_cz=depol2(0.013), one_h=depol1(2e-3), one_p=depol1(2e-3),
                              one_check=depol1(2e-3), meas=3e-3, spect=0.062, reset=5e-3),
}
MODEL_NAMES = list(MODELS)


# --------------------------------------------------------------------------- labelling
def label_locations(circuit_path, mechanisms):
    """Label each mechanism by the operation it belongs to (circuit order = model order)."""
    from lrb.correction.compact_dem import read_run_circuit as read_circuit
    ops = read_circuit(circuit_path).operations
    noise = []

    def previous_gate(i):
        j = i - 1
        while ops[j].gate_name in ("N1", "N2"):
            j -= 1
        return ops[j]

    for i, op in enumerate(ops):
        if op.gate_name not in ("N1", "N2"):
            continue
        pr = op.params or {}
        scale = float(pr.get("prob_scale", "nan"))
        chan = pr.get("noise_channel", pr.get("channel", "d"))
        if op.gate_name == "N2":
            prev = previous_gate(i)
            pair = {int(op.qudit_index), int(op.target_index)}
            cls = "2q_check" if pair & {5, 6, 7, 8} else ("2q_SWAP" if prev.gate_name == "SWAP" else "2q_CZ")
            if cls == "2q_CZ" and not prev.gate_name.startswith("CZ"):
                raise ValueError(f"unexpected logical two-qutrit gate {prev.gate_name}")
        elif chan == "d" and scale == 0.1:
            prev = previous_gate(i)
            q = int(op.qudit_index)
            cls = "1q_check" if q >= 5 else "1q_" + prev.gate_name.split("_")[0]
        elif chan == "d" and scale == 2.0:
            j = i + 1
            while ops[j].gate_name not in ("M", "RESET"):
                j += 1
            q = int(op.qudit_index)
            if ops[j].gate_name == "RESET":
                cls = "idle_reset_data"
            elif q >= 5:
                cls = "idle_meas_anc"
            else:
                cls = "idle_meas_data_last" if int(ops[j].qudit_index) == 8 else "idle_meas_data_other"
        elif chan == "d" and scale == 1.0:
            cls = "anc_depol"
        elif chan == "f" and scale == 5.0:
            cls = "meas_flip"
        elif chan == "f" and scale == 2.0:
            cls = "reset_flip"
        else:
            raise ValueError(f"unclassified noise op {op.gate_name} {chan} {scale}")
        noise.append((op.gate_name, int(op.qudit_index),
                      -1 if op.target_index is None else int(op.target_index), chan, scale, cls))
    if len(noise) != len(mechanisms):
        raise ValueError(f"{circuit_path}: {len(noise)} noise ops vs {len(mechanisms)} mechanisms")
    for (g, q, t, ch, sc, _), mech in zip(noise, mechanisms):
        if (g, q, t, ch, sc) != (mech.gate, mech.qudit, mech.target, mech.channel, mech.prob_scale):
            raise ValueError(f"{circuit_path}: noise op / mechanism mismatch {(g, q, t, ch, sc)}")
    return [n[-1] for n in noise]


def chain_for_model(structure, labels, model):
    """Exact chain model with per-location probabilities and Weyl distributions."""
    from lrb.correction import chain_dem
    mechs, ab, act = [], [], np.zeros(len(structure.mechanisms))
    for i, (mech, lab) in enumerate(zip(structure.mechanisms, labels)):
        q, kind = model[lab]
        a, b = structure.ab[i]
        if kind == "z" and mech.gate == "N1" and mech.channel == "d":
            rows = np.flatnonzero(mech.vectors[:, 0] == 0)          # Z^1, Z^2 only
            mech = dataclasses.replace(mech, vectors=mech.vectors[rows],
                                       detector_effects=mech.detector_effects[rows],
                                       logical_effects=mech.logical_effects[rows])
            a, b = a[rows], b[rows]
        mechs.append(mech)
        ab.append((a, b))
        act[i] = q
    final_b = 0 if structure.layout.terminal_index is None else chain_dem.TERMINAL_WIDTH
    kernels = chain_dem.build_kernels(mechs, structure.slots, ab, act, structure.layout.num_rounds, final_b)
    return chain_dem.ChainModel(p=1.0, layout=structure.layout, transfer=structure.transfer,
                                transfer_perm=structure.transfer_perm, kernels=kernels,
                                slot_of_mechanism=structure.slots, mechanisms=mechs, activation=act,
                                diagnostics=structure.diagnostics, terminal_perm=structure.terminal_perm)


def metrics_from_joint(joint):
    joint = np.asarray(joint, dtype=float)
    acc = float(joint.sum())
    if acc <= 0:
        return acc, np.nan, np.nan
    z = (joint / acc) @ W3
    return acc, float(z.real), float(abs(z))


def run_sequence(seq, shots, seed, check_validation):
    from lrb.correction import decoders, forward, sampler, study
    rows = []
    info = {}
    for di in USE_DI:
        m = DEPTHS[di]
        prepared = study.load_prepared(os.path.join(CACHE, "structures", f"c{seq:02d}_d{di}.pkl"))
        st = prepared.structure
        labels = label_locations(os.path.join(RUN, "experiments", "LRB", str(seq), "4", f"{di}.chp"), st.mechanisms)
        if check_validation and di == 1:
            ref = st.at(1e-3)
            mine = chain_for_model(st, labels, MODELS["si1000_1e-3"])
            info["kernel_maxdiff"] = max(float(np.max(np.abs(x - y))) for x, y in zip(ref.kernels, mine.kernels))
        weights = {}
        for lab, mech in zip(labels, st.mechanisms):
            weights[lab] = weights.get(lab, 0.0) + mech.prob_scale
        info[f"weights_m{m}"] = weights
        for mi, name in enumerate(MODEL_NAMES):
            chain = chain_for_model(st, labels, MODELS[name])
            cache = decoders._SlotMatrices(chain)
            u1 = metrics_from_joint(decoders.lrbd_exact(chain, decoders.unif_mask(m, 1), cache))
            c1 = metrics_from_joint(decoders.lrbd_exact(chain, decoders.const_mask(m, 1), cache))
            raw = metrics_from_joint(forward.constrained_joint(chain, [None] * chain.num_rounds))
            rng = np.random.default_rng([seed, seq, di, mi])
            det, logical = sampler.sample_shots(chain.mechanisms, chain.activation, shots, rng)
            syn = sampler.syndrome_indices(det, chain.layout.detector_index)
            post, _ = forward.forward_posteriors_shared(chain, syn)
            idx = np.arange(shots)
            l_mld = post.argmax(axis=1)
            s_mld = float(post[idx, l_mld].mean())
            s_mld_sd = float(post[idx, l_mld].std(ddof=1) / np.sqrt(shots))
            term_mask = decoders.const_mask(m, 1)
            post_t, _ = forward.forward_posteriors_shared(chain, syn, term_mask)
            l_t = post_t.argmax(axis=1)
            s_term = float(post[idx, l_t].mean())
            n_round = sum(chain.activation[i] for i, lab in enumerate(labels))
            rows.append(dict(model=name, seq=seq, m=m, acc_u1=u1[0], lam_u1=u1[1], leg_u1=u1[2],
                             acc_c1=c1[0], lam_c1=c1[1], leg_c1=c1[2], lam_raw=raw[1],
                             lam_mld=(3 * s_mld - 1) / 2, lam_mld_se=1.5 * s_mld_sd,
                             lam_mld_term=(3 * s_term - 1) / 2, faults_total=float(n_round)))
    return rows, info


def fit_exp(ms, ys, floor=0.03):
    ms = np.asarray(ms, float); ys = np.asarray(ys, float)
    ok = np.isfinite(ys) & (ys > floor)
    if ok.sum() < 2:
        return np.nan, np.nan
    slope, icpt = np.polyfit(ms[ok], np.log(ys[ok]), 1)
    return float(np.exp(slope)), float(np.exp(icpt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=6)
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261028)
    args = ap.parse_args()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_sequence, s, args.shots, args.seed, s == 0) for s in range(args.seqs)]
        results = [f.result() for f in futs]
    rows = [r for res, _ in results for r in res]
    info0 = results[0][1]
    out_csv = os.path.join(HERE, "h2_platform_noise_study.csv")
    with open(out_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"sequences {args.seqs}, depths {[DEPTHS[i] for i in USE_DI]}, MLD shots {args.shots}, "
          f"wall {time.time() - t0:.0f} s; wrote {out_csv}")

    print("=" * 100)
    print("V1  validation of the relabelled-model path")
    print(f"  max |kernel difference| between structure.at(1e-3) and the SI1000 relabelled model (seq 0, m = 2): "
          f"{info0['kernel_maxdiff']:.2e}")
    for m in (0, 2, 10):
        wts = info0.get(f"weights_m{m}")
        if wts:
            rounds = m + 1
            per = {k: round(v / rounds, 2) for k, v in sorted(wts.items())}
            print(f"  m = {m}: SI1000 weight per round (sum of prob_scale / (m+1)) by class: {per}")

    def csv_row(policy):
        with open(os.path.join(RUN, "results", "LRB", "4", f"{policy}_check_data", "1.csv")) as fh:
            r = list(csv.reader(fh))
        return [float(x) for x in r[1][1:]], [float(x) for x in r[3][1:]]

    fu, ru = csv_row("unif"); fc, rc = csv_row("const")
    print("  SI1000 p = 1e-3, mean over my sequences vs baseline CSV (30 sequences, 1e6 shots):")
    for di in USE_DI:
        m = DEPTHS[di]
        sel = [r for r in rows if r["model"] == "si1000_1e-3" and r["m"] == m]
        print(f"    m = {m:2d}: unif1 rejection {1 - np.mean([r['acc_u1'] for r in sel]):.4f} (CSV {ru[di]:.4f}), "
              f"legacy F {np.mean([r['leg_u1'] for r in sel]):.5f} (CSV {fu[di]:.5f}); const1 rejection "
              f"{1 - np.mean([r['acc_c1'] for r in sel]):.4f} (CSV {rc[di]:.4f}), legacy F "
              f"{np.mean([r['leg_c1'] for r in sel]):.5f} (CSV {fc[di]:.5f})")
    print("  MLD (all rounds) at p = 1e-3 versus the decoder study's saved results for the same circuits:")
    for s in range(min(args.seqs, 3)):
        z = np.load(os.path.join(CACHE, "results_baseline", f"c{s:02d}_p0p001.npz"))
        for di in (1, 3, 4):
            rb = z[f"d{di}__mld_all_rb"]; theirs = (3 * rb[0] / rb.sum() - 1) / 2
            mine = [r for r in rows if r["model"] == "si1000_1e-3" and r["seq"] == s and r["m"] == DEPTHS[di]][0]
            print(f"    seq {s} m = {DEPTHS[di]:2d}: mine {mine['lam_mld']:.5f} +- {mine['lam_mld_se']:.5f}, "
                  f"decoder study {theirs:.5f} ({int(z[f'd{di}__shots'])} shots)")

    print("=" * 100)
    print("R1  per-depth means over sequences (lam = signed fidelity Re E[w^R]; acc = acceptance)")
    summary = {}
    for name in MODEL_NAMES:
        print(f"  model {name}")
        print(f"    {'m':>3s} {'faults':>7s} {'acc_u1':>9s} {'lam_u1':>8s} {'acc_c1':>8s} {'lam_c1':>8s} "
              f"{'lam_raw':>8s} {'lam_mld':>8s} {'lam_mld_term':>12s}")
        agg = {}
        for di in USE_DI:
            m = DEPTHS[di]
            sel = [r for r in rows if r["model"] == name and r["m"] == m]
            agg[m] = {k: float(np.nanmean([r[k] for r in sel])) for k in
                      ("faults_total", "acc_u1", "lam_u1", "acc_c1", "lam_c1", "lam_raw", "lam_mld", "lam_mld_term")}
            a = agg[m]
            print(f"    {m:3d} {a['faults_total']:7.3f} {a['acc_u1']:9.3e} {a['lam_u1']:8.5f} {a['acc_c1']:8.4f} "
                  f"{a['lam_c1']:8.5f} {a['lam_raw']:8.5f} {a['lam_mld']:8.5f} {a['lam_mld_term']:12.5f}")
        summary[name] = agg

    print("=" * 100)
    print("R2  fitted decays lam(m) = B f^m (log-linear fit over depths with lam > 0.03); r = 2(1 - f)/3 per")
    print("    depth step (one forward logical Clifford with its check round plus one inverse logical Clifford,")
    print("    the brief's convention); acceptance per depth step a")
    for name in MODEL_NAMES:
        agg = summary[name]
        ms = sorted(agg)
        out = []
        for key in ("lam_mld", "lam_mld_term", "lam_raw", "lam_u1", "lam_c1"):
            f, B = fit_exp(ms, [agg[m][key] for m in ms])
            out.append(f"{key} r = {2 * (1 - f) / 3:.3g} (B {B:.3f})" if np.isfinite(f) else f"{key} r = n/a")
        a_u1, _ = fit_exp(ms, [agg[m]["acc_u1"] for m in ms], floor=1e-12)
        a_c1, _ = fit_exp(ms, [agg[m]["acc_c1"] for m in ms], floor=1e-12)
        print(f"  {name:13s}: " + "; ".join(out) + f"; a_u1 {a_u1:.3f}, a_c1 {a_c1:.3f}")


if __name__ == "__main__":
    main()
