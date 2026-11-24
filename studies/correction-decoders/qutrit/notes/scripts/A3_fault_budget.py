#!/usr/bin/env python3
"""A3: per-segment fault budget of the baseline LRB-D circuits (read-only).

Parses baseline .chp circuit files directly and sums the activation
probabilities of the N1/N2 noise locations in each time segment:
  forward Clifford k (k = 1..m), check round r (r = 0..m-1, incl. reset layer),
  the unchecked inverse block (all m inverse Cliffords), and the final round m.
Faults are also split by type: data-qutrit locations, ancilla-only locations,
measurement-result flips (N1 'f' right before M) and reset flips (N1 'f'
after RESET).  The numbers are sums of first-order fault probabilities (the
expected number of activated locations), in units of the swept p.

Usage: module load python/3.11; python A3_fault_budget.py [--p-idx 8] [--cliff 0]
"""

from __future__ import annotations

import argparse

RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
       "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
PS = [0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 7.5e-3, 0.01, 0.0125, 0.015, 0.02,
      0.03, 0.04, 0.05]
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
GATES_1Q = {"I", "X", "X_INV", "Z", "Z_INV", "H", "H_INV", "P", "P_INV"}
GATES_2Q = {"CNOT", "CNOT_INV", "CZ", "CZ_INV", "SWAP"}


def parse(path):
    ops = []
    for raw in open(path):
        t = raw.split()
        if not t or t[0] in ("#", "Randomly-generated", "d", "DETECTOR",
                             "LOGICAL_OBSERVABLE", "TICK"):
            continue
        args = [int(x) for x in t[1:] if "=" not in x]
        params = dict(x.split("=", 1) for x in t[1:] if "=" in x)
        params = {k: v.strip('"') for k, v in params.items()}
        ops.append((t[0], args, params))
    return ops


def budget(path, p):
    ops = parse(path)
    segs = []          # list of [label, totals dict]
    state = "encode"
    n_round = 0
    cur = None

    def new(label):
        d = {"label": label, "data": 0.0, "anc": 0.0, "mflip": 0.0, "rflip": 0.0}
        segs.append(d)
        return d

    last = None
    for i, (name, args, params) in enumerate(ops):
        is_gate = name in GATES_1Q or name in GATES_2Q
        touches_anc = any(a >= 5 for a in args)
        if is_gate and touches_anc and state != "round":
            state = "round"
            cur = new(f"round {n_round}")
        elif is_gate and not touches_anc and state in ("round", "encode_done"):
            # data-only gate after a round: next logical block (or terminal)
            state = "gates"
            cur = new("gates")
        elif state == "encode" and is_gate and name == "CNOT" and args == [4, 2]:
            state = "encode_done"
            continue
        if name == "M" and args[0] == 8:
            n_round += 1
        if name in ("N1", "N2"):
            prob = float(params["prob"])
            ch = params.get("noise_channel", params.get("channel", "d"))
            if cur is None:
                cur = new("gates")
            if name == "N1" and ch == "f":
                nxt = ops[i + 1][0] if i + 1 < len(ops) else ""
                key = "mflip" if nxt == "M" else "rflip"
                cur[key] += prob
            elif any(a < 5 for a in args):
                cur["data"] += prob
            else:
                cur["anc"] += prob
        last = name
    # drop empty terminal segment(s) and label gate blocks
    segs = [s for s in segs if (s["data"] + s["anc"] + s["mflip"] + s["rflip"]) > 0
            or s["label"].startswith("round")]
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-idx", type=int, default=8)
    ap.add_argument("--cliff", type=int, default=0)
    ap.add_argument("--depth-idx", type=int, nargs="*", default=[1, 8])
    a = ap.parse_args()
    p = PS[a.p_idx]
    for di in a.depth_idx:
        m = DEPTHS[di]
        path = f"{RUN}/experiments/LRB/{a.cliff}/{a.p_idx}/{di}.chp"
        segs = budget(path, p)
        print(f"# Clifford seq {a.cliff}, depth m={m}, p={p}: expected number of "
              f"activated fault locations per segment, in units of p")
        tot = {"gate_fwd": 0.0, "rounds": 0.0, "inverse": 0.0, "final": 0.0}
        gate_blocks = [s for s in segs if s["label"] == "gates"]
        rounds = [s for s in segs if s["label"].startswith("round")]
        for s in segs:
            t = s["data"] + s["anc"] + s["mflip"] + s["rflip"]
            print(f"  {s['label']:10s} total {t/p:7.2f}  data {s['data']/p:7.2f}  "
                  f"ancilla {s['anc']/p:6.2f}  meas-flip {s['mflip']/p:6.2f}  "
                  f"reset-flip {s['rflip']/p:6.2f}")
        if m > 0:
            fwd = gate_blocks[:m]
            inv = gate_blocks[m] if len(gate_blocks) > m else None
            f_tot = sum(s["data"] + s["anc"] for s in fwd) / p
            r_tot = sum(s["data"] + s["anc"] + s["mflip"] + s["rflip"]
                        for s in rounds[:m]) / p
            r_data = sum(s["data"] for s in rounds[:m]) / p
            i_tot = (inv["data"] + inv["anc"]) / p if inv else 0.0
            fin = rounds[m]
            fin_tot = (fin["data"] + fin["anc"] + fin["mflip"] + fin["rflip"]) / p
            print(f"  summary: forward Cliffords {f_tot:.1f} p ({f_tot/m:.2f} p per "
                  f"Clifford); rounds 0..m-1 {r_tot:.1f} p ({r_tot/m:.1f} p per round,"
                  f" of which data {r_data/m:.1f} p); inverse block {i_tot:.1f} p "
                  f"({i_tot/m:.2f} p per Clifford); final round {fin_tot:.1f} p")


if __name__ == "__main__":
    main()
