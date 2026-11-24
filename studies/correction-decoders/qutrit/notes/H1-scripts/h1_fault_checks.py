"""Single-fault analysis of three ways to implement logical P's CZ(2,3).

Weyl operators X^x Z^z on n qutrits are tracked as (x, z) in Z_3^{2n}
(phases dropped). Conjugation rules, checked by hand on basis states:
  CSUM(c->t) |a,b> = |a, a+b>:  x_t += x_c ; z_c -= z_t
  CSUM^-1(c->t):                x_t -= x_c ; z_c += z_t
  CZ(c,t)  |a,b> = w^{ab}|a,b>: z_t += x_c ; z_c += x_t
  CZ^-1(c,t):                   z_t -= x_c ; z_c -= x_t
For every gate in a gadget and every one of the 80 non-identity two-qutrit
Weyl errors placed right after it, the error is propagated to the end of the
gadget. A fault is "silent logical" if the data part has trivial syndrome,
acts non-trivially on the logical qutrit, and leaves no X-type component on
an ancilla/relay (which would be seen when that ancilla is next measured or
when it is checked for cleanliness).

Gadgets (wire labels: data 0-4, extra qutrits 5+):
  native : CZ(2,3)                                 (fold coupling native)
  relay  : CSUM(3->r) CZ(2,r) CSUM^-1(3->r)        (one relay qutrit r=9)
  bus    : planar 3x3 layout, data 0 used as a bus through ancillas 5 and 7:
           CSUM(2->7) CSUM(3->5) CSUM(7->0) CZ(0,5) CSUM^-1(7->0) CZ^-1(0,5)
           CSUM^-1(2->7) CSUM^-1(3->5)
Also checked: the ideal gadgets implement CZ(2,3) exactly (basis-state test).
"""

from __future__ import annotations

import itertools

import numpy as np

N = 10  # wires 0..9
STABS = {  # (sx, sz) on data 0..4
    "S_Z1": (np.zeros(5, int), np.array([1, -1, 0, -1, 0])),
    "S_Z2": (np.zeros(5, int), np.array([0, 1, 1, 0, -1])),
    "S_X1": (np.array([1, 1, -1, 0, 0]), np.zeros(5, int)),
    "S_X2": (np.array([0, -1, 0, 1, -1]), np.zeros(5, int)),
}
LOGICALS = {
    "X_L": (np.array([1, 0, 0, 1, 0]), np.zeros(5, int)),
    "Z_L": (np.zeros(5, int), np.array([1, 0, 1, 0, 0])),
}


def symp(x, z, sx, sz):
    return int((x @ sz - z @ sx) % 3)


def apply(gate, c, t, x, z):
    if gate == "CSUM":
        x[t] = (x[t] + x[c]) % 3
        z[c] = (z[c] - z[t]) % 3
    elif gate == "CSUM_INV":
        x[t] = (x[t] - x[c]) % 3
        z[c] = (z[c] + z[t]) % 3
    elif gate == "CZ":
        z[t] = (z[t] + x[c]) % 3
        z[c] = (z[c] + x[t]) % 3
    elif gate == "CZ_INV":
        z[t] = (z[t] - x[c]) % 3
        z[c] = (z[c] - x[t]) % 3
    else:
        raise ValueError(gate)


def apply_basis(gate, c, t, state, phase):
    """Act on a computational basis state; phase accumulates exponent of w."""
    a, b = state[c], state[t]
    if gate == "CSUM":
        state[t] = (a + b) % 3
    elif gate == "CSUM_INV":
        state[t] = (b - a) % 3
    elif gate == "CZ":
        phase += a * b
    elif gate == "CZ_INV":
        phase -= a * b
    return state, phase % 3


GADGETS = {
    "native": [("CZ", 2, 3)],
    "relay": [("CSUM", 3, 9), ("CZ", 2, 9), ("CSUM_INV", 3, 9)],
    "bus": [("CSUM", 2, 7), ("CSUM", 3, 5), ("CSUM", 7, 0), ("CZ", 0, 5),
            ("CSUM_INV", 7, 0), ("CZ_INV", 0, 5), ("CSUM_INV", 2, 7),
            ("CSUM_INV", 3, 5)],
}


def check_ideal(gadget):
    """Gadget maps |x, anc=0> to w^{x2 x3}|x, anc=0> for all data x."""
    for xs in itertools.product(range(3), repeat=5):
        state = list(xs) + [0] * (N - 5)
        phase = 0
        for g, c, t in gadget:
            state, phase = apply_basis(g, c, t, state, phase)
        if state[:5] != list(xs) or any(state[5:]):
            return False
        if phase != (xs[2] * xs[3]) % 3:
            return False
    return True


def analyse(gadget):
    counts = {"total": 0, "detected": 0, "trivial": 0, "silent_logical": 0}
    for k, (g, c, t) in enumerate(gadget):
        for e in itertools.product(range(3), repeat=4):
            if not any(e):
                continue
            x = np.zeros(N, int)
            z = np.zeros(N, int)
            x[c], z[c], x[t], z[t] = e
            for g2, c2, t2 in gadget[k + 1:]:
                apply(g2, c2, t2, x, z)
            xd, zd = x[:5], z[:5]
            syn = [symp(xd, zd, *s) for s in STABS.values()]
            anc_x = any(x[5:] % 3)
            log = [symp(xd, zd, *l) for l in LOGICALS.values()]
            counts["total"] += 1
            if any(syn) or anc_x:
                counts["detected"] += 1
            elif any(log):
                counts["silent_logical"] += 1
            else:
                counts["trivial"] += 1
    return counts


def all_errors_on_23_detected():
    for e in itertools.product(range(3), repeat=4):
        if not any(e):
            continue
        x = np.zeros(5, int)
        z = np.zeros(5, int)
        x[2], z[2], x[3], z[3] = e
        if not any(symp(x, z, *s) for s in STABS.values()):
            return False
    return True


def main():
    print("Every non-identity Weyl error supported on {2,3} is detected:",
          all_errors_on_23_detected())
    for name, gadget in GADGETS.items():
        print(f"{name}: ideal action is CZ(2,3) with clean ancillas: "
              f"{check_ideal(gadget)}; two-qutrit gates: {len(gadget)}; "
              f"single-fault outcomes: {analyse(gadget)}")


if __name__ == "__main__":
    main()
