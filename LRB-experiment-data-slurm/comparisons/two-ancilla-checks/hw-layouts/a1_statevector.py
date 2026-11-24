"""Exact statevector check that A1's stabilizer measurements are non-destructive.

A random (non-stabilizer) logical qutrit state is put in the code space, ancillas E
(wire 5) and N (wire 6) start in |0>, and the A1 check circuits are applied gate by
gate on all 9 qutrits (3^9 amplitudes), with projective ancilla readouts.
Reports per check: readout probabilities, data fidelity with the input logical
state, and <G1..G4> on the data. Then a gate-by-gate trace of G1 (the
read-before / read-after check) and G2 shows where the information goes mid-check.
"""
import ast
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import hw_layouts_statevector as SV  # noqa: E402

E, N = 5, 6
LAB = {0: "q0", 1: "q1", 2: "q2", 3: "q3", 4: "q4", E: "E", N: "N"}
ANC = {"G1": N, "G2": E, "G3": N, "G4": E}


def checks():
    out = {}
    for line in (HERE / "encoder_A1_circuits.txt").read_text().splitlines():
        if line[:1] == "G":
            tag, rest = line.split(" ", 1)
            out[tag] = [("g", *op) for op in ast.literal_eval(rest)] + [("m", ANC[tag], tag), ("r", ANC[tag])]
    return out


def fresh(psi0):
    psi = np.zeros((3,) * SV.NW, dtype=complex)
    psi[(slice(None),) * SV.ND + (0,) * (SV.NW - SV.ND)] = psi0.reshape((3,) * SV.ND)
    return psi


def main():
    rng = np.random.default_rng(1116)
    evals, evecs = np.linalg.eigh(SV.P)
    code = evecs[:, evals > 0.5]
    chk = checks()
    print(f"code space dimension {code.shape[1]} (one logical qutrit)")
    worst_f, worst_p = 1.0, 0.0
    for trial in range(10):
        coeff = rng.normal(size=3) + 1j * rng.normal(size=3)
        psi0 = code @ (coeff / np.linalg.norm(coeff))
        psi = fresh(psi0)
        if trial == 0:
            print(f"random logical state: <X_L>={np.vdot(psi0, SV.XL @ psi0):.3f} "
                  f"<Z_L>={np.vdot(psi0, SV.ZL @ psi0):.3f}\n\nround 1, check by check:")
        for rnd in range(3):
            for tag in ("G1", "G2", "G3", "G4"):
                psi, log = SV.run(psi, chk[tag], rng)
                d = SV.data_state(psi)
                worst_f = min(worst_f, abs(np.vdot(psi0, d)) ** 2)
                worst_p = max(worst_p, 1 - log[0][2][0])
                if trial == 0 and rnd == 0:
                    print(f"  {tag} ({LAB[ANC[tag]]}): " + SV.report_check(psi, psi0, log))
        if trial == 0:
            print(f"  after 3 rounds: fidelity {abs(np.vdot(psi0, SV.data_state(psi))) ** 2:.12f}")
    print(f"\n10 random logical states x 3 rounds: min fidelity {worst_f:.12f}, "
          f"max P(readout != 0) {worst_p:.1e}")

    coeff = rng.normal(size=3) + 1j * rng.normal(size=3)
    psi0 = code @ (coeff / np.linalg.norm(coeff))

    def tr(o, s):
        pn = np.sum(np.abs(np.take(s, 0, axis=N)) ** 2)
        pe = np.sum(np.abs(np.take(s, 0, axis=E)) ** 2)
        st = " ".join(f"<{k}>={SV.expect(s, M):+.2f}" for k, M in SV.G.items())
        print(f"  {o[1]:8s} {LAB[o[2]]:>2s}" + (f"->{LAB[o[3]]:2s}" if len(o) > 3 else "    ")
              + f"  P(N=0)={pn:.3f} P(E=0)={pe:.3f}  data in code space {SV.expect(s, SV.P):.3f}  {st}")

    for tag in ("G1", "G2"):
        print(f"\n{tag} gate by gate (stabilizers evaluated on the data alone):")
        psi, log = SV.run(fresh(psi0), chk[tag], rng, trace=tr)
        print("  readout: " + SV.report_check(psi, psi0, log))


if __name__ == "__main__":
    main()
