"""Zero noise: what reading one half of a split stabilizer does to the data.

A. Couple W to q3 alone and read W, then read q3 directly: W always equals
   -z3, i.e. reading W is a measurement of data qutrit q3. Afterwards the whole
   G4 (measured correctly, by the data relay) is random.
B. Five rounds of the user's schedule (G2+G3, then G4, then G1) with separate
   readouts, then every data qutrit read in the X basis, no postselection.
C. A single Z4 error after round 2 of 5: with separate readouts nothing tells
   it apart from the random G1/G4 values and G2, G3 never see it, while the
   logical X value changes; with the data relay G4 flags it in rounds 3-5.
"""
import sys
from collections import Counter
from pathlib import Path

from sdim.program import Program

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scheduled_sites_test import (E, N, S, W, Builder, Code, g2_on_e,  # noqa: E402
                                  g3_on_n, separate)

SHOTS = 3000


def relay_g4(b):
    b.g("H", E); b.g("H", S)
    b.g("CNOT", S, 3); b.g("CNOT_INV", 4, S)
    b.g("CNOT_INV", E, 1); b.g("CNOT_INV", E, 4)
    b.g("CNOT", 4, S); b.g("CNOT_INV", S, 3)
    b.g("H_INV", E); b.g("H_INV", S)
    b.m("G4a", E); b.m("G4b", S)


def relay_g1(b):
    b.g("CNOT_INV", 3, W); b.g("CNOT", W, 0)
    b.g("CNOT", 0, N); b.g("CNOT_INV", 1, N)
    b.g("CNOT_INV", W, 0); b.g("CNOT", 3, W)
    b.m("G1a", N); b.m("G1b", W)


def relay_g4_then_g1(b):
    relay_g4(b); relay_g1(b)


def simulate(b):
    m = Program(b.c).simulate(shots=SHOTS)[0]
    return lambda label, r, s: int(
        m[b.reads[label][r][0]][b.reads[label][r][1]][s].measurement_value)


def dist(values):
    return dict(sorted(Counter(values).items()))


def test_a():
    print("== A: read W after coupling it to q3 only ==")
    b = Builder(Code.logical_plus_initial_state())
    b.g("CNOT_INV", 3, W); b.m("W", W); b.m("z3", 3)
    v = simulate(b)
    same = sum((v("W", 0, s) + v("z3", 0, s)) % 3 == 0 for s in range(SHOTS))
    print(f"  W readout == -z3 (q3 read directly)      {same}/{SHOTS}, "
          f"W values {dist(v('W', 0, s) for s in range(SHOTS))}")
    b = Builder(Code.logical_plus_initial_state())
    b.g("CNOT_INV", 3, W); b.m("W", W); b.reset(W)
    relay_g4(b)
    v = simulate(b)
    print(f"  whole G4 after reading W                 "
          f"{dist((v('G4a', 0, s) + v('G4b', 0, s)) % 3 for s in range(SHOTS))}")
    b = Builder(Code.logical_plus_initial_state())
    relay_g1(b); b.reset(N, W); relay_g4(b)
    v = simulate(b)
    print(f"  control: whole G1 first, then whole G4   "
          f"{dist((v('G4a', 0, s) + v('G4b', 0, s)) % 3 for s in range(SHOTS))}")


def rounds_then_read_all(step2, rounds=5, error=None, error_after=2):
    b = Builder(Code.logical_plus_initial_state())
    for r in range(rounds):
        if error and r == error_after:
            b.g(error[1], error[0])
        g3_on_n(b); g2_on_e(b); b.reset(N, E)
        step2(b); b.reset(N, W, E, S)
    for q in range(5):
        b.g("H_INV", q); b.m(f"x{q}", q)
    v = simulate(b)
    x = [[v(f"x{q}", 0, s) for q in range(5)] for s in range(SHOTS)]
    checks = {g: [[(v(f"{g}a", r, s) + v(f"{g}b", r, s)) % 3 for s in range(SHOTS)]
                  for r in range(rounds)] for g in ("G1", "G4")}
    for g in ("G2", "G3"):
        checks[g] = [[v(g, r, s) for s in range(SHOTS)] for r in range(rounds)]
    return x, checks


def report(name, x, checks):
    print(f"  {name}")
    for g in ("G1", "G2", "G3", "G4"):
        per_round = [len(set(r)) for r in checks[g]]
        flagged = sum(any(r[s] for r in checks[g]) for s in range(SHOTS))
        print(f"    {g}: distinct values per round {per_round}, "
              f"shots with a nonzero value {flagged}/{SHOTS}")
    print(f"    final X0 X3       {dist((a[0] + a[3]) % 3 for a in x)}")
    print(f"    final X0 X1 X4    {dist((a[0] + a[1] + a[4]) % 3 for a in x)}")
    print(f"    final G4 (data)   {dist((-a[1] + a[3] - a[4]) % 3 for a in x)}")
    print(f"    final G3 (data)   {dist((a[0] + a[1] - a[2]) % 3 for a in x)}")


if __name__ == "__main__":
    test_a()
    print("\n== B: five rounds, G2+G3 then G4 then G1, no postselection ==")
    report("separate readouts", *rounds_then_read_all(separate("", "", "G4")))
    report("data relay", *rounds_then_read_all(relay_g4_then_g1))
    print("\n== C: one Z4 error after round 2 of 5 ==")
    report("separate readouts", *rounds_then_read_all(separate("", "", "G4"), error=(4, "Z")))
    report("data relay", *rounds_then_read_all(relay_g4_then_g1, error=(4, "Z")))
