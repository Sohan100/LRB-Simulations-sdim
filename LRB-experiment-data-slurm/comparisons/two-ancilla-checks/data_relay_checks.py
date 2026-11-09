"""More zero-noise checks of the scheduled data relay (transfer CNOT_INV q4->S):
logical H sandwich, five rounds, relay readouts, single data errors."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scheduled_sites_test import SHOTS, data_relay, run, summary  # noqa: E402

relay = data_relay("CNOT_INV")
summary("data relay, logical H sandwich", relay, sandwich=True)
summary("data relay, five rounds", relay, rounds=5)
_, _, _, relay_zero = run(relay, rounds=5)
print(f"W and S read 0 in every round: {sum(relay_zero)}/{SHOTS}")

print("\nsingle data error before round 1 (data relay, round-1 syndromes G1 G2 G3 G4):")
for wire in range(5):
    for pauli in ("X", "Z"):
        out, *_ = run(relay, rounds=1, error=(wire, pauli), shots=200)
        synd = tuple(sorted(set(out[g][0]))[0] if len(set(out[g][0])) == 1 else "random"
                     for g in ("G1", "G2", "G3", "G4"))
        print(f"  {pauli} on q{wire}: {synd}")
