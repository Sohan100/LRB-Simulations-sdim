"""Zero noise, schedule G2+G3, then G4, then G1 on the four reused sites:
separate readouts vs the data relay. Two and five rounds."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scheduled_sites_test import E, N, S, W, separate, summary  # noqa: E402


def relay_g4_then_g1(b):
    b.g("H", E); b.g("H", S)
    b.g("CNOT", S, 3); b.g("CNOT_INV", 4, S)
    b.g("CNOT_INV", E, 1); b.g("CNOT_INV", E, 4)
    b.g("CNOT", 4, S); b.g("CNOT_INV", S, 3)
    b.g("H_INV", E); b.g("H_INV", S)
    b.m("G4a", E); b.m("G4b", S)
    b.g("CNOT_INV", 3, W); b.g("CNOT", W, 0)
    b.g("CNOT", 0, N); b.g("CNOT_INV", 1, N)
    b.g("CNOT_INV", W, 0); b.g("CNOT", 3, W)
    b.m("G1a", N); b.m("G1b", W)


for rounds in (2, 5):
    summary(f"separate readout, G4 then G1, {rounds} rounds", separate("", "", "G4"), rounds=rounds)
    summary(f"data relay, G4 then G1, {rounds} rounds", relay_g4_then_g1, rounds=rounds)
summary("data relay, G4 then G1, logical H sandwich", relay_g4_then_g1, sandwich=True)
