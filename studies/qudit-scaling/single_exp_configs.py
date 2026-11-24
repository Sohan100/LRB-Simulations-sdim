"""Configurations of the Part 2 single-exponential analysis."""
CONFIGS = {
    # label: (merge family, merge label, code, layout, noise)
    "qutrit-one-si1000": ("si1000", "one-ancilla-si1000", "qutrit", "one-ancilla", "SI1000"),
    "qutrit-one-sd6": ("sd6", "one-ancilla-sd6", "qutrit", "one-ancilla", "SD6"),
    "qutrit-two-si1000": ("layout-b-si1000", "layout-b-si1000", "qutrit", "two-ancilla", "SI1000"),
    "qutrit-two-sd6": ("layout-b-sd6", "layout-b-sd6", "qutrit", "two-ancilla", "SD6"),
    "qubit-one-si1000": ("qubit-si1000", "qubit-one-ancilla-si1000", "qubit", "one-ancilla", "SI1000"),
    "qubit-one-sd6": ("qubit-sd6", "qubit-one-ancilla-sd6", "qubit", "one-ancilla", "SD6"),
    "qubit-two-si1000": ("qubit-layout-b-si1000", "qubit-layout-b-si1000", "qubit", "two-ancilla", "SI1000"),
    "qubit-two-sd6": ("qubit-layout-b-sd6", "qubit-layout-b-sd6", "qubit", "two-ancilla", "SD6"),
    # one-ancilla layout with the five-moment round (the paper's round since 2026-11-19)
    "qutrit-one5-si1000": ("prod5b-si1000", "prod5b-one-ancilla-si1000", "qutrit", "one-ancilla, five-moment", "SI1000"),
    "qutrit-one5-sd6": ("prod5b-sd6", "prod5b-one-ancilla-sd6", "qutrit", "one-ancilla, five-moment", "SD6"),
    "qubit-one5-si1000": ("qubit-prod5b-si1000", "qubit-prod5b-one-ancilla-si1000", "qubit", "one-ancilla, five-moment", "SI1000"),
    "qubit-one5-sd6": ("qubit-prod5b-sd6", "qubit-prod5b-one-ancilla-sd6", "qubit", "one-ancilla, five-moment", "SD6"),
}
