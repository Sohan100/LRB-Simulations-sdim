"""Circuit-level memory experiments for the folded and the rotated surface code.

The folded surface code is the unrotated surface code [[d^2 + (d-1)^2, 1, d]] folded along its
diagonal. Folding moves qubits but keeps every ancilla-data coupling, so its memory circuit is the
unrotated code's circuit. The rotated code [[d^2, 1, d]] is the usual baseline.

Both codes are built by the same function from three ingredients (data qubits, checks with a
four-step CNOT schedule, logical operators) and receive the same noise:

* SD6 (standard depolarizing, CX gate set) and SI1000 (superconducting inspired, CZ gate set) of
  Gidney, Newman, Fowler and Broughton, "A Fault-Tolerant Honeycomb Memory", Quantum 5, 605 (2021),
  Tables 1 and 2. Noise rules stack, so a qubit idle during a measurement or reset moment gets
  Idle and ResonatorIdle.
* One round: reset ancillas, basis change, four CNOT layers, basis change, measure ancillas.
  Measurement and reset are separate moments, as in the LRB runs of this repository.
* Under SI1000 every CX(c, t) becomes H(t) CZ(c, t) H(t); Hadamards on the same qubit in
  neighbouring single-qubit moments cancel.

For the folded code a fold-transversal logical gate can be applied between every two rounds:
* "H": H on every data qubit. The SWAPs of the fold pairs are a relabelling of the two layers, so
  no gate is needed for them; instead every ancilla measures the other check type from then on.
  The H shares a time step with the ancilla basis change, so it adds no moment.
* "S": CZ on every fold pair (one extra moment) and S on the fold qubits (in the same time step as
  the ancilla basis change). S and S^dag alternate, so an even number of gates returns X_L.

Detectors. By default every check gets a detector. With detectors="sector" only the checks whose
starting type equals the memory basis do: the Z checks of a Z memory, whose detectors see the X
errors that flip Z_L, and the X checks of an X memory. A matching decoder decodes the X and Z
sectors separately, so for plain and H memories the two choices give the same predictions. After a
logical S the X check A satisfies X_A(after) = X_A(before) Z_fold(A)(after), so its detector takes
three measurements; with both sectors this produces hyperedges that matching cannot use, which is
why S memories are always run with detectors="sector".

S memories are run in the Z basis only. In the X basis an X error between two S gates becomes a Z
error on its mirror qubit at the second CZ and flips X_L; only the Z checks see it, so decoding
needs both sectors at once (correlated decoding of transversal gates), not independent matching.
A CZ never turns a Z error into an X error, so the Z basis has no such coupling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import stim

Coord = tuple[int, int]
Moment = list  # list of (gate, targets); targets are qubit indices, or pairs for CX/CZ


# --------------------------------------------------------------------------------------------
# Codes
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Check:
    ancilla: Coord
    basis: str                              # "X" or "Z"
    schedule: tuple[Coord | None, ...]      # data qubit touched in CNOT layer 0..3, or None


@dataclass(frozen=True)
class SurfaceCode:
    name: str
    distance: int
    data: tuple[Coord, ...]
    checks: tuple[Check, ...]
    logical_x: tuple[Coord, ...]
    logical_z: tuple[Coord, ...]

    @property
    def num_qubits(self) -> int:
        return len(self.data) + len(self.checks)


def folded_surface_code(d: int) -> SurfaceCode:
    """Unrotated surface code in the unfolded coordinates of the layout note.

    Sites (u, v), 0 <= u, v <= 2d-2, u to the right and v downwards. Data on u+v even, X checks on
    (u odd, v even), Z checks on (u even, v odd). Every check visits its neighbours in the order
    north, west, east, south; the hook left after two steps is a diagonal pair, which never
    shortens a logical string. The fold is (u, v) <-> (v, u).
    """
    L = 2 * d - 2
    data = tuple((u, v) for v in range(L + 1) for u in range(L + 1) if (u + v) % 2 == 0)
    dset = set(data)
    order = ((0, -1), (-1, 0), (1, 0), (0, 1))          # N, W, E, S
    checks = []
    for v in range(L + 1):
        for u in range(L + 1):
            if (u + v) % 2 == 0:
                continue
            basis = "X" if u % 2 == 1 else "Z"
            sched = tuple((u + du, v + dv) if (u + du, v + dv) in dset else None for du, dv in order)
            checks.append(Check((u, v), basis, sched))
    logical_x = tuple(q for q in data if q[0] == 0)     # a column, weight d
    logical_z = tuple(q for q in data if q[1] == 0)     # a row, weight d
    return SurfaceCode("folded", d, data, tuple(checks), logical_x, logical_z)


def rotated_surface_code(d: int) -> SurfaceCode:
    """Rotated surface code: data at (2i+1, 2j+1), ancillas at even (x, y).

    X checks visit NW, NE, SW, SE and Z checks NW, SW, NE, SE, so X hooks are horizontal and
    Z hooks vertical, perpendicular to the logical of the same type.
    """
    data = tuple((2 * i + 1, 2 * j + 1) for j in range(d) for i in range(d))
    dset = set(data)
    x_order = ((-1, -1), (1, -1), (-1, 1), (1, 1))
    z_order = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    checks = []
    for b in range(d + 1):
        for a in range(d + 1):
            x, y = 2 * a, 2 * b
            basis = "X" if (a + b) % 2 == 0 else "Z"
            on_tb = b in (0, d) and 0 < a < d
            on_lr = a in (0, d) and 0 < b < d
            interior = 0 < a < d and 0 < b < d
            if not (interior or (on_tb and basis == "X") or (on_lr and basis == "Z")):
                continue
            order = x_order if basis == "X" else z_order
            sched = tuple((x + dx, y + dy) if (x + dx, y + dy) in dset else None for dx, dy in order)
            checks.append(Check((x, y), basis, sched))
    logical_x = tuple(q for q in data if q[0] == 1)     # left column
    logical_z = tuple(q for q in data if q[1] == 1)     # top row
    return SurfaceCode("rotated", d, data, tuple(checks), logical_x, logical_z)


CODES = {"folded": folded_surface_code, "rotated": rotated_surface_code}


# --------------------------------------------------------------------------------------------
# Noise models
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class NoiseModel:
    name: str
    two_qubit_gate: str       # native entangling gate for the check circuit
    p2: float                 # DEPOLARIZE2 after any two-qubit gate
    p1: float                 # DEPOLARIZE1 after a single-qubit Clifford
    p_init: float             # X_ERROR after reset
    p_meas: float             # X_ERROR before measurement
    p_idle: float             # DEPOLARIZE1 on a qubit idle in any moment
    p_resonator_idle: float   # extra DEPOLARIZE1 on a qubit idle in a measure/reset moment

    @staticmethod
    def sd6(p: float) -> "NoiseModel":
        return NoiseModel("SD6", "CX", p, p, p, p, p, 0.0)

    @staticmethod
    def si1000(p: float) -> "NoiseModel":
        return NoiseModel("SI1000", "CZ", p, p / 10, 2 * p, 5 * p, p / 10, 2 * p)

    @staticmethod
    def by_name(name: str, p: float) -> "NoiseModel":
        return {"SD6": NoiseModel.sd6, "SI1000": NoiseModel.si1000}[name.upper()](p)

    @staticmethod
    def noiseless(gate: str = "CX") -> "NoiseModel":
        return NoiseModel("none", gate, 0, 0, 0, 0, 0, 0)


ONE_QUBIT = ("H", "S", "S_DAG")
TWO_QUBIT = ("CX", "CZ")


# --------------------------------------------------------------------------------------------
# Rounds as lists of moments
# --------------------------------------------------------------------------------------------
def _round(code: SurfaceCode, idx: dict, types: dict, first: bool, basis: str,
           data_1q: list) -> list:
    """Moments of one check round; data_1q is [(gate, qubits)] done during the basis change."""
    anc = [idx[c.ancilla] for c in code.checks]
    xanc = [idx[c.ancilla] for c in code.checks if types[c.ancilla] == "X"]
    reset = anc + ([idx[q] for q in code.data] if first else [])
    h = set(xanc)
    if first and basis == "X":
        h |= {idx[q] for q in code.data}
    basis_change = []
    for g, qs in data_1q:
        if g == "H":
            h ^= set(qs)
        else:
            basis_change.append((g, list(qs)))
    moments = [[("R", reset)], [("H", sorted(h))] + basis_change]
    for k in range(4):
        pairs = []
        for c in code.checks:
            q = c.schedule[k]
            if q is not None:
                a, dq = idx[c.ancilla], idx[q]
                pairs.append((a, dq) if types[c.ancilla] == "X" else (dq, a))
        moments.append([("CX", pairs)])
    moments.append([("H", xanc)])
    moments.append([("M", anc)])
    return moments


def _compile_cz(moments: list) -> list:
    """CX(c,t) -> H(t) CZ H(t); neighbouring single-qubit moments merge and H.H cancels.

    A merge is only made when it keeps one gate per qubit per moment."""
    out: list = []
    for mom in moments:
        if len(mom) == 1 and mom[0][0] == "CX":
            pairs = mom[0][1]
            tgt = [t for _, t in pairs]
            out += [[("H", tgt)], [("CZ", pairs)], [("H", tgt)]]
        else:
            out.append([(g, list(t)) for g, t in mom])

    def single(m):
        return all(g in ONE_QUBIT for g, _ in m)

    merged: list = []
    for mom in out:
        if merged and single(mom) and single(merged[-1]):
            prev = {g: set(t) for g, t in merged[-1]}
            cur = {g: set(t) for g, t in mom}
            h = prev.get("H", set()) ^ cur.get("H", set())
            others = {g: prev.get(g, set()) | cur.get(g, set()) for g in ("S", "S_DAG")}
            busy = [q for g in ("S", "S_DAG") for q in others[g]]
            if len(busy) == len(set(busy)) and not (set(busy) & h):
                merged[-1] = [("H", sorted(h))] + [(g, sorted(others[g])) for g in ("S", "S_DAG") if others[g]]
                continue
        merged.append(mom)
    return [[(g, t) for g, t in m if t] for m in merged if any(t for _, t in m)]


def _emit(circ: stim.Circuit, moment: list, nq: int, noise: NoiseModel) -> None:
    used: set = set()
    ops = []
    for g, tg in moment:
        flat = [q for pair in tg for q in pair] if g in TWO_QUBIT else list(tg)
        if not flat:
            continue
        assert not (used & set(flat)), "a qubit is used twice in one moment"
        used |= set(flat)
        ops.append((g, flat))
    if not ops:
        return
    collapse = any(g in ("R", "M") for g, _ in ops)
    for g, flat in ops:
        if g == "M" and noise.p_meas > 0:
            circ.append("X_ERROR", flat, noise.p_meas)
    for g, flat in ops:
        circ.append(g, flat)
    for g, flat in ops:
        if g == "R" and noise.p_init > 0:
            circ.append("X_ERROR", flat, noise.p_init)
        elif g in ONE_QUBIT and noise.p1 > 0:
            circ.append("DEPOLARIZE1", flat, noise.p1)
        elif g in TWO_QUBIT and noise.p2 > 0:
            circ.append("DEPOLARIZE2", flat, noise.p2)
    idle = [q for q in range(nq) if q not in used]
    if idle and noise.p_idle > 0:
        circ.append("DEPOLARIZE1", idle, noise.p_idle)
    if idle and collapse and noise.p_resonator_idle > 0:
        circ.append("DEPOLARIZE1", idle, noise.p_resonator_idle)
    circ.append("TICK")


def fold_pairs(code: SurfaceCode) -> tuple[list[Coord], list[tuple[Coord, Coord]]]:
    """Fold qubits and fold pairs ((u, v) with u < v, and its mirror) of the folded code."""
    fold = [q for q in code.data if q[0] == q[1]]
    pairs = [(q, (q[1], q[0])) for q in code.data if q[0] < q[1]]
    return fold, pairs


def memory_circuit(code: SurfaceCode, rounds: int, basis: str, noise: NoiseModel,
                   logical_gate: str | None = None, detectors: str = "all") -> stim.Circuit:
    """Prepare a logical basis state, run `rounds` check rounds, measure every data qubit.

    logical_gate (folded code only): "H" or "S", applied between every two rounds.
    detectors: "all" or "sector" (see the module docstring); S forces "sector".
    """
    assert basis in ("X", "Z") and rounds >= 1 and detectors in ("all", "sector")
    if logical_gate == "S":
        assert basis == "Z", "S memories need correlated decoding in the X basis; run them in Z"
        detectors = "sector"
    if logical_gate is not None:
        assert code.name == "folded" and logical_gate in ("H", "S")
    qubits = list(code.data) + [c.ancilla for c in code.checks]
    idx = {q: i for i, q in enumerate(qubits)}
    nq = len(qubits)
    circ = stim.Circuit()
    for q, i in idx.items():
        circ.append("QUBIT_COORDS", [i], [q[0], q[1]])
    types = {c.ancilla: c.basis for c in code.checks}
    keep = {c.ancilla for c in code.checks if detectors == "all" or c.basis == basis}
    # mirror check of every check: the fold maps an X check onto a Z check and back
    mirror = {c.ancilla: (c.ancilla[1], c.ancilla[0]) for c in code.checks}
    fold, pairs = fold_pairs(code) if code.name == "folded" else ([], [])

    meas = 0
    last: dict = {}
    cur_basis = basis          # basis of the logical state in the current frame
    pending: list = []         # data 1q gates for the next basis-change moment
    last_gate = None
    for r in range(rounds):
        moments = _round(code, idx, types, r == 0, basis, pending)
        if noise.two_qubit_gate == "CZ":
            moments = _compile_cz(moments)
        for m in moments:
            _emit(circ, m, nq, noise)
        rec = {c.ancilla: meas + j for j, c in enumerate(code.checks)}
        meas += len(code.checks)
        for c in code.checks:
            a = c.ancilla
            if a not in keep:
                continue
            tg = [stim.target_rec(rec[a] - meas)]
            if r == 0:
                if types[a] != basis:
                    continue
            else:
                tg.append(stim.target_rec(last[a] - meas))
                if last_gate == "S" and types[a] == "X":
                    # S maps X_A to X_A Z_{fold(A)}; the mirror Z check is unchanged by S, so its
                    # value in this round stands in for its value at the gate
                    tg.append(stim.target_rec(rec[mirror[a]] - meas))
            circ.append("DETECTOR", tg, [a[0], a[1], r])
        last = rec
        circ.append("SHIFT_COORDS", [], [0, 0, 1])
        pending, last_gate = [], None
        if logical_gate is not None and r < rounds - 1:
            if logical_gate == "H":
                pending = [("H", [idx[q] for q in code.data])]
                types = {a: ("Z" if t == "X" else "X") for a, t in types.items()}
                cur_basis = "Z" if cur_basis == "X" else "X"
            else:
                g = "S" if (r % 2 == 0) else "S_DAG"
                _emit(circ, [("CZ", [(idx[a], idx[b]) for a, b in pairs])], nq, noise)
                pending = [(g, [idx[q] for q in fold])]
            last_gate = logical_gate
    # final data measurement in the current frame's basis
    dq = [idx[q] for q in code.data]
    if cur_basis == "X":
        _emit(circ, [("H", dq)], nq, noise)
    _emit(circ, [("M", dq)], nq, noise)
    drec = {q: meas + j for j, q in enumerate(code.data)}
    meas += len(code.data)
    for c in code.checks:
        if types[c.ancilla] != cur_basis or c.ancilla not in keep:
            continue
        tg = [stim.target_rec(last[c.ancilla] - meas)]
        tg += [stim.target_rec(drec[q] - meas) for q in c.schedule if q is not None]
        circ.append("DETECTOR", tg, [c.ancilla[0], c.ancilla[1], 0])
    # A logical H keeps the support of the logical and changes only its Pauli type (X on the
    # column becomes Z on the column), so the observable is always the starting logical's support,
    # read out in the current frame's basis.
    logical = code.logical_x if basis == "X" else code.logical_z
    circ.append("OBSERVABLE_INCLUDE", [stim.target_rec(drec[q] - meas) for q in logical], 0)
    return circ


def build(code_name: str, d: int, noise_name: str, p: float, basis: str, rounds: int | None = None,
          logical_gate: str | None = None, detectors: str = "all") -> stim.Circuit:
    code = CODES[code_name](d)
    return memory_circuit(code, rounds if rounds is not None else d, basis,
                          NoiseModel.by_name(noise_name, p), logical_gate, detectors)


def round_depth(code_name: str, noise_name: str) -> int:
    """Moments per check round (after the first)."""
    code = CODES[code_name](3)
    idx = {q: i for i, q in enumerate(list(code.data) + [c.ancilla for c in code.checks])}
    moments = _round(code, idx, {c.ancilla: c.basis for c in code.checks}, False, "Z", [])
    if NoiseModel.by_name(noise_name, 0.001).two_qubit_gate == "CZ":
        moments = _compile_cz(moments)
    return len([m for m in moments if any(t for _, t in m)])


def tasks(codes: Iterable[str], distances: Iterable[int], noises: Iterable[str], ps: Iterable[float],
          bases: Iterable[str] = ("X", "Z"), logical_gate: str | None = None, detectors: str = "all"):
    import sinter
    for noise in noises:
        for code in codes:
            for d in distances:
                for p in ps:
                    for b in bases:
                        circ = build(code, d, noise, p, b, logical_gate=logical_gate, detectors=detectors)
                        yield sinter.Task(
                            circuit=circ,
                            json_metadata=dict(code=code, d=d, noise=noise, p=p, basis=b, rounds=d,
                                               qubits=CODES[code](d).num_qubits,
                                               gate=logical_gate or "none",
                                               detectors="sector" if logical_gate == "S" else detectors))
