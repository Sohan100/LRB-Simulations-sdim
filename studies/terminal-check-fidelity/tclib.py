"""Exact engine for terminal-check (const-0) LRB-D on the folded [[5,1,2]] code.

Everything here is read-only with respect to ``src/lrb`` and the run folders.
Circuits are assembled in memory from the production generator's own pieces
(``LRBCircuitGenerator`` segments and noise scheduling), so no file is written
and the ``working-folder*.txt`` pointers are never touched.

Key fact used throughout (Section 2 of REPORT.md): a terminal-check circuit
has two detectors (the X-type stabilizers evaluated from the final X-basis
data readout) and one logical observable (logical X). sdim's exact DEM gives
every noise gate as a "subgroup mechanism" on Z_d^3 = (D0, D1, L): with
probability pi it adds a uniformly random element of a subgroup. The Fourier
transform (character) of the outcome distribution is therefore a product,

    chi_k = prod_over_noise_gates (1 - pi_g)^[k is non-trivial on gate g's subgroup],

for k in Z_d^3. ``CircuitChars`` stores, per circuit, the number of noise
gates of each (kind, scale) type on which each character is non-trivial, so
the exact outcome distribution follows for every p without sampling.

Outcome indexing: o = (s1, s2, l) -> s1 * d^2 + s2 * d + l, same for k.
"""
from __future__ import annotations

import functools
import itertools
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # studies/terminal-check-fidelity -> repo root
sys.path.insert(0, os.path.join(REPO, "src"))
MAIN_DATA = os.path.join(REPO, "LRB-experiment-data-slurm")

from sdim.circuit import Circuit  # noqa: E402
from sdim.dem import DetectorErrorModel  # noqa: E402

from lrb.circuit_generator import (  # noqa: E402
    LRBCircuitGenerator,
)
from lrb.code_simulation_profiles import CodeSimulationProfileRegistry  # noqa: E402
from lrb.dem_simulation import reference_shot  # noqa: E402

CODE_NAMES = {"qubit": "folded_qubit", "qutrit": "folded_qutrit"}
RUNS = {
    ("qutrit", "si1000"): "Run-2026-10-28-10-23-30-folded_qutrit-single-ancilla-si1000-sdim140",
    ("qutrit", "sd6"): "Run-2026-10-28-10-24-46-folded_qutrit-single-ancilla-sd6-sdim140",
    ("qubit", "si1000"): "Run-2026-10-31-10-12-55-folded_qubit-single-ancilla-si1000-sdim140",
    ("qubit", "sd6"): "Run-2026-10-31-10-13-03-folded_qubit-single-ancilla-sd6-sdim140",
}


# ----------------------------------------------------------------- Cliffords

def _token_unitaries(d: int) -> dict:
    """Single-qudit token unitaries in sdim's conventions (d = 2 or 3).

    Only used to find the compiled recovery word; every compiled circuit is
    checked afterwards against sdim's own noiseless reference shot.
    """
    w = np.exp(2j * np.pi / d)
    base = {
        "I": np.eye(d, dtype=complex),
        "X": np.roll(np.eye(d), 1, axis=0).astype(complex),
        "Z": np.diag([w ** j for j in range(d)]),
        "H": np.array([[w ** (j * k) for k in range(d)] for j in range(d)]) / np.sqrt(d),
        "P": np.diag([1, 1j]) if d == 2 else np.diag([1, 1, w]),
    }
    out = dict(base)
    for name, u in base.items():
        if name != "I":
            out[f"{name}_INV"] = u.conj().T
    return out


def _phase_key(u: np.ndarray) -> tuple:
    flat = u.reshape(-1)
    pivot = flat[int(np.argmax(np.abs(flat) > 1e-9))]
    v = flat * (abs(pivot) / pivot)
    return tuple(np.round(v.real, 6) + 0.0) + tuple(np.round(v.imag, 6) + 0.0)


class Recovery:
    """Catalog word that inverts a product of catalog words (d = 2 or 3)."""

    def __init__(self, cd):
        tok = _token_unitaries(cd.dimension)
        self.cd = cd
        self.u = {}
        for word in cd.clifford_strings:
            m = np.eye(cd.dimension, dtype=complex)
            for g in cd.clifford_to_gate_sequence(word):
                m = tok[g] @ m
            self.u[word] = m
        self.by_key = {}
        for word, m in self.u.items():
            self.by_key.setdefault(_phase_key(m), word)
        assert len(self.by_key) == len(self.u), "catalog has duplicates up to phase"

    def word(self, words) -> str:
        total = np.eye(self.cd.dimension, dtype=complex)
        for w in words:
            total = self.u[w] @ total
        return self.by_key[_phase_key(total.conj().T)]


@functools.lru_cache(maxsize=None)
def setup(code: str, noise: str):
    """Return (generator, code_definition, recovery table) for a code and noise model."""
    profile = CodeSimulationProfileRegistry.resolve_code_profile(CODE_NAMES[code])
    cd = profile.code_definition
    gen = LRBCircuitGenerator(code_definition=cd, noise_model=noise)
    return gen, cd, Recovery(cd)


def catalog(code: str) -> list:
    return list(setup(code, "si1000")[1].clifford_strings)


# ------------------------------------------------------------------ circuits

CROSSTALK_TAG = 0.0009765625  # prob_scale used only to label injected correlated noise


def build_circuit(code: str, noise: str, segments, noisy_prep: bool = False,
                  noisy_meas: bool = False, crosstalk=()) -> Circuit:
    """Assemble one terminal-check circuit.

    ``segments`` is a list of ``(word, mode)`` in execution order, with mode
    ``"fwd"`` (the catalog word), ``"inv"`` (its token-wise mirror inverse,
    as production uses) each optionally suffixed with ``"_ideal"`` to leave
    that Clifford noiseless (used only for ground-truth constructions).

    Preparation of |+>_L and the terminal X-basis data readout are ideal
    unless ``noisy_prep`` / ``noisy_meas`` ask for the generator's own
    circuit-level noise on them (data resets + encoder, and the final basis
    change + measurement). The circuit carries ``prob_scale`` metadata with
    all probabilities zero; use ``CircuitChars`` or ``set_p``.

    ``crosstalk`` is a tuple of qudit pairs; after every noisy Clifford one
    extra two-qudit depolarizing gate is added on each pair, labelled by
    ``prob_scale = CROSSTALK_TAG`` so its strength can be set independently
    of p (robustness tests only; production circuits have none).
    """
    gen, cd, _ = setup(code, noise)
    c = Circuit(dimension=cd.dimension, num_qudits=cd.encoded_num_qudits)
    if noisy_prep:
        prep = Circuit(dimension=cd.dimension, num_qudits=cd.encoded_num_qudits)
        for wire in range(5):
            prep.add_gate("RESET", wire)
        prep = prep + cd.logical_plus_initial_state()
        c = c + gen.apply_circuit_level_noise(prep)
    else:
        c = c + cd.logical_plus_initial_state()
    for word, mode in segments:
        base, _, flag = mode.partition("_")
        if base == "fwd":
            gates = cd.clifford_to_gate_sequence(word)
        elif base == "inv":
            gates = cd.clifford_inverse_map[word]
        else:
            raise ValueError(mode)
        seg = gen._logical_clifford_circuit(gates)
        c = c + (seg if flag == "ideal" else gen.apply_circuit_level_noise(seg))
        if flag != "ideal":
            for q0, q1 in crosstalk:
                c.add_gate("N2", q0, q1, prob=0.0)
                c.operations[-1].params["prob_scale"] = CROSSTALK_TAG
    term = cd.terminal_const0_measurement()
    c = c + (gen.apply_circuit_level_noise(term, noisy_measurements=True) if noisy_meas else term)
    gen.append_lrb_const0_detector_events(c, 0)
    return c


def mirror_segments(words) -> list:
    """C1..Cm then Cm^-1..C1^-1 (production closing)."""
    return [(w, "fwd") for w in words] + [(w, "inv") for w in reversed(words)]


def compiled_segments(code: str, words) -> list:
    """C1..Cm then one catalog recovery Clifford (standard RB closing).

    Depth 0 has no gate at all, like production's compiled mode.
    """
    if not words:
        return []
    rec = setup(code, "si1000")[2]
    return [(w, "fwd") for w in words] + [(rec.word(words), "fwd")]


def set_p(code: str, noise: str, circuit: Circuit, p: float) -> None:
    setup(code, noise)[0].update_noise_param(circuit, p)


# ------------------------------------------------------------ exact characters

_SRC = re.compile(r"@(\d+):")


def _pi(kind: str, q, d: int):
    """Subgroup firing probability of a noise gate with error probability q."""
    if kind == "N2":
        return q * d ** 4 / (d ** 4 - 1)
    if kind == "N1d":
        return q * d ** 2 / (d ** 2 - 1)
    return q * d / (d - 1)  # N1f, N1p


@functools.lru_cache(maxsize=None)
def char_grid(d: int) -> np.ndarray:
    """All k in Z_d^3 as rows (k_s1, k_s2, k_l), index k_s1 d^2 + k_s2 d + k_l."""
    return np.array(list(itertools.product(range(d), repeat=3)), dtype=np.int64)


@functools.lru_cache(maxsize=None)
def fourier_matrix(d: int) -> np.ndarray:
    """F[o, k] = omega^(-k.o) / d^3, so P = F @ chi."""
    g = char_grid(d)
    return np.exp(-2j * np.pi * ((g @ g.T) % d) / d) / d ** 3


class CircuitChars:
    """Exact p-dependence of one terminal-check circuit's outcome distribution.

    ``types`` lists (kind, scale) noise-gate classes; ``n[t, k]`` counts the
    gates of class t on which character k is non-trivial.
    """

    def __init__(self, code: str, noise: str, circuit: Circuit):
        gen, cd, _ = setup(code, noise)
        self.d = d = cd.dimension
        gen.update_noise_param(circuit, 1e-3)
        _, ref = reference_shot(circuit)
        assert all(v % d == 0 for v in ref["detectors"]), "noiseless syndrome not trivial"
        assert all(v % d == 0 for v in ref["logicals"]), "noiseless logical outcome not 0"
        dem = DetectorErrorModel.from_circuit(circuit, merge=False)
        assert dem.num_detectors == 2 and dem.num_observables == 1
        grid = char_grid(d)
        acc: dict = {}
        # sdim's program IR drops identity gates, so a mechanism's "@index"
        # counts operations other than "I". Map back and check every match.
        ir_ops = [op for op in circuit.operations if op.gate_name != "I"]
        for mech in dem.mechanisms:
            idx = [int(i) for i in _SRC.findall(mech.source)]
            assert len(idx) == 1, mech.source
            op = ir_ops[idx[0]]
            assert op.gate_name in ("N1", "N2"), (mech.source, op.gate_name)
            assert f":q{int(op.qudit_index)}" in mech.source, (mech.source, op.qudit_index)
            if op.gate_name == "N2":
                kind = "N2"
            else:
                kind = "N1" + str(op.params.get("noise_channel", "d"))
            scale = float(op.params["prob_scale"])
            nontrivial = np.zeros(d ** 3, dtype=bool)
            for g in mech.generators:
                vec = np.zeros(3, dtype=np.int64)
                for target, coeff in g.items():
                    vec[int(target)] = int(coeff)
                nontrivial |= ((grid @ vec) % d) != 0
            acc.setdefault((kind, scale), np.zeros(d ** 3, dtype=np.int64))
            acc[(kind, scale)] += nontrivial
        self.types = sorted(acc)
        self.n = np.array([acc[t] for t in self.types], dtype=np.int64).reshape(len(self.types), d ** 3)

    def chars(self, p, scale_override=None) -> np.ndarray:
        """chi_k(p) for all k; ``p`` scalar or array (leading axis)."""
        p = np.asarray(p, dtype=float)
        logchi = np.zeros(p.shape + (self.d ** 3,))
        for (kind, scale), n in zip(self.types, self.n):
            s = scale if scale_override is None else scale_override.get((kind, scale), scale)
            logchi = logchi + np.log1p(-_pi(kind, s * p, self.d))[..., None] * n
        return np.exp(logchi)

    def dist(self, p) -> np.ndarray:
        """Exact outcome distribution over (s1, s2, l), flattened to d^3."""
        return dist_from_chars(self.chars(p), self.d)


def dist_from_chars(chi: np.ndarray, d: int) -> np.ndarray:
    return np.real(chi @ fourier_matrix(d).T)


def chars_from_dist(P: np.ndarray, d: int) -> np.ndarray:
    """chi_k = sum_o P(o) omega^(k.o); real part (noise here is inversion symmetric)."""
    g = char_grid(d)
    return np.real(P @ np.exp(2j * np.pi * ((g @ g.T) % d) / d))


def chars_from_dem(dem) -> np.ndarray:
    """Characters of a DEM with explicit probabilities (e.g. a production .chp)."""
    d = dem.dimension
    grid = char_grid(d)
    logchi = np.zeros(d ** 3)
    for mech in dem.mechanisms:
        nontrivial = np.zeros(d ** 3, dtype=bool)
        for g in mech.generators:
            vec = np.zeros(3, dtype=np.int64)
            for target, coeff in g.items():
                vec[int(target)] = int(coeff)
            nontrivial |= ((grid @ vec) % d) != 0
        logchi += np.log1p(-mech.probability) * nontrivial
    return np.exp(logchi)


# --------------------------------------------------------- derived quantities

def index_sets(d: int):
    """Character index sets: S (logical part 0, non-zero syndrome part) and
    L (logical part 1), plus outcome indices of accepted shots by logical value."""
    g = char_grid(d)
    S = np.where((g[:, 2] == 0) & ((g[:, 0] != 0) | (g[:, 1] != 0)))[0]
    L = np.where(g[:, 2] == 1)[0]
    acc = np.where((g[:, 0] == 0) & (g[:, 1] == 0))[0]  # ordered by logical value
    return S, L, acc


def accept_and_polarization(P: np.ndarray, d: int):
    """Acceptance A, joint J = E[Re w^l 1_acc] and accepted polarization J / A.

    For d = 2 the polarization is <X_L> of accepted shots; for d = 3 it is the
    signed polarization (3 P(0|acc) - 1) / 2. Leading axes of P are kept.
    """
    _, _, acc = index_sets(d)
    pa = P[..., acc]
    A = pa.sum(-1)
    J = pa[..., 0] - pa[..., 1:].sum(-1) / (d - 1)
    return A, J, J / A


def infidelity_from_polarization(lam, d: int):
    """Average-gate-infidelity scale of a polarization loss: (d-1)/d (1 - lam)."""
    return (d - 1) / d * (1.0 - np.asarray(lam))
