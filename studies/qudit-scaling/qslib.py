"""General-q folded [[5,1,2]]_q code: Clifford catalogue, code profile, sampler.

Everything here builds on the repository's production source (``src/lrb``,
plus the fit and merge helpers in ``scripts/``).  The study first ran against
a frozen copy of that source taken from the one-ancilla-prod5b worktree on
2026-11-19; that copy is byte-identical to the repository's ``src/lrb`` and
the three helper scripts.  Nothing under ``src/`` or any run folder is
written.

Clifford catalogue
------------------
The production catalogues (q = 2: 24 elements, q = 3: 216) are a Pauli prefix
X^a Z^b times one representative word per element of SL(2, Z_q), with
descriptor words whose rightmost symbol acts first.  The symplectic words are
the lexicographically smallest shortest *positive* words in {H, S}
(``H < S``), and the Pauli prefix is ``"X" * a + "Z" * b``.  ``catalogue(q,
"production")`` reproduces both production tuples exactly (checked in
``01_catalogue_checks.py``) and extends the rule to any prime q.

``catalogue(q, "inverse")`` is the alternative compile that may also use the
native inverse tokens H_INV, P_INV, X_INV, Z_INV (lower-case letters in the
descriptor).  For q = 2 it coincides with the production catalogue.
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # studies/qudit-scaling -> repo root
REPO_SRC = os.path.join(REPO, "src")
REPO_SCRIPTS = os.path.join(REPO, "scripts")
DATA = os.path.join(REPO, "LRB-experiment-data-slurm")
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

# Descriptor letter -> sdim token.  Upper case as in production; lower case
# marks the inverse token (used only by the "inverse" compile).
LETTER_TOKEN = {
    "H": "H", "S": "P", "P": "P", "X": "X", "Z": "Z", "I": "I",
    "h": "H_INV", "s": "P_INV", "p": "P_INV", "x": "X_INV", "z": "Z_INV",
}
TOKEN_INVERSE = {
    "H": "H_INV", "H_INV": "H", "P": "P_INV", "P_INV": "P",
    "X": "X_INV", "X_INV": "X", "Z": "Z_INV", "Z_INV": "Z", "I": "I",
}


# ---------------------------------------------------------------------------
# Symplectic action of the one-qudit tokens, from sdim's own unitaries.
# ---------------------------------------------------------------------------
def token_unitaries(q: int) -> dict[str, np.ndarray]:
    from sdim import unitary as su

    h = np.asarray(su.generate_h_matrix(q), dtype=complex)
    p = np.asarray(su.generate_p_matrix(q), dtype=complex)
    x = np.asarray(su.generate_x_matrix(q), dtype=complex)
    z = np.asarray(su.generate_z_matrix(q), dtype=complex)
    out = {"H": h, "P": p, "X": x, "Z": z, "I": np.eye(q, dtype=complex)}
    for name in ("H", "P", "X", "Z"):
        out[name + "_INV"] = out[name].conj().T
    return out


def _pauli(q: int, a: int, b: int, u: dict) -> np.ndarray:
    return np.linalg.matrix_power(u["X"], a % q) @ np.linalg.matrix_power(u["Z"], b % q)


def symplectic_of(unit: np.ndarray, q: int, u: dict) -> np.ndarray:
    """2x2 matrix M over Z_q with U X^a Z^b U^dag ~ X^{a'} Z^{b'}, (a',b') = M (a,b)."""
    cols = []
    for a, b in ((1, 0), (0, 1)):
        target = unit @ _pauli(q, a, b, u) @ unit.conj().T
        found = None
        for a2 in range(q):
            for b2 in range(q):
                cand = _pauli(q, a2, b2, u)
                ov = np.vdot(cand, target) / q
                if abs(abs(ov) - 1.0) < 1e-8:
                    found = (a2, b2)
                    break
            if found:
                break
        assert found is not None
        cols.append(found)
    return np.array([[cols[0][0], cols[1][0]], [cols[0][1], cols[1][1]]], dtype=np.int64) % q


@lru_cache(maxsize=None)
def token_symplectics(q: int) -> dict[str, tuple]:
    u = token_unitaries(q)
    return {name: tuple(map(tuple, symplectic_of(u[name], q, u)))
            for name in ("H", "P", "H_INV", "P_INV")}


def _mat(t) -> np.ndarray:
    return np.array(t, dtype=np.int64)


def _key(m: np.ndarray, q: int) -> tuple:
    return tuple(int(v) for v in (m % q).reshape(-1))


def word_symplectic(word: str, q: int) -> np.ndarray:
    """Descriptor acts right to left: M = M(w0) M(w1) ... M(wn)."""
    sym = token_symplectics(q)
    m = np.eye(2, dtype=np.int64)
    for letter in word:
        tok = LETTER_TOKEN[letter]
        if tok in ("X", "Z", "X_INV", "Z_INV", "I"):
            continue
        m = (m @ _mat(sym[tok])) % q
    return m % q


def sl2_order(q: int) -> int:
    return q * (q * q - 1)


@lru_cache(maxsize=None)
def symplectic_words(q: int, alphabet: str = "HS") -> tuple[str, ...]:
    """Lexicographically smallest shortest word for each element of SL(2, q).

    Words are enumerated by length, and within a length in lexicographic order
    of ``alphabet`` (production uses "HS"; the inverse compile "HShs" ordered
    H < S < h < s).  The identity is the word "I".
    """
    order = {c: i for i, c in enumerate(alphabet)}
    target = sl2_order(q)
    seen = {_key(np.eye(2, dtype=np.int64), q): "I"}
    words = ["I"]
    frontier = [("", np.eye(2, dtype=np.int64))]
    length = 0
    while len(seen) < target:
        length += 1
        new_frontier = []
        # extend every shortest word of the previous length on the right; the
        # lexicographic order of words of length L is the order of (prefix,
        # last letter) when prefixes are themselves sorted.
        cands = []
        for prefix, m in frontier:
            for c in alphabet:
                w = prefix + c
                cands.append((tuple(order[x] for x in w), w))
        cands.sort()
        for _, w in cands:
            m = word_symplectic(w, q)
            k = _key(m, q)
            if k not in seen:
                seen[k] = w
                words.append(w)
                new_frontier.append((w, m))
        # Lexicographically smallest shortest words are prefix-closed, so only
        # the words that reached a new element need extending.
        frontier = new_frontier
        if length > 4 * q + 8:
            raise RuntimeError("BFS did not close")
    assert len(words) == target, (len(words), target)
    return tuple(words)


def pauli_words(q: int, compile_mode: str) -> tuple[str, ...]:
    """Pauli prefixes in the production order: b outer, a inner (X^a Z^b)."""
    out = []
    for b in range(q):
        for a in range(q):
            if compile_mode == "production":
                w = "X" * a + "Z" * b
            elif compile_mode == "inverse":
                wa = "X" * a if a <= q - a else "x" * (q - a)
                wb = "Z" * b if b <= q - b else "z" * (q - b)
                w = wa + wb
            else:
                raise ValueError(compile_mode)
            out.append(w)
    return tuple(out)


@lru_cache(maxsize=None)
def catalogue(q: int, compile_mode: str = "production") -> tuple[str, ...]:
    if compile_mode == "production":
        sym = symplectic_words(q, "HS")
    elif compile_mode == "inverse":
        sym = symplectic_words(q, "HShs")
    else:
        raise ValueError(compile_mode)
    out = []
    for pw in pauli_words(q, compile_mode):
        for sw in sym:
            out.append(pw + sw if sw != "I" else (pw or "I"))
    # Production descriptors spell the phase gate with the sdim symbol P.
    return tuple(w.replace("S", "P").replace("s", "p") for w in out)


def gate_sequence(descriptor: str) -> list[str]:
    """Execution-order sdim tokens (rightmost descriptor letter acts first)."""
    return [LETTER_TOKEN[c] for c in reversed(descriptor)]


def inverse_sequence(descriptor: str) -> list[str]:
    """Production inverse convention: same letters, each token inverted, in
    descriptor order (i.e. the execution-order reverse of gate_sequence)."""
    return [TOKEN_INVERSE[LETTER_TOKEN[c]] for c in descriptor]


def token_count(descriptor: str) -> int:
    return len(descriptor)


def mean_tokens(q: int, compile_mode: str) -> float:
    return float(np.mean([token_count(w) for w in catalogue(q, compile_mode)]))


# ---------------------------------------------------------------------------
# Code profile for arbitrary q (one-ancilla layout, production round).
# ---------------------------------------------------------------------------
ROUNDS = ("seven", "five")  # one-ancilla check round: published seven-moment, or five-moment
# Default round for the analysis scripts (20/22/23/24); QSCAL_ROUND=five selects the five-moment round.
ROUND_DEFAULT = os.environ.get("QSCAL_ROUND", "seven")
ROUND_SUFFIX = "" if ROUND_DEFAULT == "seven" else "_five"


def _round_class(q: int, round_name: str):
    from lrb import code_definitions as defns

    if round_name == "seven":
        return type(f"FoldedQ{q}DetectionCode", (defns.FoldedQutritDetectionCode,), {"dimension": q})
    if round_name == "five":
        return type(f"FoldedQ{q}FiveMomentDetectionCode", (defns.FoldedQutritProd5bDetectionCode,),
                    {"dimension": q})
    raise ValueError(round_name)


def _check_blocks(cls, round_name: str):
    if round_name == "seven":
        return ((cls.z_ALT_stabilizer_measurement_ancillae_circuit, {5, 6}),
                (cls.x_stabilizer_measurement_ancillae_circuit, {7, 8}))
    return ((cls.prod5b_check_round_circuit, {5, 6, 7, 8}),)


def _definition(q, words, round_name):
    from lrb.code_simulation_profiles import CodeSimulationProfileRegistry

    base = CodeSimulationProfileRegistry.resolve_code_profile("folded_qutrit").code_definition
    cls = _round_class(q, round_name)
    return replace(
        base,
        dimension=q,
        clifford_strings=tuple(words),
        clifford_to_gate_sequence=gate_sequence,
        clifford_inverse_map={w: inverse_sequence(w) for w in words},
        logical_plus_initial_state=cls.logical_plus_initial_state,
        logical_gate_circuit=cls.single_qudit_logical_gate_circuit,
        stabilizer_check_blocks=_check_blocks(cls, round_name),
        reset_measurement_wires=cls.reset_measurement_wires,
        terminal_measurement=cls.terminal_logical_x_measurement_circuit,
        terminal_const0_measurement=cls.terminal_const0_logical_x_measurement_circuit,
    )


@lru_cache(maxsize=None)
def code_definition(q: int, compile_mode: str = "production", round_name: str | None = None):
    return _definition(q, catalogue(q, compile_mode), round_name or ROUND_DEFAULT)


def generator(q: int, noise: str, compile_mode: str = "production", round_name: str | None = None):
    from lrb.circuit_generator import LRBCircuitGenerator

    return LRBCircuitGenerator(code_definition=code_definition(q, compile_mode, round_name),
                               noise_model=noise)


# ---------------------------------------------------------------------------
# Large q: random catalogue words without enumerating the whole catalogue.
# ---------------------------------------------------------------------------
def random_words(q: int, compile_mode: str, n: int, seed: int) -> list[str]:
    """n uniformly random catalogue words (Pauli prefix x SL(2,q) word)."""
    rng = np.random.default_rng(seed)
    sym = symplectic_words(q, "HS" if compile_mode == "production" else "HShs")
    out = []
    for _ in range(n):
        a, b = (int(x) for x in rng.integers(0, q, 2))
        if compile_mode == "production":
            pw = "X" * a + "Z" * b
        else:
            pw = ("X" * a if a <= q - a else "x" * (q - a)) + ("Z" * b if b <= q - b else "z" * (q - b))
        sw = sym[int(rng.integers(0, len(sym)))]
        w = pw + sw if sw != "I" else (pw or "I")
        out.append(w.replace("S", "P").replace("s", "p"))
    return out


def mean_tokens_exact(q: int, compile_mode: str) -> float:
    """Mean descriptor length over the full catalogue, without enumerating it."""
    sym = symplectic_words(q, "HS" if compile_mode == "production" else "HShs")
    sym_len = np.mean([0 if w == "I" else len(w) for w in sym])
    if compile_mode == "production":
        pauli = 2 * np.mean(np.arange(q))
    else:
        pauli = 2 * np.mean([min(a, q - a) for a in range(q)])
    # the identity word "I" has length 1 when the Pauli prefix is empty
    return float(sym_len + pauli + 1.0 / (q * q * len(sym)))


def code_definition_for_words(q: int, words, compile_mode: str = "production",
                              round_name: str | None = None):
    """Like code_definition(), restricted to the given words (cheap at large q)."""
    return _definition(q, tuple(dict.fromkeys(words)), round_name or ROUND_DEFAULT)
