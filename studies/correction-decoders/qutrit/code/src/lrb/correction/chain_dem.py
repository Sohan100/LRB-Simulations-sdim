"""Chain-structured detector error models for repeated-round LRB circuits.

``lrb.dem_simulation.CompactDetectorErrorModel`` stores, for every noise
location, the Z_3 shift that each unit Weyl component causes on every raw
stabilizer detector and on the terminal logical observable. The detectors of
an ordinary LRB circuit are raw ancilla outcomes, so a persistent data error
shows up in every later round. This module rewrites the model as a chain:

* every location is assigned to the *slot* ``r`` of the first round whose
  detectors it touches;
* inside its slot a location acts on the raw syndrome of round ``r`` (``a``)
  and on the persistent syndrome carried into round ``r + 1`` (``b``), and
  shifts the logical observable (``l``);
* between rounds the persistent syndrome is transported by a linear map
  ``T[r + 1]`` (the syndrome action of the noiseless gates in between), which
  is learned from the model itself and then checked against every location.

With this structure the observed history obeys
``s_r = h_r + sum a``, ``h_{r+1} = T[r+1] h_r + sum b``, which makes exact
maximum-likelihood decoding a forward recursion over 81 x 3 states (see
``lrb.correction.forward``). The construction fails loudly if any location
violates the chain assumptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product
from typing import Sequence

import numpy as np

from . import zmod3

SYNDROME_WIDTH = 4
SYNDROME_STATES = 3**SYNDROME_WIDTH
KERNEL_SHAPE = (3, SYNDROME_STATES, SYNDROME_STATES)  # (l, b, a)

_ROUND_LABEL = re.compile(r"^lrb_stab_r(\d+)_w(\d+)$")
_CONST0_LABEL = re.compile(r"^lrb_const0_xstab_(\d+)$")
_TERMINAL_LABEL = re.compile(r"^lrb_term_xstab_(\d+)$")
TERMINAL_WIDTH = 2


@dataclass(frozen=True)
class Mechanism:
    """
    One independent noise location as a distribution over effect vectors.

    Attributes:
        location_index (int): Index into the compact model's location list.
        gate (str): ``N1`` or ``N2``.
        qudit (int): First wire the noise acts on.
        target (int): Second wire for ``N2`` locations, else ``-1``.
        channel (str): SDIM channel tag (``d``, ``f`` or ``p``).
        prob_scale (float): Multiplier of the swept parameter ``p``.
        vectors (np.ndarray): ``(n_v, width)`` non-zero Weyl vectors the
            location can apply once active, each equally likely.
        detector_effects (np.ndarray): ``(n_v, n_det)`` raw detector shifts.
        logical_effects (np.ndarray): ``(n_v,)`` logical shifts.
        unit_detector (np.ndarray): ``(width, n_det)`` unit-component
            detector responses.
        unit_logical (np.ndarray): ``(width,)`` unit-component logical
            responses.
    """

    location_index: int
    gate: str
    qudit: int
    target: int
    channel: str
    prob_scale: float
    vectors: np.ndarray
    detector_effects: np.ndarray
    logical_effects: np.ndarray
    unit_detector: np.ndarray
    unit_logical: np.ndarray

    def activation_probability(self, p: float) -> float:
        """Return the probability that this location applies a non-identity."""
        value = float(p) * self.prob_scale
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid local probability {value} at p={p}")
        return value


def location_vectors(gate: str, channel: str) -> np.ndarray:
    """
    Return the equally likely non-zero Weyl vectors of one noise location.

    The conventions mirror ``lrb.dem_simulation.sample_location_fault_vectors``:
    ``N1 d`` is uniform over the 8 non-zero ``(x, z)`` pairs, ``N1 f`` shifts
    only component 0, ``N1 p`` only component 1, and ``N2`` with the
    generator's depolarizing ``prob_dist`` is uniform over the 80 non-zero
    4-component vectors.

    Args:
        gate (str): ``N1`` or ``N2``.
        channel (str): SDIM channel tag.

    Returns:
        np.ndarray: ``(n_v, width)`` integer array.
    """
    if gate == "N1":
        if channel == "d":
            return np.array(
                [v for v in product(range(3), repeat=2) if any(v)],
                dtype=np.int64,
            )
        if channel == "f":
            return np.array([[1, 0], [2, 0]], dtype=np.int64)
        if channel == "p":
            return np.array([[0, 1], [0, 2]], dtype=np.int64)
        raise ValueError(f"unsupported N1 channel {channel}")
    if gate == "N2":
        return np.array(
            [v for v in product(range(3), repeat=4) if any(v)],
            dtype=np.int64,
        )
    raise ValueError(f"unsupported noise gate {gate}")


def _check_uniform_n2(params: dict) -> None:
    """Raise unless an N2 location uses uniform two-qutrit depolarizing noise."""
    prob_dist = params.get("prob_dist")
    if prob_dist is None:
        return
    tail = np.asarray(prob_dist, dtype=float)[1:]
    if tail.size and not np.allclose(tail, tail[0], rtol=1e-9, atol=1e-15):
        raise ValueError("chain model expects uniform two-qutrit depolarizing noise")


def extract_mechanisms(model) -> list[Mechanism]:
    """
    Extract the location list of a compact DEM as dense effect tables.

    Args:
        model: ``CompactDetectorErrorModel`` built at any non-zero ``p``.

    Returns:
        list[Mechanism]: One entry per active location, in model order.
    """
    n_det = model.num_detectors
    mechanisms: list[Mechanism] = []
    for index, location in enumerate(model.locations):
        if location.gate == "N2":
            _check_uniform_n2(location.params)
        width = location.vector_width
        unit_detector = np.zeros((width, n_det), dtype=np.int64)
        unit_logical = np.zeros(width, dtype=np.int64)
        for component, terms in enumerate(location.component_terms):
            for detector, coefficient in terms.detector:
                unit_detector[component, detector] = coefficient % 3
            for logical, coefficient in terms.logical:
                if logical != 0:
                    raise ValueError("chain model expects a single logical observable")
                unit_logical[component] = coefficient % 3
        vectors = location_vectors(location.gate, location.channel)
        prob_scale = float(location.params.get("prob_scale", 1.0))
        mechanisms.append(
            Mechanism(
                location_index=index,
                gate=location.gate,
                qudit=location.qudit,
                target=location.target,
                channel=location.channel,
                prob_scale=prob_scale,
                vectors=vectors,
                detector_effects=(vectors @ unit_detector) % 3,
                logical_effects=(vectors @ unit_logical) % 3,
                unit_detector=unit_detector,
                unit_logical=unit_logical,
            )
        )
    return mechanisms


@dataclass(frozen=True)
class DetectorLayout:
    """
    Round structure of an LRB detector list.

    Attributes:
        num_rounds (int): Number of syndrome rounds.
        width (int): Detectors per round.
        detector_index (np.ndarray): ``(num_rounds, width)`` detector indices.
        kind (str): ``rounds`` for ancilla-check circuits, ``const0`` for the
            direct terminal X-stabilizer circuits.
        terminal_index (np.ndarray | None): Detector indices of the optional
            noiseless terminal X-stabilizers (variant circuits that read all
            five data qutrits at the end), or ``None``.
    """

    num_rounds: int
    width: int
    detector_index: np.ndarray
    kind: str
    terminal_index: np.ndarray | None = None


def detector_layout(labels: Sequence[str]) -> DetectorLayout:
    """
    Parse detector labels into a round-by-wire layout.

    Args:
        labels (Sequence[str]): Detector labels of a compact DEM.

    Returns:
        DetectorLayout: Parsed layout.
    """
    const0 = [_CONST0_LABEL.match(label) for label in labels]
    if labels and all(const0):
        order = np.argsort([int(match.group(1)) for match in const0])
        return DetectorLayout(1, len(labels), order[None, :].astype(np.int64), "const0")
    entries = []
    terminal: list[tuple[int, int]] = []
    for index, label in enumerate(labels):
        match = _ROUND_LABEL.match(label)
        if match is None:
            term = _TERMINAL_LABEL.match(label)
            if term is None:
                raise ValueError(f"unexpected detector label {label}")
            terminal.append((int(term.group(1)), index))
            continue
        entries.append((int(match.group(1)), int(match.group(2)), index))
    rounds = sorted({entry[0] for entry in entries})
    wires = sorted({entry[1] for entry in entries})
    if rounds != list(range(len(rounds))):
        raise ValueError("detector rounds are not contiguous")
    table = np.full((len(rounds), len(wires)), -1, dtype=np.int64)
    for round_index, wire, index in entries:
        table[round_index, wires.index(wire)] = index
    if np.any(table < 0):
        raise ValueError("missing detector in round layout")
    terminal_index = None
    if terminal:
        terminal.sort()
        if [t[0] for t in terminal] != list(range(TERMINAL_WIDTH)):
            raise ValueError("terminal X-stabilizer detectors must be 0..1")
        terminal_index = np.array([t[1] for t in terminal], dtype=np.int64)
    return DetectorLayout(len(rounds), len(wires), table, "rounds", terminal_index)


@dataclass
class ChainModel:
    """
    Chain-structured DEM of one circuit at one noise strength.

    Attributes:
        p (float): Swept physical noise parameter.
        layout (DetectorLayout): Round structure of the detectors.
        transfer (list[np.ndarray | None]): ``transfer[r]`` maps the
            persistent syndrome at round ``r - 1`` to round ``r``
            (``transfer[0]`` is unused and ``None``).
        transfer_perm (list[np.ndarray | None]): Same maps as permutations of
            the 81 syndrome indices.
        kernels (list[np.ndarray]): ``kernels[r]`` has shape ``(3, 81, 81)``
            indexed ``[l, b, a]``: the joint probability that the faults of
            slot ``r`` shift the logical by ``l``, the carried syndrome by
            ``b`` and the raw syndrome of round ``r`` by ``a``. The last slot
            has no next round: its middle axis has length 1, or 9 when the
            circuit has terminal X-stabilizer detectors, in which case it
            indexes the slot's shift of the terminal X-syndrome.
        slot_of_mechanism (np.ndarray): Slot index per mechanism.
        mechanisms (list[Mechanism]): Underlying mechanisms.
        activation (np.ndarray): Activation probability per mechanism.
        terminal_perm (np.ndarray | None): Map from a round-``m`` persistent
            syndrome index to the terminal X-syndrome index it produces, or
            ``None`` without terminal detectors.
    """

    p: float
    layout: DetectorLayout
    transfer: list
    transfer_perm: list
    kernels: list
    slot_of_mechanism: np.ndarray
    mechanisms: list
    activation: np.ndarray
    diagnostics: dict = field(default_factory=dict)
    terminal_perm: np.ndarray | None = None

    @property
    def num_rounds(self) -> int:
        return self.layout.num_rounds

    @property
    def has_terminal(self) -> bool:
        return self.terminal_perm is not None


def _round_effects(effects: np.ndarray, layout: DetectorLayout) -> np.ndarray:
    """Reshape ``(..., n_det)`` effects into ``(..., rounds, width)``."""
    return effects[..., layout.detector_index]


def learn_transfer_maps(
    mechanisms: Sequence[Mechanism],
    layout: DetectorLayout,
) -> tuple[list, dict]:
    """
    Learn the syndrome transport maps between consecutive rounds.

    For ``r >= 1`` the map ``T[r + 1]`` is fitted from every unit component
    whose first detected round is at most ``r - 1``: such a component is a
    persistent data error at both rounds ``r`` and ``r + 1`` (or already
    gone), so its two raw syndromes are related by the noiseless gate action.
    ``T[1]`` is never needed because no fault precedes slot 0.

    Args:
        mechanisms (Sequence[Mechanism]): Extracted mechanisms.
        layout (DetectorLayout): Detector round structure.

    Returns:
        tuple[list, dict]: Transfer matrices and fit diagnostics.
    """
    rounds = layout.num_rounds
    units = np.concatenate([m.unit_detector for m in mechanisms], axis=0)
    per_round = _round_effects(units, layout)  # (n_units, rounds, width)
    nonzero_round = np.any(per_round != 0, axis=2)
    first = np.where(nonzero_round.any(axis=1), nonzero_round.argmax(axis=1), rounds)
    transfer: list = [None] * rounds
    ranks: dict[int, int] = {}
    for target in range(2, rounds):
        source = target - 1
        rows = first <= source - 1
        x_rows = per_round[rows, source, :]
        y_rows = per_round[rows, target, :]
        keep = np.any(x_rows != 0, axis=1) | np.any(y_rows != 0, axis=1)
        x_rows, y_rows = x_rows[keep], y_rows[keep]
        matrix = zmod3.solve_linear_map(x_rows, y_rows)
        if matrix is None:
            raise ValueError(f"no linear syndrome map from round {source} to {target}")
        ranks[target] = int(np.linalg.matrix_rank(x_rows)) if len(x_rows) else 0
        transfer[target] = matrix
    if rounds >= 2:
        transfer[1] = np.eye(layout.width, dtype=np.int64)
    return transfer, {"transfer_input_rank": ranks}


def learn_terminal_map(mechanisms: Sequence[Mechanism], layout: DetectorLayout) -> np.ndarray | None:
    """
    Learn the linear map from the last round's syndrome to the terminal one.

    Only components first detected before the last round are used: they are
    persistent data errors at the last round, and the noiseless terminal
    readout of all data qutrits sees exactly their X-stabilizer part.

    Args:
        mechanisms (Sequence[Mechanism]): Extracted mechanisms.
        layout (DetectorLayout): Detector layout with terminal detectors.

    Returns:
        np.ndarray | None: ``(2, 4)`` map over Z_3, or ``None`` without
        terminal detectors.
    """
    if layout.terminal_index is None:
        return None
    rounds = layout.num_rounds
    units = np.concatenate([m.unit_detector for m in mechanisms], axis=0)
    per_round = _round_effects(units, layout)
    nonzero_round = np.any(per_round != 0, axis=2)
    first = np.where(nonzero_round.any(axis=1), nonzero_round.argmax(axis=1), rounds)
    rows = first <= rounds - 2
    x_rows = per_round[rows, rounds - 1, :]
    y_rows = units[rows][:, layout.terminal_index]
    if not np.any(rows):
        # Depth-0 circuits have a single round; no earlier fault exists, so
        # the map is irrelevant. Use the X-stabilizer projection guess.
        return np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.int64)
    matrix = zmod3.solve_linear_map(x_rows, y_rows)
    if matrix is None:
        raise ValueError("terminal X-syndrome is not a linear image of the last round")
    return matrix


def _projection_to_perm(matrix: np.ndarray, width_in: int) -> np.ndarray:
    digits = zmod3.index_to_digits(np.arange(3**width_in), width_in)
    return zmod3.digits_to_index((digits @ matrix.T) % 3)


def _matrix_to_perm(matrix: np.ndarray, width: int) -> np.ndarray:
    digits = zmod3.index_to_digits(np.arange(3**width), width)
    mapped = (digits @ matrix.T) % 3
    perm = zmod3.digits_to_index(mapped)
    if len(np.unique(perm)) != perm.size:
        raise ValueError("syndrome transfer map is not invertible")
    return perm


def assign_slots(
    mechanisms: Sequence[Mechanism],
    layout: DetectorLayout,
    transfer: Sequence,
    terminal_map: np.ndarray | None = None,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Assign each mechanism to a slot and verify the chain assumptions.

    Args:
        mechanisms (Sequence[Mechanism]): Extracted mechanisms.
        layout (DetectorLayout): Detector round structure.
        transfer (Sequence): Learned transfer maps.
        terminal_map (np.ndarray | None): Learned last-round-to-terminal map.

    Returns:
        tuple: Slot per mechanism, and per mechanism the ``(a, b)`` effects
        for each of its vectors. In the last slot ``b`` is the terminal
        X-syndrome shift (two digits) or an empty ``(n_v, 0)`` array.

    Raises:
        ValueError: If any vector's later-round (or terminal) effects are not
            the transported image of its slot ``r + 1`` effect.
    """
    rounds = layout.num_rounds
    slots = np.empty(len(mechanisms), dtype=np.int64)
    ab: list[tuple[np.ndarray, np.ndarray]] = []
    for index, mechanism in enumerate(mechanisms):
        per_round = _round_effects(mechanism.detector_effects, layout)
        touched = np.flatnonzero(np.any(per_round != 0, axis=(0, 2)))
        slot = int(touched[0]) if touched.size else rounds - 1
        slots[index] = slot
        a = per_round[:, slot, :]
        if slot + 1 < rounds:
            b = per_round[:, slot + 1, :]
        elif layout.terminal_index is not None:
            b = mechanism.detector_effects[:, layout.terminal_index] % 3
        else:
            b = np.zeros((a.shape[0], 0), dtype=np.int64)
        carried = b.copy()
        for later in range(slot + 2, rounds):
            carried = (carried @ transfer[later].T) % 3
            if np.any(carried != per_round[:, later, :]):
                raise ValueError(
                    f"mechanism {index} ({mechanism.gate} on {mechanism.qudit}) "
                    f"breaks the chain assumption at round {later}"
                )
        if layout.terminal_index is not None and slot < rounds - 1:
            expected = (per_round[:, rounds - 1, :] @ terminal_map.T) % 3
            if np.any(expected != mechanism.detector_effects[:, layout.terminal_index] % 3):
                raise ValueError(
                    f"mechanism {index} ({mechanism.gate} on {mechanism.qudit}) "
                    "breaks the terminal-readout assumption"
                )
        ab.append((a, b))
    return slots, ab


def build_kernels(
    mechanisms: Sequence[Mechanism],
    slots: np.ndarray,
    ab: Sequence[tuple[np.ndarray, np.ndarray]],
    activation: np.ndarray,
    num_rounds: int,
    final_b_width: int = 0,
) -> list[np.ndarray]:
    """
    Convolve the mechanisms of every slot into its ``(l, b, a)`` kernel.

    The convolution is done directly in probability space (no Fourier
    transform), so every kernel entry is an exact non-negative sum.

    Args:
        mechanisms (Sequence[Mechanism]): Extracted mechanisms.
        slots (np.ndarray): Slot per mechanism.
        ab (Sequence): ``(a, b)`` round effects per mechanism.
        activation (np.ndarray): Activation probability per mechanism.
        num_rounds (int): Number of syndrome rounds.
        final_b_width (int): Digits of the last slot's ``b`` axis (0 without
            terminal detectors, 2 with them).

    Returns:
        list[np.ndarray]: One ``(3, 81, 81)`` kernel per slot; the last one
        is ``(3, 1, 81)`` or ``(3, 9, 81)`` (see ``ChainModel``).
    """
    width = SYNDROME_WIDTH
    sub = zmod3.subtraction_table(width)
    kernels = []
    for slot in range(num_rounds):
        indices = np.flatnonzero(slots == slot)
        b_width = width if slot < num_rounds - 1 else final_b_width
        sub_b = sub if b_width == width else zmod3.subtraction_table(b_width)
        kernel = np.zeros((3, 3**b_width, SYNDROME_STATES), dtype=np.float64)
        kernel[0, 0, 0] = 1.0
        for index in indices:
            q = float(activation[index])
            if q == 0.0:
                continue
            mechanism = mechanisms[index]
            a, b = ab[index]
            a_idx = zmod3.digits_to_index(a)
            b_idx = zmod3.digits_to_index(b)
            l_val = mechanism.logical_effects % 3
            atoms: dict[tuple[int, int, int], float] = {}
            weight = q / len(mechanism.vectors)
            for key in zip(l_val.tolist(), b_idx.tolist(), a_idx.tolist()):
                atoms[key] = atoms.get(key, 0.0) + weight
            zero_weight = (1.0 - q) + atoms.pop((0, 0, 0), 0.0)
            updated = zero_weight * kernel
            if atoms:
                keys = np.array(list(atoms.keys()), dtype=np.int64)
                weights = np.array(list(atoms.values()), dtype=np.float64)
                l_perm = (np.arange(3)[None, :] - keys[:, 0:1]) % 3
                b_perm = sub_b[:, keys[:, 1]].T
                a_perm = sub[:, keys[:, 2]].T
                shifted = kernel[
                    l_perm[:, :, None, None],
                    b_perm[:, None, :, None],
                    a_perm[:, None, None, :],
                ]
                updated += np.tensordot(weights, shifted, axes=1)
            kernel = updated
        kernels.append(kernel)
    return kernels


@dataclass
class ChainStructure:
    """
    Noise-strength-independent part of a chain model.

    Attributes:
        layout (DetectorLayout): Detector round structure.
        transfer (list): Learned syndrome transfer matrices.
        transfer_perm (list): Transfer maps as index permutations.
        slots (np.ndarray): Slot per mechanism.
        ab (list): Per-mechanism ``(a, b)`` round effects of each vector.
        mechanisms (list[Mechanism]): Underlying mechanisms.
        diagnostics (dict): Fit and slot diagnostics.
    """

    layout: DetectorLayout
    transfer: list
    transfer_perm: list
    slots: np.ndarray
    ab: list
    mechanisms: list
    diagnostics: dict
    terminal_map: np.ndarray | None = None
    terminal_perm: np.ndarray | None = None

    def at(self, p: float) -> ChainModel:
        """Build the chain model at swept noise strength ``p``."""
        activation = np.array([m.activation_probability(p) for m in self.mechanisms])
        final_b_width = 0 if self.layout.terminal_index is None else TERMINAL_WIDTH
        kernels = build_kernels(self.mechanisms, self.slots, self.ab, activation,
                                self.layout.num_rounds, final_b_width)
        return ChainModel(
            p=float(p),
            layout=self.layout,
            transfer=self.transfer,
            transfer_perm=self.transfer_perm,
            kernels=kernels,
            slot_of_mechanism=self.slots,
            mechanisms=self.mechanisms,
            activation=activation,
            diagnostics=self.diagnostics,
            terminal_perm=self.terminal_perm,
        )


def build_chain_structure(model, mechanisms: list[Mechanism] | None = None) -> ChainStructure:
    """
    Learn transfer maps and slot assignments of one compact DEM.

    Args:
        model: ``CompactDetectorErrorModel`` built at any non-zero ``p`` (only
            the response structure and ``prob_scale`` metadata are used).
        mechanisms (list[Mechanism] | None): Pre-extracted mechanisms.

    Returns:
        ChainStructure: Structure reusable at every noise strength.
    """
    if mechanisms is None:
        mechanisms = extract_mechanisms(model)
    layout = detector_layout(model.detector_labels)
    if layout.kind != "rounds":
        raise ValueError("use build_const0_model for const=0 circuits")
    if layout.width != SYNDROME_WIDTH:
        raise ValueError("chain model expects four detectors per round")
    transfer, diagnostics = learn_transfer_maps(mechanisms, layout)
    terminal_map = learn_terminal_map(mechanisms, layout)
    slots, ab = assign_slots(mechanisms, layout, transfer, terminal_map)
    perms = [None] + [
        _matrix_to_perm(matrix, layout.width) for matrix in transfer[1:]
    ]
    terminal_perm = (None if terminal_map is None
                     else _projection_to_perm(terminal_map, layout.width))
    diagnostics["slot_counts"] = np.bincount(slots, minlength=layout.num_rounds).tolist()
    return ChainStructure(layout, transfer, perms, slots, ab, list(mechanisms), diagnostics,
                          terminal_map, terminal_perm)


def build_chain_model(model, p: float, mechanisms: list[Mechanism] | None = None) -> ChainModel:
    """
    Build the chain model of one compact DEM at noise strength ``p``.

    Args:
        model: ``CompactDetectorErrorModel`` built at any non-zero ``p`` (only
            the response structure and ``prob_scale`` metadata are used).
        p (float): Swept physical noise parameter to evaluate.
        mechanisms (list[Mechanism] | None): Pre-extracted mechanisms.

    Returns:
        ChainModel: Kernels, transfer maps, and slot assignment.
    """
    return build_chain_structure(model, mechanisms).at(p)


@dataclass
class Const0Model:
    """
    Joint distribution of the two terminal X-stabilizers and the logical shift.

    Attributes:
        p (float): Swept physical noise parameter.
        joint (np.ndarray): ``(3, 9)`` array indexed ``[l, syndrome_index]``.
        mechanisms (list[Mechanism]): Underlying mechanisms.
        activation (np.ndarray): Activation probability per mechanism.
        layout (DetectorLayout): Detector layout (one round of width 2).
    """

    p: float
    joint: np.ndarray
    mechanisms: list
    activation: np.ndarray
    layout: DetectorLayout


def build_const0_model(model, p: float, mechanisms: list[Mechanism] | None = None) -> Const0Model:
    """
    Build the exact joint (logical, terminal X-syndrome) distribution.

    Args:
        model: ``CompactDetectorErrorModel`` of a ``const=0`` circuit.
        p (float): Swept physical noise parameter.
        mechanisms (list[Mechanism] | None): Pre-extracted mechanisms.

    Returns:
        Const0Model: Exact joint distribution.
    """
    if mechanisms is None:
        mechanisms = extract_mechanisms(model)
    layout = detector_layout(model.detector_labels)
    if layout.kind != "const0":
        raise ValueError("not a const=0 detector layout")
    activation = np.array([m.activation_probability(p) for m in mechanisms])
    joint = const0_joint_from_mechanisms(mechanisms, layout, p)
    return Const0Model(float(p), joint, mechanisms, activation, layout)


def const0_joint_from_mechanisms(
    mechanisms: Sequence[Mechanism],
    layout: DetectorLayout,
    p: float,
) -> np.ndarray:
    """
    Exact joint of the logical shift and the terminal X-syndrome.

    Args:
        mechanisms (Sequence[Mechanism]): Mechanisms of a const0 circuit.
        layout (DetectorLayout): Its one-round detector layout.
        p (float): Swept noise parameter.

    Returns:
        np.ndarray: ``(3, 3**width)`` array indexed ``[l, syndrome_index]``.
    """
    width = layout.width
    sub = zmod3.subtraction_table(width)
    joint = np.zeros((3, 3**width), dtype=np.float64)
    joint[0, 0] = 1.0
    for mechanism in mechanisms:
        q = mechanism.activation_probability(p)
        if q == 0.0:
            continue
        s_idx = zmod3.digits_to_index(mechanism.detector_effects[:, layout.detector_index[0]])
        atoms: dict[tuple[int, int], float] = {}
        weight = q / len(mechanism.vectors)
        for key in zip((mechanism.logical_effects % 3).tolist(), s_idx.tolist()):
            atoms[key] = atoms.get(key, 0.0) + weight
        updated = ((1.0 - q) + atoms.pop((0, 0), 0.0)) * joint
        for (l_shift, s_shift), w in atoms.items():
            updated += w * joint[np.ix_((np.arange(3) - l_shift) % 3, sub[:, s_shift])]
        joint = updated
    return joint
