"""Tests for the Steane [[7,1,3]] LRB circuits and decoders (work package K).

Run with ``module load python/3.11`` and
``OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest
-q tests/test_qubit512_steane.py``.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

stim = pytest.importorskip("stim")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import code as C  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402
from lrb_qubit import steane_circuits as SV  # noqa: E402
from lrb_qubit import steane_code as SC  # noqa: E402

NAIVE = SC.SteaneCode("naive")
FLAG = SC.SteaneCode("flag")
SEQUENCES = E.clifford_sequences(3, 22, 20261028)
VARIANTS = [(NAIVE, "sequential"), (NAIVE, "parallel"),
            (FLAG, "sequential"), (FLAG, "parallel")]


# ----------------------------------------------------------------------
# The code
# ----------------------------------------------------------------------
def test_stabilizers_and_logicals():
    stabs = [NAIVE.stabilizer_pauli(s) for s in range(6)]
    for a, b in itertools.combinations(stabs, 2):
        assert a.commutes(b)
    x_l, z_l = NAIVE.logical_pauli("X"), NAIVE.logical_pauli("Z")
    assert all(s.commutes(x_l) and s.commutes(z_l) for s in stabs)
    assert not x_l.commutes(z_l)
    matrix = np.array([np.concatenate(s.to_numpy()) for s in stabs], float)
    assert np.linalg.matrix_rank(matrix) == 6
    # X^7 and Z^7 are the same logical classes as the weight-3 operators.
    assert NAIVE.logical_class(stim.PauliString("XXXXXXX")) == ("X", 1)
    assert NAIVE.logical_class(stim.PauliString("ZZZZZZZ")) == ("Z", 1)


def test_code_distance_three_and_weight_two_equivalence():
    for weight in (1, 2):
        for qubits in itertools.combinations(range(7), weight):
            for paulis in itertools.product("XYZ", repeat=weight):
                p = stim.PauliString(7)
                for q, ch in zip(qubits, paulis):
                    p[q] = ch
                cls = NAIVE.logical_class(p)
                assert cls is None, (p, cls)  # never an undetected logical
    # Every weight-2 X error is a weight-1 X error times X_L (why hooks are
    # fatal for naive extraction).
    for (c, d), e in SC.weight_two_equivalences().items():
        p = stim.PauliString(7)
        for q in (c, d, e):
            p[q] = "X"
        assert NAIVE.logical_class(p) == ("X", 1)


def _logical_map(code, ops):
    tableau = code.data_tableau(ops)
    return (code.logical_class(tableau(code.logical_pauli("X"))),
            code.logical_class(tableau(code.logical_pauli("Z"))))


def _stabilizers_preserved(code, ops):
    tableau = code.data_tableau(ops)
    for s in range(6):
        image = tableau(code.stabilizer_pauli(s))
        if code.logical_class(image) != ("I", 1):
            return False
    return True


def test_transversal_gate_conventions():
    h = [("H", (q,)) for q in range(7)]
    s = [("S", (q,)) for q in range(7)]
    s_dag = [("S_DAG", (q,)) for q in range(7)]
    assert _logical_map(NAIVE, h) == (("Z", 1), ("X", 1))
    # Transversal S is logical S^dagger; logical S is transversal S_DAG.
    assert _logical_map(NAIVE, s) == (("Y", -1), ("Z", 1))
    assert _logical_map(NAIVE, s_dag) == (("Y", 1), ("Z", 1))
    assert NAIVE.logical_gate_operations("P") == s_dag
    for ops in (h, s, s_dag):
        assert _stabilizers_preserved(NAIVE, ops)
    assert _logical_map(NAIVE, NAIVE.logical_gate_operations("X")) == (
        ("X", 1), ("Z", -1))
    assert _logical_map(NAIVE, NAIVE.logical_gate_operations("Z")) == (
        ("X", -1), ("Z", 1))


def _lead_logical_map(descriptor):
    """Logical action of a descriptor in the lead's [[5,1,2]] model."""
    tableau = C.data_tableau(C.logical_clifford_operations(descriptor))
    x = stim.PauliString(C.LOGICAL_X)
    z = stim.PauliString(C.LOGICAL_Z)

    def cls(pauli):
        for name, logical in (("X", x), ("Y", x * z * 1j), ("Z", z)):
            rest = pauli * logical
            try:
                subset = C.decompose_in_stabilizers(rest)
            except ValueError:
                continue
            product = stim.PauliString(5)
            for k in subset:
                product *= C.stabilizer_pauli(k)
            return name, (1 if rest == product else -1)
        return None

    return cls(tableau(x)), cls(tableau(z))


@pytest.mark.parametrize("pauli_gates", SC.PAULI_GATES)
def test_clifford_catalog_matches_lead_code(pauli_gates):
    """Each descriptor implements the same logical Clifford as in the lead's
    [[5,1,2]] model, and its token-by-token inverse undoes it."""
    code = SC.SteaneCode("naive", pauli_gates=pauli_gates)
    images = set()
    for descriptor in C.SINGLE_QUBIT_CLIFFORDS:
        ops = code.logical_clifford_operations(descriptor)
        assert _stabilizers_preserved(code, ops)
        mine = _logical_map(code, ops)
        assert mine == _lead_logical_map(descriptor), descriptor
        images.add(mine)
        both = ops + code.logical_clifford_operations(descriptor,
                                                      inverse=True)
        assert _logical_map(code, both) == (("X", 1), ("Z", 1))
    assert len(images) == 24


def test_encoder_prepares_plus_logical():
    sim = stim.TableauSimulator()
    circuit = stim.Circuit()
    for gate, targets in NAIVE.encoding_operations():
        circuit.append(gate, list(targets))
    sim.do(circuit)
    for s in range(6):
        assert sim.peek_observable_expectation(NAIVE.stabilizer_pauli(s)) == 1
    assert sim.peek_observable_expectation(NAIVE.logical_pauli("X")) == 1


# ----------------------------------------------------------------------
# Circuits
# ----------------------------------------------------------------------
@pytest.mark.parametrize("code,schedule", VARIANTS)
def test_noiseless_circuits_are_deterministic(code, schedule):
    for depth in (0, 1, 3, 6):
        lc = SV.build_lrb_circuit(code, SEQUENCES[0], depth, 0.0,
                                  schedule=schedule)
        dets, obs = lc.circuit.compile_detector_sampler(seed=1).sample(
            64, separate_observables=True)
        assert not dets.any() and not obs.any()
        noisy = SV.build_lrb_circuit(code, SEQUENCES[0], depth, 1e-3,
                                     schedule=schedule)
        noisy.circuit.detector_error_model()  # raises if non-deterministic
        expected = (6 + (6 if code.extraction == "flag" else 0)) * (
            depth + 1) + 3
        assert noisy.num_detectors == expected


def test_measurement_layers_per_round():
    def idle_layers(circuit):
        return sum(1 for inst in circuit.flattened()
                   if inst.name == "DEPOLARIZE1"
                   and np.isclose(inst.gate_args_copy()[0], 2e-3)
                   and len(inst.targets_copy()) > 2)
    for code in (NAIVE, FLAG):
        seq = SV.build_lrb_circuit(code, SEQUENCES[0], 0, 1e-3,
                                   schedule="sequential")
        par = SV.build_lrb_circuit(code, SEQUENCES[0], 0, 1e-3,
                                   schedule="parallel")
        # Terminal round only (no reset): 6 vs 1 measurement layers.
        assert idle_layers(seq.circuit) == 6
        assert idle_layers(par.circuit) == 1


@pytest.mark.parametrize("code,schedule", VARIANTS)
def test_raw_syndromes_match_measurement_records(code, schedule):
    lc = SV.build_lrb_circuit(code, SEQUENCES[1], 4, 4e-3, schedule=schedule)
    meas = lc.circuit.compile_sampler(seed=5).sample(2000)
    dets, _ = lc.circuit.compile_m2d_converter().convert(
        measurements=meas, separate_observables=True)
    raw = lc.raw_syndromes(dets)
    assert np.array_equal(raw, meas[:, lc.syndrome_records])
    if code.extraction == "flag":
        assert np.array_equal(lc.raw_flags(dets), meas[:, lc.flag_records])


@pytest.mark.parametrize("schedule", SV.SCHEDULES)
def test_circuit_distance(schedule):
    """Naive extraction: distance 2 (hooks); flagged: distance 3."""
    for code, expected in ((NAIVE, 2), (FLAG, 3)):
        for depth in (0, 2):
            lc = SV.build_lrb_circuit(code, SEQUENCES[0], depth, 1e-3,
                                      schedule=schedule)
            report = SV.distance_report(lc.circuit)
            assert report["exact_lower_bound"] == expected
            assert report["stim_search"] == expected
            assert report["shortest_graphlike"] == expected


def test_first_order_structure():
    for schedule in SV.SCHEDULES:
        flag = SV.build_lrb_circuit(FLAG, SEQUENCES[2], 4, 1e-6,
                                    schedule=schedule)
        summary = SV.first_order_summary(flag.circuit.detector_error_model())
        assert summary["U1"] == 0 and summary["E1"] == 0
        assert summary["A2"] == 0 and summary["R1"] > 0
        naive = SV.build_lrb_circuit(NAIVE, SEQUENCES[2], 4, 1e-6,
                                     schedule=schedule)
        summary = SV.first_order_summary(naive.circuit.detector_error_model())
        assert summary["U1"] == 0 and summary["E1"] > 0


# ----------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------
def test_slot_mld_equals_lead_transfer_matrix_without_flags():
    lc = SV.build_lrb_circuit(NAIVE, SEQUENCES[2], 4, 2e-3)
    dem = lc.circuit.detector_error_model()
    slots, _ = lc.detector_slots()
    lead = M.TransferMatrixMLD(dem, slots)
    mine = SV.mld_for(lc, dem)
    dets, _ = lc.circuit.compile_detector_sampler(seed=5).sample(
        5000, separate_observables=True)
    a = lead.log_likelihood_ratio(dets)
    b = mine.log_likelihood_ratio(dets)
    finite = np.isfinite(a)
    assert np.array_equal(finite, np.isfinite(b))
    assert np.max(np.abs(a[finite] - b[finite])) < 1e-9


def _restrict(circuit, keep):
    out = stim.Circuit()
    index = 0
    for inst in circuit.flattened():
        if inst.name == "DETECTOR":
            if keep[index]:
                out.append(inst)
            index += 1
        else:
            out.append(inst)
    return out


@pytest.mark.parametrize("code,depth,schedule", [
    (FLAG, 0, "sequential"), (FLAG, 0, "parallel"), (NAIVE, 1, "sequential"),
    (NAIVE, 1, "parallel")])
def test_slot_mld_equals_brute_force(code, depth, schedule):
    lc = SV.build_lrb_circuit(code, SEQUENCES[1], depth, 5e-3,
                              schedule=schedule)
    dem = lc.circuit.detector_error_model()
    assert dem.num_detectors <= 15
    dets, _ = lc.circuit.compile_detector_sampler(seed=3).sample(
        3000, separate_observables=True)
    llr = SV.mld_for(lc, dem).log_likelihood_ratio(dets)
    exact = SV.brute_force_llr(dem, dets)
    finite = np.isfinite(exact)
    assert np.array_equal(finite, np.isfinite(llr))
    assert np.max(np.abs(llr[finite] - exact[finite])) < 1e-9


def test_slot_mld_flag_transition_inside_the_circuit():
    """Round-0 flags conditioning the step into round 1 (not the final
    slot): depth 1, keeping round-0 syndromes and flags and round-1
    syndromes (18 detectors, brute-forceable)."""
    lc = SV.build_lrb_circuit(FLAG, SEQUENCES[2], 1, 5e-3,
                              schedule="sequential")
    keep = np.zeros(lc.num_detectors, dtype=bool)
    keep[lc.syndrome_columns[0]] = True
    keep[lc.flag_columns[0]] = True
    keep[lc.syndrome_columns[1]] = True
    circuit = _restrict(lc.circuit, keep)
    dem = circuit.detector_error_model()
    assert dem.num_detectors == 18
    slots_full, trans_full = lc.detector_slots()
    slots, trans = slots_full[keep], trans_full[keep]
    decoder = SV.SlotMLD(dem, slots, trans)
    assert decoder.num_slots == 2 and 1 in decoder.groups
    dets, _ = circuit.compile_detector_sampler(seed=9).sample(
        3000, separate_observables=True)
    llr = decoder.log_likelihood_ratio(dets)
    exact = SV.brute_force_llr(dem, dets)
    finite = np.isfinite(exact)
    assert dets[:, 6:12].any()  # some shots raise flags
    assert np.max(np.abs(llr[finite] - exact[finite])) < 1e-9


def test_flagged_mld_corrects_every_single_fault():
    """Distance 3 in practice: exact MLD decodes every single DEM mechanism
    of the flagged circuit correctly (and not all of the naive one)."""
    for code, all_correct in ((FLAG, True), (NAIVE, False)):
        lc = SV.build_lrb_circuit(code, SEQUENCES[0], 3, 1e-4,
                                  schedule="parallel")
        dem = lc.circuit.detector_error_model()
        decoder = SV.mld_for(lc, dem)
        mechs = [m for m in M.dem_mechanisms(dem) if m.probability > 0]
        dense = np.zeros((len(mechs), dem.num_detectors), dtype=bool)
        for i, m in enumerate(mechs):
            dense[i, list(m.detectors)] = True
        flips = np.array([m.flips_observable for m in mechs])
        wrong = (decoder.log_likelihood_ratio(dense) < 0) != flips
        assert (not wrong.any()) == all_correct


# ----------------------------------------------------------------------
# Physical RB and counting
# ----------------------------------------------------------------------
def test_physical_rb_exact_formula_matches_sampling():
    seq = SEQUENCES[0]
    depth, p_rb = 10, 0.05
    exact = SV.physical_rb_fidelity(seq, depth, p_rb)
    circuit = Q.build_physical_rb_circuit(seq, depth, p_rb)
    _, obs = circuit.compile_detector_sampler(seed=4).sample(
        200_000, separate_observables=True)
    sampled = abs(1 - 2 * obs[:, 0].mean())
    sigma = 2 * np.sqrt(obs[:, 0].mean() * (1 - obs[:, 0].mean()) / 200_000)
    assert abs(sampled - exact) < 5 * sigma
    # Ensemble decay: ~0.339 p_RB error per step.
    eps = (1 - SV.physical_rb_decay(1e-3)) / 2
    assert 0.33e-3 < eps < 0.35e-3


def test_run_sequence_counts_are_consistent():
    result = SV.run_sequence(FLAG, "parallel", 2e-3, SEQUENCES[0], [0, 2],
                             4000, decode_shots=2000, seed=3)
    for name, table in result.tables.items():
        assert (table.accepted <= table.shots).all()
        assert (table.errors <= table.accepted).all()
    unif1 = result.tables["lrbd_unif_1"]
    final = result.tables["lrbd_unif_1_final"]
    noflag = result.tables["lrbd_unif_1_noflag"]
    assert (final.accepted <= unif1.accepted).all()
    assert (unif1.accepted <= noflag.accepted).all()
    assert result.tables["mld"].shots.sum() == 2 * 2000
    hist = result.gap_hist["mld"]
    assert hist.sum() == 2 * 2000
    assert hist[..., 1].sum() == result.tables["mld"].errors.sum()
