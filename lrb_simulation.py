"""Simulation and post-processing utilities for logical RB workflows."""

from __future__ import annotations

import ast
import csv
import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sdim.circuit import Circuit
from sdim.circuit_io import read_circuit
from sdim.program import Program

from experiment_setup import ExperimentSetupManager


NORMAL_RB_SHOTS = 10000


class NoiseModelUtils:
    """
    Utilities for normalizing and validating noise-model parameters.

    Attributes:
        None: This class is stateless and stores no persistent attributes.

    Methods:
        two_qudit_depol_prob_dist(dimension, prob): Build a full two-qudit
            depolarizing probability distribution.
        ensure_noise_params(circuit, default_prob): Ensure N2 operations carry
            a valid ``prob_dist`` payload.
    """

    @staticmethod
    def two_qudit_depol_prob_dist(dimension: int, prob: float):
        """
        Two qudit depol prob dist.
        
        Args:
            dimension (Any): Input argument.
            prob (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        n = dimension**4
        if n <= 1:
            return [1.0]
        if prob < 0.0 or prob > 1.0:
            raise ValueError(f"Invalid depolarizing prob {prob}")
        tail = prob / (n - 1)
        dist = [tail] * n
        dist[0] = 1.0 - prob
        return dist

    @staticmethod
    def ensure_noise_params(circuit, default_prob: float = None):
        """
        Ensure noise params.
        
        Args:
            circuit (Any): Input argument.
            default_prob (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        for op in circuit.operations:
            if op.gate_name != 'N2' and getattr(op, 'gate_id', None) != 18:
                continue
            if op.params is None:
                op.params = {}

            prob_dist = op.params.get('prob_dist', None)
            if isinstance(prob_dist, str):
                try:
                    parsed = ast.literal_eval(prob_dist)
                    op.params['prob_dist'] = parsed
                    prob_dist = parsed
                except Exception:
                    prob_dist = None

            if isinstance(
                    prob_dist,
                (list, tuple)) and len(prob_dist) == circuit.dimension**4:
                continue

            prob = op.params.get('prob', default_prob)
            if prob is None:
                raise ValueError(
                    "N2 gate missing 'prob' and 'prob_dist' params; "
                    "cannot expand."
                )
            op.params['prob_dist'] = NoiseModelUtils.two_qudit_depol_prob_dist(
                circuit.dimension, float(prob))


@dataclass(frozen=True)
class LRBUnpackSpec:
    """
    Specification describing logical/stabilizer unpack behavior.

    Attributes:
        stabilizer_wires (tuple[int, ...]): Wires measured for stabilizer
            checks.
        logical_measurement_wires (tuple[int, ...]): Wires used for logical
            readout.
        logical_outcome_fn (Callable[[list[int], int], int]): Function mapping
            logical measurements and depth to an integer logical outcome.
        check_stride (int): Round-to-round stride in measurement records.
        check_round_start (int): First check round index to inspect.
        check_rounds_offset (int): Offset applied to depth for round count.
        logical_measurement_round (int): Round index used for logical readout.

    Methods:
        This dataclass is declarative and defines no custom methods.
    """

    stabilizer_wires: tuple[int, ...]
    logical_measurement_wires: tuple[int, ...]
    logical_outcome_fn: Callable[[list[int], int], int]
    check_stride: int = 2
    check_round_start: int = 0
    check_rounds_offset: int = 1
    logical_measurement_round: int = 0


@dataclass
class LRBSimulationEngine:
    """
    Object-oriented facade for RB/LRB simulation workflows.

    Attributes:
        normal_rb_shots (int): Default shot count used by physical RB runs.

    Methods:
        ensure_noise_parameters(circuit, default_prob): Normalize N2 noise
            parameters in one circuit.
        run_lrb(...): Execute logical RB simulations for prepared circuits.
        run_rb(...): Execute physical RB simulations for prepared circuits.
        evaluate_uniform_postselection(...): Build uniform-interval acceptance
            table.
        evaluate_constant_postselection(...): Build constant-count acceptance
            table.
        apply_postselection(...): Apply acceptance decisions to measurement
            records.
        run_round(...): Execute one resumable probability-index experiment
            round.
    """

    normal_rb_shots: int = NORMAL_RB_SHOTS

    def ensure_noise_parameters(self,
                                circuit: Circuit,
                                default_prob: float | None = None) -> None:
        """
        Ensure noise parameters.
        
        Args:
            circuit (Any): Input argument.
            default_prob (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        NoiseModelUtils.ensure_noise_params(circuit, default_prob=default_prob)

    def run_lrb(self,
                experiments,
                depths: list[int],
                shots: int,
                unpack_func: Callable | None = None,
                partial_progress_folder: str = "./prog"):
        """
        Run lrb.
        
        Args:
            experiments (Any): Input argument.
            depths (Any): Input argument.
            shots (Any): Input argument.
            unpack_func (Any): Input argument.
            partial_progress_folder (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        if unpack_func is None:
            raise ValueError("run_lrb requires an explicit unpack_func.")
        return LRBSimulationPipeline.LRB(
            experiments=experiments,
            depths=depths,
            shots=shots,
            unpack_func=unpack_func,
            partial_progress_folder=partial_progress_folder,
        )

    def run_rb(self,
               experiments,
               depths: list[int],
               shots: int | None = None) -> np.ndarray:
        """
        Run rb.
        
        Args:
            experiments (Any): Input argument.
            depths (Any): Input argument.
            shots (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        actual_shots = self.normal_rb_shots if shots is None else shots
        return LRBSimulationPipeline.RB(experiments=experiments,
                                        depths=depths,
                                        shots=actual_shots)

    def evaluate_uniform_postselection(
        self,
        stabilizer_check_record: np.ndarray,
        depths: list[int],
        interval: int,
    ):
        """
        Evaluate uniform postselection.
        
        Args:
            stabilizer_check_record (Any): Input argument.
            depths (Any): Input argument.
            interval (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        return LRBSimulationPipeline.postselect_uniform_interval(
            stabilizer_check_record, depths, interval)

    def evaluate_constant_postselection(self,
                                        stabilizer_check_record: np.ndarray,
                                        depths: list[int],
                                        num_checks: int):
        """
        Evaluate constant postselection.
        
        Args:
            stabilizer_check_record (Any): Input argument.
            depths (Any): Input argument.
            num_checks (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        return LRBSimulationPipeline.postselect_constant_number(
            stabilizer_check_record, depths, num_checks)

    def apply_postselection(self, measurement_record: np.ndarray,
                            decision_table: np.ndarray):
        """
        Apply postselection.
        
        Args:
            measurement_record (Any): Input argument.
            decision_table (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        return LRBSimulationPipeline.postselect_record(measurement_record,
                                                       decision_table)

    def run_round(
        self,
        stab_checks_const,
        stab_checks_unif,
        batch_size: int,
        error_prob: float,
        error_prob_ind: int,
        num_cliff_seq: int,
        depths: list[int],
        num_shots: int,
        filter_trivial_shots: bool,
        lrb_experiment_folder_path: str,
        rb_experiment_folder_path: str,
        lrb_results_folder_path: str,
        rb_results_folder_path: str,
        partial_progress_folder_path: str,
        unpack_func: Callable | None = None,
        logical_dimension: int = 3,
    ) -> int:
        """
        Run round.
        
        Args:
            stab_checks_const (Any): Input argument.
            stab_checks_unif (Any): Input argument.
            batch_size (Any): Input argument.
            error_prob (Any): Input argument.
            error_prob_ind (Any): Input argument.
            num_cliff_seq (Any): Input argument.
            depths (Any): Input argument.
            num_shots (Any): Input argument.
            filter_trivial_shots (Any): Input argument.
            lrb_experiment_folder_path (Any): Input argument.
            rb_experiment_folder_path (Any): Input argument.
            lrb_results_folder_path (Any): Input argument.
            rb_results_folder_path (Any): Input argument.
            partial_progress_folder_path (Any): Input argument.
            unpack_func (Any): Input argument.
            logical_dimension (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        if unpack_func is None:
            raise ValueError("run_round requires an explicit unpack_func.")
        return LRBSimulationPipeline.run_LRB_round(
            stab_checks_const=stab_checks_const,
            stab_checks_unif=stab_checks_unif,
            BATCH_SIZE=batch_size,
            error_prob=error_prob,
            error_prob_ind=error_prob_ind,
            num_cliff_seq=num_cliff_seq,
            depths=depths,
            num_shots=num_shots,
            filter_trivial_shots=filter_trivial_shots,
            LRB_experiment_folder_path=lrb_experiment_folder_path,
            RB_experiment_folder_path=rb_experiment_folder_path,
            LRB_results_folder_path=lrb_results_folder_path,
            RB_results_folder_path=rb_results_folder_path,
            partial_progress_folder_path=partial_progress_folder_path,
            unpack_func=unpack_func,
            logical_dimension=logical_dimension,
        )


class LRBSimulationPipeline:
    """
    Static helpers for simulation, postselection, and statistics.

    Attributes:
        None: This class is stateless and stores no persistent attributes.

    Methods:
        unpack_measurement_results_from_spec(...): Generic measurement
            unpacker.
        LRB/RB(...): Execute logical or physical RB simulations.
        postselect_* methods: Build and apply stabilizer-based acceptance
            tables.
        extract_* and *_stats methods: Convert raw records into summary
            statistics.
        run_LRB_round(...): Execute one full resumable LRB/RB round.
    """
    @staticmethod
    def unpack_measurement_results_from_spec(
        results,
        depth: int,
        shots: int,
        spec: LRBUnpackSpec,
    ) -> tuple[list[list[int]], list[int]]:
        """
        Unpack stabilizer checks and logical outcomes from simulator results.

        Args:
            results (Any): Raw simulator measurement tensor.
            depth (int): Benchmark depth for the current circuit.
            shots (int): Number of simulated shots.
            spec (LRBUnpackSpec): Wire and indexing specification used to
                unpack results.

        Returns:
            tuple[list[list[int]], list[int]]: Stabilizer-pass decisions per
            shot and logical outcomes per shot.

        Raises:
            IndexError: If unpack indices are incompatible with result shape.
            ValueError: If logical outcome mapping cannot produce integers.
        """
        accept_decisions: list[list[int]] = []
        measurement_values: list[int] = []
        check_round_limit = depth + spec.check_rounds_offset
    
        for shot_idx in range(shots):
            shot_checks: list[int] = []
    
            for check_round in range(
                spec.check_round_start, check_round_limit
            ):
                check_offset = check_round * spec.check_stride
                stab_values = [
                    results[wire][check_offset][shot_idx].measurement_value
                    for wire in spec.stabilizer_wires
                ]
                shot_checks.append(
                    int(all(value == 0 for value in stab_values))
                )
    
            logical_measurements = [
                results[wire][spec.logical_measurement_round][shot_idx]
                .measurement_value for wire in spec.logical_measurement_wires
            ]
            measurement_values.append(
                int(spec.logical_outcome_fn(logical_measurements, depth)))
            accept_decisions.append(shot_checks)
    
        return accept_decisions, measurement_values
    @staticmethod
    def LRB(
            experiments,
            depths: list[int],
            shots: int,
            unpack_func: Callable,
            partial_progress_folder='./prog'):
        # 2D circuit table: first index is Clifford index, second is
        # experiment index, and entry i uses depth depths[i].
    
        """
        LRB.
        
        Args:
            experiments (Any): Input argument.
            depths (Any): Input argument.
            shots (Any): Input argument.
            unpack_func (Any): Input argument.
            partial_progress_folder (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        depths.sort()
    
        max_depth = depths[-1]
        num_depths = len(depths)
        num_cliff = len(experiments)
    
        data_table_shape = (num_cliff, num_depths, shots)
        check_table_shape = (num_cliff, num_depths, shots, max_depth + 1)
        measurement_record = np.zeros(data_table_shape, dtype=np.int64)
        # 1 records a pass, -1 records a fail, 0 is a default value that by the
        # end of LRB means it was an unused slot for depth < depths[-1].
        stabilizer_check_record = np.zeros(check_table_shape, dtype=np.int64)
    
        start_ng = 0
        start_k = 0
    
        for ng in range(start_ng, num_cliff):
            #print(f'Computing Clifford sequence {ng}')
            for k in range(start_k, num_depths):
    
                c = experiments[ng][k]
    
                # Run the circuits over many shots
                NoiseModelUtils.ensure_noise_params(c)
    
                results = Program(c).simulate(shots=shots)
    
                #print("Cliff sequence is " + str(ng))
                #print("Depth / file is " + str(k))
                # Unpack the measurements
                #print(f"From LRB, we're passing current depth as {depths[k]}")
                stab_checks, m_values = unpack_func(results, depths[k], shots)
    
                # Record in table
                for s in range(shots):
                    measurement_record[ng, k, s] = m_values[s]
                    for d in range(len(stab_checks[s])):
                        stabilizer_check_record[
                            ng, k, s, d
                        ] = stab_checks[s][d]
    
        return measurement_record, stabilizer_check_record
    
    
    # An interval of 0 introduces no extra checks
    @staticmethod
    def postselect_uniform_interval(stabilizer_check_record, depths, interval):
    
        """
        Postselect uniform interval.
        
        Args:
            stabilizer_check_record (Any): Input argument.
            depths (Any): Input argument.
            interval (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        decision_table = np.zeros(stabilizer_check_record.shape[0:3],
                                  dtype=np.int64)
        num_cliff = stabilizer_check_record.shape[0]
        num_depths = stabilizer_check_record.shape[1]
        shots = stabilizer_check_record.shape[2]
    
        # make loop more general with nditer multi indices and slicing
        for ng in range(num_cliff):
            for k in range(num_depths):
                for s in range(shots):
    
                    # Terminal Stabilizer Check (after inversion step)
                    stab_pass = stabilizer_check_record[ng, k, s, depths[k]]
    
                    # Introduce extra checks only when depth is high enough.
                    # enough for the interval!
                    if depths[k] >= interval and interval > 0:
                        num_checks = depths[k] // interval
    
                        # intermediate checks
                        for j in range(1, num_checks +
                                       1):  # include the last forward check
                            idx = j * interval - 1  # forward segment index
                            # never touch the terminal slot
                            if idx < depths[k]:
                                stab_pass *= (
                                    stabilizer_check_record[ng, k, s, idx]
                                )
    
                    decision_table[ng, k, s] = stab_pass
    
        return decision_table
    
    
    # Number of checks 0 introduces no extra checks (obvious)
    @staticmethod
    def postselect_constant_number(
        stabilizer_check_record,
        depths,
        num_checks,
    ):
    
        """
        Postselect constant number.
        
        Args:
            stabilizer_check_record (Any): Input argument.
            depths (Any): Input argument.
            num_checks (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        decision_table = np.zeros(stabilizer_check_record.shape[0:3],
                                  dtype=np.int64)
        num_cliff = stabilizer_check_record.shape[0]
        num_depths = stabilizer_check_record.shape[1]
        shots = stabilizer_check_record.shape[2]
    
        # make loop more general with nditer multi indices and slicing
        for ng in range(num_cliff):
            for k in range(num_depths):
                for s in range(shots):
    
                    # Terminal Stabilizer Check (after inversion step)
                    stab_pass = stabilizer_check_record[ng, k, s, depths[k]]
    
                    # Do extra checks only if we can accomodate them
                    if num_checks > 0 and num_checks <= depths[k]:
                        interval = depths[k] // num_checks
    
                        # intermediate checks
                        for j in range(1, num_checks):
                            stab_pass = (
                                stab_pass
                                * stabilizer_check_record[
                                    ng, k, s, (j * interval) - 1
                                ]
                            )
    
                    decision_table[ng, k, s] = stab_pass
    
        return decision_table
    
    
    @staticmethod
    def postselect_record(measurement_record, decision_table):
    
        """
        Postselect record.
        
        Args:
            measurement_record (Any): Input argument.
            decision_table (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        out = measurement_record.copy()
        out[decision_table == 0] = -1
        return out
    
    
    @staticmethod
    def RB(
            experiments,
            depths: list[int],
            shots: int):
        # 2D circuit table: first index is Clifford index, second is
        # experiment index, and entry i uses depth depths[i].
    
        """
        RB.
        
        Args:
            experiments (Any): Input argument.
            depths (Any): Input argument.
            shots (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        depths.sort()
        num_depths = len(depths)
        num_cliff = len(experiments)
        data_table_shape = (num_cliff, num_depths, shots)
        measurement_record = np.zeros(data_table_shape, dtype=np.int64)
    
        for ng in range(num_cliff):
            for k in range(num_depths):
                c = experiments[ng][k]
                # Run the circuits over many shots
                NoiseModelUtils.ensure_noise_params(c)
    
                results = Program(c).simulate(shots=shots)
                # Record in table
                for s in range(shots):
                    measurement_record[ng, k,
                                       s] = results[0][0][s].measurement_value
    
        return measurement_record
    
    
    @staticmethod
    def write_raw_data(filename, measurement_record, stabilizer_check_record):
        """
        Write raw data.
        
        Args:
            filename (Any): Input argument.
            measurement_record (Any): Input argument.
            stabilizer_check_record (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        np.savez(filename,
                 measurement_record=measurement_record,
                 stabilizer_check_record=stabilizer_check_record)
    
    
    @staticmethod
    def read_raw_data(filename):
        """
        Read raw data.
        
        Args:
            filename (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        f = filename
        if os.path.exists(f):
            raw_data = np.load(f)
            return raw_data['measurement_record'], raw_data[
                'stabilizer_check_record']
    
        return None
    
    
    # Document everything shortly after finishing
    # This function unpacks the 3D data generated above into fidelities in a
    # 2D array.
    @staticmethod
    def extract_statistics(measurement_record, dimension: int | None = None):
    
        """
        Extract statistics.
        
        Args:
            measurement_record (Any): Input argument.
            dimension (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        shape = (measurement_record.shape[1], measurement_record.shape[0])
        shots = measurement_record.shape[2]
        fidelities = np.zeros(shape, dtype=np.float64)
        rejected_runs = np.zeros(shape, dtype=np.float64)
    
        valid_data = measurement_record[measurement_record >= 0]
        if dimension is None:
            dimension = int(valid_data.max()) + 1 if valid_data.size else 1
        elif valid_data.size and int(valid_data.max()) >= dimension:
            raise ValueError(
                "Measurement record contains outcomes outside the provided "
                "dimension."
            )
    
        omega = np.exp((2 * np.pi * (1j)) / dimension)
        phase_factors = omega**np.arange(dimension)
    
        fidelity_stats = []
        rejected_stats = []
    
        for k in range(shape[0]):
    
            #print(f"Calculating fidelities for depth: {depths[k]}")
    
            # Calculate fidelities over all Clifford runs at this depth
            for ng in range(shape[1]):
                data = measurement_record[ng, k]
                accepted = data[data >= 0]
                rejected_runs[k, ng] = (shots - accepted.size) / shots
    
                if accepted.size == 0:
                    fidelities[k, ng] = 0.0
                    continue
    
                tallies = np.bincount(accepted, minlength=dimension)
                fidelity = np.absolute(np.dot(tallies, phase_factors) /
                                       accepted.size)
    
                fidelities[k, ng] = fidelity
    
            # Calculate stats at given depth
            fidelity_stats.append({})
            rejected_stats.append({})
    
            #print(f"fidelity is: {fidelities[k]}")
    
            fidelity_stats[-1]['mean'] = np.mean(fidelities[k], axis=(0))
            rejected_stats[-1]['mean'] = np.mean(rejected_runs[k], axis=(0))
    
            if shape[1] == 1:
                fidelity_stats[-1]['std'] = None
                rejected_stats[-1]['std'] = None
            else:
                fidelity_stats[-1]['std'] = np.std(fidelities[k], axis=(0))
                rejected_stats[-1]['std'] = np.std(rejected_runs[k], axis=(0))
    
        return fidelity_stats, fidelities, rejected_stats, rejected_runs
    
    
    @staticmethod
    def extract_lrb_counts(measurement_record, dimension: int = 3):
    
        """
        Extract lrb counts.
        
        Args:
            measurement_record (Any): Input argument.
            dimension (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        shape = (measurement_record.shape[1], measurement_record.shape[0],
                 dimension)
        shots = measurement_record.shape[2]
        m_counts = np.zeros(shape, dtype=np.int32)
        rejected_runs = np.zeros(shape[:2], dtype=np.int32)
    
        for k in range(shape[0]):
    
            #print(f"Calculating fidelities for depth: {depths[k]}")
    
            # Calculate fidelities over all Clifford runs at this depth
            for ng in range(shape[1]):
                data = measurement_record[ng, k]
                #tallies = [0 for j in range(dimension)]
                tallies = m_counts[k, ng]
    
                for i in range(shots):
                    if data[i] == -1:
                        rejected_runs[k, ng] += 1
                    else:
                        tallies[data[i]] += 1
    
                # Fidelity calculation
                # num_accepted_runs = sum(tallies)
                # omega = np.exp( (2 * np.pi * (1j)) / dimension)
                # normalized_tallies = [
                #     tallies[i] / num_accepted_runs for i in range(dimension)
                # ]
                # fidelity_summands = [
                #     (omega ** i) * normalized_tallies[i]
                #     for i in range(dimension)
                # ]
                # fidelity = np.absolute(sum(fidelity_summands))
    
        return m_counts, rejected_runs
    
    
    @staticmethod
    def lrb_counts_to_statistics(measurement_record,
                                 rejected_runs,
                                 BATCH_SIZE,
                                 num_unfiltered_shots,
                                 filter_trivial_shots: bool = False,
                                 dimension: int = 3):
    
        """
        Lrb counts to statistics.
        
        Args:
            measurement_record (Any): Input argument.
            rejected_runs (Any): Input argument.
            BATCH_SIZE (Any): Input argument.
            num_unfiltered_shots (Any): Input argument.
            filter_trivial_shots (Any): Input argument.
            dimension (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        shape = (measurement_record.shape[0], measurement_record.shape[1])
    
        #raw_shots = measurement_record.shape[2]
        num_trivial_shots = num_unfiltered_shots // BATCH_SIZE
        offset = 0 if (num_unfiltered_shots % BATCH_SIZE) == 0 else 1
        num_trivial_shots += offset
        shots = (
            num_unfiltered_shots if not filter_trivial_shots else
            num_unfiltered_shots - num_trivial_shots
        )
    
        fidelities = np.zeros(shape, dtype=np.float64)
    
        fidelity_stats = []
        rejected_stats = []
    
        for k in range(shape[0]):
    
            #print(f"Calculating fidelities for depth: {depths[k]}")
    
            # Calculate fidelities over all Clifford runs at this depth
            for ng in range(shape[1]):
                tallies = measurement_record[k, ng]
    
                if filter_trivial_shots:
                    tallies[0] -= num_trivial_shots
                    rejected_runs[k, ng] += num_trivial_shots
                    assert (tallies[0] >= 0)
    
                # sanity check filtered data
                # print(f"tallies {tallies} and sum {sum(tallies)}")
                # print(f"rejected runs {rejected_runs[k, ng]}")
                assert (sum(tallies) +
                        rejected_runs[k, ng] == num_unfiltered_shots)
    
                # Fidelity calculation
                num_accepted_runs = sum(tallies)
                omega = np.exp((2 * np.pi * (1j)) / dimension)
                normalized_tallies = [
                    tallies[i] / num_accepted_runs for i in range(dimension)
                ]
                fidelity_summands = [(omega**i) * normalized_tallies[i]
                                     for i in range(dimension)]
                fidelity = np.absolute(sum(fidelity_summands))
    
                fidelities[k, ng] = fidelity
    
            #rejected_runs /= shots
            normalized_rejected_runs = rejected_runs / shots
            # Calculate stats at given depth
            fidelity_stats.append({})
            rejected_stats.append({})
    
            #print(f"fidelity is: {fidelities[k]}")
    
            fidelity_stats[-1]['mean'] = np.mean(fidelities[k], axis=(0))
            rejected_stats[-1]['mean'] = np.mean(normalized_rejected_runs[k],
                                                 axis=(0))
    
            if shape[1] == 1:
                fidelity_stats[-1]['std'] = None
                rejected_stats[-1]['std'] = None
            else:
                fidelity_stats[-1]['std'] = np.std(fidelities[k], axis=(0))
                rejected_stats[-1]['std'] = np.std(fidelities[k], axis=(0))
    
        return fidelity_stats, fidelities, rejected_stats, rejected_runs
    
    
    @staticmethod
    def write_stats(filename, prob, fidelity_stats, rejected_stats):
        """
        Write stats.
        
        Args:
            filename (Any): Input argument.
            prob (Any): Input argument.
            fidelity_stats (Any): Input argument.
            rejected_stats (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        with open(filename, "w") as csvfile:
            len_fid = len(fidelity_stats)
            len_rej = len(rejected_stats)
            writer = csv.writer(csvfile)
            data = []
            data.append(["Probability", prob])
            data.append(["Fidelity averages"] +
                        [fidelity_stats[i]['mean'] for i in range(len_fid)])
            data.append(["Fidelity Standard Deviations"] +
                        [fidelity_stats[i]['std'] for i in range(len_fid)])
            data.append(["Rejected Runs"] +
                        [rejected_stats[i]['mean'] for i in range(len_rej)])
            data.append(["Rejected Standard Deviations"] +
                        [rejected_stats[i]['std'] for i in range(len_rej)])
            writer.writerows(data)
    
    
    @staticmethod
    def read_stats(filename):
        """
        Read stats.
        
        Args:
            filename (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        with open(filename, "r") as csvfile:
            reader = csv.reader(csvfile)
            # Format is above
            data = []
            for line in reader:
                data.append(line)
    
            prob = float(data[0][1])
    
            fidelity_stats = []
            rejected_stats = []
            len_data = len(data[1]) - 1
    
            for i in range(len_data):
                fidelity_stats.append({})
                rejected_stats.append({})
    
                fidelity_stats[-1]['mean'] = float(data[1][i + 1])
                fidelity_stats[-1]['std'] = float(data[2][i + 1])
    
                rejected_stats[-1]['mean'] = float(data[3][i + 1])
                rejected_stats[-1]['std'] = float(data[4][i + 1])
    
        return prob, fidelity_stats, rejected_stats
    
    
    @staticmethod
    def process_lrb_counts(measurement_results, stabilizer_record, depths,
                           stab_check_array, stab_checks_are_const, folder,
                           dimension: int = 3):
    
        """
        Process lrb counts.
        
        Args:
            measurement_results (Any): Input argument.
            stabilizer_record (Any): Input argument.
            depths (Any): Input argument.
            stab_check_array (Any): Input argument.
            stab_checks_are_const (Any): Input argument.
            folder (Any): Input argument.
            dimension (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        shape = (measurement_results.shape[1], measurement_results.shape[0],
                 dimension)
    
        path_suffix = (
            "const_check_data/" if stab_checks_are_const else
            "unif_check_data/"
        )
        folder_path = folder + path_suffix
    
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
    
        for check in stab_check_array:
    
            if stab_checks_are_const:
                decision_table = (
                    LRBSimulationPipeline.postselect_constant_number(
                    stabilizer_check_record=stabilizer_record,
                    depths=depths,
                    num_checks=check
                    )
                )
            else:
                decision_table = (
                    LRBSimulationPipeline.postselect_uniform_interval(
                    stabilizer_check_record=stabilizer_record,
                    depths=depths,
                    interval=check
                    )
                )

            postselected_results = LRBSimulationPipeline.postselect_record(
                measurement_record=measurement_results,
                decision_table=decision_table)
            data_from_experiment, rejected_runs = (
                LRBSimulationPipeline.extract_lrb_counts(
                    measurement_record=postselected_results,
                    dimension=dimension))
    
            data_filename = folder_path + str(check) + ".npy"
            rejected_filename = folder_path + str(check) + "_rejected.npy"
    
            if not os.path.exists(data_filename):
                arr = np.zeros(shape, dtype=np.int32)
                np.save(data_filename, arr)
    
            if not os.path.exists(rejected_filename):
                arr = np.zeros(shape[:2], dtype=np.int32)
                np.save(rejected_filename, arr)
    
            counts_from_disk = np.load(data_filename)
            data_from_experiment = data_from_experiment + counts_from_disk
            np.save(data_filename, data_from_experiment)
    
            rejected_counts_from_disk = np.load(rejected_filename)
            rejected_runs = rejected_runs + rejected_counts_from_disk
            np.save(rejected_filename, rejected_runs)
    
            #print(f"Rejected Runs (TOTAL) is :\n\n{rejected_runs}")
    
        return 0
    
    
    @staticmethod
    def write_lrb_stats(read_directory, save_directory, prob, params,
                        BATCH_SIZE, num_unfiltered_shots,
                        filter_trivial_shots, dimension: int = 3):
    
        """
        Write lrb stats.
        
        Args:
            read_directory (Any): Input argument.
            save_directory (Any): Input argument.
            prob (Any): Input argument.
            params (Any): Input argument.
            BATCH_SIZE (Any): Input argument.
            num_unfiltered_shots (Any): Input argument.
            filter_trivial_shots (Any): Input argument.
            dimension (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        for check in params:
            measurement_results = np.load(read_directory + str(check) + ".npy")
            rejected_runs = np.load(
                read_directory + str(check) + "_rejected.npy"
            )
            fidelity_stats, _, rejected_stats, _ = (
                LRBSimulationPipeline.lrb_counts_to_statistics(
                    measurement_results,
                    rejected_runs,
                    BATCH_SIZE,
                    num_unfiltered_shots,
                    filter_trivial_shots,
                    dimension=dimension,
                )
            )
            LRBSimulationPipeline.write_stats(
                save_directory + str(check) + ".csv",
                prob,
                fidelity_stats,
                rejected_stats,
            )
    
    
    @staticmethod
    def run_LRB_round(
        stab_checks_const,
        stab_checks_unif,
        BATCH_SIZE,
        error_prob,
        error_prob_ind,
        num_cliff_seq,
        depths,
        num_shots,
        filter_trivial_shots,
        LRB_experiment_folder_path,
        RB_experiment_folder_path,
        LRB_results_folder_path,
        RB_results_folder_path,
        partial_progress_folder_path,
        unpack_func=None,
        logical_dimension: int = 3,
    ):
        """
        Run LRB round.
        
        Args:
            stab_checks_const (Any): Input argument.
            stab_checks_unif (Any): Input argument.
            BATCH_SIZE (Any): Input argument.
            error_prob (Any): Input argument.
            error_prob_ind (Any): Input argument.
            num_cliff_seq (Any): Input argument.
            depths (Any): Input argument.
            num_shots (Any): Input argument.
            filter_trivial_shots (Any): Input argument.
            LRB_experiment_folder_path (Any): Input argument.
            RB_experiment_folder_path (Any): Input argument.
            LRB_results_folder_path (Any): Input argument.
            RB_results_folder_path (Any): Input argument.
            partial_progress_folder_path (Any): Input argument.
            unpack_func (Any): Input argument.
            logical_dimension (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        if unpack_func is None:
            raise ValueError("run_LRB_round requires an explicit unpack_func.")

        LRB_experiments = []
        RB_experiments = []
        num_depths = len(depths)
    
        # Create progress if not found
        if not os.path.exists(partial_progress_folder_path +
                              "shots_processed.txt"):
            ExperimentSetupManager.write_single_param(
                num_shots,
                partial_progress_folder_path + "shots_processed.txt",
            )
    
        # Load in progress
        shots_to_process = int(
            ExperimentSetupManager.fetch_single_param(
                partial_progress_folder_path + "shots_processed.txt"))
        print(f"We have to process {shots_to_process} shots.")
    
        # Determine whether experiment is done
        if shots_to_process > 0:
    
            # Otherwise read in all LRB experiment data
            for i in range(num_cliff_seq):
                # make new list
                LRB_experiments_from_single_sequence = []
                LRB_cliff_path = LRB_experiment_folder_path + \
                    str(i) + '/' + str(error_prob_ind) + '/'
    
                # iterate through depths
                for j in range(num_depths):
                    LRB_single_experiment_path = (
                        LRB_cliff_path + str(j) + ".chp"
                    )
                    LRB_c = read_circuit(
                        os.path.abspath(LRB_single_experiment_path))
                    LRB_experiments_from_single_sequence.append(LRB_c)
    
                LRB_experiments.append(LRB_experiments_from_single_sequence)
    
            # Read in stabilizer check parameters
            # stab_checks_const = fetch_list(
            #     partial_progress_folder_path + "/check_const.txt"
            # )
            # stab_checks_unif = fetch_list(
            #     partial_progress_folder_path + "/check_unif.txt"
            # )
    
            # Run shots in batches of size BATCH_SIZE
            while shots_to_process > 0:
    
                # Determine how many shots to run
                batch = (
                    BATCH_SIZE if shots_to_process > BATCH_SIZE else
                    shots_to_process
                )
    
                # Run experiment
                print(
                    f"Resuming experiments for error probability {error_prob}"
                )
                start_time = time.time()
                LRB_experiment_results = LRBSimulationPipeline.LRB(
                    experiments=LRB_experiments,
                    depths=depths,
                    shots=batch,
                    unpack_func=unpack_func,
                    partial_progress_folder=partial_progress_folder_path)
                end_time = time.time()
                print(f"Finished in {str(end_time - start_time)} seconds!")
    
                # Process partial results into progress files
                # Constant checks
                LRBSimulationPipeline.process_lrb_counts(
                    measurement_results=LRB_experiment_results[0],
                    stabilizer_record=LRB_experiment_results[1],
                    depths=depths,
                    stab_check_array=stab_checks_const,
                    stab_checks_are_const=True,
                    folder=partial_progress_folder_path,
                    dimension=logical_dimension)
                # Uniform checks
                LRBSimulationPipeline.process_lrb_counts(
                    measurement_results=LRB_experiment_results[0],
                    stabilizer_record=LRB_experiment_results[1],
                    depths=depths,
                    stab_check_array=stab_checks_unif,
                    stab_checks_are_const=False,
                    folder=partial_progress_folder_path,
                    dimension=logical_dimension)
    
                # Update shots processed
                shots_to_process -= batch
    
                with open(partial_progress_folder_path + "shots_processed.txt",
                          "w") as progress_file:
                    progress_file.write(str(shots_to_process))
                    print(
                        f"Progress saved, need to process "
                        f"{shots_to_process} more shots."
                    )
    
        # Calculate and write LRB stats
        const_save_dir = LRB_results_folder_path + \
            str(error_prob_ind) + "/const_check_data/"
        unif_save_dir = LRB_results_folder_path + \
            str(error_prob_ind) + "/unif_check_data/"
    
        if not os.path.exists(const_save_dir):
            os.mkdir(const_save_dir)
        if not os.path.exists(unif_save_dir):
            os.mkdir(unif_save_dir)
    
        LRBSimulationPipeline.write_lrb_stats(
            partial_progress_folder_path + "const_check_data/", const_save_dir,
            error_prob, stab_checks_const, BATCH_SIZE, num_shots,
            filter_trivial_shots, dimension=logical_dimension)
        LRBSimulationPipeline.write_lrb_stats(
            partial_progress_folder_path + "unif_check_data/", unif_save_dir,
            error_prob, stab_checks_unif, BATCH_SIZE, num_shots,
            filter_trivial_shots, dimension=logical_dimension)
        print(
            "Wrote logical test data to the following directories:\n"
            f"{const_save_dir}\n"
            f"{unif_save_dir}"
        )
    
        # LRB_writefile = (
        #     LRB_results_folder_path + str(error_prob_ind) + '.csv'
        # )
        # LRB_raw_data_file = (
        #     LRB_results_folder_path + str(error_prob_ind)
        #     + '_raw_LRB_data.npz'
        # )
        # write_raw_data(
        #     LRB_raw_data_file,
        #     LRB_experiment_results[0],
        #     LRB_experiment_results[1],
        # )
        # print(f"Wrote logical test raw data  to {LRB_raw_data_file}")
    
        # Complete RB
        for i in range(num_cliff_seq):
    
            # make new list
            RB_experiments_from_single_sequence = []
            RB_cliff_path = RB_experiment_folder_path + \
                str(i) + '/' + str(error_prob_ind) + '/'
    
            # iterate through depths
            for j in range(num_depths):
                RB_single_experiment_path = RB_cliff_path + str(j) + ".chp"
                RB_c = read_circuit(os.path.abspath(RB_single_experiment_path))
                RB_experiments_from_single_sequence.append(RB_c)
    
            RB_experiments.append(RB_experiments_from_single_sequence)
    
        print(f"Running physical circuit...")
        start_time = time.time()
        RB_experiment_results = LRBSimulationPipeline.RB(
            experiments=RB_experiments,
            depths=depths,
            shots=NORMAL_RB_SHOTS,
        )
        end_time = time.time()
        print(f"Finished in {str(end_time - start_time)} seconds!")
    
        # Calculate and write RB stats
        RB_writefile = RB_results_folder_path + str(error_prob_ind) + '.csv'
        RB_f_stats, _, RB_r_stats, _ = (
            LRBSimulationPipeline.extract_statistics(
                measurement_record=RB_experiment_results,
                dimension=None,
            )
        )
        LRBSimulationPipeline.write_stats(filename=RB_writefile,
                                          prob=error_prob,
                                          fidelity_stats=RB_f_stats,
                                          rejected_stats=RB_r_stats)
        print(f"Wrote physical test results to {RB_writefile}")
    
        return 0
    
# Shared default engine for scripts that prefer object-style calls.
DEFAULT_SIM_ENGINE = LRBSimulationEngine()
