"""Op lists of the one-ancilla rounds compared in README section "One ancilla per check on a square lattice".

Wires: data q0..q4 = 0..4, a1 = 5 (G1), a2 = 6 (G2), a3 = 7 (G3), a4 = 8 (G4).
Format as ../hw_layouts_check.py: ("g", gate, wires...), ("m", wire, tag), ("r", wire).
"""

# The paper's one-ancilla round (same layout and 12 gates as the LRB generator round),
# packed into 5 gate moments instead of 7 (prod_reschedule.py, first zero-silent schedule).
PROD5 = [('r', 5), ('r', 6), ('r', 7), ('r', 8), ('g', 'CNOT', 0, 5), ('g', 'H', 7), ('g', 'H', 8), ('g', 'CNOT_INV', 1, 5), ('g', 'CNOT', 7, 0), ('g', 'CNOT_INV', 8, 4), ('g', 'CNOT_INV', 3, 5), ('g', 'CNOT_INV', 4, 6), ('g', 'CNOT_INV', 7, 2), ('g', 'CNOT_INV', 8, 1), ('g', 'CNOT', 2, 6), ('g', 'CNOT', 7, 1), ('g', 'CNOT', 8, 3), ('g', 'CNOT', 1, 6), ('g', 'H_INV', 7), ('g', 'H_INV', 8), ('m', 5, 'G1'), ('m', 6, 'G2'), ('m', 7, 'G3'), ('m', 8, 'G4')]

# Square-lattice one-ancilla layouts (oa_transfer_search.py --safe, oa_finalists.py): 19 two-qutrit
# gates (4 data-data, 5 ancilla-ancilla transfers), 9 gate moments, one reset and one readout layer.
# Layout SQ14 (x right, y up; couplers = drawn lattice neighbours that the round uses):
#     a3  a4  q1
#     q3  q0  q4
#     q2  a2  a1
# data couplers q1-q4 (window CNOT_INV 1->4) and q2-q3 (window CNOT 3->2); a1: q4; a2: q0, q2;
# a3: q3; a4: q0, q1; ancilla couplers a1-a2, a3-a4.  data {1: (4, 4), 2: (2, 2), 3: (2, 3), 4: (4, 3), 0: (3, 3)}  ancillas a1..a4 ((4, 2), (3, 2), (2, 4), (3, 4))
SQ14 = [('r', 5), ('r', 6), ('r', 7), ('r', 8), ('g', 'H', 7), ('g', 'H', 8), ('g', 'CNOT_INV', 7, 3), ('g', 'CNOT', 8, 1), ('g', 'CNOT_INV', 1, 4), ('g', 'CNOT', 8, 0), ('g', 'CNOT_INV', 7, 8), ('g', 'CNOT', 8, 7), ('g', 'CNOT_INV', 4, 5), ('g', 'CNOT_INV', 8, 1), ('g', 'CNOT', 1, 4), ('g', 'CNOT', 3, 2), ('g', 'CNOT', 7, 3), ('g', 'CNOT', 2, 6), ('g', 'CNOT_INV', 3, 2), ('g', 'CNOT_INV', 0, 6), ('g', 'CNOT_INV', 6, 5), ('g', 'CNOT', 5, 6), ('g', 'CNOT_INV', 4, 5), ('g', 'CNOT', 2, 6), ('g', 'CNOT', 6, 5), ('g', 'H_INV', 7), ('g', 'H_INV', 8), ('m', 5, 'G1'), ('m', 6, 'G2'), ('m', 7, 'G3'), ('m', 8, 'G4')]

# Layout SQ01:
#     a4  q0  q1
#     a3  q4  a1
#     q2  q3  a2
# data couplers q0-q1 (window CNOT_INV 0->1) and q2-q3 (window CNOT_INV 2->3); a1: q1, q4; a2: q3;
# a3: q2, q4; a4: q0; ancilla couplers a1-a2, a3-a4.  data {1: (4, 4), 2: (2, 2), 0: (3, 4), 3: (3, 2), 4: (3, 3)}  ancillas a1..a4 ((4, 3), (4, 2), (2, 3), (2, 4))
SQ01 = [('r', 5), ('r', 6), ('r', 7), ('r', 8), ('g', 'H', 7), ('g', 'H', 8), ('g', 'CNOT_INV', 4, 5), ('g', 'CNOT', 1, 5), ('g', 'CNOT', 3, 6), ('g', 'CNOT', 5, 6), ('g', 'CNOT_INV', 0, 1), ('g', 'CNOT', 8, 0), ('g', 'CNOT_INV', 6, 5), ('g', 'CNOT_INV', 1, 5), ('g', 'CNOT', 0, 1), ('g', 'CNOT_INV', 2, 3), ('g', 'CNOT_INV', 7, 2), ('g', 'CNOT_INV', 3, 6), ('g', 'CNOT', 7, 4), ('g', 'CNOT', 2, 3), ('g', 'CNOT', 8, 7), ('g', 'CNOT_INV', 7, 8), ('g', 'CNOT', 8, 0), ('g', 'CNOT_INV', 7, 2), ('g', 'CNOT_INV', 8, 7), ('g', 'H_INV', 7), ('g', 'H_INV', 8), ('m', 5, 'G1'), ('m', 6, 'G2'), ('m', 7, 'G3'), ('m', 8, 'G4')]

# Three data CNOTs per round (oa_lower_bound.py '1,1,1;2,1;3' 18, finalists_L3.out): 15 two-qutrit
# gates (6 data-data, 9 ancilla-data, no ancilla-ancilla coupler), 9 gate moments.
# SQ15A (L3_candidates.txt line 2):
#      .  a1  q3   .
#      .  q1  q2  a2
#     a3  q0  a4  q4
# windows [CNOT 1->2, CNOT_INV 0->1] (nested) and [CNOT 2->3]; a1: q1, q3; a2: q2, q4; a3: q0;
# a4: q0, q2, q4.
SQ15A = [('r', 5), ('r', 6), ('r', 7), ('r', 8), ('g', 'H', 7), ('g', 'H', 8), ('g', 'CNOT', 1, 2), ('g', 'CNOT_INV', 4, 6), ('g', 'CNOT', 2, 6), ('g', 'CNOT_INV', 0, 1), ('g', 'CNOT_INV', 1, 5), ('g', 'CNOT_INV', 3, 5), ('g', 'CNOT', 7, 0), ('g', 'CNOT_INV', 8, 4), ('g', 'CNOT_INV', 8, 0), ('g', 'CNOT', 0, 1), ('g', 'CNOT_INV', 1, 2), ('g', 'CNOT', 8, 0), ('g', 'CNOT', 2, 3), ('g', 'CNOT_INV', 8, 2), ('g', 'CNOT_INV', 2, 3), ('g', 'H_INV', 7), ('g', 'H_INV', 8), ('m', 5, 'G1'), ('m', 6, 'G2'), ('m', 7, 'G3'), ('m', 8, 'G4')]

# SQ15B (L3_candidates.txt line 5):
#      .  a2  q2   .
#      .  q1  q3  a1
#     a4  q4  a3  q0
# windows [CNOT 1->3, CNOT_INV 4->1] (nested) and [CNOT 3->2]; a1: q0, q3; a2: q1, q2;
# a3: q0, q3, q4; a4: q4.
SQ15B = [('r', 5), ('r', 6), ('r', 7), ('r', 8), ('g', 'H', 7), ('g', 'H', 8), ('g', 'CNOT', 1, 3), ('g', 'CNOT_INV', 4, 1), ('g', 'CNOT_INV', 8, 4), ('g', 'CNOT', 7, 4), ('g', 'CNOT_INV', 3, 5), ('g', 'CNOT', 1, 6), ('g', 'CNOT', 2, 6), ('g', 'CNOT', 4, 1), ('g', 'CNOT', 0, 5), ('g', 'CNOT_INV', 1, 3), ('g', 'CNOT', 7, 0), ('g', 'CNOT', 3, 2), ('g', 'CNOT_INV', 7, 4), ('g', 'CNOT', 7, 3), ('g', 'CNOT_INV', 3, 2), ('g', 'H_INV', 7), ('g', 'H_INV', 8), ('m', 5, 'G1'), ('m', 6, 'G2'), ('m', 7, 'G3'), ('m', 8, 'G4')]

# Best of the 52 zero-silent 5-moment schedules by silent fault pairs (rank_prod5.py):
# 346 two-qutrit-gate pairs, against 410 for PROD5 and 506 for the generator's round.
# Also the best of all 31,104 correct schedules of this round (oa_schedule_search.py):
# lowest exact leading-order logical error per round under SI1000 and SD6 (tied with 3 others).
PROD5B = [('r', 5), ('r', 6), ('r', 7), ('r', 8), ('g', 'CNOT_INV', 1, 5), ('g', 'H', 7), ('g', 'H', 8), ('g', 'CNOT', 0, 5), ('g', 'CNOT_INV', 7, 2), ('g', 'CNOT_INV', 8, 1), ('g', 'CNOT_INV', 3, 5), ('g', 'CNOT', 2, 6), ('g', 'CNOT', 7, 0), ('g', 'CNOT_INV', 8, 4), ('g', 'CNOT_INV', 4, 6), ('g', 'CNOT', 7, 1), ('g', 'CNOT', 8, 3), ('g', 'CNOT', 1, 6), ('g', 'H_INV', 7), ('g', 'H_INV', 8), ('m', 5, 'G1'), ('m', 6, 'G2'), ('m', 7, 'G3'), ('m', 8, 'G4')]

ROUNDS = {"production-5": PROD5, "production-5b": PROD5B, "sq14": SQ14, "sq01": SQ01, "sq15a": SQ15A, "sq15b": SQ15B}
