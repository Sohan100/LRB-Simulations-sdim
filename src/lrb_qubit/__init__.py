"""Qubit counterpart of the folded ``[[5,1,2]]_3`` LRB-D workflow, built on Stim.

The package mirrors the qutrit pipeline in ``src/lrb`` on qubits:

``code``
    The qubit ``[[5,1,2]]`` folded (planar d=2) surface code, its
    fold-transversal logical gates, and the 24-element single-qubit Clifford
    catalog.
``circuits``
    Stim circuit builders for encoded LRB sequences and matched physical RB,
    with the repo's generalized SI1000 noise translated to qubits, the same
    check-round layout, and detector/observable annotations for decoding.
``baseline``
    LRB-D postselection policies (``const`` / ``unif``) and statistics in the
    same format as the qutrit pipeline.
"""
