"""Reading run-folder circuits for the correction study under sdim 1.4.0.

``chain_dem`` rescales every noise gate with its ``prob_scale``. sdim 1.4.0 does not write that
field and rejects it on reading, so ``compact_dem.read_run_circuit`` restores it: from ``prob / p``
for files written by sdim 1.4.0, and from the stored field for older files.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("sdim")

from sdim.circuit_io import write_circuit  # noqa: E402

from lrb.circuit_generator import LRBCircuitGenerator  # noqa: E402
from lrb.code_definitions import single_qutrit_cliffords_exhaustive_strings as CATALOG  # noqa: E402
from lrb.correction.compact_dem import read_run_circuit  # noqa: E402
from lrb.correction.variants import build_variant_circuit  # noqa: E402

STRINGS = [CATALOG[5], CATALOG[17], CATALOG[24]]


def _noise(circuit):
    ops = [op for op in circuit.operations if op.gate_name in ("N1", "N2")]
    return ([(op.gate_name, op.params.get("noise_channel")) for op in ops],
            [float(op.params["prob_scale"]) for op in ops])


@pytest.fixture(scope="module")
def circuit():
    return build_variant_circuit(STRINGS, 2, p=2e-3)


def test_sdim140_file_recovers_generator_scales(circuit, tmp_path):
    path = LRBCircuitGenerator.write_circuit_file(circuit, str(tmp_path), "0.chp")
    with open(path) as handle:
        assert "prob_scale" not in handle.read()
    names, scales = _noise(read_run_circuit(path, 2e-3))
    assert names == _noise(circuit)[0]
    assert scales == pytest.approx(_noise(circuit)[1])
    assert set(scales) == {0.1, 1.0, 2.0, 5.0}
    with pytest.raises(ValueError, match="pass the run's p"):
        read_run_circuit(path)


@pytest.mark.parametrize("quoted", [False, True])
def test_file_with_stored_scales(circuit, tmp_path, quoted):
    # plain sdim write_circuit keeps prob_scale (as the files written before sdim 1.4.0
    # did, quoted), and sdim 1.4.0's own reader rejects it
    path = write_circuit(circuit=circuit, output_file="0.chp", comment="",
                         directory=str(tmp_path))
    if quoted:
        with open(path) as handle:
            text = handle.read()
        with open(path, "w") as handle:
            handle.write(re.sub(r"prob_scale=(\S+)", r'prob_scale="\1"', text))
    names, scales = _noise(read_run_circuit(path))
    assert names == _noise(circuit)[0]
    assert scales == pytest.approx(_noise(circuit)[1])
