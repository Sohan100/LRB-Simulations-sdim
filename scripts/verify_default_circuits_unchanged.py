"""Check that the default circuit generator still reproduces a run's .chp files.

Production circuits were drawn with an unseeded ``random.choice``, so there
is no seed to replay. Instead this script recovers each sampled sequence's
Clifford words from the run's own files: it renders every catalog word with
the current generator and matches the rendered text against the deepest
circuit (check rounds delimit LRB Cliffords; RB is parsed with backtracking
and pinned by every depth's inverse block). It then regenerates the LRB,
const=0 and RB circuits from those words with the default settings and
compares them byte for byte with the files on disk, at the requested
probability indices.

Nothing is written outside a temporary directory. Usage:

    python scripts/verify_default_circuits_unchanged.py RUN_FOLDER [--seqs 0,17] [--p-indices 0,6,14]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from sdim.circuit import Circuit  # noqa: E402

from lrb.circuit_generator import LRBCircuitGenerator  # noqa: E402
from lrb.code_simulation_profiles import CodeSimulationProfileRegistry  # noqa: E402

HEADER_LINES = 3  # comment, '#', 'd <dim> qudits=<n>'


def _read_param(run: str, name: str) -> str:
    with open(os.path.join(run, name)) as fh:
        return fh.read().strip()


def generator_for_run(run: str, **kwargs) -> LRBCircuitGenerator:
    profile = CodeSimulationProfileRegistry.resolve_code_profile(_read_param(run, "code_name.txt"))
    return LRBCircuitGenerator(code_definition=profile.code_definition,
                               noise_model=_read_param(run, "noise_model.txt"), **kwargs)


class Renderer:
    """Render circuits to .chp text exactly as generate_tests writes them."""

    def __init__(self, gen: LRBCircuitGenerator):
        self.gen = gen
        self.tmp = tempfile.mkdtemp(prefix="chp_verify_")

    def text(self, circuit, prob=None) -> str:
        if prob is not None:
            self.gen.update_noise_param(circuit, prob)
        path = self.gen.write_circuit_file(circuit, self.tmp, "c.chp")
        with open(path) as fh:
            return fh.read()

    def body(self, circuit, prob=0.0) -> list[str]:
        return self.text(circuit, prob).splitlines()[HEADER_LINES:]


def _chp(run, family, seq, p_idx, depth_idx) -> str:
    with open(os.path.join(run, "experiments", family, str(seq), str(p_idx), f"{depth_idx}.chp")) as fh:
        return fh.read()


def decode_lrb_words(run: str, gen: LRBCircuitGenerator, seq: int) -> list[str]:
    """Recover the forward Clifford words of one LRB sequence from its p-index-0 files."""
    cd = gen.code_definition
    depths = [int(x) for x in _read_param(run, "depths.txt").split(",")]
    p0 = float(_read_param(run, "probs.txt").split(",")[0])
    r = Renderer(gen)
    lines = _chp(run, "LRB", seq, 0, len(depths) - 1).splitlines()[HEADER_LINES:]
    blank = Circuit(dimension=cd.dimension, num_qudits=cd.encoded_num_qudits)
    offset = len(r.body(blank + cd.logical_plus_initial_state(), p0))
    check = r.body(gen._noisy_segment(gen._check_round_circuit(), True), p0)
    cands = {w: r.body(gen._noisy_segment(gen._logical_clifford_circuit(cd.clifford_to_gate_sequence(w)), True), p0)
             + check for w in cd.clifford_strings}
    words = []
    for _ in range(max(depths)):
        hits = [w for w, c in cands.items() if lines[offset:offset + len(c)] == c]
        if len(hits) != 1:
            raise RuntimeError(f"seq {seq}: {len(hits)} catalog words match at Clifford {len(words) + 1}")
        words.append(hits[0])
        offset += len(cands[hits[0]])
    return words


def decode_rb_words(run: str, gen: LRBCircuitGenerator, seq: int) -> list[str]:
    """Recover one RB sequence's words; adjacent words can merge in the text,
    so backtrack and pin boundaries with every depth's file."""
    cd = gen.code_definition
    depths = [int(x) for x in _read_param(run, "depths.txt").split(",")]
    p0 = float(_read_param(run, "probs.txt").split(",")[0])
    r = Renderer(gen)
    target = {d: _chp(run, "RB", seq, 0, i) for i, d in enumerate(depths)}
    lines = target[max(depths)].splitlines()[HEADER_LINES:]
    cands = {}
    for w in cd.clifford_strings:
        seg = Circuit(dimension=cd.dimension, num_qudits=cd.physical_num_qudits)
        for g in cd.clifford_to_gate_sequence(w):
            cd.apply_physical_gate(seg, g)
        cands[w] = r.body(gen._noisy_segment(seg, True), p0)

    def rb_text(words, depth):
        gen.generate_random_clifford_strings = lambda n: list(words[:n])
        try:
            circ = gen.generate_rb_clifford_sequence(depths=[depth], with_noise=True)[0]
        finally:
            del gen.generate_random_clifford_strings
        return r.text(circ, p0)

    def dfs(words, offset):
        n = len(words)
        if n in target and n > 0 and rb_text(words, n) != target[n]:
            return None
        if n == max(depths):
            return words
        for w, c in cands.items():
            if lines[offset:offset + len(c)] == c:
                got = dfs(words + [w], offset + len(c))
                if got is not None:
                    return got
        return None

    words = dfs([], 0)
    if words is None:
        raise RuntimeError(f"seq {seq}: no RB word sequence reproduces the files")
    return words


def compare_sequence(run, gen, seq, p_indices) -> list[str]:
    """Regenerate one sequence with default settings; return mismatching files."""
    depths = [int(x) for x in _read_param(run, "depths.txt").split(",")]
    probs = [float(x) for x in _read_param(run, "probs.txt").split(",")]
    r = Renderer(gen)
    bad = []
    words = decode_lrb_words(run, gen, seq)
    families = {
        "LRB": gen.generate_lrb_clifford_sequence(depths=depths, with_noise=True, clifford_strings=words),
        "LRB_const0": gen.generate_lrb_const0_clifford_sequence(depths=depths, with_noise=True,
                                                                clifford_strings=words),
    }
    rb_words = decode_rb_words(run, gen, seq)
    gen.generate_random_clifford_strings = lambda n: list(rb_words[:n])
    try:
        families["RB"] = gen.generate_rb_clifford_sequence(depths=depths, with_noise=True)
    finally:
        del gen.generate_random_clifford_strings
    for family, circuits in families.items():
        if not os.path.isdir(os.path.join(run, "experiments", family)):
            continue
        for p_idx in p_indices:
            for i, circ in enumerate(circuits):
                if r.text(circ, probs[p_idx]) != _chp(run, family, seq, p_idx, i):
                    bad.append(f"{family}/{seq}/{p_idx}/{i}.chp")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--seqs", default="0,17")
    ap.add_argument("--p-indices", default="0,6,14")
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    gen = generator_for_run(run)
    n_probs = len(_read_param(run, "probs.txt").split(","))
    p_indices = [int(x) for x in a.p_indices.split(",") if int(x) < n_probs]
    total_bad = []
    for seq in (int(x) for x in a.seqs.split(",")):
        bad = compare_sequence(run, gen, seq, p_indices)
        total_bad += bad
        print(f"{os.path.basename(run)} seq {seq}: {'identical' if not bad else f'{len(bad)} files differ'}")
    if total_bad:
        print("\n".join(total_bad[:20]))
        sys.exit(1)


if __name__ == "__main__":
    main()
