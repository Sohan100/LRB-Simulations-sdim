"""Copy an LRB run with the same Clifford sequences but another profile's check round.

The new run is paired with the source run. It has the same Clifford words in every
sequence, the same depths, probabilities, shots, check policies and noise model; only the
code profile (here: the check round) changes.
- LRB words are decoded from the source run's own files with the source profile's
  generator (``verify_default_circuits_unchanged.decode_lrb_words``).
- Before anything is written, each sequence is regenerated with the source profile and
  compared byte for byte with the source files (LRB and LRB_const0, three p-indices), so
  a decoding error cannot pass silently.
- LRB and LRB_const0 are then regenerated from those words with the target profile and
  written exactly as ``LRBCircuitGenerator.generate_tests`` writes them.
- RB circuits do not involve the code and are copied byte for byte.
- Folders, done markers and parameter files are as ``ExperimentSetupManager.run_setup``
  makes them; ``code_name.txt`` names the target profile. ``round_variant_manifest.json``
  records the source and the checks.

Usage::

    PYTHONNOUSERSITE=1 .venv-sdim140/bin/python scripts/make_round_variant.py \\
        --source LRB-experiment-data-slurm/Run-2026-10-28-10-23-30-folded_qutrit-single-ancilla-si1000-sdim140 \\
        --code-name folded_qutrit_prod5b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import verify_default_circuits_unchanged as V  # noqa: E402
from lrb.circuit_generator import LRBCircuitGenerator  # noqa: E402
from lrb.code_simulation_profiles import CodeSimulationProfileRegistry  # noqa: E402

PARAM_FILES = ("probs.txt", "depths.txt", "shots.txt", "num_cliffs.txt", "check_const.txt",
               "check_unif.txt", "noise_model.txt", "inverse_mode.txt", "run_instructions.txt",
               "d.txt")


def _list(run: Path, name: str) -> list[str]:
    return [v for v in (run / name).read_text().strip().split(",") if v.strip()]


def _sequence(args):
    src, dest, code_name, seq = args
    src_s, dest_s = str(src), str(dest)
    depths = [int(x) for x in _list(src, "depths.txt")]
    probs = [float(x) for x in _list(src, "probs.txt")]
    noise = (src / "noise_model.txt").read_text().strip()
    families = [f for f in ("LRB", "LRB_const0") if (src / "experiments" / f).is_dir()]
    old = V.generator_for_run(src_s)
    words = V.decode_lrb_words(src_s, old, seq)
    # 1. the decoded words must reproduce the source files
    r = V.Renderer(old)
    check_p = sorted({0, len(probs) // 2, len(probs) - 1})
    regen = {"LRB": old.generate_lrb_clifford_sequence(depths=depths, with_noise=True,
                                                       clifford_strings=words),
             "LRB_const0": old.generate_lrb_const0_clifford_sequence(depths=depths, with_noise=True,
                                                                     clifford_strings=words)}
    for fam in families:
        for p_idx in check_p:
            for i, circ in enumerate(regen[fam]):
                if r.text(circ, probs[p_idx]) != V._chp(src_s, fam, seq, p_idx, i):
                    raise RuntimeError(f"seq {seq}: source {fam}/{p_idx}/{i} not reproduced")
    # 2. regenerate with the target profile and write as generate_tests does
    prof = CodeSimulationProfileRegistry.resolve_code_profile(code_name)
    new = LRBCircuitGenerator(code_definition=prof.code_definition, noise_model=noise)
    built = {"LRB": new.generate_lrb_clifford_sequence(depths=depths, with_noise=True,
                                                       clifford_strings=words),
             "LRB_const0": new.generate_lrb_const0_clifford_sequence(depths=depths, with_noise=True,
                                                                     clifford_strings=words)}
    n = 0
    for fam in families:
        for p_idx, prob in enumerate(probs):
            path = os.path.join(dest_s, "experiments", fam, str(seq), str(p_idx))
            os.makedirs(path, exist_ok=True)
            for i, circ in enumerate(built[fam]):
                new.update_noise_param(circ, prob)
                new.write_circuit_file(circ, path, f"{i}.chp")
                n += 1
    shutil.copytree(src / "experiments" / "RB" / str(seq), dest / "experiments" / "RB" / str(seq))
    return seq, words, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--code-name", required=True)
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()
    src = a.source.resolve()
    src_code = (src / "code_name.txt").read_text().strip()
    CodeSimulationProfileRegistry.resolve_code_profile(a.code_name)
    base = re.sub(r"^Run-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-", "", src.name)
    if not base.startswith(src_code + "-"):
        raise SystemExit(f"run name {src.name} does not start with its code name {src_code}")
    dest = src.parent / f"Run-{time.strftime('%Y-%m-%d-%H-%M-%S')}-{a.code_name}{base[len(src_code):]}"
    dest.mkdir(parents=False, exist_ok=False)
    for name in PARAM_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dest / name)
    (dest / "code_name.txt").write_text(a.code_name)
    n_probs = len(_list(src, "probs.txt"))
    for sub in ("experiments/LRB", "experiments/RB", "results/LRB", "results/RB", "progress"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
    if (src / "experiments" / "LRB_const0").is_dir():
        (dest / "experiments" / "LRB_const0").mkdir()
    for idx in range(n_probs):
        (dest / "results" / "LRB" / str(idx)).mkdir(exist_ok=True)
        (dest / "progress" / str(idx)).mkdir(exist_ok=True)
        (dest / "progress" / str(idx) / "done.txt").write_text("0")
    n_seq = int((src / "num_cliffs.txt").read_text().strip())
    jobs = [(src, dest, a.code_name, s) for s in range(n_seq)]
    words, files = {}, 0
    with Pool(min(a.workers, n_seq)) as pool:
        for seq, w, n in pool.imap_unordered(_sequence, jobs):
            words[seq] = w
            files += n
            print(f"seq {seq}: {len(w)} words decoded and source reproduced; {n} circuits written",
                  flush=True)
    manifest = dict(
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), source_run=str(src),
        source_code_name=src_code, code_name=a.code_name,
        pairing="same Clifford words per sequence (decoded from the source files); RB copied byte for byte",
        source_reproduced="every sequence, LRB and LRB_const0 at p-indices first/middle/last",
        lrb_circuits_written=files,
        clifford_words={str(k): words[k] for k in sorted(words)},
        script=str(Path(__file__).resolve()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (dest / "round_variant_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(dest)


if __name__ == "__main__":
    main()
