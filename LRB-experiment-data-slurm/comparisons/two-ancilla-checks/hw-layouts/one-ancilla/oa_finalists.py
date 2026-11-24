"""Turn search candidates into scheduled, fault-safe rounds.

For a candidate line of oa_transfer_search.py / oa_frame_search.py output: enumerate
minimum-cost sector sequences (oa_build.sector_paths), interleave them at random
with the data-gate windows, screen with the symplectic checker (oa_symbolic: round
correct, 0 silent single faults), and keep the shallowest by ASAP gate moments.

Usage: oa_finalists.py OUTFILE N_CANDIDATES [SAMPLES]  -> prints the best op list per
candidate (repr) for later sdim verification and benchmarking (oa_verify_bench.py).
"""
import ast
import random
import re
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oa_build as B  # noqa: E402
import oa_frame_search as FS  # noqa: E402
import oa_symbolic as SY  # noqa: E402
import oa_transfer_search as TS  # noqa: E402

GATE = re.compile(r"(CNOT|CNOT_INV)\((\d)->(\d)\)")


def parse_line(line):
    m = re.match(r"\s+(\d+) gates .*?frames \[(.*?)\]\s+data couplers (\[.*?\]).*?"
                 r"G1:(\[.*?\])(?:/\d+)?\s+G2:(\[.*?\])(?:/\d+)?\s+G3:(\[.*?\])(?:/\d+)?\s+G4:(\[.*?\])(?:/\d+)?"
                 r"(?:\s+a1-a2 (\w) a3-a4 (\w))?\s+data (\{.*?\}) anc (\(.*\))", line)
    if not m:
        return None
    total, frames, edges, g1, g2, g3, g4, c12, c34, data, anc = m.groups()
    timeline = []
    for w in frames.split("|"):
        gs = [(int(c), int(t), 1 if n == "CNOT" else 2) for n, c, t in GATE.findall(w)]
        if gs:
            timeline.append(tuple(gs))
    return dict(total=int(total), timeline=tuple(timeline), edges=ast.literal_eval(edges),
                groups=[frozenset(ast.literal_eval(g)) for g in (g1, g2, g3, g4)],
                c12=c12 == "Y", c34=c34 == "Y", data=ast.literal_eval(data), anc=ast.literal_eval(anc))


def asap_moments(ops):
    """Gate moments of the ASAP packing used by the LRB generator (resets first,
    final readout deferred); 1q and 2q gates share moments."""
    ready = {}
    nm = 0
    for op in ops:
        if op[0] != "g":
            continue
        ws = op[2:]
        t = max(ready.get(w, 0) for w in ws) + 1
        for w in ws:
            ready[w] = t
        nm = max(nm, t)
    return nm


def best_round(cand, samples=20000, seed=1):
    segs = TS.segments(cand["timeline"])
    T1, T2, T3, T4 = cand["groups"]
    zc, zp = B.sector_paths("Z", FS.CHECKS["G1"][1], FS.CHECKS["G2"][1], T1, T2, cand["c12"], segs, limit=3000)
    xc, xp = B.sector_paths("X", FS.CHECKS["G3"][1], FS.CHECKS["G4"][1], T3, T4, cand["c34"], segs, limit=3000)
    if not zp or not xp:
        return None
    rng = random.Random(seed)
    best = None
    tried = ok = 0
    for _ in range(samples):
        ops = B.assemble(rng.choice(zp), rng.choice(xp), cand["timeline"], rng)
        tried += 1
        if not SY.round_ok(ops):
            continue
        ok += 1
        nm = asap_moments(ops)
        if best is not None and nm >= best[0]:
            continue
        bad, total = SY.silent(ops)
        if bad == 0:
            best = (nm, ops)
    return dict(zc=zc, xc=xc, nz=len(zp), nx=len(xp), tried=tried, correct=ok, best=best)


def _work(args):
    i, cand, samples = args
    return i, cand, best_round(cand, samples)


if __name__ == "__main__":
    path, n = sys.argv[1], int(sys.argv[2])
    samples = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    cands = [c for c in map(parse_line, Path(path).read_text().splitlines()) if c][:n]
    with Pool(min(120, len(cands))) as pool:
        for i, cand, res in pool.imap(_work, [(i, c, samples) for i, c in enumerate(cands)]):
            two_q = None
            if res and res["best"]:
                two_q = sum(1 for op in res["best"][1] if op[0] == "g" and len(op) == 4)
            print(f"[{i}] {cand['total']} gates  frames {cand['timeline']}  groups {[sorted(g) for g in cand['groups']]}"
                  f"  c12 {cand['c12']} c34 {cand['c34']}  data {cand['data']} anc {cand['anc']}", flush=True)
            if not res:
                print("    no sector sequences", flush=True)
                continue
            print(f"    sector costs Z {res['zc']} X {res['xc']}; {res['nz']} x {res['nx']} sequences; "
                  f"{res['correct']}/{res['tried']} interleavings correct", flush=True)
            if res["best"]:
                print(f"    BEST {two_q} two-qutrit gates, {res['best'][0]} gate moments (ASAP), 0 silent", flush=True)
                print(f"    OPS {res['best'][1]}", flush=True)
            else:
                print("    no zero-silent interleaving found", flush=True)
