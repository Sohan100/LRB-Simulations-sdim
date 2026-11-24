"""Lower bound on the round size for larger data-frame sets (safe pairs only).

For every timeline of data-CNOT windows on the safe pairs 01, 04, 12, 13, 14, 23 with
L data CNOTs in total, a relaxed sector search gives a lower bound on reads +
transfers: both ancillas of a sector may touch all five data qutrits and are coupled
(oa_transfer_search.sector_bfs with T = {q0..q4}). Any layout with that timeline
needs at least 2L + lb_Z + lb_X two-qutrit gates. Timelines whose bound is <= MAX are
then searched exactly (touched sets, colouring, embedding) with
oa_transfer_search.analyse.

Usage: oa_lower_bound.py 'SHAPES' MAX [NPRINT]   (e.g. '1,1,1;2,1;3' 19)
"""
import sys
from multiprocessing import Pool

import oa_frame_search as FS
import oa_transfer_search as TS

UNSAFE = {frozenset(p) for p in [(0, 2), (0, 3), (2, 4), (3, 4)]}
FS.GATES[:] = [g for g in FS.GATES if frozenset(g[:2]) not in UNSAFE]


def bound(args):
    timeline, cap = args
    L = sum(len(w) for w in timeline)
    edges = sorted({(2, 3)} | {tuple(sorted(g[:2])) for w in timeline for g in w})
    if not TS.colour_ok(edges, [], []):
        return timeline, None
    segs = TS.segments(timeline)
    full = frozenset(range(5))
    lbs = []
    for kind, a, b in (("Z", "G1", "G2"), ("X", "G3", "G4")):
        m, s = TS.read_moves(kind, segs, full)
        tgt = TS.enc(FS.CHECKS[a][1]) * TS.NV + TS.enc(FS.CHECKS[b][1])
        lbs.append(int(TS.sector_bfs(len(segs), m, s, m, s, True, tgt, TS.ADD, TS.MUL, max(cap - 2 * L, 1))))
    return timeline, (2 * L + sum(lbs), lbs[0], lbs[1])


def pruned_table(kind, tP, tQ, segs, maxc):
    """TS.sector_table with the same filters, visiting touched-set pairs from large to
    small and skipping every pair below one that already failed (cost is monotone:
    fewer touched data can only cost more)."""
    target = TS.enc(tP) * TS.NV + TS.enc(tQ)
    cache = {T: TS.read_moves(kind, segs, T) for T in TS.SUBSETS}
    pairs = []
    for TP in TS.SUBSETS:
        for TQ in TS.SUBSETS:
            if len(TP & TQ) > 2:
                continue
            for coupled in (False, True):
                if coupled and (TP & TQ or len(TP) > 3 or len(TQ) > 3):
                    continue
                pairs.append((TP, TQ, coupled))
    pairs.sort(key=lambda t: -(len(t[0]) + len(t[1])))
    failing = {False: [], True: []}
    out = {}
    for TP, TQ, coupled in pairs:
        if any(TP <= A and TQ <= B for A, B in failing[coupled]):
            continue
        pm, ps = cache[TP]
        qm, qs = cache[TQ]
        c = int(TS.sector_bfs(len(segs), pm, ps, qm, qs, coupled, target, TS.ADD, TS.MUL, maxc))
        if c <= maxc:
            out[(TP, TQ, coupled)] = c
        else:
            failing[coupled].append((TP, TQ))
    return out


def _exact(args):
    """TS.analyse with pruned sector tables and per-sector caps from the other sector's
    lower bound."""
    timeline, cap, lbz, lbx = args
    L = sum(len(w) for w in timeline)
    budget = cap - 2 * L
    segs = TS.segments(timeline)
    zt = pruned_table("Z", FS.CHECKS["G1"][1], FS.CHECKS["G2"][1], segs, max(budget - lbx, 1))
    if not zt:
        return []
    zmin = min(zt.values())
    xt = pruned_table("X", FS.CHECKS["G3"][1], FS.CHECKS["G4"][1], segs, max(budget - zmin, 1))
    if not xt:
        return []
    orig = TS.sector_table
    tables = {"Z": zt, "X": xt}
    TS.sector_table = lambda kind, *a, **k: tables[kind]
    try:
        return TS.analyse(timeline, cap, True)
    finally:
        TS.sector_table = orig


if __name__ == "__main__":
    shapes = [tuple(int(x) for x in a.split(",")) for a in sys.argv[1].split(";")]
    cap = int(sys.argv[2])
    survivors = []
    with Pool(110) as pool:
        for shape in shapes:
            jobs = [(t, cap) for t in FS.timelines(shape)]
            hist = {}
            for t, res in pool.imap_unordered(bound, jobs, chunksize=16):
                lb = res[0] if res else None
                key = lb if lb is not None and lb <= cap else f">{cap}"
                hist[key] = hist.get(key, 0) + 1
                if lb is not None and lb <= cap:
                    survivors.append((t, res[1], res[2]))
            print(f"windows {shape}: {len(jobs)} timelines; lower bound histogram "
                  f"{dict(sorted(hist.items(), key=lambda kv: str(kv[0])))}", flush=True)
        print(f"{len(survivors)} timelines with lower bound <= {cap}; exact search on them", flush=True)
        rows = []
        done = 0
        for res in pool.imap_unordered(_exact, [(t, cap, lz, lx) for t, lz, lx in survivors], chunksize=1):
            rows += res
            done += 1
            if done % 2000 == 0:
                print(f"  exact: {done}/{len(survivors)} timelines, {len(rows)} layouts so far", flush=True)
    rows.sort(key=lambda r: r[0])
    print(f"embeddable layouts with <= {cap} two-qutrit gates: {len(rows)}"
          + (f"; best {rows[0][0]}" if rows else ""), flush=True)
    nprint = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    for total, t, groups, c12, c34, zc, xc, edges, (pos, anc) in rows[:nprint]:
        print(f"  {total:2d} gates (Z {zc}, X {xc}, data {total - zc - xc})  frames [{TS.fmt_timeline(t)}]  "
              f"data couplers {edges}  "
              + "  ".join(f"{FS.NAMES[j]}:{sorted(g)}" for j, g in enumerate(groups))
              + f"  a1-a2 {'Y' if c12 else 'n'} a3-a4 {'Y' if c34 else 'n'}  data {pos} anc {anc}", flush=True)
