"""Co-design ranking for extra trees only (non-CNOT 4-gate encoders, e.g. star at q2)."""
import sys
sys.path.insert(0, ".")
import codesign as CD  # noqa: E402

if __name__ == "__main__":
    rows = []
    for s in sys.argv[1:]:
        T = frozenset(frozenset((int(e[0]), int(e[1]))) for e in s.split(","))
        name, n_emb, r = CD._tree_rows(T)
        print(f"tree {name:12s}: {n_emb} lattice embeddings of T+23, {len(r)} candidates", flush=True)
        rows += r
    rows.sort(key=lambda r: r[0])
    seen = set()
    for r in rows:
        k = (r[1], tuple(sorted(r[2].items())), r[5][0], r[6][0])
        if k in seen:
            continue
        seen.add(k)
        total, T, pos, extra, safe, A, B, cs = r
        print(f"  {total:2d} gates  tree {T:12s} data {pos}  extra {extra or '-'}  safe {safe}  "
              f"anc1 {A[0]} touches {A[1]} reads {A[2]}  anc2 {B[0]} touches {B[1]} reads {B[2]}  per-check {cs}")
        if len(seen) >= 15:
            break
