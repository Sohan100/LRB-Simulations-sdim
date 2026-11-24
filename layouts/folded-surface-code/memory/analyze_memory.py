"""Turn the sinter statistics into per-round logical error rates, Lambda, thresholds and footprints.

Per-round error: a memory run of r = d rounds fails with probability P_L; the per-round rate is
eps = (1 - (1 - 2 P_L)^(1/r)) / 2 (sinter.shot_error_rate_to_piece_error_rate). X and Z memories
are combined as eps = (eps_X + eps_Z) / 2, the error of a random Pauli eigenstate.

Inputs: the sinter files in results/raw/ (kept locally, not in git) or, when they are absent, the
per-task counts in results/stats.csv, which this script writes and which hold every shot and error
count the analysis uses.

Writes results/summary.csv, lambda.csv, thresholds.csv, teraquop.csv, gates.csv and stats.csv.
Run `python analyze_memory.py --extras` to write all tables and print the comparisons.
"""
from __future__ import annotations

import collections
import csv
import math
import os
import sys

import numpy as np
import sinter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import memory_circuits as mc  # noqa: E402

RESULTS = os.path.join(HERE, "results")
RAW = os.path.join(RESULTS, "raw")
STATS = os.path.join(RESULTS, "stats.csv")


def _raw_files(prefix: str) -> list[str]:
    if not os.path.isdir(RAW):
        return []
    return [os.path.join(RAW, f) for f in sorted(os.listdir(RAW)) if f.startswith(prefix) and f.endswith(".csv")]


def _stats_rows(experiment: str):
    with open(STATS, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["experiment"] == experiment:
                yield r


def qubits(code: str, d: int) -> int:
    return mc.CODES[code](d).num_qubits


def per_round(errors: int, shots: int, rounds: int) -> float:
    return sinter.shot_error_rate_to_piece_error_rate(errors / shots, pieces=rounds) if shots else float("nan")


def ci_per_round(errors: int, shots: int, rounds: int, z: float = 1.0) -> tuple[float, float]:
    """Wilson interval (z standard errors) on P_L, mapped to the per-round rate."""
    if shots == 0:
        return float("nan"), float("nan")
    ph = errors / shots
    den = 1 + z * z / shots
    c = (ph + z * z / (2 * shots)) / den
    h = z * math.sqrt(ph * (1 - ph) / shots + z * z / (4 * shots * shots)) / den
    lo, hi = max(c - h, 0.0), min(c + h, 0.4999)
    return per_round(lo * shots, shots, rounds), per_round(hi * shots, shots, rounds)


def load() -> dict:
    """{(noise, code, d, p): dict(shots, errors per basis, eps, eps_lo, eps_hi)}"""
    tot = collections.defaultdict(lambda: [0, 0])
    raw = _raw_files("memory-")
    if raw:
        for s in sinter.read_stats_from_csv_files(*raw):
            m = s.json_metadata
            k = (m["noise"], m["code"], m["d"], m["p"], m["basis"])
            tot[k][0] += s.shots
            tot[k][1] += s.errors
    else:
        for r in _stats_rows("memory"):
            k = (r["noise"], r["code"], int(r["d"]), float(r["p"]), r["basis"])
            tot[k][0] += int(r["shots"])
            tot[k][1] += int(r["errors"])
    rows = {}
    keys = sorted({k[:4] for k in tot})
    for k in keys:
        noise, code, d, p = k
        e, lo, hi, sh, er = [], [], [], [], []
        for b in ("X", "Z"):
            shots, errors = tot.get(k + (b,), (0, 0))
            e.append(per_round(errors, shots, d))
            l, h = ci_per_round(errors, shots, d)
            lo.append(l)
            hi.append(h)
            sh.append(shots)
            er.append(errors)
        rows[k] = dict(noise=noise, code=code, d=d, p=p, qubits=qubits(code, d),
                       shots_x=sh[0], errors_x=er[0], shots_z=sh[1], errors_z=er[1],
                       eps_x=e[0], eps_z=e[1], eps=float(np.mean(e)),
                       eps_lo=float(np.mean(lo)), eps_hi=float(np.mean(hi)))
    return rows


def lambda_fit(rows: dict, noise: str, code: str, p: float, dmin: int = 5, min_errors: int = 20):
    """Fit log eps = a - (d+1)/2 log Lambda over the distances with enough errors.

    Returns (Lambda, a, distances used) or None.
    """
    pts = [(r["d"], r["eps"]) for (n, c, d, pp), r in rows.items()
           if n == noise and c == code and pp == p and d >= dmin
           and min(r["errors_x"], r["errors_z"]) >= min_errors and r["eps"] > 0]
    pts.sort()
    if len(pts) < 2:
        return None
    ds = np.array([x for x, _ in pts], float)
    y = np.log([e for _, e in pts])
    A = np.vstack([np.ones_like(ds), -(ds + 1) / 2]).T
    (a, logL), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.exp(logL)), float(a), [int(x) for x in ds]


def threshold(rows: dict, noise: str, code: str, d1: int, d2: int):
    """p where the per-round curves of d1 and d2 cross (linear interpolation in log-log)."""
    ps = sorted({p for (n, c, d, p) in rows if n == noise and c == code})
    prev = None
    for p in ps:
        a, b = rows.get((noise, code, d1, p)), rows.get((noise, code, d2, p))
        if not a or not b or a["eps"] <= 0 or b["eps"] <= 0:
            continue
        g = math.log(b["eps"]) - math.log(a["eps"])
        if prev is not None and prev[1] < 0 <= g:
            p0, g0 = prev
            t = g0 / (g0 - g)
            return float(math.exp(math.log(p0) + t * (math.log(p) - math.log(p0))))
        prev = (p, g)
    return None


def footprint(rows: dict, noise: str, code: str, p: float, target: float):
    """Qubits needed for eps <= target, from the Lambda fit (d odd, extrapolated)."""
    fit = lambda_fit(rows, noise, code, p)
    if fit is None or fit[0] <= 1:
        return None
    L, a, _ = fit
    for d in range(3, 201, 2):
        if math.exp(a) * L ** (-(d + 1) / 2) <= target:
            return d, qubits(code, d)
    return None


def write_summary(rows: dict) -> str:
    path = os.path.join(RESULTS, "summary.csv")
    keys = ["noise", "code", "d", "p", "qubits", "eps", "eps_lo", "eps_hi", "eps_x", "eps_z",
            "shots_x", "errors_x", "shots_z", "errors_z"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for k in sorted(rows):
            w.writerow({kk: rows[k][kk] for kk in keys})
    return path


if __name__ == "__main__":
    rows = load()
    print("wrote", write_summary(rows))
    for noise in ("SD6", "SI1000"):
        print("\n==", noise)
        ps = sorted({k[3] for k in rows if k[0] == noise})
        for code in ("folded", "rotated"):
            print(" ", code)
            for d in (3, 5, 7, 9, 11):
                line = "    d=%2d n=%3d " % (d, qubits(code, d))
                for p in ps:
                    r = rows.get((noise, code, d, p))
                    line += " %.1e%s" % (r["eps"], "*" if min(r["errors_x"], r["errors_z"]) < 20 else " ") if r else "    -    "
                print(line)
            print("    p:        " + " ".join("%8.4f" % p for p in ps))
        for code in ("folded", "rotated"):
            th = [threshold(rows, noise, code, d, d + 2) for d in (3, 5, 7, 9)]
            print("  threshold", code, ["%.4f" % t if t else None for t in th])
        for p in ps:
            out = []
            for code in ("folded", "rotated"):
                f = lambda_fit(rows, noise, code, p)
                out.append("%s Lambda %.2f (d %s)" % (code, f[0], f[2]) if f else "%s -" % code)
            print("  p=%.4f  %s" % (p, "   ".join(out)))


# --------------------------------------------------------------------------------------------
# Logical gates between rounds (folded code)
# --------------------------------------------------------------------------------------------
def load_gates() -> dict:
    """{(noise, gate, d, p, basis): (shots, errors)} from results/gates-*.csv"""
    tot = collections.defaultdict(lambda: [0, 0])
    raw = _raw_files("gates-")
    if raw:
        for s in sinter.read_stats_from_csv_files(*raw):
            m = s.json_metadata
            k = (m["noise"], m.get("gate", "none"), m["d"], m["p"], m["basis"])
            tot[k][0] += s.shots
            tot[k][1] += s.errors
    elif os.path.exists(STATS):
        for r in _stats_rows("gates"):
            k = (r["noise"], r["gate"], int(r["d"]), float(r["p"]), r["basis"])
            tot[k][0] += int(r["shots"])
            tot[k][1] += int(r["errors"])
    return dict(tot)


def gate_table(gates: dict) -> list[dict]:
    """Per-round error with a logical gate between rounds, against the plain memory (Z basis,
    sector detectors), for every (noise, d, p)."""
    out = []
    keys = sorted({(n, d, p) for (n, g, d, p, b) in gates})
    for n, d, p in keys:
        row = dict(noise=n, d=d, p=p)
        for g, b in (("none", "Z"), ("H", "Z"), ("H", "X"), ("S", "Z")):
            sh, er = gates.get((n, g, d, p, b), (0, 0))
            row["eps_%s_%s" % (g, b)] = per_round(er, sh, d) if sh else float("nan")
            row["err_%s_%s" % (g, b)] = er
            row["shots_%s_%s" % (g, b)] = sh
        out.append(row)
    return out


def teraquop(rows: dict, noise: str, code: str, p: float, block: float = 1e-12):
    """Smallest odd d with d * eps(d) <= block (a d-round block fails with probability <= 1e-12,
    the teraquop target of Gidney et al. 2021), from the Lambda fit. Returns (d, qubits)."""
    fit = lambda_fit(rows, noise, code, p)
    if fit is None or fit[0] <= 1:
        return None
    L, a, _ = fit
    for d in range(3, 401, 2):
        if d * math.exp(a) * L ** (-(d + 1) / 2) <= block:
            return d, qubits(code, d)
    return None


def eps_at_qubits(rows: dict, noise: str, code: str, p: float, n: float) -> float:
    """Per-round error at n physical qubits, from the Lambda fit with d solved from n."""
    fit = lambda_fit(rows, noise, code, p)
    if fit is None:
        return float("nan")
    L, a, _ = fit
    if code == "folded":
        d = (math.sqrt(n) + 1) / 2          # n = (2d - 1)^2
    else:
        d = math.sqrt((n + 1) / 2)          # n = 2 d^2 - 1
    return math.exp(a) * L ** (-(d + 1) / 2)


def write_tables(rows: dict, gates: dict) -> None:
    with open(os.path.join(RESULTS, "lambda.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["noise", "code", "p", "Lambda", "log_prefactor", "distances"])
        for noise in ("SD6", "SI1000"):
            for code in ("folded", "rotated"):
                for p in sorted({k[3] for k in rows if k[0] == noise}):
                    f = lambda_fit(rows, noise, code, p)
                    if f:
                        w.writerow([noise, code, p, "%.4f" % f[0], "%.4f" % f[1], " ".join(map(str, f[2]))])
    with open(os.path.join(RESULTS, "thresholds.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["noise", "code", "d_pair", "p_cross"])
        for noise in ("SD6", "SI1000"):
            for code in ("folded", "rotated"):
                for d in (3, 5, 7, 9):
                    t = threshold(rows, noise, code, d, d + 2)
                    w.writerow([noise, code, "%d-%d" % (d, d + 2), "%.5f" % t if t else ""])
    with open(os.path.join(RESULTS, "teraquop.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["noise", "p", "code", "d", "qubits"])
        for noise in ("SD6", "SI1000"):
            for p in (0.0005, 0.001, 0.0015, 0.002):
                for code in ("folded", "rotated"):
                    t = teraquop(rows, noise, code, p)
                    w.writerow([noise, p, code, t[0] if t else "", t[1] if t else ""])
    gt = gate_table(gates)
    if gt:
        with open(os.path.join(RESULTS, "gates.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(gt[0]))
            w.writeheader()
            for r in gt:
                w.writerow(r)
    # compact per-task statistics: the data behind every number above
    with open(os.path.join(RESULTS, "stats.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "noise", "code", "gate", "d", "p", "basis", "rounds", "shots", "errors"])
        for k, r in sorted(rows.items()):
            for b in ("X", "Z"):
                w.writerow(["memory", k[0], k[1], "none", k[2], k[3], b, k[2], r["shots_" + b.lower()], r["errors_" + b.lower()]])
        for (n, g, d, p, b), (sh, er) in sorted(gates.items()):
            w.writerow(["gates", n, "folded", g, d, p, b, d, sh, er])


def report_extras(rows: dict, gates: dict) -> None:
    print("\n== teraquop footprint (d-round block failure <= 1e-12)")
    for noise in ("SD6", "SI1000"):
        for p in (0.0005, 0.001, 0.0015, 0.002):
            f, r = teraquop(rows, noise, "folded", p), teraquop(rows, noise, "rotated", p)
            print("  %-6s p=%.4f  folded d=%s n=%s   rotated d=%s n=%s   ratio %.2f" % (
                noise, p, f and f[0], f and f[1], r and r[0], r and r[1], (f[1] / r[1]) if f and r else float("nan")))
    print("\n== per-round error at equal qubit count (fits), p = 0.001")
    for noise in ("SD6", "SI1000"):
        for n in (169, 289, 441, 1000):
            ef = eps_at_qubits(rows, noise, "folded", 0.001, n)
            er = eps_at_qubits(rows, noise, "rotated", 0.001, n)
            print("  %-6s n=%5d  folded %.2e  rotated %.2e  folded/rotated %.2f" % (noise, n, ef, er, ef / er))
    print("\n== logical gate between every two rounds (folded code), per-round error")
    for r in gate_table(gates):
        print("  %-6s d=%d p=%.3f  plain(Z) %.2e  H(Z) %.2e  H(X) %.2e  S(Z) %.2e   S/plain %.2f  [errors %d %d %d %d]" % (
            r["noise"], r["d"], r["p"], r["eps_none_Z"], r["eps_H_Z"], r["eps_H_X"], r["eps_S_Z"],
            r["eps_S_Z"] / r["eps_none_Z"] if r["eps_none_Z"] else float("nan"),
            r["err_none_Z"], r["err_H_Z"], r["err_H_X"], r["err_S_Z"]))


if __name__ == "__main__" and "--extras" in sys.argv:
    rows = load()
    gates = load_gates()
    write_tables(rows, gates)
    report_extras(rows, gates)
