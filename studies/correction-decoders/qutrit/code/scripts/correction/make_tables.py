"""Markdown tables for the report from the committed result CSVs."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "docs" / "correction-decoders" / "qutrit-512" / "results"

NAMES = {
    "lrbd:unif1": "LRB-D, every round (unif 1)",
    "lrbd:unif2": "LRB-D, every 2nd round (unif 2)",
    "lrbd:unif4": "LRB-D, every 4th round (unif 4)",
    "lrbd:const1": "LRB-D, final round only (const 1)",
    "lrbd:const0": "LRB-D, no ancilla checks (const 0)",
    "lrbd:xtriv": "LRB-D, final round X-part only",
    "bfd:all": "Blip-forgiving detection (BFD)",
    "bfd:xtriv": "BFD, final round X-part only",
    "soft:0.97": "MLD, reject if confidence < 0.97",
    "soft:0.9": "MLD, reject if confidence < 0.9",
    "soft:0.5": "MLD, reject if confidence < 0.5",
    "mld:all": "Full MLD correction (all rounds)",
    "mld:unif4": "Full MLD, every 4th round only",
    "mld:terminal": "Full MLD, final round only",
    "mld:const0": "Full MLD, const 0 circuits",
    "win:2": "Causal hard-decision, lookahead 2",
    "win:1": "Causal hard-decision, lookahead 1",
    "win:0": "Causal hard-decision, lookahead 0",
    "raw": "No decoding, no rejection",
}


def read(path):
    """Read a CSV, falling back to its ``.gz`` sibling."""
    path = Path(path)
    if not path.exists() and Path(str(path) + ".gz").exists():
        path = Path(str(path) + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return list(csv.DictReader(handle))


def fnum(x, digits=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:
        return "n/a"
    return f"{v:.{digits}g}" if abs(v) < 1e-3 or abs(v) >= 1e4 else f"{v:.{digits + 1}g}"


def main() -> int:
    fits = read(RESULTS / "fits_baseline.csv")
    curves = read(RESULTS / "curves_baseline.csv")
    rb = {round(float(r["p_lrb"]), 12): float(r["rb_r"]) for r in read(RESULTS / "rb_reference.csv")}
    lines = []
    for p in (2e-4, 5e-4, 1e-3, 2e-3, 5e-3):
        lines += [f"#### p = {p:g} (physical RB at p_RB = {5 * p:g}: r = {rb.get(round(p, 12), float('nan')):.3g} per depth step)", "",
                  "| Strategy | r per depth step [95% CI] | acceptance per depth step | accepted P(correct) at m = 0 | accepted P(correct) at m = 22 | acceptance at m = 22 |",
                  "|---|---|---|---|---|---|"]
        for key, name in NAMES.items():
            fit = next((r for r in fits if r["strategy"] == key and abs(float(r["p"]) - p) < 1e-12), None)
            cur = {int(r["depth"]): r for r in curves if r["strategy"] == key and abs(float(r["p"]) - p) < 1e-12}
            if fit is None or not cur:
                continue
            c0, c22 = cur.get(0), cur.get(22)
            lines.append(
                f"| {name} | {fnum(fit.get('signed_r'))} [{fnum(fit.get('signed_r_lo'))}, {fnum(fit.get('signed_r_hi'))}] "
                f"| {fnum(fit.get('acc_per_step'), 3)} | {fnum(c0['survival_mean'], 5) if c0 else 'n/a'} "
                f"| {fnum(c22['survival_mean'], 5) if c22 else 'n/a'} | {fnum(c22['acc_mean'], 3) if c22 else 'n/a'} |")
        lines.append("")
    (RESULTS / "decoder_tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:30]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
