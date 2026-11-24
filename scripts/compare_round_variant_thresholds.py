"""Paired p* comparison: the one-ancilla layout's five-moment round against the seven-moment round.

The five-moment round (internal name PROD5B, profiles folded_qutrit_prod5b and
folded_qubit_prod5b) is the published seven-moment one-ancilla round with its gates
reordered. The five-moment runs are paired copies of the published one-ancilla runs
(``make_round_variant.py``): sequence s has the same Clifford words in both, and the RB
circuits are byte-identical. So:

- LRB: one bootstrap draw of sequences (with replacement) is applied to both rounds,
  separately for the production and the high-p-extension sequence sets;
- RB: both rounds are compared with one common RB curve, the published run's RB
  (the five-moment runs re-simulated identical RB circuits; using one curve removes that
  sampling noise from the difference), resampled with the same draw;
- every curve is refit and p*(k) recomputed with the published fit settings and threshold
  estimator, on the merged production + extension (qutrit) or refinement
  (qubit) tables (``merge_highp_threshold_extension.py`` families ``qutrit``/``prod5b``
  and ``qubit``/``qubit-prod5b``).

Reported per noise model and interval k = 1..4: the published p* of both rounds (each
against its own RB), the paired p* with the common RB, and the bootstrap distribution of
the difference and the ratio (SE, 68% and 95% percentile intervals). Needs the sequence
bootstrap RB caches of all four labels (``sequence_bootstrap_thresholds.py``).
Outputs: LRB-experiment-data-slurm/comparisons/sdim140-prod5b/paired_thresholds.{csv,md}
(qutrit) and paired_thresholds_qubit.{csv,md} (qubit).

Usage (repo root)::

    PYTHONNOUSERSITE=1 .venv-sdim140/bin/python scripts/compare_round_variant_thresholds.py \
        --family qutrit|qubit [--reps 1000]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_highp_threshold_extension as merge  # noqa: E402
import sequence_bootstrap_thresholds as sb  # noqa: E402

PAIRS = {
    "qutrit": {"si1000": ("one-ancilla-si1000", "prod5b-one-ancilla-si1000"),
               "sd6": ("one-ancilla-sd6", "prod5b-one-ancilla-sd6")},
    "qubit": {"si1000": ("qubit-one-ancilla-si1000", "qubit-prod5b-one-ancilla-si1000"),
              "sd6": ("qubit-one-ancilla-sd6", "qubit-prod5b-one-ancilla-sd6")},
}
STEM = {"qutrit": "paired_thresholds", "qubit": "paired_thresholds_qubit"}
OUT = merge.COMPARISONS / "sdim140-prod5b"


def load(label: str) -> sb.Configuration:
    conf = sb.Configuration(label, sb.configurations()[label], sb.OUT_ROOT / label)
    for src in (conf.prod, conf.ext):
        src.load_lrb()
        if not src.cache.exists():
            raise SystemExit(f"run sequence_bootstrap_thresholds.py {label} first ({src.cache})")
        src.load_or_simulate_rb()
    return conf


def published(conf: sb.Configuration) -> dict[int, float]:
    return {k: float(conf.published.loc[conf.published["check_num"] == k, "threshold_p"].iloc[0])
            for k in sb.CHECKS}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--family", choices=sorted(PAIRS), default="qutrit")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for noise, (cur_label, new_label) in PAIRS[a.family].items():
        start = time.time()
        cur, new = load(cur_label), load(new_label)
        cur.validate()
        new.validate()
        pub_cur, pub_new = published(cur), published(new)
        # same grids and sequence counts, so draws and rows line up
        assert [r[2] for r in cur.rows] == [r[2] for r in new.rows]
        for a_src, b_src in ((cur.prod, new.prod), (cur.ext, new.ext)):
            assert a_src.depths == b_src.depths
            assert int(a_src.plotter.num_cliffs) == int(b_src.plotter.num_cliffs)
        # common RB: the published (seven-moment) run's
        new.prod.rb, new.ext.rb = cur.prod.rb, cur.ext.rb
        new.rb_published = dict(cur.rb_published)
        full = {key: None for key in ("lrb_prod", "lrb_ext", "rb_prod", "rb_ext")}
        base_cur, base_new = cur.evaluate(full), new.evaluate(full)
        num = {"prod": int(cur.prod.plotter.num_cliffs), "ext": int(cur.ext.plotter.num_cliffs)}
        rng = np.random.default_rng(a.seed)
        reps = {k: [] for k in sb.CHECKS}
        for _ in range(a.reps):
            draws = {f"{kind}_{name}": rng.integers(0, num[name], num[name])
                     for kind in ("lrb", "rb") for name in ("prod", "ext")}
            ec, en = cur.evaluate(draws), new.evaluate(draws)
            for k in sb.CHECKS:
                reps[k].append((ec[k][0], en[k][0]))
        for k in sb.CHECKS:
            v = np.asarray(reps[k], dtype=float)
            ok = np.isfinite(v).all(axis=1)
            d, r = v[ok, 1] - v[ok, 0], v[ok, 1] / v[ok, 0]
            qd = np.percentile(d, [2.5, 16, 84, 97.5])
            qr = np.percentile(r, [2.5, 16, 84, 97.5])
            rows.append({
                "noise": noise, "interval": k,
                "p_star_seven_moment_published": pub_cur[k], "p_star_five_moment_published": pub_new[k],
                "p_star_seven_moment_paired": base_cur[k][0], "p_star_five_moment_paired": base_new[k][0],
                "method_seven_moment": base_cur[k][1], "method_five_moment": base_new[k][1],
                "diff": base_new[k][0] - base_cur[k][0], "diff_se": float(np.std(d, ddof=1)),
                "diff_ci68_low": qd[1], "diff_ci68_high": qd[2],
                "diff_ci95_low": qd[0], "diff_ci95_high": qd[3],
                "ratio": base_new[k][0] / base_cur[k][0], "ratio_se": float(np.std(r, ddof=1)),
                "ratio_ci95_low": qr[0], "ratio_ci95_high": qr[3],
                "valid_fraction": float(ok.mean()), "replicates": a.reps,
            })
            x = rows[-1]
            print(f"{a.family} {noise} k={k}: p* seven-moment {x['p_star_seven_moment_paired']:.4f} -> five-moment "
                  f"{x['p_star_five_moment_paired']:.4f} (published {pub_cur[k]:.4f} -> {pub_new[k]:.4f}); "
                  f"diff {x['diff']:+.4f} +- {x['diff_se']:.4f}, 95% [{qd[0]:+.4f}, {qd[3]:+.4f}]; "
                  f"ratio {x['ratio']:.3f}, 95% [{qr[0]:.3f}, {qr[3]:.3f}]", flush=True)
        print(f"  {noise} done in {time.time() - start:.0f} s", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / f"{STEM[a.family]}.csv", index=False)
    code = "$[[5,1,2]]_3$ (qutrit)" if a.family == "qutrit" else "$[[5,1,2]]_2$ (qubit)"
    lines = [
        f"# {code}, one-ancilla layout: the five-moment round vs the seven-moment round, paired pseudo-thresholds",
        "",
        f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} by `scripts/compare_round_variant_thresholds.py`"
        f" ({a.reps} paired sequence-bootstrap replicates, seed {a.seed}).",
        "",
        "Same Clifford sequences, grids, noise and estimator; only the check round differs."
        " The seven-moment round is the one of the published runs. `paired` p* use one common RB"
        " curve (the seven-moment run's); `published` p* use each run's own RB."
        " diff = five-moment - seven-moment, with its paired bootstrap SE and 95% interval.",
        "",
        "| noise | k | p* seven-moment | p* five-moment | diff (SE) | diff 95% | ratio (95%) | published seven -> five |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for x in rows:
        lines.append(
            f"| {x['noise'].upper()} | {x['interval']} | {x['p_star_seven_moment_paired']:.4f} |"
            f" {x['p_star_five_moment_paired']:.4f} | {x['diff']:+.4f} ({x['diff_se']:.4f}) |"
            f" [{x['diff_ci95_low']:+.4f}, {x['diff_ci95_high']:+.4f}] |"
            f" {x['ratio']:.3f} [{x['ratio_ci95_low']:.3f}, {x['ratio_ci95_high']:.3f}] |"
            f" {x['p_star_seven_moment_published']:.4f} -> {x['p_star_five_moment_published']:.4f} |")
    (OUT / f"{STEM[a.family]}.md").write_text("\n".join(lines) + "\n")
    print(OUT / f"{STEM[a.family]}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
