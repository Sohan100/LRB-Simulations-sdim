"""Error of the mean for pseudo-thresholds: bootstrap over Clifford sequences.

The notebook's threshold intervals resample every depth and every p
independently. The same 30 random Clifford sequences are used at every depth
and every p, though, so their errors are correlated. This script resamples
whole sequences instead: one draw of 30 sequences (with replacement) is
applied to every depth and every p of a sequence set, the per-depth means
are rebuilt from each sequence's own data, every LRB and RB curve is refit
with the notebook's fit settings (depth 0 included; even depths only for
k >= 2 in extension runs, as in the production merge), and the crossing is
recomputed with the notebook's threshold estimator. The spread of p* over
replicates is the error of the mean over the random-sequence ensemble.

Sequence sets, resampled independently: production LRB, extension LRB,
production RB, extension RB (RB circuits use their own Clifford draws).
LRB per-sequence data are the checkpointed outcome tallies, so the full,
unresampled data reproduce the published values exactly (checked). RB
per-sequence results were not checkpointed and the sampler is unseeded, so
RB is re-simulated once (10,000 shots per sequence, as in production) and
only its replicate-to-replicate shifts are used, centred on the published
RB error rates. A leave-one-sequence-out jackknife is reported as a check.

Outputs go to LRB-experiment-data-slurm/comparisons/sequence-bootstrap/.

Usage (repo root):
    PYTHONNOUSERSITE=1 .venv-sdim140/bin/python \\
        scripts/sequence_bootstrap_thresholds.py <label> [--reps 1000]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_highp_threshold_extension as merge  # noqa: E402
from lrb.lrb_simulation import LRBSimulationPipeline  # noqa: E402

try:
    from sdim.circuit_io import read_circuit
except ImportError:  # pragma: no cover - sdim is required at runtime
    read_circuit = None

OUT_ROOT = merge.COMPARISONS / "sequence-bootstrap"
CHECKS = (1, 2, 3, 4)
RB_SHOTS = 10000
SUMMARY_NAME = merge.SUMMARY_NAME


def configurations() -> dict[str, dict]:
    out = {}
    for campaign in merge.CAMPAIGNS.values():
        for label, (production, extension) in campaign["pairs"].items():
            out[label] = {
                "production": production,
                "extension": extension,
                "p_split": campaign["p_split"],
                "p_max": campaign.get("p_max", np.inf),
                "threshold_config": campaign.get(
                    "threshold_config", merge.THRESHOLD_CONFIG),
                "merged_dir": campaign["out"] / label,
            }
    return out


def per_sequence_fidelity(tallies: np.ndarray) -> np.ndarray:
    """|sum_i omega^i n_i| / N per (depth, sequence); NaN with no accepted shot.

    The last axis holds one tally per logical outcome, so its length is the
    dimension d and omega = exp(2 pi i / d).
    """
    tallies = np.asarray(tallies, dtype=float)
    dimension = tallies.shape[-1]
    omega_powers = np.exp(2j * np.pi * np.arange(dimension) / dimension)
    accepted = tallies.sum(axis=-1)
    amplitude = np.abs(tallies @ omega_powers)
    with np.errstate(invalid="ignore", divide="ignore"):
        return amplitude / accepted


class Source:
    """One run folder: per-sequence LRB fidelities and RB results by p index."""

    def __init__(self, run: str, cache_dir: Path) -> None:
        self.run = run
        self.path = merge.RUNS_ROOT / run
        self.plotter = merge.plotter_for(run)
        self.depths = list(self.plotter.depths)
        self.probabilities = [float(p) for p in self.plotter.probabilities]
        self.cache = cache_dir / f"rb_per_sequence_{run}.npz"
        self.lrb: dict[tuple[int, int], np.ndarray] = {}
        self.rb: dict[int, np.ndarray] = {}

    def load_lrb(self) -> None:
        for idx in range(len(self.probabilities)):
            folder = (self.path / "progress" / str(idx) / ".checkpoint_current"
                      / "unif_check_data")
            for k in CHECKS:
                tallies = np.load(folder / f"{k}.npy")
                self.lrb[(idx, k)] = per_sequence_fidelity(tallies)

    def load_or_simulate_rb(self) -> None:
        if self.cache.exists():
            data = np.load(self.cache)
            self.rb = {int(key[3:]): data[key] for key in data.files}
            return
        num_cliffs = int(self.plotter.num_cliffs)
        for idx in range(len(self.probabilities)):
            experiments = [
                [read_circuit(str(self.path / "experiments" / "RB" / str(seq)
                                  / str(idx) / f"{d}.chp"))
                 for d in range(len(self.depths))]
                for seq in range(num_cliffs)
            ]
            record = LRBSimulationPipeline.RB(
                experiments=experiments,
                depths=list(self.depths),
                shots=RB_SHOTS,
                simulation_backend="dem",
            )
            _, fidelities, _, _ = LRBSimulationPipeline.extract_statistics(
                record, dimension=int(self.plotter.d_dim))
            self.rb[idx] = np.asarray(fidelities, dtype=float)
        np.savez(self.cache, **{f"idx{k}": v for k, v in self.rb.items()})


def stats_list(fidelity: np.ndarray, keep_depth: np.ndarray) -> list[dict]:
    means = fidelity.mean(axis=1)
    stds = fidelity.std(axis=1)
    means = np.where(keep_depth, means, np.nan)
    stds = np.where(keep_depth, stds, np.nan)
    return [{"mean": float(m), "std": float(s)} for m, s in zip(means, stds)]


def error_rate(plotter, fit) -> float:
    if fit is None or not np.isfinite(fit.get("f", np.nan)):
        return np.nan
    return 1.0 - plotter._decay_to_fidelity(float(fit["f"]))


def threshold_from_errors(plotter, cfg, p, lrb_err, rb_err) -> tuple[float, str]:
    """The notebook's point-estimate chain for one interval."""
    x = np.asarray(p, dtype=float)
    lrb = np.asarray(lrb_err, dtype=float)
    rb = np.asarray(rb_err, dtype=float)
    finite = plotter._finite_mask(x, x, lrb, rb)
    x, lrb, rb = x[finite], lrb[finite], rb[finite]
    if len(x) < 2:
        return np.nan, "insufficient_points"
    order = np.argsort(x)
    x, lrb, rb = x[order], lrb[order], rb[order]
    cut = int(min(
        plotter._trim_garbage_tail_by_monotone_dip(
            x, lrb, cfg.tail_min_prefix, cfg.tail_drop_tol_abs,
            cfg.tail_drop_tol_rel),
        plotter._trim_garbage_tail_by_monotone_dip(
            x, rb, cfg.tail_min_prefix, cfg.tail_drop_tol_abs,
            cfg.tail_drop_tol_rel),
    ))
    dummy = np.zeros(cut)
    common_x, lrb_c, rb_c, _, _ = plotter._interpolate_lrb_rb_on_overlap(
        lrb_x=x[:cut], lrb_err=lrb[:cut], rb_x=x[:cut], rb_err=rb[:cut],
        lrb_extra=(dummy,), rb_extra=(dummy,))
    if len(common_x) < 2:
        return np.nan, "insufficient_overlap_after_tail_trim"
    start = plotter._first_threshold_start_index(
        lrb_c, rb_c, cfg.ignore_first_n, cfg.err_floor)
    thr = plotter._estimate_monotone_threshold_in_p(
        common_x, lrb_c, rb_c, tol=cfg.tol,
        require_consecutive=cfg.require_consecutive, start_idx=start)
    return float(thr["threshold_p"]), str(thr["method"])


class Configuration:
    def __init__(self, label: str, spec: dict, out_dir: Path) -> None:
        self.label = label
        self.p_split = spec["p_split"]
        self.p_max = spec["p_max"]
        self.merged = pd.read_csv(spec["merged_dir"] / "unif_lrb_vs_rb_table_merged.csv")
        self.published = pd.read_csv(spec["merged_dir"] / SUMMARY_NAME)
        self.prod = Source(spec["production"], out_dir)
        self.ext = Source(spec["extension"], out_dir)
        self.cfg = spec["threshold_config"]
        # (source, p index) for every row of the merged table, in p order.
        self.rows: list[tuple[Source, int, float]] = []
        for p in sorted(self.merged["p"].unique()):
            src = (self.ext if merge.from_extension(p, self.p_split, self.p_max)
                   else self.prod)
            idx = int(np.argmin(np.abs(np.asarray(src.probabilities) - p)))
            assert abs(src.probabilities[idx] - p) < 1e-12, (src.run, p)
            self.rows.append((src, idx, float(p)))
        self.rb_published = {
            float(r.p): float(r.rb_error_rate)
            for r in self.merged[self.merged["check_num"] == 1].itertuples()}

    def keep_mask(self, src: Source, k: int) -> np.ndarray:
        depths = np.asarray(src.depths)
        if src is self.ext and k >= 2:
            return depths % 2 == 0
        return np.ones(len(depths), dtype=bool)

    def evaluate(self, draws: dict[str, np.ndarray]) -> dict[int, tuple[float, str]]:
        """p*(k) for one set of sequence draws (None = full sample)."""
        rb_err = []
        for src, idx, p in self.rows:
            seqs = draws["rb_" + ("prod" if src is self.prod else "ext")]
            full = src.rb[idx]
            sample = full if seqs is None else full[:, seqs]
            all_depths = np.ones(len(src.depths), dtype=bool)
            fit_sample, _ = src.plotter._fit_rb_from_stats(stats_list(sample, all_depths))
            fit_full, _ = src.plotter._fit_rb_from_stats(stats_list(full, all_depths))
            # Centre on the published RB value; RB was re-simulated.
            rb_err.append(self.rb_published[p]
                          + error_rate(src.plotter, fit_sample)
                          - error_rate(src.plotter, fit_full))
        out = {}
        for k in CHECKS:
            lrb_err = []
            for src, idx, p in self.rows:
                seqs = draws["lrb_" + ("prod" if src is self.prod else "ext")]
                full = src.lrb[(idx, k)]
                sample = full if seqs is None else full[:, seqs]
                fit, _ = src.plotter._fit_lrb_from_stats(
                    stats_list(sample, self.keep_mask(src, k)))
                lrb_err.append(error_rate(src.plotter, fit))
            p = [row[2] for row in self.rows]
            out[k] = threshold_from_errors(self.prod.plotter, self.cfg, p,
                                           lrb_err, rb_err)
        return out

    def validate(self) -> None:
        """Full-sample LRB must reproduce the published table and p*."""
        worst_mean = worst_err = 0.0
        for src, idx, p in self.rows:
            for k in CHECKS:
                stats = merge.LRBResultsPlotter._read_fidelity_stats_file(str(
                    src.path / "results" / "LRB" / str(idx) / "unif_check_data"
                    / f"{k}.csv"))
                published_means = np.array([s["mean"] for s in stats], dtype=float)
                ours = src.lrb[(idx, k)].mean(axis=1)
                ok = np.isfinite(published_means) & np.isfinite(ours)
                if ok.any():
                    worst_mean = max(worst_mean, float(np.max(np.abs(
                        published_means[ok] - ours[ok]))))
                fit, _ = src.plotter._fit_lrb_from_stats(
                    stats_list(src.lrb[(idx, k)], self.keep_mask(src, k)))
                ours_err = error_rate(src.plotter, fit)
                row = self.merged[(self.merged["check_num"] == k)
                                  & (np.isclose(self.merged["p"], p))]
                published_err = float(row["lrb_error_rate"].iloc[0])
                if np.isfinite(published_err) or np.isfinite(ours_err):
                    worst_err = max(worst_err, abs(published_err - ours_err)
                                    if np.isfinite(published_err) and np.isfinite(ours_err)
                                    else np.inf)
        identity = self.evaluate({key: None for key in
                                  ("lrb_prod", "lrb_ext", "rb_prod", "rb_ext")})
        print(f"  check: max |mean - published| = {worst_mean:.2e}, "
              f"max |LRB error - published| = {worst_err:.2e}")
        for k in CHECKS:
            published = float(self.published.loc[
                self.published["check_num"] == k, "threshold_p"].iloc[0])
            print(f"  check k={k}: p* full sample {identity[k][0]:.6f} vs "
                  f"published {published:.6f}")
            if not np.isclose(identity[k][0], published, rtol=0, atol=1e-9):
                raise RuntimeError(f"{self.label} k={k}: full-sample p* does "
                                   "not reproduce the published value")
        if worst_mean > 1e-9 or worst_err > 1e-9:
            raise RuntimeError(f"{self.label}: full-sample LRB data do not "
                               "reproduce the published table")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("label", choices=sorted(configurations()))
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args(argv)

    out_dir = OUT_ROOT / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    conf = Configuration(args.label, configurations()[args.label], out_dir)
    start = time.time()
    for src in (conf.prod, conf.ext):
        src.load_lrb()
        src.load_or_simulate_rb()
    print(f"=== {args.label}: data ready in {time.time() - start:.0f} s")
    conf.validate()

    num = {name: int(src.plotter.num_cliffs) for name, src in
           (("prod", conf.prod), ("ext", conf.ext))}
    rng = np.random.default_rng(args.seed)
    replicates = []
    for rep in range(args.reps):
        draws = {f"{kind}_{name}": rng.integers(0, num[name], num[name])
                 for kind in ("lrb", "rb") for name in ("prod", "ext")}
        for k, (p_star, method) in conf.evaluate(draws).items():
            replicates.append({"rep": rep, "check_num": k,
                               "p_star": p_star, "method": method})
    reps = pd.DataFrame(replicates)
    reps.to_csv(out_dir / "bootstrap_replicates.csv", index=False)

    # Leave-one-sequence-out jackknife, one sequence set at a time.
    jack_var = {k: 0.0 for k in CHECKS}
    full = {name: None for name in ("lrb_prod", "lrb_ext", "rb_prod", "rb_ext")}
    for key in full:
        n = num[key.split("_")[1]]
        values = {k: [] for k in CHECKS}
        for drop in range(n):
            draws = dict(full)
            draws[key] = np.array([s for s in range(n) if s != drop])
            for k, (p_star, _) in conf.evaluate(draws).items():
                values[k].append(p_star)
        for k in CHECKS:
            v = np.asarray(values[k], dtype=float)
            v = v[np.isfinite(v)]
            if len(v) > 1:
                jack_var[k] += (len(v) - 1) / len(v) * np.sum((v - v.mean()) ** 2)

    rows = []
    for k in CHECKS:
        published = conf.published[conf.published["check_num"] == k].iloc[0]
        values = reps.loc[reps["check_num"] == k, "p_star"].to_numpy(dtype=float)
        valid = values[np.isfinite(values)]
        q = np.percentile(valid, [2.5, 16, 50, 84, 97.5]) if len(valid) else [np.nan] * 5
        rows.append({
            "configuration": args.label,
            "interval": k,
            "p_star": float(published["threshold_p"]),
            "seq_boot_se": float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan,
            "seq_boot_ci68_low": q[1], "seq_boot_ci68_high": q[3],
            "seq_boot_ci95_low": q[0], "seq_boot_ci95_high": q[4],
            "seq_boot_median": q[2],
            "seq_boot_valid_fraction": len(valid) / max(len(values), 1),
            "seq_jackknife_se": float(np.sqrt(jack_var[k])),
            "previous_ci68_low": float(published["threshold_p_ci_low"]),
            "previous_ci68_high": float(published["threshold_p_ci_high"]),
            "replicates": len(values),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "sequence_bootstrap_summary.csv", index=False)
    for r in rows:
        print(f"  k={r['interval']}: p*={r['p_star']:.4f}  sequence-bootstrap SE "
              f"{r['seq_boot_se']:.4f} (jackknife {r['seq_jackknife_se']:.4f}), "
              f"68% [{r['seq_boot_ci68_low']:.4f}, {r['seq_boot_ci68_high']:.4f}], "
              f"previous 68% [{r['previous_ci68_low']:.4f}, {r['previous_ci68_high']:.4f}], "
              f"valid {r['seq_boot_valid_fraction']:.3f}")
    print(f"  done in {time.time() - start:.0f} s; outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
