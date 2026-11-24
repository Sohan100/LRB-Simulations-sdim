"""Pipeline estimator for this study's sampled data.

Uses the production plotter's own fit and threshold code
(``src/lrb/lrb_plotting.py`` and the notebook's crossing chain as
reimplemented in ``scripts/sequence_bootstrap_thresholds.py``):

* LRB: per-sequence |sum_i omega^i n_i|/N, mean and std over sequences per
  depth, fit y = b f^m with offset 0, weights 1/std^2, all depths incl. m = 0
  (even depths only for k >= 2 on extension-style grids when requested).
* RB: same statistic, fit with offset fixed at 0 (production FIT_CONFIG).
* error rate r = (d-1)(1-f)/d per forward+inverse pair.
* p*: the notebook crossing chain with SCALE_FREE_THRESHOLD_CONFIG
  (tol 0, no error floor), the dimension-neutral rule used for the qubit
  campaigns.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import qslib  # noqa: F401
sys.path.insert(0, qslib.REPO_SCRIPTS)
import merge_highp_threshold_extension as merge  # noqa: E402
import sequence_bootstrap_thresholds as sbt  # noqa: E402
from lrb.lrb_plotting import LRBResultsPlotter  # noqa: E402

FIT_CONFIG = merge.FIT_CONFIG
SCALE_FREE = merge.SCALE_FREE_THRESHOLD_CONFIG
THRESHOLD_CONFIG = merge.THRESHOLD_CONFIG
stats_list = sbt.stats_list
threshold_from_errors = sbt.threshold_from_errors


def fake_plotter(folder: Path, q: int, depths, probs, n_seq: int, noise: str,
                 code_name: str = "folded_qudit") -> LRBResultsPlotter:
    """A plotter bound to a parameter-only folder (no circuits, no results)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "depths.txt").write_text(",".join(str(int(d)) for d in depths))
    (folder / "probs.txt").write_text(",".join(repr(float(p)) for p in probs))
    (folder / "d.txt").write_text(str(int(q)))
    (folder / "num_cliffs.txt").write_text(str(int(n_seq)))
    (folder / "code_name.txt").write_text(code_name)
    (folder / "noise_model.txt").write_text(noise)
    (folder / "check_unif.txt").write_text("1,2,3,4,5,6")
    (folder / "check_const.txt").write_text("1")
    return LRBResultsPlotter(working_folder=str(folder) + "/", fit_config=FIT_CONFIG)


def error_rate(plotter, fit) -> float:
    return sbt.error_rate(plotter, fit)


def fit_lrb(plotter, pol: np.ndarray, keep: np.ndarray | None = None):
    """pol: (depth, seq) per-sequence polarization (NaN = no accepted shot)."""
    if keep is None:
        keep = np.ones(pol.shape[0], dtype=bool)
    fit, _ = plotter._fit_lrb_from_stats(stats_list(pol, keep))
    return fit, error_rate(plotter, fit)


def fit_rb(plotter, pol: np.ndarray):
    fit, _ = plotter._fit_rb_from_stats(stats_list(pol, np.ones(pol.shape[0], dtype=bool)))
    return fit, error_rate(plotter, fit)


def rb_exact_polarization(q: int, noise: str, words_by_seq, depths, p: float,
                          compile_mode: str = "production") -> np.ndarray:
    """Exact per-sequence RB polarization: prod_i lambda^(2 n_i).

    Every token of a physical RB Clifford is followed by DEP1 at s*p
    (s = 0.1 SI1000, 1 SD6; the identity token gets idle noise at the same
    scale), preparation and readout are ideal, and DEP1 commutes with every
    Clifford, so the Z polarization after the mirror circuit is lambda^N with
    N the total token count and lambda = 1 - s p q^2/(q^2-1).
    """
    s = 0.1 if noise == "si1000" else 1.0
    lam = 1.0 - s * p * q * q / (q * q - 1.0)
    out = np.zeros((len(depths), len(words_by_seq)))
    for j, words in enumerate(words_by_seq):
        n = np.array([qslib.token_count(w) for w in words])
        for i, d in enumerate(depths):
            out[i, j] = lam ** (2 * int(n[:d].sum()))
    return out
