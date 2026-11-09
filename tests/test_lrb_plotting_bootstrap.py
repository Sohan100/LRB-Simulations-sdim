"""The RB error-rate bootstrap refits with the same model as the RB point fit."""

from __future__ import annotations

import numpy as np

from lrb.lrb_plotting import LRBPlotFitConfig, LRBResultsPlotter


DEPTHS = np.array([0, 2, 4, 6, 10, 14, 18, 20, 22], dtype=float)


def make_plotter(rb_a_fixed: float | None) -> LRBResultsPlotter:
    """Build a plotter with only the attributes the fit helpers read."""
    plotter = LRBResultsPlotter.__new__(LRBResultsPlotter)
    plotter.fit_config = LRBPlotFitConfig(
        a_fixed=0.0,
        rb_a_fixed=rb_a_fixed,
        min_fit_points_lrb=2,
        min_fit_points_rb=3,
        f_min=1e-8,
        f_max=1.0,
    )
    plotter.num_cliffs = 30
    plotter.d_dim = 3
    return plotter


def rb_bootstrap(plotter: LRBResultsPlotter, means, stds):
    return plotter._bootstrap_error_rate_from_fit_series(
        depths=DEPTHS,
        means=means,
        std_values=stds,
        fit_kind="rb",
        n_boot=300,
        ci_level=0.68,
        use_sem=True,
        rng=np.random.default_rng(12345),
    )


def test_fixed_offset_rb_bootstrap_brackets_point_estimate():
    plotter = make_plotter(rb_a_fixed=0.0)
    means = 0.995 ** DEPTHS
    stds = np.full(len(DEPTHS), 2e-3)
    stds[0] = 0.0
    point = plotter._fit_decay_parameter_fixed_a(
        DEPTHS, means, stds, a_fixed=0.0)
    error_rate = 1.0 - plotter._decay_to_fidelity(point["f"])

    std, low, high = rb_bootstrap(plotter, means, stds)

    assert low <= error_rate <= high
    # A free-offset refit trades the offset against the decay and inflates
    # the spread to the size of the error rate itself.
    assert std < 0.1 * error_rate


def test_free_offset_rb_bootstrap_still_available():
    plotter = make_plotter(rb_a_fixed=None)
    means = 0.2 + 0.8 * 0.99 ** DEPTHS
    stds = np.full(len(DEPTHS), 2e-3)
    point = plotter._fit_decay_parameter_free_a(DEPTHS, means, stds)
    assert abs(point["a"] - 0.2) < 1e-3

    std, low, high = rb_bootstrap(plotter, means, stds)

    assert np.isfinite(std) and low <= high
