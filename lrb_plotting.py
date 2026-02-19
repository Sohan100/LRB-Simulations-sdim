"""Plotting helpers for LRB/RB result analysis.

This module defines object-oriented plotting utilities that read one prepared
LRB run folder and generate summary figures for uniform and constant
stabilizer-check strategies.

Uniform-check plots include fitted LRB and RB decay curves.
Constant-check plots intentionally do not use fitting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from experiment_setup import ExperimentSetupManager
from lrb_simulation import LRBSimulationPipeline

try:
    import pandas as pd
except Exception:
    pd = None


# Fit controls are separate so notebooks can tune behavior without touching
# plotting logic.
@dataclass(frozen=True)
class LRBPlotFitConfig:
    """
    Fit configuration for LRB and RB plotting workflows.

    Attributes:
        a_fixed (float): Fixed LRB asymptote used in fixed-offset LRB fits.
        y_ceiling (float | None): Optional upper bound for LRB fit points.
        min_fit_points_lrb (int): Minimum samples required for LRB fitting.
        min_fit_points_rb (int): Minimum samples required for RB fitting.
        f_min (float): Lower decay-parameter bound used in fit scans.
        f_max (float): Upper decay-parameter bound used in fit scans.

    Methods:
        This dataclass is declarative and defines no custom methods.
    """

    a_fixed: float = 0.0
    y_ceiling: float | None = 1.0
    min_fit_points_lrb: int = 2
    min_fit_points_rb: int = 2
    f_min: float = 1e-8
    f_max: float = 1.000000


@dataclass(frozen=True)
class LRBThresholdConfig:
    """
    Configuration for monotone threshold estimation from unif error-rate data.

    Attributes:
        tol (float): Allowed negative margin for ``lrb_err - rb_err``.
        require_consecutive (int): Required consecutive onset points.
        ignore_first_n (int): Number of initial sorted-p points to ignore.
        err_floor (float | None): Optional minimum error level gate.
        tail_min_prefix (int): Prefix length protected from tail trimming.
        tail_drop_tol_abs (float): Absolute drop threshold for tail trim.
        tail_drop_tol_rel (float): Relative drop threshold for tail trim.
        p_min (float | None): Optional lower p bound filter.
        p_max (float | None): Optional upper p bound filter.
        min_lrb_n_points_keep (int): Optional LRB fit-point filter.
        min_rb_n_points_keep (int): Optional RB fit-point filter.
        zoom_half_window_points (int): Half-window around onset for zoom.
        zoom_pad_frac (float): Fractional padding in zoomed error plot.
        min_zoom_span (float): Minimum x/y span for zoomed axes.
        p_window_before_points (int): Points before onset in parametric plot.
        p_window_after_points (int): Points after onset in parametric plot.
        min_points_in_view (int): Minimum samples shown in parametric view.
        view_pad_frac (float): Fractional axis padding in parametric view.

    Methods:
        This dataclass is declarative and defines no custom methods.
    """

    tol: float = 5e-4
    require_consecutive: int = 2
    ignore_first_n: int = 3
    err_floor: float | None = 2e-3
    tail_min_prefix: int = 6
    tail_drop_tol_abs: float = 0.01
    tail_drop_tol_rel: float = 0.25
    p_min: float | None = None
    p_max: float | None = None
    min_lrb_n_points_keep: int = 0
    min_rb_n_points_keep: int = 0
    zoom_half_window_points: int = 4
    zoom_pad_frac: float = 0.25
    min_zoom_span: float = 0.01
    p_window_before_points: int = 8
    p_window_after_points: int = 5
    min_points_in_view: int = 6
    view_pad_frac: float = 0.15


class LRBResultsPlotter:
    """
    Generate LRB/RB summary plots from one completed run folder.

    The class loads run metadata once, reads per-probability CSV results, and
    produces plots for each stabilizer-check policy value.

    Attributes:
        working_folder (str): Prepared run folder path.
        fit_config (LRBPlotFitConfig): Fitting controls for uniform plots.
        depths (list[int]): Loaded benchmark depths.
        probabilities (list[float]): Loaded error-probability sweep.
        stab_const (list[int]): Constant-check policy values.
        stab_unif (list[int]): Uniform-check policy values.
        lrb_root (str): Root path for LRB results.
        rb_root (str): Root path for RB results.
        out_dir (str): Output folder for generated plots.
        d_dim (int): Logical dimension used for fidelity conversion.

    Methods:
        plot_all_unif_checks(show=True): Plot all uniform checks with fits.
        plot_all_const_checks(show=True): Plot all constant checks without
            fits.
        plot_one_unif_check(check_num, show=True): Plot one uniform check.
        plot_one_const_check(check_num, show=True): Plot one constant check.
    """

    def __init__(
        self,
        working_folder: str,
        fit_config: LRBPlotFitConfig | None = None,
    ) -> None:
        """
        Initialize the run-specific plotter and load run parameters.

        Args:
            working_folder (str): Path to one run directory.
            fit_config (LRBPlotFitConfig | None): Optional fit controls.

        Returns:
            None: Constructor populates object state.

        Raises:
            FileNotFoundError: If required run metadata is missing.
            ValueError: If run metadata cannot be parsed.
        """
        self.working_folder = working_folder
        self.fit_config = fit_config or LRBPlotFitConfig()
        # Load all run metadata once; plotting calls reuse cached attributes.
        self._load_run_params()

    def _load_run_params(self) -> None:
        """
        Load run metadata, result roots, and output directory paths.

        Args:
            None: Uses object configuration.

        Returns:
            None: Updates object attributes.

        Raises:
            ValueError: If run metadata files contain malformed values.
        """
        self.depths = [
            int(v) for v in ExperimentSetupManager.fetch_list(
                os.path.join(self.working_folder, "depths.txt"))
        ]
        self.probabilities = [
            float(v) for v in ExperimentSetupManager.fetch_list(
                os.path.join(self.working_folder, "probs.txt"))
        ]

        # Check arrays are optional in legacy runs, so handle empty files.
        const_raw = ExperimentSetupManager.fetch_list(
            os.path.join(self.working_folder, "check_const.txt")
        )
        unif_raw = ExperimentSetupManager.fetch_list(
            os.path.join(self.working_folder, "check_unif.txt")
        )

        self.stab_const = [] if not const_raw else [int(v) for v in const_raw]
        self.stab_unif = [] if not unif_raw else [int(v) for v in unif_raw]

        self.lrb_root = os.path.join(self.working_folder, "results", "LRB")
        self.rb_root = os.path.join(self.working_folder, "results", "RB")
        self.out_dir = os.path.join(self.working_folder, "results", "plots")
        os.makedirs(self.out_dir, exist_ok=True)

        # Keep a sane fallback dimension when d.txt is absent.
        d_path = os.path.join(self.working_folder, "d.txt")
        self.d_dim = 3
        if os.path.exists(d_path):
            value = ExperimentSetupManager.fetch_single_param(d_path)
            if value not in ("", "0"):
                self.d_dim = int(value)

    @staticmethod
    def _extract_series(
        stats_list: list[dict[str, Any]] | None,
        field: str,
        n_depths: int,
    ) -> list[float]:
        """
        Extract one field across depth-indexed stats entries.

        Args:
            stats_list (list[dict[str, Any]] | None): Per-depth stats list.
            field (str): Field name to extract.
            n_depths (int): Expected number of depths.

        Returns:
            list[float]: Extracted values, with NaN for missing entries.

        Raises:
            ValueError: Not raised directly by this method.
        """
        values: list[float] = []
        for depth_index in range(n_depths):
            value = np.nan
            if stats_list is not None and depth_index < len(stats_list):
                entry = stats_list[depth_index]
                if entry is not None and field in entry:
                    raw = entry[field]
                    if raw is not None:
                        value = float(raw)
            values.append(value)
        return values

    @staticmethod
    def _mask_invalid(
        depths: list[int],
        means: list[float],
        errors: list[float],
    ) -> tuple[list[int], list[float], list[float]]:
        """
        Remove rows containing invalid mean/error values.

        Args:
            depths (list[int]): Depth axis values.
            means (list[float]): Mean values aligned to depths.
            errors (list[float]): Standard-deviation values aligned to depths.

        Returns:
            tuple[list[int], list[float], list[float]]: Filtered arrays.

        Raises:
            ValueError: Not raised directly by this method.
        """
        x: list[int] = []
        y: list[float] = []
        e: list[float] = []
        for depth, mean, err in zip(depths, means, errors):
            if np.isnan(mean) or np.isnan(err):
                continue
            x.append(depth)
            y.append(mean)
            e.append(err)
        return x, y, e

    @staticmethod
    def _safe_weights(errors: np.ndarray) -> np.ndarray:
        """
        Convert standard deviations into numerically safe WLS weights.

        Args:
            errors (np.ndarray): Error-vector input.

        Returns:
            np.ndarray: Weight vector ``1 / sigma^2`` with finite values.

        Raises:
            ValueError: Not raised directly by this method.
        """
        err = np.asarray(errors, dtype=float)
        err = np.where(np.isnan(err), np.inf, err)
        finite = err[np.isfinite(err)]
        median = float(np.median(finite)) if len(finite) else 1.0
        floor = max(median * 1e-3, 1e-12)
        err = np.where(err <= 0, floor, err)
        return 1.0 / (err * err)

    @staticmethod
    def _apply_ceiling_cut(
        x: list[int],
        y: list[float],
        e: list[float],
        y_ceiling: float | None,
    ) -> tuple[list[int], list[float], list[float]]:
        """
        Filter out points above a configured y-value ceiling.

        Args:
            x (list[int]): Depth values.
            y (list[float]): Mean values.
            e (list[float]): Error values.
            y_ceiling (float | None): Ceiling or ``None`` for no cut.

        Returns:
            tuple[list[int], list[float], list[float]]: Filtered arrays.

        Raises:
            ValueError: Not raised directly by this method.
        """
        if y_ceiling is None:
            return x, y, e
        x_out: list[int] = []
        y_out: list[float] = []
        e_out: list[float] = []
        for depth, mean, err in zip(x, y, e):
            if mean <= y_ceiling:
                x_out.append(depth)
                y_out.append(mean)
                e_out.append(err)
        return x_out, y_out, e_out

    def _fit_decay_parameter_fixed_a(
        self,
        depths: list[int],
        means: list[float],
        errors: list[float],
    ) -> dict[str, float] | None:
        """
        Fit ``y = a_fixed + b * f^depth`` using a weighted f-grid scan.

        Args:
            depths (list[int]): Input depth axis.
            means (list[float]): Mean values for fitting.
            errors (list[float]): Standard deviations for weighting.

        Returns:
            dict[str, float] | None: Best-fit parameters, or ``None``.

        Raises:
            ValueError: Not raised directly by this method.
        """
        x = np.asarray(depths, dtype=float)
        y = np.asarray(means, dtype=float)
        e = np.asarray(errors, dtype=float)

        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
        x, y, e = x[ok], y[ok], e[ok]

        if len(x) < self.fit_config.min_fit_points_lrb:
            return None

        w = self._safe_weights(e)
        y0 = y - float(self.fit_config.a_fixed)
        low = float(self.fit_config.f_min)
        high = float(self.fit_config.f_max)
        best: dict[str, float] | None = None

        for _ in range(5):
            # Coarse-to-fine grid scan around the current best decay value.
            f_grid = np.linspace(low, high, 2001)
            z = np.power(f_grid[None, :], x[:, None])
            num = np.sum(w[:, None] * z * y0[:, None], axis=0)
            den = np.sum(w[:, None] * z * z, axis=0)
            den = np.where(den <= 0, np.nan, den)
            b = num / den
            resid = y0[:, None] - b[None, :] * z
            chi2 = np.sum(w[:, None] * resid * resid, axis=0)
            chi2 = np.where(np.isfinite(chi2), chi2, np.inf)

            idx = int(np.argmin(chi2))
            f_best = float(f_grid[idx])
            best = {
                "f": f_best,
                "a": float(self.fit_config.a_fixed),
                "b": float(b[idx]),
                "chi2": float(chi2[idx]),
                "n_points": float(len(x)),
            }

            span = (high - low) / 10.0
            low = max(self.fit_config.f_min, f_best - span)
            high = min(self.fit_config.f_max, f_best + span)

        return best

    def _fit_decay_parameter_free_a(
        self,
        depths: list[int],
        means: list[float],
        errors: list[float],
    ) -> dict[str, float] | None:
        """
        Fit ``y = a + b * f^depth`` using a weighted f-grid scan.

        Args:
            depths (list[int]): Input depth axis.
            means (list[float]): Mean values for fitting.
            errors (list[float]): Standard deviations for weighting.

        Returns:
            dict[str, float] | None: Best-fit parameters, or ``None``.

        Raises:
            ValueError: Not raised directly by this method.
        """
        x = np.asarray(depths, dtype=float)
        y = np.asarray(means, dtype=float)
        e = np.asarray(errors, dtype=float)

        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
        x, y, e = x[ok], y[ok], e[ok]

        if len(x) < self.fit_config.min_fit_points_rb:
            return None

        w = self._safe_weights(e)
        s = float(np.sum(w))
        sy = float(np.sum(w * y))
        low = float(self.fit_config.f_min)
        high = float(self.fit_config.f_max)
        best: dict[str, float] | None = None

        for _ in range(5):
            # Scan decay values and solve (a, b) analytically at each point.
            f_grid = np.linspace(low, high, 2001)
            z = np.power(f_grid[None, :], x[:, None])
            sz = np.sum(w[:, None] * z, axis=0)
            szz = np.sum(w[:, None] * z * z, axis=0)
            szy = np.sum(w[:, None] * z * y[:, None], axis=0)
            det = (s * szz) - (sz * sz)
            det = np.where(np.abs(det) < 1e-20, np.nan, det)
            a = (szz * sy - sz * szy) / det
            b = (s * szy - sz * sy) / det
            resid = y[:, None] - (a[None, :] + b[None, :] * z)
            chi2 = np.sum(w[:, None] * resid * resid, axis=0)
            chi2 = np.where(np.isfinite(chi2), chi2, np.inf)

            idx = int(np.argmin(chi2))
            f_best = float(f_grid[idx])
            best = {
                "f": f_best,
                "a": float(a[idx]),
                "b": float(b[idx]),
                "chi2": float(chi2[idx]),
                "n_points": float(len(x)),
            }

            span = (high - low) / 10.0
            low = max(self.fit_config.f_min, f_best - span)
            high = min(self.fit_config.f_max, f_best + span)

        return best

    def _fit_lrb_from_stats(
        self,
        stats_list: list[dict[str, Any]] | None,
    ) -> tuple[dict[str, float] | None, tuple[list[int], list[float],
                                              list[float]]]:
        """
        Build filtered LRB series and apply fixed-a fitting.

        Args:
            stats_list (list[dict[str, Any]] | None): Per-depth LRB stats.

        Returns:
            tuple[dict[str, float] | None, tuple[list[int], list[float],
                list[float]]]: Fit result and plotted data vectors.

        Raises:
            ValueError: Not raised directly by this method.
        """
        means = self._extract_series(stats_list, "mean", len(self.depths))
        errs = self._extract_series(stats_list, "std", len(self.depths))
        x, y, e = self._mask_invalid(self.depths, means, errs)
        x, y, e = self._apply_ceiling_cut(x, y, e, self.fit_config.y_ceiling)
        fit = self._fit_decay_parameter_fixed_a(x, y, e)
        return fit, (x, y, e)

    def _fit_rb_from_stats(
        self,
        stats_list: list[dict[str, Any]] | None,
    ) -> tuple[dict[str, float] | None, tuple[list[int], list[float],
                                              list[float]]]:
        """
        Build filtered RB series and apply free-a fitting.

        Args:
            stats_list (list[dict[str, Any]] | None): Per-depth RB stats.

        Returns:
            tuple[dict[str, float] | None, tuple[list[int], list[float],
                list[float]]]: Fit result and plotted data vectors.

        Raises:
            ValueError: Not raised directly by this method.
        """
        means = self._extract_series(stats_list, "mean", len(self.depths))
        errs = self._extract_series(stats_list, "std", len(self.depths))
        x, y, e = self._mask_invalid(self.depths, means, errs)
        fit = self._fit_decay_parameter_free_a(x, y, e)
        return fit, (x, y, e)

    def _decay_to_fidelity(self, decay: float) -> float:
        """
        Convert fitted decay value into average fidelity.

        Args:
            decay (float): Decay parameter ``f``.

        Returns:
            float: Average fidelity computed from ``f`` and ``d_dim``.

        Raises:
            ValueError: Not raised directly by this method.
        """
        return (1.0 + (self.d_dim - 1.0) * float(decay)) / float(self.d_dim)

    def _read_pair(
        self,
        check_type: str,
        check_num: int,
        prob_index: int,
    ) -> tuple[bool, tuple[Any, Any, Any], bool, tuple[Any, Any, Any]]:
        """
        Read one LRB results file and optional RB companion file.

        Args:
            check_type (str): ``"unif"`` or ``"const"``.
            check_num (int): Check policy value.
            prob_index (int): Probability index.

        Returns:
            tuple[bool, tuple[Any, Any, Any], bool, tuple[Any, Any, Any]]:
            Read-status flags and parsed result triples.

        Raises:
            ValueError: If ``check_type`` is unsupported.
        """
        if check_type not in ("unif", "const"):
            raise ValueError(f"Unsupported check_type: {check_type}")

        # Uniform and constant results are stored in parallel subfolders.
        data_dir = "unif_check_data" if check_type == "unif" else \
            "const_check_data"

        lrb_path = os.path.join(
            self.lrb_root,
            str(prob_index),
            data_dir,
            f"{check_num}.csv",
        )
        rb_path = os.path.join(self.rb_root, f"{prob_index}.csv")

        ok_lrb = False
        ok_rb = False
        lrb = (None, None, None)
        rb = (None, None, None)

        if os.path.exists(lrb_path):
            lrb = LRBSimulationPipeline.read_stats(lrb_path)
            ok_lrb = True
        else:
            print(f"[WARN] missing LRB file: {lrb_path}")

        if os.path.exists(rb_path):
            rb = LRBSimulationPipeline.read_stats(rb_path)
            ok_rb = True
        else:
            print(f"[WARN] missing RB file: {rb_path}")

        return ok_lrb, lrb, ok_rb, rb

    def _build_rows_for_check(
        self,
        check_type: str,
        check_num: int,
    ) -> list[dict[str, Any]]:
        """
        Collect parsed series data for all probabilities at one check value.

        Args:
            check_type (str): ``"unif"`` or ``"const"``.
            check_num (int): Selected check policy value.

        Returns:
            list[dict[str, Any]]: Parsed per-probability plotting rows.

        Raises:
            ValueError: Propagated for unsupported check type.
        """
        rows: list[dict[str, Any]] = []
        for prob_index, prob in enumerate(self.probabilities):
            ok_lrb, lrb, ok_rb, rb = self._read_pair(
                check_type=check_type,
                check_num=check_num,
                prob_index=prob_index,
            )
            if not ok_lrb:
                continue

            _, lrb_f, lrb_r = lrb
            rb_f = rb[1] if ok_rb else None
            rows.append(
                {
                    "p": prob,
                    "lrb_f": lrb_f,
                    "lrb_r": lrb_r,
                    "rb_f": rb_f,
                }
            )
        return rows

    def _plot_rejected_panel(
        self,
        ax: Any,
        lrb_r_stats: list[dict[str, Any]] | None,
        prob_value: float,
    ) -> None:
        """
        Plot rejected-run counts for one probability row.

        Args:
            ax (Any): Matplotlib axis target.
            lrb_r_stats (list[dict[str, Any]] | None): Rejected-run stats.
            prob_value (float): Probability value for title annotation.

        Returns:
            None: Draws directly on the axis.

        Raises:
            ValueError: Not raised directly by this method.
        """
        means = self._extract_series(lrb_r_stats, "mean", len(self.depths))
        errs = self._extract_series(lrb_r_stats, "std", len(self.depths))
        x, y, e = self._mask_invalid(self.depths, means, errs)
        ax.errorbar(x, y, yerr=e, fmt="-o")
        y_max = (max(y) * 1.1) if y else 1.0
        ax.set_ylim(0, y_max if y_max > 0 else 1.0)
        ax.set_title(f"Rejected runs (p = {prob_value:.3g})")
        ax.set_ylabel("# rejected runs")

    def _plot_data_only_series(
        self,
        ax: Any,
        lrb_f_stats: list[dict[str, Any]] | None,
        rb_f_stats: list[dict[str, Any]] | None,
    ) -> None:
        """
        Plot LRB and RB data points only, without any fitted curves.

        Args:
            ax (Any): Matplotlib axis target.
            lrb_f_stats (list[dict[str, Any]] | None): LRB fidelity stats.
            rb_f_stats (list[dict[str, Any]] | None): RB fidelity stats.

        Returns:
            None: Draws directly on the axis.

        Raises:
            ValueError: Not raised directly by this method.
        """
        lrb_means = self._extract_series(lrb_f_stats, "mean", len(self.depths))
        lrb_errs = self._extract_series(lrb_f_stats, "std", len(self.depths))
        x_l, y_l, e_l = self._mask_invalid(self.depths, lrb_means, lrb_errs)
        ax.errorbar(x_l, y_l, yerr=e_l, fmt="o", label="LRB")

        if rb_f_stats is not None:
            rb_means = self._extract_series(
                rb_f_stats, "mean", len(self.depths))
            rb_errs = self._extract_series(rb_f_stats, "std", len(self.depths))
            x_r, y_r, e_r = self._mask_invalid(self.depths, rb_means, rb_errs)
            ax.errorbar(x_r, y_r, yerr=e_r, fmt="s", label="RB")

        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Expectation Value of Logical X")
        ax.legend(fontsize=8)

    def _plot_fit_series(
        self,
        ax: Any,
        lrb_f_stats: list[dict[str, Any]] | None,
        rb_f_stats: list[dict[str, Any]] | None,
    ) -> None:
        """
        Plot fitted LRB and RB decay curves with fidelity legend labels.

        Args:
            ax (Any): Matplotlib axis target.
            lrb_f_stats (list[dict[str, Any]] | None): LRB fidelity stats.
            rb_f_stats (list[dict[str, Any]] | None): RB fidelity stats.

        Returns:
            None: Draws directly on the axis.

        Raises:
            ValueError: Not raised directly by this method.
        """
        lrb_fit, (x_l, y_l, e_l) = self._fit_lrb_from_stats(lrb_f_stats)
        if lrb_fit is not None:
            f_l = lrb_fit["f"]
            fid_l = self._decay_to_fidelity(f_l)
            lrb_label = f"LRB (f={f_l:.6f}, F={fid_l:.6f})"
        else:
            lrb_label = "LRB (fit failed)"
        lrb_data = ax.errorbar(x_l, y_l, yerr=e_l, fmt="o", label=lrb_label)
        if lrb_fit is not None:
            x_fit = np.array(sorted(set(x_l)), dtype=float)
            y_fit = lrb_fit["a"] + lrb_fit["b"] * np.power(lrb_fit["f"], x_fit)
            ax.plot(
                x_fit,
                y_fit,
                linewidth=1,
                color=lrb_data.lines[0].get_color(),
            )

        if rb_f_stats is not None:
            rb_fit, (x_r, y_r, e_r) = self._fit_rb_from_stats(rb_f_stats)
            if rb_fit is not None:
                f_r = rb_fit["f"]
                fid_r = self._decay_to_fidelity(f_r)
                rb_label = f"RB (f={f_r:.6f}, F={fid_r:.6f})"
            else:
                rb_label = "RB (fit failed)"
            rb_data = ax.errorbar(x_r, y_r, yerr=e_r, fmt="s", label=rb_label)
            if rb_fit is not None:
                x_fit = np.array(sorted(set(x_r)), dtype=float)
                y_fit = rb_fit["a"] + rb_fit["b"] * np.power(
                    rb_fit["f"], x_fit)
                ax.plot(
                    x_fit,
                    y_fit,
                    linewidth=1,
                    color=rb_data.lines[0].get_color(),
                )

        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Expectation Value of Logical X")
        ax.legend(fontsize=8)

    def _plot_check_rows(
        self,
        rows: list[dict[str, Any]],
        check_type: str,
        check_num: int,
        out_path: str,
        use_fits: bool,
        show: bool,
    ) -> str | None:
        """
        Render the full two-column summary plot for one check value.

        Args:
            rows (list[dict[str, Any]]): Per-probability row data.
            check_type (str): ``"unif"`` or ``"const"``.
            check_num (int): Check value.
            out_path (str): PDF output path.
            use_fits (bool): Whether to draw fitted curves.
            show (bool): Whether to display interactive figures.

        Returns:
            str | None: Output path if created; otherwise ``None``.

        Raises:
            ValueError: If ``rows`` are malformed.
        """
        if not rows:
            print(
                f"[INFO] no data for {check_type} check={check_num}; "
                "skipping plot."
            )
            return None

        n_rows = len(rows)
        # Column 0: logical signal curves. Column 1: rejected-run counts.
        fig, axs = plt.subplots(
            nrows=n_rows,
            ncols=2,
            figsize=(10, n_rows * 3),
            sharex="col",
            constrained_layout=True,
        )

        def cell(row: int, col: int) -> Any:
            if n_rows > 1:
                return axs[row][col]
            return axs[col]

        for row_index, row in enumerate(rows):
            prob_value = float(row["p"])
            lrb_f = row["lrb_f"]
            lrb_r = row["lrb_r"]
            rb_f = row["rb_f"]

            ax_main = cell(row_index, 0)
            if use_fits:
                # Uniform plots use fitted overlays; const plots are data-only.
                self._plot_fit_series(
                    ax_main,
                    lrb_f_stats=lrb_f,
                    rb_f_stats=rb_f,
                )
            else:
                self._plot_data_only_series(
                    ax_main, lrb_f_stats=lrb_f, rb_f_stats=rb_f)
            ax_main.set_title(f"p = {prob_value:.3g}")

            ax_rej = cell(row_index, 1)
            self._plot_rejected_panel(
                ax=ax_rej,
                lrb_r_stats=lrb_r,
                prob_value=prob_value,
            )

        cell(n_rows - 1, 0).set_xlabel("Depth")
        cell(n_rows - 1, 1).set_xlabel("Depth")

        mode = "Fit" if use_fits else "NoFit"
        fig.suptitle(
            f"LRB vs RB ({check_type}={check_num}, q={self.d_dim}, {mode})",
            fontsize=14,
            y=1.01,
        )
        fig.savefig(out_path, dpi=300)
        if show:
            plt.show()
        plt.close(fig)
        print(f"[OK] wrote {out_path}")
        return out_path

    def plot_one_unif_check(
        self,
        check_num: int,
        show: bool = True,
    ) -> str | None:
        """
        Plot one uniform-check summary with LRB and RB fitted curves.

        Args:
            check_num (int): Uniform check interval.
            show (bool): Whether to display the figure.

        Returns:
            str | None: Output path if created; otherwise ``None``.

        Raises:
            ValueError: Propagated for invalid check values.
        """
        rows = self._build_rows_for_check(
            check_type="unif",
            check_num=check_num,
        )
        out_path = os.path.join(
            self.out_dir,
            f"unif-{check_num}-Summary-Graph-Fit.pdf",
        )
        return self._plot_check_rows(
            rows=rows,
            check_type="unif",
            check_num=check_num,
            out_path=out_path,
            use_fits=True,
            show=show,
        )

    def plot_one_const_check(
        self,
        check_num: int,
        show: bool = True,
    ) -> str | None:
        """
        Plot one constant-check summary with data-only LRB/RB traces.

        Args:
            check_num (int): Constant number of stabilizer checks.
            show (bool): Whether to display the figure.

        Returns:
            str | None: Output path if created; otherwise ``None``.

        Raises:
            ValueError: Propagated for invalid check values.
        """
        rows = self._build_rows_for_check(
            check_type="const",
            check_num=check_num,
        )
        out_path = os.path.join(
            self.out_dir,
            f"const-{check_num}-Summary-Graph-NoFit.pdf",
        )
        return self._plot_check_rows(
            rows=rows,
            check_type="const",
            check_num=check_num,
            out_path=out_path,
            use_fits=False,
            show=show,
        )

    def plot_all_unif_checks(self, show: bool = True) -> list[str]:
        """
        Generate uniform-check summary plots for all configured values.

        Args:
            show (bool): Whether to display figures during generation.

        Returns:
            list[str]: Output paths for plots that were generated.

        Raises:
            ValueError: Propagated from single-plot generation.
        """
        checks = self.stab_unif if self.stab_unif else list(range(1, 23))
        outputs: list[str] = []
        for check_num in checks:
            out_path = self.plot_one_unif_check(check_num=check_num, show=show)
            if out_path is not None:
                outputs.append(out_path)
        return outputs

    def plot_all_const_checks(self, show: bool = True) -> list[str]:
        """
        Generate constant-check summary plots for all configured values.

        Args:
            show (bool): Whether to display figures during generation.

        Returns:
            list[str]: Output paths for plots that were generated.

        Raises:
            ValueError: Propagated from single-plot generation.
        """
        checks = self.stab_const if self.stab_const else list(range(0, 23))
        outputs: list[str] = []
        for check_num in checks:
            out_path = self.plot_one_const_check(
                check_num=check_num,
                show=show,
            )
            if out_path is not None:
                outputs.append(out_path)
        return outputs

    def _safe_metrics_from_fit(
        self,
        fit: dict[str, float] | None,
    ) -> dict[str, float]:
        """
        Convert a decay fit dictionary into fidelity/error metrics.

        Args:
            fit (dict[str, float] | None): Fitted decay record.

        Returns:
            dict[str, float]: Normalized metrics with NaN fallbacks.

        Raises:
            ValueError: Not raised directly by this method.
        """
        if fit is None:
            return {
                "f": np.nan,
                "fid": np.nan,
                "err": np.nan,
                "a": np.nan,
                "b": np.nan,
                "chi2": np.nan,
                "n_points": 0.0,
            }
        decay = float(fit["f"])
        fidelity = self._decay_to_fidelity(decay)
        return {
            "f": decay,
            "fid": fidelity,
            "err": 1.0 - fidelity,
            "a": float(fit.get("a", np.nan)),
            "b": float(fit.get("b", np.nan)),
            "chi2": float(fit.get("chi2", np.nan)),
            "n_points": float(fit.get("n_points", 0.0)),
        }

    @staticmethod
    def _read_fidelity_stats_file(path: str) -> list[dict[str, Any]] | None:
        """
        Read one stats CSV and return fidelity stats if available.

        Args:
            path (str): Input CSV file path.

        Returns:
            list[dict[str, Any]] | None: Parsed fidelity stats or ``None``.

        Raises:
            OSError: Propagated if file read fails unexpectedly.
        """
        if not os.path.exists(path):
            return None
        _, fidelity_stats, _ = LRBSimulationPipeline.read_stats(path)
        return fidelity_stats

    def build_unif_lrb_vs_rb_table_mixed_fits(
        self,
        write_per_check_tables: bool = True,
    ) -> str:
        """
        Build a per-unif-check table with mixed fits and error-rate columns.

        LRB uses fixed-a fitting and RB uses free-a fitting.

        Args:
            write_per_check_tables (bool): Whether to write per-check CSVs.

        Returns:
            str: Output path of the all-check aggregate CSV.

        Raises:
            RuntimeError: If pandas is unavailable.
        """
        if pd is None:
            raise RuntimeError("pandas is required for table generation.")

        # RB is check-independent, so fit once per probability for reuse.
        checks = self.stab_unif if self.stab_unif else list(range(1, 23))
        rb_cache: dict[int, dict[str, float]] = {}
        for prob_index, prob in enumerate(self.probabilities):
            rb_path = os.path.join(self.rb_root, f"{prob_index}.csv")
            rb_stats = self._read_fidelity_stats_file(rb_path)
            rb_fit = None
            if rb_stats is not None:
                rb_fit, _ = self._fit_rb_from_stats(rb_stats)
            metrics = self._safe_metrics_from_fit(rb_fit)
            metrics["p"] = float(prob)
            rb_cache[prob_index] = metrics

        rows: list[dict[str, float | int | str]] = []
        for check_num in checks:
            for prob_index, prob in enumerate(self.probabilities):
                lrb_path = os.path.join(
                    self.lrb_root,
                    str(prob_index),
                    "unif_check_data",
                    f"{check_num}.csv",
                )
                lrb_stats = self._read_fidelity_stats_file(lrb_path)
                lrb_fit = None
                if lrb_stats is not None:
                    lrb_fit, _ = self._fit_lrb_from_stats(lrb_stats)
                lrb_m = self._safe_metrics_from_fit(lrb_fit)
                rb_m = rb_cache[prob_index]

                rows.append(
                    {
                        "check_num": int(check_num),
                        "p_index": int(prob_index),
                        "p": float(prob),
                        "lrb_f": lrb_m["f"],
                        "lrb_fidelity": lrb_m["fid"],
                        "lrb_error_rate": lrb_m["err"],
                        "lrb_a": lrb_m["a"],
                        "lrb_b": lrb_m["b"],
                        "lrb_chi2": lrb_m["chi2"],
                        "lrb_n_points": int(lrb_m["n_points"]),
                        "rb_f": rb_m["f"],
                        "rb_fidelity": rb_m["fid"],
                        "rb_error_rate": rb_m["err"],
                        "rb_a": rb_m["a"],
                        "rb_b": rb_m["b"],
                        "rb_chi2": rb_m["chi2"],
                        "rb_n_points": int(rb_m["n_points"]),
                    }
                )

        # Persist one all-check table plus optional per-check views.
        frame = pd.DataFrame(rows).sort_values(
            ["check_num", "p"]).reset_index(drop=True)
        out_all = os.path.join(
            self.out_dir,
            "unif_lrb_vs_rb_table_all_mixed_fits.csv",
        )
        frame.to_csv(out_all, index=False)
        print(f"[OK] wrote {out_all}")

        if write_per_check_tables:
            columns = [
                "p",
                "lrb_fidelity",
                "lrb_error_rate",
                "rb_fidelity",
                "rb_error_rate",
                "lrb_f",
                "rb_f",
                "lrb_a",
                "rb_a",
                "lrb_n_points",
                "rb_n_points",
            ]
            for check_num in checks:
                sub = frame[frame["check_num"] == check_num][columns].copy()
                out_one = os.path.join(
                    self.out_dir,
                    f"unif-{check_num}-lrb-vs-rb-table-mixed-fits.csv",
                )
                sub.to_csv(out_one, index=False)
                print(f"[OK] wrote {out_one}")

        return out_all

    @staticmethod
    def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
        """
        Build a finite-value mask across all arrays.

        Args:
            arrays (np.ndarray): Arrays to validate.

        Returns:
            np.ndarray: Boolean mask of jointly finite entries.

        Raises:
            ValueError: Not raised directly by this method.
        """
        mask = None
        for array in arrays:
            current = np.isfinite(np.asarray(array, dtype=float))
            mask = current if mask is None else (mask & current)
        return mask

    @staticmethod
    def _trim_garbage_tail_by_monotone_dip(
        p: np.ndarray,
        series: np.ndarray,
        min_prefix: int,
        drop_tol_abs: float,
        drop_tol_rel: float,
    ) -> int:
        """
        Trim at first significant tail drop after a protected prefix.

        Args:
            p (np.ndarray): Probability values.
            series (np.ndarray): Error-rate series.
            min_prefix (int): Prefix length not eligible for trimming.
            drop_tol_abs (float): Absolute drop threshold.
            drop_tol_rel (float): Relative drop threshold.

        Returns:
            int: Cut index where data up to ``[0:cut]`` is retained.

        Raises:
            ValueError: Not raised directly by this method.
        """
        p_arr = np.asarray(p, dtype=float)
        s_arr = np.asarray(series, dtype=float)
        mask = LRBResultsPlotter._finite_mask(p_arr, s_arr)
        _, s_arr = p_arr[mask], s_arr[mask]
        if len(s_arr) == 0:
            return 0

        running_max = float(s_arr[0])
        for index in range(1, len(s_arr)):
            running_max = max(running_max, float(s_arr[index - 1]))
            if index < int(min_prefix):
                continue
            allowed_drop = max(
                float(drop_tol_abs),
                float(drop_tol_rel) * running_max,
            )
            if float(s_arr[index]) < running_max - allowed_drop:
                return int(index)
        return int(len(s_arr))

    @staticmethod
    def _first_threshold_start_index(
        lrb_err: np.ndarray,
        rb_err: np.ndarray,
        ignore_first_n: int,
        err_floor: float | None,
    ) -> int:
        """
        Compute threshold-search start index from prefix and error floor.

        Args:
            lrb_err (np.ndarray): LRB error-rate series.
            rb_err (np.ndarray): RB error-rate series.
            ignore_first_n (int): Initial points to skip.
            err_floor (float | None): Optional minimum scale for thresholding.

        Returns:
            int: Start index for threshold search.

        Raises:
            ValueError: Not raised directly by this method.
        """
        n_points = len(lrb_err)
        start = max(0, int(ignore_first_n))
        if err_floor is None:
            return min(start, n_points)
        for idx in range(n_points):
            if lrb_err[idx] >= err_floor or rb_err[idx] >= err_floor:
                start = max(start, idx)
                break
        return min(start, n_points)

    @staticmethod
    def _estimate_monotone_threshold_in_p(
        p: np.ndarray,
        lrb_err: np.ndarray,
        rb_err: np.ndarray,
        tol: float,
        require_consecutive: int,
        start_idx: int,
    ) -> dict[str, float | str | int | None]:
        """
        Estimate the first monotone-worse threshold in probability space.

        Args:
            p (np.ndarray): Probability values.
            lrb_err (np.ndarray): LRB error rates.
            rb_err (np.ndarray): RB error rates.
            tol (float): Allowed negative margin for ``lrb-rb``.
            require_consecutive (int): Consecutive onset requirement.
            start_idx (int): First index eligible for thresholding.

        Returns:
            dict[str, float | str | int | None]: Threshold metadata.

        Raises:
            ValueError: Not raised directly by this method.
        """
        p_arr = np.asarray(p, dtype=float)
        l_arr = np.asarray(lrb_err, dtype=float)
        r_arr = np.asarray(rb_err, dtype=float)
        mask = LRBResultsPlotter._finite_mask(p_arr, l_arr, r_arr)
        p_arr, l_arr, r_arr = p_arr[mask], l_arr[mask], r_arr[mask]
        if len(p_arr) < 2 or start_idx >= len(p_arr):
            return {
                "threshold_p": np.nan,
                "threshold_rb_error": np.nan,
                "threshold_lrb_error": np.nan,
                "method": "insufficient_points",
                "onset_index": None,
            }

        order = np.argsort(p_arr)
        p_arr, l_arr, r_arr = p_arr[order], l_arr[order], r_arr[order]
        diff = l_arr - r_arr
        good = diff >= -float(tol)

        # suffix_all_good[i] means every point from i onward is acceptable.
        suffix_all_good = np.empty(len(good), dtype=bool)
        running = True
        for idx in range(len(good) - 1, -1, -1):
            running = running and bool(good[idx])
            suffix_all_good[idx] = running

        candidates = [
            int(idx) for idx in np.where(suffix_all_good)[0]
            if idx >= start_idx
        ]
        if not candidates:
            # Fall back to the start index with fewest tail violations.
            tail_violations = np.array(
                [np.sum(~good[idx:]) for idx in range(len(good))])
            idx = int(start_idx + np.argmin(tail_violations[start_idx:]))
            return {
                "threshold_p": float(p_arr[idx]),
                "threshold_rb_error": float(r_arr[idx]),
                "threshold_lrb_error": float(l_arr[idx]),
                "method": "no_permanent_regime_min_tail_violations",
                "onset_index": int(idx),
            }

        onset = None
        needed = int(require_consecutive)
        for idx in candidates:
            if needed <= 1:
                onset = idx
                break
            if idx + needed <= len(good) and np.all(good[idx:idx + needed]):
                onset = idx
                break

        if onset is None:
            onset = candidates[0]
            method = "permanent_regime_no_consecutive"
        else:
            method = "permanent_regime"

        if onset == 0:
            return {
                "threshold_p": float(p_arr[0]),
                "threshold_rb_error": float(r_arr[0]),
                "threshold_lrb_error": float(l_arr[0]),
                "method": method + "_at_start",
                "onset_index": int(onset),
            }

        d0 = float(diff[onset - 1])
        d1 = float(diff[onset])
        if not (np.isfinite(d0) and np.isfinite(d1)):
            return {
                "threshold_p": float(p_arr[onset]),
                "threshold_rb_error": float(r_arr[onset]),
                "threshold_lrb_error": float(l_arr[onset]),
                "method": method + "_no_interp",
                "onset_index": int(onset),
            }
        if d0 == d1 or d0 * d1 > 0:
            return {
                "threshold_p": float(p_arr[onset]),
                "threshold_rb_error": float(r_arr[onset]),
                "threshold_lrb_error": float(l_arr[onset]),
                "method": method + "_no_interp",
                "onset_index": int(onset),
            }

        t = -d0 / (d1 - d0)
        p_star = float(
            p_arr[onset - 1] + t * (p_arr[onset] - p_arr[onset - 1]))
        rb_star = float(
            r_arr[onset - 1] + t * (r_arr[onset] - r_arr[onset - 1]))
        lrb_star = float(
            l_arr[onset - 1] + t * (l_arr[onset] - l_arr[onset - 1]))

        return {
            "threshold_p": p_star,
            "threshold_rb_error": rb_star,
            "threshold_lrb_error": lrb_star,
            "method": method + "_interp",
            "onset_index": int(onset),
        }

    @staticmethod
    def _zoom_bounds_from_indices(
        x: np.ndarray,
        y_list: list[np.ndarray],
        i0: int,
        i1: int,
        pad_frac: float,
        min_span: float,
    ) -> tuple[float, float, float, float]:
        """
        Compute padded x/y bounds around a selected index window.

        Args:
            x (np.ndarray): x-axis values.
            y_list (list[np.ndarray]): y-axis series list.
            i0 (int): Window start index.
            i1 (int): Window end index (exclusive).
            pad_frac (float): Fractional axis padding.
            min_span (float): Minimum axis span.

        Returns:
            tuple[float, float, float, float]: ``xmin, xmax, ymin, ymax``.

        Raises:
            ValueError: Not raised directly by this method.
        """
        xw = np.asarray(x[i0:i1], dtype=float)
        ys = [np.asarray(y[i0:i1], dtype=float) for y in y_list]
        xmin = float(np.nanmin(xw))
        xmax = float(np.nanmax(xw))
        ymin = float(np.nanmin([np.nanmin(series) for series in ys]))
        ymax = float(np.nanmax([np.nanmax(series) for series in ys]))

        x_span = max(xmax - xmin, float(min_span))
        y_span = max(ymax - ymin, float(min_span))
        xmin -= pad_frac * x_span
        xmax += pad_frac * x_span
        ymin -= pad_frac * y_span
        ymax += pad_frac * y_span
        xmin = max(0.0, xmin)
        ymin = max(0.0, ymin)
        return xmin, xmax, ymin, ymax

    def _plot_error_vs_p_threshold(
        self,
        p: np.ndarray,
        lrb_err: np.ndarray,
        rb_err: np.ndarray,
        check_num: int,
        threshold: dict[str, float | str | int | None],
        cfg: LRBThresholdConfig,
        show: bool,
    ) -> str:
        """
        Plot error-rate curves versus p with threshold-focused zoom.

        Args:
            p (np.ndarray): Sorted probability values.
            lrb_err (np.ndarray): LRB error-rate values.
            rb_err (np.ndarray): RB error-rate values.
            check_num (int): Uniform check value.
            threshold (dict[str, float | str | int | None]): Threshold record.
            cfg (LRBThresholdConfig): Plot and threshold controls.
            show (bool): Whether to display the figure.

        Returns:
            str: Generated output PDF path.

        Raises:
            ValueError: Not raised directly by this method.
        """
        p_arr = np.asarray(p, dtype=float)
        l_arr = np.asarray(lrb_err, dtype=float)
        r_arr = np.asarray(rb_err, dtype=float)
        plt.figure(figsize=(8.5, 5.5))
        plt.plot(p_arr, l_arr, "-o", markersize=4, label="LRB error rate")
        plt.plot(p_arr, r_arr, "-s", markersize=4, label="RB error rate")

        threshold_p = float(threshold["threshold_p"])
        onset = threshold.get("onset_index", None)
        if np.isfinite(threshold_p):
            plt.axvline(
                threshold_p,
                linestyle="--",
                alpha=0.7,
                label=f"$p\\approx{threshold_p:.4g}$",
            )
            plt.plot(
                [threshold_p],
                [threshold["threshold_lrb_error"]],
                "o",
                markersize=9,
                alpha=0.7,
            )
            plt.plot(
                [threshold_p],
                [threshold["threshold_rb_error"]],
                "o",
                markersize=9,
                alpha=0.7,
            )

        if onset is None:
            # Use nearest threshold-p sample as a zoom anchor.
            if np.isfinite(threshold_p):
                onset = int(np.argmin(np.abs(p_arr - threshold_p)))
            else:
                onset = len(p_arr) // 2

        i0 = max(0, int(onset) - int(cfg.zoom_half_window_points))
        i1 = min(len(p_arr), int(onset) + int(cfg.zoom_half_window_points) + 1)
        xmin, xmax, ymin, ymax = self._zoom_bounds_from_indices(
            p_arr,
            [l_arr, r_arr],
            i0,
            i1,
            cfg.zoom_pad_frac,
            cfg.min_zoom_span,
        )
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel("physical noise parameter p")
        plt.ylabel("error rate (1 - fidelity)")
        plt.title(f"unif={check_num}: error rates vs p (zoom near threshold)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(
            self.out_dir,
            f"unif-{check_num}-error-vs-p-threshold-monotone.pdf",
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        print(f"[OK] wrote {out_path}")
        return out_path

    def _plot_lrb_vs_rb_parametric_threshold(
        self,
        p: np.ndarray,
        lrb_err: np.ndarray,
        rb_err: np.ndarray,
        check_num: int,
        threshold: dict[str, float | str | int | None],
        cfg: LRBThresholdConfig,
        show: bool,
    ) -> str | None:
        """
        Plot parametric LRB-vs-RB error rates in a p-window near threshold.

        Args:
            p (np.ndarray): Sorted probability values.
            lrb_err (np.ndarray): LRB error-rate values.
            rb_err (np.ndarray): RB error-rate values.
            check_num (int): Uniform check value.
            threshold (dict[str, float | str | int | None]): Threshold record.
            cfg (LRBThresholdConfig): Plot and threshold controls.
            show (bool): Whether to display the figure.

        Returns:
            str | None: Output path, or ``None`` if not enough data.

        Raises:
            ValueError: Not raised directly by this method.
        """
        p_arr = np.asarray(p, dtype=float)
        l_arr = np.asarray(lrb_err, dtype=float)
        r_arr = np.asarray(rb_err, dtype=float)

        onset = threshold.get("onset_index", None)
        tp = float(threshold.get("threshold_p", np.nan))
        x0 = float(threshold.get("threshold_rb_error", np.nan))
        y0 = float(threshold.get("threshold_lrb_error", np.nan))
        if onset is None:
            # Prefer geometric threshold anchors over midpoint fallback.
            if np.isfinite(x0):
                onset = int(np.argmin(np.abs(r_arr - x0)))
            elif np.isfinite(tp):
                onset = int(np.argmin(np.abs(p_arr - tp)))
            else:
                onset = max(0, len(p_arr) // 2)

        n_points = len(p_arr)
        i0 = max(0, int(onset) - int(cfg.p_window_before_points))
        i1 = min(n_points, int(onset) + int(cfg.p_window_after_points) + 1)
        while (i1 - i0) < int(cfg.min_points_in_view) and (
            i0 > 0 or i1 < n_points
        ):
            if i0 > 0:
                i0 -= 1
            if (i1 - i0) < int(cfg.min_points_in_view) and i1 < n_points:
                i1 += 1

        r_win = r_arr[i0:i1]
        l_win = l_arr[i0:i1]
        mask = np.isfinite(r_win) & np.isfinite(l_win)
        r_win = r_win[mask]
        l_win = l_win[mask]
        if len(r_win) < 2:
            print(
                f"[WARN] unif={check_num}: "
                "not enough points for focused plot"
            )
            return None

        xmin = float(np.min(r_win))
        xmax = float(np.max(r_win))
        ymin = float(np.min(l_win))
        ymax = float(np.max(l_win))
        x_span = max(xmax - xmin, cfg.min_zoom_span)
        y_span = max(ymax - ymin, cfg.min_zoom_span)
        xmin = max(0.0, xmin - cfg.view_pad_frac * x_span)
        xmax = xmax + cfg.view_pad_frac * x_span
        ymin = max(0.0, ymin - cfg.view_pad_frac * y_span)
        ymax = ymax + cfg.view_pad_frac * y_span

        plt.figure(figsize=(7.2, 6.2))
        # Plot the RB-vs-LRB trajectory in increasing-p order.
        plt.plot(r_win, l_win, "-o", markersize=4, label="(RB(p), LRB(p))")
        lo = max(xmin, ymin)
        hi = min(xmax, ymax)
        if hi > lo:
            plt.plot([lo, hi], [lo, hi], "--", linewidth=1, label="y = x")
        if np.isfinite(x0) and np.isfinite(y0):
            plt.plot(
                [x0],
                [y0],
                "o",
                markersize=10,
                alpha=0.8,
                label=f"$\\epsilon_{{RB}}\\approx{x0:.4g}$",
            )
            plt.axvline(x0, linestyle="--", alpha=0.6)
            plt.axhline(y0, linestyle="--", alpha=0.6)

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel("RB error rate (1 - fidelity)")
        plt.ylabel("LRB error rate (1 - fidelity)")
        plt.title(f"unif={check_num}: LRB vs RB (p-window near threshold)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(
            self.out_dir,
            f"unif-{check_num}-lrb-vs-rb-threshold-monotone.pdf",
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        print(f"[OK] wrote {out_path}")
        return out_path

    def _find_unif_table_csv(self) -> str | None:
        """
        Resolve the preferred unif summary table path in the plot directory.

        Args:
            None: Uses object output directory.

        Returns:
            str | None: Existing table path or ``None`` if not found.

        Raises:
            ValueError: Not raised directly by this method.
        """
        candidates = [
            "unif_lrb_vs_rb_table_all_mixed_fits.csv",
            "unif_lrb_vs_rb_table_all.csv",
        ]
        for name in candidates:
            path = os.path.join(self.out_dir, name)
            if os.path.exists(path):
                return path
        return None

    def plot_all_unif_threshold_graphs(
        self,
        threshold_config: LRBThresholdConfig | None = None,
        table_csv_path: str | None = None,
        show: bool = True,
    ) -> str:
        """
        Build threshold graphs and summary CSV for all uniform checks.

        Args:
            threshold_config (LRBThresholdConfig | None): Optional controls.
            table_csv_path (str | None): Optional input table path.
            show (bool): Whether to display generated figures.

        Returns:
            str: Path to the generated threshold summary CSV.

        Raises:
            RuntimeError: If pandas is unavailable.
            FileNotFoundError: If no unif table is available.
        """
        if pd is None:
            raise RuntimeError("pandas is required for threshold plotting.")

        cfg = threshold_config or LRBThresholdConfig()
        if table_csv_path is None:
            table_csv_path = self._find_unif_table_csv()
        if table_csv_path is None:
            # Auto-build the table when the notebook has not generated it yet.
            table_csv_path = self.build_unif_lrb_vs_rb_table_mixed_fits(
                write_per_check_tables=False)
        if not os.path.exists(table_csv_path):
            raise FileNotFoundError(f"Missing table CSV: {table_csv_path}")

        frame = pd.read_csv(table_csv_path)
        required_cols = ["check_num", "p", "lrb_error_rate", "rb_error_rate"]
        for col in required_cols:
            if col not in frame.columns:
                raise KeyError(f"Missing column '{col}' in {table_csv_path}")

        has_lrb_n = "lrb_n_points" in frame.columns
        has_rb_n = "rb_n_points" in frame.columns
        checks = sorted([int(v) for v in frame["check_num"].dropna().unique()])

        summary_rows: list[dict[str, float | int | str]] = []
        for check_num in checks:
            sub = frame[frame["check_num"] == check_num].copy()
            if cfg.p_min is not None:
                sub = sub[sub["p"] >= cfg.p_min]
            if cfg.p_max is not None:
                sub = sub[sub["p"] <= cfg.p_max]
            if has_lrb_n and cfg.min_lrb_n_points_keep > 0:
                sub = sub[sub["lrb_n_points"] >= cfg.min_lrb_n_points_keep]
            if has_rb_n and cfg.min_rb_n_points_keep > 0:
                sub = sub[sub["rb_n_points"] >= cfg.min_rb_n_points_keep]

            p = sub["p"].to_numpy(dtype=float)
            lrb_err = sub["lrb_error_rate"].to_numpy(dtype=float)
            rb_err = sub["rb_error_rate"].to_numpy(dtype=float)
            finite = self._finite_mask(p, lrb_err, rb_err)
            p, lrb_err, rb_err = p[finite], lrb_err[finite], rb_err[finite]
            if len(p) < 2:
                summary_rows.append(
                    {
                        "check_num": int(check_num),
                        "threshold_p": np.nan,
                        "threshold_rb_error_rate": np.nan,
                        "threshold_lrb_error_rate": np.nan,
                        "method": "insufficient_points",
                        "n_points_used": int(len(p)),
                    }
                )
                continue

            order = np.argsort(p)
            p, lrb_err, rb_err = p[order], lrb_err[order], rb_err[order]

            cut_l = self._trim_garbage_tail_by_monotone_dip(
                p,
                lrb_err,
                cfg.tail_min_prefix,
                cfg.tail_drop_tol_abs,
                cfg.tail_drop_tol_rel,
            )
            cut_r = self._trim_garbage_tail_by_monotone_dip(
                p,
                rb_err,
                cfg.tail_min_prefix,
                cfg.tail_drop_tol_abs,
                cfg.tail_drop_tol_rel,
            )
            cut_idx = int(min(cut_l, cut_r))
            p_trim = p[:cut_idx]
            lrb_trim = lrb_err[:cut_idx]
            rb_trim = rb_err[:cut_idx]
            if len(p_trim) < 2:
                summary_rows.append(
                    {
                        "check_num": int(check_num),
                        "threshold_p": np.nan,
                        "threshold_rb_error_rate": np.nan,
                        "threshold_lrb_error_rate": np.nan,
                        "method": "insufficient_points_after_tail_trim",
                        "n_points_used": int(len(p_trim)),
                    }
                )
                continue

            start_idx = self._first_threshold_start_index(
                lrb_trim,
                rb_trim,
                cfg.ignore_first_n,
                cfg.err_floor,
            )
            thr = self._estimate_monotone_threshold_in_p(
                p_trim,
                lrb_trim,
                rb_trim,
                tol=cfg.tol,
                require_consecutive=cfg.require_consecutive,
                start_idx=start_idx,
            )

            summary_rows.append(
                {
                    "check_num": int(check_num),
                    "threshold_p": float(thr["threshold_p"]),
                    "threshold_rb_error_rate": float(
                        thr["threshold_rb_error"]),
                    "threshold_lrb_error_rate": float(
                        thr["threshold_lrb_error"]),
                    "method": str(thr["method"]),
                    "start_idx_used": int(start_idx),
                    "tail_cut_idx": int(cut_idx),
                    "n_points_used": int(len(p_trim)),
                }
            )
            self._plot_error_vs_p_threshold(
                p=p_trim,
                lrb_err=lrb_trim,
                rb_err=rb_trim,
                check_num=check_num,
                threshold=thr,
                cfg=cfg,
                show=show,
            )
            self._plot_lrb_vs_rb_parametric_threshold(
                p=p_trim,
                lrb_err=lrb_trim,
                rb_err=rb_trim,
                check_num=check_num,
                threshold=thr,
                cfg=cfg,
                show=show,
            )

        summary = pd.DataFrame(summary_rows).sort_values(
            "check_num").reset_index(drop=True)
        out_summary = os.path.join(
            self.out_dir,
            "unif_thresholds_summary_monotone_trim_zoom_pwindow.csv",
        )
        summary.to_csv(out_summary, index=False)
        print(f"[OK] wrote {out_summary}")
        return out_summary
