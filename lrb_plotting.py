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

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
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
    f_max: float = 1.0000000000


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
        bootstrap_reps_error (int): Bootstrap replicates for point error bars.
        bootstrap_reps_threshold (int): Bootstrap replicates for threshold bars.
        bootstrap_ci_level (float): Central CI level (for example, ``0.68``).
        bootstrap_use_sem (bool): Use SEM instead of std for bootstrap noise.
        bootstrap_seed (int | None): Optional deterministic RNG seed.
        paper_mode (bool): Render unif threshold error-vs-p plots in one grid.
        paper_cols (int): Number of subplot columns in paper mode.
        paper_panel_width_in (float): Width (in) allocated per subplot column.
        paper_panel_height_in (float): Height (in) allocated per subplot row.
        paper_title_fontsize (float): Subplot title font size in paper mode.
        paper_axis_label_fontsize (float): Axis-label font size in paper mode.
        paper_tick_fontsize (float): Tick-label font size in paper mode.
        paper_legend_fontsize (float): Legend font size in paper mode.
        paper_line_width (float): Default line width in paper mode.
        paper_marker_size (float): Default marker size in paper mode.

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
    bootstrap_reps_error: int = 200
    bootstrap_reps_threshold: int = 300
    bootstrap_ci_level: float = 0.68
    bootstrap_use_sem: bool = True
    bootstrap_seed: int | None = 12345
    paper_mode: bool = False
    paper_cols: int = 3
    paper_panel_width_in: float = 3.2
    paper_panel_height_in: float = 2.5
    paper_title_fontsize: float = 7.4
    paper_axis_label_fontsize: float = 6.8
    paper_tick_fontsize: float = 6.1
    paper_legend_fontsize: float = 5.9
    paper_line_width: float = 0.9
    paper_marker_size: float = 3.0


@dataclass(frozen=True)
class LRBPaperPlotConfig:
    """
    Layout/style controls for paper-ready unif/const summary figures.

    Attributes:
        enabled (bool): Enable paper-mode rendering and pagination.
        column_layout (str): ``"single"`` or ``"double"`` figure width.
        rows_per_page (int): Number of probability rows per PDF page.
        pairs_per_row (int): Number of (main,rejected) panel pairs per row.
        num_prob_samples (int): Number of probability points to plot per check.
        prob_sample_indices (tuple[int, ...]): Explicit probability row indices.
        row_height_in (float): Height in inches allocated to one probability row.
        panel_title_fontsize (float): Font size for per-panel titles.
        axis_label_fontsize (float): Font size for axis labels.
        tick_fontsize (float): Font size for tick labels.
        legend_fontsize (float): Font size for legends.
        line_width (float): Default line width in paper mode.
        marker_size (float): Default marker size in paper mode.
        errorbar_capsize (float): Default errorbar cap size in paper mode.
        show_legend_first_row_only (bool): Keep legends only on first row/page.
        compact_fit_legend (bool): Use short fit legend labels in paper mode.
        legend_on_rejected_panel (bool): Draw LRB/RB legend on rejected panel.

    Methods:
        This dataclass is declarative and defines no custom methods.
    """

    enabled: bool = False
    column_layout: str = "double"
    rows_per_page: int = 5
    pairs_per_row: int = 1
    num_prob_samples: int = 0
    prob_sample_indices: tuple[int, ...] = ()
    row_height_in: float = 1.7
    panel_title_fontsize: float = 7.5
    axis_label_fontsize: float = 6.8
    tick_fontsize: float = 6.2
    legend_fontsize: float = 5.9
    line_width: float = 0.9
    marker_size: float = 2.8
    errorbar_capsize: float = 1.4
    show_legend_first_row_only: bool = True
    compact_fit_legend: bool = True
    legend_on_rejected_panel: bool = True


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
    _CODE_TITLE_BY_NAME: dict[str, str] = {
        "qgrm_3_1_2": r"$[[3,1,2]]_3$",
        "folded_qutrit": r"$[[5,1,2]]_3$",
    }

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

        # Number of random circuits per depth; used for SEM-based bootstrap.
        self.num_cliffs = 1
        num_cliffs_path = os.path.join(self.working_folder, "num_cliffs.txt")
        if os.path.exists(num_cliffs_path):
            value = ExperimentSetupManager.fetch_single_param(num_cliffs_path)
            if value not in ("", "0"):
                self.num_cliffs = max(1, int(float(value)))

        code_name_path = os.path.join(self.working_folder, "code_name.txt")
        self.code_name = "unknown_code"
        if os.path.exists(code_name_path):
            value = ExperimentSetupManager.fetch_single_param(code_name_path)
            if value not in ("", "0"):
                self.code_name = value
        self.code_title = self._format_code_title(self.code_name)

    @classmethod
    def _format_code_title(cls, code_name: str) -> str:
        """
        Format the code-family fragment used in figure titles.

        Args:
            code_name (str): Run code identifier.

        Returns:
            str: Pretty title token, falling back to ``code_name``.

        Raises:
            ValueError: Not raised directly by this method.
        """
        return cls._CODE_TITLE_BY_NAME.get(code_name, code_name)

    def _title_context(self, check_type: str, check_num: int) -> str:
        """
        Build shared title context with check type, q, and code family.

        Args:
            check_type (str): ``"unif"`` or ``"const"``.
            check_num (int): Check policy value.

        Returns:
            str: Title-context fragment.

        Raises:
            ValueError: Not raised directly by this method.
        """
        if check_type == "unif":
            check_label = "uniform interval check"
        elif check_type == "const":
            check_label = "constant check"
        else:
            check_label = "check"
        return f"{check_label} = {check_num}, q={self.d_dim}, {self.code_title}"

    @staticmethod
    def _filter_checks_by_range(
        checks: list[int],
        check_min: int | None = None,
        check_max: int | None = None,
    ) -> list[int]:
        """
        Filter a check-number list by an optional inclusive range.

        Args:
            checks (list[int]): Check values to filter.
            check_min (int | None): Optional inclusive lower bound.
            check_max (int | None): Optional inclusive upper bound.

        Returns:
            list[int]: Sorted filtered check values.

        Raises:
            ValueError: If ``check_min > check_max``.
        """
        if check_min is not None and check_max is not None:
            if int(check_min) > int(check_max):
                raise ValueError("check_min must be <= check_max.")
        out = sorted([int(v) for v in checks])
        if check_min is not None:
            out = [v for v in out if v >= int(check_min)]
        if check_max is not None:
            out = [v for v in out if v <= int(check_max)]
        return out

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
        Plot rejected-run proportions for one probability row.

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
        y_arr = np.clip(np.asarray(y, dtype=float), 0.0, 1.0)
        e_arr = np.asarray(e, dtype=float)

        # CSV stores run-to-run std across random sequences; for uncertainty of
        # the displayed mean proportion, use SEM.
        n_seq = max(int(getattr(self, "num_cliffs", 1)), 1)
        if n_seq > 1:
            e_arr = e_arr / np.sqrt(float(n_seq))
        e_arr = np.where(np.isfinite(e_arr), np.maximum(0.0, e_arr), 0.0)

        low = np.maximum(0.0, y_arr - e_arr)
        high = np.minimum(1.0, y_arr + e_arr)
        yerr = np.vstack([y_arr - low, high - y_arr])

        ax.errorbar(x, y_arr, yerr=yerr, fmt="-o", capsize=2)
        # Use a data-adaptive range so local variation is visible while
        # retaining physical bounds with slight display headroom.
        if len(y_arr) > 0:
            y_min_raw = float(np.nanmin(low))
            y_max_raw = float(np.nanmax(high))
            span = max(y_max_raw - y_min_raw, 0.02)
            pad = max(0.05 * span, 0.01)
            y_min = y_min_raw - pad
            y_max = y_max_raw + pad
            y_min = max(-0.03, y_min)
            y_max = min(1.03, y_max)

            # Snap to edges when data are near physical limits.
            if y_min_raw <= 0.02:
                y_min = -0.03
            if y_max_raw >= 0.98:
                y_max = 1.03
            if y_max <= y_min:
                y_min, y_max = -0.03, 1.03
        else:
            y_min, y_max = -0.03, 1.03
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Rejected runs (p = {prob_value:.3g})")
        ax.set_ylabel("Proportion of Rejected Runs")

    @staticmethod
    def _legend_loc_for_curve_band(
        x_candidates: list[float],
        y_candidates: list[float],
        y_min: float,
        y_max: float,
    ) -> str:
        """
        Choose legend corner with minimal data occupancy.

        Args:
            x_candidates (list[float]): X coordinates of plotted data/fit points.
            y_candidates (list[float]): Y coordinates of plotted data/fit points.
            y_min (float): Current axis lower bound.
            y_max (float): Current axis upper bound.

        Returns:
            str: Legend location string.

        Raises:
            ValueError: Not raised directly by this method.
        """
        if not x_candidates or not y_candidates:
            return "best"
        x_arr = np.asarray(x_candidates, dtype=float)
        y_arr = np.asarray(y_candidates, dtype=float)
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_arr = x_arr[mask]
        y_arr = y_arr[mask]
        if len(x_arr) == 0:
            return "best"

        x_mid = 0.5 * (float(np.min(x_arr)) + float(np.max(x_arr)))
        y_mid = 0.5 * (float(y_min) + float(y_max))

        scores = {
            "upper right": int(np.sum((x_arr >= x_mid) & (y_arr >= y_mid))),
            "upper left": int(np.sum((x_arr < x_mid) & (y_arr >= y_mid))),
            "lower right": int(np.sum((x_arr >= x_mid) & (y_arr < y_mid))),
            "lower left": int(np.sum((x_arr < x_mid) & (y_arr < y_mid))),
        }
        order = ["upper right", "upper left", "lower right", "lower left"]
        return min(order, key=lambda loc: scores[loc])

    @staticmethod
    def _ensure_legend_above_curves(
        ax: Any,
        legend: Any,
        x_top_candidates: list[float],
        y_top_candidates: list[float],
        y_max_cap: float = 1.5,
    ) -> None:
        """
        Increase y-max so the legend box sits above plotted curve envelope.

        Args:
            ax (Any): Matplotlib axis target.
            legend (Any): Legend object returned by ``ax.legend``.
            x_top_candidates (list[float]): X positions of top envelope points.
            y_top_candidates (list[float]): Top envelope y-values.
            y_max_cap (float): Maximum allowed y-axis upper bound.

        Returns:
            None: Updates axis limits in place.

        Raises:
            ValueError: Not raised directly by this method.
        """
        if legend is None:
            return
        if not x_top_candidates or not y_top_candidates:
            return

        x_arr = np.asarray(x_top_candidates, dtype=float)
        y_arr = np.asarray(y_top_candidates, dtype=float)
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_arr = x_arr[mask]
        y_arr = y_arr[mask]
        if len(x_arr) == 0:
            return

        fig = ax.figure
        for _ in range(3):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox_disp = legend.get_window_extent(renderer=renderer)
            bbox_axes = bbox_disp.transformed(ax.transAxes.inverted())
            frac_bottom = float(bbox_axes.y0)
            if frac_bottom <= 1e-6:
                return

            bbox_data = bbox_disp.transformed(ax.transData.inverted())
            x0 = float(bbox_data.x0)
            x1 = float(bbox_data.x1)
            in_box_x = (x_arr >= min(x0, x1)) & (x_arr <= max(x0, x1))
            if np.any(in_box_x):
                y_curve_top = float(np.nanmax(y_arr[in_box_x]))
            else:
                y_curve_top = float(np.nanmax(y_arr))

            y_min, y_max = ax.get_ylim()
            span = max(float(y_max - y_min), 1e-9)
            # Small positive gap so legend does not graze the curve envelope.
            clearance = max(0.025 * span, 0.0015)
            desired_bottom = y_curve_top + clearance
            current_bottom = y_min + frac_bottom * span
            if current_bottom >= desired_bottom:
                return

            required_span = (desired_bottom - y_min) / frac_bottom
            new_ymax = min(y_min + max(span, required_span), y_max_cap)
            if new_ymax <= y_max + 1e-12:
                return
            ax.set_ylim(y_min, new_ymax)

    def _plot_data_only_series(
        self,
        ax: Any,
        lrb_f_stats: list[dict[str, Any]] | None,
        rb_f_stats: list[dict[str, Any]] | None,
        show_legend: bool = True,
        legend_fontsize: float = 8.0,
    ) -> None:
        """
        Plot LRB and RB data points only, without any fitted curves.

        Args:
            ax (Any): Matplotlib axis target.
            lrb_f_stats (list[dict[str, Any]] | None): LRB fidelity stats.
            rb_f_stats (list[dict[str, Any]] | None): RB fidelity stats.
            show_legend (bool): Whether to draw legend for this panel.
            legend_fontsize (float): Legend font size.

        Returns:
            None: Draws directly on the axis.

        Raises:
            ValueError: Not raised directly by this method.
        """
        lrb_means = self._extract_series(lrb_f_stats, "mean", len(self.depths))
        lrb_errs = self._extract_series(lrb_f_stats, "std", len(self.depths))
        x_l, y_l, e_l = self._mask_invalid(self.depths, lrb_means, lrb_errs)
        x_top_candidates = [float(v) for v in x_l]
        y_top_candidates = [float(yv + ev) for yv, ev in zip(y_l, e_l)]
        ax.errorbar(x_l, y_l, yerr=e_l, fmt="-o", label="LRB")
        y_low_candidates = [float(yv - ev) for yv, ev in zip(y_l, e_l)]
        y_high_candidates = [float(yv + ev) for yv, ev in zip(y_l, e_l)]

        if rb_f_stats is not None:
            rb_means = self._extract_series(
                rb_f_stats, "mean", len(self.depths))
            rb_errs = self._extract_series(rb_f_stats, "std", len(self.depths))
            x_r, y_r, e_r = self._mask_invalid(self.depths, rb_means, rb_errs)
            x_top_candidates.extend(float(v) for v in x_r)
            y_top_candidates.extend(float(yv + ev) for yv, ev in zip(y_r, e_r))
            ax.errorbar(x_r, y_r, yerr=e_r, fmt="-s", label="RB")
            y_low_candidates.extend(float(yv - ev) for yv, ev in zip(y_r, e_r))
            y_high_candidates.extend(
                float(yv + ev) for yv, ev in zip(y_r, e_r))

        if y_low_candidates and y_high_candidates:
            y_min_raw = float(np.nanmin(y_low_candidates))
            y_max_raw = float(np.nanmax(y_high_candidates))
            span = max(y_max_raw - y_min_raw, 0.05)
            pad_bottom = max(0.05 * span, 0.006)
            pad_top = max(0.03 * span, 0.002)
            y_min = max(-0.03, y_min_raw - pad_bottom)
            y_max = min(1.35, y_max_raw + pad_top)
            if y_max <= y_min:
                y_min, y_max = -0.03, 1.06
        else:
            y_min, y_max = -0.03, 1.06
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Expectation Value of Logical X")
        if show_legend:
            legend = ax.legend(
                fontsize=float(legend_fontsize),
                loc="upper right",
                framealpha=0.95,
                facecolor="white",
                edgecolor="0.7",
                borderaxespad=0.3,
            )
            self._ensure_legend_above_curves(
                ax=ax,
                legend=legend,
                x_top_candidates=x_top_candidates,
                y_top_candidates=y_top_candidates,
            )

    def _plot_fit_series(
        self,
        ax: Any,
        lrb_f_stats: list[dict[str, Any]] | None,
        rb_f_stats: list[dict[str, Any]] | None,
        show_legend: bool = True,
        legend_fontsize: float = 8.0,
        compact_legend: bool = False,
    ) -> None:
        """
        Plot fitted LRB and RB decay curves with fidelity legend labels.

        Args:
            ax (Any): Matplotlib axis target.
            lrb_f_stats (list[dict[str, Any]] | None): LRB fidelity stats.
            rb_f_stats (list[dict[str, Any]] | None): RB fidelity stats.
            show_legend (bool): Whether to draw legend for this panel.
            legend_fontsize (float): Legend font size.
            compact_legend (bool): Use compact fit labels in legend.

        Returns:
            None: Draws directly on the axis.

        Raises:
            ValueError: Not raised directly by this method.
        """
        lrb_fit, (x_l, y_l, e_l) = self._fit_lrb_from_stats(lrb_f_stats)
        x_top_candidates = [float(v) for v in x_l]
        y_top_candidates = [float(yv + ev) for yv, ev in zip(y_l, e_l)]
        y_low_candidates = [float(yv - ev) for yv, ev in zip(y_l, e_l)]
        y_high_candidates = [float(yv + ev) for yv, ev in zip(y_l, e_l)]
        if lrb_fit is not None:
            f_l = lrb_fit["f"]
            fid_l = self._decay_to_fidelity(f_l)
            if compact_legend:
                lrb_label = f"LRB (F={fid_l:.4f})"
            else:
                lrb_label = f"LRB (f={f_l:.6f}, F={fid_l:.6f})"
        else:
            lrb_label = "LRB"
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
            x_top_candidates.extend(float(v) for v in x_fit)
            y_top_candidates.extend(float(v) for v in y_fit)
            y_low_candidates.extend(float(v) for v in y_fit)
            y_high_candidates.extend(float(v) for v in y_fit)

        if rb_f_stats is not None:
            rb_fit, (x_r, y_r, e_r) = self._fit_rb_from_stats(rb_f_stats)
            x_top_candidates.extend(float(v) for v in x_r)
            y_top_candidates.extend(float(yv + ev) for yv, ev in zip(y_r, e_r))
            y_low_candidates.extend(float(yv - ev) for yv, ev in zip(y_r, e_r))
            y_high_candidates.extend(
                float(yv + ev) for yv, ev in zip(y_r, e_r))
            if rb_fit is not None:
                f_r = rb_fit["f"]
                fid_r = self._decay_to_fidelity(f_r)
                if compact_legend:
                    rb_label = f"RB (F={fid_r:.4f})"
                else:
                    rb_label = f"RB (f={f_r:.6f}, F={fid_r:.6f})"
            else:
                rb_label = "RB"
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
                x_top_candidates.extend(float(v) for v in x_fit)
                y_top_candidates.extend(float(v) for v in y_fit)
                y_low_candidates.extend(float(v) for v in y_fit)
                y_high_candidates.extend(float(v) for v in y_fit)

        if y_low_candidates and y_high_candidates:
            y_min_raw = float(np.nanmin(y_low_candidates))
            y_max_raw = float(np.nanmax(y_high_candidates))
            span = max(y_max_raw - y_min_raw, 0.05)
            pad_bottom = max(0.05 * span, 0.006)
            pad_top = max(0.03 * span, 0.002)
            y_min = max(-0.03, y_min_raw - pad_bottom)
            y_max = min(1.35, y_max_raw + pad_top)
            if y_max <= y_min:
                y_min, y_max = -0.03, 1.06
        else:
            y_min, y_max = -0.03, 1.06
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Expectation Value of Logical X")
        if show_legend:
            legend = ax.legend(
                fontsize=float(legend_fontsize),
                loc="upper right",
                framealpha=0.95,
                facecolor="white",
                edgecolor="0.7",
                borderaxespad=0.3,
            )
            self._ensure_legend_above_curves(
                ax=ax,
                legend=legend,
                x_top_candidates=x_top_candidates,
                y_top_candidates=y_top_candidates,
            )

    def _plot_check_rows(
        self,
        rows: list[dict[str, Any]],
        check_type: str,
        check_num: int,
        out_path: str,
        use_fits: bool,
        show: bool,
        paper_config: LRBPaperPlotConfig | None = None,
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
            paper_config (LRBPaperPlotConfig | None): Optional paper settings.

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

        paper = paper_config or LRBPaperPlotConfig()
        plot_rows = list(rows)
        if paper.enabled and len(paper.prob_sample_indices) > 0:
            n_total = len(plot_rows)
            idx = []
            for i in paper.prob_sample_indices:
                ii = int(i)
                if 0 <= ii < n_total and ii not in idx:
                    idx.append(ii)
            if idx:
                plot_rows = [plot_rows[i] for i in idx]
        elif paper.enabled and int(paper.num_prob_samples) > 0:
            n_total = len(plot_rows)
            n_keep = min(int(paper.num_prob_samples), n_total)
            if n_keep < n_total:
                idx = np.linspace(0, n_total - 1, n_keep, dtype=int)
                idx = sorted(set(int(i) for i in idx))
                while len(idx) < n_keep:
                    idx.append(n_total - 1)
                    idx = sorted(set(idx))
                idx = idx[:n_keep]
                plot_rows = [plot_rows[i] for i in idx]

        pairs_per_row = 1 if not paper.enabled else max(
            1, int(paper.pairs_per_row))
        rows_per_page = len(plot_rows) if not paper.enabled else max(
            1, int(paper.rows_per_page))
        items_per_page = rows_per_page * pairs_per_row
        page_chunks = [
            plot_rows[i:i + items_per_page]
            for i in range(0, len(plot_rows), items_per_page)
        ]

        def figure_width() -> float:
            if not paper.enabled:
                return 10.0
            layout = str(paper.column_layout).strip().lower()
            if layout == "single":
                return 3.35
            if layout == "double":
                return 7.0
            return 10.0

        rc_updates: dict[str, float] = {}
        if paper.enabled:
            rc_updates = {
                "font.size": float(paper.tick_fontsize),
                "axes.labelsize": float(paper.axis_label_fontsize),
                "xtick.labelsize": float(paper.tick_fontsize),
                "ytick.labelsize": float(paper.tick_fontsize),
                "legend.fontsize": float(paper.legend_fontsize),
                "lines.linewidth": float(paper.line_width),
                "lines.markersize": float(paper.marker_size),
                "errorbar.capsize": float(paper.errorbar_capsize),
            }

        with mpl.rc_context(rc=rc_updates):
            if paper.enabled:
                if os.path.exists(out_path):
                    os.remove(out_path)
                with PdfPages(out_path) as pdf:
                    for page_idx, page_rows in enumerate(page_chunks):
                        n_items = len(page_rows)
                        n_rows = int(np.ceil(n_items / float(pairs_per_row)))
                        n_cols = 2 * pairs_per_row
                        fig, axs = plt.subplots(
                            nrows=n_rows,
                            ncols=n_cols,
                            figsize=(figure_width(),
                                     max(2.4, float(paper.row_height_in) * n_rows)),
                            sharex="col",
                            constrained_layout=True,
                        )

                        def cell(row: int, col: int) -> Any:
                            if n_rows == 1:
                                return axs[col]
                            return axs[row][col]

                        for item_index, row in enumerate(page_rows):
                            row_index = item_index // pairs_per_row
                            pair_index = item_index % pairs_per_row
                            prob_value = float(row["p"])
                            lrb_f = row["lrb_f"]
                            lrb_r = row["lrb_r"]
                            rb_f = row["rb_f"]
                            show_legend = True if use_fits else (
                                not paper.show_legend_first_row_only
                                or row_index == 0
                            )
                            legend_on_rejected = (
                                bool(paper.legend_on_rejected_panel)
                                and bool(show_legend)
                            )
                            main_show_legend = bool(show_legend) and not legend_on_rejected

                            ax_main = cell(row_index, 2 * pair_index)
                            if use_fits:
                                self._plot_fit_series(
                                    ax_main,
                                    lrb_f_stats=lrb_f,
                                    rb_f_stats=rb_f,
                                    show_legend=main_show_legend,
                                    legend_fontsize=float(paper.legend_fontsize),
                                    compact_legend=bool(paper.compact_fit_legend),
                                )
                            else:
                                self._plot_data_only_series(
                                    ax_main,
                                    lrb_f_stats=lrb_f,
                                    rb_f_stats=rb_f,
                                    show_legend=main_show_legend,
                                    legend_fontsize=float(paper.legend_fontsize),
                                )
                            ax_main.set_title(
                                f"p = {prob_value:.3g}",
                                fontsize=float(paper.panel_title_fontsize),
                            )
                            ax_main.yaxis.label.set_size(
                                float(paper.axis_label_fontsize))
                            ax_main.tick_params(labelsize=float(paper.tick_fontsize))

                            ax_rej = cell(row_index, 2 * pair_index + 1)
                            self._plot_rejected_panel(
                                ax=ax_rej,
                                lrb_r_stats=lrb_r,
                                prob_value=prob_value,
                            )
                            ax_rej.set_title(
                                f"Rejected runs (p = {prob_value:.3g})",
                                fontsize=float(paper.panel_title_fontsize),
                            )
                            ax_rej.yaxis.label.set_size(
                                float(paper.axis_label_fontsize))
                            ax_rej.tick_params(labelsize=float(paper.tick_fontsize))
                            if legend_on_rejected:
                                handles, labels = ax_main.get_legend_handles_labels()
                                if handles:
                                    rej_loc = "lower right"
                                    if len(ax_rej.lines) > 0:
                                        rej_line = ax_rej.lines[0]
                                        rej_x = np.asarray(
                                            rej_line.get_xdata(), dtype=float)
                                        rej_y = np.asarray(
                                            rej_line.get_ydata(), dtype=float)
                                        rej_x_vals = [
                                            float(v)
                                            for v in rej_x[np.isfinite(rej_x)]
                                        ]
                                        rej_y_vals = [
                                            float(v)
                                            for v in rej_y[np.isfinite(rej_y)]
                                        ]
                                        if rej_x_vals and rej_y_vals:
                                            y_rej_min, y_rej_max = ax_rej.get_ylim()
                                            rej_loc = self._legend_loc_for_curve_band(
                                                x_candidates=rej_x_vals,
                                                y_candidates=rej_y_vals,
                                                y_min=float(y_rej_min),
                                                y_max=float(y_rej_max),
                                            )
                                    ax_rej.legend(
                                        handles,
                                        labels,
                                        fontsize=float(paper.legend_fontsize),
                                        loc=rej_loc,
                                        framealpha=0.95,
                                        facecolor="white",
                                        edgecolor="0.7",
                                        borderaxespad=0.25,
                                        handlelength=1.35,
                                        handletextpad=0.35,
                                        labelspacing=0.25,
                                    )

                        # Hide unused trailing panel pairs on partial final row.
                        total_pairs = n_rows * pairs_per_row
                        for pair_slot in range(n_items, total_pairs):
                            rr = pair_slot // pairs_per_row
                            cc = pair_slot % pairs_per_row
                            cell(rr, 2 * cc).set_visible(False)
                            cell(rr, 2 * cc + 1).set_visible(False)

                        for col in range(n_cols):
                            cell(n_rows - 1, col).set_xlabel(
                                "Depth",
                                fontsize=float(paper.axis_label_fontsize),
                            )

                        # In paper mode keep the panel grid clean: no global header.
                        pdf.savefig(fig, dpi=300, bbox_inches="tight")
                        if show:
                            plt.show()
                        plt.close(fig)
            else:
                n_rows = len(rows)
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
                        self._plot_fit_series(
                            ax_main,
                            lrb_f_stats=lrb_f,
                            rb_f_stats=rb_f,
                            compact_legend=False,
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

                fig.suptitle(
                    f"LRB vs RB ({self._title_context(check_type, check_num)})",
                    fontsize=14,
                    y=1.01,
                )
                self._save_pdf_overwrite(fig=fig, out_path=out_path, dpi=300)
                if show:
                    plt.show()
                plt.close(fig)
        print(f"[OK] wrote {out_path}")
        return out_path

    def plot_one_unif_check(
        self,
        check_num: int,
        show: bool = True,
        paper_config: LRBPaperPlotConfig | None = None,
    ) -> str | None:
        """
        Plot one uniform-check summary with LRB and RB fitted curves.

        Args:
            check_num (int): Uniform check interval.
            show (bool): Whether to display the figure.
            paper_config (LRBPaperPlotConfig | None): Optional paper settings.

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
            paper_config=paper_config,
        )

    def plot_one_const_check(
        self,
        check_num: int,
        show: bool = True,
        paper_config: LRBPaperPlotConfig | None = None,
    ) -> str | None:
        """
        Plot one constant-check summary with data-only LRB/RB traces.

        Args:
            check_num (int): Constant number of stabilizer checks.
            show (bool): Whether to display the figure.
            paper_config (LRBPaperPlotConfig | None): Optional paper settings.

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
            paper_config=paper_config,
        )

    def plot_all_unif_checks(
        self,
        show: bool = True,
        check_min: int | None = None,
        check_max: int | None = None,
        paper_config: LRBPaperPlotConfig | None = None,
    ) -> list[str]:
        """
        Generate uniform-check summary plots for all configured values.

        Args:
            show (bool): Whether to display figures during generation.
            check_min (int | None): Optional inclusive lower check bound.
            check_max (int | None): Optional inclusive upper check bound.
            paper_config (LRBPaperPlotConfig | None): Optional paper settings.

        Returns:
            list[str]: Output paths for plots that were generated.

        Raises:
            ValueError: Propagated from single-plot generation.
        """
        checks = self.stab_unif if self.stab_unif else list(range(1, 23))
        checks = self._filter_checks_by_range(
            checks=checks,
            check_min=check_min,
            check_max=check_max,
        )
        outputs: list[str] = []
        for check_num in checks:
            out_path = self.plot_one_unif_check(
                check_num=check_num,
                show=show,
                paper_config=paper_config,
            )
            if out_path is not None:
                outputs.append(out_path)
        return outputs

    def plot_all_const_checks(
        self,
        show: bool = True,
        check_min: int | None = None,
        check_max: int | None = None,
        paper_config: LRBPaperPlotConfig | None = None,
    ) -> list[str]:
        """
        Generate constant-check summary plots for all configured values.

        Args:
            show (bool): Whether to display figures during generation.
            check_min (int | None): Optional inclusive lower check bound.
            check_max (int | None): Optional inclusive upper check bound.
            paper_config (LRBPaperPlotConfig | None): Optional paper settings.

        Returns:
            list[str]: Output paths for plots that were generated.

        Raises:
            ValueError: Propagated from single-plot generation.
        """
        checks = self.stab_const if self.stab_const else list(range(0, 23))
        checks = self._filter_checks_by_range(
            checks=checks,
            check_min=check_min,
            check_max=check_max,
        )
        outputs: list[str] = []
        for check_num in checks:
            out_path = self.plot_one_const_check(
                check_num=check_num,
                show=show,
                paper_config=paper_config,
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
        check_min: int | None = None,
        check_max: int | None = None,
        bootstrap_reps: int = 0,
        bootstrap_ci_level: float = 0.68,
        bootstrap_use_sem: bool = True,
        bootstrap_seed: int | None = None,
    ) -> str:
        """
        Build a per-unif-check table with mixed fits and error-rate columns.

        LRB uses fixed-a fitting and RB uses free-a fitting.

        Args:
            write_per_check_tables (bool): Whether to write per-check CSVs.
            check_min (int | None): Optional inclusive lower check bound.
            check_max (int | None): Optional inclusive upper check bound.
            bootstrap_reps (int): Bootstrap replicates for point CIs.
            bootstrap_ci_level (float): Central CI level for point CIs.
            bootstrap_use_sem (bool): Use SEM instead of std for bootstrap.
            bootstrap_seed (int | None): Optional deterministic bootstrap seed.

        Returns:
            str: Output path of the all-check aggregate CSV.

        Raises:
            RuntimeError: If pandas is unavailable.
        """
        if pd is None:
            raise RuntimeError("pandas is required for table generation.")

        # RB is check-independent, so fit once per probability for reuse.
        checks = self.stab_unif if self.stab_unif else list(range(1, 23))
        checks = self._filter_checks_by_range(
            checks=checks,
            check_min=check_min,
            check_max=check_max,
        )
        if not checks:
            raise ValueError("No unif checks found in the requested range.")
        ci_level = self._normalize_ci_level(bootstrap_ci_level)
        rng = np.random.default_rng(bootstrap_seed)
        rb_cache: dict[int, dict[str, float]] = {}
        for prob_index, prob in enumerate(self.probabilities):
            rb_path = os.path.join(self.rb_root, f"{prob_index}.csv")
            rb_stats = self._read_fidelity_stats_file(rb_path)
            rb_fit = None
            rb_xyz = ([], [], [])
            if rb_stats is not None:
                rb_fit, rb_xyz = self._fit_rb_from_stats(rb_stats)
            metrics = self._safe_metrics_from_fit(rb_fit)
            rb_std, rb_lo, rb_hi = np.nan, np.nan, np.nan
            if int(bootstrap_reps) > 0 and rb_stats is not None:
                rb_std, rb_lo, rb_hi = self._bootstrap_error_rate_from_fit_series(
                    depths=rb_xyz[0],
                    means=rb_xyz[1],
                    std_values=rb_xyz[2],
                    fit_kind="rb",
                    n_boot=int(bootstrap_reps),
                    ci_level=ci_level,
                    use_sem=bool(bootstrap_use_sem),
                    rng=rng,
                )
            metrics["p"] = float(prob)
            metrics["error_rate_std_boot"] = float(rb_std)
            metrics["error_rate_ci_low"] = float(rb_lo)
            metrics["error_rate_ci_high"] = float(rb_hi)
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
                lrb_xyz = ([], [], [])
                if lrb_stats is not None:
                    lrb_fit, lrb_xyz = self._fit_lrb_from_stats(lrb_stats)
                lrb_m = self._safe_metrics_from_fit(lrb_fit)
                lrb_std, lrb_lo, lrb_hi = np.nan, np.nan, np.nan
                if int(bootstrap_reps) > 0 and lrb_stats is not None:
                    lrb_std, lrb_lo, lrb_hi = (
                        self._bootstrap_error_rate_from_fit_series(
                            depths=lrb_xyz[0],
                            means=lrb_xyz[1],
                            std_values=lrb_xyz[2],
                            fit_kind="lrb",
                            n_boot=int(bootstrap_reps),
                            ci_level=ci_level,
                            use_sem=bool(bootstrap_use_sem),
                            rng=rng,
                        )
                    )
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
                        "lrb_error_rate_std_boot": float(lrb_std),
                        "lrb_error_rate_ci_low": float(lrb_lo),
                        "lrb_error_rate_ci_high": float(lrb_hi),
                        "rb_f": rb_m["f"],
                        "rb_fidelity": rb_m["fid"],
                        "rb_error_rate": rb_m["err"],
                        "rb_a": rb_m["a"],
                        "rb_b": rb_m["b"],
                        "rb_chi2": rb_m["chi2"],
                        "rb_n_points": int(rb_m["n_points"]),
                        "rb_error_rate_std_boot": float(
                            rb_m["error_rate_std_boot"]),
                        "rb_error_rate_ci_low": float(
                            rb_m["error_rate_ci_low"]),
                        "rb_error_rate_ci_high": float(
                            rb_m["error_rate_ci_high"]),
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
                "lrb_error_rate_std_boot",
                "lrb_error_rate_ci_low",
                "lrb_error_rate_ci_high",
                "rb_error_rate_std_boot",
                "rb_error_rate_ci_low",
                "rb_error_rate_ci_high",
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
    def _save_pdf_overwrite(
        fig: Any,
        out_path: str,
        dpi: int = 300,
        bbox_inches: str | None = None,
    ) -> None:
        """
        Save one PDF and explicitly replace any existing file at the same path.

        Args:
            fig (Any): Matplotlib figure object.
            out_path (str): Destination PDF path.
            dpi (int): DPI passed to ``savefig``.
            bbox_inches (str | None): Optional bbox policy.

        Returns:
            None: Writes file to disk.

        Raises:
            PermissionError: If another application is locking the PDF.
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
            save_kwargs: dict[str, Any] = {"dpi": dpi}
            if bbox_inches is not None:
                save_kwargs["bbox_inches"] = bbox_inches
            fig.savefig(out_path, **save_kwargs)
        except PermissionError as exc:
            raise PermissionError(
                f"Could not overwrite '{out_path}'. "
                "Close any app locking the PDF and rerun."
            ) from exc

    @staticmethod
    def _normalize_ci_level(ci_level: float) -> float:
        """
        Clamp CI level to a numerically safe open interval.

        Args:
            ci_level (float): Requested central confidence-interval level.

        Returns:
            float: Clamped CI level in ``(0, 1)``.

        Raises:
            ValueError: Not raised directly by this method.
        """
        return float(np.clip(float(ci_level), 1e-6, 1.0 - 1e-6))

    @staticmethod
    def _bootstrap_sigma(
        std_values: np.ndarray,
        n_sequences: int,
        use_sem: bool,
    ) -> np.ndarray:
        """
        Convert per-depth standard deviations into bootstrap noise scales.

        Args:
            std_values (np.ndarray): Per-depth standard deviation values.
            n_sequences (int): Number of random sequences per depth.
            use_sem (bool): Whether to scale std by ``sqrt(n_sequences)``.

        Returns:
            np.ndarray: Positive finite bootstrap sigma values.

        Raises:
            ValueError: Not raised directly by this method.
        """
        sigma = np.asarray(std_values, dtype=float).copy()
        if use_sem and int(n_sequences) > 1:
            sigma = sigma / np.sqrt(float(n_sequences))
        finite_pos = sigma[np.isfinite(sigma) & (sigma > 0)]
        floor = float(max(np.median(finite_pos) * 1e-3, 1e-12)) \
            if len(finite_pos) else 1e-6
        sigma = np.where(~np.isfinite(sigma) | (sigma <= 0.0), floor, sigma)
        return sigma

    @staticmethod
    def _bootstrap_interval(
        samples: np.ndarray,
        ci_level: float,
    ) -> tuple[float, float, float]:
        """
        Compute bootstrap std and central CI bounds from finite samples.

        Args:
            samples (np.ndarray): Bootstrap sample values.
            ci_level (float): Central CI level.

        Returns:
            tuple[float, float, float]: ``(std, ci_low, ci_high)``.

        Raises:
            ValueError: Not raised directly by this method.
        """
        arr = np.asarray(samples, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return np.nan, np.nan, np.nan
        ci = LRBResultsPlotter._normalize_ci_level(ci_level)
        alpha = (1.0 - ci) / 2.0
        lo = float(np.quantile(arr, alpha))
        hi = float(np.quantile(arr, 1.0 - alpha))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        return std, lo, hi

    @staticmethod
    def _to_asymmetric_yerr(
        y: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
    ) -> np.ndarray:
        """
        Convert lower/upper CI arrays into Matplotlib ``yerr`` shape.

        Args:
            y (np.ndarray): Point estimates.
            low (np.ndarray): Lower CI bounds.
            high (np.ndarray): Upper CI bounds.

        Returns:
            np.ndarray: Array of shape ``(2, n)`` with nonnegative errors.

        Raises:
            ValueError: Not raised directly by this method.
        """
        y_arr = np.asarray(y, dtype=float)
        lo = np.asarray(low, dtype=float)
        hi = np.asarray(high, dtype=float)
        lower = np.where(np.isfinite(lo), np.maximum(0.0, y_arr - lo), 0.0)
        upper = np.where(np.isfinite(hi), np.maximum(0.0, hi - y_arr), 0.0)
        return np.vstack([lower, upper])

    def _bootstrap_error_rate_from_fit_series(
        self,
        depths: list[int] | np.ndarray,
        means: list[float] | np.ndarray,
        std_values: list[float] | np.ndarray,
        fit_kind: str,
        n_boot: int,
        ci_level: float,
        use_sem: bool,
        rng: np.random.Generator,
    ) -> tuple[float, float, float]:
        """
        Bootstrap error-rate uncertainty for one fitted LRB/RB point.

        Args:
            depths (list[int] | np.ndarray): Depth values used in fit.
            means (list[float] | np.ndarray): Mean fidelities per depth.
            std_values (list[float] | np.ndarray): Std values per depth.
            fit_kind (str): ``"lrb"`` (fixed-a) or ``"rb"`` (free-a).
            n_boot (int): Number of bootstrap replicates.
            ci_level (float): Central CI level.
            use_sem (bool): Whether to use SEM bootstrap noise.
            rng (np.random.Generator): RNG used for reproducible resampling.

        Returns:
            tuple[float, float, float]: ``(std, ci_low, ci_high)``.

        Raises:
            ValueError: If ``fit_kind`` is unsupported.
        """
        if int(n_boot) <= 0:
            return np.nan, np.nan, np.nan
        x = np.asarray(depths, dtype=float)
        y = np.asarray(means, dtype=float)
        e = np.asarray(std_values, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
        x, y, e = x[finite], y[finite], e[finite]
        if fit_kind == "lrb" and self.fit_config.y_ceiling is not None:
            keep = y <= float(self.fit_config.y_ceiling)
            x, y, e = x[keep], y[keep], e[keep]

        min_points = (
            self.fit_config.min_fit_points_lrb
            if fit_kind == "lrb"
            else self.fit_config.min_fit_points_rb
        )
        if len(x) < int(min_points):
            return np.nan, np.nan, np.nan

        sigma = self._bootstrap_sigma(
            std_values=e,
            n_sequences=self.num_cliffs,
            use_sem=use_sem,
        )
        fit_errors = sigma if use_sem else e
        boot_err: list[float] = []
        for _ in range(int(n_boot)):
            y_boot = y + rng.normal(loc=0.0, scale=sigma, size=len(y))
            y_boot = np.clip(y_boot, 0.0, 1.0)
            if fit_kind == "lrb":
                fit = self._fit_decay_parameter_fixed_a(
                    depths=x,
                    means=y_boot,
                    errors=fit_errors,
                )
            elif fit_kind == "rb":
                fit = self._fit_decay_parameter_free_a(
                    depths=x,
                    means=y_boot,
                    errors=fit_errors,
                )
            else:
                raise ValueError(f"Unsupported fit_kind: {fit_kind}")
            if fit is None:
                continue
            decay = float(fit["f"])
            if not np.isfinite(decay):
                continue
            err_val = 1.0 - self._decay_to_fidelity(decay)
            if np.isfinite(err_val):
                boot_err.append(float(err_val))

        return self._bootstrap_interval(
            samples=np.asarray(boot_err, dtype=float),
            ci_level=ci_level,
        )

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

    @staticmethod
    def _fit_lrb_curve_global(
        p: np.ndarray,
        y: np.ndarray,
    ) -> tuple[dict[str, float] | None, str]:
        """
        Fit a global exponential-with-offset model for LRB error-vs-p data.

        Model:
            ``y(p) = c + a * exp(b * p)``

        The fit uses a small grid search over ``c`` and linear regression in
        log-space for ``a`` and ``b`` at each ``c``. The best-SSE candidate is
        returned.

        Args:
            p (np.ndarray): Probability values.
            y (np.ndarray): LRB error-rate values.

        Returns:
            tuple[dict[str, float] | None, str]: Fitted model parameters and a
                human-readable model label; ``None`` if no fit succeeds.

        Raises:
            ValueError: Not raised directly by this method.
        """
        p_arr = np.asarray(p, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        mask = np.isfinite(p_arr) & np.isfinite(y_arr)
        p_arr = p_arr[mask]
        y_arr = y_arr[mask]
        if len(p_arr) < 2:
            return None, "fit unavailable"
        y_min = float(np.min(y_arr))
        y_max = float(np.max(y_arr))
        span = max(y_max - y_min, 1e-9)

        # Candidate offsets below the observed floor.
        c_low = y_min - 0.5 * span
        c_high = y_min - 1e-10
        c_grid = np.linspace(c_low, c_high, 200)

        best_sse = np.inf
        best_params: dict[str, float] | None = None
        for c in c_grid:
            shifted = y_arr - c
            if np.any(shifted <= 0):
                continue
            try:
                b, log_a = np.polyfit(p_arr, np.log(shifted), 1)
            except Exception:
                continue
            if not (np.isfinite(b) and np.isfinite(log_a)):
                continue
            a = float(np.exp(log_a))
            y_hat = c + a * np.exp(b * p_arr)
            if not np.all(np.isfinite(y_hat)):
                continue
            sse = float(np.sum((y_arr - y_hat) ** 2))
            if sse < best_sse:
                best_sse = sse
                best_params = {
                    "a": float(a),
                    "b": float(b),
                    "c": float(c),
                }

        if best_params is None:
            return None, "fit unavailable"
        return best_params, "LRB exponential fit"

    def _bootstrap_threshold_p_interval(
        self,
        p: np.ndarray,
        lrb_err: np.ndarray,
        rb_err: np.ndarray,
        lrb_err_std: np.ndarray,
        rb_err_std: np.ndarray,
        cfg: LRBThresholdConfig,
        n_boot: int,
        ci_level: float,
        rng: np.random.Generator,
    ) -> tuple[float, float, float]:
        """
        Bootstrap pseudo-threshold uncertainty from pointwise error-rate std.

        Args:
            p (np.ndarray): Sorted probability values.
            lrb_err (np.ndarray): Baseline LRB error-rate series.
            rb_err (np.ndarray): Baseline RB error-rate series.
            lrb_err_std (np.ndarray): Pointwise LRB std proxy.
            rb_err_std (np.ndarray): Pointwise RB std proxy.
            cfg (LRBThresholdConfig): Threshold controls for each replicate.
            n_boot (int): Number of bootstrap replicates.
            ci_level (float): Central CI level.
            rng (np.random.Generator): RNG used for reproducible resampling.

        Returns:
            tuple[float, float, float]: ``(std, ci_low, ci_high)`` in ``p``.

        Raises:
            ValueError: Not raised directly by this method.
        """
        if int(n_boot) <= 0:
            return np.nan, np.nan, np.nan
        p_arr = np.asarray(p, dtype=float)
        l_arr = np.asarray(lrb_err, dtype=float)
        r_arr = np.asarray(rb_err, dtype=float)
        l_std = np.asarray(lrb_err_std, dtype=float)
        r_std = np.asarray(rb_err_std, dtype=float)

        l_std = self._bootstrap_sigma(
            std_values=l_std,
            n_sequences=1,
            use_sem=False,
        )
        r_std = self._bootstrap_sigma(
            std_values=r_std,
            n_sequences=1,
            use_sem=False,
        )

        p_samples: list[float] = []
        for _ in range(int(n_boot)):
            l_boot = np.clip(
                l_arr + rng.normal(loc=0.0, scale=l_std, size=len(l_arr)),
                0.0,
                np.inf,
            )
            r_boot = np.clip(
                r_arr + rng.normal(loc=0.0, scale=r_std, size=len(r_arr)),
                0.0,
                np.inf,
            )
            start_idx = self._first_threshold_start_index(
                lrb_err=l_boot,
                rb_err=r_boot,
                ignore_first_n=cfg.ignore_first_n,
                err_floor=cfg.err_floor,
            )
            threshold = self._estimate_monotone_threshold_in_p(
                p=p_arr,
                lrb_err=l_boot,
                rb_err=r_boot,
                tol=cfg.tol,
                require_consecutive=cfg.require_consecutive,
                start_idx=start_idx,
            )
            threshold_p = float(threshold.get("threshold_p", np.nan))
            if np.isfinite(threshold_p):
                p_samples.append(threshold_p)

        return self._bootstrap_interval(
            samples=np.asarray(p_samples, dtype=float),
            ci_level=ci_level,
        )

    def _draw_error_vs_p_threshold_axis(
        self,
        ax: Any,
        p: np.ndarray,
        lrb_err: np.ndarray,
        rb_err: np.ndarray,
        lrb_err_ci_low: np.ndarray | None,
        lrb_err_ci_high: np.ndarray | None,
        rb_err_ci_low: np.ndarray | None,
        rb_err_ci_high: np.ndarray | None,
        check_num: int,
        threshold: dict[str, float | str | int | None],
        cfg: LRBThresholdConfig,
        title: str | None = None,
        show_legend: bool = True,
    ) -> None:
        """
        Draw one threshold error-vs-p panel on an existing axis.

        Args:
            ax (Any): Matplotlib axis target.
            p (np.ndarray): Sorted probability values.
            lrb_err (np.ndarray): LRB error-rate values.
            rb_err (np.ndarray): RB error-rate values.
            lrb_err_ci_low (np.ndarray | None): Optional LRB lower CI bounds.
            lrb_err_ci_high (np.ndarray | None): Optional LRB upper CI bounds.
            rb_err_ci_low (np.ndarray | None): Optional RB lower CI bounds.
            rb_err_ci_high (np.ndarray | None): Optional RB upper CI bounds.
            check_num (int): Uniform check value.
            threshold (dict[str, float | str | int | None]): Threshold record.
            cfg (LRBThresholdConfig): Plot and threshold controls.
            title (str | None): Optional axis title override.
            show_legend (bool): Whether to draw a legend.

        Returns:
            None: Draws directly on the axis.

        Raises:
            ValueError: Not raised directly by this method.
        """
        p_arr = np.asarray(p, dtype=float)
        l_arr = np.asarray(lrb_err, dtype=float)
        r_arr = np.asarray(rb_err, dtype=float)
        ax.set_axisbelow(True)
        has_lrb_ci = lrb_err_ci_low is not None and lrb_err_ci_high is not None
        has_rb_ci = rb_err_ci_low is not None and rb_err_ci_high is not None
        if has_lrb_ci:
            lrb_yerr = self._to_asymmetric_yerr(
                y=l_arr,
                low=np.asarray(lrb_err_ci_low, dtype=float),
                high=np.asarray(lrb_err_ci_high, dtype=float),
            )
            ax.errorbar(
                p_arr,
                l_arr,
                yerr=lrb_yerr,
                fmt="-o",
                markersize=4,
                capsize=2,
                elinewidth=0.9,
                label="LRB error rate",
                zorder=2,
            )
        else:
            ax.plot(
                p_arr,
                l_arr,
                "-o",
                markersize=4,
                label="LRB error rate",
                zorder=2,
            )

        if has_rb_ci:
            rb_yerr = self._to_asymmetric_yerr(
                y=r_arr,
                low=np.asarray(rb_err_ci_low, dtype=float),
                high=np.asarray(rb_err_ci_high, dtype=float),
            )
            ax.errorbar(
                p_arr,
                r_arr,
                yerr=rb_yerr,
                fmt="-s",
                markersize=4,
                capsize=2,
                elinewidth=0.9,
                label="RB error rate",
                zorder=2,
            )
        else:
            ax.plot(
                p_arr,
                r_arr,
                "-s",
                markersize=4,
                label="RB error rate",
                zorder=2,
            )

        threshold_p = float(threshold["threshold_p"])
        onset = threshold.get("onset_index", None)
        if np.isfinite(threshold_p):
            ax.axvline(
                threshold_p,
                linestyle="--",
                alpha=0.7,
                label=f"$p^*\\approx{threshold_p:.4g}$",
                zorder=1.5,
            )
            ax.plot(
                [threshold_p],
                [threshold["threshold_lrb_error"]],
                "o",
                markersize=9,
                alpha=0.7,
                color="tab:blue",
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=8,
            )
            ax.plot(
                [threshold_p],
                [threshold["threshold_rb_error"]],
                "o",
                markersize=9,
                alpha=0.7,
                color="tab:red",
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=9,
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
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Physical Noise Parameter p")
        ax.set_ylabel("Error Rate (1 - Fidelity)")
        if title is None:
            title = (
                "LRB vs RB "
                f"({self._title_context('unif', check_num)}): "
                "error rates vs p"
            )
        ax.set_title(title)
        ax.grid(True, zorder=0)
        if show_legend:
            ax.legend()

    def _plot_error_vs_p_threshold(
        self,
        p: np.ndarray,
        lrb_err: np.ndarray,
        rb_err: np.ndarray,
        lrb_err_ci_low: np.ndarray | None,
        lrb_err_ci_high: np.ndarray | None,
        rb_err_ci_low: np.ndarray | None,
        rb_err_ci_high: np.ndarray | None,
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
            lrb_err_ci_low (np.ndarray | None): Optional LRB lower CI bounds.
            lrb_err_ci_high (np.ndarray | None): Optional LRB upper CI bounds.
            rb_err_ci_low (np.ndarray | None): Optional RB lower CI bounds.
            rb_err_ci_high (np.ndarray | None): Optional RB upper CI bounds.
            check_num (int): Uniform check value.
            threshold (dict[str, float | str | int | None]): Threshold record.
            cfg (LRBThresholdConfig): Plot and threshold controls.
            show (bool): Whether to display the figure.

        Returns:
            str: Generated output PDF path.

        Raises:
            ValueError: Not raised directly by this method.
        """
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        self._draw_error_vs_p_threshold_axis(
            ax=ax,
            p=p,
            lrb_err=lrb_err,
            rb_err=rb_err,
            lrb_err_ci_low=lrb_err_ci_low,
            lrb_err_ci_high=lrb_err_ci_high,
            rb_err_ci_low=rb_err_ci_low,
            rb_err_ci_high=rb_err_ci_high,
            check_num=check_num,
            threshold=threshold,
            cfg=cfg,
            title=None,
            show_legend=True,
        )
        fig.tight_layout()

        out_path = os.path.join(
            self.out_dir,
            f"unif-{check_num}-error-vs-p-threshold-monotone.pdf",
        )
        self._save_pdf_overwrite(
            fig=fig,
            out_path=out_path,
            dpi=300,
            bbox_inches="tight",
        )
        if show:
            plt.show()
        plt.close(fig)
        print(f"[OK] wrote {out_path}")
        return out_path

    def _plot_error_vs_p_threshold_paper_grid(
        self,
        rows: list[dict[str, Any]],
        cfg: LRBThresholdConfig,
        show: bool,
    ) -> str | None:
        """
        Render all unif threshold error-vs-p panels on one paper-style canvas.

        Args:
            rows (list[dict[str, Any]]): Per-check threshold plotting records.
            cfg (LRBThresholdConfig): Threshold/paper controls.
            show (bool): Whether to display the figure.

        Returns:
            str | None: Output PDF path, or ``None`` if no rows are available.

        Raises:
            ValueError: Not raised directly by this method.
        """
        if not rows:
            return None
        n_items = len(rows)
        n_cols = max(1, int(cfg.paper_cols))
        n_rows = int(np.ceil(n_items / float(n_cols)))
        fig_w = max(5.5, float(cfg.paper_panel_width_in) * n_cols)
        fig_h = max(3.2, float(cfg.paper_panel_height_in) * n_rows)

        rc_updates: dict[str, float] = {
            "font.size": float(cfg.paper_tick_fontsize),
            "axes.labelsize": float(cfg.paper_axis_label_fontsize),
            "xtick.labelsize": float(cfg.paper_tick_fontsize),
            "ytick.labelsize": float(cfg.paper_tick_fontsize),
            "legend.fontsize": float(cfg.paper_legend_fontsize),
            "lines.linewidth": float(cfg.paper_line_width),
            "lines.markersize": float(cfg.paper_marker_size),
        }

        with mpl.rc_context(rc=rc_updates):
            fig, axs = plt.subplots(
                nrows=n_rows,
                ncols=n_cols,
                figsize=(fig_w, fig_h),
                constrained_layout=True,
            )
            axs_arr = np.atleast_2d(axs)
            if n_rows == 1:
                axs_arr = axs_arr.reshape(1, n_cols)
            if n_cols == 1:
                axs_arr = axs_arr.reshape(n_rows, 1)

            for idx, row in enumerate(rows):
                rr = idx // n_cols
                cc = idx % n_cols
                ax = axs_arr[rr, cc]
                check_num = int(row["check_num"])
                self._draw_error_vs_p_threshold_axis(
                    ax=ax,
                    p=np.asarray(row["p"], dtype=float),
                    lrb_err=np.asarray(row["lrb_err"], dtype=float),
                    rb_err=np.asarray(row["rb_err"], dtype=float),
                    lrb_err_ci_low=row["lrb_err_ci_low"],
                    lrb_err_ci_high=row["lrb_err_ci_high"],
                    rb_err_ci_low=row["rb_err_ci_low"],
                    rb_err_ci_high=row["rb_err_ci_high"],
                    check_num=check_num,
                    threshold=row["threshold"],
                    cfg=cfg,
                    title=f"Uniform Interval Check = {check_num}",
                    show_legend=True,
                )
                ax.title.set_fontsize(float(cfg.paper_title_fontsize))

            for idx in range(n_items, n_rows * n_cols):
                rr = idx // n_cols
                cc = idx % n_cols
                axs_arr[rr, cc].set_visible(False)

            out_path = os.path.join(
                self.out_dir,
                "unif-threshold-error-vs-p-paper-grid.pdf",
            )
            self._save_pdf_overwrite(
                fig=fig,
                out_path=out_path,
                dpi=300,
                bbox_inches="tight",
            )
            if show:
                plt.show()
            plt.close(fig)

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
        plt.plot(
            r_win,
            l_win,
            "-o",
            markersize=4,
            label="(RB(p), LRB(p)) p-order",
        )
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
        plt.xlabel("RB Error Rate (1 - Fidelity)")
        plt.ylabel("LRB Error Rate (1 - Fidelity)")
        plt.title(
            "LRB vs RB "
            f"({self._title_context('unif', check_num)})"
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(
            self.out_dir,
            f"unif-{check_num}-lrb-vs-rb-threshold-monotone.pdf",
        )
        self._save_pdf_overwrite(
            fig=plt.gcf(),
            out_path=out_path,
            dpi=300,
            bbox_inches="tight",
        )
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
        check_min: int | None = None,
        check_max: int | None = None,
    ) -> str:
        """
        Build threshold graphs and summary CSV for all uniform checks.

        Args:
            threshold_config (LRBThresholdConfig | None): Optional controls.
            table_csv_path (str | None): Optional input table path.
            show (bool): Whether to display generated figures.
            check_min (int | None): Optional inclusive lower check bound.
            check_max (int | None): Optional inclusive upper check bound.
            Paper mode:
            When ``threshold_config.paper_mode`` is true, this method skips
            per-check threshold PDF outputs and writes one combined
            error-vs-p grid:
            ``unif-threshold-error-vs-p-paper-grid.pdf``.

        Returns:
            str: Path to the generated threshold summary CSV.

        Raises:
            RuntimeError: If pandas is unavailable.
            FileNotFoundError: If no unif table is available.
        """
        if pd is None:
            raise RuntimeError("pandas is required for threshold plotting.")

        cfg = threshold_config or LRBThresholdConfig()
        ci_level = self._normalize_ci_level(cfg.bootstrap_ci_level)
        rng = np.random.default_rng(cfg.bootstrap_seed)
        point_ci_cols = [
            "lrb_error_rate_std_boot",
            "lrb_error_rate_ci_low",
            "lrb_error_rate_ci_high",
            "rb_error_rate_std_boot",
            "rb_error_rate_ci_low",
            "rb_error_rate_ci_high",
        ]
        if table_csv_path is None:
            table_csv_path = self._find_unif_table_csv()
        if table_csv_path is None:
            # Auto-build the table when the notebook has not generated it yet.
            table_csv_path = self.build_unif_lrb_vs_rb_table_mixed_fits(
                write_per_check_tables=False,
                check_min=check_min,
                check_max=check_max,
                bootstrap_reps=cfg.bootstrap_reps_error,
                bootstrap_ci_level=ci_level,
                bootstrap_use_sem=cfg.bootstrap_use_sem,
                bootstrap_seed=cfg.bootstrap_seed,
            )
        if not os.path.exists(table_csv_path):
            raise FileNotFoundError(f"Missing table CSV: {table_csv_path}")

        frame = pd.read_csv(table_csv_path)
        required_cols = ["check_num", "p", "lrb_error_rate", "rb_error_rate"]
        for col in required_cols:
            if col not in frame.columns:
                raise KeyError(f"Missing column '{col}' in {table_csv_path}")
        has_point_ci = all(col in frame.columns for col in point_ci_cols)
        if cfg.bootstrap_reps_error > 0 and not has_point_ci:
            table_csv_path = self.build_unif_lrb_vs_rb_table_mixed_fits(
                write_per_check_tables=False,
                check_min=check_min,
                check_max=check_max,
                bootstrap_reps=cfg.bootstrap_reps_error,
                bootstrap_ci_level=ci_level,
                bootstrap_use_sem=cfg.bootstrap_use_sem,
                bootstrap_seed=cfg.bootstrap_seed,
            )
            frame = pd.read_csv(table_csv_path)
            has_point_ci = all(col in frame.columns for col in point_ci_cols)

        has_lrb_n = "lrb_n_points" in frame.columns
        has_rb_n = "rb_n_points" in frame.columns
        checks = sorted([int(v) for v in frame["check_num"].dropna().unique()])
        checks = self._filter_checks_by_range(
            checks=checks,
            check_min=check_min,
            check_max=check_max,
        )
        if not checks:
            raise ValueError("No unif checks found in the requested range.")

        summary_rows: list[dict[str, float | int | str]] = []
        paper_rows: list[dict[str, Any]] = []
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
            if has_point_ci:
                lrb_ci_low = sub["lrb_error_rate_ci_low"].to_numpy(dtype=float)
                lrb_ci_high = sub["lrb_error_rate_ci_high"].to_numpy(
                    dtype=float)
                rb_ci_low = sub["rb_error_rate_ci_low"].to_numpy(dtype=float)
                rb_ci_high = sub["rb_error_rate_ci_high"].to_numpy(dtype=float)
                lrb_std_boot = sub["lrb_error_rate_std_boot"].to_numpy(
                    dtype=float)
                rb_std_boot = sub["rb_error_rate_std_boot"].to_numpy(
                    dtype=float)
            else:
                lrb_ci_low = np.full_like(lrb_err, np.nan, dtype=float)
                lrb_ci_high = np.full_like(lrb_err, np.nan, dtype=float)
                rb_ci_low = np.full_like(rb_err, np.nan, dtype=float)
                rb_ci_high = np.full_like(rb_err, np.nan, dtype=float)
                lrb_std_boot = np.full_like(lrb_err, np.nan, dtype=float)
                rb_std_boot = np.full_like(rb_err, np.nan, dtype=float)
            finite = self._finite_mask(p, lrb_err, rb_err)
            p = p[finite]
            lrb_err = lrb_err[finite]
            rb_err = rb_err[finite]
            lrb_ci_low = lrb_ci_low[finite]
            lrb_ci_high = lrb_ci_high[finite]
            rb_ci_low = rb_ci_low[finite]
            rb_ci_high = rb_ci_high[finite]
            lrb_std_boot = lrb_std_boot[finite]
            rb_std_boot = rb_std_boot[finite]
            if len(p) < 2:
                summary_rows.append(
                    {
                        "check_num": int(check_num),
                        "threshold_p": np.nan,
                        "threshold_p_std_boot": np.nan,
                        "threshold_p_ci_low": np.nan,
                        "threshold_p_ci_high": np.nan,
                        "threshold_rb_error_rate": np.nan,
                        "threshold_lrb_error_rate": np.nan,
                        "method": "insufficient_points",
                        "n_points_used": int(len(p)),
                    }
                )
                continue

            order = np.argsort(p)
            p = p[order]
            lrb_err = lrb_err[order]
            rb_err = rb_err[order]
            lrb_ci_low = lrb_ci_low[order]
            lrb_ci_high = lrb_ci_high[order]
            rb_ci_low = rb_ci_low[order]
            rb_ci_high = rb_ci_high[order]
            lrb_std_boot = lrb_std_boot[order]
            rb_std_boot = rb_std_boot[order]

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
            lrb_ci_low_trim = lrb_ci_low[:cut_idx]
            lrb_ci_high_trim = lrb_ci_high[:cut_idx]
            rb_ci_low_trim = rb_ci_low[:cut_idx]
            rb_ci_high_trim = rb_ci_high[:cut_idx]
            lrb_std_trim = lrb_std_boot[:cut_idx]
            rb_std_trim = rb_std_boot[:cut_idx]
            if len(p_trim) < 2:
                summary_rows.append(
                    {
                        "check_num": int(check_num),
                        "threshold_p": np.nan,
                        "threshold_p_std_boot": np.nan,
                        "threshold_p_ci_low": np.nan,
                        "threshold_p_ci_high": np.nan,
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
            threshold_std, threshold_lo, threshold_hi = np.nan, np.nan, np.nan
            if has_point_ci and cfg.bootstrap_reps_threshold > 0:
                threshold_std, threshold_lo, threshold_hi = (
                    self._bootstrap_threshold_p_interval(
                        p=p_trim,
                        lrb_err=lrb_trim,
                        rb_err=rb_trim,
                        lrb_err_std=lrb_std_trim,
                        rb_err_std=rb_std_trim,
                        cfg=cfg,
                        n_boot=cfg.bootstrap_reps_threshold,
                        ci_level=ci_level,
                        rng=rng,
                    )
                )

            summary_rows.append(
                {
                    "check_num": int(check_num),
                    "threshold_p": float(thr["threshold_p"]),
                    "threshold_p_std_boot": float(threshold_std),
                    "threshold_p_ci_low": float(threshold_lo),
                    "threshold_p_ci_high": float(threshold_hi),
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
            if cfg.paper_mode:
                paper_rows.append(
                    {
                        "check_num": int(check_num),
                        "p": np.asarray(p_trim, dtype=float),
                        "lrb_err": np.asarray(lrb_trim, dtype=float),
                        "rb_err": np.asarray(rb_trim, dtype=float),
                        "lrb_err_ci_low": (
                            np.asarray(lrb_ci_low_trim, dtype=float)
                            if has_point_ci
                            else None
                        ),
                        "lrb_err_ci_high": (
                            np.asarray(lrb_ci_high_trim, dtype=float)
                            if has_point_ci
                            else None
                        ),
                        "rb_err_ci_low": (
                            np.asarray(rb_ci_low_trim, dtype=float)
                            if has_point_ci
                            else None
                        ),
                        "rb_err_ci_high": (
                            np.asarray(rb_ci_high_trim, dtype=float)
                            if has_point_ci
                            else None
                        ),
                        "threshold": thr,
                    }
                )
            else:
                self._plot_error_vs_p_threshold(
                    p=p_trim,
                    lrb_err=lrb_trim,
                    rb_err=rb_trim,
                    lrb_err_ci_low=lrb_ci_low_trim if has_point_ci else None,
                    lrb_err_ci_high=lrb_ci_high_trim if has_point_ci else None,
                    rb_err_ci_low=rb_ci_low_trim if has_point_ci else None,
                    rb_err_ci_high=rb_ci_high_trim if has_point_ci else None,
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

        if cfg.paper_mode:
            self._plot_error_vs_p_threshold_paper_grid(
                rows=paper_rows,
                cfg=cfg,
                show=show,
            )

        summary = pd.DataFrame(summary_rows).sort_values(
            "check_num").reset_index(drop=True)
        for _, row in summary.iterrows():
            p_star = float(row.get("threshold_p", np.nan))
            if not np.isfinite(p_star):
                continue
            std_boot = float(row.get("threshold_p_std_boot", np.nan))
            ci_low = float(row.get("threshold_p_ci_low", np.nan))
            ci_high = float(row.get("threshold_p_ci_high", np.nan))
            if np.isfinite(std_boot):
                unc = std_boot
            elif np.isfinite(ci_low) and np.isfinite(ci_high):
                unc = 0.5 * abs(ci_high - ci_low)
            else:
                unc = np.nan
            check_num = int(row.get("check_num", -1))
            if np.isfinite(unc):
                print(
                    f"[THRESHOLD] Uniform Interval Check = {check_num}: "
                    f"p* = {p_star:.6g} +- {unc:.3g}"
                )
            else:
                print(
                    f"[THRESHOLD] Uniform Interval Check = {check_num}: "
                    f"p* = {p_star:.6g} +- NaN"
                )
        out_summary = os.path.join(
            self.out_dir,
            "unif_thresholds_summary_monotone_trim_zoom_pwindow.csv",
        )
        summary.to_csv(out_summary, index=False)
        print(f"[OK] wrote {out_summary}")
        return out_summary

    def plot_unif_thresholds_monotone_worse_trim_tail_zoom_focus_pwindow(
        self,
        threshold_config: LRBThresholdConfig | None = None,
        table_csv_path: str | None = None,
        show: bool = True,
        check_min: int | None = None,
        check_max: int | None = None,
    ) -> str:
        """
        Compatibility wrapper for the standalone monotone-threshold script flow.

        This method keeps the same behavior and outputs as
        ``plot_all_unif_threshold_graphs``:
          - ``unif-<CHECK>-error-vs-p-threshold-monotone.pdf``
          - ``unif-<CHECK>-lrb-vs-rb-threshold-monotone.pdf``
          - ``unif_thresholds_summary_monotone_trim_zoom_pwindow.csv``
        """
        return self.plot_all_unif_threshold_graphs(
            threshold_config=threshold_config,
            table_csv_path=table_csv_path,
            show=show,
            check_min=check_min,
            check_max=check_max,
        )

    def plot_unif_pseudo_thresholds_vs_interval_check(
        self,
        check_min: int = 1,
        check_max: int = 4,
        summary_csv_path: str | None = None,
        do_fit: bool = True,
        fit_model: str = "exp",
        fit_degree: int = 1,
        paper_mode: bool = False,
        show: bool = True,
    ) -> str:
        """
        Plot pseudo-threshold ``p`` versus uniform interval-check number.

        The method reads the threshold summary CSV, keeps checks in
        ``[check_min, check_max]``, and optionally overlays a fit model
        (exponential by default, polynomial optional). If the threshold
        summary does not exist yet, it is generated automatically.

        Args:
            check_min (int): Inclusive lower bound of unif check numbers.
            check_max (int): Inclusive upper bound of unif check numbers.
            summary_csv_path (str | None): Optional threshold summary CSV path.
            do_fit (bool): Whether to overlay a polynomial fit.
            fit_model (str): ``"exp"`` or ``"poly"`` fit model.
            fit_degree (int): Requested polynomial degree for fitting.
            paper_mode (bool): If true, suppress the plot title/header.
            show (bool): Whether to display the figure.

        Returns:
            str: Output PDF path for the pseudo-threshold fit plot.

        Raises:
            RuntimeError: If pandas is unavailable.
            FileNotFoundError: If summary CSV is missing and cannot be built.
            ValueError: If no finite thresholds exist in the selected range.
        """
        if pd is None:
            raise RuntimeError("pandas is required for pseudo-threshold plots.")
        if check_min > check_max:
            raise ValueError("check_min must be <= check_max.")
        fit_model = fit_model.strip().lower()
        if fit_model not in ("exp", "poly"):
            raise ValueError("fit_model must be 'exp' or 'poly'.")

        if summary_csv_path is None:
            summary_csv_path = os.path.join(
                self.out_dir,
                "unif_thresholds_summary_monotone_trim_zoom_pwindow.csv",
            )
        if not os.path.exists(summary_csv_path):
            summary_csv_path = self.plot_all_unif_threshold_graphs(show=False)
        if not os.path.exists(summary_csv_path):
            raise FileNotFoundError(f"Missing summary CSV: {summary_csv_path}")

        frame = pd.read_csv(summary_csv_path)
        for col in ("check_num", "threshold_p"):
            if col not in frame.columns:
                raise KeyError(f"Missing column '{col}' in {summary_csv_path}")

        sub = frame[
            (frame["check_num"] >= int(check_min))
            & (frame["check_num"] <= int(check_max))
        ].copy()
        sub = sub[np.isfinite(sub["threshold_p"])].copy()
        sub = sub.sort_values("check_num").reset_index(drop=True)
        if sub.empty:
            raise ValueError(
                "No finite pseudo-threshold points in the selected check range."
            )

        x = sub["check_num"].to_numpy(dtype=float)
        y = sub["threshold_p"].to_numpy(dtype=float)
        has_threshold_ci = (
            "threshold_p_ci_low" in sub.columns
            and "threshold_p_ci_high" in sub.columns
        )

        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        ax.set_axisbelow(True)
        if has_threshold_ci:
            yerr = self._to_asymmetric_yerr(
                y=y,
                low=sub["threshold_p_ci_low"].to_numpy(dtype=float),
                high=sub["threshold_p_ci_high"].to_numpy(dtype=float),
            )
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o",
                markersize=6,
                capsize=3,
                elinewidth=1.0,
                alpha=0.9,
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=3,
                label=r"Pseudo-threshold $p^*$ data",
            )
        else:
            ax.scatter(
                x,
                y,
                s=36,
                zorder=3,
                edgecolors="white",
                linewidths=0.6,
                label=r"Pseudo-threshold $p^*$ data",
            )

        # Print check-wise pseudo-threshold statements in "# +- #" form.
        if "threshold_p_std_boot" in sub.columns:
            unc_arr = sub["threshold_p_std_boot"].to_numpy(dtype=float)
        elif has_threshold_ci:
            lo_arr = sub["threshold_p_ci_low"].to_numpy(dtype=float)
            hi_arr = sub["threshold_p_ci_high"].to_numpy(dtype=float)
            unc_arr = 0.5 * np.abs(hi_arr - lo_arr)
        else:
            unc_arr = np.full_like(y, np.nan, dtype=float)
        for k, yv, uv in zip(x, y, unc_arr):
            if np.isfinite(uv):
                print(
                    f"[PSEUDO] Uniform Interval Check = {int(k)}: "
                    f"p* = {float(yv):.6g} +- {float(uv):.3g}"
                )
            else:
                print(
                    f"[PSEUDO] Uniform Interval Check = {int(k)}: "
                    f"p* = {float(yv):.6g} +- NaN"
                )

        degree = max(0, int(fit_degree))
        degree = min(degree, len(x) - 1)
        if do_fit and len(x) >= 2:
            x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 200)
            if fit_model == "exp":
                positive = y > 0.0
                if int(np.sum(positive)) < 2:
                    print(
                        "[FIT] skipped exponential fit: need at least two "
                        "positive pseudo-threshold values."
                    )
                else:
                    slope, intercept = np.polyfit(
                        x[positive],
                        np.log(y[positive]),
                        1,
                    )
                    y_fit = np.exp(intercept + slope * x_fit)
                    ax.plot(
                        x_fit,
                        y_fit,
                        "--",
                        linewidth=1.5,
                        zorder=2,
                        label="exp fit",
                    )

                    y_hat = np.exp(intercept + slope * x)
                    ss_res = float(np.sum((y - y_hat) ** 2))
                    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                    r2 = np.nan if ss_tot <= 0 else 1.0 - (ss_res / ss_tot)
                    print(
                        "[FIT] exponential pseudo-threshold fit: "
                        f"p ~= exp({intercept:.6g} + {slope:.6g}*k), "
                        f"R^2={r2:.6g}"
                    )
            elif degree >= 1:
                coeff = np.polyfit(x, y, degree)
                poly = np.poly1d(coeff)
                y_fit = poly(x_fit)
                ax.plot(
                    x_fit,
                    y_fit,
                    "--",
                    linewidth=1.5,
                    zorder=2,
                    label="poly fit",
                )

                y_hat = poly(x)
                ss_res = float(np.sum((y - y_hat) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = np.nan if ss_tot <= 0 else 1.0 - (ss_res / ss_tot)
                if degree == 1:
                    slope = float(coeff[0])
                    intercept = float(coeff[1])
                    print(
                        "[FIT] linear pseudo-threshold fit: "
                        f"p ~= {slope:.6g}*k + {intercept:.6g}, R^2={r2:.6g}"
                    )
                else:
                    print(
                        "[FIT] polynomial pseudo-threshold fit: "
                        f"degree={degree}, R^2={r2:.6g}"
                    )
            else:
                print(
                    "[FIT] skipped polynomial fit: "
                    "insufficient points or degree=0."
                )
        else:
            print("[FIT] fit disabled by do_fit=False.")

        ax.set_xlabel("Uniform Interval Check Number")
        ax.set_ylabel(r"Pseudo-threshold $p^*$")
        if not paper_mode:
            ax.set_title(
                r"Pseudo-threshold $p^*$ vs Uniform Interval Check "
                f"(q={self.d_dim}, {self.code_title})"
            )
        tick_start = int(np.floor(np.min(x)))
        tick_end = int(np.ceil(np.max(x)))
        ax.set_xticks(list(range(tick_start, tick_end + 1)))
        ax.grid(True, zorder=0)
        ax.legend()
        fig.tight_layout()

        out_path = os.path.join(
            self.out_dir,
            f"unif-{int(check_min)}-to-{int(check_max)}-"
            "pseudo-threshold-vs-interval-check-fit.pdf",
        )
        self._save_pdf_overwrite(
            fig=fig,
            out_path=out_path,
            dpi=300,
            bbox_inches="tight",
        )
        if show:
            plt.show()
        plt.close(fig)
        print(f"[OK] wrote {out_path}")
        return out_path
