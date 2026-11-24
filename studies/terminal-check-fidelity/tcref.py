"""Exact reference values the estimators are compared with.

true unit infidelity   eps_L(1): logical infidelity of one depth unit (one
                       mirror pair, or one Clifford for the compiled closing)
                       between a perfect preparation and a perfect check,
                       pooled over the catalog in a random frame.
coincidence infidelity eps_co: what the block-rate mode fit converges to with
                       infinite data, i.e. the part of eps_L(1) that comes
                       from coincidences of independent errors at the mean
                       rates. Defined operationally as the block fit of the
                       exact sequence-averaged distribution on a designed
                       depth grid with ideal SPAM.
F_cov = eps_L(1) / eps_co: the factor by which the spread of error content
                       between Cliffords raises the average over single
                       Cliffords above eps_co.
"""
from __future__ import annotations

import functools

import numpy as np

import tcfit as F
import tcsteps as S


def design_depths(discard_per_step: float, n: int = 8, reach: float = 1.0, cap: int = 200) -> list:
    """Depth grid 0..m_max with m_max ~ reach / (rejection rate per depth unit).

    At m_max the acceptance has dropped to about exp(-reach) of its depth-0
    value. Depths are spread as (j/n)^1.5 so shallow ones are included, and
    depths 0 and 1 are always present.
    """
    rate = -np.log1p(-min(discard_per_step, 0.9))
    m_max = int(np.clip(np.ceil(reach / rate), 4, cap))
    grid = np.unique(np.round(m_max * np.linspace(0, 1, n) ** 1.5).astype(int))
    return [int(m) for m in np.unique(np.concatenate([[0, 1], grid]))]


@functools.lru_cache(maxsize=None)
def _model(code, noise):
    return S.StepModel(code, noise)


@functools.lru_cache(maxsize=None)
def reference(code: str, noise: str, p: float, protocol: str = "mirror") -> dict:
    sm = _model(code, noise)
    d = sm.d
    gt = sm.ground_truth(p)
    mirror = protocol == "mirror"
    true_unit = gt["pair_stat"] if mirror else gt["single_stat"]
    discard = gt["pair_stat_discard"] if mirror else gt["single_stat_discard"]
    depths = design_depths(discard)
    x = np.array(depths, float)
    if not mirror:
        x = np.where(x > 0, x + 1, 0.0)
    ex = sm.exact_curve(p, depths, protocol)
    co = F.mode_fit(x, ex["P"] * 1e12, d, "block")["eps1"]
    return {"true_unit": true_unit, "coincidence": co, "f_cov": true_unit / co, "discard": discard,
            "depths": depths}
