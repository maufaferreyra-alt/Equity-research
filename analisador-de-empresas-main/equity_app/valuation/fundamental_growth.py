"""
Damodaran fundamental growth estimation.

Sustainable growth rate derived from reinvestment economics:
    g_FCFF = Reinvestment Rate x ROIC
    g_FCFE = Retention Ratio x ROE

This anchors growth assumptions in business fundamentals rather than
extrapolating historical revenue trends.

Ref: Damodaran, "Investment Valuation", 3rd ed, Ch 11
("Estimating Growth").
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from analysis.koller_reorg import ReorganizedFinancials
from analysis.lifecycle_classifier import LifecycleStage
from analysis.ratios import _get, cagr


@dataclass
class FundamentalGrowth:
    g_fcff: float                       # Reinvestment Rate x ROIC
    g_fcfe: float                       # Retention Ratio x ROE
    g_revenue_historical: float
    g_consensus: Optional[float]        # Analyst estimate (currently unused)
    recommended_g_explicit: float       # High-growth phase (5-10y)
    recommended_g_stable: float         # Perpetuity, <= risk_free_rate
    rationale: str


def _clamp_growth(g: float, lo: float = -0.05, hi: float = 0.30) -> float:
    if g is None or not np.isfinite(g):
        return float("nan")
    return float(max(lo, min(g, hi)))


def _nan_mean(values) -> float:
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def estimate_fundamental_growth(
    reorg: ReorganizedFinancials,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    *,
    stage: LifecycleStage,
    risk_free_rate: float = 0.04,
    cash: Optional[pd.DataFrame] = None,
) -> FundamentalGrowth:
    """Anchor the DCF explicit-period growth in business fundamentals.

    The four candidate growth measures are blended differently by
    lifecycle stage (see the body for the per-stage rule). The
    ``risk_free_rate`` parameter is only used to cap the *perpetuity*
    growth — the explicit-period number returned here is left free,
    so the caller can apply ``min(recommended_g, risk_free_rate)`` at
    the terminal-stage boundary.
    """
    # ---- g_fcff = mean(reinvestment_rate_3y) * mean(ROIC_3y) ----
    rr_3y = reorg.reinvestment_rate.replace([np.inf, -np.inf], np.nan).dropna().tail(3)
    roic_3y = reorg.roic.replace([np.inf, -np.inf], np.nan).dropna().tail(3)
    if not rr_3y.empty and not roic_3y.empty:
        g_fcff_raw = float(rr_3y.mean()) * float(roic_3y.mean())
    else:
        g_fcff_raw = float("nan")
    g_fcff = _clamp_growth(g_fcff_raw)

    # ---- g_fcfe = retention_ratio * ROE ----
    net_income = _get(income, "net_income")
    equity = _get(balance, "total_equity")
    # dividendsPaid is on the cash-flow statement; fall back to the
    # income statement only because some adapters denormalize it there.
    dividends = _get(cash, "dividends_paid") if cash is not None else None
    if dividends is None:
        dividends = _get(income, "dividends_paid")
    g_fcfe_raw = float("nan")
    if net_income is not None and equity is not None:
        avg_eq = equity.rolling(window=2, min_periods=1).mean()
        roe = net_income / avg_eq.where(avg_eq != 0)
        if dividends is not None:
            # Dividends are typically negative on the cash-flow statement;
            # abs() normalizes both sign conventions.
            ratio = dividends.abs() / net_income.where(net_income > 0)
            retention = (1.0 - ratio).clip(lower=0.0, upper=1.0)
        else:
            retention = pd.Series(1.0, index=net_income.index)
        g_fcfe_series = (retention * roe).replace([np.inf, -np.inf], np.nan).dropna().tail(3)
        if not g_fcfe_series.empty:
            g_fcfe_raw = float(g_fcfe_series.mean())
    g_fcfe = _clamp_growth(g_fcfe_raw)

    # ---- g_revenue_historical = 5y revenue CAGR ----
    revenue = _get(income, "revenue")
    g_rev = cagr(revenue, periods=4) if revenue is not None else float("nan")
    if not np.isfinite(g_rev) and revenue is not None:
        g_rev = cagr(revenue)
    g_revenue_historical = _clamp_growth(g_rev)

    # ---- Stage-aware blend ----
    if stage in ("young_growth", "high_growth"):
        recommended = _nan_mean([g_fcff, g_fcfe, g_revenue_historical])
        if np.isfinite(recommended):
            recommended = min(recommended, 0.25)
    elif stage in ("mature_growth", "mature_stable"):
        recommended = _nan_mean([g_fcff, g_fcfe])
    elif stage == "cyclical":
        candidates = [g for g in (g_fcff, g_revenue_historical, 0.05)
                      if g is not None and np.isfinite(g)]
        recommended = float(min(candidates)) if candidates else float("nan")
    elif stage == "declining":
        if np.isfinite(g_revenue_historical):
            recommended = min(g_revenue_historical, 0.0)
        else:
            recommended = 0.0
    else:
        recommended = _nan_mean([g_fcff, g_fcfe, g_revenue_historical])

    recommended_g_explicit = _clamp_growth(recommended)

    # ---- Perpetuity / stable growth ----
    # Damodaran's cap: g_stable <= risk_free_rate, since no company
    # can outgrow the economy in perpetuity. Cyclicals get a softer
    # floor (~inflation-like) because their through-cycle terminal
    # growth shouldn't collapse to zero just because the explicit
    # period was bounded at 5%.
    if stage == "cyclical":
        if np.isfinite(g_revenue_historical):
            stable_candidate = min(g_revenue_historical, risk_free_rate * 1.25)
        else:
            stable_candidate = risk_free_rate * 1.25
    else:
        if np.isfinite(recommended_g_explicit):
            stable_candidate = min(recommended_g_explicit, risk_free_rate)
        else:
            stable_candidate = risk_free_rate
    recommended_g_stable = _clamp_growth(stable_candidate)

    # ---- Rationale ----
    blend_label = {
        "young_growth": "avg(g_FCFF, g_FCFE, g_hist) capped at 25%",
        "high_growth": "avg(g_FCFF, g_FCFE, g_hist) capped at 25%",
        "mature_growth": "avg(g_FCFF, g_FCFE) - fundamentals anchor",
        "mature_stable": "avg(g_FCFF, g_FCFE) - fundamentals anchor",
        "cyclical": "through-cycle min(g_FCFF, g_hist, 5%)",
        "declining": "min(g_hist, 0)",
    }.get(stage, "blended fundamentals")
    stable_label = (
        f"min(g_hist, rf*1.25)={recommended_g_stable:.2%}"
        if stage == "cyclical"
        else f"min(g_explicit, rf)={recommended_g_stable:.2%}"
    )

    rationale = (
        f"{stage}: g_FCFF={g_fcff:.2%}, g_FCFE={g_fcfe:.2%}, "
        f"g_hist={g_revenue_historical:.2%} -> "
        f"explicit={recommended_g_explicit:.2%} ({blend_label}); "
        f"stable={recommended_g_stable:.2%} ({stable_label}, rf={risk_free_rate:.2%})."
    )

    return FundamentalGrowth(
        g_fcff=g_fcff,
        g_fcfe=g_fcfe,
        g_revenue_historical=g_revenue_historical,
        g_consensus=None,
        recommended_g_explicit=recommended_g_explicit,
        recommended_g_stable=recommended_g_stable,
        rationale=rationale,
    )
